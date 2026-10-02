//! Primary columns of a large log, built in a stream.
//!
//! [`build_primary_columns`] produces the primary columns that one claim
//! holds for a log of `leaves` leaves whose values come from a seed. It walks
//! the Merkle tree once, computes every label, and adds each cell's
//! contribution to the column entry that reads it. The encoding is linear and
//! the contributions can be added in any order, so the tree is never held in
//! memory: the state is the columns and one stack per worker.
//!
//! The result equals, entry for entry, the columns a maintainer holds after
//! the same leaves are appended.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;

use crate::Fp;
use crate::auth::AuthParams;
use crate::auth::allocation::default_value_at_index;
use crate::auth::mapping::{self, NodeLinearization};
use crate::auth::merkle::{default_hash, leaf_hash, node_hash};
use crate::coded_array::axis::segment_axis_indices;
use crate::field::fp_zero;
use crate::private::layout::PrivateLayout;
use crate::private::pir::{PirShape, or_entry_into_cells};
use crate::unauth::CodedArrayParams;
use crate::value::{Value256, decode_value, encode_value};

/// Leaves per worker task, as a power of two.
pub const CHUNK_BITS: u32 = 16;

/// The value of leaf `index` of the log with seed `seed`: a nonzero field
/// element derived from both.
pub fn bulk_leaf_value(seed: u64, index: u64) -> Value256 {
    // SplitMix64 over (seed, index, limb).
    let mut bytes = [0u8; 32];
    for (limb, chunk) in bytes.chunks_exact_mut(8).enumerate() {
        let mut x = seed
            ^ index.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ (limb as u64).wrapping_mul(0xd1b5_4a32_d192_ed03);
        x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^= x >> 31;
        chunk.copy_from_slice(&x.to_le_bytes());
    }
    let mut value = decode_value(bytes);
    if value == fp_zero() {
        value = Fp::from(1u64);
    }
    encode_value(value)
}

/// The primary columns of one claim for a whole log.
pub struct BulkColumns {
    /// `segments * kappa` entries: the primary column of each segment.
    pub entries: Vec<Fp>,
    /// Number of segments.
    pub segments: usize,
    /// Root of the Merkle tree.
    pub root: Value256,
}

/// Adds the contribution of cells to the column entries of one claim.
struct ColumnEncoder<'a> {
    params: &'a CodedArrayParams,
    column_identifier: Fp,
    /// Z-axis and W-axis index of each logical offset of a segment.
    axes_of_offset: Vec<(u16, u16)>,
    /// Z-axis Lagrange basis at the column identifier, by Z-scale index.
    basis_by_scale: HashMap<usize, Vec<Fp>>,
}

impl<'a> ColumnEncoder<'a> {
    fn new(params: &'a CodedArrayParams, registry_index: usize) -> Self {
        let axes_of_offset = (0..params.n_grid)
            .map(|offset| {
                let grid = params.logical_grid_index(offset);
                (
                    params.z_index_for_grid_index(grid) as u16,
                    params.w_index_for_grid_index(grid) as u16,
                )
            })
            .collect();
        Self {
            params,
            column_identifier: params.shard_registry[registry_index].z,
            axes_of_offset,
            basis_by_scale: HashMap::new(),
        }
    }

    /// Add the cell at `array_index` with content `value` to `entries`,
    /// which starts at segment `first_segment`.
    fn add_cell(&mut self, entries: &mut [Fp], first_segment: usize, array_index: u64, value: Fp) {
        let kappa = self.params.need;
        let segment = (array_index / self.params.n_grid as u64) as usize;
        let offset = (array_index % self.params.n_grid as u64) as usize;
        let (z, w) = self.axes_of_offset[offset];
        let (z_scale, _) = segment_axis_indices(segment);
        let params = self.params;
        let column_identifier = self.column_identifier;
        let basis = self.basis_by_scale.entry(z_scale).or_insert_with(|| {
            let mut basis = vec![fp_zero(); kappa];
            params.z_basis_at(&params.segment_axes(segment), column_identifier, &mut basis);
            basis
        });
        entries[(segment - first_segment) * kappa + w as usize] += basis[z as usize] * value;
    }
}

/// Shape of the tree walk.
struct Walk<'a> {
    auth: &'a AuthParams,
    leaves: u64,
    seed: u64,
}

impl Walk<'_> {
    fn height(&self) -> u32 {
        self.auth.tree_height as u32
    }

    /// Whether the node at `(level, pos)` has a filled leaf below it.
    fn is_active(&self, level: u32, pos: u64) -> bool {
        let shift = self.height() - level;
        if shift >= 64 {
            return pos == 0;
        }
        (pos as u128) << shift < self.leaves as u128
    }

    /// Nodes of a subtree whose root is at `level`.
    fn subtree_nodes(&self, level: u32) -> u64 {
        (1u64 << (self.height() - level + 1)) - 1
    }

    /// Label of the subtree rooted at `(level, pos)`, whose root sits at
    /// `array_index`; every active node of the subtree is passed to `cell`.
    fn subtree(
        &self,
        level: u32,
        pos: u64,
        array_index: u64,
        cell: &mut dyn FnMut(u64, Fp),
    ) -> Value256 {
        if !self.is_active(level, pos) {
            return default_hash(&self.auth.defaults, level as usize);
        }
        if level == self.height() {
            let value = bulk_leaf_value(self.seed, pos);
            cell(array_index, decode_value(value));
            return leaf_hash(pos, value);
        }
        let left = self.subtree(level + 1, 2 * pos, array_index + 1, cell);
        let right = self.subtree(
            level + 1,
            2 * pos + 1,
            array_index + 1 + self.subtree_nodes(level + 1),
            cell,
        );
        let label = node_hash(left, right);
        cell(array_index, decode_value(label));
        label
    }

    /// Like [`Self::subtree`] for the levels above `chunk_level`, taking the
    /// labels at `chunk_level` from `chunk_labels`.
    fn upper(
        &self,
        level: u32,
        pos: u64,
        array_index: u64,
        chunk_level: u32,
        chunk_labels: &[Value256],
        cell: &mut dyn FnMut(u64, Fp),
    ) -> Value256 {
        if !self.is_active(level, pos) {
            return default_hash(&self.auth.defaults, level as usize);
        }
        if level == chunk_level {
            return chunk_labels[pos as usize];
        }
        let left = self.upper(level + 1, 2 * pos, array_index + 1, chunk_level, chunk_labels, cell);
        // The right child is reached only when it is active, and then the
        // left subtree is full and its size fits the array.
        let right = if self.is_active(level + 1, 2 * pos + 1) {
            self.upper(
                level + 1,
                2 * pos + 1,
                array_index + 1 + self.subtree_nodes(level + 1),
                chunk_level,
                chunk_labels,
                cell,
            )
        } else {
            default_hash(&self.auth.defaults, level as usize + 1)
        };
        let label = node_hash(left, right);
        cell(array_index, decode_value(label));
        label
    }
}

/// Build the primary columns of claim `registry_index` for the log of
/// `leaves` leaves with values [`bulk_leaf_value`]`(seed, i)`, on `threads`
/// worker threads.
pub fn build_primary_columns(
    params: &CodedArrayParams,
    auth: &AuthParams,
    node_linearization: NodeLinearization,
    leaves: u64,
    seed: u64,
    registry_index: usize,
    threads: usize,
) -> BulkColumns {
    build_primary_columns_in_chunks(
        params,
        auth,
        node_linearization,
        leaves,
        seed,
        registry_index,
        threads,
        CHUNK_BITS,
    )
}

/// [`build_primary_columns`] with worker tasks of `2^chunk_bits` leaves. The
/// result does not depend on `chunk_bits`.
#[allow(clippy::too_many_arguments)]
pub fn build_primary_columns_in_chunks(
    params: &CodedArrayParams,
    auth: &AuthParams,
    node_linearization: NodeLinearization,
    leaves: u64,
    seed: u64,
    registry_index: usize,
    threads: usize,
    chunk_bits: u32,
) -> BulkColumns {
    let log = BulkLog::build(
        params,
        auth,
        node_linearization,
        leaves,
        seed,
        registry_index,
        threads,
        chunk_bits,
    );
    BulkColumns {
        segments: log.segments(),
        root: log.root,
        entries: log.entries,
    }
}

/// The primary columns of one claim for a seeded log that can grow.
///
/// [`BulkLog::extend`] appends leaves: it replaces the contributions of the
/// labels above the worker tasks and of the cells after the last node, and
/// walks only the tasks that hold appended leaves. The result equals a build
/// at the longer length.
pub struct BulkLog<'a> {
    params: &'a CodedArrayParams,
    auth: &'a AuthParams,
    node_linearization: NodeLinearization,
    seed: u64,
    registry_index: usize,
    chunk_bits: u32,
    leaves: u64,
    chunk_labels: Vec<Value256>,
    entries: Vec<Fp>,
    root: Value256,
}

impl<'a> BulkLog<'a> {
    /// Build the columns of a log of `leaves` leaves.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        params: &'a CodedArrayParams,
        auth: &'a AuthParams,
        node_linearization: NodeLinearization,
        leaves: u64,
        seed: u64,
        registry_index: usize,
        threads: usize,
        chunk_bits: u32,
    ) -> Self {
        let height = auth.tree_height as u32;
        assert!(height < 64, "tree too tall");
        let mut log = Self {
            params,
            auth,
            node_linearization,
            seed,
            registry_index,
            chunk_bits: chunk_bits.min(height),
            leaves: 0,
            chunk_labels: Vec::new(),
            entries: Vec::new(),
            root: default_hash(&auth.defaults, 0),
        };
        log.extend(leaves, threads);
        log
    }

    /// Number of leaves of the log.
    pub fn leaves(&self) -> u64 {
        self.leaves
    }

    /// Number of segments of the columns.
    pub fn segments(&self) -> usize {
        self.entries.len() / self.params.need
    }

    /// The primary-column entries, `kappa` per segment.
    pub fn entries(&self) -> &[Fp] {
        &self.entries
    }

    /// Root of the Merkle tree.
    pub fn root(&self) -> Value256 {
        self.root
    }

    fn walk(&self, leaves: u64) -> Walk<'a> {
        Walk {
            auth: self.auth,
            leaves,
            seed: self.seed,
        }
    }

    /// Add (`sign = 1`) or remove (`sign = -1`) the contributions of the
    /// labels above the worker tasks and of the cells after the last node,
    /// for a log of `leaves` leaves. Returns the root.
    fn apply_upper(&mut self, leaves: u64, negate: bool) -> Value256 {
        let height = self.auth.tree_height as u32;
        let chunk_level = height - self.chunk_bits;
        let n_grid = self.params.n_grid as u64;
        let cursor = mapping::cursor(self.node_linearization, height as u16, leaves);
        let segments = cursor.div_ceil(n_grid);
        let walk = self.walk(leaves);
        let mut encoder = ColumnEncoder::new(self.params, self.registry_index);
        let entries = &mut self.entries;
        let chunk_labels = &self.chunk_labels;
        let mut cell = |index: u64, value: Fp| {
            encoder.add_cell(entries, 0, index, if negate { -value } else { value });
        };
        let root = walk.upper(0, 0, 0, chunk_level, chunk_labels, &mut cell);
        for index in cursor..segments * n_grid {
            let value =
                decode_value(default_value_at_index(self.auth, self.node_linearization, index));
            cell(index, value);
        }
        root
    }

    /// Walk the worker tasks `chunks` of a log of `leaves` leaves and add
    /// (or remove) their contributions; when adding, record their labels.
    fn apply_chunks(
        &mut self,
        leaves: u64,
        chunks: std::ops::Range<usize>,
        negate: bool,
        threads: usize,
    ) {
        if chunks.is_empty() {
            return;
        }
        let height = self.auth.tree_height as u32;
        let chunk_level = height - self.chunk_bits;
        let kappa = self.params.need;
        let n_grid = self.params.n_grid as u64;
        let total_segments = self.entries.len() / kappa;
        let walk = self.walk(leaves);
        let params = self.params;
        let registry_index = self.registry_index;
        let next_chunk = AtomicUsize::new(chunks.start);
        let (sender, receiver) = mpsc::channel::<(usize, Value256, usize, Vec<Fp>)>();
        let entries = &mut self.entries;
        let chunk_labels = &mut self.chunk_labels;
        std::thread::scope(|scope| {
            for _ in 0..threads.max(1).min(chunks.len()) {
                let sender = sender.clone();
                let walk = &walk;
                let next_chunk = &next_chunk;
                let chunks = chunks.clone();
                scope.spawn(move || {
                    let mut encoder = ColumnEncoder::new(params, registry_index);
                    loop {
                        let chunk = next_chunk.fetch_add(1, Ordering::Relaxed);
                        if chunk >= chunks.end {
                            break;
                        }
                        let root_index = mapping::preorder_idx(
                            height as u16,
                            chunk_level as u16,
                            chunk as u64,
                        );
                        let last_index = root_index + walk.subtree_nodes(chunk_level) - 1;
                        let first_segment = (root_index / n_grid) as usize;
                        let local_segments = ((last_index / n_grid) as usize)
                            .min(total_segments - 1)
                            - first_segment
                            + 1;
                        let mut local = vec![fp_zero(); local_segments * kappa];
                        let label = walk.subtree(
                            chunk_level,
                            chunk as u64,
                            root_index,
                            &mut |index, value| {
                                encoder.add_cell(&mut local, first_segment, index, value)
                            },
                        );
                        if sender.send((chunk, label, first_segment, local)).is_err() {
                            break;
                        }
                    }
                });
            }
            drop(sender);
            // The two ends of a task may share a segment with its
            // neighbours, so the entries are added here.
            for (chunk, label, first_segment, local) in receiver {
                if !negate {
                    chunk_labels[chunk] = label;
                }
                for (entry, value) in entries[first_segment * kappa..].iter_mut().zip(local) {
                    if negate {
                        *entry -= value;
                    } else {
                        *entry += value;
                    }
                }
            }
        });
    }

    /// Grow the log to `new_leaves` leaves.
    pub fn extend(&mut self, new_leaves: u64, threads: usize) {
        let height = self.auth.tree_height as u32;
        assert!(new_leaves >= self.leaves, "the log shrank");
        assert!(new_leaves > 0, "the log has no filled leaf");
        assert!(new_leaves <= 1u64 << height, "leaves exceed the tree");
        if new_leaves == self.leaves {
            return;
        }
        let kappa = self.params.need;
        let n_grid = self.params.n_grid as u64;
        let chunk_leaves = 1u64 << self.chunk_bits;
        let old_leaves = self.leaves;
        let old_chunks = old_leaves.div_ceil(chunk_leaves) as usize;
        let new_chunks = new_leaves.div_ceil(chunk_leaves) as usize;
        // A task that is partly filled is walked again at the new length.
        let first_changed = (old_leaves / chunk_leaves) as usize;

        if old_leaves > 0 {
            self.apply_upper(old_leaves, true);
            self.apply_chunks(old_leaves, first_changed..old_chunks, true, threads);
        }

        let cursor = mapping::cursor(self.node_linearization, height as u16, new_leaves);
        let segments = cursor.div_ceil(n_grid) as usize;
        self.entries.resize(segments * kappa, fp_zero());
        self.chunk_labels.resize(new_chunks, [0u8; 32]);
        self.leaves = new_leaves;
        self.apply_chunks(new_leaves, first_changed..new_chunks, false, threads);
        self.root = self.apply_upper(new_leaves, false);
    }
}

/// The PIR cells of a server that stores the whole tree of the seeded log,
/// for a [`PrivateLayout::whole_tree`] layout, and the root of the tree.
///
/// Every cell of the maintained array is one entry of its record: a leaf
/// holds its value and an inner node its label. The cells after the last
/// node of the last record hold the default content of their positions.
pub fn build_tree_cells(
    auth: &AuthParams,
    node_linearization: NodeLinearization,
    seed: u64,
    layout: &PrivateLayout,
    threads: usize,
) -> (Vec<u16>, Value256) {
    let leaves = layout.leaves();
    let height = auth.tree_height as u32;
    assert!(height < 64, "tree too tall");
    let record_cells = PirShape::of(layout).record_cells;
    let cells_per_record = layout.cells_per_record() as u64;
    let records = layout.records();
    let mut cells = vec![0u16; records * record_cells];
    let walk = Walk { auth, leaves, seed };
    let chunk_bits = CHUNK_BITS.min(height);
    let chunk_level = height - chunk_bits;
    let chunks = leaves.div_ceil(1u64 << chunk_bits) as usize;
    let mut chunk_labels = vec![[0u8; 32]; chunks];

    let write = |cells: &mut Vec<u16>, index: u64, value: Fp| {
        let record = (index / cells_per_record) as usize;
        or_entry_into_cells(
            &mut cells[record * record_cells..(record + 1) * record_cells],
            (index % cells_per_record) as usize,
            value,
        );
    };

    let next_chunk = AtomicUsize::new(0);
    // A bounded channel keeps the workers from running ahead of the writer.
    let (sender, receiver) = mpsc::sync_channel::<(usize, Value256, Vec<(u64, Fp)>)>(threads.max(1));
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1).min(chunks) {
            let sender = sender.clone();
            let walk = &walk;
            let next_chunk = &next_chunk;
            scope.spawn(move || {
                loop {
                    let chunk = next_chunk.fetch_add(1, Ordering::Relaxed);
                    if chunk >= chunks {
                        break;
                    }
                    let root_index =
                        mapping::preorder_idx(height as u16, chunk_level as u16, chunk as u64);
                    let mut local = Vec::with_capacity(walk.subtree_nodes(chunk_level) as usize);
                    let label = walk.subtree(chunk_level, chunk as u64, root_index, &mut |index, value| {
                        local.push((index, value))
                    });
                    if sender.send((chunk, label, local)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(sender);
        for (chunk, label, local) in receiver {
            chunk_labels[chunk] = label;
            for (index, value) in local {
                write(&mut cells, index, value);
            }
        }
    });

    let mut upper_cells = Vec::new();
    let root = walk.upper(0, 0, 0, chunk_level, &chunk_labels, &mut |index, value| {
        upper_cells.push((index, value))
    });
    for (index, value) in upper_cells {
        write(&mut cells, index, value);
    }
    let cursor = mapping::cursor(node_linearization, height as u16, leaves);
    for index in cursor..records as u64 * cells_per_record {
        let value = decode_value(default_value_at_index(auth, node_linearization, index));
        write(&mut cells, index, value);
    }
    (cells, root)
}
