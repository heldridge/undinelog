//! Serving reads at a threshold above the stored one.
//!
//! A store at threshold `kappa` answers a read at threshold `2^j kappa` by
//! stitching the fragments it holds while it serves the request. Its state
//! does not change. `j` stitches group `4^j` source segments into one
//! stitched segment, and unrolling the stitch gives each stitched entry as a
//! combination of `2^j` source entries:
//!
//! ```text
//! L_T[w] = sum_m E_m(C) * L_{(m, v)}[w >> j]
//! R_T[z] = sum_m F_m(w) * R_{(u, m)}[z >> j]
//! ```
//!
//! For a column entry the low bits of `w` fix the W-path `v` of the source
//! segments and `m` ranges over their `2^j` Z-paths; `E_m(C)` is the product
//! of the `j` Z-axis selectors along path `m`. Row entries are symmetric. A
//! point share is therefore `2^j` multiplications per claim, and a whole
//! stitched column costs about `2 * 4^j kappa`.
//!
//! Source segments beyond the allocated ones hold the array's default values.
//! The caller supplies those values, so every store answers with the
//! fragment of the same polynomial whatever its threshold and allocation.

use std::borrow::Cow;
use std::collections::HashMap;

use super::{
    ClaimBasisCache, Index, UnauthOptimizedReadRequest, UnauthOptimizedReadResponse,
    UnauthReadObjectContribution, UnauthReadObjectId, UnauthStore, collect_local_shard_entries,
    logical_offset_to_grid_index, public_values_per_segment,
};
use crate::coded_array::axis::{axis_scale_from_index, axis_selectors, segment_axis_indices};
use crate::coded_array::params::{CodedArrayParams, ShardRegEntry};
use crate::coded_array::segment::{encode_primary_column, stitch_primary_column};
use crate::coded_array::store::RowKey;
use crate::field::{Fp, fp_one, fp_zero};
use crate::shares::{RowShare, SegmentPrimaryShard, ShardShare};

/// Default field value of a public index beyond a store's allocated segments.
pub type DefaultCellFn<'a> = &'a dyn Fn(Index) -> Fp;

/// Selector products of one axis for the `2^levels` paths under a stitched
/// segment.
///
/// Entry `m` is the product over the `levels` stitches of the selector of
/// the half that path `m` takes: bit `p` of `m` is the half taken at the
/// stitch from level `p` to level `p + 1`. `scale_index` is the stitched
/// segment's scale index on the axis and `x` the claim's assignment on it.
///
/// # Complexity
/// `2^levels - 1` selector evaluations.
fn stitch_weights(params: &CodedArrayParams, levels: usize, scale_index: usize, x: Fp) -> Vec<Fp> {
    let mut nodes = vec![(scale_index, fp_one())];
    for level in (1..=levels).rev() {
        // The scale ladder of level `level` is the stored ladder from entry
        // `level` on.
        let inv_ladder = &params.inv_scale_ladder_roots[level..];
        let child_need = params.need << (level - 1);
        let mut children = Vec::with_capacity(2 * nodes.len());
        for (scale, weight) in nodes {
            let inv_rho = axis_scale_from_index(inv_ladder, scale);
            let (s0, s1) = axis_selectors(child_need, inv_rho, x);
            children.push((scale << 1, weight * s0));
            children.push(((scale << 1) | 1, weight * s1));
        }
        nodes = children;
    }
    nodes.into_iter().map(|(_, weight)| weight).collect()
}

/// Source segment under stitched segment `target` with the given paths:
/// base-four digit `p` of the result is `2 u_p + v_p`, where `u_p` is bit `p`
/// of `z_path` and `v_p` is bit `p` of `w_path`.
fn source_segment(target: usize, levels: usize, z_path: usize, w_path: usize) -> usize {
    let mut seg = target;
    for p in (0..levels).rev() {
        seg = (seg << 2) | (((z_path >> p) & 1) << 1) | ((w_path >> p) & 1);
    }
    seg
}

/// Reverse the low `levels` bits of `bits`.
fn reverse_path_bits(bits: usize, levels: usize) -> usize {
    (0..levels).fold(0, |path, q| path | (((bits >> q) & 1) << (levels - 1 - q)))
}

/// Source fragments of one store, including the default source segments
/// beyond its allocated ones.
struct SourceFragments<'a> {
    store: &'a UnauthStore,
    default_cell: Option<DefaultCellFn<'a>>,
    /// Cells of default source segments in grid order.
    default_cells: HashMap<usize, Vec<Fp>>,
    /// Lagrange bases used to encode default source segments.
    bases: ClaimBasisCache,
}

impl<'a> SourceFragments<'a> {
    /// Cells of default source segment `seg` in grid order.
    fn default_segment_cells(&mut self, default_cell: DefaultCellFn<'a>, seg: usize) -> &[Fp] {
        let params = &self.store.params;
        self.default_cells.entry(seg).or_insert_with(|| {
            let span = public_values_per_segment(params);
            let base = seg as u64 * span as u64;
            let mut cells = vec![fp_zero(); params.n_grid];
            for offset in 0..span {
                cells[logical_offset_to_grid_index(params, offset)] =
                    default_cell(base + offset as u64);
            }
            cells
        })
    }

    /// Entry `w_index` of the claim's primary column for source segment
    /// `seg`, or `None` when the claim's column is missing.
    fn column_entry(
        &mut self,
        registry_index: usize,
        entry: &ShardRegEntry,
        seg: usize,
        w_index: usize,
    ) -> Option<Fp> {
        let store = self.store;
        let params = &store.params;
        if seg < store.segments {
            return store
                .coded
                .shard_segment(params, &entry.key, seg)
                .map(|column| column[w_index]);
        }
        let Some(default_cell) = self.default_cell else {
            return Some(fp_zero());
        };
        // One default value per cell of the column's W-index.
        let base = seg as u64 * public_values_per_segment(params) as u64;
        let basis = self.bases.z_basis(params, registry_index, seg);
        let mut acc = fp_zero();
        for (z_index, weight) in basis.iter().enumerate() {
            let grid_index = params.grid_index_for_axes(z_index, w_index);
            let offset = params.logical_offset_for_grid_index(grid_index);
            acc += default_cell(base + offset as u64) * *weight;
        }
        Some(acc)
    }

    /// The claim's primary column for source segment `seg`: `Ok(None)` for a
    /// zero default segment, `Err(())` when the claim's column is missing.
    fn column(
        &mut self,
        registry_index: usize,
        entry: &ShardRegEntry,
        seg: usize,
    ) -> Result<Option<Cow<'a, [Fp]>>, ()> {
        let store = self.store;
        let params = &store.params;
        if seg < store.segments {
            return store
                .coded
                .shard_segment(params, &entry.key, seg)
                .map(|column| Some(Cow::Borrowed(column)))
                .ok_or(());
        }
        let Some(default_cell) = self.default_cell else {
            return Ok(None);
        };
        let basis = self.bases.z_basis(params, registry_index, seg).to_vec();
        let cells = self.default_segment_cells(default_cell, seg);
        let mut column = vec![fp_zero(); params.need];
        encode_primary_column(cells, &basis, &mut column);
        Ok(Some(Cow::Owned(column)))
    }

    /// Entry `z_index` of one auxiliary row for source segment `seg`, or
    /// `None` when the row is missing.
    fn row_entry(
        &mut self,
        row_key: &RowKey,
        row_label_index: usize,
        seg: usize,
        z_index: usize,
    ) -> Option<Fp> {
        let store = self.store;
        let params = &store.params;
        if seg < store.segments {
            return store
                .coded
                .row_segment(params, row_key, seg)
                .map(|row| row[z_index]);
        }
        let Some(default_cell) = self.default_cell else {
            return Some(fp_zero());
        };
        // One default value per cell of the row's Z-index.
        let base = seg as u64 * public_values_per_segment(params) as u64;
        let basis = self.bases.w_basis(params, row_label_index, seg);
        let mut acc = fp_zero();
        for (w_index, weight) in basis.iter().enumerate() {
            let grid_index = params.grid_index_for_axes(z_index, w_index);
            let offset = params.logical_offset_for_grid_index(grid_index);
            acc += default_cell(base + offset as u64) * *weight;
        }
        Some(acc)
    }
}

/// One store viewed `levels` stitches above its stored threshold, for the
/// objects of one request.
pub(crate) struct StitchedView<'a> {
    levels: usize,
    sources: SourceFragments<'a>,
    /// `E_m(C)` keyed by `(registry_index, z_scale_index)` of the stitched
    /// segment.
    column_weights: HashMap<(usize, usize), Vec<Fp>>,
    /// `F_m(w)` keyed by `(row_label_index, w_scale_index)` of the stitched
    /// segment.
    row_weights: HashMap<(usize, usize), Vec<Fp>>,
}

impl<'a> StitchedView<'a> {
    pub(crate) fn new(
        store: &'a UnauthStore,
        levels: usize,
        default_cell: Option<DefaultCellFn<'a>>,
    ) -> Self {
        Self {
            levels,
            sources: SourceFragments {
                store,
                default_cell,
                default_cells: HashMap::new(),
                bases: ClaimBasisCache::default(),
            },
            column_weights: HashMap::new(),
            row_weights: HashMap::new(),
        }
    }

    /// First source segment of stitched segment `target`, when the stitched
    /// segment exists: it does once its first source segment is allocated.
    fn first_source_segment(&self, target: usize) -> Option<usize> {
        let shift = 2 * self.levels;
        (target <= usize::MAX >> shift)
            .then(|| target << shift)
            .filter(|&first| first < self.sources.store.segments)
    }

    /// Stitched point shares of every local claim for one public index.
    pub(crate) fn point_shares(&mut self, idx: Index) -> Vec<ShardShare> {
        let store = self.sources.store;
        let params = &store.params;
        let levels = self.levels;
        let span = public_values_per_segment(params) as u64;
        let Ok(seg) = usize::try_from(idx / span) else {
            return Vec::new();
        };
        let target = seg >> (2 * levels);
        let Some(first) = self.first_source_segment(target) else {
            return Vec::new();
        };
        let grid_index = logical_offset_to_grid_index(params, (idx % span) as usize);
        let w_index = params.w_index_for_grid_index(grid_index);
        let (z_scale_index, w_scale_index) = segment_axis_indices(seg);
        let w_path = w_scale_index & ((1 << levels) - 1);
        let target_z_scale = z_scale_index >> levels;

        let claims = collect_local_shard_entries(params, &store.coded, Some(first));
        let mut shares = Vec::with_capacity(claims.len());
        for (registry_index, entry) in claims {
            let weights = self
                .column_weights
                .entry((registry_index, target_z_scale))
                .or_insert_with(|| stitch_weights(params, levels, target_z_scale, entry.z));
            let mut value = Some(fp_zero());
            for (z_path, weight) in weights.iter().enumerate() {
                let source = source_segment(target, levels, z_path, w_path);
                let source_entry =
                    self.sources
                        .column_entry(registry_index, &entry, source, w_index);
                value = value
                    .zip(source_entry)
                    .map(|(acc, source_entry)| acc + *weight * source_entry);
            }
            if value.is_some() {
                shares.push(ShardShare {
                    registry_index,
                    value,
                });
            }
        }
        shares
    }

    /// Stitched primary columns of every local claim for one stitched
    /// segment.
    pub(crate) fn segment_columns(&mut self, target: usize) -> Vec<SegmentPrimaryShard> {
        let Some(first) = self.first_source_segment(target) else {
            return Vec::new();
        };
        let store = self.sources.store;
        let params = &store.params;
        let claims = collect_local_shard_entries(params, &store.coded, Some(first));
        let mut shards = Vec::with_capacity(claims.len());
        for (registry_index, entry) in claims {
            let fragments = (0..(1usize << (2 * self.levels)))
                .map(|position| {
                    self.sources
                        .column(registry_index, &entry, first + position)
                })
                .collect::<Result<Vec<_>, ()>>();
            let Ok(fragments) = fragments else {
                continue;
            };
            shards.push(SegmentPrimaryShard {
                registry_index,
                values: stitch_columns(params, self.levels, target, fragments, entry.z),
            });
        }
        shards
    }

    /// Stitched row entries of every local claim for one aligned block of
    /// the stitched layout.
    pub(crate) fn block_rows(&mut self, block: Index) -> Vec<RowShare> {
        let store = self.sources.store;
        let params = &store.params;
        let levels = self.levels;
        let target_need = (params.need << levels) as u64;
        let Ok(target) = usize::try_from(block / target_need) else {
            return Vec::new();
        };
        let Some(first) = self.first_source_segment(target) else {
            return Vec::new();
        };
        // Entry `2i + u` of a stitched row comes from the rows of half `u`,
        // so the low bits of the stitched Z-index give the Z-path from the
        // top stitch down.
        let z_index = (block % target_need) as usize;
        let z_path = reverse_path_bits(z_index, levels);
        let source_z_index = z_index >> levels;
        let target_w_scale = segment_axis_indices(target).1;

        let mut rows = store
            .coded
            .row_segments_for_segment(params, first)
            .into_iter()
            .filter_map(|(row_key, _)| {
                let row_label_index = params.row_label_index_for_bytes(row_key.row_label_bytes)?;
                Some((*row_key, row_label_index))
            })
            .collect::<Vec<_>>();
        rows.sort_by_key(|(row_key, _)| row_key.row_label_bytes);
        rows.dedup_by_key(|(row_key, _)| row_key.row_label_bytes);

        let mut shares = Vec::with_capacity(rows.len());
        for (row_key, row_label_index) in rows {
            let row_label = params.row_labels[row_label_index];
            let weights = self
                .row_weights
                .entry((row_label_index, target_w_scale))
                .or_insert_with(|| stitch_weights(params, levels, target_w_scale, row_label));
            let mut value = Some(fp_zero());
            for (w_path, weight) in weights.iter().enumerate() {
                let source = source_segment(target, levels, z_path, w_path);
                let source_entry =
                    self.sources
                        .row_entry(&row_key, row_label_index, source, source_z_index);
                value = value
                    .zip(source_entry)
                    .map(|(acc, source_entry)| acc + *weight * source_entry);
            }
            if let Some(value) = value {
                shares.push(RowShare { row_label, value });
            }
        }
        shares
    }
}

/// Stitch the `4^levels` source columns of one claim under stitched segment
/// `target` into its stitched column, one level at a time.
///
/// `fragments[c]` is the column of source segment `4^levels * target + c`, or
/// `None` when it is zero. `x` is the claim's column identifier.
fn stitch_columns(
    params: &CodedArrayParams,
    levels: usize,
    target: usize,
    fragments: Vec<Option<Cow<'_, [Fp]>>>,
    x: Fp,
) -> Vec<Fp> {
    let mut current = fragments;
    for level in 1..=levels {
        let inv_ladder = &params.inv_scale_ladder_roots[level..];
        let child_need = params.need << (level - 1);
        let first_parent = target << (2 * (levels - level));
        // A selector depends on the parent only through its Z-axis scale.
        let mut selectors_by_scale: HashMap<usize, (Fp, Fp)> = HashMap::new();
        let mut parents = Vec::with_capacity(current.len() / 4);
        for (position, children) in current.chunks_exact(4).enumerate() {
            if children.iter().all(Option::is_none) {
                parents.push(None);
                continue;
            }
            let (z_scale_index, _) = segment_axis_indices(first_parent + position);
            let selectors = *selectors_by_scale.entry(z_scale_index).or_insert_with(|| {
                axis_selectors(
                    child_need,
                    axis_scale_from_index(inv_ladder, z_scale_index),
                    x,
                )
            });
            let mut column = vec![fp_zero(); 2 * child_need];
            stitch_primary_column(
                selectors,
                [
                    children[0].as_deref(),
                    children[1].as_deref(),
                    children[2].as_deref(),
                    children[3].as_deref(),
                ],
                &mut column,
            );
            parents.push(Some(Cow::Owned(column)));
        }
        current = parents;
    }
    match current.pop().flatten() {
        Some(column) => column.into_owned(),
        None => vec![fp_zero(); params.need << levels],
    }
}

impl UnauthStore {
    /// Number of stitches from the stored threshold up to `kappa`, when this
    /// store serves reads at `kappa`.
    ///
    /// A store serves its own threshold and every threshold `2^j kappa` above
    /// it up to the registry size, the largest threshold a read can meet.
    pub fn stitch_levels_to(&self, kappa: usize) -> Option<usize> {
        let need = self.params.need;
        if kappa < need || !kappa.is_power_of_two() {
            return None;
        }
        let levels = (kappa / need).trailing_zeros() as usize;
        if levels == 0 {
            return Some(0);
        }
        (kappa <= self.params.shard_registry.len()
            && levels <= self.params.inv_scale_ladder_roots.len())
        .then_some(levels)
    }

    /// Point shares for each index at `levels` stitches above the stored
    /// threshold.
    pub(crate) fn handle_multi_read_stitched(
        &self,
        idxs: &[Index],
        levels: usize,
        default_cell: Option<DefaultCellFn<'_>>,
    ) -> Vec<Vec<ShardShare>> {
        let mut view = StitchedView::new(self, levels, default_cell);
        idxs.iter().map(|&idx| view.point_shares(idx)).collect()
    }
}

/// Serve the ordered objects of an optimized read request at the threshold it
/// names, stitching on the fly when that threshold is above the stored one.
///
/// `default_cell` gives the values of source segments beyond the allocated
/// ones; `None` means zero.
pub fn handle_optimized_read_request_with_defaults(
    store: &UnauthStore,
    request: &UnauthOptimizedReadRequest,
    default_cell: Option<DefaultCellFn<'_>>,
) -> UnauthOptimizedReadResponse {
    let Some(levels) = store.stitch_levels_to(request.kappa) else {
        return UnauthOptimizedReadResponse::default();
    };
    let mut view = StitchedView::new(store, levels, default_cell);
    let mut objects = Vec::with_capacity(request.objects.len());
    for (object_pos, object) in request.objects.iter().enumerate() {
        objects.push(match *object {
            UnauthReadObjectId::Segment { segment } => UnauthReadObjectContribution::Segment {
                object_pos,
                shards: if levels == 0 {
                    store.handle_read_segment_raw(segment)
                } else {
                    view.segment_columns(segment)
                },
            },
            UnauthReadObjectId::Block { block } => UnauthReadObjectContribution::Block {
                object_pos,
                rows: if levels == 0 {
                    store.handle_read_block(block)
                } else {
                    view.block_rows(block)
                },
            },
            UnauthReadObjectId::Point { array_index } => UnauthReadObjectContribution::Point {
                object_pos,
                shares: if levels == 0 {
                    store.handle_read(array_index)
                } else {
                    view.point_shares(array_index)
                },
            },
        });
    }
    UnauthOptimizedReadResponse {
        helper_index: None,
        objects,
    }
}
