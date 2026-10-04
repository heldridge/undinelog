//! Blocks, patterns, and records of the private-opening database.
//!
//! # Blocks
//! A maintainer's database is its primary columns. A *block* is the primary
//! columns of `segments_per_record` consecutive segments of one claim:
//! `segments_per_record * kappa` stored entries. The cell at array index `p`
//! is read from the entry `(p / segment_cells, w_index(p % segment_cells))`,
//! exactly as in a public point read.
//!
//! # Patterns and records
//! The leaves are cut into groups of `2^group_bits` consecutive leaves, and a
//! *pattern* is one group. A group is a subtree, and its nodes occupy one
//! contiguous range of the preorder array that is shorter than a block, so it
//! lies in at most two consecutive blocks. The opening of a leaf of the group
//! consists of a lower part inside that subtree, whose cells depend on the
//! leaf, and of one sibling at each level above the subtree, which depends
//! on the group alone. The *record* of a pattern lists, in fixed *slots*:
//! - slots 0 and 1: the blocks that hold the group's subtree, each in the
//!   slot of its parity;
//! - slot `2 + j`: the entry of the sibling of the group's ancestor `j`
//!   levels above the subtree root, for the levels inside the filled part of
//!   the tree. Such a sibling is an *upper node*, named by its level and its
//!   position within the level.
//!
//! A slot is empty when its sibling is a default node or when the subtree
//! fits in one block. The same record serves every leaf of the group, so a
//! request names only the pattern.

use crate::auth::AuthParams;
use crate::auth::mapping::{self, NodeLinearization};
use crate::auth::merkle::{NodeKey, leaf_key, parent, sibling};
use crate::unauth::CodedArrayParams;

/// Why a layout cannot be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutError {
    /// `segments_per_record` must be a power of two.
    SegmentsPerRecord(usize),
    /// The log has no filled leaf.
    EmptyLog,
    /// The log holds more leaves than the tree has.
    TooManyLeaves { leaves: u64, tree_height: usize },
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayoutError::SegmentsPerRecord(value) => {
                write!(f, "segments per record must be a power of two, got {value}")
            }
            LayoutError::EmptyLog => write!(f, "the log has no filled leaf"),
            LayoutError::TooManyLeaves { leaves, tree_height } => {
                write!(f, "{leaves} leaves exceed a tree of height {tree_height}")
            }
        }
    }
}

impl std::error::Error for LayoutError {}

/// One stored entry needed by an opening, located in the pattern's row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannedEntry {
    /// The Merkle node whose cell this entry decodes.
    pub node: NodeKey,
    /// The node's index in the maintained array.
    pub array_index: u64,
    /// Slot of the pattern's row that holds the entry's record.
    pub slot: usize,
    /// Index of the entry within that record.
    pub entry_in_record: usize,
}

/// The database layout of private openings for one log length.
///
/// It depends only on public values: the tree height, the number of filled
/// leaves, the threshold, and the record size. A maintainer and a client that
/// agree on these derive the same layout.
#[derive(Clone, Debug)]
pub struct PrivateLayout {
    tree_height: usize,
    node_linearization: NodeLinearization,
    leaves: u64,
    /// Stored entries per segment: `kappa` for a claim's primary columns.
    kappa: usize,
    segment_cells: usize,
    segments_per_record: usize,
    group_bits: u32,
    upper_slots: usize,
    records: usize,
    patterns: usize,
    /// W-axis index of each logical offset of a segment.
    w_index_of_offset: Vec<u32>,
}

/// Index of the entry whose symbols fill a slot above the group's subtree.
pub const UPPER_ENTRY: usize = 0;

/// Height of the active prefix at which the public grouping starts to
/// grow: a block is one segment up to `2^GROUPING_BASE_BITS` leaves and
/// doubles with every two further levels, so it grows with the square root
/// of the log.
pub const GROUPING_BASE_BITS: u32 = 25;

/// The public grouping: segments per block for a log of `leaves` leaves.
/// Every client and every maintainer derive it from the log length in the
/// anchor, so one query serves all maintainers of a registry.
pub fn segments_per_block(leaves: u64) -> usize {
    let height = if leaves <= 1 { 0 } else { (leaves - 1).ilog2() + 1 };
    1usize << (height.saturating_sub(GROUPING_BASE_BITS) / 2)
}

impl PrivateLayout {
    /// Layout for a log of `leaves` filled leaves at the threshold of
    /// `params`, with the public grouping [`segments_per_block`].
    pub fn for_log(
        params: &CodedArrayParams,
        auth: &AuthParams,
        node_linearization: NodeLinearization,
        leaves: u64,
    ) -> Result<Self, LayoutError> {
        Self::new(params, auth, node_linearization, leaves, segments_per_block(leaves))
    }

    /// Layout for a log of `leaves` filled leaves at the threshold of
    /// `params`, with blocks of `segments_per_record` segments.
    pub fn new(
        params: &CodedArrayParams,
        auth: &AuthParams,
        node_linearization: NodeLinearization,
        leaves: u64,
        segments_per_record: usize,
    ) -> Result<Self, LayoutError> {
        if segments_per_record == 0 || !segments_per_record.is_power_of_two() {
            return Err(LayoutError::SegmentsPerRecord(segments_per_record));
        }
        if leaves == 0 {
            return Err(LayoutError::EmptyLog);
        }
        let tree_height = auth.tree_height;
        if tree_height < 64 && leaves > (1u64 << tree_height) {
            return Err(LayoutError::TooManyLeaves { leaves, tree_height });
        }
        let segment_cells = params.n_grid;
        let record_cells = segments_per_record * segment_cells;
        // Largest subtree with at most `record_cells` nodes: `2^(g+1) - 1`.
        let group_bits = ((record_cells + 1).ilog2() - 1).min(tree_height as u32);
        // Smallest `a` with `leaves <= 2^a`: levels above it hold default
        // siblings for every leaf.
        let active_bits = if leaves <= 1 { 0 } else { (leaves - 1).ilog2() + 1 };
        let upper_slots = active_bits.saturating_sub(group_bits) as usize;
        let cells = mapping::cursor(node_linearization, tree_height as u16, leaves);
        let segments = cells.div_ceil(segment_cells as u64) as usize;
        let records = segments.div_ceil(segments_per_record);
        let patterns = leaves.div_ceil(1u64 << group_bits) as usize;
        let w_index_of_offset = (0..segment_cells)
            .map(|offset| params.w_index_for_grid_index(params.logical_grid_index(offset)) as u32)
            .collect();
        Ok(Self {
            tree_height,
            node_linearization,
            leaves,
            kappa: params.s,
            segment_cells,
            segments_per_record,
            group_bits,
            upper_slots,
            records,
            patterns,
            w_index_of_offset,
        })
    }

    /// Layout for a server that stores the whole tree, with no coding: a
    /// record is `cells_per_record` consecutive cells of the maintained
    /// array, and every cell is one entry. The patterns, rows, and slots are
    /// those of [`Self::new`] for records of that many cells.
    pub fn whole_tree(
        auth: &AuthParams,
        node_linearization: NodeLinearization,
        leaves: u64,
        cells_per_record: usize,
    ) -> Result<Self, LayoutError> {
        if cells_per_record == 0 || !cells_per_record.is_power_of_two() {
            return Err(LayoutError::SegmentsPerRecord(cells_per_record));
        }
        if leaves == 0 {
            return Err(LayoutError::EmptyLog);
        }
        let tree_height = auth.tree_height;
        if tree_height < 64 && leaves > (1u64 << tree_height) {
            return Err(LayoutError::TooManyLeaves { leaves, tree_height });
        }
        let group_bits = ((cells_per_record + 1).ilog2() - 1).min(tree_height as u32);
        let active_bits = if leaves <= 1 { 0 } else { (leaves - 1).ilog2() + 1 };
        let cells = mapping::cursor(node_linearization, tree_height as u16, leaves);
        let upper_slots = active_bits.saturating_sub(group_bits) as usize;
        let patterns = leaves.div_ceil(1u64 << group_bits) as usize;
        Ok(Self {
            tree_height,
            node_linearization,
            leaves,
            kappa: cells_per_record,
            segment_cells: cells_per_record,
            segments_per_record: 1,
            group_bits,
            upper_slots,
            records: cells.div_ceil(cells_per_record as u64) as usize,
            patterns,
            w_index_of_offset: (0..cells_per_record as u32).collect(),
        })
    }

    /// Number of filled leaves.
    pub fn leaves(&self) -> u64 {
        self.leaves
    }

    /// Threshold of the fragments this layout reads.
    pub fn kappa(&self) -> usize {
        self.kappa
    }

    /// Segments per record.
    pub fn segments_per_record(&self) -> usize {
        self.segments_per_record
    }

    /// Stored entries per record.
    pub fn entries_per_record(&self) -> usize {
        self.segments_per_record * self.kappa
    }

    /// Cells of the maintained array per record.
    pub fn cells_per_record(&self) -> usize {
        self.segments_per_record * self.segment_cells
    }

    /// Leaves per pattern, as a power of two.
    pub fn group_bits(&self) -> u32 {
        self.group_bits
    }

    /// Number of records of one claim.
    pub fn records(&self) -> usize {
        self.records
    }

    /// Number of patterns.
    pub fn patterns(&self) -> usize {
        self.patterns
    }

    /// Slots per record: two blocks and one entry per level above the
    /// group's subtree.
    pub fn slots(&self) -> usize {
        2 + self.upper_slots
    }

    /// Levels above the group's subtree with stored siblings.
    pub fn upper_slots(&self) -> usize {
        self.upper_slots
    }

    /// Stored nodes `level` levels above the group roots, numbered by
    /// position from the left. The number only grows when the log grows.
    pub fn upper_count(&self, level: usize) -> usize {
        assert!(level < self.upper_slots, "level above the stored siblings");
        ((self.patterns - 1) >> level) + 1
    }

    /// The Merkle node `level` levels above the group roots at position
    /// `pos`.
    pub fn upper_node(&self, level: usize, pos: usize) -> NodeKey {
        assert!(pos < self.upper_count(level), "upper node out of range");
        NodeKey {
            level: self.tree_height as u32 - self.group_bits - level as u32,
            pos: pos as u64,
        }
    }

    /// The patterns whose opening reads an upper node: the groups below
    /// its sibling, a contiguous range.
    pub fn upper_patterns(&self, level: usize, pos: usize) -> std::ops::Range<usize> {
        let start = ((pos ^ 1) << level).min(self.patterns);
        let end = (((pos ^ 1) + 1) << level).min(self.patterns);
        start..end
    }

    /// Array index of an upper node.
    pub fn upper_array_index(&self, level: usize, pos: usize) -> u64 {
        self.array_index(self.upper_node(level, pos))
            .expect("an upper node is active")
    }

    /// Index, in the entries of the claim, of the entry that reads an upper
    /// node: the block times the entries per block, plus the entry within
    /// the block.
    pub fn upper_entry_index(&self, level: usize, pos: usize) -> usize {
        self.entry_index(self.upper_array_index(level, pos))
    }

    /// Pattern of a leaf.
    pub fn pattern_of(&self, leaf: u64) -> usize {
        assert!(leaf < self.leaves, "leaf outside the filled prefix");
        (leaf >> self.group_bits) as usize
    }

    /// Root of the subtree of one pattern.
    fn group_root(&self, pattern: usize) -> NodeKey {
        NodeKey {
            level: self.tree_height as u32 - self.group_bits,
            pos: pattern as u64,
        }
    }

    fn array_index(&self, node: NodeKey) -> Option<u64> {
        mapping::idx_if_active(
            self.node_linearization,
            self.tree_height as u16,
            self.leaves,
            node.level as u16,
            node.pos,
        )
    }

    fn record_of_index(&self, array_index: u64) -> usize {
        (array_index / self.cells_per_record() as u64) as usize
    }

    /// The row of a pattern: the record of each slot, or `None` for an empty
    /// slot.
    pub fn row(&self, pattern: usize) -> Vec<Option<usize>> {
        assert!(pattern < self.patterns, "pattern out of range");
        let mut row = vec![None; self.slots()];
        let root = self.group_root(pattern);
        // The subtree root is active because the group holds a filled leaf.
        let first_index = self
            .array_index(root)
            .expect("the subtree root of a pattern is active");
        let first_record = self.record_of_index(first_index);
        row[first_record % 2] = Some(first_record);
        let subtree_cells = (1u64 << (self.group_bits + 1)) - 1;
        let last_record = self.record_of_index(first_index + subtree_cells - 1);
        if last_record != first_record && last_record < self.records {
            row[last_record % 2] = Some(last_record);
        }
        for level in 0..self.upper_slots {
            let pos = (pattern >> level) ^ 1;
            if pos < self.upper_count(level) {
                row[2 + level] = Some(pos);
            }
        }
        row
    }

    /// The records of all patterns as one table of `patterns x slots` ids,
    /// with `u32::MAX` for an empty slot: block numbers in slots 0 and 1,
    /// positions of upper nodes within their level in the slots above. It equals [`Self::row`] for every
    /// pattern and is computed level by level.
    pub fn rows_table(&self) -> Vec<u32> {
        const NO_RECORD: u32 = u32::MAX;
        let slots = self.slots();
        let mut table = vec![NO_RECORD; self.patterns * slots];
        let root_level = self.tree_height as u32 - self.group_bits;
        let subtree_cells = (1u64 << (self.group_bits + 1)) - 1;
        // Array index of every group root: the nodes of one level, from the
        // indices of their parents.
        let top_level = root_level - self.upper_slots as u32;
        // Above `top_level` the filled part is the left spine: node
        // `(level, 0)` sits at index `level`.
        let mut indices: Vec<u64> = vec![top_level as u64];
        for level in top_level + 1..=root_level {
            let left_nodes = (1u64 << (self.tree_height as u32 - level + 1)) - 1;
            let width = ((self.patterns as u64 - 1) >> (root_level - level)) as usize + 1;
            let mut next = Vec::with_capacity(width);
            for pos in 0..width {
                let parent = indices[pos / 2];
                next.push(if pos % 2 == 0 {
                    parent + 1
                } else {
                    parent + 1 + left_nodes
                });
            }
            indices = next;
        }
        for pattern in 0..self.patterns {
            let row = &mut table[pattern * slots..(pattern + 1) * slots];
            let first_index = indices[pattern];
            let first_record = self.record_of_index(first_index);
            row[first_record % 2] = first_record as u32;
            let last_record = self.record_of_index(first_index + subtree_cells - 1);
            if last_record != first_record && last_record < self.records {
                row[last_record % 2] = last_record as u32;
            }
            for level in 0..self.upper_slots {
                let pos = (pattern >> level) ^ 1;
                if pos < self.upper_count(level) {
                    row[2 + level] = pos as u32;
                }
            }
        }
        table
    }

    /// The records whose cells differ between a log of `old_leaves` leaves
    /// and this longer log: the records from the old end of the array on,
    /// and those that hold a label on the path of the first appended leaf.
    pub fn records_changed_by_append(&self, old_leaves: u64) -> Vec<usize> {
        assert!(old_leaves <= self.leaves, "the log shrank");
        if old_leaves == self.leaves {
            return Vec::new();
        }
        let old_cursor = mapping::cursor(
            self.node_linearization,
            self.tree_height as u16,
            old_leaves,
        );
        let mut changed = (self.record_of_index(old_cursor)..self.records).collect::<Vec<_>>();
        let first_new = self.record_of_index(old_cursor);
        let mut key = leaf_key(self.tree_height as u32, old_leaves);
        loop {
            if let Some(index) = self.array_index(key) {
                let record = self.record_of_index(index);
                if record < first_new && !changed.contains(&record) {
                    changed.push(record);
                }
            }
            if key.level == 0 {
                break;
            }
            key = parent(key);
        }
        changed.sort_unstable();
        changed
    }

    /// The stored entries that the opening of `leaf` reads, located in the
    /// record of the leaf's pattern.
    ///
    /// The nodes are the leaf and the sibling at every level, in the order of
    /// a public opening, without the default ones.
    pub fn opening_plan(&self, leaf: u64) -> Vec<PlannedEntry> {
        let pattern = self.pattern_of(leaf);
        let row = self.row(pattern);
        let root_level = self.tree_height as u32 - self.group_bits;
        let mut plan = Vec::with_capacity(self.tree_height + 1);
        let mut key = leaf_key(self.tree_height as u32, leaf);
        for step in 0..=self.tree_height {
            let node = if step == 0 { key } else { sibling(key) };
            if let Some(array_index) = self.array_index(node) {
                if node.level > root_level {
                    let record = self.record_of_index(array_index);
                    let slot = record % 2;
                    assert_eq!(row[slot], Some(record), "opening cell outside its pattern record");
                    plan.push(PlannedEntry {
                        node,
                        array_index,
                        slot,
                        entry_in_record: self.entry_in_record(array_index),
                    });
                } else {
                    // Upper siblings beyond the filled levels are default
                    // nodes and never reach this branch.
                    let level = (root_level - node.level) as usize;
                    let slot = 2 + level;
                    assert_eq!(
                        row[slot],
                        Some(node.pos as usize),
                        "opening node outside its pattern record"
                    );
                    plan.push(PlannedEntry {
                        node,
                        array_index,
                        slot,
                        entry_in_record: UPPER_ENTRY,
                    });
                }
            }
            if step > 0 {
                key = parent(key);
            }
            if step == self.tree_height {
                break;
            }
        }
        plan
    }

    /// Index, in the entries of the claim, of the entry that reads
    /// `array_index`: the block times the entries per block, plus the entry
    /// within the block.
    pub fn entry_index(&self, array_index: u64) -> usize {
        self.record_of_index(array_index) * self.entries_per_record()
            + self.entry_in_record(array_index)
    }

    /// Index, within its record, of the entry that reads `array_index`.
    pub fn entry_in_record(&self, array_index: u64) -> usize {
        let segment = (array_index / self.segment_cells as u64) as usize;
        let offset = (array_index % self.segment_cells as u64) as usize;
        (segment % self.segments_per_record) * self.kappa + self.w_index_of_offset[offset] as usize
    }
}
