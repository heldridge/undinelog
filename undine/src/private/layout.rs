//! Records and patterns of the private-opening database.
//!
//! # Records
//! A maintainer's database is its primary columns. A *record* is the primary
//! columns of `segments_per_record` consecutive segments of one claim:
//! `segments_per_record * kappa` stored entries. The cell at array index `p`
//! is read from the entry `(p / segment_cells, w_index(p % segment_cells))`,
//! exactly as in a public point read.
//!
//! # Patterns
//! The leaves are cut into groups of `2^group_bits` consecutive leaves, and a
//! *pattern* is one group. A group is a subtree, and its nodes occupy one
//! contiguous range of the preorder array that is shorter than a record, so it
//! lies in at most two consecutive records. The opening of every leaf of the
//! group needs cells of that subtree and one sibling at each level above it.
//! The row of a pattern therefore lists, in fixed *slots*:
//! - slots 0 and 1: the records that hold the group's subtree;
//! - slot `2 + j`: the record that holds the sibling of the group's ancestor
//!   `j` levels above the subtree root, for the levels inside the filled part
//!   of the tree.
//!
//! A slot is empty when its sibling is a default node or when the subtree
//! fits in one record. The same row serves every leaf of the group, so a
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

impl PrivateLayout {
    /// Layout for a log of `leaves` filled leaves at the threshold of `params`.
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
        Ok(Self {
            tree_height,
            node_linearization,
            leaves,
            kappa: cells_per_record,
            segment_cells: cells_per_record,
            segments_per_record: 1,
            group_bits,
            upper_slots: active_bits.saturating_sub(group_bits) as usize,
            records: cells.div_ceil(cells_per_record as u64) as usize,
            patterns: leaves.div_ceil(1u64 << group_bits) as usize,
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

    /// Slots per row.
    pub fn slots(&self) -> usize {
        2 + self.upper_slots
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
        row[0] = Some(first_record);
        let subtree_cells = (1u64 << (self.group_bits + 1)) - 1;
        let last_record = self.record_of_index(first_index + subtree_cells - 1);
        if last_record != first_record && last_record < self.records {
            row[1] = Some(last_record);
        }
        let mut ancestor = root;
        for slot in 2..self.slots() {
            if let Some(index) = self.array_index(sibling(ancestor)) {
                row[slot] = Some(self.record_of_index(index));
            }
            ancestor = parent(ancestor);
        }
        row
    }

    /// The rows of all patterns as one table of `patterns x slots` record
    /// numbers, with `u32::MAX` for an empty slot. It equals [`Self::row`]
    /// for every pattern and is computed level by level.
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
        let mut levels: Vec<Vec<u64>> = vec![indices.clone()];
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
            // One more node per level may be an active sibling.
            if width % 2 == 1 {
                let parent = indices[width / 2];
                next.push(parent + 1 + left_nodes);
            }
            levels.push(next.clone());
            indices = next;
        }
        let cursor = mapping::cursor(self.node_linearization, self.tree_height as u16, self.leaves);
        for pattern in 0..self.patterns {
            let row = &mut table[pattern * slots..(pattern + 1) * slots];
            let first_index = levels[(root_level - top_level) as usize][pattern];
            let first_record = self.record_of_index(first_index);
            row[0] = first_record as u32;
            let last_record = self.record_of_index(first_index + subtree_cells - 1);
            if last_record != first_record && last_record < self.records {
                row[1] = last_record as u32;
            }
            for slot in 2..slots {
                let level = root_level - (slot as u32 - 2);
                let sibling = (pattern as u64 >> (root_level - level)) ^ 1;
                let index = levels[(level - top_level) as usize][sibling as usize];
                let shift = self.tree_height as u32 - level;
                // A sibling is active when its first leaf is filled.
                if (sibling << shift) < self.leaves && index < cursor {
                    row[slot] = self.record_of_index(index) as u32;
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
    /// row of the leaf's pattern.
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
                let record = self.record_of_index(array_index);
                let slot = if node.level > root_level {
                    if row[0] == Some(record) { 0 } else { 1 }
                } else {
                    2 + (root_level - node.level) as usize
                };
                // Upper siblings beyond the filled levels are default nodes
                // and never reach this branch.
                assert_eq!(row[slot], Some(record), "opening cell outside its pattern row");
                plan.push(PlannedEntry {
                    node,
                    array_index,
                    slot,
                    entry_in_record: self.entry_in_record(array_index),
                });
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

    /// Index, within its record, of the entry that reads `array_index`.
    pub fn entry_in_record(&self, array_index: u64) -> usize {
        let segment = (array_index / self.segment_cells as u64) as usize;
        let offset = (array_index % self.segment_cells as u64) as usize;
        (segment % self.segments_per_record) * self.kappa + self.w_index_of_offset[offset] as usize
    }
}
