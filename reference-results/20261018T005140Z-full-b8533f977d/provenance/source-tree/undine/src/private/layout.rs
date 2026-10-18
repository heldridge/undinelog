//! Groups, patterns, and records of the private-opening database.
//!
//! # Groups
//! A maintainer's database is its primary columns. A *group* is the unit of
//! storage and of retrieval: the primary columns of `segments_per_record`
//! consecutive segments of one claim, `segments_per_record * kappa` stored
//! entries (the code calls a group a record of the store). Under the public
//! grouping a group is one segment. The cell at array index `p` is read from
//! the entry `(p / segment_cells, w_index(p % segment_cells))`, exactly as
//! in a public point read.
//!
//! # Patterns and records
//! A *pattern* is the combination of groups that holds the openings of the
//! `2^pattern_bits` leaves below one node. The subtree of that node is one
//! contiguous range of the preorder array, so it meets at most
//! `group_slots = span + 1` consecutive groups, where `span` is the number
//! of groups it fills. The opening of a leaf of the pattern consists of a
//! lower part inside that subtree, whose cells depend on the leaf, and of
//! one sibling at each level above the subtree, which depends on the
//! pattern alone. The *record* of a pattern lists, in fixed *slots*:
//! - slots `0..group_slots`: the groups that the subtree meets, group `g`
//!   in slot `g % group_slots`;
//! - slot `group_slots + j`: the entry of the sibling of the pattern root's
//!   ancestor `j` levels above the root, for the levels inside the filled
//!   part of the tree. Such a sibling is an *upper node*, named by its
//!   level and its position within the level.
//!
//! A slot is empty when its sibling is a default node or when the subtree
//! meets fewer groups. The same record serves every leaf of the pattern,
//! so a request names only the pattern.

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
    /// The subtree of a pattern must fill at least one group and fit in the
    /// tree.
    PatternBits { pattern_bits: u32, group_cells: usize },
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
            LayoutError::PatternBits { pattern_bits, group_cells } => write!(
                f,
                "patterns of 2^{pattern_bits} leaves do not fill a group of {group_cells} cells"
            ),
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
    /// Slots of a record that hold groups: the groups a subtree fills, and
    /// one more.
    group_slots: usize,
    upper_slots: usize,
    records: usize,
    patterns: usize,
    /// W-axis index of each logical offset of a segment.
    w_index_of_offset: Vec<u32>,
}

/// Index of the entry whose symbols fill a slot above the group's subtree.
pub const UPPER_ENTRY: usize = 0;

/// How a fragment is cut for private openings: the group and the pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grouping {
    /// Segments per group.
    pub segments_per_group: usize,
    /// Leaves per pattern, as a power of two.
    pub pattern_bits: u32,
}

/// The public pattern size balances the client's upload against its
/// download. The query has one coefficient of 7 bytes per pattern,
/// `7 * leaves / 2^g` bytes for patterns of `2^g` leaves. The record holds
/// the columns of the pattern's subtree, `2^(g+1) / kappa` entries at 110
/// bytes each in the reply (32 bytes carried at 12 KiB per 2^11 symbols of
/// 14 bits). The two are equal at `2^g = sqrt(0.032 * leaves * kappa)`, in
/// powers of two `g = floor((h + log2 kappa - PATTERN_SHIFT) / 2)` for the
/// height `h` of the active prefix.
pub const PATTERN_SHIFT: i64 = 5;

/// The public pattern size for a log of `leaves` leaves at threshold
/// `kappa`, a power of two: the balance, and at least the pattern whose
/// subtree fills one segment, `2 log2 kappa - 1`, which returns two columns,
/// the least a maintainer returns.
pub fn pattern_bits(leaves: u64, kappa: usize) -> u32 {
    assert!(kappa.is_power_of_two(), "the threshold is a power of two");
    let height = if leaves <= 1 { 0 } else { (leaves - 1).ilog2() + 1 };
    let log_kappa = i64::from(kappa.ilog2());
    let balance = (i64::from(height) + log_kappa - PATTERN_SHIFT).div_euclid(2);
    balance.max(2 * log_kappa - 1).max(0) as u32
}

impl Grouping {
    /// The public grouping of a log of `leaves` leaves at threshold `kappa`:
    /// the segment as the group and the public pattern size. Every client
    /// and every maintainer derive it from the log length in the anchor and
    /// the threshold, so one query serves every maintainer of a registry.
    pub fn public(leaves: u64, kappa: usize) -> Self {
        Self {
            segments_per_group: 1,
            pattern_bits: pattern_bits(leaves, kappa),
        }
    }

    /// Groups of `segments_per_group` segments of `segment_cells` cells with
    /// the largest pattern whose subtree fits in one group, so that a
    /// record holds two groups.
    pub fn one_group(segments_per_group: usize, segment_cells: usize) -> Self {
        Self {
            segments_per_group,
            pattern_bits: (segments_per_group * segment_cells + 1).ilog2().saturating_sub(1),
        }
    }
}

impl PrivateLayout {
    /// Layout for a log of `leaves` filled leaves at the threshold of
    /// `params`, with the public grouping [`Grouping::public`].
    pub fn for_log(
        params: &CodedArrayParams,
        auth: &AuthParams,
        node_linearization: NodeLinearization,
        leaves: u64,
    ) -> Result<Self, LayoutError> {
        Self::with_grouping(
            params,
            auth,
            node_linearization,
            leaves,
            Grouping::public(leaves, params.s),
        )
    }

    /// Layout for a log of `leaves` filled leaves at the threshold of
    /// `params`, with groups of `segments_per_record` segments and the
    /// largest pattern whose subtree fits in one group.
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
        Self::with_grouping(
            params,
            auth,
            node_linearization,
            leaves,
            Grouping::one_group(segments_per_record, params.n_grid),
        )
    }

    /// Layout for a log of `leaves` filled leaves at the threshold of
    /// `params`, with the groups and the pattern size of `grouping`.
    pub fn with_grouping(
        params: &CodedArrayParams,
        auth: &AuthParams,
        node_linearization: NodeLinearization,
        leaves: u64,
        grouping: Grouping,
    ) -> Result<Self, LayoutError> {
        let segments_per_record = grouping.segments_per_group;
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
        let (group_bits, group_slots) =
            Self::pattern_of_grouping(grouping.pattern_bits, record_cells, tree_height)?;
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
            group_slots,
            upper_slots,
            records,
            patterns,
            w_index_of_offset,
        })
    }

    /// The pattern size clipped to the tree, and the group slots of its
    /// record: a subtree of `2^(g+1) - 1` cells fills `span` groups of
    /// `group_cells` cells and meets at most `span + 1`.
    fn pattern_of_grouping(
        pattern_bits: u32,
        group_cells: usize,
        tree_height: usize,
    ) -> Result<(u32, usize), LayoutError> {
        let group_bits = pattern_bits.min(tree_height as u32);
        let subtree_cells = (1u64 << (group_bits + 1)) - 1;
        // A pattern below the tree's height fills at least one group; a
        // tree smaller than a group is one pattern.
        if group_bits < tree_height as u32 && subtree_cells + 1 < group_cells as u64 {
            return Err(LayoutError::PatternBits {
                pattern_bits,
                group_cells,
            });
        }
        let span = subtree_cells.div_ceil(group_cells as u64) as usize;
        Ok((group_bits, span + 1))
    }

    /// Layout for a server that stores the whole tree, with no coding: a
    /// group is `cells_per_record` consecutive cells of the maintained
    /// array, and every cell is one entry. The pattern is the largest whose
    /// subtree fits in one group.
    pub fn whole_tree(
        auth: &AuthParams,
        node_linearization: NodeLinearization,
        leaves: u64,
        cells_per_record: usize,
    ) -> Result<Self, LayoutError> {
        if cells_per_record == 0 || !cells_per_record.is_power_of_two() {
            return Err(LayoutError::SegmentsPerRecord(cells_per_record));
        }
        Self::whole_tree_with_pattern(
            auth,
            node_linearization,
            leaves,
            cells_per_record,
            Grouping::one_group(1, cells_per_record).pattern_bits,
        )
    }

    /// Layout for a server that stores the whole tree, with groups of
    /// `cells_per_record` cells and patterns of `2^pattern_bits` leaves.
    pub fn whole_tree_with_pattern(
        auth: &AuthParams,
        node_linearization: NodeLinearization,
        leaves: u64,
        cells_per_record: usize,
        pattern_bits: u32,
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
        let (group_bits, group_slots) =
            Self::pattern_of_grouping(pattern_bits, cells_per_record, tree_height)?;
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
            group_slots,
            upper_slots,
            records: cells.div_ceil(cells_per_record as u64) as usize,
            patterns,
            w_index_of_offset: (0..cells_per_record as u32).collect(),
        })
    }

    /// The groups and the pattern size of this layout.
    pub fn grouping(&self) -> Grouping {
        Grouping {
            segments_per_group: self.segments_per_record,
            pattern_bits: self.group_bits,
        }
    }

    /// Slots of a record that hold groups.
    pub fn group_slots(&self) -> usize {
        self.group_slots
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

    /// Slots per record: the groups of the pattern's subtree and one entry
    /// per level above it.
    pub fn slots(&self) -> usize {
        self.group_slots + self.upper_slots
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
        let subtree_cells = (1u64 << (self.group_bits + 1)) - 1;
        let last_record = self.record_of_index(first_index + subtree_cells - 1);
        for record in first_record..=last_record.min(self.records - 1) {
            row[record % self.group_slots] = Some(record);
        }
        for level in 0..self.upper_slots {
            let pos = (pattern >> level) ^ 1;
            if pos < self.upper_count(level) {
                row[self.group_slots + level] = Some(pos);
            }
        }
        row
    }

    /// The records of all patterns as one table of `patterns x slots` ids,
    /// with `u32::MAX` for an empty slot: group numbers in the group slots,
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
            let last_record = self.record_of_index(first_index + subtree_cells - 1);
            for record in first_record..=last_record.min(self.records - 1) {
                row[record % self.group_slots] = record as u32;
            }
            for level in 0..self.upper_slots {
                let pos = (pattern >> level) ^ 1;
                if pos < self.upper_count(level) {
                    row[self.group_slots + level] = pos as u32;
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
                    let slot = record % self.group_slots;
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
                    let slot = self.group_slots + level;
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
