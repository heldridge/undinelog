//! Exact selected records for client-only benchmarks.
//!
//! Encode only segments touched by the requested openings. The callers use
//! preorder layouts. The input callback supplies real cells within the
//! allocated prefix; segment defaults and record padding follow BulkLog.

#![allow(dead_code)] // Benchmarks and the independent equivalence test use different methods.

use paper_a_optimized::auth::mapping::{self, NodeLinearization};
use paper_a_optimized::merkle::default_hash;
use paper_a_optimized::private::pir::{bytes_from_cells, bytes_to_cells};
use paper_a_optimized::private::{PirShape, PrivateLayout, entries_to_cells, tree_segment_entries};
use paper_a_optimized::{AuthParams, CodedArrayParams, Fp, decode_value, fp_zero};
use std::collections::{BTreeSet, HashMap};

const ORDER: NodeLinearization = NodeLinearization::Preorder;

pub struct ClaimRecords {
    per_record: usize,
    /// `records[record][claim]`: the claim's entries of the record.
    records: HashMap<usize, Vec<Vec<Fp>>>,
}

impl ClaimRecords {
    pub fn new(
        params: &CodedArrayParams,
        auth: &AuthParams,
        layout: &PrivateLayout,
        claims: &[usize],
        leaves: impl IntoIterator<Item = u64>,
        cell: &dyn Fn(u32, u64) -> Fp,
    ) -> Self {
        let per_record = layout.entries_per_record();
        let kappa = layout.kappa();
        let shape = PirShape::of(layout);
        let mut wanted = BTreeSet::new();
        for leaf in leaves {
            for (slot, id) in layout.row(layout.pattern_of(leaf)).into_iter().enumerate() {
                match id {
                    Some(record) if slot < shape.group_slots => {
                        wanted.insert(record);
                    }
                    Some(pos) => {
                        let index = layout.upper_entry_index(slot - shape.group_slots, pos);
                        wanted.insert(index / per_record);
                    }
                    None => {}
                }
            }
        }
        let height = auth.tree_height as u16;
        let segments_total =
            mapping::cursor(ORDER, height, layout.leaves()).div_ceil(params.n_grid as u64) as usize;
        let segments_per_record = layout.segments_per_record();
        let segments = wanted
            .iter()
            .flat_map(|&record| record * segments_per_record..(record + 1) * segments_per_record)
            .filter(|&segment| segment < segments_total)
            .collect::<Vec<_>>();
        let cursor = mapping::cursor(ORDER, height, layout.leaves());
        // The final allocated segment includes default cells beyond the
        // active prefix. Records beyond that segment are zero padding.
        let cell = |level: u32, pos: u64| {
            if mapping::preorder_idx(height, level as u16, pos) < cursor {
                cell(level, pos)
            } else if level == u32::from(height) {
                fp_zero()
            } else {
                decode_value(default_hash(&auth.defaults, level as usize))
            }
        };
        let encoded = tree_segment_entries(params, auth, &cell, claims, &segments);
        let by_segment = segments
            .iter()
            .copied()
            .zip(encoded)
            .collect::<HashMap<_, _>>();
        let records = wanted
            .iter()
            .map(|&record| {
                let entries = (0..claims.len())
                    .map(|claim| {
                        let mut values = vec![fp_zero(); per_record];
                        for k in 0..segments_per_record {
                            if let Some(segment) =
                                by_segment.get(&(record * segments_per_record + k))
                            {
                                values[k * kappa..(k + 1) * kappa].copy_from_slice(&segment[claim]);
                            }
                        }
                        values
                    })
                    .collect();
                (record, entries)
            })
            .collect();
        Self {
            per_record,
            records,
        }
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn entry(&self, claim: usize, entry: usize) -> Fp {
        self.records[&(entry / self.per_record)][claim][entry % self.per_record]
    }

    fn record_cells(
        &self,
        shape: &PirShape,
        claim: usize,
        record: usize,
        patch: Option<(usize, Fp)>,
    ) -> Vec<u16> {
        let mut entries = self.records[&record][claim].clone();
        if let Some((index, value)) = patch {
            if index / self.per_record == record {
                entries[index % self.per_record] = value;
            }
        }
        let mut cells = vec![0u16; shape.record_cells];
        entries_to_cells(&entries, &mut cells);
        cells
    }

    /// The row the query of `pattern` selects from claim `claim`: the
    /// record of each group slot and the entry of each upper slot, as
    /// `PirDatabase` lays them out.
    pub fn slots(&self, layout: &PrivateLayout, claim: usize, pattern: usize) -> Vec<Vec<u64>> {
        self.slots_with_patch(layout, claim, pattern, None)
    }

    /// Apply the same changed stored entry wherever the row reads it,
    /// including an upper-node slot that also occurs in a group record.
    pub fn slots_with_patch(
        &self,
        layout: &PrivateLayout,
        claim: usize,
        pattern: usize,
        patch: Option<(usize, Fp)>,
    ) -> Vec<Vec<u64>> {
        let shape = PirShape::of(layout);
        layout
            .row(pattern)
            .iter()
            .enumerate()
            .map(|(slot, id)| {
                let mut cells = vec![0u16; shape.slot_width(slot)];
                match id {
                    Some(record) if slot < shape.group_slots => {
                        cells = self.record_cells(&shape, claim, *record, patch);
                    }
                    Some(pos) => {
                        let index = layout.upper_entry_index(slot - shape.group_slots, *pos);
                        let block = self
                            .record_cells(&shape, claim, index / self.per_record, patch)
                            .into_iter()
                            .map(u64::from)
                            .collect::<Vec<_>>();
                        bytes_to_cells(
                            bytes_from_cells(&block, index % self.per_record),
                            &mut cells,
                        );
                    }
                    None => {}
                }
                assert_eq!(cells.len(), shape.slot_width(slot), "cells of slot {slot}");
                cells.into_iter().map(u64::from).collect()
            })
            .collect()
    }
}
