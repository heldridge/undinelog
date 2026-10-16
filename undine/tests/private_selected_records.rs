//! Client fixtures must match whole fragments, including malicious shares.
#![cfg(feature = "private")]

#[path = "../benches/support/private_records.rs"]
mod private_records;

use paper_a_optimized::core::NodeLinearization;
use paper_a_optimized::merkle::default_hash;
use paper_a_optimized::private::{BulkLog, PirDatabase, PirShape, PrivateLayout, SuffixTree};
use paper_a_optimized::{
    AuthParams, CodedArrayConfig, Fp, build_coded_params, build_shard_registry_deterministic,
    decode_value, fp_one, fp_zero,
};
use private_records::ClaimRecords;

fn full_slots(layout: &PrivateLayout, entries: &[Fp], pattern: usize) -> Vec<Vec<u64>> {
    let shape = PirShape::of(layout);
    let cells = PirDatabase::cells_from_entries(layout, entries);
    let upper = PirDatabase::upper_from_cells(layout, &cells);
    layout
        .row(pattern)
        .iter()
        .enumerate()
        .map(|(slot, id)| {
            let width = shape.slot_width(slot);
            match id {
                None => vec![0; width],
                Some(id) => {
                    let source = if slot < shape.group_slots {
                        &cells
                    } else {
                        &upper[slot - shape.group_slots]
                    };
                    source[id * width..(id + 1) * width]
                        .iter()
                        .copied()
                        .map(u64::from)
                        .collect()
                }
            }
        })
        .collect()
}

#[test]
fn selected_records_equal_full_fragments_and_corruptions() {
    const SEED: u64 = 42;
    for (kappa, height, leaves, defaults) in [
        (2, 7, 1, 1),
        (2, 7, 37, 37),
        (4, 9, 257, 257),
        (2, 50, 33, 33),
        (4, 50, 257, 257),
        (4, 9, 512, 0),
        (4, 9, 512, 507),
    ] {
        let params = build_coded_params(&CodedArrayConfig {
            kappa,
            shard_count: 3 * kappa,
            shard_registry: build_shard_registry_deterministic(3 * kappa),
            party_registry_indices: vec![],
            seed: SEED,
        });
        let auth = AuthParams::new(height);
        // Nonconsecutive claims detect accidental use of a claim's position
        // in the selected list as its registry identifier.
        let claims = [0, kappa, 3 * kappa - 1];
        let bulk = BulkLog::build_for_claims(
            &params,
            &auth,
            NodeLinearization::Preorder,
            leaves,
            defaults,
            SEED,
            &claims,
            2,
            3,
        );
        let tree = (defaults < leaves).then(|| SuffixTree::new(&auth, defaults, SEED));
        let cell = |level, pos| match &tree {
            Some(tree) => tree.cell(level, pos),
            None if level as usize == height => fp_zero(),
            None => decode_value(default_hash(&auth.defaults, level as usize)),
        };
        for groups in [1, 2, 4] {
            let layout =
                PrivateLayout::new(&params, &auth, NodeLinearization::Preorder, leaves, groups)
                    .unwrap();
            let indices = [0, leaves / 2, leaves - 1];
            let selected = ClaimRecords::new(&params, &auth, &layout, &claims, indices, &cell);
            for index in indices {
                let pattern = layout.pattern_of(index);
                let plan = layout.opening_plan(index);
                for claim in 0..claims.len() {
                    let original = bulk.claim_entries(claim);
                    assert_eq!(
                        selected.slots(&layout, claim, pattern),
                        full_slots(&layout, original, pattern)
                    );
                    for entry in &plan {
                        let offset = layout.entry_index(entry.array_index);
                        assert_eq!(selected.entry(claim, offset), original[offset]);
                        let changed = original[offset] + fp_one();
                        let mut corrupted = original.to_vec();
                        corrupted[offset] = changed;
                        assert_eq!(
                            selected.slots_with_patch(
                                &layout,
                                claim,
                                pattern,
                                Some((offset, changed))
                            ),
                            full_slots(&layout, &corrupted, pattern),
                            "height {height}, leaves {leaves}, groups {groups}, claim {claim}, offset {offset}",
                        );
                    }
                }
            }
        }
    }
}
