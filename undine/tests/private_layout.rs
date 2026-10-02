//! The private-opening layout names exactly the entries of a public opening.
#![cfg(feature = "private")]

#[path = "support/cluster.rs"]
mod cluster;

use cluster::{LocalServer, written_cluster};
use paper_a_optimized::core::NodeLinearization;
use paper_a_optimized::private::{PrivateLayout, local_claims, record_entries};
use paper_a_optimized::unauth::service::{UnauthRequest, UnauthResponse, UnauthService};
use paper_a_optimized::unauth::{CodedArrayParams, build_kappa_double_params};
use paper_a_optimized::{AuthParams, Fp};
use std::collections::HashMap;

/// Check, for every leaf and every server, that the entries of a public
/// opening are the entries the layout locates in the pattern's records.
fn assert_layout_matches_public_openings(
    params: &CodedArrayParams,
    auth: &AuthParams,
    servers: &[LocalServer],
    leaves: usize,
    segments_per_record: usize,
) {
    let layout = PrivateLayout::new(
        params,
        auth,
        NodeLinearization::Preorder,
        leaves as u64,
        segments_per_record,
    )
    .expect("layout");
    assert_eq!(layout.kappa(), params.need);

    // The rows table is the row of every pattern.
    let table = layout.rows_table();
    assert_eq!(table.len(), layout.patterns() * layout.slots());
    for pattern in 0..layout.patterns() {
        let from_table = table[pattern * layout.slots()..(pattern + 1) * layout.slots()]
            .iter()
            .map(|&record| (record != u32::MAX).then_some(record as usize))
            .collect::<Vec<_>>();
        assert_eq!(from_table, layout.row(pattern), "rows table, pattern {pattern}");
    }
    assert_eq!(
        layout.patterns(),
        leaves.div_ceil(1 << layout.group_bits())
    );

    for server in servers {
        let guard = server.inner.lock().expect("server lock");
        let store = &guard.store.unauth;
        let claims = local_claims(store);
        assert_eq!(claims, vec![server.label]);
        assert_eq!(layout.records(), store.segments().div_ceil(segments_per_record));
        let mut records: HashMap<usize, Vec<Fp>> = HashMap::new();

        for leaf in 0..leaves as u64 {
            let public = guard.handle_read(leaf);
            let public_entries = public
                .nodes
                .iter()
                .filter(|reply| !reply.shares.is_empty())
                .collect::<Vec<_>>();
            let plan = layout.opening_plan(leaf);
            let row = layout.row(layout.pattern_of(leaf));
            assert_eq!(row.len(), layout.slots());
            assert_eq!(
                plan.iter().map(|entry| entry.node).collect::<Vec<_>>(),
                public_entries.iter().map(|reply| reply.node).collect::<Vec<_>>(),
                "leaf {leaf}: the plan covers the stored nodes of the public opening"
            );
            for (entry, reply) in plan.iter().zip(public_entries) {
                assert_eq!(reply.shares.len(), 1);
                let share = &reply.shares[0];
                assert_eq!(share.registry_index(), server.label);
                let record = row[entry.slot].expect("planned slot holds a record");
                let entries = records
                    .entry(record)
                    .or_insert_with(|| record_entries(store, &layout, server.label, record));
                assert_eq!(entries.len(), layout.entries_per_record());
                assert_eq!(
                    Some(entries[entry.entry_in_record]),
                    share.value(),
                    "leaf {leaf}, node {:?}: record {record} entry {}",
                    entry.node,
                    entry.entry_in_record
                );
            }
        }
    }

    // The patterns that place one record in one upper slot are consecutive:
    // they are the leaves of one subtree.
    for slot in 2..layout.slots() {
        let mut last_pattern_of: HashMap<usize, usize> = HashMap::new();
        for pattern in 0..layout.patterns() {
            if let Some(record) = layout.row(pattern)[slot] {
                if let Some(last) = last_pattern_of.insert(record, pattern) {
                    // A record may hold sibling roots of several subtrees, so
                    // runs of one record may repeat; within the sibling of one
                    // subtree they are consecutive.
                    assert!(last < pattern);
                }
            }
        }
    }
}

#[test]
fn layout_matches_public_openings_at_the_base_threshold() {
    for &(kappa, tree_height, leaves) in &[
        (2usize, 6usize, 1usize),
        (2, 6, 2),
        (2, 6, 3),
        (2, 6, 11),
        (2, 6, 37),
        (2, 6, 64),
        (4, 7, 23),
        (4, 7, 100),
        (2, 50, 21),
        (4, 50, 70),
    ] {
        let (params, auth, servers, _values, _root) =
            written_cluster(kappa, 3 * kappa, tree_height, leaves, 77);
        for segments_per_record in [1usize, 2, 4] {
            assert_layout_matches_public_openings(
                &params,
                &auth,
                &servers[..2],
                leaves,
                segments_per_record,
            );
        }
    }
}

#[test]
fn layout_matches_public_openings_after_a_stitch() {
    let leaves = 45usize;
    let (params, auth, servers, _values, _root) = written_cluster(2, 12, 7, leaves, 78);
    for server in &servers {
        let mut guard = server.inner.lock().expect("server lock");
        match UnauthService::handle_request(&mut *guard, UnauthRequest::KappaDouble) {
            UnauthResponse::KappaDouble { ok: true } => {}
            other => panic!("unexpected kappa-double response: {:?}", other),
        }
    }
    let doubled = build_kappa_double_params(&params).expect("doubled params");
    for segments_per_record in [1usize, 2] {
        assert_layout_matches_public_openings(
            &doubled,
            &auth,
            &servers[..2],
            leaves,
            segments_per_record,
        );
    }
}

#[test]
fn layout_sizes_follow_the_record_size() {
    let (params, auth, _servers, _values, _root) = written_cluster(4, 12, 9, 300, 79);
    let layout =
        PrivateLayout::new(&params, &auth, NodeLinearization::Preorder, 300, 1).expect("layout");
    // A record of kappa^2 = 16 cells holds a subtree of 8 leaves (15 nodes).
    assert_eq!(layout.group_bits(), 3);
    assert_eq!(layout.patterns(), 38);
    // 300 leaves fill 9 levels; the levels above the 3 of a group are 6.
    assert_eq!(layout.slots(), 2 + 6);
    assert_eq!(layout.entries_per_record(), 4);
    let larger =
        PrivateLayout::new(&params, &auth, NodeLinearization::Preorder, 300, 4).expect("layout");
    assert_eq!(larger.group_bits(), 5);
    assert_eq!(larger.patterns(), 10);
    assert_eq!(larger.slots(), 2 + 4);
    assert!(larger.records() <= layout.records().div_ceil(4));
}

#[test]
fn layout_matches_public_openings_next_to_powers_of_two_in_a_deep_tree() {
    // A deep tree with few leaves: the filled part is a left spine, and the
    // number of filled levels changes right at a power of two.
    for exponent in [3u32, 4, 5] {
        for leaves in [(1usize << exponent) - 1, 1 << exponent, (1 << exponent) + 1] {
            let (params, auth, servers, _values, _root) = written_cluster(2, 6, 50, leaves, 80);
            for segments_per_record in [1usize, 2] {
                assert_layout_matches_public_openings(
                    &params,
                    &auth,
                    &servers[..1],
                    leaves,
                    segments_per_record,
                );
            }
        }
    }
}

#[test]
fn whole_tree_layout_locates_every_cell_of_an_opening() {
    use paper_a_optimized::auth::mapping;
    for &(tree_height, leaves, cells_per_record) in &[
        (6usize, 1u64, 4usize),
        (6, 37, 4),
        (6, 64, 8),
        (50, 33, 4),
        (50, 300, 16),
        (50, 1025, 64),
    ] {
        let auth = AuthParams::new(tree_height);
        let layout = PrivateLayout::whole_tree(
            &auth,
            NodeLinearization::Preorder,
            leaves,
            cells_per_record,
        )
        .expect("layout");
        assert_eq!(layout.entries_per_record(), cells_per_record);
        assert_eq!(layout.cells_per_record(), cells_per_record);
        let cursor = mapping::cursor(NodeLinearization::Preorder, tree_height as u16, leaves);
        assert_eq!(layout.records(), cursor.div_ceil(cells_per_record as u64) as usize);
        let table = layout.rows_table();
        for pattern in 0..layout.patterns() {
            let from_table = table[pattern * layout.slots()..(pattern + 1) * layout.slots()]
                .iter()
                .map(|&record| (record != u32::MAX).then_some(record as usize))
                .collect::<Vec<_>>();
            assert_eq!(from_table, layout.row(pattern));
        }
        for leaf in 0..leaves {
            let row = layout.row(layout.pattern_of(leaf));
            let plan = layout.opening_plan(leaf);
            // The leaf itself is always stored; its entry is its cell.
            assert_eq!(plan[0].node.pos, leaf);
            for entry in &plan {
                let record = row[entry.slot].expect("planned slot holds a record");
                assert_eq!(
                    record as u64 * cells_per_record as u64 + entry.entry_in_record as u64,
                    entry.array_index
                );
            }
        }
    }
}
