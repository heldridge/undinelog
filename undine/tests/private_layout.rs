//! The private-opening layout names exactly the entries of a public opening.
#![cfg(feature = "private")]

#[path = "support/cluster.rs"]
mod cluster;

use cluster::{LocalServer, written_cluster};
use paper_a_optimized::core::NodeLinearization;
use paper_a_optimized::private::layout::UPPER_ENTRY;
use paper_a_optimized::private::{Grouping, PrivateLayout, local_claims, record_entries};
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
    assert_grouping_matches_public_openings(
        params,
        auth,
        servers,
        leaves,
        Grouping::one_group(segments_per_record, params.n_grid),
    );
}

/// The same check for any groups and pattern size.
fn assert_grouping_matches_public_openings(
    params: &CodedArrayParams,
    auth: &AuthParams,
    servers: &[LocalServer],
    leaves: usize,
    grouping: Grouping,
) {
    let segments_per_record = grouping.segments_per_group;
    let layout = PrivateLayout::with_grouping(
        params,
        auth,
        NodeLinearization::Preorder,
        leaves as u64,
        grouping,
    )
    .expect("layout");
    assert_eq!(layout.kappa(), params.need);
    assert_eq!(layout.grouping().segments_per_group, segments_per_record);
    let group_slots = layout.group_slots();
    // A subtree of 2^(g+1) - 1 cells fills `span` groups and meets one more.
    let subtree_cells = (1usize << (layout.group_bits() + 1)) - 1;
    assert_eq!(group_slots, subtree_cells.div_ceil(layout.cells_per_record()) + 1);

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
                let id = row[entry.slot].expect("planned slot holds an item");
                // A group slot names the group and the entry within it; an
                // upper slot names the node, whose entry is one field element.
                let (record, entry_in_record) = if entry.slot < group_slots {
                    (id, entry.entry_in_record)
                } else {
                    assert_eq!(entry.entry_in_record, UPPER_ENTRY);
                    assert_eq!(
                        layout.upper_array_index(entry.slot - group_slots, id),
                        entry.array_index
                    );
                    let index = layout.upper_entry_index(entry.slot - group_slots, id);
                    (
                        index / layout.entries_per_record(),
                        index % layout.entries_per_record(),
                    )
                };
                let entries = records
                    .entry(record)
                    .or_insert_with(|| record_entries(store, &layout, server.label, record));
                assert_eq!(entries.len(), layout.entries_per_record());
                assert_eq!(
                    Some(entries[entry_in_record]),
                    share.value(),
                    "leaf {leaf}, node {:?}: record {record} entry {entry_in_record}",
                    entry.node,
                );
            }
        }
    }

    // The patterns that read one upper node are the groups below its
    // sibling, a contiguous range.
    for slot in group_slots..layout.slots() {
        let level = slot - group_slots;
        for pos in 0..layout.upper_count(level) {
            let readers = (0..layout.patterns())
                .filter(|&pattern| layout.row(pattern)[slot] == Some(pos))
                .collect::<Vec<_>>();
            assert_eq!(
                readers,
                layout.upper_patterns(level, pos).collect::<Vec<_>>(),
                "level {level}, node {pos}"
            );
        }
    }
    // Group `g` sits in slot `g mod group_slots`, the groups of a pattern
    // are consecutive, and a group lies in the records of at most two
    // patterns.
    let mut holders = vec![0usize; layout.records()];
    for pattern in 0..layout.patterns() {
        let row = layout.row(pattern);
        let mut groups = (0..group_slots).filter_map(|slot| {
            row[slot].inspect(|&group| assert_eq!(group % group_slots, slot))
        }).collect::<Vec<_>>();
        groups.sort_unstable();
        assert!(!groups.is_empty());
        assert!(groups.windows(2).all(|pair| pair[1] == pair[0] + 1), "consecutive groups");
        for group in groups {
            holders[group] += 1;
        }
    }
    assert!(holders.iter().all(|&count| count <= 2), "a group lies in at most two patterns");
}

/// Patterns whose subtree spans several groups: records of `span + 1`
/// groups, checked against the public openings like the others.
#[test]
fn layout_matches_public_openings_for_patterns_of_several_groups() {
    for &(kappa, tree_height, leaves) in &[
        (2usize, 7usize, 37usize),
        (2, 7, 100),
        (4, 8, 150),
        (2, 50, 70),
    ] {
        let (params, auth, servers, _values, _root) =
            written_cluster(kappa, 3 * kappa, tree_height, leaves, 78);
        let smallest = Grouping::one_group(1, params.n_grid).pattern_bits;
        for extra in [1u32, 2, 3] {
            assert_grouping_matches_public_openings(
                &params,
                &auth,
                &servers[..2],
                leaves,
                Grouping {
                    segments_per_group: 1,
                    pattern_bits: smallest + extra,
                },
            );
        }
        // Groups of two segments with patterns of four groups.
        assert_grouping_matches_public_openings(
            &params,
            &auth,
            &servers[..2],
            leaves,
            Grouping {
                segments_per_group: 2,
                pattern_bits: Grouping::one_group(2, params.n_grid).pattern_bits + 2,
            },
        );
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
                let id = row[entry.slot].expect("planned slot holds an item");
                if entry.slot < 2 {
                    assert_eq!(
                        id as u64 * cells_per_record as u64 + entry.entry_in_record as u64,
                        entry.array_index
                    );
                } else {
                    assert_eq!(entry.entry_in_record, UPPER_ENTRY);
                    assert_eq!(layout.upper_array_index(entry.slot - 2, id), entry.array_index);
                }
            }
        }
    }
}

/// The public pattern size balances the client's upload against its
/// download, 2^floor((h + log2 kappa - 5) / 2) leaves, and is at least the
/// pattern whose subtree fills one segment, 2^(2 log2 kappa - 1) leaves.
#[test]
fn public_pattern_size_balances_upload_against_download() {
    use paper_a_optimized::private::pattern_bits;
    // kappa = 64: patterns of 2^13 to 2^16 leaves on logs of 2^25 to 2^31.
    assert_eq!(pattern_bits(1 << 25, 64), 13);
    assert_eq!(pattern_bits((1 << 25) + 1, 64), 13);
    assert_eq!(pattern_bits(1 << 27, 64), 14);
    assert_eq!(pattern_bits(1 << 29, 64), 15);
    assert_eq!(pattern_bits(1 << 31, 64), 16);
    // A smaller threshold has smaller patterns that span more segments.
    assert_eq!(pattern_bits(1 << 27, 8), 12);
    // A large threshold or a short log: the smallest pattern.
    assert_eq!(pattern_bits(1 << 25, 128), 13);
    assert_eq!(pattern_bits(1 << 25, 256), 15);
    assert_eq!(pattern_bits(1 << 15, 64), 11);
    // The single server (kappa = 1).
    assert_eq!(pattern_bits(1 << 25, 1), 10);
    assert_eq!(pattern_bits(1, 1), 0);
    let public = Grouping::public(1 << 31, 64);
    assert_eq!((public.segments_per_group, public.pattern_bits), (1, 16));
}
