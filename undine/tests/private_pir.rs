//! A maintainer's private answer equals YPIR's answer over the table of
//! complete openings, and a client decodes the entries of a public opening.
#![cfg(feature = "private")]

#[path = "support/cluster.rs"]
mod cluster;

use cluster::written_cluster;
use paper_a_optimized::core::NodeLinearization;
use paper_a_optimized::private::{
    Grouping, PirClient, PirDatabase, PirError, PirShape, PrivateLayout, TableDatabase,
    entry_from_cells,
};

fn assert_private_answers(kappa: usize, tree_height: usize, leaves: usize, segments_per_record: usize) {
    assert_private_answers_for_patterns(kappa, tree_height, leaves, segments_per_record, 0);
}

/// The same check with patterns `extra_bits` larger than one group, so that
/// a record holds more than two groups.
fn assert_private_answers_for_patterns(
    kappa: usize,
    tree_height: usize,
    leaves: usize,
    segments_per_record: usize,
    extra_bits: u32,
) {
    let (params, auth, servers, _values, _root) =
        written_cluster(kappa, 3 * kappa, tree_height, leaves, 91);
    let mut grouping = Grouping::one_group(segments_per_record, params.n_grid);
    grouping.pattern_bits += extra_bits;
    let layout = PrivateLayout::with_grouping(
        &params,
        &auth,
        NodeLinearization::Preorder,
        leaves as u64,
        grouping,
    )
    .expect("layout");
    assert_eq!(PirShape::of(&layout).group_slots, layout.group_slots());
    let shape = PirShape::of(&layout);
    let server = &servers[1];
    let guard = server.inner.lock().expect("server lock");
    let store = &guard.store.unauth;

    let database = PirDatabase::build(store, &layout, server.label);
    assert_eq!(database.shape(), shape);
    let table = TableDatabase::from_database(&database);

    // Preprocessing over the stored records equals YPIR's over the table.
    assert!(
        database.hint_values() == table.hint_values(),
        "hint differs from the table's"
    );

    let client = PirClient::generate(shape);
    let pack_keys = client.pack_keys();
    let mut patterns = vec![0, shape.patterns / 2, shape.patterns - 1];
    patterns.dedup();
    for pattern in patterns {
        let query = client.query(pattern);
        assert!(
            database.first_pass(&query).expect("first pass")
                == table.first_pass(&query).expect("table first pass"),
            "first pass differs from the table's for pattern {pattern}"
        );
        let response = database.answer(&query, &pack_keys).expect("answer");
        assert!(
            response == table.answer(&query, &pack_keys).expect("table answer"),
            "response differs from the table's for pattern {pattern}"
        );

        // The client decodes the record of the pattern: its blocks and the
        // entries of its upper nodes.
        let slots = client.decode(&response).expect("decode");
        let row = layout.row(pattern);
        assert_eq!(slots.len(), row.len());
        for (slot, item) in row.iter().enumerate() {
            let expected = match item {
                Some(id) => database
                    .slot_cells(slot, *id)
                    .iter()
                    .map(|&cell| cell as u64)
                    .collect::<Vec<_>>(),
                None => vec![0u64; shape.slot_width(slot)],
            };
            assert!(slots[slot] == expected, "pattern {pattern}, slot {slot}");
        }

        // The decoded entries are those of a public opening, for every leaf
        // of the pattern.
        let first_leaf = (pattern as u64) << layout.group_bits();
        let last_leaf = (first_leaf + (1u64 << layout.group_bits())).min(leaves as u64);
        for leaf in first_leaf..last_leaf {
            let public = guard.handle_read(leaf);
            let public_entries = public
                .nodes
                .iter()
                .filter(|reply| !reply.shares.is_empty())
                .collect::<Vec<_>>();
            let plan = layout.opening_plan(leaf);
            assert_eq!(plan.len(), public_entries.len());
            for (entry, reply) in plan.iter().zip(public_entries) {
                assert_eq!(entry.node, reply.node);
                assert_eq!(
                    Some(entry_from_cells(&slots[entry.slot], entry.entry_in_record)),
                    reply.shares[0].value(),
                    "leaf {leaf}, node {:?}",
                    entry.node
                );
            }
        }
    }
}

#[test]
fn private_answer_equals_the_table_and_decodes_public_entries() {
    assert_private_answers(2, 8, 90, 1);
    assert_private_answers(4, 8, 200, 1);
    assert_private_answers(4, 50, 150, 2);
}

#[test]
fn private_answer_equals_the_table_for_patterns_of_several_groups() {
    assert_private_answers_for_patterns(2, 8, 90, 1, 1);
    assert_private_answers_for_patterns(2, 8, 200, 1, 3);
    assert_private_answers_for_patterns(4, 50, 150, 1, 2);
    assert_private_answers_for_patterns(4, 8, 200, 2, 1);
}

#[test]
fn malformed_requests_and_responses_are_rejected() {
    let (params, auth, servers, _values, _root) = written_cluster(2, 6, 6, 20, 92);
    let layout = PrivateLayout::new(&params, &auth, NodeLinearization::Preorder, 20, 1)
        .expect("layout");
    let shape = PirShape::of(&layout);
    let guard = servers[0].inner.lock().expect("server lock");
    let database = PirDatabase::build(&guard.store.unauth, &layout, 0);
    let client = PirClient::generate(shape);
    let pack_keys = client.pack_keys();
    let query = client.query(0);

    assert_eq!(
        database.answer(&query[1..], &pack_keys),
        Err(PirError::Query)
    );
    let mut short_keys = pack_keys.clone();
    short_keys.words.pop();
    assert_eq!(database.answer(&query, &short_keys), Err(PirError::PackKeys));

    let mut response = database.answer(&query, &pack_keys).expect("answer");
    assert!(client.decode(&response).is_ok());
    response[0].pop();
    assert_eq!(client.decode(&response), Err(PirError::Response));
    assert_eq!(client.decode(&[]), Err(PirError::Response));
}
