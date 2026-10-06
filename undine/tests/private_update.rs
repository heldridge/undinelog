//! After an append, the updated preprocessing equals a fresh one.
#![cfg(feature = "private")]

#[path = "support/cluster.rs"]
mod cluster;

use cluster::{LocalServer, reader, written_cluster};
use paper_a_optimized::core::NodeLinearization;
use paper_a_optimized::private::{Grouping, PirClient, PirDatabase, PirShape, PrivateLayout};
use paper_a_optimized::unauth::CodedArrayParams;
use paper_a_optimized::{AuthParams, Fp, encode_value};

fn layout_of(
    params: &CodedArrayParams,
    auth: &AuthParams,
    leaves: usize,
    segments_per_record: usize,
) -> PrivateLayout {
    // The largest pattern that fits in one group, enlarged by the bits a
    // test asks for, so that its subtree spans several groups.
    let mut grouping = Grouping::one_group(segments_per_record, params.n_grid);
    grouping.pattern_bits += EXTRA_PATTERN_BITS.with(|bits| bits.get());
    PrivateLayout::with_grouping(
        params,
        auth,
        NodeLinearization::Preorder,
        leaves as u64,
        grouping,
    )
    .expect("layout")
}

thread_local! {
    static EXTRA_PATTERN_BITS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Build the database of `server` for the current log of `leaves` leaves.
fn fresh_database(
    params: &CodedArrayParams,
    auth: &AuthParams,
    server: &LocalServer,
    leaves: usize,
    segments_per_record: usize,
) -> PirDatabase {
    let guard = server.inner.lock().expect("server lock");
    PirDatabase::build(
        &guard.store.unauth,
        &layout_of(params, auth, leaves, segments_per_record),
        server.label,
    )
}

fn assert_same_database(updated: &PirDatabase, fresh: &PirDatabase, context: &str) {
    assert_eq!(updated.shape(), fresh.shape(), "{context}: shape");
    assert!(
        updated.hint_values() == fresh.hint_values(),
        "{context}: the updated hint differs from a fresh one"
    );
    let shape = fresh.shape();
    let client = PirClient::generate(shape);
    let pack_keys = client.pack_keys();
    let mut patterns = vec![0, shape.patterns / 2, shape.patterns - 1];
    patterns.dedup();
    for pattern in patterns {
        assert_eq!(updated.row(pattern), fresh.row(pattern), "{context}: row");
        let query = client.query(pattern);
        assert!(
            updated.answer(&query, &pack_keys).expect("updated answer")
                == fresh.answer(&query, &pack_keys).expect("fresh answer"),
            "{context}: the updated response differs for pattern {pattern}"
        );
    }
}

/// Append in steps and compare the updated database with a fresh one after
/// each step.
fn assert_updates_equal_fresh(
    kappa: usize,
    tree_height: usize,
    segments_per_record: usize,
    lengths: &[usize],
) {
    let (params, auth, servers, _values, _root) =
        written_cluster(kappa, 3 * kappa, tree_height, lengths[0], 111);
    let server = &servers[2];
    let mut client = reader(&params, &auth, lengths[0], &servers);
    let mut database = fresh_database(&params, &auth, server, lengths[0], segments_per_record);

    for window in lengths.windows(2) {
        let (from, to) = (window[0], window[1]);
        let appended = (from..to)
            .map(|idx| encode_value(Fp::from(9000 + 7 * idx as u64)))
            .collect::<Vec<_>>();
        assert_eq!(client.append_leaves(&appended).expect("append"), from as u64);

        {
            let guard = server.inner.lock().expect("server lock");
            let layout = layout_of(&params, &auth, to, segments_per_record);
            let changed = layout.records_changed_by_append(from as u64);
            database.update(&guard.store.unauth, &layout, server.label, Some(&changed));
        }
        let fresh = fresh_database(&params, &auth, server, to, segments_per_record);
        assert_same_database(
            &database,
            &fresh,
            &format!("kappa {kappa}, {from} -> {to} leaves"),
        );
    }
}

#[test]
fn updates_within_the_same_dimensions_equal_fresh_preprocessing() {
    assert_updates_equal_fresh(2, 8, 1, &[20, 21, 27, 31]);
    assert_updates_equal_fresh(4, 50, 1, &[40, 57]);
}

#[test]
fn updates_that_add_a_slot_equal_fresh_preprocessing() {
    // Crossing a power of two fills one more level: the rows gain a slot.
    let shape_at = |leaves: usize| {
        let (params, auth, _servers, _values, _root) = written_cluster(2, 6, 8, leaves, 112);
        PirShape::of(&layout_of(&params, &auth, leaves, 1))
    };
    assert!(shape_at(33).slots > shape_at(32).slots);
    assert_updates_equal_fresh(2, 8, 1, &[30, 32, 33, 64, 65]);
}

#[test]
fn updates_that_grow_the_database_dimensions_equal_fresh_preprocessing() {
    // More columns: with records of four segments at kappa = 4, the seventh
    // slot takes the row past 2048 cells.
    assert_updates_equal_fresh(4, 11, 4, &[500, 530]);
    // More rows: at kappa = 2 a pattern is two leaves, so 4096 leaves fill
    // the 2048 rows of the smallest database.
    assert_updates_equal_fresh(2, 13, 1, &[4090, 4100]);
}

#[test]
fn updates_of_patterns_that_span_several_groups_equal_fresh_preprocessing() {
    for extra in [1u32, 2] {
        EXTRA_PATTERN_BITS.with(|bits| bits.set(extra));
        // The same dimensions, a new slot, and a new level of the tree.
        assert_updates_equal_fresh(2, 8, 1, &[20, 21, 27, 31, 33, 64, 65]);
        assert_updates_equal_fresh(4, 50, 1, &[40, 57, 130]);
        assert_updates_equal_fresh(4, 11, 2, &[500, 530]);
    }
    EXTRA_PATTERN_BITS.with(|bits| bits.set(0));
}

#[test]
fn a_server_updates_its_state_after_an_append() {
    let kappa = 2usize;
    let leaves = 20usize;
    let (params, auth, servers, mut values, _root) =
        written_cluster(kappa, 3 * kappa, 8, leaves, 113);
    let mut client = reader(&params, &auth, leaves, &servers);
    assert_eq!(client.read_private(7).expect("private opening"), values[7]);

    let appended = (0..15u64)
        .map(|idx| encode_value(Fp::from(77_000 + idx)))
        .collect::<Vec<_>>();
    client.append_leaves(&appended).expect("append");
    values.extend(appended);

    // The servers bring their preprocessing to the longer log, and the
    // private openings still equal the public ones.
    for index in [0u64, 7, 19, 20, 34] {
        let (value, path) = client
            .read_private_with_path(index, None)
            .expect("private opening after the append");
        assert_eq!(value, values[index as usize]);
        let (public_value, public_path) = client.read_with_path(index).expect("public opening");
        assert_eq!(value, public_value);
        assert_eq!(path.siblings, public_path.siblings);
        assert_eq!(path.root, public_path.root);
    }
    // Every server that answered after the append updated its state once.
    let mut updated = 0;
    for server in &servers {
        let guard = server.inner.lock().expect("server lock");
        if let Some(state) = guard.private_state() {
            if state.leaves() == values.len() as u64 {
                assert_eq!(state.updates(), 1);
                updated += 1;
            }
        }
    }
    assert!(updated >= kappa);
}
