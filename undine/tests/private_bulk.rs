//! The bulk builder gives the primary columns of the append path.
#![cfg(feature = "private")]

#[path = "support/cluster.rs"]
mod cluster;

use cluster::{build_params, build_servers, reader};
use paper_a_optimized::core::NodeLinearization;
use paper_a_optimized::private::{
    PirDatabase, PrivateLayout, build_primary_columns, build_primary_columns_in_chunks,
    bulk_leaf_value, record_entries,
};
use paper_a_optimized::private::bulk::{BulkLog, CHUNK_BITS};
use paper_a_optimized::{AuthParams, Fp};

const SEED: u64 = 0x5eed;

/// Write the bulk log through the servers and compare one claim's stored
/// primary columns, and the root, with the bulk builder's.
fn assert_bulk_equals_append_path(
    kappa: usize,
    tree_height: usize,
    leaves: usize,
    threads: usize,
    chunk_bits: u32,
) {
    let params = build_params(kappa, 3 * kappa, 121);
    let auth = AuthParams::new(tree_height);
    let servers = build_servers(&params, &auth, leaves);
    let mut client = reader(&params, &auth, leaves, &servers);
    let updates = (0..leaves as u64)
        .map(|index| (index, bulk_leaf_value(SEED, index)))
        .collect::<Vec<_>>();
    client.write_batch(&updates).expect("write the bulk log");
    let root = client.trusted_root().expect("root after write");

    let server = &servers[1];
    let guard = server.inner.lock().expect("server lock");
    let store = &guard.store.unauth;
    let layout = PrivateLayout::new(&params, &auth, NodeLinearization::Preorder, leaves as u64, 1)
        .expect("layout");
    let stored = (0..store.segments())
        .flat_map(|segment| record_entries(store, &layout, server.label, segment))
        .collect::<Vec<Fp>>();

    let bulk = build_primary_columns_in_chunks(
        &params,
        &auth,
        NodeLinearization::Preorder,
        leaves as u64,
        SEED,
        server.label,
        threads,
        chunk_bits,
    );
    let context = format!(
        "kappa {kappa}, height {tree_height}, {leaves} leaves, chunks of 2^{chunk_bits}"
    );
    assert_eq!(bulk.root, root, "{context}: root");
    assert_eq!(bulk.segments, store.segments(), "{context}: segments");
    assert!(bulk.entries == stored, "{context}: primary columns differ");

    // The database built from the bulk columns is the one built from the
    // store.
    let from_bulk = PirDatabase::from_entries(&layout, &bulk.entries);
    let from_store = PirDatabase::build(store, &layout, server.label);
    assert_eq!(from_bulk.shape(), from_store.shape());
    assert!(from_bulk.hint_values() == from_store.hint_values(), "{context}: hint");

    // The hint does not depend on the number of threads that compute it.
    let cells = PirDatabase::cells_from_entries(&layout, &bulk.entries);
    let (threaded, _) = PirDatabase::from_cells_timed(&layout, cells, 3);
    assert!(threaded.hint_values() == from_store.hint_values(), "{context}: threaded hint");
}

#[test]
fn bulk_columns_equal_the_append_path() {
    for &(kappa, tree_height, leaves) in &[
        (2usize, 6usize, 1usize),
        (2, 6, 2),
        (2, 6, 37),
        (2, 6, 64),
        (4, 9, 300),
        (2, 50, 33),
        (4, 50, 257),
    ] {
        // One worker task for the whole log, and tasks of eight leaves on
        // three threads.
        assert_bulk_equals_append_path(kappa, tree_height, leaves, 1, CHUNK_BITS);
        assert_bulk_equals_append_path(kappa, tree_height, leaves, 3, 3);
    }
}

/// A log whose first `default_leaves` leaves are allocated and unwritten,
/// the rest written: the bulk columns of a default prefix equal the
/// append path's.
fn assert_default_prefix_equals_append_path(
    kappa: usize,
    tree_height: usize,
    leaves: usize,
    default_leaves: usize,
    chunk_bits: u32,
) {
    let params = build_params(kappa, 3 * kappa, 121);
    let auth = AuthParams::new(tree_height);
    let servers = build_servers(&params, &auth, leaves);
    let mut client = reader(&params, &auth, leaves, &servers);
    let updates = (default_leaves as u64..leaves as u64)
        .map(|index| (index, bulk_leaf_value(SEED, index)))
        .collect::<Vec<_>>();
    if !updates.is_empty() {
        client.write_batch(&updates).expect("write the written leaves");
    }

    let server = &servers[1];
    let guard = server.inner.lock().expect("server lock");
    let root = guard.store.root;
    let store = &guard.store.unauth;
    let layout = PrivateLayout::new(&params, &auth, NodeLinearization::Preorder, leaves as u64, 1)
        .expect("layout");
    let stored = (0..store.segments())
        .flat_map(|segment| record_entries(store, &layout, server.label, segment))
        .collect::<Vec<Fp>>();

    let bulk = BulkLog::build_with_default_prefix(
        &params,
        &auth,
        NodeLinearization::Preorder,
        leaves as u64,
        default_leaves as u64,
        SEED,
        server.label,
        3,
        chunk_bits,
    );
    let context = format!(
        "kappa {kappa}, height {tree_height}, {leaves} leaves, {default_leaves} default, chunks of 2^{chunk_bits}"
    );
    assert_eq!(bulk.root(), root, "{context}: root");
    assert_eq!(bulk.segments(), store.segments(), "{context}: segments");
    assert!(bulk.entries() == stored, "{context}: primary columns differ");
}

#[test]
fn default_prefix_columns_equal_the_append_path() {
    for &(kappa, tree_height, leaves, default_leaves) in &[
        (2usize, 6usize, 37usize, 37usize),
        (2, 6, 37, 20),
        (2, 6, 64, 1),
        (4, 9, 300, 256),
        (4, 9, 300, 300),
        (2, 50, 33, 33),
        (4, 50, 257, 100),
    ] {
        assert_default_prefix_equals_append_path(kappa, tree_height, leaves, default_leaves, CHUNK_BITS);
        assert_default_prefix_equals_append_path(kappa, tree_height, leaves, default_leaves, 3);
    }
}

#[test]
fn bulk_columns_are_independent_of_threads_and_chunks() {
    let params = build_params(2, 6, 122);
    let auth = AuthParams::new(20);
    // More than one default worker task of 2^16 leaves.
    let leaves = (1u64 << 16) + 12_345;
    let order = NodeLinearization::Preorder;
    let one = build_primary_columns(&params, &auth, order, leaves, SEED, 3, 1);
    let many = build_primary_columns(&params, &auth, order, leaves, SEED, 3, 5);
    let small = build_primary_columns_in_chunks(&params, &auth, order, leaves, SEED, 3, 4, 9);
    assert_eq!(one.root, many.root);
    assert!(one.entries == many.entries);
    assert_eq!(one.root, small.root);
    assert!(one.entries == small.entries);
}

#[test]
fn a_whole_tree_database_answers_with_the_cells_of_the_public_opening() {
    use paper_a_optimized::private::{PirClient, PirShape, build_tree_cells, entry_from_cells};
    use paper_a_optimized::{decode_value, merkle};

    for &(tree_height, leaves, cells_per_record) in &[(7usize, 100usize, 8usize), (50, 150, 16)] {
        // The reference: the same log written through coded servers.
        let params = build_params(2, 6, 126);
        let auth = AuthParams::new(tree_height);
        let servers = build_servers(&params, &auth, leaves);
        let mut client = reader(&params, &auth, leaves, &servers);
        let updates = (0..leaves as u64)
            .map(|index| (index, bulk_leaf_value(SEED, index)))
            .collect::<Vec<_>>();
        client.write_batch(&updates).expect("write the bulk log");
        let root = client.trusted_root().expect("root");

        let order = NodeLinearization::Preorder;
        let layout = PrivateLayout::whole_tree(&auth, order, leaves as u64, cells_per_record)
            .expect("layout");
        let (cells, tree_root) = build_tree_cells(&auth, order, SEED, &layout, 3);
        assert_eq!(tree_root, root);
        let database = PirDatabase::from_cells(&layout, cells);
        let shape = PirShape::of(&layout);
        let pir_client = PirClient::generate(shape);
        let pack_keys = pir_client.pack_keys();

        for leaf in [0u64, 1, leaves as u64 / 2, leaves as u64 - 1] {
            let query = pir_client.query(layout.pattern_of(leaf));
            let response = database.answer(&query, &pack_keys).expect("answer");
            let slots = pir_client.decode(&response).expect("decode");
            let (value, path) = client.read_with_path(leaf).expect("public opening");
            // The decoded cells are the leaf value and the stored siblings.
            let plan = layout.opening_plan(leaf);
            assert_eq!(
                entry_from_cells(&slots[plan[0].slot], plan[0].entry_in_record),
                decode_value(value)
            );
            for entry in &plan[1..] {
                let step = tree_height - entry.node.level as usize;
                assert_eq!(
                    entry_from_cells(&slots[entry.slot], entry.entry_in_record),
                    decode_value(path.siblings[step]),
                    "leaf {leaf}, sibling at level {}",
                    entry.node.level
                );
            }
            assert_eq!(
                merkle::recompute_root(
                    tree_height as u32,
                    leaf,
                    value,
                    &path.siblings,
                    &auth.defaults
                ),
                tree_root
            );
        }
    }
}

#[test]
fn an_extended_bulk_log_equals_a_build_at_the_longer_length() {
    use paper_a_optimized::private::BulkLog;
    let params = build_params(4, 12, 125);
    let auth = AuthParams::new(50);
    let order = NodeLinearization::Preorder;
    for chunk_bits in [3u32, 5] {
        // Lengths that end inside a worker task, on its boundary, and past a
        // power of two.
        let lengths = [1u64, 7, 8, 24, 29, 64, 65, 200, 256, 300];
        let mut log = BulkLog::build(&params, &auth, order, lengths[0], SEED, 7, 2, chunk_bits);
        for &leaves in &lengths {
            log.extend(leaves, 3);
            let fresh =
                build_primary_columns_in_chunks(&params, &auth, order, leaves, SEED, 7, 1, 9);
            assert_eq!(log.leaves(), leaves);
            assert_eq!(log.root(), fresh.root, "{leaves} leaves, chunks of 2^{chunk_bits}");
            assert_eq!(log.segments(), fresh.segments);
            assert!(
                log.entries() == fresh.entries.as_slice(),
                "{leaves} leaves, chunks of 2^{chunk_bits}: extended columns differ"
            );
        }
    }
}

#[test]
fn a_log_built_for_several_claims_equals_each_claim_s_own_build() {
    let params = build_params(4, 12, 125);
    let auth = AuthParams::new(50);
    let order = NodeLinearization::Preorder;
    let claims = [7usize, 0, 11, 3];
    for &(leaves, default_leaves) in &[(300u64, 0u64), (300, 300), (257, 100)] {
        let mut together = BulkLog::build_for_claims(
            &params, &auth, order, leaves, default_leaves, SEED, &claims, 3, 5,
        );
        let longer = leaves + 40;
        for extended in [leaves, longer] {
            together.extend(extended, 2);
            for (position, &claim) in claims.iter().enumerate() {
                let alone = BulkLog::build_with_default_prefix(
                    &params, &auth, order, extended, default_leaves, SEED, claim, 1, 9,
                );
                assert_eq!(together.root(), alone.root(), "{extended} leaves");
                assert!(
                    together.claim_entries(position) == alone.entries(),
                    "{extended} leaves, claim {claim}: columns differ"
                );
            }
        }
        let columns = together.into_claim_columns();
        assert_eq!(columns.len(), claims.len());
        for (position, &claim) in claims.iter().enumerate() {
            let alone = BulkLog::build_with_default_prefix(
                &params, &auth, order, longer, default_leaves, SEED, claim, 1, 9,
            );
            assert_eq!(columns[position].segments, alone.segments());
            assert_eq!(columns[position].root, alone.root());
            assert!(columns[position].entries == alone.entries(), "into_claim_columns");
        }
    }
}

#[test]
fn an_update_from_the_changed_records_equals_a_fresh_database() {
    let params = build_params(4, 12, 124);
    let auth = AuthParams::new(50);
    let order = NodeLinearization::Preorder;
    for &(old_leaves, new_leaves, segments_per_record) in &[
        (40u64, 41u64, 1usize),
        (100, 164, 1),
        (512, 530, 4),
        (1000, 1100, 2),
        // More than 128 new patterns: the sums of a slot are reduced on the
        // way.
        (100, 1500, 1),
    ] {
        let layout_at = |leaves: u64| {
            PrivateLayout::new(&params, &auth, order, leaves, segments_per_record).expect("layout")
        };
        let columns_at = |leaves: u64| {
            build_primary_columns_in_chunks(&params, &auth, order, leaves, SEED, 5, 2, 4)
        };
        let old = columns_at(old_leaves);
        let new = columns_at(new_leaves);
        let new_layout = layout_at(new_leaves);
        let fresh = PirDatabase::from_entries(&new_layout, &new.entries);
        let changed = new_layout.records_changed_by_append(old_leaves);
        assert!(changed.len() < new_layout.records() || new_layout.records() <= 3);

        let context = format!("{old_leaves} -> {new_leaves} leaves");
        let mut by_changed = PirDatabase::from_entries(&layout_at(old_leaves), &old.entries);
        by_changed.update_from_entries(&new_layout, &new.entries, Some(&changed));
        assert_eq!(by_changed.shape(), fresh.shape(), "{context}");
        assert!(
            by_changed.hint_values() == fresh.hint_values(),
            "{context}: update from the changed records"
        );
        let mut by_scan = PirDatabase::from_entries(&layout_at(old_leaves), &old.entries);
        by_scan.update_from_entries(&new_layout, &new.entries, None);
        assert!(
            by_scan.hint_values() == fresh.hint_values(),
            "{context}: update from all records"
        );
        for pattern in [0, new_layout.patterns() - 1] {
            assert_eq!(by_changed.row(pattern), fresh.row(pattern), "{context}");
            for (slot, id) in fresh.row(pattern).into_iter().enumerate() {
                if let Some(id) = id {
                    assert!(by_changed.slot_cells(slot, id) == fresh.slot_cells(slot, id));
                }
            }
        }
    }
}

/// Peak resident memory of this process, in MiB.
fn peak_rss_mib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|line| line.starts_with("VmHWM:"))
                .and_then(|line| line.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .map_or(0, |kib| kib / 1024)
}

/// Time and memory of the bulk builder, the preprocessing, and one answer on
/// a large log. Run with
/// `PRIVATE_PROBE_LEAVES_LOG2=27 cargo test --release --test private_bulk -- --ignored --nocapture`.
#[test]
#[ignore]
fn large_log_probe() {
    use paper_a_optimized::private::{PirClient, PirShape};
    use std::time::Instant;

    let leaves_log2: u32 = std::env::var("PRIVATE_PROBE_LEAVES_LOG2")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(22);
    let kappa: usize = std::env::var("PRIVATE_PROBE_KAPPA")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(64);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let leaves = 1u64 << leaves_log2;
    let params = build_params(kappa, kappa, 123);
    let auth = AuthParams::new(50);
    let order = NodeLinearization::Preorder;

    let start = Instant::now();
    let bulk = build_primary_columns(&params, &auth, order, leaves, SEED, 0, threads);
    println!(
        "leaves 2^{leaves_log2}, kappa {kappa}: bulk columns in {:.1} s on {threads} threads, {} segments, peak {} MiB",
        start.elapsed().as_secs_f64(),
        bulk.segments,
        peak_rss_mib()
    );

    let layout = PrivateLayout::new(&params, &auth, order, leaves, 1).expect("layout");
    let shape = PirShape::of(&layout);
    println!(
        "patterns {}, slots {}, record cells {}, database {} x {}",
        shape.patterns,
        shape.slots,
        shape.record_cells,
        shape.db_rows(),
        shape.db_cols()
    );
    let cells = PirDatabase::cells_from_entries(&layout, &bulk.entries);
    drop(bulk);
    let (database, times) = PirDatabase::from_cells_timed(&layout, cells, threads);
    println!(
        "preprocessing: hint {:.1} s, packing tables {:.1} s, peak {} MiB",
        times.hint.as_secs_f64(),
        times.packing.as_secs_f64(),
        peak_rss_mib()
    );

    let client = PirClient::generate(shape);
    let start = Instant::now();
    let pack_keys = client.pack_keys();
    let query = client.query(shape.patterns / 3);
    println!("client keys and query in {:.3} s", start.elapsed().as_secs_f64());
    let start = Instant::now();
    let first_pass = database.first_pass(&query).expect("first pass");
    println!("first pass in {:.3} s", start.elapsed().as_secs_f64());
    drop(first_pass);
    let start = Instant::now();
    let response = database.answer(&query, &pack_keys).expect("answer");
    println!("answer in {:.3} s", start.elapsed().as_secs_f64());
    let start = Instant::now();
    let slots = client.decode(&response).expect("decode");
    println!("decode in {:.3} s", start.elapsed().as_secs_f64());

    // The decoded row holds the records of the pattern.
    let row = layout.row(shape.patterns / 3);
    for (slot, item) in row.iter().enumerate() {
        if let Some(id) = item {
            let expected = database
                .slot_cells(slot, *id)
                .iter()
                .map(|&cell| cell as u64)
                .collect::<Vec<_>>();
            assert!(slots[slot] == expected, "slot {slot} decodes its item");
        }
    }
    println!("peak {} MiB", peak_rss_mib());
}
