//! Allocation append-only test with multiple segments and writes.

mod support;

use paper_a_optimized::auth::mapping;
use paper_a_optimized::{
    AuthConfig, CodedArrayConfig, Config, DeltaMode, Fp, ShardKey, Value256,
    build_shard_registry_deterministic, random_value,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use support::{System, sanity_dump, setup};

/// Snapshot of store state for append-only verification.
struct StoreSnapshot {
    root: Value256,
    segments: Vec<(ShardKey, Vec<Vec<Fp>>)>,
}

/// Compute the expected number of segments from the preorder cursor.
fn expected_segment_count(system: &System) -> usize {
    let height = system.auth.tree_height as u16;
    let cursor = mapping::cursor(
        system.node_linearization,
        height,
        system.allocated_leaves as u64,
    );
    let segment_len = system.params.segment_len as u64;
    let needed = cursor / segment_len + u64::from(cursor % segment_len != 0);
    usize::try_from(needed).expect("segment count fits usize")
}

/// Capture a subset of shard vectors for the store.
fn snapshot_store(system: &System, max_keys: usize) -> StoreSnapshot {
    let root = system.store.root;
    let total = system.params.shard_registry.len();
    let sample_count = std::cmp::min(max_keys, total);
    let mut segments = Vec::with_capacity(sample_count);
    for entry in system
        .params
        .shard_registry
        .iter()
        .copied()
        .take(sample_count)
    {
        let shard_segments = system
            .store
            .unauth
            .coded
            .clone_shard_family_segments(&system.params, &entry.key)
            .expect("missing shard");
        segments.push((entry.key, shard_segments));
    }
    StoreSnapshot { root, segments }
}

/// Assert that allocation appended segments without overwriting existing data.
fn assert_append_only(system: &System, snapshot: &StoreSnapshot) {
    assert_eq!(system.store.root, snapshot.root);
    let expected_segments = expected_segment_count(system);
    for (key, before_segments) in snapshot.segments.iter() {
        let after_segments = system
            .store
            .unauth
            .coded
            .clone_shard_family_segments(&system.params, key)
            .expect("missing shard after allocation");
        assert!(after_segments.len() >= before_segments.len());
        assert_eq!(
            &after_segments[..before_segments.len()],
            &before_segments[..]
        );
        assert_eq!(after_segments.len(), expected_segments);
    }
}

/// Sample a value distinct from `old`.
fn distinct_value(rng: &mut ChaCha20Rng, old: Value256) -> Value256 {
    let mut new = random_value(rng);
    while new == old {
        new = random_value(rng);
    }
    new
}

#[test]
/// Allocate multiple segments and ensure previous values remain stable.
fn allocation_append_only() {
    let h: usize = 32;
    let alloc_leaf_batch: usize = 1 << 10;
    let segment_len: usize = 2 * alloc_leaf_batch;
    let n_grid = segment_len;
    let mut s: usize = 256;
    let g: usize = 8;
    while (n_grid - 1) / s + 1 > g {
        s *= 2;
    }

    let shard_registry = build_shard_registry_deterministic(g, s, n_grid);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = Config {
        auth: AuthConfig {
            tree_height: h,
            alloc_leaf_batch,
            node_linearization: mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            segment_len,
            s,
            shard_count: g,
            shard_registry,
            party_registry_indices,
            seed: 0,
        },
    };

    let mut system = setup(cfg);
    let mut rng = ChaCha20Rng::seed_from_u64(123);
    let snapshot_count = std::cmp::min(g, 8);

    let root_before = system.store.root;
    let (old0, path0) = system.read(0).expect("read before write 0");
    let new0 = distinct_value(&mut rng, old0);
    let update0 = system
        .build_broadcast_update(0, old0, new0, path0, DeltaMode::Aggregate)
        .expect("build update 0");
    system
        .apply_broadcast_update(&update0)
        .expect("apply update 0");
    let root_after_first = system.store.root;
    assert_ne!(root_after_first, root_before);

    let last_leaf_chunk0 = alloc_leaf_batch - 1;
    let (old1, path1) = system
        .read(last_leaf_chunk0)
        .expect("read before write last chunk 0");
    let new1 = distinct_value(&mut rng, old1);
    let update1 = system
        .build_broadcast_update(last_leaf_chunk0, old1, new1, path1, DeltaMode::Aggregate)
        .expect("build update last chunk 0");
    system
        .apply_broadcast_update(&update1)
        .expect("apply update last chunk 0");
    let root_after_second = system.store.root;
    assert_ne!(root_after_second, root_after_first);

    let snapshot0 = snapshot_store(&system, snapshot_count);
    sanity_dump(
        &system,
        &[
            0,
            last_leaf_chunk0,
            alloc_leaf_batch,
            2 * alloc_leaf_batch - 1,
        ],
    );

    system.allocate_segment().expect("allocate chunk 1");

    assert_append_only(&system, &snapshot0);
    let (read0, _) = system.read(0).expect("read 0 after chunk 1");
    assert_eq!(read0, new0);
    let (read1, _) = system
        .read(last_leaf_chunk0)
        .expect("read last chunk 0 after chunk 1");
    assert_eq!(read1, new1);

    let first_leaf_chunk1 = alloc_leaf_batch;
    let last_leaf_chunk1 = 2 * alloc_leaf_batch - 1;
    let root_before_chunk1_write = system.store.root;
    let (old2, path2) = system
        .read(first_leaf_chunk1)
        .expect("read before write first chunk 1");
    let new2 = distinct_value(&mut rng, old2);
    let update2 = system
        .build_broadcast_update(first_leaf_chunk1, old2, new2, path2, DeltaMode::Aggregate)
        .expect("build update first chunk 1");
    system
        .apply_broadcast_update(&update2)
        .expect("apply update first chunk 1");
    let root_after_chunk1_first = system.store.root;
    assert_ne!(root_after_chunk1_first, root_before_chunk1_write);

    let (old3, path3) = system
        .read(last_leaf_chunk1)
        .expect("read before write last chunk 1");
    let new3 = distinct_value(&mut rng, old3);
    let update3 = system
        .build_broadcast_update(last_leaf_chunk1, old3, new3, path3, DeltaMode::Aggregate)
        .expect("build update last chunk 1");
    system
        .apply_broadcast_update(&update3)
        .expect("apply update last chunk 1");
    let root_after_chunk1_second = system.store.root;
    assert_ne!(root_after_chunk1_second, root_after_chunk1_first);

    let (read2, _) = system
        .read(first_leaf_chunk1)
        .expect("read first chunk 1 after writes");
    assert_eq!(read2, new2);
    let (read3, _) = system
        .read(last_leaf_chunk1)
        .expect("read last chunk 1 after writes");
    assert_eq!(read3, new3);

    let snapshot1 = snapshot_store(&system, snapshot_count);
    sanity_dump(
        &system,
        &[0, last_leaf_chunk0, first_leaf_chunk1, last_leaf_chunk1],
    );

    system.allocate_segment().expect("allocate chunk 2");

    assert_append_only(&system, &snapshot1);
    let (read0_again, _) = system.read(0).expect("read 0 after chunk 2");
    assert_eq!(read0_again, new0);
    let (read1_again, _) = system
        .read(last_leaf_chunk0)
        .expect("read last chunk 0 after chunk 2");
    assert_eq!(read1_again, new1);
    let (read2_again, _) = system
        .read(first_leaf_chunk1)
        .expect("read first chunk 1 after chunk 2");
    assert_eq!(read2_again, new2);
    let (read3_again, _) = system
        .read(last_leaf_chunk1)
        .expect("read last chunk 1 after chunk 2");
    assert_eq!(read3_again, new3);
}
