//! Ensure preorder indices remain stable as segments are appended.

mod support;

use paper_a_optimized::{
    AuthConfig, CodedArrayConfig, Config, DeltaMode, Fp, ShardKey, Value256,
    build_shard_registry_deterministic, random_value,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use support::{System, setup};

/// Snapshot of a single shard vector for stability checks.
struct StoreSnapshot {
    root: Value256,
    key: ShardKey,
    segments: Vec<Vec<Fp>>,
}

/// Capture the store's first shard vector.
fn snapshot_store(system: &System) -> StoreSnapshot {
    let key = system.params.shard_registry[0].key;
    let segments = system
        .store
        .unauth
        .coded
        .clone_shard_family_segments(&system.params, &key)
        .expect("missing shard");
    StoreSnapshot {
        root: system.store.root,
        key,
        segments,
    }
}

/// Assert segment data was appended without mutation.
fn assert_append_only(system: &System, snapshot: &StoreSnapshot) {
    assert_eq!(system.store.root, snapshot.root);
    let after_segments = system
        .store
        .unauth
        .coded
        .clone_shard_family_segments(&system.params, &snapshot.key)
        .expect("missing shard after allocation");
    assert!(after_segments.len() >= snapshot.segments.len());
    assert_eq!(
        &after_segments[..snapshot.segments.len()],
        &snapshot.segments[..]
    );
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
/// Append segments and verify old indices/values remain stable.
fn preorder_indices_stable_under_allocation() {
    let h: usize = 3;
    let alloc_leaf_batch: usize = 2;
    let kappa: usize = 2;
    let g: usize = 4;

    let shard_registry = build_shard_registry_deterministic(g);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = Config {
        auth: AuthConfig {
            tree_height: h,
            alloc_leaf_batch,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            kappa,
            shard_count: g,
            shard_registry,
            party_registry_indices,
            seed: 0,
        },
    };
    let mut system = setup(cfg);
    let mut rng = ChaCha20Rng::seed_from_u64(123);

    let (old0, path0) = system.read(0).expect("read before write 0");
    let new0 = distinct_value(&mut rng, old0);
    let update0 = system
        .build_broadcast_update(0, old0, new0, path0, DeltaMode::Aggregate)
        .expect("build update 0");
    system
        .apply_broadcast_update(&update0)
        .expect("apply update 0");

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

    let snapshot0 = snapshot_store(&system);
    system.allocate_segment().expect("allocate chunk 1");
    assert_append_only(&system, &snapshot0);
    let (read0, _) = system.read(0).expect("read 0 after chunk 1");
    assert_eq!(read0, new0);
    let (read1, _) = system
        .read(last_leaf_chunk0)
        .expect("read last chunk 0 after chunk 1");
    assert_eq!(read1, new1);

    let snapshot1 = snapshot_store(&system);
    system.allocate_segment().expect("allocate chunk 2");
    assert_append_only(&system, &snapshot1);
    let (read0_again, _) = system.read(0).expect("read 0 after chunk 2");
    assert_eq!(read0_again, new0);
    let (read1_again, _) = system
        .read(last_leaf_chunk0)
        .expect("read last chunk 0 after chunk 2");
    assert_eq!(read1_again, new1);
}
