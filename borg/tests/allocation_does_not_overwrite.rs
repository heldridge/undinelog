//! Ensure append-only allocation never overwrites existing shard data.

mod support;

use paper_a_optimized::{
    AuthConfig, CodedArrayConfig, Config, DeltaMode, build_shard_registry_deterministic,
    random_value,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use support::setup;

#[test]
/// Allocate a new segment and ensure previous shard vectors remain unchanged.
fn allocation_does_not_overwrite() {
    let h: usize = 32;
    let alloc_leaf_batch: usize = 1 << 10;
    let segment_len: usize = 2 * alloc_leaf_batch;
    let n_grid = segment_len;
    let mut s: usize = 256;
    let g: usize = 8;
    // Keep g small while ensuring params.need <= g for reads at this N_grid.
    while (n_grid - 1) / s + 1 > g {
        s *= 2;
    }
    let shard_registry = build_shard_registry_deterministic(g, s, n_grid);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = Config {
        auth: AuthConfig {
            tree_height: h,
            alloc_leaf_batch,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
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

    let root_before = system.store.root;
    let mut rng = ChaCha20Rng::seed_from_u64(123);

    let (old0, path0) = system.read(0).expect("read before write 0");
    let mut new0 = random_value(&mut rng);
    while new0 == old0 {
        new0 = random_value(&mut rng);
    }
    let update0 = system
        .build_broadcast_update(0, old0, new0, path0, DeltaMode::Aggregate)
        .expect("build update 0");
    system
        .apply_broadcast_update(&update0)
        .expect("apply update 0");
    let root_after_first = system.store.root;
    assert_ne!(root_after_first, root_before);

    let last_leaf = alloc_leaf_batch - 1;
    let (old1, path1) = system.read(last_leaf).expect("read before write last");
    let mut new1 = random_value(&mut rng);
    while new1 == old1 {
        new1 = random_value(&mut rng);
    }
    let update1 = system
        .build_broadcast_update(last_leaf, old1, new1, path1, DeltaMode::Aggregate)
        .expect("build update last");
    system
        .apply_broadcast_update(&update1)
        .expect("apply update last");

    let root_snapshot = system.store.root;
    let snapshot_count = std::cmp::min(g, 4);
    let mut snapshots = Vec::with_capacity(snapshot_count);
    for idx in 0..snapshot_count {
        let entry = system.params.shard_registry[idx];
        let segments = system
            .store
            .unauth
            .coded
            .clone_shard_family_segments(&system.params, &entry.key)
            .expect("missing shard");
        let old_len = segments.len();
        snapshots.push((entry.key, segments, old_len));
    }

    system.allocate_segment().expect("second allocation");

    assert_eq!(system.store.root, root_snapshot);
    let (read0, _) = system.read(0).expect("read 0 after allocation");
    assert_eq!(read0, new0);
    let (read1, _) = system.read(last_leaf).expect("read last after allocation");
    assert_eq!(read1, new1);

    for (key, before_segments, old_len) in snapshots {
        let after_segments = system
            .store
            .unauth
            .coded
            .clone_shard_family_segments(&system.params, &key)
            .expect("missing shard after allocation");
        assert!(after_segments.len() >= old_len);
        assert_eq!(&after_segments[..old_len], &before_segments[..]);
    }
}
