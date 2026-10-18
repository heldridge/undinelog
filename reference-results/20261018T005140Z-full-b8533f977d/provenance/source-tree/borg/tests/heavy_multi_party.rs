//! End-to-end single-store tests for larger parameter sizes.

mod support;

use paper_a_optimized::DefaultHashes;
use paper_a_optimized::merkle::recompute_root;
use paper_a_optimized::{
    AuthConfig, CodedArrayConfig, Config, DeltaMode, ShardRegEntry, Value256,
    build_shard_registry_deterministic, random_value,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::time::Instant;
use support::{setup, setup_with_timing};

/// Verify that an authentication path matches the provided root.
fn verify_auth_path(
    height: u32,
    defaults: &DefaultHashes,
    index: usize,
    value: Value256,
    siblings: &[Value256],
    root: Value256,
) -> bool {
    recompute_root(height, index as u64, value, siblings, defaults) == root
}

/// Build a deterministic shard registry for tests.
fn build_registry(g: usize, s: usize, segment_len: usize) -> Vec<ShardRegEntry> {
    let n_grid = segment_len;
    build_shard_registry_deterministic(g, s, n_grid)
}

/// Log2 helper for power-of-two sizes.
fn log2_pow2(n: usize) -> usize {
    assert!(n.is_power_of_two());
    usize::BITS as usize - 1 - n.leading_zeros() as usize
}

/// Run a single-store read/write loop and validate invariants.
fn run_single_store(
    cfg: Config,
    k: usize,
    expected_n_grid: usize,
    expected_delta: usize,
    expected_need: usize,
) {
    let mut system = setup(cfg.clone());
    let mut rng = ChaCha20Rng::seed_from_u64(cfg.coded.seed);
    let leaf_count = system.allocated_leaves;

    let height = system.auth.tree_height as u32;
    let defaults = system.auth.defaults.clone();
    assert_eq!(system.params.n_grid, expected_n_grid);
    assert_eq!(system.params.delta, expected_delta);
    assert_eq!(system.params.need, expected_need);
    assert!(system.params.need > 1);
    assert_eq!(system.params.delta + 1, system.params.need);
    assert!(system.params.shard_registry.len() >= system.params.need);

    for _ in 0..k {
        let i = rng.gen_range(0..leaf_count);
        let (old, path) = system.read(i).expect("read");
        assert!(verify_auth_path(
            height,
            &defaults,
            i,
            old,
            &path.siblings,
            path.root
        ));

        let new = random_value(&mut rng);
        let update = system
            .build_broadcast_update(i, old, new, path.clone(), DeltaMode::Aggregate)
            .expect("build update");
        system
            .apply_broadcast_update(&update)
            .expect("apply update");

        let (after, path2) = system.read(i).expect("read2");
        assert_eq!(after, new);
        assert!(verify_auth_path(
            height,
            &defaults,
            i,
            after,
            &path2.siblings,
            path2.root
        ));
    }

    let i = rng.gen_range(0..leaf_count);
    let (value, path) = system.read(i).expect("final read");
    assert!(verify_auth_path(
        height,
        &defaults,
        i,
        value,
        &path.siblings,
        path.root
    ));

    assert_eq!(system.store.root, path.root);
}

#[test]
#[ignore]
/// Heavy end-to-end test for large parameters (ignored by default).
fn heavy_single_store_end_to_end() {
    // L ~= 1e6, but the spec requires power-of-two sizes.
    let l = 1_048_576usize; // 2^20
    let s = 1_024usize; // 2^10
    let g = 4_096usize;
    let party_registry_indices = vec![(0..g).collect()];
    let shard_registry = build_registry(g, s, 2 * l);
    let h = log2_pow2(l);
    let cfg = Config {
        auth: AuthConfig {
            tree_height: h,
            alloc_leaf_batch: l,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            segment_len: 2 * l,
            s,
            shard_count: g,
            shard_registry,
            party_registry_indices,
            seed: 42,
        },
    };

    let expected_n_grid = 2_097_152usize; // 2^21
    run_single_store(cfg, 5, expected_n_grid, 2_047, 2_048);
}

#[test]
/// Medium end-to-end test for CI-friendly sizes.
fn medium_single_store_end_to_end() {
    let l = 16_384usize; // 2^14
    let s = 256usize; // 2^8
    let g = 256usize;
    let party_registry_indices = vec![(0..g).collect()];
    let shard_registry = build_registry(g, s, 2 * l);
    let h = log2_pow2(l);
    let cfg = Config {
        auth: AuthConfig {
            tree_height: h,
            alloc_leaf_batch: l,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            segment_len: 2 * l,
            s,
            shard_count: g,
            shard_registry,
            party_registry_indices,
            seed: 42,
        },
    };

    let expected_n_grid = 32_768usize; // 2^15
    run_single_store(cfg, 3, expected_n_grid, 127, 128);
}

#[test]
#[ignore]
/// Timing-heavy setup test (ignored by default).
fn heavy_setup_timing_option_b() {
    let l = 1_048_576usize; // 2^20
    let s = 1_024usize; // 2^10
    let g = 256usize;
    let party_registry_indices = vec![(0..g).collect()];
    let shard_registry = build_registry(g, s, 2 * l);
    let h = log2_pow2(l);
    let cfg = Config {
        auth: AuthConfig {
            tree_height: h,
            alloc_leaf_batch: l,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            segment_len: 2 * l,
            s,
            shard_count: g,
            shard_registry,
            party_registry_indices,
            seed: 42,
        },
    };

    let total_start = Instant::now();
    let (system, timings) = setup_with_timing(cfg);
    let total = total_start.elapsed();

    println!("tree build: {:?}", timings.tree_build);
    println!("ifft: {:?}", timings.ifft);
    println!("shard eval: {:?}", timings.shard_eval);
    println!("row eval: {:?}", timings.row_eval);
    println!("total setup: {:?}", total);

    assert_eq!(system.store.root, system.auth.defaults.by_level[0]);
}
