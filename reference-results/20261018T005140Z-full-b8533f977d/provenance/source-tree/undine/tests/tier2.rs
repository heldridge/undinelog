//! Tier-2 correctness tests covering read/write behavior and allocation.

mod support;

use paper_a_optimized::{
    AuthConfig, AuthPath, CodedArrayConfig, Config, build_shard_registry_deterministic,
    random_value,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use support::setup;

#[test]
/// Smoke test: small read/write round-trip and path verification.
fn test_a_small_read_write() {
    let cfg = Config {
        auth: AuthConfig {
            tree_height: 1,
            alloc_leaf_batch: 2,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            kappa: 2,
            shard_count: 2,
            shard_registry: build_shard_registry_deterministic(2),
            party_registry_indices: vec![vec![0, 1]],
            seed: 7,
        },
    };
    let mut system = setup(cfg);
    let i = 0;
    let (old, path) = system.read(i).expect("read");
    let mut rng = ChaCha20Rng::seed_from_u64(1);
    let new = random_value(&mut rng);
    system.write(i, old, new, path).expect("write");
    let (after, new_path) = system.read(i).expect("read2");
    assert_eq!(after, new);
    let recomputed = paper_a_optimized::merkle::recompute_root(
        system.auth.tree_height as u32,
        i as u64,
        after,
        &new_path.siblings,
        &system.auth.defaults,
    );
    assert_eq!(recomputed, new_path.root);
}

#[test]
/// Read a larger tree and verify the returned path root.
fn test_b_larger_read() {
    let cfg = Config {
        auth: AuthConfig {
            tree_height: 5,
            alloc_leaf_batch: 32,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            kappa: 8,
            shard_count: 8,
            shard_registry: build_shard_registry_deterministic(8),
            party_registry_indices: vec![vec![0, 1, 2, 3, 4, 5, 6, 7]],
            seed: 9,
        },
    };
    let system = setup(cfg);
    let (value, path) = system.read(5).expect("read");
    let recomputed = paper_a_optimized::merkle::recompute_root(
        system.auth.tree_height as u32,
        5,
        value,
        &path.siblings,
        &system.auth.defaults,
    );
    assert_eq!(recomputed, path.root);
    assert_eq!(path.root, system.store.root);
}

#[test]
/// Reject writes if the auth path is tampered with.
fn test_c_bad_write_rejected() {
    let cfg = Config {
        auth: AuthConfig {
            tree_height: 1,
            alloc_leaf_batch: 2,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            kappa: 2,
            shard_count: 2,
            shard_registry: build_shard_registry_deterministic(2),
            party_registry_indices: vec![vec![0, 1]],
            seed: 11,
        },
    };
    let mut system = setup(cfg);
    let i = 1;
    let (old, mut path) = system.read(i).expect("read");
    let mut rng = ChaCha20Rng::seed_from_u64(2);
    let new = random_value(&mut rng);

    path.siblings[0][0] ^= 0x01;
    let root_before = system.store.root;
    assert!(system.write(i, old, new, path).is_err());
    assert_eq!(system.store.root, root_before);
}

#[test]
/// Reads fail when there are fewer than `need` shards available.
fn test_d_not_enough_shards() {
    let cfg = Config {
        auth: AuthConfig {
            tree_height: 5,
            alloc_leaf_batch: 32,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            kappa: 8,
            shard_count: 7,
            shard_registry: build_shard_registry_deterministic(7),
            party_registry_indices: vec![vec![0, 1, 2, 3, 4, 5, 6]],
            seed: 13,
        },
    };
    let system = setup(cfg);
    let err = system.read(3).expect_err("should fail");
    match err {
        paper_a_optimized::ReadError::NotEnoughShards => {}
        _ => panic!("unexpected error"),
    }
}

#[test]
/// Allocation gates reads/writes to the allocated segment only.
fn test_e_allocate_once_gates_access() {
    let cfg = Config {
        auth: AuthConfig {
            tree_height: 3,
            alloc_leaf_batch: 4,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            kappa: 2,
            shard_count: 8,
            shard_registry: build_shard_registry_deterministic(8),
            party_registry_indices: vec![vec![0, 1, 2, 3, 4, 5, 6, 7]],
            seed: 17,
        },
    };
    let mut system = setup(cfg);
    let (old, path) = system.read(3).expect("read in allocated range");
    let mut rng = ChaCha20Rng::seed_from_u64(77);
    let new = random_value(&mut rng);
    system
        .write(3, old, new, path)
        .expect("write in allocated range");

    let (value, path) = system.read(4).expect("read outside allocated range");
    let zero = paper_a_optimized::encode_value(paper_a_optimized::Fp::from(0u64));
    assert_eq!(value, zero);
    let recomputed = paper_a_optimized::merkle::recompute_root(
        system.auth.tree_height as u32,
        4,
        value,
        &path.siblings,
        &system.auth.defaults,
    );
    assert_eq!(recomputed, system.store.root);
    let err = system
        .write(
            4,
            old,
            new,
            AuthPath {
                index: 3,
                siblings: vec![],
                root: [0u8; 32],
            },
        )
        .expect_err("write outside allocated range");
    assert!(matches!(err, paper_a_optimized::WriteError::InvalidIndex));
}

#[test]
/// Allocation twice extends the readable/writable range.
fn test_f_allocate_twice_extends_range() {
    let cfg = Config {
        auth: AuthConfig {
            tree_height: 3,
            alloc_leaf_batch: 4,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            kappa: 2,
            shard_count: 8,
            shard_registry: build_shard_registry_deterministic(8),
            party_registry_indices: vec![vec![0, 1, 2, 3, 4, 5, 6, 7]],
            seed: 19,
        },
    };
    let mut system = setup(cfg);
    system.allocate_segment().expect("second allocation");
    let (old, path) = system.read(6).expect("read after second allocation");
    let mut rng = ChaCha20Rng::seed_from_u64(99);
    let new = random_value(&mut rng);
    system
        .write(6, old, new, path)
        .expect("write after second allocation");
    let (after, _) = system.read(6).expect("read after write");
    assert_eq!(after, new);
}

#[test]
/// Ensure roots remain consistent after writes.
fn test_g_root_consistency() {
    let cfg = Config {
        auth: AuthConfig {
            tree_height: 4,
            alloc_leaf_batch: 16,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            kappa: 4,
            shard_count: 8,
            shard_registry: build_shard_registry_deterministic(8),
            party_registry_indices: vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
            seed: 23,
        },
    };
    let mut system = setup(cfg);
    let (old, path) = system.read(5).expect("read");
    let mut rng = ChaCha20Rng::seed_from_u64(1234);
    let new = random_value(&mut rng);
    system.write(5, old, new, path).expect("write");
    assert_eq!(system.store.root, system.read(5).expect("read").1.root);
}
