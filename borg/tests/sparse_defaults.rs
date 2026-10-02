//! Tests for sparse Merkle defaults and level-only hash semantics.

mod support;

use paper_a_optimized::merkle::{build_default_hashes, default_hash, leaf_label, node_hash};
use paper_a_optimized::{
    AuthConfig, CodedArrayConfig, Config, DeltaMode, ShardRegEntry,
    build_shard_registry_deterministic, random_value,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::collections::HashMap;
use support::setup;

fn log2_pow2(n: usize) -> usize {
    assert!(n.is_power_of_two());
    usize::BITS as usize - 1 - n.leading_zeros() as usize
}

/// Build a deterministic shard registry for a given segment size.
fn build_registry(g: usize, s: usize, segment_len: usize) -> Vec<ShardRegEntry> {
    let n_grid = segment_len;
    build_shard_registry_deterministic(g, s, n_grid)
}

#[test]
/// Default hashes should depend only on the level, not position.
fn default_hashes_are_level_only() {
    let height = 6usize;
    let defaults = build_default_hashes(height);
    assert_eq!(defaults.by_level.len(), height + 1);
    for level in 0..height {
        let expected = node_hash(
            default_hash(&defaults, level + 1),
            default_hash(&defaults, level + 1),
        );
        assert_eq!(default_hash(&defaults, level), expected);
    }
}

#[test]
/// The empty tree root must match the level-0 default hash.
fn empty_root_matches_default() {
    let cfg = Config {
        auth: AuthConfig {
            tree_height: 3,
            alloc_leaf_batch: 8,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            segment_len: 16,
            s: 4,
            shard_count: 8,
            shard_registry: build_registry(8, 4, 16),
            party_registry_indices: vec![vec![0, 1, 2, 3, 4, 5, 6, 7]],
            seed: 77,
        },
    };
    let system = setup(cfg);
    let root_default = default_hash(&system.auth.defaults, 0);
    assert_eq!(system.store.root, root_default);
}

fn reference_root(
    defaults: &paper_a_optimized::DefaultHashes,
    height: u32,
    updates: &HashMap<u64, paper_a_optimized::Value256>,
) -> paper_a_optimized::Value256 {
    let mut updates_vec: Vec<(u64, paper_a_optimized::Value256)> =
        updates.iter().map(|(&i, &v)| (i, v)).collect();

    fn compute_node(
        level: u32,
        height: u32,
        updates: &[(u64, paper_a_optimized::Value256)],
        defaults: &paper_a_optimized::DefaultHashes,
    ) -> paper_a_optimized::Value256 {
        // Recursively compute the sparse Merkle root for the provided updates.
        if updates.is_empty() {
            return default_hash(defaults, level as usize);
        }
        if level == height {
            let (idx, value) = updates[0];
            return leaf_label(defaults, height, idx, value);
        }
        let shift = height - level - 1;
        let mut left = Vec::new();
        let mut right = Vec::new();
        for &(idx, value) in updates {
            if ((idx >> shift) & 1) == 0 {
                left.push((idx, value));
            } else {
                right.push((idx, value));
            }
        }
        let left_label = compute_node(level + 1, height, &left, defaults);
        let right_label = compute_node(level + 1, height, &right, defaults);
        node_hash(left_label, right_label)
    }

    if updates_vec.len() > 1 {
        updates_vec.sort_by_key(|(idx, _)| *idx);
    }
    compute_node(0, height, &updates_vec, defaults)
}

#[test]
/// Sparse updates applied through the system should match a reference root.
fn sparse_updates_match_reference_root() {
    let cfg = Config {
        auth: AuthConfig {
            tree_height: 4,
            alloc_leaf_batch: 8,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            segment_len: 16,
            s: 4,
            shard_count: 8,
            shard_registry: build_registry(8, 4, 16),
            party_registry_indices: vec![vec![0, 1, 2, 3, 4, 5, 6, 7]],
            seed: 99,
        },
    };
    let mut system = setup(cfg);
    system.allocate_segment().expect("allocate segment");
    let height = system.auth.tree_height as u32;
    let defaults = system.auth.defaults.clone();

    let mut rng = ChaCha20Rng::seed_from_u64(123);
    let mut updates: HashMap<u64, paper_a_optimized::Value256> = HashMap::new();
    for &idx in &[1usize, 3, 7, 12] {
        let (old, path) = system.read(idx).expect("read");
        let mut new = random_value(&mut rng);
        if new == old {
            new[0] ^= 0x01;
        }
        let update = system
            .build_broadcast_update(idx, old, new, path, DeltaMode::Aggregate)
            .expect("build update");
        system
            .apply_broadcast_update(&update)
            .expect("apply update");
        updates.insert(idx as u64, new);
    }

    let reference = reference_root(&defaults, height, &updates);
    assert_eq!(system.store.root, reference);
}

#[test]
/// Dense and sparse default roots should match for an empty tree.
fn dense_vs_sparse_empty_tree_matches() {
    let l_max = 8usize;
    let height = log2_pow2(l_max);
    let defaults = build_default_hashes(height);
    let root_sparse = default_hash(&defaults, 0);

    let mut level = vec![default_hash(&defaults, height); l_max];
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks_exact(2) {
            next.push(node_hash(pair[0], pair[1]));
        }
        level = next;
    }
    let root_dense = level[0];

    let cfg = Config {
        auth: AuthConfig {
            tree_height: height,
            alloc_leaf_batch: l_max,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            segment_len: 2 * l_max,
            s: 4,
            shard_count: 8,
            shard_registry: build_registry(8, 4, 2 * l_max),
            party_registry_indices: vec![vec![0, 1, 2, 3, 4, 5, 6, 7]],
            seed: 7,
        },
    };
    let system = setup(cfg);

    assert_eq!(root_dense, root_sparse);
    assert_eq!(system.store.root, root_sparse);
}
