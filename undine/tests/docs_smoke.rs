//! Documentation-aligned smoke tests for core invariants.
//!
//! This test intentionally exercises a small subset of the API to keep the
//! documentation and invariants anchored in code.

mod support;

use paper_a_optimized::auth::mapping;
use paper_a_optimized::merkle::{NodeKey, leaf_key, parent, sibling};
use paper_a_optimized::{
    AuthConfig, BatchWriteOpts, CodedArrayConfig, Config, DeltaMode, LeafUpdateBatch,
    MerkleTreeProof, Value256, build_shard_registry_deterministic, random_value,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::collections::HashMap;
use support::{sanity_dump, setup};

/// Build a frontier multiproof from multiple authentication paths.
fn build_proof_from_paths(
    height: u32,
    root: Value256,
    paths: &[(usize, Vec<Value256>)],
) -> MerkleTreeProof {
    let mut nodes_map: HashMap<NodeKey, Value256> = HashMap::new();
    for (index, siblings) in paths {
        let mut v = leaf_key(height, *index as u64);
        for &sib_content in siblings {
            let sib = sibling(v);
            if let Some(existing) = nodes_map.insert(sib, sib_content) {
                assert_eq!(existing, sib_content, "conflicting proof node {:?}", sib);
            }
            v = parent(v);
        }
    }
    let mut nodes: Vec<(NodeKey, Value256)> = nodes_map.into_iter().collect();
    nodes.sort_by_key(|(node_id, _)| *node_id);
    MerkleTreeProof { root, nodes }
}

#[test]
fn docs_smoke_invariants() {
    let h: usize = 3;
    let alloc_leaf_batch: usize = 4;
    let kappa: usize = 2;
    let g: usize = 4;
    let shard_registry = build_shard_registry_deterministic(g);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = Config {
        auth: AuthConfig {
            tree_height: h,
            alloc_leaf_batch,
            node_linearization: mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            kappa,
            shard_count: g,
            shard_registry,
            party_registry_indices,
            seed: 123,
        },
    };

    let mut system = setup(cfg);
    {
        let params = &system.params;
        // Basic parameter invariants tied to the documentation.
        assert!(system.alloc_leaf_batch.is_power_of_two());
        assert!(params.need.is_power_of_two());
        assert_eq!(params.need, kappa);
        assert_eq!(params.s, params.need);
        assert_eq!(params.n_grid, params.need * params.need);
        assert_eq!(params.segment_len, params.n_grid);

        // Cursor monotonicity and preorder roundtrip over the active prefix.
        let cursor0 = mapping::cursor(
            system.node_linearization,
            system.auth.tree_height as u16,
            system.allocated_leaves as u64,
        );
        for idx in 0..cursor0 {
            let (level, pos) = mapping::node_from_idx(
                system.node_linearization,
                system.auth.tree_height as u16,
                idx,
            );
            let idx2 = mapping::idx_if_active(
                system.node_linearization,
                system.auth.tree_height as u16,
                system.allocated_leaves as u64,
                level,
                pos,
            )
            .expect("active index");
            assert_eq!(idx, idx2);
        }
    }

    // Read + write round-trip on a single leaf.
    let (old, path) = system.read(0).expect("read leaf 0");
    let mut rng = ChaCha20Rng::seed_from_u64(999);
    let mut new = random_value(&mut rng);
    while new == old {
        new = random_value(&mut rng);
    }
    let update0 = system
        .build_broadcast_update(0, old, new, path, DeltaMode::Aggregate)
        .expect("build update leaf 0");
    system
        .apply_broadcast_update(&update0)
        .expect("apply update leaf 0");
    let (after, _) = system.read(0).expect("read leaf 0 after write");
    assert_eq!(after, new);

    // Batch write with a minimal multiproof.
    let (old0, path0) = system.read(0).expect("read leaf 0 for batch");
    let (old1, path1) = system.read(1).expect("read leaf 1 for batch");
    let mut new0 = random_value(&mut rng);
    if new0 == old0 {
        new0[0] ^= 0x01;
    }
    let mut new1 = random_value(&mut rng);
    if new1 == old1 {
        new1[0] ^= 0x02;
    }
    let updates = vec![
        LeafUpdateBatch {
            index: 0,
            old: old0,
            new: new0,
        },
        LeafUpdateBatch {
            index: 1,
            old: old1,
            new: new1,
        },
    ];
    let proof = build_proof_from_paths(
        system.auth.tree_height as u32,
        system.store.root,
        &[(0, path0.siblings), (1, path1.siblings)],
    );
    let update_batch = system
        .build_broadcast_update_batch(
            updates,
            proof,
            BatchWriteOpts {
                delta_mode: DeltaMode::Aggregate,
            },
        )
        .expect("build batch update");
    system
        .apply_broadcast_update_batch(&update_batch)
        .expect("apply batch update");
    let (after0, _) = system.read(0).expect("read leaf 0 after batch");
    let (after1, _) = system.read(1).expect("read leaf 1 after batch");
    assert_eq!(after0, new0);
    assert_eq!(after1, new1);

    // Allocation monotonicity and append-only segment counts.
    let first_key = system.params.shard_registry[0].key;
    let seg_len_before = system
        .store
        .unauth
        .coded
        .shard_family_segment_count(&first_key)
        .expect("missing shard");
    let cursor_before = mapping::cursor(
        system.node_linearization,
        system.auth.tree_height as u16,
        system.allocated_leaves as u64,
    );

    system.allocate_segment().expect("allocate segment 1");
    let seg_len_after = system
        .store
        .unauth
        .coded
        .shard_family_segment_count(&first_key)
        .expect("missing shard after allocation");
    let cursor_after = mapping::cursor(
        system.node_linearization,
        system.auth.tree_height as u16,
        system.allocated_leaves as u64,
    );
    assert!(seg_len_after >= seg_len_before);
    assert!(cursor_after >= cursor_before);

    // Optional verbose output if SANITY=1.
    sanity_dump(&system, &[0, 1, alloc_leaf_batch - 1]);
}
