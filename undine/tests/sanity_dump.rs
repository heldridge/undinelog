//! Sanity checks for allocation/mapping plus optional debug output.

mod support;

use paper_a_optimized::auth::mapping;
use paper_a_optimized::{
    AuthConfig, CodedArrayConfig, Config, build_shard_registry_deterministic, merkle::leaf_key,
};
use support::{System, sanity_dump, setup};

/// Assert that preorder mapping and segment counts are internally consistent.
fn assert_mapping(system: &System) {
    let params = &system.params;
    let h = system.auth.tree_height as u16;
    let k = system.allocated_leaves as u64;
    let cursor = mapping::cursor(system.node_linearization, h, k);
    let segment_span = paper_a_optimized::unauth::public_values_per_segment(params) as u64;
    let needed_segments = cursor / segment_span + u64::from(cursor % segment_span != 0);

    let seg_count = system
        .store
        .unauth
        .coded
        .shard_keys()
        .next()
        .and_then(|key| system.store.unauth.coded.shard_family_segment_count(key))
        .unwrap_or(0);
    assert_eq!(seg_count as u64, needed_segments);

    for idx in 0..cursor {
        let (level, pos) = mapping::node_from_idx(system.node_linearization, h, idx);
        let idx2 = mapping::idx_if_active(system.node_linearization, h, k, level, pos)
            .expect("active index");
        assert_eq!(idx2, idx);
        let seg = idx / segment_span;
        let off = idx % segment_span;
        assert_eq!(idx, seg * segment_span + off);
    }

    let sample_leaf = if k > 0 { k - 1 } else { 0 };
    let mut key = leaf_key(system.auth.tree_height as u32, sample_leaf);
    loop {
        let idx =
            mapping::idx_if_active(system.node_linearization, h, k, key.level as u16, key.pos)
                .expect("active path node");
        let seg = idx / segment_span;
        let off = idx % segment_span;
        let (lvl2, pos2) =
            mapping::node_from_idx(system.node_linearization, h, seg * segment_span + off);
        assert_eq!(lvl2 as u32, key.level);
        assert_eq!(pos2, key.pos);
        if key.level == 0 {
            break;
        }
        key = paper_a_optimized::merkle::parent(key);
    }
}

#[test]
/// Ensure sanity_dump paths remain valid across allocations.
fn sanity_dump_alloc() {
    let h: usize = 4;
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
            seed: 0,
        },
    };
    let mut system = setup(cfg);
    sanity_dump(&system, &[0, 1, 2, alloc_leaf_batch - 1]);
    assert_mapping(&system);

    system.allocate_segment().expect("second allocation");
    sanity_dump(&system, &[0, 1, 2, alloc_leaf_batch - 1]);
    assert_mapping(&system);
}

#[test]
/// Exhaustively check preorder mapping roundtrip on a small tree.
fn preorder_roundtrip_small_tree() {
    let h: u16 = 4;
    let total = mapping::subtree_size(0, h) as u64;
    for idx in 0..total {
        let (level, pos) = mapping::node_from_preorder_idx(h, idx);
        let idx2 = mapping::preorder_idx(h, level, pos);
        assert_eq!(idx, idx2);
    }
}
