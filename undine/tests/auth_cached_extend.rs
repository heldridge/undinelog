use paper_a_optimized::auth::mapping::{self, NodeLinearization};
use paper_a_optimized::auth::service::AuthServer;
use paper_a_optimized::unauth::{ClaimBundle, UnauthStore};
use paper_a_optimized::{
    AuthParams, CodedArrayConfig, build_coded_params, build_shard_registry_deterministic,
    default_hash, encode_value, fp_zero,
};

fn build_params() -> paper_a_optimized::CodedArrayParams {
    let kappa = 2usize;
    let shard_count = 4usize;
    let shard_registry = build_shard_registry_deterministic(shard_count);
    let cfg = CodedArrayConfig {
        kappa,
        shard_count,
        shard_registry,
        party_registry_indices: vec![(0..shard_count).collect()],
        seed: 11,
    };
    build_coded_params(&cfg)
}

fn auth_default_segment(
    auth: &AuthParams,
    params: &paper_a_optimized::CodedArrayParams,
    node_linearization: NodeLinearization,
    seg: usize,
) -> Vec<[u8; 32]> {
    let zero = encode_value(fp_zero());
    let mut t_seg = vec![zero; params.n_grid];
    let height = auth.tree_height as u16;
    let total_nodes = (1u128 << ((auth.tree_height + 1) as u32)) - 1;
    let total_nodes_u64 = Some(total_nodes as u64);
    let seg_base = seg as u64 * paper_a_optimized::unauth::public_values_per_segment(params) as u64;
    for logical_offset in 0..paper_a_optimized::unauth::public_values_per_segment(params) {
        let idx = seg_base + logical_offset as u64;
        let grid_index =
            paper_a_optimized::unauth::logical_offset_to_grid_index(params, logical_offset);
        if let Some(total) = total_nodes_u64 {
            if idx >= total {
                t_seg[grid_index] = zero;
                continue;
            }
        }
        let (level, _) = mapping::node_from_idx(node_linearization, height, idx);
        if level == height {
            t_seg[grid_index] = zero;
            continue;
        }
        t_seg[grid_index] = default_hash(&auth.defaults, level as usize);
    }
    t_seg
}

#[test]
fn auth_server_cached_extend_matches_generic_full_materialization() {
    let params = build_params();
    let auth = AuthParams::new(5);
    let node_linearization = NodeLinearization::Preorder;
    let allocated_leaves = 13usize;
    let cursor = mapping::cursor(
        node_linearization,
        auth.tree_height as u16,
        allocated_leaves as u64,
    );

    let server = AuthServer::new(
        params.clone(),
        auth.clone(),
        node_linearization,
        allocated_leaves,
    )
    .expect("auth server");

    let mut oracle = UnauthStore::new(params.clone());
    oracle
        .extend_with_segments(cursor, |seg| {
            auth_default_segment(&auth, &params, node_linearization, seg)
        })
        .expect("oracle extend");

    assert_eq!(server.store.unauth.len(), oracle.len());
    assert_eq!(server.store.unauth.segments(), oracle.segments());
    assert!(
        server
            .store
            .unauth
            .coded
            .equals_for_tests(&params, &oracle.coded)
    );
}

#[test]
fn auth_server_cached_extend_matches_generic_filtered_materialization() {
    let params = build_params();
    let auth = AuthParams::new(5);
    let node_linearization = NodeLinearization::Preorder;
    let allocated_leaves = 13usize;
    let assigned_indices = vec![0usize, 2usize];
    let cursor = mapping::cursor(
        node_linearization,
        auth.tree_height as u16,
        allocated_leaves as u64,
    );

    let server = AuthServer::new_with_assignments(
        params.clone(),
        auth.clone(),
        node_linearization,
        allocated_leaves,
        Some(assigned_indices.clone()),
    )
    .expect("auth server");

    let mut oracle = UnauthStore::new(params.clone());
    oracle
        .extend_with_segments_filtered(cursor, &assigned_indices, |seg| {
            auth_default_segment(&auth, &params, node_linearization, seg)
        })
        .expect("oracle extend");

    assert_eq!(server.store.unauth.len(), oracle.len());
    assert_eq!(server.store.unauth.segments(), oracle.segments());
    assert!(
        server
            .store
            .unauth
            .coded
            .equals_for_tests(&params, &oracle.coded)
    );
}

#[test]
fn auth_server_cached_extend_matches_generic_claim_materialization() {
    let params = build_params();
    let auth = AuthParams::new(5);
    let node_linearization = NodeLinearization::Preorder;
    let allocated_leaves = 13usize;
    let claims = vec![
        ClaimBundle::exact_for_registry_index(&params, 0).expect("claim 0"),
        ClaimBundle::exact_for_registry_index(&params, 1).expect("claim 1"),
    ];
    let cursor = mapping::cursor(
        node_linearization,
        auth.tree_height as u16,
        allocated_leaves as u64,
    );

    let server = AuthServer::new_with_assignments_and_claims(
        params.clone(),
        auth.clone(),
        node_linearization,
        allocated_leaves,
        None,
        Some(claims.clone()),
    )
    .expect("auth server");

    let mut oracle = UnauthStore::new_with_claims(params.clone(), claims.clone());
    oracle
        .extend_with_segments_for_claims(cursor, &claims, |seg| {
            auth_default_segment(&auth, &params, node_linearization, seg)
        })
        .expect("oracle extend");

    assert_eq!(server.store.unauth.len(), oracle.len());
    assert_eq!(server.store.unauth.segments(), oracle.segments());
    assert!(
        server
            .store
            .unauth
            .coded
            .equals_for_tests(&params, &oracle.coded)
    );
}
