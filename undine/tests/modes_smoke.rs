//! Smoke test covering both unauthenticated and authenticated modes.

use paper_a_optimized::auth::service::AuthServer;
use paper_a_optimized::auth::{AuthParams, mapping};
use paper_a_optimized::coded_array::params::CodedArrayConfig;
use paper_a_optimized::merkle::leaf_key;
use paper_a_optimized::net::wire::{
    AuthReadRequest, NetMaybeFpBlob, UnauthOptimizedReadRequest as NetUnauthOptimizedReadRequest,
    UnauthReadObjectId as NetUnauthReadObjectId,
};
use paper_a_optimized::net::{Message, Response};
use paper_a_optimized::unauth::{DeltaMode, UnauthDecodeParams, UnauthStore};
use paper_a_optimized::{Fp, ShardShare, build_shard_registry_deterministic, decode_value};

fn decode_from_shares(
    params: &paper_a_optimized::CodedArrayParams,
    idx: u64,
    owned_indices: &[usize],
    values: &NetMaybeFpBlob,
) -> Fp {
    let shares: Vec<ShardShare> = owned_indices
        .iter()
        .zip(values.0.iter().copied())
        .map(|(&registry_index, value)| ShardShare::new(registry_index, value))
        .collect();
    params
        .decode_index_from_shares(idx, &shares)
        .expect("decode from shares")
}

#[test]
fn modes_smoke_basic() {
    let kappa: usize = 2;
    let initial_len: u64 = 4;
    let alloc_leaf_batch: usize = 4;
    let g = 2;
    let shard_registry = build_shard_registry_deterministic(g);
    let party_registry_indices = vec![vec![0, 1]];
    let cfg = CodedArrayConfig {
        kappa,
        shard_count: g,
        shard_registry: shard_registry.clone(),
        party_registry_indices: party_registry_indices.clone(),
        seed: 42,
    };
    let params = paper_a_optimized::build_coded_params(&cfg);

    // Unauthenticated: apply delta and decode.
    let mut unauth_store = UnauthStore::new(params.clone());
    unauth_store.extend(initial_len).expect("extend unauth");
    unauth_store
        .apply_deltas(&[(1, Fp::from(9u64))], DeltaMode::Aggregate)
        .expect("apply delta");
    let value = decode_value(unauth_store.read(1).expect("read unauth"));
    assert_eq!(value, Fp::from(9u64));

    // Authenticated: read a default leaf from a single party.
    let auth = AuthParams::new(3);
    let mut auth_party = AuthServer::new(
        params.clone(),
        auth,
        mapping::NodeLinearization::Preorder,
        alloc_leaf_batch,
    )
    .expect("auth party");
    let height = auth_party.auth.tree_height as u32;
    let leaf = 0u64;
    let resp = match paper_a_optimized::net::auth::handle_message(
        &mut auth_party,
        Message::AuthRead {
            req: AuthReadRequest {
                leaf,
                kappa: kappa as u64,
            },
        },
    ) {
        Response::AuthRead { resp } => resp,
        other => panic!("unexpected response: {:?}", other),
    };
    let leaf_node = leaf_key(height, leaf);
    let idx = mapping::idx_if_active(
        mapping::NodeLinearization::Preorder,
        auth_party.auth.tree_height as u16,
        alloc_leaf_batch as u64,
        leaf_node.level as u16,
        leaf_node.pos,
    )
    .expect("active leaf");
    let value = decode_from_shares(
        &params,
        idx,
        &party_registry_indices[0],
        &resp.nodes[0].values,
    );
    assert_eq!(value, Fp::from(0u64));
}

#[test]
fn auth_server_exposes_unauth_maintenance_but_not_block_api() {
    let kappa: usize = 2;
    let alloc_leaf_batch: usize = 4;
    let g = 2;
    let shard_registry = build_shard_registry_deterministic(g);
    let party_registry_indices = vec![vec![0, 1]];
    let cfg = CodedArrayConfig {
        kappa,
        shard_count: g,
        shard_registry,
        party_registry_indices,
        seed: 77,
    };
    let params = paper_a_optimized::build_coded_params(&cfg);

    let auth = AuthParams::new(3);
    let mut auth_party = AuthServer::new(
        params.clone(),
        auth,
        mapping::NodeLinearization::Preorder,
        alloc_leaf_batch,
    )
    .expect("auth party");

    let resp =
        paper_a_optimized::net::unauth::handle_message(&mut auth_party, Message::UnauthGetLen);
    assert!(matches!(resp, Response::UnauthLen { .. }));

    let resp = paper_a_optimized::net::unauth::handle_message(
        &mut auth_party,
        Message::UnauthRepairShard {
            seg: 0,
            registry_index: 0,
        },
    );
    let Response::UnauthRepairShard { values } = resp else {
        panic!("unexpected response");
    };
    assert_eq!(values.len(), kappa);

    let resp = paper_a_optimized::net::unauth::handle_message(
        &mut auth_party,
        Message::UnauthReadBlock { block: 0 },
    );
    assert!(matches!(resp, Response::Error { .. }));

    let resp =
        paper_a_optimized::net::unauth::handle_message(&mut auth_party, Message::UnauthKappaDouble);
    assert!(matches!(resp, Response::UnauthKappaDouble { ok: true }));
    assert_eq!(auth_party.params.need, 2 * kappa);
    assert_eq!(auth_party.store.unauth.params.need, 2 * kappa);
}

/// A party at threshold `kappa` answers network reads that name `2 kappa`
/// exactly as a party that stores the same array at `2 kappa`.
#[test]
fn network_reads_at_a_higher_threshold_match_a_party_stored_at_it() {
    let kappa: usize = 2;
    let alloc_leaf_batch: usize = 6;
    let g = 8;
    let params = paper_a_optimized::build_coded_params(&CodedArrayConfig {
        kappa,
        shard_count: g,
        shard_registry: build_shard_registry_deterministic(g),
        party_registry_indices: vec![(0..g).collect()],
        seed: 91,
    });
    let target = paper_a_optimized::unauth::build_params_for_kappa(&params, 2 * kappa)
        .expect("target params");
    let build_party = |params: &paper_a_optimized::CodedArrayParams| {
        AuthServer::new(
            params.clone(),
            AuthParams::new(4),
            mapping::NodeLinearization::Preorder,
            alloc_leaf_batch,
        )
        .expect("auth party")
    };
    let mut source_party = build_party(&params);
    let mut target_party = build_party(&target);

    for leaf in 0..alloc_leaf_batch as u64 {
        let message = Message::AuthRead {
            req: AuthReadRequest {
                leaf,
                kappa: target.need as u64,
            },
        };
        let read = |party: &mut AuthServer| match paper_a_optimized::net::auth::handle_message(
            party,
            message.clone(),
        ) {
            Response::AuthRead { resp } => resp
                .nodes
                .into_iter()
                .map(|node| ((node.node.level, node.node.pos), node.values))
                .collect::<Vec<_>>(),
            other => panic!("unexpected response: {:?}", other),
        };
        let on_the_fly = read(&mut source_party);
        assert!(on_the_fly.iter().any(|(_, values)| !values.0.is_empty()));
        assert_eq!(on_the_fly, read(&mut target_party));

        let read_only =
            |party: &AuthServer| match paper_a_optimized::net::auth::handle_read_only_message(
                party,
                message.clone(),
            ) {
                Some(Response::AuthRead { resp }) => resp
                    .nodes
                    .into_iter()
                    .map(|node| ((node.node.level, node.node.pos), node.values))
                    .collect::<Vec<_>>(),
                other => panic!("unexpected response: {:?}", other),
            };
        assert_eq!(read_only(&source_party), on_the_fly);
    }

    let message = Message::UnauthOptimizedRead {
        req: NetUnauthOptimizedReadRequest {
            kappa: target.need as u64,
            objects: vec![
                NetUnauthReadObjectId::Segment { segment: 0 },
                NetUnauthReadObjectId::Point { array_index: 5 },
            ],
        },
    };
    let read = |party: &AuthServer| match paper_a_optimized::net::auth::handle_read_only_message(
        party,
        message.clone(),
    ) {
        Some(Response::UnauthOptimizedRead { resp }) => resp,
        other => panic!("unexpected response: {:?}", other),
    };
    let on_the_fly = read(&source_party);
    assert_eq!(on_the_fly.objects.len(), 2);
    assert_eq!(on_the_fly, read(&target_party));
}
