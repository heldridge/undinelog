//! Smoke test covering both unauthenticated and authenticated modes.

use ark_ff::Field;
use paper_a_optimized::auth::service::AuthServer;
use paper_a_optimized::auth::{AuthParams, mapping};
use paper_a_optimized::coded_array::params::CodedArrayConfig;
use paper_a_optimized::coded_array::rs::{barycentric_eval, precompute_barycentric};
use paper_a_optimized::merkle::leaf_key;
use paper_a_optimized::net::wire::{AuthReadRequest, NetMaybeFpBlob};
use paper_a_optimized::net::{Message, Response};
use paper_a_optimized::unauth::{DeltaMode, UnauthStore};
use paper_a_optimized::{Fp, build_shard_registry_deterministic, decode_value};

fn decode_from_shares(
    params: &paper_a_optimized::CodedArrayParams,
    j: usize,
    owned_indices: &[usize],
    values: &NetMaybeFpBlob,
) -> Fp {
    let alpha = params.x[j];
    let z_eval = alpha.pow([params.s as u64]);
    let mut z_values = Vec::new();
    let mut r_values = Vec::new();
    for (&registry_index, y) in owned_indices.iter().zip(values.0.iter().copied()) {
        let Some(y) = y else {
            continue;
        };
        let entry = params.shard_registry[registry_index];
        z_values.push(entry.z);
        r_values.push(y);
        if r_values.len() >= params.need {
            break;
        }
    }
    let weights = precompute_barycentric(&z_values);
    barycentric_eval(&weights, &r_values, z_eval)
}

#[test]
fn modes_smoke_basic() {
    let segment_len: usize = 8;
    let n_grid = segment_len;
    let initial_len: u64 = 4;
    let alloc_leaf_batch: usize = 4;
    let s = 8;
    let g = 1;
    let shard_registry = build_shard_registry_deterministic(g, s, n_grid);
    let party_registry_indices = vec![vec![0]];
    let cfg = CodedArrayConfig {
        segment_len,
        s,
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
            req: AuthReadRequest { leaf },
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
    let j = (idx % params.segment_len as u64) as usize;
    let value = decode_from_shares(
        &params,
        j,
        &party_registry_indices[0],
        &resp.nodes[0].values,
    );
    assert_eq!(value, Fp::from(0u64));
}

#[test]
fn auth_server_exposes_unauth_maintenance_but_not_block_api() {
    let segment_len: usize = 8;
    let n_grid = segment_len;
    let alloc_leaf_batch: usize = 4;
    let s = 8;
    let g = 1;
    let shard_registry = build_shard_registry_deterministic(g, s, n_grid);
    let party_registry_indices = vec![vec![0]];
    let cfg = CodedArrayConfig {
        segment_len,
        s,
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
    assert_eq!(values.len(), s);

    let resp = paper_a_optimized::net::unauth::handle_message(
        &mut auth_party,
        Message::UnauthReadBlock { block: 0 },
    );
    assert!(matches!(resp, Response::Error { .. }));

    let resp = paper_a_optimized::net::unauth::handle_message(
        &mut auth_party,
        Message::UnauthMigrate {
            target_s: 4,
            partner_shards: Vec::new(),
            helper_rows: Vec::new(),
        },
    );
    assert!(matches!(resp, Response::UnauthMigrate { ok: true }));
    assert_eq!(auth_party.params.s, 4);
    assert_eq!(auth_party.store.unauth.params.s, 4);
}
