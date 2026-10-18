//! Authenticated server rejects invalid proofs.

mod support;

use paper_a_optimized::auth::service::AuthServer;
use paper_a_optimized::net::wire::{AuthWriteRequest, NetAuthPath, NetValue};
use paper_a_optimized::net::{Message, Response};
use paper_a_optimized::{
    AuthBroadcastUpdate, AuthConfig, CodedArrayConfig, Config, DeltaMode, Fp,
    build_shard_registry_deterministic, encode_value,
};
use support::setup;

fn update_to_request(update: &AuthBroadcastUpdate) -> AuthWriteRequest {
    AuthWriteRequest {
        old: NetValue(update.old),
        new: NetValue(update.new),
        path: NetAuthPath {
            index: update.index as u64,
            siblings: update.path.siblings.iter().copied().map(NetValue).collect(),
            root: NetValue(update.path.root),
        },
    }
}

#[test]
fn auth_server_rejects_invalid_root() {
    let h = 3;
    let alloc_leaf_batch = 4;
    let kappa = 2;
    let g = 2;
    let shard_registry = build_shard_registry_deterministic(g);
    let party_registry_indices = vec![vec![0, 1]];
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
            party_registry_indices: party_registry_indices.clone(),
            seed: 99,
        },
    };

    let system = setup(cfg);
    let (old, path) = system.read(0).expect("read");
    let new = encode_value(Fp::from(7u64));
    let update = system
        .build_broadcast_update(0, old, new, path, DeltaMode::Aggregate)
        .expect("build update");

    let mut bad = update.clone();
    bad.path.root = encode_value(Fp::from(123u64));

    let mut party = AuthServer::new(
        system.params.clone(),
        system.auth.clone(),
        system.node_linearization,
        system.allocated_leaves,
    )
    .expect("server");
    let root_before = party.store.root;

    let resp = match paper_a_optimized::net::auth::handle_message(
        &mut party,
        Message::AuthWrite {
            req: update_to_request(&bad),
        },
    ) {
        Response::AuthWrite { resp } => resp,
        other => panic!("unexpected response: {:?}", other),
    };
    assert!(!resp.ok, "invalid proof should be rejected");
    assert_eq!(party.store.root, root_before, "root must not change");
}
