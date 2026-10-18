//! Restored setup must be identical to an append-built baseline.
#![cfg(all(feature = "bench", feature = "private"))]

#[path = "../benches/support/auth_fixture.rs"]
mod auth_fixture;
#[path = "support/cluster.rs"]
mod cluster;
#[path = "../benches/support/fixture_cache.rs"]
mod fixture_cache;

use paper_a_optimized::core::NodeLinearization;
use paper_a_optimized::private::BulkLog;
use paper_a_optimized::{AuthParams, encode_value};

#[test]
fn restored_auth_fixture_has_identical_primary_and_auxiliary_state() {
    let params = cluster::build_params(4, 12, 42);
    let auth = AuthParams::new(50);
    let servers = cluster::build_servers(&params, &auth, 257);
    let mut client = cluster::reader(&params, &auth, 257, &servers);
    let values = (0..257)
        .map(|i| (i, encode_value((i + 1).into())))
        .collect::<Vec<_>>();
    client.write_batch(&values).unwrap();
    for server in &servers {
        let original = server.inner.lock().unwrap();
        let fixture = auth_fixture::AuthFixture::capture(&original);
        let bytes = bincode::serde::encode_to_vec(&fixture, bincode::config::standard()).unwrap();
        let (fixture, used): (auth_fixture::AuthFixture, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(used, bytes.len());
        let restored = fixture.restore(
            &params,
            &auth,
            NodeLinearization::Preorder,
            original.store.unauth.claim_bundles(),
        );
        assert_eq!(restored.allocated_leaves(), original.allocated_leaves());
        assert_eq!(restored.store.root, original.store.root);
        assert!(
            restored
                .store
                .unauth
                .coded
                .equals_for_tests(&params, &original.store.unauth.coded)
        );
    }
}

#[test]
fn restored_bulk_fixture_extends_like_a_fresh_build() {
    let params = cluster::build_params(4, 12, 42);
    let auth = AuthParams::new(50);
    let order = NodeLinearization::Preorder;
    for defaults in [0, 11, 257] {
        let original =
            BulkLog::build_for_claims(&params, &auth, order, 257, defaults, 42, &[0, 3, 11], 2, 3);
        let mut bytes = Vec::new();
        original.write_benchmark_fixture(&mut bytes);
        let mut restored =
            BulkLog::read_benchmark_fixture(&params, &auth, order, &mut bytes.as_slice());
        assert_eq!(restored.root(), original.root());
        for claim in 0..3 {
            assert_eq!(restored.claim_entries(claim), original.claim_entries(claim));
        }
        restored.extend(300, 2);
        let fresh =
            BulkLog::build_for_claims(&params, &auth, order, 300, defaults, 42, &[0, 3, 11], 2, 3);
        assert_eq!(restored.root(), fresh.root());
        for claim in 0..3 {
            assert_eq!(restored.claim_entries(claim), fresh.claim_entries(claim));
        }
        // The loaded, mutated instance did not alter its immutable fixture.
        let again = BulkLog::read_benchmark_fixture(&params, &auth, order, &mut bytes.as_slice());
        assert_eq!(again.leaves(), 257);
        assert_eq!(again.root(), original.root());
    }
}
