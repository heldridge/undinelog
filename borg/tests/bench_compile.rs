//! Compile-only test to ensure bench modules build under the `bench` feature.

#![cfg(feature = "bench")]
#![allow(dead_code)]

#[path = "../benches/support/mod.rs"]
mod support;

#[path = "../benches/bench_unauth_retrieval.rs"]
mod bench_unauth_retrieval;

#[path = "../benches/bench_unauth_migrate.rs"]
mod bench_unauth_migrate;

#[path = "../benches/bench_unauth_onboarding.rs"]
mod bench_unauth_onboarding;

#[path = "../benches/bench_unauth_update.rs"]
mod bench_unauth_update;

#[path = "../benches/bench_unauth_block_update.rs"]
mod bench_unauth_block_update;

#[path = "../benches/bench_unauth_locality.rs"]
mod bench_unauth_locality;

#[path = "../benches/bench_auth_archive_local.rs"]
mod bench_auth_archive_local;

#[path = "../benches/bench_append_only_log_client_point_open_local.rs"]
mod bench_append_only_log_client_point_open_local;

#[path = "../benches/bench_rs_error_correction.rs"]
mod bench_rs_error_correction;

#[test]
/// Smoke test: the benches compile under `--features bench`.
fn bench_entrypoints_compile() {
    assert!(true);
}
