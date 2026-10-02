//! Compile-only test to ensure bench modules build under the `bench` feature.

#![cfg(feature = "bench")]
#![allow(dead_code)]

#[path = "../benches/support/mod.rs"]
mod support;

#[path = "../benches/bench_append_only_log_client_point_open_local.rs"]
mod bench_append_only_log_client_point_open_local;

#[path = "../benches/bench_append_only_log_on_the_fly_open_local.rs"]
mod bench_append_only_log_on_the_fly_open_local;

#[test]
/// Smoke test: the benches compile under `--features bench`.
fn bench_entrypoints_compile() {
    assert!(true);
}
