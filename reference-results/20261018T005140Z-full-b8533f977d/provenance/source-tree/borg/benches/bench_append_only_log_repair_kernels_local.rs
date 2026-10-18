//! Local append-only-log repair kernel benchmark.
//!
//! This suite warms a real append-only-log cluster through the actual aligned
//! bulk_updates-append path, then measures only the two local kernels we care about:
//! - one live helper serving its repair contribution for one claim
//! - one wiped claimant reconstructing and installing that claim state
//!
//! Warmup, helper selection, and per-claim planning are fixed setup.

mod support;

use crate::support::append_only_log_repair::{
    BENCH_SEED, DEFAULT_COMMITTED_BULK_UPDATE_BATCHES, DEFAULT_HELPER_SURPLUS, DEFAULT_S,
    DEFAULT_SEGMENT_LEN, DEFAULT_TRIALS, MetricStats, RepairKernelsFixture, build_setup, git_sha,
    parse_usize_env, stats_from_samples, timestamp_unix_ms, write_json,
};
use serde::Serialize;
use std::collections::BTreeSet;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

#[derive(Clone, Debug, Serialize)]
struct BenchMeta {
    seed: u64,
    trials: usize,
    committed_bulk_update_batches: usize,
    helper_surplus: usize,
    max_errors: usize,
    corrupt_helper_count: usize,
    timestamp_unix_ms: u64,
    git: Option<String>,
    notes: String,
    metrics: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct BenchEntry {
    segment_len: usize,
    s: usize,
    n_grid: usize,
    need: usize,
    shard_count: usize,
    repair_helper_count: usize,
    bulk_updates_bytes: usize,
    bulk_updates_size: usize,
    committed_bulk_update_batches: usize,
    allocated_leaves: usize,
    live_unauth_len: u64,
    segment_count: usize,
    target_registry_index: usize,
    target_row_count: usize,
    helper_registry_index: usize,
    max_errors: usize,
    corrupt_helper_count: usize,
    correcting_repair_helper_count: usize,
    helper_row_replies_per_segment: usize,
    helper_eval_scalars_per_segment: usize,
    helper_row_reply_wire_bytes: usize,
    helper_eval_reply_wire_bytes: usize,
    helper_assist_total_wire_bytes: usize,
    recover_helper_reply_bytes_modeled: u64,
    recover_correcting_helper_reply_bytes_modeled: Option<u64>,
    full_bulk_updates_download_bytes: u64,
    helper_assist_core_ns: MetricStats,
    helper_assist_net_adapter_ns: MetricStats,
    recoverer_recover_kernel_ns: MetricStats,
    recoverer_recover_correcting_clean_kernel_ns: Option<MetricStats>,
    recoverer_recover_correcting_corrupt_kernel_ns: Option<MetricStats>,
    recoverer_install_kernel_ns: MetricStats,
    recoverer_recover_install_kernel_ns: MetricStats,
    bulk_updates_rebuild_from_downloaded_batches_ns: MetricStats,
}

#[derive(Clone, Debug, Serialize)]
struct BenchResults {
    meta: BenchMeta,
    entries: Vec<BenchEntry>,
}

fn parse_metric_filter(name: &str, allowed: &[&str]) -> BTreeSet<String> {
    let Some(raw) = std::env::var(name).ok() else {
        return allowed.iter().map(|value| (*value).to_string()).collect();
    };
    let selected = raw
        .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
        .filter(|value| !value.is_empty())
        .map(|value| value.to_string())
        .collect::<BTreeSet<_>>();
    assert!(
        !selected.is_empty(),
        "{name} must select at least one metric"
    );
    for metric in &selected {
        assert!(
            allowed.contains(&metric.as_str()),
            "invalid {name} metric `{metric}`"
        );
    }
    selected
}

fn bench_repair_kernels(
    trials: usize,
    committed_bulk_update_batches: usize,
    helper_surplus: usize,
    max_errors: usize,
    metrics: &BTreeSet<String>,
) -> BenchEntry {
    let segment_len = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_REPAIR_SEGMENT_LEN",
        DEFAULT_SEGMENT_LEN,
    );
    let s = parse_usize_env("BENCH_APPEND_ONLY_LOG_REPAIR_S", DEFAULT_S);
    let setup = build_setup(segment_len, s, helper_surplus);
    let mut fixture =
        RepairKernelsFixture::build(&setup, committed_bulk_update_batches, 0, 1, max_errors);
    fixture.validate_recovery();

    let mut helper_core_samples = Vec::with_capacity(trials);
    if metrics.contains("helper_core") {
        for _ in 0..trials {
            let start = Instant::now();
            let replies = fixture.helper_core_assist_once();
            helper_core_samples.push(start.elapsed().as_nanos() as u64);
            black_box(replies);
        }
    }

    let mut helper_net_adapter_samples = Vec::with_capacity(trials);
    if metrics.contains("helper_rpc") {
        for _ in 0..trials {
            let start = Instant::now();
            let replies = fixture.helper_rpc_assist_once();
            let row_bytes = paper_a_optimized::net::wire::encode_binary(&replies.0)
                .expect("serialize repair row response");
            let eval_bytes = paper_a_optimized::net::wire::encode_binary(&replies.1)
                .expect("serialize repair evaluation response");
            helper_net_adapter_samples.push(start.elapsed().as_nanos() as u64);
            assert_eq!(
                row_bytes.len() + eval_bytes.len(),
                fixture.helper_assist_total_wire_bytes
            );
            black_box((replies, row_bytes, eval_bytes));
        }
    }

    let repaired_segments_template = if metrics.contains("install") {
        let repaired = fixture.recover_segments_once();
        let recovered = fixture.install_segments_once(repaired.clone());
        black_box(recovered);
        Some(repaired)
    } else {
        None
    };

    let mut recover_kernel_samples = Vec::with_capacity(trials);
    if metrics.contains("recover") {
        for _ in 0..trials {
            let start = Instant::now();
            let repaired = fixture.recover_segments_once();
            recover_kernel_samples.push(start.elapsed().as_nanos() as u64);
            black_box(repaired);
        }
    }

    let recover_correcting_clean_kernel_ns =
        if max_errors > 0 && metrics.contains("recover_correcting_clean") {
            let mut samples = Vec::with_capacity(trials);
            for _ in 0..trials {
                let start = Instant::now();
                let repaired = fixture
                    .recover_segments_correcting_clean_once()
                    .expect("correcting clean fixture");
                samples.push(start.elapsed().as_nanos() as u64);
                black_box(repaired);
            }
            Some(stats_from_samples(samples))
        } else {
            None
        };

    let recover_correcting_corrupt_kernel_ns =
        if max_errors > 0 && metrics.contains("recover_correcting_corrupt") {
            let mut samples = Vec::with_capacity(trials);
            for _ in 0..trials {
                let start = Instant::now();
                let repaired = fixture
                    .recover_segments_correcting_corrupt_once()
                    .expect("correcting corrupt fixture");
                samples.push(start.elapsed().as_nanos() as u64);
                black_box(repaired);
            }
            Some(stats_from_samples(samples))
        } else {
            None
        };

    let mut install_kernel_samples = Vec::with_capacity(trials);
    if let Some(repaired_segments_template) = repaired_segments_template {
        for _ in 0..trials {
            let start = Instant::now();
            let recovered = fixture.install_segments_once(repaired_segments_template.clone());
            install_kernel_samples.push(start.elapsed().as_nanos() as u64);
            black_box(recovered);
        }
    }

    let mut recover_install_kernel_samples = Vec::with_capacity(trials);
    if metrics.contains("recover_install") {
        for _ in 0..trials {
            let start = Instant::now();
            let recovered = fixture.recover_and_install_once();
            recover_install_kernel_samples.push(start.elapsed().as_nanos() as u64);
            black_box(recovered);
        }
    }

    let mut bulk_updates_rebuild_samples = Vec::with_capacity(trials);
    if metrics.contains("rebuild") {
        for _ in 0..trials {
            let start = Instant::now();
            let rebuilt = fixture.rebuild_from_downloaded_bulk_updates_once();
            bulk_updates_rebuild_samples.push(start.elapsed().as_nanos() as u64);
            black_box(rebuilt);
        }
    }

    BenchEntry {
        segment_len: fixture.segment_len,
        s: fixture.s,
        n_grid: fixture.n_grid,
        need: fixture.need,
        shard_count: fixture.shard_count,
        repair_helper_count: fixture.repair_helper_count,
        bulk_updates_bytes: fixture.bulk_updates_bytes,
        bulk_updates_size: fixture.bulk_updates_size,
        committed_bulk_update_batches: fixture.committed_bulk_update_batches,
        allocated_leaves: fixture.allocated_leaves,
        live_unauth_len: fixture.live_unauth_len,
        segment_count: fixture.segment_count,
        target_registry_index: fixture.target_registry_index,
        target_row_count: fixture.target_row_count,
        helper_registry_index: fixture.helper_registry_index,
        max_errors: fixture.max_errors,
        corrupt_helper_count: fixture.corrupt_helper_count,
        correcting_repair_helper_count: fixture.correcting_repair_helper_count,
        helper_row_replies_per_segment: fixture.helper_row_replies_per_segment,
        helper_eval_scalars_per_segment: fixture.helper_eval_scalars_per_segment,
        helper_row_reply_wire_bytes: fixture.helper_row_reply_wire_bytes,
        helper_eval_reply_wire_bytes: fixture.helper_eval_reply_wire_bytes,
        helper_assist_total_wire_bytes: fixture.helper_assist_total_wire_bytes,
        recover_helper_reply_bytes_modeled: fixture.recover_helper_reply_bytes_modeled,
        recover_correcting_helper_reply_bytes_modeled: fixture
            .recover_correcting_helper_reply_bytes_modeled,
        full_bulk_updates_download_bytes: fixture.full_bulk_updates_download_bytes,
        helper_assist_core_ns: stats_from_samples(helper_core_samples),
        helper_assist_net_adapter_ns: stats_from_samples(helper_net_adapter_samples),
        recoverer_recover_kernel_ns: stats_from_samples(recover_kernel_samples),
        recoverer_recover_correcting_clean_kernel_ns: recover_correcting_clean_kernel_ns,
        recoverer_recover_correcting_corrupt_kernel_ns: recover_correcting_corrupt_kernel_ns,
        recoverer_install_kernel_ns: stats_from_samples(install_kernel_samples),
        recoverer_recover_install_kernel_ns: stats_from_samples(recover_install_kernel_samples),
        bulk_updates_rebuild_from_downloaded_batches_ns: stats_from_samples(
            bulk_updates_rebuild_samples,
        ),
    }
}

fn run_benchmark() {
    let trials = parse_usize_env("BENCH_APPEND_ONLY_LOG_REPAIR_TRIALS", DEFAULT_TRIALS);
    let committed_bulk_update_batches = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_REPAIR_COMMITTED_BULK_UPDATE_BATCHES",
        DEFAULT_COMMITTED_BULK_UPDATE_BATCHES,
    );
    let requested_helper_surplus = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_REPAIR_HELPER_SURPLUS",
        DEFAULT_HELPER_SURPLUS,
    );
    let max_errors = std::env::var("BENCH_APPEND_ONLY_LOG_REPAIR_MAX_ERRORS")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(0);
    let helper_surplus = if max_errors == 0 {
        requested_helper_surplus
    } else {
        requested_helper_surplus.max(2 * max_errors + 2)
    };
    let metrics = parse_metric_filter(
        "BENCH_APPEND_ONLY_LOG_REPAIR_METRICS",
        &[
            "helper_core",
            "helper_rpc",
            "recover",
            "recover_correcting_clean",
            "recover_correcting_corrupt",
            "install",
            "recover_install",
            "rebuild",
        ],
    );

    let entry = bench_repair_kernels(
        trials,
        committed_bulk_update_batches,
        helper_surplus,
        max_errors,
        &metrics,
    );
    let results = BenchResults {
        meta: BenchMeta {
            seed: BENCH_SEED,
            trials,
            committed_bulk_update_batches,
            helper_surplus,
            max_errors,
            corrupt_helper_count: if max_errors == 0 { 0 } else { max_errors },
            timestamp_unix_ms: timestamp_unix_ms(),
            git: git_sha(),
            notes: "local append-only-log repair kernels: warm a real one-claim-per-party cluster through aligned bulk updates appends, then time selected helper and wiped-claimant repair operations; the helper RPC metric includes response serialization, while warmup, helper selection, TCP, and per-claim planning stay outside the timed region. Set BENCH_APPEND_ONLY_LOG_REPAIR_MAX_ERRORS>0 to enable simulated client-side correction fixtures.".to_string(),
            metrics: metrics.into_iter().collect(),
        },
        entries: vec![entry],
    };

    write_json(
        Path::new("results/bench_append_only_log_repair_kernels_local.json"),
        &results,
    );
}

fn main() {
    run_benchmark();
}
