//! Local app-only benchmark for the AppendOnlyLog fixed-bulk updates append path.
//!
//! This separates:
//! - exact bulk updates codec handling
//! - generic aligned subtree append
//! - prepared fixed-bulk updates append with app-local preprocessing

use append_only_log_demo::codec::{
    bulk_updates_bytes_as_leaves, encode_bulk_updates_leaves, encode_bulk_updates_leaves_into,
};
use append_only_log_demo::frontier::PreparedBulkUpdatesAppend;
use append_only_log_demo::layout::{BULK_UPDATES_BYTES, BULK_UPDATES_SIZE};
use append_only_log_demo::registry::AppendOnlyLogFrontierLeafState;
use paper_a_optimized::auth::client::{AuthClient, AuthTransport, ClientError as AuthClientError};
use paper_a_optimized::auth::mapping::NodeLinearization;
use paper_a_optimized::auth::service::{
    AuthBatchWriteRequest, AuthBatchWriteResponse, AuthFreshSubtreeWriteRequest, AuthReadResponse,
    AuthRequest, AuthResponse, AuthServer, AuthService, AuthSubtreeReadResponse,
    AuthSubtreeUpperPathReadResponse, AuthWriteRequest, AuthWriteResponse,
};
use paper_a_optimized::unauth::ClaimBundle;
use paper_a_optimized::{
    AuthConfig, AuthParams, CodedArrayConfig, Config, build_coded_params,
    build_shard_registry_deterministic,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde::Serialize;
use std::collections::BTreeSet;
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const BENCH_SEED: u64 = 42;
const DEFAULT_TREE_HEIGHT: usize = 50;
const DEFAULT_TRIALS: usize = 1;
const DEFAULT_SEGMENT_LEN: usize = 4096;
const DEFAULT_S: usize = 64;
const DEFAULT_LOCAL_CLAIMS: usize = 0;
const DEFAULT_SKIP_GENERIC: bool = false;
const DEFAULT_SEQUENCE_BLOCKS: usize = 64;
const DEFAULT_SEQUENCE_WARMUP_BLOCKS: usize = 0;

#[derive(Clone)]
struct LocalServer {
    label: String,
    inner: Arc<Mutex<AuthServer>>,
}

#[derive(Clone)]
struct LocalTransport;

impl AuthTransport for LocalTransport {
    type Server = LocalServer;

    fn supports_subtree_upper_path_reads(&self) -> bool {
        true
    }

    fn supports_fresh_subtree_writes(&self) -> bool {
        true
    }

    fn server_label(&self, server: &Self::Server) -> String {
        server.label.clone()
    }

    fn auth_get_len(&self, server: &Self::Server) -> Result<u64, AuthClientError> {
        let mut guard = server.inner.lock().expect("auth server lock");
        match AuthService::handle_request(&mut *guard, AuthRequest::GetLen) {
            AuthResponse::Len { len } => Ok(len),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!(
                "invalid auth len response: {:?}",
                other
            ))),
        }
    }

    fn auth_read(
        &self,
        server: &Self::Server,
        index: u64,
    ) -> Result<AuthReadResponse, AuthClientError> {
        let mut guard = server.inner.lock().expect("auth server lock");
        match AuthService::handle_request(&mut *guard, AuthRequest::Read { index }) {
            AuthResponse::Read { resp } => Ok(resp),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!(
                "invalid auth read response: {:?}",
                other
            ))),
        }
    }

    fn auth_read_subtree(
        &self,
        server: &Self::Server,
        start_index: u64,
        leaf_count: usize,
    ) -> Result<AuthSubtreeReadResponse, AuthClientError> {
        let mut guard = server.inner.lock().expect("auth server lock");
        match AuthService::handle_request(
            &mut *guard,
            AuthRequest::ReadSubtree {
                start_index,
                leaf_count,
            },
        ) {
            AuthResponse::ReadSubtree { resp } => Ok(resp),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!(
                "invalid auth subtree read response: {:?}",
                other
            ))),
        }
    }

    fn auth_read_subtree_upper_path(
        &self,
        server: &Self::Server,
        start_index: u64,
        leaf_count: usize,
    ) -> Result<AuthSubtreeUpperPathReadResponse, AuthClientError> {
        let mut guard = server.inner.lock().expect("auth server lock");
        match AuthService::handle_request(
            &mut *guard,
            AuthRequest::ReadSubtreeUpperPath {
                start_index,
                leaf_count,
            },
        ) {
            AuthResponse::ReadSubtreeUpperPath { resp } => Ok(resp),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!(
                "invalid auth subtree upper-path response: {:?}",
                other
            ))),
        }
    }

    fn unauth_multi_read(
        &self,
        server: &Self::Server,
        indices: &[u64],
    ) -> Result<Vec<Vec<paper_a_optimized::ShardShare>>, AuthClientError> {
        let mut guard = server.inner.lock().expect("auth server lock");
        match AuthService::handle_request(
            &mut *guard,
            AuthRequest::UnauthMultiRead {
                idxs: indices.to_vec(),
            },
        ) {
            AuthResponse::UnauthMultiRead { shares } => Ok(shares),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!(
                "invalid unauth multi-read response: {:?}",
                other
            ))),
        }
    }

    fn auth_write(
        &self,
        server: &Self::Server,
        req: &AuthWriteRequest,
    ) -> Result<AuthWriteResponse, AuthClientError> {
        let mut guard = server.inner.lock().expect("auth server lock");
        match AuthService::handle_request(&mut *guard, AuthRequest::Write { req: req.clone() }) {
            AuthResponse::Write { resp } => Ok(resp),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!(
                "invalid auth write response: {:?}",
                other
            ))),
        }
    }

    fn auth_write_batch(
        &self,
        server: &Self::Server,
        req: &AuthBatchWriteRequest,
    ) -> Result<AuthBatchWriteResponse, AuthClientError> {
        let mut guard = server.inner.lock().expect("auth server lock");
        match AuthService::handle_request(&mut *guard, AuthRequest::WriteBatch { req: req.clone() })
        {
            AuthResponse::WriteBatch { resp } => Ok(resp),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!(
                "invalid auth batch write response: {:?}",
                other
            ))),
        }
    }

    fn auth_write_fresh_subtree(
        &self,
        server: &Self::Server,
        req: &AuthFreshSubtreeWriteRequest,
    ) -> Result<AuthBatchWriteResponse, AuthClientError> {
        let mut guard = server.inner.lock().expect("auth server lock");
        match AuthService::handle_request(
            &mut *guard,
            AuthRequest::WriteFreshSubtree { req: req.clone() },
        ) {
            AuthResponse::WriteFreshSubtree { resp } => Ok(resp),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!(
                "invalid auth fresh subtree write response: {:?}",
                other
            ))),
        }
    }

    fn auth_extend(&self, server: &Self::Server, new_len: u64) -> Result<bool, AuthClientError> {
        let mut guard = server.inner.lock().expect("auth server lock");
        match AuthService::handle_request(
            &mut *guard,
            AuthRequest::Extend {
                new_leaves: new_len,
            },
        ) {
            AuthResponse::Extend { ok } => Ok(ok),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!(
                "invalid auth extend response: {:?}",
                other
            ))),
        }
    }
}

#[derive(Clone)]
struct BenchSetup {
    cfg: Config,
    params: paper_a_optimized::CodedArrayParams,
    auth: AuthParams,
}

#[derive(Clone, Debug, Serialize)]
struct MetricStats {
    mean: u64,
    p50: u64,
    p95: u64,
    trials: u64,
    samples: Vec<u64>,
}

#[derive(Clone, Debug, Serialize)]
struct BenchResults {
    meta: BenchMeta,
    entries: Vec<BenchEntry>,
}

#[derive(Clone, Debug, Serialize)]
struct BenchMeta {
    seed: u64,
    trials: usize,
    sequence_blocks: usize,
    sequence_warmup_blocks: usize,
    timestamp_unix_ms: u64,
    git: Option<String>,
    notes: String,
    skip_generic: bool,
    metrics: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct BenchEntry {
    segment_len: usize,
    s: usize,
    n_grid: usize,
    need: usize,
    shard_count: usize,
    local_claims: usize,
    codec_old_ns: MetricStats,
    codec_scratch_ns: MetricStats,
    codec_borrow_ns: MetricStats,
    generic_next_bulk_updates_append_ns: MetricStats,
    prepared_next_bulk_updates_append_ns: MetricStats,
    prepare_next_bulk_updates_slack_ns: MetricStats,
    prepared_with_slack_next_bulk_updates_append_ns: MetricStats,
    prepared_speedup_vs_generic_mean: f64,
    prepared_with_slack_speedup_vs_prepared_mean: f64,
}

fn percentile(sorted: &[u64], pct: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (pct * (sorted.len() - 1)) / 100;
    sorted[idx]
}

fn stats_from_samples(mut samples: Vec<u64>) -> MetricStats {
    if samples.is_empty() {
        return MetricStats {
            mean: 0,
            p50: 0,
            p95: 0,
            trials: 0,
            samples: Vec::new(),
        };
    }
    let sum: u128 = samples.iter().map(|&v| v as u128).sum();
    let mean = (sum / samples.len() as u128) as u64;
    let raw_samples = samples.clone();
    samples.sort_unstable();
    MetricStats {
        mean,
        p50: percentile(&samples, 50),
        p95: percentile(&samples, 95),
        trials: samples.len() as u64,
        samples: raw_samples,
    }
}

fn git_sha() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let sha = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if sha.is_empty() { None } else { Some(sha) }
}

fn parse_bool_env(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => default,
        },
        Err(_) => default,
    }
}

fn parse_usize_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn parse_usize_env_allow_zero(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(default)
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

fn build_registry(
    shard_count: usize,
    s: usize,
    segment_len: usize,
) -> Vec<paper_a_optimized::ShardRegEntry> {
    let n_grid = segment_len;
    build_shard_registry_deterministic(shard_count, s, n_grid)
}

fn build_config(segment_len: usize, s: usize) -> Config {
    let tree_height = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_BULK_UPDATES_APPEND_TREE_HEIGHT",
        DEFAULT_TREE_HEIGHT,
    );
    let n_grid = segment_len;
    assert!(s.is_power_of_two(), "s must be a power of two");
    assert!(s <= n_grid, "s must fit inside n_grid");
    assert_eq!(n_grid % s, 0, "s must divide n_grid");
    let need = (n_grid - 1) / s + 1;
    let shard_count = need;
    Config {
        auth: AuthConfig {
            tree_height,
            alloc_leaf_batch: segment_len / 2,
            node_linearization: NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            segment_len,
            s,
            shard_count,
            shard_registry: build_registry(shard_count, s, segment_len),
            party_registry_indices: (0..shard_count).map(|idx| vec![idx]).collect(),
            seed: BENCH_SEED,
        },
    }
}

fn build_setup(segment_len: usize, s: usize) -> BenchSetup {
    let cfg = build_config(segment_len, s);
    let params = build_coded_params(&cfg.coded);
    let auth = AuthParams::new(cfg.auth.tree_height);
    BenchSetup { cfg, params, auth }
}

fn claim_for_registry_index(
    params: &paper_a_optimized::CodedArrayParams,
    registry_index: usize,
) -> ClaimBundle {
    ClaimBundle::exact_for_registry_index(params, registry_index)
        .expect("exact claim should exist for registry index")
}

fn build_claim_server(setup: &BenchSetup, local_claims: usize) -> AuthServer {
    let params = &setup.params;
    let effective_claims = if local_claims == 0 {
        params.shard_registry.len()
    } else {
        local_claims.min(params.shard_registry.len())
    };
    let claims = (0..effective_claims)
        .map(|idx| claim_for_registry_index(&params, idx))
        .collect();
    AuthServer::new_with_assignments_and_claims(
        params.clone(),
        setup.auth.clone(),
        setup.cfg.auth.node_linearization,
        0,
        None,
        Some(claims),
    )
    .expect("build claim server")
}

fn make_client(server: AuthServer, setup: &BenchSetup) -> AuthClient<LocalTransport> {
    let server = LocalServer {
        label: "local".to_string(),
        inner: Arc::new(Mutex::new(server)),
    };
    AuthClient::new(
        LocalTransport,
        setup.params.clone(),
        setup.auth.clone(),
        setup.cfg.auth.node_linearization,
        0,
        server.clone(),
        vec![server],
    )
}

fn build_exact_bulk_updates(seed: u64) -> Vec<u8> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut bulk_updates = Vec::with_capacity(BULK_UPDATES_BYTES);
    for _ in 0..BULK_UPDATES_SIZE {
        bulk_updates.extend_from_slice(&paper_a_optimized::random_value(&mut rng));
    }
    bulk_updates
}

fn warm_first_bulk_updates_client(
    setup: &BenchSetup,
    local_claims: usize,
    bulk_updates0_leaves: &[paper_a_optimized::Value256],
) -> (AuthClient<LocalTransport>, AppendOnlyLogFrontierLeafState) {
    let mut planner = PreparedBulkUpdatesAppend::new(setup.cfg.auth.tree_height).expect("planner");
    let mut client = make_client(build_claim_server(setup, local_claims), setup);
    let frontier0 = planner
        .append_from_frontier(&mut client, 0, None, bulk_updates0_leaves)
        .expect("warm first bulk_updates");
    assert_eq!(client.trusted_root(), Some(frontier0.path.root));
    (client, frontier0)
}

fn bench_point(
    trials: usize,
    setup: &BenchSetup,
    local_claims: usize,
    skip_generic: bool,
    sequence_blocks: usize,
    sequence_warmup_blocks: usize,
    metrics: &BTreeSet<String>,
) -> BenchEntry {
    assert!(
        sequence_blocks > sequence_warmup_blocks,
        "sequence_blocks must exceed sequence_warmup_blocks"
    );
    let effective_local_claims = if local_claims == 0 {
        setup.params.shard_registry.len()
    } else {
        local_claims.min(setup.params.shard_registry.len())
    };
    let generic_available = effective_local_claims >= setup.params.need;
    let run_generic = metrics.contains("generic") && !skip_generic && generic_available;

    let bulk_updates0 = build_exact_bulk_updates(BENCH_SEED ^ 0xB17C0);
    let bulk_updates1 = build_exact_bulk_updates(BENCH_SEED ^ 0xB17C1);
    let bulk_updates0_leaves =
        bulk_updates_bytes_as_leaves(&bulk_updates0).expect("bulk_updates0 leaves");
    let bulk_updates1_leaves =
        bulk_updates_bytes_as_leaves(&bulk_updates1).expect("bulk_updates1 leaves");

    let mut codec_old = Vec::with_capacity(trials);
    let mut codec_scratch = Vec::with_capacity(trials);
    let mut codec_borrow = Vec::with_capacity(trials);
    if metrics.contains("codec") {
        for _ in 0..trials {
            let start = Instant::now();
            let leaves = black_box(encode_bulk_updates_leaves(&bulk_updates0).expect("encode old"));
            codec_old.push(start.elapsed().as_nanos() as u64);
            assert_eq!(leaves.len(), BULK_UPDATES_SIZE);

            let mut scratch = Vec::with_capacity(BULK_UPDATES_SIZE);
            let start = Instant::now();
            black_box(
                encode_bulk_updates_leaves_into(&bulk_updates0, &mut scratch).expect("encode into"),
            );
            codec_scratch.push(start.elapsed().as_nanos() as u64);
            assert_eq!(scratch.len(), BULK_UPDATES_SIZE);

            let start = Instant::now();
            let leaves =
                black_box(bulk_updates_bytes_as_leaves(&bulk_updates0).expect("borrow leaves"));
            codec_borrow.push(start.elapsed().as_nanos() as u64);
            assert_eq!(leaves.len(), BULK_UPDATES_SIZE);
        }
    }

    let mut prepare_slack_samples = Vec::with_capacity(trials);
    if metrics.contains("prepare_slack") {
        for _ in 0..trials {
            let (mut client, _) =
                warm_first_bulk_updates_client(setup, local_claims, &bulk_updates0_leaves);
            let warm_root = client.trusted_root().expect("warm root");
            let start = Instant::now();
            client
                .extend(2 * BULK_UPDATES_SIZE as u64)
                .expect("prepare next bulk_updates slack");
            prepare_slack_samples.push(start.elapsed().as_nanos() as u64);
            assert_eq!(client.trusted_root(), Some(warm_root));
        }
    }

    let mut generic_samples = Vec::with_capacity(trials);
    if run_generic {
        for _ in 0..trials {
            let (mut client, _) =
                warm_first_bulk_updates_client(setup, local_claims, &bulk_updates0_leaves);
            let start = Instant::now();
            let appended_at =
                black_box(client.append_subtree(&bulk_updates1_leaves)).expect("generic append");
            generic_samples.push(start.elapsed().as_nanos() as u64);
            assert_eq!(appended_at, BULK_UPDATES_SIZE as u64);
        }
    }

    let sampled_blocks_per_trial = sequence_blocks - sequence_warmup_blocks;
    let mut prepared_samples = Vec::with_capacity(trials * sampled_blocks_per_trial);
    if metrics.contains("prepared") {
        for _ in 0..trials {
            prepared_samples.extend(sequence_append_samples(
                setup,
                local_claims,
                &bulk_updates1_leaves,
                sequence_blocks,
                sequence_warmup_blocks,
                false,
            ));
        }
    }

    let mut prepared_with_slack_samples = Vec::with_capacity(trials * sampled_blocks_per_trial);
    if metrics.contains("prepared_with_slack") {
        for _ in 0..trials {
            prepared_with_slack_samples.extend(sequence_append_samples(
                setup,
                local_claims,
                &bulk_updates1_leaves,
                sequence_blocks,
                sequence_warmup_blocks,
                true,
            ));
        }
    }

    let codec_old_stats = stats_from_samples(codec_old);
    let codec_scratch_stats = stats_from_samples(codec_scratch);
    let codec_borrow_stats = stats_from_samples(codec_borrow);
    let generic_stats = stats_from_samples(generic_samples);
    let prepared_stats = stats_from_samples(prepared_samples);
    let prepare_slack_stats = stats_from_samples(prepare_slack_samples);
    let prepared_with_slack_stats = stats_from_samples(prepared_with_slack_samples);
    let prepared_speedup_vs_generic_mean = if generic_stats.mean == 0 || prepared_stats.mean == 0 {
        0.0
    } else {
        generic_stats.mean as f64 / prepared_stats.mean as f64
    };
    let prepared_with_slack_speedup_vs_prepared_mean =
        if prepared_stats.mean == 0 || prepared_with_slack_stats.mean == 0 {
            0.0
        } else {
            prepared_stats.mean as f64 / prepared_with_slack_stats.mean as f64
        };
    BenchEntry {
        segment_len: setup.cfg.coded.segment_len,
        s: setup.cfg.coded.s,
        n_grid: setup.params.n_grid,
        need: setup.params.need,
        shard_count: setup.params.shard_registry.len(),
        local_claims: effective_local_claims,
        codec_old_ns: codec_old_stats,
        codec_scratch_ns: codec_scratch_stats,
        codec_borrow_ns: codec_borrow_stats,
        generic_next_bulk_updates_append_ns: generic_stats,
        prepared_next_bulk_updates_append_ns: prepared_stats,
        prepare_next_bulk_updates_slack_ns: prepare_slack_stats,
        prepared_with_slack_next_bulk_updates_append_ns: prepared_with_slack_stats,
        prepared_speedup_vs_generic_mean,
        prepared_with_slack_speedup_vs_prepared_mean,
    }
}

fn sequence_append_samples(
    setup: &BenchSetup,
    local_claims: usize,
    bulk_updates_leaves: &[paper_a_optimized::Value256],
    sequence_blocks: usize,
    sequence_warmup_blocks: usize,
    prepare_slack: bool,
) -> Vec<u64> {
    let mut client = make_client(build_claim_server(setup, local_claims), setup);
    let mut planner = PreparedBulkUpdatesAppend::new(setup.cfg.auth.tree_height).expect("planner");
    let mut frontier: Option<AppendOnlyLogFrontierLeafState> = None;
    let mut frontier_len = 0u64;
    let mut samples = Vec::with_capacity(sequence_blocks - sequence_warmup_blocks);

    for block_index in 0..sequence_blocks {
        if prepare_slack {
            client
                .extend(frontier_len + BULK_UPDATES_SIZE as u64)
                .expect("prepare next bulk_updates slack");
        }

        let start = Instant::now();
        let next_frontier = if prepare_slack {
            black_box(planner.append_from_prepared_frontier(
                &mut client,
                frontier_len,
                frontier.as_ref(),
                bulk_updates_leaves,
            ))
            .expect("prepared sequence append with slack")
        } else {
            black_box(planner.append_from_frontier(
                &mut client,
                frontier_len,
                frontier.as_ref(),
                bulk_updates_leaves,
            ))
            .expect("prepared sequence append")
        };
        let elapsed_ns = start.elapsed().as_nanos() as u64;

        frontier_len += BULK_UPDATES_SIZE as u64;
        assert_eq!(next_frontier.path.index + 1, frontier_len as usize);
        frontier = Some(next_frontier);
        if block_index >= sequence_warmup_blocks {
            samples.push(elapsed_ns);
        }
    }

    samples
}

fn main() {
    let trials = std::env::var("BENCH_APPEND_ONLY_LOG_BULK_UPDATES_APPEND_TRIALS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_TRIALS);
    let segment_len = std::env::var("BENCH_APPEND_ONLY_LOG_BULK_UPDATES_APPEND_SEGMENT_LEN")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_SEGMENT_LEN);
    let s = std::env::var("BENCH_APPEND_ONLY_LOG_BULK_UPDATES_APPEND_S")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_S);
    let local_claims = std::env::var("BENCH_APPEND_ONLY_LOG_BULK_UPDATES_APPEND_LOCAL_CLAIMS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_LOCAL_CLAIMS);
    let skip_generic = parse_bool_env(
        "BENCH_APPEND_ONLY_LOG_BULK_UPDATES_APPEND_SKIP_GENERIC",
        DEFAULT_SKIP_GENERIC,
    );
    let sequence_blocks = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_BULK_UPDATES_APPEND_SEQUENCE_BLOCKS",
        DEFAULT_SEQUENCE_BLOCKS,
    );
    let sequence_warmup_blocks = parse_usize_env_allow_zero(
        "BENCH_APPEND_ONLY_LOG_BULK_UPDATES_APPEND_SEQUENCE_WARMUP_BLOCKS",
        DEFAULT_SEQUENCE_WARMUP_BLOCKS,
    );
    let metrics = parse_metric_filter(
        "BENCH_APPEND_ONLY_LOG_BULK_UPDATES_APPEND_METRICS",
        &[
            "codec",
            "prepare_slack",
            "generic",
            "prepared",
            "prepared_with_slack",
        ],
    );

    let setup = build_setup(segment_len, s);
    let entry = bench_point(
        trials,
        &setup,
        local_claims,
        skip_generic,
        sequence_blocks,
        sequence_warmup_blocks,
        &metrics,
    );
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let results = BenchResults {
        meta: BenchMeta {
            seed: BENCH_SEED,
            trials,
            sequence_blocks,
            sequence_warmup_blocks,
            timestamp_unix_ms: timestamp_ms,
            git: git_sha(),
            notes: "append-only-log local benchmark with fresh live-server trials: each trial appends a sequence of fixed bulk-update blocks and summarizes per-block append cost over the configured live-process samples; the slack path performs suffix preallocation before each timed append".to_string(),
            skip_generic,
            metrics: metrics.into_iter().collect(),
        },
        entries: vec![entry],
    };

    let path = Path::new("results/bench_append_only_log_bulk_updates_append_local.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create results dir");
    }
    let data = serde_json::to_vec_pretty(&results).expect("serialize results");
    fs::write(path, data).expect("write results");
    println!(
        "{}",
        serde_json::to_string_pretty(&results).expect("render results")
    );
}
