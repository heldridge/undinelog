//! Local append-only-log range-opening benchmark.
//!
//! The setup warms one-claim-per-helper auth servers through the real
//! append-only-log frontier append path. The measured operations are:
//! - one helper serving the planned compact unauthenticated range objects
//! - client-side threshold Combine+Verify over exactly kappa helper replies
//! - client-side error-correction Combine+Verify over enough replies to tolerate
//!   kappa corrupted helpers, using Merkle-guided helper exclusion
//!
//! The client-side timing mirrors the point-opening benchmark: no TCP, network
//! transmission, or timed helper work is included in Combine+Verify. The older
//! localhost client benchmark is only a broad workflow smoke.

#[path = "support/auth_fixture.rs"]
mod auth_fixture;
#[path = "support/fixture_cache.rs"]
mod fixture_cache;
#[path = "support/local_setup.rs"]
mod local_setup;
use auth_fixture::AuthFixture;
use fixture_cache::FixtureCache;

use append_only_log_demo::codec::{bulk_updates_bytes_as_leaves, leaves_to_bulk_updates_bytes};
use append_only_log_demo::frontier::PreparedBulkUpdatesAppend;
use append_only_log_demo::layout::{
    BULK_UPDATES_BYTES, BULK_UPDATES_SIZE, FIELD_ELEMENTS_PER_CODEC_GROUP, LEAF_BYTES,
    PAYLOAD_BYTES_PER_CODEC_GROUP,
};
use paper_a_optimized::auth::client::{AuthClient, AuthTransport, ClientError as AuthClientError};
use paper_a_optimized::auth::mapping::NodeLinearization;
use paper_a_optimized::auth::service::{
    AuthBatchWriteRequest, AuthBatchWriteResponse, AuthFreshSubtreeWriteRequest, AuthReadResponse,
    AuthRequest, AuthResponse, AuthServer, AuthService, AuthSubtreeReadResponse,
    AuthSubtreeUpperPathReadResponse, AuthWriteRequest, AuthWriteResponse,
};
use paper_a_optimized::unauth::{
    ClaimBundle, UnauthOptimizedReadRequest, UnauthOptimizedReadResponse,
    UnauthReadObjectContribution, UnauthReadObjectId, error_correcting_share_target,
};
use paper_a_optimized::{
    AuthConfig, AuthParams, CodedArrayConfig, Config, RowShare, ShardShare, Value256,
    build_coded_params, build_shard_registry_deterministic, fp_one,
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
const DEFAULT_TRIALS: usize = 10;
const DEFAULT_KAPPA: usize = 64;
const DEFAULT_COMMITTED_BULK_UPDATE_BATCHES: usize = 3;

#[derive(Clone)]
struct LocalServer {
    label: String,
    inner: Arc<Mutex<AuthServer>>,
}

#[derive(Clone, Default)]
struct LocalTransport {
    captured_request: Option<Arc<Mutex<Option<UnauthOptimizedReadRequest>>>>,
}

impl AuthTransport for LocalTransport {
    type Server = LocalServer;

    fn supports_fresh_subtree_writes(&self) -> bool {
        true
    }

    fn supports_unauth_optimized_reads(&self) -> bool {
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
                "invalid auth len response: {other:?}"
            ))),
        }
    }

    fn auth_read(
        &self,
        server: &Self::Server,
        index: u64,
        kappa: usize,
    ) -> Result<AuthReadResponse, AuthClientError> {
        let mut guard = server.inner.lock().expect("auth server lock");
        match AuthService::handle_request(&mut *guard, AuthRequest::Read { index, kappa }) {
            AuthResponse::Read { resp } => Ok(resp),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!(
                "invalid auth read response: {other:?}"
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
                "invalid auth subtree response: {other:?}"
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
                "invalid auth subtree upper-path response: {other:?}"
            ))),
        }
    }

    fn unauth_multi_read(
        &self,
        server: &Self::Server,
        indices: &[u64],
    ) -> Result<Vec<Vec<ShardShare>>, AuthClientError> {
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
                "invalid unauth multi-read response: {other:?}"
            ))),
        }
    }

    fn unauth_optimized_read(
        &self,
        server: &Self::Server,
        request: &UnauthOptimizedReadRequest,
    ) -> Result<UnauthOptimizedReadResponse, AuthClientError> {
        if let Some(captured) = &self.captured_request {
            *captured.lock().expect("capture lock") = Some(request.clone());
        }
        let guard = server.inner.lock().expect("auth server lock");
        Ok(guard.handle_optimized_read(request))
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
                "invalid auth write response: {other:?}"
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
                "invalid auth batch-write response: {other:?}"
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
                "invalid auth fresh-subtree response: {other:?}"
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
                "invalid auth extend response: {other:?}"
            ))),
        }
    }
}

#[derive(Clone)]
struct NoopTransport;

impl AuthTransport for NoopTransport {
    type Server = usize;

    fn server_label(&self, server: &Self::Server) -> String {
        format!("replay-{server}")
    }

    fn auth_get_len(&self, _server: &Self::Server) -> Result<u64, AuthClientError> {
        Err(AuthClientError::Net(
            "pre-collected benchmark does not fetch length".to_string(),
        ))
    }

    fn auth_read(
        &self,
        _server: &Self::Server,
        _index: u64,
        _kappa: usize,
    ) -> Result<AuthReadResponse, AuthClientError> {
        Err(AuthClientError::Net(
            "pre-collected benchmark does not fetch reads".to_string(),
        ))
    }

    fn auth_read_subtree(
        &self,
        _server: &Self::Server,
        _start_index: u64,
        _leaf_count: usize,
    ) -> Result<AuthSubtreeReadResponse, AuthClientError> {
        Err(AuthClientError::Net(
            "pre-collected benchmark does not fetch subtree reads".to_string(),
        ))
    }

    fn unauth_multi_read(
        &self,
        _server: &Self::Server,
        _indices: &[u64],
    ) -> Result<Vec<Vec<ShardShare>>, AuthClientError> {
        Err(AuthClientError::Net(
            "pre-collected benchmark does not fetch unauth reads".to_string(),
        ))
    }

    fn auth_write(
        &self,
        _server: &Self::Server,
        _req: &AuthWriteRequest,
    ) -> Result<AuthWriteResponse, AuthClientError> {
        Err(AuthClientError::Net(
            "pre-collected benchmark does not write".to_string(),
        ))
    }

    fn auth_write_batch(
        &self,
        _server: &Self::Server,
        _req: &AuthBatchWriteRequest,
    ) -> Result<AuthBatchWriteResponse, AuthClientError> {
        Err(AuthClientError::Net(
            "pre-collected benchmark does not batch-write".to_string(),
        ))
    }

    fn auth_extend(&self, _server: &Self::Server, _new_len: u64) -> Result<bool, AuthClientError> {
        Err(AuthClientError::Net(
            "pre-collected benchmark does not extend".to_string(),
        ))
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
    inner_repetitions: usize,
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
    committed_bulk_update_batches: usize,
    query_bulk_update_batch: usize,
    query_byte_offset: usize,
    query_global_byte_offset: usize,
    query_byte_len: usize,
    query_leaf_start: u64,
    query_leaf_count: usize,
    timestamp_unix_ms: u64,
    git: Option<String>,
    notes: String,
    metrics: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct BenchEntry {
    #[serde(rename = "Seg")]
    seg: usize,
    #[serde(rename = "encoded_cells_N")]
    encoded_cells_n: usize,
    encoded_segment_bytes: usize,
    kappa: usize,
    primary_slice_field_elements: usize,
    registry_helper_count: usize,
    bulk_updates_bytes: usize,
    inserted_values: usize,
    range_bytes: usize,
    range_leaf_count: usize,
    request_objects: usize,
    request_segments: usize,
    request_points: usize,
    threshold_helper_replies: usize,
    helper_segment_shards: usize,
    helper_point_shares: usize,
    helper_response_field_elements: usize,
    helper_response_serialized_bytes: usize,
    correcting_helper_replies: usize,
    bad_helper_replies: usize,
    proof_nodes: usize,
    client_returned_leaves: usize,
    helper_range_share_core_ns: MetricStats,
    helper_range_share_serialized_ns: MetricStats,
    client_threshold_combine_verify_ns: MetricStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    split_range_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    split_threshold_request_segments: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    split_threshold_request_points: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    split_threshold_helper_response_field_elements: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    split_threshold_combine_verify_ns: Option<MetricStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    split_correction_budget_combine_verify_ns: Option<MetricStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    split_kappa_failures_filtered_combine_verify_ns: Option<MetricStats>,
    client_correction_budget_combine_verify_ns: MetricStats,
    client_kappa_failures_filtered_combine_verify_ns: MetricStats,
}

struct WarmedRange {
    allocated_leaves: usize,
    trusted_root: Value256,
    request: UnauthOptimizedReadRequest,
    full_response: UnauthOptimizedReadResponse,
    threshold_responses: Vec<UnauthOptimizedReadResponse>,
    query_leaf_start: u64,
    query_leaf_count: usize,
    query_byte_len: usize,
    query_local_byte_offset: usize,
    expected_bytes: Vec<u8>,
    proof_nodes: usize,
}

struct WarmedCluster {
    servers: Vec<LocalServer>,
    allocated_leaves: usize,
    trusted_root: Value256,
}

#[derive(Clone, Copy)]
struct QuerySpec {
    query_bulk_updates: usize,
    query_byte_offset: usize,
    query_global_byte_offset: usize,
    query_byte_len: usize,
    query_leaf_start: u64,
    query_leaf_count: usize,
    query_local_byte_offset: usize,
}

fn percentile(sorted: &[u64], pct: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    sorted[(pct * (sorted.len() - 1)) / 100]
}

fn stats_from_samples(mut samples: Vec<u64>) -> MetricStats {
    if samples.is_empty() {
        return MetricStats {
            mean: 0,
            p50: 0,
            p95: 0,
            trials: 0,
            samples: Vec::new(),
            inner_repetitions: 1,
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
        inner_repetitions: 1,
    }
}

fn bench_metric<T>(trials: usize, mut f: impl FnMut() -> T) -> MetricStats {
    let mut samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        let start = Instant::now();
        black_box(f());
        samples.push(start.elapsed().as_nanos() as u64);
    }
    stats_from_samples(samples)
}

/// Each batch mean is one sample; inner calls never increase the CI's n.
fn bench_helper_metric<T>(trials: usize, mut f: impl FnMut() -> T) -> MetricStats {
    const INNER: usize = 32;
    let mut samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        let start = Instant::now();
        for _ in 0..INNER {
            // Keep response destruction in the scope, as in bench_metric.
            black_box(f());
        }
        samples.push((start.elapsed().as_nanos() / INNER as u128).max(1) as u64);
    }
    let mut metric = stats_from_samples(samples);
    metric.inner_repetitions = INNER;
    metric
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

fn parse_usize_list_env(name: &str) -> Option<Vec<usize>> {
    let values = std::env::var(name)
        .ok()?
        .split(',')
        .filter_map(|raw| {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return None;
            }
            trimmed.parse::<usize>().ok().filter(|value| *value > 0)
        })
        .collect::<Vec<_>>();
    (!values.is_empty()).then_some(values)
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

fn build_config(kappa: usize) -> Config {
    let tree_height = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_RANGE_OPEN_TREE_HEIGHT",
        DEFAULT_TREE_HEIGHT,
    );
    assert!(kappa.is_power_of_two(), "kappa must be a power of two");
    assert!(kappa >= 2, "kappa must be at least two");
    // The segment size is derived from the threshold.
    let seg = kappa * kappa;
    let registry_helper_count = 3 * kappa;
    Config {
        auth: AuthConfig {
            tree_height,
            alloc_leaf_batch: seg / 2,
            node_linearization: NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            kappa,
            shard_count: registry_helper_count,
            shard_registry: build_shard_registry_deterministic(registry_helper_count),
            party_registry_indices: (0..registry_helper_count).map(|idx| vec![idx]).collect(),
            seed: BENCH_SEED,
        },
    }
}

fn build_setup(kappa: usize) -> BenchSetup {
    let cfg = build_config(kappa);
    let params = build_coded_params(&cfg.coded);
    let auth = AuthParams::new(cfg.auth.tree_height);
    BenchSetup { cfg, params, auth }
}

fn make_full_server(setup: &BenchSetup) -> LocalServer {
    LocalServer {
        label: "full-local".to_string(),
        inner: Arc::new(Mutex::new(
            AuthServer::new(
                setup.params.clone(),
                setup.auth.clone(),
                setup.cfg.auth.node_linearization,
                0,
            )
            .expect("build full auth server"),
        )),
    }
}

/// Warm one helper: a server bound to the claim at registry index 0 that
/// receives the same bulk-update appends as the full server.
fn build_warmed_helper(setup: &BenchSetup, committed_bulk_update_batches: usize) -> AuthServer {
    let claim = ClaimBundle::exact_for_registry_index(&setup.params, 0)
        .expect("exact claim should exist for registry index 0");
    let cache = range_fixture_cache(setup, committed_bulk_update_batches, "helper");
    if let Some(fixture) = cache.load::<AuthFixture>() {
        return fixture.restore(
            &setup.params,
            &setup.auth,
            setup.cfg.auth.node_linearization,
            Some(vec![claim]),
        );
    }
    let server = AuthServer::new_with_assignments_and_claims(
        setup.params.clone(),
        setup.auth.clone(),
        setup.cfg.auth.node_linearization,
        0,
        None,
        Some(vec![claim]),
    )
    .expect("build helper auth server");
    let servers = vec![LocalServer {
        label: "helper-local".to_string(),
        inner: Arc::new(Mutex::new(server)),
    }];
    let mut client = make_client(&servers, setup, None);
    let mut planner = PreparedBulkUpdatesAppend::new(setup.cfg.auth.tree_height).expect("planner");
    let mut frontier = None;
    let mut frontier_len = 0u64;
    for bulk_updates_idx in 0..committed_bulk_update_batches {
        let bulk_updates =
            build_exact_bulk_updates(BENCH_SEED ^ 0xA110000 ^ bulk_updates_idx as u64);
        let leaves = bulk_updates_bytes_as_leaves(&bulk_updates).expect("bulk updates leaves");
        let next_frontier = planner
            .append_from_frontier(&mut client, frontier_len, frontier.as_ref(), &leaves)
            .expect("append committed bulk update to the helper");
        frontier_len += BULK_UPDATES_SIZE as u64;
        frontier = Some(next_frontier);
    }
    drop(client);
    let local = servers.into_iter().next().expect("one setup helper");
    let helper = local_setup::take_local_server(local.inner);
    if cache.enabled() {
        cache.store(&AuthFixture::capture(&helper));
    }
    helper
}

/// One helper's answer to a range request, serialized for the wire.
fn serialized_helper_response(
    helper: &AuthServer,
    request: &UnauthOptimizedReadRequest,
) -> Vec<u8> {
    optimized_response_wire_bytes(&helper.handle_optimized_read(request))
}

fn make_client(
    servers: &[LocalServer],
    setup: &BenchSetup,
    captured_request: Option<Arc<Mutex<Option<UnauthOptimizedReadRequest>>>>,
) -> AuthClient<LocalTransport> {
    AuthClient::new(
        LocalTransport { captured_request },
        setup.params.clone(),
        setup.auth.clone(),
        setup.cfg.auth.node_linearization,
        0,
        servers[0].clone(),
        servers.to_vec(),
    )
}

fn make_processing_client(
    setup: &BenchSetup,
    allocated_leaves: usize,
    trusted_root: Value256,
) -> AuthClient<NoopTransport> {
    let mut client = AuthClient::new(
        NoopTransport,
        setup.params.clone(),
        setup.auth.clone(),
        setup.cfg.auth.node_linearization,
        allocated_leaves,
        0,
        Vec::new(),
    );
    client.set_trusted_root(trusted_root);
    client
}

fn build_exact_bulk_updates(seed: u64) -> Vec<u8> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut bulk_updates = Vec::with_capacity(BULK_UPDATES_BYTES);
    for _ in 0..BULK_UPDATES_SIZE {
        bulk_updates.extend_from_slice(&paper_a_optimized::random_value(&mut rng));
    }
    bulk_updates
}

fn byte_range_to_leaf_range(offset: usize, len: usize) -> (usize, usize, usize) {
    let end = offset.checked_add(len).expect("byte range overflow");
    let start_group = offset / PAYLOAD_BYTES_PER_CODEC_GROUP;
    let end_group = end.div_ceil(PAYLOAD_BYTES_PER_CODEC_GROUP);
    let start_leaf = start_group * FIELD_ELEMENTS_PER_CODEC_GROUP;
    let end_leaf = end_group * FIELD_ELEMENTS_PER_CODEC_GROUP;
    let local_start = offset % PAYLOAD_BYTES_PER_CODEC_GROUP;
    (start_leaf, end_leaf - start_leaf, local_start)
}

fn exact_committed_range_bytes(global_byte_offset: usize, len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    let mut cursor = global_byte_offset;
    while out.len() < len {
        let bulk_updates_idx = cursor / BULK_UPDATES_BYTES;
        let local_offset = cursor % BULK_UPDATES_BYTES;
        let bulk_updates =
            build_exact_bulk_updates(BENCH_SEED ^ 0xA110000 ^ bulk_updates_idx as u64);
        let take = (len - out.len()).min(BULK_UPDATES_BYTES - local_offset);
        out.extend_from_slice(&bulk_updates[local_offset..local_offset + take]);
        cursor += take;
    }
    out
}

fn response_counts(response: &UnauthOptimizedReadResponse) -> (usize, usize, usize, usize) {
    let mut segment_shards = 0usize;
    let mut block_rows = 0usize;
    let mut point_shares = 0usize;
    let mut field_elements = 0usize;
    for contribution in &response.objects {
        match contribution {
            UnauthReadObjectContribution::Segment { shards, .. } => {
                segment_shards += shards.len();
                field_elements += shards.iter().map(|shard| shard.values.len()).sum::<usize>();
            }
            UnauthReadObjectContribution::Block { rows, .. } => {
                block_rows += rows.len();
                field_elements += rows.len();
            }
            UnauthReadObjectContribution::Point { shares, .. } => {
                let nonempty = shares
                    .iter()
                    .filter(|share| share.value().is_some())
                    .count();
                point_shares += nonempty;
                field_elements += nonempty;
            }
        }
    }
    (segment_shards, block_rows, point_shares, field_elements)
}

fn optimized_response_wire_bytes(response: &UnauthOptimizedReadResponse) -> Vec<u8> {
    let net_response =
        paper_a_optimized::net::unauth::core_optimized_response_to_net(response.clone())
            .expect("pack optimized response");
    let wire_response =
        paper_a_optimized::net::Response::UnauthOptimizedRead { resp: net_response };
    paper_a_optimized::net::wire::encode_binary(&wire_response)
        .expect("encode optimized range response")
}

fn optimized_response_wire_len(response: &UnauthOptimizedReadResponse) -> usize {
    optimized_response_wire_bytes(response).len()
}

fn request_counts(request: &UnauthOptimizedReadRequest) -> (usize, usize, usize) {
    let mut segments = 0usize;
    let mut blocks = 0usize;
    let mut points = 0usize;
    for object in &request.objects {
        match object {
            UnauthReadObjectId::Segment { .. } => segments += 1,
            UnauthReadObjectId::Block { .. } => blocks += 1,
            UnauthReadObjectId::Point { .. } => points += 1,
        }
    }
    (segments, blocks, points)
}

fn split_full_optimized_response(
    full_response: &UnauthOptimizedReadResponse,
    helper_count: usize,
) -> Vec<UnauthOptimizedReadResponse> {
    (0..helper_count)
        .map(|registry_index| {
            let objects = full_response
                .objects
                .iter()
                .map(|contribution| match contribution {
                    UnauthReadObjectContribution::Segment { object_pos, shards } => {
                        UnauthReadObjectContribution::Segment {
                            object_pos: *object_pos,
                            shards: shards
                                .iter()
                                .filter(|shard| shard.registry_index == registry_index)
                                .cloned()
                                .collect(),
                        }
                    }
                    UnauthReadObjectContribution::Block { object_pos, rows } => {
                        UnauthReadObjectContribution::Block {
                            object_pos: *object_pos,
                            rows: rows.clone(),
                        }
                    }
                    UnauthReadObjectContribution::Point { object_pos, shares } => {
                        UnauthReadObjectContribution::Point {
                            object_pos: *object_pos,
                            shares: shares
                                .iter()
                                .filter(|share| share.registry_index() == registry_index)
                                .cloned()
                                .collect(),
                        }
                    }
                })
                .collect();
            UnauthOptimizedReadResponse {
                helper_index: Some(registry_index),
                objects,
            }
        })
        .collect()
}

fn corrupt_optimized_responses(
    responses: &[UnauthOptimizedReadResponse],
    bad_helper_count: usize,
) -> Vec<UnauthOptimizedReadResponse> {
    let mut corrupted = responses.to_vec();
    for (bad_pos, response) in corrupted.iter_mut().take(bad_helper_count).enumerate() {
        if response.objects.is_empty() {
            continue;
        }
        let object_pos_to_corrupt = bad_pos % response.objects.len();
        for (object_pos, contribution) in response.objects.iter_mut().enumerate() {
            if object_pos != object_pos_to_corrupt {
                continue;
            }
            match contribution {
                UnauthReadObjectContribution::Segment { shards, .. } => {
                    for shard in shards {
                        let mut delta = fp_one();
                        for value in &mut shard.values {
                            *value += delta;
                            delta += fp_one();
                        }
                    }
                }
                UnauthReadObjectContribution::Block { rows, .. } => {
                    for row in rows {
                        *row = RowShare::new(row.row_label(), row.value() + fp_one());
                    }
                }
                UnauthReadObjectContribution::Point { shares, .. } => {
                    for share in shares {
                        let corrupted_value = share.value().map(|value| value + fp_one());
                        *share = ShardShare::new(share.registry_index(), corrupted_value);
                    }
                }
            }
        }
    }
    corrupted
}

fn assert_processed_range(
    leaves: Vec<Value256>,
    proof: paper_a_optimized::auth::AuthRangeProof,
    warmed: &WarmedRange,
) {
    assert_verified_range_shape(&leaves, &proof, warmed);
    let bytes = leaves_to_bulk_updates_bytes(&leaves).expect("range bytes");
    assert_eq!(
        bytes[warmed.query_local_byte_offset
            ..warmed.query_local_byte_offset + warmed.query_byte_len],
        warmed.expected_bytes
    );
}

fn assert_verified_range_shape(
    leaves: &[Value256],
    proof: &paper_a_optimized::auth::AuthRangeProof,
    warmed: &WarmedRange,
) {
    assert_eq!(proof.root, warmed.trusted_root);
    assert_eq!(proof.start_index, warmed.query_leaf_start);
    assert_eq!(proof.leaf_count, warmed.query_leaf_count);
    assert_eq!(leaves.len(), warmed.query_leaf_count);
}

fn range_fixture_cache(setup: &BenchSetup, batches: usize, role: &str) -> FixtureCache {
    FixtureCache::new(
        "public-range",
        &(
            role,
            BENCH_SEED,
            setup.auth.tree_height,
            setup.params.need,
            setup.params.segment_len,
            setup.params.shard_registry.len(),
            BULK_UPDATES_BYTES,
            batches,
        ),
    )
}

fn build_warmed_cluster(setup: &BenchSetup, committed_bulk_update_batches: usize) -> WarmedCluster {
    assert!(committed_bulk_update_batches > 0);
    let cache = range_fixture_cache(setup, committed_bulk_update_batches, "cluster");
    if let Some(fixture) = cache.load::<AuthFixture>() {
        let server = fixture.restore(
            &setup.params,
            &setup.auth,
            setup.cfg.auth.node_linearization,
            None,
        );
        return WarmedCluster {
            allocated_leaves: server.allocated_leaves() as usize,
            trusted_root: server.store.root,
            servers: vec![LocalServer {
                label: "full-local".into(),
                inner: Arc::new(Mutex::new(server)),
            }],
        };
    }
    let servers = vec![make_full_server(setup)];
    let mut client = make_client(&servers, setup, None);
    let mut planner = PreparedBulkUpdatesAppend::new(setup.cfg.auth.tree_height).expect("planner");
    let mut frontier = None;
    let mut frontier_len = 0u64;

    for bulk_updates_idx in 0..committed_bulk_update_batches {
        let bulk_updates =
            build_exact_bulk_updates(BENCH_SEED ^ 0xA110000 ^ bulk_updates_idx as u64);
        let leaves = bulk_updates_bytes_as_leaves(&bulk_updates).expect("bulk updates leaves");
        let next_frontier = planner
            .append_from_frontier(&mut client, frontier_len, frontier.as_ref(), &leaves)
            .expect("append committed bulk update");
        frontier_len += BULK_UPDATES_SIZE as u64;
        frontier = Some(next_frontier);
    }

    let trusted_root = client.trusted_root().expect("trusted root after appends");
    let allocated_leaves = client
        .current_len()
        .expect("allocated length after appends") as usize;

    if cache.enabled() {
        cache.store(&AuthFixture::capture(
            &servers[0].inner.lock().expect("cluster fixture"),
        ));
    }
    WarmedCluster {
        servers,
        allocated_leaves,
        trusted_root,
    }
}

fn query_spec(
    setup: &BenchSetup,
    committed_bulk_update_batches: usize,
    query_byte_len_override: Option<usize>,
) -> QuerySpec {
    let query_bulk_updates_default = if committed_bulk_update_batches >= 2 {
        1
    } else {
        0
    };
    let query_bulk_updates = parse_usize_env_allow_zero(
        "BENCH_APPEND_ONLY_LOG_RANGE_OPEN_QUERY_BULK_UPDATES",
        query_bulk_updates_default,
    )
    .min(committed_bulk_update_batches - 1);
    let total_committed_bytes = committed_bulk_update_batches
        .checked_mul(BULK_UPDATES_BYTES)
        .expect("committed byte count overflow");
    let query_byte_offset =
        parse_usize_env_allow_zero("BENCH_APPEND_ONLY_LOG_RANGE_OPEN_BYTE_OFFSET", 0)
            .min(BULK_UPDATES_BYTES - 1);
    let query_global_byte_offset = query_bulk_updates
        .checked_mul(BULK_UPDATES_BYTES)
        .and_then(|base| base.checked_add(query_byte_offset))
        .expect("query byte offset overflow");
    assert!(
        query_global_byte_offset < total_committed_bytes,
        "query start exceeds committed bytes"
    );
    let default_range_len = (setup.params.segment_len * LEAF_BYTES)
        .min(total_committed_bytes - query_global_byte_offset);
    let max_len = total_committed_bytes - query_global_byte_offset;
    let query_byte_len = query_byte_len_override
        .unwrap_or_else(|| {
            parse_usize_env(
                "BENCH_APPEND_ONLY_LOG_RANGE_OPEN_BYTE_LEN",
                default_range_len,
            )
        })
        .min(max_len);
    let (start_leaf, leaf_count, local_start) =
        byte_range_to_leaf_range(query_global_byte_offset, query_byte_len);

    QuerySpec {
        query_bulk_updates,
        query_byte_offset,
        query_global_byte_offset,
        query_byte_len,
        query_leaf_start: start_leaf as u64,
        query_leaf_count: leaf_count,
        query_local_byte_offset: local_start,
    }
}

fn split_query_specs(spec: QuerySpec, split_count: usize) -> Vec<QuerySpec> {
    assert!(split_count > 0);
    assert_eq!(
        spec.query_byte_len % split_count,
        0,
        "split ranges must divide the query byte length"
    );
    let split_len = spec.query_byte_len / split_count;
    assert!(split_len > 0);
    (0..split_count)
        .map(|split| {
            let query_global_byte_offset = spec
                .query_global_byte_offset
                .checked_add(split.checked_mul(split_len).expect("split offset overflow"))
                .expect("split global byte offset overflow");
            let (start_leaf, leaf_count, local_start) =
                byte_range_to_leaf_range(query_global_byte_offset, split_len);
            QuerySpec {
                query_bulk_updates: query_global_byte_offset / BULK_UPDATES_BYTES,
                query_byte_offset: query_global_byte_offset % BULK_UPDATES_BYTES,
                query_global_byte_offset,
                query_byte_len: split_len,
                query_leaf_start: start_leaf as u64,
                query_leaf_count: leaf_count,
                query_local_byte_offset: local_start,
            }
        })
        .collect()
}

fn build_warmed_range(setup: &BenchSetup, cluster: &WarmedCluster, spec: QuerySpec) -> WarmedRange {
    let expected_bytes =
        exact_committed_range_bytes(spec.query_global_byte_offset, spec.query_byte_len);
    let captured_request = Arc::new(Mutex::new(None));
    let mut read_client = make_client(&cluster.servers, setup, Some(Arc::clone(&captured_request)));
    read_client.set_trusted_root(cluster.trusted_root);
    let (leaves, proof) = read_client
        .read_range_with_proof(spec.query_leaf_start, spec.query_leaf_count)
        .expect("warm range open");
    let range_bytes = leaves_to_bulk_updates_bytes(&leaves).expect("range bytes");
    assert_eq!(
        range_bytes
            [spec.query_local_byte_offset..spec.query_local_byte_offset + spec.query_byte_len],
        expected_bytes
    );

    let request = captured_request
        .lock()
        .expect("request capture lock")
        .clone()
        .expect("captured optimized request");
    let full_response = {
        let guard = cluster.servers[0].inner.lock().expect("auth server lock");
        guard.handle_optimized_read(&request)
    };
    let threshold_responses = split_full_optimized_response(&full_response, setup.params.need);
    WarmedRange {
        allocated_leaves: cluster.allocated_leaves,
        trusted_root: cluster.trusted_root,
        request,
        full_response,
        threshold_responses,
        query_leaf_start: spec.query_leaf_start,
        query_leaf_count: spec.query_leaf_count,
        query_byte_len: spec.query_byte_len,
        query_local_byte_offset: spec.query_local_byte_offset,
        expected_bytes,
        proof_nodes: proof.proof_nodes.len(),
    }
}

fn bench_range_open(
    trials: usize,
    setup: &BenchSetup,
    helper: &AuthServer,
    warmed: WarmedRange,
    split_ranges: Option<&[WarmedRange]>,
    metrics: &BTreeSet<String>,
) -> BenchEntry {
    let core_response = warmed
        .threshold_responses
        .first()
        .cloned()
        .expect("threshold helper response");
    let (helper_segment_shards, helper_block_rows, helper_point_shares, helper_response_elements) =
        response_counts(&core_response);
    assert_eq!(
        helper_block_rows, 0,
        "range planner should use segments plus point leftovers, not row-backed blocks"
    );
    let helper_response_serialized_bytes = optimized_response_wire_len(&core_response);

    // The helper holds one claim of the same log. Its answer must be the
    // share of that claim in the full server's answer.
    assert_eq!(
        serialized_helper_response(helper, &warmed.request),
        optimized_response_wire_bytes(&core_response),
        "the helper's answer differs from its share of the full answer"
    );
    let helper_range_share_core_ns = if metrics.contains("core") {
        bench_helper_metric(trials, || helper.handle_optimized_read(&warmed.request))
    } else {
        stats_from_samples(Vec::new())
    };
    let helper_range_share_serialized_ns = if metrics.contains("serialized") {
        bench_helper_metric(trials, || {
            serialized_helper_response(helper, &warmed.request)
        })
    } else {
        stats_from_samples(Vec::new())
    };
    let threshold_helper_replies = setup.params.need;
    let correcting_helper_replies =
        error_correcting_share_target(setup.params.need, setup.params.need);
    let bad_helper_replies = setup.params.need;
    assert_eq!(warmed.threshold_responses.len(), threshold_helper_replies);
    let correcting_responses =
        split_full_optimized_response(&warmed.full_response, correcting_helper_replies);
    assert_eq!(correcting_responses.len(), correcting_helper_replies);
    let correcting_bad_responses =
        corrupt_optimized_responses(&correcting_responses, bad_helper_replies);
    let split_correcting_responses = split_ranges.map(|ranges| {
        ranges
            .iter()
            .map(|range| {
                let responses =
                    split_full_optimized_response(&range.full_response, correcting_helper_replies);
                assert_eq!(responses.len(), correcting_helper_replies);
                responses
            })
            .collect::<Vec<_>>()
    });
    let split_correcting_bad_responses = split_correcting_responses.as_ref().map(|all_responses| {
        all_responses
            .iter()
            .map(|responses| corrupt_optimized_responses(responses, bad_helper_replies))
            .collect::<Vec<_>>()
    });

    let mut check_client =
        make_processing_client(setup, warmed.allocated_leaves, warmed.trusted_root);
    let (check_leaves, check_proof) = check_client
        .read_range_with_proof_from_optimized_responses(
            warmed.query_leaf_start,
            warmed.query_leaf_count,
            &warmed.threshold_responses,
        )
        .expect("threshold pre-collected range open");
    assert_processed_range(check_leaves, check_proof, &warmed);

    let mut correcting_clean_client =
        make_processing_client(setup, warmed.allocated_leaves, warmed.trusted_root);
    let (correcting_clean_leaves, correcting_clean_proof) = correcting_clean_client
        .read_range_with_proof_from_optimized_responses_correcting(
            warmed.query_leaf_start,
            warmed.query_leaf_count,
            &correcting_responses,
            setup.params.need,
        )
        .expect("no-corrupted-helper error-correction pre-collected range open");
    assert_processed_range(correcting_clean_leaves, correcting_clean_proof, &warmed);

    let mut correcting_bad_filtered_client =
        make_processing_client(setup, warmed.allocated_leaves, warmed.trusted_root);
    let (correcting_bad_filtered_leaves, correcting_bad_filtered_proof) =
        correcting_bad_filtered_client
            .read_range_with_proof_from_optimized_responses_correcting(
                warmed.query_leaf_start,
                warmed.query_leaf_count,
                &correcting_bad_responses,
                setup.params.need,
            )
            .expect("corrupted-helper error-correction pre-collected range open");
    assert_processed_range(
        correcting_bad_filtered_leaves,
        correcting_bad_filtered_proof,
        &warmed,
    );

    let client_threshold_combine_verify_ns = if metrics.contains("threshold") {
        bench_metric(trials, || {
            let mut client =
                make_processing_client(setup, warmed.allocated_leaves, warmed.trusted_root);
            let (leaves, proof) = client
                .read_range_with_proof_from_optimized_responses(
                    warmed.query_leaf_start,
                    warmed.query_leaf_count,
                    &warmed.threshold_responses,
                )
                .expect("range open");
            assert_verified_range_shape(&leaves, &proof, &warmed);
        })
    } else {
        stats_from_samples(Vec::new())
    };
    let split_threshold_combine_verify_ns = split_ranges.map(|ranges| {
        if metrics.contains("split_threshold") {
            bench_metric(trials, || {
                for range in ranges {
                    let mut client =
                        make_processing_client(setup, range.allocated_leaves, range.trusted_root);
                    let (leaves, proof) = client
                        .read_range_with_proof_from_optimized_responses(
                            range.query_leaf_start,
                            range.query_leaf_count,
                            &range.threshold_responses,
                        )
                        .expect("split range open");
                    assert_verified_range_shape(&leaves, &proof, range);
                }
            })
        } else {
            stats_from_samples(Vec::new())
        }
    });
    let split_correction_budget_combine_verify_ns =
        match (split_ranges, split_correcting_responses.as_ref()) {
            (Some(ranges), Some(all_responses)) if metrics.contains("split_correcting_clean") => {
                Some(bench_metric(trials, || {
                    for (range, responses) in ranges.iter().zip(all_responses.iter()) {
                        let mut client = make_processing_client(
                            setup,
                            range.allocated_leaves,
                            range.trusted_root,
                        );
                        let (leaves, proof) = client
                            .read_range_with_proof_from_optimized_responses_correcting(
                                range.query_leaf_start,
                                range.query_leaf_count,
                                responses,
                                setup.params.need,
                            )
                            .expect("split no-corrupted-helper error-correction range open");
                        assert_verified_range_shape(&leaves, &proof, range);
                    }
                }))
            }
            (Some(_), Some(_)) => Some(stats_from_samples(Vec::new())),
            _ => None,
        };
    let client_correction_budget_combine_verify_ns = if metrics.contains("correcting_clean") {
        bench_metric(trials, || {
            let mut client =
                make_processing_client(setup, warmed.allocated_leaves, warmed.trusted_root);
            let (leaves, proof) = client
                .read_range_with_proof_from_optimized_responses_correcting(
                    warmed.query_leaf_start,
                    warmed.query_leaf_count,
                    &correcting_responses,
                    setup.params.need,
                )
                .expect("no-corrupted-helper error-correction pre-collected range open");
            assert_verified_range_shape(&leaves, &proof, &warmed);
        })
    } else {
        stats_from_samples(Vec::new())
    };
    let client_kappa_failures_filtered_combine_verify_ns = if metrics.contains("correcting_corrupt")
    {
        bench_metric(trials, || {
            let mut client =
                make_processing_client(setup, warmed.allocated_leaves, warmed.trusted_root);
            let (leaves, proof) = client
                .read_range_with_proof_from_optimized_responses_correcting(
                    warmed.query_leaf_start,
                    warmed.query_leaf_count,
                    &correcting_bad_responses,
                    setup.params.need,
                )
                .expect("corrupted-helper error-correction pre-collected range open");
            assert_verified_range_shape(&leaves, &proof, &warmed);
        })
    } else {
        stats_from_samples(Vec::new())
    };
    let split_kappa_failures_filtered_combine_verify_ns =
        match (split_ranges, split_correcting_bad_responses.as_ref()) {
            (Some(ranges), Some(all_responses)) if metrics.contains("split_correcting_corrupt") => {
                Some(bench_metric(trials, || {
                    for (range, responses) in ranges.iter().zip(all_responses.iter()) {
                        let mut client = make_processing_client(
                            setup,
                            range.allocated_leaves,
                            range.trusted_root,
                        );
                        let (leaves, proof) = client
                            .read_range_with_proof_from_optimized_responses_correcting(
                                range.query_leaf_start,
                                range.query_leaf_count,
                                responses,
                                setup.params.need,
                            )
                            .expect("split bad-helper filtered correcting range open");
                        assert_verified_range_shape(&leaves, &proof, range);
                    }
                }))
            }
            (Some(_), Some(_)) => Some(stats_from_samples(Vec::new())),
            _ => None,
        };

    let (request_segments, request_blocks, request_points) = request_counts(&warmed.request);
    assert_eq!(
        request_blocks, 0,
        "range planner should use segments plus point leftovers, not row-backed blocks"
    );
    let (
        split_range_count,
        split_threshold_request_segments,
        split_threshold_request_points,
        split_threshold_helper_response_field_elements,
    ) = split_ranges
        .map(|ranges| {
            let mut segments = 0usize;
            let mut points = 0usize;
            let mut fields = 0usize;
            for range in ranges {
                let (range_segments, range_blocks, range_points) = request_counts(&range.request);
                assert_eq!(
                    range_blocks, 0,
                    "range planner should use segments plus point leftovers, not row-backed blocks"
                );
                segments += range_segments;
                points += range_points;
                fields += range
                    .threshold_responses
                    .first()
                    .map(response_counts)
                    .map(|(_, _, _, elements)| elements)
                    .unwrap_or(0);
            }
            (
                Some(ranges.len()),
                Some(segments),
                Some(points),
                Some(fields),
            )
        })
        .unwrap_or((None, None, None, None));
    BenchEntry {
        seg: setup.params.segment_len,
        encoded_cells_n: setup.params.n_grid,
        encoded_segment_bytes: setup.params.n_grid * LEAF_BYTES,
        kappa: setup.params.need,
        primary_slice_field_elements: setup.params.s,
        registry_helper_count: setup.params.shard_registry.len(),
        bulk_updates_bytes: BULK_UPDATES_BYTES,
        inserted_values: BULK_UPDATES_SIZE,
        range_bytes: warmed.query_byte_len,
        range_leaf_count: warmed.query_leaf_count,
        request_objects: warmed.request.objects.len(),
        request_segments,
        request_points,
        threshold_helper_replies,
        helper_segment_shards,
        helper_point_shares,
        helper_response_field_elements: helper_response_elements,
        helper_response_serialized_bytes,
        correcting_helper_replies,
        bad_helper_replies,
        proof_nodes: warmed.proof_nodes,
        client_returned_leaves: warmed.query_leaf_count,
        helper_range_share_core_ns,
        helper_range_share_serialized_ns,
        client_threshold_combine_verify_ns,
        split_range_count,
        split_threshold_request_segments,
        split_threshold_request_points,
        split_threshold_helper_response_field_elements,
        split_threshold_combine_verify_ns,
        split_correction_budget_combine_verify_ns,
        split_kappa_failures_filtered_combine_verify_ns,
        client_correction_budget_combine_verify_ns,
        client_kappa_failures_filtered_combine_verify_ns,
    }
}

fn write_benchmark_results(meta: &BenchMeta, entries: &[BenchEntry]) {
    let results = BenchResults {
        meta: meta.clone(),
        entries: entries.to_vec(),
    };
    let out_dir = Path::new("results");
    let _ = fs::create_dir_all(out_dir);
    let path = out_dir.join("bench_append_only_log_range_open_local.json");
    let tmp_path = out_dir.join("bench_append_only_log_range_open_local.json.tmp");
    let data =
        serde_json::to_vec_pretty(&results).expect("serialize bench_append_only_log_range_open");
    fs::write(&tmp_path, data).expect("write bench_append_only_log_range_open_local.json.tmp");
    fs::rename(&tmp_path, &path).expect("publish bench_append_only_log_range_open_local.json");
}

fn run_benchmark() {
    let trials = parse_usize_env("BENCH_APPEND_ONLY_LOG_RANGE_OPEN_TRIALS", DEFAULT_TRIALS);
    let kappa = parse_usize_env("BENCH_APPEND_ONLY_LOG_RANGE_OPEN_KAPPA", DEFAULT_KAPPA);
    let committed_bulk_update_batches = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_RANGE_OPEN_COMMITTED_BULK_UPDATE_BATCHES",
        DEFAULT_COMMITTED_BULK_UPDATE_BATCHES,
    );
    let split_count = parse_usize_env("BENCH_APPEND_ONLY_LOG_RANGE_OPEN_SPLIT_COUNT", 1);
    let metrics = parse_metric_filter(
        "BENCH_APPEND_ONLY_LOG_RANGE_OPEN_METRICS",
        &[
            "core",
            "serialized",
            "threshold",
            "correcting_clean",
            "correcting_corrupt",
            "split_threshold",
            "split_correcting_clean",
            "split_correcting_corrupt",
        ],
    );
    let setup = build_setup(kappa);

    let query_byte_lens = parse_usize_list_env("BENCH_APPEND_ONLY_LOG_RANGE_OPEN_BYTE_LEN_LIST")
        .unwrap_or_else(|| {
            vec![parse_usize_env(
                "BENCH_APPEND_ONLY_LOG_RANGE_OPEN_BYTE_LEN",
                (setup.params.segment_len * LEAF_BYTES).min(BULK_UPDATES_BYTES),
            )]
        });
    let specs = query_byte_lens
        .into_iter()
        .map(|query_byte_len| {
            query_spec(&setup, committed_bulk_update_batches, Some(query_byte_len))
        })
        .collect::<Vec<_>>();
    let first_spec = specs
        .first()
        .copied()
        .expect("at least one range length to benchmark");

    let cluster = build_warmed_cluster(&setup, committed_bulk_update_batches);
    let helper = build_warmed_helper(&setup, committed_bulk_update_batches);
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let meta = BenchMeta {
        seed: BENCH_SEED,
        trials,
        committed_bulk_update_batches,
        query_bulk_update_batch: first_spec.query_bulk_updates,
        query_byte_offset: first_spec.query_byte_offset,
        query_global_byte_offset: first_spec.query_global_byte_offset,
        query_byte_len: first_spec.query_byte_len,
        query_leaf_start: first_spec.query_leaf_start,
        query_leaf_count: first_spec.query_leaf_count,
        timestamp_unix_ms: timestamp_ms,
        git: git_sha(),
        notes: "clean append-only-log range-opening benchmark: warm one full local app state and one one-claim helper from the same real bulk-update appends; time the helper's answer to the captured optimized range request (handler, and handler plus serialization), split the full server's answer into registry-index helper replies, and time the local client verified range-open metrics for each requested range length; results are flushed after each completed range length".to_string(),
        metrics: metrics.iter().cloned().collect(),
    };

    let mut entries = Vec::new();
    write_benchmark_results(&meta, &entries);
    for spec in specs {
        let warmed = build_warmed_range(&setup, &cluster, spec);
        let entry = if split_count > 1 {
            let split_ranges = split_query_specs(spec, split_count)
                .into_iter()
                .map(|split_spec| build_warmed_range(&setup, &cluster, split_spec))
                .collect::<Vec<_>>();
            bench_range_open(
                trials,
                &setup,
                &helper,
                warmed,
                Some(&split_ranges),
                &metrics,
            )
        } else {
            bench_range_open(trials, &setup, &helper, warmed, None, &metrics)
        };
        entries.push(entry);
        write_benchmark_results(&meta, &entries);
    }
}

fn main() {
    run_benchmark();
}

#[test]
#[ignore]
fn bench_append_only_log_range_open_local() {
    run_benchmark();
}
