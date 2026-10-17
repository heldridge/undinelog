//! Local per-server benchmark for one authenticated point-opening request in the
//! append-only-log application.
//!
//! The setup warms a single claim-bound server by appending several committed
//! log bulk update batches through the real app frontier path. The timed operation is then a
//! single-leaf authenticated open request against that warmed server.

#[path = "support/local_setup.rs"]
mod local_setup;

use append_only_log_demo::codec::bulk_updates_bytes_as_leaves;
use append_only_log_demo::frontier::PreparedBulkUpdatesAppend;
use append_only_log_demo::layout::{
    BULK_UPDATES_BYTES, BULK_UPDATES_SIZE, bulk_updates_start_leaf_index,
};
use paper_a_optimized::auth::client::{AuthClient, AuthTransport, ClientError as AuthClientError};
use paper_a_optimized::auth::mapping::NodeLinearization;
use paper_a_optimized::auth::service::{
    AuthBatchWriteRequest, AuthBatchWriteResponse, AuthFreshSubtreeWriteRequest, AuthReadResponse,
    AuthRequest, AuthResponse, AuthServer, AuthService, AuthSubtreeReadResponse,
    AuthSubtreeUpperPathReadResponse, AuthWriteRequest, AuthWriteResponse,
};
use paper_a_optimized::coded_array::params::ROWS_PER_CLAIM;
use paper_a_optimized::net;
use paper_a_optimized::net::wire::AuthReadRequest as NetAuthReadRequest;
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
const DEFAULT_TRIALS: usize = 20;
const DEFAULT_KAPPA: usize = 64;
const DEFAULT_LOCAL_CLAIMS: usize = 1;
const DEFAULT_COMMITTED_BULK_UPDATE_BATCHES: usize = 3;

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
        kappa: usize,
    ) -> Result<AuthReadResponse, AuthClientError> {
        let mut guard = server.inner.lock().expect("auth server lock");
        match AuthService::handle_request(&mut *guard, AuthRequest::Read { index, kappa }) {
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
    committed_bulk_update_batches: usize,
    query_bulk_update_batch: usize,
    query_value_offset: usize,
    query_value_index: u64,
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
    #[serde(rename = "RowsPerClaim")]
    rows_per_claim: usize,
    auxiliary_field_elements_per_claim: usize,
    helper_count: usize,
    local_claims: usize,
    bulk_updates_bytes: usize,
    inserted_values: usize,
    auth_nodes_returned: usize,
    array_shares_returned: usize,
    serialized_response_bytes: usize,
    helper_open_share_core_ns: MetricStats,
    helper_open_share_serialized_ns: MetricStats,
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

fn parse_usize_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
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

fn build_config(kappa: usize) -> Config {
    let tree_height = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_POINT_OPEN_TREE_HEIGHT",
        DEFAULT_TREE_HEIGHT,
    );
    assert!(kappa.is_power_of_two(), "kappa must be a power of two");
    assert!(kappa >= 2, "kappa must be at least two");
    // The segment size is derived from the threshold.
    let seg = kappa * kappa;
    let helper_count = kappa;
    Config {
        auth: AuthConfig {
            tree_height,
            alloc_leaf_batch: seg / 2,
            node_linearization: NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            kappa,
            shard_count: helper_count,
            shard_registry: build_shard_registry_deterministic(helper_count),
            party_registry_indices: (0..helper_count).map(|idx| vec![idx]).collect(),
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
        .map(|idx| claim_for_registry_index(params, idx))
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

fn make_local_server(server: AuthServer) -> LocalServer {
    LocalServer {
        label: "local".to_string(),
        inner: Arc::new(Mutex::new(server)),
    }
}

fn make_client(server: LocalServer, setup: &BenchSetup) -> AuthClient<LocalTransport> {
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

fn build_warmed_server(
    setup: &BenchSetup,
    local_claims: usize,
    committed_bulk_update_batches: usize,
) -> (AuthServer, u64) {
    assert!(
        committed_bulk_update_batches > 0,
        "committed_bulk_update_batches must be positive"
    );
    let local = make_local_server(build_claim_server(setup, local_claims));
    let mut client = make_client(local.clone(), setup);
    let mut planner = PreparedBulkUpdatesAppend::new(setup.cfg.auth.tree_height).expect("planner");
    let mut frontier = None;
    let mut frontier_len = 0u64;

    for bulk_updates_idx in 0..committed_bulk_update_batches {
        let bulk_updates =
            build_exact_bulk_updates(BENCH_SEED ^ 0xA110000 ^ bulk_updates_idx as u64);
        let leaves = bulk_updates_bytes_as_leaves(&bulk_updates).expect("bulk updates leaves");
        let next_frontier = planner
            .append_from_frontier(&mut client, frontier_len, frontier.as_ref(), &leaves)
            .expect("append committed bulk_updates");
        frontier_len += BULK_UPDATES_SIZE as u64;
        frontier = Some(next_frontier);
    }

    let query_bulk_updates_default = if committed_bulk_update_batches >= 2 {
        1
    } else {
        0
    };
    let query_bulk_updates = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_POINT_OPEN_QUERY_BULK_UPDATES",
        query_bulk_updates_default,
    )
    .min(committed_bulk_update_batches - 1);
    let query_leaf_offset = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_POINT_OPEN_QUERY_LEAF_OFFSET",
        BULK_UPDATES_SIZE / 2,
    )
    .min(BULK_UPDATES_SIZE - 1);
    let query_leaf_index =
        bulk_updates_start_leaf_index(query_bulk_updates as u64) + query_leaf_offset as u64;
    drop(client);
    let server = local_setup::take_local_server(local.inner);
    (server, query_leaf_index)
}

fn count_response_shares(resp: &net::Response) -> (usize, usize) {
    match resp {
        net::Response::AuthRead { resp } => {
            let node_count = resp.nodes.len();
            let share_count = resp
                .nodes
                .iter()
                .map(|node| node.values.0.iter().filter(|value| value.is_some()).count())
                .sum();
            (node_count, share_count)
        }
        other => panic!("unexpected response type for point-open benchmark: {other:?}"),
    }
}

fn bench_point_open(
    trials: usize,
    setup: &BenchSetup,
    local_claims: usize,
    metrics: &BTreeSet<String>,
) -> BenchEntry {
    let effective_local_claims = if local_claims == 0 {
        setup.params.shard_registry.len()
    } else {
        local_claims.min(setup.params.shard_registry.len())
    };
    let committed_bulk_update_batches = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_POINT_OPEN_COMMITTED_BULK_UPDATE_BATCHES",
        DEFAULT_COMMITTED_BULK_UPDATE_BATCHES,
    );
    let (server, query_leaf_index) =
        build_warmed_server(setup, local_claims, committed_bulk_update_batches);
    let query_bulk_updates_default = if committed_bulk_update_batches >= 2 {
        1
    } else {
        0
    };
    let query_bulk_updates = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_POINT_OPEN_QUERY_BULK_UPDATES",
        query_bulk_updates_default,
    )
    .min(committed_bulk_update_batches - 1);
    let query_leaf_offset = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_POINT_OPEN_QUERY_LEAF_OFFSET",
        BULK_UPDATES_SIZE / 2,
    )
    .min(BULK_UPDATES_SIZE - 1);

    let core_resp = server.handle_read(query_leaf_index);
    let core_nodes = core_resp.nodes.len();
    let core_share_count: usize = core_resp.nodes.iter().map(|node| node.shares.len()).sum();

    let req = net::Message::AuthRead {
        req: NetAuthReadRequest {
            leaf: query_leaf_index,
            kappa: server.params.need as u64,
        },
    };
    let rpc_resp = net::auth::handle_read_only_message(&server, req.clone())
        .expect("point-open read-only response");
    let (response_nodes, response_nonempty_shares) = count_response_shares(&rpc_resp);
    assert_eq!(response_nodes, core_nodes);
    assert_eq!(response_nonempty_shares, core_share_count);
    let response_wire_bytes = net::wire::encode_binary(&rpc_resp)
        .expect("encode point-open response")
        .len();

    let mut core_samples = Vec::with_capacity(trials);
    if metrics.contains("core") {
        for _ in 0..trials {
            let start = Instant::now();
            let resp = black_box(server.handle_read(query_leaf_index));
            core_samples.push(start.elapsed().as_nanos() as u64);
            assert_eq!(resp.nodes.len(), core_nodes);
        }
    }

    let mut rpc_samples = Vec::with_capacity(trials);
    if metrics.contains("serialized") {
        for _ in 0..trials {
            let start = Instant::now();
            let resp = net::auth::handle_read_only_message(&server, req.clone())
                .expect("point-open read-only response");
            let encoded =
                black_box(net::wire::encode_binary(&resp).expect("serialize point-open response"));
            rpc_samples.push(start.elapsed().as_nanos() as u64);
            let (nodes, shares) = count_response_shares(&resp);
            assert_eq!(nodes, response_nodes);
            assert_eq!(shares, response_nonempty_shares);
            assert_eq!(encoded.len(), response_wire_bytes);
        }
    }

    let _ = (query_bulk_updates, query_leaf_offset);
    BenchEntry {
        seg: setup.params.segment_len,
        encoded_cells_n: setup.params.n_grid,
        encoded_segment_bytes: setup.params.n_grid * 32,
        kappa: setup.params.need,
        primary_slice_field_elements: setup.params.s,
        rows_per_claim: ROWS_PER_CLAIM,
        auxiliary_field_elements_per_claim: ROWS_PER_CLAIM * setup.params.row_len(),
        helper_count: setup.params.shard_registry.len(),
        local_claims: effective_local_claims,
        bulk_updates_bytes: BULK_UPDATES_BYTES,
        inserted_values: BULK_UPDATES_SIZE,
        auth_nodes_returned: response_nodes,
        array_shares_returned: response_nonempty_shares,
        serialized_response_bytes: response_wire_bytes,
        helper_open_share_core_ns: stats_from_samples(core_samples),
        helper_open_share_serialized_ns: stats_from_samples(rpc_samples),
    }
}

fn run_benchmark() {
    let trials = parse_usize_env("BENCH_APPEND_ONLY_LOG_POINT_OPEN_TRIALS", DEFAULT_TRIALS);
    let kappa = parse_usize_env("BENCH_APPEND_ONLY_LOG_POINT_OPEN_KAPPA", DEFAULT_KAPPA);
    let local_claims = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_POINT_OPEN_LOCAL_CLAIMS",
        DEFAULT_LOCAL_CLAIMS,
    );
    let committed_bulk_update_batches = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_POINT_OPEN_COMMITTED_BULK_UPDATE_BATCHES",
        DEFAULT_COMMITTED_BULK_UPDATE_BATCHES,
    );
    let query_bulk_updates_default = if committed_bulk_update_batches >= 2 {
        1
    } else {
        0
    };
    let query_bulk_updates = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_POINT_OPEN_QUERY_BULK_UPDATES",
        query_bulk_updates_default,
    )
    .min(committed_bulk_update_batches - 1);
    let query_leaf_offset = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_POINT_OPEN_QUERY_LEAF_OFFSET",
        BULK_UPDATES_SIZE / 2,
    )
    .min(BULK_UPDATES_SIZE - 1);
    let query_leaf_index =
        bulk_updates_start_leaf_index(query_bulk_updates as u64) + query_leaf_offset as u64;
    let metrics = parse_metric_filter(
        "BENCH_APPEND_ONLY_LOG_POINT_OPEN_METRICS",
        &["core", "serialized"],
    );

    let setup = build_setup(kappa);
    let point = bench_point_open(trials, &setup, local_claims, &metrics);

    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let results = BenchResults {
        meta: BenchMeta {
            seed: BENCH_SEED,
            trials,
            committed_bulk_update_batches,
            query_bulk_update_batch: query_bulk_updates,
            query_value_offset: query_leaf_offset,
            query_value_index: query_leaf_index,
            timestamp_unix_ms: timestamp_ms,
            git: git_sha(),
            notes: "local append-only-log per-server point-open benchmark: warm one claim-bound server through real bulk updates appends, then time one authenticated leaf-open request on that single server via the core handler and/or the read-only RPC adapter including response serialization".to_string(),
            metrics: metrics.into_iter().collect(),
        },
        entries: vec![point],
    };

    let out_dir = Path::new("results");
    let _ = fs::create_dir_all(out_dir);
    let path = out_dir.join("bench_append_only_log_point_open_local.json");
    let data = serde_json::to_vec_pretty(&results)
        .expect("serialize bench_append_only_log_point_open_local");
    fs::write(path, data).expect("write bench_append_only_log_point_open_local.json");
}

fn main() {
    run_benchmark();
}

#[test]
#[ignore]
fn bench_append_only_log_point_open_local() {
    run_benchmark();
}
