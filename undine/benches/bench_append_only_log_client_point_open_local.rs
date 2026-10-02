//! Simulated client-side point-open benchmark for the append-only-log app.
//!
//! The setup warms local auth state from canonical field-entry bulk-update batches through
//! the real append-only-log frontier append path, captures one full read reply, splits
//! it into simulated helper replies, and then times only client processing of
//! those already-available replies: share aggregation,
//! Reed-Solomon decoding, Merkle path reconstruction, and trusted-root checks.
//! No TCP and no timed server handler work are included.

use append_only_log_demo::codec::bulk_updates_bytes_as_leaves;
use append_only_log_demo::frontier::PreparedBulkUpdatesAppend;
use append_only_log_demo::layout::{
    BULK_UPDATES_BYTES, BULK_UPDATES_SIZE, bulk_updates_start_leaf_index,
};
use paper_a_optimized::auth::client::{AuthClient, AuthTransport, ClientError as AuthClientError};
use paper_a_optimized::auth::mapping::NodeLinearization;
use paper_a_optimized::auth::service::{
    AuthBatchWriteRequest, AuthBatchWriteResponse, AuthFreshSubtreeWriteRequest, AuthReadNodeReply,
    AuthReadResponse, AuthRequest, AuthResponse, AuthServer, AuthService, AuthSubtreeReadResponse,
    AuthSubtreeUpperPathReadResponse, AuthWriteRequest, AuthWriteResponse,
};
use paper_a_optimized::unauth::error_correcting_share_target;
use paper_a_optimized::{
    AuthConfig, AuthParams, CodedArrayConfig, Config, ShardShare, Value256, build_coded_params,
    build_shard_registry_deterministic, fp_one,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde::Serialize;
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
                "invalid auth subtree response: {:?}",
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
    zero_bad_helper_replies: usize,
    correcting_helper_replies: usize,
    bad_helper_replies: usize,
    auth_nodes_per_helper: usize,
    array_shares_per_helper: usize,
    zero_bad_array_shares: usize,
    correcting_array_shares: usize,
    client_zero_bad_helpers_combine_verify_ns: MetricStats,
    client_kappa_bad_helpers_combine_verify_ns: MetricStats,
}

struct WarmedCluster {
    response: AuthReadResponse,
    allocated_leaves: usize,
    trusted_root: Value256,
    query_leaf_index: u64,
    expected_value: Value256,
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

fn bench_metric<T>(trials: usize, mut f: impl FnMut() -> T) -> MetricStats {
    let mut samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        let start = Instant::now();
        black_box(f());
        samples.push(start.elapsed().as_nanos() as u64);
    }
    stats_from_samples(samples)
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

fn build_config(kappa: usize) -> Config {
    let tree_height = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_CLIENT_POINT_OPEN_TREE_HEIGHT",
        DEFAULT_TREE_HEIGHT,
    );
    assert!(kappa.is_power_of_two(), "kappa must be a power of two");
    assert!(kappa >= 2, "kappa must be at least two");
    // The segment size is derived from the threshold.
    let seg = kappa * kappa;
    let registry_helper_count = error_correcting_share_target(kappa, kappa);
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

fn make_local_server(label: String, server: AuthServer) -> LocalServer {
    LocalServer {
        label,
        inner: Arc::new(Mutex::new(server)),
    }
}

fn make_warming_client(servers: &[LocalServer], setup: &BenchSetup) -> AuthClient<LocalTransport> {
    AuthClient::new(
        LocalTransport,
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

fn build_warmed_cluster(setup: &BenchSetup, committed_bulk_update_batches: usize) -> WarmedCluster {
    assert!(
        committed_bulk_update_batches > 0,
        "committed_bulk_update_batches must be positive"
    );
    let servers = vec![make_local_server(
        "full-local".to_string(),
        AuthServer::new(
            setup.params.clone(),
            setup.auth.clone(),
            setup.cfg.auth.node_linearization,
            0,
        )
        .expect("build full auth server"),
    )];
    let mut client = make_warming_client(&servers, setup);
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
        "BENCH_APPEND_ONLY_LOG_CLIENT_POINT_OPEN_QUERY_BULK_UPDATES",
        query_bulk_updates_default,
    )
    .min(committed_bulk_update_batches - 1);
    let query_value_offset = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_CLIENT_POINT_OPEN_QUERY_VALUE_OFFSET",
        BULK_UPDATES_SIZE / 2,
    )
    .min(BULK_UPDATES_SIZE - 1);
    let query_leaf_index =
        bulk_updates_start_leaf_index(query_bulk_updates as u64) + query_value_offset as u64;
    let query_bulk_updates_bytes =
        build_exact_bulk_updates(BENCH_SEED ^ 0xA110000 ^ query_bulk_updates as u64);
    let query_leaves =
        bulk_updates_bytes_as_leaves(&query_bulk_updates_bytes).expect("query bulk updates leaves");
    let expected_value = query_leaves[query_value_offset];
    let allocated_leaves = client
        .current_len()
        .expect("allocated length after appends") as usize;
    let trusted_root = client.trusted_root().expect("trusted root after appends");
    let (check_value, check_path) = client
        .read_with_path(query_leaf_index)
        .expect("local client point open after warmup");
    assert_eq!(check_value, expected_value);
    assert_eq!(check_path.root, trusted_root);
    let response = {
        let guard = servers[0].inner.lock().expect("auth server lock");
        guard.handle_read(query_leaf_index)
    };

    WarmedCluster {
        response,
        allocated_leaves,
        trusted_root,
        query_leaf_index,
        expected_value,
    }
}

fn capture_responses(cluster: &WarmedCluster, helper_count: usize) -> Vec<AuthReadResponse> {
    (0..helper_count)
        .map(|registry_index| AuthReadResponse {
            nodes: cluster
                .response
                .nodes
                .iter()
                .map(|node| AuthReadNodeReply {
                    node: node.node,
                    shares: node
                        .shares
                        .iter()
                        .filter(|share| share.registry_index() == registry_index)
                        .cloned()
                        .collect(),
                })
                .collect(),
        })
        .collect()
}

fn corrupt_responses(
    responses: &[AuthReadResponse],
    bad_helper_count: usize,
) -> Vec<AuthReadResponse> {
    let mut corrupted = responses.to_vec();
    for (bad_pos, resp) in corrupted.iter_mut().take(bad_helper_count).enumerate() {
        if resp.nodes.is_empty() {
            continue;
        }
        let node_pos_to_corrupt = bad_pos % resp.nodes.len();
        if let Some(node) = resp.nodes.get_mut(node_pos_to_corrupt) {
            for share in &mut node.shares {
                let corrupted_value = share.value().map(|value| value + fp_one());
                *share = ShardShare::new(share.registry_index(), corrupted_value);
            }
        }
    }
    corrupted
}

fn count_nonempty_shares(responses: &[AuthReadResponse]) -> usize {
    responses
        .iter()
        .flat_map(|resp| &resp.nodes)
        .map(|node| {
            node.shares
                .iter()
                .filter(|share| share.value().is_some())
                .count()
        })
        .sum()
}

fn response_nodes_per_helper(responses: &[AuthReadResponse]) -> usize {
    responses.first().map(|resp| resp.nodes.len()).unwrap_or(0)
}

fn response_shares_per_helper(responses: &[AuthReadResponse]) -> usize {
    responses
        .first()
        .map(|resp| {
            resp.nodes
                .iter()
                .map(|node| {
                    node.shares
                        .iter()
                        .filter(|share| share.value().is_some())
                        .count()
                })
                .sum()
        })
        .unwrap_or(0)
}

fn assert_processed_read(
    result: (Value256, paper_a_optimized::AuthPath),
    expected_value: Value256,
    trusted_root: Value256,
) {
    let (value, path) = result;
    assert_eq!(value, expected_value);
    assert_eq!(path.root, trusted_root);
}

fn bench_client_point_open(trials: usize, setup: &BenchSetup) -> BenchEntry {
    let committed_bulk_update_batches = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_CLIENT_POINT_OPEN_COMMITTED_BULK_UPDATE_BATCHES",
        DEFAULT_COMMITTED_BULK_UPDATE_BATCHES,
    );
    let cluster = build_warmed_cluster(setup, committed_bulk_update_batches);
    let zero_bad_helper_replies = setup.params.need;
    let correcting_helper_replies =
        error_correcting_share_target(setup.params.need, setup.params.need);
    let bad_helper_replies = setup.params.need;

    let zero_bad_responses = capture_responses(&cluster, zero_bad_helper_replies);
    let correcting_responses = capture_responses(&cluster, correcting_helper_replies);
    let correcting_bad_responses = corrupt_responses(&correcting_responses, bad_helper_replies);

    let expected_value = cluster.expected_value;
    let trusted_root = cluster.trusted_root;

    let mut zero_bad_client = make_processing_client(setup, cluster.allocated_leaves, trusted_root);
    let zero_bad_check = zero_bad_client
        .read_with_path_from_responses(cluster.query_leaf_index, &zero_bad_responses)
        .expect("zero-bad-helper pre-collected point open");
    assert_processed_read(zero_bad_check, expected_value, trusted_root);

    let mut correcting_bad_client =
        make_processing_client(setup, cluster.allocated_leaves, trusted_root);
    let correcting_bad_check = correcting_bad_client
        .read_with_path_from_responses_correcting(
            cluster.query_leaf_index,
            &correcting_bad_responses,
            setup.params.need,
        )
        .expect("bad-helper correcting pre-collected point open");
    assert_processed_read(correcting_bad_check, expected_value, trusted_root);

    let client_zero_bad_helpers_combine_verify_ns = bench_metric(trials, || {
        let result = zero_bad_client
            .read_with_path_from_responses(cluster.query_leaf_index, &zero_bad_responses)
            .expect("zero-bad-helper pre-collected point open");
        assert_processed_read(result, expected_value, trusted_root);
    });
    let client_kappa_bad_helpers_combine_verify_ns = bench_metric(trials, || {
        let result = correcting_bad_client
            .read_with_path_from_responses_correcting(
                cluster.query_leaf_index,
                &correcting_bad_responses,
                setup.params.need,
            )
            .expect("bad-helper correcting pre-collected point open");
        assert_processed_read(result, expected_value, trusted_root);
    });

    BenchEntry {
        seg: setup.params.segment_len,
        encoded_cells_n: setup.params.n_grid,
        encoded_segment_bytes: setup.params.n_grid * 32,
        kappa: setup.params.need,
        primary_slice_field_elements: setup.params.s,
        registry_helper_count: setup.params.shard_registry.len(),
        bulk_updates_bytes: BULK_UPDATES_BYTES,
        inserted_values: BULK_UPDATES_SIZE,
        zero_bad_helper_replies,
        correcting_helper_replies,
        bad_helper_replies,
        auth_nodes_per_helper: response_nodes_per_helper(&correcting_responses),
        array_shares_per_helper: response_shares_per_helper(&correcting_responses),
        zero_bad_array_shares: count_nonempty_shares(&zero_bad_responses),
        correcting_array_shares: count_nonempty_shares(&correcting_responses),
        client_zero_bad_helpers_combine_verify_ns,
        client_kappa_bad_helpers_combine_verify_ns,
    }
}

fn run_benchmark() {
    let trials = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_CLIENT_POINT_OPEN_TRIALS",
        DEFAULT_TRIALS,
    );
    let kappa = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_CLIENT_POINT_OPEN_KAPPA",
        DEFAULT_KAPPA,
    );
    let committed_bulk_update_batches = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_CLIENT_POINT_OPEN_COMMITTED_BULK_UPDATE_BATCHES",
        DEFAULT_COMMITTED_BULK_UPDATE_BATCHES,
    );
    let query_bulk_updates_default = if committed_bulk_update_batches >= 2 {
        1
    } else {
        0
    };
    let query_bulk_updates = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_CLIENT_POINT_OPEN_QUERY_BULK_UPDATES",
        query_bulk_updates_default,
    )
    .min(committed_bulk_update_batches - 1);
    let query_value_offset = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_CLIENT_POINT_OPEN_QUERY_VALUE_OFFSET",
        BULK_UPDATES_SIZE / 2,
    )
    .min(BULK_UPDATES_SIZE - 1);
    let query_leaf_index =
        bulk_updates_start_leaf_index(query_bulk_updates as u64) + query_value_offset as u64;

    let setup = build_setup(kappa);
    let point = bench_client_point_open(trials, &setup);

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
            query_value_offset,
            query_value_index: query_leaf_index,
            timestamp_unix_ms: timestamp_ms,
            git: git_sha(),
            notes: "simulated append-only-log client point-open benchmark: warm local auth state from canonical field-entry bulk-update batches through the app codec and frontier path, split a full read reply into helper replies, then time client-side Combine+Verify with kappa replies and 0 bad helpers, and with 3*kappa replies containing kappa staggered bad helpers".to_string(),
        },
        entries: vec![point],
    };

    let out_dir = Path::new("results");
    let _ = fs::create_dir_all(out_dir);
    let path = out_dir.join("bench_append_only_log_client_point_open_local.json");
    let data = serde_json::to_vec_pretty(&results)
        .expect("serialize bench_append_only_log_client_point_open_local");
    fs::write(path, data).expect("write bench_append_only_log_client_point_open_local.json");
}

fn main() {
    run_benchmark();
}

#[test]
#[ignore]
fn bench_append_only_log_client_point_open_local() {
    run_benchmark();
}
