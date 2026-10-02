//! Local authenticated-array point-opening benchmark.
//!
//! This benchmark uses the auth layer directly, without AppendOnlyLog. It
//! materializes a fresh active prefix through one aligned fresh-subtree
//! transition, then measures one helper's `OpenShare` response and the client's
//! threshold `Combine+Verify` over pre-collected helper responses.

use paper_a_optimized::auth::client::{AuthClient, AuthTransport, ClientError};
use paper_a_optimized::auth::mapping::NodeLinearization;
use paper_a_optimized::auth::merkle::{NodeKey, default_hash, parent};
use paper_a_optimized::auth::service::{
    AuthBatchWriteRequest, AuthBatchWriteResponse, AuthFreshSubtreeWriteRequest, AuthReadResponse,
    AuthRequest, AuthResponse, AuthServer, AuthService, AuthSubtreeReadResponse,
    AuthSubtreeUpperPathReadResponse, AuthWriteRequest, AuthWriteResponse,
};
use paper_a_optimized::unauth::{ClaimBundle, DeltaMode};
use paper_a_optimized::{
    AuthConfig, AuthParams, AuthSubtreeProof, CodedArrayConfig, Config, ShardShare, Value256,
    build_coded_params, build_shard_registry_deterministic, random_value,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde::Serialize;
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const BENCH_SEED: u64 = 42;
const DEFAULT_TRIALS: usize = 20;
const TREE_HEIGHT: usize = 50;
const SEGMENT_LEN: usize = 128;
const S: usize = 16;
const ACTIVE_FIELD_ELEMENTS: usize = 262_144;
const QUERY_INDEX: u64 = (ACTIVE_FIELD_ELEMENTS / 2) as u64;

#[derive(Clone, Default)]
struct OfflineTransport;

impl AuthTransport for OfflineTransport {
    type Server = ();

    fn supports_subtree_upper_path_reads(&self) -> bool {
        false
    }

    fn supports_fresh_subtree_writes(&self) -> bool {
        false
    }

    fn server_label(&self, _server: &Self::Server) -> String {
        "offline".to_string()
    }

    fn auth_get_len(&self, _server: &Self::Server) -> Result<u64, ClientError> {
        Err(ClientError::Net("offline benchmark transport".to_string()))
    }

    fn auth_read(
        &self,
        _server: &Self::Server,
        _index: u64,
    ) -> Result<AuthReadResponse, ClientError> {
        Err(ClientError::Net("offline benchmark transport".to_string()))
    }

    fn auth_read_subtree(
        &self,
        _server: &Self::Server,
        _start_index: u64,
        _leaf_count: usize,
    ) -> Result<AuthSubtreeReadResponse, ClientError> {
        Err(ClientError::Net("offline benchmark transport".to_string()))
    }

    fn auth_read_subtree_upper_path(
        &self,
        _server: &Self::Server,
        _start_index: u64,
        _leaf_count: usize,
    ) -> Result<AuthSubtreeUpperPathReadResponse, ClientError> {
        Err(ClientError::Net("offline benchmark transport".to_string()))
    }

    fn unauth_multi_read(
        &self,
        _server: &Self::Server,
        _indices: &[u64],
    ) -> Result<Vec<Vec<ShardShare>>, ClientError> {
        Err(ClientError::Net("offline benchmark transport".to_string()))
    }

    fn auth_write(
        &self,
        _server: &Self::Server,
        _req: &AuthWriteRequest,
    ) -> Result<AuthWriteResponse, ClientError> {
        Err(ClientError::Net("offline benchmark transport".to_string()))
    }

    fn auth_write_batch(
        &self,
        _server: &Self::Server,
        _req: &AuthBatchWriteRequest,
    ) -> Result<AuthBatchWriteResponse, ClientError> {
        Err(ClientError::Net("offline benchmark transport".to_string()))
    }

    fn auth_write_fresh_subtree(
        &self,
        _server: &Self::Server,
        _req: &AuthFreshSubtreeWriteRequest,
    ) -> Result<AuthBatchWriteResponse, ClientError> {
        Err(ClientError::Net("offline benchmark transport".to_string()))
    }

    fn auth_extend(&self, _server: &Self::Server, _new_len: u64) -> Result<bool, ClientError> {
        Err(ClientError::Net("offline benchmark transport".to_string()))
    }
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
struct BenchMeta {
    seed: u64,
    trials: usize,
    timestamp_unix_ms: u64,
    git: Option<String>,
    notes: String,
}

#[derive(Clone, Debug, Serialize)]
struct BenchEntry {
    tree_height: usize,
    segment_len: usize,
    n_grid: usize,
    s: usize,
    need: usize,
    active_field_elements: usize,
    query_index: u64,
    helper_count: usize,
    response_nodes_per_helper: usize,
    response_nonempty_shares_per_helper: usize,
    helper_open_share_ns: MetricStats,
    client_combine_verify_ns: MetricStats,
}

#[derive(Clone, Debug, Serialize)]
struct BenchResults {
    meta: BenchMeta,
    entries: Vec<BenchEntry>,
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

fn parse_trials_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn build_config() -> Config {
    let n_grid = SEGMENT_LEN;
    let need = (n_grid - 1) / S + 1;
    Config {
        auth: AuthConfig {
            tree_height: TREE_HEIGHT,
            alloc_leaf_batch: (SEGMENT_LEN + 1) / 2,
            node_linearization: NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            segment_len: SEGMENT_LEN,
            s: S,
            shard_count: need,
            shard_registry: build_shard_registry_deterministic(need, S, n_grid),
            party_registry_indices: (0..need).map(|idx| vec![idx]).collect(),
            seed: BENCH_SEED,
        },
    }
}

fn claim_for_registry_index(
    params: &paper_a_optimized::CodedArrayParams,
    registry_index: usize,
) -> ClaimBundle {
    ClaimBundle::exact_for_registry_index(params, registry_index)
        .expect("exact claim should exist for registry index")
}

fn build_claim_server(cfg: &Config, registry_index: usize) -> AuthServer {
    let params = build_coded_params(&cfg.coded);
    let claim = claim_for_registry_index(&params, registry_index);
    AuthServer::new_with_assignments_and_claims(
        params,
        AuthParams::new(cfg.auth.tree_height),
        cfg.auth.node_linearization,
        0,
        None,
        Some(vec![claim]),
    )
    .expect("build claim server")
}

fn build_values(count: usize) -> Vec<Value256> {
    let mut rng = ChaCha20Rng::seed_from_u64(BENCH_SEED ^ 0xA717_0001);
    (0..count).map(|_| random_value(&mut rng)).collect()
}

fn build_default_subtree_proof(auth: &AuthParams, leaf_count: usize) -> AuthSubtreeProof {
    assert!(leaf_count.is_power_of_two());
    let subtree_height = leaf_count.trailing_zeros() as usize;
    let root_level = auth.tree_height - subtree_height;
    let mut upper_siblings = Vec::with_capacity(root_level);
    let mut key = NodeKey {
        level: root_level as u32,
        pos: 0,
    };
    for _ in 0..root_level {
        upper_siblings.push(default_hash(&auth.defaults, key.level as usize));
        key = parent(key);
    }
    AuthSubtreeProof {
        start_index: 0,
        leaf_count,
        upper_siblings,
        root: default_hash(&auth.defaults, 0),
    }
}

fn extend_server(server: &mut AuthServer, new_len: usize) {
    match server.handle_request(AuthRequest::Extend {
        new_leaves: new_len as u64,
    }) {
        AuthResponse::Extend { ok } => assert!(ok),
        AuthResponse::Error { message } => panic!("extend failed: {message}"),
        other => panic!("unexpected extend response: {other:?}"),
    }
}

fn write_transition(server: &mut AuthServer, req: &AuthFreshSubtreeWriteRequest) {
    match server.handle_request(AuthRequest::WriteFreshSubtree { req: req.clone() }) {
        AuthResponse::WriteFreshSubtree { resp } => {
            assert!(resp.ok, "write failed: {:?}", resp.error)
        }
        AuthResponse::Error { message } => panic!("write failed: {message}"),
        other => panic!("unexpected write response: {other:?}"),
    }
}

fn warm_servers(cfg: &Config) -> Vec<AuthServer> {
    let params = build_coded_params(&cfg.coded);
    let auth = AuthParams::new(cfg.auth.tree_height);
    let values = build_values(ACTIVE_FIELD_ELEMENTS);
    let req = AuthFreshSubtreeWriteRequest {
        leaves: values,
        proof: build_default_subtree_proof(&auth, ACTIVE_FIELD_ELEMENTS),
        delta_mode: DeltaMode::Aggregate,
    };
    (0..params.need)
        .map(|idx| {
            let mut server = build_claim_server(cfg, idx);
            extend_server(&mut server, ACTIVE_FIELD_ELEMENTS);
            write_transition(&mut server, &req);
            server
        })
        .collect()
}

fn count_nonempty(resp: &AuthReadResponse) -> usize {
    resp.nodes
        .iter()
        .map(|node| {
            node.shares
                .iter()
                .filter(|share| share.value().is_some())
                .count()
        })
        .sum()
}

fn bench_point_open(trials: usize) -> BenchEntry {
    let cfg = build_config();
    let params = build_coded_params(&cfg.coded);
    let servers = warm_servers(&cfg);
    let root = servers[0].store.root;

    let response_template = servers[0].handle_read(QUERY_INDEX);
    let response_nodes = response_template.nodes.len();
    let response_nonempty = count_nonempty(&response_template);
    let responses = servers
        .iter()
        .map(|server| server.handle_read(QUERY_INDEX))
        .collect::<Vec<_>>();

    let mut helper_samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        let start = Instant::now();
        let resp = black_box(servers[0].handle_read(QUERY_INDEX));
        helper_samples.push(start.elapsed().as_nanos() as u64);
        assert_eq!(resp.nodes.len(), response_nodes);
    }

    let mut client_samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        let mut client = AuthClient::new(
            OfflineTransport,
            params.clone(),
            AuthParams::new(cfg.auth.tree_height),
            cfg.auth.node_linearization,
            ACTIVE_FIELD_ELEMENTS,
            (),
            vec![()],
        );
        client.set_trusted_root(root);
        let start = Instant::now();
        let opened = black_box(
            client
                .read_with_path_from_responses(QUERY_INDEX, &responses)
                .expect("combine point opening"),
        );
        client_samples.push(start.elapsed().as_nanos() as u64);
        assert_eq!(opened.1.root, root);
    }

    BenchEntry {
        tree_height: TREE_HEIGHT,
        segment_len: SEGMENT_LEN,
        n_grid: params.n_grid,
        s: S,
        need: params.need,
        active_field_elements: ACTIVE_FIELD_ELEMENTS,
        query_index: QUERY_INDEX,
        helper_count: params.need,
        response_nodes_per_helper: response_nodes,
        response_nonempty_shares_per_helper: response_nonempty,
        helper_open_share_ns: stats_from_samples(helper_samples),
        client_combine_verify_ns: stats_from_samples(client_samples),
    }
}

fn main() {
    let trials = parse_trials_env("BENCH_AUTH_POINT_OPEN_TRIALS", DEFAULT_TRIALS);
    let entry = bench_point_open(trials);
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let results = BenchResults {
        meta: BenchMeta {
            seed: BENCH_SEED,
            trials,
            timestamp_unix_ms: timestamp_ms,
            git: git_sha(),
            notes: "auth-layer-only EDVC point-opening benchmark: warm k claim-bound helpers through one fresh-subtree transition, then time one helper OpenShare response and threshold client Combine+Verify over pre-collected helper responses".to_string(),
        },
        entries: vec![entry],
    };
    let out_dir = Path::new("results");
    let _ = fs::create_dir_all(out_dir);
    let path = out_dir.join("bench_auth_point_open_local.json");
    let data = serde_json::to_vec_pretty(&results).expect("serialize bench_auth_point_open_local");
    fs::write(path, data).expect("write bench_auth_point_open_local.json");
}
