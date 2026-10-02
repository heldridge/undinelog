use append_only_log_demo::codec::bulk_updates_bytes_as_leaves;
use append_only_log_demo::frontier::PreparedBulkUpdatesAppend;
use append_only_log_demo::layout::{BULK_UPDATES_BYTES, BULK_UPDATES_SIZE};
use paper_a_optimized::auth::client::{AuthClient, AuthTransport, ClientError as AuthClientError};
use paper_a_optimized::auth::mapping::NodeLinearization;
use paper_a_optimized::auth::service::{
    AuthBatchWriteRequest, AuthBatchWriteResponse, AuthFreshSubtreeWriteRequest, AuthReadResponse,
    AuthRequest, AuthResponse, AuthServer, AuthService, AuthSubtreeReadResponse,
    AuthSubtreeUpperPathReadResponse, AuthWriteRequest, AuthWriteResponse,
};
use paper_a_optimized::coded_array::params::ROWS_PER_CLAIM;
use paper_a_optimized::net;
use paper_a_optimized::net::wire::{NetFpBlob, NetRowMatrixReply};
use paper_a_optimized::unauth::client::{
    ClaimRecoveryPlan, ClientError as UnauthClientError, UnauthClient as CoreUnauthClient,
    UnauthTransport,
};
use paper_a_optimized::unauth::service::{
    RowReadRequest, UnauthRequest, UnauthResponse, UnauthService,
};
use paper_a_optimized::unauth::{
    ClaimBundle, ClaimSegmentInstallPlan, RecoveredClaimSegmentState, UnauthStore,
    error_correcting_share_target,
};
use paper_a_optimized::{
    AuthConfig, AuthParams, CodedArrayConfig, CodedArrayParams, Config, Fp, RowShare, Value256,
    build_coded_params, build_shard_registry_deterministic, encode_value,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

pub const BENCH_SEED: u64 = 42;
pub const DEFAULT_TREE_HEIGHT: usize = 50;
pub const DEFAULT_TRIALS: usize = 5;
pub const DEFAULT_KAPPA: usize = 64;
pub const DEFAULT_COMMITTED_BULK_UPDATE_BATCHES: usize = 3;
pub const DEFAULT_HELPER_SURPLUS: usize = 1;
pub const FIELD_BYTES: u64 = 32;

#[derive(Clone)]
struct LocalAuthServer {
    party_id: u64,
    inner: Arc<Mutex<AuthServer>>,
}

#[derive(Clone)]
struct LocalAuthTransport;

impl AuthTransport for LocalAuthTransport {
    type Server = LocalAuthServer;

    fn supports_subtree_upper_path_reads(&self) -> bool {
        true
    }

    fn supports_fresh_subtree_writes(&self) -> bool {
        true
    }

    fn server_label(&self, server: &Self::Server) -> String {
        server.party_id.to_string()
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
struct ReplayServer {
    party_id: u64,
}

#[derive(Clone)]
struct ReplayTransport {
    logical_len: u64,
    target_z: Fp,
    target_row_labels: Arc<Vec<Fp>>,
    row_matrix_replies: Arc<HashMap<(u64, usize), Vec<Vec<Vec<paper_a_optimized::RowShare>>>>>,
    shard_eval_replies: Arc<HashMap<(u64, usize), HashMap<Value256, Vec<Fp>>>>,
    corrupt_party_ids: Arc<HashSet<u64>>,
}

impl ReplayTransport {
    fn unsupported(op: &str) -> UnauthClientError {
        UnauthClientError::InvalidResponse(format!("unsupported replay transport operation: {op}"))
    }

    fn corrupt_row_reply(reply: &mut [Vec<RowShare>], party_id: u64) {
        for batch in reply {
            for row in batch {
                *row = RowShare::new(
                    row.row_label(),
                    row.value() + Fp::from(40_000u64 + party_id),
                );
            }
        }
    }

    fn corrupt_eval_reply(reply: &mut [Vec<Fp>], party_id: u64) {
        for batch in reply {
            for value in batch {
                *value += Fp::from(50_000u64 + party_id);
            }
        }
    }
}

impl UnauthTransport for ReplayTransport {
    type Server = ReplayServer;

    fn server_label(&self, server: &Self::Server) -> String {
        server.party_id.to_string()
    }

    fn unauth_get_len(&self, _server: &Self::Server) -> Result<u64, UnauthClientError> {
        Ok(self.logical_len)
    }

    fn unauth_read(
        &self,
        _server: &Self::Server,
        _idx: u64,
    ) -> Result<Vec<paper_a_optimized::ShardShare>, UnauthClientError> {
        Err(Self::unsupported("unauth_read"))
    }

    fn unauth_read_block(
        &self,
        _server: &Self::Server,
        _block: u64,
    ) -> Result<Vec<paper_a_optimized::RowShare>, UnauthClientError> {
        Err(Self::unsupported("unauth_read_block"))
    }

    fn unauth_read_rows_at_z(
        &self,
        _server: &Self::Server,
        _seg: usize,
        _z_eval: Fp,
    ) -> Result<Vec<paper_a_optimized::RowShare>, UnauthClientError> {
        Err(Self::unsupported("unauth_read_rows_at_z"))
    }

    fn unauth_read_rows_batch_at_z(
        &self,
        _server: &Self::Server,
        _seg: usize,
        _z_evals: &[Fp],
    ) -> Result<Vec<Vec<paper_a_optimized::RowShare>>, UnauthClientError> {
        Err(Self::unsupported("unauth_read_rows_batch_at_z"))
    }

    fn unauth_read_rows_matrix(
        &self,
        server: &Self::Server,
        requests: &[(usize, Vec<Fp>)],
    ) -> Result<Vec<Vec<Vec<paper_a_optimized::RowShare>>>, UnauthClientError> {
        let mut out = Vec::with_capacity(requests.len());
        for (seg, z_evals) in requests {
            if z_evals.len() != 1 || z_evals[0] != self.target_z {
                return Err(Self::unsupported("unexpected row-matrix z set"));
            }
            let rows = self
                .row_matrix_replies
                .get(&(server.party_id, *seg))
                .cloned()
                .ok_or_else(|| {
                    UnauthClientError::Net("missing replay row-matrix reply".to_string())
                })?;
            if rows.len() != 1 {
                return Err(Self::unsupported(
                    "unexpected replay row-matrix request shape",
                ));
            }
            let mut reply = rows
                .into_iter()
                .next()
                .expect("one replay row-matrix request");
            if self.corrupt_party_ids.contains(&server.party_id) {
                Self::corrupt_row_reply(&mut reply, server.party_id);
            }
            out.push(reply);
        }
        Ok(out)
    }

    fn unauth_multi_read(
        &self,
        _server: &Self::Server,
        _idxs: &[u64],
    ) -> Result<Vec<Vec<paper_a_optimized::ShardShare>>, UnauthClientError> {
        Err(Self::unsupported("unauth_multi_read"))
    }

    fn unauth_repair_shard(
        &self,
        _server: &Self::Server,
        _seg: usize,
        _registry_index: usize,
    ) -> Result<Vec<Fp>, UnauthClientError> {
        Err(Self::unsupported("unauth_repair_shard"))
    }

    fn unauth_repair_shard_batch(
        &self,
        _server: &Self::Server,
        _seg: usize,
        _registry_indices: &[usize],
    ) -> Result<Vec<Vec<Fp>>, UnauthClientError> {
        Err(Self::unsupported("unauth_repair_shard_batch"))
    }

    fn unauth_eval_shards_at_row_labels(
        &self,
        server: &Self::Server,
        seg: usize,
        col_labels: &[Fp],
        row_labels: &[Fp],
    ) -> Result<Vec<Vec<Fp>>, UnauthClientError> {
        if row_labels != self.target_row_labels.as_slice() {
            return Err(Self::unsupported("unexpected shard-eval row-label set"));
        }
        let Some(by_registry) = self.shard_eval_replies.get(&(server.party_id, seg)) else {
            return Err(UnauthClientError::Net(
                "missing replay shard-eval reply".to_string(),
            ));
        };
        col_labels
            .iter()
            .map(|col_label| {
                let mut reply = by_registry
                    .get(&encode_value(*col_label))
                    .cloned()
                    .ok_or_else(|| {
                        UnauthClientError::Net(format!(
                            "missing replay shard eval {:?} for helper {} segment {}",
                            col_label, server.party_id, seg
                        ))
                    })?;
                if self.corrupt_party_ids.contains(&server.party_id) {
                    Self::corrupt_eval_reply(std::slice::from_mut(&mut reply), server.party_id);
                }
                Ok(reply)
            })
            .collect()
    }

    fn unauth_apply_deltas(
        &self,
        _server: &Self::Server,
        _deltas: &[(u64, Fp)],
        _mode: paper_a_optimized::unauth::DeltaMode,
    ) -> Result<bool, UnauthClientError> {
        Err(Self::unsupported("unauth_apply_deltas"))
    }

    fn unauth_extend(
        &self,
        _server: &Self::Server,
        _new_len: u64,
    ) -> Result<bool, UnauthClientError> {
        Err(Self::unsupported("unauth_extend"))
    }
}

#[derive(Clone)]
pub struct BenchSetup {
    pub cfg: Config,
    pub params: CodedArrayParams,
    pub auth: AuthParams,
}

#[derive(Clone, Debug, Serialize)]
pub struct MetricStats {
    pub mean: u64,
    pub p50: u64,
    pub p95: u64,
    pub trials: u64,
    pub samples: Vec<u64>,
}

pub struct RepairKernelsFixture {
    setup: BenchSetup,
    helper_core_server: AuthServer,
    helper_rpc_server: AuthServer,
    recover_fixture: RecoverFixture,
    recover_correcting_clean_fixture: Option<RecoverFixture>,
    recover_correcting_corrupt_fixture: Option<RecoverFixture>,
    bulk_updates_leaves: Vec<Vec<Value256>>,
    expected_auth_root: Value256,
    helper_z: Fp,
    helper_col_label: Fp,
    target_row_labels: Vec<Fp>,
    pub kappa: usize,
    pub segment_len: usize,
    pub n_grid: usize,
    pub need: usize,
    pub shard_count: usize,
    pub repair_helper_count: usize,
    pub bulk_updates_bytes: usize,
    pub bulk_updates_size: usize,
    pub committed_bulk_update_batches: usize,
    pub allocated_leaves: usize,
    pub live_unauth_len: u64,
    pub segment_count: usize,
    pub target_registry_index: usize,
    pub target_row_count: usize,
    pub helper_registry_index: usize,
    pub helper_row_replies_per_segment: usize,
    pub helper_eval_scalars_per_segment: usize,
    pub helper_row_reply_wire_bytes: usize,
    pub helper_eval_reply_wire_bytes: usize,
    pub helper_assist_total_wire_bytes: usize,
    pub recover_helper_reply_bytes_modeled: u64,
    pub recover_correcting_helper_reply_bytes_modeled: Option<u64>,
    pub full_bulk_updates_download_bytes: u64,
    pub max_errors: usize,
    pub corrupt_helper_count: usize,
    pub correcting_repair_helper_count: usize,
}

struct RecoverFixture {
    client: CoreUnauthClient<ReplayTransport>,
    empty_target: UnauthStore,
    expected_target: UnauthStore,
    recovery_plan: ClaimRecoveryPlan,
    install_plan: ClaimSegmentInstallPlan,
    segs: Vec<usize>,
    max_errors: usize,
}

pub fn percentile(sorted: &[u64], pct: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (pct * (sorted.len() - 1)) / 100;
    sorted[idx]
}

pub fn stats_from_samples(mut samples: Vec<u64>) -> MetricStats {
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

pub fn git_sha() -> Option<String> {
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

pub fn parse_usize_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

pub fn timestamp_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub fn write_json<T: Serialize>(path: &Path, value: &T) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let data = serde_json::to_vec_pretty(value).expect("serialize benchmark json");
    fs::write(path, data).expect("write benchmark json");
}

fn build_config(kappa: usize, helper_surplus: usize) -> Config {
    let tree_height = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_REPAIR_TREE_HEIGHT",
        DEFAULT_TREE_HEIGHT,
    );
    assert!(kappa.is_power_of_two(), "kappa must be a power of two");
    assert!(kappa >= 2, "kappa must be at least two");
    // The segment size is derived from the threshold.
    let segment_len = kappa * kappa;
    let shard_count = kappa + helper_surplus;
    Config {
        auth: AuthConfig {
            tree_height,
            alloc_leaf_batch: segment_len / 2,
            node_linearization: NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            kappa,
            shard_count,
            shard_registry: build_shard_registry_deterministic(shard_count),
            party_registry_indices: (0..shard_count).map(|idx| vec![idx]).collect(),
            seed: BENCH_SEED,
        },
    }
}

pub fn build_setup(kappa: usize, helper_surplus: usize) -> BenchSetup {
    let cfg = build_config(kappa, helper_surplus);
    let params = build_coded_params(&cfg.coded);
    let auth = AuthParams::new(cfg.auth.tree_height);
    BenchSetup { cfg, params, auth }
}

pub fn claim_for_registry_index(params: &CodedArrayParams, registry_index: usize) -> ClaimBundle {
    ClaimBundle::exact_for_registry_index(params, registry_index)
        .expect("exact claim should exist for registry index")
}

fn build_claim_server(setup: &BenchSetup, registry_index: usize) -> AuthServer {
    let claims = vec![claim_for_registry_index(&setup.params, registry_index)];
    AuthServer::new_with_assignments_and_claims(
        setup.params.clone(),
        setup.auth.clone(),
        setup.cfg.auth.node_linearization,
        0,
        None,
        Some(claims),
    )
    .expect("build claim server")
}

fn make_local_server(party_id: u64, server: AuthServer) -> LocalAuthServer {
    LocalAuthServer {
        party_id,
        inner: Arc::new(Mutex::new(server)),
    }
}

fn make_cluster_client(
    servers: &[LocalAuthServer],
    setup: &BenchSetup,
) -> AuthClient<LocalAuthTransport> {
    AuthClient::new(
        LocalAuthTransport,
        setup.params.clone(),
        setup.auth.clone(),
        setup.cfg.auth.node_linearization,
        0,
        servers[0].clone(),
        servers.to_vec(),
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

pub fn build_warmed_cluster(
    setup: &BenchSetup,
    committed_bulk_update_batches: usize,
) -> Vec<AuthServer> {
    assert!(
        committed_bulk_update_batches > 0,
        "committed_bulk_update_batches must be positive"
    );
    let local_servers: Vec<_> = (0..setup.params.shard_registry.len())
        .map(|registry_index| {
            make_local_server(
                registry_index as u64,
                build_claim_server(setup, registry_index),
            )
        })
        .collect();
    let mut client = make_cluster_client(&local_servers, setup);
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
    local_servers
        .iter()
        .map(|server| server.inner.lock().expect("server lock").clone())
        .collect()
}

pub fn build_warmed_cluster_for_registry_indices(
    setup: &BenchSetup,
    committed_bulk_update_batches: usize,
    registry_indices: &[usize],
) -> BTreeMap<usize, AuthServer> {
    assert!(
        committed_bulk_update_batches > 0,
        "committed_bulk_update_batches must be positive"
    );
    let local_servers: Vec<_> = registry_indices
        .iter()
        .copied()
        .map(|registry_index| {
            assert!(
                registry_index < setup.params.shard_registry.len(),
                "registry index out of range"
            );
            make_local_server(
                registry_index as u64,
                build_claim_server(setup, registry_index),
            )
        })
        .collect();
    let mut client = make_cluster_client(&local_servers, setup);
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
    registry_indices
        .iter()
        .copied()
        .zip(
            local_servers
                .iter()
                .map(|server| server.inner.lock().expect("server lock").clone()),
        )
        .collect()
}

fn helper_row_matrix_request(segs: &[usize], z_eval: Fp) -> UnauthRequest {
    UnauthRequest::ReadRowsMatrix {
        requests: segs
            .iter()
            .copied()
            .map(|seg| RowReadRequest {
                seg,
                z_evals: vec![z_eval],
            })
            .collect(),
    }
}

fn helper_eval_matrix_request(segs: &[usize], col_label: Fp, row_labels: &[Fp]) -> UnauthRequest {
    UnauthRequest::EvalShardsAtRowLabelsMatrix {
        segs: segs.to_vec(),
        col_labels: vec![col_label],
        row_labels: row_labels.to_vec(),
    }
}

fn core_row_matrix_reply(
    server: &mut AuthServer,
    segs: &[usize],
    z_eval: Fp,
) -> Vec<Vec<Vec<paper_a_optimized::RowShare>>> {
    match server
        .store
        .unauth
        .handle_request(helper_row_matrix_request(segs, z_eval))
    {
        UnauthResponse::ReadRowsMatrix { rows } => rows,
        UnauthResponse::Error { message } => panic!("row helper matrix core error: {message}"),
        other => panic!("unexpected row helper matrix core response: {other:?}"),
    }
}

fn core_eval_matrix_reply(
    server: &mut AuthServer,
    segs: &[usize],
    col_label: Fp,
    row_labels: &[Fp],
) -> Vec<Vec<Vec<Fp>>> {
    match server
        .store
        .unauth
        .handle_request(helper_eval_matrix_request(segs, col_label, row_labels))
    {
        UnauthResponse::EvalShardsAtRowLabelsMatrix { values } => values,
        UnauthResponse::Error { message } => {
            panic!("shard-eval helper matrix core error: {message}")
        }
        other => panic!("unexpected shard-eval helper matrix core response: {other:?}"),
    }
}

fn rpc_row_matrix_reply(server: &mut AuthServer, segs: &[usize], z_eval: Fp) -> net::Response {
    match server
        .store
        .unauth
        .handle_request(helper_row_matrix_request(segs, z_eval))
    {
        UnauthResponse::ReadRowsMatrix { rows } => net::Response::UnauthReadRowsMatrix {
            rows: NetRowMatrixReply {
                values: NetFpBlob(
                    rows.into_iter()
                        .flat_map(|req_rows| {
                            req_rows
                                .into_iter()
                                .flat_map(|batch| batch.into_iter().map(|row| row.value()))
                        })
                        .collect(),
                ),
            },
        },
        UnauthResponse::Error { message } => panic!("row helper matrix rpc error: {message}"),
        other => panic!("unexpected row helper matrix rpc response: {other:?}"),
    }
}

fn rpc_eval_matrix_reply(
    server: &mut AuthServer,
    segs: &[usize],
    col_label: Fp,
    row_labels: &[Fp],
) -> net::Response {
    match server
        .store
        .unauth
        .handle_request(helper_eval_matrix_request(segs, col_label, row_labels))
    {
        UnauthResponse::EvalShardsAtRowLabelsMatrix { values } => {
            net::Response::UnauthEvalShardsAtRowLabelsMatrix {
                values: NetFpBlob(
                    values
                        .into_iter()
                        .flat_map(|per_seg| per_seg.into_iter().flat_map(|batch| batch.into_iter()))
                        .collect(),
                ),
            }
        }
        UnauthResponse::Error { message } => {
            panic!("shard-eval helper matrix rpc error: {message}")
        }
        other => panic!("unexpected shard-eval helper matrix rpc response: {other:?}"),
    }
}

fn build_recover_fixture(
    setup: &BenchSetup,
    warmed: &[AuthServer],
    target_registry_index: usize,
    max_errors: usize,
    corrupt_helper_count: usize,
) -> RecoverFixture {
    assert!(
        corrupt_helper_count <= max_errors,
        "cannot corrupt more helpers than max_errors"
    );
    let target_claim = claim_for_registry_index(&setup.params, target_registry_index);
    let target_z = setup.params.shard_registry[target_registry_index].z;
    let target_row_label_indices = target_claim
        .row_label_indices(&setup.params)
        .expect("target claim rows");
    let target_row_labels: Vec<_> = target_row_label_indices
        .iter()
        .map(|&idx| setup.params.row_labels[idx])
        .collect();
    let live_unauth_len = warmed[target_registry_index].store.unauth.len();
    let segment_count = live_unauth_len.div_ceil(setup.params.segment_len as u64) as usize;
    let helper_limit = if max_errors == 0 {
        setup.params.need
    } else {
        error_correcting_share_target(setup.params.need, max_errors)
    };
    let active_helper_ids: Vec<u64> = (0..warmed.len())
        .filter(|&registry_index| registry_index != target_registry_index)
        .take(helper_limit)
        .map(|registry_index| registry_index as u64)
        .collect();
    assert_eq!(
        active_helper_ids.len(),
        helper_limit,
        "not enough helpers for requested error-correcting recovery"
    );
    let corrupt_party_ids = active_helper_ids
        .iter()
        .copied()
        .take(corrupt_helper_count)
        .collect::<HashSet<_>>();
    let helper_servers: Vec<_> = active_helper_ids
        .iter()
        .copied()
        .map(|party_id| ReplayServer { party_id })
        .collect();
    let helper_owned_indices: Vec<_> = active_helper_ids
        .iter()
        .copied()
        .map(|party_id| vec![party_id as usize])
        .collect();

    let mut row_matrix_replies = HashMap::new();
    let mut shard_eval_replies = HashMap::new();
    for &party_id in &active_helper_ids {
        let registry_index = party_id as usize;
        let mut helper = warmed[registry_index].clone();
        for seg in 0..segment_count {
            let rows = core_row_matrix_reply(&mut helper, &[seg], target_z);
            row_matrix_replies.insert((party_id, seg), rows);

            let helper_col_label = setup.params.shard_registry[registry_index].z;
            let shard_values =
                core_eval_matrix_reply(&mut helper, &[seg], helper_col_label, &target_row_labels);
            let mut by_registry = HashMap::new();
            for values in shard_values {
                for batch in values {
                    by_registry.insert(encode_value(helper_col_label), batch);
                }
            }
            shard_eval_replies.insert((party_id, seg), by_registry);
        }
    }

    let transport = ReplayTransport {
        logical_len: live_unauth_len,
        target_z,
        target_row_labels: Arc::new(target_row_labels),
        row_matrix_replies: Arc::new(row_matrix_replies),
        shard_eval_replies: Arc::new(shard_eval_replies),
        corrupt_party_ids: Arc::new(corrupt_party_ids),
    };
    let client = CoreUnauthClient::new_with_owned_indices(
        transport,
        setup.params.clone(),
        helper_servers[0].clone(),
        helper_servers,
        helper_owned_indices,
        live_unauth_len,
    );
    let mut empty_target =
        UnauthStore::new_with_claims(setup.params.clone(), vec![target_claim.clone()]);
    empty_target
        .extend(live_unauth_len)
        .expect("extend empty target store");
    let install_plan = empty_target
        .claim_segment_install_plan()
        .expect("target install plan");
    let recovery_plan = if max_errors == 0 {
        client
            .prepare_claim_recovery(&target_claim)
            .expect("target claim recovery plan")
    } else {
        client
            .prepare_claim_recovery_correcting(&target_claim, max_errors)
            .expect("target correcting claim recovery plan")
    };
    let segs: Vec<_> = (0..segment_count).collect();

    RecoverFixture {
        client,
        empty_target,
        expected_target: warmed[target_registry_index].store.unauth.clone(),
        recovery_plan,
        install_plan,
        segs,
        max_errors,
    }
}

impl RecoverFixture {
    fn recover_segments_once(&self) -> Vec<(usize, RecoveredClaimSegmentState)> {
        if self.max_errors == 0 {
            self.client
                .repair_claim_segments_fp_with_plan(&self.segs, &self.recovery_plan)
                .expect("repair full claim segments")
        } else {
            self.client
                .repair_claim_segments_fp_with_plan_correcting(
                    &self.segs,
                    &self.recovery_plan,
                    self.max_errors,
                )
                .expect("repair full claim segments with correction")
        }
    }

    fn install_segments_once(
        &self,
        repaired_segments: Vec<(usize, RecoveredClaimSegmentState)>,
    ) -> UnauthStore {
        let mut recovered = self.empty_target.clone();
        for (seg, repaired) in repaired_segments {
            recovered
                .install_recovered_claim_segment_state_owned_with_plan(
                    seg,
                    &self.install_plan,
                    repaired,
                )
                .expect("install repaired claim segment");
        }
        recovered
    }

    fn recover_and_install_once(&self) -> UnauthStore {
        self.install_segments_once(self.recover_segments_once())
    }
}

impl RepairKernelsFixture {
    pub fn build(
        setup: &BenchSetup,
        committed_bulk_update_batches: usize,
        target_registry_index: usize,
        helper_registry_index: usize,
        max_errors: usize,
    ) -> Self {
        let warmed = build_warmed_cluster(setup, committed_bulk_update_batches);
        assert!(
            warmed.len() >= setup.params.need + 1,
            "repair benchmark requires at least one helper beyond need"
        );

        let bulk_updates_leaves: Vec<Vec<Value256>> = (0..committed_bulk_update_batches)
            .map(|bulk_updates_idx| {
                let bulk_updates =
                    build_exact_bulk_updates(BENCH_SEED ^ 0xA110000 ^ bulk_updates_idx as u64);
                bulk_updates_bytes_as_leaves(&bulk_updates)
                    .expect("bulk updates leaves")
                    .iter()
                    .copied()
                    .collect()
            })
            .collect();

        let target_claim = claim_for_registry_index(&setup.params, target_registry_index);
        let helper_col_label = setup.params.shard_registry[helper_registry_index].z;
        let helper_z = setup.params.shard_registry[target_registry_index].z;
        let live_unauth_len = warmed[target_registry_index].store.unauth.len();
        let segment_count = live_unauth_len.div_ceil(setup.params.segment_len as u64) as usize;
        let segs: Vec<_> = (0..segment_count).collect();
        let target_row_label_indices = target_claim
            .row_label_indices(&setup.params)
            .expect("target claim rows");
        let target_row_labels: Vec<_> = target_row_label_indices
            .iter()
            .map(|&idx| setup.params.row_labels[idx])
            .collect();
        let target_row_count = target_row_label_indices.len();

        let mut helper_core_server = warmed[helper_registry_index].clone();
        let mut helper_rpc_server = warmed[helper_registry_index].clone();

        let core_row_probe = core_row_matrix_reply(&mut helper_core_server, &segs, helper_z);
        let core_shard_probe = core_eval_matrix_reply(
            &mut helper_core_server,
            &segs,
            helper_col_label,
            &target_row_labels,
        );
        let helper_row_replies_per_segment = core_row_probe
            .first()
            .and_then(|batches| batches.first())
            .map(|rows| rows.len())
            .unwrap_or(0);
        let helper_eval_scalars_per_segment = core_shard_probe
            .first()
            .and_then(|values| values.first())
            .map(|values| values.len())
            .unwrap_or(0);

        let rpc_row_probe = rpc_row_matrix_reply(&mut helper_rpc_server, &segs, helper_z);
        let rpc_shard_probe = rpc_eval_matrix_reply(
            &mut helper_rpc_server,
            &segs,
            helper_col_label,
            &target_row_labels,
        );
        let helper_row_reply_wire_bytes = net::wire::encode_binary(&rpc_row_probe)
            .expect("encode helper row reply")
            .len();
        let helper_eval_reply_wire_bytes = net::wire::encode_binary(&rpc_shard_probe)
            .expect("encode helper shard-eval reply")
            .len();

        let recover_fixture = build_recover_fixture(setup, &warmed, target_registry_index, 0, 0);
        let recover_correcting_clean_fixture = (max_errors > 0)
            .then(|| build_recover_fixture(setup, &warmed, target_registry_index, max_errors, 0));
        let recover_correcting_corrupt_fixture = (max_errors > 0).then(|| {
            build_recover_fixture(
                setup,
                &warmed,
                target_registry_index,
                max_errors,
                max_errors,
            )
        });
        let row_helpers_per_segment = setup
            .params
            .s
            .div_ceil(helper_row_replies_per_segment.max(1));
        let recover_helper_reply_bytes_modeled = segment_count as u64
            * ((row_helpers_per_segment * helper_row_replies_per_segment) as u64
                + (setup.params.need * target_row_count) as u64)
            * FIELD_BYTES;
        let recover_correcting_helper_reply_bytes_modeled = (max_errors > 0).then(|| {
            let correcting_row_points = error_correcting_share_target(
                setup.params.s,
                max_errors.saturating_mul(ROWS_PER_CLAIM),
            );
            let correcting_row_helpers_per_segment =
                correcting_row_points.div_ceil(helper_row_replies_per_segment.max(1));
            let correcting_eval_helpers =
                error_correcting_share_target(setup.params.need, max_errors);
            segment_count as u64
                * ((correcting_row_helpers_per_segment * helper_row_replies_per_segment) as u64
                    + (correcting_eval_helpers * target_row_count) as u64)
                * FIELD_BYTES
        });

        Self {
            setup: setup.clone(),
            helper_core_server: warmed[helper_registry_index].clone(),
            helper_rpc_server: warmed[helper_registry_index].clone(),
            recover_fixture,
            recover_correcting_clean_fixture,
            recover_correcting_corrupt_fixture,
            bulk_updates_leaves,
            expected_auth_root: warmed[target_registry_index].store.root,
            helper_z,
            helper_col_label,
            target_row_labels,
            kappa: setup.cfg.coded.kappa,
            segment_len: setup.params.segment_len,
            n_grid: setup.params.n_grid,
            need: setup.params.need,
            shard_count: setup.params.shard_registry.len(),
            repair_helper_count: setup.params.need,
            bulk_updates_bytes: BULK_UPDATES_BYTES,
            bulk_updates_size: BULK_UPDATES_SIZE,
            committed_bulk_update_batches,
            allocated_leaves: committed_bulk_update_batches * BULK_UPDATES_SIZE,
            live_unauth_len,
            segment_count,
            target_registry_index,
            target_row_count,
            helper_registry_index,
            helper_row_replies_per_segment,
            helper_eval_scalars_per_segment,
            helper_row_reply_wire_bytes,
            helper_eval_reply_wire_bytes,
            helper_assist_total_wire_bytes: helper_row_reply_wire_bytes
                + helper_eval_reply_wire_bytes,
            recover_helper_reply_bytes_modeled,
            recover_correcting_helper_reply_bytes_modeled,
            full_bulk_updates_download_bytes: (committed_bulk_update_batches * BULK_UPDATES_BYTES)
                as u64,
            max_errors,
            corrupt_helper_count: if max_errors == 0 { 0 } else { max_errors },
            correcting_repair_helper_count: if max_errors == 0 {
                0
            } else {
                error_correcting_share_target(setup.params.need, max_errors)
            },
        }
    }

    pub fn helper_core_assist_once(
        &mut self,
    ) -> (
        Vec<Vec<Vec<paper_a_optimized::RowShare>>>,
        Vec<Vec<Vec<Fp>>>,
    ) {
        let segs = self.segs();
        let rows = core_row_matrix_reply(&mut self.helper_core_server, &segs, self.helper_z);
        let evals = core_eval_matrix_reply(
            &mut self.helper_core_server,
            &segs,
            self.helper_col_label,
            &self.target_row_labels,
        );
        (rows, evals)
    }

    pub fn helper_rpc_assist_once(&mut self) -> (net::Response, net::Response) {
        let segs = self.segs();
        let row_resp = rpc_row_matrix_reply(&mut self.helper_rpc_server, &segs, self.helper_z);
        let shard_resp = rpc_eval_matrix_reply(
            &mut self.helper_rpc_server,
            &segs,
            self.helper_col_label,
            &self.target_row_labels,
        );
        (row_resp, shard_resp)
    }

    pub fn recover_segments_once(&self) -> Vec<(usize, RecoveredClaimSegmentState)> {
        self.recover_fixture.recover_segments_once()
    }

    pub fn recover_segments_correcting_clean_once(
        &self,
    ) -> Option<Vec<(usize, RecoveredClaimSegmentState)>> {
        self.recover_correcting_clean_fixture
            .as_ref()
            .map(RecoverFixture::recover_segments_once)
    }

    pub fn recover_segments_correcting_corrupt_once(
        &self,
    ) -> Option<Vec<(usize, RecoveredClaimSegmentState)>> {
        self.recover_correcting_corrupt_fixture
            .as_ref()
            .map(RecoverFixture::recover_segments_once)
    }

    pub fn install_segments_once(
        &self,
        repaired_segments: Vec<(usize, RecoveredClaimSegmentState)>,
    ) -> UnauthStore {
        self.recover_fixture
            .install_segments_once(repaired_segments)
    }

    pub fn recover_and_install_once(&self) -> UnauthStore {
        self.recover_fixture.recover_and_install_once()
    }

    pub fn rebuild_from_downloaded_bulk_updates_once(&self) -> AuthServer {
        let local_server = make_local_server(
            self.target_registry_index as u64,
            build_claim_server(&self.setup, self.target_registry_index),
        );
        let local_servers = vec![local_server.clone()];
        let mut client = make_cluster_client(&local_servers, &self.setup);
        let mut planner =
            PreparedBulkUpdatesAppend::new(self.setup.cfg.auth.tree_height).expect("planner");
        let mut frontier = None;
        let mut frontier_len = 0u64;
        for leaves in &self.bulk_updates_leaves {
            let next_frontier = planner
                .append_from_frontier(&mut client, frontier_len, frontier.as_ref(), leaves)
                .expect("append downloaded bulk_updates");
            frontier_len += BULK_UPDATES_SIZE as u64;
            frontier = Some(next_frontier);
        }
        local_server.inner.lock().expect("server lock").clone()
    }

    pub fn validate_recovery(&self) {
        let recovered = self.recover_and_install_once();
        self.validate_recovered_store(&recovered);

        if let Some(fixture) = &self.recover_correcting_clean_fixture {
            let recovered = fixture.recover_and_install_once();
            self.validate_recovered_store(&recovered);
        }
        if let Some(fixture) = &self.recover_correcting_corrupt_fixture {
            let recovered = fixture.recover_and_install_once();
            self.validate_recovered_store(&recovered);
        }

        let rebuilt = self.rebuild_from_downloaded_bulk_updates_once();
        assert_eq!(rebuilt.store.root, self.expected_auth_root);
        assert_eq!(
            rebuilt.store.unauth.len(),
            self.recover_fixture.expected_target.len()
        );
        assert_eq!(
            rebuilt.store.unauth.segments(),
            self.recover_fixture.expected_target.segments()
        );
        assert!(rebuilt.store.unauth.coded.equals_for_tests(
            &self.recover_fixture.expected_target.params,
            &self.recover_fixture.expected_target.coded,
        ));
    }

    fn validate_recovered_store(&self, recovered: &UnauthStore) {
        assert!(recovered.coded.equals_for_tests(
            &self.recover_fixture.expected_target.params,
            &self.recover_fixture.expected_target.coded,
        ));
    }

    fn segs(&self) -> Vec<usize> {
        self.recover_fixture.segs.clone()
    }
}
