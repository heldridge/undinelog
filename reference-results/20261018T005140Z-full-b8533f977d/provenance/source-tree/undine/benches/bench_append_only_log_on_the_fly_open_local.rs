//! Local per-server benchmark for openings served on the fly at a threshold
//! above the stored one, in the append-only-log application.
//!
//! One claim-bound server stores the log at threshold `kappa`, warmed by
//! appending committed bulk update batches through the real app frontier
//! path. A client then opens at `2^j kappa`. The timed operation is the
//! server's answer to one point-opening request and to one range-opening
//! request at that threshold, including response serialization, while its
//! state stays at `kappa`.
//!
//! Each on-the-fly answer is timed next to the answer of a copy of the server
//! that ran `j` kappa-doubles and so stores the log at the opened threshold.
//! The two answers are checked to be identical byte for byte. The time to run
//! those `j` kappa-doubles is reported as the cost the on-the-fly path avoids.
//!
//! The output has one entry per answer (`query` is `point` or `range`) and
//! opened threshold, and one `stitch` entry per threshold above the stored
//! one. Every timed series has exactly `TRIALS` samples; the trials of all
//! series are interleaved.

#[path = "support/local_setup.rs"]
mod local_setup;

use append_only_log_demo::codec::bulk_updates_bytes_as_leaves;
use append_only_log_demo::frontier::PreparedBulkUpdatesAppend;
use append_only_log_demo::layout::{
    BULK_UPDATES_BYTES, BULK_UPDATES_SIZE, FIELD_ELEMENTS_PER_CODEC_GROUP,
    PAYLOAD_BYTES_PER_CODEC_GROUP, bulk_updates_start_leaf_index,
};
use paper_a_optimized::auth::client::{AuthClient, AuthTransport, ClientError as AuthClientError};
use paper_a_optimized::auth::mapping::NodeLinearization;
use paper_a_optimized::auth::service::{
    AuthBatchWriteRequest, AuthBatchWriteResponse, AuthFreshSubtreeWriteRequest, AuthReadResponse,
    AuthRequest, AuthResponse, AuthServer, AuthService, AuthSubtreeReadResponse,
    AuthSubtreeUpperPathReadResponse, AuthWriteRequest, AuthWriteResponse,
};
use paper_a_optimized::net;
use paper_a_optimized::net::wire::AuthReadRequest as NetAuthReadRequest;
use paper_a_optimized::unauth::service::{UnauthRequest, UnauthResponse, UnauthService};
use paper_a_optimized::unauth::{
    ClaimBundle, UnauthOptimizedReadRequest, UnauthOptimizedReadResponse, UnauthReadObjectId,
    build_params_for_kappa,
};
use paper_a_optimized::{
    AuthConfig, AuthParams, CodedArrayConfig, CodedArrayParams, Config, build_coded_params,
    build_shard_registry_deterministic,
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
/// Stored threshold.
const DEFAULT_KAPPA: usize = 64;
/// Openings are measured at `2^j kappa` for `j` up to this many stitches.
const DEFAULT_MAX_LEVELS: usize = 3;
const DEFAULT_COMMITTED_BULK_UPDATE_BATCHES: usize = 3;
const DEFAULT_RANGE_BYTE_LENS: [usize; 2] = [1 << 20, 4 << 20];
const ENV_PREFIX: &str = "BENCH_APPEND_ONLY_LOG_ON_THE_FLY_OPEN";
/// A point answer lasts only a few microseconds. Each retained sample is the
/// mean of this many calls; inner calls are not independent observations.
const POINT_CALLS_PER_SAMPLE: usize = 32;

#[derive(Clone)]
struct LocalServer {
    label: String,
    inner: Arc<Mutex<AuthServer>>,
}

/// Transport that records the optimized read request a client plans.
#[derive(Clone, Default)]
struct LocalTransport {
    captured_request: Option<Arc<Mutex<Option<UnauthOptimizedReadRequest>>>>,
}

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

    fn supports_unauth_optimized_reads(&self) -> bool {
        true
    }

    fn unauth_optimized_read(
        &self,
        server: &Self::Server,
        request: &UnauthOptimizedReadRequest,
    ) -> Result<UnauthOptimizedReadResponse, AuthClientError> {
        if let Some(captured) = &self.captured_request {
            *captured.lock().expect("request capture lock") = Some(request.clone());
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
    params: CodedArrayParams,
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
    point_calls_per_sample: usize,
    seed: u64,
    trials: usize,
    stored_kappa: usize,
    max_levels: usize,
    registry_claims: usize,
    local_claims: usize,
    committed_bulk_update_batches: usize,
    stored_segments: usize,
    query_value_index: u64,
    timestamp_unix_ms: u64,
    git: Option<String>,
    notes: String,
}

/// One measured answer at one opened threshold.
///
/// `query` is `"point"` for the point opening and `"range"` for a range
/// opening of `query_byte_len` bytes. At `levels = 0` the server answers at
/// its stored threshold, so only the stored series is reported.
#[derive(Clone, Debug, Serialize)]
struct AnswerEntry {
    query: &'static str,
    stored_kappa: usize,
    open_kappa: usize,
    /// Stitches between the stored threshold and the opened one.
    levels: usize,
    query_byte_len: usize,
    query_leaf_start: u64,
    query_leaf_count: usize,
    request_segments: usize,
    request_points: usize,
    response_field_elements: usize,
    serialized_response_bytes: usize,
    /// Server stored at `stored_kappa`, request at `open_kappa`.
    #[serde(skip_serializing_if = "Option::is_none")]
    helper_on_the_fly_serialized_ns: Option<MetricStats>,
    /// Server stored at `open_kappa`, request at `open_kappa`.
    helper_stored_at_open_kappa_serialized_ns: MetricStats,
}

/// Running `levels` kappa-doubles on the server's whole state: the work an
/// on-the-fly answer does not do.
#[derive(Clone, Debug, Serialize)]
struct StitchEntry {
    query: &'static str,
    stored_kappa: usize,
    open_kappa: usize,
    levels: usize,
    /// Source segments combined into one segment of the opened threshold.
    source_segments_per_open_segment: usize,
    /// Segments of the opened layout that cover the stored log.
    open_segments: usize,
    stitch_whole_state_ns: MetricStats,
}

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
enum BenchEntry {
    Answer(AnswerEntry),
    Stitch(StitchEntry),
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

/// One timed operation: an answer of a server to a request, or the stitch of
/// the stored server's whole state.
enum Timed<'a> {
    Point {
        server: &'a AuthServer,
        req: &'a net::Message,
    },
    Range {
        server: &'a AuthServer,
        request: &'a UnauthOptimizedReadRequest,
    },
    Stitch {
        stored: &'a AuthServer,
        levels: usize,
    },
}

impl Timed<'_> {
    /// Run the operation once and return its duration in nanoseconds.
    fn sample(&self) -> u64 {
        match self {
            Timed::Point { server, req } => {
                // Warm the request once, then batch these short operations.
                black_box(serialized_point_response(server, req));
                let start = Instant::now();
                for _ in 0..POINT_CALLS_PER_SAMPLE {
                    black_box(serialized_point_response(black_box(server), black_box(req)));
                }
                (start.elapsed().as_nanos() / POINT_CALLS_PER_SAMPLE as u128) as u64
            }
            Timed::Range { server, request } => {
                black_box(serialized_range_response(server, request));
                let start = Instant::now();
                black_box(serialized_range_response(server, request));
                start.elapsed().as_nanos() as u64
            }
            Timed::Stitch { stored, levels } => {
                let mut server = (*stored).clone();
                let start = Instant::now();
                kappa_double_levels(&mut server, *levels);
                let elapsed = start.elapsed().as_nanos() as u64;
                black_box(server);
                elapsed
            }
        }
    }
}

/// Take `trials` samples of every operation. Each trial visits every
/// operation once, so a slow stretch of the machine touches one sample of
/// each metric instead of all samples of one.
fn sample_interleaved(trials: usize, operations: &[Timed<'_>]) -> Vec<MetricStats> {
    let mut samples = vec![Vec::with_capacity(trials); operations.len()];
    for _ in 0..trials {
        for (operation, samples) in operations.iter().zip(samples.iter_mut()) {
            samples.push(operation.sample());
        }
    }
    samples.into_iter().map(stats_from_samples).collect()
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

fn parse_usize_env(suffix: &str, default: usize) -> usize {
    std::env::var(format!("{ENV_PREFIX}_{suffix}"))
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn parse_usize_list_env(suffix: &str, default: &[usize]) -> Vec<usize> {
    let Some(raw) = std::env::var(format!("{ENV_PREFIX}_{suffix}")).ok() else {
        return default.to_vec();
    };
    let values = raw
        .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse::<usize>()
                .expect("list entry must be an integer")
        })
        .collect::<Vec<_>>();
    assert!(
        !values.is_empty(),
        "{ENV_PREFIX}_{suffix} must not be empty"
    );
    values
}

/// Config at the stored threshold. The registry holds `2^max_levels kappa`
/// claims, enough for an opening at the highest measured threshold.
fn build_setup(kappa: usize, max_levels: usize) -> BenchSetup {
    let tree_height = parse_usize_env("TREE_HEIGHT", DEFAULT_TREE_HEIGHT);
    assert!(kappa.is_power_of_two(), "kappa must be a power of two");
    assert!(kappa >= 2, "kappa must be at least two");
    let registry_claims = kappa << max_levels;
    let cfg = Config {
        auth: AuthConfig {
            tree_height,
            alloc_leaf_batch: kappa * kappa / 2,
            node_linearization: NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            kappa,
            shard_count: registry_claims,
            shard_registry: build_shard_registry_deterministic(registry_claims),
            party_registry_indices: (0..registry_claims).map(|idx| vec![idx]).collect(),
            seed: BENCH_SEED,
        },
    };
    let params = build_coded_params(&cfg.coded);
    let auth = AuthParams::new(cfg.auth.tree_height);
    BenchSetup { cfg, params, auth }
}

fn build_claim_server(setup: &BenchSetup, local_claims: usize) -> AuthServer {
    let params = &setup.params;
    let claims = (0..local_claims)
        .map(|idx| {
            ClaimBundle::exact_for_registry_index(params, idx)
                .expect("exact claim should exist for registry index")
        })
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

fn make_client(
    server: LocalServer,
    setup: &BenchSetup,
    params: &CodedArrayParams,
    captured_request: Option<Arc<Mutex<Option<UnauthOptimizedReadRequest>>>>,
) -> AuthClient<LocalTransport> {
    AuthClient::new(
        LocalTransport { captured_request },
        params.clone(),
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

/// Warm one claim-bound server at the stored threshold through real bulk
/// update appends.
fn build_warmed_server(
    setup: &BenchSetup,
    local_claims: usize,
    committed_bulk_update_batches: usize,
) -> AuthServer {
    let local = make_local_server(build_claim_server(setup, local_claims));
    let mut client = make_client(local.clone(), setup, &setup.params, None);
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
    drop(client);
    local_setup::take_local_server(local.inner)
}

/// Run `levels` kappa-doubles on a server through its service interface.
fn kappa_double_levels(server: &mut AuthServer, levels: usize) {
    for _ in 0..levels {
        match UnauthService::handle_request(server, UnauthRequest::KappaDouble) {
            UnauthResponse::KappaDouble { ok: true } => {}
            other => panic!("unexpected kappa-double response: {other:?}"),
        }
    }
}

fn byte_range_to_leaf_range(offset: usize, len: usize) -> (usize, usize) {
    let end = offset.checked_add(len).expect("byte range overflow");
    let start_group = offset / PAYLOAD_BYTES_PER_CODEC_GROUP;
    let end_group = end.div_ceil(PAYLOAD_BYTES_PER_CODEC_GROUP);
    let start_leaf = start_group * FIELD_ELEMENTS_PER_CODEC_GROUP;
    let end_leaf = end_group * FIELD_ELEMENTS_PER_CODEC_GROUP;
    (start_leaf, end_leaf - start_leaf)
}

/// The optimized read request a client at the threshold of `params` plans
/// for one leaf range. The server holds one claim, so the read itself cannot
/// complete; only the planned request is kept.
fn plan_range_request(
    setup: &BenchSetup,
    params: &CodedArrayParams,
    server: &AuthServer,
    leaf_start: u64,
    leaf_count: usize,
) -> UnauthOptimizedReadRequest {
    let captured = Arc::new(Mutex::new(None));
    let mut client = make_client(
        make_local_server(server.clone()),
        setup,
        params,
        Some(Arc::clone(&captured)),
    );
    let _ = client.read_range_with_proof(leaf_start, leaf_count);
    let request = captured
        .lock()
        .expect("request capture lock")
        .clone()
        .expect("captured optimized request");
    assert_eq!(request.kappa, params.need);
    request
}

/// One server's serialized answer to an authenticated point-opening request.
fn serialized_point_response(server: &AuthServer, req: &net::Message) -> Vec<u8> {
    let resp = net::auth::handle_read_only_message(server, req.clone())
        .expect("point-open read-only response");
    net::wire::encode_binary(&resp).expect("serialize point-open response")
}

/// One server's serialized answer to an optimized range-opening request.
fn serialized_range_response(server: &AuthServer, request: &UnauthOptimizedReadRequest) -> Vec<u8> {
    let response = server.handle_optimized_read(request);
    let net_response =
        net::unauth::core_optimized_response_to_net(response).expect("pack optimized response");
    net::wire::encode_binary(&net::Response::UnauthOptimizedRead { resp: net_response })
        .expect("serialize range-open response")
}

fn response_field_elements(response: &UnauthOptimizedReadResponse) -> usize {
    use paper_a_optimized::unauth::UnauthReadObjectContribution as Contribution;
    response
        .objects
        .iter()
        .map(|contribution| match contribution {
            Contribution::Segment { shards, .. } => {
                shards.iter().map(|shard| shard.values.len()).sum::<usize>()
            }
            Contribution::Block { rows, .. } => rows.len(),
            Contribution::Point { shares, .. } => shares.len(),
        })
        .sum()
}

/// One planned range opening at one threshold.
struct RangeCase {
    query_byte_len: usize,
    leaf_start: u64,
    leaf_count: usize,
    request: UnauthOptimizedReadRequest,
}

/// Everything measured at one opened threshold.
struct LevelCase {
    levels: usize,
    open_kappa: usize,
    /// Copy of the stored server after `levels` kappa-doubles.
    stitched: AuthServer,
    point_req: net::Message,
    ranges: Vec<RangeCase>,
}

fn build_level_case(
    setup: &BenchSetup,
    stored: &AuthServer,
    levels: usize,
    query_leaf_index: u64,
    range_byte_lens: &[usize],
) -> LevelCase {
    let open_kappa = setup.params.need << levels;
    let open_params = build_params_for_kappa(&setup.params, open_kappa).expect("open params");
    assert_eq!(
        stored.store.unauth.stitch_levels_to(open_kappa),
        Some(levels)
    );

    // The reference server stores the same log at the opened threshold.
    let mut stitched = stored.clone();
    kappa_double_levels(&mut stitched, levels);
    assert_eq!(stitched.params.need, open_kappa);
    let point_req = net::Message::AuthRead {
        req: NetAuthReadRequest {
            leaf: query_leaf_index,
            kappa: open_kappa as u64,
        },
    };
    assert_eq!(
        serialized_point_response(stored, &point_req),
        serialized_point_response(&stitched, &point_req),
        "on-the-fly point opening differs from the stitched server's at kappa = {open_kappa}"
    );

    let ranges = range_byte_lens
        .iter()
        .map(|&query_byte_len| {
            let (leaf_start, leaf_count) = byte_range_to_leaf_range(0, query_byte_len);
            let leaf_start = bulk_updates_start_leaf_index(0) + leaf_start as u64;
            let request = plan_range_request(setup, &open_params, stored, leaf_start, leaf_count);
            assert_eq!(
                serialized_range_response(stored, &request),
                serialized_range_response(&stitched, &request),
                "on-the-fly range opening differs from the stitched server's at kappa = {open_kappa}"
            );
            RangeCase {
                query_byte_len,
                leaf_start,
                leaf_count,
                request,
            }
        })
        .collect();

    LevelCase {
        levels,
        open_kappa,
        stitched,
        point_req,
        ranges,
    }
}

/// Time every answer of every case, and the stitch of the whole state for
/// every threshold above the stored one, and assemble the result entries.
fn bench_cases(trials: usize, stored: &AuthServer, cases: &[LevelCase]) -> Vec<BenchEntry> {
    // Per case, in the order the entries are assembled below: the point
    // answer, then each range answer, on-the-fly before stored; then the
    // stitch. At the stored threshold only the stored answer exists.
    let mut operations = Vec::new();
    for case in cases {
        let on_the_fly = (case.levels > 0).then_some(stored);
        for server in on_the_fly.into_iter().chain([&case.stitched]) {
            operations.push(Timed::Point {
                server,
                req: &case.point_req,
            });
        }
        for range in &case.ranges {
            for server in on_the_fly.into_iter().chain([&case.stitched]) {
                operations.push(Timed::Range {
                    server,
                    request: &range.request,
                });
            }
        }
        if case.levels > 0 {
            operations.push(Timed::Stitch {
                stored,
                levels: case.levels,
            });
        }
    }
    let mut stats = sample_interleaved(trials, &operations).into_iter();
    let mut next = || stats.next().expect("one stats entry per operation");

    let stored_kappa = stored.params.need;
    let mut entries = Vec::new();
    for case in cases {
        let query_leaf_index = match &case.point_req {
            net::Message::AuthRead { req } => req.leaf,
            other => panic!("unexpected point request: {other:?}"),
        };
        let core_point = stored.handle_read_at(query_leaf_index, case.open_kappa);
        let array_shares_returned: usize =
            core_point.nodes.iter().map(|node| node.shares.len()).sum();
        assert!(array_shares_returned > 0);
        entries.push(BenchEntry::Answer(AnswerEntry {
            query: "point",
            stored_kappa,
            open_kappa: case.open_kappa,
            levels: case.levels,
            query_byte_len: 0,
            query_leaf_start: query_leaf_index,
            query_leaf_count: 1,
            request_segments: 0,
            request_points: core_point.nodes.len(),
            response_field_elements: array_shares_returned,
            serialized_response_bytes: serialized_point_response(stored, &case.point_req).len(),
            helper_on_the_fly_serialized_ns: (case.levels > 0).then(&mut next),
            helper_stored_at_open_kappa_serialized_ns: next(),
        }));
        for range in &case.ranges {
            let request_segments = range
                .request
                .objects
                .iter()
                .filter(|object| matches!(object, UnauthReadObjectId::Segment { .. }))
                .count();
            entries.push(BenchEntry::Answer(AnswerEntry {
                query: "range",
                stored_kappa,
                open_kappa: case.open_kappa,
                levels: case.levels,
                query_byte_len: range.query_byte_len,
                query_leaf_start: range.leaf_start,
                query_leaf_count: range.leaf_count,
                request_segments,
                request_points: range.request.objects.len() - request_segments,
                response_field_elements: response_field_elements(
                    &stored.handle_optimized_read(&range.request),
                ),
                serialized_response_bytes: serialized_range_response(stored, &range.request).len(),
                helper_on_the_fly_serialized_ns: (case.levels > 0).then(&mut next),
                helper_stored_at_open_kappa_serialized_ns: next(),
            }));
        }
        if case.levels > 0 {
            entries.push(BenchEntry::Stitch(StitchEntry {
                query: "stitch",
                stored_kappa,
                open_kappa: case.open_kappa,
                levels: case.levels,
                source_segments_per_open_segment: 1 << (2 * case.levels),
                open_segments: case.stitched.store.unauth.segments(),
                stitch_whole_state_ns: next(),
            }));
        }
    }
    entries
}

fn run_benchmark() {
    let trials = parse_usize_env("TRIALS", DEFAULT_TRIALS);
    let kappa = parse_usize_env("KAPPA", DEFAULT_KAPPA);
    let max_levels = parse_usize_env("MAX_LEVELS", DEFAULT_MAX_LEVELS);
    let local_claims = parse_usize_env("LOCAL_CLAIMS", 1);
    let committed_bulk_update_batches = parse_usize_env(
        "COMMITTED_BULK_UPDATE_BATCHES",
        DEFAULT_COMMITTED_BULK_UPDATE_BATCHES,
    );
    let range_byte_lens = parse_usize_list_env("RANGE_BYTE_LEN_LIST", &DEFAULT_RANGE_BYTE_LENS);
    for &len in &range_byte_lens {
        assert!(
            len <= committed_bulk_update_batches * BULK_UPDATES_BYTES,
            "range of {len} bytes exceeds the committed log"
        );
    }
    let query_bulk_updates = committed_bulk_update_batches / 2;
    let query_leaf_index =
        bulk_updates_start_leaf_index(query_bulk_updates as u64) + (BULK_UPDATES_SIZE / 2) as u64;

    let setup = build_setup(kappa, max_levels);
    let stored = build_warmed_server(&setup, local_claims, committed_bulk_update_batches);
    let cases = (0..=max_levels)
        .map(|levels| build_level_case(&setup, &stored, levels, query_leaf_index, &range_byte_lens))
        .collect::<Vec<_>>();
    let entries = bench_cases(trials, &stored, &cases);

    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let results = BenchResults {
        meta: BenchMeta {
            point_calls_per_sample: POINT_CALLS_PER_SAMPLE,
            seed: BENCH_SEED,
            trials,
            stored_kappa: kappa,
            max_levels,
            registry_claims: setup.params.shard_registry.len(),
            local_claims,
            committed_bulk_update_batches,
            stored_segments: stored.store.unauth.segments(),
            query_value_index: query_leaf_index,
            timestamp_unix_ms: timestamp_ms,
            git: git_sha(),
            notes: "local append-only-log per-server on-the-fly opening benchmark: warm one claim-bound server at the stored threshold through real bulk updates appends, then time its serialized answers to one point-opening request and to range-opening requests at 2^j times the stored threshold, next to the answers of a copy that ran j kappa-doubles".to_string(),
        },
        entries,
    };

    let out_dir = Path::new("results");
    let _ = fs::create_dir_all(out_dir);
    let path = out_dir.join("bench_append_only_log_on_the_fly_open_local.json");
    let data = serde_json::to_vec_pretty(&results)
        .expect("serialize bench_append_only_log_on_the_fly_open_local");
    fs::write(path, data).expect("write bench_append_only_log_on_the_fly_open_local.json");
}

fn main() {
    run_benchmark();
}

#[test]
#[ignore]
fn bench_append_only_log_on_the_fly_open_local() {
    run_benchmark();
}
