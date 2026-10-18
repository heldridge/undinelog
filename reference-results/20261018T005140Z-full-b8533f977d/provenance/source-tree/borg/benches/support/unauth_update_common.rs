//! Unified unauthenticated sparse-update benchmark suite.
//!
//! This suite measures the current migrated unauthenticated update path:
//! - public batch updates through the unauth client/service boundary,
//! - maintained-row updates together with primary-column updates,
//! - aggregate and FFT-forced update modes as a within-system comparison,
//! - one structured aligned block update versus one batched point-decomposition
//!   request for the same logical block write.
//!
//! Communication is modeled from the current delta-broadcast message shape,
//! while correctness is checked by comparing each claim-bound local store
//! against the corresponding subset of an expected full store after update.

use super::rs_striped_baseline::{
    default_helper_indices, encode_segment_from_values, fair_layout_for_params,
};
use ark_ff::UniformRand;
use paper_a_optimized::unauth::client::{
    ClientError, UnauthClient as CoreUnauthClient, UnauthTransport,
};
use paper_a_optimized::unauth::service::{UnauthRequest, UnauthResponse, UnauthService};
use paper_a_optimized::unauth::{ClaimBundle, DeltaMode, UnauthStore};
use paper_a_optimized::{
    CodedArrayConfig, CodedArrayParams, Fp, Value256, build_coded_params,
    build_shard_registry_deterministic, decode_value, encode_value,
};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::Serialize;
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub const BENCH_SEED: u64 = 42;
pub const DEFAULT_TRIALS: usize = 3;
pub const DEFAULT_K_VALUES: &[usize] = &[8, 16, 32];
pub const DEFAULT_S_VALUES: &[usize] = &[4, 8, 16, 32];
pub const DEFAULT_SEGMENT_LENS: &[usize] = &[128, 1024, 2048, 4096];
pub const DEFAULT_BATCH_SIZES: &[usize] = &[1, 8, 64];
const FIELD_BYTES: u64 = 32;
const INDEX_BYTES: u64 = 8;

type UnauthClient = CoreUnauthClient<LocalTransport>;

#[derive(Clone)]
struct SharedStore(Arc<Mutex<UnauthStore>>);

impl SharedStore {
    fn new(store: UnauthStore) -> Self {
        Self(Arc::new(Mutex::new(store)))
    }

    fn borrow(&self) -> MutexGuard<'_, UnauthStore> {
        self.0.lock().expect("bench store lock")
    }

    fn borrow_mut(&self) -> MutexGuard<'_, UnauthStore> {
        self.0.lock().expect("bench store lock")
    }
}

#[derive(Clone)]
struct LocalServer {
    party_id: u64,
    store: SharedStore,
}

#[derive(Clone)]
struct LocalTransport;

impl UnauthTransport for LocalTransport {
    type Server = LocalServer;

    fn server_label(&self, server: &Self::Server) -> String {
        server.party_id.to_string()
    }

    fn unauth_get_len(&self, server: &Self::Server) -> Result<u64, ClientError> {
        match server
            .store
            .borrow_mut()
            .handle_request(UnauthRequest::GetLen)
        {
            UnauthResponse::Len { len } => Ok(len),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid len response: {:?}",
                other
            ))),
        }
    }

    fn unauth_read(
        &self,
        server: &Self::Server,
        idx: u64,
    ) -> Result<Vec<paper_a_optimized::ShardShare>, ClientError> {
        match server
            .store
            .borrow_mut()
            .handle_request(UnauthRequest::Read { idx })
        {
            UnauthResponse::Read { shares } => Ok(shares),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid read response: {:?}",
                other
            ))),
        }
    }

    fn unauth_read_block(
        &self,
        server: &Self::Server,
        block: u64,
    ) -> Result<Vec<paper_a_optimized::RowShare>, ClientError> {
        match server
            .store
            .borrow_mut()
            .handle_request(UnauthRequest::ReadBlock { block })
        {
            UnauthResponse::ReadBlock { rows } => Ok(rows),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid block response: {:?}",
                other
            ))),
        }
    }

    fn unauth_read_rows_at_z(
        &self,
        server: &Self::Server,
        seg: usize,
        z_eval: Fp,
    ) -> Result<Vec<paper_a_optimized::RowShare>, ClientError> {
        match server
            .store
            .borrow_mut()
            .handle_request(UnauthRequest::ReadRowsAtZ { seg, z_eval })
        {
            UnauthResponse::ReadRowsAtZ { rows } => Ok(rows),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid rows-at-z response: {:?}",
                other
            ))),
        }
    }

    fn unauth_read_rows_batch_at_z(
        &self,
        server: &Self::Server,
        seg: usize,
        z_evals: &[Fp],
    ) -> Result<Vec<Vec<paper_a_optimized::RowShare>>, ClientError> {
        match server
            .store
            .borrow_mut()
            .handle_request(UnauthRequest::ReadRowsBatchAtZ {
                seg,
                z_evals: z_evals.to_vec(),
            }) {
            UnauthResponse::ReadRowsBatchAtZ { rows } => Ok(rows),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid rows-batch response: {:?}",
                other
            ))),
        }
    }

    fn unauth_read_rows_matrix(
        &self,
        server: &Self::Server,
        requests: &[(usize, Vec<Fp>)],
    ) -> Result<Vec<Vec<Vec<paper_a_optimized::RowShare>>>, ClientError> {
        match server
            .store
            .borrow_mut()
            .handle_request(UnauthRequest::ReadRowsMatrix {
                requests: requests
                    .iter()
                    .map(
                        |(seg, z_evals)| paper_a_optimized::unauth::service::RowReadRequest {
                            seg: *seg,
                            z_evals: z_evals.clone(),
                        },
                    )
                    .collect(),
            }) {
            UnauthResponse::ReadRowsMatrix { rows } => Ok(rows),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid rows-matrix response: {:?}",
                other
            ))),
        }
    }

    fn unauth_multi_read(
        &self,
        server: &Self::Server,
        idxs: &[u64],
    ) -> Result<Vec<Vec<paper_a_optimized::ShardShare>>, ClientError> {
        match server
            .store
            .borrow_mut()
            .handle_request(UnauthRequest::MultiRead {
                idxs: idxs.to_vec(),
            }) {
            UnauthResponse::MultiRead { shares } => Ok(shares),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid multi-read response: {:?}",
                other
            ))),
        }
    }

    fn unauth_repair_shard(
        &self,
        server: &Self::Server,
        seg: usize,
        registry_index: usize,
    ) -> Result<Vec<Fp>, ClientError> {
        match server
            .store
            .borrow_mut()
            .handle_request(UnauthRequest::RepairShard {
                seg,
                registry_index,
            }) {
            UnauthResponse::RepairShard { values } => Ok(values),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid repair response: {:?}",
                other
            ))),
        }
    }

    fn unauth_apply_deltas(
        &self,
        server: &Self::Server,
        deltas: &[(u64, Fp)],
        mode: DeltaMode,
    ) -> Result<bool, ClientError> {
        match server
            .store
            .borrow_mut()
            .handle_request(UnauthRequest::ApplyDeltas {
                deltas: deltas.to_vec(),
                mode,
            }) {
            UnauthResponse::ApplyDeltas { ok } => Ok(ok),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid apply response: {:?}",
                other
            ))),
        }
    }

    fn unauth_apply_block_deltas(
        &self,
        server: &Self::Server,
        block: u64,
        deltas: &[(usize, Fp)],
        mode: DeltaMode,
    ) -> Result<bool, ClientError> {
        match server
            .store
            .borrow_mut()
            .handle_request(UnauthRequest::ApplyBlockDeltas {
                block,
                deltas: deltas.to_vec(),
                mode,
            }) {
            UnauthResponse::ApplyBlockDeltas { ok } => Ok(ok),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid block apply response: {:?}",
                other
            ))),
        }
    }

    fn unauth_extend(&self, server: &Self::Server, new_len: u64) -> Result<bool, ClientError> {
        match server
            .store
            .borrow_mut()
            .handle_request(UnauthRequest::Extend { new_len })
        {
            UnauthResponse::Extend { ok } => Ok(ok),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid extend response: {:?}",
                other
            ))),
        }
    }

    fn unauth_migrate(
        &self,
        server: &Self::Server,
        target_s: usize,
        partner_shards: &[paper_a_optimized::unauth::PartnerShardMigration],
        helper_rows: &[paper_a_optimized::unauth::HelperRowMigration],
    ) -> Result<bool, ClientError> {
        match server
            .store
            .borrow_mut()
            .handle_request(UnauthRequest::Migrate {
                target_s,
                partner_shards: partner_shards.to_vec(),
                helper_rows: helper_rows.to_vec(),
            }) {
            UnauthResponse::Migrate { ok } => Ok(ok),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid migrate response: {:?}",
                other
            ))),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct MetricStats {
    mean: u64,
    p50: u64,
    p95: u64,
    trials: u64,
}

#[derive(Clone, Debug, Serialize)]
struct UpdateCommStats {
    server_count: u64,
    delta_count: u64,
    modeled_upload_bytes: u64,
    modeled_rpc_count: u64,
}

#[derive(Clone, Debug, Serialize)]
struct BenchEntry {
    segment_len: usize,
    message_size: usize,
    n_grid: usize,
    s: usize,
    k: usize,
    derived_s: usize,
    need: usize,
    shard_count: usize,
    rows_per_shard: usize,
    logical_len: u64,
    layer: String,
    operation: String,
    variant: Option<String>,
    baseline_kind: Option<String>,
    comparison_scope: Option<String>,
    decode_scope: Option<String>,
    batch_size: usize,
    update_mode: String,
    block_slots_touched: Option<usize>,
    point_count_equivalent: Option<usize>,
    comparison_role: Option<String>,
    download_bytes: Option<u64>,
    upload_bytes: Option<u64>,
    servers_contacted: Option<u64>,
    requires_read_modify_write: Option<bool>,
    update_ns: MetricStats,
    updates_per_sec_mean: f64,
    update_comm: UpdateCommStats,
}

#[derive(Clone, Debug, Serialize)]
struct BenchMeta {
    seed: u64,
    trials: usize,
    timestamp_unix_ms: u64,
    git: Option<String>,
    notes: String,
    k_values: Vec<usize>,
    s_values: Vec<usize>,
    segment_lens: Vec<usize>,
    batch_sizes: Vec<usize>,
    claim_model: String,
    claims_per_party: usize,
    effective_dimension_note: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct BenchResults {
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
        };
    }
    let sum: u128 = samples.iter().map(|&v| v as u128).sum();
    let mean = (sum / samples.len() as u128) as u64;
    samples.sort_unstable();
    MetricStats {
        mean,
        p50: percentile(&samples, 50),
        p95: percentile(&samples, 95),
        trials: samples.len() as u64,
    }
}

fn bench_metric<F, T>(trials: usize, mut f: F) -> MetricStats
where
    F: FnMut() -> T,
{
    let mut samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        let start = Instant::now();
        black_box(f());
        samples.push(start.elapsed().as_nanos() as u64);
    }
    stats_from_samples(samples)
}

fn bench_mutating_stores<F>(stores: &mut [UnauthStore], mut f: F) -> MetricStats
where
    F: FnMut(&mut UnauthStore),
{
    let mut samples = Vec::with_capacity(stores.len());
    for store in stores.iter_mut() {
        let start = Instant::now();
        black_box(f(store));
        samples.push(start.elapsed().as_nanos() as u64);
    }
    stats_from_samples(samples)
}

fn bench_client_workflow<F>(
    trials: usize,
    params: &CodedArrayParams,
    logical_len: u64,
    store_templates: &[UnauthStore],
    expected: &UnauthStore,
    mut f: F,
) -> MetricStats
where
    F: FnMut(&mut UnauthClient),
{
    let mut samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        let (mut client, servers) = build_client(params, logical_len, store_templates);
        let start = Instant::now();
        f(&mut client);
        samples.push(start.elapsed().as_nanos() as u64);
        for server in &servers {
            let store = server.store.borrow();
            assert_store_matches_expected_subset(&store, expected);
        }
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

pub fn parse_list_env(name: &str, default: &[usize]) -> Vec<usize> {
    let Ok(raw) = std::env::var(name) else {
        return default.to_vec();
    };
    let mut values = Vec::new();
    for token in raw.split(|c: char| c == ',' || c == ';' || c.is_whitespace()) {
        if token.is_empty() {
            continue;
        }
        let value = token
            .parse::<usize>()
            .unwrap_or_else(|_| panic!("invalid {} value `{}`", name, token));
        values.push(value);
    }
    if values.is_empty() {
        default.to_vec()
    } else {
        values
    }
}

pub fn parse_trials_env(name: &str) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_TRIALS)
}

fn build_params(segment_len: usize, s: usize) -> CodedArrayParams {
    let n_grid = segment_len;
    assert!(s.is_power_of_two(), "s must be a power of two");
    assert!(s <= n_grid, "s must fit inside n_grid");
    let shard_count = (n_grid - 1) / s + 1;
    let shard_registry = build_shard_registry_deterministic(shard_count, s, n_grid);
    let party_registry_indices = (0..shard_count).map(|idx| vec![idx]).collect();
    let cfg = CodedArrayConfig {
        segment_len,
        s,
        shard_count,
        shard_registry,
        party_registry_indices,
        seed: BENCH_SEED,
    };
    build_coded_params(&cfg)
}

fn build_params_from_k(segment_len: usize, k: usize) -> CodedArrayParams {
    let n_grid = segment_len;
    assert!(k.is_power_of_two(), "k must be a power of two");
    assert!(k <= n_grid, "k must fit inside n_grid");
    assert_eq!(n_grid % k, 0, "k must divide n_grid");
    build_params(segment_len, n_grid / k)
}

fn effective_k(params: &CodedArrayParams) -> usize {
    params.need
}

fn derived_s(params: &CodedArrayParams) -> usize {
    let k = effective_k(params);
    assert_eq!(params.n_grid % k, 0, "k must divide n_grid");
    let derived = params.n_grid / k;
    assert_eq!(derived, params.s, "derived s must match params.s");
    derived
}

fn build_seed_deltas(len: u64, seed: u64) -> Vec<(u64, Fp)> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut deltas = Vec::with_capacity(len as usize);
    for idx in 0..len {
        deltas.push((idx, Fp::rand(&mut rng)));
    }
    deltas
}

fn build_logical_values(len: usize, deltas: &[(u64, Fp)]) -> Vec<Value256> {
    let mut values = vec![encode_value(paper_a_optimized::fp_zero()); len];
    for &(idx, delta) in deltas {
        let slot = &mut values[idx as usize];
        *slot = encode_value(decode_value(*slot) + delta);
    }
    values
}

fn build_update_batch(len: u64, batch_size: usize, seed: u64) -> Vec<(u64, Fp)> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut deltas = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        deltas.push((rng.next_u64() % len, Fp::rand(&mut rng)));
    }
    deltas
}

fn build_block_update_deltas(s: usize, seed: u64) -> Vec<(usize, Fp)> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut deltas = Vec::with_capacity(s);
    for offset in 0..s {
        deltas.push((offset, Fp::rand(&mut rng)));
    }
    deltas
}

fn block_update_to_point_deltas(
    params: &CodedArrayParams,
    block: u64,
    deltas: &[(usize, Fp)],
) -> Vec<(u64, Fp)> {
    let pos = paper_a_optimized::unauth::semantic_block_position(params, block);
    let seg_base = pos.segment as u64 * params.segment_len as u64;
    deltas
        .iter()
        .map(|(offset_in_block, delta)| {
            let logical_offset = pos.block_in_segment * params.s + offset_in_block;
            (seg_base + logical_offset as u64, *delta)
        })
        .collect()
}

fn build_full_store(params: &CodedArrayParams, len: u64, deltas: &[(u64, Fp)]) -> UnauthStore {
    let mut store = UnauthStore::new(params.clone());
    store.extend(len).expect("extend full store");
    store
        .apply_deltas(deltas, DeltaMode::Aggregate)
        .expect("seed full store");
    store
}

fn build_single_claim_store(
    params: &CodedArrayParams,
    len: u64,
    deltas: &[(u64, Fp)],
) -> UnauthStore {
    let claim = ClaimBundle::exact_for_registry_index(params, 0)
        .expect("exact claim should exist for registry index");
    let mut store = UnauthStore::new_with_claims(params.clone(), vec![claim]);
    store.extend(len).expect("extend single-claim store");
    store
        .apply_deltas(deltas, DeltaMode::Aggregate)
        .expect("seed single-claim store");
    store
}

fn build_local_store_templates(
    params: &CodedArrayParams,
    len: u64,
    deltas: &[(u64, Fp)],
) -> Vec<UnauthStore> {
    let mut stores = Vec::with_capacity(params.shard_registry.len());
    for registry_index in 0..params.shard_registry.len() {
        let claim = ClaimBundle::exact_for_registry_index(params, registry_index)
            .expect("exact claim should exist for registry index");
        let mut store = UnauthStore::new_with_claims(params.clone(), vec![claim]);
        store.extend(len).expect("extend claim-bound store");
        store
            .apply_deltas(deltas, DeltaMode::Aggregate)
            .expect("seed claim-bound store");
        stores.push(store);
    }
    stores
}

fn wrap_local_servers(stores: &[UnauthStore]) -> Vec<LocalServer> {
    stores
        .iter()
        .cloned()
        .enumerate()
        .map(|(registry_index, store)| LocalServer {
            party_id: registry_index as u64,
            store: SharedStore::new(store),
        })
        .collect()
}

fn build_client(
    params: &CodedArrayParams,
    len: u64,
    store_templates: &[UnauthStore],
) -> (UnauthClient, Vec<LocalServer>) {
    let servers = wrap_local_servers(store_templates);
    let client = CoreUnauthClient::new(
        LocalTransport,
        params.clone(),
        servers[0].clone(),
        servers.clone(),
        len,
    );
    (client, servers)
}

fn assert_store_matches_expected_subset(store: &UnauthStore, expected: &UnauthStore) {
    assert_eq!(store.len(), expected.len());
    assert_eq!(store.params.s, expected.params.s);
    let store_shards = store.coded.as_nested_shards_for_tests(&store.params);
    let expected_shards = expected.coded.as_nested_shards_for_tests(&expected.params);
    for (key, segments) in &store_shards {
        let expected_segments = expected_shards.get(key).expect("expected shard");
        assert_eq!(segments, expected_segments);
    }
    let store_rows = store.coded.as_nested_rows_for_tests(&store.params);
    let expected_rows = expected.coded.as_nested_rows_for_tests(&expected.params);
    for (key, segments) in &store_rows {
        let expected_segments = expected_rows.get(key).expect("expected row");
        assert_eq!(segments, expected_segments);
    }
}

fn modeled_update_comm(batch_size: usize, server_count: usize) -> UpdateCommStats {
    UpdateCommStats {
        server_count: server_count as u64,
        delta_count: batch_size as u64,
        modeled_upload_bytes: server_count as u64 * batch_size as u64 * (INDEX_BYTES + FIELD_BYTES),
        modeled_rpc_count: server_count as u64,
    }
}

fn updates_per_sec_mean(batch_size: usize, stats: &MetricStats) -> f64 {
    if stats.mean == 0 {
        return 0.0;
    }
    batch_size as f64 / (stats.mean as f64 / 1_000_000_000.0)
}

fn mode_name(mode: DeltaMode) -> &'static str {
    match mode {
        DeltaMode::Aggregate => "aggregate",
        DeltaMode::InterpolateFFT => "fft",
    }
}

fn modeled_block_update_comm(
    delta_count: usize,
    server_count: usize,
    block_update: bool,
) -> UpdateCommStats {
    let request_count = server_count as u64;
    let modeled_upload_bytes = if block_update {
        server_count as u64 * (INDEX_BYTES + delta_count as u64 * (INDEX_BYTES + FIELD_BYTES))
    } else {
        server_count as u64 * delta_count as u64 * (INDEX_BYTES + FIELD_BYTES)
    };
    UpdateCommStats {
        server_count: server_count as u64,
        delta_count: delta_count as u64,
        modeled_upload_bytes,
        modeled_rpc_count: request_count,
    }
}

fn modeled_practical_stripe_comm(
    baseline_k: usize,
    codeword_count: usize,
    shard_count: usize,
) -> (u64, u64, u64) {
    let download_bytes = baseline_k as u64 * codeword_count as u64 * FIELD_BYTES;
    let upload_bytes = shard_count as u64 * codeword_count as u64 * FIELD_BYTES;
    let servers_contacted = shard_count as u64;
    (download_bytes, upload_bytes, servers_contacted)
}

fn apply_point_deltas_to_values(values: &mut [Value256], deltas: &[(u64, Fp)]) {
    for &(idx, delta) in deltas {
        let slot = &mut values[idx as usize];
        *slot = encode_value(decode_value(*slot) + delta);
    }
}

fn apply_block_deltas_to_values(
    params: &CodedArrayParams,
    block: u64,
    values: &mut [Value256],
    deltas: &[(usize, Fp)],
) {
    let point_deltas = block_update_to_point_deltas(params, block, deltas);
    apply_point_deltas_to_values(values, &point_deltas);
}

fn bench_practical_stripe_update<F>(
    trials: usize,
    params: &CodedArrayParams,
    seed_values: &[Value256],
    mut apply: F,
) -> MetricStats
where
    F: FnMut(&CodedArrayParams, &mut [Value256]),
{
    let rs_layout = fair_layout_for_params(params, params.segment_len);
    let helper_indices = default_helper_indices(&rs_layout);
    let rs_segment = encode_segment_from_values(&rs_layout, seed_values);
    let decoder = rs_layout.decoder(&helper_indices);
    let target_template =
        vec![vec![paper_a_optimized::fp_zero(); rs_layout.codeword_count]; rs_layout.n];

    let mut samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        let mut target_servers = target_template.clone();
        let start = Instant::now();
        let mut decoded = decoder.decode_full_segment(&rs_layout, &rs_segment);
        apply(params, &mut decoded);
        let restriped = encode_segment_from_values(&rs_layout, &decoded);
        for (target, source) in target_servers
            .iter_mut()
            .zip(restriped.shard_vectors.iter())
        {
            target.copy_from_slice(source);
        }
        black_box(target_servers);
        samples.push(start.elapsed().as_nanos() as u64);
    }
    stats_from_samples(samples)
}

fn run_point_entry(
    segment_len: usize,
    s: usize,
    batch_size: usize,
    mode: DeltaMode,
    trials: usize,
) -> BenchEntry {
    let params = build_params(segment_len, s);
    let message_size = params.segment_len;
    let k = effective_k(&params);
    let derived_s = derived_s(&params);
    let logical_len = params.segment_len as u64;
    let seed_deltas = build_seed_deltas(
        logical_len,
        BENCH_SEED ^ segment_len as u64 ^ ((s as u64) << 32),
    );
    let update_deltas = build_update_batch(
        logical_len,
        batch_size,
        BENCH_SEED ^ 0x55AA55AAu64 ^ segment_len as u64 ^ ((s as u64) << 16),
    );

    let store_templates = build_local_store_templates(&params, logical_len, &seed_deltas);
    let mut expected = build_full_store(&params, logical_len, &seed_deltas);
    expected
        .apply_deltas(&update_deltas, mode)
        .expect("apply expected deltas");

    let update_ns = bench_metric(trials, || {
        let (mut client, servers) = build_client(&params, logical_len, &store_templates);
        client
            .write_batch(&update_deltas, mode)
            .expect("timed unauth update");
        for server in &servers {
            let store = server.store.borrow();
            assert_store_matches_expected_subset(&store, &expected);
        }
    });

    BenchEntry {
        segment_len: params.segment_len,
        message_size,
        n_grid: params.n_grid,
        s: params.s,
        k,
        derived_s,
        need: params.need,
        shard_count: params.shard_registry.len(),
        rows_per_shard: params.rows_per_shard,
        logical_len,
        layer: "encoded_state_algorithm_level".to_string(),
        operation: "point_update_batch".to_string(),
        variant: None,
        baseline_kind: None,
        comparison_scope: None,
        decode_scope: None,
        batch_size,
        update_mode: mode_name(mode).to_string(),
        block_slots_touched: None,
        point_count_equivalent: Some(batch_size),
        comparison_role: Some("headline".to_string()),
        download_bytes: None,
        upload_bytes: None,
        servers_contacted: None,
        requires_read_modify_write: None,
        updates_per_sec_mean: updates_per_sec_mean(batch_size, &update_ns),
        update_ns,
        update_comm: modeled_update_comm(batch_size, params.shard_registry.len()),
    }
}

fn run_block_pair(segment_len: usize, s: usize, trials: usize) -> Vec<BenchEntry> {
    let params = build_params(segment_len, s);
    let message_size = params.segment_len;
    let k = effective_k(&params);
    let derived_s = derived_s(&params);
    let logical_len = params.segment_len as u64;
    let seed_deltas = build_seed_deltas(
        logical_len,
        BENCH_SEED ^ 0xABCD1234u64 ^ segment_len as u64 ^ ((s as u64) << 32),
    );
    let block_deltas = build_block_update_deltas(
        params.s,
        BENCH_SEED ^ 0x13572468u64 ^ segment_len as u64 ^ ((s as u64) << 16),
    );
    let point_deltas = block_update_to_point_deltas(&params, 0, &block_deltas);

    let base_store = build_single_claim_store(&params, logical_len, &seed_deltas);

    let mut expected = base_store.clone();
    expected
        .apply_deltas(&point_deltas, DeltaMode::Aggregate)
        .expect("apply expected block deltas");

    let mut structured_check = base_store.clone();
    structured_check
        .apply_block_deltas(0, &block_deltas, DeltaMode::Aggregate)
        .expect("apply structured block deltas");
    assert_store_matches_expected_subset(&structured_check, &expected);

    let mut structured_trials = vec![base_store.clone(); trials];
    let structured_ns = bench_mutating_stores(&mut structured_trials, |store| {
        store
            .apply_block_deltas(0, &block_deltas, DeltaMode::Aggregate)
            .expect("timed aligned block update");
    });
    for store in &structured_trials {
        assert_store_matches_expected_subset(store, &expected);
    }

    let mut point_trials = vec![base_store; trials];
    let point_ns = bench_mutating_stores(&mut point_trials, |store| {
        store
            .apply_deltas(&point_deltas, DeltaMode::Aggregate)
            .expect("timed batched point decomposition update");
    });
    for store in &point_trials {
        assert_store_matches_expected_subset(store, &expected);
    }

    let make_entry =
        |operation: &str, comparison_role: &str, update_ns: MetricStats, block_update: bool| {
            BenchEntry {
                segment_len: params.segment_len,
                message_size,
                n_grid: params.n_grid,
                s: params.s,
                k,
                derived_s,
                need: params.need,
                shard_count: 1,
                rows_per_shard: params.row_label_indices_for_registry_index(0).len(),
                logical_len,
                layer: "encoded_state_algorithm_level".to_string(),
                operation: operation.to_string(),
                variant: None,
                baseline_kind: None,
                comparison_scope: None,
                decode_scope: None,
                batch_size: params.s,
                update_mode: "aggregate".to_string(),
                block_slots_touched: Some(params.s),
                point_count_equivalent: Some(params.s),
                comparison_role: Some(comparison_role.to_string()),
                download_bytes: None,
                upload_bytes: None,
                servers_contacted: None,
                requires_read_modify_write: None,
                updates_per_sec_mean: updates_per_sec_mean(params.s, &update_ns),
                update_ns,
                update_comm: modeled_block_update_comm(params.s, 1, block_update),
            }
        };

    vec![
        make_entry(
            "aligned_block_as_batch_points",
            "batch_point_baseline",
            point_ns,
            false,
        ),
        make_entry(
            "aligned_block_update_structured",
            "structured_headline",
            structured_ns,
            true,
        ),
    ]
}

fn run_paper_point_update_pair(segment_len: usize, k: usize, trials: usize) -> Vec<BenchEntry> {
    let params = build_params_from_k(segment_len, k);
    let message_size = params.segment_len;
    let derived_s = derived_s(&params);
    let logical_len = params.segment_len as u64;
    let seed_deltas = build_seed_deltas(
        logical_len,
        BENCH_SEED ^ 0x1111AAAAu64 ^ segment_len as u64 ^ ((k as u64) << 20),
    );
    let update_deltas = build_update_batch(
        logical_len,
        1,
        BENCH_SEED ^ 0x2222BBBBu64 ^ segment_len as u64 ^ ((k as u64) << 24),
    );
    let seed_values = build_logical_values(params.segment_len, &seed_deltas);

    let store_templates = build_local_store_templates(&params, logical_len, &seed_deltas);
    let mut expected = build_full_store(&params, logical_len, &seed_deltas);
    expected
        .apply_deltas(&update_deltas, DeltaMode::Aggregate)
        .expect("apply expected point deltas");

    let impl_update_ns = bench_client_workflow(
        trials,
        &params,
        logical_len,
        &store_templates,
        &expected,
        |client| {
            client
                .write_batch(&update_deltas, DeltaMode::Aggregate)
                .expect("timed implemented point update");
        },
    );
    let impl_updates_per_sec = updates_per_sec_mean(1, &impl_update_ns);

    let baseline_update_ns =
        bench_practical_stripe_update(trials, &params, &seed_values, |_, values| {
            apply_point_deltas_to_values(values, &update_deltas);
        });
    let baseline_updates_per_sec = updates_per_sec_mean(1, &baseline_update_ns);

    let rs_layout = fair_layout_for_params(&params, params.segment_len);
    let (baseline_download, baseline_upload, baseline_servers) =
        modeled_practical_stripe_comm(rs_layout.k, rs_layout.codeword_count, rs_layout.n);
    let implemented_upload =
        modeled_update_comm(update_deltas.len(), params.shard_registry.len()).modeled_upload_bytes;

    vec![
        BenchEntry {
            segment_len: params.segment_len,
            message_size,
            n_grid: params.n_grid,
            s: params.s,
            k: params.need,
            derived_s,
            need: params.need,
            shard_count: params.shard_registry.len(),
            rows_per_shard: params.rows_per_shard,
            logical_len,
            layer: "system_level".to_string(),
            operation: "point_update".to_string(),
            variant: Some("implemented_point_update".to_string()),
            baseline_kind: Some("rs_stripe_practical".to_string()),
            comparison_scope: Some("system_workflow".to_string()),
            decode_scope: Some("semantic_point".to_string()),
            batch_size: 1,
            update_mode: "aggregate".to_string(),
            block_slots_touched: None,
            point_count_equivalent: Some(1),
            comparison_role: Some("headline".to_string()),
            download_bytes: Some(0),
            upload_bytes: Some(implemented_upload),
            servers_contacted: Some(params.shard_registry.len() as u64),
            requires_read_modify_write: Some(false),
            update_ns: impl_update_ns,
            updates_per_sec_mean: impl_updates_per_sec,
            update_comm: modeled_update_comm(update_deltas.len(), params.shard_registry.len()),
        },
        BenchEntry {
            segment_len: params.segment_len,
            message_size,
            n_grid: params.n_grid,
            s: params.s,
            k: params.need,
            derived_s,
            need: params.need,
            shard_count: params.shard_registry.len(),
            rows_per_shard: params.rows_per_shard,
            logical_len,
            layer: "system_level".to_string(),
            operation: "point_update".to_string(),
            variant: Some("rs_stripe_practical_point_update".to_string()),
            baseline_kind: Some("rs_stripe_practical".to_string()),
            comparison_scope: Some("system_workflow".to_string()),
            decode_scope: Some("full_stripe".to_string()),
            batch_size: 1,
            update_mode: "stripe_rmw".to_string(),
            block_slots_touched: None,
            point_count_equivalent: Some(1),
            comparison_role: Some("headline".to_string()),
            download_bytes: Some(baseline_download),
            upload_bytes: Some(baseline_upload),
            servers_contacted: Some(baseline_servers),
            requires_read_modify_write: Some(true),
            update_ns: baseline_update_ns,
            updates_per_sec_mean: baseline_updates_per_sec,
            update_comm: UpdateCommStats {
                server_count: baseline_servers,
                delta_count: 1,
                modeled_upload_bytes: baseline_upload,
                modeled_rpc_count: baseline_servers,
            },
        },
    ]
}

fn run_paper_block_update_pair(segment_len: usize, k: usize, trials: usize) -> Vec<BenchEntry> {
    let params = build_params_from_k(segment_len, k);
    let message_size = params.segment_len;
    let derived_s = derived_s(&params);
    let logical_len = params.segment_len as u64;
    let seed_deltas = build_seed_deltas(
        logical_len,
        BENCH_SEED ^ 0x3333CCCCu64 ^ segment_len as u64 ^ ((k as u64) << 20),
    );
    let block_deltas = build_block_update_deltas(
        params.s,
        BENCH_SEED ^ 0x4444DDDDu64 ^ segment_len as u64 ^ ((k as u64) << 24),
    );
    let seed_values = build_logical_values(params.segment_len, &seed_deltas);

    let store_templates = build_local_store_templates(&params, logical_len, &seed_deltas);
    let mut expected = build_full_store(&params, logical_len, &seed_deltas);
    expected
        .apply_block_deltas(0, &block_deltas, DeltaMode::Aggregate)
        .expect("apply expected block deltas");

    let impl_update_ns = bench_client_workflow(
        trials,
        &params,
        logical_len,
        &store_templates,
        &expected,
        |client| {
            client
                .write_block(0, &block_deltas, DeltaMode::Aggregate)
                .expect("timed implemented block update");
        },
    );
    let impl_updates_per_sec = updates_per_sec_mean(params.s, &impl_update_ns);

    let baseline_update_ns =
        bench_practical_stripe_update(trials, &params, &seed_values, |params, values| {
            apply_block_deltas_to_values(params, 0, values, &block_deltas);
        });
    let baseline_updates_per_sec = updates_per_sec_mean(params.s, &baseline_update_ns);

    let rs_layout = fair_layout_for_params(&params, params.segment_len);
    let (baseline_download, baseline_upload, baseline_servers) =
        modeled_practical_stripe_comm(rs_layout.k, rs_layout.codeword_count, rs_layout.n);
    let implemented_upload =
        modeled_block_update_comm(block_deltas.len(), params.shard_registry.len(), true)
            .modeled_upload_bytes;

    vec![
        BenchEntry {
            segment_len: params.segment_len,
            message_size,
            n_grid: params.n_grid,
            s: params.s,
            k: params.need,
            derived_s,
            need: params.need,
            shard_count: params.shard_registry.len(),
            rows_per_shard: params.rows_per_shard,
            logical_len,
            layer: "system_level".to_string(),
            operation: "aligned_block_update".to_string(),
            variant: Some("implemented_block_update".to_string()),
            baseline_kind: Some("rs_stripe_practical".to_string()),
            comparison_scope: Some("system_workflow".to_string()),
            decode_scope: Some("semantic_aligned_block".to_string()),
            batch_size: params.s,
            update_mode: "aggregate".to_string(),
            block_slots_touched: Some(params.s),
            point_count_equivalent: Some(params.s),
            comparison_role: Some("headline".to_string()),
            download_bytes: Some(0),
            upload_bytes: Some(implemented_upload),
            servers_contacted: Some(params.shard_registry.len() as u64),
            requires_read_modify_write: Some(false),
            update_ns: impl_update_ns,
            updates_per_sec_mean: impl_updates_per_sec,
            update_comm: modeled_block_update_comm(
                block_deltas.len(),
                params.shard_registry.len(),
                true,
            ),
        },
        BenchEntry {
            segment_len: params.segment_len,
            message_size,
            n_grid: params.n_grid,
            s: params.s,
            k: params.need,
            derived_s,
            need: params.need,
            shard_count: params.shard_registry.len(),
            rows_per_shard: params.rows_per_shard,
            logical_len,
            layer: "system_level".to_string(),
            operation: "aligned_block_update".to_string(),
            variant: Some("rs_stripe_practical_block_update".to_string()),
            baseline_kind: Some("rs_stripe_practical".to_string()),
            comparison_scope: Some("system_workflow".to_string()),
            decode_scope: Some("full_stripe".to_string()),
            batch_size: params.s,
            update_mode: "stripe_rmw".to_string(),
            block_slots_touched: Some(params.s),
            point_count_equivalent: Some(params.s),
            comparison_role: Some("headline".to_string()),
            download_bytes: Some(baseline_download),
            upload_bytes: Some(baseline_upload),
            servers_contacted: Some(baseline_servers),
            requires_read_modify_write: Some(true),
            update_ns: baseline_update_ns,
            updates_per_sec_mean: baseline_updates_per_sec,
            update_comm: UpdateCommStats {
                server_count: baseline_servers,
                delta_count: params.s as u64,
                modeled_upload_bytes: baseline_upload,
                modeled_rpc_count: baseline_servers,
            },
        },
    ]
}

pub fn build_point_results(
    trials: usize,
    s_values: Vec<usize>,
    segment_lens: Vec<usize>,
    batch_sizes: Vec<usize>,
) -> BenchResults {
    let mut entries = Vec::new();
    for &segment_len in &segment_lens {
        for &s in &s_values {
            if !s.is_power_of_two() {
                continue;
            }
            if s > segment_len {
                continue;
            }
            for &batch_size in &batch_sizes {
                entries.push(run_point_entry(
                    segment_len,
                    s,
                    batch_size,
                    DeltaMode::Aggregate,
                    trials,
                ));
                entries.push(run_point_entry(
                    segment_len,
                    s,
                    batch_size,
                    DeltaMode::InterpolateFFT,
                    trials,
                ));
            }
        }
    }

    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    BenchResults {
        meta: BenchMeta {
            seed: BENCH_SEED,
            trials,
            timestamp_unix_ms: timestamp_ms,
            git: git_sha(),
            notes: "unauth point-update suite over the public update path, covering generic batched point updates and aggregate versus FFT update modes".to_string(),
            k_values: Vec::new(),
            s_values,
            segment_lens,
            batch_sizes,
            claim_model: "one_claim_per_party".to_string(),
            claims_per_party: 1,
            effective_dimension_note: "k = need = n_col and derived_s = n_grid / k under the current one-claim-per-party artifact model".to_string(),
        },
        entries,
    }
}

pub fn build_block_results(
    trials: usize,
    s_values: Vec<usize>,
    segment_lens: Vec<usize>,
) -> BenchResults {
    let mut entries = Vec::new();
    for &segment_len in &segment_lens {
        for &s in &s_values {
            if !s.is_power_of_two() {
                continue;
            }
            if s > segment_len {
                continue;
            }
            let params = build_params(segment_len, s);
            let block_deltas = build_block_update_deltas(
                params.s,
                BENCH_SEED ^ 0x13572468u64 ^ segment_len as u64 ^ ((s as u64) << 16),
            );
            let point_deltas = block_update_to_point_deltas(&params, 0, &block_deltas);
            if point_deltas
                .iter()
                .any(|(idx, _)| *idx >= params.segment_len as u64)
            {
                continue;
            }
            entries.extend(run_block_pair(segment_len, s, trials));
        }
    }

    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    BenchResults {
        meta: BenchMeta {
            seed: BENCH_SEED,
            trials,
            timestamp_unix_ms: timestamp_ms,
            git: git_sha(),
            notes: "unauth block-update suite on one claimed local store, comparing one structured aligned block update against decomposing the same logical block write into one batched point-update apply while excluding setup and validation from the timed region".to_string(),
            k_values: Vec::new(),
            s_values,
            segment_lens,
            batch_sizes: Vec::new(),
            claim_model: "one_claim_per_party".to_string(),
            claims_per_party: 1,
            effective_dimension_note: "k = need = n_col and derived_s = n_grid / k under the current one-claim-per-party artifact model".to_string(),
        },
        entries,
    }
}

pub fn build_paper_update_results(
    trials: usize,
    k_values: Vec<usize>,
    segment_lens: Vec<usize>,
) -> BenchResults {
    let mut entries = Vec::new();
    for &segment_len in &segment_lens {
        for &k in &k_values {
            let n_grid = segment_len;
            if !k.is_power_of_two() || k == 0 || k > n_grid || n_grid % k != 0 {
                continue;
            }
            entries.extend(run_paper_point_update_pair(segment_len, k, trials));
            entries.extend(run_paper_block_update_pair(segment_len, k, trials));
        }
    }

    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    BenchResults {
        meta: BenchMeta {
            seed: BENCH_SEED,
            trials,
            timestamp_unix_ms: timestamp_ms,
            git: git_sha(),
            notes: "paper-facing unauth update suite for Figure M4, comparing semantic point and aligned-block updates against the practical stripe read/decode/modify/re-encode/send baseline".to_string(),
            k_values,
            s_values: Vec::new(),
            segment_lens,
            batch_sizes: Vec::new(),
            claim_model: "one_claim_per_party".to_string(),
            claims_per_party: 1,
            effective_dimension_note: "paper-facing update figure uses a segment_len × k matrix with derived_s = n_grid / k under the current one-claim-per-party artifact model".to_string(),
        },
        entries,
    }
}

pub fn write_results(path_name: &str, results: &BenchResults) {
    let out_dir = Path::new("results");
    let _ = fs::create_dir_all(out_dir);
    let path = out_dir.join(path_name);
    let data = serde_json::to_vec_pretty(results).expect("serialize update bench results");
    fs::write(path, data).expect("write update bench results");
}
