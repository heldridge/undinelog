//! Benchmark-only encoding baselines for the unauthenticated array.
//!
//! This suite compares the optimized coset/decomposition encoding against two
//! local comparators that are intentionally kept out of `src/`:
//! - striped and single-polynomial univariate segment codes with cached
//!   Lagrange updates;
//! - a generic bivariate code with row/column-shaped claims and naive/cached
//!   product updates.

mod support;

use crate::support::generic_bivariate_baseline::{BivariateUpdateMode, GenericBivariateBaseline};
use crate::support::rs_striped_baseline::{encode_segment_from_values, fair_layout_for_params};
use crate::support::state_metrics::{StoreFootprint, measure_store_footprint};
use crate::support::univariate_segment_baseline::{
    CompactLagrangeCache, UnivariateSegmentBaseline, UnivariateUpdateMode, encode_field_values,
    eval_on_domain, padded_domain_values,
};
use paper_a_optimized::coded_array::rs::{barycentric_eval, precompute_barycentric};
use paper_a_optimized::unauth::{
    ClaimBundle, DeltaMode, UnauthStore, block_grid_indices, build_segment_double_params,
};
use paper_a_optimized::{
    CodedArrayConfig, CodedArrayParams, Fp, Value256, build_coded_params,
    build_shard_registry_deterministic, decode_value, encode_value,
};
use serde::Serialize;
use std::collections::BTreeSet;
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const FIELD_BYTES: u64 = 32;
const DEFAULT_TRIALS: usize = 3;
const DEFAULT_SEGMENT_LENS: &[usize] = &[128];
const DEFAULT_K_VALUES: &[usize] = &[8];
const UPDATE_COUNT: usize = 32;
const POINT_INDEX: usize = 17;

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
    trials: usize,
    segment_lens: Vec<usize>,
    k_values: Vec<usize>,
    update_count: usize,
    timestamp_unix_ms: u64,
    git: Option<String>,
    command: Vec<String>,
    notes: String,
}

#[derive(Clone, Debug, Serialize)]
struct BenchEntry {
    model: String,
    operation: String,
    segment_len: usize,
    source_k: usize,
    target_k: Option<usize>,
    source_shard_len: usize,
    target_shard_len: Option<usize>,
    update_mode: Option<String>,
    latency_ns: MetricStats,
    maintained_bytes_per_party: u64,
    cache_bytes_per_party: u64,
    helper_download_bytes: u64,
    notes: String,
}

#[derive(Clone, Debug, Serialize)]
struct BenchResults {
    meta: BenchMeta,
    entries: Vec<BenchEntry>,
}

#[derive(Clone, Debug)]
struct StripedUnivariateCachedUpdate {
    k: usize,
    source_points: Vec<Fp>,
    shard_point: Fp,
    shard_values: Vec<Fp>,
    basis_cache: CompactLagrangeCache,
}

impl StripedUnivariateCachedUpdate {
    fn new(params: &CodedArrayParams, logical_values: &[Value256]) -> Self {
        let layout = fair_layout_for_params(params, logical_values.len());
        let segment = encode_segment_from_values(&layout, logical_values);
        let shard_point = layout.shard_points[0];
        let source_weights = precompute_barycentric(&layout.source_points).w;
        let basis_cache = CompactLagrangeCache::new(
            &layout.source_points,
            &source_weights,
            std::slice::from_ref(&shard_point),
        );
        Self {
            k: layout.k,
            source_points: layout.source_points,
            shard_point,
            shard_values: segment.shard_vectors[0].clone(),
            basis_cache,
        }
    }

    fn apply_updates(&mut self, updates: &[(u64, Fp)]) {
        for &(idx, delta) in updates {
            let idx = idx as usize;
            let codeword_idx = idx / self.k;
            let source_idx = idx % self.k;
            let basis = self.basis_cache.lagrange_basis(
                &self.source_points,
                std::slice::from_ref(&self.shard_point),
                0,
                source_idx,
            );
            self.shard_values[codeword_idx] += delta * basis;
        }
    }

    fn maintained_bytes_per_party(&self) -> u64 {
        self.shard_values.len() as u64 * FIELD_BYTES
    }

    fn cache_bytes_per_party(&self) -> u64 {
        self.basis_cache.field_element_count() as u64 * FIELD_BYTES
    }
}

fn zero_stats() -> MetricStats {
    MetricStats {
        mean: 0,
        p50: 0,
        p95: 0,
        trials: 0,
        samples: Vec::new(),
    }
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

fn bench_with_setup<Model, Setup, Work, Ret>(
    trials: usize,
    mut setup: Setup,
    mut work: Work,
) -> MetricStats
where
    Setup: FnMut() -> Model,
    Work: FnMut(&mut Model) -> Ret,
{
    let mut samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        let mut model = setup();
        let start = Instant::now();
        let ret = work(&mut model);
        black_box(ret);
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
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn parse_usize_list(var: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(var)
        .ok()
        .map(|raw| {
            raw.split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

fn parse_bool(var: &str) -> bool {
    std::env::var(var)
        .ok()
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            matches!(value.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn output_path() -> String {
    std::env::var("BENCH_UNAUTH_ENCODING_BASELINES_OUT")
        .unwrap_or_else(|_| "results/bench_unauth_encoding_baselines.json".to_string())
}

fn build_params_for_k(
    segment_len: usize,
    k: usize,
    shard_count: usize,
) -> Option<CodedArrayParams> {
    let n_grid = segment_len;
    if k == 0 || n_grid % k != 0 {
        return None;
    }
    let s = n_grid / k;
    Some(build_coded_params(&CodedArrayConfig {
        segment_len,
        s,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count, s, n_grid),
        party_registry_indices: (0..shard_count).map(|idx| vec![idx]).collect(),
        seed: 91,
    }))
}

fn initial_values(segment_len: usize) -> Vec<Fp> {
    (0..segment_len)
        .map(|idx| Fp::from((idx as u64 + 1) * 17))
        .collect()
}

fn update_batch(segment_len: usize) -> Vec<(u64, Fp)> {
    (0..UPDATE_COUNT)
        .map(|round| {
            let idx = (POINT_INDEX + round * 13) % segment_len;
            (idx as u64, Fp::from((round as u64 + 5) * 19))
        })
        .collect()
}

fn apply_updates_to_values(values: &mut [Fp], updates: &[(u64, Fp)]) {
    for &(idx, delta) in updates {
        values[idx as usize] += delta;
    }
}

fn segment_double_grid_values(params: &CodedArrayParams, source_segment: usize) -> Vec<Value256> {
    (0..params.n_grid)
        .map(|idx| {
            let raw = ((source_segment as u64 + 1) * 10_000) + idx as u64 + 1;
            encode_value(Fp::from(raw * 17))
        })
        .collect()
}

fn segment_double_target_values(
    params: &CodedArrayParams,
    target_params: &CodedArrayParams,
) -> (Vec<Value256>, Vec<Value256>, Vec<Value256>) {
    let left = segment_double_grid_values(params, 0);
    let right = segment_double_grid_values(params, 1);
    let mut target = Vec::with_capacity(target_params.n_grid);
    target.extend_from_slice(&left);
    target.extend_from_slice(&right);
    target.resize(target_params.n_grid, encode_value(Fp::from(0u64)));
    (left, right, target)
}

fn decode_univariate_logical_values(
    model: &UnivariateSegmentBaseline,
    logical_len: usize,
    helper_count: usize,
) -> Vec<Value256> {
    assert!(helper_count <= model.parties.len());
    let mut points = Vec::with_capacity(helper_count * model.parties[0].values.len());
    let mut values = Vec::with_capacity(points.capacity());
    for party in model.parties.iter().take(helper_count) {
        points.extend_from_slice(&party.eval_points);
        values.extend_from_slice(&party.values);
    }
    assert!(points.len() >= model.n_grid);
    let points = &points[..model.n_grid];
    let values = &values[..model.n_grid];
    let weights = precompute_barycentric(points);
    model
        .domain
        .iter()
        .take(logical_len)
        .map(|&point| encode_value(barycentric_eval(&weights, values, point)))
        .collect()
}

fn target_univariate_party_values(
    params: &CodedArrayParams,
    target_values: &[Value256],
    registry_index: usize,
) -> Vec<Fp> {
    let target =
        crate::support::univariate_segment_baseline::target_party_template(params, registry_index);
    let domain_values = padded_domain_values(params, target_values);
    target
        .eval_points
        .iter()
        .copied()
        .map(|point| eval_on_domain(&params.x, &domain_values, point))
        .collect()
}

fn build_segment_double_claim_store(
    params: &CodedArrayParams,
    left: &[Value256],
    right: &[Value256],
) -> UnauthStore {
    let claim = ClaimBundle::exact_for_registry_index(params, 0).expect("claim for registry 0");
    let mut store = UnauthStore::new_with_claims(params.clone(), vec![claim]);
    store
        .extend_with_segments((2 * params.n_grid) as u64, |seg| {
            if seg == 0 {
                left.to_vec()
            } else {
                right.to_vec()
            }
        })
        .expect("extend segment-double source store");
    store
}

fn aligned_offsets(params: &CodedArrayParams) -> Vec<usize> {
    let offsets = block_grid_indices(params, 0);
    assert!(
        offsets.iter().all(|&offset| offset < params.segment_len),
        "encoding baseline aligned-open benchmark expects block 0 to contain only active indices"
    );
    offsets
}

fn touched_striped_codewords(k: usize, offsets: &[usize]) -> usize {
    offsets
        .iter()
        .map(|idx| idx / k)
        .collect::<BTreeSet<_>>()
        .len()
}

fn touched_bivariate_w_queries(params: &CodedArrayParams, offsets: &[usize]) -> usize {
    offsets
        .iter()
        .map(|idx| idx % params.s)
        .collect::<BTreeSet<_>>()
        .len()
}

fn univariate_read_block_from_parties(
    model: &UnivariateSegmentBaseline,
    offsets: &[usize],
    helper_count: usize,
) -> Vec<Fp> {
    assert!(helper_count <= model.parties.len());
    let mut points = Vec::with_capacity(helper_count * model.parties[0].values.len());
    let mut values = Vec::with_capacity(points.capacity());
    for party in model.parties.iter().take(helper_count) {
        points.extend_from_slice(&party.eval_points);
        values.extend_from_slice(&party.values);
    }
    assert!(points.len() >= model.n_grid);
    let points = &points[..model.n_grid];
    let values = &values[..model.n_grid];
    let weights = precompute_barycentric(points);
    offsets
        .iter()
        .map(|&offset| barycentric_eval(&weights, values, model.domain[offset]))
        .collect()
}

fn generic_bivariate_read_offsets_from_claims(
    model: &GenericBivariateBaseline,
    offsets: &[usize],
    helper_count: usize,
) -> Vec<Fp> {
    assert!(helper_count <= model.claims.len());
    assert!(
        helper_count >= model.source_z_domain.len(),
        "generic bivariate block read needs enough claim-column evaluations"
    );
    let claims = &model.claims[..helper_count];
    let z_points = claims
        .iter()
        .map(|claim| claim.z_column)
        .collect::<Vec<_>>();
    let z_weights = precompute_barycentric(&z_points);
    let w_weights_by_claim = claims
        .iter()
        .map(|claim| precompute_barycentric(&claim.w_column_points))
        .collect::<Vec<_>>();
    let mut shares_by_w = vec![None; model.source_w_domain.len()];

    offsets
        .iter()
        .map(|&offset| {
            assert!(offset < model.segment_len);
            let source_w_len = model.source_w_domain.len();
            let source_z_idx = offset / source_w_len;
            let source_w_idx = offset % source_w_len;
            if shares_by_w[source_w_idx].is_none() {
                let target_w = model.source_w_domain[source_w_idx];
                let shares = claims
                    .iter()
                    .zip(w_weights_by_claim.iter())
                    .map(|(claim, w_weights)| {
                        barycentric_eval(&w_weights, &claim.column_values, target_w)
                    })
                    .collect::<Vec<_>>();
                shares_by_w[source_w_idx] = Some(shares);
            }
            let shares = shares_by_w[source_w_idx]
                .as_ref()
                .expect("bivariate W shares cached");
            barycentric_eval(&z_weights, shares, model.source_z_domain[source_z_idx])
        })
        .collect()
}

fn build_claim_store(params: &CodedArrayParams, logical_values: &[Value256]) -> UnauthStore {
    let claim = ClaimBundle::exact_for_registry_index(params, 0).expect("claim for registry 0");
    let mut store = UnauthStore::new_with_claims(params.clone(), vec![claim]);
    store
        .extend(params.segment_len as u64)
        .expect("extend claim store");
    let deltas = logical_values
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, value)| (idx as u64, decode_value(value)))
        .collect::<Vec<_>>();
    store
        .apply_deltas(&deltas, DeltaMode::Aggregate)
        .expect("seed claim store");
    store
}

fn build_empty_claim_store(params: &CodedArrayParams) -> UnauthStore {
    let claim = ClaimBundle::exact_for_registry_index(params, 0).expect("claim for registry 0");
    let mut store = UnauthStore::new_with_claims(params.clone(), vec![claim]);
    store
        .extend(params.segment_len as u64)
        .expect("extend empty claim store");
    store
}

fn build_full_store_with_claims(
    params: &CodedArrayParams,
    logical_values: &[Value256],
    claims: Vec<ClaimBundle>,
) -> UnauthStore {
    let mut store = UnauthStore::new_with_claims(params.clone(), claims);
    store
        .extend(params.segment_len as u64)
        .expect("extend store");
    let deltas = logical_values
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, value)| (idx as u64, decode_value(value)))
        .collect::<Vec<_>>();
    store
        .apply_deltas(&deltas, DeltaMode::Aggregate)
        .expect("seed store");
    store
}

fn build_empty_store_with_claims(
    params: &CodedArrayParams,
    claims: Vec<ClaimBundle>,
) -> UnauthStore {
    let mut store = UnauthStore::new_with_claims(params.clone(), claims);
    store
        .extend(params.segment_len as u64)
        .expect("extend empty store");
    store
}

fn all_claims(params: &CodedArrayParams) -> Vec<ClaimBundle> {
    (0..params.shard_registry.len())
        .map(|idx| ClaimBundle::exact_for_registry_index(params, idx).expect("claim"))
        .collect()
}

fn first_claims(params: &CodedArrayParams, count: usize) -> Vec<ClaimBundle> {
    (0..count)
        .map(|idx| ClaimBundle::exact_for_registry_index(params, idx).expect("claim"))
        .collect()
}

fn rows_per_claim(params: &CodedArrayParams) -> usize {
    usize::max(1, params.s.div_ceil(params.need))
}

fn algebraic_store_bytes(footprint: &StoreFootprint) -> u64 {
    footprint.primary_bytes + footprint.auxiliary_bytes
}

fn projected_univariate_maintained_bytes(params: &CodedArrayParams) -> u64 {
    params.s as u64 * FIELD_BYTES
}

fn projected_univariate_cache_bytes(params: &CodedArrayParams) -> u64 {
    (params.n_grid + params.s) as u64 * FIELD_BYTES
}

fn projected_striped_univariate_maintained_bytes(params: &CodedArrayParams) -> u64 {
    params.segment_len.div_ceil(params.need) as u64 * FIELD_BYTES
}

fn projected_striped_univariate_cache_bytes(params: &CodedArrayParams) -> u64 {
    (params.need + 1) as u64 * FIELD_BYTES
}

fn projected_generic_bivariate_maintained_bytes(params: &CodedArrayParams) -> u64 {
    (params.s + rows_per_claim(params) * params.need) as u64 * FIELD_BYTES
}

fn projected_generic_bivariate_cache_bytes(params: &CodedArrayParams) -> u64 {
    let rows_per_claim = rows_per_claim(params);
    (3 * params.need + 3 * params.s + rows_per_claim + 1) as u64 * FIELD_BYTES
}

fn validate_models(
    params: &CodedArrayParams,
    expected_values: &[Fp],
    updates: &[(u64, Fp)],
    helper_count: usize,
) {
    let logical_values = encode_field_values(&initial_values(params.segment_len));
    let mut univariate = UnivariateSegmentBaseline::new(
        params,
        &logical_values,
        helper_count,
        UnivariateUpdateMode::Cached,
    );
    univariate.apply_updates(updates);
    assert_eq!(
        univariate.read_point_from_parties(POINT_INDEX, helper_count),
        expected_values[POINT_INDEX]
    );

    let mut bivariate = GenericBivariateBaseline::new(
        params,
        &logical_values,
        helper_count,
        BivariateUpdateMode::CachedSeparated,
    );
    bivariate.apply_updates(updates);
    assert_eq!(
        bivariate.read_point_from_claims(POINT_INDEX, helper_count),
        expected_values[POINT_INDEX]
    );

    let mut full = UnauthStore::new(params.clone());
    full.extend(params.segment_len as u64)
        .expect("extend validation store");
    full.apply_deltas(
        &logical_values
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, value)| (idx as u64, decode_value(value)))
            .collect::<Vec<_>>(),
        DeltaMode::Aggregate,
    )
    .expect("seed validation store");
    full.apply_deltas(updates, DeltaMode::Aggregate)
        .expect("update validation store");
    assert_eq!(
        decode_value(full.read(POINT_INDEX as u64).expect("read validation")),
        expected_values[POINT_INDEX]
    );
}

fn validate_optimized_updates_from_zero(
    params: &CodedArrayParams,
    expected_values: &[Fp],
    updates: &[(u64, Fp)],
) {
    let mut full = UnauthStore::new(params.clone());
    full.extend(params.segment_len as u64)
        .expect("extend validation store");
    full.apply_deltas(updates, DeltaMode::Aggregate)
        .expect("update validation store");
    assert_eq!(
        decode_value(full.read(POINT_INDEX as u64).expect("read validation")),
        expected_values[POINT_INDEX]
    );
}

fn projected_baseline_entries(
    params: &CodedArrayParams,
    segment_len: usize,
    source_k: usize,
) -> Vec<BenchEntry> {
    [
        (
            "striped_univariate_cached",
            "cached",
            projected_striped_univariate_maintained_bytes(params),
            projected_striped_univariate_cache_bytes(params),
            "projected striped-RS cache footprint only; runtime skipped by large-segment mode",
        ),
        (
            "univariate_segment_cached",
            "cached",
            projected_univariate_maintained_bytes(params),
            projected_univariate_cache_bytes(params),
            "projected cache footprint only; runtime skipped by large-segment mode",
        ),
        (
            "generic_bivariate_cached",
            "cached_separated",
            projected_generic_bivariate_maintained_bytes(params),
            projected_generic_bivariate_cache_bytes(params),
            "projected separated-cache footprint only; runtime skipped by large-segment mode",
        ),
    ]
    .into_iter()
    .map(
        |(model, mode, maintained_bytes, cache_bytes, notes)| BenchEntry {
            model: model.to_string(),
            operation: "project_update_cache_footprint".to_string(),
            segment_len,
            source_k,
            target_k: None,
            source_shard_len: params.s,
            target_shard_len: None,
            update_mode: Some(mode.to_string()),
            latency_ns: zero_stats(),
            maintained_bytes_per_party: maintained_bytes,
            cache_bytes_per_party: cache_bytes,
            helper_download_bytes: 0,
            notes: notes.to_string(),
        },
    )
    .collect()
}

fn run_large_segment_entry(
    segment_len: usize,
    source_k: usize,
    trials: usize,
    skip_materialization: bool,
    measure_setup: bool,
) -> Vec<BenchEntry> {
    let target_k = source_k * 2;
    let shard_count = target_k * 2 + 2;
    let Some(params) = build_params_for_k(segment_len, source_k, shard_count) else {
        return Vec::new();
    };
    let Some(target_params) = build_params_for_k(segment_len, target_k, shard_count) else {
        return Vec::new();
    };
    if params.s < 2 {
        return Vec::new();
    }
    let updates = update_batch(segment_len);
    let mut expected = vec![Fp::from(0u64); segment_len];
    apply_updates_to_values(&mut expected, &updates);
    validate_optimized_updates_from_zero(&params, &expected, &updates);

    let mut entries = projected_baseline_entries(&params, segment_len, source_k);

    let optimized_model = build_empty_claim_store(&params);
    let optimized_footprint = measure_store_footprint(&optimized_model);
    let optimized_setup = if measure_setup {
        bench_with_setup(trials, || (), |_| build_empty_claim_store(&params))
    } else {
        zero_stats()
    };
    entries.push(BenchEntry {
        model: "coset_decomposition".to_string(),
        operation: "build_state_and_update_cache".to_string(),
        segment_len,
        source_k,
        target_k: None,
        source_shard_len: params.s,
        target_shard_len: None,
        update_mode: Some("coset_patch".to_string()),
        latency_ns: optimized_setup,
        maintained_bytes_per_party: algebraic_store_bytes(&optimized_footprint),
        cache_bytes_per_party: 0,
        helper_download_bytes: 0,
        notes: "measured large-segment empty-state extension for one advertised responsibility without a dense update cache"
            .to_string(),
    });

    let optimized_update = bench_with_setup(
        trials,
        || optimized_model.clone(),
        |model| {
            model
                .apply_deltas(&updates, DeltaMode::Aggregate)
                .expect("optimized updates")
        },
    );
    entries.push(BenchEntry {
        model: "coset_decomposition".to_string(),
        operation: "apply_public_point_update_batch".to_string(),
        segment_len,
        source_k,
        target_k: None,
        source_shard_len: params.s,
        target_shard_len: None,
        update_mode: Some("coset_patch".to_string()),
        latency_ns: optimized_update,
        maintained_bytes_per_party: algebraic_store_bytes(&optimized_footprint),
        cache_bytes_per_party: 0,
        helper_download_bytes: 0,
        notes: format!(
            "{} public point updates on one advertised responsibility",
            updates.len()
        ),
    });

    if skip_materialization {
        return entries;
    }

    let claims = vec![ClaimBundle::exact_for_registry_index(&params, 0).expect("source claim")];
    let mut claim_bound = build_empty_store_with_claims(&params, claims);
    claim_bound
        .apply_deltas(&updates, DeltaMode::Aggregate)
        .expect("update claim-bound source store");
    let mut full = build_empty_store_with_claims(&params, all_claims(&params));
    full.apply_deltas(&updates, DeltaMode::Aggregate)
        .expect("update full helper store");
    let mut full_target = full.clone();
    full_target
        .migrate(target_params.s, &[])
        .expect("full target migrate");
    let source_entry = params.shard_registry[0];
    let source_primary = claim_bound
        .coded
        .shard_segment(&params, &source_entry.key, 0)
        .expect("source primary column")
        .to_vec();
    let target_claim =
        ClaimBundle::exact_for_registry_index(&target_params, 0).expect("target child claim");
    let target_row_indices = target_claim
        .row_label_indices(&target_params)
        .expect("target child row labels");
    assert_eq!(target_row_indices.len(), rows_per_claim(&target_params));
    let target_row_labels = target_row_indices
        .iter()
        .map(|&idx| target_params.row_labels[idx])
        .collect::<Vec<_>>();
    let helper_download =
        (target_params.s + target_params.need * target_row_labels.len()) as u64 * FIELD_BYTES;
    let optimized_kappa_rescale = bench_with_setup(
        trials,
        || (full_target.clone(), source_primary.clone()),
        |(model, primary)| {
            let second_half = primary[target_params.s..].to_vec();
            assert_eq!(second_half.len(), target_params.s);
            for &target_row_label in &target_row_labels {
                let row = model
                    .repair_row_from_shards(0, target_row_label)
                    .expect("target row repair");
                assert_eq!(row.len(), target_params.need);
            }
        },
    );
    entries.push(BenchEntry {
        model: "coset_decomposition".to_string(),
        operation: "materialize_target_responsibility".to_string(),
        segment_len,
        source_k,
        target_k: Some(target_k),
        source_shard_len: params.s,
        target_shard_len: Some(target_params.s),
        update_mode: None,
        latency_ns: optimized_kappa_rescale,
        maintained_bytes_per_party: algebraic_store_bytes(&optimized_footprint),
        cache_bytes_per_party: 0,
        helper_download_bytes: helper_download,
        notes:
            "measured helper-assisted coset/decomposition kappa-rescale using partner material and all target-row repairs"
                .to_string(),
    });

    entries
}

fn run_entry(
    segment_len: usize,
    source_k: usize,
    trials: usize,
    large_segment_mode: bool,
    skip_materialization: bool,
    skip_naive_updates: bool,
    measure_setup: bool,
) -> Vec<BenchEntry> {
    if large_segment_mode {
        return run_large_segment_entry(
            segment_len,
            source_k,
            trials,
            skip_materialization,
            measure_setup,
        );
    }

    let target_k = source_k * 2;
    let shard_count = target_k * 2 + 2;
    let Some(params) = build_params_for_k(segment_len, source_k, shard_count) else {
        return Vec::new();
    };
    let Some(target_params) = build_params_for_k(segment_len, target_k, shard_count) else {
        return Vec::new();
    };
    let segment_double_params =
        build_segment_double_params(&params).expect("segment-double target params");
    if params.s < 2 {
        return Vec::new();
    }
    let logical_values = encode_field_values(&initial_values(segment_len));
    let updates = update_batch(segment_len);
    let mut expected = initial_values(segment_len);
    apply_updates_to_values(&mut expected, &updates);
    validate_models(&params, &expected, &updates, source_k);

    let striped_cached_model = StripedUnivariateCachedUpdate::new(&params, &logical_values);
    let univariate_cached_model = UnivariateSegmentBaseline::new(
        &params,
        &logical_values,
        source_k,
        UnivariateUpdateMode::Cached,
    );
    let bivariate_cached_model = GenericBivariateBaseline::new(
        &params,
        &logical_values,
        source_k,
        BivariateUpdateMode::CachedSeparated,
    );
    let univariate_cached_update_model =
        UnivariateSegmentBaseline::new(&params, &logical_values, 1, UnivariateUpdateMode::Cached);
    let bivariate_cached_update_model = GenericBivariateBaseline::new(
        &params,
        &logical_values,
        1,
        BivariateUpdateMode::CachedSeparated,
    );
    let optimized_model = build_claim_store(&params, &logical_values);
    let optimized_read_model =
        build_full_store_with_claims(&params, &logical_values, first_claims(&params, source_k));
    let optimized_footprint = measure_store_footprint(&optimized_model);
    let striped_layout = fair_layout_for_params(&params, logical_values.len());
    let striped_segment = encode_segment_from_values(&striped_layout, &logical_values);
    let striped_decoder = striped_layout.decoder(&(0..source_k).collect::<Vec<_>>());
    let block_offsets = aligned_offsets(&params);
    let expected_block = block_offsets
        .iter()
        .map(|&offset| logical_values[offset])
        .collect::<Vec<_>>();

    let mut entries = Vec::new();
    let setup_cases = [
        (
            "striped_univariate_cached",
            "cached",
            striped_cached_model.maintained_bytes_per_party(),
            striped_cached_model.cache_bytes_per_party(),
            if measure_setup {
                bench_with_setup(
                    trials,
                    || (),
                    |_| StripedUnivariateCachedUpdate::new(&params, &logical_values),
                )
            } else {
                zero_stats()
            },
            "builds one shard vector and compact per-helper Lagrange update cache for degree-k stripes",
        ),
        (
            "univariate_segment_cached",
            "cached",
            univariate_cached_update_model.maintained_bytes_per_party(),
            univariate_cached_update_model.cache_bytes_per_party(),
            if measure_setup {
                bench_with_setup(
                    trials,
                    || (),
                    |_| {
                        UnivariateSegmentBaseline::new(
                            &params,
                            &logical_values,
                            1,
                            UnivariateUpdateMode::Cached,
                        )
                    },
                )
            } else {
                zero_stats()
            },
            "builds one maintained slice and compact per-helper Lagrange update cache",
        ),
        (
            "generic_bivariate_cached",
            "cached_separated",
            bivariate_cached_update_model.maintained_bytes_per_claim(),
            bivariate_cached_update_model.cache_bytes_per_claim(),
            if measure_setup {
                bench_with_setup(
                    trials,
                    || (),
                    |_| {
                        GenericBivariateBaseline::new(
                            &params,
                            &logical_values,
                            1,
                            BivariateUpdateMode::CachedSeparated,
                        )
                    },
                )
            } else {
                zero_stats()
            },
            "builds one row/column-shaped claim state and separated Lagrange caches",
        ),
        (
            "coset_decomposition",
            "coset_patch",
            algebraic_store_bytes(&optimized_footprint),
            0,
            if measure_setup {
                bench_with_setup(
                    trials,
                    || (),
                    |_| build_claim_store(&params, &logical_values),
                )
            } else {
                zero_stats()
            },
            "builds one advertised responsibility without a dense update cache",
        ),
    ];
    for (model, mode, maintained_bytes, cache_bytes, latency_ns, notes) in setup_cases {
        entries.push(BenchEntry {
            model: model.to_string(),
            operation: "build_state_and_update_cache".to_string(),
            segment_len,
            source_k,
            target_k: None,
            source_shard_len: params.s,
            target_shard_len: None,
            update_mode: Some(mode.to_string()),
            latency_ns,
            maintained_bytes_per_party: maintained_bytes,
            cache_bytes_per_party: cache_bytes,
            helper_download_bytes: 0,
            notes: notes.to_string(),
        });
    }

    let mut update_cases = vec![
        (
            "striped_univariate_cached",
            "cached",
            striped_cached_model.maintained_bytes_per_party(),
            striped_cached_model.cache_bytes_per_party(),
            bench_with_setup(
                trials,
                || striped_cached_model.clone(),
                |model| model.apply_updates(&updates),
            ),
        ),
        (
            "univariate_segment_cached",
            "cached",
            univariate_cached_update_model.maintained_bytes_per_party(),
            univariate_cached_update_model.cache_bytes_per_party(),
            bench_with_setup(
                trials,
                || univariate_cached_update_model.clone(),
                |model| model.apply_updates(&updates),
            ),
        ),
        (
            "generic_bivariate_cached",
            "cached_separated",
            bivariate_cached_update_model.maintained_bytes_per_claim(),
            bivariate_cached_update_model.cache_bytes_per_claim(),
            bench_with_setup(
                trials,
                || bivariate_cached_update_model.clone(),
                |model| model.apply_updates(&updates),
            ),
        ),
    ];
    if !skip_naive_updates {
        let univariate_naive_update_model = UnivariateSegmentBaseline::new(
            &params,
            &logical_values,
            1,
            UnivariateUpdateMode::Naive,
        );
        update_cases.push((
            "univariate_segment_naive",
            "naive",
            univariate_naive_update_model.maintained_bytes_per_party(),
            univariate_naive_update_model.cache_bytes_per_party(),
            bench_with_setup(
                trials,
                || univariate_naive_update_model.clone(),
                |model| model.apply_updates(&updates),
            ),
        ));
        let bivariate_naive_update_model =
            GenericBivariateBaseline::new(&params, &logical_values, 1, BivariateUpdateMode::Naive);
        update_cases.push((
            "generic_bivariate_naive",
            "naive",
            bivariate_naive_update_model.maintained_bytes_per_claim(),
            bivariate_naive_update_model.cache_bytes_per_claim(),
            bench_with_setup(
                trials,
                || bivariate_naive_update_model.clone(),
                |model| model.apply_updates(&updates),
            ),
        ));
    }
    for (model, mode, maintained_bytes, cache_bytes, latency_ns) in update_cases {
        let responsibility = if model == "striped_univariate_cached" {
            "one maintained shard vector"
        } else {
            "one maintained responsibility"
        };
        entries.push(BenchEntry {
            model: model.to_string(),
            operation: "apply_public_point_update_batch".to_string(),
            segment_len,
            source_k,
            target_k: None,
            source_shard_len: params.s,
            target_shard_len: None,
            update_mode: Some(mode.to_string()),
            latency_ns,
            maintained_bytes_per_party: maintained_bytes,
            cache_bytes_per_party: cache_bytes,
            helper_download_bytes: 0,
            notes: format!(
                "{} public point updates on {}",
                updates.len(),
                responsibility
            ),
        });
    }

    let optimized_update = bench_with_setup(
        trials,
        || optimized_model.clone(),
        |model| {
            model
                .apply_deltas(&updates, DeltaMode::Aggregate)
                .expect("optimized updates")
        },
    );
    entries.push(BenchEntry {
        model: "coset_decomposition".to_string(),
        operation: "apply_public_point_update_batch".to_string(),
        segment_len,
        source_k,
        target_k: None,
        source_shard_len: params.s,
        target_shard_len: None,
        update_mode: Some("coset_patch".to_string()),
        latency_ns: optimized_update,
        maintained_bytes_per_party: algebraic_store_bytes(&optimized_footprint),
        cache_bytes_per_party: 0,
        helper_download_bytes: 0,
        notes: format!(
            "{} public point updates on one advertised responsibility",
            updates.len()
        ),
    });

    let point_cases = [
        (
            "striped_univariate_cached",
            striped_cached_model.maintained_bytes_per_party(),
            striped_cached_model.cache_bytes_per_party(),
            source_k as u64 * FIELD_BYTES,
            bench_with_setup(
                trials,
                || striped_segment.clone(),
                |segment| {
                    assert_eq!(
                        striped_decoder.read_point(&striped_layout, segment, POINT_INDEX),
                        logical_values[POINT_INDEX]
                    );
                },
            ),
        ),
        (
            "univariate_segment_cached",
            univariate_cached_model.maintained_bytes_per_party(),
            univariate_cached_model.cache_bytes_per_party(),
            source_k as u64 * params.s as u64 * FIELD_BYTES,
            bench_with_setup(
                trials,
                || univariate_cached_model.clone(),
                |model| {
                    assert_eq!(
                        model.read_point_from_parties(POINT_INDEX, source_k),
                        initial_values(segment_len)[POINT_INDEX]
                    );
                },
            ),
        ),
        (
            "generic_bivariate_cached",
            bivariate_cached_model.maintained_bytes_per_claim(),
            bivariate_cached_model.cache_bytes_per_claim(),
            source_k as u64 * FIELD_BYTES,
            bench_with_setup(
                trials,
                || bivariate_cached_model.clone(),
                |model| {
                    assert_eq!(
                        model.read_point_from_claims(POINT_INDEX, source_k),
                        initial_values(segment_len)[POINT_INDEX]
                    );
                },
            ),
        ),
        (
            "coset_decomposition",
            algebraic_store_bytes(&optimized_footprint),
            0,
            source_k as u64 * FIELD_BYTES,
            bench_with_setup(
                trials,
                || optimized_read_model.clone(),
                |model| {
                    assert_eq!(
                        model
                            .read(POINT_INDEX as u64)
                            .expect("optimized point read"),
                        logical_values[POINT_INDEX]
                    );
                },
            ),
        ),
    ];
    for (model, maintained_bytes, cache_bytes, helper_download_bytes, latency_ns) in point_cases {
        entries.push(BenchEntry {
            model: model.to_string(),
            operation: "client_point_open".to_string(),
            segment_len,
            source_k,
            target_k: None,
            source_shard_len: params.s,
            target_shard_len: None,
            update_mode: None,
            latency_ns,
            maintained_bytes_per_party: maintained_bytes,
            cache_bytes_per_party: cache_bytes,
            helper_download_bytes,
            notes: format!("client reconstructs logical index {}", POINT_INDEX),
        });
    }

    let aligned_cases = [
        (
            "striped_univariate_cached",
            striped_cached_model.maintained_bytes_per_party(),
            striped_cached_model.cache_bytes_per_party(),
            touched_striped_codewords(source_k, &block_offsets) as u64
                * source_k as u64
                * FIELD_BYTES,
            bench_with_setup(
                trials,
                || striped_segment.clone(),
                |segment| {
                    assert_eq!(
                        striped_decoder.read_block(&striped_layout, segment, &block_offsets),
                        expected_block
                    );
                },
            ),
        ),
        (
            "univariate_segment_cached",
            univariate_cached_model.maintained_bytes_per_party(),
            univariate_cached_model.cache_bytes_per_party(),
            source_k as u64 * params.s as u64 * FIELD_BYTES,
            bench_with_setup(
                trials,
                || univariate_cached_model.clone(),
                |model| {
                    let values =
                        univariate_read_block_from_parties(model, &block_offsets, source_k);
                    assert_eq!(
                        values,
                        block_offsets
                            .iter()
                            .map(|&offset| initial_values(segment_len)[offset])
                            .collect::<Vec<_>>()
                    );
                },
            ),
        ),
        (
            "generic_bivariate_cached",
            bivariate_cached_model.maintained_bytes_per_claim(),
            bivariate_cached_model.cache_bytes_per_claim(),
            source_k as u64
                * touched_bivariate_w_queries(&params, &block_offsets) as u64
                * FIELD_BYTES,
            bench_with_setup(
                trials,
                || bivariate_cached_model.clone(),
                |model| {
                    let values =
                        generic_bivariate_read_offsets_from_claims(model, &block_offsets, source_k);
                    assert_eq!(
                        values,
                        block_offsets
                            .iter()
                            .map(|&offset| initial_values(segment_len)[offset])
                            .collect::<Vec<_>>()
                    );
                },
            ),
        ),
        (
            "coset_decomposition",
            algebraic_store_bytes(&optimized_footprint),
            0,
            source_k as u64 * rows_per_claim(&params) as u64 * FIELD_BYTES,
            bench_with_setup(
                trials,
                || optimized_read_model.clone(),
                |model| {
                    assert_eq!(
                        model.read_block(0).expect("optimized block read"),
                        expected_block
                    );
                },
            ),
        ),
    ];
    for (model, maintained_bytes, cache_bytes, helper_download_bytes, latency_ns) in aligned_cases {
        entries.push(BenchEntry {
            model: model.to_string(),
            operation: "client_aligned_open".to_string(),
            segment_len,
            source_k,
            target_k: None,
            source_shard_len: params.s,
            target_shard_len: None,
            update_mode: None,
            latency_ns,
            maintained_bytes_per_party: maintained_bytes,
            cache_bytes_per_party: cache_bytes,
            helper_download_bytes,
            notes: format!(
                "client reconstructs aligned block 0 containing {} values",
                block_offsets.len()
            ),
        });
    }

    let univariate_repair_target =
        crate::support::univariate_segment_baseline::target_party_template(&params, source_k);
    let generic_repair_z_domain =
        crate::support::generic_bivariate_baseline::source_z_domain(&params);
    let generic_repair_target = crate::support::generic_bivariate_baseline::claim_template(
        &params,
        source_k,
        &generic_repair_z_domain,
        &params.h,
    );
    let optimized_repair_helpers =
        build_full_store_with_claims(&params, &logical_values, first_claims(&params, source_k));
    let optimized_repair_claim =
        ClaimBundle::exact_for_registry_index(&params, source_k).expect("repair target claim");
    let optimized_repair_row_indices = optimized_repair_claim
        .row_label_indices(&params)
        .expect("repair row labels");
    let repair_cases = [
        (
            "striped_univariate_cached",
            striped_cached_model.maintained_bytes_per_party(),
            striped_cached_model.cache_bytes_per_party(),
            source_k as u64 * striped_layout.codeword_count as u64 * FIELD_BYTES,
            bench_with_setup(
                trials,
                || striped_segment.clone(),
                |segment| {
                    let repaired = striped_decoder.reconstruct_shard_vector(
                        &striped_layout,
                        segment,
                        source_k,
                    );
                    assert_eq!(repaired, segment.shard_vectors[source_k]);
                },
            ),
        ),
        (
            "univariate_segment_cached",
            univariate_cached_model.maintained_bytes_per_party(),
            univariate_cached_model.cache_bytes_per_party(),
            source_k as u64 * params.s as u64 * FIELD_BYTES,
            bench_with_setup(
                trials,
                || univariate_cached_model.clone(),
                |model| {
                    let repaired =
                        model.repair_party_from_helpers(&univariate_repair_target, source_k);
                    assert_eq!(repaired.len(), params.s);
                },
            ),
        ),
        (
            "generic_bivariate_cached",
            bivariate_cached_model.maintained_bytes_per_claim(),
            bivariate_cached_model.cache_bytes_per_claim(),
            (params.s + rows_per_claim(&params) * source_k) as u64 * FIELD_BYTES,
            bench_with_setup(
                trials,
                || bivariate_cached_model.clone(),
                |model| {
                    let repaired = model
                        .repair_claim_same_geometry_from_helpers(&generic_repair_target, source_k);
                    assert_eq!(repaired.column_values.len(), params.s);
                    assert_eq!(repaired.row_values.len(), rows_per_claim(&params));
                },
            ),
        ),
        (
            "coset_decomposition",
            algebraic_store_bytes(&optimized_footprint),
            0,
            (params.s + rows_per_claim(&params) * source_k) as u64 * FIELD_BYTES,
            bench_with_setup(
                trials,
                || optimized_repair_helpers.clone(),
                |model| {
                    let entry = params.shard_registry[optimized_repair_claim.registry_index];
                    let primary = model
                        .repair_primary_shard_from_rows(0, entry)
                        .expect("optimized primary repair");
                    assert_eq!(primary.len(), params.s);
                    let rows = optimized_repair_row_indices
                        .iter()
                        .map(|&row_idx| {
                            model
                                .repair_row_from_shards(0, params.row_labels[row_idx])
                                .expect("optimized row repair")
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(rows.len(), rows_per_claim(&params));
                    for row in rows {
                        assert_eq!(row.len(), params.need);
                    }
                },
            ),
        ),
    ];
    for (model, maintained_bytes, cache_bytes, helper_download_bytes, latency_ns) in repair_cases {
        entries.push(BenchEntry {
            model: model.to_string(),
            operation: "repair_current_responsibility".to_string(),
            segment_len,
            source_k,
            target_k: None,
            source_shard_len: params.s,
            target_shard_len: None,
            update_mode: None,
            latency_ns,
            maintained_bytes_per_party: maintained_bytes,
            cache_bytes_per_party: cache_bytes,
            helper_download_bytes,
            notes: "maintainer reconstructs one current responsibility from helper data"
                .to_string(),
        });
    }

    let univariate_target = crate::support::univariate_segment_baseline::target_party_template(
        &target_params,
        source_k,
    );
    let target_striped_layout = fair_layout_for_params(&target_params, logical_values.len());
    let target_striped_segment =
        encode_segment_from_values(&target_striped_layout, &logical_values);
    let generic_target_z_domain =
        crate::support::generic_bivariate_baseline::source_z_domain(&target_params);
    let generic_target = crate::support::generic_bivariate_baseline::claim_template(
        &target_params,
        source_k,
        &generic_target_z_domain,
        &target_params.h,
    );
    let kappa_rescale_cases = [
        (
            "striped_univariate_cached",
            striped_cached_model.maintained_bytes_per_party(),
            striped_cached_model.cache_bytes_per_party(),
            source_k as u64 * striped_layout.codeword_count as u64 * FIELD_BYTES,
            bench_with_setup(
                trials,
                || striped_segment.clone(),
                |segment| {
                    let target = striped_decoder.restripe_target_shards(
                        &striped_layout,
                        segment,
                        &target_striped_layout,
                        &[source_k],
                    );
                    assert_eq!(target[0], target_striped_segment.shard_vectors[source_k]);
                },
            ),
        ),
        (
            "univariate_segment_cached",
            univariate_cached_model.maintained_bytes_per_party(),
            univariate_cached_model.cache_bytes_per_party(),
            source_k as u64 * params.s as u64 * FIELD_BYTES,
            bench_with_setup(
                trials,
                || univariate_cached_model.clone(),
                |model| {
                    let repaired = model.repair_party_from_helpers(&univariate_target, source_k);
                    assert_eq!(repaired.len(), target_params.s);
                },
            ),
        ),
        (
            "generic_bivariate_cached",
            bivariate_cached_model.maintained_bytes_per_claim(),
            bivariate_cached_model.cache_bytes_per_claim(),
            source_k as u64 * params.s as u64 * FIELD_BYTES,
            bench_with_setup(
                trials,
                || bivariate_cached_model.clone(),
                |model| {
                    let repaired = model.kappa_rescale_target_claim_by_decode_reencode(
                        &target_params,
                        generic_target.registry_index,
                        source_k,
                    );
                    assert_eq!(repaired.column_values.len(), target_params.s);
                    assert_eq!(repaired.row_values.len(), rows_per_claim(&target_params));
                },
            ),
        ),
    ];
    for (model, maintained_bytes, cache_bytes, helper_download_bytes, latency_ns) in
        kappa_rescale_cases
    {
        entries.push(BenchEntry {
            model: model.to_string(),
            operation: "materialize_target_responsibility".to_string(),
            segment_len,
            source_k,
            target_k: Some(target_k),
            source_shard_len: params.s,
            target_shard_len: Some(target_params.s),
            update_mode: None,
            latency_ns,
            maintained_bytes_per_party: maintained_bytes,
            cache_bytes_per_party: cache_bytes,
            helper_download_bytes,
            notes: if model == "striped_univariate_cached" {
                "baseline decodes and re-encodes the striped segment under the target k".to_string()
            } else if model == "generic_bivariate_cached" {
                "baseline decodes the old bivariate segment and re-encodes the complete target claim, including its target row-label bundle".to_string()
            } else {
                "baseline materializes target responsibility by generic interpolation".to_string()
            },
        });
    }

    let claims = vec![ClaimBundle::exact_for_registry_index(&params, 0).expect("source claim")];
    let mut claim_bound = build_full_store_with_claims(&params, &logical_values, claims);
    claim_bound
        .apply_deltas(&updates, DeltaMode::Aggregate)
        .expect("update claim-bound source store");
    let mut full = build_full_store_with_claims(&params, &logical_values, all_claims(&params));
    full.apply_deltas(&updates, DeltaMode::Aggregate)
        .expect("update full helper store");
    let mut full_target = full.clone();
    full_target
        .migrate(target_params.s, &[])
        .expect("full target migrate");
    let source_entry = params.shard_registry[0];
    let source_primary = claim_bound
        .coded
        .shard_segment(&params, &source_entry.key, 0)
        .expect("source primary column")
        .to_vec();
    let target_claim =
        ClaimBundle::exact_for_registry_index(&target_params, 0).expect("target child claim");
    let target_row_indices = target_claim
        .row_label_indices(&target_params)
        .expect("target child row labels");
    assert_eq!(target_row_indices.len(), rows_per_claim(&target_params));
    let target_row_labels = target_row_indices
        .iter()
        .map(|&idx| target_params.row_labels[idx])
        .collect::<Vec<_>>();
    let helper_download =
        (target_params.s + target_params.need * target_row_labels.len()) as u64 * FIELD_BYTES;
    let optimized_kappa_rescale = bench_with_setup(
        trials,
        || (full_target.clone(), source_primary.clone()),
        |(model, primary)| {
            let second_half = primary[target_params.s..].to_vec();
            assert_eq!(second_half.len(), target_params.s);
            for &target_row_label in &target_row_labels {
                let row = model
                    .repair_row_from_shards(0, target_row_label)
                    .expect("target row repair");
                assert_eq!(row.len(), target_params.need);
            }
        },
    );
    entries.push(BenchEntry {
        model: "coset_decomposition".to_string(),
        operation: "materialize_target_responsibility".to_string(),
        segment_len,
        source_k,
        target_k: Some(target_k),
        source_shard_len: params.s,
        target_shard_len: Some(target_params.s),
        update_mode: None,
        latency_ns: optimized_kappa_rescale,
        maintained_bytes_per_party: algebraic_store_bytes(&optimized_footprint),
        cache_bytes_per_party: 0,
        helper_download_bytes: helper_download,
        notes: "construction-specific split primary conversion plus all target-row repairs"
            .to_string(),
    });

    let reverse_source_k = target_k;
    let reverse_target_k = source_k;
    let reverse_striped_model = StripedUnivariateCachedUpdate::new(&target_params, &logical_values);
    let reverse_univariate_model = UnivariateSegmentBaseline::new(
        &target_params,
        &logical_values,
        reverse_source_k,
        UnivariateUpdateMode::Cached,
    );
    let reverse_bivariate_model = GenericBivariateBaseline::new(
        &target_params,
        &logical_values,
        reverse_source_k,
        BivariateUpdateMode::CachedSeparated,
    );
    let reverse_source_layout = fair_layout_for_params(&target_params, logical_values.len());
    let reverse_source_segment =
        encode_segment_from_values(&reverse_source_layout, &logical_values);
    let reverse_decoder = reverse_source_layout.decoder(&(0..reverse_source_k).collect::<Vec<_>>());
    let reverse_target_layout = fair_layout_for_params(&params, logical_values.len());
    let reverse_target_segment =
        encode_segment_from_values(&reverse_target_layout, &logical_values);
    let reverse_univariate_target =
        crate::support::univariate_segment_baseline::target_party_template(&params, 0);
    let reverse_generic_target_z_domain =
        crate::support::generic_bivariate_baseline::source_z_domain(&params);
    let reverse_generic_target = crate::support::generic_bivariate_baseline::claim_template(
        &params,
        0,
        &reverse_generic_target_z_domain,
        &params.h,
    );
    let reverse_optimized_footprint = measure_store_footprint(&optimized_repair_helpers);
    let reverse_target_row_labels = optimized_repair_row_indices
        .iter()
        .map(|&idx| params.row_labels[idx])
        .collect::<Vec<_>>();
    assert_eq!(reverse_target_row_labels.len(), rows_per_claim(&params));
    let reverse_cases = [
        (
            "striped_univariate_cached",
            reverse_striped_model.maintained_bytes_per_party(),
            reverse_striped_model.cache_bytes_per_party(),
            reverse_source_k as u64 * reverse_source_layout.codeword_count as u64 * FIELD_BYTES,
            bench_with_setup(
                trials,
                || reverse_source_segment.clone(),
                |segment| {
                    let target = reverse_decoder.restripe_target_shards(
                        &reverse_source_layout,
                        segment,
                        &reverse_target_layout,
                        &[0],
                    );
                    assert_eq!(target[0], reverse_target_segment.shard_vectors[0]);
                },
            ),
        ),
        (
            "univariate_segment_cached",
            reverse_univariate_model.maintained_bytes_per_party(),
            reverse_univariate_model.cache_bytes_per_party(),
            reverse_source_k as u64 * target_params.s as u64 * FIELD_BYTES,
            bench_with_setup(
                trials,
                || reverse_univariate_model.clone(),
                |model| {
                    let repaired = model
                        .repair_party_from_helpers(&reverse_univariate_target, reverse_source_k);
                    assert_eq!(repaired.len(), params.s);
                },
            ),
        ),
        (
            "generic_bivariate_cached",
            reverse_bivariate_model.maintained_bytes_per_claim(),
            reverse_bivariate_model.cache_bytes_per_claim(),
            reverse_source_k as u64 * target_params.s as u64 * FIELD_BYTES,
            bench_with_setup(
                trials,
                || reverse_bivariate_model.clone(),
                |model| {
                    let repaired = model.kappa_rescale_target_claim_by_decode_reencode(
                        &params,
                        reverse_generic_target.registry_index,
                        reverse_source_k,
                    );
                    assert_eq!(repaired.column_values.len(), params.s);
                    assert_eq!(repaired.row_values.len(), rows_per_claim(&params));
                },
            ),
        ),
        (
            "coset_decomposition",
            algebraic_store_bytes(&reverse_optimized_footprint),
            0,
            params.need as u64 * reverse_target_row_labels.len() as u64 * FIELD_BYTES,
            bench_with_setup(
                trials,
                || {
                    (
                        optimized_repair_helpers.clone(),
                        vec![Fp::from(0u64); params.s],
                    )
                },
                |(model, primary)| {
                    let kept_primary = primary.iter().step_by(2).copied().collect::<Vec<_>>();
                    assert_eq!(kept_primary.len(), target_params.s);
                    for &reverse_target_row_label in &reverse_target_row_labels {
                        let row = model
                            .repair_row_from_shards(0, reverse_target_row_label)
                            .expect("optimized target-row repair");
                        assert_eq!(row.len(), params.need);
                    }
                },
            ),
        ),
    ];
    for (model, maintained_bytes, cache_bytes, helper_download_bytes, latency_ns) in reverse_cases {
        entries.push(BenchEntry {
            model: model.to_string(),
            operation: "materialize_target_responsibility".to_string(),
            segment_len,
            source_k: reverse_source_k,
            target_k: Some(reverse_target_k),
            source_shard_len: target_params.s,
            target_shard_len: Some(params.s),
            update_mode: None,
            latency_ns,
            maintained_bytes_per_party: maintained_bytes,
            cache_bytes_per_party: cache_bytes,
            helper_download_bytes,
            notes: if model == "coset_decomposition" {
                "construction-specific local primary conversion plus all target-row repairs"
                    .to_string()
            } else if model == "striped_univariate_cached" {
                "baseline decodes and re-encodes the striped segment under the target k".to_string()
            } else if model == "generic_bivariate_cached" {
                "baseline decodes the old bivariate segment and re-encodes the complete target claim, including its target row-label bundle".to_string()
            } else {
                "baseline materializes target responsibility by generic interpolation".to_string()
            },
        });
    }

    let (segment_double_left, segment_double_right, segment_double_target_values) =
        segment_double_target_values(&params, &segment_double_params);
    let segment_double_source_layout = fair_layout_for_params(&params, segment_double_left.len());
    let segment_double_left_segment =
        encode_segment_from_values(&segment_double_source_layout, &segment_double_left);
    let segment_double_right_segment =
        encode_segment_from_values(&segment_double_source_layout, &segment_double_right);
    let segment_double_decoder =
        segment_double_source_layout.decoder(&(0..source_k).collect::<Vec<_>>());
    let segment_double_target_layout =
        fair_layout_for_params(&segment_double_params, segment_double_target_values.len());
    let segment_double_left_univariate = UnivariateSegmentBaseline::new(
        &params,
        &segment_double_left,
        source_k,
        UnivariateUpdateMode::Cached,
    );
    let segment_double_right_univariate = UnivariateSegmentBaseline::new(
        &params,
        &segment_double_right,
        source_k,
        UnivariateUpdateMode::Cached,
    );
    let segment_double_left_bivariate = GenericBivariateBaseline::new(
        &params,
        &segment_double_left,
        source_k,
        BivariateUpdateMode::CachedSeparated,
    );
    let segment_double_right_bivariate = GenericBivariateBaseline::new(
        &params,
        &segment_double_right,
        source_k,
        BivariateUpdateMode::CachedSeparated,
    );
    let segment_double_optimized_source =
        build_segment_double_claim_store(&params, &segment_double_left, &segment_double_right);
    let segment_double_optimized_footprint =
        measure_store_footprint(&segment_double_optimized_source);
    let segment_double_cases = [
        (
            "striped_univariate_cached",
            2 * segment_double_left_segment.shard_vectors[0].len() as u64 * FIELD_BYTES,
            striped_cached_model.cache_bytes_per_party(),
            2 * source_k as u64 * segment_double_source_layout.codeword_count as u64 * FIELD_BYTES,
            bench_with_setup(
                trials,
                || {
                    (
                        segment_double_left_segment.clone(),
                        segment_double_right_segment.clone(),
                    )
                },
                |(left, right)| {
                    let mut decoded = segment_double_decoder
                        .decode_full_segment(&segment_double_source_layout, left);
                    decoded.extend(
                        segment_double_decoder
                            .decode_full_segment(&segment_double_source_layout, right),
                    );
                    decoded.resize(
                        segment_double_target_values.len(),
                        encode_value(Fp::from(0u64)),
                    );
                    let target =
                        encode_segment_from_values(&segment_double_target_layout, &decoded);
                    assert_eq!(
                        target.shard_vectors[0].len(),
                        segment_double_target_layout.codeword_count
                    );
                },
            ),
        ),
        (
            "univariate_segment_cached",
            2 * segment_double_left_univariate.maintained_bytes_per_party(),
            segment_double_left_univariate.cache_bytes_per_party(),
            2 * source_k as u64 * params.s as u64 * FIELD_BYTES,
            bench_with_setup(
                trials,
                || {
                    (
                        segment_double_left_univariate.clone(),
                        segment_double_right_univariate.clone(),
                    )
                },
                |(left, right)| {
                    let mut decoded =
                        decode_univariate_logical_values(left, segment_double_left.len(), source_k);
                    decoded.extend(decode_univariate_logical_values(
                        right,
                        segment_double_right.len(),
                        source_k,
                    ));
                    decoded.resize(
                        segment_double_target_values.len(),
                        encode_value(Fp::from(0u64)),
                    );
                    let target =
                        target_univariate_party_values(&segment_double_params, &decoded, 0);
                    assert_eq!(target.len(), segment_double_params.s);
                },
            ),
        ),
        (
            "generic_bivariate_cached",
            2 * segment_double_left_bivariate.maintained_bytes_per_claim(),
            segment_double_left_bivariate.cache_bytes_per_claim(),
            2 * source_k as u64 * params.s as u64 * FIELD_BYTES,
            bench_with_setup(
                trials,
                || {
                    (
                        segment_double_left_bivariate.clone(),
                        segment_double_right_bivariate.clone(),
                    )
                },
                |(left, right)| {
                    let target = left.segment_double_target_claim_by_decode_reencode(
                        right,
                        &segment_double_params,
                        0,
                        source_k,
                    );
                    assert_eq!(target.column_values.len(), segment_double_params.s);
                    assert_eq!(
                        target.row_values.len(),
                        rows_per_claim(&segment_double_params)
                    );
                },
            ),
        ),
        (
            "coset_decomposition",
            algebraic_store_bytes(&segment_double_optimized_footprint),
            0,
            0,
            bench_with_setup(
                trials,
                || segment_double_optimized_source.clone(),
                |model| {
                    model.segment_double().expect("optimized segment-double");
                    assert_eq!(model.params.segment_len, segment_double_params.segment_len);
                    assert_eq!(model.params.need, segment_double_params.need);
                    assert_eq!(model.params.s, segment_double_params.s);
                    assert_eq!(model.segments(), 1);
                },
            ),
        ),
    ];
    for (model, maintained_bytes, cache_bytes, helper_download_bytes, latency_ns) in
        segment_double_cases
    {
        entries.push(BenchEntry {
            model: model.to_string(),
            operation: "materialize_segment_double_responsibility".to_string(),
            segment_len,
            source_k,
            target_k: Some(segment_double_params.need),
            source_shard_len: params.s,
            target_shard_len: Some(segment_double_params.s),
            update_mode: None,
            latency_ns,
            maintained_bytes_per_party: maintained_bytes,
            cache_bytes_per_party: cache_bytes,
            helper_download_bytes,
            notes: if model == "coset_decomposition" {
                "construction-specific local segment-double stitch of two adjacent source responsibilities"
                    .to_string()
            } else {
                "baseline decodes two adjacent source segment responsibilities and re-encodes one target responsibility"
                    .to_string()
            },
        });
    }

    entries
}

fn main() {
    let trials = std::env::var("BENCH_UNAUTH_ENCODING_BASELINES_TRIALS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(DEFAULT_TRIALS);
    let segment_lens = parse_usize_list(
        "BENCH_UNAUTH_ENCODING_BASELINES_SEGMENT_LENS",
        DEFAULT_SEGMENT_LENS,
    );
    let k_values = parse_usize_list("BENCH_UNAUTH_ENCODING_BASELINES_K_VALUES", DEFAULT_K_VALUES);
    let large_segment_mode = parse_bool("BENCH_UNAUTH_ENCODING_BASELINES_LARGE_SEGMENT_MODE");
    let skip_materialization = parse_bool("BENCH_UNAUTH_ENCODING_BASELINES_SKIP_MATERIALIZATION");
    let skip_naive_updates = parse_bool("BENCH_UNAUTH_ENCODING_BASELINES_SKIP_NAIVE_UPDATES");
    let measure_setup = std::env::var("BENCH_UNAUTH_ENCODING_BASELINES_MEASURE_SETUP")
        .ok()
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "no" | "off")
        })
        .unwrap_or(true);
    let mut entries = Vec::new();
    for &segment_len in &segment_lens {
        for &k in &k_values {
            entries.extend(run_entry(
                segment_len,
                k,
                trials,
                large_segment_mode,
                skip_materialization,
                skip_naive_updates,
                measure_setup,
            ));
        }
    }
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let results = BenchResults {
        meta: BenchMeta {
            trials,
            segment_lens,
            k_values,
            update_count: UPDATE_COUNT,
            timestamp_unix_ms: timestamp_ms,
            git: git_sha(),
            command: std::env::args().collect(),
            notes: format!(
                "benchmark-only unauthenticated-array encoding baselines: striped univariate RS, single-polynomial univariate Lagrange, generic bivariate product patch, and optimized coset/decomposition state; large_segment_mode={}, skip_materialization={}, skip_naive_updates={}, measure_setup={}",
                large_segment_mode, skip_materialization, skip_naive_updates, measure_setup
            ),
        },
        entries,
    };
    let path = output_path();
    let out_dir = Path::new(&path).parent().unwrap_or_else(|| Path::new("."));
    let _ = fs::create_dir_all(out_dir);
    let data =
        serde_json::to_vec_pretty(&results).expect("serialize bench_unauth_encoding_baselines");
    fs::write(path, data).expect("write bench_unauth_encoding_baselines.json");
}
