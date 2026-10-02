//! Local one-server append-only-log kappa-rescale benchmark.
//!
//! The app-level kappa-rescale smoke test intentionally exercises the whole
//! local cluster. This benchmark measures the per-maintainer work instead, for
//! one destination server that holds one claim:
//! - doubling `kappa` is the four-segment stitch. The server combines its own
//!   fragments; no helper is involved and nothing is downloaded.
//! - halving `kappa` undoes the stitch. The missing half of the server's
//!   source state is decoded from the partial-stitch evaluations of `2 kappa`
//!   helper claims that are at the doubled threshold, two field elements per
//!   helper and stitched segment. The benchmark also times the undo kernel
//!   when the half comes as the source state of the two diagonal segments,
//!   which is what helpers at the lower threshold supply through ordinary
//!   repair.

mod support;

use crate::support::append_only_log_repair::{
    BENCH_SEED, DEFAULT_KAPPA, FIELD_BYTES, MetricStats, build_setup,
    build_warmed_cluster_for_registry_indices, git_sha, parse_usize_env, stats_from_samples,
    timestamp_unix_ms, write_json,
};
use append_only_log_demo::layout::{BULK_UPDATES_BYTES, BULK_UPDATES_SIZE};
use paper_a_optimized::Fp;
use paper_a_optimized::net::wire::{self, NetFpBlob};
use paper_a_optimized::net::{Message, Response};
use paper_a_optimized::unauth::service::{UnauthRequest, UnauthResponse, UnauthService};
use paper_a_optimized::unauth::{
    ClaimBundle, CodedArrayParams, LeadingSourceDecoder, PartialStitchEvals,
    RecoveredClaimSegmentState, SourceHalfState, UnauthStore,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

const DEFAULT_KAPPA_RESCALE_COMMITTED_BULK_UPDATE_BATCHES: usize = 3;
const DEFAULT_KAPPA_RESCALE_TRIALS: usize = 3;
/// Registry index of the destination server's claim.
const RECEIVER_REGISTRY_INDEX: usize = 0;

#[derive(Clone, Debug, Serialize)]
struct BenchMeta {
    seed: u64,
    trials: usize,
    committed_bulk_update_batches: usize,
    timestamp_unix_ms: u64,
    git: Option<String>,
    notes: String,
    metrics: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct BenchEntry {
    /// Source threshold; the receiver doubles it and then halves it back.
    kappa: usize,
    target_kappa: usize,
    segment_len: usize,
    target_segment_len: usize,
    source_need: usize,
    target_need: usize,
    source_shard_count: usize,
    bulk_updates_bytes: usize,
    bulk_updates_size: usize,
    committed_bulk_update_batches: usize,
    live_unauth_len: u64,
    segment_count: usize,
    target_segment_count: usize,
    receiver_registry_index: usize,
    receiver_source_state_bytes: u64,
    receiver_target_state_bytes: u64,
    kappa_double_helper_count: usize,
    kappa_double_helper_eval_raw_bytes_total: u64,
    local_kappa_double_total_ns: MetricStats,
    kappa_halve_helper_count: usize,
    kappa_halve_helper_eval_scalars_total: u64,
    kappa_halve_helper_eval_raw_bytes_total: u64,
    kappa_halve_helper_eval_one_helper_serialized_bytes: usize,
    kappa_halve_helper_eval_one_helper_serialized_ns: MetricStats,
    kappa_halve_helper_eval_all_helpers_serial_ns: MetricStats,
    kappa_halve_decode_source_half_ns: MetricStats,
    kappa_halve_unstitch_install_ns: MetricStats,
    kappa_halve_prepared_total_ns: MetricStats,
    kappa_halve_diagonal_source_raw_bytes_total: u64,
    kappa_halve_diagonal_unstitch_install_ns: MetricStats,
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

/// Warmed one-claim stores at the source threshold, the receiver's stitched
/// store, and the helpers' stitched stores.
struct OneServerKappaRescaleFixture {
    source_params: CodedArrayParams,
    target_params: CodedArrayParams,
    /// The receiver at the source threshold.
    source_store: UnauthStore,
    /// The receiver after the stitch.
    stitched_store: UnauthStore,
    /// Helpers at the doubled threshold, by registry index.
    stitched_helpers: BTreeMap<usize, UnauthStore>,
    receiver_claim: ClaimBundle,
    receiver_col_label: Fp,
    receiver_row_label: Fp,
    stitched_segs: Vec<usize>,
}

fn build_warmed_claim_stores_batched(
    setup: &crate::support::append_only_log_repair::BenchSetup,
    committed_bulk_update_batches: usize,
    registry_indices: &[usize],
) -> BTreeMap<usize, UnauthStore> {
    let batch_size =
        parse_usize_env("BENCH_APPEND_ONLY_LOG_KAPPA_RESCALE_WARM_BATCH_SIZE", 16).max(1);
    let mut stores = BTreeMap::new();
    for batch in registry_indices.chunks(batch_size) {
        let warmed =
            build_warmed_cluster_for_registry_indices(setup, committed_bulk_update_batches, batch);
        for (registry_index, server) in warmed {
            stores.insert(registry_index, server.store.unauth);
        }
    }
    stores
}

fn build_fixture(
    kappa: usize,
    committed_bulk_update_batches: usize,
) -> OneServerKappaRescaleFixture {
    // Undoing the stitch decodes at the doubled threshold, so it needs
    // `2 kappa` helper claims besides the receiver's own.
    let helper_count =
        2 * kappa + parse_usize_env("BENCH_APPEND_ONLY_LOG_KAPPA_RESCALE_EXTRA_HELPERS", 0);
    let helper_surplus = helper_count + 1 - kappa;
    let setup = build_setup(kappa, helper_surplus);
    let registry_indices = (0..=helper_count).collect::<Vec<_>>();
    let mut warmed_stores =
        build_warmed_claim_stores_batched(&setup, committed_bulk_update_batches, &registry_indices);

    let source_store = warmed_stores
        .remove(&RECEIVER_REGISTRY_INDEX)
        .expect("receiver store warmed");
    let mut stitched_store = source_store.clone();
    stitched_store
        .kappa_double()
        .expect("receiver kappa-double");
    let stitched_helpers = warmed_stores
        .into_iter()
        .map(|(registry_index, mut store)| {
            store.kappa_double().expect("helper kappa-double");
            (registry_index, store)
        })
        .collect::<BTreeMap<_, _>>();

    let target_params = stitched_store.params.clone();
    let stitched_segs = (0..stitched_store.segments()).collect();
    OneServerKappaRescaleFixture {
        receiver_claim: ClaimBundle::exact_for_registry_index(
            &setup.params,
            RECEIVER_REGISTRY_INDEX,
        )
        .expect("receiver claim"),
        receiver_col_label: setup.params.shard_registry[RECEIVER_REGISTRY_INDEX].z,
        receiver_row_label: setup.params.row_labels[RECEIVER_REGISTRY_INDEX],
        source_params: setup.params,
        target_params,
        source_store,
        stitched_store,
        stitched_helpers,
        stitched_segs,
    }
}

impl OneServerKappaRescaleFixture {
    /// The receiver's local stitch: every group of four source segments
    /// becomes one segment at the doubled threshold.
    fn kappa_double_once(&self, mut store: UnauthStore) -> UnauthStore {
        store.kappa_double().expect("kappa-double");
        store
    }

    fn partial_stitch_message(&self) -> Message {
        Message::UnauthEvalPartialStitches {
            segs: self.stitched_segs.iter().map(|&seg| seg as u64).collect(),
            col_labels: NetFpBlob(vec![self.receiver_col_label]),
            row_labels: NetFpBlob(vec![self.receiver_row_label]),
        }
    }

    /// One helper's response to the receiver's request, through the RPC
    /// adapter and serialized for the wire.
    fn helper_serialized_reply_once(&self, helper: &mut UnauthStore) -> Vec<u8> {
        let response =
            paper_a_optimized::net::unauth::handle_message(helper, self.partial_stitch_message());
        assert!(
            matches!(response, Response::UnauthEvalPartialStitches { .. }),
            "unexpected partial-stitch response"
        );
        wire::encode_binary(&response).expect("serialize partial-stitch response")
    }

    /// One helper's evaluations, without the wire adapter.
    fn helper_evals_once(&self, helper: &mut UnauthStore) -> Vec<Vec<PartialStitchEvals>> {
        match helper.handle_request(UnauthRequest::EvalPartialStitches {
            segs: self.stitched_segs.clone(),
            col_labels: vec![self.receiver_col_label],
            row_labels: vec![self.receiver_row_label],
        }) {
            UnauthResponse::EvalPartialStitches { evals } => evals,
            other => panic!("unexpected partial-stitch response: {:?}", other),
        }
    }

    /// Evaluations of every helper, grouped by stitched segment.
    fn collect_helper_evals_once(
        &self,
        helpers: &mut BTreeMap<usize, UnauthStore>,
    ) -> Vec<Vec<PartialStitchEvals>> {
        let mut evals_by_seg = vec![Vec::new(); self.stitched_segs.len()];
        for helper in helpers.values_mut() {
            for (seg_pos, helper_evals) in self.helper_evals_once(helper).into_iter().enumerate() {
                evals_by_seg[seg_pos].extend(helper_evals);
            }
        }
        evals_by_seg
    }

    /// Decode the receiver's leading source half of every stitched segment.
    fn decode_source_halves_once(
        &self,
        evals_by_seg: &[Vec<PartialStitchEvals>],
    ) -> Vec<SourceHalfState> {
        let params = &self.target_params;
        let registry_index = self.receiver_claim.registry_index;
        let row_label_index = self.receiver_claim.row_label_index();
        // Every segment is decoded from the same helper claims.
        let helper_indices = evals_by_seg
            .first()
            .map(|helper_evals| {
                helper_evals
                    .iter()
                    .take(params.need)
                    .map(|helper| helper.registry_index)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let decoder = LeadingSourceDecoder::new(params, &helper_indices).expect("helper decoder");
        self.stitched_segs
            .iter()
            .zip(evals_by_seg.iter())
            .map(|(&seg, helper_evals)| {
                let helper_evals = &helper_evals[..params.need];
                let row_evals = helper_evals
                    .iter()
                    .map(|helper| helper.row_evals[0])
                    .collect::<Vec<_>>();
                let column_evals = helper_evals
                    .iter()
                    .map(|helper| helper.column_evals[0])
                    .collect::<Vec<_>>();
                let fragments = decoder
                    .decode(params, seg, &row_evals, &column_evals)
                    .expect("decode leading source half");
                SourceHalfState::Leading {
                    columns_00: vec![(registry_index, fragments.column_00)],
                    columns_01: vec![(registry_index, fragments.column_01)],
                    rows_00: vec![(row_label_index, fragments.row_00)],
                    rows_10: vec![(row_label_index, fragments.row_10)],
                }
            })
            .collect()
    }

    /// Undo the stitch on the receiver from prepared source halves.
    fn kappa_halve_once(
        &self,
        mut store: UnauthStore,
        source_halves: &[SourceHalfState],
    ) -> UnauthStore {
        store.kappa_halve(source_halves).expect("kappa-halve");
        store
    }

    /// The receiver's source state for the two diagonal source segments of
    /// every stitched segment, as ordinary repair at the lower threshold
    /// returns it.
    fn diagonal_source_halves(&self) -> Vec<SourceHalfState> {
        let params = &self.source_params;
        let entry = params.shard_registry[self.receiver_claim.registry_index];
        let row_label_index = self.receiver_claim.row_label_index();
        let row_key = paper_a_optimized::RowKey::new(entry.key, params.row_labels[row_label_index]);
        let segment_state = |seg: usize| {
            // A source segment beyond the store is padding of the stitch.
            let column = self
                .source_store
                .coded
                .shard_segment(params, &entry.key, seg)
                .map(|values| values.to_vec())
                .unwrap_or_else(|| vec![paper_a_optimized::fp_zero(); params.s]);
            let row = self
                .source_store
                .coded
                .row_segment(params, &row_key, seg)
                .map(|values| values.to_vec())
                .unwrap_or_else(|| vec![paper_a_optimized::fp_zero(); params.row_len()]);
            RecoveredClaimSegmentState {
                shards: vec![(self.receiver_claim.registry_index, column)],
                rows: vec![(row_label_index, row)],
            }
        };
        self.stitched_segs
            .iter()
            .map(|&parent| SourceHalfState::Diagonal {
                segment_00: segment_state(4 * parent),
                segment_11: segment_state(4 * parent + 3),
            })
            .collect()
    }

    fn assert_same_coded_state(&self, actual: &UnauthStore, expected: &UnauthStore) {
        assert_eq!(actual.len(), expected.len());
        assert_eq!(actual.segments(), expected.segments());
        assert_eq!(actual.params.need, expected.params.need);
        assert!(
            actual
                .coded
                .equals_for_tests(&expected.params, &expected.coded)
        );
    }

    /// The stitch followed by either undo returns the receiver's source
    /// state exactly.
    fn validate(&self, leading_halves: &[SourceHalfState], diagonal_halves: &[SourceHalfState]) {
        let stitched = self.kappa_double_once(self.source_store.clone());
        self.assert_same_coded_state(&stitched, &self.stitched_store);
        let from_leading = self.kappa_halve_once(self.stitched_store.clone(), leading_halves);
        self.assert_same_coded_state(&from_leading, &self.source_store);
        let from_diagonal = self.kappa_halve_once(self.stitched_store.clone(), diagonal_halves);
        self.assert_same_coded_state(&from_diagonal, &self.source_store);
    }
}

fn bench_one_server_kappa_rescale(
    trials: usize,
    committed_bulk_update_batches: usize,
    metrics: &BTreeSet<String>,
) -> BenchEntry {
    let kappa = parse_usize_env("BENCH_APPEND_ONLY_LOG_KAPPA_RESCALE_KAPPA", DEFAULT_KAPPA);
    let fixture = build_fixture(kappa, committed_bulk_update_batches);
    let mut helpers = fixture.stitched_helpers.clone();
    let helper_evals_template = fixture.collect_helper_evals_once(&mut helpers);
    let leading_halves_template = fixture.decode_source_halves_once(&helper_evals_template);
    let diagonal_halves_template = fixture.diagonal_source_halves();
    fixture.validate(&leading_halves_template, &diagonal_halves_template);

    let mut double_samples = Vec::with_capacity(trials);
    if metrics.contains("receiver_double") {
        for _ in 0..trials {
            let store = fixture.source_store.clone();
            let start = Instant::now();
            let stitched = fixture.kappa_double_once(store);
            double_samples.push(start.elapsed().as_nanos() as u64);
            black_box(stitched);
        }
    }

    let first_helper_index = *helpers.keys().next().expect("at least one helper");
    let one_helper_serialized_bytes = fixture
        .helper_serialized_reply_once(helpers.get_mut(&first_helper_index).expect("first helper"))
        .len();
    let mut one_helper_samples = Vec::with_capacity(trials);
    if metrics.contains("helper_halve") {
        let helper = helpers.get_mut(&first_helper_index).expect("first helper");
        for _ in 0..trials {
            let start = Instant::now();
            let encoded = fixture.helper_serialized_reply_once(helper);
            one_helper_samples.push(start.elapsed().as_nanos() as u64);
            black_box(encoded);
        }
    }

    let mut all_helper_samples = Vec::with_capacity(trials);
    if metrics.contains("all_helpers_halve") {
        for _ in 0..trials {
            let start = Instant::now();
            let evals = fixture.collect_helper_evals_once(&mut helpers);
            all_helper_samples.push(start.elapsed().as_nanos() as u64);
            black_box(evals);
        }
    }

    let mut decode_samples = Vec::with_capacity(trials);
    let mut install_samples = Vec::with_capacity(trials);
    if metrics.contains("component_halve") {
        for _ in 0..trials {
            let start = Instant::now();
            let halves = fixture.decode_source_halves_once(&helper_evals_template);
            decode_samples.push(start.elapsed().as_nanos() as u64);
            black_box(halves);
        }
        for _ in 0..trials {
            let store = fixture.stitched_store.clone();
            let start = Instant::now();
            let halved = fixture.kappa_halve_once(store, &leading_halves_template);
            install_samples.push(start.elapsed().as_nanos() as u64);
            black_box(halved);
        }
    }

    let mut halve_total_samples = Vec::with_capacity(trials);
    if metrics.contains("receiver_halve") {
        for _ in 0..trials {
            let store = fixture.stitched_store.clone();
            let start = Instant::now();
            let halves = fixture.decode_source_halves_once(&helper_evals_template);
            let halved = fixture.kappa_halve_once(store, &halves);
            halve_total_samples.push(start.elapsed().as_nanos() as u64);
            black_box(halved);
        }
    }

    let mut diagonal_install_samples = Vec::with_capacity(trials);
    if metrics.contains("receiver_halve_diagonal") {
        for _ in 0..trials {
            let store = fixture.stitched_store.clone();
            let start = Instant::now();
            let halved = fixture.kappa_halve_once(store, &diagonal_halves_template);
            diagonal_install_samples.push(start.elapsed().as_nanos() as u64);
            black_box(halved);
        }
    }

    let source_segments = fixture.source_store.segments();
    let target_segments = fixture.stitched_store.segments();
    let source_kappa = fixture.source_params.need;
    let target_kappa = fixture.target_params.need;
    // Two field elements per helper claim and stitched segment.
    let helper_eval_scalars_total = 2 * helpers.len() as u64 * target_segments as u64;
    // The diagonal half is the source column and row of two source segments
    // per stitched segment.
    let diagonal_source_scalars_total = 2 * 2 * source_kappa as u64 * target_segments as u64;
    BenchEntry {
        kappa: source_kappa,
        target_kappa,
        segment_len: fixture.source_params.segment_len,
        target_segment_len: fixture.target_params.segment_len,
        source_need: source_kappa,
        target_need: target_kappa,
        source_shard_count: fixture.source_params.shard_registry.len(),
        bulk_updates_bytes: BULK_UPDATES_BYTES,
        bulk_updates_size: BULK_UPDATES_SIZE,
        committed_bulk_update_batches,
        live_unauth_len: fixture.source_store.len(),
        segment_count: source_segments,
        target_segment_count: target_segments,
        receiver_registry_index: fixture.receiver_claim.registry_index,
        receiver_source_state_bytes: source_segments as u64 * 2 * source_kappa as u64 * FIELD_BYTES,
        receiver_target_state_bytes: target_segments as u64 * 2 * target_kappa as u64 * FIELD_BYTES,
        kappa_double_helper_count: 0,
        kappa_double_helper_eval_raw_bytes_total: 0,
        local_kappa_double_total_ns: stats_from_samples(double_samples),
        kappa_halve_helper_count: helpers.len(),
        kappa_halve_helper_eval_scalars_total: helper_eval_scalars_total,
        kappa_halve_helper_eval_raw_bytes_total: helper_eval_scalars_total * FIELD_BYTES,
        kappa_halve_helper_eval_one_helper_serialized_bytes: one_helper_serialized_bytes,
        kappa_halve_helper_eval_one_helper_serialized_ns: stats_from_samples(one_helper_samples),
        kappa_halve_helper_eval_all_helpers_serial_ns: stats_from_samples(all_helper_samples),
        kappa_halve_decode_source_half_ns: stats_from_samples(decode_samples),
        kappa_halve_unstitch_install_ns: stats_from_samples(install_samples),
        kappa_halve_prepared_total_ns: stats_from_samples(halve_total_samples),
        kappa_halve_diagonal_source_raw_bytes_total: diagonal_source_scalars_total * FIELD_BYTES,
        kappa_halve_diagonal_unstitch_install_ns: stats_from_samples(diagonal_install_samples),
    }
}

fn run_benchmark() {
    let trials = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_KAPPA_RESCALE_TRIALS",
        DEFAULT_KAPPA_RESCALE_TRIALS,
    );
    let committed_bulk_update_batches = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_KAPPA_RESCALE_COMMITTED_BULK_UPDATE_BATCHES",
        DEFAULT_KAPPA_RESCALE_COMMITTED_BULK_UPDATE_BATCHES,
    );
    let metrics = parse_metric_filter(
        "BENCH_APPEND_ONLY_LOG_KAPPA_RESCALE_METRICS",
        &[
            "receiver_double",
            "helper_halve",
            "all_helpers_halve",
            "component_halve",
            "receiver_halve",
            "receiver_halve_diagonal",
        ],
    );
    let entry = bench_one_server_kappa_rescale(trials, committed_bulk_update_batches, &metrics);
    let results = BenchResults {
        meta: BenchMeta {
            seed: BENCH_SEED,
            trials,
            committed_bulk_update_batches,
            timestamp_unix_ms: timestamp_unix_ms(),
            git: git_sha(),
            notes: "local append-only-log one-server reconfiguration: warm real one-claim-per-party append-only-log state, then time the destination maintainer's local four-segment stitch (kappa-double), one helper's partial-stitch response construction and serialization, and the destination maintainer's undo of the stitch (kappa-halve) from prepared helper responses or from diagonal source state; whole-cluster coordination is intentionally excluded".to_string(),
            metrics: metrics.into_iter().collect(),
        },
        entries: vec![entry],
    };
    write_json(
        Path::new("results/bench_append_only_log_one_server_kappa_rescale_local.json"),
        &results,
    );
}

fn main() {
    run_benchmark();
}
