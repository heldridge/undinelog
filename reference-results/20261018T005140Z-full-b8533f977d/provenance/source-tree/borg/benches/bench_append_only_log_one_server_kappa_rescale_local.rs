//! Local one-server append-only-log kappa-rescale benchmark.
//!
//! The app-level kappa-rescale smoke test intentionally exercises the whole
//! local cluster. This benchmark measures the per-maintainer work instead:
//! one destination server keeps its left child under `s -> s/2`, receives the
//! compact helper evaluations needed for its retained rows, and installs the
//! kappa-double one-claim state.

mod support;

use crate::support::append_only_log_repair::{
    BENCH_SEED, FIELD_BYTES, MetricStats, build_setup, build_warmed_cluster_for_registry_indices,
    claim_for_registry_index, git_sha, parse_usize_env, stats_from_samples, timestamp_unix_ms,
    write_json,
};
use append_only_log_demo::layout::{BULK_UPDATES_BYTES, BULK_UPDATES_SIZE};
use paper_a_optimized::Fp;
use paper_a_optimized::net::wire::{self, NetFp, NetKappaDoubleRowEvalMigration};
use paper_a_optimized::unauth::{
    ClaimBundle, CodedArrayParams, KappaDoubleOneClaimLeftPlan, KappaDoubleRowEvalMigration,
    KappaHalveOneClaimWithPartnerPlan, PartnerShardMigration, ShardKey, TargetRowEvalMigration,
    UnauthStore, build_migrated_params, build_segment_double_params,
    kappa_double_child_registry_indices, kappa_halve_parent_registry_index,
    shard_value_from_vector_batch,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

const DEFAULT_KAPPA_RESCALE_SEGMENT_LEN: usize = 128;
const DEFAULT_KAPPA_RESCALE_SOURCE_S: usize = 16;
const DEFAULT_KAPPA_RESCALE_COMMITTED_BULK_UPDATE_BATCHES: usize = 3;
const DEFAULT_KAPPA_RESCALE_TRIALS: usize = 3;

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
    segment_len: usize,
    source_s: usize,
    target_s: usize,
    n_grid: usize,
    source_need: usize,
    target_need: usize,
    source_shard_count: usize,
    bulk_updates_bytes: usize,
    bulk_updates_size: usize,
    committed_bulk_update_batches: usize,
    live_unauth_len: u64,
    segment_count: usize,
    target_source_registry_index: usize,
    target_registry_index: usize,
    target_row_count: usize,
    helper_target_count: usize,
    helper_eval_scalars_total: u64,
    helper_eval_raw_bytes_total: u64,
    helper_eval_one_helper_serialized_ns: MetricStats,
    helper_eval_all_helpers_serial_ns: MetricStats,
    local_retained_child_materialize_ns: MetricStats,
    local_row_hydrate_install_ns: MetricStats,
    local_kappa_double_total_ns: MetricStats,
    segment_double_target_segment_len: usize,
    segment_double_target_s: usize,
    segment_double_target_need: usize,
    segment_double_target_segment_count: usize,
    segment_double_helper_eval_raw_bytes_total: u64,
    local_segment_double_total_ns: MetricStats,
    kappa_halve_source_s: usize,
    kappa_halve_target_s: usize,
    kappa_halve_source_need: usize,
    kappa_halve_target_need: usize,
    kappa_halve_target_row_count: usize,
    kappa_halve_helper_count: usize,
    kappa_halve_helper_eval_raw_bytes_total: u64,
    kappa_halve_helper_eval_one_helper_serialized_ns: MetricStats,
    kappa_halve_helper_eval_all_helpers_serial_ns: MetricStats,
    kappa_halve_primary_materialize_ns: MetricStats,
    kappa_halve_row_hydrate_install_ns: MetricStats,
    kappa_halve_prepared_total_ns: MetricStats,
}

#[derive(Clone, Debug, Serialize)]
struct BenchResults {
    meta: BenchMeta,
    entries: Vec<BenchEntry>,
}

fn serialize_kappa_double_reply(reply: &KappaDoubleRowEvalMigration) -> Vec<u8> {
    let reply = NetKappaDoubleRowEvalMigration {
        registry_index: reply.registry_index as u32,
        segments: reply
            .segments
            .iter()
            .map(|segment| segment.iter().copied().map(NetFp).collect())
            .collect(),
    };
    wire::encode_binary(&reply).expect("serialize kappa-double helper response")
}

fn serialize_kappa_halve_reply(reply: &TargetRowEvalMigration) -> Vec<u8> {
    let reply = NetKappaDoubleRowEvalMigration {
        registry_index: reply.registry_index as u32,
        segments: reply
            .segments
            .iter()
            .map(|segment| segment.iter().copied().map(NetFp).collect())
            .collect(),
    };
    wire::encode_binary(&reply).expect("serialize kappa-halve helper response")
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

struct OneServerKappaDoubleFixture {
    source_store: UnauthStore,
    helper_stores: BTreeMap<usize, UnauthStore>,
    source_params: CodedArrayParams,
    target_params: CodedArrayParams,
    target_s: usize,
    segs: Vec<usize>,
    target_registry_index: usize,
    target_claim: ClaimBundle,
    target_row_labels: Vec<paper_a_optimized::Fp>,
    helper_target_indices: Vec<usize>,
    kappa_double_plan: KappaDoubleOneClaimLeftPlan,
}

struct OneServerKappaHalveFixture {
    source_store: UnauthStore,
    helper_stores: BTreeMap<usize, UnauthStore>,
    source_params: CodedArrayParams,
    target_params: CodedArrayParams,
    target_s: usize,
    target_need: usize,
    target_row_labels: Vec<Fp>,
    segs: Vec<usize>,
    partner_source_registry_index: usize,
    partner_helper_registry_index: usize,
    helper_target_indices: Vec<usize>,
    kappa_halve_plan: KappaHalveOneClaimWithPartnerPlan,
}

fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

fn source_helper_surplus_for_target_need(segment_len: usize, source_s: usize) -> usize {
    let n_grid = segment_len;
    let source_need = ceil_div(n_grid, source_s);
    let target_need = ceil_div(n_grid, source_s / 2);
    target_need.saturating_sub(source_need)
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
    segment_len: usize,
    source_s: usize,
    committed_bulk_update_batches: usize,
) -> OneServerKappaDoubleFixture {
    assert!(source_s >= 2 && source_s.is_power_of_two());
    let target_s = source_s / 2;
    let helper_surplus = source_helper_surplus_for_target_need(segment_len, source_s)
        + parse_usize_env("BENCH_APPEND_ONLY_LOG_KAPPA_RESCALE_EXTRA_HELPERS", 0);
    let setup = build_setup(segment_len, source_s, helper_surplus);
    let target_params = build_migrated_params(&setup.params, target_s).expect("target params");
    let (target_registry_index, _) = kappa_double_child_registry_indices(0);
    let target_claim = claim_for_registry_index(&target_params, target_registry_index);
    let target_row_label_indices = target_claim
        .row_label_indices(&target_params)
        .expect("target claim rows");
    let target_row_labels = target_row_label_indices
        .iter()
        .map(|&idx| target_params.row_labels[idx])
        .collect::<Vec<_>>();
    let target_need = target_params.need;
    let mut helper_target_indices = Vec::with_capacity(target_need.saturating_sub(1));
    for source_registry_index in 1..setup.params.shard_registry.len() {
        let (left_child, _) = kappa_double_child_registry_indices(source_registry_index);
        helper_target_indices.push(left_child);
        if helper_target_indices.len() + 1 >= target_need {
            break;
        }
    }
    assert_eq!(
        helper_target_indices.len() + 1,
        target_need,
        "not enough helper target children for one-server kappa-double"
    );
    let mut required_source_indices = BTreeSet::new();
    required_source_indices.insert(0usize);
    for &target_registry_index in &helper_target_indices {
        required_source_indices.insert(kappa_halve_parent_registry_index(target_registry_index));
    }
    let required_source_indices = required_source_indices.into_iter().collect::<Vec<_>>();
    let mut warmed_stores = build_warmed_claim_stores_batched(
        &setup,
        committed_bulk_update_batches,
        &required_source_indices,
    );
    let source_store = warmed_stores.remove(&0).expect("source registry 0 warmed");
    let kappa_double_plan = source_store
        .prepare_kappa_double_one_claim_left(target_s, &helper_target_indices)
        .expect("prepared kappa-double plan");
    let segment_count = source_store.segments();
    let segs = (0..segment_count).collect();
    OneServerKappaDoubleFixture {
        source_store,
        helper_stores: warmed_stores,
        source_params: setup.params,
        target_params,
        target_s,
        segs,
        target_registry_index,
        target_claim,
        target_row_labels,
        helper_target_indices,
        kappa_double_plan,
    }
}

fn build_kappa_halve_fixture(
    segment_len: usize,
    source_s: usize,
    committed_bulk_update_batches: usize,
) -> OneServerKappaHalveFixture {
    assert!(source_s >= 1 && source_s.is_power_of_two());
    let target_s = source_s * 2;
    let setup = build_setup(segment_len, source_s, 0);
    let target_params = build_migrated_params(&setup.params, target_s).expect("target params");
    let source_need = setup.params.need;
    let target_claim = claim_for_registry_index(&target_params, 0);
    let target_row_label_indices = target_claim
        .row_label_indices(&target_params)
        .expect("target claim rows");
    let target_row_labels = target_row_label_indices
        .iter()
        .map(|&idx| target_params.row_labels[idx])
        .collect::<Vec<_>>();
    let target_need = target_params.need;
    let helper_target_indices = (1..target_need).collect::<Vec<_>>();
    let partner_source_registry_index = 0usize;
    let partner_helper_registry_index = 1usize;
    assert!(source_need >= 2 * target_need);
    let required_source_indices = (0..source_need).collect::<Vec<_>>();
    let mut warmed_stores = build_warmed_claim_stores_batched(
        &setup,
        committed_bulk_update_batches,
        &required_source_indices,
    );
    let source_store = warmed_stores.remove(&0).expect("source registry 0 warmed");
    let kappa_halve_plan = source_store
        .prepare_kappa_halve_one_claim_with_partner(target_s, &helper_target_indices)
        .expect("prepared kappa-halve plan");
    let segment_count = source_store.segments();
    let segs = (0..segment_count).collect();
    OneServerKappaHalveFixture {
        source_store,
        helper_stores: warmed_stores,
        source_params: setup.params,
        target_params,
        target_s,
        target_need,
        target_row_labels,
        segs,
        partner_source_registry_index,
        partner_helper_registry_index,
        helper_target_indices,
        kappa_halve_plan,
    }
}

impl OneServerKappaDoubleFixture {
    fn helper_eval_for_target_from_store(
        &self,
        target_registry_index: usize,
        helper: &UnauthStore,
    ) -> KappaDoubleRowEvalMigration {
        assert_eq!(
            helper.claim_bundles().map(|claims| claims.len()),
            Some(1),
            "helper benchmark expects one claim"
        );
        let values_by_seg = helper
            .handle_eval_kappa_double_target_at_row_labels_matrix(
                self.target_s,
                &self.segs,
                &[target_registry_index],
                &self.target_row_labels,
            )
            .expect("helper kappa-double target eval");
        let segments = values_by_seg
            .into_iter()
            .map(|per_seg| {
                assert_eq!(per_seg.len(), 1);
                per_seg.into_iter().next().expect("one target")
            })
            .collect();
        KappaDoubleRowEvalMigration {
            registry_index: target_registry_index,
            segments,
        }
    }

    fn helper_eval_for_target(&self, target_registry_index: usize) -> KappaDoubleRowEvalMigration {
        let source_parent = kappa_halve_parent_registry_index(target_registry_index);
        let helper = self
            .helper_stores
            .get(&source_parent)
            .expect("warmed helper parent");
        self.helper_eval_for_target_from_store(target_registry_index, helper)
    }

    fn collect_helper_evals_once(&self) -> Vec<KappaDoubleRowEvalMigration> {
        self.helper_target_indices
            .iter()
            .copied()
            .map(|target_registry_index| self.helper_eval_for_target(target_registry_index))
            .collect()
    }

    fn materialize_once(&self) -> (UnauthStore, Vec<ClaimBundle>) {
        self.source_store
            .bench_materialize_kappa_double_one_claim_left_with_plan(&self.kappa_double_plan)
            .expect("materialize retained child")
    }

    fn hydrate_once(
        &self,
        helper_evals: &[KappaDoubleRowEvalMigration],
        _target_claims: &[ClaimBundle],
    ) -> UnauthStore {
        let (mut target_store, _) = self.materialize_once();
        target_store
            .bench_hydrate_kappa_double_one_claim_left_rows_with_plan(
                helper_evals,
                &self.kappa_double_plan,
            )
            .expect("hydrate retained rows");
        target_store
    }

    fn kappa_double_total_once(&self, helper_evals: &[KappaDoubleRowEvalMigration]) -> UnauthStore {
        let mut store = self.source_store.clone();
        store
            .migrate_kappa_double_one_claim_left_with_plan(&self.kappa_double_plan, helper_evals)
            .expect("one-server kappa-double");
        store
    }

    fn segment_double_once(&self) -> UnauthStore {
        let mut store = self.source_store.clone();
        store.segment_double().expect("one-server segment-double");
        store
    }

    fn validate(
        &self,
        helper_evals: &[KappaDoubleRowEvalMigration],
        target_claims: &[ClaimBundle],
    ) {
        let phased = self.hydrate_once(helper_evals, target_claims);
        let total = self.kappa_double_total_once(helper_evals);
        assert_eq!(phased.len(), total.len());
        assert_eq!(phased.segments(), total.segments());
        assert!(phased.coded.equals_for_tests(&phased.params, &total.coded));

        let segment_doubled = self.segment_double_once();
        let segment_double_params =
            build_segment_double_params(&self.source_params).expect("segment-double params");
        assert_eq!(
            segment_doubled.params.segment_len,
            segment_double_params.segment_len
        );
        assert_eq!(segment_doubled.params.n_grid, segment_double_params.n_grid);
        assert_eq!(segment_doubled.params.s, segment_double_params.s);
        assert_eq!(segment_doubled.params.need, segment_double_params.need);
        assert_eq!(
            segment_doubled.params.layout_base_n_grid,
            segment_double_params.layout_base_n_grid
        );
        assert_eq!(
            segment_doubled.params.layout_level,
            segment_double_params.layout_level
        );
        assert_eq!(segment_doubled.len(), self.source_store.len());
        assert_eq!(
            segment_doubled.segments(),
            self.source_store.segments().div_ceil(2)
        );
        assert_eq!(segment_doubled.params.s, self.source_params.s);
        assert_eq!(segment_doubled.params.need, self.source_params.need * 2);
        assert_eq!(segment_doubled.params.n_grid, self.source_params.n_grid * 2);
    }
}

impl OneServerKappaHalveFixture {
    fn collect_primary_segments(&self, helper: &UnauthStore, key: ShardKey) -> Vec<Vec<Fp>> {
        self.segs
            .iter()
            .map(|&seg| {
                helper
                    .coded
                    .shard_segment(&self.source_params, &key, seg)
                    .expect("primary segment material")
                    .to_vec()
            })
            .collect()
    }

    fn partner_primary_from_store(&self, helper: &UnauthStore) -> PartnerShardMigration {
        assert_eq!(
            helper.claim_bundles().map(|claims| claims.len()),
            Some(1),
            "helper benchmark expects one claim"
        );
        let partner_entry = self.source_params.shard_registry[self.partner_helper_registry_index];
        let partner_segments = self.collect_primary_segments(helper, partner_entry.key);
        PartnerShardMigration {
            registry_index: self.partner_source_registry_index,
            segments: partner_segments,
        }
    }

    fn partner_primary_once(&self) -> PartnerShardMigration {
        let helper = self
            .helper_stores
            .get(&self.partner_helper_registry_index)
            .expect("partner helper store");
        self.partner_primary_from_store(helper)
    }

    fn helper_eval_for_target_from_source_pair(
        &self,
        target_registry_index: usize,
    ) -> TargetRowEvalMigration {
        let left_source_index = 2 * target_registry_index;
        let right_source_index = left_source_index + 1;
        let left_entry = self.source_params.shard_registry[left_source_index];
        let right_entry = self.source_params.shard_registry[right_source_index];
        let left_store = self
            .helper_stores
            .get(&left_source_index)
            .expect("left source helper store");
        let right_store = self
            .helper_stores
            .get(&right_source_index)
            .expect("right source helper store");
        let target_entry = self.target_params.shard_registry[target_registry_index];
        let mut segments = Vec::with_capacity(self.segs.len());
        for &seg in &self.segs {
            let left = left_store
                .coded
                .shard_segment(&self.source_params, &left_entry.key, seg)
                .expect("left source primary segment");
            let right = right_store
                .coded
                .shard_segment(&self.source_params, &right_entry.key, seg)
                .expect("right source primary segment");
            let mut merged = Vec::with_capacity(left.len() + right.len());
            for (l, r) in left.iter().zip(right.iter()) {
                merged.push(*l);
                merged.push(*r);
            }
            let evals = shard_value_from_vector_batch(
                &self.target_params,
                target_entry,
                &merged,
                &self.target_row_labels,
            )
            .into_iter()
            .map(|value| value.expect("target row eval"))
            .collect();
            segments.push(evals);
        }
        TargetRowEvalMigration {
            registry_index: target_registry_index,
            segments,
        }
    }

    fn collect_helper_evals_once(&self) -> Vec<TargetRowEvalMigration> {
        self.helper_target_indices
            .iter()
            .copied()
            .map(|target_registry_index| {
                self.helper_eval_for_target_from_source_pair(target_registry_index)
            })
            .collect()
    }

    fn materialize_primary_once(&self, partner: &PartnerShardMigration) -> UnauthStore {
        self.source_store
            .bench_materialize_kappa_halve_one_claim_with_partner(
                &self.kappa_halve_plan,
                &[partner.clone()],
            )
            .expect("materialize direct kappa-halve primary")
    }

    fn hydrate_rows_once(
        &self,
        mut store: UnauthStore,
        helper_evals: &[TargetRowEvalMigration],
    ) -> UnauthStore {
        store
            .bench_hydrate_kappa_halve_one_claim_rows_with_plan(
                helper_evals,
                &self.kappa_halve_plan,
            )
            .expect("hydrate direct kappa-halve rows");
        store
    }

    fn kappa_halve_prepared_total_once(
        &self,
        partner: &PartnerShardMigration,
        helper_evals: &[TargetRowEvalMigration],
    ) -> UnauthStore {
        let store = self.materialize_primary_once(partner);
        self.hydrate_rows_once(store, helper_evals)
    }

    fn validate(&self, partner: &PartnerShardMigration, helper_evals: &[TargetRowEvalMigration]) {
        let materialized = self.materialize_primary_once(partner);
        let phased = self.hydrate_rows_once(materialized, helper_evals);
        let total = self.kappa_halve_prepared_total_once(partner, helper_evals);
        assert_eq!(phased.len(), total.len());
        assert_eq!(phased.segments(), total.segments());
        assert!(phased.coded.equals_for_tests(&phased.params, &total.coded));
    }
}

fn bench_one_server_kappa_rescale(
    trials: usize,
    committed_bulk_update_batches: usize,
    metrics: &BTreeSet<String>,
) -> BenchEntry {
    let segment_len = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_KAPPA_RESCALE_SEGMENT_LEN",
        DEFAULT_KAPPA_RESCALE_SEGMENT_LEN,
    );
    let source_s = parse_usize_env(
        "BENCH_APPEND_ONLY_LOG_KAPPA_RESCALE_SOURCE_S",
        DEFAULT_KAPPA_RESCALE_SOURCE_S,
    );
    let fixture = build_fixture(segment_len, source_s, committed_bulk_update_batches);
    let helper_evals_template = fixture.collect_helper_evals_once();
    let (materialized_template, target_claims) = fixture.materialize_once();
    fixture.validate(&helper_evals_template, &target_claims);
    black_box(&materialized_template);

    let mut one_helper_samples = Vec::with_capacity(trials);
    let first_helper_target = fixture.helper_target_indices[0];
    let first_helper_source_parent = kappa_halve_parent_registry_index(first_helper_target);
    let first_helper_store = fixture
        .helper_stores
        .get(&first_helper_source_parent)
        .expect("first helper store");
    if metrics.contains("helper_double") {
        for _ in 0..trials {
            let start = Instant::now();
            let reply =
                fixture.helper_eval_for_target_from_store(first_helper_target, first_helper_store);
            let encoded = serialize_kappa_double_reply(&reply);
            one_helper_samples.push(start.elapsed().as_nanos() as u64);
            black_box(encoded);
        }
    }

    let mut all_helper_samples = Vec::with_capacity(trials);
    if metrics.contains("all_helpers_double") {
        for _ in 0..trials {
            let mut elapsed_ns = 0u64;
            for &target_registry_index in &fixture.helper_target_indices {
                let source_parent = kappa_halve_parent_registry_index(target_registry_index);
                let helper = fixture
                    .helper_stores
                    .get(&source_parent)
                    .expect("helper store");
                let start = Instant::now();
                let reply =
                    fixture.helper_eval_for_target_from_store(target_registry_index, helper);
                elapsed_ns = elapsed_ns.saturating_add(start.elapsed().as_nanos() as u64);
                black_box(reply);
            }
            all_helper_samples.push(elapsed_ns);
        }
    }

    let mut materialize_samples = Vec::with_capacity(trials);
    if metrics.contains("component_double") {
        for _ in 0..trials {
            let start = Instant::now();
            let materialized = fixture.materialize_once();
            materialize_samples.push(start.elapsed().as_nanos() as u64);
            black_box(materialized);
        }
    }

    let mut hydrate_samples = Vec::with_capacity(trials);
    if metrics.contains("component_double") {
        for _ in 0..trials {
            let start = Instant::now();
            let hydrated = fixture.hydrate_once(&helper_evals_template, &target_claims);
            hydrate_samples.push(start.elapsed().as_nanos() as u64);
            black_box(hydrated);
        }
    }

    let mut total_samples = Vec::with_capacity(trials);
    if metrics.contains("receiver_double") {
        for _ in 0..trials {
            let start = Instant::now();
            let total = fixture.kappa_double_total_once(&helper_evals_template);
            total_samples.push(start.elapsed().as_nanos() as u64);
            black_box(total);
        }
    }

    let segment_double_template = fixture.segment_double_once();
    let mut segment_double_samples = Vec::with_capacity(trials);
    if metrics.contains("segment_double") {
        for _ in 0..trials {
            let start = Instant::now();
            let segment_doubled = fixture.segment_double_once();
            segment_double_samples.push(start.elapsed().as_nanos() as u64);
            black_box(segment_doubled);
        }
    }

    let helper_eval_scalars_total = fixture.helper_target_indices.len() as u64
        * fixture.segs.len() as u64
        * fixture.target_row_labels.len() as u64;

    let kappa_halve_fixture =
        build_kappa_halve_fixture(segment_len, fixture.target_s, committed_bulk_update_batches);
    let kappa_halve_partner_template = kappa_halve_fixture.partner_primary_once();
    let kappa_halve_helper_evals_template = kappa_halve_fixture.collect_helper_evals_once();
    kappa_halve_fixture.validate(
        &kappa_halve_partner_template,
        &kappa_halve_helper_evals_template,
    );
    let kappa_halve_materialized_template =
        kappa_halve_fixture.materialize_primary_once(&kappa_halve_partner_template);

    let mut kappa_halve_one_helper_samples = Vec::with_capacity(trials);
    let first_kappa_halve_helper_index = kappa_halve_fixture.helper_target_indices[0];
    if metrics.contains("helper_halve") {
        for _ in 0..trials {
            let start = Instant::now();
            let reply = kappa_halve_fixture
                .helper_eval_for_target_from_source_pair(first_kappa_halve_helper_index);
            let encoded = serialize_kappa_halve_reply(&reply);
            kappa_halve_one_helper_samples.push(start.elapsed().as_nanos() as u64);
            black_box(encoded);
        }
    }

    let mut kappa_halve_all_helper_samples = Vec::with_capacity(trials);
    if metrics.contains("all_helpers_halve") {
        for _ in 0..trials {
            let mut elapsed_ns = 0u64;
            for &target_registry_index in &kappa_halve_fixture.helper_target_indices {
                let start = Instant::now();
                let reply = kappa_halve_fixture
                    .helper_eval_for_target_from_source_pair(target_registry_index);
                elapsed_ns = elapsed_ns.saturating_add(start.elapsed().as_nanos() as u64);
                black_box(reply);
            }
            kappa_halve_all_helper_samples.push(elapsed_ns);
        }
    }

    let mut kappa_halve_hydrate_rows_samples = Vec::with_capacity(trials);
    if metrics.contains("component_halve") {
        for _ in 0..trials {
            let start = Instant::now();
            let hydrated = kappa_halve_fixture.hydrate_rows_once(
                kappa_halve_materialized_template.clone(),
                &kappa_halve_helper_evals_template,
            );
            kappa_halve_hydrate_rows_samples.push(start.elapsed().as_nanos() as u64);
            black_box(hydrated);
        }
    }

    let mut kappa_halve_primary_materialize_samples = Vec::with_capacity(trials);
    if metrics.contains("component_halve") {
        for _ in 0..trials {
            let start = Instant::now();
            let kappa_halved =
                kappa_halve_fixture.materialize_primary_once(&kappa_halve_partner_template);
            kappa_halve_primary_materialize_samples.push(start.elapsed().as_nanos() as u64);
            black_box(kappa_halved);
        }
    }

    let mut kappa_halve_prepared_total_samples = Vec::with_capacity(trials);
    if metrics.contains("receiver_halve") {
        for _ in 0..trials {
            let start = Instant::now();
            let kappa_halved = kappa_halve_fixture.kappa_halve_prepared_total_once(
                &kappa_halve_partner_template,
                &kappa_halve_helper_evals_template,
            );
            kappa_halve_prepared_total_samples.push(start.elapsed().as_nanos() as u64);
            black_box(kappa_halved);
        }
    }

    let kappa_halve_partner_primary_scalars_total =
        kappa_halve_fixture.segs.len() as u64 * kappa_halve_fixture.source_params.s as u64;
    let kappa_halve_helper_eval_scalars_total = kappa_halve_fixture.helper_target_indices.len()
        as u64
        * kappa_halve_fixture.segs.len() as u64
        * kappa_halve_fixture.target_row_labels.len() as u64;

    BenchEntry {
        segment_len,
        source_s,
        target_s: fixture.target_s,
        n_grid: fixture.source_params.n_grid,
        source_need: fixture.source_params.need,
        target_need: fixture.target_params.need,
        source_shard_count: fixture.source_params.shard_registry.len(),
        bulk_updates_bytes: BULK_UPDATES_BYTES,
        bulk_updates_size: BULK_UPDATES_SIZE,
        committed_bulk_update_batches,
        live_unauth_len: fixture.source_store.len(),
        segment_count: fixture.segs.len(),
        target_source_registry_index: 0,
        target_registry_index: fixture.target_registry_index,
        target_row_count: fixture
            .target_claim
            .row_label_indices(&fixture.target_params)
            .expect("target rows")
            .len(),
        helper_target_count: fixture.helper_target_indices.len(),
        helper_eval_scalars_total,
        helper_eval_raw_bytes_total: helper_eval_scalars_total * FIELD_BYTES,
        helper_eval_one_helper_serialized_ns: stats_from_samples(one_helper_samples),
        helper_eval_all_helpers_serial_ns: stats_from_samples(all_helper_samples),
        local_retained_child_materialize_ns: stats_from_samples(materialize_samples),
        local_row_hydrate_install_ns: stats_from_samples(hydrate_samples),
        local_kappa_double_total_ns: stats_from_samples(total_samples),
        segment_double_target_segment_len: segment_double_template.params.segment_len,
        segment_double_target_s: segment_double_template.params.s,
        segment_double_target_need: segment_double_template.params.need,
        segment_double_target_segment_count: segment_double_template.segments(),
        segment_double_helper_eval_raw_bytes_total: 0,
        local_segment_double_total_ns: stats_from_samples(segment_double_samples),
        kappa_halve_source_s: kappa_halve_fixture.source_params.s,
        kappa_halve_target_s: kappa_halve_fixture.target_s,
        kappa_halve_source_need: kappa_halve_fixture.source_params.need,
        kappa_halve_target_need: kappa_halve_fixture.target_need,
        kappa_halve_target_row_count: kappa_halve_fixture.target_row_labels.len(),
        kappa_halve_helper_count: kappa_halve_fixture.helper_target_indices.len() + 1,
        kappa_halve_helper_eval_raw_bytes_total: (kappa_halve_partner_primary_scalars_total
            + kappa_halve_helper_eval_scalars_total)
            * FIELD_BYTES,
        kappa_halve_helper_eval_one_helper_serialized_ns: stats_from_samples(
            kappa_halve_one_helper_samples,
        ),
        kappa_halve_helper_eval_all_helpers_serial_ns: stats_from_samples(
            kappa_halve_all_helper_samples,
        ),
        kappa_halve_primary_materialize_ns: stats_from_samples(
            kappa_halve_primary_materialize_samples,
        ),
        kappa_halve_row_hydrate_install_ns: stats_from_samples(kappa_halve_hydrate_rows_samples),
        kappa_halve_prepared_total_ns: stats_from_samples(kappa_halve_prepared_total_samples),
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
            "helper_double",
            "all_helpers_double",
            "component_double",
            "receiver_double",
            "segment_double",
            "helper_halve",
            "all_helpers_halve",
            "component_halve",
            "receiver_halve",
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
            notes: "local append-only-log one-server reconfiguration: warm real one-claim-per-party append-only-log state, then time selected helper response construction and serialization plus destination-maintainer metrics for adjacent kappa changes and local segment-double; whole-cluster coordination is intentionally excluded".to_string(),
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
