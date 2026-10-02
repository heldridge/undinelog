//! Unauthenticated storage layer built over the coded-array machinery.
//!
//! This module exposes a minimal array-like API that knows nothing about
//! Merkle trees or authentication. Values are stored as coded segments,
//! partitioned by a public segment span of `segment_len == n_grid` values.
//!
//! Client orchestration lives in `crate::unauth::client`, with transport
//! adapters in `crate::net::unauth`.

pub mod api;
pub mod client;
mod kappa_rescale;
pub mod layout;
pub mod service;

pub use crate::coded_array::params::{
    CodedArrayConfig, CodedArrayParams, PaperGeometry, PaperGeometryError, RowFamilyId,
    SetupTimings, ShardKey, ShardRegEntry, build_coded_params,
    build_migrated_party_registry_indices, build_migrated_shard_registry,
    build_shard_registry_deterministic, compute_initial_rows_option_b,
    compute_initial_shards_option_b, derive_paper_geometry, kappa_double_child_registry_indices,
    kappa_halve_parent_registry_index, registry_supports_sibling_kappa_halve,
    validate_coded_config,
};
pub use crate::coded_array::reconstruct::{
    DecodeCache, ReadError as CodedReadError, decode_segment_value_with_cache,
};
pub use crate::coded_array::rs::{
    BarycentricWeights, barycentric_eval, barycentric_eval_pair_many, precompute_barycentric,
};
pub use crate::coded_array::shard_maths::{
    DeltaMode, apply_deltas_to_segment, apply_deltas_to_segment_with_mode, shard_value_from_store,
    shard_value_from_store_batch, shard_value_from_store_batch_on_grid_coset,
    shard_value_from_vector, shard_value_from_vector_batch,
};
pub use crate::coded_array::store::{CodedStore, RowKey};
use crate::field::{Fp, batch_invert, fp_one, fp_pow, fp_zero};
use crate::shares::{RowShare, SegmentPrimaryShard, ShardShare};
use crate::value::{Value256, decode_value, encode_value};
pub use api::{LogicalValueFamilyKey, UnauthArray, UnauthArrayParams, UnauthDecodeParams};
use ark_ff::{FftField, Field};
pub use kappa_rescale::{KappaDoubleOneClaimLeftPlan, KappaDoubleRowEvalMigration};
#[cfg(feature = "bench")]
pub use kappa_rescale::{KappaHalveOneClaimWithPartnerPlan, TargetRowEvalMigration};
pub use layout::{
    SemanticBlockPosition, SemanticPosition, aligned_block_indices, block_alphas,
    block_alphas_for_segment, block_grid_indices, block_in_segment_from_column_coordinate,
    grid_index_to_alpha_for_segment, logical_block_to_column_coordinate,
    logical_block_to_column_coordinate_for_segment, logical_blocks_per_segment,
    logical_offset_to_alpha, logical_offset_to_alpha_for_segment,
    logical_offset_to_column_coordinate, logical_offset_to_column_coordinate_for_segment,
    logical_offset_to_grid_index, logical_offset_to_grid_index_for_s, public_index_to_segment,
    public_values_per_segment, segment_domain_rho, semantic_block_position, semantic_position,
    semantic_position_for_s,
};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;
use std::time::Instant;

thread_local! {
    static ROW_HELPER_SCRATCH: RefCell<RowHelperScratch> = RefCell::new(RowHelperScratch::default());
}

const ROW_INTERPOLATE_THRESHOLD: usize = 16;
const ROW_MATERIALIZE_COEFF_THRESHOLD: usize = 16;
const COSET_INTERPOLATE_THRESHOLD: usize = 64;

#[derive(Default)]
struct RowHelperScratch {
    scale_cache: Vec<Option<Fp>>,
    scale_cache_touched: Vec<usize>,
    coeffs: Vec<Fp>,
    subgroup_values: Vec<Fp>,
    batch_values: Vec<Fp>,
}

/// Logical index into the unauthenticated array.
pub type Index = u64;
/// Encoded 256-bit value stored at each index.
pub type Value = Value256;
/// Delta update at a given index.
pub type Delta = (Index, Fp);
/// Delta update at one offset within an aligned semantic block.
pub type BlockDelta = (usize, Fp);

/// Errors returned when extending storage.
#[derive(Clone, Debug)]
pub enum ExtendError {
    SegmentOverflow,
    InvalidFamilyValues,
}

/// Errors returned when migrating between unauth modes.
#[derive(Clone, Debug)]
pub enum MigrateError {
    InvalidTargetS,
    InvalidSegmentMerge,
    InvalidClaim,
    InvalidPartnerShards,
    MissingPartnerShards,
    MissingLocalRows,
    ExtendFailed(ExtendError),
}

/// Errors returned when writing deltas.
#[derive(Clone, Debug)]
pub enum WriteError {
    InvalidIndex,
}

/// Errors returned by unauthenticated reads.
#[derive(Clone, Debug)]
pub enum ReadError {
    InvalidIndex,
    NotEnoughShards,
}

/// Errors returned when installing recovered claim-bound state.
#[derive(Clone, Debug)]
pub enum InstallClaimStateError {
    MissingClaims,
    InvalidIndex,
    MissingShard(usize),
    MissingRow(usize),
    UnexpectedShard(usize),
    UnexpectedRow(usize),
    InvalidShardLength {
        registry_index: usize,
        expected: usize,
        actual: usize,
    },
    InvalidRowLength {
        row_label_index: usize,
        expected: usize,
        actual: usize,
    },
}

/// Parameters shared by all unauthenticated parties/clients.
#[derive(Clone)]
pub struct UnauthParams {
    pub params: CodedArrayParams,
}

impl UnauthParams {
    /// Build parameters from a coded-array config.
    pub fn from_config(cfg: CodedArrayConfig) -> Self {
        let params = crate::coded_array::params::build_coded_params(&cfg);
        Self { params }
    }
}

/// One local claim bundle: a primary column plus its maintained row family.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ClaimBundle {
    pub registry_index: usize,
    pub row_family_id: RowFamilyId,
}

/// Fp-native recovered state for one claim over one allocated segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveredClaimSegmentState {
    pub shards: Vec<(usize, Vec<Fp>)>,
    pub rows: Vec<(usize, Vec<Fp>)>,
}

/// Prevalidated claim-install shape for one claim-bound store.
#[derive(Clone, Debug)]
pub struct ClaimSegmentInstallPlan {
    expected_shards: HashSet<usize>,
    expected_rows: HashSet<usize>,
    expected_shard_len: usize,
    expected_row_len: usize,
}

/// Result of an oversampled shard decode that identified inconsistent helpers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CorrectedShardDecode {
    pub value: Fp,
    pub bad_registry_indices: Vec<usize>,
    pub share_count: usize,
}

/// Number of distinct helper shares to request for an error-correcting read.
///
/// Correcting `e` arbitrary bad Reed-Solomon replies requires `need + 2e`
/// distinct shares.
pub fn error_correcting_share_target(need: usize, max_errors: usize) -> usize {
    need.saturating_add(max_errors.saturating_mul(2))
}

/// Client-side plan for reading maintained-array indices through a compact
/// segment/point object cover.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnauthOptimizedReadPlan {
    pub requested_indices: Vec<Index>,
    pub len: Index,
    pub share_target: usize,
    pub max_errors: Option<usize>,
    pub block_row_target: usize,
    pub segment_shard_len: usize,
    pub registry_len: usize,
    pub request: UnauthOptimizedReadRequest,
    pub cover: UnauthOptimizedReadCover,
}

/// Ordered server-facing unauth read objects. Servers serve this request as-is
/// and do not re-plan from local state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnauthOptimizedReadRequest {
    pub objects: Vec<UnauthReadObjectId>,
}

/// Internal cover buckets selected by the unauth planner.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UnauthOptimizedReadCover {
    pub segments: Vec<usize>,
    pub blocks: Vec<Index>,
    pub points: Vec<Index>,
}

/// Compact identity for one planned unauth read object.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnauthReadObjectId {
    Segment { segment: usize },
    Block { block: Index },
    Point { array_index: Index },
}

/// Contributions from one helper for a planned optimized read request.
#[derive(Clone, Debug, Default)]
pub struct UnauthOptimizedReadResponse {
    /// Client-local helper position for this response, when known.
    ///
    /// This is not a wire field. Network clients attach it after receiving the
    /// reply so later client-side correction can quarantine a whole helper.
    pub helper_index: Option<usize>,
    pub objects: Vec<UnauthReadObjectContribution>,
}

/// Bad optimized-read contribution quarantine set tracked by the client.
#[derive(Clone, Debug, Default)]
pub struct UnauthReadBadFilter {
    pub helper_indices: HashSet<usize>,
    pub registry_indices: HashSet<usize>,
}

impl UnauthReadBadFilter {
    fn known_bad_count(&self) -> usize {
        self.helper_indices.len().max(self.registry_indices.len())
    }

    pub fn clear(&mut self) {
        self.helper_indices.clear();
        self.registry_indices.clear();
    }

    pub fn extend(&mut self, other: UnauthReadBadFilter) {
        self.helper_indices.extend(other.helper_indices);
        self.registry_indices.extend(other.registry_indices);
    }

    pub fn overlaps(&self, other: &UnauthReadBadFilter) -> bool {
        self.helper_indices
            .iter()
            .any(|helper_index| other.helper_indices.contains(helper_index))
            || self
                .registry_indices
                .iter()
                .any(|registry_index| other.registry_indices.contains(registry_index))
    }
}

/// Candidate values decoded from one optimized-read object.
///
/// Candidate decode is intentionally a cheap threshold decode. It records only
/// the helpers and registry indices used to form the candidate; the caller must
/// certify the candidate values against its own commitment before keeping them.
pub struct UnauthCandidateObjectDecode {
    pub values: Vec<(Index, Value256)>,
    pub provenance: UnauthReadBadFilter,
}

struct SegmentShardSelection {
    shards: Vec<SegmentPrimaryShard>,
    provenance: UnauthReadBadFilter,
}

struct PointShareSelection {
    shares: Vec<ShardShare>,
    provenance: UnauthReadBadFilter,
}

struct BlockRowSelection {
    rows: Vec<RowShare>,
    helper_indices: HashSet<usize>,
}

/// One helper contribution keyed by the object position in the client plan.
#[derive(Clone, Debug)]
pub enum UnauthReadObjectContribution {
    Segment {
        object_pos: usize,
        shards: Vec<SegmentPrimaryShard>,
    },
    Block {
        object_pos: usize,
        rows: Vec<RowShare>,
    },
    Point {
        object_pos: usize,
        shares: Vec<ShardShare>,
    },
}

#[derive(Clone, Copy)]
struct IndexedReadContribution<'a> {
    helper_index: Option<usize>,
    contribution: &'a UnauthReadObjectContribution,
}

pub(crate) struct UnauthOptimizedResponseIndex<'a> {
    by_object: Vec<Vec<IndexedReadContribution<'a>>>,
}

impl<'a> UnauthOptimizedResponseIndex<'a> {
    pub(crate) fn new(object_count: usize, responses: &'a [UnauthOptimizedReadResponse]) -> Self {
        let mut by_object = (0..object_count).map(|_| Vec::new()).collect::<Vec<_>>();
        for response in responses {
            for contribution in &response.objects {
                let object_pos = contribution_object_pos(contribution);
                if let Some(entries) = by_object.get_mut(object_pos) {
                    entries.push(IndexedReadContribution {
                        helper_index: response.helper_index,
                        contribution,
                    });
                }
            }
        }
        Self { by_object }
    }

    fn contributions(&self, object_pos: usize) -> &[IndexedReadContribution<'a>] {
        self.by_object
            .get(object_pos)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

fn contribution_object_pos(contribution: &UnauthReadObjectContribution) -> usize {
    match contribution {
        UnauthReadObjectContribution::Segment { object_pos, .. }
        | UnauthReadObjectContribution::Block { object_pos, .. }
        | UnauthReadObjectContribution::Point { object_pos, .. } => *object_pos,
    }
}

fn helper_is_filtered(
    helper_index: Option<usize>,
    bad_filter: Option<&UnauthReadBadFilter>,
) -> bool {
    bad_filter.is_some_and(|filter| {
        helper_index.is_some_and(|helper_index| filter.helper_indices.contains(&helper_index))
    })
}

/// Build a deterministic segment/point cover for maintained-array reads.
///
/// `requested_indices` in the returned plan are sorted and deduplicated. Values
/// decoded from the plan are returned in exactly that order. Indices outside
/// `len` stay in `requested_indices` but are not fetched; they decode as the
/// unauth default value.
pub fn plan_optimized_read_indices(
    params: &CodedArrayParams,
    len: Index,
    indices: &[Index],
    share_target: usize,
    max_errors: Option<usize>,
) -> UnauthOptimizedReadPlan {
    let share_target = usize::max(
        share_target,
        match max_errors {
            Some(max_errors) => error_correcting_share_target(params.need, max_errors),
            None => params.need,
        },
    );
    let mut requested_indices = indices.to_vec();
    requested_indices.sort_unstable();
    requested_indices.dedup();

    let mut by_segment: BTreeMap<usize, Vec<Index>> = BTreeMap::new();
    for &array_index in &requested_indices {
        if array_index >= len {
            continue;
        }
        let (segment, _) = public_index_to_segment(params, array_index);
        by_segment.entry(segment).or_default().push(array_index);
    }

    let mut cover = UnauthOptimizedReadCover::default();
    let segment_cost_cells = params.s;

    for (segment, segment_indices) in by_segment {
        let needed_in_segment = segment_indices.len();
        // Costs are all multiplied by `share_target`, so the segment threshold
        // reduces to a requested-cell count. The range API starts from complete
        // segments and point leftovers; row-backed aligned blocks remain an
        // explicit lower-level unauth object, not an automatic range tier.
        if needed_in_segment >= segment_cost_cells {
            cover.segments.push(segment);
            continue;
        }
        cover.points.extend(segment_indices);
    }

    cover.segments.sort_unstable();
    cover.segments.dedup();
    cover.blocks.sort_unstable();
    cover.blocks.dedup();
    cover.points.sort_unstable();
    cover.points.dedup();

    let mut objects =
        Vec::with_capacity(cover.segments.len() + cover.blocks.len() + cover.points.len());
    objects.extend(
        cover
            .segments
            .iter()
            .copied()
            .map(|segment| UnauthReadObjectId::Segment { segment }),
    );
    objects.extend(
        cover
            .blocks
            .iter()
            .copied()
            .map(|block| UnauthReadObjectId::Block { block }),
    );
    objects.extend(
        cover
            .points
            .iter()
            .copied()
            .map(|array_index| UnauthReadObjectId::Point { array_index }),
    );

    UnauthOptimizedReadPlan {
        requested_indices,
        len,
        share_target,
        max_errors,
        block_row_target: match max_errors {
            Some(max_errors) => error_correcting_share_target(params.s, max_errors),
            None => params.s,
        },
        segment_shard_len: params.s,
        registry_len: params.shard_registry.len(),
        request: UnauthOptimizedReadRequest { objects },
        cover,
    }
}

/// Serve the ordered objects in an optimized unauth read request from a store.
pub fn handle_optimized_read_request(
    store: &UnauthStore,
    request: &UnauthOptimizedReadRequest,
) -> UnauthOptimizedReadResponse {
    let mut objects = Vec::with_capacity(request.objects.len());
    for (object_pos, object) in request.objects.iter().enumerate() {
        match *object {
            UnauthReadObjectId::Segment { segment } => {
                objects.push(UnauthReadObjectContribution::Segment {
                    object_pos,
                    shards: store.handle_read_segment_raw(segment),
                });
            }
            UnauthReadObjectId::Block { block } => {
                objects.push(UnauthReadObjectContribution::Block {
                    object_pos,
                    rows: store.handle_read_block(block),
                });
            }
            UnauthReadObjectId::Point { array_index } => {
                objects.push(UnauthReadObjectContribution::Point {
                    object_pos,
                    shares: store.handle_read(array_index),
                });
            }
        }
    }
    UnauthOptimizedReadResponse {
        helper_index: None,
        objects,
    }
}

/// Return true once the collected opaque helper replies satisfy every planned
/// unauth object.
pub fn optimized_read_satisfied(
    plan: &UnauthOptimizedReadPlan,
    responses: &[UnauthOptimizedReadResponse],
) -> bool {
    let response_index = UnauthOptimizedResponseIndex::new(plan.request.objects.len(), responses);
    for (object_pos, object) in plan.request.objects.iter().enumerate() {
        match object {
            UnauthReadObjectId::Segment { .. } => {
                if count_segment_shards_for_object(
                    plan,
                    &response_index,
                    object_pos,
                    plan.share_target,
                    None,
                ) < plan.share_target
                {
                    return false;
                }
            }
            UnauthReadObjectId::Block { .. } => {
                if count_block_rows_for_object(
                    &response_index,
                    object_pos,
                    plan.block_row_target,
                    None,
                ) < plan.block_row_target
                {
                    return false;
                }
            }
            UnauthReadObjectId::Point { .. } => {
                if count_point_shares_for_object(
                    &response_index,
                    object_pos,
                    plan.share_target,
                    None,
                ) < plan.share_target
                {
                    return false;
                }
            }
        }
    }
    true
}

/// Decode an optimized unauth read plan into values ordered by
/// `plan.requested_indices`.
///
/// Segment objects are decoded in bulk from primary slices through the
/// two-variable lens. Correcting segment-object decode treats bad helpers as
/// bad primary columns: it finds shared bad-column positions at segment
/// granularity, then reconstructs the pinned rows from the remaining columns.
/// Callers that authenticate decoded labels still rely on the final root check
/// as the end-to-end correctness gate.
pub fn decode_optimized_read_plan(
    params: &CodedArrayParams,
    plan: &UnauthOptimizedReadPlan,
    responses: &[UnauthOptimizedReadResponse],
) -> Result<Vec<Value256>, CodedReadError> {
    decode_optimized_read_plan_internal(
        params,
        plan,
        responses,
        plan.max_errors.is_some(),
        UnauthReadBadFilter::default(),
    )
}

/// Return object positions in the order a client should try local correction.
///
/// Segment objects come first because one segment-object correction can repair
/// every requested value covered by that object.
pub fn optimized_read_object_correction_order(plan: &UnauthOptimizedReadPlan) -> Vec<usize> {
    let mut object_positions = plan
        .request
        .objects
        .iter()
        .enumerate()
        .filter_map(|(object_pos, object)| {
            matches!(object, UnauthReadObjectId::Segment { .. }).then_some(object_pos)
        })
        .collect::<Vec<_>>();
    object_positions.extend(plan.request.objects.iter().enumerate().filter_map(
        |(object_pos, object)| {
            (!matches!(object, UnauthReadObjectId::Segment { .. })).then_some(object_pos)
        },
    ));
    object_positions
}

fn planned_object_pos_covering_index(
    params: &CodedArrayParams,
    plan: &UnauthOptimizedReadPlan,
    array_index: Index,
    skipped_object_positions: Option<&HashSet<usize>>,
) -> Option<usize> {
    if array_index >= plan.len {
        return None;
    }
    let (segment, _) = public_index_to_segment(params, array_index);
    if let Ok(pos) = plan.cover.segments.binary_search(&segment) {
        if !skipped_object_positions.is_some_and(|skipped| skipped.contains(&pos)) {
            return Some(pos);
        }
    }

    let block_offset = plan.cover.segments.len();
    for (block_pos, &block) in plan.cover.blocks.iter().enumerate() {
        let object_pos = block_offset + block_pos;
        if skipped_object_positions.is_some_and(|skipped| skipped.contains(&object_pos)) {
            continue;
        }
        if aligned_block_indices(params, block, plan.len)
            .into_iter()
            .flatten()
            .any(|idx| idx == array_index)
        {
            return Some(object_pos);
        }
    }

    let point_offset = block_offset + plan.cover.blocks.len();
    plan.cover
        .points
        .binary_search(&array_index)
        .ok()
        .map(|point_pos| point_offset + point_pos)
        .filter(|object_pos| {
            !skipped_object_positions.is_some_and(|skipped| skipped.contains(object_pos))
        })
}

/// Decode one planned optimized-read object into `(array_index, value)` pairs.
///
/// The object identity, covered indices, and contribution shape remain unauth
/// details. Callers only receive decoded maintained-array values for the
/// requested indices covered by that object.
pub fn decode_optimized_read_object_values(
    params: &CodedArrayParams,
    plan: &UnauthOptimizedReadPlan,
    responses: &[UnauthOptimizedReadResponse],
    object_pos: usize,
    bad_filter: Option<&mut UnauthReadBadFilter>,
) -> Result<Vec<(Index, Value256)>, CodedReadError> {
    let response_index = UnauthOptimizedResponseIndex::new(plan.request.objects.len(), responses);
    decode_optimized_read_object_values_indexed(
        params,
        plan,
        &response_index,
        object_pos,
        bad_filter,
    )
}

pub(crate) fn decode_optimized_read_object_values_indexed(
    params: &CodedArrayParams,
    plan: &UnauthOptimizedReadPlan,
    response_index: &UnauthOptimizedResponseIndex<'_>,
    object_pos: usize,
    bad_filter: Option<&mut UnauthReadBadFilter>,
) -> Result<Vec<(Index, Value256)>, CodedReadError> {
    let mut bad_filter = bad_filter;
    let object = plan
        .request
        .objects
        .get(object_pos)
        .ok_or(CodedReadError::NotEnoughShards)?;
    let decode_cache = DecodeCache::new();

    match *object {
        UnauthReadObjectId::Segment { segment } => {
            let segment_indices = requested_indices_for_segment(params, plan, segment);
            let shards = collect_segment_shards_for_object(
                plan,
                response_index,
                object_pos,
                plan.share_target,
                bad_filter.as_deref(),
            );
            let remaining_errors = plan.max_errors.map(|max_errors| {
                max_errors.saturating_sub(
                    bad_filter
                        .as_ref()
                        .map(|filter| filter.known_bad_count())
                        .unwrap_or(0),
                )
            });
            let registry_before = bad_filter
                .as_ref()
                .map(|filter| filter.registry_indices.clone())
                .unwrap_or_default();
            let decoded = decode_segment_object_values(
                params,
                segment,
                &segment_indices,
                &shards,
                remaining_errors,
                &decode_cache,
                bad_filter
                    .as_deref_mut()
                    .map(|filter| &mut filter.registry_indices),
            )?;
            promote_new_bad_registry_indices_to_helpers(
                response_index,
                object_pos,
                &registry_before,
                bad_filter.as_deref_mut(),
            );
            if decoded.len() != segment_indices.len() {
                return Err(CodedReadError::NotEnoughShards);
            }
            Ok(segment_indices.into_iter().zip(decoded).collect())
        }
        UnauthReadObjectId::Block { block } => {
            let requested_set: HashSet<Index> = plan.requested_indices.iter().copied().collect();
            let rows = collect_block_rows_for_object(
                response_index,
                object_pos,
                plan.block_row_target,
                bad_filter.as_deref(),
            );
            let decoded = match plan.max_errors {
                Some(max_errors) => {
                    let remaining_errors = max_errors.saturating_sub(
                        bad_filter
                            .as_ref()
                            .map(|filter| filter.known_bad_count())
                            .unwrap_or(0),
                    );
                    if remaining_errors == 0 {
                        decode_block_from_rows(params, block, &rows)?
                    } else {
                        decode_block_from_rows_correcting(params, block, &rows, remaining_errors)?
                    }
                }
                None => decode_block_from_rows(params, block, &rows)?,
            };
            let mut out = Vec::new();
            for (offset, maybe_index) in aligned_block_indices(params, block, plan.len)
                .into_iter()
                .enumerate()
            {
                let Some(array_index) = maybe_index else {
                    continue;
                };
                if requested_set.contains(&array_index) {
                    out.push((array_index, encode_value(decoded[offset])));
                }
            }
            Ok(out)
        }
        UnauthReadObjectId::Point { array_index } => {
            let shares = collect_point_shares_for_object(
                response_index,
                object_pos,
                plan.share_target,
                bad_filter.as_deref(),
            );
            let (_, logical_offset) = public_index_to_segment(params, array_index);
            let z_eval = params.decode_index_eval(array_index);
            let value = match plan.max_errors {
                Some(max_errors) => {
                    let remaining_errors = bad_filter
                        .as_ref()
                        .map(|filter| max_errors.saturating_sub(filter.known_bad_count()))
                        .unwrap_or(max_errors);
                    let available_errors = shares.len().saturating_sub(params.need) / 2;
                    let decode_errors = remaining_errors.min(available_errors);
                    if decode_errors == 0 {
                        decode_from_shares_at_z_with_cache(
                            params,
                            z_eval,
                            &shares,
                            Some(&decode_cache),
                        )?
                    } else {
                        let decoded = decode_from_shares_at_z_with_error_correction(
                            params,
                            z_eval,
                            &shares,
                            decode_errors,
                        )?;
                        if let Some(filter) = bad_filter.as_deref_mut() {
                            let registry_before = filter.registry_indices.clone();
                            filter.registry_indices.extend(decoded.bad_registry_indices);
                            promote_new_bad_registry_indices_to_helpers(
                                response_index,
                                object_pos,
                                &registry_before,
                                Some(filter),
                            );
                        }
                        decoded.value
                    }
                }
                None => decode_from_shares_at_z_with_cache(
                    params,
                    z_eval,
                    &shares,
                    Some(&decode_cache),
                )?,
            };
            debug_assert_eq!(
                z_eval,
                logical_offset_to_column_coordinate_for_segment(
                    params,
                    public_index_to_segment(params, array_index).0,
                    logical_offset
                )
            );
            Ok(vec![(array_index, encode_value(value))])
        }
    }
}

/// Decode the planned object that covers a requested maintained-array index.
///
/// The returned object position is an opaque handle the caller can use only to
/// avoid retrying the same object during local diagnosis.
pub fn decode_optimized_read_object_covering_index(
    params: &CodedArrayParams,
    plan: &UnauthOptimizedReadPlan,
    responses: &[UnauthOptimizedReadResponse],
    array_index: Index,
    skipped_object_positions: &HashSet<usize>,
    bad_filter: Option<&mut UnauthReadBadFilter>,
) -> Result<(usize, Vec<(Index, Value256)>), CodedReadError> {
    let response_index = UnauthOptimizedResponseIndex::new(plan.request.objects.len(), responses);
    decode_optimized_read_object_covering_index_indexed(
        params,
        plan,
        &response_index,
        array_index,
        skipped_object_positions,
        bad_filter,
    )
}

pub(crate) fn decode_optimized_read_object_covering_index_indexed(
    params: &CodedArrayParams,
    plan: &UnauthOptimizedReadPlan,
    response_index: &UnauthOptimizedResponseIndex<'_>,
    array_index: Index,
    skipped_object_positions: &HashSet<usize>,
    bad_filter: Option<&mut UnauthReadBadFilter>,
) -> Result<(usize, Vec<(Index, Value256)>), CodedReadError> {
    let mut bad_filter = bad_filter;
    let object_pos = planned_object_pos_covering_index(
        params,
        plan,
        array_index,
        Some(skipped_object_positions),
    )
    .ok_or(CodedReadError::NotEnoughShards)?;
    let decoded = decode_optimized_read_object_values_indexed(
        params,
        plan,
        response_index,
        object_pos,
        bad_filter.as_deref_mut(),
    )?;
    Ok((object_pos, decoded))
}

/// Decode a candidate object using a cheap threshold decode.
pub fn decode_optimized_read_object_candidate_covering_index(
    params: &CodedArrayParams,
    plan: &UnauthOptimizedReadPlan,
    responses: &[UnauthOptimizedReadResponse],
    array_index: Index,
    bad_filter: Option<&UnauthReadBadFilter>,
    decode_cache: &DecodeCache,
) -> Result<(usize, UnauthCandidateObjectDecode), CodedReadError> {
    let response_index = UnauthOptimizedResponseIndex::new(plan.request.objects.len(), responses);
    decode_optimized_read_object_candidate_covering_index_indexed(
        params,
        plan,
        &response_index,
        array_index,
        bad_filter,
        decode_cache,
    )
}

pub(crate) fn decode_optimized_read_object_candidate_covering_index_indexed(
    params: &CodedArrayParams,
    plan: &UnauthOptimizedReadPlan,
    response_index: &UnauthOptimizedResponseIndex<'_>,
    array_index: Index,
    bad_filter: Option<&UnauthReadBadFilter>,
    decode_cache: &DecodeCache,
) -> Result<(usize, UnauthCandidateObjectDecode), CodedReadError> {
    let object_pos = planned_object_pos_covering_index(params, plan, array_index, None)
        .ok_or(CodedReadError::NotEnoughShards)?;
    let decoded = decode_optimized_read_object_candidate_values(
        params,
        plan,
        response_index,
        object_pos,
        bad_filter,
        decode_cache,
    )?;
    Ok((object_pos, decoded))
}

fn decode_optimized_read_object_candidate_values(
    params: &CodedArrayParams,
    plan: &UnauthOptimizedReadPlan,
    response_index: &UnauthOptimizedResponseIndex<'_>,
    object_pos: usize,
    bad_filter: Option<&UnauthReadBadFilter>,
    decode_cache: &DecodeCache,
) -> Result<UnauthCandidateObjectDecode, CodedReadError> {
    let object = plan
        .request
        .objects
        .get(object_pos)
        .ok_or(CodedReadError::NotEnoughShards)?;

    match *object {
        UnauthReadObjectId::Segment { segment } => {
            let segment_indices = requested_indices_for_segment(params, plan, segment);
            let selection = collect_segment_shards_for_object_with_provenance(
                plan,
                response_index,
                object_pos,
                params.need,
                bad_filter,
            );
            let values = decode_segment_object_values_bulk_candidate(
                params,
                segment,
                &segment_indices,
                &selection.shards,
                decode_cache,
            )?;
            Ok(UnauthCandidateObjectDecode {
                values: segment_indices.into_iter().zip(values).collect(),
                provenance: selection.provenance,
            })
        }
        UnauthReadObjectId::Block { block } => {
            let requested_set: HashSet<Index> = plan.requested_indices.iter().copied().collect();
            let selection = collect_block_rows_for_object_with_provenance(
                response_index,
                object_pos,
                plan.block_row_target,
                bad_filter,
            );
            let decoded = decode_block_from_rows(params, block, &selection.rows)?;
            let mut values = Vec::new();
            for (offset, maybe_index) in aligned_block_indices(params, block, plan.len)
                .into_iter()
                .enumerate()
            {
                let Some(array_index) = maybe_index else {
                    continue;
                };
                if requested_set.contains(&array_index) {
                    values.push((array_index, encode_value(decoded[offset])));
                }
            }
            Ok(UnauthCandidateObjectDecode {
                values,
                provenance: UnauthReadBadFilter {
                    helper_indices: selection.helper_indices,
                    registry_indices: HashSet::new(),
                },
            })
        }
        UnauthReadObjectId::Point { array_index } => {
            let selection = collect_point_shares_for_object_with_provenance(
                response_index,
                object_pos,
                params.need,
                bad_filter,
            );
            let (_, logical_offset) = public_index_to_segment(params, array_index);
            let z_eval = params.decode_index_eval(array_index);
            let value = decode_point_candidate(params, z_eval, &selection.shares)?;
            debug_assert_eq!(
                z_eval,
                logical_offset_to_column_coordinate_for_segment(
                    params,
                    public_index_to_segment(params, array_index).0,
                    logical_offset
                )
            );
            Ok(UnauthCandidateObjectDecode {
                values: vec![(array_index, encode_value(value))],
                provenance: selection.provenance,
            })
        }
    }
}

fn decode_point_candidate(
    params: &CodedArrayParams,
    z_eval: Fp,
    shares: &[ShardShare],
) -> Result<Fp, CodedReadError> {
    let mut points = Vec::new();
    let mut seen = HashSet::new();
    for share in shares {
        let Some(y) = share.value else {
            continue;
        };
        if share.registry_index >= params.shard_registry.len() || !seen.insert(share.registry_index)
        {
            continue;
        }
        let entry = params.shard_registry[share.registry_index];
        points.push((entry.z, y));
    }
    if points.len() < params.need {
        return Err(CodedReadError::NotEnoughShards);
    }
    let coefficients = crate::coded_array::rs::interpolate_coefficients_from_exact_points(
        &points[..params.need],
        params.need,
    )
    .ok_or(CodedReadError::NotEnoughShards)?;
    Ok(poly_eval_coefficients(&coefficients, z_eval))
}

fn decode_segment_object_values_bulk_candidate(
    params: &CodedArrayParams,
    segment: usize,
    requested_indices: &[Index],
    shards: &[SegmentPrimaryShard],
    decode_cache: &DecodeCache,
) -> Result<Vec<Value256>, CodedReadError> {
    let z_degree = params.n_grid / params.s;
    if z_degree != params.need {
        return Err(CodedReadError::NotEnoughShards);
    }

    let mut selected = Vec::with_capacity(params.need);
    let mut seen_z = HashSet::new();
    for shard in shards {
        let Some(&entry) = params.shard_registry.get(shard.registry_index) else {
            continue;
        };
        if shard.values.len() != params.s {
            continue;
        }
        if !seen_z.insert(entry.key.z_bytes) {
            continue;
        }
        selected.push((entry, shard.values.as_slice()));
        if selected.len() == params.need {
            break;
        }
    }
    if selected.len() < params.need {
        return Err(CodedReadError::NotEnoughShards);
    }

    let root_s = fp_pow(params.omega, z_degree as u64);
    let z_values = selected
        .iter()
        .map(|(entry, _)| entry.z)
        .collect::<Vec<_>>();
    let shard_keys = selected
        .iter()
        .map(|(entry, _)| entry.key)
        .collect::<Vec<_>>();
    let weights = decode_cache.get_weights_for_keys_and_z(&shard_keys, &z_values);
    let mut row_samples = vec![fp_zero(); params.s * params.need];
    for (sample_pos, (entry, values)) in selected.iter().enumerate() {
        let mut coeffs = values.to_vec();
        crate::coded_array::params::inverse_fft_in_place(&mut coeffs, root_s);
        let inv_a = entry.a.inverse().ok_or(CodedReadError::NotEnoughShards)?;
        let mut inv_a_pow = fp_one();
        for row in 0..params.s {
            row_samples[row * params.need + sample_pos] = coeffs[row] * inv_a_pow;
            inv_a_pow *= inv_a;
        }
    }

    let rho = segment_domain_rho(params, segment);
    let rho_s = fp_pow(rho, params.s as u64);
    let z_root = fp_pow(params.omega, params.s as u64);
    let mut z_targets = Vec::with_capacity(z_degree);
    let mut z_eval = rho_s;
    for _ in 0..z_degree {
        z_targets.push(z_eval);
        z_eval *= z_root;
    }
    let row_values_by_target = evaluate_candidate_rows(params, &weights, &row_samples, &z_targets);
    let grid_values = materialize_segment_grid_from_candidate_rows(
        params,
        segment,
        &row_values_by_target,
        root_s,
        z_degree,
    );

    requested_segment_values_from_grid(params, segment, requested_indices, &grid_values)
}

fn evaluate_candidate_rows(
    params: &CodedArrayParams,
    weights: &BarycentricWeights,
    row_samples: &[Fp],
    z_targets: &[Fp],
) -> Vec<Vec<Fp>> {
    let interpolation_matrix = barycentric_eval_matrix(weights, z_targets);
    let mut row_values_by_target = Vec::with_capacity(params.s);
    for row in 0..params.s {
        let row_values = &row_samples[row * params.need..(row + 1) * params.need];
        let mut values = Vec::with_capacity(z_targets.len());
        for target_pos in 0..z_targets.len() {
            let row_factors =
                &interpolation_matrix[target_pos * params.need..(target_pos + 1) * params.need];
            let mut value = fp_zero();
            for sample_pos in 0..params.need {
                value += row_factors[sample_pos] * row_values[sample_pos];
            }
            values.push(value);
        }
        row_values_by_target.push(values);
    }
    row_values_by_target
}

fn materialize_segment_grid_from_candidate_rows(
    params: &CodedArrayParams,
    segment: usize,
    row_values_by_target: &[Vec<Fp>],
    root_s: Fp,
    z_degree: usize,
) -> Vec<Fp> {
    let mut grid_values = vec![fp_zero(); params.n_grid];
    let rho = segment_domain_rho(params, segment);
    let mut coset_base = rho;
    for z_pos in 0..z_degree {
        let mut w_coeffs = vec![fp_zero(); params.s];
        let mut base_pow = fp_one();
        for row in 0..params.s {
            w_coeffs[row] = row_values_by_target[row][z_pos] * base_pow;
            base_pow *= coset_base;
        }
        crate::coded_array::params::fft_in_place(&mut w_coeffs, root_s);
        for (offset_in_coset, value) in w_coeffs.into_iter().enumerate() {
            let grid_index = z_pos + offset_in_coset * z_degree;
            grid_values[grid_index] = value;
        }
        coset_base *= params.omega;
    }
    grid_values
}

fn requested_segment_values_from_grid(
    params: &CodedArrayParams,
    segment: usize,
    requested_indices: &[Index],
    grid_values: &[Fp],
) -> Result<Vec<Value256>, CodedReadError> {
    let mut out = Vec::with_capacity(requested_indices.len());
    for &array_index in requested_indices {
        let (idx_segment, logical_offset) = public_index_to_segment(params, array_index);
        if idx_segment != segment || logical_offset >= public_values_per_segment(params) {
            return Err(CodedReadError::NotEnoughShards);
        }
        let grid_index = logical_offset_to_grid_index(params, logical_offset);
        out.push(encode_value(grid_values[grid_index]));
    }
    Ok(out)
}

fn helper_indices_for_registry_indices(
    response_index: &UnauthOptimizedResponseIndex<'_>,
    object_pos: usize,
    registry_indices: &HashSet<usize>,
) -> HashSet<usize> {
    if registry_indices.is_empty() {
        return HashSet::new();
    }
    let mut helper_indices = HashSet::new();
    for indexed in response_index.contributions(object_pos) {
        let Some(helper_index) = indexed.helper_index else {
            continue;
        };
        if contribution_contains_registry_index(indexed.contribution, registry_indices) {
            helper_indices.insert(helper_index);
        }
    }
    helper_indices
}

fn contribution_contains_registry_index(
    contribution: &UnauthReadObjectContribution,
    registry_indices: &HashSet<usize>,
) -> bool {
    match contribution {
        UnauthReadObjectContribution::Segment { shards, .. } => shards
            .iter()
            .any(|shard| registry_indices.contains(&shard.registry_index)),
        UnauthReadObjectContribution::Point { shares, .. } => shares
            .iter()
            .any(|share| registry_indices.contains(&share.registry_index)),
        _ => false,
    }
}

fn promote_new_bad_registry_indices_to_helpers(
    response_index: &UnauthOptimizedResponseIndex<'_>,
    object_pos: usize,
    registry_before: &HashSet<usize>,
    bad_filter: Option<&mut UnauthReadBadFilter>,
) {
    let Some(bad_filter) = bad_filter else {
        return;
    };
    let new_registry_indices = bad_filter
        .registry_indices
        .difference(registry_before)
        .copied()
        .collect::<HashSet<_>>();
    bad_filter
        .helper_indices
        .extend(helper_indices_for_registry_indices(
            response_index,
            object_pos,
            &new_registry_indices,
        ));
}

/// Benchmark-only variant that disables cross-object bad-helper quarantine.
#[cfg(feature = "bench")]
pub fn decode_optimized_read_plan_no_filter(
    params: &CodedArrayParams,
    plan: &UnauthOptimizedReadPlan,
    responses: &[UnauthOptimizedReadResponse],
) -> Result<Vec<Value256>, CodedReadError> {
    decode_optimized_read_plan_internal(
        params,
        plan,
        responses,
        false,
        UnauthReadBadFilter::default(),
    )
}

fn decode_optimized_read_plan_internal(
    params: &CodedArrayParams,
    plan: &UnauthOptimizedReadPlan,
    responses: &[UnauthOptimizedReadResponse],
    filter_bad_helpers: bool,
    initial_bad_filter: UnauthReadBadFilter,
) -> Result<Vec<Value256>, CodedReadError> {
    let response_index = UnauthOptimizedResponseIndex::new(plan.request.objects.len(), responses);
    let mut bad_filter = initial_bad_filter;

    loop {
        let mut values_by_pos = vec![None; plan.requested_indices.len()];
        let decode_cache = DecodeCache::new();
        let mut restart_with_filter = false;

        for (object_pos, object) in plan.request.objects.iter().enumerate() {
            let bad_count_before = bad_filter.known_bad_count();
            match *object {
                UnauthReadObjectId::Segment { segment } => {
                    let segment_indices = requested_indices_for_segment(params, plan, segment);
                    let shards = collect_segment_shards_for_object(
                        plan,
                        &response_index,
                        object_pos,
                        plan.share_target,
                        filter_bad_helpers.then_some(&bad_filter),
                    );
                    let remaining_errors = plan.max_errors.map(|max_errors| {
                        if filter_bad_helpers {
                            max_errors.saturating_sub(bad_filter.known_bad_count())
                        } else {
                            max_errors
                        }
                    });
                    let registry_before = bad_filter.registry_indices.clone();
                    let bad_registry_filter = (filter_bad_helpers && plan.max_errors.is_some())
                        .then_some(&mut bad_filter.registry_indices);
                    let decoded = decode_segment_object_values(
                        params,
                        segment,
                        &segment_indices,
                        &shards,
                        remaining_errors,
                        &decode_cache,
                        bad_registry_filter,
                    )?;
                    if filter_bad_helpers && plan.max_errors.is_some() {
                        promote_new_bad_registry_indices_to_helpers(
                            &response_index,
                            object_pos,
                            &registry_before,
                            Some(&mut bad_filter),
                        );
                    }
                    fill_requested_values_for_sorted_indices(
                        plan,
                        &mut values_by_pos,
                        &segment_indices,
                        decoded,
                    )?;
                }
                UnauthReadObjectId::Block { block } => {
                    let rows = collect_block_rows_for_object(
                        &response_index,
                        object_pos,
                        plan.block_row_target,
                        filter_bad_helpers.then_some(&bad_filter),
                    );
                    let decoded = match plan.max_errors {
                        Some(max_errors) => {
                            let remaining_errors = if filter_bad_helpers {
                                max_errors.saturating_sub(bad_filter.known_bad_count())
                            } else {
                                max_errors
                            };
                            if remaining_errors == 0 {
                                decode_block_from_rows(params, block, &rows)?
                            } else {
                                decode_block_from_rows_correcting(
                                    params,
                                    block,
                                    &rows,
                                    remaining_errors,
                                )?
                            }
                        }
                        None => decode_block_from_rows(params, block, &rows)?,
                    };
                    let block_indices = aligned_block_indices(params, block, plan.len);
                    for (offset, maybe_index) in block_indices.into_iter().enumerate() {
                        let Some(array_index) = maybe_index else {
                            continue;
                        };
                        fill_requested_value_if_planned(
                            plan,
                            &mut values_by_pos,
                            array_index,
                            encode_value(decoded[offset]),
                        );
                    }
                }
                UnauthReadObjectId::Point { array_index } => {
                    let shares = collect_point_shares_for_object(
                        &response_index,
                        object_pos,
                        plan.share_target,
                        filter_bad_helpers.then_some(&bad_filter),
                    );
                    let (_, logical_offset) = public_index_to_segment(params, array_index);
                    let z_eval = params.decode_index_eval(array_index);
                    let value = match plan.max_errors {
                        Some(max_errors) => {
                            let remaining_errors = if filter_bad_helpers {
                                max_errors.saturating_sub(bad_filter.known_bad_count())
                            } else {
                                max_errors
                            };
                            let available_errors = shares.len().saturating_sub(params.need) / 2;
                            let decode_errors = remaining_errors.min(available_errors);
                            if decode_errors == 0 {
                                decode_from_shares_at_z_with_cache(
                                    params,
                                    z_eval,
                                    &shares,
                                    Some(&decode_cache),
                                )?
                            } else {
                                let decoded = decode_from_shares_at_z_with_error_correction(
                                    params,
                                    z_eval,
                                    &shares,
                                    decode_errors,
                                )?;
                                if filter_bad_helpers {
                                    let registry_before = bad_filter.registry_indices.clone();
                                    bad_filter
                                        .registry_indices
                                        .extend(decoded.bad_registry_indices);
                                    promote_new_bad_registry_indices_to_helpers(
                                        &response_index,
                                        object_pos,
                                        &registry_before,
                                        Some(&mut bad_filter),
                                    );
                                }
                                decoded.value
                            }
                        }
                        None => decode_from_shares_at_z_with_cache(
                            params,
                            z_eval,
                            &shares,
                            Some(&decode_cache),
                        )?,
                    };
                    debug_assert_eq!(
                        z_eval,
                        logical_offset_to_column_coordinate_for_segment(
                            params,
                            public_index_to_segment(params, array_index).0,
                            logical_offset
                        )
                    );
                    fill_requested_value(
                        plan,
                        &mut values_by_pos,
                        array_index,
                        encode_value(value),
                    )?;
                }
            }

            if bad_filter.known_bad_count() > bad_count_before {
                restart_with_filter = true;
                break;
            }
        }

        if restart_with_filter {
            continue;
        }

        let mut out = Vec::with_capacity(plan.requested_indices.len());
        for (pos, &array_index) in plan.requested_indices.iter().enumerate() {
            if array_index >= plan.len {
                out.push(encode_value(fp_zero()));
            } else {
                out.push(
                    values_by_pos[pos]
                        .take()
                        .ok_or(CodedReadError::NotEnoughShards)?,
                );
            }
        }
        return Ok(out);
    }
}

fn fill_requested_values_for_sorted_indices(
    plan: &UnauthOptimizedReadPlan,
    values_by_pos: &mut [Option<Value256>],
    indices: &[Index],
    values: Vec<Value256>,
) -> Result<(), CodedReadError> {
    if indices.len() != values.len() {
        return Err(CodedReadError::NotEnoughShards);
    }
    let Some(&first) = indices.first() else {
        return Ok(());
    };
    let mut pos = lower_bound_index(&plan.requested_indices, first);
    for (&array_index, value) in indices.iter().zip(values.into_iter()) {
        if pos >= plan.requested_indices.len() || plan.requested_indices[pos] != array_index {
            fill_requested_value(plan, values_by_pos, array_index, value)?;
        } else {
            values_by_pos[pos] = Some(value);
            pos += 1;
        }
    }
    Ok(())
}

fn fill_requested_value_if_planned(
    plan: &UnauthOptimizedReadPlan,
    values_by_pos: &mut [Option<Value256>],
    array_index: Index,
    value: Value256,
) {
    if let Ok(pos) = plan.requested_indices.binary_search(&array_index) {
        values_by_pos[pos] = Some(value);
    }
}

fn fill_requested_value(
    plan: &UnauthOptimizedReadPlan,
    values_by_pos: &mut [Option<Value256>],
    array_index: Index,
    value: Value256,
) -> Result<(), CodedReadError> {
    let pos = plan
        .requested_indices
        .binary_search(&array_index)
        .map_err(|_| CodedReadError::NotEnoughShards)?;
    values_by_pos[pos] = Some(value);
    Ok(())
}

fn collect_point_shares_for_object_with_provenance(
    response_index: &UnauthOptimizedResponseIndex<'_>,
    object_pos: usize,
    target: usize,
    bad_filter: Option<&UnauthReadBadFilter>,
) -> PointShareSelection {
    let mut shares = Vec::new();
    let mut provenance = UnauthReadBadFilter::default();
    let mut seen = HashSet::new();
    for indexed in response_index.contributions(object_pos) {
        if helper_is_filtered(indexed.helper_index, bad_filter) {
            continue;
        }
        let UnauthReadObjectContribution::Point { shares: reply, .. } = indexed.contribution else {
            continue;
        };
        for share in reply {
            if bad_filter
                .is_some_and(|filter| filter.registry_indices.contains(&share.registry_index))
            {
                continue;
            }
            if share.value.is_some() && seen.insert(share.registry_index) {
                if let Some(helper_index) = indexed.helper_index {
                    provenance.helper_indices.insert(helper_index);
                }
                provenance.registry_indices.insert(share.registry_index);
                shares.push(share.clone());
                if shares.len() >= target {
                    return PointShareSelection { shares, provenance };
                }
            }
        }
    }
    PointShareSelection { shares, provenance }
}

fn count_point_shares_for_object(
    response_index: &UnauthOptimizedResponseIndex<'_>,
    object_pos: usize,
    target: usize,
    bad_filter: Option<&UnauthReadBadFilter>,
) -> usize {
    let mut count = 0usize;
    let mut seen = HashSet::new();
    for indexed in response_index.contributions(object_pos) {
        if helper_is_filtered(indexed.helper_index, bad_filter) {
            continue;
        }
        let UnauthReadObjectContribution::Point { shares: reply, .. } = indexed.contribution else {
            continue;
        };
        for share in reply {
            if bad_filter
                .is_some_and(|filter| filter.registry_indices.contains(&share.registry_index))
            {
                continue;
            }
            if share.value.is_some() && seen.insert(share.registry_index) {
                count += 1;
                if count >= target {
                    return count;
                }
            }
        }
    }
    count
}

fn collect_point_shares_for_object(
    response_index: &UnauthOptimizedResponseIndex<'_>,
    object_pos: usize,
    target: usize,
    bad_filter: Option<&UnauthReadBadFilter>,
) -> Vec<ShardShare> {
    let mut shares = Vec::new();
    let mut seen = HashSet::new();
    for indexed in response_index.contributions(object_pos) {
        if helper_is_filtered(indexed.helper_index, bad_filter) {
            continue;
        }
        let UnauthReadObjectContribution::Point { shares: reply, .. } = indexed.contribution else {
            continue;
        };
        for share in reply {
            if bad_filter
                .is_some_and(|filter| filter.registry_indices.contains(&share.registry_index))
            {
                continue;
            }
            if share.value.is_some() && seen.insert(share.registry_index) {
                shares.push(share.clone());
                if shares.len() >= target {
                    return shares;
                }
            }
        }
    }
    shares
}

fn collect_block_rows_for_object_with_provenance(
    response_index: &UnauthOptimizedResponseIndex<'_>,
    object_pos: usize,
    target: usize,
    bad_filter: Option<&UnauthReadBadFilter>,
) -> BlockRowSelection {
    let mut rows = Vec::new();
    let mut helper_indices = HashSet::new();
    let mut seen = HashSet::new();
    for indexed in response_index.contributions(object_pos) {
        if helper_is_filtered(indexed.helper_index, bad_filter) {
            continue;
        }
        let UnauthReadObjectContribution::Block { rows: reply, .. } = indexed.contribution else {
            continue;
        };
        for row in reply {
            if seen.insert(encode_value(row.row_label)) {
                if let Some(helper_index) = indexed.helper_index {
                    helper_indices.insert(helper_index);
                }
                rows.push(row.clone());
                if rows.len() >= target {
                    rows.sort_by_key(|row| encode_value(row.row_label));
                    return BlockRowSelection {
                        rows,
                        helper_indices,
                    };
                }
            }
        }
    }
    rows.sort_by_key(|row| encode_value(row.row_label));
    BlockRowSelection {
        rows,
        helper_indices,
    }
}

fn count_block_rows_for_object(
    response_index: &UnauthOptimizedResponseIndex<'_>,
    object_pos: usize,
    target: usize,
    bad_filter: Option<&UnauthReadBadFilter>,
) -> usize {
    let mut count = 0usize;
    let mut seen = HashSet::new();
    for indexed in response_index.contributions(object_pos) {
        if helper_is_filtered(indexed.helper_index, bad_filter) {
            continue;
        }
        let UnauthReadObjectContribution::Block { rows: reply, .. } = indexed.contribution else {
            continue;
        };
        for row in reply {
            if seen.insert(encode_value(row.row_label)) {
                count += 1;
                if count >= target {
                    return count;
                }
            }
        }
    }
    count
}

fn collect_block_rows_for_object(
    response_index: &UnauthOptimizedResponseIndex<'_>,
    object_pos: usize,
    target: usize,
    bad_filter: Option<&UnauthReadBadFilter>,
) -> Vec<RowShare> {
    let mut rows = Vec::new();
    let mut seen = HashSet::new();
    for indexed in response_index.contributions(object_pos) {
        if helper_is_filtered(indexed.helper_index, bad_filter) {
            continue;
        }
        let UnauthReadObjectContribution::Block { rows: reply, .. } = indexed.contribution else {
            continue;
        };
        for row in reply {
            if seen.insert(encode_value(row.row_label)) {
                rows.push(row.clone());
                if rows.len() >= target {
                    rows.sort_by_key(|row| encode_value(row.row_label));
                    return rows;
                }
            }
        }
    }
    rows.sort_by_key(|row| encode_value(row.row_label));
    rows
}

fn collect_segment_shards_for_object_with_provenance(
    plan: &UnauthOptimizedReadPlan,
    response_index: &UnauthOptimizedResponseIndex<'_>,
    object_pos: usize,
    target: usize,
    bad_filter: Option<&UnauthReadBadFilter>,
) -> SegmentShardSelection {
    let mut shards = Vec::new();
    let mut provenance = UnauthReadBadFilter::default();
    let mut seen = HashSet::new();
    for indexed in response_index.contributions(object_pos) {
        if helper_is_filtered(indexed.helper_index, bad_filter) {
            continue;
        }
        let UnauthReadObjectContribution::Segment { shards: reply, .. } = indexed.contribution
        else {
            continue;
        };
        for shard in reply {
            if bad_filter
                .is_some_and(|filter| filter.registry_indices.contains(&shard.registry_index))
            {
                continue;
            }
            if shard.registry_index < plan.registry_len
                && shard.values.len() == plan.segment_shard_len
                && seen.insert(shard.registry_index)
            {
                if let Some(helper_index) = indexed.helper_index {
                    provenance.helper_indices.insert(helper_index);
                }
                provenance.registry_indices.insert(shard.registry_index);
                shards.push(shard.clone());
                if shards.len() >= target {
                    shards.sort_by_key(|shard| shard.registry_index);
                    return SegmentShardSelection { shards, provenance };
                }
            }
        }
    }
    shards.sort_by_key(|shard| shard.registry_index);
    SegmentShardSelection { shards, provenance }
}

fn count_segment_shards_for_object(
    plan: &UnauthOptimizedReadPlan,
    response_index: &UnauthOptimizedResponseIndex<'_>,
    object_pos: usize,
    target: usize,
    bad_filter: Option<&UnauthReadBadFilter>,
) -> usize {
    let mut count = 0usize;
    let mut seen = HashSet::new();
    for indexed in response_index.contributions(object_pos) {
        if helper_is_filtered(indexed.helper_index, bad_filter) {
            continue;
        }
        let UnauthReadObjectContribution::Segment { shards: reply, .. } = indexed.contribution
        else {
            continue;
        };
        for shard in reply {
            if bad_filter
                .is_some_and(|filter| filter.registry_indices.contains(&shard.registry_index))
            {
                continue;
            }
            if shard.registry_index < plan.registry_len
                && shard.values.len() == plan.segment_shard_len
                && seen.insert(shard.registry_index)
            {
                count += 1;
                if count >= target {
                    return count;
                }
            }
        }
    }
    count
}

fn collect_segment_shards_for_object(
    plan: &UnauthOptimizedReadPlan,
    response_index: &UnauthOptimizedResponseIndex<'_>,
    object_pos: usize,
    target: usize,
    bad_filter: Option<&UnauthReadBadFilter>,
) -> Vec<SegmentPrimaryShard> {
    let mut shards = Vec::new();
    let mut seen = HashSet::new();
    for indexed in response_index.contributions(object_pos) {
        if helper_is_filtered(indexed.helper_index, bad_filter) {
            continue;
        }
        let UnauthReadObjectContribution::Segment { shards: reply, .. } = indexed.contribution
        else {
            continue;
        };
        for shard in reply {
            if bad_filter
                .is_some_and(|filter| filter.registry_indices.contains(&shard.registry_index))
            {
                continue;
            }
            if shard.registry_index < plan.registry_len
                && shard.values.len() == plan.segment_shard_len
                && seen.insert(shard.registry_index)
            {
                shards.push(shard.clone());
                if shards.len() >= target {
                    shards.sort_by_key(|shard| shard.registry_index);
                    return shards;
                }
            }
        }
    }
    shards.sort_by_key(|shard| shard.registry_index);
    shards
}

fn requested_indices_for_segment(
    params: &CodedArrayParams,
    plan: &UnauthOptimizedReadPlan,
    segment: usize,
) -> Vec<Index> {
    let segment_len = public_values_per_segment(params) as Index;
    let start = (segment as Index).saturating_mul(segment_len);
    let end = start.saturating_add(segment_len).min(plan.len);
    if start >= end {
        return Vec::new();
    }
    let lo = lower_bound_index(&plan.requested_indices, start);
    let hi = lower_bound_index(&plan.requested_indices, end);
    plan.requested_indices[lo..hi]
        .iter()
        .copied()
        .filter(|&array_index| array_index < plan.len)
        .collect()
}

fn lower_bound_index(indices: &[Index], target: Index) -> usize {
    let mut lo = 0usize;
    let mut hi = indices.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if indices[mid] < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn decode_segment_object_values(
    params: &CodedArrayParams,
    segment: usize,
    requested_indices: &[Index],
    shards: &[SegmentPrimaryShard],
    max_errors: Option<usize>,
    decode_cache: &DecodeCache,
    bad_registry_filter: Option<&mut HashSet<usize>>,
) -> Result<Vec<Value256>, CodedReadError> {
    if requested_indices.is_empty() {
        return Ok(Vec::new());
    }

    let required = match max_errors {
        Some(remaining_errors) => error_correcting_share_target(params.need, remaining_errors),
        None => params.need,
    };
    if shards.len() < required {
        return Err(CodedReadError::NotEnoughShards);
    }
    if max_errors.is_none() || max_errors == Some(0) {
        return decode_segment_object_values_bulk_honest(
            params,
            segment,
            requested_indices,
            shards,
            decode_cache,
        );
    }
    let max_errors = max_errors.expect("checked above");
    if params.n_grid / params.s == params.need {
        return decode_segment_object_values_bulk_correcting(
            params,
            segment,
            requested_indices,
            shards,
            max_errors,
            bad_registry_filter,
        );
    }

    decode_segment_object_values_pointwise_correcting(
        params,
        segment,
        requested_indices,
        shards,
        max_errors,
        bad_registry_filter,
    )
}

fn decode_segment_object_values_pointwise_correcting(
    params: &CodedArrayParams,
    segment: usize,
    requested_indices: &[Index],
    shards: &[SegmentPrimaryShard],
    max_errors: usize,
    mut bad_registry_filter: Option<&mut HashSet<usize>>,
) -> Result<Vec<Value256>, CodedReadError> {
    let mut alphas = Vec::with_capacity(requested_indices.len());
    let mut z_evals = Vec::with_capacity(requested_indices.len());
    for &array_index in requested_indices {
        let (idx_segment, logical_offset) = public_index_to_segment(params, array_index);
        if idx_segment != segment {
            return Err(CodedReadError::NotEnoughShards);
        }
        let alpha = logical_offset_to_alpha_for_segment(params, segment, logical_offset);
        alphas.push(alpha);
        z_evals.push(params.column_coordinate(alpha));
    }

    let mut shares_by_pos = vec![Vec::<ShardShare>::new(); requested_indices.len()];
    for shard in shards {
        if bad_registry_filter
            .as_ref()
            .is_some_and(|filter| filter.contains(&shard.registry_index))
        {
            continue;
        }
        let Some(&entry) = params.shard_registry.get(shard.registry_index) else {
            continue;
        };
        let replies = shard_value_from_vector_batch(params, entry, &shard.values, &alphas);
        for (pos, reply) in replies.into_iter().enumerate() {
            let Some(value) = reply else {
                continue;
            };
            shares_by_pos[pos].push(ShardShare {
                registry_index: shard.registry_index,
                value: Some(value),
            });
        }
    }

    let mut out = Vec::with_capacity(requested_indices.len());
    for (pos, shares) in shares_by_pos.iter().enumerate() {
        let available_errors = shares.len().saturating_sub(params.need) / 2;
        let decode_errors = max_errors.min(available_errors);
        let value = if decode_errors == 0 {
            decode_from_shares_at_z_with_cache(params, z_evals[pos], shares, None)?
        } else {
            let decoded = decode_from_shares_at_z_with_error_correction(
                params,
                z_evals[pos],
                shares,
                decode_errors,
            )?;
            if let Some(filter) = bad_registry_filter.as_deref_mut() {
                filter.extend(decoded.bad_registry_indices);
            }
            decoded.value
        };
        out.push(encode_value(value));
    }
    Ok(out)
}

fn decode_segment_object_values_bulk_correcting(
    params: &CodedArrayParams,
    segment: usize,
    requested_indices: &[Index],
    shards: &[SegmentPrimaryShard],
    max_errors: usize,
    mut bad_registry_filter: Option<&mut HashSet<usize>>,
) -> Result<Vec<Value256>, CodedReadError> {
    let z_degree = params.n_grid / params.s;
    if z_degree != params.need {
        return Err(CodedReadError::NotEnoughShards);
    }

    let mut selected = Vec::new();
    let mut seen_z = HashSet::new();
    for shard in shards {
        if bad_registry_filter
            .as_ref()
            .is_some_and(|filter| filter.contains(&shard.registry_index))
        {
            continue;
        }
        let Some(&entry) = params.shard_registry.get(shard.registry_index) else {
            continue;
        };
        if shard.values.len() != params.s {
            continue;
        }
        if !seen_z.insert(entry.key.z_bytes) {
            continue;
        }
        selected.push((shard.registry_index, entry, shard.values.as_slice()));
    }
    if selected.len() < params.need {
        return Err(CodedReadError::NotEnoughShards);
    }

    let root_s = fp_pow(params.omega, z_degree as u64);
    let mut row_samples = vec![fp_zero(); params.s * selected.len()];
    for (sample_pos, (_, entry, values)) in selected.iter().enumerate() {
        let mut coeffs = values.to_vec();
        crate::coded_array::params::inverse_fft_in_place(&mut coeffs, root_s);
        let inv_a = entry.a.inverse().ok_or(CodedReadError::NotEnoughShards)?;
        let mut inv_a_pow = fp_one();
        for row in 0..params.s {
            row_samples[row * selected.len() + sample_pos] = coeffs[row] * inv_a_pow;
            inv_a_pow *= inv_a;
        }
    }

    let shared_bad_sample_positions =
        detect_bad_segment_sample_positions(params, &selected, &row_samples, params.s, max_errors);
    let shared_bad_sample_positions = shared_bad_sample_positions
        .into_iter()
        .collect::<HashSet<_>>();
    if let Some(filter) = bad_registry_filter.as_deref_mut() {
        filter.extend(
            shared_bad_sample_positions
                .iter()
                .filter_map(|&sample_pos| selected.get(sample_pos).map(|(idx, _, _)| *idx)),
        );
    }

    let rho = segment_domain_rho(params, segment);
    let rho_s = fp_pow(rho, params.s as u64);
    let z_root = fp_pow(params.omega, params.s as u64);
    let mut z_targets = Vec::with_capacity(z_degree);
    let mut z_eval = rho_s;
    for _ in 0..z_degree {
        z_targets.push(z_eval);
        z_eval *= z_root;
    }

    let row_values_by_target = match decode_segment_rows_exact_with_known_bad_positions(
        params,
        &selected,
        &row_samples,
        params.s,
        &z_targets,
        &shared_bad_sample_positions,
        bad_registry_filter.as_ref().map(|filter| &**filter),
    ) {
        Some(row_values) => row_values,
        None => {
            let mut row_values_by_target = Vec::with_capacity(params.s);
            for row in 0..params.s {
                let mut points = Vec::with_capacity(selected.len());
                let mut point_registry_indices = Vec::with_capacity(selected.len());
                for (sample_pos, (registry_index, entry, _)) in selected.iter().enumerate() {
                    if shared_bad_sample_positions.contains(&sample_pos) {
                        continue;
                    }
                    if bad_registry_filter
                        .as_ref()
                        .is_some_and(|filter| filter.contains(registry_index))
                    {
                        continue;
                    }
                    points.push((entry.z, row_samples[row * selected.len() + sample_pos]));
                    point_registry_indices.push(*registry_index);
                }
                if points.len() < params.need {
                    return Err(CodedReadError::NotEnoughShards);
                }
                let decoded = match crate::coded_array::rs::rs_decode_eval_with_errors(
                    &points,
                    params.need,
                    fp_zero(),
                    0,
                ) {
                    Ok(decoded) => decoded,
                    Err(_) => {
                        let mut fallback_points = Vec::with_capacity(selected.len());
                        let mut fallback_registry_indices = Vec::with_capacity(selected.len());
                        for (sample_pos, (registry_index, entry, _)) in selected.iter().enumerate()
                        {
                            if bad_registry_filter
                                .as_ref()
                                .is_some_and(|filter| filter.contains(registry_index))
                                && !shared_bad_sample_positions.contains(&sample_pos)
                            {
                                continue;
                            }
                            fallback_points
                                .push((entry.z, row_samples[row * selected.len() + sample_pos]));
                            fallback_registry_indices.push(*registry_index);
                        }
                        if fallback_points.len() < params.need + 2 * max_errors {
                            return Err(CodedReadError::NotEnoughShards);
                        }
                        let decoded = crate::coded_array::rs::rs_decode_eval_with_errors(
                            &fallback_points,
                            params.need,
                            fp_zero(),
                            max_errors,
                        )
                        .map_err(|_| CodedReadError::NotEnoughShards)?;
                        point_registry_indices = fallback_registry_indices;
                        decoded
                    }
                };
                if let Some(filter) = bad_registry_filter.as_deref_mut() {
                    filter.extend(
                        decoded
                            .bad_indices
                            .iter()
                            .filter_map(|&idx| point_registry_indices.get(idx).copied()),
                    );
                }
                row_values_by_target.push(eval_coefficients_on_points(
                    params,
                    &decoded.coefficients,
                    &z_targets,
                ));
            }
            row_values_by_target
        }
    };

    let mut grid_values = vec![fp_zero(); params.n_grid];
    let mut coset_base = rho;
    for z_pos in 0..z_degree {
        let mut w_coeffs = vec![fp_zero(); params.s];
        let mut base_pow = fp_one();
        for row in 0..params.s {
            w_coeffs[row] = row_values_by_target[row][z_pos] * base_pow;
            base_pow *= coset_base;
        }
        crate::coded_array::params::fft_in_place(&mut w_coeffs, root_s);
        for (offset_in_coset, value) in w_coeffs.into_iter().enumerate() {
            let grid_index = z_pos + offset_in_coset * z_degree;
            grid_values[grid_index] = value;
        }
        coset_base *= params.omega;
    }

    let mut out = Vec::with_capacity(requested_indices.len());
    for &array_index in requested_indices {
        let (idx_segment, logical_offset) = public_index_to_segment(params, array_index);
        if idx_segment != segment || logical_offset >= public_values_per_segment(params) {
            return Err(CodedReadError::NotEnoughShards);
        }
        let grid_index = logical_offset_to_grid_index(params, logical_offset);
        out.push(encode_value(grid_values[grid_index]));
    }
    Ok(out)
}

fn decode_segment_rows_exact_with_known_bad_positions(
    params: &CodedArrayParams,
    selected: &[(usize, ShardRegEntry, &[Fp])],
    row_samples: &[Fp],
    row_count: usize,
    z_targets: &[Fp],
    bad_sample_positions: &HashSet<usize>,
    bad_registry_filter: Option<&HashSet<usize>>,
) -> Option<Vec<Vec<Fp>>> {
    let good_positions = selected
        .iter()
        .enumerate()
        .filter_map(|(sample_pos, (registry_index, _, _))| {
            (!bad_sample_positions.contains(&sample_pos)
                && !bad_registry_filter.is_some_and(|filter| filter.contains(registry_index)))
            .then_some(sample_pos)
        })
        .collect::<Vec<_>>();
    if good_positions.len() < params.need {
        return None;
    }

    let basis_positions = &good_positions[..params.need];
    let basis_z = basis_positions
        .iter()
        .map(|&sample_pos| selected[sample_pos].1.z)
        .collect::<Vec<_>>();
    let basis = interpolation_basis_coefficients(&basis_z);

    let mut rows = Vec::with_capacity(row_count);
    for row in 0..row_count {
        let mut coefficients = vec![fp_zero(); params.need];
        for (basis_pos, &sample_pos) in basis_positions.iter().enumerate() {
            let y = row_samples[row * selected.len() + sample_pos];
            for (out, basis_coeff) in coefficients.iter_mut().zip(basis[basis_pos].iter()) {
                *out += y * *basis_coeff;
            }
        }
        for &sample_pos in good_positions.iter().skip(params.need) {
            let z = selected[sample_pos].1.z;
            let expected = poly_eval_coefficients(&coefficients, z);
            if expected != row_samples[row * selected.len() + sample_pos] {
                return None;
            }
        }
        rows.push(eval_coefficients_on_points(
            params,
            &coefficients,
            z_targets,
        ));
    }
    Some(rows)
}

fn interpolation_basis_coefficients(z: &[Fp]) -> Vec<Vec<Fp>> {
    if z.is_empty() {
        return Vec::new();
    }
    let weights = precompute_barycentric(z);
    let mut root_poly = vec![fp_one()];
    for &x in z {
        let mut next = vec![fp_zero(); root_poly.len() + 1];
        for (idx, coeff) in root_poly.iter().copied().enumerate() {
            next[idx] -= coeff * x;
            next[idx + 1] += coeff;
        }
        root_poly = next;
    }

    let degree = z.len();
    let mut basis = vec![vec![fp_zero(); degree]; degree];
    for (point_idx, &x) in z.iter().enumerate() {
        let scale = weights.w[point_idx];
        let mut quotient = vec![fp_zero(); degree];
        quotient[degree - 1] = root_poly[degree];
        for coeff_idx in (1..degree).rev() {
            quotient[coeff_idx - 1] = root_poly[coeff_idx] + x * quotient[coeff_idx];
        }
        for (out, basis_coeff) in basis[point_idx].iter_mut().zip(quotient.into_iter()) {
            *out = scale * basis_coeff;
        }
    }
    basis
}

fn poly_eval_coefficients(coefficients: &[Fp], z: Fp) -> Fp {
    let mut acc = fp_zero();
    for coeff in coefficients.iter().rev() {
        acc = acc * z + *coeff;
    }
    acc
}

fn detect_bad_segment_sample_positions(
    params: &CodedArrayParams,
    selected: &[(usize, ShardRegEntry, &[Fp])],
    row_samples: &[Fp],
    row_count: usize,
    max_errors: usize,
) -> Vec<usize> {
    if max_errors == 0 || selected.len() < params.need + 2 * max_errors {
        return Vec::new();
    }

    let challenges = [Fp::from(17u64), Fp::from(257u64)];
    let mut bad_positions = HashSet::new();
    for challenge in challenges {
        let mut points = Vec::with_capacity(selected.len());
        for (sample_pos, (_, entry, _)) in selected.iter().enumerate() {
            let mut acc = fp_zero();
            let mut challenge_pow = fp_one();
            for row in 0..row_count {
                acc += row_samples[row * selected.len() + sample_pos] * challenge_pow;
                challenge_pow *= challenge;
            }
            points.push((entry.z, acc));
        }

        let Ok(decoded) = crate::coded_array::rs::rs_decode_eval_with_errors(
            &points,
            params.need,
            fp_zero(),
            max_errors,
        ) else {
            continue;
        };
        bad_positions.extend(decoded.bad_indices);
        if bad_positions.len() >= max_errors {
            break;
        }
    }

    bad_positions.into_iter().collect()
}

fn decode_segment_object_values_bulk_honest(
    params: &CodedArrayParams,
    segment: usize,
    requested_indices: &[Index],
    shards: &[SegmentPrimaryShard],
    decode_cache: &DecodeCache,
) -> Result<Vec<Value256>, CodedReadError> {
    let z_degree = params.n_grid / params.s;
    if z_degree != params.need {
        return Err(CodedReadError::NotEnoughShards);
    }

    let mut selected = Vec::with_capacity(params.need);
    let mut seen_z = HashSet::new();
    for shard in shards {
        let Some(&entry) = params.shard_registry.get(shard.registry_index) else {
            continue;
        };
        if shard.values.len() != params.s {
            continue;
        }
        if !seen_z.insert(entry.key.z_bytes) {
            continue;
        }
        selected.push((entry, shard.values.as_slice()));
        if selected.len() == params.need {
            break;
        }
    }
    if selected.len() < params.need {
        return Err(CodedReadError::NotEnoughShards);
    }

    let root_s = fp_pow(params.omega, z_degree as u64);
    let z_values = selected
        .iter()
        .map(|(entry, _)| entry.z)
        .collect::<Vec<_>>();
    let shard_keys = selected
        .iter()
        .map(|(entry, _)| entry.key)
        .collect::<Vec<_>>();
    let weights = decode_cache.get_weights_for_keys_and_z(&shard_keys, &z_values);
    let mut row_samples = vec![fp_zero(); params.s * params.need];
    for (sample_pos, (entry, values)) in selected.iter().enumerate() {
        let mut coeffs = values.to_vec();
        crate::coded_array::params::inverse_fft_in_place(&mut coeffs, root_s);
        let inv_a = entry.a.inverse().ok_or(CodedReadError::NotEnoughShards)?;
        let mut inv_a_pow = fp_one();
        for row in 0..params.s {
            row_samples[row * params.need + sample_pos] = coeffs[row] * inv_a_pow;
            inv_a_pow *= inv_a;
        }
    }

    let rho = segment_domain_rho(params, segment);
    let rho_s = fp_pow(rho, params.s as u64);
    let z_root = fp_pow(params.omega, params.s as u64);
    let mut z_targets = Vec::with_capacity(z_degree);
    let mut z_eval = rho_s;
    for _ in 0..z_degree {
        z_targets.push(z_eval);
        z_eval *= z_root;
    }
    let interpolation_matrix = barycentric_eval_matrix(&weights, &z_targets);

    let mut grid_values = vec![fp_zero(); params.n_grid];
    let mut coset_base = rho;
    for z_pos in 0..z_degree {
        let row_factors = &interpolation_matrix[z_pos * params.need..(z_pos + 1) * params.need];
        let mut w_coeffs = vec![fp_zero(); params.s];
        let mut base_pow = fp_one();
        for row in 0..params.s {
            let row_values = &row_samples[row * params.need..(row + 1) * params.need];
            let mut row_value = fp_zero();
            for sample_pos in 0..params.need {
                row_value += row_factors[sample_pos] * row_values[sample_pos];
            }
            w_coeffs[row] = row_value * base_pow;
            base_pow *= coset_base;
        }
        crate::coded_array::params::fft_in_place(&mut w_coeffs, root_s);
        for (offset_in_coset, value) in w_coeffs.into_iter().enumerate() {
            let grid_index = z_pos + offset_in_coset * z_degree;
            grid_values[grid_index] = value;
        }
        coset_base *= params.omega;
    }

    let mut out = Vec::with_capacity(requested_indices.len());
    for &array_index in requested_indices {
        let (idx_segment, logical_offset) = public_index_to_segment(params, array_index);
        if idx_segment != segment || logical_offset >= public_values_per_segment(params) {
            return Err(CodedReadError::NotEnoughShards);
        }
        let grid_index = logical_offset_to_grid_index(params, logical_offset);
        out.push(encode_value(grid_values[grid_index]));
    }
    Ok(out)
}

fn barycentric_eval_matrix(weights: &BarycentricWeights, z_evals: &[Fp]) -> Vec<Fp> {
    let n = weights.z.len();
    if n == 0 {
        return Vec::new();
    }
    let mut matrix = vec![fp_zero(); z_evals.len() * n];
    let mut pending = Vec::new();
    let mut denoms = Vec::new();

    for (eval_pos, &z_eval) in z_evals.iter().enumerate() {
        if let Some(sample_pos) = weights.z.iter().position(|&z_i| z_i == z_eval) {
            matrix[eval_pos * n + sample_pos] = fp_one();
        } else {
            pending.push(eval_pos);
            denoms.extend(weights.z.iter().map(|&z_i| z_eval - z_i));
        }
    }

    batch_invert(&mut denoms);
    let mut final_denoms = Vec::with_capacity(pending.len());
    for (pending_pos, &eval_pos) in pending.iter().enumerate() {
        let start = pending_pos * n;
        let matrix_start = eval_pos * n;
        let mut den = fp_zero();
        for sample_pos in 0..n {
            let factor = weights.w[sample_pos] * denoms[start + sample_pos];
            matrix[matrix_start + sample_pos] = factor;
            den += factor;
        }
        final_denoms.push(den);
    }

    batch_invert(&mut final_denoms);
    for (pending_pos, &eval_pos) in pending.iter().enumerate() {
        let inv_den = final_denoms[pending_pos];
        let matrix_start = eval_pos * n;
        for sample_pos in 0..n {
            matrix[matrix_start + sample_pos] *= inv_den;
        }
    }
    matrix
}

impl ClaimBundle {
    /// Build the default exact claim bundle for one registry index.
    pub fn exact_for_registry_index(
        params: &CodedArrayParams,
        registry_index: usize,
    ) -> Option<Self> {
        Some(Self {
            registry_index,
            row_family_id: params.row_family_id_for_registry_index(registry_index)?,
        })
    }

    /// Expand the claim's maintained row family into explicit row-label indices.
    pub fn row_label_indices(&self, params: &CodedArrayParams) -> Option<Vec<usize>> {
        params.row_label_indices_for_row_family(self.row_family_id)
    }

    /// Borrow the claim's exact maintained row family without rebuilding it.
    pub fn row_label_indices_slice<'a>(&self, params: &'a CodedArrayParams) -> Option<&'a [usize]> {
        params.row_label_indices_for_row_family_slice(self.row_family_id)
    }
}

/// Per-claim partner shard data supplied during one kappa-halve migration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartnerShardMigration {
    pub registry_index: usize,
    pub segments: Vec<Vec<Fp>>,
}

/// Full source-row data supplied during kappa-halve when local rows are insufficient.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HelperRowMigration {
    pub row_label: Fp,
    pub segments: Vec<Vec<Fp>>,
}

type FullMaterializationCache = HashMap<Vec<Value256>, (Vec<Vec<Fp>>, Vec<Vec<Fp>>)>;
type ClaimMaterializationCache =
    HashMap<(Vec<ClaimBundle>, Vec<Value256>), (HashMap<usize, Vec<Fp>>, HashMap<usize, Vec<Fp>>)>;
type FullFamilyMaterializationCache = HashMap<LogicalValueFamilyKey, (Vec<Vec<Fp>>, Vec<Vec<Fp>>)>;
type ClaimFamilyMaterializationCache = HashMap<
    (Vec<ClaimBundle>, LogicalValueFamilyKey),
    (HashMap<usize, Vec<Fp>>, HashMap<usize, Vec<Fp>>),
>;
type FamilyCoeffCache = HashMap<LogicalValueFamilyKey, Vec<Fp>>;

fn segment_values_are_layout_invariant(values: &[Value256]) -> bool {
    values
        .split_first()
        .is_none_or(|(first, rest)| rest.iter().all(|value| value == first))
}

fn family_key_is_layout_invariant(family_key: &LogicalValueFamilyKey) -> bool {
    family_key
        .codes()
        .split_first()
        .is_none_or(|(first, rest)| rest.iter().all(|code| code == first))
}

/// Shard storage for the unauthenticated layer.
pub struct UnauthStore {
    pub params: CodedArrayParams,
    pub coded: CodedStore,
    claims: Option<Vec<ClaimBundle>>,
    len: Index,
    segments: usize,
    full_materialization_cache: FullMaterializationCache,
    claim_materialization_cache: ClaimMaterializationCache,
    full_family_materialization_cache: FullFamilyMaterializationCache,
    claim_family_materialization_cache: ClaimFamilyMaterializationCache,
    family_coeff_cache: FamilyCoeffCache,
}

impl Clone for UnauthStore {
    fn clone(&self) -> Self {
        Self {
            params: self.params.clone(),
            coded: self.coded.clone(),
            claims: self.claims.clone(),
            len: self.len,
            segments: self.segments,
            full_materialization_cache: HashMap::new(),
            claim_materialization_cache: HashMap::new(),
            full_family_materialization_cache: HashMap::new(),
            claim_family_materialization_cache: HashMap::new(),
            family_coeff_cache: HashMap::new(),
        }
    }
}

impl UnauthStore {
    /// Create an empty store holding all shard entries.
    pub fn new(params: CodedArrayParams) -> Self {
        Self {
            params,
            coded: CodedStore::new(),
            claims: None,
            len: 0,
            segments: 0,
            full_materialization_cache: HashMap::new(),
            claim_materialization_cache: HashMap::new(),
            full_family_materialization_cache: HashMap::new(),
            claim_family_materialization_cache: HashMap::new(),
            family_coeff_cache: HashMap::new(),
        }
    }

    /// Create an empty store materializing only the supplied claim bundles.
    pub fn new_with_claims(params: CodedArrayParams, claims: Vec<ClaimBundle>) -> Self {
        let claims = normalize_claim_bundles(&params, claims);
        Self {
            params,
            coded: CodedStore::new(),
            claims: Some(claims),
            len: 0,
            segments: 0,
            full_materialization_cache: HashMap::new(),
            claim_materialization_cache: HashMap::new(),
            full_family_materialization_cache: HashMap::new(),
            claim_family_materialization_cache: HashMap::new(),
            family_coeff_cache: HashMap::new(),
        }
    }

    /// Return the current logical length.
    pub fn len(&self) -> Index {
        self.len
    }

    /// Install recovered claim-bound shard and row state for one allocated segment.
    pub fn install_claim_segment_state(
        &mut self,
        seg: usize,
        repaired_shards: &[(usize, Vec<Value>)],
        repaired_rows: &[(usize, Vec<Value>)],
    ) -> Result<(), InstallClaimStateError> {
        let repaired_shards_fp: Vec<_> = repaired_shards
            .iter()
            .map(|(registry_index, values)| {
                (
                    *registry_index,
                    values.iter().copied().map(decode_value).collect::<Vec<_>>(),
                )
            })
            .collect();
        let repaired_rows_fp: Vec<_> = repaired_rows
            .iter()
            .map(|(row_label_index, values)| {
                (
                    *row_label_index,
                    values.iter().copied().map(decode_value).collect::<Vec<_>>(),
                )
            })
            .collect();
        self.install_claim_segment_state_fp(seg, &repaired_shards_fp, &repaired_rows_fp)
    }

    /// Precompute the exact shard/row shape expected by this claim-bound store.
    pub fn claim_segment_install_plan(
        &self,
    ) -> Result<ClaimSegmentInstallPlan, InstallClaimStateError> {
        let claims = self
            .claims
            .as_deref()
            .ok_or(InstallClaimStateError::MissingClaims)?;
        let expected_shards: HashSet<_> = claims.iter().map(|claim| claim.registry_index).collect();
        let mut expected_rows = HashSet::new();
        for claim in claims {
            let row_label_indices = claim
                .row_label_indices_slice(&self.params)
                .ok_or(InstallClaimStateError::InvalidIndex)?;
            expected_rows.extend(row_label_indices.iter().copied());
        }
        Ok(ClaimSegmentInstallPlan {
            expected_shards,
            expected_rows,
            expected_shard_len: self.params.s,
            expected_row_len: self.params.row_eval_points.len(),
        })
    }

    /// Install recovered claim-bound shard and row state for one allocated segment.
    pub fn install_claim_segment_state_fp(
        &mut self,
        seg: usize,
        repaired_shards: &[(usize, Vec<Fp>)],
        repaired_rows: &[(usize, Vec<Fp>)],
    ) -> Result<(), InstallClaimStateError> {
        let plan = self.claim_segment_install_plan()?;
        self.install_claim_segment_state_fp_with_plan(seg, &plan, repaired_shards, repaired_rows)
    }

    /// Install one Fp-native recovered claim segment state object.
    pub fn install_recovered_claim_segment_state(
        &mut self,
        seg: usize,
        recovered: &RecoveredClaimSegmentState,
    ) -> Result<(), InstallClaimStateError> {
        self.install_claim_segment_state_fp(seg, &recovered.shards, &recovered.rows)
    }

    /// Install one Fp-native recovered claim segment state object using a precomputed plan.
    pub fn install_recovered_claim_segment_state_with_plan(
        &mut self,
        seg: usize,
        plan: &ClaimSegmentInstallPlan,
        recovered: &RecoveredClaimSegmentState,
    ) -> Result<(), InstallClaimStateError> {
        self.install_claim_segment_state_fp_with_plan(seg, plan, &recovered.shards, &recovered.rows)
    }

    /// Install one owned Fp-native recovered claim segment state object using a precomputed plan.
    pub fn install_recovered_claim_segment_state_owned_with_plan(
        &mut self,
        seg: usize,
        plan: &ClaimSegmentInstallPlan,
        recovered: RecoveredClaimSegmentState,
    ) -> Result<(), InstallClaimStateError> {
        if seg >= self.segments {
            return Err(InstallClaimStateError::InvalidIndex);
        }
        let claims = self
            .claims
            .as_deref()
            .ok_or(InstallClaimStateError::MissingClaims)?;
        let shard_map = validate_owned_recovered_shards(plan, recovered.shards)?;
        let row_map = validate_owned_recovered_rows(plan, recovered.rows)?;
        overwrite_segment_state_for_claims_sparse(
            &self.params,
            &mut self.coded,
            seg,
            claims,
            &shard_map,
            &row_map,
        );
        Ok(())
    }

    fn install_claim_segment_state_fp_with_plan(
        &mut self,
        seg: usize,
        plan: &ClaimSegmentInstallPlan,
        repaired_shards: &[(usize, Vec<Fp>)],
        repaired_rows: &[(usize, Vec<Fp>)],
    ) -> Result<(), InstallClaimStateError> {
        if seg >= self.segments {
            return Err(InstallClaimStateError::InvalidIndex);
        }
        let claims = self
            .claims
            .as_deref()
            .ok_or(InstallClaimStateError::MissingClaims)?;
        let shard_map = validate_recovered_shards(plan, repaired_shards)?;
        let row_map = validate_recovered_rows(plan, repaired_rows)?;
        overwrite_segment_state_for_claims_sparse(
            &self.params,
            &mut self.coded,
            seg,
            claims,
            &shard_map,
            &row_map,
        );
        Ok(())
    }

    pub(crate) fn overwrite_segment_from_values(
        &mut self,
        seg: usize,
        t_seg: &[Value256],
    ) -> Result<(), WriteError> {
        if seg >= self.segments {
            return Err(WriteError::InvalidIndex);
        }
        match self.claims.clone() {
            Some(claims) => {
                let (shards, rows) = materialize_claim_segment_state_for_segment(
                    &self.params,
                    seg,
                    t_seg,
                    &claims,
                    None,
                );
                overwrite_segment_state_for_claims_sparse(
                    &self.params,
                    &mut self.coded,
                    seg,
                    &claims,
                    &shards,
                    &rows,
                );
            }
            None => {
                let (shards, rows) =
                    materialize_segment_state_for_segment(&self.params, seg, t_seg, None);
                let all_indices: Vec<usize> = (0..self.params.shard_registry.len()).collect();
                let all_claims = default_claim_bundles_for_indices(&self.params, &all_indices);
                let shard_map = shards
                    .into_iter()
                    .enumerate()
                    .collect::<HashMap<usize, Vec<Fp>>>();
                let row_map = rows
                    .into_iter()
                    .enumerate()
                    .collect::<HashMap<usize, Vec<Fp>>>();
                overwrite_segment_state_for_claims_sparse(
                    &self.params,
                    &mut self.coded,
                    seg,
                    &all_claims,
                    &shard_map,
                    &row_map,
                );
            }
        }
        Ok(())
    }

    pub(crate) fn add_segment_delta_from_values(
        &mut self,
        seg: usize,
        t_seg_delta: &[Value256],
    ) -> Result<(), WriteError> {
        if seg >= self.segments {
            return Err(WriteError::InvalidIndex);
        }
        match self.claims.clone() {
            Some(claims) => {
                let (shards, rows) = materialize_claim_segment_state_for_segment(
                    &self.params,
                    seg,
                    t_seg_delta,
                    &claims,
                    None,
                );
                accumulate_segment_state_for_claims_sparse(
                    &self.params,
                    &mut self.coded,
                    seg,
                    &claims,
                    &shards,
                    &rows,
                );
            }
            None => {
                let (shards, rows) =
                    materialize_segment_state_for_segment(&self.params, seg, t_seg_delta, None);
                let all_indices: Vec<usize> = (0..self.params.shard_registry.len()).collect();
                accumulate_segment_state(
                    &self.params,
                    &mut self.coded,
                    seg,
                    &all_indices,
                    &shards,
                    &rows,
                );
            }
        }
        Ok(())
    }

    /// Add encoded field deltas to a contiguous logical array range.
    pub fn add_range_deltas(
        &mut self,
        start: Index,
        deltas: &[Value256],
    ) -> Result<(), WriteError> {
        if deltas.is_empty() {
            return Ok(());
        }
        let len = deltas.len() as u64;
        let end = start.checked_add(len).ok_or(WriteError::InvalidIndex)?;
        if end > self.len {
            return Err(WriteError::InvalidIndex);
        }
        let b = public_values_per_segment(&self.params) as u64;
        let first_seg = start / b;
        let last_seg = (end - 1) / b;
        for seg in first_seg..=last_seg {
            let seg_base = seg * b;
            let mut segment_deltas = default_segment_values(&self.params);
            for logical_offset in 0..public_values_per_segment(&self.params) {
                let idx = seg_base + logical_offset as u64;
                if idx < start || idx >= end {
                    continue;
                }
                let grid_index = logical_offset_to_grid_index(&self.params, logical_offset);
                segment_deltas[grid_index] = deltas[(idx - start) as usize];
            }
            let seg = usize::try_from(seg).map_err(|_| WriteError::InvalidIndex)?;
            self.add_segment_delta_from_values(seg, &segment_deltas)?;
        }
        Ok(())
    }

    /// Write final values into a range whose current contents are known to be
    /// `default_at_index`. Whole touched segments are overwritten directly;
    /// boundary fragments are applied as deltas without reading neighboring
    /// values.
    pub fn write_fresh_range_from_defaults<F>(
        &mut self,
        start: Index,
        values: &[Value256],
        mode: DeltaMode,
        mut default_at_index: F,
    ) -> Result<(), WriteError>
    where
        F: FnMut(Index) -> Value256,
    {
        if values.is_empty() {
            return Ok(());
        }
        let len = values.len() as u64;
        let end = start.checked_add(len).ok_or(WriteError::InvalidIndex)?;
        if end > self.len {
            return Err(WriteError::InvalidIndex);
        }
        let b = public_values_per_segment(&self.params) as u64;
        let first_seg = start / b;
        let last_seg = (end - 1) / b;
        let mut sparse_boundary_deltas = Vec::new();
        for seg in first_seg..=last_seg {
            let seg_base = seg * b;
            let seg_end = seg_base.checked_add(b).ok_or(WriteError::InvalidIndex)?;
            let seg = usize::try_from(seg).map_err(|_| WriteError::InvalidIndex)?;
            let write_start = start.max(seg_base);
            let write_end = end.min(seg_end);
            let write_count =
                usize::try_from(write_end - write_start).map_err(|_| WriteError::InvalidIndex)?;
            if start <= seg_base && seg_end <= end {
                let segment_values = self.params.materialize_logical_segment(|logical_offset| {
                    values[(seg_base + logical_offset as u64 - start) as usize]
                });
                self.overwrite_segment_from_values(seg, &segment_values)?;
                continue;
            }

            if write_count <= ROW_INTERPOLATE_THRESHOLD {
                for idx in write_start..write_end {
                    let new_value = values[(idx - start) as usize];
                    let default_value = default_at_index(idx);
                    let delta = decode_value(new_value) - decode_value(default_value);
                    if delta != fp_zero() {
                        sparse_boundary_deltas.push((idx, delta));
                    }
                }
                continue;
            }

            let mut segment_deltas = default_segment_values(&self.params);
            for logical_offset in 0..public_values_per_segment(&self.params) {
                let idx = seg_base + logical_offset as u64;
                if idx < start || idx >= end {
                    continue;
                }
                let new_value = values[(idx - start) as usize];
                let default_value = default_at_index(idx);
                let delta = decode_value(new_value) - decode_value(default_value);
                let grid_index = logical_offset_to_grid_index(&self.params, logical_offset);
                segment_deltas[grid_index] = encode_value(delta);
            }
            self.add_segment_delta_from_values(seg, &segment_deltas)?;
        }
        if !sparse_boundary_deltas.is_empty() {
            self.apply_deltas(&sparse_boundary_deltas, mode)?;
        }
        Ok(())
    }

    /// Add values into an inactive range that is physically present but known
    /// to contain zero padding, without changing the public logical length.
    pub(crate) fn hydrate_inactive_zero_range(
        &mut self,
        start: Index,
        values: &[Value256],
        mode: DeltaMode,
    ) -> Result<(), WriteError> {
        if values.is_empty() {
            return Ok(());
        }
        let end = start
            .checked_add(values.len() as u64)
            .ok_or(WriteError::InvalidIndex)?;
        if start < self.len {
            return Err(WriteError::InvalidIndex);
        }
        let capacity = (self.segments as u64)
            .checked_mul(public_values_per_segment(&self.params) as u64)
            .ok_or(WriteError::InvalidIndex)?;
        if end > capacity {
            return Err(WriteError::InvalidIndex);
        }

        let deltas = values
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(offset, value)| {
                let delta = decode_value(value);
                (delta != fp_zero()).then_some((start + offset as u64, delta))
            })
            .collect::<Vec<_>>();
        let old_len = self.len;
        self.len = end;
        let result = self.apply_deltas(&deltas, mode);
        self.len = old_len;
        result
    }

    /// Overwrite a contiguous logical array range, preserving existing values
    /// outside the range in any touched segment.
    pub fn overwrite_range(&mut self, start: Index, values: &[Value256]) -> Result<(), WriteError> {
        if values.is_empty() {
            return Ok(());
        }
        let len = values.len() as u64;
        let end = start.checked_add(len).ok_or(WriteError::InvalidIndex)?;
        if end > self.len {
            return Err(WriteError::InvalidIndex);
        }
        let b = public_values_per_segment(&self.params) as u64;
        let first_seg = start / b;
        let last_seg = (end - 1) / b;
        for seg in first_seg..=last_seg {
            let seg_base = seg * b;
            let mut segment_values = default_segment_values(&self.params);
            for logical_offset in 0..public_values_per_segment(&self.params) {
                let idx = seg_base + logical_offset as u64;
                let value = if idx < start || idx >= end {
                    if idx < self.len {
                        self.read(idx).map_err(|_| WriteError::InvalidIndex)?
                    } else {
                        encode_value(fp_zero())
                    }
                } else {
                    values[(idx - start) as usize]
                };
                let grid_index = logical_offset_to_grid_index(&self.params, logical_offset);
                segment_values[grid_index] = value;
            }
            let seg = usize::try_from(seg).map_err(|_| WriteError::InvalidIndex)?;
            self.overwrite_segment_from_values(seg, &segment_values)?;
        }
        Ok(())
    }

    /// Return the current number of allocated segments.
    pub fn segments(&self) -> usize {
        self.segments
    }

    /// Return the currently configured claim bundles, if this store is claim-bound.
    pub fn claim_bundles(&self) -> Option<Vec<ClaimBundle>> {
        self.claims.clone()
    }

    fn clear_materialization_caches(&mut self) {
        self.full_materialization_cache.clear();
        self.claim_materialization_cache.clear();
        self.full_family_materialization_cache.clear();
        self.claim_family_materialization_cache.clear();
        self.family_coeff_cache.clear();
    }

    /// Restrict a claim-bound store to a deterministic subset of claims.
    ///
    /// Kappa-rescale may temporarily materialize more state than the next
    /// registry assigns to this server. This method performs the activation
    /// step: keep only the target claims and delete all other primary and row
    /// state. It does not change `params`, `len`, or `segments`.
    pub fn retain_claim_bundles(&mut self, claims: Vec<ClaimBundle>) -> Result<(), MigrateError> {
        if self.claims.is_none() {
            return Err(MigrateError::InvalidClaim);
        }
        let claims = normalize_claim_bundles(&self.params, claims);
        for claim in &claims {
            let Some(entry) = self.params.shard_registry.get(claim.registry_index) else {
                return Err(MigrateError::InvalidClaim);
            };
            let Some(row_label_indices) = claim.row_label_indices_slice(&self.params) else {
                return Err(MigrateError::InvalidClaim);
            };
            if !self.coded.contains_shard(&entry.key) {
                return Err(MigrateError::MissingPartnerShards);
            }
            for &row_label_idx in row_label_indices {
                let row_key = RowKey::new(entry.key, self.params.row_labels[row_label_idx]);
                if !self.coded.contains_row(&row_key) {
                    return Err(MigrateError::MissingLocalRows);
                }
            }
        }

        let keep_shards: HashSet<_> = claims
            .iter()
            .map(|claim| self.params.shard_registry[claim.registry_index].key)
            .collect();
        let mut keep_rows = HashSet::new();
        for claim in &claims {
            let entry = self.params.shard_registry[claim.registry_index];
            let row_label_indices = claim
                .row_label_indices_slice(&self.params)
                .ok_or(MigrateError::InvalidClaim)?;
            for &row_label_idx in row_label_indices {
                keep_rows.insert(RowKey::new(
                    entry.key,
                    self.params.row_labels[row_label_idx],
                ));
            }
        }
        self.coded.retain_shards(|key| keep_shards.contains(key));
        self.coded.retain_rows(|key| keep_rows.contains(key));
        self.claims = Some(claims);
        self.clear_materialization_caches();
        Ok(())
    }

    /// Restrict a claim-bound store to exact claims for registry indices.
    pub fn retain_registry_indices(
        &mut self,
        registry_indices: &[usize],
    ) -> Result<(), MigrateError> {
        let mut claims = Vec::with_capacity(registry_indices.len());
        for &registry_index in registry_indices {
            claims.push(
                ClaimBundle::exact_for_registry_index(&self.params, registry_index)
                    .ok_or(MigrateError::InvalidClaim)?,
            );
        }
        self.retain_claim_bundles(claims)
    }

    /// Extend the array length, appending any newly required segments.
    pub fn extend(&mut self, new_len: Index) -> Result<(), ExtendError> {
        if new_len <= self.len {
            return Ok(());
        }
        match self.claims.clone() {
            Some(claims) => self.extend_default_segments_for_claims(new_len, &claims),
            None => self.extend_default_segments(new_len),
        }
    }

    /// Change the current encoded state to the adjacent mode parameter `s`.
    ///
    /// Prefer `kappa_double` or `kappa_halve` for the two supported adjacent
    /// directions. This generic form is kept for compatibility and internal
    /// callers that already computed the target `s`.
    pub fn migrate(
        &mut self,
        target_s: usize,
        partner_shards: &[PartnerShardMigration],
    ) -> Result<(), MigrateError> {
        self.migrate_with_helper_rows(target_s, partner_shards, &[])
    }

    /// Change the current encoded state to the adjacent mode parameter `s`,
    /// optionally supplying `kappa_halve` helper rows.
    pub fn migrate_with_helper_rows(
        &mut self,
        target_s: usize,
        partner_shards: &[PartnerShardMigration],
        helper_rows: &[HelperRowMigration],
    ) -> Result<(), MigrateError> {
        let source_claims = self.effective_claims_for_params(&self.params);
        let new_params = build_migrated_params(&self.params, target_s)?;
        let migrated = if target_s > self.params.s {
            self.kappa_halve_encoded_state(new_params, partner_shards, helper_rows)?
        } else {
            let target_claims = split_claim_bundles_for_kappa_double_target_mode(
                &self.params,
                &new_params,
                &source_claims,
            );
            let mut migrated = self.kappa_double_encoded_state(new_params)?;
            migrated.hydrate_rows_from_target_columns_with_claims(
                partner_shards,
                Some(&target_claims),
            )?;
            migrated
        };
        *self = migrated;
        Ok(())
    }

    /// Double the segment size with the local factor-two layout.
    ///
    /// This keeps `s` fixed, doubles `n_grid`/`need`, and places adjacent
    /// logical source segments on the even/odd physical grids of the target
    /// segment.
    pub fn segment_double(&mut self) -> Result<(), MigrateError> {
        self.segment_double_with_helpers(&[])
    }

    /// Double the segment size with optional compatibility helper payloads.
    ///
    /// The local factor-two operation does not currently consume helper
    /// shards, so non-empty helper input is rejected.
    pub fn segment_double_with_helpers(
        &mut self,
        helper_shards: &[PartnerShardMigration],
    ) -> Result<(), MigrateError> {
        if !helper_shards.is_empty() {
            return Err(MigrateError::InvalidSegmentMerge);
        }
        let new_params = build_segment_double_params(&self.params)?;
        let migrated = self.segment_double_encoded_state(new_params)?;
        *self = migrated;
        Ok(())
    }

    /// Double `kappa` while keeping the segment size fixed.
    ///
    /// Algebraically this is the old adjacent `s -> s/2` migration.
    pub fn kappa_double(
        &mut self,
        helper_shards: &[PartnerShardMigration],
    ) -> Result<(), MigrateError> {
        if self.params.s <= 1 {
            return Err(MigrateError::InvalidTargetS);
        }
        self.migrate(self.params.s / 2, helper_shards)
    }

    /// Halve `kappa` while keeping the segment size fixed.
    ///
    /// Algebraically this is the old adjacent `s -> 2s` migration.
    pub fn kappa_halve(
        &mut self,
        partner_shards: &[PartnerShardMigration],
        helper_rows: &[HelperRowMigration],
    ) -> Result<(), MigrateError> {
        let target_s = self
            .params
            .s
            .checked_mul(2)
            .ok_or(MigrateError::InvalidTargetS)?;
        self.migrate_with_helper_rows(target_s, partner_shards, helper_rows)
    }

    /// Change to the `kappa_double` target by splitting columns only.
    ///
    /// This is the kappa-double stage-1 primitive used by the network layer before
    /// helper-column hydration. It updates claims into the target lens and
    /// clears all row materialization.
    pub fn migrate_kappa_double_columns(&mut self, target_s: usize) -> Result<(), MigrateError> {
        let new_params = build_migrated_params(&self.params, target_s)?;
        if target_s >= self.params.s {
            return Err(MigrateError::InvalidTargetS);
        }
        let migrated = self.kappa_double_encoded_state(new_params)?;
        *self = migrated;
        Ok(())
    }

    /// Rebuild all currently owned target rows from target-mode columns.
    ///
    /// `helper_shards` may supply extra target columns when the local store
    /// does not hold enough target columns to decode its maintained rows.
    pub fn hydrate_rows_from_target_columns(
        &mut self,
        helper_shards: &[PartnerShardMigration],
    ) -> Result<(), MigrateError> {
        self.hydrate_rows_from_target_columns_with_claims(helper_shards, None)
    }

    fn hydrate_rows_from_target_columns_with_claims(
        &mut self,
        helper_shards: &[PartnerShardMigration],
        claims_override: Option<&[ClaimBundle]>,
    ) -> Result<(), MigrateError> {
        let claims = claims_override
            .map(|claims| normalize_claim_bundles(&self.params, claims.to_vec()))
            .unwrap_or_else(|| self.effective_claims_for_params(&self.params));
        self.coded.clear_rows();
        if claims.is_empty() {
            return Ok(());
        }

        let source_shards = collect_target_shards_for_row_hydration(
            &self.params,
            &self.coded,
            self.segments,
            helper_shards,
        )?;
        let target_row_owners = collect_unique_target_row_owners(&self.params, &claims)?;
        if target_row_owners.is_empty() {
            return Ok(());
        }
        let hydrated_rows = hydrate_unique_row_family_from_target_columns(
            &self.params,
            &self.coded,
            self.segments,
            &source_shards,
            &target_row_owners,
        )?;
        for (row_label_bytes, (row_label, owner_key)) in target_row_owners {
            let row_segments = hydrated_rows
                .get(&row_label_bytes)
                .ok_or(MigrateError::MissingLocalRows)?;
            self.coded.insert_row_family_segments(
                &self.params,
                RowKey::new(owner_key, row_label),
                row_segments.clone(),
            );
        }
        Ok(())
    }

    /// Read a value from the array.
    ///
    /// Returns the default value when `i >= len`.
    pub fn read(&self, i: Index) -> Result<Value, CodedReadError> {
        if i >= self.len {
            return Ok(encode_value(fp_zero()));
        }
        let (seg, logical_offset) = self.index_to_segment(i);
        let alpha = logical_offset_to_alpha_for_segment(&self.params, seg, logical_offset);
        let z_eval = self.params.column_coordinate(alpha);
        let shares = self.collect_shard_replies(seg, alpha);
        let value = decode_from_shares_at_z(&self.params, z_eval, &shares)?;
        Ok(encode_value(value))
    }

    /// Read a value using the raw FFT-grid layout.
    ///
    /// Raw reads address allocated segment storage, not the public logical
    /// length. This matters after layout permutations, where a live partial
    /// tail segment can place active cells at high raw grid indices.
    pub fn read_raw(&self, i: Index) -> Result<Value, CodedReadError> {
        let (seg, j) = self.index_to_segment(i);
        if seg >= self.segments {
            return Ok(encode_value(fp_zero()));
        }
        let alpha = grid_index_to_alpha_for_segment(&self.params, seg, j);
        let z_eval = self.params.column_coordinate(alpha);
        let shares = self.collect_shard_replies(seg, alpha);
        let value = decode_from_shares_at_z(&self.params, z_eval, &shares)?;
        Ok(encode_value(value))
    }

    /// Read one aligned semantic block using the maintained row view.
    pub fn read_block(&self, block: Index) -> Result<Vec<Value>, CodedReadError> {
        let pos = semantic_block_position(&self.params, block);
        if pos.segment >= self.segments {
            return Ok(Vec::new());
        }
        let rows = self.handle_read_block(block);
        let values = decode_block_from_rows(&self.params, block, &rows)?;
        Ok(values.into_iter().map(encode_value).collect())
    }

    /// Apply a batch of deltas using the provided update mode.
    pub fn apply_deltas(
        &mut self,
        deltas: &[(Index, Fp)],
        mode: DeltaMode,
    ) -> Result<(), WriteError> {
        if deltas.is_empty() {
            return Ok(());
        }
        let b = public_values_per_segment(&self.params) as u64;

        if self.logical_deltas_are_sorted_and_valid(deltas) {
            return self.apply_sorted_logical_deltas(deltas, b, mode);
        }

        self.apply_unsorted_logical_deltas(deltas, b, mode)
    }

    fn logical_deltas_are_sorted_and_valid(&self, deltas: &[(Index, Fp)]) -> bool {
        let mut previous = None;
        for &(idx, _) in deltas {
            if idx >= self.len {
                return false;
            }
            if let Some(previous_idx) = previous {
                if idx < previous_idx {
                    return false;
                }
            }
            previous = Some(idx);
        }
        true
    }

    fn apply_sorted_logical_deltas(
        &mut self,
        deltas: &[(Index, Fp)],
        b: u64,
        mode: DeltaMode,
    ) -> Result<(), WriteError> {
        let mut current_seg = None;
        let mut segment_deltas =
            Vec::with_capacity(deltas.len().min(public_values_per_segment(&self.params)));

        for &(idx, delta) in deltas {
            let seg = idx / b;
            if current_seg != Some(seg) {
                if let Some(seg) = current_seg {
                    self.apply_segment_grid_deltas(seg, &segment_deltas, mode)?;
                    segment_deltas.clear();
                }
                current_seg = Some(seg);
            }
            let logical_offset = (idx % b) as usize;
            let grid_index = logical_offset_to_grid_index(&self.params, logical_offset);
            segment_deltas.push((grid_index, delta));
        }

        if let Some(seg) = current_seg {
            self.apply_segment_grid_deltas(seg, &segment_deltas, mode)?;
        }

        Ok(())
    }

    fn apply_unsorted_logical_deltas(
        &mut self,
        deltas: &[(Index, Fp)],
        b: u64,
        mode: DeltaMode,
    ) -> Result<(), WriteError> {
        let mut by_seg: HashMap<u64, Vec<(usize, Fp)>> = HashMap::new();
        for &(idx, delta) in deltas {
            if idx >= self.len {
                return Err(WriteError::InvalidIndex);
            }
            let seg = idx / b;
            let logical_offset = (idx % b) as usize;
            let grid_index = logical_offset_to_grid_index(&self.params, logical_offset);
            by_seg.entry(seg).or_default().push((grid_index, delta));
        }
        let mut segments: Vec<(u64, Vec<(usize, Fp)>)> = by_seg.into_iter().collect();
        segments.sort_by_key(|(seg, _)| *seg);
        for (_, deltas) in segments.iter_mut() {
            deltas.sort_by_key(|(j, _)| *j);
        }
        for (seg, deltas) in segments {
            self.apply_segment_grid_deltas(seg, &deltas, mode)?;
        }
        Ok(())
    }

    fn apply_segment_grid_deltas(
        &mut self,
        seg: u64,
        deltas: &[(usize, Fp)],
        mode: DeltaMode,
    ) -> Result<(), WriteError> {
        let seg_usize = usize::try_from(seg).map_err(|_| WriteError::InvalidIndex)?;
        let rho = segment_domain_rho(&self.params, seg_usize);
        if rho != fp_one() {
            for &(grid_index, _) in deltas {
                if grid_index >= self.params.n_grid {
                    return Err(WriteError::InvalidIndex);
                }
            }
            if should_apply_coset_deltas_via_coefficients(mode, deltas.len()) {
                self.apply_segment_grid_deltas_on_coset_via_coefficients(seg_usize, rho, deltas);
                return Ok(());
            }
            apply_deltas_to_segment_on_coset(&self.params, &mut self.coded, seg_usize, rho, deltas);
            apply_row_deltas_to_segment_on_coset(
                &self.params,
                &mut self.coded,
                seg_usize,
                rho,
                deltas,
            );
            return Ok(());
        }
        apply_deltas_to_segment_with_mode(&self.params, &mut self.coded, seg_usize, deltas, mode);
        apply_row_deltas_to_segment(&self.params, &mut self.coded, seg_usize, deltas);
        Ok(())
    }

    fn apply_segment_grid_deltas_on_coset_via_coefficients(
        &mut self,
        seg: usize,
        rho: Fp,
        deltas: &[(usize, Fp)],
    ) {
        let coeffs = segment_delta_coeffs_from_grid_deltas_on_coset(&self.params, rho, deltas);
        match self.claims.clone() {
            Some(claims) => {
                let (shards, rows) = materialize_claim_segment_state_from_coeffs(
                    &self.params,
                    &coeffs,
                    &claims,
                    None,
                );
                accumulate_segment_state_for_claims_sparse(
                    &self.params,
                    &mut self.coded,
                    seg,
                    &claims,
                    &shards,
                    &rows,
                );
            }
            None => {
                let shards = crate::coded_array::params::compute_initial_shards_from_coeffs(
                    &self.params,
                    &coeffs,
                    None,
                );
                let rows = crate::coded_array::params::compute_initial_rows_from_coeffs(
                    &self.params,
                    &coeffs,
                    None,
                );
                let all_indices: Vec<usize> = (0..self.params.shard_registry.len()).collect();
                accumulate_segment_state(
                    &self.params,
                    &mut self.coded,
                    seg,
                    &all_indices,
                    &shards,
                    &rows,
                );
            }
        }
    }

    /// Apply a delta patch to one aligned semantic block in the current mode.
    pub fn apply_block_deltas(
        &mut self,
        block: Index,
        deltas: &[BlockDelta],
        mode: DeltaMode,
    ) -> Result<(), WriteError> {
        if deltas.is_empty() {
            return Ok(());
        }
        let pos = semantic_block_position(&self.params, block);
        if pos.segment >= self.segments {
            return Err(WriteError::InvalidIndex);
        }
        let seg_base = pos.segment as u64 * public_values_per_segment(&self.params) as u64;
        let block_grid_indices = block_grid_indices(&self.params, pos.block_in_segment);
        let mut grid_deltas = Vec::with_capacity(deltas.len());
        let mut block_delta_values = vec![fp_zero(); self.params.s];
        for &(offset_in_block, delta) in deltas {
            if offset_in_block >= self.params.s {
                return Err(WriteError::InvalidIndex);
            }
            let grid_index = block_grid_indices[offset_in_block];
            if grid_index >= public_values_per_segment(&self.params) {
                return Err(WriteError::InvalidIndex);
            }
            let logical_offset = self.params.logical_offset_for_grid_index(grid_index);
            let idx = seg_base + logical_offset as u64;
            if idx >= self.len {
                return Err(WriteError::InvalidIndex);
            }
            grid_deltas.push((grid_index, delta));
            block_delta_values[offset_in_block] += delta;
        }
        grid_deltas.sort_by_key(|(j, _)| *j);
        if segment_domain_rho(&self.params, pos.segment) != fp_one() {
            return self.apply_segment_grid_deltas(pos.segment as u64, &grid_deltas, mode);
        }
        if matches!(mode, DeltaMode::Aggregate) {
            let eval_points = block_alphas(&self.params, pos.block_in_segment);
            let weights = precompute_barycentric(&eval_points);
            apply_block_deltas_to_segment(
                &self.params,
                &mut self.coded,
                pos.segment,
                pos.block_in_segment,
                &block_delta_values,
                &weights,
            );
            apply_block_row_deltas_to_segment(
                &self.params,
                &mut self.coded,
                pos.segment,
                pos.block_in_segment,
                &block_delta_values,
                &weights,
            );
        } else {
            apply_block_deltas_via_points(
                &self.params,
                &mut self.coded,
                pos.segment,
                &grid_deltas,
                mode,
            );
        }
        Ok(())
    }

    /// Apply a batch of deltas using the raw FFT-grid layout.
    pub fn apply_deltas_raw(
        &mut self,
        deltas: &[(Index, Fp)],
        mode: DeltaMode,
    ) -> Result<(), WriteError> {
        if deltas.is_empty() {
            return Ok(());
        }
        let trace = std::env::var("AUTH_WRITE_TRACE")
            .map(|value| value == "1")
            .unwrap_or(false);
        let total_start = trace.then(Instant::now);
        let group_start = trace.then(Instant::now);
        let b = public_values_per_segment(&self.params) as u64;
        let mut by_seg: HashMap<u64, Vec<(usize, Fp)>> = HashMap::new();
        for &(idx, delta) in deltas {
            if idx >= self.len {
                return Err(WriteError::InvalidIndex);
            }
            let seg = idx / b;
            let j = (idx % b) as usize;
            by_seg.entry(seg).or_default().push((j, delta));
        }
        let mut segments: Vec<(u64, Vec<(usize, Fp)>)> = by_seg.into_iter().collect();
        segments.sort_by_key(|(seg, _)| *seg);
        for (_, deltas) in segments.iter_mut() {
            deltas.sort_by_key(|(j, _)| *j);
        }
        let group_elapsed = group_start.map(|start| start.elapsed());
        let mut shard_apply_secs = 0.0f64;
        let row_apply_secs = 0.0f64;
        for (seg, deltas) in segments {
            let shard_start = trace.then(Instant::now);
            self.apply_segment_grid_deltas(seg, &deltas, mode)?;
            if let Some(start) = shard_start {
                shard_apply_secs += start.elapsed().as_secs_f64();
            }
        }
        if trace {
            eprintln!(
                "[auth-write-trace] unauth_apply_raw deltas={} group_sort_ms={:.3} shard_apply_ms={:.3} row_apply_ms={:.3} total_ms={:.3}",
                deltas.len(),
                group_elapsed.unwrap().as_secs_f64() * 1_000.0,
                shard_apply_secs * 1_000.0,
                row_apply_secs * 1_000.0,
                total_start.unwrap().elapsed().as_secs_f64() * 1_000.0,
            );
        }
        Ok(())
    }

    /// Apply a single delta at index `i`.
    pub fn write_delta(&mut self, i: Index, delta: Fp) -> Result<(), WriteError> {
        self.write_batch_deltas(&[(i, delta)], DeltaMode::Aggregate)
    }

    /// Apply a single delta inside one aligned semantic block.
    pub fn write_block_delta(
        &mut self,
        block: Index,
        offset_in_block: usize,
        delta: Fp,
    ) -> Result<(), WriteError> {
        self.apply_block_deltas(block, &[(offset_in_block, delta)], DeltaMode::Aggregate)
    }

    /// Apply a single delta using the raw FFT-grid layout.
    pub fn write_delta_raw(&mut self, i: Index, delta: Fp) -> Result<(), WriteError> {
        self.write_batch_deltas_raw(&[(i, delta)], DeltaMode::Aggregate)
    }

    /// Apply a batch of deltas using the provided update mode.
    pub fn write_batch_deltas(
        &mut self,
        deltas: &[(Index, Fp)],
        mode: DeltaMode,
    ) -> Result<(), WriteError> {
        self.apply_deltas(deltas, mode)
    }

    /// Apply a batch of aligned semantic-block deltas.
    pub fn write_block_batch_deltas(
        &mut self,
        block: Index,
        deltas: &[BlockDelta],
        mode: DeltaMode,
    ) -> Result<(), WriteError> {
        self.apply_block_deltas(block, deltas, mode)
    }

    /// Apply a batch of deltas using the raw FFT-grid layout.
    pub fn write_batch_deltas_raw(
        &mut self,
        deltas: &[(Index, Fp)],
        mode: DeltaMode,
    ) -> Result<(), WriteError> {
        self.apply_deltas_raw(deltas, mode)
    }

    /// Handle a read request by returning shard values for this store.
    pub(crate) fn handle_read(&self, idx: Index) -> Vec<ShardShare> {
        let b = public_values_per_segment(&self.params) as u64;
        let seg = idx / b;
        let logical_offset = (idx % b) as usize;
        let seg_usize = match usize::try_from(seg) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        if seg_usize >= self.segments {
            return Vec::new();
        }
        let alpha = logical_offset_to_alpha_for_segment(&self.params, seg_usize, logical_offset);
        self.collect_shard_replies(seg_usize, alpha)
    }

    /// Handle a batch read request by returning shard values for each index.
    pub(crate) fn handle_multi_read(&self, idxs: &[Index]) -> Vec<Vec<ShardShare>> {
        self.handle_read_batch_inner(idxs, true)
    }

    /// Handle an aligned block read request by returning maintained row replies.
    pub(crate) fn handle_read_block(&self, block: Index) -> Vec<RowShare> {
        let pos = semantic_block_position(&self.params, block);
        if pos.segment >= self.segments {
            return Vec::new();
        }
        let z_eval = logical_block_to_column_coordinate_for_segment(&self.params, block);
        collect_row_samples_batch_at_z(&self.params, &self.coded, pos.segment, &[z_eval])
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    /// Return the locally stored primary shard vectors for one segment.
    ///
    /// This path is a raw dump only: it does not evaluate shard replies and it
    /// does not repair missing shards from rows.
    pub(crate) fn handle_read_segment_raw(&self, seg: usize) -> Vec<SegmentPrimaryShard> {
        if seg >= self.segments {
            return Vec::new();
        }
        let mut shards = Vec::new();
        for (registry_index, entry) in
            collect_local_shard_entries(&self.params, &self.coded, Some(seg))
        {
            if let Some(values) = self.coded.shard_segment(&self.params, &entry.key, seg) {
                shards.push(SegmentPrimaryShard {
                    registry_index,
                    values: values.to_vec(),
                });
            }
        }
        shards
    }

    /// Return maintained row replies at one explicit column coordinate.
    pub(crate) fn handle_read_rows_at_z(&self, seg: usize, z_eval: Fp) -> Vec<RowShare> {
        if seg >= self.segments {
            return Vec::new();
        }
        collect_row_samples_at_z(&self.params, &self.coded, seg, z_eval)
    }

    /// Return maintained row replies for multiple explicit column coordinates.
    pub(crate) fn handle_read_rows_batch_at_z(
        &self,
        seg: usize,
        z_evals: &[Fp],
    ) -> Vec<Vec<RowShare>> {
        if seg >= self.segments {
            return vec![Vec::new(); z_evals.len()];
        }
        collect_row_samples_batch_at_z(&self.params, &self.coded, seg, z_evals)
    }

    /// Return maintained row replies for a batch of segment/column requests.
    pub(crate) fn handle_read_rows_matrix(
        &self,
        requests: &[crate::unauth::service::RowReadRequest],
    ) -> Vec<Vec<Vec<RowShare>>> {
        let mut rows = requests
            .iter()
            .map(|request| vec![Vec::new(); request.z_evals.len()])
            .collect::<Vec<_>>();
        let mut by_seg = BTreeMap::<usize, Vec<(usize, usize, Fp)>>::new();
        for (request_pos, request) in requests.iter().enumerate() {
            for (z_pos, &z_eval) in request.z_evals.iter().enumerate() {
                by_seg
                    .entry(request.seg)
                    .or_default()
                    .push((request_pos, z_pos, z_eval));
            }
        }
        for (seg, items) in by_seg {
            let mut unique_z_evals = Vec::new();
            let mut unique_index_by_bytes = HashMap::new();
            let mut scatter = Vec::with_capacity(items.len());
            for &(_, _, z_eval) in &items {
                let batch_pos = match unique_index_by_bytes.entry(encode_value(z_eval)) {
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        let idx = unique_z_evals.len();
                        unique_z_evals.push(z_eval);
                        entry.insert(idx);
                        idx
                    }
                    std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
                };
                scatter.push(batch_pos);
            }
            let batches = self.handle_read_rows_batch_at_z(seg, &unique_z_evals);
            for ((request_pos, z_pos, _), batch_pos) in items.into_iter().zip(scatter.into_iter()) {
                rows[request_pos][z_pos] = batches.get(batch_pos).cloned().unwrap_or_default();
            }
        }
        rows
    }

    /// Compute one repaired primary shard from locally maintained rows.
    pub(crate) fn handle_repair_shard(
        &self,
        seg: usize,
        registry_index: usize,
    ) -> Result<Vec<Fp>, CodedReadError> {
        if let Some(claims) = &self.claims {
            let is_claimed = claims
                .iter()
                .any(|claim| claim.registry_index == registry_index);
            if !is_claimed {
                return Err(CodedReadError::NotEnoughShards);
            }
        }
        let Some(&entry) = self.params.shard_registry.get(registry_index) else {
            return Err(CodedReadError::NotEnoughShards);
        };
        if let Some(values) = self.coded.shard_segment(&self.params, &entry.key, seg) {
            return Ok(values.to_vec());
        }
        self.repair_primary_shard_from_rows(seg, entry)
    }

    /// Compute a batch of repaired primary shards from local state.
    pub(crate) fn handle_repair_shard_batch(
        &self,
        seg: usize,
        registry_indices: &[usize],
    ) -> Result<Vec<Vec<Fp>>, CodedReadError> {
        if seg >= self.segments {
            return Err(CodedReadError::NotEnoughShards);
        }
        let mut values = vec![Vec::new(); registry_indices.len()];
        let mut missing_positions = Vec::new();
        let mut missing_entries = Vec::new();
        for (pos, &registry_index) in registry_indices.iter().enumerate() {
            if let Some(claims) = &self.claims {
                let is_claimed = claims
                    .iter()
                    .any(|claim| claim.registry_index == registry_index);
                if !is_claimed {
                    return Err(CodedReadError::NotEnoughShards);
                }
            }
            let Some(&entry) = self.params.shard_registry.get(registry_index) else {
                return Err(CodedReadError::NotEnoughShards);
            };
            if let Some(shard_values) = self.coded.shard_segment(&self.params, &entry.key, seg) {
                values[pos] = shard_values.to_vec();
                continue;
            }
            missing_positions.push(pos);
            missing_entries.push(entry);
        }
        if missing_entries.is_empty() {
            return Ok(values);
        }
        let z_evals: Vec<_> = missing_entries.iter().map(|entry| entry.z).collect();
        let row_batches = collect_row_samples_batch_at_z(&self.params, &self.coded, seg, &z_evals);
        let repaired =
            decode_primary_shards_from_row_batches(&self.params, &missing_entries, &row_batches)?;
        for (pos, repaired_values) in missing_positions.into_iter().zip(repaired.into_iter()) {
            values[pos] = repaired_values;
        }
        Ok(values)
    }

    /// Evaluate one batch of owned primary shards at the requested row labels.
    pub(crate) fn handle_eval_shards_at_row_labels(
        &self,
        seg: usize,
        col_labels: &[Fp],
        row_labels: &[Fp],
    ) -> Result<Vec<Vec<Fp>>, CodedReadError> {
        if seg >= self.segments {
            return Err(CodedReadError::NotEnoughShards);
        }
        let mut out = Vec::with_capacity(col_labels.len());
        for &col_label in col_labels {
            let Some((registry_index, entry)) = self
                .params
                .shard_registry
                .iter()
                .copied()
                .enumerate()
                .find(|(_, entry)| entry.z == col_label)
            else {
                return Err(CodedReadError::NotEnoughShards);
            };
            if let Some(claims) = &self.claims {
                let is_claimed = claims
                    .iter()
                    .any(|claim| claim.registry_index == registry_index);
                if !is_claimed {
                    return Err(CodedReadError::NotEnoughShards);
                }
            }
            let local_shard = self.coded.shard_segment(&self.params, &entry.key, seg);
            let repaired_shard;
            let shard_values = if let Some(values) = local_shard {
                values
            } else {
                repaired_shard = self.handle_repair_shard(seg, registry_index)?;
                repaired_shard.as_slice()
            };
            let evals = if row_labels.len() == 1 {
                vec![shard_value_from_vector(
                    &self.params,
                    entry,
                    shard_values,
                    row_labels[0],
                )]
            } else {
                shard_value_from_vector_batch(&self.params, entry, shard_values, row_labels)
            };
            let mut row_values = Vec::with_capacity(row_labels.len());
            for value in evals {
                let Some(value) = value else {
                    return Err(CodedReadError::NotEnoughShards);
                };
                row_values.push(value);
            }
            out.push(row_values);
        }
        Ok(out)
    }

    /// Evaluate one fixed batch of owned primary shards at the requested row labels across many segments.
    pub(crate) fn handle_eval_shards_at_row_labels_matrix(
        &self,
        segs: &[usize],
        col_labels: &[Fp],
        row_labels: &[Fp],
    ) -> Result<Vec<Vec<Vec<Fp>>>, CodedReadError> {
        let mut out = Vec::with_capacity(segs.len());
        for &seg in segs {
            out.push(self.handle_eval_shards_at_row_labels(seg, col_labels, row_labels)?);
        }
        Ok(out)
    }

    /// Reconstruct one missing primary shard from maintained rows.
    pub fn repair_primary_shard_from_rows(
        &self,
        seg: usize,
        entry: ShardRegEntry,
    ) -> Result<Vec<Fp>, CodedReadError> {
        if seg >= self.segments {
            return Err(CodedReadError::NotEnoughShards);
        }
        let rows = collect_row_samples_at_z(&self.params, &self.coded, seg, entry.z);
        decode_primary_shard_from_rows(&self.params, entry, &rows)
    }

    /// Reconstruct one maintained row from primary shards.
    pub fn repair_row_from_shards(
        &self,
        seg: usize,
        row_label: Fp,
    ) -> Result<Vec<Fp>, CodedReadError> {
        if seg >= self.segments {
            return Err(CodedReadError::NotEnoughShards);
        }
        decode_row_from_shards(&self.params, &self.coded, seg, row_label)
    }

    /// Handle a batch read request using the raw FFT-grid layout.
    #[allow(dead_code)]
    pub(crate) fn handle_read_batch_raw(&self, idxs: &[Index]) -> Vec<Vec<ShardShare>> {
        self.handle_read_batch_inner(idxs, false)
    }

    fn handle_read_batch_inner(
        &self,
        idxs: &[Index],
        semantic_layout: bool,
    ) -> Vec<Vec<ShardShare>> {
        if idxs.is_empty() {
            return Vec::new();
        }
        let mut results = Vec::with_capacity(idxs.len());
        for _ in 0..idxs.len() {
            results.push(Vec::new());
        }
        let b = public_values_per_segment(&self.params) as u64;
        let mut by_seg: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (pos, &idx) in idxs.iter().enumerate() {
            let seg = idx / b;
            let logical_offset = (idx % b) as usize;
            let j = if semantic_layout {
                logical_offset_to_grid_index(&self.params, logical_offset)
            } else {
                logical_offset
            };
            let seg_usize = match usize::try_from(seg) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if seg_usize >= self.segments {
                continue;
            }
            by_seg.entry(seg_usize).or_default().push((pos, j));
        }
        for (seg, items) in by_seg {
            let rho = segment_domain_rho(&self.params, seg);
            let grid_indices = items.iter().map(|&(_, j)| j).collect::<Vec<_>>();
            let local_shards = collect_local_shard_entries(&self.params, &self.coded, Some(seg));
            for shares in &mut results {
                shares.reserve(local_shards.len());
            }
            for (reg_idx, entry) in local_shards {
                let replies = shard_value_from_store_batch_on_grid_coset(
                    &self.params,
                    &self.coded,
                    entry,
                    seg,
                    rho,
                    &grid_indices,
                );
                for (idx_in_group, reply) in replies.into_iter().enumerate() {
                    let pos = items[idx_in_group].0;
                    if let Some(value) = reply {
                        results[pos].push(ShardShare {
                            registry_index: reg_idx,
                            value: Some(value),
                        });
                    }
                }
            }
        }
        results
    }

    fn collect_shard_replies(&self, seg: usize, alpha: Fp) -> Vec<ShardShare> {
        let local_shards = collect_local_shard_entries(&self.params, &self.coded, Some(seg));
        let mut shares = Vec::with_capacity(local_shards.len());
        for (idx, entry) in local_shards {
            if let Some(value) =
                shard_value_from_store(&self.params, &self.coded, entry, seg, alpha)
            {
                shares.push(ShardShare {
                    registry_index: idx,
                    value: Some(value),
                });
            }
        }
        shares
    }

    /// Append segments up to the size needed for `new_len`, using a builder.
    pub fn extend_with_segments<F>(
        &mut self,
        new_len: Index,
        build_segment: F,
    ) -> Result<(), ExtendError>
    where
        F: FnMut(usize) -> Vec<Value256>,
    {
        self.extend_with_segments_timed(new_len, build_segment, None)
    }

    /// Append storage up to `new_len`, deriving each logical array value from
    /// its public index. Segment construction remains internal to unauth.
    pub fn extend_with_values<F>(
        &mut self,
        new_len: Index,
        value_at_index: F,
    ) -> Result<(), ExtendError>
    where
        F: FnMut(Index) -> Value256,
    {
        self.extend_with_values_timed(new_len, value_at_index, None)
    }

    /// Timed variant of [`Self::extend_with_values`].
    pub fn extend_with_values_timed<F>(
        &mut self,
        new_len: Index,
        mut value_at_index: F,
        mut timings: Option<&mut SetupTimings>,
    ) -> Result<(), ExtendError>
    where
        F: FnMut(Index) -> Value256,
    {
        if let Some(claims) = self.claims.clone() {
            return self.extend_with_values_for_claims_timed(
                new_len,
                &claims,
                value_at_index,
                timings,
            );
        }
        if new_len <= self.len {
            return Ok(());
        }
        let params = self.params.clone();
        let b = public_values_per_segment(&params) as u64;
        let needed_segments = new_len / b + u64::from(new_len % b != 0);
        let needed_segments =
            usize::try_from(needed_segments).map_err(|_| ExtendError::SegmentOverflow)?;
        let all_indices: Vec<usize> = (0..params.shard_registry.len()).collect();
        let all_claims = default_claim_bundles_for_indices(&params, &all_indices);
        reserve_segment_state_for_claims(
            &params,
            &mut self.coded,
            &all_claims,
            needed_segments - self.segments,
        );
        for seg in self.segments..needed_segments {
            let t_seg = logical_segment_from_index_values(&params, seg, &mut value_at_index);
            let cacheable = segment_domain_rho(&params, seg) == fp_one()
                || segment_values_are_layout_invariant(&t_seg);
            if cacheable {
                if let Some((shards, rows)) = self.full_materialization_cache.get(&t_seg) {
                    append_segment_state_for_claims(
                        &params,
                        &mut self.coded,
                        seg,
                        &all_claims,
                        shards,
                        rows,
                    );
                    continue;
                }
            }

            let materialized =
                materialize_segment_state_for_segment(&params, seg, &t_seg, timings.as_deref_mut());
            append_segment_state_for_claims(
                &params,
                &mut self.coded,
                seg,
                &all_claims,
                &materialized.0,
                &materialized.1,
            );
            if cacheable {
                self.full_materialization_cache.insert(t_seg, materialized);
            }
        }
        self.segments = needed_segments;
        self.len = new_len;
        Ok(())
    }

    /// Append storage up to `new_len`, using caller-provided logical family
    /// keys to reuse materialized coded state without exposing the coding
    /// layout to the caller.
    pub fn extend_with_value_families_timed<FK, FV>(
        &mut self,
        new_len: Index,
        mut family_key_for_range: FK,
        mut logical_values_for_range: FV,
        mut timings: Option<&mut SetupTimings>,
    ) -> Result<(), ExtendError>
    where
        FK: FnMut(Range<Index>) -> LogicalValueFamilyKey,
        FV: FnMut(Range<Index>, &LogicalValueFamilyKey) -> Vec<Value256>,
    {
        if let Some(claims) = self.claims.clone() {
            return self.extend_with_value_families_for_claims_timed(
                new_len,
                &claims,
                family_key_for_range,
                logical_values_for_range,
                timings,
            );
        }
        if new_len <= self.len {
            return Ok(());
        }
        let params = self.params.clone();
        let b = public_values_per_segment(&params) as u64;
        let needed_segments = new_len / b + u64::from(new_len % b != 0);
        let needed_segments =
            usize::try_from(needed_segments).map_err(|_| ExtendError::SegmentOverflow)?;
        let all_indices: Vec<usize> = (0..params.shard_registry.len()).collect();
        let all_claims = default_claim_bundles_for_indices(&params, &all_indices);
        reserve_segment_state_for_claims(
            &params,
            &mut self.coded,
            &all_claims,
            needed_segments - self.segments,
        );
        for seg in self.segments..needed_segments {
            let range = logical_segment_range(&params, seg)?;
            let family_key = family_key_for_range(range.clone());
            let cacheable = segment_domain_rho(&params, seg) == fp_one()
                || family_key_is_layout_invariant(&family_key);
            if cacheable {
                if let Some((shards, rows)) =
                    self.full_family_materialization_cache.get(&family_key)
                {
                    append_segment_state_for_claims(
                        &params,
                        &mut self.coded,
                        seg,
                        &all_claims,
                        shards,
                        rows,
                    );
                    continue;
                }
            }

            let base_coeffs = if let Some(coeffs) = self.family_coeff_cache.get(&family_key) {
                coeffs.clone()
            } else {
                let logical_values = logical_values_for_range(range, &family_key);
                let t_seg = logical_segment_from_logical_values(&params, &logical_values)?;
                let coeffs = crate::coded_array::params::compute_segment_coeffs_from_values(
                    &params,
                    &t_seg,
                    timings.as_deref_mut(),
                );
                self.family_coeff_cache
                    .insert(family_key.clone(), coeffs.clone());
                coeffs
            };
            let coeffs = coeffs_for_segment_rho(&base_coeffs, segment_domain_rho(&params, seg));
            let materialized =
                materialize_segment_state_from_coeffs(&params, &coeffs, timings.as_deref_mut());
            append_segment_state_for_claims(
                &params,
                &mut self.coded,
                seg,
                &all_claims,
                &materialized.0,
                &materialized.1,
            );
            if cacheable {
                self.full_family_materialization_cache
                    .insert(family_key, materialized);
            }
        }
        self.segments = needed_segments;
        self.len = new_len;
        Ok(())
    }

    /// Append storage for selected registry indices, deriving logical values by
    /// public index.
    pub fn extend_with_values_filtered_timed<F>(
        &mut self,
        new_len: Index,
        shard_indices: &[usize],
        mut value_at_index: F,
        mut timings: Option<&mut SetupTimings>,
    ) -> Result<(), ExtendError>
    where
        F: FnMut(Index) -> Value256,
    {
        if new_len <= self.len {
            return Ok(());
        }
        let params = self.params.clone();
        let b = public_values_per_segment(&params) as u64;
        let needed_segments = new_len / b + u64::from(new_len % b != 0);
        let needed_segments =
            usize::try_from(needed_segments).map_err(|_| ExtendError::SegmentOverflow)?;
        let mut indices = shard_indices.to_vec();
        indices.sort_unstable();
        indices.dedup();
        if indices.is_empty() {
            self.segments = needed_segments;
            self.len = new_len;
            return Ok(());
        }
        let claims = default_claim_bundles_for_indices(&params, &indices);
        reserve_segment_state_for_claims(
            &params,
            &mut self.coded,
            &claims,
            needed_segments - self.segments,
        );
        for seg in self.segments..needed_segments {
            let t_seg = logical_segment_from_index_values(&params, seg, &mut value_at_index);
            let cacheable = segment_domain_rho(&params, seg) == fp_one()
                || segment_values_are_layout_invariant(&t_seg);
            if cacheable {
                if let Some((shards, rows)) = self.full_materialization_cache.get(&t_seg) {
                    append_segment_state_for_claims(
                        &params,
                        &mut self.coded,
                        seg,
                        &claims,
                        shards,
                        rows,
                    );
                    continue;
                }
            }

            let materialized =
                materialize_segment_state_for_segment(&params, seg, &t_seg, timings.as_deref_mut());
            append_segment_state_for_claims(
                &params,
                &mut self.coded,
                seg,
                &claims,
                &materialized.0,
                &materialized.1,
            );
            if cacheable {
                self.full_materialization_cache.insert(t_seg, materialized);
            }
        }
        self.segments = needed_segments;
        self.len = new_len;
        Ok(())
    }

    /// Append storage for selected registry indices, using logical family keys
    /// to reuse materialized state.
    pub fn extend_with_value_families_filtered_timed<FK, FV>(
        &mut self,
        new_len: Index,
        shard_indices: &[usize],
        mut family_key_for_range: FK,
        mut logical_values_for_range: FV,
        mut timings: Option<&mut SetupTimings>,
    ) -> Result<(), ExtendError>
    where
        FK: FnMut(Range<Index>) -> LogicalValueFamilyKey,
        FV: FnMut(Range<Index>, &LogicalValueFamilyKey) -> Vec<Value256>,
    {
        if new_len <= self.len {
            return Ok(());
        }
        let params = self.params.clone();
        let b = public_values_per_segment(&params) as u64;
        let needed_segments = new_len / b + u64::from(new_len % b != 0);
        let needed_segments =
            usize::try_from(needed_segments).map_err(|_| ExtendError::SegmentOverflow)?;
        let mut indices = shard_indices.to_vec();
        indices.sort_unstable();
        indices.dedup();
        if indices.is_empty() {
            self.segments = needed_segments;
            self.len = new_len;
            return Ok(());
        }
        let claims = default_claim_bundles_for_indices(&params, &indices);
        reserve_segment_state_for_claims(
            &params,
            &mut self.coded,
            &claims,
            needed_segments - self.segments,
        );
        for seg in self.segments..needed_segments {
            let range = logical_segment_range(&params, seg)?;
            let family_key = family_key_for_range(range.clone());
            let cacheable = segment_domain_rho(&params, seg) == fp_one()
                || family_key_is_layout_invariant(&family_key);
            if cacheable {
                if let Some((shards, rows)) =
                    self.full_family_materialization_cache.get(&family_key)
                {
                    append_segment_state_for_claims(
                        &params,
                        &mut self.coded,
                        seg,
                        &claims,
                        shards,
                        rows,
                    );
                    continue;
                }
            }

            let base_coeffs = if let Some(coeffs) = self.family_coeff_cache.get(&family_key) {
                coeffs.clone()
            } else {
                let logical_values = logical_values_for_range(range, &family_key);
                let t_seg = logical_segment_from_logical_values(&params, &logical_values)?;
                let coeffs = crate::coded_array::params::compute_segment_coeffs_from_values(
                    &params,
                    &t_seg,
                    timings.as_deref_mut(),
                );
                self.family_coeff_cache
                    .insert(family_key.clone(), coeffs.clone());
                coeffs
            };
            let coeffs = coeffs_for_segment_rho(&base_coeffs, segment_domain_rho(&params, seg));
            let materialized =
                materialize_segment_state_from_coeffs(&params, &coeffs, timings.as_deref_mut());
            append_segment_state_for_claims(
                &params,
                &mut self.coded,
                seg,
                &claims,
                &materialized.0,
                &materialized.1,
            );
            if cacheable {
                self.full_family_materialization_cache
                    .insert(family_key, materialized);
            }
        }
        self.segments = needed_segments;
        self.len = new_len;
        Ok(())
    }

    /// Append claim-bound storage, deriving logical values by public index.
    pub fn extend_with_values_for_claims_timed<F>(
        &mut self,
        new_len: Index,
        claims: &[ClaimBundle],
        mut value_at_index: F,
        mut timings: Option<&mut SetupTimings>,
    ) -> Result<(), ExtendError>
    where
        F: FnMut(Index) -> Value256,
    {
        if new_len <= self.len {
            return Ok(());
        }
        let params = self.params.clone();
        let b = public_values_per_segment(&params) as u64;
        let needed_segments = new_len / b + u64::from(new_len % b != 0);
        let needed_segments =
            usize::try_from(needed_segments).map_err(|_| ExtendError::SegmentOverflow)?;
        if claims.is_empty() {
            self.segments = needed_segments;
            self.len = new_len;
            return Ok(());
        }
        reserve_segment_state_for_claims(
            &params,
            &mut self.coded,
            claims,
            needed_segments - self.segments,
        );
        let claim_cache_key = claims.to_vec();
        for seg in self.segments..needed_segments {
            let t_seg = logical_segment_from_index_values(&params, seg, &mut value_at_index);
            let cache_key = (claim_cache_key.clone(), t_seg.clone());
            let cacheable = segment_domain_rho(&params, seg) == fp_one()
                || segment_values_are_layout_invariant(&t_seg);
            if cacheable {
                if let Some((shards, rows)) = self.claim_materialization_cache.get(&cache_key) {
                    append_segment_state_for_claims_sparse(
                        &params,
                        &mut self.coded,
                        seg,
                        claims,
                        shards,
                        rows,
                    );
                    continue;
                }
            }

            let materialized = materialize_claim_segment_state_for_segment(
                &params,
                seg,
                &t_seg,
                claims,
                timings.as_deref_mut(),
            );
            append_segment_state_for_claims_sparse(
                &params,
                &mut self.coded,
                seg,
                claims,
                &materialized.0,
                &materialized.1,
            );
            if cacheable {
                self.claim_materialization_cache
                    .insert(cache_key, materialized);
            }
        }
        self.segments = needed_segments;
        self.len = new_len;
        Ok(())
    }

    /// Append claim-bound storage, using logical family keys to reuse
    /// materialized state for repeated value families.
    pub fn extend_with_value_families_for_claims_timed<FK, FV>(
        &mut self,
        new_len: Index,
        claims: &[ClaimBundle],
        mut family_key_for_range: FK,
        mut logical_values_for_range: FV,
        mut timings: Option<&mut SetupTimings>,
    ) -> Result<(), ExtendError>
    where
        FK: FnMut(Range<Index>) -> LogicalValueFamilyKey,
        FV: FnMut(Range<Index>, &LogicalValueFamilyKey) -> Vec<Value256>,
    {
        if new_len <= self.len {
            return Ok(());
        }
        let params = self.params.clone();
        let b = public_values_per_segment(&params) as u64;
        let needed_segments = new_len / b + u64::from(new_len % b != 0);
        let needed_segments =
            usize::try_from(needed_segments).map_err(|_| ExtendError::SegmentOverflow)?;
        if claims.is_empty() {
            self.segments = needed_segments;
            self.len = new_len;
            return Ok(());
        }
        reserve_segment_state_for_claims(
            &params,
            &mut self.coded,
            claims,
            needed_segments - self.segments,
        );
        let claim_cache_key = claims.to_vec();
        for seg in self.segments..needed_segments {
            let range = logical_segment_range(&params, seg)?;
            let family_key = family_key_for_range(range.clone());
            let cache_key = (claim_cache_key.clone(), family_key.clone());
            let cacheable = segment_domain_rho(&params, seg) == fp_one()
                || family_key_is_layout_invariant(&family_key);
            if cacheable {
                if let Some((shards, rows)) =
                    self.claim_family_materialization_cache.get(&cache_key)
                {
                    append_segment_state_for_claims_sparse(
                        &params,
                        &mut self.coded,
                        seg,
                        claims,
                        shards,
                        rows,
                    );
                    continue;
                }
            }

            let base_coeffs = if let Some(coeffs) = self.family_coeff_cache.get(&family_key) {
                coeffs.clone()
            } else {
                let logical_values = logical_values_for_range(range, &family_key);
                let t_seg = logical_segment_from_logical_values(&params, &logical_values)?;
                let coeffs = crate::coded_array::params::compute_segment_coeffs_from_values(
                    &params,
                    &t_seg,
                    timings.as_deref_mut(),
                );
                self.family_coeff_cache
                    .insert(family_key.clone(), coeffs.clone());
                coeffs
            };
            let coeffs = coeffs_for_segment_rho(&base_coeffs, segment_domain_rho(&params, seg));
            let materialized = materialize_claim_segment_state_from_coeffs(
                &params,
                &coeffs,
                claims,
                timings.as_deref_mut(),
            );
            append_segment_state_for_claims_sparse(
                &params,
                &mut self.coded,
                seg,
                claims,
                &materialized.0,
                &materialized.1,
            );
            if cacheable {
                self.claim_family_materialization_cache
                    .insert(cache_key, materialized);
            }
        }
        self.segments = needed_segments;
        self.len = new_len;
        Ok(())
    }

    /// Append segments, storing shards only for the selected registry indices.
    pub fn extend_with_segments_filtered<F>(
        &mut self,
        new_len: Index,
        shard_indices: &[usize],
        build_segment: F,
    ) -> Result<(), ExtendError>
    where
        F: FnMut(usize) -> Vec<Value256>,
    {
        self.extend_with_segments_filtered_timed(new_len, shard_indices, build_segment, None)
    }

    /// Append segments, storing state only for the selected explicit claim bundles.
    pub fn extend_with_segments_for_claims<F>(
        &mut self,
        new_len: Index,
        claims: &[ClaimBundle],
        build_segment: F,
    ) -> Result<(), ExtendError>
    where
        F: FnMut(usize) -> Vec<Value256>,
    {
        self.extend_with_claims(new_len, claims, build_segment)
    }

    /// Append segments up to the size needed for `new_len`, using a builder and optional timings.
    pub fn extend_with_segments_timed<F>(
        &mut self,
        new_len: Index,
        mut build_segment: F,
        mut timings: Option<&mut SetupTimings>,
    ) -> Result<(), ExtendError>
    where
        F: FnMut(usize) -> Vec<Value256>,
    {
        if let Some(claims) = self.claims.clone() {
            return self.extend_with_claims_timed(new_len, &claims, build_segment, timings);
        }
        if new_len <= self.len {
            return Ok(());
        }
        let b = public_values_per_segment(&self.params) as u64;
        let needed_segments = new_len / b + u64::from(new_len % b != 0);
        let needed_segments =
            usize::try_from(needed_segments).map_err(|_| ExtendError::SegmentOverflow)?;
        let all_indices: Vec<usize> = (0..self.params.shard_registry.len()).collect();
        let all_claims = default_claim_bundles_for_indices(&self.params, &all_indices);
        reserve_segment_state_for_claims(
            &self.params,
            &mut self.coded,
            &all_claims,
            needed_segments - self.segments,
        );
        for seg in self.segments..needed_segments {
            let t_seg = build_segment(seg);
            let (shards, rows) = materialize_segment_state_for_segment(
                &self.params,
                seg,
                &t_seg,
                timings.as_deref_mut(),
            );
            append_segment_state_for_claims(
                &self.params,
                &mut self.coded,
                seg,
                &all_claims,
                &shards,
                &rows,
            );
        }
        self.segments = needed_segments;
        self.len = new_len;
        Ok(())
    }

    /// Append segments, storing shards only for selected registry indices.
    pub fn extend_with_segments_filtered_timed<F>(
        &mut self,
        new_len: Index,
        shard_indices: &[usize],
        mut build_segment: F,
        mut timings: Option<&mut SetupTimings>,
    ) -> Result<(), ExtendError>
    where
        F: FnMut(usize) -> Vec<Value256>,
    {
        if new_len <= self.len {
            return Ok(());
        }
        let b = public_values_per_segment(&self.params) as u64;
        let needed_segments = new_len / b + u64::from(new_len % b != 0);
        let needed_segments =
            usize::try_from(needed_segments).map_err(|_| ExtendError::SegmentOverflow)?;
        let mut indices = shard_indices.to_vec();
        indices.sort_unstable();
        indices.dedup();
        if indices.is_empty() {
            self.segments = needed_segments;
            self.len = new_len;
            return Ok(());
        }
        let claims = default_claim_bundles_for_indices(&self.params, &indices);
        reserve_segment_state_for_claims(
            &self.params,
            &mut self.coded,
            &claims,
            needed_segments - self.segments,
        );
        for seg in self.segments..needed_segments {
            let t_seg = build_segment(seg);
            let (shards, rows) = materialize_segment_state_for_segment(
                &self.params,
                seg,
                &t_seg,
                timings.as_deref_mut(),
            );
            append_segment_state_for_claims(
                &self.params,
                &mut self.coded,
                seg,
                &claims,
                &shards,
                &rows,
            );
        }
        self.segments = needed_segments;
        self.len = new_len;
        Ok(())
    }

    fn index_to_segment(&self, i: Index) -> (usize, usize) {
        public_index_to_segment(&self.params, i)
    }

    fn extend_with_claims<F>(
        &mut self,
        new_len: Index,
        claims: &[ClaimBundle],
        build_segment: F,
    ) -> Result<(), ExtendError>
    where
        F: FnMut(usize) -> Vec<Value256>,
    {
        self.extend_with_claims_timed(new_len, claims, build_segment, None)
    }

    fn extend_with_claims_timed<F>(
        &mut self,
        new_len: Index,
        claims: &[ClaimBundle],
        mut build_segment: F,
        mut timings: Option<&mut SetupTimings>,
    ) -> Result<(), ExtendError>
    where
        F: FnMut(usize) -> Vec<Value256>,
    {
        if new_len <= self.len {
            return Ok(());
        }
        let b = public_values_per_segment(&self.params) as u64;
        let needed_segments = new_len / b + u64::from(new_len % b != 0);
        let needed_segments =
            usize::try_from(needed_segments).map_err(|_| ExtendError::SegmentOverflow)?;
        if claims.is_empty() {
            self.segments = needed_segments;
            self.len = new_len;
            return Ok(());
        }
        reserve_segment_state_for_claims(
            &self.params,
            &mut self.coded,
            claims,
            needed_segments - self.segments,
        );
        for seg in self.segments..needed_segments {
            let t_seg = build_segment(seg);
            let (shards, rows) = materialize_claim_segment_state_for_segment(
                &self.params,
                seg,
                &t_seg,
                claims,
                timings.as_deref_mut(),
            );
            append_segment_state_for_claims_sparse(
                &self.params,
                &mut self.coded,
                seg,
                claims,
                &shards,
                &rows,
            );
        }
        self.segments = needed_segments;
        self.len = new_len;
        Ok(())
    }

    fn effective_claims_for_params(&self, params: &CodedArrayParams) -> Vec<ClaimBundle> {
        match &self.claims {
            Some(claims) => normalize_claim_bundles(params, claims.clone()),
            None => {
                derive_claim_bundles_from_coded_state(params, &self.coded).unwrap_or_else(|| {
                    default_claim_bundles_for_indices(
                        params,
                        &(0..params.shard_registry.len()).collect::<Vec<_>>(),
                    )
                })
            }
        }
    }

    fn extend_default_segments(&mut self, new_len: Index) -> Result<(), ExtendError> {
        let b = public_values_per_segment(&self.params) as u64;
        let needed_segments = new_len / b + u64::from(new_len % b != 0);
        let needed_segments =
            usize::try_from(needed_segments).map_err(|_| ExtendError::SegmentOverflow)?;
        if needed_segments <= self.segments {
            self.len = new_len;
            return Ok(());
        }

        let t_seg = default_segment_values(&self.params);
        let all_indices: Vec<usize> = (0..self.params.shard_registry.len()).collect();
        let all_claims = default_claim_bundles_for_indices(&self.params, &all_indices);
        reserve_segment_state_for_claims(
            &self.params,
            &mut self.coded,
            &all_claims,
            needed_segments - self.segments,
        );
        for seg in self.segments..needed_segments {
            let (shards, rows) =
                materialize_segment_state_for_segment(&self.params, seg, &t_seg, None);
            append_segment_state_for_claims(
                &self.params,
                &mut self.coded,
                seg,
                &all_claims,
                &shards,
                &rows,
            );
        }
        self.segments = needed_segments;
        self.len = new_len;
        Ok(())
    }

    fn extend_default_segments_for_claims(
        &mut self,
        new_len: Index,
        claims: &[ClaimBundle],
    ) -> Result<(), ExtendError> {
        let b = public_values_per_segment(&self.params) as u64;
        let needed_segments = new_len / b + u64::from(new_len % b != 0);
        let needed_segments =
            usize::try_from(needed_segments).map_err(|_| ExtendError::SegmentOverflow)?;
        if claims.is_empty() {
            self.segments = needed_segments;
            self.len = new_len;
            return Ok(());
        }
        if needed_segments <= self.segments {
            self.len = new_len;
            return Ok(());
        }

        let t_seg = default_segment_values(&self.params);
        reserve_segment_state_for_claims(
            &self.params,
            &mut self.coded,
            claims,
            needed_segments - self.segments,
        );
        for seg in self.segments..needed_segments {
            let (shards, rows) = materialize_claim_segment_state_for_segment(
                &self.params,
                seg,
                &t_seg,
                claims,
                None,
            );
            append_segment_state_for_claims_sparse(
                &self.params,
                &mut self.coded,
                seg,
                claims,
                &shards,
                &rows,
            );
        }
        self.segments = needed_segments;
        self.len = new_len;
        Ok(())
    }

    fn kappa_halve_encoded_state(
        &self,
        new_params: CodedArrayParams,
        partner_shards: &[PartnerShardMigration],
        helper_rows: &[HelperRowMigration],
    ) -> Result<UnauthStore, MigrateError> {
        let source_claims = self.effective_claims_for_params(&self.params);
        let source_rows = if self.segments > 0 {
            Some(collect_source_rows_for_kappa_halve(
                &self.params,
                &self.coded,
                self.segments,
                helper_rows,
            )?)
        } else {
            None
        };
        let sibling_kappa_halve = registry_supports_sibling_kappa_halve(&self.params);
        let target_claims = if sibling_kappa_halve {
            merge_claim_bundles_for_kappa_halve(&self.params, &new_params, &source_claims)
        } else {
            let target_indices: Vec<_> = source_claims
                .iter()
                .map(|claim| claim.registry_index)
                .collect();
            default_claim_bundles_for_indices(&new_params, &target_indices)
        };
        let mut partner_by_index = HashMap::new();
        for partner in partner_shards {
            if partner.registry_index >= self.params.shard_registry.len()
                || partner.segments.len() != self.segments
            {
                return Err(MigrateError::InvalidPartnerShards);
            }
            if partner
                .segments
                .iter()
                .any(|segment_values| segment_values.len() != self.params.s)
            {
                return Err(MigrateError::InvalidPartnerShards);
            }
            partner_by_index.insert(partner.registry_index, partner);
        }
        let mut partner_scratch: HashMap<usize, Vec<Vec<Fp>>> = HashMap::new();

        let mut migrated = if self.claims.is_some() {
            UnauthStore::new_with_claims(new_params.clone(), target_claims.clone())
        } else {
            UnauthStore::new(new_params.clone())
        };
        let source_claims_by_registry: HashMap<usize, &ClaimBundle> = source_claims
            .iter()
            .map(|claim| (claim.registry_index, claim))
            .collect();
        for target_claim in &target_claims {
            let registry_index = target_claim.registry_index;
            let (source_registry_index, partner_registry_index) = if sibling_kappa_halve {
                let left = 2 * registry_index;
                (left, Some(left + 1))
            } else {
                (registry_index, None)
            };
            if !source_claims_by_registry.contains_key(&source_registry_index) {
                return Err(MigrateError::MissingPartnerShards);
            }
            let old_entry = self.params.shard_registry[source_registry_index];
            let new_entry = new_params.shard_registry[registry_index];
            let partner_entry = kappa_halve_partner_entry(&self.params, old_entry);
            let mut new_shards = Vec::with_capacity(self.segments);
            for seg in 0..self.segments {
                let old_values = self
                    .coded
                    .shard_segment(&self.params, &old_entry.key, seg)
                    .ok_or(MigrateError::MissingPartnerShards)?;
                let partner_values = if let Some(partner_registry_index) = partner_registry_index {
                    let partner_key = self.params.shard_registry[partner_registry_index].key;
                    if let Some(local_partner_values) =
                        self.coded.shard_segment(&self.params, &partner_key, seg)
                    {
                        local_partner_values
                    } else {
                        match partner_by_index.get(&source_registry_index) {
                            Some(partner) => partner
                                .segments
                                .get(seg)
                                .ok_or(MigrateError::InvalidPartnerShards)?,
                            None => {
                                let repaired = partner_scratch
                                    .entry(source_registry_index)
                                    .or_insert_with(|| {
                                        let mut repaired = Vec::with_capacity(self.segments);
                                        for source_seg in 0..self.segments {
                                            let values = source_rows
                                                .as_ref()
                                                .and_then(|rows| {
                                                    repair_primary_shard_from_source_rows(
                                                        &self.params,
                                                        &self.coded,
                                                        rows,
                                                        source_seg,
                                                        partner_entry,
                                                    )
                                                    .ok()
                                                })
                                                .unwrap_or_default();
                                            repaired.push(values);
                                        }
                                        repaired
                                    });
                                let values = repaired
                                    .get(seg)
                                    .ok_or(MigrateError::MissingPartnerShards)?;
                                if values.len() != self.params.s {
                                    return Err(MigrateError::MissingPartnerShards);
                                }
                                values
                            }
                        }
                    }
                } else {
                    match partner_by_index.get(&source_registry_index) {
                        Some(partner) => partner
                            .segments
                            .get(seg)
                            .ok_or(MigrateError::InvalidPartnerShards)?,
                        None => {
                            let repaired = partner_scratch
                                .entry(source_registry_index)
                                .or_insert_with(|| {
                                    let mut repaired = Vec::with_capacity(self.segments);
                                    for source_seg in 0..self.segments {
                                        let values = source_rows
                                            .as_ref()
                                            .and_then(|rows| {
                                                repair_primary_shard_from_source_rows(
                                                    &self.params,
                                                    &self.coded,
                                                    rows,
                                                    source_seg,
                                                    partner_entry,
                                                )
                                                .ok()
                                            })
                                            .unwrap_or_default();
                                        repaired.push(values);
                                    }
                                    repaired
                                });
                            let values = repaired
                                .get(seg)
                                .ok_or(MigrateError::MissingPartnerShards)?;
                            if values.len() != self.params.s {
                                return Err(MigrateError::MissingPartnerShards);
                            }
                            values
                        }
                    }
                };
                let kappa_halved = kappa_halve_primary_shard(old_values, partner_values);
                new_shards.push(kappa_halved);
            }
            migrated
                .coded
                .insert_shard_family_segments(&new_params, new_entry.key, new_shards);
        }

        if self.segments > 0 {
            let source_rows = source_rows
                .as_ref()
                .expect("source rows were collected when segments > 0");
            let target_row_owners = collect_unique_target_row_owners(&new_params, &target_claims)?;
            let migrated_rows = migrate_unique_kappa_halve_row_family(
                &self.params,
                &new_params,
                &self.coded,
                self.segments,
                source_rows,
                &target_row_owners,
            )?;
            for (row_label_bytes, (row_label, owner_key)) in target_row_owners {
                let row_segments = migrated_rows
                    .get(&row_label_bytes)
                    .ok_or(MigrateError::MissingLocalRows)?;
                migrated.coded.insert_row_family_segments(
                    &new_params,
                    RowKey::new(owner_key, row_label),
                    row_segments.clone(),
                );
            }
        }
        migrated.len = self.len;
        migrated.segments = self.segments;
        Ok(migrated)
    }

    fn kappa_double_encoded_state(
        &self,
        new_params: CodedArrayParams,
    ) -> Result<UnauthStore, MigrateError> {
        let source_claims = self.effective_claims_for_params(&self.params);
        let target_claims = split_claim_bundles_for_kappa_double_target_mode(
            &self.params,
            &new_params,
            &source_claims,
        );
        let mut migrated = if self.claims.is_some() {
            UnauthStore::new_with_claims(new_params.clone(), target_claims.clone())
        } else {
            UnauthStore::new(new_params.clone())
        };
        for claim in &source_claims {
            let registry_index = claim.registry_index;
            let old_entry = self.params.shard_registry[registry_index];
            let (left_target_idx, right_target_idx) =
                kappa_double_child_registry_indices(registry_index);
            let left_entry = new_params.shard_registry[left_target_idx];
            let right_entry = new_params.shard_registry[right_target_idx];
            let mut left_segments = Vec::with_capacity(self.segments);
            let mut right_segments = Vec::with_capacity(self.segments);
            for seg in 0..self.segments {
                let old_values = self
                    .coded
                    .shard_segment(&self.params, &old_entry.key, seg)
                    .ok_or(MigrateError::MissingLocalRows)?;
                let (left, right) = kappa_double_primary_shard_pair(old_values);
                left_segments.push(left);
                right_segments.push(right);
            }
            migrated
                .coded
                .insert_shard_family_segments(&new_params, left_entry.key, left_segments);
            migrated.coded.insert_shard_family_segments(
                &new_params,
                right_entry.key,
                right_segments,
            );
        }
        migrated.len = self.len;
        migrated.segments = self.segments;
        Ok(migrated)
    }

    fn segment_double_encoded_state(
        &self,
        new_params: CodedArrayParams,
    ) -> Result<UnauthStore, MigrateError> {
        if self.claims.is_none() {
            return self.segment_double_full_state(new_params);
        }
        let source_claims = self.effective_claims_for_params(&self.params);
        let target_indices: Vec<_> = source_claims
            .iter()
            .map(|claim| claim.registry_index)
            .collect();
        let target_claims = default_claim_bundles_for_indices(&new_params, &target_indices);
        let target_segments = self.segments.div_ceil(2);
        let mut migrated = UnauthStore::new_with_claims(new_params.clone(), target_claims.clone());
        for claim in &source_claims {
            let registry_index = claim.registry_index;
            let old_entry = self.params.shard_registry[registry_index];
            let new_entry = new_params.shard_registry[registry_index];
            if old_entry.key != new_entry.key
                || old_entry.a != new_entry.a
                || old_entry.z != new_entry.z
            {
                return Err(MigrateError::InvalidSegmentMerge);
            }
            let target_family = self.segment_double_primary_family(
                &new_params,
                target_segments,
                registry_index,
                old_entry.key,
            )?;
            migrated
                .coded
                .insert_shard_family_segments(&new_params, new_entry.key, target_family);
        }
        migrated.len = self.len;
        migrated.segments = target_segments;
        self.segment_double_rows_into(&mut migrated, &new_params, target_segments, &target_claims)?;
        Ok(migrated)
    }

    fn segment_double_full_state(
        &self,
        new_params: CodedArrayParams,
    ) -> Result<UnauthStore, MigrateError> {
        let target_segments = self.segments.div_ceil(2);
        let mut migrated = UnauthStore::new(new_params.clone());
        let all_indices: Vec<usize> = (0..new_params.shard_registry.len()).collect();
        let all_claims = default_claim_bundles_for_indices(&new_params, &all_indices);
        reserve_segment_state_for_claims(
            &new_params,
            &mut migrated.coded,
            &all_claims,
            target_segments,
        );
        for (registry_index, entry) in new_params.shard_registry.iter().copied().enumerate() {
            let old_entry = self.params.shard_registry[registry_index];
            let target_family = self.segment_double_primary_family(
                &new_params,
                target_segments,
                registry_index,
                old_entry.key,
            )?;
            migrated
                .coded
                .insert_shard_family_segments(&new_params, entry.key, target_family);
        }
        migrated.len = self.len;
        migrated.segments = target_segments;
        self.segment_double_rows_into(&mut migrated, &new_params, target_segments, &all_claims)?;
        Ok(migrated)
    }

    fn segment_double_rows_into(
        &self,
        migrated: &mut UnauthStore,
        new_params: &CodedArrayParams,
        target_segments: usize,
        target_claims: &[ClaimBundle],
    ) -> Result<(), MigrateError> {
        migrated.coded.clear_rows();
        if target_claims.is_empty() {
            return Ok(());
        }
        let inv_two = merge_selector_half()?;
        for claim in target_claims {
            let registry_index = claim.registry_index;
            let old_entry = self.params.shard_registry[registry_index];
            let new_entry = new_params.shard_registry[registry_index];
            if old_entry.key != new_entry.key {
                return Err(MigrateError::InvalidSegmentMerge);
            }
            let row_label_indices = claim
                .row_label_indices_slice(new_params)
                .ok_or(MigrateError::InvalidClaim)?;
            for &target_row_label_idx in row_label_indices {
                let row_label = new_params.row_labels[target_row_label_idx];
                let row_label_bytes = encode_value(row_label);
                if self
                    .params
                    .row_label_index_for_bytes(row_label_bytes)
                    .is_none()
                {
                    return Err(MigrateError::MissingLocalRows);
                }
                let source_row_key = RowKey::new(old_entry.key, row_label);
                let mut row_segments = Vec::with_capacity(target_segments);
                for target_seg in 0..target_segments {
                    row_segments.push(self.segment_double_row_segment(
                        new_params,
                        target_seg,
                        &source_row_key,
                        inv_two,
                    )?);
                }
                migrated.coded.insert_row_family_segments(
                    new_params,
                    RowKey::new(new_entry.key, row_label),
                    row_segments,
                );
            }
        }
        Ok(())
    }

    fn segment_double_row_segment(
        &self,
        new_params: &CodedArrayParams,
        target_seg: usize,
        source_row_key: &RowKey,
        inv_two: Fp,
    ) -> Result<Vec<Fp>, MigrateError> {
        let left_seg = target_seg * 2;
        let right_seg = left_seg + 1;
        let zero_segment;
        let left = self
            .coded
            .row_segment(&self.params, source_row_key, left_seg)
            .ok_or(MigrateError::MissingLocalRows)?;
        let right = if right_seg < self.segments {
            self.coded
                .row_segment(&self.params, source_row_key, right_seg)
                .ok_or(MigrateError::MissingLocalRows)?
        } else {
            zero_segment = vec![fp_zero(); self.params.need];
            &zero_segment
        };
        if left.len() != self.params.need || right.len() != self.params.need {
            return Err(MigrateError::InvalidSegmentMerge);
        }

        let rho = segment_domain_rho(new_params, target_seg);
        let c_rho = fp_pow(rho, self.params.s as u64);
        let inv_c_rho = c_rho.inverse().ok_or(MigrateError::InvalidSegmentMerge)?;
        let mut out = Vec::with_capacity(new_params.need);
        let (left_values, right_values) = barycentric_eval_pair_many(
            &self.params.row_eval_bary_weights,
            left,
            right,
            &new_params.row_eval_points,
        );
        for ((&z_eval, &left_value), &right_value) in new_params
            .row_eval_points
            .iter()
            .zip(left_values.iter())
            .zip(right_values.iter())
        {
            let q = fp_pow(z_eval * inv_c_rho, self.params.need as u64);
            let e0 = (fp_one() + q) * inv_two;
            let e1 = (fp_one() - q) * inv_two;
            out.push(e0 * left_value + e1 * right_value);
        }
        Ok(out)
    }

    fn segment_double_primary_family(
        &self,
        new_params: &CodedArrayParams,
        target_segments: usize,
        registry_index: usize,
        source_key: ShardKey,
    ) -> Result<Vec<Vec<Fp>>, MigrateError> {
        let inv_two = merge_selector_half()?;
        let mut segments = Vec::with_capacity(target_segments);
        for target_seg in 0..target_segments {
            segments.push(self.segment_double_primary_segment(
                new_params,
                target_seg,
                registry_index,
                source_key,
                inv_two,
            )?);
        }
        Ok(segments)
    }

    fn segment_double_primary_segment(
        &self,
        new_params: &CodedArrayParams,
        target_seg: usize,
        registry_index: usize,
        source_key: ShardKey,
        inv_two: Fp,
    ) -> Result<Vec<Fp>, MigrateError> {
        let left_seg = target_seg * 2;
        let right_seg = left_seg + 1;
        let zero_segment;
        let left = self
            .coded
            .shard_segment(&self.params, &source_key, left_seg)
            .ok_or(MigrateError::MissingPartnerShards)?;
        let right = if right_seg < self.segments {
            self.coded
                .shard_segment(&self.params, &source_key, right_seg)
                .ok_or(MigrateError::MissingPartnerShards)?
        } else {
            zero_segment = vec![fp_zero(); self.params.s];
            &zero_segment
        };
        if left.len() != self.params.s || right.len() != self.params.s {
            return Err(MigrateError::InvalidSegmentMerge);
        }
        let entry = new_params.shard_registry[registry_index];
        let rho = segment_domain_rho(new_params, target_seg);
        let c_rho = fp_pow(rho, self.params.s as u64);
        let inv_c_rho = c_rho.inverse().ok_or(MigrateError::InvalidSegmentMerge)?;
        let q = fp_pow(entry.z * inv_c_rho, self.params.need as u64);
        let e0 = (fp_one() + q) * inv_two;
        let e1 = (fp_one() - q) * inv_two;
        let mut out = Vec::with_capacity(new_params.s);
        for (&left_value, &right_value) in left.iter().zip(right.iter()) {
            out.push(e0 * left_value + e1 * right_value);
        }
        Ok(out)
    }

    fn kappa_double_encoded_state_retaining_left_children(
        &self,
        new_params: CodedArrayParams,
        source_claims: &[ClaimBundle],
        target_claims: &[ClaimBundle],
    ) -> Result<UnauthStore, MigrateError> {
        let mut migrated = if self.claims.is_some() {
            UnauthStore::new_with_claims(new_params.clone(), target_claims.to_vec())
        } else {
            UnauthStore::new(new_params.clone())
        };
        for claim in source_claims {
            let registry_index = claim.registry_index;
            let old_entry = self.params.shard_registry[registry_index];
            let (left_target_idx, _) = kappa_double_child_registry_indices(registry_index);
            let left_entry = new_params.shard_registry[left_target_idx];
            let mut left_segments = Vec::with_capacity(self.segments);
            for seg in 0..self.segments {
                let old_values = self
                    .coded
                    .shard_segment(&self.params, &old_entry.key, seg)
                    .ok_or(MigrateError::MissingLocalRows)?;
                left_segments.push(kappa_double_primary_shard_child(old_values, true));
            }
            migrated
                .coded
                .insert_shard_family_segments(&new_params, left_entry.key, left_segments);
        }
        migrated.len = self.len;
        migrated.segments = self.segments;
        Ok(migrated)
    }
}

fn merge_selector_half() -> Result<Fp, MigrateError> {
    Fp::from(2u64)
        .inverse()
        .ok_or(MigrateError::InvalidSegmentMerge)
}

fn validate_recovered_shards(
    plan: &ClaimSegmentInstallPlan,
    repaired_shards: &[(usize, Vec<Fp>)],
) -> Result<HashMap<usize, Vec<Fp>>, InstallClaimStateError> {
    let mut shard_map = HashMap::new();
    for (registry_index, values) in repaired_shards {
        if !plan.expected_shards.contains(registry_index) {
            return Err(InstallClaimStateError::UnexpectedShard(*registry_index));
        }
        if values.len() != plan.expected_shard_len {
            return Err(InstallClaimStateError::InvalidShardLength {
                registry_index: *registry_index,
                expected: plan.expected_shard_len,
                actual: values.len(),
            });
        }
        shard_map.insert(*registry_index, values.clone());
    }
    for &registry_index in &plan.expected_shards {
        if !shard_map.contains_key(&registry_index) {
            return Err(InstallClaimStateError::MissingShard(registry_index));
        }
    }
    Ok(shard_map)
}

fn validate_recovered_rows(
    plan: &ClaimSegmentInstallPlan,
    repaired_rows: &[(usize, Vec<Fp>)],
) -> Result<HashMap<usize, Vec<Fp>>, InstallClaimStateError> {
    let mut row_map = HashMap::new();
    for (row_label_index, values) in repaired_rows {
        if !plan.expected_rows.contains(row_label_index) {
            return Err(InstallClaimStateError::UnexpectedRow(*row_label_index));
        }
        if values.len() != plan.expected_row_len {
            return Err(InstallClaimStateError::InvalidRowLength {
                row_label_index: *row_label_index,
                expected: plan.expected_row_len,
                actual: values.len(),
            });
        }
        row_map.insert(*row_label_index, values.clone());
    }
    for &row_label_index in &plan.expected_rows {
        if !row_map.contains_key(&row_label_index) {
            return Err(InstallClaimStateError::MissingRow(row_label_index));
        }
    }
    Ok(row_map)
}

fn validate_owned_recovered_shards(
    plan: &ClaimSegmentInstallPlan,
    repaired_shards: Vec<(usize, Vec<Fp>)>,
) -> Result<HashMap<usize, Vec<Fp>>, InstallClaimStateError> {
    let mut shard_map = HashMap::new();
    for (registry_index, values) in repaired_shards {
        if !plan.expected_shards.contains(&registry_index) {
            return Err(InstallClaimStateError::UnexpectedShard(registry_index));
        }
        if values.len() != plan.expected_shard_len {
            return Err(InstallClaimStateError::InvalidShardLength {
                registry_index,
                expected: plan.expected_shard_len,
                actual: values.len(),
            });
        }
        shard_map.insert(registry_index, values);
    }
    for &registry_index in &plan.expected_shards {
        if !shard_map.contains_key(&registry_index) {
            return Err(InstallClaimStateError::MissingShard(registry_index));
        }
    }
    Ok(shard_map)
}

fn validate_owned_recovered_rows(
    plan: &ClaimSegmentInstallPlan,
    repaired_rows: Vec<(usize, Vec<Fp>)>,
) -> Result<HashMap<usize, Vec<Fp>>, InstallClaimStateError> {
    let mut row_map = HashMap::new();
    for (row_label_index, values) in repaired_rows {
        if !plan.expected_rows.contains(&row_label_index) {
            return Err(InstallClaimStateError::UnexpectedRow(row_label_index));
        }
        if values.len() != plan.expected_row_len {
            return Err(InstallClaimStateError::InvalidRowLength {
                row_label_index,
                expected: plan.expected_row_len,
                actual: values.len(),
            });
        }
        row_map.insert(row_label_index, values);
    }
    for &row_label_index in &plan.expected_rows {
        if !row_map.contains_key(&row_label_index) {
            return Err(InstallClaimStateError::MissingRow(row_label_index));
        }
    }
    Ok(row_map)
}

fn default_segment_values(params: &CodedArrayParams) -> Vec<Value256> {
    let zero = encode_value(fp_zero());
    vec![zero; params.n_grid]
}

fn logical_segment_range(
    params: &CodedArrayParams,
    seg: usize,
) -> Result<Range<Index>, ExtendError> {
    let seg = Index::try_from(seg).map_err(|_| ExtendError::SegmentOverflow)?;
    let b = public_values_per_segment(params) as Index;
    let start = seg.checked_mul(b).ok_or(ExtendError::SegmentOverflow)?;
    let end = start.checked_add(b).ok_or(ExtendError::SegmentOverflow)?;
    Ok(start..end)
}

fn logical_segment_from_logical_values(
    params: &CodedArrayParams,
    logical_values: &[Value256],
) -> Result<Vec<Value256>, ExtendError> {
    if logical_values.len() != public_values_per_segment(params) {
        return Err(ExtendError::InvalidFamilyValues);
    }
    let mut segment = default_segment_values(params);
    for (logical_offset, value) in logical_values.iter().copied().enumerate() {
        let grid_index = logical_offset_to_grid_index(params, logical_offset);
        segment[grid_index] = value;
    }
    Ok(segment)
}

fn logical_segment_from_index_values<F>(
    params: &CodedArrayParams,
    seg: usize,
    value_at_index: &mut F,
) -> Vec<Value256>
where
    F: FnMut(Index) -> Value256,
{
    let mut segment = default_segment_values(params);
    let seg_base = seg as u64 * public_values_per_segment(params) as u64;
    for logical_offset in 0..public_values_per_segment(params) {
        let grid_index = logical_offset_to_grid_index(params, logical_offset);
        segment[grid_index] = value_at_index(seg_base + logical_offset as u64);
    }
    segment
}

fn overwrite_segment_state_for_claims_sparse(
    params: &CodedArrayParams,
    coded: &mut CodedStore,
    seg: usize,
    claims: &[ClaimBundle],
    shards: &HashMap<usize, Vec<Fp>>,
    rows: &HashMap<usize, Vec<Fp>>,
) {
    for claim in claims {
        let idx = claim.registry_index;
        let entry = params.shard_registry[idx];
        let shard_values = shards
            .get(&idx)
            .expect("missing selected shard values for claim-bound overwrite");
        coded.overwrite_shard_segment(params, entry.key, seg, shard_values);

        let row_label_indices = claim
            .row_label_indices_slice(params)
            .expect("claim-bound stores require exact row-family assignments");
        for &row_label_idx in row_label_indices {
            let row_values = rows
                .get(&row_label_idx)
                .expect("missing selected row values for claim-bound overwrite");
            let row_key = RowKey::new(entry.key, params.row_labels[row_label_idx]);
            coded.overwrite_row_segment(params, row_key, seg, row_values);
        }
    }
}

fn accumulate_segment_state(
    params: &CodedArrayParams,
    coded: &mut CodedStore,
    seg: usize,
    shard_indices: &[usize],
    shards: &[Vec<Fp>],
    rows: &[Vec<Fp>],
) {
    let claims = default_claim_bundles_for_indices(params, shard_indices);
    for claim in &claims {
        let idx = claim.registry_index;
        let entry = params.shard_registry[idx];
        let shard_values = &shards[idx];
        let segment = coded
            .shard_segment_mut(params, &entry.key, seg)
            .expect("delta application requires pre-existing shard segment");
        accumulate_segment_values(segment, shard_values);

        let row_label_indices = claim
            .row_label_indices_slice(params)
            .expect("claim-bound stores require exact row-family assignments");
        for &row_label_idx in row_label_indices {
            let row_key = RowKey::new(entry.key, params.row_labels[row_label_idx]);
            let row_segment = coded
                .row_segment_mut(params, &row_key, seg)
                .expect("delta application requires pre-existing row segment");
            accumulate_segment_values(row_segment, &rows[row_label_idx]);
        }
    }
}

fn accumulate_segment_state_for_claims_sparse(
    params: &CodedArrayParams,
    coded: &mut CodedStore,
    seg: usize,
    claims: &[ClaimBundle],
    shards: &HashMap<usize, Vec<Fp>>,
    rows: &HashMap<usize, Vec<Fp>>,
) {
    for claim in claims {
        let idx = claim.registry_index;
        let entry = params.shard_registry[idx];
        let shard_values = shards
            .get(&idx)
            .expect("missing selected shard values for claim-bound delta");
        let segment = coded
            .shard_segment_mut(params, &entry.key, seg)
            .expect("delta application requires pre-existing shard segment");
        accumulate_segment_values(segment, shard_values);

        let row_label_indices = claim
            .row_label_indices_slice(params)
            .expect("claim-bound stores require exact row-family assignments");
        for &row_label_idx in row_label_indices {
            let row_values = rows
                .get(&row_label_idx)
                .expect("missing selected row values for claim-bound delta");
            let row_key = RowKey::new(entry.key, params.row_labels[row_label_idx]);
            let row_segment = coded
                .row_segment_mut(params, &row_key, seg)
                .expect("delta application requires pre-existing row segment");
            accumulate_segment_values(row_segment, row_values);
        }
    }
}

fn accumulate_segment_values(dst: &mut [Fp], delta: &[Fp]) {
    assert_eq!(
        dst.len(),
        delta.len(),
        "segment delta application requires matching segment widths",
    );
    for (dst_value, delta_value) in dst.iter_mut().zip(delta.iter()) {
        *dst_value += *delta_value;
    }
}

/// Build coded params for a fixed-segment kappa rescale while preserving shard representatives.
///
/// Prefer `build_kappa_double_params` or `build_kappa_halve_params` for the two
/// supported adjacent directions. This generic target-`s` form is kept as a
/// compatibility helper.
pub fn build_migrated_params(
    params: &CodedArrayParams,
    target_s: usize,
) -> Result<CodedArrayParams, MigrateError> {
    let n_grid = params.n_grid;
    let kappa_halve_target = params.s.checked_mul(2);
    let can_kappa_double = params.s > 1 && target_s == params.s / 2;
    let can_kappa_halve = kappa_halve_target == Some(target_s) && target_s <= n_grid;
    if target_s == 0
        || !target_s.is_power_of_two()
        || target_s > n_grid
        || !(can_kappa_double || can_kappa_halve)
    {
        return Err(MigrateError::InvalidTargetS);
    }
    let shard_registry = build_migrated_shard_registry(params, target_s);
    let party_registry_indices = build_migrated_party_registry_indices(params, target_s);
    let shard_count = shard_registry.len();
    let cfg = CodedArrayConfig {
        segment_len: params.segment_len,
        s: target_s,
        shard_count,
        shard_registry,
        party_registry_indices,
        seed: 0,
    };
    let mut migrated = build_coded_params(&cfg);
    migrated.logical_to_grid_index = params.logical_to_grid_index.clone();
    migrated.grid_index_to_logical_offset = params.grid_index_to_logical_offset.clone();
    migrated.layout_base_n_grid = params.layout_base_n_grid;
    migrated.layout_level = params.layout_level;
    Ok(migrated)
}

/// Build coded params for `kappa -> 2kappa` with fixed segment size.
pub fn build_kappa_double_params(
    params: &CodedArrayParams,
) -> Result<CodedArrayParams, MigrateError> {
    if params.s <= 1 {
        return Err(MigrateError::InvalidTargetS);
    }
    build_migrated_params(params, params.s / 2)
}

/// Build coded params for `kappa -> kappa/2` with fixed segment size.
pub fn build_kappa_halve_params(
    params: &CodedArrayParams,
) -> Result<CodedArrayParams, MigrateError> {
    let target_s = params
        .s
        .checked_mul(2)
        .ok_or(MigrateError::InvalidTargetS)?;
    build_migrated_params(params, target_s)
}

/// Build coded params for segment doubling while preserving the canonical coset layout.
pub fn build_segment_double_params(
    params: &CodedArrayParams,
) -> Result<CodedArrayParams, MigrateError> {
    if params.segment_len != params.n_grid {
        return Err(MigrateError::InvalidSegmentMerge);
    }
    let target_n_grid = params
        .n_grid
        .checked_mul(2)
        .ok_or(MigrateError::InvalidSegmentMerge)?;
    let two_adicity = <Fp as FftField>::TWO_ADICITY as usize;
    if target_n_grid.trailing_zeros() as usize > two_adicity {
        return Err(MigrateError::InvalidSegmentMerge);
    }
    let target_segment_len = target_n_grid;
    let cfg = CodedArrayConfig {
        segment_len: target_segment_len,
        s: params.s,
        shard_count: params.shard_registry.len(),
        shard_registry: params.shard_registry.clone(),
        party_registry_indices: params.party_registry_indices.clone(),
        seed: 0,
    };
    let mut target = build_coded_params(&cfg);
    if target.rows_per_shard > params.rows_per_shard
        || target.row_family_count != params.row_family_count
    {
        return Err(MigrateError::InvalidSegmentMerge);
    }
    let mut target_row_labels = Vec::with_capacity(
        target
            .row_family_count
            .saturating_mul(target.rows_per_shard),
    );
    for family_id in 0..target.row_family_count {
        let old_start = family_id
            .checked_mul(params.rows_per_shard)
            .ok_or(MigrateError::InvalidSegmentMerge)?;
        let old_end = old_start
            .checked_add(target.rows_per_shard)
            .ok_or(MigrateError::InvalidSegmentMerge)?;
        if old_end > params.row_labels.len() {
            return Err(MigrateError::InvalidSegmentMerge);
        }
        target_row_labels.extend_from_slice(&params.row_labels[old_start..old_end]);
    }
    for &row_label in &target_row_labels {
        if row_label == fp_zero() || fp_pow(row_label, target.n_grid as u64) == fp_one() {
            return Err(MigrateError::InvalidSegmentMerge);
        }
    }
    target.row_labels = target_row_labels;
    target.refresh_row_metadata_caches();
    target
        .apply_canonical_logical_layout(params.layout_base_n_grid)
        .map_err(|_| MigrateError::InvalidSegmentMerge)?;
    if target.n_grid != target_n_grid
        || target.s != params.s
        || target.need
            != params
                .need
                .checked_mul(2)
                .ok_or(MigrateError::InvalidSegmentMerge)?
        || target.segment_len != target.n_grid
        || target.shard_registry.len() < target.need
        || target.h != params.h
        || target.layout_level
            != params
                .layout_level
                .checked_add(1)
                .ok_or(MigrateError::InvalidSegmentMerge)?
    {
        return Err(MigrateError::InvalidSegmentMerge);
    }
    for entry in &target.shard_registry {
        if fp_pow(entry.a, target.n_grid as u64) == fp_one() {
            return Err(MigrateError::InvalidSegmentMerge);
        }
    }
    Ok(target)
}

fn kappa_halve_partner_entry(params: &CodedArrayParams, entry: ShardRegEntry) -> ShardRegEntry {
    let twist = fp_pow(params.omega, (params.n_grid / (2 * params.s)) as u64);
    let a = entry.a * twist;
    let z = -entry.z;
    ShardRegEntry {
        a,
        z,
        key: ShardKey {
            a_bytes: encode_value(a),
            z_bytes: encode_value(z),
        },
    }
}

fn kappa_halve_primary_shard(old_values: &[Fp], partner_values: &[Fp]) -> Vec<Fp> {
    let mut out = Vec::with_capacity(old_values.len() * 2);
    for (left, right) in old_values.iter().zip(partner_values.iter()) {
        out.push(*left);
        out.push(*right);
    }
    out
}

fn kappa_double_primary_shard_pair(old_values: &[Fp]) -> (Vec<Fp>, Vec<Fp>) {
    let mut left = Vec::with_capacity(old_values.len() / 2);
    let mut right = Vec::with_capacity(old_values.len() / 2);
    for (idx, value) in old_values.iter().enumerate() {
        if idx % 2 == 0 {
            left.push(*value);
        } else {
            right.push(*value);
        }
    }
    (left, right)
}

fn kappa_double_primary_shard_child(old_values: &[Fp], left_child: bool) -> Vec<Fp> {
    let mut out = Vec::with_capacity(old_values.len() / 2);
    let parity = if left_child { 0 } else { 1 };
    for (idx, value) in old_values.iter().enumerate() {
        if idx % 2 == parity {
            out.push(*value);
        }
    }
    out
}

enum SourceRowFamily<'a> {
    Local(RowKey),
    Helper(&'a HelperRowMigration),
}

struct SourceRowsForKappaHalve<'a> {
    rows: BTreeMap<Value256, SourceRowFamily<'a>>,
}

fn source_row_segment<'a>(
    params: &CodedArrayParams,
    coded: &'a CodedStore,
    source: &SourceRowFamily<'a>,
    seg: usize,
) -> Result<&'a [Fp], MigrateError> {
    match source {
        SourceRowFamily::Local(row_key) => coded
            .row_segment(params, row_key, seg)
            .ok_or(MigrateError::MissingLocalRows),
        SourceRowFamily::Helper(helper) => {
            let row_values = helper
                .segments
                .get(seg)
                .ok_or(MigrateError::InvalidPartnerShards)?;
            if row_values.len() != params.row_eval_points.len() {
                return Err(MigrateError::InvalidPartnerShards);
            }
            Ok(row_values)
        }
    }
}

fn validate_local_row_family(
    params: &CodedArrayParams,
    coded: &CodedStore,
    segments: usize,
    row_key: &RowKey,
) -> Result<(), MigrateError> {
    if coded.row_family_segment_count(row_key) != Some(segments) {
        return Err(MigrateError::MissingLocalRows);
    }
    for seg in 0..segments {
        let row_values = coded
            .row_segment(params, row_key, seg)
            .ok_or(MigrateError::MissingLocalRows)?;
        if row_values.len() != params.row_eval_points.len() {
            return Err(MigrateError::MissingLocalRows);
        }
    }
    Ok(())
}

fn source_row_families_match(
    params: &CodedArrayParams,
    coded: &CodedStore,
    segments: usize,
    left: &SourceRowFamily<'_>,
    right: &SourceRowFamily<'_>,
) -> Result<bool, MigrateError> {
    for seg in 0..segments {
        if source_row_segment(params, coded, left, seg)?
            != source_row_segment(params, coded, right, seg)?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn collect_source_rows_for_kappa_halve<'a>(
    params: &CodedArrayParams,
    coded: &CodedStore,
    segments: usize,
    helper_rows: &'a [HelperRowMigration],
) -> Result<SourceRowsForKappaHalve<'a>, MigrateError> {
    let mut rows = BTreeMap::<Value256, SourceRowFamily<'a>>::new();
    for row_key in coded.row_keys().copied() {
        validate_local_row_family(params, coded, segments, &row_key)?;
        let source = SourceRowFamily::Local(row_key);
        match rows.entry(row_key.row_label_bytes) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(source);
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                if !source_row_families_match(params, coded, segments, entry.get(), &source)? {
                    return Err(MigrateError::MissingLocalRows);
                }
            }
        }
    }
    for helper in helper_rows {
        if helper.segments.len() != segments
            || helper
                .segments
                .iter()
                .any(|row_values| row_values.len() != params.row_eval_points.len())
        {
            return Err(MigrateError::InvalidPartnerShards);
        }
        let source = SourceRowFamily::Helper(helper);
        match rows.entry(encode_value(helper.row_label)) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(source);
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                if !source_row_families_match(params, coded, segments, entry.get(), &source)? {
                    return Err(MigrateError::InvalidPartnerShards);
                }
            }
        }
    }
    Ok(SourceRowsForKappaHalve { rows })
}

fn repair_primary_shard_from_source_rows(
    params: &CodedArrayParams,
    coded: &CodedStore,
    source_rows: &SourceRowsForKappaHalve<'_>,
    seg: usize,
    entry: ShardRegEntry,
) -> Result<Vec<Fp>, MigrateError> {
    let mut rows = Vec::with_capacity(params.s);
    for (&row_label_bytes, row_family) in &source_rows.rows {
        let row_values = source_row_segment(params, coded, row_family, seg)?;
        rows.push(RowShare {
            row_label: decode_value(row_label_bytes),
            value: barycentric_eval(&params.row_eval_bary_weights, row_values, entry.z),
        });
        if rows.len() >= params.s {
            break;
        }
    }
    if rows.len() < params.s {
        return Err(MigrateError::MissingLocalRows);
    }
    decode_primary_shard_from_rows(params, entry, &rows)
        .map_err(|_| MigrateError::MissingPartnerShards)
}

enum TargetShardFamily<'a> {
    Local(ShardKey),
    Helper(&'a PartnerShardMigration),
}

struct TargetShardsForRowHydration<'a> {
    shards: BTreeMap<usize, TargetShardFamily<'a>>,
}

fn target_shard_segment<'a>(
    params: &CodedArrayParams,
    coded: &'a CodedStore,
    source: &TargetShardFamily<'a>,
    seg: usize,
) -> Result<&'a [Fp], MigrateError> {
    match source {
        TargetShardFamily::Local(key) => coded
            .shard_segment(params, key, seg)
            .ok_or(MigrateError::MissingLocalRows),
        TargetShardFamily::Helper(helper) => {
            let segment_values = helper
                .segments
                .get(seg)
                .ok_or(MigrateError::InvalidPartnerShards)?;
            if segment_values.len() != params.s {
                return Err(MigrateError::InvalidPartnerShards);
            }
            Ok(segment_values)
        }
    }
}

fn validate_local_shard_family(
    params: &CodedArrayParams,
    coded: &CodedStore,
    segments: usize,
    key: &ShardKey,
) -> Result<(), MigrateError> {
    if coded.shard_family_segment_count(key) != Some(segments) {
        return Err(MigrateError::MissingLocalRows);
    }
    for seg in 0..segments {
        let shard_values = coded
            .shard_segment(params, key, seg)
            .ok_or(MigrateError::MissingLocalRows)?;
        if shard_values.len() != params.s {
            return Err(MigrateError::MissingLocalRows);
        }
    }
    Ok(())
}

fn target_shard_families_match(
    params: &CodedArrayParams,
    coded: &CodedStore,
    segments: usize,
    left: &TargetShardFamily<'_>,
    right: &TargetShardFamily<'_>,
) -> Result<bool, MigrateError> {
    for seg in 0..segments {
        if target_shard_segment(params, coded, left, seg)?
            != target_shard_segment(params, coded, right, seg)?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn collect_target_shards_for_row_hydration<'a>(
    params: &CodedArrayParams,
    coded: &CodedStore,
    segments: usize,
    helper_shards: &'a [PartnerShardMigration],
) -> Result<TargetShardsForRowHydration<'a>, MigrateError> {
    let mut shards = BTreeMap::<usize, TargetShardFamily<'a>>::new();
    for (registry_index, entry) in params.shard_registry.iter().copied().enumerate() {
        if coded.contains_shard(&entry.key) {
            validate_local_shard_family(params, coded, segments, &entry.key)?;
            shards.insert(registry_index, TargetShardFamily::Local(entry.key));
        }
    }
    for helper in helper_shards {
        if helper.registry_index >= params.shard_registry.len() || helper.segments.len() != segments
        {
            return Err(MigrateError::InvalidPartnerShards);
        }
        if helper
            .segments
            .iter()
            .any(|segment_values| segment_values.len() != params.s)
        {
            return Err(MigrateError::InvalidPartnerShards);
        }
        let source = TargetShardFamily::Helper(helper);
        match shards.entry(helper.registry_index) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(source);
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                if !target_shard_families_match(params, coded, segments, entry.get(), &source)? {
                    return Err(MigrateError::InvalidPartnerShards);
                }
            }
        }
    }
    Ok(TargetShardsForRowHydration { shards })
}

#[allow(dead_code)]
fn validate_segment_double_target_shards(
    params: &CodedArrayParams,
    helper_shards: &[PartnerShardMigration],
    segments: usize,
) -> Result<HashMap<usize, Vec<Vec<Fp>>>, MigrateError> {
    let mut by_index = HashMap::new();
    for helper in helper_shards {
        if helper.registry_index >= params.shard_registry.len() || helper.segments.len() != segments
        {
            return Err(MigrateError::InvalidPartnerShards);
        }
        if helper
            .segments
            .iter()
            .any(|segment_values| segment_values.len() != params.s)
        {
            return Err(MigrateError::InvalidPartnerShards);
        }
        match by_index.entry(helper.registry_index) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(helper.segments.clone());
            }
            std::collections::hash_map::Entry::Occupied(entry) => {
                if entry.get() != &helper.segments {
                    return Err(MigrateError::InvalidPartnerShards);
                }
            }
        }
    }
    Ok(by_index)
}

fn collect_unique_target_row_owners(
    params: &CodedArrayParams,
    claims: &[ClaimBundle],
) -> Result<BTreeMap<Value256, (Fp, ShardKey)>, MigrateError> {
    let mut owners_by_label = BTreeMap::<Value256, (Fp, ShardKey)>::new();
    for claim in claims {
        let Some(&entry) = params.shard_registry.get(claim.registry_index) else {
            return Err(MigrateError::InvalidPartnerShards);
        };
        let Some(row_label_indices) = claim.row_label_indices_slice(params) else {
            return Err(MigrateError::InvalidPartnerShards);
        };
        for &row_label_idx in row_label_indices {
            let Some(&row_label) = params.row_labels.get(row_label_idx) else {
                return Err(MigrateError::InvalidPartnerShards);
            };
            let row_label_bytes = encode_value(row_label);
            match owners_by_label.entry(row_label_bytes) {
                std::collections::btree_map::Entry::Vacant(slot) => {
                    slot.insert((row_label, entry.key));
                }
                std::collections::btree_map::Entry::Occupied(existing) => {
                    if existing.get() != &(row_label, entry.key) {
                        return Err(MigrateError::InvalidPartnerShards);
                    }
                }
            }
        }
    }
    Ok(owners_by_label)
}

fn collect_target_row_owners_by_index(
    params: &CodedArrayParams,
    claims: &[ClaimBundle],
) -> Result<Vec<(usize, Fp, ShardKey)>, MigrateError> {
    let mut owners_by_index = BTreeMap::<usize, (Fp, ShardKey)>::new();
    for claim in claims {
        let Some(&entry) = params.shard_registry.get(claim.registry_index) else {
            return Err(MigrateError::InvalidPartnerShards);
        };
        let Some(row_label_indices) = claim.row_label_indices_slice(params) else {
            return Err(MigrateError::InvalidPartnerShards);
        };
        for &row_label_idx in row_label_indices {
            let Some(&row_label) = params.row_labels.get(row_label_idx) else {
                return Err(MigrateError::InvalidPartnerShards);
            };
            match owners_by_index.entry(row_label_idx) {
                std::collections::btree_map::Entry::Vacant(slot) => {
                    slot.insert((row_label, entry.key));
                }
                std::collections::btree_map::Entry::Occupied(existing) => {
                    if existing.get() != &(row_label, entry.key) {
                        return Err(MigrateError::InvalidPartnerShards);
                    }
                }
            }
        }
    }
    Ok(owners_by_index
        .into_iter()
        .map(|(row_label_idx, (row_label, key))| (row_label_idx, row_label, key))
        .collect())
}

fn hydrate_unique_row_family_from_target_columns(
    params: &CodedArrayParams,
    coded: &CodedStore,
    segments: usize,
    source_shards: &TargetShardsForRowHydration<'_>,
    target_row_owners: &BTreeMap<Value256, (Fp, ShardKey)>,
) -> Result<BTreeMap<Value256, Vec<Vec<Fp>>>, MigrateError> {
    if source_shards.shards.is_empty() && target_row_owners.is_empty() {
        return Ok(BTreeMap::new());
    }

    let mut hydrated = BTreeMap::<Value256, Vec<Vec<Fp>>>::new();
    for (&row_label_bytes, (row_label, _)) in target_row_owners {
        let mut row_segments = Vec::with_capacity(segments);
        for seg in 0..segments {
            let row_values =
                decode_row_from_target_shards(params, coded, source_shards, seg, *row_label)
                    .map_err(|_| MigrateError::MissingLocalRows)?;
            row_segments.push(row_values);
        }
        hydrated.insert(row_label_bytes, row_segments);
    }
    Ok(hydrated)
}

fn migrate_unique_kappa_halve_row_family(
    old_params: &CodedArrayParams,
    new_params: &CodedArrayParams,
    coded: &CodedStore,
    segments: usize,
    source_rows: &SourceRowsForKappaHalve<'_>,
    target_row_owners: &BTreeMap<Value256, (Fp, ShardKey)>,
) -> Result<BTreeMap<Value256, Vec<Vec<Fp>>>, MigrateError> {
    let source_labels: Vec<Fp> = source_rows.rows.keys().copied().map(decode_value).collect();
    if source_labels.is_empty() && target_row_owners.is_empty() {
        return Ok(BTreeMap::new());
    }

    let source_families: Vec<&SourceRowFamily<'_>> = source_rows.rows.values().collect();

    let source_monomials = source_monomial_matrix(&source_labels, old_params.s);
    let mut eval_coeffs = BTreeMap::<Value256, Vec<Fp>>::new();
    for (&row_label_bytes, (row_label, _)) in target_row_owners {
        let target_moments = kappa_halve_target_moments(*row_label, old_params.s);
        let coeffs = solve_linear_system(&source_monomials, &target_moments)
            .ok_or(MigrateError::MissingLocalRows)?;
        eval_coeffs.insert(row_label_bytes, coeffs);
    }

    let inv_two = Fp::from(2u64).inverse().expect("two is invertible");
    let mut migrated = BTreeMap::<Value256, Vec<Vec<Fp>>>::new();
    for (&row_label_bytes, (row_label, _)) in target_row_owners {
        let coeffs = eval_coeffs
            .get(&row_label_bytes)
            .ok_or(MigrateError::MissingLocalRows)?;
        let row_scale = fp_pow(*row_label, old_params.s as u64);
        let mut row_segments = Vec::with_capacity(segments);
        let mut source_segments = Vec::with_capacity(source_families.len());
        for seg in 0..segments {
            source_segments.clear();
            for source_family in &source_families {
                source_segments.push(source_row_segment(old_params, coded, source_family, seg)?);
            }
            let mut values = Vec::with_capacity(new_params.row_eval_points.len());
            for target_idx in 0..new_params.row_eval_points.len() {
                let (pos_idx, neg_idx) = old_params
                    .kappa_halve_row_eval_preimage_indices(target_idx)
                    .ok_or(MigrateError::MissingLocalRows)?;
                let mut pos = fp_zero();
                let mut neg = fp_zero();
                for (coeff, row_values) in coeffs.iter().zip(source_segments.iter()) {
                    pos += *coeff * row_values[pos_idx];
                    neg += *coeff * row_values[neg_idx];
                }
                let z = old_params.row_eval_points[pos_idx];
                let avg = (pos + neg) * inv_two;
                let diff = (pos - neg) * inv_two;
                values.push(
                    avg + row_scale * diff * z.inverse().expect("off-grid point is non-zero"),
                );
            }
            row_segments.push(values);
        }
        migrated.insert(row_label_bytes, row_segments);
    }
    Ok(migrated)
}

fn source_monomial_matrix(source_labels: &[Fp], degree: usize) -> Vec<Vec<Fp>> {
    let mut rows = Vec::with_capacity(degree);
    for power in 0..degree {
        let mut row = Vec::with_capacity(source_labels.len());
        for &label in source_labels {
            row.push(fp_pow(label, power as u64));
        }
        rows.push(row);
    }
    rows
}

fn kappa_halve_target_moments(row_label: Fp, old_s: usize) -> Vec<Fp> {
    let mut moments = Vec::with_capacity(old_s);
    let mut cur = fp_one();
    for _ in 0..old_s {
        moments.push(cur);
        cur *= row_label;
    }
    moments
}

fn solve_linear_system(matrix: &[Vec<Fp>], rhs: &[Fp]) -> Option<Vec<Fp>> {
    if matrix.len() != rhs.len() {
        return None;
    }
    let vars = matrix.first().map_or(0, Vec::len);
    if matrix.iter().any(|row| row.len() != vars) {
        return None;
    }
    if vars == 0 {
        return rhs.iter().all(|value| *value == fp_zero()).then(Vec::new);
    }

    let mut augmented = Vec::with_capacity(matrix.len());
    for (row, &value) in matrix.iter().zip(rhs.iter()) {
        let mut augmented_row = row.clone();
        augmented_row.push(value);
        augmented.push(augmented_row);
    }

    let mut pivot_rows = Vec::<(usize, usize)>::new();
    let mut pivot_row = 0usize;
    for col in 0..vars {
        let pivot = (pivot_row..augmented.len()).find(|&row| augmented[row][col] != fp_zero());
        let Some(pivot_idx) = pivot else {
            continue;
        };
        augmented.swap(pivot_row, pivot_idx);
        let inv = augmented[pivot_row][col].inverse()?;
        for value in &mut augmented[pivot_row][col..=vars] {
            *value *= inv;
        }
        let pivot_values = augmented[pivot_row].clone();
        for row in 0..augmented.len() {
            if row == pivot_row || augmented[row][col] == fp_zero() {
                continue;
            }
            let factor = augmented[row][col];
            for idx in col..=vars {
                augmented[row][idx] -= factor * pivot_values[idx];
            }
        }
        pivot_rows.push((pivot_row, col));
        pivot_row += 1;
        if pivot_row == augmented.len() {
            break;
        }
    }

    for row in &augmented {
        if row[..vars].iter().all(|value| *value == fp_zero()) && row[vars] != fp_zero() {
            return None;
        }
    }

    let mut solution = vec![fp_zero(); vars];
    for (row, col) in pivot_rows {
        solution[col] = augmented[row][vars];
    }
    Some(solution)
}

fn materialize_segment_state(
    params: &CodedArrayParams,
    t_seg: &[Value256],
    mut timings: Option<&mut SetupTimings>,
) -> (Vec<Vec<Fp>>, Vec<Vec<Fp>>) {
    let coeffs = crate::coded_array::params::compute_segment_coeffs_from_values(
        params,
        t_seg,
        timings.as_deref_mut(),
    );
    materialize_segment_state_from_coeffs(params, &coeffs, timings)
}

fn materialize_segment_state_from_coeffs(
    params: &CodedArrayParams,
    coeffs: &[Fp],
    mut timings: Option<&mut SetupTimings>,
) -> (Vec<Vec<Fp>>, Vec<Vec<Fp>>) {
    let shards = crate::coded_array::params::compute_initial_shards_from_coeffs(
        params,
        coeffs,
        timings.as_deref_mut(),
    );
    let rows = crate::coded_array::params::compute_initial_rows_from_coeffs(
        params,
        coeffs,
        timings.as_deref_mut(),
    );
    if let Some(timing) = timings {
        let row_label_count = rows.len();
        if row_label_count > 0 {
            if crate::coded_array::params::uses_fft_domain_for_full_row_materialization(params) {
                timing.record_fft_domain_row_materialization(
                    params.row_family_count,
                    row_label_count,
                    row_label_count,
                    crate::coded_array::params::full_row_fft_domain_size(params),
                );
            } else {
                timing.record_sparse_row_materialization(
                    params.row_family_count,
                    row_label_count,
                    row_label_count,
                );
            }
        }
    }
    (shards, rows)
}

fn materialize_segment_state_for_segment(
    params: &CodedArrayParams,
    seg: usize,
    t_seg: &[Value256],
    mut timings: Option<&mut SetupTimings>,
) -> (Vec<Vec<Fp>>, Vec<Vec<Fp>>) {
    let rho = segment_domain_rho(params, seg);
    if rho == fp_one() {
        return materialize_segment_state(params, t_seg, timings);
    }
    let coeffs = crate::coded_array::params::compute_segment_coeffs_from_values_on_coset(
        params,
        t_seg,
        rho,
        timings.as_deref_mut(),
    );
    materialize_segment_state_from_coeffs(params, &coeffs, timings)
}

fn materialize_claim_segment_state_for_segment(
    params: &CodedArrayParams,
    seg: usize,
    t_seg: &[Value256],
    claims: &[ClaimBundle],
    timings: Option<&mut SetupTimings>,
) -> (HashMap<usize, Vec<Fp>>, HashMap<usize, Vec<Fp>>) {
    materialize_claim_segment_state_on_rho(
        params,
        t_seg,
        claims,
        segment_domain_rho(params, seg),
        timings,
    )
}

fn materialize_claim_segment_state_on_rho(
    params: &CodedArrayParams,
    t_seg: &[Value256],
    claims: &[ClaimBundle],
    rho: Fp,
    mut timings: Option<&mut SetupTimings>,
) -> (HashMap<usize, Vec<Fp>>, HashMap<usize, Vec<Fp>>) {
    let coeffs = crate::coded_array::params::compute_segment_coeffs_from_values_on_coset(
        params,
        t_seg,
        rho,
        timings.as_deref_mut(),
    );

    materialize_claim_segment_state_from_coeffs(params, &coeffs, claims, timings)
}

fn materialize_claim_segment_state_from_coeffs(
    params: &CodedArrayParams,
    coeffs: &[Fp],
    claims: &[ClaimBundle],
    mut timings: Option<&mut SetupTimings>,
) -> (HashMap<usize, Vec<Fp>>, HashMap<usize, Vec<Fp>>) {
    let mut registry_indices = Vec::with_capacity(claims.len());
    let total_row_labels = claims
        .iter()
        .map(|claim| {
            claim
                .row_label_indices_slice(params)
                .expect("claim-bound stores require exact row-family assignments")
                .len()
        })
        .sum();
    let mut row_label_indices = Vec::with_capacity(total_row_labels);
    for claim in claims {
        registry_indices.push(claim.registry_index);
        row_label_indices.extend_from_slice(
            claim
                .row_label_indices_slice(params)
                .expect("claim-bound stores require exact row-family assignments"),
        );
    }
    let use_fft_domain = crate::coded_array::params::should_materialize_rows_via_fft_domain(
        params,
        &row_label_indices,
    );

    let shards = crate::coded_array::params::compute_selected_shards_from_coeffs(
        params,
        coeffs,
        &registry_indices,
        None,
    )
    .into_iter()
    .collect();
    let rows: HashMap<usize, Vec<Fp>> =
        crate::coded_array::params::compute_selected_rows_from_coeffs(
            params,
            coeffs,
            &row_label_indices,
            timings.as_deref_mut(),
        )
        .into_iter()
        .collect();
    if let Some(timing) = timings {
        let row_label_count = rows.len();
        if row_label_count > 0 {
            // Counting distinct row families is only for timing metadata; keep
            // it off the untimed append/materialization path.
            let row_family_count = count_distinct_row_families(claims);
            if use_fft_domain {
                timing.record_fft_domain_row_materialization(
                    row_family_count,
                    row_label_count,
                    row_label_indices.len(),
                    crate::coded_array::params::selected_row_fft_domain_size(
                        params,
                        &row_label_indices,
                    ),
                );
            } else {
                timing.record_sparse_row_materialization(
                    row_family_count,
                    row_label_count,
                    row_label_indices.len(),
                );
            }
        }
    }
    (shards, rows)
}

fn count_distinct_row_families(claims: &[ClaimBundle]) -> usize {
    let mut family_ids = HashSet::new();
    for claim in claims {
        family_ids.insert(claim.row_family_id);
    }
    family_ids.len()
}

fn reserve_segment_state_for_claims(
    params: &CodedArrayParams,
    coded: &mut CodedStore,
    claims: &[ClaimBundle],
    additional_segments: usize,
) {
    if additional_segments == 0 {
        return;
    }
    for claim in claims {
        let idx = claim.registry_index;
        let entry = params.shard_registry[idx];
        coded.reserve_shard_segments(params, entry.key, additional_segments);

        let row_label_indices = claim
            .row_label_indices_slice(params)
            .expect("claim-bound stores require exact row-family assignments");
        for &row_label_idx in row_label_indices {
            let row_key = RowKey::new(entry.key, params.row_labels[row_label_idx]);
            coded.reserve_row_segments(params, row_key, additional_segments);
        }
    }
}

fn normalize_claim_bundles(
    params: &CodedArrayParams,
    mut claims: Vec<ClaimBundle>,
) -> Vec<ClaimBundle> {
    let mut by_registry = BTreeMap::<usize, RowFamilyId>::new();
    for claim in claims.drain(..) {
        let expected_family_id = params
            .row_family_id_for_registry_index(claim.registry_index)
            .expect("claim-bound stores require exact row-family assignments");
        assert_eq!(
            claim.row_family_id, expected_family_id,
            "claim-bound stores require registry-aligned exact row families"
        );
        by_registry
            .entry(claim.registry_index)
            .and_modify(|existing| {
                assert_eq!(
                    *existing, claim.row_family_id,
                    "duplicate claims for one registry index must agree on row family"
                );
            })
            .or_insert(claim.row_family_id);
    }
    let normalized: Vec<_> = by_registry
        .into_iter()
        .map(|(registry_index, row_family_id)| ClaimBundle {
            registry_index,
            row_family_id,
        })
        .collect();
    assert!(
        claims_have_globally_unique_row_labels(params, &normalized),
        "claim-bound stores require globally unique row label indices across claims",
    );
    normalized
}

fn default_claim_bundles_for_indices(
    params: &CodedArrayParams,
    shard_indices: &[usize],
) -> Vec<ClaimBundle> {
    let mut claims = Vec::with_capacity(shard_indices.len());
    for &idx in shard_indices {
        if let Some(claim) = ClaimBundle::exact_for_registry_index(params, idx) {
            claims.push(claim);
        }
    }
    if claims.is_empty() {
        Vec::new()
    } else {
        normalize_claim_bundles(params, claims)
    }
}

fn derive_claim_bundles_from_coded_state(
    params: &CodedArrayParams,
    coded: &CodedStore,
) -> Option<Vec<ClaimBundle>> {
    if coded.is_empty() {
        return None;
    }

    let row_label_index_by_bytes: HashMap<_, _> = params
        .row_labels
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, row_label)| (encode_value(row_label), idx))
        .collect();

    let mut by_registry = BTreeMap::<usize, Vec<usize>>::new();
    for (registry_index, entry) in params.shard_registry.iter().copied().enumerate() {
        if coded.contains_shard(&entry.key) {
            by_registry.entry(registry_index).or_default();
        }
    }
    for row_key in coded.row_keys() {
        let registry_index = params.shard_registry_by_key.get(&row_key.owner).copied()?;
        let row_label_index = row_label_index_by_bytes
            .get(&row_key.row_label_bytes)
            .copied()?;
        by_registry
            .entry(registry_index)
            .or_default()
            .push(row_label_index);
    }

    if by_registry.is_empty() {
        return None;
    }

    let mut claims = Vec::with_capacity(by_registry.len());
    for (registry_index, row_label_indices) in by_registry {
        let explicit_rows = if row_label_indices.is_empty() {
            params.row_label_indices_for_registry_index(registry_index)
        } else {
            let mut rows = row_label_indices;
            rows.sort_unstable();
            rows.dedup();
            rows
        };
        let exact_family_id = params.row_family_id_for_row_label_indices(&explicit_rows)?;
        claims.push(ClaimBundle {
            registry_index,
            row_family_id: exact_family_id,
        });
    }
    Some(normalize_claim_bundles(params, claims))
}

fn claims_have_globally_unique_row_labels(
    params: &CodedArrayParams,
    claims: &[ClaimBundle],
) -> bool {
    let mut seen = HashSet::new();
    for claim in claims {
        let Some(row_label_indices) = claim.row_label_indices_slice(params) else {
            return false;
        };
        for &row_label_index in row_label_indices {
            if !seen.insert(row_label_index) {
                return false;
            }
        }
    }
    true
}

pub(crate) fn split_claim_bundles_for_kappa_double_target_mode(
    old_params: &CodedArrayParams,
    new_params: &CodedArrayParams,
    claims: &[ClaimBundle],
) -> Vec<ClaimBundle> {
    let mut projected = Vec::with_capacity(claims.len() * 2);
    for claim in claims {
        let (left, right) = kappa_double_child_registry_indices(claim.registry_index);
        let _ = old_params;
        projected.push(
            ClaimBundle::exact_for_registry_index(new_params, left)
                .expect("left child claim must exist under kappa-double"),
        );
        projected.push(
            ClaimBundle::exact_for_registry_index(new_params, right)
                .expect("right child claim must exist under kappa-double"),
        );
    }
    normalize_claim_bundles(new_params, projected)
}

pub(crate) fn merge_claim_bundles_for_kappa_halve(
    old_params: &CodedArrayParams,
    new_params: &CodedArrayParams,
    claims: &[ClaimBundle],
) -> Vec<ClaimBundle> {
    let mut by_parent: HashMap<usize, Vec<&ClaimBundle>> = HashMap::new();
    for claim in claims {
        by_parent
            .entry(kappa_halve_parent_registry_index(claim.registry_index))
            .or_default()
            .push(claim);
    }
    let mut merged = Vec::with_capacity(by_parent.len());
    for (registry_index, family) in by_parent {
        let _ = family;
        let _ = old_params;
        merged.push(
            ClaimBundle::exact_for_registry_index(new_params, registry_index)
                .expect("kappa-halve parent claim must exist"),
        );
    }
    normalize_claim_bundles(new_params, merged)
}

fn append_segment_state_for_claims(
    params: &CodedArrayParams,
    coded: &mut CodedStore,
    seg: usize,
    claims: &[ClaimBundle],
    shards: &[Vec<Fp>],
    rows: &[Vec<Fp>],
) {
    for claim in claims {
        let idx = claim.registry_index;
        let entry = params.shard_registry[idx];
        coded.append_shard_segment(params, entry.key, seg, &shards[idx]);

        let row_label_indices = claim
            .row_label_indices_slice(params)
            .expect("claim-bound stores require exact row-family assignments");
        for &row_label_idx in row_label_indices {
            let row_key = RowKey::new(entry.key, params.row_labels[row_label_idx]);
            coded.append_row_segment(params, row_key, seg, &rows[row_label_idx]);
        }
    }
}

fn append_segment_state_for_claims_sparse(
    params: &CodedArrayParams,
    coded: &mut CodedStore,
    seg: usize,
    claims: &[ClaimBundle],
    shards: &HashMap<usize, Vec<Fp>>,
    rows: &HashMap<usize, Vec<Fp>>,
) {
    for claim in claims {
        let idx = claim.registry_index;
        let entry = params.shard_registry[idx];
        let shard_values = shards
            .get(&idx)
            .expect("missing selected shard values for claim-bound append");
        coded.append_shard_segment(params, entry.key, seg, shard_values);

        let row_label_indices = claim
            .row_label_indices_slice(params)
            .expect("claim-bound stores require exact row-family assignments");
        for &row_label_idx in row_label_indices {
            let row_values = rows
                .get(&row_label_idx)
                .expect("missing selected row values for claim-bound append");
            let row_key = RowKey::new(entry.key, params.row_labels[row_label_idx]);
            coded.append_row_segment(params, row_key, seg, row_values);
        }
    }
}

fn should_apply_coset_deltas_via_coefficients(mode: DeltaMode, delta_count: usize) -> bool {
    matches!(mode, DeltaMode::InterpolateFFT) && delta_count >= COSET_INTERPOLATE_THRESHOLD
}

fn segment_delta_coeffs_from_grid_deltas_on_coset(
    params: &CodedArrayParams,
    rho: Fp,
    deltas: &[(usize, Fp)],
) -> Vec<Fp> {
    let mut coeffs = vec![fp_zero(); params.n_grid];
    for &(grid_index, delta) in deltas {
        debug_assert!(grid_index < params.n_grid);
        if grid_index < params.n_grid {
            coeffs[grid_index] += delta;
        }
    }

    crate::coded_array::params::inverse_fft_in_place(&mut coeffs, params.omega);
    if rho == fp_one() {
        return coeffs;
    }

    // The IFFT gives Q with Q(x_j) = delta_j. On the segment coset we need
    // D(rho * x_j) = delta_j, so D(X) = Q(X / rho).
    let inv_rho = rho
        .inverse()
        .expect("segment domain scale must be non-zero");
    let mut scale = fp_one();
    for coeff in &mut coeffs {
        *coeff *= scale;
        scale *= inv_rho;
    }
    coeffs
}

fn coeffs_for_segment_rho(base_coeffs: &[Fp], rho: Fp) -> Vec<Fp> {
    let mut coeffs = base_coeffs.to_vec();
    if rho == fp_one() {
        return coeffs;
    }
    let inv_rho = rho
        .inverse()
        .expect("segment domain scale must be non-zero");
    let mut scale = fp_one();
    for coeff in &mut coeffs {
        *coeff *= scale;
        scale *= inv_rho;
    }
    coeffs
}

fn apply_block_deltas_via_points(
    params: &CodedArrayParams,
    coded: &mut CodedStore,
    seg: usize,
    deltas: &[(usize, Fp)],
    mode: DeltaMode,
) {
    apply_deltas_to_segment_with_mode(params, coded, seg, deltas, mode);
    apply_row_deltas_to_segment(params, coded, seg, deltas);
}

fn apply_deltas_to_segment_on_coset(
    params: &CodedArrayParams,
    coded: &mut CodedStore,
    seg: usize,
    rho: Fp,
    deltas: &[(usize, Fp)],
) {
    if deltas.is_empty() {
        return;
    }
    let rho_n = fp_pow(rho, params.n_grid as u64);
    let inv_rho_n = rho_n
        .inverse()
        .expect("segment domain scale must be non-zero");
    coded.for_each_shard_segment_mut(params, seg, |shard_key, y_vec| {
        let entry_idx = params
            .shard_registry_by_key
            .get(shard_key)
            .copied()
            .expect("missing shard entry");
        let entry = params.shard_registry[entry_idx];
        let k_rho = fp_pow(entry.a, params.n_grid as u64) - rho_n;
        let betas = params.h.iter().map(|&h| entry.a * h).collect::<Vec<_>>();
        let mut denoms = vec![fp_zero(); params.s];
        for &(grid_index, delta) in deltas {
            let alpha = rho * params.grid_point(grid_index);
            let weight = alpha * params.inv_n_grid * inv_rho_n;
            let scale = delta * weight * k_rho;
            let mut zero_pos = None;
            for (t, denom) in denoms.iter_mut().enumerate() {
                *denom = betas[t] - alpha;
                if *denom == fp_zero() {
                    zero_pos = Some(t);
                }
            }
            if let Some(zero_pos) = zero_pos {
                for (t, value) in y_vec.iter_mut().enumerate() {
                    if t == zero_pos {
                        *value += delta;
                    } else {
                        *value += scale * denoms[t].inverse().expect("distinct point");
                    }
                }
            } else {
                batch_invert(&mut denoms);
                for (value, inv_denom) in y_vec.iter_mut().zip(denoms.iter()) {
                    *value += scale * *inv_denom;
                }
            }
        }
    });
}

fn row_update_basis_for_column_coset(
    params: &CodedArrayParams,
    source_column: Fp,
    coset_vanishing_value: Fp,
    out: &mut [Fp],
) {
    debug_assert_eq!(out.len(), params.row_eval_points.len());
    if out.is_empty() {
        return;
    }
    let inv_n = Fp::from(params.induced_z_grid_size() as u64)
        .inverse()
        .expect("induced Z-grid size is non-zero");
    let inv_coset_vanishing = coset_vanishing_value
        .inverse()
        .expect("column coset vanishing value is non-zero");
    let common = source_column * inv_n * inv_coset_vanishing;
    let mut active = Vec::new();
    let mut denoms = Vec::new();
    for (idx, (value, &z_eval)) in out
        .iter_mut()
        .zip(params.row_eval_points.iter())
        .enumerate()
    {
        if z_eval == source_column {
            *value = fp_one();
            continue;
        }
        let numerator = fp_pow(z_eval, params.induced_z_grid_size() as u64) - coset_vanishing_value;
        if numerator == fp_zero() {
            *value = fp_zero();
            continue;
        }
        *value = common * numerator;
        active.push(idx);
        denoms.push(z_eval - source_column);
    }
    batch_invert(&mut denoms);
    for (idx, inv_denom) in active.into_iter().zip(denoms.into_iter()) {
        out[idx] *= inv_denom;
    }
}

fn apply_row_deltas_to_segment_on_coset(
    params: &CodedArrayParams,
    coded: &mut CodedStore,
    seg: usize,
    rho: Fp,
    deltas: &[(usize, Fp)],
) {
    if deltas.is_empty() || coded.row_family_count() == 0 {
        return;
    }
    let mut active_rows = collect_active_row_entries_mut(params, coded, seg);
    if active_rows.is_empty() {
        return;
    }

    let coset_vanishing_value = fp_pow(rho, params.n_grid as u64);
    let mut basis = vec![fp_zero(); params.row_eval_points.len()];
    for &(grid_index, delta) in deltas {
        let alpha = rho * params.grid_point(grid_index);
        let source_column = params.column_coordinate(alpha);
        row_update_basis_for_column_coset(params, source_column, coset_vanishing_value, &mut basis);
        let alpha_pow = if params.s == 1 {
            fp_one()
        } else {
            fp_pow(alpha, (params.s - 1) as u64)
        };
        let denom_common = Fp::from(params.s as u64) * alpha_pow;
        let mut active_scales = Vec::new();
        let mut denoms = Vec::new();
        for row in &mut active_rows {
            let mut scale = fp_zero();
            if row.row_label == alpha {
                scale = fp_one();
            } else {
                let numerator = fp_pow(row.row_label, params.s as u64) - source_column;
                if numerator != fp_zero() {
                    scale = numerator;
                    active_scales.push(scale);
                    denoms.push(denom_common * (row.row_label - alpha));
                    continue;
                }
            }
            if scale == fp_zero() {
                continue;
            }
            let scaled_delta = delta * scale;
            let row_values = &mut *row.row_values;
            for (value, basis_value) in row_values.iter_mut().zip(basis.iter()) {
                *value += scaled_delta * *basis_value;
            }
        }
        if !active_scales.is_empty() {
            batch_invert(&mut denoms);
            let mut active_idx = 0usize;
            for row in &mut active_rows {
                if row.row_label == alpha {
                    continue;
                }
                let numerator = fp_pow(row.row_label, params.s as u64) - source_column;
                if numerator == fp_zero() {
                    continue;
                }
                let scale = active_scales[active_idx] * denoms[active_idx];
                active_idx += 1;
                let scaled_delta = delta * scale;
                let row_values = &mut *row.row_values;
                for (value, basis_value) in row_values.iter_mut().zip(basis.iter()) {
                    *value += scaled_delta * *basis_value;
                }
            }
        }
    }
}

fn column_basis_at_z(params: &CodedArrayParams, source_column: Fp, z_eval: Fp) -> Fp {
    if z_eval == source_column {
        return fp_one();
    }
    let numerator = fp_pow(z_eval, params.induced_z_grid_size() as u64) - fp_one();
    if numerator == fp_zero() {
        return fp_zero();
    }
    let denom = z_eval - source_column;
    let inv_n = Fp::from(params.induced_z_grid_size() as u64)
        .inverse()
        .expect("induced Z-grid size is non-zero");
    source_column * inv_n * numerator * denom.inverse().expect("distinct column coordinate")
}

fn apply_block_deltas_to_segment(
    params: &CodedArrayParams,
    coded: &mut CodedStore,
    seg: usize,
    block_in_segment: usize,
    delta_values: &[Fp],
    weights: &crate::coded_array::rs::BarycentricWeights,
) {
    if coded.shard_family_count() == 0 {
        return;
    }
    let source_column = params.column_coordinate_for_grid_index(block_in_segment);
    coded.for_each_shard_segment_mut(params, seg, |shard_key, y_vec| {
        let entry_idx = params
            .shard_registry_by_key
            .get(shard_key)
            .copied()
            .expect("missing shard entry");
        let entry = params.shard_registry[entry_idx];
        let lambda = column_basis_at_z(params, source_column, entry.z);
        if lambda == fp_zero() {
            return;
        }
        for (t, value) in y_vec.iter_mut().enumerate() {
            let eval_point = entry.a * params.h[t];
            let delta_value = barycentric_eval(weights, delta_values, eval_point);
            *value += lambda * delta_value;
        }
    });
}

fn apply_block_row_deltas_to_segment(
    params: &CodedArrayParams,
    coded: &mut CodedStore,
    seg: usize,
    block_in_segment: usize,
    delta_values: &[Fp],
    weights: &crate::coded_array::rs::BarycentricWeights,
) {
    if coded.row_family_count() == 0 {
        return;
    }

    let mut active_rows = collect_active_row_entries_mut(params, coded, seg);
    if active_rows.is_empty() {
        return;
    }

    ROW_HELPER_SCRATCH.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        if scratch.scale_cache.len() != params.row_labels.len() {
            scratch.scale_cache.clear();
            scratch.scale_cache.resize(params.row_labels.len(), None);
        }
        let basis = params.row_update_basis_at_grid_index(block_in_segment);
        scratch.scale_cache_touched.clear();
        for row in &mut active_rows {
            let row_label_index = row.row_label_index;
            let scale = match scratch.scale_cache[row_label_index] {
                Some(scale) => scale,
                None => {
                    let scale = barycentric_eval(weights, delta_values, row.row_label);
                    scratch.scale_cache[row_label_index] = Some(scale);
                    scratch.scale_cache_touched.push(row_label_index);
                    scale
                }
            };
            if scale == fp_zero() {
                continue;
            }
            let row_values = &mut *row.row_values;
            for (value, basis_value) in row_values.iter_mut().zip(basis.iter()) {
                *value += scale * *basis_value;
            }
        }
        let touched: Vec<_> = scratch.scale_cache_touched.drain(..).collect();
        for row_label_index in touched {
            scratch.scale_cache[row_label_index] = None;
        }
    });
}

fn apply_row_deltas_to_segment(
    params: &CodedArrayParams,
    coded: &mut CodedStore,
    seg: usize,
    deltas: &[(usize, Fp)],
) {
    if deltas.is_empty() || coded.row_family_count() == 0 {
        return;
    }

    let mut active_rows = collect_active_row_entries_mut(params, coded, seg);
    if active_rows.is_empty() {
        return;
    }

    if deltas.len() >= ROW_INTERPOLATE_THRESHOLD {
        apply_row_deltas_to_segment_interpolated(params, &mut active_rows, deltas);
        return;
    }

    ROW_HELPER_SCRATCH.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        if scratch.scale_cache.len() != params.row_labels.len() {
            scratch.scale_cache.clear();
            scratch.scale_cache.resize(params.row_labels.len(), None);
        }
        for &(grid_index, delta) in deltas {
            let basis = params.row_update_basis_at_grid_index(grid_index);
            scratch.scale_cache_touched.clear();
            for row in &mut active_rows {
                let row_label_index = row.row_label_index;
                let scale = match scratch.scale_cache[row_label_index] {
                    Some(scale) => scale,
                    None => {
                        let scale = crate::coded_array::params::row_update_scale_for_grid_index(
                            params,
                            grid_index,
                            row.row_label,
                        );
                        scratch.scale_cache[row_label_index] = Some(scale);
                        scratch.scale_cache_touched.push(row_label_index);
                        scale
                    }
                };
                if scale == fp_zero() {
                    continue;
                }
                let row_values = &mut *row.row_values;
                let scaled_delta = delta * scale;
                for (value, basis_value) in row_values.iter_mut().zip(basis.iter()) {
                    *value += scaled_delta * *basis_value;
                }
            }
            let touched: Vec<_> = scratch.scale_cache_touched.drain(..).collect();
            for row_label_index in touched {
                scratch.scale_cache[row_label_index] = None;
            }
        }
    });
}

fn apply_row_deltas_to_segment_interpolated(
    params: &CodedArrayParams,
    active_rows: &mut [ActiveRowMut<'_>],
    deltas: &[(usize, Fp)],
) {
    ROW_HELPER_SCRATCH.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        if scratch.coeffs.len() != params.n_grid {
            scratch.coeffs.resize(params.n_grid, fp_zero());
        }
        for value in scratch.coeffs.iter_mut() {
            *value = fp_zero();
        }
        for &(grid_index, delta) in deltas {
            if grid_index < params.n_grid {
                scratch.coeffs[grid_index] += delta;
            }
        }
        crate::coded_array::params::inverse_fft_in_place(&mut scratch.coeffs, params.omega);

        let row_label_indices = active_rows
            .iter()
            .map(|row| row.row_label_index)
            .collect::<Vec<_>>();
        let row_deltas = crate::coded_array::params::compute_selected_rows_from_coeffs(
            params,
            &scratch.coeffs,
            &row_label_indices,
            None,
        )
        .into_iter()
        .collect::<HashMap<_, _>>();

        for row in active_rows {
            let Some(delta_row) = row_deltas.get(&row.row_label_index) else {
                continue;
            };
            let row_values = &mut *row.row_values;
            debug_assert_eq!(row_values.len(), delta_row.len());
            for (value, delta) in row_values.iter_mut().zip(delta_row.iter()) {
                *value += *delta;
            }
        }
    });
}

#[derive(Clone, Copy)]
struct ActiveRow<'a> {
    row_label_bytes: Value256,
    row_label: Fp,
    row_values: &'a [Fp],
}

struct ActiveRowMut<'a> {
    row_label_index: usize,
    row_label: Fp,
    row_values: &'a mut [Fp],
}

fn collect_active_row_entries<'a>(
    params: &'a CodedArrayParams,
    coded: &'a CodedStore,
    seg: usize,
) -> Vec<ActiveRow<'a>> {
    let row_segments = coded.row_segments_for_segment(params, seg);
    let mut rows = Vec::with_capacity(row_segments.len());
    for (row_key, row_values) in row_segments {
        let Some(row_label_index) = params.row_label_index_for_bytes(row_key.row_label_bytes)
        else {
            continue;
        };
        rows.push(ActiveRow {
            row_label_bytes: row_key.row_label_bytes,
            row_label: params.row_labels[row_label_index],
            row_values,
        });
    }
    rows.sort_by_key(|row| row.row_label_bytes);
    rows.dedup_by_key(|row| row.row_label_bytes);
    rows
}

fn collect_local_shard_entries(
    params: &CodedArrayParams,
    coded: &CodedStore,
    seg: Option<usize>,
) -> Vec<(usize, ShardRegEntry)> {
    let mut shards = Vec::with_capacity(coded.shard_family_count());
    for key in coded.shard_keys() {
        if let Some(seg) = seg {
            if coded.shard_segment(params, key, seg).is_none() {
                continue;
            }
        }
        let Some(&registry_index) = params.shard_registry_by_key.get(key) else {
            continue;
        };
        let Some(&entry) = params.shard_registry.get(registry_index) else {
            continue;
        };
        shards.push((registry_index, entry));
    }
    shards.sort_unstable_by_key(|(registry_index, _)| *registry_index);
    shards.dedup_by_key(|(registry_index, _)| *registry_index);
    shards
}

fn collect_active_row_entries_mut<'a>(
    params: &'a CodedArrayParams,
    coded: &'a mut CodedStore,
    seg: usize,
) -> Vec<ActiveRowMut<'a>> {
    let row_segments = coded.row_segments_mut_for_segment(params, seg);
    let mut rows = Vec::with_capacity(row_segments.len());
    for (row_key, row_values) in row_segments {
        let Some(row_label_index) = params.row_label_index_for_bytes(row_key.row_label_bytes)
        else {
            continue;
        };
        rows.push(ActiveRowMut {
            row_label_index,
            row_label: params.row_labels[row_label_index],
            row_values,
        });
    }
    rows
}

fn collect_row_samples_at_z(
    params: &CodedArrayParams,
    coded: &CodedStore,
    seg: usize,
    z_eval: Fp,
) -> Vec<RowShare> {
    if coded.row_family_count() == 0 {
        return Vec::new();
    }
    let active_rows = collect_active_row_entries(params, coded, seg);
    let mut rows = Vec::with_capacity(active_rows.len());
    for row in active_rows {
        rows.push(RowShare {
            row_label: row.row_label,
            value: barycentric_eval(&params.row_eval_bary_weights, row.row_values, z_eval),
        });
    }
    rows
}

fn induced_z_subgroup_indices(params: &CodedArrayParams, z_evals: &[Fp]) -> Option<Vec<usize>> {
    let mut indices = Vec::with_capacity(z_evals.len());
    for &z_eval in z_evals {
        let idx = params
            .induced_z_subgroup_index_by_bytes
            .get(&encode_value(z_eval))
            .copied()?;
        indices.push(idx);
    }
    Some(indices)
}

fn evaluate_row_on_induced_z_subgroup_into(
    params: &CodedArrayParams,
    row_values: &[Fp],
    coeffs: &mut Vec<Fp>,
    out: &mut [Fp],
) {
    debug_assert_eq!(row_values.len(), params.row_eval_points.len());
    debug_assert_eq!(out.len(), params.induced_z_grid_size());
    let z_root = fp_pow(params.omega, params.s as u64);
    coeffs.clear();
    coeffs.extend_from_slice(row_values);
    crate::coded_array::params::inverse_fft_in_place(coeffs.as_mut_slice(), z_root);

    let base_z = params.row_eval_points[0];
    let inv_base_z = base_z
        .inverse()
        .expect("row evaluation coset base is non-zero");
    let mut inv_base_pow = fp_one();
    for coeff in coeffs.iter_mut() {
        *coeff *= inv_base_pow;
        inv_base_pow *= inv_base_z;
    }

    crate::coded_array::params::fft_in_place(coeffs.as_mut_slice(), z_root);
    out.copy_from_slice(coeffs.as_slice());
}

fn evaluate_stored_row_at_z_evals_from_coefficients(
    params: &CodedArrayParams,
    row_values: &[Fp],
    z_evals: &[Fp],
    out: &mut [Fp],
    coeffs: &mut Vec<Fp>,
) {
    debug_assert_eq!(row_values.len(), params.row_eval_points.len());
    debug_assert_eq!(out.len(), z_evals.len());
    let z_root = fp_pow(params.omega, params.s as u64);
    coeffs.clear();
    coeffs.extend_from_slice(row_values);
    crate::coded_array::params::inverse_fft_in_place(coeffs.as_mut_slice(), z_root);

    let base_z = params.row_eval_points[0];
    let inv_base_z = base_z
        .inverse()
        .expect("row evaluation coset base is non-zero");
    let mut inv_base_pow = fp_one();
    for coeff in coeffs.iter_mut() {
        *coeff *= inv_base_pow;
        inv_base_pow *= inv_base_z;
    }

    for (value, &z_eval) in out.iter_mut().zip(z_evals.iter()) {
        let mut acc = fp_zero();
        for coeff in coeffs.iter().rev() {
            acc = acc * z_eval + *coeff;
        }
        *value = acc;
    }
}

fn collect_row_samples_batch_generic(
    params: &CodedArrayParams,
    coded: &CodedStore,
    seg: usize,
    z_evals: &[Fp],
) -> Vec<Vec<RowShare>> {
    if coded.row_family_count() == 0 || z_evals.is_empty() {
        return vec![Vec::new(); z_evals.len()];
    }

    ROW_HELPER_SCRATCH.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        let active_rows = collect_active_row_entries(params, coded, seg);
        let mut batches = (0..z_evals.len())
            .map(|_| Vec::with_capacity(active_rows.len()))
            .collect::<Vec<_>>();
        let RowHelperScratch {
            coeffs,
            batch_values,
            ..
        } = &mut *scratch;
        batch_values.resize(active_rows.len() * z_evals.len(), fp_zero());
        let z_root = fp_pow(params.omega, params.s as u64);
        let use_coefficient_eval = z_evals.len() > 1
            && params.need >= ROW_MATERIALIZE_COEFF_THRESHOLD
            && params.row_eval_points.len() == params.need
            && crate::coded_array::params::row_eval_points_are_structured(params, z_root);
        for (row_pos, row) in active_rows.iter().enumerate() {
            let base = row_pos * z_evals.len();
            let row_slice = &mut batch_values[base..base + z_evals.len()];
            if use_coefficient_eval {
                evaluate_stored_row_at_z_evals_from_coefficients(
                    params,
                    row.row_values,
                    z_evals,
                    row_slice,
                    coeffs,
                );
            } else {
                for (value, &z_eval) in row_slice.iter_mut().zip(z_evals.iter()) {
                    *value =
                        barycentric_eval(&params.row_eval_bary_weights, row.row_values, z_eval);
                }
            }
        }
        for (row_pos, row) in active_rows.iter().enumerate() {
            let base = row_pos * z_evals.len();
            for (batch, &value) in batches
                .iter_mut()
                .zip(batch_values[base..base + z_evals.len()].iter())
            {
                batch.push(RowShare {
                    row_label: row.row_label,
                    value,
                });
            }
        }
        batches
    })
}

fn collect_row_samples_batch_on_induced_z_grid(
    params: &CodedArrayParams,
    coded: &CodedStore,
    seg: usize,
    subgroup_indices: &[usize],
) -> Vec<Vec<RowShare>> {
    if coded.row_family_count() == 0 || subgroup_indices.is_empty() {
        return vec![Vec::new(); subgroup_indices.len()];
    }

    ROW_HELPER_SCRATCH.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        if scratch.subgroup_values.len() != params.induced_z_grid_size() {
            scratch
                .subgroup_values
                .resize(params.induced_z_grid_size(), fp_zero());
        }
        let active_rows = collect_active_row_entries(params, coded, seg);
        let mut batches = (0..subgroup_indices.len())
            .map(|_| Vec::with_capacity(active_rows.len()))
            .collect::<Vec<_>>();
        let RowHelperScratch {
            coeffs,
            subgroup_values,
            batch_values,
            ..
        } = &mut *scratch;
        batch_values.resize(active_rows.len() * subgroup_indices.len(), fp_zero());
        for (row_pos, row) in active_rows.iter().enumerate() {
            evaluate_row_on_induced_z_subgroup_into(
                params,
                row.row_values,
                coeffs,
                subgroup_values,
            );
            let base = row_pos * subgroup_indices.len();
            for (offset, &idx) in subgroup_indices.iter().enumerate() {
                let value = subgroup_values[idx];
                batch_values[base + offset] = value;
            }
        }
        for (row_pos, row) in active_rows.iter().enumerate() {
            let base = row_pos * subgroup_indices.len();
            for (batch, &value) in batches
                .iter_mut()
                .zip(batch_values[base..base + subgroup_indices.len()].iter())
            {
                batch.push(RowShare {
                    row_label: row.row_label,
                    value,
                });
            }
        }
        batches
    })
}

fn collect_row_samples_batch_at_z(
    params: &CodedArrayParams,
    coded: &CodedStore,
    seg: usize,
    z_evals: &[Fp],
) -> Vec<Vec<RowShare>> {
    if z_evals.is_empty() {
        return Vec::new();
    }

    let mut unique_z_evals = Vec::with_capacity(z_evals.len());
    let mut unique_index_by_bytes = HashMap::with_capacity(z_evals.len());
    let mut scatter = Vec::with_capacity(z_evals.len());
    for &z_eval in z_evals {
        let batch_pos = match unique_index_by_bytes.entry(encode_value(z_eval)) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                let idx = unique_z_evals.len();
                unique_z_evals.push(z_eval);
                entry.insert(idx);
                idx
            }
            std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
        };
        scatter.push(batch_pos);
    }

    let unique_batches = match induced_z_subgroup_indices(params, &unique_z_evals) {
        Some(subgroup_indices) => {
            collect_row_samples_batch_on_induced_z_grid(params, coded, seg, &subgroup_indices)
        }
        None => collect_row_samples_batch_generic(params, coded, seg, &unique_z_evals),
    };

    if unique_z_evals.len() == z_evals.len() {
        unique_batches
    } else {
        scatter
            .into_iter()
            .map(|batch_pos| unique_batches[batch_pos].clone())
            .collect()
    }
}

/// Decode a segment value from shard values.
pub(crate) fn decode_from_shares_at_z<P: UnauthDecodeParams>(
    params: &P,
    z_eval: Fp,
    shares: &[ShardShare],
) -> Result<Fp, CodedReadError> {
    decode_from_shares_at_z_with_cache(params, z_eval, shares, None)
}

/// Decode a segment value from oversampled shard values while correcting up to
/// `max_errors` arbitrary bad helper replies.
pub fn decode_from_shares_with_error_correction<P: UnauthDecodeParams>(
    params: &P,
    j: usize,
    shares: &[ShardShare],
    max_errors: usize,
) -> Result<CorrectedShardDecode, CodedReadError> {
    let z_eval = params.decode_eval(j);
    decode_from_shares_at_z_with_error_correction(params, z_eval, shares, max_errors)
}

pub fn decode_from_shares_at_z_with_error_correction<P: UnauthDecodeParams>(
    params: &P,
    z_eval: Fp,
    shares: &[ShardShare],
    max_errors: usize,
) -> Result<CorrectedShardDecode, CodedReadError> {
    let shard_registry = params.shard_registry();
    let mut seen = HashSet::new();
    let mut points = Vec::new();
    let mut registry_indices = Vec::new();
    for share in shares {
        let Some(y) = share.value else {
            continue;
        };
        if share.registry_index >= shard_registry.len() || !seen.insert(share.registry_index) {
            continue;
        }
        let entry = shard_registry[share.registry_index];
        points.push((entry.z, y));
        registry_indices.push(share.registry_index);
    }
    if points.len() < params.need() + 2 * max_errors {
        return Err(CodedReadError::NotEnoughShards);
    }
    let decoded = crate::coded_array::rs::rs_decode_eval_with_errors(
        &points,
        params.need(),
        z_eval,
        max_errors,
    )
    .map_err(|_| CodedReadError::NotEnoughShards)?;
    let mut bad_registry_indices = decoded
        .bad_indices
        .into_iter()
        .filter_map(|idx| registry_indices.get(idx).copied())
        .collect::<Vec<_>>();
    bad_registry_indices.sort_unstable();
    Ok(CorrectedShardDecode {
        value: decoded.value,
        bad_registry_indices,
        share_count: points.len(),
    })
}

pub(crate) fn decode_from_shares_at_z_with_cache<P: UnauthDecodeParams>(
    params: &P,
    z_eval: Fp,
    shares: &[ShardShare],
    cache: Option<&DecodeCache>,
) -> Result<Fp, CodedReadError> {
    let mut keys = Vec::new();
    let mut z_values = Vec::new();
    let mut r_values = Vec::new();
    let shard_registry = params.shard_registry();
    for share in shares {
        let Some(y) = share.value else {
            continue;
        };
        if share.registry_index >= shard_registry.len() {
            continue;
        }
        let entry = shard_registry[share.registry_index];
        keys.push(entry.key);
        z_values.push(entry.z);
        r_values.push(y);
        if r_values.len() >= params.need() {
            break;
        }
    }
    if r_values.len() < params.need() {
        return Err(CodedReadError::NotEnoughShards);
    }
    let weights = match cache {
        Some(cache) => cache.get_weights_for_keys_and_z(&keys, &z_values),
        None => precompute_barycentric(&z_values),
    };
    Ok(barycentric_eval(&weights, &r_values, z_eval))
}

fn decode_slice_from_rows(
    params: &CodedArrayParams,
    rows: &[RowShare],
    eval_points: &[Fp],
) -> Result<Vec<Fp>, CodedReadError> {
    let mut row_labels = Vec::new();
    let mut row_label_indices = Vec::new();
    let mut row_values = Vec::new();
    let mut seen = HashSet::new();
    for row in rows {
        let label_bytes = encode_value(row.row_label);
        if !seen.insert(label_bytes) {
            continue;
        }
        let Some(row_label_index) = params.row_label_index_for_bytes(label_bytes) else {
            continue;
        };
        row_labels.push(row.row_label);
        row_label_indices.push(row_label_index);
        row_values.push(row.value);
        if row_values.len() >= params.s {
            break;
        }
    }
    if row_values.len() < params.s {
        return Err(CodedReadError::NotEnoughShards);
    }

    if let Some(values) =
        try_decode_slice_from_rows_fft(params, &row_label_indices, &row_values, eval_points)
    {
        return Ok(values);
    }

    let weights = precompute_barycentric(&row_labels);
    let mut values = Vec::with_capacity(eval_points.len());
    for &eval_point in eval_points {
        values.push(barycentric_eval(&weights, &row_values, eval_point));
    }
    Ok(values)
}

fn decode_primary_shard_from_rows(
    params: &CodedArrayParams,
    entry: ShardRegEntry,
    rows: &[RowShare],
) -> Result<Vec<Fp>, CodedReadError> {
    let eval_points: Vec<_> = (0..params.s).map(|t| entry.a * params.h[t]).collect();
    decode_slice_from_rows(params, rows, &eval_points)
}

fn try_decode_slice_from_rows_fft(
    params: &CodedArrayParams,
    row_label_indices: &[usize],
    row_values: &[Fp],
    eval_points: &[Fp],
) -> Option<Vec<Fp>> {
    if params.s == 0
        || row_values.len() < params.s
        || row_label_indices.len() < params.s
        || eval_points.len() != params.s
        || params.row_label_coset_bases.is_empty()
    {
        return None;
    }

    let root = fp_pow(params.omega, (params.n_grid / params.s) as u64);
    let target_base = *eval_points.first()?;
    let mut cur = fp_one();
    for &eval_point in eval_points {
        if eval_point != target_base * cur {
            return None;
        }
        cur *= root;
    }

    let mut by_coset = vec![vec![None; params.s]; params.row_label_coset_bases.len()];
    let mut counts = vec![0usize; params.row_label_coset_bases.len()];
    let mut chosen_coset = None;
    for (&row_label_index, &row_value) in row_label_indices.iter().zip(row_values.iter()) {
        let coset_idx = row_label_index / params.s;
        let offset = row_label_index % params.s;
        let Some(row_values_by_offset) = by_coset.get_mut(coset_idx) else {
            continue;
        };
        if row_values_by_offset[offset].replace(row_value).is_none() {
            counts[coset_idx] += 1;
            if counts[coset_idx] == params.s {
                chosen_coset = Some(coset_idx);
                break;
            }
        }
    }

    let coset_idx = chosen_coset?;
    let source_base = params.row_label_coset_bases[coset_idx];
    let source_values = &by_coset[coset_idx];
    let mut coeffs = Vec::with_capacity(params.s);
    for maybe_value in source_values {
        coeffs.push((*maybe_value)?);
    }

    crate::coded_array::params::inverse_fft_in_place(coeffs.as_mut_slice(), root);
    let inv_source_base = source_base.inverse()?;
    let mut inv_source_pow = fp_one();
    for coeff in coeffs.iter_mut() {
        *coeff *= inv_source_pow;
        inv_source_pow *= inv_source_base;
    }

    let mut values = vec![fp_zero(); params.s];
    let mut target_pow = fp_one();
    for (idx, &coeff) in coeffs.iter().enumerate() {
        values[idx] = coeff * target_pow;
        target_pow *= target_base;
    }
    crate::coded_array::params::fft_in_place(values.as_mut_slice(), root);
    Some(values)
}

fn decode_primary_shards_from_row_batches(
    params: &CodedArrayParams,
    entries: &[ShardRegEntry],
    row_batches: &[Vec<RowShare>],
) -> Result<Vec<Vec<Fp>>, CodedReadError> {
    debug_assert_eq!(entries.len(), row_batches.len());
    entries
        .iter()
        .copied()
        .zip(row_batches.iter())
        .map(|(entry, rows)| decode_primary_shard_from_rows(params, entry, rows))
        .collect()
}

#[allow(dead_code)]
/// Decode one aligned semantic block from maintained row replies.
pub(crate) fn decode_block_from_rows(
    params: &CodedArrayParams,
    block: Index,
    rows: &[RowShare],
) -> Result<Vec<Fp>, CodedReadError> {
    let pos = semantic_block_position(params, block);
    let eval_points = block_alphas_for_segment(params, pos.segment, pos.block_in_segment);
    decode_slice_from_rows(params, rows, &eval_points)
}

fn decode_block_from_rows_correcting(
    params: &CodedArrayParams,
    block: Index,
    rows: &[RowShare],
    max_errors: usize,
) -> Result<Vec<Fp>, CodedReadError> {
    let pos = semantic_block_position(params, block);
    let eval_points = block_alphas_for_segment(params, pos.segment, pos.block_in_segment);
    decode_slice_from_rows_correcting(params, rows, &eval_points, max_errors)
}

fn decode_slice_from_rows_correcting(
    params: &CodedArrayParams,
    rows: &[RowShare],
    eval_points: &[Fp],
    max_errors: usize,
) -> Result<Vec<Fp>, CodedReadError> {
    let mut seen = HashSet::new();
    let mut points = Vec::new();
    for row in rows {
        let label_bytes = encode_value(row.row_label);
        if params.row_label_index_for_bytes(label_bytes).is_none() || !seen.insert(label_bytes) {
            continue;
        }
        points.push((row.row_label, row.value));
        if points.len() >= params.s + 2 * max_errors {
            break;
        }
    }
    if points.len() < params.s + 2 * max_errors {
        return Err(CodedReadError::NotEnoughShards);
    }

    let decoded = crate::coded_array::rs::rs_decode_eval_with_errors(
        &points,
        params.s,
        eval_points.first().copied().unwrap_or_else(fp_zero),
        max_errors,
    )
    .map_err(|_| CodedReadError::NotEnoughShards)?;
    Ok(eval_coefficients_on_points(
        params,
        &decoded.coefficients,
        eval_points,
    ))
}

fn eval_coefficients_on_points(
    params: &CodedArrayParams,
    coefficients: &[Fp],
    eval_points: &[Fp],
) -> Vec<Fp> {
    if eval_points.is_empty() {
        return Vec::new();
    }
    if coefficients.len() <= eval_points.len() && eval_points.len().is_power_of_two() {
        let root = fp_pow(params.omega, (params.n_grid / eval_points.len()) as u64);
        let base = eval_points[0];
        let mut cur = base;
        let mut structured = true;
        for &eval_point in eval_points {
            if eval_point != cur {
                structured = false;
                break;
            }
            cur *= root;
        }
        if structured {
            let mut values = vec![fp_zero(); eval_points.len()];
            values[..coefficients.len()].copy_from_slice(coefficients);
            let mut base_pow = fp_one();
            for coeff in values.iter_mut() {
                *coeff *= base_pow;
                base_pow *= base;
            }
            crate::coded_array::params::fft_in_place(&mut values, root);
            return values;
        }
    }

    eval_points
        .iter()
        .map(|&eval_point| eval_coefficients(coefficients, eval_point))
        .collect()
}

fn eval_coefficients(coefficients: &[Fp], x: Fp) -> Fp {
    let mut acc = fp_zero();
    for &coeff in coefficients.iter().rev() {
        acc *= x;
        acc += coeff;
    }
    acc
}

pub(crate) fn evaluate_shard_at_row_label(
    params: &CodedArrayParams,
    registry_index: usize,
    shard_values: &[Fp],
    row_label: Fp,
) -> Fp {
    debug_assert_eq!(shard_values.len(), params.s);
    let query = row_label * params.inv_a[registry_index];
    barycentric_eval(&params.h_bary_weights, shard_values, query)
}

fn decode_row_from_shards(
    params: &CodedArrayParams,
    coded: &CodedStore,
    seg: usize,
    row_label: Fp,
) -> Result<Vec<Fp>, CodedReadError> {
    let mut z_values = Vec::new();
    let mut row_values = Vec::new();
    for (registry_index, entry) in params.shard_registry.iter().copied().enumerate() {
        let Some(shard_values) = coded.shard_segment(params, &entry.key, seg) else {
            continue;
        };
        z_values.push(entry.z);
        row_values.push(evaluate_shard_at_row_label(
            params,
            registry_index,
            shard_values,
            row_label,
        ));
        if row_values.len() >= params.need {
            break;
        }
    }
    interpolate_row_from_samples(params, &z_values, &row_values)
}

fn decode_row_from_target_shards(
    params: &CodedArrayParams,
    coded: &CodedStore,
    source_shards: &TargetShardsForRowHydration<'_>,
    seg: usize,
    row_label: Fp,
) -> Result<Vec<Fp>, CodedReadError> {
    let mut z_values = Vec::new();
    let mut row_values = Vec::new();
    for (&registry_index, source) in &source_shards.shards {
        let Some(&_entry) = params.shard_registry.get(registry_index) else {
            return Err(CodedReadError::NotEnoughShards);
        };
        let shard_values = target_shard_segment(params, coded, source, seg)
            .map_err(|_| CodedReadError::NotEnoughShards)?;
        z_values.push(params.shard_registry[registry_index].z);
        row_values.push(evaluate_shard_at_row_label(
            params,
            registry_index,
            shard_values,
            row_label,
        ));
        if row_values.len() >= params.need {
            break;
        }
    }
    interpolate_row_from_samples(params, &z_values, &row_values)
}

fn interpolate_row_from_samples(
    params: &CodedArrayParams,
    z_values: &[Fp],
    row_values: &[Fp],
) -> Result<Vec<Fp>, CodedReadError> {
    materialize_row_on_eval_points_from_samples(params, z_values, row_values, None)
}

pub(crate) fn materialize_row_on_eval_points_from_coefficients(
    params: &CodedArrayParams,
    coefficients: &[Fp],
) -> Vec<Fp> {
    crate::coded_array::params::evaluate_row_coefficients_on_row_eval_points(params, coefficients)
}

pub(crate) fn materialize_row_on_eval_points_from_samples(
    params: &CodedArrayParams,
    z_values: &[Fp],
    row_values: &[Fp],
    weights: Option<&BarycentricWeights>,
) -> Result<Vec<Fp>, CodedReadError> {
    if row_values.len() < params.need {
        return Err(CodedReadError::NotEnoughShards);
    }

    if z_values.len() < params.need {
        return Err(CodedReadError::NotEnoughShards);
    }

    if params.need >= ROW_MATERIALIZE_COEFF_THRESHOLD {
        let points = z_values
            .iter()
            .copied()
            .zip(row_values.iter().copied())
            .take(params.need)
            .collect::<Vec<_>>();
        if let Some(coefficients) =
            crate::coded_array::rs::interpolate_coefficients_from_exact_points(&points, params.need)
        {
            return Ok(materialize_row_on_eval_points_from_coefficients(
                params,
                &coefficients,
            ));
        }
    }

    let owned_weights;
    let weights = match weights {
        Some(weights) => weights,
        None => {
            owned_weights = precompute_barycentric(&z_values[..params.need]);
            &owned_weights
        }
    };
    if row_values.len() < weights.z.len() {
        return Err(CodedReadError::NotEnoughShards);
    }
    let row_values = &row_values[..weights.z.len()];
    let mut repaired = Vec::with_capacity(params.row_eval_points.len());
    for &z_eval in &params.row_eval_points {
        repaired.push(barycentric_eval(&weights, &row_values, z_eval));
    }
    Ok(repaired)
}

/// Read multiple indices from the coded store, returning encoded values.
pub fn read_batch_from_store(
    params: &CodedArrayParams,
    store: &CodedStore,
    idxs: &[Index],
    decode_cache: Option<&DecodeCache>,
) -> Result<Vec<Value256>, ReadError> {
    if idxs.is_empty() {
        return Ok(Vec::new());
    }
    let b = public_values_per_segment(params) as u64;
    let mut by_seg: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    let mut z_evals = vec![fp_zero(); idxs.len()];
    for (pos, &idx) in idxs.iter().enumerate() {
        let seg = usize::try_from(idx / b).map_err(|_| ReadError::InvalidIndex)?;
        let logical_offset = usize::try_from(idx % b).map_err(|_| ReadError::InvalidIndex)?;
        let grid_index = logical_offset_to_grid_index(params, logical_offset);
        by_seg.entry(seg).or_default().push((pos, grid_index));
    }
    let mut shares_by_pos: Vec<Vec<ShardShare>> = vec![Vec::new(); idxs.len()];
    for (seg, items) in by_seg {
        if items.is_empty() {
            continue;
        }
        let mut remaining = items.len();
        let rho = segment_domain_rho(params, seg);
        let grid_indices = items
            .iter()
            .map(|&(pos, grid_index)| {
                let alpha = rho * params.grid_point(grid_index);
                z_evals[pos] = params.column_coordinate(alpha);
                grid_index
            })
            .collect::<Vec<_>>();
        for (reg_idx, entry) in collect_local_shard_entries(params, store, Some(seg)) {
            if remaining == 0 {
                break;
            }
            let replies = shard_value_from_store_batch_on_grid_coset(
                params,
                store,
                entry,
                seg,
                rho,
                &grid_indices,
            );
            for (idx_in_group, reply) in replies.into_iter().enumerate() {
                let pos = items[idx_in_group].0;
                if shares_by_pos[pos].len() >= params.need {
                    continue;
                }
                let Some(value) = reply else {
                    continue;
                };
                shares_by_pos[pos].push(ShardShare {
                    registry_index: reg_idx,
                    value: Some(value),
                });
                if shares_by_pos[pos].len() == params.need {
                    remaining -= 1;
                }
            }
        }
    }
    let mut values = Vec::with_capacity(idxs.len());
    for (pos, _idx) in idxs.iter().enumerate() {
        let value = decode_from_shares_at_z_with_cache(
            params,
            z_evals[pos],
            &shares_by_pos[pos],
            decode_cache,
        )
        .map_err(|_| ReadError::NotEnoughShards)?;
        values.push(encode_value(value));
    }
    Ok(values)
}

/// Read multiple indices from the coded store using the raw FFT-grid layout.
pub fn read_batch_from_store_raw(
    params: &CodedArrayParams,
    store: &CodedStore,
    idxs: &[Index],
    decode_cache: Option<&DecodeCache>,
) -> Result<Vec<Value256>, ReadError> {
    if idxs.is_empty() {
        return Ok(Vec::new());
    }
    let b = public_values_per_segment(params) as u64;
    let mut by_seg: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    let mut z_evals = vec![fp_zero(); idxs.len()];
    for (pos, &idx) in idxs.iter().enumerate() {
        let seg = usize::try_from(idx / b).map_err(|_| ReadError::InvalidIndex)?;
        let j = usize::try_from(idx % b).map_err(|_| ReadError::InvalidIndex)?;
        by_seg.entry(seg).or_default().push((pos, j));
    }
    let mut shares_by_pos: Vec<Vec<ShardShare>> = vec![Vec::new(); idxs.len()];
    for (seg, items) in by_seg {
        if items.is_empty() {
            continue;
        }
        let mut remaining = items.len();
        let rho = segment_domain_rho(params, seg);
        let grid_indices = items
            .iter()
            .map(|&(pos, j)| {
                let alpha = rho * params.grid_point(j);
                z_evals[pos] = params.column_coordinate(alpha);
                j
            })
            .collect::<Vec<_>>();
        let mut seen_keys: HashSet<ShardKey> = HashSet::new();
        for (reg_idx, entry) in params.shard_registry.iter().copied().enumerate() {
            if remaining == 0 {
                break;
            }
            if !seen_keys.insert(entry.key) {
                continue;
            }
            let replies = shard_value_from_store_batch_on_grid_coset(
                params,
                store,
                entry,
                seg,
                rho,
                &grid_indices,
            );
            for (idx_in_group, reply) in replies.into_iter().enumerate() {
                let pos = items[idx_in_group].0;
                if shares_by_pos[pos].len() >= params.need {
                    continue;
                }
                let Some(value) = reply else {
                    continue;
                };
                shares_by_pos[pos].push(ShardShare {
                    registry_index: reg_idx,
                    value: Some(value),
                });
                if shares_by_pos[pos].len() == params.need {
                    remaining -= 1;
                }
            }
        }
    }
    let mut values = Vec::with_capacity(idxs.len());
    for (pos, _idx) in idxs.iter().enumerate() {
        let value = decode_from_shares_at_z_with_cache(
            params,
            z_evals[pos],
            &shares_by_pos[pos],
            decode_cache,
        )
        .map_err(|_| ReadError::NotEnoughShards)?;
        values.push(encode_value(value));
    }
    Ok(values)
}

/// Legacy alias for the single-store type.
pub type UnauthStorage = UnauthStore;

#[cfg(test)]
mod tests {
    use super::*;

    fn optimized_test_params() -> CodedArrayParams {
        let segment_len = 16;
        let s = 8;
        let shard_count = 4;
        let shard_registry = build_shard_registry_deterministic(shard_count, s, segment_len);
        build_coded_params(&CodedArrayConfig {
            segment_len,
            s,
            shard_count,
            shard_registry,
            party_registry_indices: vec![(0..shard_count).collect()],
            seed: 99,
        })
    }

    fn populated_store(params: CodedArrayParams, len: Index) -> UnauthStore {
        let mut store = UnauthStore::new(params);
        store.extend(len).expect("extend optimized test store");
        let deltas = (0..len)
            .map(|idx| (idx, Fp::from(idx + 1)))
            .collect::<Vec<_>>();
        store
            .apply_deltas(&deltas, DeltaMode::Aggregate)
            .expect("populate optimized test store");
        store
    }

    fn segment_responses_one_shard_per_helper(
        store: &UnauthStore,
        request: &UnauthOptimizedReadRequest,
    ) -> Vec<UnauthOptimizedReadResponse> {
        let response = handle_optimized_read_request(store, request);
        let mut responses = Vec::new();
        for contribution in response.objects {
            let UnauthReadObjectContribution::Segment { object_pos, shards } = contribution else {
                panic!("expected only segment contributions in optimized segment test");
            };
            for (helper_index, shard) in shards.into_iter().enumerate() {
                responses.push(UnauthOptimizedReadResponse {
                    helper_index: Some(helper_index),
                    objects: vec![UnauthReadObjectContribution::Segment {
                        object_pos,
                        shards: vec![shard],
                    }],
                });
            }
        }
        responses
    }

    #[test]
    fn optimized_planner_sorts_dedups_and_selects_segments_and_points() {
        let params = optimized_test_params();
        let len = 48;
        let block_indices = aligned_block_indices(&params, 2, len)
            .into_iter()
            .flatten()
            .take(params.rows_per_shard)
            .collect::<Vec<_>>();
        let mut indices = (0..params.s as u64).collect::<Vec<_>>();
        indices.extend(block_indices.iter().copied());
        indices.extend([33, 35, 33, 1_000]);

        let plan = plan_optimized_read_indices(&params, len, &indices, params.need, None);

        let mut expected_requested = indices;
        expected_requested.sort_unstable();
        expected_requested.dedup();
        assert_eq!(plan.requested_indices, expected_requested);
        assert_eq!(plan.cover.segments, vec![0]);
        assert!(plan.cover.blocks.is_empty());
        let mut expected_points = block_indices;
        expected_points.extend([33, 35]);
        expected_points.sort_unstable();
        expected_points.dedup();
        assert_eq!(plan.cover.points, expected_points);
        assert_eq!(plan.request.objects.len(), 1 + plan.cover.points.len());
    }

    #[test]
    fn optimized_correcting_planner_keeps_segment_point_cover() {
        let params = optimized_test_params();
        let len = 48;
        let block_indices = aligned_block_indices(&params, 2, len)
            .into_iter()
            .flatten()
            .take(params.rows_per_shard)
            .collect::<Vec<_>>();

        let share_target = error_correcting_share_target(params.need, 1);
        let plan = plan_optimized_read_indices(&params, len, &block_indices, share_target, Some(1));

        assert!(plan.cover.segments.is_empty());
        assert!(plan.cover.blocks.is_empty());
        assert_eq!(plan.cover.points, block_indices);
        assert_eq!(plan.block_row_target, params.s + 2);
    }

    #[test]
    fn optimized_handle_and_decode_matches_repeated_store_reads() {
        let params = optimized_test_params();
        let len = 48;
        let store = populated_store(params.clone(), len);
        let block_indices = aligned_block_indices(&params, 2, len)
            .into_iter()
            .flatten()
            .take(params.rows_per_shard)
            .collect::<Vec<_>>();
        let mut indices = (0..params.s as u64).collect::<Vec<_>>();
        indices.extend(block_indices);
        indices.extend([33, 35, 1_000]);

        let plan = plan_optimized_read_indices(&params, len, &indices, params.need, None);
        let response = handle_optimized_read_request(&store, &plan.request);

        assert!(optimized_read_satisfied(&plan, &[response.clone()]));
        let optimized = decode_optimized_read_plan(&params, &plan, &[response]).expect("decode");
        let expected = plan
            .requested_indices
            .iter()
            .map(|&idx| {
                if idx < len {
                    store.read(idx).expect("read expected optimized value")
                } else {
                    encode_value(fp_zero())
                }
            })
            .collect::<Vec<_>>();

        assert_eq!(optimized, expected);
    }

    #[test]
    fn optimized_segment_bulk_decode_matches_scaled_segment_reads() {
        let params = optimized_test_params();
        let len = 48;
        let store = populated_store(params.clone(), len);
        let indices = (params.n_grid as u64..2 * params.n_grid as u64).collect::<Vec<_>>();

        let plan = plan_optimized_read_indices(&params, len, &indices, params.need, None);
        assert_eq!(plan.cover.segments, vec![1]);
        assert!(plan.cover.blocks.is_empty());
        assert!(plan.cover.points.is_empty());

        let response = handle_optimized_read_request(&store, &plan.request);
        assert!(optimized_read_satisfied(&plan, &[response.clone()]));
        let optimized = decode_optimized_read_plan(&params, &plan, &[response]).expect("decode");
        let expected = plan
            .requested_indices
            .iter()
            .map(|&idx| store.read(idx).expect("read expected scaled segment value"))
            .collect::<Vec<_>>();

        assert_eq!(optimized, expected);
    }

    #[test]
    fn optimized_correcting_segment_decode_recovers_corrupt_helper() {
        let params = optimized_test_params();
        let len = 48;
        let store = populated_store(params.clone(), len);
        let indices = (params.n_grid as u64..2 * params.n_grid as u64).collect::<Vec<_>>();
        let share_target = error_correcting_share_target(params.need, 1);
        let plan = plan_optimized_read_indices(&params, len, &indices, share_target, Some(1));
        assert_eq!(plan.cover.segments, vec![1]);

        let mut responses = segment_responses_one_shard_per_helper(&store, &plan.request);
        let UnauthReadObjectContribution::Segment { shards, .. } = responses[0]
            .objects
            .first_mut()
            .expect("segment contribution")
        else {
            panic!("expected segment contribution");
        };
        let shard = shards.first_mut().expect("first shard");
        let mut delta = fp_one();
        for value in &mut shard.values {
            *value += delta;
            delta += fp_one();
        }

        assert!(optimized_read_satisfied(&plan, &responses));
        let optimized = decode_optimized_read_plan(&params, &plan, &responses).expect("decode");
        let expected = plan
            .requested_indices
            .iter()
            .map(|&idx| store.read(idx).expect("read expected scaled segment value"))
            .collect::<Vec<_>>();
        assert_eq!(optimized, expected);

        #[cfg(feature = "bench")]
        {
            let no_filter =
                decode_optimized_read_plan_no_filter(&params, &plan, &responses).expect("decode");
            assert_eq!(no_filter, expected);
        }
    }

    #[test]
    fn optimized_candidate_segment_decode_uses_threshold_helpers_only() {
        let params = optimized_test_params();
        let len = 48;
        let store = populated_store(params.clone(), len);
        let indices = (params.n_grid as u64..2 * params.n_grid as u64).collect::<Vec<_>>();
        let share_target = error_correcting_share_target(params.need, 1);
        let plan = plan_optimized_read_indices(&params, len, &indices, share_target, Some(1));
        assert_eq!(plan.cover.segments, vec![1]);

        let mut responses = segment_responses_one_shard_per_helper(&store, &plan.request);
        let corrupted_helper_index = params.need;
        let UnauthReadObjectContribution::Segment { shards, .. } = responses
            [corrupted_helper_index]
            .objects
            .first_mut()
            .expect("segment contribution")
        else {
            panic!("expected segment contribution");
        };
        for value in &mut shards[0].values {
            *value += fp_one();
        }

        let (_, candidate) = decode_optimized_read_object_candidate_covering_index(
            &params,
            &plan,
            &responses,
            indices[0],
            None,
            &DecodeCache::new(),
        )
        .expect("candidate decode");
        let expected = plan
            .requested_indices
            .iter()
            .map(|&idx| store.read(idx).expect("read expected scaled segment value"))
            .collect::<Vec<_>>();
        let candidate_values = candidate
            .values
            .iter()
            .map(|(_, value)| *value)
            .collect::<Vec<_>>();

        assert_eq!(candidate_values, expected);
        assert_eq!(
            candidate.provenance.registry_indices,
            (0..params.need).collect::<HashSet<_>>()
        );
        assert_eq!(
            candidate.provenance.helper_indices,
            (0..params.need).collect::<HashSet<_>>()
        );
    }
}
