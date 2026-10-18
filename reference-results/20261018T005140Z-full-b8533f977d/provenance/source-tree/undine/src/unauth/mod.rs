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
pub mod layout;
pub mod service;
mod stitched_read;

use crate::coded_array::axis::{
    axis_selectors, coset_lagrange_basis_at, dot, invert_selectors, segment_axis_indices,
};
use crate::coded_array::params::segment_cells_from_values;
pub use crate::coded_array::params::{
    CodedArrayConfig, CodedArrayParams, PaperGeometry, PaperGeometryError, SegmentAxes,
    SetupTimings, ShardKey, ShardRegEntry, build_coded_params, build_shard_registry_deterministic,
    compute_initial_rows, compute_initial_shards, derive_paper_geometry, validate_coded_config,
};
pub use crate::coded_array::reconstruct::{
    DecodeCache, ReadError as CodedReadError, decode_segment_value_with_cache,
};
pub use crate::coded_array::rs::{
    BarycentricWeights, barycentric_eval, barycentric_eval_pair_many, precompute_barycentric,
};
use crate::coded_array::segment::{
    encode_auxiliary_row, encode_primary_column, evaluate_partial_stitch, stitch_auxiliary_row,
    stitch_primary_column, unstitch_auxiliary_rows, unstitch_auxiliary_rows_from_leading,
    unstitch_primary_columns, unstitch_primary_columns_from_leading,
};
pub use crate::coded_array::shard_maths::{
    DeltaMode, apply_deltas_to_segment, apply_deltas_to_segment_with_mode,
    apply_row_deltas_to_segment, shard_value_from_store, shard_value_from_store_batch,
    shard_value_from_vector, shard_value_from_vector_batch,
};
use crate::coded_array::shard_maths::{row_value_at_column, shard_value_at_row_label};
pub use crate::coded_array::store::{CodedStore, RowKey};
use crate::field::{Fp, batch_invert, fp_one, fp_zero};
use crate::shares::{RowShare, SegmentPrimaryShard, ShardShare};
use crate::value::{Value256, decode_value, encode_value};
pub use api::{LogicalValueFamilyKey, UnauthArray, UnauthArrayParams, UnauthDecodeParams};
pub use layout::{
    SemanticBlockPosition, SemanticPosition, aligned_block_indices, block_alphas_for_segment,
    block_grid_indices, block_in_segment_from_column_coordinate,
    logical_block_to_column_coordinate_for_segment, logical_blocks_per_segment,
    logical_offset_to_column_coordinate_for_segment, logical_offset_to_grid_index,
    public_index_to_segment, public_values_per_segment, semantic_block_position, semantic_position,
};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;
use std::time::Instant;
pub use stitched_read::{DefaultCellFn, handle_optimized_read_request_with_defaults};

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

/// Errors returned when doubling or halving the threshold.
#[derive(Clone, Debug)]
pub enum MigrateError {
    /// The target threshold is not supported by the field or the layout ladder.
    InvalidTargetKappa,
    /// The store is at the base level of its layout ladder and has no source
    /// segments to return to.
    AtBaseLevel,
    InvalidClaim,
    /// Supplied source fragments do not match the store's claims or widths.
    InvalidSourceFragments,
    /// Source fragments are missing for some stitched segment.
    MissingSourceFragments,
    MissingLocalShards,
    MissingLocalRows,
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

/// One local claim bundle: a primary column plus its one auxiliary row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClaimBundle {
    pub registry_index: usize,
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
///
/// `kappa` is the threshold the client reads at. Object identifiers are
/// positions in the layout of that threshold, and a store at a lower
/// threshold answers by stitching on the fly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnauthOptimizedReadRequest {
    pub kappa: usize,
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
        request: UnauthOptimizedReadRequest {
            kappa: params.need,
            objects,
        },
        cover,
    }
}

/// Serve the ordered objects in an optimized unauth read request from a
/// store whose inactive cells are zero.
pub fn handle_optimized_read_request(
    store: &UnauthStore,
    request: &UnauthOptimizedReadRequest,
) -> UnauthOptimizedReadResponse {
    handle_optimized_read_request_with_defaults(store, request, None)
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
    decode_segment_object_values_bulk_honest(
        params,
        segment,
        requested_indices,
        shards,
        decode_cache,
    )
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
    match max_errors {
        None | Some(0) => decode_segment_object_values_bulk_honest(
            params,
            segment,
            requested_indices,
            shards,
            decode_cache,
        ),
        Some(max_errors) => decode_segment_object_values_bulk_correcting(
            params,
            segment,
            requested_indices,
            shards,
            max_errors,
            bad_registry_filter,
        ),
    }
}

/// Decode one segment from oversampled primary columns while correcting up
/// to `max_errors` bad helper columns.
///
/// For every W-axis index `j`, the helpers' entries `L_C[j]` are samples of
/// the degree-`< kappa` polynomial `P(Z, eta_j)` at their column identifiers.
/// A bad helper corrupts the same sample position for every `j`, so bad
/// positions are located once on random linear combinations of the `kappa`
/// codewords and the remaining positions are decoded exactly.
fn decode_segment_object_values_bulk_correcting(
    params: &CodedArrayParams,
    segment: usize,
    requested_indices: &[Index],
    shards: &[SegmentPrimaryShard],
    max_errors: usize,
    mut bad_registry_filter: Option<&mut HashSet<usize>>,
) -> Result<Vec<Value256>, CodedReadError> {
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

    let shared_bad_sample_positions =
        detect_bad_segment_sample_positions(params, &selected, max_errors);
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

    let z_targets = segment_z_axis_points(params, segment);
    let grid_values = match decode_segment_grid_exact_with_known_bad_positions(
        params,
        &selected,
        &z_targets,
        &shared_bad_sample_positions,
        bad_registry_filter.as_ref().map(|filter| &**filter),
    ) {
        Some(grid_values) => grid_values,
        None => {
            let mut grid_values = vec![fp_zero(); params.n_grid];
            for w_index in 0..params.s {
                let mut points = Vec::with_capacity(selected.len());
                let mut point_registry_indices = Vec::with_capacity(selected.len());
                for (sample_pos, (registry_index, entry, values)) in selected.iter().enumerate() {
                    if shared_bad_sample_positions.contains(&sample_pos) {
                        continue;
                    }
                    if bad_registry_filter
                        .as_ref()
                        .is_some_and(|filter| filter.contains(registry_index))
                    {
                        continue;
                    }
                    points.push((entry.z, values[w_index]));
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
                        for (sample_pos, (registry_index, entry, values)) in
                            selected.iter().enumerate()
                        {
                            if bad_registry_filter
                                .as_ref()
                                .is_some_and(|filter| filter.contains(registry_index))
                                && !shared_bad_sample_positions.contains(&sample_pos)
                            {
                                continue;
                            }
                            fallback_points.push((entry.z, values[w_index]));
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
                let values_on_z_axis =
                    eval_coefficients_on_points(params, &decoded.coefficients, &z_targets);
                for (z_index, value) in values_on_z_axis.into_iter().enumerate() {
                    grid_values[params.grid_index_for_axes(z_index, w_index)] = value;
                }
            }
            grid_values
        }
    };

    requested_segment_values_from_grid(params, segment, requested_indices, &grid_values)
}

/// Return the Z-axis points of one segment in axis order.
pub(crate) fn segment_z_axis_points(params: &CodedArrayParams, segment: usize) -> Vec<Fp> {
    let rho_z = params.segment_axes(segment).rho_z;
    params.axis.iter().map(|&point| rho_z * point).collect()
}

/// Interpolate every cell of one segment from `need` primary columns.
///
/// `matrix` maps samples at the helpers' column identifiers to values on the
/// segment's Z-axis; it is shared by all `kappa` W-axis indices.
fn segment_grid_from_columns(
    params: &CodedArrayParams,
    columns: &[&[Fp]],
    matrix: &[Fp],
) -> Vec<Fp> {
    debug_assert_eq!(columns.len(), params.need);
    let mut grid_values = vec![fp_zero(); params.n_grid];
    for (z_index, cells) in grid_values.chunks_exact_mut(params.s).enumerate() {
        let factors = &matrix[z_index * params.need..(z_index + 1) * params.need];
        for (&factor, column) in factors.iter().zip(columns.iter()) {
            for (cell, &sample) in cells.iter_mut().zip(column.iter()) {
                *cell += factor * sample;
            }
        }
    }
    grid_values
}

fn decode_segment_grid_exact_with_known_bad_positions(
    params: &CodedArrayParams,
    selected: &[(usize, ShardRegEntry, &[Fp])],
    z_targets: &[Fp],
    bad_sample_positions: &HashSet<usize>,
    bad_registry_filter: Option<&HashSet<usize>>,
) -> Option<Vec<Fp>> {
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
    let weights = precompute_barycentric(&basis_z);
    let basis_columns = basis_positions
        .iter()
        .map(|&sample_pos| selected[sample_pos].2)
        .collect::<Vec<_>>();

    // Every remaining good helper must agree with the interpolant of the
    // basis helpers on all `kappa` codewords.
    let check_positions = &good_positions[params.need..];
    if !check_positions.is_empty() {
        let check_z = check_positions
            .iter()
            .map(|&sample_pos| selected[sample_pos].1.z)
            .collect::<Vec<_>>();
        let check_matrix = barycentric_eval_matrix(&weights, &check_z);
        for (check_pos, &sample_pos) in check_positions.iter().enumerate() {
            let factors = &check_matrix[check_pos * params.need..(check_pos + 1) * params.need];
            let mut expected = vec![fp_zero(); params.s];
            for (&factor, column) in factors.iter().zip(basis_columns.iter()) {
                for (value, &sample) in expected.iter_mut().zip(column.iter()) {
                    *value += factor * sample;
                }
            }
            if expected.as_slice() != selected[sample_pos].2 {
                return None;
            }
        }
    }

    let matrix = barycentric_eval_matrix(&weights, z_targets);
    Some(segment_grid_from_columns(params, &basis_columns, &matrix))
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
    max_errors: usize,
) -> Vec<usize> {
    if max_errors == 0 || selected.len() < params.need + 2 * max_errors {
        return Vec::new();
    }

    let challenges = [Fp::from(17u64), Fp::from(257u64)];
    let mut bad_positions = HashSet::new();
    for challenge in challenges {
        let mut points = Vec::with_capacity(selected.len());
        for (_, entry, values) in selected.iter() {
            let mut acc = fp_zero();
            for &value in values.iter().rev() {
                acc = acc * challenge + value;
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

/// Decode one segment from `need` honest primary columns.
///
/// For every W-axis index `j`, the helpers' entries `L_C[j]` are samples of
/// `P(Z, eta_j)` at their column identifiers; one shared interpolation matrix
/// maps them to the segment's Z-axis.
fn decode_segment_object_values_bulk_honest(
    params: &CodedArrayParams,
    segment: usize,
    requested_indices: &[Index],
    shards: &[SegmentPrimaryShard],
    decode_cache: &DecodeCache,
) -> Result<Vec<Value256>, CodedReadError> {
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

    let z_values = selected
        .iter()
        .map(|(entry, _)| entry.z)
        .collect::<Vec<_>>();
    let shard_keys = selected
        .iter()
        .map(|(entry, _)| entry.key)
        .collect::<Vec<_>>();
    let columns = selected
        .iter()
        .map(|(_, values)| *values)
        .collect::<Vec<_>>();
    let weights = decode_cache.get_weights_for_keys_and_z(&shard_keys, &z_values);
    let z_targets = segment_z_axis_points(params, segment);
    let matrix = barycentric_eval_matrix(&weights, &z_targets);
    let grid_values = segment_grid_from_columns(params, &columns, &matrix);

    requested_segment_values_from_grid(params, segment, requested_indices, &grid_values)
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
    /// Build the claim bundle for one registry index.
    pub fn exact_for_registry_index(
        params: &CodedArrayParams,
        registry_index: usize,
    ) -> Option<Self> {
        (registry_index < params.shard_registry.len()).then_some(Self { registry_index })
    }

    /// Return the index of the claim's auxiliary row label.
    pub fn row_label_index(&self) -> usize {
        self.registry_index
    }

    /// Return the claim's auxiliary row label index as a list.
    pub fn row_label_indices(&self, params: &CodedArrayParams) -> Option<Vec<usize>> {
        (self.registry_index < params.row_labels.len()).then(|| vec![self.row_label_index()])
    }
}

/// Half of a store's source state for one stitched segment.
///
/// A stitched fragment holds half of the claim's source state. Undoing the
/// stitch needs the other half, which comes in one of two shapes depending on
/// who supplies it. Columns are `(registry_index, values)` and rows are
/// `(row_label_index, values)`, one entry per claim of the store, each with
/// the lower threshold's `kappa / 2` values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceHalfState {
    /// Source state of the diagonal segments `(0, 0)` and `(1, 1)`.
    ///
    /// This is ordinary state under the same claims at the lower threshold:
    /// either still local, or recovered with the repair procedure from
    /// maintainers at the lower threshold.
    Diagonal {
        segment_00: RecoveredClaimSegmentState,
        segment_11: RecoveredClaimSegmentState,
    },
    /// Primary columns of segments `(0, 0)` and `(0, 1)` and auxiliary rows of
    /// segments `(0, 0)` and `(1, 0)`.
    ///
    /// This is what maintainers at the current (higher) threshold supply
    /// through partial-stitch evaluations; see
    /// [`decode_leading_source_half`].
    Leading {
        columns_00: Vec<(usize, Vec<Fp>)>,
        columns_01: Vec<(usize, Vec<Fp>)>,
        rows_00: Vec<(usize, Vec<Fp>)>,
        rows_10: Vec<(usize, Vec<Fp>)>,
    },
}

/// One helper claim's partial-stitch evaluations for one stitched segment.
///
/// With the helper claim at `(C_h, w_h)` and requested targets `(C_k, w_k)`:
/// `row_evals[k] = S_0(C_k, w_h)` is read off the helper's auxiliary row, and
/// `column_evals[k] = S'_0(C_h, w_k)` off its primary column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartialStitchEvals {
    /// Registry index of the helper claim.
    pub registry_index: usize,
    /// `S_0(C_k, w_h)` for every requested column identifier `C_k`.
    pub row_evals: Vec<Fp>,
    /// `S'_0(C_h, w_k)` for every requested row label `w_k`.
    pub column_evals: Vec<Fp>,
}

/// Encoded state of one segment keyed by registry index (columns) and by
/// row-label index (rows).
type SparseSegmentState = (HashMap<usize, Vec<Fp>>, HashMap<usize, Vec<Fp>>);

/// Encoded state reused across segments that carry one logical value family.
///
/// A primary column depends on its segment only through the segment's Z-axis
/// scale, and an auxiliary row only through its W-axis scale. Columns are
/// therefore cached per Z-scale index and rows per W-scale index. Among `N`
/// consecutive segments there are about `sqrt(N)` distinct scales on each
/// axis, so a repeated family is encoded about `sqrt(N)` times instead of
/// `N` times.
struct FamilyEncoding {
    /// Decoded cells of the family, in grid order.
    cells: Vec<Fp>,
    /// All cells are equal; the encoding is then the same for every scale.
    constant: bool,
    /// Primary columns keyed by `(z_scale_index, registry_index)`.
    columns: HashMap<(usize, usize), Vec<Fp>>,
    /// Auxiliary rows keyed by `(w_scale_index, row_label_index)`.
    rows: HashMap<(usize, usize), Vec<Fp>>,
}

impl FamilyEncoding {
    fn new(params: &CodedArrayParams, t_seg: &[Value256]) -> Self {
        Self {
            cells: segment_cells_from_values(params, t_seg),
            constant: segment_values_are_layout_invariant(t_seg),
            columns: HashMap::new(),
            rows: HashMap::new(),
        }
    }
}

type FamilyMaterializationCache = HashMap<LogicalValueFamilyKey, FamilyEncoding>;

/// Lagrange bases of local claims, reused across segments.
///
/// The basis of a column identifier depends on the segment only through its
/// Z-axis scale, and the basis of a row label only through its W-axis scale.
/// Among `N` consecutive segments there are about `sqrt(N)` distinct scales
/// on each axis, so encoding and updating most segments needs no basis
/// computation and no field inversion.
#[derive(Default)]
struct ClaimBasisCache {
    /// `Lambda^Z(C)` keyed by `(registry_index, z_scale_index)`.
    z: HashMap<(usize, usize), Vec<Fp>>,
    /// `Lambda^W(w)` keyed by `(row_label_index, w_scale_index)`.
    w: HashMap<(usize, usize), Vec<Fp>>,
}

impl ClaimBasisCache {
    /// `Lambda^Z(C)` for the claim at `registry_index` on segment `seg`.
    fn z_basis(&mut self, params: &CodedArrayParams, registry_index: usize, seg: usize) -> &[Fp] {
        let (z_scale_index, _) = segment_axis_indices(seg);
        self.z
            .entry((registry_index, z_scale_index))
            .or_insert_with(|| {
                let mut basis = vec![fp_zero(); params.need];
                params.z_basis_at(
                    &params.segment_axes(seg),
                    params.shard_registry[registry_index].z,
                    &mut basis,
                );
                basis
            })
    }

    /// `Lambda^W(w)` for the row label at `row_label_index` on segment `seg`.
    fn w_basis(&mut self, params: &CodedArrayParams, row_label_index: usize, seg: usize) -> &[Fp] {
        let (_, w_scale_index) = segment_axis_indices(seg);
        self.w
            .entry((row_label_index, w_scale_index))
            .or_insert_with(|| {
                let mut basis = vec![fp_zero(); params.need];
                params.w_basis_at(
                    &params.segment_axes(seg),
                    params.row_labels[row_label_index],
                    &mut basis,
                );
                basis
            })
    }

    fn clear(&mut self) {
        self.z.clear();
        self.w.clear();
    }
}

fn segment_values_are_layout_invariant(values: &[Value256]) -> bool {
    values
        .split_first()
        .is_none_or(|(first, rest)| rest.iter().all(|value| value == first))
}

/// Lagrange bases of requested off-axis points, reused across the segments
/// of one helper request that share an axis scale.
#[derive(Default)]
struct RequestBasisCache {
    z: HashMap<(Value256, usize), Vec<Fp>>,
    w: HashMap<(Value256, usize), Vec<Fp>>,
}

impl RequestBasisCache {
    /// `Lambda^Z(x)` on the Z-axis of segment `seg`.
    fn z_basis(&mut self, params: &CodedArrayParams, seg: usize, x: Fp) -> &[Fp] {
        let (z_scale_index, _) = segment_axis_indices(seg);
        self.z
            .entry((encode_value(x), z_scale_index))
            .or_insert_with(|| {
                let mut basis = vec![fp_zero(); params.need];
                params.z_basis_at(&params.segment_axes(seg), x, &mut basis);
                basis
            })
    }

    /// `Lambda^W(x)` on the W-axis of segment `seg`.
    fn w_basis(&mut self, params: &CodedArrayParams, seg: usize, x: Fp) -> &[Fp] {
        let (_, w_scale_index) = segment_axis_indices(seg);
        self.w
            .entry((encode_value(x), w_scale_index))
            .or_insert_with(|| {
                let mut basis = vec![fp_zero(); params.need];
                params.w_basis_at(&params.segment_axes(seg), x, &mut basis);
                basis
            })
    }
}

/// Which axis of a segment an interpolation targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SegmentAxis {
    Z,
    W,
}

/// Interpolation matrices from one fixed set of `kappa` sample points to the
/// axis points of segments.
///
/// The matrix depends on the segment only through the scale of the target
/// axis, so it is computed once per distinct scale and reused: repairing one
/// more segment with a known scale costs `kappa^2` multiplications and no
/// inversion.
pub(crate) struct AxisInterpolationCache {
    axis: SegmentAxis,
    weights: BarycentricWeights,
    matrices: HashMap<usize, Vec<Fp>>,
}

impl AxisInterpolationCache {
    pub(crate) fn new(axis: SegmentAxis, points: &[Fp]) -> Self {
        Self {
            axis,
            weights: precompute_barycentric(points),
            matrices: HashMap::new(),
        }
    }

    /// The sample points this cache interpolates from.
    pub(crate) fn points(&self) -> &[Fp] {
        &self.weights.z
    }

    /// Interpolate `samples` (one per sample point) and return the values on
    /// the target axis of segment `seg`.
    pub(crate) fn values_on_axis(
        &mut self,
        params: &CodedArrayParams,
        seg: usize,
        samples: &[Fp],
    ) -> Vec<Fp> {
        let n = self.weights.z.len();
        debug_assert_eq!(samples.len(), n);
        let (z_scale_index, w_scale_index) = segment_axis_indices(seg);
        let scale_index = match self.axis {
            SegmentAxis::Z => z_scale_index,
            SegmentAxis::W => w_scale_index,
        };
        let axis = self.axis;
        let weights = &self.weights;
        let matrix = self.matrices.entry(scale_index).or_insert_with(|| {
            let targets = match axis {
                SegmentAxis::Z => segment_z_axis_points(params, seg),
                SegmentAxis::W => block_alphas_for_segment(params, seg),
            };
            barycentric_eval_matrix(weights, &targets)
        });
        matrix
            .chunks_exact(n)
            .map(|factors| dot(factors, samples))
            .collect()
    }
}

/// Shard storage for the unauthenticated layer.
pub struct UnauthStore {
    pub params: CodedArrayParams,
    pub coded: CodedStore,
    claims: Option<Vec<ClaimBundle>>,
    len: Index,
    segments: usize,
    /// First public index of the zero padding left by a stitch, if any. From
    /// there to the end of the allocated segments the cells hold zero rather
    /// than the values an extension writes; the next extension that reaches
    /// the padding fills it in.
    zero_padding_from: Option<Index>,
    family_materialization_cache: FamilyMaterializationCache,
    claim_bases: ClaimBasisCache,
}

impl Clone for UnauthStore {
    fn clone(&self) -> Self {
        Self {
            params: self.params.clone(),
            coded: self.coded.clone(),
            claims: self.claims.clone(),
            len: self.len,
            segments: self.segments,
            zero_padding_from: self.zero_padding_from,
            family_materialization_cache: HashMap::new(),
            claim_bases: ClaimBasisCache::default(),
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
            zero_padding_from: None,
            family_materialization_cache: HashMap::new(),
            claim_bases: ClaimBasisCache::default(),
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
            zero_padding_from: None,
            family_materialization_cache: HashMap::new(),
            claim_bases: ClaimBasisCache::default(),
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
        Ok(claim_segment_install_plan_for(&self.params, claims))
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
        let claims = self.local_claims();
        let (shards, rows) = encode_segment_state_for_claims(
            &self.params,
            &mut self.claim_bases,
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
        let end = start
            .checked_add(deltas.len() as u64)
            .ok_or(WriteError::InvalidIndex)?;
        if end > self.len {
            return Err(WriteError::InvalidIndex);
        }
        let sparse_deltas = deltas
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(offset, value)| {
                let delta = decode_value(value);
                (delta != fp_zero()).then_some((start + offset as u64, delta))
            })
            .collect::<Vec<_>>();
        self.apply_deltas(&sparse_deltas, DeltaMode::InterpolateFFT)
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
        let mut boundary_deltas = Vec::new();
        for seg in first_seg..=last_seg {
            let seg_base = seg * b;
            let seg_end = seg_base.checked_add(b).ok_or(WriteError::InvalidIndex)?;
            let seg = usize::try_from(seg).map_err(|_| WriteError::InvalidIndex)?;
            if start <= seg_base && seg_end <= end {
                let segment_values = self.params.materialize_logical_segment(|logical_offset| {
                    values[(seg_base + logical_offset as u64 - start) as usize]
                });
                self.overwrite_segment_from_values(seg, &segment_values)?;
                continue;
            }

            // A closed-form delta costs one multiplication per claim, so a
            // boundary fragment of any size is applied cell by cell.
            for idx in start.max(seg_base)..end.min(seg_end) {
                let new_value = values[(idx - start) as usize];
                let default_value = default_at_index(idx);
                let delta = decode_value(new_value) - decode_value(default_value);
                if delta != fp_zero() {
                    boundary_deltas.push((idx, delta));
                }
            }
        }
        if !boundary_deltas.is_empty() {
            self.apply_deltas(&boundary_deltas, mode)?;
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
        if result.is_ok() {
            // The caller filled this part of the padding.
            if let Some(padding_start) = self.zero_padding_from {
                if start <= padding_start && padding_start < end {
                    self.zero_padding_from = (end < capacity).then_some(end);
                }
            }
        }
        result
    }

    /// Bring the zero padding left by a stitch to the values an extension
    /// writes there, so that growing into it needs no further work.
    ///
    /// `logical_values_for_segment(seg)` returns the values of one segment in
    /// public order.
    fn hydrate_zero_padding<F>(
        &mut self,
        mut logical_values_for_segment: F,
    ) -> Result<(), ExtendError>
    where
        F: FnMut(usize) -> Result<Vec<Value256>, ExtendError>,
    {
        let Some(padding_start) = self.zero_padding_from.take() else {
            return Ok(());
        };
        if self.coded.is_empty() {
            return Ok(());
        }
        let span = public_values_per_segment(&self.params) as u64;
        let capacity = self.segments as u64 * span;
        let first_seg =
            usize::try_from(padding_start / span).map_err(|_| ExtendError::SegmentOverflow)?;
        let mut deltas = Vec::new();
        for seg in first_seg..self.segments {
            let values = logical_values_for_segment(seg)?;
            if values.len() != span as usize {
                return Err(ExtendError::InvalidFamilyValues);
            }
            let seg_base = seg as u64 * span;
            for (offset, value) in values.into_iter().enumerate() {
                let idx = seg_base + offset as u64;
                let delta = decode_value(value);
                if idx >= padding_start && delta != fp_zero() {
                    deltas.push((idx, delta));
                }
            }
        }
        let old_len = self.len;
        self.len = capacity;
        let result = self.apply_deltas(&deltas, DeltaMode::InterpolateFFT);
        self.len = old_len;
        result.map_err(|_| ExtendError::SegmentOverflow)
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
        self.family_materialization_cache.clear();
        self.claim_bases.clear();
    }

    /// Return the claims whose state this store holds.
    ///
    /// A claim-bound store returns its configured claims. Otherwise the claims
    /// are read off the coded state, and an empty store is taken to hold every
    /// registry entry.
    fn local_claims(&self) -> Vec<ClaimBundle> {
        match &self.claims {
            Some(claims) => claims.clone(),
            None => derive_claim_bundles_from_coded_state(&self.params, &self.coded)
                .unwrap_or_else(|| all_claim_bundles(&self.params)),
        }
    }

    /// Restrict a claim-bound store to a deterministic subset of claims.
    ///
    /// This keeps only the target claims and deletes all other primary and row
    /// state. It does not change `params`, `len`, or `segments`.
    pub fn retain_claim_bundles(&mut self, claims: Vec<ClaimBundle>) -> Result<(), MigrateError> {
        if self.claims.is_none() {
            return Err(MigrateError::InvalidClaim);
        }
        for claim in &claims {
            if claim.registry_index >= self.params.shard_registry.len() {
                return Err(MigrateError::InvalidClaim);
            }
        }
        let claims = normalize_claim_bundles(&self.params, claims);
        let mut keep_shards = HashSet::new();
        let mut keep_rows = HashSet::new();
        for claim in &claims {
            let entry = self.params.shard_registry[claim.registry_index];
            let row_key = RowKey::new(entry.key, self.params.row_labels[claim.row_label_index()]);
            if !self.coded.contains_shard(&entry.key) {
                return Err(MigrateError::MissingLocalShards);
            }
            if !self.coded.contains_row(&row_key) {
                return Err(MigrateError::MissingLocalRows);
            }
            keep_shards.insert(entry.key);
            keep_rows.insert(row_key);
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
        let claims = self.extension_claims();
        let zero_segment = default_segment_values(&self.params);
        self.extend_claims_from_segments(new_len, &claims, |_| zero_segment.clone(), None)
    }

    /// Double `kappa` with the local four-segment stitch.
    ///
    /// Every group of four consecutive segments at threshold `kappa` becomes
    /// one segment at threshold `2 kappa`, so the segment size goes from
    /// `kappa^2` to `4 kappa^2`. Each claim keeps its column identifier and
    /// row label and combines its own four fragments entry by entry; nothing
    /// is downloaded. A claim stores `4 kappa` field elements for the stitched
    /// segment in place of `8 kappa` for its four source segments. A partially
    /// filled last group is stitched with zero fragments for its missing
    /// segments.
    ///
    /// The public logical order is preserved: source segment `4b + c` becomes
    /// the `c`-th quarter of stitched segment `b`.
    pub fn kappa_double(&mut self) -> Result<(), MigrateError> {
        let new_params = build_kappa_double_params(&self.params)?;
        let claims = self.local_claims();
        let source_need = self.params.need;
        let target_segments = self.segments.div_ceil(4);
        let mut coded = CodedStore::new();
        reserve_segment_state_for_claims(&new_params, &mut coded, &claims, target_segments);
        let mut column = vec![fp_zero(); new_params.s];
        let mut row = vec![fp_zero(); new_params.row_len()];
        // A selector depends on the stitched segment only through the scale
        // of its axis, so each claim needs it once per distinct scale.
        let mut selectors_z_by_scale: HashMap<usize, (Fp, Fp)> = HashMap::new();
        let mut selectors_w_by_scale: HashMap<usize, (Fp, Fp)> = HashMap::new();
        for claim in &claims {
            let entry = self.params.shard_registry[claim.registry_index];
            let row_label = self.params.row_labels[claim.row_label_index()];
            let row_key = RowKey::new(entry.key, row_label);
            selectors_z_by_scale.clear();
            selectors_w_by_scale.clear();
            for parent in 0..target_segments {
                let (z_scale_index, w_scale_index) = segment_axis_indices(parent);
                let selectors_z = *selectors_z_by_scale
                    .entry(z_scale_index)
                    .or_insert_with(|| {
                        let inv_rho_z = new_params.segment_axes(parent).inv_rho_z;
                        axis_selectors(source_need, inv_rho_z, entry.z)
                    });
                let selectors_w = *selectors_w_by_scale
                    .entry(w_scale_index)
                    .or_insert_with(|| {
                        let inv_rho_w = new_params.segment_axes(parent).inv_rho_w;
                        axis_selectors(source_need, inv_rho_w, row_label)
                    });
                let mut child_columns: [Option<&[Fp]>; 4] = [None; 4];
                let mut child_rows: [Option<&[Fp]>; 4] = [None; 4];
                for position in 0..4 {
                    let child = 4 * parent + position;
                    if child >= self.segments {
                        continue;
                    }
                    child_columns[position] = Some(
                        self.coded
                            .shard_segment(&self.params, &entry.key, child)
                            .ok_or(MigrateError::MissingLocalShards)?,
                    );
                    child_rows[position] = Some(
                        self.coded
                            .row_segment(&self.params, &row_key, child)
                            .ok_or(MigrateError::MissingLocalRows)?,
                    );
                }
                stitch_primary_column(selectors_z, child_columns, &mut column);
                stitch_auxiliary_row(selectors_w, child_rows, &mut row);
                coded.append_shard_segment(&new_params, entry.key, parent, &column);
                coded.append_row_segment(&new_params, row_key, parent, &row);
            }
        }
        let source_capacity = self.segments as u64 * public_values_per_segment(&self.params) as u64;
        let target_capacity =
            target_segments as u64 * public_values_per_segment(&new_params) as u64;
        self.zero_padding_from = self
            .zero_padding_from
            .or((source_capacity < target_capacity).then_some(source_capacity));
        self.params = new_params;
        self.coded = coded;
        self.segments = target_segments;
        self.clear_materialization_caches();
        Ok(())
    }

    /// Halve `kappa` by undoing the four-segment stitch.
    ///
    /// Every segment at threshold `kappa` becomes its four source segments at
    /// threshold `kappa / 2`. `source_halves[b]` holds, for the claims of this
    /// store, the half of the source state of stitched segment `b` that the
    /// stitched fragment does not determine; the other half follows entry by
    /// entry. Source segments that lie entirely beyond the logical length are
    /// padding and are dropped.
    pub fn kappa_halve(&mut self, source_halves: &[SourceHalfState]) -> Result<(), MigrateError> {
        let new_params = build_kappa_halve_params(&self.params)?;
        if source_halves.len() != self.segments {
            return Err(MigrateError::MissingSourceFragments);
        }
        let claims = self.local_claims();
        let plan = claim_segment_install_plan_for(&new_params, &claims);
        let source_need = new_params.need;
        let source_span = public_values_per_segment(&new_params) as u64;
        let target_segments = usize::try_from(self.len.div_ceil(source_span))
            .map_err(|_| MigrateError::InvalidTargetKappa)?
            .min(4 * self.segments);

        let validated = |shards: &[(usize, Vec<Fp>)]| {
            validate_recovered_shards(&plan, shards)
                .map_err(|_| MigrateError::InvalidSourceFragments)
        };
        let validated_rows = |rows: &[(usize, Vec<Fp>)]| {
            validate_recovered_rows(&plan, rows).map_err(|_| MigrateError::InvalidSourceFragments)
        };

        let mut coded = CodedStore::new();
        reserve_segment_state_for_claims(&new_params, &mut coded, &claims, target_segments);
        // Selectors and their inverses, per claim and per axis scale of the
        // stitched segment.
        let mut selectors_z_by_scale: HashMap<(usize, usize), ((Fp, Fp), (Fp, Fp))> =
            HashMap::new();
        let mut selectors_w_by_scale: HashMap<(usize, usize), ((Fp, Fp), (Fp, Fp))> =
            HashMap::new();
        for (parent, source_half) in source_halves.iter().enumerate() {
            let (z_scale_index, w_scale_index) = segment_axis_indices(parent);
            // Known fragments of this stitched segment, by registry index and
            // by row-label index.
            let (diagonal, columns_a, columns_b, rows_a, rows_b) = match source_half {
                SourceHalfState::Diagonal {
                    segment_00,
                    segment_11,
                } => (
                    true,
                    validated(&segment_00.shards)?,
                    validated(&segment_11.shards)?,
                    validated_rows(&segment_00.rows)?,
                    validated_rows(&segment_11.rows)?,
                ),
                SourceHalfState::Leading {
                    columns_00,
                    columns_01,
                    rows_00,
                    rows_10,
                } => (
                    false,
                    validated(columns_00)?,
                    validated(columns_01)?,
                    validated_rows(rows_00)?,
                    validated_rows(rows_10)?,
                ),
            };
            for claim in &claims {
                let registry_index = claim.registry_index;
                let row_label_index = claim.row_label_index();
                let entry = self.params.shard_registry[registry_index];
                let row_label = self.params.row_labels[row_label_index];
                let row_key = RowKey::new(entry.key, row_label);
                let (selectors_z, inverse_selectors_z) = *selectors_z_by_scale
                    .entry((registry_index, z_scale_index))
                    .or_insert_with(|| {
                        let inv_rho_z = self.params.segment_axes(parent).inv_rho_z;
                        let selectors = axis_selectors(source_need, inv_rho_z, entry.z);
                        (selectors, invert_selectors(selectors))
                    });
                let (selectors_w, inverse_selectors_w) = *selectors_w_by_scale
                    .entry((row_label_index, w_scale_index))
                    .or_insert_with(|| {
                        let inv_rho_w = self.params.segment_axes(parent).inv_rho_w;
                        let selectors = axis_selectors(source_need, inv_rho_w, row_label);
                        (selectors, invert_selectors(selectors))
                    });
                let stitched_column = self
                    .coded
                    .shard_segment(&self.params, &entry.key, parent)
                    .ok_or(MigrateError::MissingLocalShards)?;
                let stitched_row = self
                    .coded
                    .row_segment(&self.params, &row_key, parent)
                    .ok_or(MigrateError::MissingLocalRows)?;
                let column_a = columns_a[&registry_index].as_slice();
                let column_b = columns_b[&registry_index].as_slice();
                let row_a = rows_a[&row_label_index].as_slice();
                let row_b = rows_b[&row_label_index].as_slice();
                let (derived_columns, derived_rows);
                let (child_columns, child_rows): ([&[Fp]; 4], [&[Fp]; 4]) = if diagonal {
                    derived_columns = unstitch_primary_columns(
                        selectors_z,
                        inverse_selectors_z,
                        stitched_column,
                        column_a,
                        column_b,
                    );
                    derived_rows = unstitch_auxiliary_rows(
                        selectors_w,
                        inverse_selectors_w,
                        stitched_row,
                        row_a,
                        row_b,
                    );
                    (
                        [column_a, &derived_columns.0, &derived_columns.1, column_b],
                        [row_a, &derived_rows.0, &derived_rows.1, row_b],
                    )
                } else {
                    derived_columns = unstitch_primary_columns_from_leading(
                        selectors_z,
                        inverse_selectors_z,
                        stitched_column,
                        column_a,
                        column_b,
                    );
                    derived_rows = unstitch_auxiliary_rows_from_leading(
                        selectors_w,
                        inverse_selectors_w,
                        stitched_row,
                        row_a,
                        row_b,
                    );
                    (
                        [column_a, column_b, &derived_columns.0, &derived_columns.1],
                        [row_a, &derived_rows.0, row_b, &derived_rows.1],
                    )
                };
                for position in 0..4 {
                    let child = 4 * parent + position;
                    if child >= target_segments {
                        break;
                    }
                    coded.append_shard_segment(
                        &new_params,
                        entry.key,
                        child,
                        child_columns[position],
                    );
                    coded.append_row_segment(&new_params, row_key, child, child_rows[position]);
                }
            }
        }
        let target_capacity =
            target_segments as u64 * public_values_per_segment(&new_params) as u64;
        self.zero_padding_from = self
            .zero_padding_from
            .filter(|&padding_start| padding_start < target_capacity);
        self.params = new_params;
        self.coded = coded;
        self.segments = target_segments;
        self.clear_materialization_caches();
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
        let grid_index = logical_offset_to_grid_index(&self.params, logical_offset);
        self.read_segment_cell(seg, grid_index)
    }

    /// Read a value using the raw grid layout.
    ///
    /// Raw reads address allocated segment storage, not the public logical
    /// length. This matters after layout permutations, where a live partial
    /// tail segment can place active cells at high raw grid indices.
    pub fn read_raw(&self, i: Index) -> Result<Value, CodedReadError> {
        let (seg, grid_index) = self.index_to_segment(i);
        if seg >= self.segments {
            return Ok(encode_value(fp_zero()));
        }
        self.read_segment_cell(seg, grid_index)
    }

    fn read_segment_cell(&self, seg: usize, grid_index: usize) -> Result<Value, CodedReadError> {
        let z_index = self.params.z_index_for_grid_index(grid_index);
        let w_index = self.params.w_index_for_grid_index(grid_index);
        let z_eval = self.params.segment_axes(seg).rho_z * self.params.axis[z_index];
        let shares = self.collect_shard_replies(seg, w_index);
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
        for (seg, deltas) in segments {
            self.apply_segment_grid_deltas(seg, &deltas, mode)?;
        }
        Ok(())
    }

    /// Apply `(grid_index, delta)` updates to one segment: every local primary
    /// column and every local auxiliary row changes in one entry per delta.
    ///
    /// A delta `Delta` at cell `(z_i, eta_j)` adds `Delta * Lambda^Z_i(C)` to
    /// entry `j` of the column of `C` and `Delta * Lambda^W_j(w)` to entry `i`
    /// of the row of `w`. The bases come from the per-scale cache, so the
    /// steady-state cost is two multiplications per delta and claim.
    fn apply_segment_grid_deltas(
        &mut self,
        seg: u64,
        deltas: &[(usize, Fp)],
        _mode: DeltaMode,
    ) -> Result<(), WriteError> {
        let seg = usize::try_from(seg).map_err(|_| WriteError::InvalidIndex)?;
        if seg >= self.segments {
            return Err(WriteError::InvalidIndex);
        }
        for &(grid_index, _) in deltas {
            if grid_index >= self.params.n_grid {
                return Err(WriteError::InvalidIndex);
            }
        }
        let params = &self.params;
        let bases = &mut self.claim_bases;
        let need = params.need;
        self.coded
            .for_each_shard_segment_mut(params, seg, |shard_key, column| {
                let registry_index = params
                    .shard_registry_by_key
                    .get(shard_key)
                    .copied()
                    .expect("missing shard entry");
                let basis = bases.z_basis(params, registry_index, seg);
                for &(grid_index, delta) in deltas {
                    column[grid_index % need] += delta * basis[grid_index / need];
                }
            });
        self.coded
            .for_each_row_segment_mut(params, seg, |row_key, row| {
                let row_label_index = params
                    .row_label_index_for_bytes(row_key.row_label_bytes)
                    .expect("missing row label");
                let basis = bases.w_basis(params, row_label_index, seg);
                for &(grid_index, delta) in deltas {
                    row[grid_index / need] += delta * basis[grid_index % need];
                }
            });
        Ok(())
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
        let mut grid_deltas = Vec::with_capacity(deltas.len());
        for &(offset_in_block, delta) in deltas {
            if offset_in_block >= self.params.s {
                return Err(WriteError::InvalidIndex);
            }
            let grid_index = self
                .params
                .grid_index_for_axes(pos.block_in_segment, offset_in_block);
            let logical_offset = self.params.logical_offset_for_grid_index(grid_index);
            if seg_base + logical_offset as u64 >= self.len {
                return Err(WriteError::InvalidIndex);
            }
            grid_deltas.push((grid_index, delta));
        }
        self.apply_segment_grid_deltas(pos.segment as u64, &grid_deltas, mode)
    }

    /// Apply a batch of deltas using the raw grid layout.
    pub fn apply_deltas_raw(
        &mut self,
        deltas: &[(Index, Fp)],
        mode: DeltaMode,
    ) -> Result<(), WriteError> {
        if deltas.is_empty() {
            return Ok(());
        }
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
        for (seg, deltas) in segments {
            self.apply_segment_grid_deltas(seg, &deltas, mode)?;
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

    /// Apply a single delta using the raw grid layout.
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

    /// Apply a batch of deltas using the raw grid layout.
    pub fn write_batch_deltas_raw(
        &mut self,
        deltas: &[(Index, Fp)],
        mode: DeltaMode,
    ) -> Result<(), WriteError> {
        self.apply_deltas_raw(deltas, mode)
    }

    /// Handle a read request by returning shard values for this store.
    pub(crate) fn handle_read(&self, idx: Index) -> Vec<ShardShare> {
        let (seg, logical_offset) = self.index_to_segment(idx);
        if seg >= self.segments {
            return Vec::new();
        }
        let grid_index = logical_offset_to_grid_index(&self.params, logical_offset);
        self.collect_shard_replies(seg, self.params.w_index_for_grid_index(grid_index))
    }

    /// Handle a batch read request by returning shard values for each index.
    pub(crate) fn handle_multi_read(&self, idxs: &[Index]) -> Vec<Vec<ShardShare>> {
        self.handle_read_batch_inner(idxs, true)
    }

    /// Handle an aligned block read request by returning maintained row replies.
    ///
    /// The block's column coordinate is a point of the segment's Z-axis, so
    /// each reply is a stored row entry.
    pub(crate) fn handle_read_block(&self, block: Index) -> Vec<RowShare> {
        let pos = semantic_block_position(&self.params, block);
        if pos.segment >= self.segments {
            return Vec::new();
        }
        collect_active_row_entries(&self.params, &self.coded, pos.segment)
            .into_iter()
            .map(|row| RowShare {
                row_label: row.row_label,
                value: row.row_values[pos.block_in_segment],
            })
            .collect()
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
        self.handle_read_rows_batch_at_z(seg, &[z_eval])
            .into_iter()
            .next()
            .unwrap_or_default()
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
        collect_row_samples_batch_at_z(
            &self.params,
            &self.coded,
            seg,
            z_evals,
            &mut RequestBasisCache::default(),
        )
    }

    /// Return maintained row replies for a batch of segment/column requests.
    pub(crate) fn handle_read_rows_matrix(
        &self,
        requests: &[crate::unauth::service::RowReadRequest],
    ) -> Vec<Vec<Vec<RowShare>>> {
        let mut bases = RequestBasisCache::default();
        requests
            .iter()
            .map(|request| {
                if request.seg >= self.segments {
                    return vec![Vec::new(); request.z_evals.len()];
                }
                collect_row_samples_batch_at_z(
                    &self.params,
                    &self.coded,
                    request.seg,
                    &request.z_evals,
                    &mut bases,
                )
            })
            .collect()
    }

    fn holds_claim(&self, registry_index: usize) -> bool {
        self.claims.as_ref().is_none_or(|claims| {
            claims
                .binary_search_by_key(&registry_index, |claim| claim.registry_index)
                .is_ok()
        })
    }

    /// Compute one repaired primary shard from locally maintained rows.
    pub(crate) fn handle_repair_shard(
        &self,
        seg: usize,
        registry_index: usize,
    ) -> Result<Vec<Fp>, CodedReadError> {
        if !self.holds_claim(registry_index) {
            return Err(CodedReadError::NotEnoughShards);
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
        registry_indices
            .iter()
            .map(|&registry_index| self.handle_repair_shard(seg, registry_index))
            .collect()
    }

    /// Evaluate one batch of owned primary shards at the requested row labels.
    ///
    /// Reply `[c][r]` is `P(C_c, w_r)` for the segment's polynomial: the
    /// sample a requester needs from this helper to repair the auxiliary row
    /// labeled `w_r`.
    pub(crate) fn handle_eval_shards_at_row_labels(
        &self,
        seg: usize,
        col_labels: &[Fp],
        row_labels: &[Fp],
    ) -> Result<Vec<Vec<Fp>>, CodedReadError> {
        self.eval_shards_at_row_labels_with(
            seg,
            col_labels,
            row_labels,
            &mut RequestBasisCache::default(),
        )
    }

    fn eval_shards_at_row_labels_with(
        &self,
        seg: usize,
        col_labels: &[Fp],
        row_labels: &[Fp],
        bases: &mut RequestBasisCache,
    ) -> Result<Vec<Vec<Fp>>, CodedReadError> {
        if seg >= self.segments {
            return Err(CodedReadError::NotEnoughShards);
        }
        let mut out = Vec::with_capacity(col_labels.len());
        for &col_label in col_labels {
            let Some(registry_index) = self
                .params
                .shard_registry
                .iter()
                .position(|entry| entry.z == col_label)
            else {
                return Err(CodedReadError::NotEnoughShards);
            };
            if !self.holds_claim(registry_index) {
                return Err(CodedReadError::NotEnoughShards);
            }
            let entry = self.params.shard_registry[registry_index];
            let repaired;
            let shard_values = match self.coded.shard_segment(&self.params, &entry.key, seg) {
                Some(values) => values,
                None => {
                    repaired = self.repair_primary_shard_from_rows(seg, entry)?;
                    repaired.as_slice()
                }
            };
            out.push(
                row_labels
                    .iter()
                    .map(|&row_label| {
                        shard_value_at_row_label(
                            shard_values,
                            bases.w_basis(&self.params, seg, row_label),
                        )
                    })
                    .collect(),
            );
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
        let mut bases = RequestBasisCache::default();
        let mut out = Vec::with_capacity(segs.len());
        for &seg in segs {
            out.push(self.eval_shards_at_row_labels_with(seg, col_labels, row_labels, &mut bases)?);
        }
        Ok(out)
    }

    /// Evaluate the partial stitches held in this store's fragments, for
    /// requesters that undo a stitch with helpers at the current threshold.
    ///
    /// The requested targets are pairs `(col_labels[k], row_labels[k])`, one
    /// per requester claim. Reply `[s]` lists, for segment `segs[s]`, one
    /// entry per local claim. Each interpolation runs over half of a coset
    /// axis: one shared basis per target and `kappa / 2` multiplications per
    /// local fragment.
    pub(crate) fn handle_eval_partial_stitches(
        &self,
        segs: &[usize],
        col_labels: &[Fp],
        row_labels: &[Fp],
    ) -> Result<Vec<Vec<PartialStitchEvals>>, CodedReadError> {
        if col_labels.len() != row_labels.len() || self.params.need < 2 {
            return Err(CodedReadError::NotEnoughShards);
        }
        let half = self.params.need / 2;
        let half_axis = self
            .params
            .axis
            .iter()
            .copied()
            .step_by(2)
            .collect::<Vec<_>>();
        let inv_half = self.params.inv_need + self.params.inv_need;
        let claims = self.local_claims();
        // Half-axis bases of all targets, per axis scale: segments of the
        // request that share a scale reuse them.
        let mut z_bases_by_scale: HashMap<usize, Vec<Fp>> = HashMap::new();
        let mut w_bases_by_scale: HashMap<usize, Vec<Fp>> = HashMap::new();
        let mut out = Vec::with_capacity(segs.len());
        for &seg in segs {
            if seg >= self.segments {
                return Err(CodedReadError::NotEnoughShards);
            }
            let (z_scale_index, w_scale_index) = segment_axis_indices(seg);
            let z_bases = z_bases_by_scale.entry(z_scale_index).or_insert_with(|| {
                let inv_rho_z = self.params.segment_axes(seg).inv_rho_z;
                let mut bases = vec![fp_zero(); col_labels.len() * half];
                for (basis, &col_label) in bases.chunks_exact_mut(half).zip(col_labels) {
                    coset_lagrange_basis_at(&half_axis, inv_half, inv_rho_z, col_label, basis);
                }
                bases
            });
            let w_bases = w_bases_by_scale.entry(w_scale_index).or_insert_with(|| {
                let inv_rho_w = self.params.segment_axes(seg).inv_rho_w;
                let mut bases = vec![fp_zero(); row_labels.len() * half];
                for (basis, &row_label) in bases.chunks_exact_mut(half).zip(row_labels) {
                    coset_lagrange_basis_at(&half_axis, inv_half, inv_rho_w, row_label, basis);
                }
                bases
            });
            let mut evals = Vec::with_capacity(claims.len());
            for claim in &claims {
                let entry = self.params.shard_registry[claim.registry_index];
                let row_key =
                    RowKey::new(entry.key, self.params.row_labels[claim.row_label_index()]);
                let column = self
                    .coded
                    .shard_segment(&self.params, &entry.key, seg)
                    .ok_or(CodedReadError::NotEnoughShards)?;
                let row = self
                    .coded
                    .row_segment(&self.params, &row_key, seg)
                    .ok_or(CodedReadError::NotEnoughShards)?;
                evals.push(PartialStitchEvals {
                    registry_index: claim.registry_index,
                    row_evals: z_bases
                        .chunks_exact(half)
                        .map(|basis| evaluate_partial_stitch(row, basis))
                        .collect(),
                    column_evals: w_bases
                        .chunks_exact(half)
                        .map(|basis| evaluate_partial_stitch(column, basis))
                        .collect(),
                });
            }
            out.push(evals);
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
        let rows = self.handle_read_rows_at_z(seg, entry.z);
        decode_primary_shard_from_rows(&self.params, seg, &rows)
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

    /// Handle a batch read request using the raw grid layout.
    #[allow(dead_code)]
    pub(crate) fn handle_read_batch_raw(&self, idxs: &[Index]) -> Vec<Vec<ShardShare>> {
        self.handle_read_batch_inner(idxs, false)
    }

    fn handle_read_batch_inner(
        &self,
        idxs: &[Index],
        semantic_layout: bool,
    ) -> Vec<Vec<ShardShare>> {
        let mut results = vec![Vec::new(); idxs.len()];
        let b = public_values_per_segment(&self.params) as u64;
        let mut by_seg: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (pos, &idx) in idxs.iter().enumerate() {
            let Ok(seg) = usize::try_from(idx / b) else {
                continue;
            };
            if seg >= self.segments {
                continue;
            }
            let offset = (idx % b) as usize;
            let grid_index = if semantic_layout {
                logical_offset_to_grid_index(&self.params, offset)
            } else {
                offset
            };
            by_seg
                .entry(seg)
                .or_default()
                .push((pos, self.params.w_index_for_grid_index(grid_index)));
        }
        for (seg, items) in by_seg {
            let local_shards = collect_local_shard_entries(&self.params, &self.coded, Some(seg));
            for &(pos, _) in &items {
                results[pos].reserve(local_shards.len());
            }
            for (registry_index, entry) in local_shards {
                let Some(column) = self.coded.shard_segment(&self.params, &entry.key, seg) else {
                    continue;
                };
                for &(pos, w_index) in &items {
                    results[pos].push(ShardShare {
                        registry_index,
                        value: Some(column[w_index]),
                    });
                }
            }
        }
        results
    }

    /// Return every local primary column's entry at one W-axis index.
    fn collect_shard_replies(&self, seg: usize, w_index: usize) -> Vec<ShardShare> {
        let local_shards = collect_local_shard_entries(&self.params, &self.coded, Some(seg));
        let mut shares = Vec::with_capacity(local_shards.len());
        for (registry_index, entry) in local_shards {
            if let Some(column) = self.coded.shard_segment(&self.params, &entry.key, seg) {
                shares.push(ShardShare {
                    registry_index,
                    value: Some(column[w_index]),
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
        value_at_index: F,
        timings: Option<&mut SetupTimings>,
    ) -> Result<(), ExtendError>
    where
        F: FnMut(Index) -> Value256,
    {
        let claims = self.extension_claims();
        self.extend_with_values_for_claims_timed(new_len, &claims, value_at_index, timings)
    }

    /// Append storage up to `new_len`, using caller-provided logical family
    /// keys to reuse materialized coded state without exposing the coding
    /// layout to the caller.
    pub fn extend_with_value_families_timed<FK, FV>(
        &mut self,
        new_len: Index,
        family_key_for_range: FK,
        logical_values_for_range: FV,
        timings: Option<&mut SetupTimings>,
    ) -> Result<(), ExtendError>
    where
        FK: FnMut(Range<Index>) -> LogicalValueFamilyKey,
        FV: FnMut(Range<Index>, &LogicalValueFamilyKey) -> Vec<Value256>,
    {
        let claims = self.extension_claims();
        self.extend_with_value_families_for_claims_timed(
            new_len,
            &claims,
            family_key_for_range,
            logical_values_for_range,
            timings,
        )
    }

    /// Append storage for selected registry indices, deriving logical values by
    /// public index.
    pub fn extend_with_values_filtered_timed<F>(
        &mut self,
        new_len: Index,
        shard_indices: &[usize],
        value_at_index: F,
        timings: Option<&mut SetupTimings>,
    ) -> Result<(), ExtendError>
    where
        F: FnMut(Index) -> Value256,
    {
        let claims = default_claim_bundles_for_indices(&self.params, shard_indices);
        self.extend_with_values_for_claims_timed(new_len, &claims, value_at_index, timings)
    }

    /// Append storage for selected registry indices, using logical family keys
    /// to reuse materialized state for repeated value families.
    pub fn extend_with_value_families_filtered_timed<FK, FV>(
        &mut self,
        new_len: Index,
        shard_indices: &[usize],
        family_key_for_range: FK,
        logical_values_for_range: FV,
        timings: Option<&mut SetupTimings>,
    ) -> Result<(), ExtendError>
    where
        FK: FnMut(Range<Index>) -> LogicalValueFamilyKey,
        FV: FnMut(Range<Index>, &LogicalValueFamilyKey) -> Vec<Value256>,
    {
        let claims = default_claim_bundles_for_indices(&self.params, shard_indices);
        self.extend_with_value_families_for_claims_timed(
            new_len,
            &claims,
            family_key_for_range,
            logical_values_for_range,
            timings,
        )
    }

    /// Append claim-bound storage, deriving logical values by public index.
    pub fn extend_with_values_for_claims_timed<F>(
        &mut self,
        new_len: Index,
        claims: &[ClaimBundle],
        mut value_at_index: F,
        timings: Option<&mut SetupTimings>,
    ) -> Result<(), ExtendError>
    where
        F: FnMut(Index) -> Value256,
    {
        let params = self.params.clone();
        self.extend_claims_from_segments(
            new_len,
            claims,
            |seg| logical_segment_from_index_values(&params, seg, &mut value_at_index),
            timings,
        )
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
        if self
            .zero_padding_from
            .is_some_and(|padding_start| new_len > padding_start)
        {
            let params = self.params.clone();
            self.hydrate_zero_padding(|seg| {
                let range = logical_segment_range(&params, seg)?;
                let family_key = family_key_for_range(range.clone());
                Ok(logical_values_for_range(range, &family_key))
            })?;
        }
        let needed_segments = self.needed_segments_for_len(new_len)?;
        if !claims.is_empty() {
            reserve_segment_state_for_claims(
                &self.params,
                &mut self.coded,
                claims,
                needed_segments.saturating_sub(self.segments),
            );
            for seg in self.segments..needed_segments {
                let range = logical_segment_range(&self.params, seg)?;
                let family_key = family_key_for_range(range.clone());
                if !self.family_materialization_cache.contains_key(&family_key) {
                    let logical_values = logical_values_for_range(range, &family_key);
                    let t_seg = logical_segment_from_logical_values(&self.params, &logical_values)?;
                    self.family_materialization_cache.insert(
                        family_key.clone(),
                        FamilyEncoding::new(&self.params, &t_seg),
                    );
                }
                let encoding = self
                    .family_materialization_cache
                    .get_mut(&family_key)
                    .expect("family encoding was just inserted");
                append_family_segment_state_for_claims(
                    &self.params,
                    &mut self.coded,
                    &mut self.claim_bases,
                    seg,
                    claims,
                    encoding,
                    timings.as_deref_mut(),
                );
            }
        }
        self.segments = self.segments.max(needed_segments);
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
        self.extend_claims_from_segments(new_len, claims, build_segment, None)
    }

    /// Append segments up to the size needed for `new_len`, using a builder and optional timings.
    pub fn extend_with_segments_timed<F>(
        &mut self,
        new_len: Index,
        build_segment: F,
        timings: Option<&mut SetupTimings>,
    ) -> Result<(), ExtendError>
    where
        F: FnMut(usize) -> Vec<Value256>,
    {
        let claims = self.extension_claims();
        self.extend_claims_from_segments(new_len, &claims, build_segment, timings)
    }

    /// Append segments, storing shards only for selected registry indices.
    pub fn extend_with_segments_filtered_timed<F>(
        &mut self,
        new_len: Index,
        shard_indices: &[usize],
        build_segment: F,
        timings: Option<&mut SetupTimings>,
    ) -> Result<(), ExtendError>
    where
        F: FnMut(usize) -> Vec<Value256>,
    {
        let claims = default_claim_bundles_for_indices(&self.params, shard_indices);
        self.extend_claims_from_segments(new_len, &claims, build_segment, timings)
    }

    fn index_to_segment(&self, i: Index) -> (usize, usize) {
        public_index_to_segment(&self.params, i)
    }

    /// Claims that an unfiltered extension materializes: the configured claims
    /// of a claim-bound store, every registry entry otherwise.
    fn extension_claims(&self) -> Vec<ClaimBundle> {
        match &self.claims {
            Some(claims) => claims.clone(),
            None => all_claim_bundles(&self.params),
        }
    }

    fn needed_segments_for_len(&self, new_len: Index) -> Result<usize, ExtendError> {
        let b = public_values_per_segment(&self.params) as u64;
        usize::try_from(new_len.div_ceil(b)).map_err(|_| ExtendError::SegmentOverflow)
    }

    /// Append the segments needed for `new_len`, encoding each from its
    /// grid-order values for the given claims.
    fn extend_claims_from_segments<F>(
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
        if self
            .zero_padding_from
            .is_some_and(|padding_start| new_len > padding_start)
        {
            let params = self.params.clone();
            self.hydrate_zero_padding(|seg| {
                let t_seg = build_segment(seg);
                if t_seg.len() != params.n_grid {
                    return Err(ExtendError::InvalidFamilyValues);
                }
                Ok((0..params.n_grid)
                    .map(|logical_offset| t_seg[params.logical_grid_index(logical_offset)])
                    .collect())
            })?;
        }
        let needed_segments = self.needed_segments_for_len(new_len)?;
        if !claims.is_empty() {
            reserve_segment_state_for_claims(
                &self.params,
                &mut self.coded,
                claims,
                needed_segments.saturating_sub(self.segments),
            );
            for seg in self.segments..needed_segments {
                let t_seg = build_segment(seg);
                let (shards, rows) = encode_segment_state_for_claims(
                    &self.params,
                    &mut self.claim_bases,
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
        }
        self.segments = self.segments.max(needed_segments);
        self.len = new_len;
        Ok(())
    }
}

fn claim_segment_install_plan_for(
    params: &CodedArrayParams,
    claims: &[ClaimBundle],
) -> ClaimSegmentInstallPlan {
    ClaimSegmentInstallPlan {
        expected_shards: claims.iter().map(|claim| claim.registry_index).collect(),
        expected_rows: claims.iter().map(|claim| claim.row_label_index()).collect(),
        expected_shard_len: params.s,
        expected_row_len: params.row_len(),
    }
}

fn validate_recovered_shards(
    plan: &ClaimSegmentInstallPlan,
    repaired_shards: &[(usize, Vec<Fp>)],
) -> Result<HashMap<usize, Vec<Fp>>, InstallClaimStateError> {
    validate_owned_recovered_shards(plan, repaired_shards.to_vec())
}

fn validate_recovered_rows(
    plan: &ClaimSegmentInstallPlan,
    repaired_rows: &[(usize, Vec<Fp>)],
) -> Result<HashMap<usize, Vec<Fp>>, InstallClaimStateError> {
    validate_owned_recovered_rows(plan, repaired_rows.to_vec())
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

/// Encode one segment's grid-order values for the given claims: the primary
/// column and the auxiliary row of each claim.
///
/// # Complexity
/// `2 kappa^2` field multiplications per claim.
fn encode_segment_state_for_claims(
    params: &CodedArrayParams,
    bases: &mut ClaimBasisCache,
    seg: usize,
    t_seg: &[Value256],
    claims: &[ClaimBundle],
    timings: Option<&mut SetupTimings>,
) -> SparseSegmentState {
    if segment_values_are_layout_invariant(t_seg) {
        // A constant segment is the constant polynomial: every column entry
        // and every row entry equals that constant.
        let constant = t_seg.first().copied().map_or_else(fp_zero, decode_value);
        let shards = claims
            .iter()
            .map(|claim| (claim.registry_index, vec![constant; params.s]))
            .collect();
        let rows = claims
            .iter()
            .map(|claim| (claim.row_label_index(), vec![constant; params.row_len()]))
            .collect();
        return (shards, rows);
    }
    let cells = segment_cells_from_values(params, t_seg);
    let start = timings.as_ref().map(|_| Instant::now());
    let shards = claims
        .iter()
        .map(|claim| {
            let mut column = vec![fp_zero(); params.s];
            encode_primary_column(
                &cells,
                bases.z_basis(params, claim.registry_index, seg),
                &mut column,
            );
            (claim.registry_index, column)
        })
        .collect();
    let columns_done = start.map(|start| start.elapsed());
    let rows = claims
        .iter()
        .map(|claim| {
            let mut row = vec![fp_zero(); params.row_len()];
            encode_auxiliary_row(
                &cells,
                bases.w_basis(params, claim.row_label_index(), seg),
                &mut row,
            );
            (claim.row_label_index(), row)
        })
        .collect();
    if let (Some(start), Some(columns_done), Some(timing)) = (start, columns_done, timings) {
        timing.shard_eval += columns_done;
        timing.row_eval += start.elapsed() - columns_done;
    }
    (shards, rows)
}

/// Append one segment of a cached value family for the given claims,
/// encoding only the columns and rows the cache does not hold yet.
fn append_family_segment_state_for_claims(
    params: &CodedArrayParams,
    coded: &mut CodedStore,
    bases: &mut ClaimBasisCache,
    seg: usize,
    claims: &[ClaimBundle],
    encoding: &mut FamilyEncoding,
    mut timings: Option<&mut SetupTimings>,
) {
    let FamilyEncoding {
        cells,
        constant,
        columns,
        rows,
    } = encoding;
    let (z_scale_index, w_scale_index) = if *constant {
        (0, 0)
    } else {
        segment_axis_indices(seg)
    };
    for claim in claims {
        let entry = params.shard_registry[claim.registry_index];
        let column = columns
            .entry((z_scale_index, claim.registry_index))
            .or_insert_with(|| {
                let start = timings.as_ref().map(|_| Instant::now());
                let mut column = vec![fp_zero(); params.s];
                encode_primary_column(
                    cells,
                    bases.z_basis(params, claim.registry_index, seg),
                    &mut column,
                );
                if let (Some(start), Some(timing)) = (start, timings.as_deref_mut()) {
                    timing.shard_eval += start.elapsed();
                }
                column
            });
        coded.append_shard_segment(params, entry.key, seg, column);

        let row_label_index = claim.row_label_index();
        let row = rows
            .entry((w_scale_index, row_label_index))
            .or_insert_with(|| {
                let start = timings.as_ref().map(|_| Instant::now());
                let mut row = vec![fp_zero(); params.row_len()];
                encode_auxiliary_row(cells, bases.w_basis(params, row_label_index, seg), &mut row);
                if let (Some(start), Some(timing)) = (start, timings.as_deref_mut()) {
                    timing.row_eval += start.elapsed();
                }
                row
            });
        let row_key = RowKey::new(entry.key, params.row_labels[row_label_index]);
        coded.append_row_segment(params, row_key, seg, row);
    }
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

        let row_label_idx = claim.row_label_index();
        let row_values = rows
            .get(&row_label_idx)
            .expect("missing selected row values for claim-bound overwrite");
        let row_key = RowKey::new(entry.key, params.row_labels[row_label_idx]);
        coded.overwrite_row_segment(params, row_key, seg, row_values);
    }
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
        let entry = params.shard_registry[claim.registry_index];
        coded.reserve_shard_segments(params, entry.key, additional_segments);
        let row_key = RowKey::new(entry.key, params.row_labels[claim.row_label_index()]);
        coded.reserve_row_segments(params, row_key, additional_segments);
    }
}

/// Sort claims by registry index and drop duplicates.
fn normalize_claim_bundles(
    params: &CodedArrayParams,
    mut claims: Vec<ClaimBundle>,
) -> Vec<ClaimBundle> {
    for claim in &claims {
        assert!(
            claim.registry_index < params.shard_registry.len(),
            "claim registry index {} out of range for {} registry entries",
            claim.registry_index,
            params.shard_registry.len()
        );
    }
    claims.sort_unstable();
    claims.dedup();
    claims
}

fn default_claim_bundles_for_indices(
    params: &CodedArrayParams,
    shard_indices: &[usize],
) -> Vec<ClaimBundle> {
    let claims = shard_indices
        .iter()
        .filter_map(|&idx| ClaimBundle::exact_for_registry_index(params, idx))
        .collect();
    normalize_claim_bundles(params, claims)
}

fn all_claim_bundles(params: &CodedArrayParams) -> Vec<ClaimBundle> {
    (0..params.shard_registry.len())
        .map(|registry_index| ClaimBundle { registry_index })
        .collect()
}

/// Read the held claims off the coded state: one claim per stored primary
/// column. Returns `None` for an empty store.
fn derive_claim_bundles_from_coded_state(
    params: &CodedArrayParams,
    coded: &CodedStore,
) -> Option<Vec<ClaimBundle>> {
    if coded.is_empty() {
        return None;
    }
    let claims = collect_local_shard_entries(params, coded, None)
        .into_iter()
        .map(|(registry_index, _)| ClaimBundle { registry_index })
        .collect::<Vec<_>>();
    (!claims.is_empty()).then_some(claims)
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

        let row_label_idx = claim.row_label_index();
        let row_values = rows
            .get(&row_label_idx)
            .expect("missing selected row values for claim-bound append");
        let row_key = RowKey::new(entry.key, params.row_labels[row_label_idx]);
        coded.append_row_segment(params, row_key, seg, row_values);
    }
}

/// Rebuild the coded params for another threshold on the same layout ladder.
///
/// The registry, the row labels, and the layout base are unchanged: a claim
/// keeps its column identifier and its row label at every level.
pub fn build_params_for_kappa(
    params: &CodedArrayParams,
    kappa: usize,
) -> Result<CodedArrayParams, MigrateError> {
    if derive_paper_geometry(kappa).is_err() {
        return Err(MigrateError::InvalidTargetKappa);
    }
    let mut rebuilt = build_coded_params(&CodedArrayConfig {
        kappa,
        shard_count: params.shard_registry.len(),
        shard_registry: params.shard_registry.clone(),
        party_registry_indices: params.party_registry_indices.clone(),
        seed: 0,
    });
    rebuilt
        .apply_canonical_logical_layout(params.layout_base_n_grid)
        .map_err(|_| MigrateError::InvalidTargetKappa)?;
    Ok(rebuilt)
}

/// Build the coded params one stitch above `params`: threshold `2 kappa` and
/// segment size `4 kappa^2`.
pub fn build_kappa_double_params(
    params: &CodedArrayParams,
) -> Result<CodedArrayParams, MigrateError> {
    let kappa = params
        .need
        .checked_mul(2)
        .ok_or(MigrateError::InvalidTargetKappa)?;
    build_params_for_kappa(params, kappa)
}

/// Build the coded params one stitch below `params`: threshold `kappa / 2`
/// and segment size `kappa^2 / 4`.
///
/// The base level of the layout ladder has no source segments, so halving is
/// defined only above it.
pub fn build_kappa_halve_params(
    params: &CodedArrayParams,
) -> Result<CodedArrayParams, MigrateError> {
    if params.layout_level == 0 {
        return Err(MigrateError::AtBaseLevel);
    }
    build_params_for_kappa(params, params.need / 2)
}

/// Leading source half of one claim for one stitched segment: the source
/// columns of segments `(0, 0)` and `(0, 1)` and the source rows of segments
/// `(0, 0)` and `(1, 0)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeadingSourceFragments {
    pub column_00: Vec<Fp>,
    pub column_01: Vec<Fp>,
    pub row_00: Vec<Fp>,
    pub row_10: Vec<Fp>,
}

/// Coefficient vectors of the Lagrange basis polynomials over `points`:
/// entry `[h * n + k]` is the coefficient of `X^k` in the basis polynomial of
/// point `h`.
///
/// # Complexity
/// `O(n^2)` field multiplications and one batch inversion.
fn interpolation_basis_coefficients(points: &[Fp]) -> Vec<Fp> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    let weights = precompute_barycentric(points);
    let mut root_poly = vec![fp_zero(); n + 1];
    root_poly[0] = fp_one();
    for (degree, &x) in points.iter().enumerate() {
        for k in (0..=degree).rev() {
            let coeff = root_poly[k];
            root_poly[k + 1] += coeff;
            root_poly[k] = -(coeff * x);
        }
    }
    let mut basis = vec![fp_zero(); n * n];
    for (h, &x) in points.iter().enumerate() {
        // Synthetic division of the root polynomial by `(X - x)`, scaled by
        // the barycentric weight of the point.
        let row = &mut basis[h * n..(h + 1) * n];
        row[n - 1] = root_poly[n];
        for k in (1..n).rev() {
            row[k - 1] = root_poly[k] + x * row[k];
        }
        for value in row.iter_mut() {
            *value *= weights.w[h];
        }
    }
    basis
}

/// Decoder of leading source halves for one fixed set of `kappa` helper
/// claims.
///
/// The helpers' row labels and column identifiers are the same for every
/// stitched segment, so the map from their samples to polynomial
/// coefficients is computed once. Decoding one segment then costs
/// `2 kappa^2` multiplications and two size-`kappa` FFTs.
pub struct LeadingSourceDecoder {
    registry_indices: Vec<usize>,
    /// Lagrange basis coefficients over the helpers' row labels.
    row_label_basis: Vec<Fp>,
    /// Lagrange basis coefficients over the helpers' column identifiers.
    column_basis: Vec<Fp>,
}

impl LeadingSourceDecoder {
    /// Build the decoder for the helper claims `registry_indices`, in the
    /// order their evaluations will be supplied. `params` are the parameters
    /// at the stitched threshold; exactly `params.need` distinct helper
    /// claims are required.
    pub fn new(
        params: &CodedArrayParams,
        registry_indices: &[usize],
    ) -> Result<Self, CodedReadError> {
        let mut seen = HashSet::new();
        if registry_indices.len() != params.need
            || registry_indices
                .iter()
                .any(|&idx| idx >= params.shard_registry.len() || !seen.insert(idx))
        {
            return Err(CodedReadError::NotEnoughShards);
        }
        let row_labels = registry_indices
            .iter()
            .map(|&idx| params.row_labels[idx])
            .collect::<Vec<_>>();
        let col_labels = registry_indices
            .iter()
            .map(|&idx| params.shard_registry[idx].z)
            .collect::<Vec<_>>();
        Ok(Self {
            registry_indices: registry_indices.to_vec(),
            row_label_basis: interpolation_basis_coefficients(&row_labels),
            column_basis: interpolation_basis_coefficients(&col_labels),
        })
    }

    /// Return the helper claims this decoder was built for, in order.
    pub fn registry_indices(&self) -> &[usize] {
        &self.registry_indices
    }

    /// Decode the leading source half of stitched segment `seg`.
    ///
    /// `row_evals[h]` is `S_0(C, w_h)` and `column_evals[h]` is
    /// `S'_0(C_h, w)` for helper claim `h` of this decoder.
    pub fn decode(
        &self,
        params: &CodedArrayParams,
        seg: usize,
        row_evals: &[Fp],
        column_evals: &[Fp],
    ) -> Result<LeadingSourceFragments, CodedReadError> {
        let n = self.registry_indices.len();
        if row_evals.len() != n || column_evals.len() != n || n != params.need {
            return Err(CodedReadError::NotEnoughShards);
        }
        let axes = params.segment_axes(seg);
        let values_on_axis = |basis: &[Fp], evals: &[Fp], rho: Fp| {
            // Coefficients of the interpolant, scaled so that one FFT over
            // the normalized axis evaluates it on the segment's axis.
            let mut values = vec![fp_zero(); n];
            for (row, &eval) in basis.chunks_exact(n).zip(evals.iter()) {
                for (value, &coeff) in values.iter_mut().zip(row.iter()) {
                    *value += eval * coeff;
                }
            }
            let mut rho_pow = fp_one();
            for value in values.iter_mut() {
                *value *= rho_pow;
                rho_pow *= rho;
            }
            crate::coded_array::params::fft_in_place(&mut values, params.omega);
            values
        };
        let column_values = values_on_axis(&self.row_label_basis, row_evals, axes.rho_w);
        let row_values = values_on_axis(&self.column_basis, column_evals, axes.rho_z);
        Ok(LeadingSourceFragments {
            column_00: column_values.iter().copied().step_by(2).collect(),
            column_01: column_values.iter().copied().skip(1).step_by(2).collect(),
            row_00: row_values.iter().copied().step_by(2).collect(),
            row_10: row_values.iter().copied().skip(1).step_by(2).collect(),
        })
    }
}

/// Decode one claim's leading source half of stitched segment `seg` from
/// partial-stitch evaluations.
///
/// `params` are the parameters at the stitched threshold `kappa`.
/// `row_samples` are `(w_h, S_0(C, w_h))`: samples of a polynomial of degree
/// `< kappa` in `W` at the helpers' row labels; its values on the two halves
/// of the segment's W-axis are the source columns of segments `(0, 0)` and
/// `(0, 1)`. `column_samples` are `(C_h, S'_0(C_h, w))`: samples of a
/// polynomial of degree `< kappa` in `Z` at the helpers' column identifiers;
/// its values on the two halves of the Z-axis are the source rows of segments
/// `(0, 0)` and `(1, 0)`.
///
/// Each list needs `kappa + 2 * max_errors` samples at distinct points.
pub fn decode_leading_source_half(
    params: &CodedArrayParams,
    seg: usize,
    row_samples: &[(Fp, Fp)],
    column_samples: &[(Fp, Fp)],
    max_errors: usize,
) -> Result<LeadingSourceFragments, CodedReadError> {
    let w_axis = block_alphas_for_segment(params, seg);
    let z_axis = segment_z_axis_points(params, seg);
    let decode_on_axis =
        |samples: &[(Fp, Fp)], axis_points: &[Fp]| -> Result<(Vec<Fp>, Vec<Fp>), CodedReadError> {
            let target = error_correcting_share_target(params.need, max_errors);
            if samples.len() < target {
                return Err(CodedReadError::NotEnoughShards);
            }
            let decoded = crate::coded_array::rs::rs_decode_eval_with_errors(
                &samples[..target],
                params.need,
                axis_points[0],
                max_errors,
            )
            .map_err(|_| CodedReadError::NotEnoughShards)?;
            let values = eval_coefficients_on_points(params, &decoded.coefficients, axis_points);
            Ok((
                values.iter().copied().step_by(2).collect(),
                values.iter().copied().skip(1).step_by(2).collect(),
            ))
        };
    let (column_00, column_01) = decode_on_axis(row_samples, &w_axis)?;
    let (row_00, row_10) = decode_on_axis(column_samples, &z_axis)?;
    Ok(LeadingSourceFragments {
        column_00,
        column_01,
        row_00,
        row_10,
    })
}

struct ActiveRow<'a> {
    row_label_bytes: Value256,
    row_label: Fp,
    row_values: &'a [Fp],
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

/// Evaluate every local auxiliary row of one segment at each column
/// coordinate: reply `[k]` holds `P(z_evals[k], w)` for every local label `w`.
///
/// A coordinate on the segment's Z-axis is a stored row entry. Any other
/// coordinate, such as the column identifier of a claim under repair, costs
/// one basis vector shared by all local rows and by all segments of the
/// request with the same Z-axis scale, and `kappa` multiplications per row.
fn collect_row_samples_batch_at_z(
    params: &CodedArrayParams,
    coded: &CodedStore,
    seg: usize,
    z_evals: &[Fp],
    bases: &mut RequestBasisCache,
) -> Vec<Vec<RowShare>> {
    if coded.row_family_count() == 0 {
        return vec![Vec::new(); z_evals.len()];
    }
    let active_rows = collect_active_row_entries(params, coded, seg);
    let inv_rho_z = params.segment_axes(seg).inv_rho_z;
    z_evals
        .iter()
        .map(|&z_eval| {
            if let Some(z_index) = params.axis_index_of(z_eval * inv_rho_z) {
                return active_rows
                    .iter()
                    .map(|row| RowShare {
                        row_label: row.row_label,
                        value: row.row_values[z_index],
                    })
                    .collect();
            }
            let basis = bases.z_basis(params, seg, z_eval);
            active_rows
                .iter()
                .map(|row| RowShare {
                    row_label: row.row_label,
                    value: row_value_at_column(row.row_values, basis),
                })
                .collect()
        })
        .collect()
}

/// Decode a segment value from shard values.
pub(crate) fn decode_from_shares_at_z<P: UnauthDecodeParams>(
    params: &P,
    z_eval: Fp,
    shares: &[ShardShare],
) -> Result<Fp, CodedReadError> {
    decode_from_shares_at_z_with_cache(params, z_eval, shares, None)
}

/// Decode the value at public index `idx` from oversampled shard values while
/// correcting up to `max_errors` arbitrary bad helper replies.
pub fn decode_from_shares_with_error_correction<P: UnauthDecodeParams>(
    params: &P,
    idx: Index,
    shares: &[ShardShare],
    max_errors: usize,
) -> Result<CorrectedShardDecode, CodedReadError> {
    let z_eval = params.decode_index_eval(idx);
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

/// Interpolate row replies over their row labels and evaluate on
/// `eval_points`.
///
/// The replies are samples `P(z, w_h)` of a polynomial of degree `< kappa` in
/// `W`, one per helper row label `w_h`.
fn decode_slice_from_rows(
    params: &CodedArrayParams,
    rows: &[RowShare],
    eval_points: &[Fp],
) -> Result<Vec<Fp>, CodedReadError> {
    let mut row_labels = Vec::with_capacity(params.s);
    let mut row_values = Vec::with_capacity(params.s);
    let mut seen = HashSet::new();
    for row in rows {
        let label_bytes = encode_value(row.row_label);
        if params.row_label_index_for_bytes(label_bytes).is_none() || !seen.insert(label_bytes) {
            continue;
        }
        row_labels.push(row.row_label);
        row_values.push(row.value);
        if row_values.len() >= params.s {
            break;
        }
    }
    if row_values.len() < params.s {
        return Err(CodedReadError::NotEnoughShards);
    }

    let weights = precompute_barycentric(&row_labels);
    let matrix = barycentric_eval_matrix(&weights, eval_points);
    Ok(matrix
        .chunks_exact(params.s)
        .map(|factors| dot(factors, &row_values))
        .collect())
}

/// Decode one primary column of `seg` from row replies evaluated at its
/// column identifier.
fn decode_primary_shard_from_rows(
    params: &CodedArrayParams,
    seg: usize,
    rows: &[RowShare],
) -> Result<Vec<Fp>, CodedReadError> {
    decode_slice_from_rows(params, rows, &block_alphas_for_segment(params, seg))
}

/// Decode one primary column of `seg` from row replies, reusing the
/// interpolation matrix across segments.
///
/// `cache` holds the matrix for the row labels of the previous call; it is
/// rebuilt when the replies come from a different set of labels.
pub(crate) fn decode_primary_shard_from_rows_cached(
    params: &CodedArrayParams,
    seg: usize,
    rows: &[RowShare],
    cache: &mut Option<AxisInterpolationCache>,
) -> Result<Vec<Fp>, CodedReadError> {
    let mut row_labels = Vec::with_capacity(params.s);
    let mut row_values = Vec::with_capacity(params.s);
    let mut seen = HashSet::new();
    for row in rows {
        let label_bytes = encode_value(row.row_label);
        if params.row_label_index_for_bytes(label_bytes).is_none() || !seen.insert(label_bytes) {
            continue;
        }
        row_labels.push(row.row_label);
        row_values.push(row.value);
        if row_values.len() >= params.s {
            break;
        }
    }
    if row_values.len() < params.s {
        return Err(CodedReadError::NotEnoughShards);
    }
    if cache
        .as_ref()
        .is_none_or(|cache| cache.points() != row_labels.as_slice())
    {
        *cache = Some(AxisInterpolationCache::new(SegmentAxis::W, &row_labels));
    }
    let cache = cache.as_mut().expect("interpolation cache was just set");
    Ok(cache.values_on_axis(params, seg, &row_values))
}

#[allow(dead_code)]
/// Decode one aligned semantic block from maintained row replies.
pub(crate) fn decode_block_from_rows(
    params: &CodedArrayParams,
    block: Index,
    rows: &[RowShare],
) -> Result<Vec<Fp>, CodedReadError> {
    let pos = semantic_block_position(params, block);
    decode_slice_from_rows(params, rows, &block_alphas_for_segment(params, pos.segment))
}

fn decode_block_from_rows_correcting(
    params: &CodedArrayParams,
    block: Index,
    rows: &[RowShare],
    max_errors: usize,
) -> Result<Vec<Fp>, CodedReadError> {
    let pos = semantic_block_position(params, block);
    let eval_points = block_alphas_for_segment(params, pos.segment);
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

/// Evaluate a coefficient vector on `eval_points`.
///
/// When the points are one full coset axis `base * omega^i`, the evaluation
/// is one FFT of size `kappa`; otherwise it is Horner's rule per point.
pub(crate) fn eval_coefficients_on_points(
    params: &CodedArrayParams,
    coefficients: &[Fp],
    eval_points: &[Fp],
) -> Vec<Fp> {
    if eval_points.is_empty() {
        return Vec::new();
    }
    if coefficients.len() <= eval_points.len() && eval_points.len() == params.need {
        let base = eval_points[0];
        let structured = eval_points
            .iter()
            .zip(params.axis.iter())
            .all(|(&eval_point, &axis_point)| eval_point == base * axis_point);
        if structured {
            let mut values = vec![fp_zero(); eval_points.len()];
            let mut base_pow = fp_one();
            for (value, &coeff) in values.iter_mut().zip(coefficients.iter()) {
                *value = coeff * base_pow;
                base_pow *= base;
            }
            crate::coded_array::params::fft_in_place(&mut values, params.omega);
            return values;
        }
    }

    eval_points
        .iter()
        .map(|&eval_point| poly_eval_coefficients(coefficients, eval_point))
        .collect()
}

/// Reconstruct the auxiliary row labeled `row_label` of one segment from the
/// local primary columns.
fn decode_row_from_shards(
    params: &CodedArrayParams,
    coded: &CodedStore,
    seg: usize,
    row_label: Fp,
) -> Result<Vec<Fp>, CodedReadError> {
    let axes = params.segment_axes(seg);
    let mut basis_w = vec![fp_zero(); params.need];
    params.w_basis_at(&axes, row_label, &mut basis_w);
    let mut z_values = Vec::with_capacity(params.need);
    let mut row_values = Vec::with_capacity(params.need);
    for entry in params.shard_registry.iter() {
        let Some(shard_values) = coded.shard_segment(params, &entry.key, seg) else {
            continue;
        };
        z_values.push(entry.z);
        row_values.push(shard_value_at_row_label(shard_values, &basis_w));
        if row_values.len() >= params.need {
            break;
        }
    }
    materialize_row_on_eval_points_from_samples(params, seg, &z_values, &row_values, None)
}

/// Interpolate samples `P(C_h, w)` of one auxiliary row over the helpers'
/// column identifiers and evaluate on the Z-axis of `seg`.
pub(crate) fn materialize_row_on_eval_points_from_samples(
    params: &CodedArrayParams,
    seg: usize,
    z_values: &[Fp],
    row_values: &[Fp],
    weights: Option<&BarycentricWeights>,
) -> Result<Vec<Fp>, CodedReadError> {
    if row_values.len() < params.need || z_values.len() < params.need {
        return Err(CodedReadError::NotEnoughShards);
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
    let matrix = barycentric_eval_matrix(weights, &segment_z_axis_points(params, seg));
    Ok(matrix
        .chunks_exact(weights.z.len())
        .map(|factors| dot(factors, row_values))
        .collect())
}

/// Read multiple indices from the coded store, returning encoded values.
pub fn read_batch_from_store(
    params: &CodedArrayParams,
    store: &CodedStore,
    idxs: &[Index],
    decode_cache: Option<&DecodeCache>,
) -> Result<Vec<Value256>, ReadError> {
    read_batch_from_store_inner(params, store, idxs, decode_cache, true)
}

/// Read multiple indices from the coded store using the raw grid layout.
pub fn read_batch_from_store_raw(
    params: &CodedArrayParams,
    store: &CodedStore,
    idxs: &[Index],
    decode_cache: Option<&DecodeCache>,
) -> Result<Vec<Value256>, ReadError> {
    read_batch_from_store_inner(params, store, idxs, decode_cache, false)
}

fn read_batch_from_store_inner(
    params: &CodedArrayParams,
    store: &CodedStore,
    idxs: &[Index],
    decode_cache: Option<&DecodeCache>,
    semantic_layout: bool,
) -> Result<Vec<Value256>, ReadError> {
    if idxs.is_empty() {
        return Ok(Vec::new());
    }
    let b = public_values_per_segment(params) as u64;
    let mut by_seg: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (pos, &idx) in idxs.iter().enumerate() {
        let seg = usize::try_from(idx / b).map_err(|_| ReadError::InvalidIndex)?;
        let offset = usize::try_from(idx % b).map_err(|_| ReadError::InvalidIndex)?;
        let grid_index = if semantic_layout {
            logical_offset_to_grid_index(params, offset)
        } else {
            offset
        };
        by_seg.entry(seg).or_default().push((pos, grid_index));
    }
    let mut z_evals = vec![fp_zero(); idxs.len()];
    let mut shares_by_pos: Vec<Vec<ShardShare>> = vec![Vec::new(); idxs.len()];
    for (seg, items) in by_seg {
        let rho_z = params.segment_axes(seg).rho_z;
        for &(pos, grid_index) in &items {
            z_evals[pos] = rho_z * params.axis[params.z_index_for_grid_index(grid_index)];
        }
        for (registry_index, entry) in collect_local_shard_entries(params, store, Some(seg))
            .into_iter()
            .take(params.need)
        {
            let Some(column) = store.shard_segment(params, &entry.key, seg) else {
                continue;
            };
            for &(pos, grid_index) in &items {
                shares_by_pos[pos].push(ShardShare {
                    registry_index,
                    value: Some(column[params.w_index_for_grid_index(grid_index)]),
                });
            }
        }
    }
    let mut values = Vec::with_capacity(idxs.len());
    for (pos, shares) in shares_by_pos.iter().enumerate() {
        let value = decode_from_shares_at_z_with_cache(params, z_evals[pos], shares, decode_cache)
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
        let shard_count = 8;
        build_coded_params(&CodedArrayConfig {
            kappa: 4,
            shard_count,
            shard_registry: build_shard_registry_deterministic(shard_count),
            party_registry_indices: vec![(0..shard_count).collect()],
            seed: 99,
        })
    }

    /// A few cells of one aligned block in segment 1: fewer than a segment
    /// read pays for, so the planner keeps them as points.
    fn sparse_block_indices(params: &CodedArrayParams, len: Index) -> Vec<Index> {
        let block = logical_blocks_per_segment(params) as u64 + 2;
        aligned_block_indices(params, block, len)
            .into_iter()
            .flatten()
            .take(params.s / 2)
            .collect()
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
        let block_indices = sparse_block_indices(&params, len);
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
        let block_indices = sparse_block_indices(&params, len);

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
        let block_indices = sparse_block_indices(&params, len);
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
