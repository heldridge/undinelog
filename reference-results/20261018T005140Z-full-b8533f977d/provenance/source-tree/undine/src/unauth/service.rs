//! Unauthenticated RPC service helpers.
//!
//! This module defines request/response types and a service trait that
//! decouples unauthenticated storage from transport/wire details.

use crate::field::Fp;
use crate::shares::{RowShare, SegmentPrimaryShard, ShardShare};
use crate::unauth::{
    DeltaMode, Index, PartialStitchEvals, SourceHalfState, UnauthOptimizedReadRequest,
    UnauthOptimizedReadResponse, UnauthStore, handle_optimized_read_request,
};

#[derive(Clone, Debug)]
pub struct RowReadRequest {
    pub seg: usize,
    pub z_evals: Vec<Fp>,
}

/// Core unauthenticated request types (transport-agnostic).
#[derive(Clone, Debug)]
pub enum UnauthRequest {
    Read {
        idx: Index,
    },
    ReadBlock {
        block: Index,
    },
    /// Return the locally stored primary shard vectors for one segment.
    /// No interpolation, evaluation, or repair fallback is performed.
    ReadSegmentRaw {
        seg: usize,
    },
    OptimizedRead {
        request: UnauthOptimizedReadRequest,
    },
    ReadRowsAtZ {
        seg: usize,
        z_eval: Fp,
    },
    ReadRowsBatchAtZ {
        seg: usize,
        z_evals: Vec<Fp>,
    },
    ReadRowsMatrix {
        requests: Vec<RowReadRequest>,
    },
    MultiRead {
        idxs: Vec<Index>,
    },
    RepairShard {
        seg: usize,
        registry_index: usize,
    },
    RepairShardBatch {
        seg: usize,
        registry_indices: Vec<usize>,
    },
    EvalShardsAtRowLabels {
        seg: usize,
        col_labels: Vec<Fp>,
        row_labels: Vec<Fp>,
    },
    EvalShardsAtRowLabelsMatrix {
        segs: Vec<usize>,
        col_labels: Vec<Fp>,
        row_labels: Vec<Fp>,
    },
    ApplyDeltas {
        deltas: Vec<(Index, Fp)>,
        mode: DeltaMode,
    },
    ApplyBlockDeltas {
        block: Index,
        deltas: Vec<(usize, Fp)>,
        mode: DeltaMode,
    },
    Extend {
        new_len: Index,
    },
    /// Double the threshold with the local four-segment stitch.
    KappaDouble,
    /// Halve the threshold by undoing the stitch. `source_halves[b]` is the
    /// half of the receiver's source state of stitched segment `b` that its
    /// stitched fragment does not determine.
    KappaHalve {
        source_halves: Vec<SourceHalfState>,
    },
    /// Evaluate the partial stitches held in the local fragments at the
    /// requested `(col_labels[k], row_labels[k])` targets, for a requester
    /// that undoes a stitch with helpers at the current threshold.
    EvalPartialStitches {
        segs: Vec<usize>,
        col_labels: Vec<Fp>,
        row_labels: Vec<Fp>,
    },
    /// Delete all local claim state except the exact target registry indices.
    RetainClaims {
        registry_indices: Vec<usize>,
    },
    GetLen,
}

/// Core unauthenticated response types (transport-agnostic).
#[derive(Clone, Debug)]
pub enum UnauthResponse {
    Read {
        shares: Vec<ShardShare>,
    },
    ReadBlock {
        rows: Vec<RowShare>,
    },
    /// Raw primary shard vectors for one segment, returned exactly as stored.
    ReadSegmentRaw {
        shards: Vec<SegmentPrimaryShard>,
    },
    OptimizedRead {
        response: UnauthOptimizedReadResponse,
    },
    ReadRowsAtZ {
        rows: Vec<RowShare>,
    },
    ReadRowsBatchAtZ {
        rows: Vec<Vec<RowShare>>,
    },
    ReadRowsMatrix {
        rows: Vec<Vec<Vec<RowShare>>>,
    },
    MultiRead {
        shares: Vec<Vec<ShardShare>>,
    },
    RepairShard {
        values: Vec<Fp>,
    },
    RepairShardBatch {
        values: Vec<Vec<Fp>>,
    },
    EvalShardsAtRowLabels {
        values: Vec<Vec<Fp>>,
    },
    EvalShardsAtRowLabelsMatrix {
        values: Vec<Vec<Vec<Fp>>>,
    },
    ApplyDeltas {
        ok: bool,
    },
    ApplyBlockDeltas {
        ok: bool,
    },
    Extend {
        ok: bool,
    },
    KappaDouble {
        ok: bool,
    },
    KappaHalve {
        ok: bool,
    },
    EvalPartialStitches {
        evals: Vec<Vec<PartialStitchEvals>>,
    },
    RetainClaims {
        ok: bool,
    },
    Len {
        len: Index,
    },
    Error {
        message: String,
    },
}

/// Unauthenticated service boundary used by transport adapters.
pub trait UnauthService {
    fn handle_request(&mut self, req: UnauthRequest) -> UnauthResponse;
}

impl UnauthService for UnauthStore {
    fn handle_request(&mut self, req: UnauthRequest) -> UnauthResponse {
        match req {
            UnauthRequest::Read { idx } => UnauthResponse::Read {
                shares: self.handle_read(idx),
            },
            UnauthRequest::ReadBlock { block } => UnauthResponse::ReadBlock {
                rows: self.handle_read_block(block),
            },
            UnauthRequest::ReadSegmentRaw { seg } => UnauthResponse::ReadSegmentRaw {
                shards: self.handle_read_segment_raw(seg),
            },
            UnauthRequest::OptimizedRead { request } => UnauthResponse::OptimizedRead {
                response: handle_optimized_read_request(self, &request),
            },
            UnauthRequest::ReadRowsAtZ { seg, z_eval } => UnauthResponse::ReadRowsAtZ {
                rows: self.handle_read_rows_at_z(seg, z_eval),
            },
            UnauthRequest::ReadRowsBatchAtZ { seg, z_evals } => UnauthResponse::ReadRowsBatchAtZ {
                rows: self.handle_read_rows_batch_at_z(seg, &z_evals),
            },
            UnauthRequest::ReadRowsMatrix { requests } => UnauthResponse::ReadRowsMatrix {
                rows: self.handle_read_rows_matrix(&requests),
            },
            UnauthRequest::MultiRead { idxs } => UnauthResponse::MultiRead {
                shares: self.handle_multi_read(&idxs),
            },
            UnauthRequest::RepairShard {
                seg,
                registry_index,
            } => match self.handle_repair_shard(seg, registry_index) {
                Ok(values) => UnauthResponse::RepairShard { values },
                Err(err) => UnauthResponse::Error {
                    message: format!("unauth repair failed: {:?}", err),
                },
            },
            UnauthRequest::RepairShardBatch {
                seg,
                registry_indices,
            } => match self.handle_repair_shard_batch(seg, &registry_indices) {
                Ok(values) => UnauthResponse::RepairShardBatch { values },
                Err(err) => UnauthResponse::Error {
                    message: format!("unauth batch repair failed: {:?}", err),
                },
            },
            UnauthRequest::EvalShardsAtRowLabels {
                seg,
                col_labels,
                row_labels,
            } => match self.handle_eval_shards_at_row_labels(seg, &col_labels, &row_labels) {
                Ok(values) => UnauthResponse::EvalShardsAtRowLabels { values },
                Err(err) => UnauthResponse::Error {
                    message: format!("unauth shard-eval failed: {:?}", err),
                },
            },
            UnauthRequest::EvalShardsAtRowLabelsMatrix {
                segs,
                col_labels,
                row_labels,
            } => {
                match self.handle_eval_shards_at_row_labels_matrix(&segs, &col_labels, &row_labels)
                {
                    Ok(values) => UnauthResponse::EvalShardsAtRowLabelsMatrix { values },
                    Err(err) => UnauthResponse::Error {
                        message: format!("unauth shard-eval matrix failed: {:?}", err),
                    },
                }
            }
            UnauthRequest::ApplyDeltas { deltas, mode } => match self.apply_deltas(&deltas, mode) {
                Ok(()) => UnauthResponse::ApplyDeltas { ok: true },
                Err(err) => UnauthResponse::Error {
                    message: format!("unauth apply failed: {:?}", err),
                },
            },
            UnauthRequest::ApplyBlockDeltas {
                block,
                deltas,
                mode,
            } => match self.apply_block_deltas(block, &deltas, mode) {
                Ok(()) => UnauthResponse::ApplyBlockDeltas { ok: true },
                Err(err) => UnauthResponse::Error {
                    message: format!("unauth block apply failed: {:?}", err),
                },
            },
            UnauthRequest::Extend { new_len } => match self.extend(new_len) {
                Ok(()) => UnauthResponse::Extend { ok: true },
                Err(err) => UnauthResponse::Error {
                    message: format!("unauth extend failed: {:?}", err),
                },
            },
            UnauthRequest::KappaDouble => match self.kappa_double() {
                Ok(()) => UnauthResponse::KappaDouble { ok: true },
                Err(err) => UnauthResponse::Error {
                    message: format!("unauth kappa-double failed: {:?}", err),
                },
            },
            UnauthRequest::KappaHalve { source_halves } => match self.kappa_halve(&source_halves) {
                Ok(()) => UnauthResponse::KappaHalve { ok: true },
                Err(err) => UnauthResponse::Error {
                    message: format!("unauth kappa-halve failed: {:?}", err),
                },
            },
            UnauthRequest::EvalPartialStitches {
                segs,
                col_labels,
                row_labels,
            } => match self.handle_eval_partial_stitches(&segs, &col_labels, &row_labels) {
                Ok(evals) => UnauthResponse::EvalPartialStitches { evals },
                Err(err) => UnauthResponse::Error {
                    message: format!("unauth partial-stitch eval failed: {:?}", err),
                },
            },
            UnauthRequest::RetainClaims { registry_indices } => {
                match self.retain_registry_indices(&registry_indices) {
                    Ok(()) => UnauthResponse::RetainClaims { ok: true },
                    Err(err) => UnauthResponse::Error {
                        message: format!("unauth retain claims failed: {:?}", err),
                    },
                }
            }
            UnauthRequest::GetLen => UnauthResponse::Len { len: self.len() },
        }
    }
}
