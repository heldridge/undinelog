//! Unauthenticated RPC service helpers.
//!
//! This module defines request/response types and a service trait that
//! decouples unauthenticated storage from transport/wire details.

use crate::field::Fp;
use crate::shares::{RowShare, SegmentPrimaryShard, ShardShare};
use crate::unauth::{
    DeltaMode, HelperRowMigration, Index, KappaDoubleRowEvalMigration, PartnerShardMigration,
    UnauthOptimizedReadRequest, UnauthOptimizedReadResponse, UnauthStore,
    handle_optimized_read_request,
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
    Migrate {
        target_s: usize,
        // Kappa-halve uses partner columns derived from the source lens.
        // Kappa-double reuses the same payload shape for target-indexed helper columns.
        partner_shards: Vec<PartnerShardMigration>,
        // Kappa-halve may additionally carry helper rows downloaded from peers.
        helper_rows: Vec<HelperRowMigration>,
    },
    SegmentDouble {
        helper_shards: Vec<PartnerShardMigration>,
    },
    KappaDoubleOneClaimLeft {
        target_s: usize,
        helper_evals: Vec<KappaDoubleRowEvalMigration>,
    },
    EvalKappaDoubleTargetAtRowLabelsMatrix {
        target_s: usize,
        segs: Vec<usize>,
        target_registry_indices: Vec<usize>,
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
    Migrate {
        ok: bool,
    },
    SegmentDouble {
        ok: bool,
    },
    KappaDoubleOneClaimLeft {
        ok: bool,
    },
    EvalKappaDoubleTargetAtRowLabelsMatrix {
        values: Vec<Vec<Vec<Fp>>>,
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
            UnauthRequest::Migrate {
                target_s,
                partner_shards,
                helper_rows,
            } => match self.migrate_with_helper_rows(target_s, &partner_shards, &helper_rows) {
                Ok(()) => UnauthResponse::Migrate { ok: true },
                Err(err) => UnauthResponse::Error {
                    message: format!("unauth migrate failed: {:?}", err),
                },
            },
            UnauthRequest::SegmentDouble { helper_shards } => {
                match self.segment_double_with_helpers(&helper_shards) {
                    Ok(()) => UnauthResponse::SegmentDouble { ok: true },
                    Err(err) => UnauthResponse::Error {
                        message: format!("unauth segment-double failed: {:?}", err),
                    },
                }
            }
            UnauthRequest::KappaDoubleOneClaimLeft {
                target_s,
                helper_evals,
            } => match self.migrate_kappa_double_one_claim_left(target_s, &helper_evals) {
                Ok(()) => UnauthResponse::KappaDoubleOneClaimLeft { ok: true },
                Err(err) => UnauthResponse::Error {
                    message: format!("unauth one-claim kappa-double failed: {:?}", err),
                },
            },
            UnauthRequest::EvalKappaDoubleTargetAtRowLabelsMatrix {
                target_s,
                segs,
                target_registry_indices,
                row_labels,
            } => match self.handle_eval_kappa_double_target_at_row_labels_matrix(
                target_s,
                &segs,
                &target_registry_indices,
                &row_labels,
            ) {
                Ok(values) => UnauthResponse::EvalKappaDoubleTargetAtRowLabelsMatrix { values },
                Err(err) => UnauthResponse::Error {
                    message: format!("unauth kappa-double target eval failed: {:?}", err),
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
