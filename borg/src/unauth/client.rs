//! Unauthenticated client orchestration with transport abstraction.

use crate::coded_array::params::{
    kappa_halve_parent_registry_index, registry_supports_sibling_kappa_halve,
};
use crate::coded_array::rs::{BarycentricWeights, precompute_barycentric};
use crate::fanout::{
    broadcast_with_failures_parallel, collect_shares_by_pos_parallel, first_ok_parallel,
    spawn_parallel_fetches,
};
use crate::field::{Fp, fp_one, fp_pow, fp_zero};
use crate::shares::{RowShare, SegmentPrimaryShard, ShardShare};
use crate::unauth::{
    ClaimBundle, CodedArrayParams, DeltaMode, HelperRowMigration, Index,
    KappaDoubleRowEvalMigration, PartnerShardMigration, RecoveredClaimSegmentState, ShardRegEntry,
    UnauthOptimizedReadRequest, UnauthOptimizedReadResponse, Value, aligned_block_indices,
    build_migrated_params, build_segment_double_params, decode_block_from_rows,
    decode_from_shares_at_z, decode_from_shares_at_z_with_error_correction,
    decode_primary_shard_from_rows, error_correcting_share_target,
    logical_block_to_column_coordinate_for_segment,
    logical_offset_to_column_coordinate_for_segment,
    materialize_row_on_eval_points_from_coefficients, materialize_row_on_eval_points_from_samples,
    public_values_per_segment, semantic_block_position, shard_value_from_vector_batch,
};
use crate::value::{Value256, encode_value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

/// Errors returned by unauthenticated client operations.
#[derive(Clone, Debug)]
pub enum ClientError {
    Net(String),
    Registry(String),
    InvalidConfig(String),
    InvalidIndex,
    NotEnoughShards,
    InvalidResponse(String),
}

/// Transport abstraction for unauthenticated client requests.
pub trait UnauthTransport: Clone {
    type Server: Clone;

    fn server_label(&self, server: &Self::Server) -> String;
    fn unauth_get_len(&self, server: &Self::Server) -> Result<u64, ClientError>;
    fn unauth_read(
        &self,
        server: &Self::Server,
        idx: Index,
    ) -> Result<Vec<ShardShare>, ClientError>;
    fn unauth_read_block(
        &self,
        server: &Self::Server,
        block: Index,
    ) -> Result<Vec<RowShare>, ClientError>;
    fn unauth_read_segment_raw(
        &self,
        server: &Self::Server,
        seg: usize,
    ) -> Result<Vec<SegmentPrimaryShard>, ClientError> {
        let _ = (server, seg);
        Err(ClientError::Net(
            "raw segment reads unsupported by transport".to_string(),
        ))
    }
    fn supports_unauth_optimized_reads(&self) -> bool {
        false
    }
    fn unauth_optimized_read(
        &self,
        server: &Self::Server,
        request: &UnauthOptimizedReadRequest,
    ) -> Result<UnauthOptimizedReadResponse, ClientError> {
        let _ = (server, request);
        Err(ClientError::Net(
            "unauth optimized reads unsupported by transport".to_string(),
        ))
    }
    fn unauth_read_rows_at_z(
        &self,
        server: &Self::Server,
        seg: usize,
        z_eval: Fp,
    ) -> Result<Vec<RowShare>, ClientError>;
    fn unauth_read_rows_batch_at_z(
        &self,
        server: &Self::Server,
        seg: usize,
        z_evals: &[Fp],
    ) -> Result<Vec<Vec<RowShare>>, ClientError>;
    fn unauth_read_rows_matrix(
        &self,
        server: &Self::Server,
        requests: &[(usize, Vec<Fp>)],
    ) -> Result<Vec<Vec<Vec<RowShare>>>, ClientError>;
    fn unauth_multi_read(
        &self,
        server: &Self::Server,
        idxs: &[Index],
    ) -> Result<Vec<Vec<ShardShare>>, ClientError>;
    fn unauth_repair_shard(
        &self,
        server: &Self::Server,
        seg: usize,
        registry_index: usize,
    ) -> Result<Vec<Fp>, ClientError>;
    fn unauth_repair_shard_batch(
        &self,
        server: &Self::Server,
        seg: usize,
        registry_indices: &[usize],
    ) -> Result<Vec<Vec<Fp>>, ClientError> {
        let mut values = Vec::with_capacity(registry_indices.len());
        for &registry_index in registry_indices {
            values.push(self.unauth_repair_shard(server, seg, registry_index)?);
        }
        Ok(values)
    }
    fn unauth_eval_shards_at_row_labels(
        &self,
        _server: &Self::Server,
        _seg: usize,
        _col_labels: &[Fp],
        _row_labels: &[Fp],
    ) -> Result<Vec<Vec<Fp>>, ClientError> {
        Err(ClientError::Net(
            "shard evaluation at row labels unsupported by transport".to_string(),
        ))
    }
    fn unauth_eval_shards_at_row_labels_matrix(
        &self,
        server: &Self::Server,
        segs: &[usize],
        col_labels: &[Fp],
        row_labels: &[Fp],
    ) -> Result<Vec<Vec<Vec<Fp>>>, ClientError> {
        let mut out = Vec::with_capacity(segs.len());
        for &seg in segs {
            out.push(self.unauth_eval_shards_at_row_labels(server, seg, col_labels, row_labels)?);
        }
        Ok(out)
    }
    fn unauth_apply_deltas(
        &self,
        server: &Self::Server,
        deltas: &[(Index, Fp)],
        mode: DeltaMode,
    ) -> Result<bool, ClientError>;
    fn unauth_apply_block_deltas(
        &self,
        _server: &Self::Server,
        _block: Index,
        _deltas: &[(usize, Fp)],
        _mode: DeltaMode,
    ) -> Result<bool, ClientError> {
        Err(ClientError::Net(
            "unauth block updates unsupported by transport".to_string(),
        ))
    }
    fn unauth_extend(&self, server: &Self::Server, new_len: Index) -> Result<bool, ClientError>;
    fn unauth_migrate(
        &self,
        server: &Self::Server,
        target_s: usize,
        partner_shards: &[PartnerShardMigration],
        helper_rows: &[HelperRowMigration],
    ) -> Result<bool, ClientError>;
    fn unauth_segment_double(
        &self,
        server: &Self::Server,
        helper_shards: &[PartnerShardMigration],
    ) -> Result<bool, ClientError> {
        let _ = (server, helper_shards);
        Err(ClientError::Net(
            "unauth segment-double unsupported by transport".to_string(),
        ))
    }
    fn unauth_kappa_double_one_claim_left(
        &self,
        _server: &Self::Server,
        _target_s: usize,
        _helper_evals: &[KappaDoubleRowEvalMigration],
    ) -> Result<bool, ClientError> {
        Err(ClientError::Net(
            "unauth one-claim kappa-double unsupported by transport".to_string(),
        ))
    }
    fn unauth_eval_kappa_double_target_at_row_labels_matrix(
        &self,
        _server: &Self::Server,
        _target_s: usize,
        _segs: &[usize],
        _target_registry_indices: &[usize],
        _row_labels: &[Fp],
    ) -> Result<Vec<Vec<Vec<Fp>>>, ClientError> {
        Err(ClientError::Net(
            "unauth kappa-double target evaluation unsupported by transport".to_string(),
        ))
    }
    fn unauth_retain_claims(
        &self,
        _server: &Self::Server,
        _registry_indices: &[usize],
    ) -> Result<bool, ClientError> {
        Err(ClientError::Net(
            "unauth retain claims unsupported by transport".to_string(),
        ))
    }
}

/// Single-server unauthenticated client that orchestrates reads/writes.
#[derive(Clone)]
pub struct UnauthClient<T: UnauthTransport> {
    pub params: CodedArrayParams,
    pub server: T::Server,
    servers: Vec<T::Server>,
    server_owned_indices: Vec<Vec<usize>>,
    len: Index,
    transport: T,
}

/// Precomputed helper-selection and interpolation plan for one recovered claim bundle.
#[derive(Clone, Debug)]
pub struct ClaimRecoveryPlan {
    claim: ClaimBundle,
    target_entry: ShardRegEntry,
    row_label_indices: Vec<usize>,
    primary_repair_server_positions: Vec<usize>,
    row_repair_entries: Vec<(usize, ShardRegEntry)>,
    row_repair_col_labels_by_server: Vec<Vec<Fp>>,
    row_repair_weights: BarycentricWeights,
}

#[derive(Clone, Debug)]
pub struct RepairedPartySegmentState {
    pub seg: usize,
    pub shards: Vec<(usize, Vec<Value>)>,
    pub rows: Vec<(usize, Vec<Value>)>,
}

#[derive(Clone, Debug)]
struct SourcedRowShare {
    row: RowShare,
    server_pos: usize,
}

#[derive(Clone, Debug)]
struct SourcedEvalValues {
    values: Vec<Fp>,
    server_pos: usize,
}

impl<T> UnauthClient<T>
where
    T: UnauthTransport + Send + Sync + 'static,
    T::Server: Clone + Send + Sync + 'static,
{
    pub fn new(
        transport: T,
        params: CodedArrayParams,
        server: T::Server,
        servers: Vec<T::Server>,
        len: Index,
    ) -> Self {
        Self::new_with_owned_indices(
            transport,
            params.clone(),
            server,
            servers,
            params.party_registry_indices.clone(),
            len,
        )
    }

    pub fn new_with_owned_indices(
        transport: T,
        params: CodedArrayParams,
        server: T::Server,
        servers: Vec<T::Server>,
        mut server_owned_indices: Vec<Vec<usize>>,
        len: Index,
    ) -> Self {
        if server_owned_indices.len() < servers.len() {
            server_owned_indices.resize(servers.len(), Vec::new());
        } else if server_owned_indices.len() > servers.len() {
            server_owned_indices.truncate(servers.len());
        }
        for indices in &mut server_owned_indices {
            indices.sort_unstable();
            indices.dedup();
        }
        Self {
            params,
            server,
            servers,
            server_owned_indices,
            len,
            transport,
        }
    }

    fn refresh_len(&mut self) -> Result<(), ClientError> {
        let transport = self.transport.clone();
        let len = first_ok_parallel(&self.servers, move |server| {
            transport.unauth_get_len(&server)
        })
        .ok_or_else(|| {
            ClientError::Net("no unauth servers responded to UnauthGetLen".to_string())
        })?;
        self.len = len;
        Ok(())
    }

    /// Precompute the stable helper selection and interpolation metadata for one claim bundle.
    pub fn prepare_claim_recovery(
        &self,
        claim: &ClaimBundle,
    ) -> Result<ClaimRecoveryPlan, ClientError> {
        self.prepare_claim_recovery_with_targets(claim, self.params.s, self.params.need)
    }

    /// Precompute a repair plan that oversamples enough helper data to correct
    /// up to `max_errors` bad RS-point replies during claim recovery.
    pub fn prepare_claim_recovery_correcting(
        &self,
        claim: &ClaimBundle,
        max_errors: usize,
    ) -> Result<ClaimRecoveryPlan, ClientError> {
        let target_entry = *self
            .params
            .shard_registry
            .get(claim.registry_index)
            .ok_or(ClientError::InvalidIndex)?;
        let row_label_indices = claim
            .row_label_indices(&self.params)
            .ok_or(ClientError::InvalidIndex)?;
        let primary_repair_server_positions = self
            .select_primary_repair_server_positions_with_target_capped(
                error_correcting_share_target(self.params.s, max_errors),
            )?;
        let row_repair_entries = self.select_row_repair_entries_with_target_distinct_servers(
            error_correcting_share_target(self.params.need, max_errors),
        )?;
        let row_repair_col_labels_by_server =
            self.plan_requested_col_labels_by_server(&row_repair_entries)?;
        let row_repair_z_evals: Vec<_> = row_repair_entries
            .iter()
            .map(|(_, entry)| entry.z)
            .collect();
        let row_repair_weights = precompute_barycentric(&row_repair_z_evals);
        Ok(ClaimRecoveryPlan {
            claim: claim.clone(),
            target_entry,
            row_label_indices,
            primary_repair_server_positions,
            row_repair_entries,
            row_repair_col_labels_by_server,
            row_repair_weights,
        })
    }

    fn prepare_claim_recovery_with_targets(
        &self,
        claim: &ClaimBundle,
        primary_repair_row_target: usize,
        row_repair_entry_target: usize,
    ) -> Result<ClaimRecoveryPlan, ClientError> {
        let target_entry = *self
            .params
            .shard_registry
            .get(claim.registry_index)
            .ok_or(ClientError::InvalidIndex)?;
        let row_label_indices = claim
            .row_label_indices(&self.params)
            .ok_or(ClientError::InvalidIndex)?;
        let primary_repair_server_positions =
            self.select_primary_repair_server_positions_with_target(primary_repair_row_target)?;
        let row_repair_entries =
            self.select_row_repair_entries_with_target(row_repair_entry_target)?;
        let row_repair_col_labels_by_server =
            self.plan_requested_col_labels_by_server(&row_repair_entries)?;
        let row_repair_z_evals: Vec<_> = row_repair_entries
            .iter()
            .map(|(_, entry)| entry.z)
            .collect();
        let row_repair_weights = precompute_barycentric(&row_repair_z_evals);
        Ok(ClaimRecoveryPlan {
            claim: claim.clone(),
            target_entry,
            row_label_indices,
            primary_repair_server_positions,
            row_repair_entries,
            row_repair_col_labels_by_server,
            row_repair_weights,
        })
    }

    fn block_indices(&self, block: Index) -> Vec<Option<Index>> {
        aligned_block_indices(&self.params, block, self.len)
    }

    fn decode_block_rows_into_values(
        &self,
        block: Index,
        rows: &[RowShare],
        logical_indices: &[Option<Index>],
    ) -> Result<Vec<Value>, ClientError> {
        let decoded = decode_block_from_rows(&self.params, block, rows)
            .map_err(|_| ClientError::NotEnoughShards)?;
        let mut results: Vec<Value256> = vec![encode_value(fp_zero()); self.params.s];
        for (offset, idx) in logical_indices.iter().enumerate() {
            if idx.is_some() {
                results[offset] = encode_value(decoded[offset]);
            }
        }
        Ok(results)
    }

    fn collect_block_rows(&self, block: Index) -> Result<Vec<RowShare>, ClientError> {
        let mut rows = Vec::with_capacity(self.params.s);
        let mut seen = HashSet::new();
        let transport = self.transport.clone();
        let fetches = spawn_parallel_fetches(&self.servers, move |server| {
            transport.unauth_read_block(&server, block)
        });
        for _ in 0..self.servers.len() {
            if rows.len() >= self.params.s {
                fetches.cancel();
                break;
            }
            let Ok((_, reply)) = fetches.recv() else {
                break;
            };
            let reply = match reply {
                Ok(reply) => reply,
                Err(_) => continue,
            };
            for row in reply {
                let label = encode_value(row.row_label);
                if seen.insert(label) {
                    rows.push(row);
                    if rows.len() >= self.params.s {
                        break;
                    }
                }
            }
            if rows.len() >= self.params.s {
                fetches.cancel();
                break;
            }
        }
        if rows.len() < self.params.s {
            return Err(ClientError::NotEnoughShards);
        }
        rows.sort_by_key(|row| encode_value(row.row_label));
        Ok(rows)
    }

    fn write_block_fallback_via_points(
        &mut self,
        block: Index,
        deltas: &[(usize, Fp)],
        mode: DeltaMode,
    ) -> Result<(), ClientError> {
        let logical_indices = self.block_indices(block);
        let mut point_deltas = Vec::with_capacity(deltas.len());
        for &(offset_in_block, delta) in deltas {
            if offset_in_block >= self.params.s {
                return Err(ClientError::InvalidIndex);
            }
            let Some(idx) = logical_indices.get(offset_in_block).and_then(|idx| *idx) else {
                return Err(ClientError::InvalidIndex);
            };
            point_deltas.push((idx, delta));
        }
        self.write_batch(&point_deltas, mode)
    }

    /// Read a value from the array.
    pub fn read(&mut self, idx: Index) -> Result<Value, ClientError> {
        self.read_with_decode_policy(idx, self.params.need, None)
    }

    /// Read a value from an oversampled helper set and correct up to
    /// `max_errors` bad shard replies.
    pub fn read_correcting(&mut self, idx: Index, max_errors: usize) -> Result<Value, ClientError> {
        let share_target = error_correcting_share_target(self.params.need, max_errors);
        self.read_with_decode_policy(idx, share_target, Some(max_errors))
    }

    fn read_with_decode_policy(
        &mut self,
        idx: Index,
        share_target: usize,
        max_errors: Option<usize>,
    ) -> Result<Value, ClientError> {
        self.refresh_len()?;
        if idx >= self.len {
            return Ok(encode_value(fp_zero()));
        }
        let b = public_values_per_segment(&self.params) as u64;
        let seg = (idx / b) as usize;
        let logical_offset = (idx % b) as usize;
        let z_eval =
            logical_offset_to_column_coordinate_for_segment(&self.params, seg, logical_offset);
        let mut shares: Vec<ShardShare> = Vec::new();
        let mut seen = HashSet::new();
        let transport = self.transport.clone();
        let fetches = spawn_parallel_fetches(&self.servers, move |server| {
            transport.unauth_read(&server, idx)
        });
        for _ in 0..self.servers.len() {
            if shares.len() >= share_target {
                fetches.cancel();
                break;
            }
            let Ok((_, reply)) = fetches.recv() else {
                break;
            };
            let reply = match reply {
                Ok(reply) => reply,
                Err(_) => continue,
            };
            for share in reply {
                if share.value.is_some() && seen.insert(share.registry_index) {
                    shares.push(share);
                    if shares.len() >= share_target {
                        break;
                    }
                }
            }
            if shares.len() >= share_target {
                fetches.cancel();
                break;
            }
        }
        let decoded = match max_errors {
            Some(max_errors) => decode_from_shares_at_z_with_error_correction(
                &self.params,
                z_eval,
                &shares,
                max_errors,
            )
            .map(|decoded| decoded.value),
            None => decode_from_shares_at_z(&self.params, z_eval, &shares),
        };
        decoded
            .map(encode_value)
            .map_err(|_| ClientError::NotEnoughShards)
    }

    /// Read multiple values from the array in one generic multi-read operation.
    pub fn multi_read(&mut self, idxs: &[Index]) -> Result<Vec<Value>, ClientError> {
        self.multi_read_with_decode_policy(idxs, self.params.need, None)
    }

    /// Read multiple values from oversampled helper sets and correct up to
    /// `max_errors` bad shard replies per decoded value.
    pub fn multi_read_correcting(
        &mut self,
        idxs: &[Index],
        max_errors: usize,
    ) -> Result<Vec<Value>, ClientError> {
        let share_target = error_correcting_share_target(self.params.need, max_errors);
        self.multi_read_with_decode_policy(idxs, share_target, Some(max_errors))
    }

    fn multi_read_with_decode_policy(
        &mut self,
        idxs: &[Index],
        share_target: usize,
        max_errors: Option<usize>,
    ) -> Result<Vec<Value>, ClientError> {
        self.refresh_len()?;
        if idxs.is_empty() {
            return Ok(Vec::new());
        }
        let mut results = vec![encode_value(fp_zero()); idxs.len()];
        let b = public_values_per_segment(&self.params) as u64;
        let mut positions = Vec::new();
        for (pos, &idx) in idxs.iter().enumerate() {
            if idx < self.len {
                positions.push((pos, idx));
            }
        }
        if positions.is_empty() {
            return Ok(results);
        }
        let mut valid_idxs = Vec::with_capacity(positions.len());
        for &(_, idx) in &positions {
            valid_idxs.push(idx);
        }
        let transport = self.transport.clone();
        let valid_idxs = valid_idxs;
        let (shares_by_idx, _) = collect_shares_by_pos_parallel(
            &self.servers,
            positions.len(),
            share_target,
            move |server| transport.unauth_multi_read(&server, &valid_idxs),
            |_, _| ClientError::InvalidResponse("unauth read batch length mismatch".to_string()),
        )?;
        for (i, (pos, idx)) in positions.iter().enumerate() {
            let seg = (*idx / b) as usize;
            let logical_offset = (idx % b) as usize;
            let z_eval =
                logical_offset_to_column_coordinate_for_segment(&self.params, seg, logical_offset);
            let decoded = match max_errors {
                Some(max_errors) => decode_from_shares_at_z_with_error_correction(
                    &self.params,
                    z_eval,
                    &shares_by_idx[i],
                    max_errors,
                )
                .map(|decoded| decoded.value),
                None => decode_from_shares_at_z(&self.params, z_eval, &shares_by_idx[i]),
            };
            let value = decoded
                .map(encode_value)
                .map_err(|_| ClientError::NotEnoughShards)?;
            results[*pos] = value;
        }
        Ok(results)
    }

    /// Alias for the generic multi-read helper.
    pub fn read_batch(&mut self, idxs: &[Index]) -> Result<Vec<Value>, ClientError> {
        self.multi_read(idxs)
    }

    /// Read one aligned semantic block in the current mode.
    pub fn read_block(&mut self, block: Index) -> Result<Vec<Value>, ClientError> {
        self.refresh_len()?;
        let logical_indices = self.block_indices(block);
        if logical_indices.iter().all(|idx| idx.is_none()) {
            return Ok(vec![encode_value(fp_zero()); self.params.s]);
        }
        let rows = self.collect_block_rows(block)?;
        self.decode_block_rows_into_values(block, &rows, &logical_indices)
    }

    /// Read the raw primary shard vectors for one segment from one party.
    pub fn read_segment_raw(
        &mut self,
        party_id: u64,
        seg: usize,
    ) -> Result<Vec<SegmentPrimaryShard>, ClientError> {
        let label = party_id.to_string();
        let server = self
            .servers
            .iter()
            .find(|server| self.transport.server_label(server) == label)
            .ok_or_else(|| ClientError::Registry(format!("party {} not found", party_id)))?;
        self.transport.unauth_read_segment_raw(server, seg)
    }

    /// Read multiple aligned semantic blocks, batching row requests by segment.
    pub fn read_block_batch(&mut self, blocks: &[Index]) -> Result<Vec<Vec<Value>>, ClientError> {
        self.refresh_len()?;
        if blocks.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = vec![vec![encode_value(fp_zero()); self.params.s]; blocks.len()];
        let mut grouped = BTreeMap::<usize, Vec<(usize, Index, Vec<Option<Index>>)>>::new();

        for (out_pos, &block) in blocks.iter().enumerate() {
            let logical_indices = self.block_indices(block);
            if logical_indices.iter().all(|idx| idx.is_none()) {
                continue;
            }
            let seg = semantic_block_position(&self.params, block).segment;
            grouped
                .entry(seg)
                .or_default()
                .push((out_pos, block, logical_indices));
        }

        if grouped.is_empty() {
            return Ok(results);
        }

        let mut requests = Vec::<(usize, Vec<Fp>)>::with_capacity(grouped.len());
        let mut request_map =
            Vec::<Vec<(usize, Index, Vec<Option<Index>>)>>::with_capacity(grouped.len());
        for (seg, mapping) in grouped {
            let z_evals = mapping
                .iter()
                .map(|(_, block, _)| {
                    logical_block_to_column_coordinate_for_segment(&self.params, *block)
                })
                .collect();
            requests.push((seg, z_evals));
            request_map.push(mapping);
        }

        let rows_by_request = self.collect_rows_matrix(&requests)?;
        if rows_by_request.len() != requests.len() {
            return Err(ClientError::InvalidResponse(
                "missing batched row-read responses".to_string(),
            ));
        }

        for (batches, mapping) in rows_by_request.into_iter().zip(request_map.into_iter()) {
            if batches.len() != mapping.len() {
                return Err(ClientError::InvalidResponse(
                    "batched row-read response length mismatch".to_string(),
                ));
            }
            for (rows, (out_pos, block, logical_indices)) in
                batches.into_iter().zip(mapping.into_iter())
            {
                results[out_pos] =
                    self.decode_block_rows_into_values(block, &rows, &logical_indices)?;
            }
        }

        Ok(results)
    }

    /// Apply a single delta at index `idx`.
    pub fn write_delta(&mut self, idx: Index, delta: Fp) -> Result<(), ClientError> {
        self.write_batch(&[(idx, delta)], DeltaMode::Aggregate)
    }

    /// Apply one aligned semantic-block delta patch.
    pub fn write_block(
        &mut self,
        block: Index,
        deltas: &[(usize, Fp)],
        mode: DeltaMode,
    ) -> Result<(), ClientError> {
        self.refresh_len()?;
        let logical_indices = self.block_indices(block);
        for &(offset_in_block, _) in deltas {
            if offset_in_block >= self.params.s {
                return Err(ClientError::InvalidIndex);
            }
            if logical_indices
                .get(offset_in_block)
                .and_then(|idx| *idx)
                .is_none()
            {
                return Err(ClientError::InvalidIndex);
            }
        }
        let transport = self.transport.clone();
        let deltas = deltas.to_vec();
        let fallback_deltas = deltas.clone();
        let failed = broadcast_with_failures_parallel(
            &self.servers,
            |server| self.transport.server_label(server),
            move |server| transport.unauth_apply_block_deltas(&server, block, &deltas, mode),
            |ok| {
                if ok {
                    Ok(())
                } else {
                    Err("unauth block write failed".to_string())
                }
            },
        );
        if !failed.is_empty() {
            let unsupported = failed
                .iter()
                .all(|(_, err)| err.contains("unauth block updates unsupported by transport"));
            if unsupported {
                return self.write_block_fallback_via_points(block, &fallback_deltas, mode);
            }
            return Err(ClientError::Net(format!(
                "unauth block write failed: {:?}",
                failed
            )));
        }
        Ok(())
    }

    /// Apply a single delta inside one aligned semantic block.
    pub fn write_block_delta(
        &mut self,
        block: Index,
        offset_in_block: usize,
        delta: Fp,
    ) -> Result<(), ClientError> {
        self.write_block(block, &[(offset_in_block, delta)], DeltaMode::Aggregate)
    }

    /// Apply a batch of deltas using the provided update mode.
    pub fn write_batch(
        &mut self,
        deltas: &[(Index, Fp)],
        mode: DeltaMode,
    ) -> Result<(), ClientError> {
        self.refresh_len()?;
        if deltas.iter().any(|(idx, _)| *idx >= self.len) {
            return Err(ClientError::InvalidIndex);
        }
        let transport = self.transport.clone();
        let deltas = deltas.to_vec();
        let failed = broadcast_with_failures_parallel(
            &self.servers,
            |server| self.transport.server_label(server),
            move |server| transport.unauth_apply_deltas(&server, &deltas, mode),
            |ok| {
                if ok {
                    Ok(())
                } else {
                    Err("unauth write failed".to_string())
                }
            },
        );
        if !failed.is_empty() {
            return Err(ClientError::Net(format!(
                "unauth write failed: {:?}",
                failed
            )));
        }
        Ok(())
    }

    /// Extend the array length, broadcasting the allocation.
    pub fn extend(&mut self, new_len: Index) -> Result<(), ClientError> {
        self.refresh_len()?;
        if new_len <= self.len {
            return Ok(());
        }
        let transport = self.transport.clone();
        let failed = broadcast_with_failures_parallel(
            &self.servers,
            |server| self.transport.server_label(server),
            move |server| transport.unauth_extend(&server, new_len),
            |ok| {
                if ok {
                    Ok(())
                } else {
                    Err("unauth extend failed".to_string())
                }
            },
        );
        if !failed.is_empty() {
            return Err(ClientError::Net(format!(
                "unauth extend failed: {:?}",
                failed
            )));
        }
        self.len = new_len;
        Ok(())
    }

    fn collect_rows_at_zs(
        &self,
        seg: usize,
        z_evals: &[Fp],
    ) -> Result<Vec<Vec<RowShare>>, ClientError> {
        self.collect_rows_matrix(&[(seg, z_evals.to_vec())])?
            .into_iter()
            .next()
            .ok_or_else(|| ClientError::InvalidResponse("missing row matrix response".to_string()))
    }

    fn collect_rows_matrix(
        &self,
        requests: &[(usize, Vec<Fp>)],
    ) -> Result<Vec<Vec<Vec<RowShare>>>, ClientError> {
        let server_positions: Vec<_> = (0..self.servers.len()).collect();
        self.collect_rows_matrix_from_server_positions(requests, &server_positions)
    }

    fn collect_rows_matrix_from_server_positions(
        &self,
        requests: &[(usize, Vec<Fp>)],
        server_positions: &[usize],
    ) -> Result<Vec<Vec<Vec<RowShare>>>, ClientError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let mut rows_by_request: Vec<Vec<Vec<RowShare>>> = requests
            .iter()
            .map(|(_, z_evals)| vec![Vec::new(); z_evals.len()])
            .collect();
        let mut seen_by_request: Vec<Vec<HashSet<_>>> = requests
            .iter()
            .map(|(_, z_evals)| (0..z_evals.len()).map(|_| HashSet::new()).collect())
            .collect();
        let transport = self.transport.clone();
        let requests = requests.to_vec();
        let requests_for_fetch = requests.clone();
        let servers: Vec<_> = server_positions
            .iter()
            .copied()
            .filter_map(|server_pos| self.servers.get(server_pos).cloned())
            .collect();
        let fetches = spawn_parallel_fetches(&servers, move |server| {
            transport.unauth_read_rows_matrix(&server, &requests_for_fetch)
        });
        for _ in 0..servers.len() {
            if rows_by_request
                .iter()
                .zip(&seen_by_request)
                .all(|(rows, _)| rows.iter().all(|batch| batch.len() >= self.params.s))
            {
                fetches.cancel();
                break;
            }
            let Ok((_, replies)) = fetches.recv() else {
                break;
            };
            let replies = match replies {
                Ok(replies) => replies,
                Err(_) => continue,
            };
            if replies.len() != requests.len() {
                fetches.cancel();
                return Err(ClientError::InvalidResponse(
                    "unauth row matrix request length mismatch".to_string(),
                ));
            }
            for (req_pos, batches) in replies.into_iter().enumerate() {
                if batches.len() != requests[req_pos].1.len() {
                    fetches.cancel();
                    return Err(ClientError::InvalidResponse(
                        "unauth row matrix batch length mismatch".to_string(),
                    ));
                }
                for (z_pos, rows) in batches.into_iter().enumerate() {
                    for row in rows {
                        let label = encode_value(row.row_label);
                        if seen_by_request[req_pos][z_pos].insert(label) {
                            rows_by_request[req_pos][z_pos].push(row);
                            if rows_by_request[req_pos][z_pos].len() >= self.params.s {
                                break;
                            }
                        }
                    }
                }
            }
        }
        Ok(rows_by_request)
    }

    fn collect_rows_matrix_from_server_positions_correcting(
        &self,
        requests: &[(usize, Vec<Fp>)],
        server_positions: &[usize],
        max_errors: usize,
        bad_servers: &HashSet<usize>,
    ) -> Result<Vec<Vec<Vec<SourcedRowShare>>>, ClientError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let known_bad_points =
            self.known_bad_row_points_for_server_positions(server_positions, bad_servers)?;
        let remaining_errors = remaining_error_budget(max_errors, known_bad_points)?;
        let target_rows = error_correcting_share_target(self.params.s, remaining_errors);
        let mut rows_by_request: Vec<Vec<Vec<SourcedRowShare>>> = requests
            .iter()
            .map(|(_, z_evals)| vec![Vec::new(); z_evals.len()])
            .collect();
        let mut seen_by_request: Vec<Vec<HashSet<_>>> = requests
            .iter()
            .map(|(_, z_evals)| (0..z_evals.len()).map(|_| HashSet::new()).collect())
            .collect();
        let transport = self.transport.clone();
        let requests = requests.to_vec();
        let requests_for_fetch = requests.clone();
        let servers: Vec<_> = server_positions
            .iter()
            .copied()
            .filter(|server_pos| !bad_servers.contains(server_pos))
            .filter_map(|server_pos| {
                self.servers
                    .get(server_pos)
                    .cloned()
                    .map(|server| (server_pos, server))
            })
            .collect();
        let fetches = spawn_parallel_fetches(&servers, move |(_, server)| {
            transport.unauth_read_rows_matrix(&server, &requests_for_fetch)
        });
        let mut replies_by_server = vec![None; servers.len()];
        for _ in 0..servers.len() {
            let Ok((server_idx, replies)) = fetches.recv() else {
                break;
            };
            let replies = match replies {
                Ok(replies) => replies,
                Err(_) => continue,
            };
            if replies.len() != requests.len() {
                fetches.cancel();
                return Err(ClientError::InvalidResponse(
                    "unauth row matrix request length mismatch".to_string(),
                ));
            }
            for (req_pos, batches) in replies.iter().enumerate() {
                if batches.len() != requests[req_pos].1.len() {
                    fetches.cancel();
                    return Err(ClientError::InvalidResponse(
                        "unauth row matrix batch length mismatch".to_string(),
                    ));
                }
            }
            replies_by_server[server_idx] = Some(replies);
        }

        for (server_idx, replies) in replies_by_server.into_iter().enumerate() {
            if rows_by_request
                .iter()
                .all(|rows| rows.iter().all(|batch| batch.len() >= target_rows))
            {
                break;
            }
            let Some(replies) = replies else {
                continue;
            };
            let server_pos = servers[server_idx].0;
            for (req_pos, batches) in replies.into_iter().enumerate() {
                for (z_pos, rows) in batches.into_iter().enumerate() {
                    if rows_by_request[req_pos][z_pos].len() >= target_rows {
                        continue;
                    }
                    let mut accepted_from_server = 0usize;
                    for row in rows {
                        if accepted_from_server >= self.params.rows_per_shard {
                            break;
                        }
                        let label = encode_value(row.row_label);
                        if seen_by_request[req_pos][z_pos].insert(label) {
                            rows_by_request[req_pos][z_pos]
                                .push(SourcedRowShare { row, server_pos });
                            accepted_from_server += 1;
                            if rows_by_request[req_pos][z_pos].len() >= target_rows {
                                break;
                            }
                        }
                    }
                }
            }
        }
        Ok(rows_by_request)
    }

    fn collect_rows_at_zs_correcting(
        &self,
        seg: usize,
        z_evals: &[Fp],
        max_errors: usize,
        bad_servers: &HashSet<usize>,
    ) -> Result<Vec<Vec<SourcedRowShare>>, ClientError> {
        let server_positions: Vec<_> = (0..self.servers.len()).collect();
        self.collect_rows_matrix_from_server_positions_correcting(
            &[(seg, z_evals.to_vec())],
            &server_positions,
            max_errors,
            bad_servers,
        )?
        .into_iter()
        .next()
        .ok_or_else(|| ClientError::InvalidResponse("missing row matrix response".to_string()))
    }

    fn decode_slice_from_sourced_rows_correcting(
        &self,
        rows: &[SourcedRowShare],
        eval_points: &[Fp],
        max_errors: usize,
        known_bad_points: usize,
        bad_servers: &mut HashSet<usize>,
    ) -> Result<Vec<Fp>, ClientError> {
        let remaining_errors = remaining_error_budget(max_errors, known_bad_points)?;
        let mut seen = HashSet::new();
        let mut points = Vec::new();
        let mut point_servers = Vec::new();
        let mut points_by_server: HashMap<usize, usize> = HashMap::new();
        for sourced in rows {
            if bad_servers.contains(&sourced.server_pos) {
                continue;
            }
            let label_bytes = encode_value(sourced.row.row_label);
            if self.params.row_label_index_for_bytes(label_bytes).is_none()
                || !seen.insert(label_bytes)
            {
                continue;
            }
            points.push((sourced.row.row_label, sourced.row.value));
            point_servers.push(sourced.server_pos);
            *points_by_server.entry(sourced.server_pos).or_insert(0) += 1;
        }
        if points.len() < self.params.s + 2 * remaining_errors {
            return Err(ClientError::NotEnoughShards);
        }

        let decoded = crate::coded_array::rs::rs_decode_eval_with_errors(
            &points,
            self.params.s,
            eval_points.first().copied().unwrap_or_else(fp_zero),
            remaining_errors,
        )
        .map_err(|_| ClientError::NotEnoughShards)?;

        let mut newly_bad_servers = HashSet::new();
        for bad_idx in decoded.bad_indices {
            if let Some(&server_pos) = point_servers.get(bad_idx) {
                newly_bad_servers.insert(server_pos);
            }
        }
        let newly_bad_points = newly_bad_servers
            .iter()
            .map(|server_pos| points_by_server.get(server_pos).copied().unwrap_or(0))
            .sum::<usize>();
        if newly_bad_points > remaining_errors {
            return Err(ClientError::NotEnoughShards);
        }
        bad_servers.extend(newly_bad_servers);

        Ok(eval_coefficients_on_structured_points(
            &self.params,
            &decoded.coefficients,
            &eval_points,
        ))
    }

    fn decode_primary_shard_from_sourced_rows_correcting(
        &self,
        entry: ShardRegEntry,
        rows: &[SourcedRowShare],
        max_errors: usize,
        known_bad_points: usize,
        bad_servers: &mut HashSet<usize>,
    ) -> Result<Vec<Fp>, ClientError> {
        let eval_points: Vec<_> = (0..self.params.s)
            .map(|t| entry.a * self.params.h[t])
            .collect();
        self.decode_slice_from_sourced_rows_correcting(
            rows,
            &eval_points,
            max_errors,
            known_bad_points,
            bad_servers,
        )
    }

    fn collect_selected_rows_matrix(
        &self,
        requests: &[(usize, Vec<Fp>)],
        selected_row_label_bytes: &[Value256],
    ) -> Result<Vec<Vec<Vec<RowShare>>>, ClientError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let selected_count = selected_row_label_bytes.len();
        let selected_pos_by_bytes: BTreeMap<_, _> = selected_row_label_bytes
            .iter()
            .copied()
            .enumerate()
            .map(|(pos, bytes)| (bytes, pos))
            .collect();
        let mut rows_by_request: Vec<Vec<Vec<Option<RowShare>>>> = requests
            .iter()
            .map(|(_, z_evals)| {
                (0..z_evals.len())
                    .map(|_| vec![None; selected_count])
                    .collect()
            })
            .collect();

        let transport = self.transport.clone();
        let requests = requests.to_vec();
        let requests_for_fetch = requests.clone();
        let selected_set: HashSet<_> = selected_row_label_bytes.iter().copied().collect();
        let servers: Vec<_> = self
            .server_owned_indices
            .iter()
            .enumerate()
            .filter(|(_, owned_indices)| {
                owned_indices.iter().any(|&registry_index| {
                    self.params
                        .row_label_indices_for_registry_index(registry_index)
                        .into_iter()
                        .any(|row_label_index| {
                            selected_set.contains(&self.params.row_label_bytes[row_label_index])
                        })
                })
            })
            .filter_map(|(server_pos, _)| self.servers.get(server_pos).cloned())
            .collect();
        let fetches = spawn_parallel_fetches(&servers, move |server| {
            transport.unauth_read_rows_matrix(&server, &requests_for_fetch)
        });
        for _ in 0..servers.len() {
            if rows_by_request.iter().all(|rows| {
                rows.iter()
                    .all(|batch| batch.iter().all(|entry| entry.is_some()))
            }) {
                fetches.cancel();
                break;
            }
            let Ok((_, replies)) = fetches.recv() else {
                break;
            };
            let replies = match replies {
                Ok(replies) => replies,
                Err(_) => continue,
            };
            if replies.len() != requests.len() {
                fetches.cancel();
                return Err(ClientError::InvalidResponse(
                    "unauth selected row matrix request length mismatch".to_string(),
                ));
            }
            for (req_pos, batches) in replies.into_iter().enumerate() {
                if batches.len() != requests[req_pos].1.len() {
                    fetches.cancel();
                    return Err(ClientError::InvalidResponse(
                        "unauth selected row matrix batch length mismatch".to_string(),
                    ));
                }
                for (z_pos, rows) in batches.into_iter().enumerate() {
                    for row in rows {
                        let label = encode_value(row.row_label);
                        let Some(&selected_pos) = selected_pos_by_bytes.get(&label) else {
                            continue;
                        };
                        if rows_by_request[req_pos][z_pos][selected_pos].is_none() {
                            rows_by_request[req_pos][z_pos][selected_pos] = Some(row);
                        }
                    }
                }
            }
        }

        let mut out = Vec::with_capacity(rows_by_request.len());
        for rows in rows_by_request {
            let mut out_batches = Vec::with_capacity(rows.len());
            for batch in rows {
                let mut out_rows = Vec::with_capacity(batch.len());
                for row in batch {
                    let Some(row) = row else {
                        return Err(ClientError::NotEnoughShards);
                    };
                    out_rows.push(row);
                }
                out_batches.push(out_rows);
            }
            out.push(out_batches);
        }
        Ok(out)
    }

    fn collect_partner_shards_for_kappa_halve(
        &self,
        target_s: usize,
    ) -> Result<Vec<Vec<PartnerShardMigration>>, ClientError> {
        build_migrated_params(&self.params, target_s)
            .map_err(|err| ClientError::InvalidConfig(format!("{:?}", err)))?;
        let mut per_server = vec![Vec::new(); self.servers.len()];
        let twist = crate::fp_pow(
            self.params.omega,
            (self.params.n_grid / (2 * self.params.s)) as u64,
        );
        for (server_pos, registry_indices) in self.server_owned_indices.iter().enumerate() {
            let segment_count = self
                .len
                .div_ceil(public_values_per_segment(&self.params) as u64)
                as usize;
            let z_evals: Vec<_> = registry_indices
                .iter()
                .map(|&registry_index| -self.params.shard_registry[registry_index].z)
                .collect();
            let requests: Vec<_> = (0..segment_count)
                .map(|seg| (seg, z_evals.clone()))
                .collect();
            let rows_by_request = self.collect_rows_matrix(&requests)?;
            for &registry_index in registry_indices {
                let source_entry = self.params.shard_registry[registry_index];
                let partner_entry = crate::ShardRegEntry {
                    a: source_entry.a * twist,
                    z: -source_entry.z,
                    key: crate::ShardKey {
                        a_bytes: crate::encode_value(source_entry.a * twist),
                        z_bytes: crate::encode_value(-source_entry.z),
                    },
                };
                let mut segments = Vec::with_capacity(segment_count);
                let z_pos = registry_indices
                    .iter()
                    .position(|idx| *idx == registry_index)
                    .ok_or_else(|| {
                        ClientError::InvalidConfig(
                            "missing kappa-halve source index in party assignment".to_string(),
                        )
                    })?;
                for seg in 0..segment_count {
                    let rows = rows_by_request[seg][z_pos].clone();
                    let values = decode_primary_shard_from_rows(&self.params, partner_entry, &rows)
                        .map_err(|_| ClientError::NotEnoughShards)?;
                    segments.push(values);
                }
                per_server[server_pos].push(PartnerShardMigration {
                    registry_index,
                    segments,
                });
            }
        }
        Ok(per_server)
    }

    fn collect_partner_shards_for_kappa_halve_correcting(
        &self,
        target_s: usize,
        max_errors: usize,
        bad_servers: &mut HashSet<usize>,
    ) -> Result<Vec<Vec<PartnerShardMigration>>, ClientError> {
        build_migrated_params(&self.params, target_s)
            .map_err(|err| ClientError::InvalidConfig(format!("{:?}", err)))?;
        let mut per_server = vec![Vec::new(); self.servers.len()];
        let twist = crate::fp_pow(
            self.params.omega,
            (self.params.n_grid / (2 * self.params.s)) as u64,
        );
        let segment_count =
            self.len
                .div_ceil(public_values_per_segment(&self.params) as u64) as usize;
        for (server_pos, registry_indices) in self.server_owned_indices.iter().enumerate() {
            if registry_indices.is_empty() {
                continue;
            }
            let entries = registry_indices
                .iter()
                .map(|&registry_index| {
                    let source_entry = self.params.shard_registry[registry_index];
                    (
                        registry_index,
                        crate::ShardRegEntry {
                            a: source_entry.a * twist,
                            z: -source_entry.z,
                            key: crate::ShardKey {
                                a_bytes: crate::encode_value(source_entry.a * twist),
                                z_bytes: crate::encode_value(-source_entry.z),
                            },
                        },
                    )
                })
                .collect::<Vec<_>>();
            let mut by_index = registry_indices
                .iter()
                .map(|&registry_index| (registry_index, Vec::with_capacity(segment_count)))
                .collect::<BTreeMap<usize, Vec<Vec<Fp>>>>();
            for seg in 0..segment_count {
                let repaired = self.repair_shard_entries_from_rows_correcting(
                    seg,
                    &entries,
                    max_errors,
                    bad_servers,
                )?;
                for &registry_index in registry_indices {
                    let values = repaired
                        .get(&registry_index)
                        .ok_or(ClientError::NotEnoughShards)?;
                    by_index
                        .get_mut(&registry_index)
                        .expect("initialized partner entry")
                        .push(values.clone());
                }
            }
            per_server[server_pos] = by_index
                .into_iter()
                .map(|(registry_index, segments)| PartnerShardMigration {
                    registry_index,
                    segments,
                })
                .collect();
        }
        Ok(per_server)
    }

    fn kappa_halve_has_local_partner_shards(&self, target_s: usize) -> bool {
        let _ = target_s;
        if target_s != self.params.s * 2 || !registry_supports_sibling_kappa_halve(&self.params) {
            return false;
        }
        self.server_owned_indices.iter().all(|indices| {
            let present: HashSet<_> = indices.iter().copied().collect();
            indices.iter().all(|&registry_index| {
                let sibling = if registry_index % 2 == 0 {
                    registry_index + 1
                } else {
                    registry_index - 1
                };
                present.contains(&sibling)
            })
        })
    }

    fn collect_helper_rows_for_kappa_halve(&self) -> Result<Vec<HelperRowMigration>, ClientError> {
        const GROW_HELPER_SEGMENT_BATCH: usize = 16;

        let segment_count =
            self.len
                .div_ceil(public_values_per_segment(&self.params) as u64) as usize;
        if segment_count == 0 {
            return Ok(Vec::new());
        }
        let discovery = self.collect_rows_matrix(&[(0, vec![self.params.row_eval_points[0]])])?;
        let mut helper_row_indices = Vec::new();
        let mut seen = HashSet::new();
        for row in discovery
            .into_iter()
            .next()
            .into_iter()
            .flat_map(|batches| batches.into_iter().flatten())
        {
            let row_label_bytes = encode_value(row.row_label);
            let Some(row_label_index) = self.params.row_label_index_for_bytes(row_label_bytes)
            else {
                continue;
            };
            if seen.insert(row_label_index) {
                helper_row_indices.push(row_label_index);
                if helper_row_indices.len() >= self.params.s {
                    break;
                }
            }
        }
        if helper_row_indices.len() < self.params.s {
            return Err(ClientError::NotEnoughShards);
        }

        let mut helper_rows: Vec<_> = helper_row_indices
            .iter()
            .map(|&row_label_index| HelperRowMigration {
                row_label: self.params.row_labels[row_label_index],
                segments: Vec::with_capacity(segment_count),
            })
            .collect();

        let helper_row_positions: BTreeMap<_, _> = helper_row_indices
            .iter()
            .copied()
            .enumerate()
            .map(|(row_pos, row_label_index)| {
                (self.params.row_label_bytes[row_label_index], row_pos)
            })
            .collect();
        let helper_row_label_bytes: Vec<_> = helper_row_indices
            .iter()
            .copied()
            .map(|row_label_index| self.params.row_label_bytes[row_label_index])
            .collect();
        let row_eval_points = self.params.row_eval_points.clone();
        let helper_count = helper_rows.len();

        for seg_start in (0..segment_count).step_by(GROW_HELPER_SEGMENT_BATCH) {
            let seg_end = (seg_start + GROW_HELPER_SEGMENT_BATCH).min(segment_count);
            let requests: Vec<_> = (seg_start..seg_end)
                .map(|seg| (seg, row_eval_points.clone()))
                .collect();
            let rows_by_request =
                self.collect_selected_rows_matrix(&requests, &helper_row_label_bytes)?;
            if rows_by_request.len() != requests.len() {
                return Err(ClientError::InvalidResponse(
                    "missing kappa-halve helper row batches".to_string(),
                ));
            }
            for batches in rows_by_request {
                if batches.len() != row_eval_points.len() {
                    return Err(ClientError::InvalidResponse(
                        "kappa-halve helper row batch length mismatch".to_string(),
                    ));
                }
                let mut segment_rows = vec![vec![fp_zero(); row_eval_points.len()]; helper_count];
                for (z_pos, rows) in batches.into_iter().enumerate() {
                    let mut found = vec![false; helper_count];
                    for row in rows {
                        let row_label_bytes = encode_value(row.row_label);
                        let Some(&row_pos) = helper_row_positions.get(&row_label_bytes) else {
                            continue;
                        };
                        segment_rows[row_pos][z_pos] = row.value;
                        found[row_pos] = true;
                    }
                    if found.iter().any(|present| !*present) {
                        return Err(ClientError::NotEnoughShards);
                    }
                }
                for (row_pos, values) in segment_rows.into_iter().enumerate() {
                    helper_rows[row_pos].segments.push(values);
                }
            }
        }

        Ok(helper_rows)
    }

    fn select_helper_row_indices_for_kappa_halve(&self) -> Result<Vec<usize>, ClientError> {
        let mut helper_row_indices = Vec::new();
        let mut seen = HashSet::new();
        for owned_indices in &self.server_owned_indices {
            for &registry_index in owned_indices {
                for row_label_index in self
                    .params
                    .row_label_indices_for_registry_index(registry_index)
                {
                    if seen.insert(row_label_index) {
                        helper_row_indices.push(row_label_index);
                        if helper_row_indices.len() >= self.params.s {
                            return Ok(helper_row_indices);
                        }
                    }
                }
            }
        }
        Err(ClientError::NotEnoughShards)
    }

    fn collect_helper_rows_for_kappa_halve_correcting(
        &self,
        max_errors: usize,
        bad_servers: &mut HashSet<usize>,
    ) -> Result<Vec<HelperRowMigration>, ClientError> {
        const GROW_HELPER_SEGMENT_BATCH: usize = 16;

        let segment_count =
            self.len
                .div_ceil(public_values_per_segment(&self.params) as u64) as usize;
        if segment_count == 0 {
            return Ok(Vec::new());
        }
        let helper_row_indices = self.select_helper_row_indices_for_kappa_halve()?;
        let mut helper_rows: Vec<_> = helper_row_indices
            .iter()
            .map(|&row_label_index| HelperRowMigration {
                row_label: self.params.row_labels[row_label_index],
                segments: Vec::with_capacity(segment_count),
            })
            .collect();

        let entries = self.select_row_repair_entries_with_target(error_correcting_share_target(
            self.params.need,
            max_errors,
        ))?;
        let requested_by_server = self.plan_requested_col_labels_by_server(&entries)?;
        for seg_start in (0..segment_count).step_by(GROW_HELPER_SEGMENT_BATCH) {
            let seg_end = (seg_start + GROW_HELPER_SEGMENT_BATCH).min(segment_count);
            let segs = (seg_start..seg_end).collect::<Vec<_>>();
            let helper_evals_by_seg = self
                .collect_shard_evaluations_matrix_from_requested_by_server_correcting(
                    &segs,
                    &requested_by_server,
                    &helper_row_indices,
                    bad_servers,
                )?;
            for helper_evals in helper_evals_by_seg {
                let repaired_rows = self.recover_rows_from_sourced_helper_evaluations_correcting(
                    &helper_evals,
                    &entries,
                    &helper_row_indices,
                    max_errors,
                    bad_servers,
                )?;
                if repaired_rows.len() != helper_rows.len() {
                    return Err(ClientError::NotEnoughShards);
                }
                for (row_pos, (_, values)) in repaired_rows.into_iter().enumerate() {
                    helper_rows[row_pos].segments.push(values);
                }
            }
        }
        Ok(helper_rows)
    }

    fn collect_helper_shards_for_kappa_double(
        &self,
        new_params: &CodedArrayParams,
    ) -> Result<Vec<Vec<PartnerShardMigration>>, ClientError> {
        let mut helper_targets_by_server = vec![Vec::new(); self.servers.len()];
        for server_pos in 0..self.servers.len() {
            let local_indices = new_params
                .party_registry_indices
                .get(server_pos)
                .cloned()
                .unwrap_or_default();
            if local_indices.len() >= new_params.need {
                continue;
            }
            let local_index_set: HashSet<_> = local_indices.into_iter().collect();
            let required_helpers = new_params.need.saturating_sub(local_index_set.len());
            for target_registry_index in 0..new_params.shard_registry.len() {
                if local_index_set.contains(&target_registry_index) {
                    continue;
                }
                helper_targets_by_server[server_pos].push(target_registry_index);
                if helper_targets_by_server[server_pos].len() >= required_helpers {
                    break;
                }
            }
            if local_index_set.len() + helper_targets_by_server[server_pos].len() < new_params.need
            {
                return Err(ClientError::NotEnoughShards);
            }
        }

        if helper_targets_by_server
            .iter()
            .all(|targets| targets.is_empty())
        {
            return Ok(vec![Vec::new(); self.servers.len()]);
        }

        let mut target_destinations_by_parent = BTreeMap::<usize, Vec<(usize, usize)>>::new();
        for (server_pos, helper_targets) in helper_targets_by_server.iter().enumerate() {
            for &target_registry_index in helper_targets {
                target_destinations_by_parent
                    .entry(kappa_halve_parent_registry_index(target_registry_index))
                    .or_default()
                    .push((server_pos, target_registry_index));
            }
        }

        let source_parent_indices: Vec<_> = target_destinations_by_parent.keys().copied().collect();
        let segment_count =
            self.len
                .div_ceil(public_values_per_segment(&self.params) as u64) as usize;
        let mut helpers_by_server =
            vec![BTreeMap::<usize, Vec<Vec<Fp>>>::new(); self.servers.len()];
        for (server_pos, helper_targets) in helper_targets_by_server.iter().enumerate() {
            for &target_registry_index in helper_targets {
                helpers_by_server[server_pos]
                    .entry(target_registry_index)
                    .or_insert_with(|| Vec::with_capacity(segment_count));
            }
        }

        for seg in 0..segment_count {
            let source_shards =
                self.collect_owned_shards_from_servers(seg, &source_parent_indices)?;
            for (&source_parent_index, destinations) in &target_destinations_by_parent {
                let old_values = source_shards
                    .get(&source_parent_index)
                    .ok_or(ClientError::NotEnoughShards)?;
                let (left_child, right_child) = split_shard_for_kappa_double(old_values);
                for &(server_pos, target_registry_index) in destinations {
                    let target_values = if target_registry_index % 2 == 0 {
                        left_child.clone()
                    } else {
                        right_child.clone()
                    };
                    helpers_by_server[server_pos]
                        .entry(target_registry_index)
                        .or_default()
                        .push(target_values);
                }
            }
        }

        let mut per_server = vec![Vec::new(); self.servers.len()];
        for (server_pos, helper_columns) in helpers_by_server.into_iter().enumerate() {
            per_server[server_pos] = helper_columns
                .into_iter()
                .map(|(registry_index, segments)| PartnerShardMigration {
                    registry_index,
                    segments,
                })
                .collect();
        }
        Ok(per_server)
    }

    fn collect_helper_shards_for_kappa_double_correcting(
        &self,
        new_params: &CodedArrayParams,
        max_errors: usize,
        bad_servers: &mut HashSet<usize>,
    ) -> Result<Vec<Vec<PartnerShardMigration>>, ClientError> {
        let mut helper_targets_by_server = vec![Vec::new(); self.servers.len()];
        for server_pos in 0..self.servers.len() {
            let local_indices = new_params
                .party_registry_indices
                .get(server_pos)
                .cloned()
                .unwrap_or_default();
            if local_indices.len() >= new_params.need {
                continue;
            }
            let local_index_set: HashSet<_> = local_indices.into_iter().collect();
            let required_helpers = new_params.need.saturating_sub(local_index_set.len());
            for target_registry_index in 0..new_params.shard_registry.len() {
                if local_index_set.contains(&target_registry_index) {
                    continue;
                }
                helper_targets_by_server[server_pos].push(target_registry_index);
                if helper_targets_by_server[server_pos].len() >= required_helpers {
                    break;
                }
            }
            if local_index_set.len() + helper_targets_by_server[server_pos].len() < new_params.need
            {
                return Err(ClientError::NotEnoughShards);
            }
        }

        if helper_targets_by_server
            .iter()
            .all(|targets| targets.is_empty())
        {
            return Ok(vec![Vec::new(); self.servers.len()]);
        }

        let mut target_destinations_by_parent = BTreeMap::<usize, Vec<(usize, usize)>>::new();
        for (server_pos, helper_targets) in helper_targets_by_server.iter().enumerate() {
            for &target_registry_index in helper_targets {
                target_destinations_by_parent
                    .entry(kappa_halve_parent_registry_index(target_registry_index))
                    .or_default()
                    .push((server_pos, target_registry_index));
            }
        }

        let source_parent_indices: Vec<_> = target_destinations_by_parent.keys().copied().collect();
        let segment_count =
            self.len
                .div_ceil(public_values_per_segment(&self.params) as u64) as usize;
        let mut helpers_by_server =
            vec![BTreeMap::<usize, Vec<Vec<Fp>>>::new(); self.servers.len()];
        for (server_pos, helper_targets) in helper_targets_by_server.iter().enumerate() {
            for &target_registry_index in helper_targets {
                helpers_by_server[server_pos]
                    .entry(target_registry_index)
                    .or_insert_with(|| Vec::with_capacity(segment_count));
            }
        }

        for seg in 0..segment_count {
            let source_shards = self.collect_owned_shards_from_servers_correcting(
                seg,
                &source_parent_indices,
                max_errors,
                bad_servers,
            )?;
            for (&source_parent_index, destinations) in &target_destinations_by_parent {
                let old_values = source_shards
                    .get(&source_parent_index)
                    .ok_or(ClientError::NotEnoughShards)?;
                let (left_child, right_child) = split_shard_for_kappa_double(old_values);
                for &(server_pos, target_registry_index) in destinations {
                    let target_values = if target_registry_index % 2 == 0 {
                        left_child.clone()
                    } else {
                        right_child.clone()
                    };
                    helpers_by_server[server_pos]
                        .entry(target_registry_index)
                        .or_default()
                        .push(target_values);
                }
            }
        }

        let mut per_server = vec![Vec::new(); self.servers.len()];
        for (server_pos, helper_columns) in helpers_by_server.into_iter().enumerate() {
            per_server[server_pos] = helper_columns
                .into_iter()
                .map(|(registry_index, segments)| PartnerShardMigration {
                    registry_index,
                    segments,
                })
                .collect();
        }
        Ok(per_server)
    }

    fn collect_helper_evals_for_one_claim_left_kappa_double(
        &self,
        new_params: &CodedArrayParams,
        target_owned_indices: &[Vec<usize>],
    ) -> Result<Vec<Vec<KappaDoubleRowEvalMigration>>, ClientError> {
        if target_owned_indices.len() != self.servers.len() {
            return Err(ClientError::InvalidConfig(format!(
                "one-claim kappa-double target has {} servers, client has {}",
                target_owned_indices.len(),
                self.servers.len()
            )));
        }

        let mut available_targets = Vec::new();
        let mut seen_targets = HashSet::new();
        for indices in target_owned_indices {
            for &target_registry_index in indices {
                if target_registry_index >= new_params.shard_registry.len() {
                    return Err(ClientError::InvalidIndex);
                }
                if seen_targets.insert(target_registry_index) {
                    available_targets.push(target_registry_index);
                }
            }
        }
        available_targets.sort_unstable();

        let mut helper_targets_by_server = vec![Vec::new(); self.servers.len()];
        for (server_pos, local_indices) in target_owned_indices.iter().enumerate() {
            let local_index_set: HashSet<_> = local_indices.iter().copied().collect();
            if local_index_set.len() >= new_params.need {
                continue;
            }
            let required_helpers = new_params.need - local_index_set.len();
            for &target_registry_index in &available_targets {
                if local_index_set.contains(&target_registry_index) {
                    continue;
                }
                helper_targets_by_server[server_pos].push(target_registry_index);
                if helper_targets_by_server[server_pos].len() >= required_helpers {
                    break;
                }
            }
            if local_index_set.len() + helper_targets_by_server[server_pos].len() < new_params.need
            {
                return Err(ClientError::NotEnoughShards);
            }
        }

        if helper_targets_by_server
            .iter()
            .all(|targets| targets.is_empty())
        {
            return Ok(vec![Vec::new(); self.servers.len()]);
        }

        let mut owner_by_source_index = std::collections::HashMap::new();
        for (server_pos, owned_indices) in self.server_owned_indices.iter().enumerate() {
            for &registry_index in owned_indices {
                owner_by_source_index
                    .entry(registry_index)
                    .or_insert(server_pos);
            }
        }

        let segment_count =
            self.len
                .div_ceil(public_values_per_segment(&self.params) as u64) as usize;
        let segs: Vec<_> = (0..segment_count).collect();
        let mut per_server = vec![Vec::new(); self.servers.len()];

        for (dest_server_pos, helper_targets) in helper_targets_by_server.iter().enumerate() {
            if helper_targets.is_empty() {
                continue;
            }
            let mut row_label_indices = Vec::new();
            let mut seen_rows = HashSet::new();
            for &target_registry_index in &target_owned_indices[dest_server_pos] {
                let Some(claim) =
                    ClaimBundle::exact_for_registry_index(new_params, target_registry_index)
                else {
                    return Err(ClientError::InvalidIndex);
                };
                let Some(indices) = claim.row_label_indices_slice(new_params) else {
                    return Err(ClientError::InvalidIndex);
                };
                for &row_label_index in indices {
                    if seen_rows.insert(row_label_index) {
                        row_label_indices.push(row_label_index);
                    }
                }
            }
            row_label_indices.sort_unstable();
            let row_labels: Vec<_> = row_label_indices
                .iter()
                .map(|&row_label_index| {
                    new_params
                        .row_labels
                        .get(row_label_index)
                        .copied()
                        .ok_or(ClientError::InvalidIndex)
                })
                .collect::<Result<_, _>>()?;

            let mut requested_by_helper_server = vec![Vec::new(); self.servers.len()];
            for &target_registry_index in helper_targets {
                let source_parent = kappa_halve_parent_registry_index(target_registry_index);
                let Some(&helper_server_pos) = owner_by_source_index.get(&source_parent) else {
                    return Err(ClientError::NotEnoughShards);
                };
                requested_by_helper_server[helper_server_pos].push(target_registry_index);
            }

            let helper_servers: Vec<_> = requested_by_helper_server
                .iter()
                .enumerate()
                .filter(|(_, requested)| !requested.is_empty())
                .filter_map(|(server_pos, _)| {
                    self.servers
                        .get(server_pos)
                        .cloned()
                        .map(|server| (server_pos, server))
                })
                .collect();
            let requested_by_helper_server = Arc::new(requested_by_helper_server);
            let segs = Arc::new(segs.clone());
            let row_labels = Arc::new(row_labels);
            let transport = self.transport.clone();
            let target_s = new_params.s;
            let fetches = spawn_parallel_fetches(&helper_servers, {
                let requested_by_helper_server = Arc::clone(&requested_by_helper_server);
                let segs = Arc::clone(&segs);
                let row_labels = Arc::clone(&row_labels);
                move |(helper_server_pos, server)| {
                    let requested = requested_by_helper_server[helper_server_pos].clone();
                    let values = transport.unauth_eval_kappa_double_target_at_row_labels_matrix(
                        &server,
                        target_s,
                        segs.as_slice(),
                        &requested,
                        row_labels.as_slice(),
                    )?;
                    Ok::<_, ClientError>((requested, values))
                }
            });

            let mut helper_map = BTreeMap::<usize, Vec<Vec<Fp>>>::new();
            for _ in 0..helper_servers.len() {
                let Ok((_, reply)) = fetches.recv() else {
                    break;
                };
                let (requested, values_by_seg) = match reply {
                    Ok(reply) => reply,
                    Err(_) => continue,
                };
                if values_by_seg.len() != segment_count {
                    fetches.cancel();
                    return Err(ClientError::InvalidResponse(
                        "kappa-double target eval segment count mismatch".to_string(),
                    ));
                }
                for (target_pos, &target_registry_index) in requested.iter().enumerate() {
                    let mut segments = Vec::with_capacity(segment_count);
                    for values_for_seg in &values_by_seg {
                        if values_for_seg.len() != requested.len() {
                            fetches.cancel();
                            return Err(ClientError::InvalidResponse(
                                "kappa-double target eval target count mismatch".to_string(),
                            ));
                        }
                        let row_values = values_for_seg[target_pos].clone();
                        if row_values.len() != row_label_indices.len() {
                            fetches.cancel();
                            return Err(ClientError::InvalidResponse(
                                "kappa-double target eval row count mismatch".to_string(),
                            ));
                        }
                        segments.push(row_values);
                    }
                    helper_map.insert(target_registry_index, segments);
                }
            }
            if helper_map.len() < helper_targets.len() {
                return Err(ClientError::NotEnoughShards);
            }
            per_server[dest_server_pos] = helper_map
                .into_iter()
                .map(|(registry_index, segments)| KappaDoubleRowEvalMigration {
                    registry_index,
                    segments,
                })
                .collect();
        }
        Ok(per_server)
    }

    fn collect_helper_evals_for_one_claim_left_kappa_double_correcting(
        &self,
        new_params: &CodedArrayParams,
        target_owned_indices: &[Vec<usize>],
        max_errors: usize,
        bad_servers: &mut HashSet<usize>,
    ) -> Result<Vec<Vec<KappaDoubleRowEvalMigration>>, ClientError> {
        if target_owned_indices.len() != self.servers.len() {
            return Err(ClientError::InvalidConfig(format!(
                "one-claim kappa-double target has {} servers, client has {}",
                target_owned_indices.len(),
                self.servers.len()
            )));
        }

        let mut available_targets = Vec::new();
        let mut seen_targets = HashSet::new();
        for indices in target_owned_indices {
            for &target_registry_index in indices {
                if target_registry_index >= new_params.shard_registry.len() {
                    return Err(ClientError::InvalidIndex);
                }
                if seen_targets.insert(target_registry_index) {
                    available_targets.push(target_registry_index);
                }
            }
        }
        available_targets.sort_unstable();

        let mut helper_targets_by_server = vec![Vec::new(); self.servers.len()];
        for (server_pos, local_indices) in target_owned_indices.iter().enumerate() {
            let local_index_set: HashSet<_> = local_indices.iter().copied().collect();
            if local_index_set.len() >= new_params.need {
                continue;
            }
            let required_helpers = new_params.need - local_index_set.len();
            for &target_registry_index in &available_targets {
                if local_index_set.contains(&target_registry_index) {
                    continue;
                }
                helper_targets_by_server[server_pos].push(target_registry_index);
                if helper_targets_by_server[server_pos].len() >= required_helpers {
                    break;
                }
            }
            if local_index_set.len() + helper_targets_by_server[server_pos].len() < new_params.need
            {
                return Err(ClientError::NotEnoughShards);
            }
        }

        if helper_targets_by_server
            .iter()
            .all(|targets| targets.is_empty())
        {
            return Ok(vec![Vec::new(); self.servers.len()]);
        }

        let segment_count =
            self.len
                .div_ceil(public_values_per_segment(&self.params) as u64) as usize;
        let mut per_server_maps = vec![BTreeMap::<usize, Vec<Vec<Fp>>>::new(); self.servers.len()];
        let mut all_source_parents = HashSet::new();
        let mut row_labels_by_dest = Vec::with_capacity(self.servers.len());
        for (dest_server_pos, helper_targets) in helper_targets_by_server.iter().enumerate() {
            if helper_targets.is_empty() {
                row_labels_by_dest.push(Vec::new());
                continue;
            }
            let mut row_label_indices = Vec::new();
            let mut seen_rows = HashSet::new();
            for &target_registry_index in &target_owned_indices[dest_server_pos] {
                let Some(claim) =
                    ClaimBundle::exact_for_registry_index(new_params, target_registry_index)
                else {
                    return Err(ClientError::InvalidIndex);
                };
                let Some(indices) = claim.row_label_indices_slice(new_params) else {
                    return Err(ClientError::InvalidIndex);
                };
                for &row_label_index in indices {
                    if seen_rows.insert(row_label_index) {
                        row_label_indices.push(row_label_index);
                    }
                }
            }
            row_label_indices.sort_unstable();
            let row_labels = row_label_indices
                .iter()
                .map(|&row_label_index| {
                    new_params
                        .row_labels
                        .get(row_label_index)
                        .copied()
                        .ok_or(ClientError::InvalidIndex)
                })
                .collect::<Result<Vec<_>, _>>()?;
            for &target_registry_index in helper_targets {
                all_source_parents.insert(kappa_halve_parent_registry_index(target_registry_index));
                per_server_maps[dest_server_pos]
                    .entry(target_registry_index)
                    .or_insert_with(|| Vec::with_capacity(segment_count));
            }
            row_labels_by_dest.push(row_labels);
        }

        let source_parent_indices = all_source_parents.into_iter().collect::<Vec<_>>();
        for seg in 0..segment_count {
            let source_shards = self.collect_owned_shards_from_servers_correcting(
                seg,
                &source_parent_indices,
                max_errors,
                bad_servers,
            )?;
            let mut child_by_target = HashMap::<usize, Vec<Fp>>::new();
            for &source_parent in &source_parent_indices {
                let old_values = source_shards
                    .get(&source_parent)
                    .ok_or(ClientError::NotEnoughShards)?;
                let (left_child, right_child) = split_shard_for_kappa_double(old_values);
                child_by_target.insert(2 * source_parent, left_child);
                child_by_target.insert(2 * source_parent + 1, right_child);
            }
            for (dest_server_pos, helper_targets) in helper_targets_by_server.iter().enumerate() {
                let row_labels = &row_labels_by_dest[dest_server_pos];
                for &target_registry_index in helper_targets {
                    let target_entry = new_params.shard_registry[target_registry_index];
                    let target_values = child_by_target
                        .get(&target_registry_index)
                        .ok_or(ClientError::NotEnoughShards)?;
                    let evals = shard_value_from_vector_batch(
                        new_params,
                        target_entry,
                        target_values,
                        row_labels,
                    )
                    .into_iter()
                    .map(|value| value.ok_or(ClientError::NotEnoughShards))
                    .collect::<Result<Vec<_>, _>>()?;
                    per_server_maps[dest_server_pos]
                        .entry(target_registry_index)
                        .or_default()
                        .push(evals);
                }
            }
        }

        Ok(per_server_maps
            .into_iter()
            .map(|helper_map| {
                helper_map
                    .into_iter()
                    .map(|(registry_index, segments)| KappaDoubleRowEvalMigration {
                        registry_index,
                        segments,
                    })
                    .collect()
            })
            .collect())
    }

    fn plan_requested_shards_by_server(
        &self,
        registry_indices: &[usize],
    ) -> Result<Vec<Vec<usize>>, ClientError> {
        if registry_indices.is_empty() {
            return Ok(vec![Vec::new(); self.servers.len()]);
        }
        let mut owner_by_index = std::collections::HashMap::new();
        for (server_pos, owned_indices) in self.server_owned_indices.iter().enumerate() {
            for &registry_index in owned_indices {
                owner_by_index.entry(registry_index).or_insert(server_pos);
            }
        }
        let mut requested_by_server = vec![Vec::new(); self.servers.len()];
        let mut seen = HashSet::new();
        for &registry_index in registry_indices {
            if !seen.insert(registry_index) {
                continue;
            }
            if registry_index >= self.params.shard_registry.len() {
                return Err(ClientError::InvalidIndex);
            }
            if let Some(&server_pos) = owner_by_index.get(&registry_index) {
                requested_by_server[server_pos].push(registry_index);
            }
        }
        Ok(requested_by_server)
    }

    fn plan_requested_col_labels_by_server(
        &self,
        entries: &[(usize, ShardRegEntry)],
    ) -> Result<Vec<Vec<Fp>>, ClientError> {
        if entries.is_empty() {
            return Ok(vec![Vec::new(); self.servers.len()]);
        }
        let mut owner_by_index = std::collections::HashMap::new();
        for (server_pos, owned_indices) in self.server_owned_indices.iter().enumerate() {
            for &registry_index in owned_indices {
                owner_by_index.entry(registry_index).or_insert(server_pos);
            }
        }
        let mut requested_by_server = vec![Vec::new(); self.servers.len()];
        let mut seen = HashSet::new();
        for &(registry_index, entry) in entries {
            if !seen.insert(entry.key.z_bytes) {
                continue;
            }
            if let Some(&server_pos) = owner_by_index.get(&registry_index) {
                requested_by_server[server_pos].push(entry.z);
            }
        }
        Ok(requested_by_server)
    }

    fn collect_shard_evaluations_from_requested_by_server(
        &self,
        seg: usize,
        requested_by_server: &[Vec<Fp>],
        row_label_indices: &[usize],
    ) -> Result<std::collections::HashMap<Value256, Vec<Fp>>, ClientError> {
        if requested_by_server
            .iter()
            .all(|requested| requested.is_empty())
        {
            return Ok(std::collections::HashMap::new());
        }
        let mut out = std::collections::HashMap::new();
        let requested_by_server = Arc::new(requested_by_server.to_vec());
        let servers: Vec<_> = self
            .servers
            .iter()
            .cloned()
            .enumerate()
            .filter(|(server_pos, _)| !requested_by_server[*server_pos].is_empty())
            .collect();
        let transport = self.transport.clone();
        let target_count = requested_by_server
            .iter()
            .map(|requested| requested.len())
            .sum();
        let row_labels: Vec<_> = row_label_indices
            .iter()
            .map(|&row_label_index| {
                self.params
                    .row_labels
                    .get(row_label_index)
                    .copied()
                    .ok_or(ClientError::InvalidIndex)
            })
            .collect::<Result<_, _>>()?;
        let fetches = spawn_parallel_fetches(&servers, {
            let requested_by_server = Arc::clone(&requested_by_server);
            let row_labels = row_labels.clone();
            move |(server_pos, server)| {
                let requested = requested_by_server[server_pos].clone();
                let values = transport.unauth_eval_shards_at_row_labels(
                    &server,
                    seg,
                    &requested,
                    &row_labels,
                )?;
                Ok::<_, ClientError>((server_pos, requested, values))
            }
        });
        for _ in 0..servers.len() {
            if out.len() >= target_count {
                fetches.cancel();
                break;
            }
            let Ok((_, reply)) = fetches.recv() else {
                break;
            };
            let (_, requested, values) = match reply {
                Ok(reply) => reply,
                Err(_) => continue,
            };
            if values.len() != requested.len() {
                fetches.cancel();
                return Err(ClientError::InvalidResponse(
                    "unauth shard-eval batch length mismatch".to_string(),
                ));
            }
            for (col_label, eval_values) in requested.into_iter().zip(values.into_iter()) {
                if eval_values.len() != row_label_indices.len() {
                    fetches.cancel();
                    return Err(ClientError::InvalidResponse(
                        "unauth shard-eval row count mismatch".to_string(),
                    ));
                }
                out.insert(encode_value(col_label), eval_values);
            }
        }
        Ok(out)
    }

    fn collect_shard_evaluations_from_requested_by_server_correcting(
        &self,
        seg: usize,
        requested_by_server: &[Vec<Fp>],
        row_label_indices: &[usize],
        bad_servers: &HashSet<usize>,
    ) -> Result<HashMap<Value256, SourcedEvalValues>, ClientError> {
        if requested_by_server
            .iter()
            .all(|requested| requested.is_empty())
        {
            return Ok(HashMap::new());
        }
        let mut out = HashMap::new();
        let requested_by_server = Arc::new(requested_by_server.to_vec());
        let servers: Vec<_> = self
            .servers
            .iter()
            .cloned()
            .enumerate()
            .filter(|(server_pos, _)| !bad_servers.contains(server_pos))
            .filter(|(server_pos, _)| !requested_by_server[*server_pos].is_empty())
            .collect();
        let transport = self.transport.clone();
        let target_count = requested_by_server
            .iter()
            .enumerate()
            .filter(|(server_pos, _)| !bad_servers.contains(server_pos))
            .map(|(_, requested)| requested.len())
            .sum();
        let row_labels: Vec<_> = row_label_indices
            .iter()
            .map(|&row_label_index| {
                self.params
                    .row_labels
                    .get(row_label_index)
                    .copied()
                    .ok_or(ClientError::InvalidIndex)
            })
            .collect::<Result<_, _>>()?;
        let fetches = spawn_parallel_fetches(&servers, {
            let requested_by_server = Arc::clone(&requested_by_server);
            let row_labels = row_labels.clone();
            move |(server_pos, server)| {
                let requested = requested_by_server[server_pos].clone();
                let values = transport.unauth_eval_shards_at_row_labels(
                    &server,
                    seg,
                    &requested,
                    &row_labels,
                )?;
                Ok::<_, ClientError>((server_pos, requested, values))
            }
        });
        for _ in 0..servers.len() {
            if out.len() >= target_count {
                fetches.cancel();
                break;
            }
            let Ok((_, reply)) = fetches.recv() else {
                break;
            };
            let (server_pos, requested, values) = match reply {
                Ok(reply) => reply,
                Err(_) => continue,
            };
            if values.len() != requested.len() {
                fetches.cancel();
                return Err(ClientError::InvalidResponse(
                    "unauth shard-eval batch length mismatch".to_string(),
                ));
            }
            for (col_label, eval_values) in requested.into_iter().zip(values.into_iter()) {
                if eval_values.len() != row_label_indices.len() {
                    fetches.cancel();
                    return Err(ClientError::InvalidResponse(
                        "unauth shard-eval row count mismatch".to_string(),
                    ));
                }
                out.insert(
                    encode_value(col_label),
                    SourcedEvalValues {
                        values: eval_values,
                        server_pos,
                    },
                );
            }
        }
        Ok(out)
    }

    fn collect_shard_evaluations_matrix_from_requested_by_server(
        &self,
        segs: &[usize],
        requested_by_server: &[Vec<Fp>],
        row_label_indices: &[usize],
    ) -> Result<Vec<std::collections::HashMap<Value256, Vec<Fp>>>, ClientError> {
        if segs.is_empty() {
            return Ok(Vec::new());
        }
        if requested_by_server
            .iter()
            .all(|requested| requested.is_empty())
        {
            return Ok(vec![std::collections::HashMap::new(); segs.len()]);
        }
        let mut out = vec![std::collections::HashMap::new(); segs.len()];
        let requested_by_server = Arc::new(requested_by_server.to_vec());
        let segs = Arc::new(segs.to_vec());
        let servers: Vec<_> = self
            .servers
            .iter()
            .cloned()
            .enumerate()
            .filter(|(server_pos, _)| !requested_by_server[*server_pos].is_empty())
            .collect();
        let transport = self.transport.clone();
        let target_count = requested_by_server
            .iter()
            .map(|requested| requested.len())
            .sum();
        let row_labels: Vec<_> = row_label_indices
            .iter()
            .map(|&row_label_index| {
                self.params
                    .row_labels
                    .get(row_label_index)
                    .copied()
                    .ok_or(ClientError::InvalidIndex)
            })
            .collect::<Result<_, _>>()?;
        let fetches = spawn_parallel_fetches(&servers, {
            let requested_by_server = Arc::clone(&requested_by_server);
            let segs = Arc::clone(&segs);
            let row_labels = row_labels.clone();
            move |(server_pos, server)| {
                let requested = requested_by_server[server_pos].clone();
                let values = transport.unauth_eval_shards_at_row_labels_matrix(
                    &server,
                    segs.as_slice(),
                    &requested,
                    &row_labels,
                )?;
                Ok::<_, ClientError>((requested, values))
            }
        });
        for _ in 0..servers.len() {
            if out.iter().all(|by_label| by_label.len() >= target_count) {
                fetches.cancel();
                break;
            }
            let Ok((_, reply)) = fetches.recv() else {
                break;
            };
            let (requested, values_by_seg) = match reply {
                Ok(reply) => reply,
                Err(_) => continue,
            };
            if values_by_seg.len() != segs.len() {
                fetches.cancel();
                return Err(ClientError::InvalidResponse(
                    "unauth shard-eval matrix segment count mismatch".to_string(),
                ));
            }
            for (seg_pos, values) in values_by_seg.into_iter().enumerate() {
                if values.len() != requested.len() {
                    fetches.cancel();
                    return Err(ClientError::InvalidResponse(
                        "unauth shard-eval matrix batch length mismatch".to_string(),
                    ));
                }
                for (col_label, eval_values) in requested.iter().copied().zip(values.into_iter()) {
                    if eval_values.len() != row_label_indices.len() {
                        fetches.cancel();
                        return Err(ClientError::InvalidResponse(
                            "unauth shard-eval matrix row count mismatch".to_string(),
                        ));
                    }
                    out[seg_pos].insert(encode_value(col_label), eval_values);
                }
            }
        }
        Ok(out)
    }

    fn collect_shard_evaluations_matrix_from_requested_by_server_correcting(
        &self,
        segs: &[usize],
        requested_by_server: &[Vec<Fp>],
        row_label_indices: &[usize],
        bad_servers: &HashSet<usize>,
    ) -> Result<Vec<HashMap<Value256, SourcedEvalValues>>, ClientError> {
        if segs.is_empty() {
            return Ok(Vec::new());
        }
        if requested_by_server
            .iter()
            .all(|requested| requested.is_empty())
        {
            return Ok(vec![HashMap::new(); segs.len()]);
        }
        let mut out = vec![HashMap::new(); segs.len()];
        let requested_by_server = Arc::new(requested_by_server.to_vec());
        let segs = Arc::new(segs.to_vec());
        let servers: Vec<_> = self
            .servers
            .iter()
            .cloned()
            .enumerate()
            .filter(|(server_pos, _)| !bad_servers.contains(server_pos))
            .filter(|(server_pos, _)| !requested_by_server[*server_pos].is_empty())
            .collect();
        let transport = self.transport.clone();
        let target_count = requested_by_server
            .iter()
            .enumerate()
            .filter(|(server_pos, _)| !bad_servers.contains(server_pos))
            .map(|(_, requested)| requested.len())
            .sum();
        let row_labels: Vec<_> = row_label_indices
            .iter()
            .map(|&row_label_index| {
                self.params
                    .row_labels
                    .get(row_label_index)
                    .copied()
                    .ok_or(ClientError::InvalidIndex)
            })
            .collect::<Result<_, _>>()?;
        let fetches = spawn_parallel_fetches(&servers, {
            let requested_by_server = Arc::clone(&requested_by_server);
            let segs = Arc::clone(&segs);
            let row_labels = row_labels.clone();
            move |(server_pos, server)| {
                let requested = requested_by_server[server_pos].clone();
                let values = transport.unauth_eval_shards_at_row_labels_matrix(
                    &server,
                    segs.as_slice(),
                    &requested,
                    &row_labels,
                )?;
                Ok::<_, ClientError>((server_pos, requested, values))
            }
        });
        for _ in 0..servers.len() {
            if out.iter().all(|by_label| by_label.len() >= target_count) {
                fetches.cancel();
                break;
            }
            let Ok((_, reply)) = fetches.recv() else {
                break;
            };
            let (server_pos, requested, values_by_seg) = match reply {
                Ok(reply) => reply,
                Err(_) => continue,
            };
            if values_by_seg.len() != segs.len() {
                fetches.cancel();
                return Err(ClientError::InvalidResponse(
                    "unauth shard-eval matrix segment count mismatch".to_string(),
                ));
            }
            for (seg_pos, values) in values_by_seg.into_iter().enumerate() {
                if values.len() != requested.len() {
                    fetches.cancel();
                    return Err(ClientError::InvalidResponse(
                        "unauth shard-eval matrix batch length mismatch".to_string(),
                    ));
                }
                for (col_label, eval_values) in requested.iter().copied().zip(values.into_iter()) {
                    if eval_values.len() != row_label_indices.len() {
                        fetches.cancel();
                        return Err(ClientError::InvalidResponse(
                            "unauth shard-eval matrix row count mismatch".to_string(),
                        ));
                    }
                    out[seg_pos].insert(
                        encode_value(col_label),
                        SourcedEvalValues {
                            values: eval_values,
                            server_pos,
                        },
                    );
                }
            }
        }
        Ok(out)
    }

    fn collect_owned_shards_from_requested_by_server(
        &self,
        seg: usize,
        requested_by_server: &[Vec<usize>],
    ) -> Result<std::collections::HashMap<usize, Vec<Fp>>, ClientError> {
        if requested_by_server
            .iter()
            .all(|requested| requested.is_empty())
        {
            return Ok(std::collections::HashMap::new());
        }
        let mut out = std::collections::HashMap::new();
        let requested_by_server = Arc::new(requested_by_server.to_vec());
        let servers: Vec<_> = self
            .servers
            .iter()
            .cloned()
            .enumerate()
            .filter(|(server_pos, _)| !requested_by_server[*server_pos].is_empty())
            .collect();
        let transport = self.transport.clone();
        let target_count = requested_by_server
            .iter()
            .map(|requested| requested.len())
            .sum();
        let fetches = spawn_parallel_fetches(&servers, {
            let requested_by_server = Arc::clone(&requested_by_server);
            move |(server_pos, server)| {
                let requested = requested_by_server[server_pos].clone();
                let values = transport.unauth_repair_shard_batch(&server, seg, &requested)?;
                Ok::<_, ClientError>((server_pos, requested, values))
            }
        });
        for _ in 0..servers.len() {
            if out.len() >= target_count {
                fetches.cancel();
                break;
            }
            let Ok((_, reply)) = fetches.recv() else {
                break;
            };
            let (_, requested, values) = match reply {
                Ok(reply) => reply,
                Err(_) => continue,
            };
            if values.len() != requested.len() {
                fetches.cancel();
                return Err(ClientError::InvalidResponse(
                    "unauth repair shard batch length mismatch".to_string(),
                ));
            }
            for (registry_index, shard_values) in requested.into_iter().zip(values.into_iter()) {
                out.insert(registry_index, shard_values);
            }
        }
        Ok(out)
    }

    fn collect_owned_shards_from_servers(
        &self,
        seg: usize,
        registry_indices: &[usize],
    ) -> Result<std::collections::HashMap<usize, Vec<Fp>>, ClientError> {
        let requested_by_server = self.plan_requested_shards_by_server(registry_indices)?;
        self.collect_owned_shards_from_requested_by_server(seg, &requested_by_server)
    }

    fn collect_owned_shards_from_servers_correcting(
        &self,
        seg: usize,
        registry_indices: &[usize],
        max_errors: usize,
        bad_servers: &mut HashSet<usize>,
    ) -> Result<HashMap<usize, Vec<Fp>>, ClientError> {
        if max_errors == 0 {
            return self
                .collect_owned_shards_from_servers(seg, registry_indices)
                .map(|values| values.into_iter().collect());
        }
        self.repair_shards_from_rows_correcting(seg, registry_indices, max_errors, bad_servers)
    }

    fn repair_shards_from_rows(
        &self,
        seg: usize,
        registry_indices: &[usize],
    ) -> Result<std::collections::HashMap<usize, Vec<Fp>>, ClientError> {
        if registry_indices.is_empty() {
            return Ok(std::collections::HashMap::new());
        }
        let mut entries = Vec::with_capacity(registry_indices.len());
        let mut z_evals = Vec::with_capacity(registry_indices.len());
        let mut seen = HashSet::new();
        for &registry_index in registry_indices {
            if !seen.insert(registry_index) {
                continue;
            }
            let Some(&entry) = self.params.shard_registry.get(registry_index) else {
                return Err(ClientError::InvalidIndex);
            };
            entries.push((registry_index, entry));
            z_evals.push(entry.z);
        }
        let rows_by_z = self.collect_rows_at_zs(seg, &z_evals)?;
        let mut out = std::collections::HashMap::new();
        for ((registry_index, entry), rows) in entries.into_iter().zip(rows_by_z.into_iter()) {
            let values = decode_primary_shard_from_rows(&self.params, entry, &rows)
                .map_err(|_| ClientError::NotEnoughShards)?;
            out.insert(registry_index, values);
        }
        Ok(out)
    }

    fn repair_shard_entries_from_rows_correcting(
        &self,
        seg: usize,
        entries: &[(usize, ShardRegEntry)],
        max_errors: usize,
        bad_servers: &mut HashSet<usize>,
    ) -> Result<HashMap<usize, Vec<Fp>>, ClientError> {
        if entries.is_empty() {
            return Ok(HashMap::new());
        }
        let z_evals = entries.iter().map(|(_, entry)| entry.z).collect::<Vec<_>>();
        let rows_by_z =
            self.collect_rows_at_zs_correcting(seg, &z_evals, max_errors, bad_servers)?;
        let all_server_positions = (0..self.servers.len()).collect::<Vec<_>>();
        let mut out = HashMap::new();
        for ((registry_index, entry), rows) in entries.iter().copied().zip(rows_by_z.into_iter()) {
            let known_bad_points =
                self.known_bad_row_points_for_server_positions(&all_server_positions, bad_servers)?;
            let values = self.decode_primary_shard_from_sourced_rows_correcting(
                entry,
                &rows,
                max_errors,
                known_bad_points,
                bad_servers,
            )?;
            out.insert(registry_index, values);
        }
        Ok(out)
    }

    fn repair_shards_from_rows_correcting(
        &self,
        seg: usize,
        registry_indices: &[usize],
        max_errors: usize,
        bad_servers: &mut HashSet<usize>,
    ) -> Result<HashMap<usize, Vec<Fp>>, ClientError> {
        if registry_indices.is_empty() {
            return Ok(HashMap::new());
        }
        let mut entries = Vec::with_capacity(registry_indices.len());
        let mut seen = HashSet::new();
        for &registry_index in registry_indices {
            if !seen.insert(registry_index) {
                continue;
            }
            let Some(&entry) = self.params.shard_registry.get(registry_index) else {
                return Err(ClientError::InvalidIndex);
            };
            entries.push((registry_index, entry));
        }
        self.repair_shard_entries_from_rows_correcting(seg, &entries, max_errors, bad_servers)
    }

    fn select_row_repair_entries(&self) -> Result<Vec<(usize, ShardRegEntry)>, ClientError> {
        self.select_row_repair_entries_with_target(self.params.need)
    }

    fn select_row_repair_entries_with_target(
        &self,
        target_count: usize,
    ) -> Result<Vec<(usize, ShardRegEntry)>, ClientError> {
        let mut entries = Vec::new();
        let mut seen_keys = HashSet::new();
        let mut owner_by_index = std::collections::HashMap::new();
        for (server_pos, owned_indices) in self.server_owned_indices.iter().enumerate() {
            for &registry_index in owned_indices {
                owner_by_index.entry(registry_index).or_insert(server_pos);
            }
        }
        for registry_index in 0..self.params.shard_registry.len() {
            if !owner_by_index.contains_key(&registry_index) {
                continue;
            }
            let entry = self.params.shard_registry[registry_index];
            if seen_keys.insert(entry.key) {
                entries.push((registry_index, entry));
                if entries.len() >= target_count {
                    break;
                }
            }
        }
        if entries.len() < target_count {
            return Err(ClientError::NotEnoughShards);
        }
        Ok(entries)
    }

    fn select_row_repair_entries_with_target_distinct_servers(
        &self,
        target_count: usize,
    ) -> Result<Vec<(usize, ShardRegEntry)>, ClientError> {
        let mut entries = Vec::new();
        let mut seen_keys = HashSet::new();
        let mut selected_servers = HashSet::new();
        let mut owner_by_index = HashMap::new();
        for (server_pos, owned_indices) in self.server_owned_indices.iter().enumerate() {
            for &registry_index in owned_indices {
                owner_by_index.entry(registry_index).or_insert(server_pos);
            }
        }
        for registry_index in 0..self.params.shard_registry.len() {
            let Some(&server_pos) = owner_by_index.get(&registry_index) else {
                continue;
            };
            if selected_servers.contains(&server_pos) {
                continue;
            }
            let entry = self.params.shard_registry[registry_index];
            if seen_keys.insert(entry.key) {
                entries.push((registry_index, entry));
                selected_servers.insert(server_pos);
                if entries.len() >= target_count {
                    break;
                }
            }
        }
        if entries.len() < target_count {
            return Err(ClientError::NotEnoughShards);
        }
        Ok(entries)
    }

    fn owner_server_pos_for_registry_index(&self, registry_index: usize) -> Option<usize> {
        self.server_owned_indices
            .iter()
            .enumerate()
            .find_map(|(server_pos, owned_indices)| {
                owned_indices
                    .contains(&registry_index)
                    .then_some(server_pos)
            })
    }

    fn known_bad_entry_points(
        &self,
        entries: &[(usize, ShardRegEntry)],
        bad_servers: &HashSet<usize>,
    ) -> Result<usize, ClientError> {
        let mut count = 0usize;
        for &(registry_index, _) in entries {
            let Some(server_pos) = self.owner_server_pos_for_registry_index(registry_index) else {
                return Err(ClientError::NotEnoughShards);
            };
            if bad_servers.contains(&server_pos) {
                count = count.saturating_add(1);
            }
        }
        Ok(count)
    }

    fn known_bad_row_points_for_server_positions(
        &self,
        server_positions: &[usize],
        bad_servers: &HashSet<usize>,
    ) -> Result<usize, ClientError> {
        let mut count = 0usize;
        for &server_pos in server_positions {
            if !bad_servers.contains(&server_pos) {
                continue;
            }
            let Some(owned_indices) = self.server_owned_indices.get(server_pos) else {
                return Err(ClientError::NotEnoughShards);
            };
            let mut rows = HashSet::new();
            for &registry_index in owned_indices {
                if registry_index >= self.params.shard_registry.len() {
                    return Err(ClientError::InvalidIndex);
                }
                for row_label_index in self
                    .params
                    .row_label_indices_for_registry_index(registry_index)
                {
                    rows.insert(row_label_index);
                    if rows.len() >= self.params.rows_per_shard {
                        break;
                    }
                }
                if rows.len() >= self.params.rows_per_shard {
                    break;
                }
            }
            count = count.saturating_add(rows.len());
        }
        Ok(count)
    }

    fn select_primary_repair_server_positions_with_target(
        &self,
        target_row_count: usize,
    ) -> Result<Vec<usize>, ClientError> {
        let mut selected = Vec::new();
        let mut covered_rows = HashSet::new();
        for (server_pos, owned_indices) in self.server_owned_indices.iter().enumerate() {
            let mut contributes = false;
            for &registry_index in owned_indices {
                for row_label_index in self
                    .params
                    .row_label_indices_for_registry_index(registry_index)
                {
                    if covered_rows.insert(row_label_index) {
                        contributes = true;
                    }
                }
            }
            if contributes {
                selected.push(server_pos);
                if covered_rows.len() >= target_row_count {
                    break;
                }
            }
        }
        if covered_rows.len() < target_row_count {
            return Err(ClientError::NotEnoughShards);
        }
        Ok(selected)
    }

    fn select_primary_repair_server_positions_with_target_capped(
        &self,
        target_row_count: usize,
    ) -> Result<Vec<usize>, ClientError> {
        let mut selected = Vec::new();
        let mut covered_rows = HashSet::new();
        for (server_pos, owned_indices) in self.server_owned_indices.iter().enumerate() {
            let mut accepted_from_server = 0usize;
            for &registry_index in owned_indices {
                if accepted_from_server >= self.params.rows_per_shard {
                    break;
                }
                for row_label_index in self
                    .params
                    .row_label_indices_for_registry_index(registry_index)
                {
                    if covered_rows.insert(row_label_index) {
                        accepted_from_server += 1;
                        if accepted_from_server >= self.params.rows_per_shard {
                            break;
                        }
                    }
                }
            }
            if accepted_from_server > 0 {
                selected.push(server_pos);
                if covered_rows.len() >= target_row_count {
                    break;
                }
            }
        }
        if covered_rows.len() < target_row_count {
            return Err(ClientError::NotEnoughShards);
        }
        Ok(selected)
    }

    fn recover_rows_from_helper_evaluations_with_weights(
        &self,
        helper_evals: &std::collections::HashMap<Value256, Vec<Fp>>,
        entries: &[(usize, ShardRegEntry)],
        row_label_indices: &[usize],
        weights: &BarycentricWeights,
    ) -> Result<Vec<(usize, Vec<Fp>)>, ClientError> {
        if row_label_indices.is_empty() {
            return Ok(Vec::new());
        }

        let mut row_values_by_target =
            vec![Vec::with_capacity(entries.len()); row_label_indices.len()];
        for (_, entry) in entries.iter().copied() {
            let Some(evals) = helper_evals.get(&entry.key.z_bytes) else {
                return Err(ClientError::NotEnoughShards);
            };
            if evals.len() != row_label_indices.len() {
                return Err(ClientError::InvalidResponse(
                    "helper shard-eval row count mismatch".to_string(),
                ));
            }
            for (row_pos, &value) in evals.iter().enumerate() {
                row_values_by_target[row_pos].push(value);
            }
        }

        let z_values = entries.iter().map(|(_, entry)| entry.z).collect::<Vec<_>>();
        let mut out = Vec::with_capacity(row_label_indices.len());
        for (&row_label_index, one_row_values) in row_label_indices
            .iter()
            .zip(row_values_by_target.into_iter())
        {
            let repaired = materialize_row_on_eval_points_from_samples(
                &self.params,
                &z_values,
                &one_row_values,
                Some(weights),
            )
            .map_err(|_| ClientError::NotEnoughShards)?;
            out.push((row_label_index, repaired));
        }
        Ok(out)
    }

    fn recover_rows_from_sourced_helper_evaluations_correcting(
        &self,
        helper_evals: &HashMap<Value256, SourcedEvalValues>,
        entries: &[(usize, ShardRegEntry)],
        row_label_indices: &[usize],
        max_errors: usize,
        bad_servers: &mut HashSet<usize>,
    ) -> Result<Vec<(usize, Vec<Fp>)>, ClientError> {
        if row_label_indices.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(row_label_indices.len());
        for (row_pos, &row_label_index) in row_label_indices.iter().enumerate() {
            let known_bad_points = self.known_bad_entry_points(entries, bad_servers)?;
            let remaining_errors = remaining_error_budget(max_errors, known_bad_points)?;
            let mut points = Vec::new();
            let mut point_servers = Vec::new();
            for (_, entry) in entries.iter().copied() {
                let Some(evals) = helper_evals.get(&entry.key.z_bytes) else {
                    continue;
                };
                if bad_servers.contains(&evals.server_pos) {
                    continue;
                }
                if evals.values.len() != row_label_indices.len() {
                    return Err(ClientError::InvalidResponse(
                        "helper shard-eval row count mismatch".to_string(),
                    ));
                }
                points.push((entry.z, evals.values[row_pos]));
                point_servers.push(evals.server_pos);
            }
            if points.len() < self.params.need + 2 * remaining_errors {
                return Err(ClientError::NotEnoughShards);
            }
            let decoded = crate::coded_array::rs::rs_decode_eval_with_errors(
                &points,
                self.params.need,
                self.params.row_eval_points[0],
                remaining_errors,
            )
            .map_err(|_| ClientError::NotEnoughShards)?;
            for bad_idx in decoded.bad_indices {
                if let Some(&server_pos) = point_servers.get(bad_idx) {
                    bad_servers.insert(server_pos);
                }
            }
            let repaired = materialize_row_on_eval_points_from_coefficients(
                &self.params,
                &decoded.coefficients,
            );
            out.push((row_label_index, repaired));
        }
        Ok(out)
    }

    /// Migrate the current unauth array under a new mode parameter `s`.
    pub fn migrate(&mut self, target_s: usize) -> Result<(), ClientError> {
        self.refresh_len()?;
        let new_params = build_migrated_params(&self.params, target_s)
            .map_err(|err| ClientError::InvalidConfig(format!("{:?}", err)))?;
        let (partner_shards, helper_rows) = if target_s > self.params.s {
            let partner_shards = if self.kappa_halve_has_local_partner_shards(target_s) {
                vec![Vec::new(); self.servers.len()]
            } else {
                self.collect_partner_shards_for_kappa_halve(target_s)?
            };
            (partner_shards, self.collect_helper_rows_for_kappa_halve()?)
        } else if target_s < self.params.s {
            (
                self.collect_helper_shards_for_kappa_double(&new_params)?,
                Vec::new(),
            )
        } else {
            (vec![Vec::new(); self.servers.len()], Vec::new())
        };
        let transport = self.transport.clone();
        let partner_shards = Arc::new(partner_shards);
        let helper_rows = Arc::new(helper_rows);
        let servers: Vec<_> = self.servers.iter().cloned().enumerate().collect();
        let failed = broadcast_with_failures_parallel(
            &servers,
            |(_, server)| self.transport.server_label(server),
            {
                let partner_shards = Arc::clone(&partner_shards);
                let helper_rows = Arc::clone(&helper_rows);
                move |(server_pos, server)| {
                    transport.unauth_migrate(
                        &server,
                        target_s,
                        &partner_shards[server_pos],
                        helper_rows.as_ref(),
                    )
                }
            },
            |ok| {
                if ok {
                    Ok(())
                } else {
                    Err("unauth migrate failed".to_string())
                }
            },
        );
        if !failed.is_empty() {
            return Err(ClientError::Net(format!(
                "unauth migrate failed: {:?}",
                failed
            )));
        }
        self.params = new_params;
        self.server_owned_indices = self.params.party_registry_indices.clone();
        Ok(())
    }

    /// Double the segment size with the local factor-two layout.
    ///
    /// This keeps the unauth layer as the black-box storage API, preserving
    /// public array indices while doubling the segment grid and threshold.
    pub fn segment_double(&mut self) -> Result<(), ClientError> {
        self.refresh_len()?;
        let new_params = build_segment_double_params(&self.params)
            .map_err(|err| ClientError::InvalidConfig(format!("{:?}", err)))?;
        let transport = self.transport.clone();
        let servers: Vec<_> = self.servers.iter().cloned().enumerate().collect();
        let failed = broadcast_with_failures_parallel(
            &servers,
            |(_, server)| self.transport.server_label(server),
            move |(_, server)| transport.unauth_segment_double(&server, &[]),
            |ok| {
                if ok {
                    Ok(())
                } else {
                    Err("unauth segment-double failed".to_string())
                }
            },
        );
        if !failed.is_empty() {
            return Err(ClientError::Net(format!(
                "unauth segment-double failed: {:?}",
                failed
            )));
        }
        self.params = new_params;
        self.server_owned_indices = self.params.party_registry_indices.clone();
        Ok(())
    }

    /// Migrate the current unauth array while correcting up to `max_errors`
    /// bad RS-point replies while collecting migration helper data.
    ///
    /// `max_errors = 0` uses the non-correcting migration path unchanged.
    pub fn migrate_correcting(
        &mut self,
        target_s: usize,
        max_errors: usize,
    ) -> Result<(), ClientError> {
        if max_errors == 0 {
            return self.migrate(target_s);
        }
        self.refresh_len()?;
        let new_params = build_migrated_params(&self.params, target_s)
            .map_err(|err| ClientError::InvalidConfig(format!("{:?}", err)))?;
        let mut bad_servers = HashSet::new();
        let (partner_shards, helper_rows) = if target_s > self.params.s {
            let partner_shards = if self.kappa_halve_has_local_partner_shards(target_s) {
                vec![Vec::new(); self.servers.len()]
            } else {
                self.collect_partner_shards_for_kappa_halve_correcting(
                    target_s,
                    max_errors,
                    &mut bad_servers,
                )?
            };
            (
                partner_shards,
                self.collect_helper_rows_for_kappa_halve_correcting(max_errors, &mut bad_servers)?,
            )
        } else if target_s < self.params.s {
            (
                self.collect_helper_shards_for_kappa_double_correcting(
                    &new_params,
                    max_errors,
                    &mut bad_servers,
                )?,
                Vec::new(),
            )
        } else {
            (vec![Vec::new(); self.servers.len()], Vec::new())
        };
        let transport = self.transport.clone();
        let partner_shards = Arc::new(partner_shards);
        let helper_rows = Arc::new(helper_rows);
        let servers: Vec<_> = self.servers.iter().cloned().enumerate().collect();
        let failed = broadcast_with_failures_parallel(
            &servers,
            |(_, server)| self.transport.server_label(server),
            {
                let partner_shards = Arc::clone(&partner_shards);
                let helper_rows = Arc::clone(&helper_rows);
                move |(server_pos, server)| {
                    transport.unauth_migrate(
                        &server,
                        target_s,
                        &partner_shards[server_pos],
                        helper_rows.as_ref(),
                    )
                }
            },
            |ok| {
                if ok {
                    Ok(())
                } else {
                    Err("unauth migrate failed".to_string())
                }
            },
        );
        if !failed.is_empty() {
            return Err(ClientError::Net(format!(
                "unauth migrate failed: {:?}",
                failed
            )));
        }
        self.params = new_params;
        self.server_owned_indices = self.params.party_registry_indices.clone();
        Ok(())
    }

    /// Reconfigure under the append-only-log one-claim policy.
    ///
    /// Kappa-double keeps only each source claim's left child.  Kappa-halve reconstructs the
    /// deleted twin from helper rows instead of shipping full partner columns,
    /// then prunes to the deterministic target assignment.
    pub fn migrate_one_claim_left(
        &mut self,
        target_s: usize,
        target_owned_indices: Vec<Vec<usize>>,
    ) -> Result<(), ClientError> {
        self.refresh_len()?;
        if target_s < self.params.s {
            let mut new_params = build_migrated_params(&self.params, target_s)
                .map_err(|err| ClientError::InvalidConfig(format!("{:?}", err)))?;
            let helper_evals = self.collect_helper_evals_for_one_claim_left_kappa_double(
                &new_params,
                &target_owned_indices,
            )?;
            let transport = self.transport.clone();
            let helper_evals = Arc::new(helper_evals);
            let servers: Vec<_> = self.servers.iter().cloned().enumerate().collect();
            let failed = broadcast_with_failures_parallel(
                &servers,
                |(_, server)| self.transport.server_label(server),
                {
                    let helper_evals = Arc::clone(&helper_evals);
                    move |(server_pos, server)| {
                        transport.unauth_kappa_double_one_claim_left(
                            &server,
                            target_s,
                            &helper_evals[server_pos],
                        )
                    }
                },
                |ok| {
                    if ok {
                        Ok(())
                    } else {
                        Err("unauth one-claim kappa-double failed".to_string())
                    }
                },
            );
            if !failed.is_empty() {
                return Err(ClientError::Net(format!(
                    "unauth one-claim kappa-double failed: {:?}",
                    failed
                )));
            }
            new_params.party_registry_indices = target_owned_indices.clone();
            self.params = new_params;
            self.server_owned_indices = target_owned_indices;
            return Ok(());
        }

        if target_s > self.params.s {
            let mut new_params = build_migrated_params(&self.params, target_s)
                .map_err(|err| ClientError::InvalidConfig(format!("{:?}", err)))?;
            let partner_shards = vec![Vec::new(); self.servers.len()];
            let helper_rows = self.collect_helper_rows_for_kappa_halve()?;
            let transport = self.transport.clone();
            let partner_shards = Arc::new(partner_shards);
            let helper_rows = Arc::new(helper_rows);
            let servers: Vec<_> = self.servers.iter().cloned().enumerate().collect();
            let failed = broadcast_with_failures_parallel(
                &servers,
                |(_, server)| self.transport.server_label(server),
                {
                    let partner_shards = Arc::clone(&partner_shards);
                    let helper_rows = Arc::clone(&helper_rows);
                    move |(server_pos, server)| {
                        transport.unauth_migrate(
                            &server,
                            target_s,
                            &partner_shards[server_pos],
                            helper_rows.as_ref(),
                        )
                    }
                },
                |ok| {
                    if ok {
                        Ok(())
                    } else {
                        Err("unauth one-claim kappa-halve failed".to_string())
                    }
                },
            );
            if !failed.is_empty() {
                return Err(ClientError::Net(format!(
                    "unauth one-claim kappa-halve failed: {:?}",
                    failed
                )));
            }
            new_params.party_registry_indices = target_owned_indices.clone();
            self.params = new_params;
            self.server_owned_indices = self.params.party_registry_indices.clone();
        } else {
            self.migrate(target_s)?;
        }
        self.retain_claims_by_server(target_owned_indices.clone())?;
        self.params.party_registry_indices = target_owned_indices;
        Ok(())
    }

    /// Reconfigure under the append-only-log one-claim policy while correcting
    /// bad RS-point replies during helper materialization.
    ///
    /// `max_errors = 0` uses the non-correcting one-claim kappa-rescale path.
    pub fn migrate_one_claim_left_correcting(
        &mut self,
        target_s: usize,
        target_owned_indices: Vec<Vec<usize>>,
        max_errors: usize,
    ) -> Result<(), ClientError> {
        if max_errors == 0 {
            return self.migrate_one_claim_left(target_s, target_owned_indices);
        }
        self.refresh_len()?;
        if target_s < self.params.s {
            let mut new_params = build_migrated_params(&self.params, target_s)
                .map_err(|err| ClientError::InvalidConfig(format!("{:?}", err)))?;
            let mut bad_servers = HashSet::new();
            let helper_evals = self
                .collect_helper_evals_for_one_claim_left_kappa_double_correcting(
                    &new_params,
                    &target_owned_indices,
                    max_errors,
                    &mut bad_servers,
                )?;
            let transport = self.transport.clone();
            let helper_evals = Arc::new(helper_evals);
            let servers: Vec<_> = self.servers.iter().cloned().enumerate().collect();
            let failed = broadcast_with_failures_parallel(
                &servers,
                |(_, server)| self.transport.server_label(server),
                {
                    let helper_evals = Arc::clone(&helper_evals);
                    move |(server_pos, server)| {
                        transport.unauth_kappa_double_one_claim_left(
                            &server,
                            target_s,
                            &helper_evals[server_pos],
                        )
                    }
                },
                |ok| {
                    if ok {
                        Ok(())
                    } else {
                        Err("unauth one-claim kappa-double failed".to_string())
                    }
                },
            );
            if !failed.is_empty() {
                return Err(ClientError::Net(format!(
                    "unauth one-claim kappa-double failed: {:?}",
                    failed
                )));
            }
            new_params.party_registry_indices = target_owned_indices.clone();
            self.params = new_params;
            self.server_owned_indices = target_owned_indices;
            return Ok(());
        }

        if target_s > self.params.s {
            let mut new_params = build_migrated_params(&self.params, target_s)
                .map_err(|err| ClientError::InvalidConfig(format!("{:?}", err)))?;
            let partner_shards = vec![Vec::new(); self.servers.len()];
            let mut bad_servers = HashSet::new();
            let helper_rows =
                self.collect_helper_rows_for_kappa_halve_correcting(max_errors, &mut bad_servers)?;
            let transport = self.transport.clone();
            let partner_shards = Arc::new(partner_shards);
            let helper_rows = Arc::new(helper_rows);
            let servers: Vec<_> = self.servers.iter().cloned().enumerate().collect();
            let failed = broadcast_with_failures_parallel(
                &servers,
                |(_, server)| self.transport.server_label(server),
                {
                    let partner_shards = Arc::clone(&partner_shards);
                    let helper_rows = Arc::clone(&helper_rows);
                    move |(server_pos, server)| {
                        transport.unauth_migrate(
                            &server,
                            target_s,
                            &partner_shards[server_pos],
                            helper_rows.as_ref(),
                        )
                    }
                },
                |ok| {
                    if ok {
                        Ok(())
                    } else {
                        Err("unauth one-claim kappa-halve failed".to_string())
                    }
                },
            );
            if !failed.is_empty() {
                return Err(ClientError::Net(format!(
                    "unauth one-claim kappa-halve failed: {:?}",
                    failed
                )));
            }
            new_params.party_registry_indices = target_owned_indices.clone();
            self.params = new_params;
            self.server_owned_indices = self.params.party_registry_indices.clone();
        } else {
            self.migrate_correcting(target_s, max_errors)?;
        }
        self.retain_claims_by_server(target_owned_indices.clone())?;
        self.params.party_registry_indices = target_owned_indices;
        Ok(())
    }

    /// Run the one-claim policy for `kappa -> 2kappa`.
    pub fn kappa_double_one_claim_left(
        &mut self,
        target_owned_indices: Vec<Vec<usize>>,
    ) -> Result<(), ClientError> {
        if self.params.s <= 1 {
            return Err(ClientError::InvalidConfig(
                "cannot kappa-double below s=1".to_string(),
            ));
        }
        self.migrate_one_claim_left(self.params.s / 2, target_owned_indices)
    }

    /// Run the one-claim policy for `kappa -> kappa/2`.
    pub fn kappa_halve_one_claim_left(
        &mut self,
        target_owned_indices: Vec<Vec<usize>>,
    ) -> Result<(), ClientError> {
        let target_s = self.params.s.checked_mul(2).ok_or_else(|| {
            ClientError::InvalidConfig("s overflow during kappa-halve".to_string())
        })?;
        self.migrate_one_claim_left(target_s, target_owned_indices)
    }

    /// Run the one-claim kappa-double policy while correcting helper data.
    pub fn kappa_double_one_claim_left_correcting(
        &mut self,
        target_owned_indices: Vec<Vec<usize>>,
        max_errors: usize,
    ) -> Result<(), ClientError> {
        if self.params.s <= 1 {
            return Err(ClientError::InvalidConfig(
                "cannot kappa-double below s=1".to_string(),
            ));
        }
        self.migrate_one_claim_left_correcting(self.params.s / 2, target_owned_indices, max_errors)
    }

    /// Run the one-claim kappa-halve policy while correcting helper data.
    pub fn kappa_halve_one_claim_left_correcting(
        &mut self,
        target_owned_indices: Vec<Vec<usize>>,
        max_errors: usize,
    ) -> Result<(), ClientError> {
        let target_s = self.params.s.checked_mul(2).ok_or_else(|| {
            ClientError::InvalidConfig("s overflow during kappa-halve".to_string())
        })?;
        self.migrate_one_claim_left_correcting(target_s, target_owned_indices, max_errors)
    }

    /// Delete temporary local state after a kappa-rescale prepare phase.
    ///
    /// `target_owned_indices` is ordered like `self.servers`; each entry is the
    /// exact target registry indices that the corresponding server should keep.
    pub fn retain_claims_by_server(
        &mut self,
        target_owned_indices: Vec<Vec<usize>>,
    ) -> Result<(), ClientError> {
        if target_owned_indices.len() != self.servers.len() {
            return Err(ClientError::InvalidConfig(format!(
                "retain plan has {} servers, client has {}",
                target_owned_indices.len(),
                self.servers.len()
            )));
        }
        let transport = self.transport.clone();
        let target_owned_indices = Arc::new(target_owned_indices);
        let servers: Vec<_> = self.servers.iter().cloned().enumerate().collect();
        let failed = broadcast_with_failures_parallel(
            &servers,
            |(_, server)| self.transport.server_label(server),
            {
                let target_owned_indices = Arc::clone(&target_owned_indices);
                move |(server_pos, server)| {
                    transport.unauth_retain_claims(&server, &target_owned_indices[server_pos])
                }
            },
            |ok| {
                if ok {
                    Ok(())
                } else {
                    Err("unauth retain claims failed".to_string())
                }
            },
        );
        if !failed.is_empty() {
            return Err(ClientError::Net(format!(
                "unauth retain claims failed: {:?}",
                failed
            )));
        }
        self.params.party_registry_indices = target_owned_indices.as_ref().clone();
        self.server_owned_indices = target_owned_indices.as_ref().clone();
        Ok(())
    }

    /// Halve `kappa` while keeping the segment size fixed (`s -> 2s`).
    pub fn kappa_halve(&mut self) -> Result<(), ClientError> {
        let target_s = self.params.s.checked_mul(2).ok_or_else(|| {
            ClientError::InvalidConfig("s overflow during kappa-halve".to_string())
        })?;
        self.migrate(target_s)
    }

    /// Halve `kappa` while correcting helper data.
    pub fn kappa_halve_correcting(&mut self, max_errors: usize) -> Result<(), ClientError> {
        let target_s = self.params.s.checked_mul(2).ok_or_else(|| {
            ClientError::InvalidConfig("s overflow during kappa-halve".to_string())
        })?;
        self.migrate_correcting(target_s, max_errors)
    }

    /// Double `kappa` while keeping the segment size fixed (`s -> s/2`).
    pub fn kappa_double(&mut self) -> Result<(), ClientError> {
        if self.params.s <= 1 {
            return Err(ClientError::InvalidConfig(
                "cannot kappa-double below s=1".to_string(),
            ));
        }
        self.migrate(self.params.s / 2)
    }

    /// Double `kappa` while correcting helper data.
    pub fn kappa_double_correcting(&mut self, max_errors: usize) -> Result<(), ClientError> {
        if self.params.s <= 1 {
            return Err(ClientError::InvalidConfig(
                "cannot kappa-double below s=1".to_string(),
            ));
        }
        self.migrate_correcting(self.params.s / 2, max_errors)
    }

    /// Ask one specific server to compute a repaired primary shard from local rows.
    pub fn repair_shard(
        &self,
        party_id: u64,
        seg: usize,
        registry_index: usize,
    ) -> Result<Vec<Value>, ClientError> {
        let label = party_id.to_string();
        let server = self
            .servers
            .iter()
            .find(|server| self.transport.server_label(server) == label)
            .ok_or_else(|| ClientError::Registry(format!("party {} not found", party_id)))?;
        let values = self
            .transport
            .unauth_repair_shard(server, seg, registry_index)?;
        Ok(values.into_iter().map(encode_value).collect())
    }

    /// Reconstruct one claimed primary shard from row replies across active live servers.
    pub fn repair_shard_from_network(
        &self,
        seg: usize,
        registry_index: usize,
    ) -> Result<Vec<Value>, ClientError> {
        let mut repaired = self.repair_shards_from_network(seg, &[registry_index])?;
        repaired
            .pop()
            .map(|(_, values)| values)
            .ok_or(ClientError::NotEnoughShards)
    }

    /// Reconstruct one claimed primary shard while correcting up to
    /// `max_errors` bad RS-point replies. `max_errors = 0` preserves the
    /// exact-threshold repair path.
    pub fn repair_shard_from_network_correcting(
        &self,
        seg: usize,
        registry_index: usize,
        max_errors: usize,
    ) -> Result<Vec<Value>, ClientError> {
        if max_errors == 0 {
            return self.repair_shard_from_network(seg, registry_index);
        }
        let mut repaired =
            self.repair_shards_from_network_correcting(seg, &[registry_index], max_errors)?;
        repaired
            .pop()
            .map(|(_, values)| values)
            .ok_or(ClientError::NotEnoughShards)
    }

    pub fn repair_shards_from_network(
        &self,
        seg: usize,
        registry_indices: &[usize],
    ) -> Result<Vec<(usize, Vec<Value>)>, ClientError> {
        if registry_indices.is_empty() {
            return Ok(Vec::new());
        }
        let mut unique_registry_indices = Vec::with_capacity(registry_indices.len());
        let mut seen = HashSet::new();
        for &registry_index in registry_indices {
            if registry_index >= self.params.shard_registry.len() {
                return Err(ClientError::InvalidIndex);
            }
            if seen.insert(registry_index) {
                unique_registry_indices.push(registry_index);
            }
        }
        let repaired = self.repair_shards_from_rows(seg, &unique_registry_indices)?;

        let mut out = Vec::with_capacity(registry_indices.len());
        for &registry_index in registry_indices {
            let Some(values) = repaired.get(&registry_index) else {
                return Err(ClientError::NotEnoughShards);
            };
            out.push((
                registry_index,
                values.iter().copied().map(encode_value).collect(),
            ));
        }
        Ok(out)
    }

    /// Reconstruct claimed primary shards while correcting up to `max_errors`
    /// bad RS-point replies across the repair session.
    pub fn repair_shards_from_network_correcting(
        &self,
        seg: usize,
        registry_indices: &[usize],
        max_errors: usize,
    ) -> Result<Vec<(usize, Vec<Value>)>, ClientError> {
        if max_errors == 0 {
            return self.repair_shards_from_network(seg, registry_indices);
        }
        if registry_indices.is_empty() {
            return Ok(Vec::new());
        }
        let mut unique_registry_indices = Vec::with_capacity(registry_indices.len());
        let mut seen = HashSet::new();
        for &registry_index in registry_indices {
            if registry_index >= self.params.shard_registry.len() {
                return Err(ClientError::InvalidIndex);
            }
            if seen.insert(registry_index) {
                unique_registry_indices.push(registry_index);
            }
        }
        let mut bad_servers = HashSet::new();
        let repaired = self.repair_shards_from_rows_correcting(
            seg,
            &unique_registry_indices,
            max_errors,
            &mut bad_servers,
        )?;

        let mut out = Vec::with_capacity(registry_indices.len());
        for &registry_index in registry_indices {
            let Some(values) = repaired.get(&registry_index) else {
                return Err(ClientError::NotEnoughShards);
            };
            out.push((
                registry_index,
                values.iter().copied().map(encode_value).collect(),
            ));
        }
        Ok(out)
    }

    /// Reconstruct one maintained row from live servers by gathering enough primary shards.
    pub fn repair_row_from_network(
        &self,
        seg: usize,
        row_label_index: usize,
    ) -> Result<Vec<Value>, ClientError> {
        let mut repaired = self.repair_rows_from_network(seg, &[row_label_index])?;
        repaired
            .pop()
            .map(|(_, values)| values)
            .ok_or(ClientError::NotEnoughShards)
    }

    /// Reconstruct one maintained row while correcting up to `max_errors` bad
    /// helper replies. `max_errors = 0` preserves the exact-threshold repair path.
    pub fn repair_row_from_network_correcting(
        &self,
        seg: usize,
        row_label_index: usize,
        max_errors: usize,
    ) -> Result<Vec<Value>, ClientError> {
        if max_errors == 0 {
            return self.repair_row_from_network(seg, row_label_index);
        }
        let mut repaired =
            self.repair_rows_from_network_correcting(seg, &[row_label_index], max_errors)?;
        repaired
            .pop()
            .map(|(_, values)| values)
            .ok_or(ClientError::NotEnoughShards)
    }

    pub fn repair_rows_from_network(
        &self,
        seg: usize,
        row_label_indices: &[usize],
    ) -> Result<Vec<(usize, Vec<Value>)>, ClientError> {
        if row_label_indices.is_empty() {
            return Ok(Vec::new());
        }
        for &row_label_index in row_label_indices {
            if self.params.row_labels.get(row_label_index).is_none() {
                return Err(ClientError::InvalidIndex);
            }
        }

        let entries = self.select_row_repair_entries()?;
        let requested_by_server = self.plan_requested_col_labels_by_server(&entries)?;
        let helper_evals = self.collect_shard_evaluations_from_requested_by_server(
            seg,
            &requested_by_server,
            row_label_indices,
        )?;
        let weights =
            precompute_barycentric(&entries.iter().map(|(_, entry)| entry.z).collect::<Vec<_>>());
        let repaired = self.recover_rows_from_helper_evaluations_with_weights(
            &helper_evals,
            &entries,
            row_label_indices,
            &weights,
        )?;
        Ok(repaired
            .into_iter()
            .map(|(row_label_index, values)| {
                (
                    row_label_index,
                    values.into_iter().map(encode_value).collect::<Vec<_>>(),
                )
            })
            .collect())
    }

    /// Reconstruct maintained rows while correcting up to `max_errors` bad
    /// helper replies across the repair session.
    pub fn repair_rows_from_network_correcting(
        &self,
        seg: usize,
        row_label_indices: &[usize],
        max_errors: usize,
    ) -> Result<Vec<(usize, Vec<Value>)>, ClientError> {
        if max_errors == 0 {
            return self.repair_rows_from_network(seg, row_label_indices);
        }
        if row_label_indices.is_empty() {
            return Ok(Vec::new());
        }
        for &row_label_index in row_label_indices {
            if self.params.row_labels.get(row_label_index).is_none() {
                return Err(ClientError::InvalidIndex);
            }
        }

        let entries = self.select_row_repair_entries_with_target_distinct_servers(
            error_correcting_share_target(self.params.need, max_errors),
        )?;
        let requested_by_server = self.plan_requested_col_labels_by_server(&entries)?;
        let mut bad_servers = HashSet::new();
        let helper_evals = self.collect_shard_evaluations_from_requested_by_server_correcting(
            seg,
            &requested_by_server,
            row_label_indices,
            &bad_servers,
        )?;
        let repaired = self.recover_rows_from_sourced_helper_evaluations_correcting(
            &helper_evals,
            &entries,
            row_label_indices,
            max_errors,
            &mut bad_servers,
        )?;
        Ok(repaired
            .into_iter()
            .map(|(row_label_index, values)| {
                (
                    row_label_index,
                    values.into_iter().map(encode_value).collect::<Vec<_>>(),
                )
            })
            .collect())
    }

    /// Reconstruct a party's claimed shards and rows across segments while
    /// sharing bad-helper provenance across shard repair, row repair, and
    /// later segments.
    pub fn repair_party_state_segments_from_network_correcting(
        &self,
        segs: &[usize],
        registry_indices: &[usize],
        row_label_indices: &[usize],
        max_errors: usize,
    ) -> Result<Vec<RepairedPartySegmentState>, ClientError> {
        if max_errors == 0 {
            let mut out = Vec::with_capacity(segs.len());
            for &seg in segs {
                out.push(RepairedPartySegmentState {
                    seg,
                    shards: self.repair_shards_from_network(seg, registry_indices)?,
                    rows: self.repair_rows_from_network(seg, row_label_indices)?,
                });
            }
            return Ok(out);
        }

        let mut unique_registry_indices = Vec::with_capacity(registry_indices.len());
        let mut seen = HashSet::new();
        for &registry_index in registry_indices {
            if registry_index >= self.params.shard_registry.len() {
                return Err(ClientError::InvalidIndex);
            }
            if seen.insert(registry_index) {
                unique_registry_indices.push(registry_index);
            }
        }
        for &row_label_index in row_label_indices {
            if self.params.row_labels.get(row_label_index).is_none() {
                return Err(ClientError::InvalidIndex);
            }
        }

        let row_repair_entries = if row_label_indices.is_empty() {
            Vec::new()
        } else {
            self.select_row_repair_entries_with_target_distinct_servers(
                error_correcting_share_target(self.params.need, max_errors),
            )?
        };
        let row_repair_requested_by_server =
            self.plan_requested_col_labels_by_server(&row_repair_entries)?;
        let mut bad_servers = HashSet::new();
        let mut out = Vec::with_capacity(segs.len());

        for &seg in segs {
            let repaired_shards_fp = if unique_registry_indices.is_empty() {
                HashMap::new()
            } else {
                self.repair_shards_from_rows_correcting(
                    seg,
                    &unique_registry_indices,
                    max_errors,
                    &mut bad_servers,
                )?
            };
            let mut shards = Vec::with_capacity(registry_indices.len());
            for &registry_index in registry_indices {
                let Some(values) = repaired_shards_fp.get(&registry_index) else {
                    return Err(ClientError::NotEnoughShards);
                };
                shards.push((
                    registry_index,
                    values.iter().copied().map(encode_value).collect(),
                ));
            }

            let rows = if row_label_indices.is_empty() {
                Vec::new()
            } else {
                let helper_evals = self
                    .collect_shard_evaluations_from_requested_by_server_correcting(
                        seg,
                        &row_repair_requested_by_server,
                        row_label_indices,
                        &bad_servers,
                    )?;
                self.recover_rows_from_sourced_helper_evaluations_correcting(
                    &helper_evals,
                    &row_repair_entries,
                    row_label_indices,
                    max_errors,
                    &mut bad_servers,
                )?
                .into_iter()
                .map(|(row_label_index, values)| {
                    (
                        row_label_index,
                        values.into_iter().map(encode_value).collect::<Vec<_>>(),
                    )
                })
                .collect()
            };

            out.push(RepairedPartySegmentState { seg, shards, rows });
        }
        Ok(out)
    }

    /// Recover one claim bundle for one segment into an Fp-native state object.
    pub fn repair_claim_segment_fp(
        &self,
        seg: usize,
        claim: &ClaimBundle,
    ) -> Result<RecoveredClaimSegmentState, ClientError> {
        let plan = self.prepare_claim_recovery(claim)?;
        self.repair_claim_segment_fp_with_plan(seg, &plan)
    }

    /// Recover one claim bundle for one segment while correcting up to
    /// `max_errors` bad RS-point replies.
    pub fn repair_claim_segment_fp_correcting(
        &self,
        seg: usize,
        claim: &ClaimBundle,
        max_errors: usize,
    ) -> Result<RecoveredClaimSegmentState, ClientError> {
        if max_errors == 0 {
            return self.repair_claim_segment_fp(seg, claim);
        }
        let plan = self.prepare_claim_recovery_correcting(claim, max_errors)?;
        self.repair_claim_segment_fp_with_plan_correcting(seg, &plan, max_errors)
    }

    /// Recover one claim bundle for one segment into an Fp-native state object.
    pub fn repair_claim_segment_fp_with_plan(
        &self,
        seg: usize,
        plan: &ClaimRecoveryPlan,
    ) -> Result<RecoveredClaimSegmentState, ClientError> {
        let mut repaired = self.repair_claim_segments_fp_with_plan(&[seg], plan)?;
        repaired.pop().map(|(_, state)| state).ok_or_else(|| {
            ClientError::InvalidResponse("missing repaired claim segment".to_string())
        })
    }

    /// Recover one claim bundle for one segment using a plan with enough
    /// oversampling to correct up to `max_errors` bad RS-point replies.
    pub fn repair_claim_segment_fp_with_plan_correcting(
        &self,
        seg: usize,
        plan: &ClaimRecoveryPlan,
        max_errors: usize,
    ) -> Result<RecoveredClaimSegmentState, ClientError> {
        if max_errors == 0 {
            return self.repair_claim_segment_fp_with_plan(seg, plan);
        }
        let mut repaired =
            self.repair_claim_segments_fp_with_plan_correcting(&[seg], plan, max_errors)?;
        repaired.pop().map(|(_, state)| state).ok_or_else(|| {
            ClientError::InvalidResponse("missing repaired claim segment".to_string())
        })
    }

    /// Recover one claim bundle across many segments into Fp-native state objects.
    pub fn repair_claim_segments_fp_with_plan(
        &self,
        segs: &[usize],
        plan: &ClaimRecoveryPlan,
    ) -> Result<Vec<(usize, RecoveredClaimSegmentState)>, ClientError> {
        if segs.is_empty() {
            return Ok(Vec::new());
        }
        let requests: Vec<_> = segs
            .iter()
            .copied()
            .map(|seg| (seg, vec![plan.target_entry.z]))
            .collect();
        let target_rows_by_seg = self.collect_rows_matrix_from_server_positions(
            &requests,
            &plan.primary_repair_server_positions,
        )?;
        let helper_evals_by_seg = self.collect_shard_evaluations_matrix_from_requested_by_server(
            segs,
            &plan.row_repair_col_labels_by_server,
            &plan.row_label_indices,
        )?;
        let mut out = Vec::with_capacity(segs.len());
        for (((&seg, target_rows), helper_evals), _) in segs
            .iter()
            .zip(target_rows_by_seg.into_iter())
            .zip(helper_evals_by_seg.into_iter())
            .zip(0..)
        {
            let rows = target_rows.into_iter().next().ok_or_else(|| {
                ClientError::InvalidResponse("missing target row batch".to_string())
            })?;
            let repaired_shard =
                decode_primary_shard_from_rows(&self.params, plan.target_entry, &rows)
                    .map_err(|_| ClientError::NotEnoughShards)?;
            let repaired_rows = self.recover_rows_from_helper_evaluations_with_weights(
                &helper_evals,
                &plan.row_repair_entries,
                &plan.row_label_indices,
                &plan.row_repair_weights,
            )?;
            out.push((
                seg,
                RecoveredClaimSegmentState {
                    shards: vec![(plan.claim.registry_index, repaired_shard)],
                    rows: repaired_rows,
                },
            ));
        }
        Ok(out)
    }

    /// Recover one claim bundle across many segments while correcting bad
    /// helper replies and excluding any helper proven bad in an earlier segment.
    pub fn repair_claim_segments_fp_with_plan_correcting(
        &self,
        segs: &[usize],
        plan: &ClaimRecoveryPlan,
        max_errors: usize,
    ) -> Result<Vec<(usize, RecoveredClaimSegmentState)>, ClientError> {
        if max_errors == 0 {
            return self.repair_claim_segments_fp_with_plan(segs, plan);
        }
        if segs.is_empty() {
            return Ok(Vec::new());
        }
        let mut bad_servers = HashSet::new();
        let mut out = Vec::with_capacity(segs.len());
        for &seg in segs {
            let target_rows = self.collect_rows_matrix_from_server_positions_correcting(
                &[(seg, vec![plan.target_entry.z])],
                &plan.primary_repair_server_positions,
                max_errors,
                &bad_servers,
            )?;
            let helper_evals = self.collect_shard_evaluations_from_requested_by_server_correcting(
                seg,
                &plan.row_repair_col_labels_by_server,
                &plan.row_label_indices,
                &bad_servers,
            )?;
            let rows = target_rows.into_iter().next().ok_or_else(|| {
                ClientError::InvalidResponse("missing target row batch".to_string())
            })?;
            let rows = rows.into_iter().next().ok_or_else(|| {
                ClientError::InvalidResponse("missing target row batch".to_string())
            })?;
            let known_bad_points = self.known_bad_row_points_for_server_positions(
                &plan.primary_repair_server_positions,
                &bad_servers,
            )?;
            let repaired_shard = self.decode_primary_shard_from_sourced_rows_correcting(
                plan.target_entry,
                &rows,
                max_errors,
                known_bad_points,
                &mut bad_servers,
            )?;
            let repaired_rows = self.recover_rows_from_sourced_helper_evaluations_correcting(
                &helper_evals,
                &plan.row_repair_entries,
                &plan.row_label_indices,
                max_errors,
                &mut bad_servers,
            )?;
            out.push((
                seg,
                RecoveredClaimSegmentState {
                    shards: vec![(plan.claim.registry_index, repaired_shard)],
                    rows: repaired_rows,
                },
            ));
        }
        Ok(out)
    }
}

fn remaining_error_budget(
    max_errors: usize,
    known_bad_points: usize,
) -> Result<usize, ClientError> {
    max_errors
        .checked_sub(known_bad_points)
        .ok_or(ClientError::NotEnoughShards)
}

fn split_shard_for_kappa_double(old_values: &[Fp]) -> (Vec<Fp>, Vec<Fp>) {
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

fn eval_coefficients_on_structured_points(
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
