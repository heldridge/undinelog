//! Authenticated RPC service helpers.
//!
//! This module defines request/response types and a service trait that
//! decouples authenticated storage from transport/wire details.

use crate::auth::allocation::{
    default_family_key_for_range, default_value_at_index, default_values_for_family_key,
};
use crate::auth::mapping::{self, NodeLinearization};
use crate::auth::merkle::{NodeKey, default_hash};
use crate::auth::read_path::{
    active_nodes_for_read, build_partial_tree_nodes, build_path_nodes, build_subtree_read_plan,
    node_to_index_if_active,
};
use crate::auth::{
    AuthParams, AuthPath, AuthStore, AuthSubtreeProof, BatchWriteOpts, LeafUpdateBatch,
    MerkleTreeProof, WriteError as AuthWriteError,
};
use crate::coded_array::params::{
    kappa_double_child_registry_indices, kappa_halve_parent_registry_index,
    registry_supports_sibling_kappa_halve,
};
use crate::shares::ShardShare;
use crate::unauth::ClaimBundle;
use crate::unauth::service::{UnauthRequest, UnauthResponse, UnauthService};
use crate::unauth::{CodedArrayParams, DeltaMode, UnauthStore};
use crate::value::Value256;
use std::collections::HashSet;

/// Core authenticated read reply for a single node.
#[derive(Clone, Debug)]
pub struct AuthReadNodeReply {
    pub node: NodeKey,
    pub shares: Vec<ShardShare>,
}

/// Core authenticated read response.
#[derive(Clone, Debug)]
pub struct AuthReadResponse {
    pub nodes: Vec<AuthReadNodeReply>,
}

/// Core authenticated read response for one aligned subtree.
#[derive(Clone, Debug)]
pub struct AuthSubtreeReadResponse {
    pub leaf_nodes: Vec<AuthReadNodeReply>,
    pub upper_path_nodes: Vec<AuthReadNodeReply>,
}

/// Core authenticated read response for only the upper path of one subtree.
#[derive(Clone, Debug)]
pub struct AuthSubtreeUpperPathReadResponse {
    pub upper_path_nodes: Vec<AuthReadNodeReply>,
}

/// Core authentication path request (wire-independent).
#[derive(Clone, Debug)]
pub struct AuthPathRequest {
    pub index: u64,
    pub siblings: Vec<Value256>,
    pub root: Value256,
}

/// Core authenticated write request.
#[derive(Clone, Debug)]
pub struct AuthWriteRequest {
    pub old: Value256,
    pub new: Value256,
    pub path: AuthPathRequest,
}

/// Core authenticated write response.
#[derive(Clone, Debug)]
pub struct AuthWriteResponse {
    pub ok: bool,
    pub root_new: Option<Value256>,
    pub error: Option<String>,
}

/// Core batch write update.
#[derive(Clone, Debug)]
pub struct AuthBatchWriteUpdate {
    pub index: u64,
    pub old: Value256,
    pub new: Value256,
}

/// Core authenticated batch write request.
#[derive(Clone, Debug)]
pub struct AuthBatchWriteRequest {
    pub updates: Vec<AuthBatchWriteUpdate>,
    pub proof: MerkleTreeProof,
    pub delta_mode: DeltaMode,
}

/// Core authenticated write request for one aligned fresh subtree.
///
/// The covered leaves are assumed to hold the default value under `proof.root`.
#[derive(Clone, Debug)]
pub struct AuthFreshSubtreeWriteRequest {
    pub leaves: Vec<Value256>,
    pub proof: AuthSubtreeProof,
    pub delta_mode: DeltaMode,
}

/// Core authenticated batch write response.
#[derive(Clone, Debug)]
pub struct AuthBatchWriteResponse {
    pub ok: bool,
    pub root_new: Option<Value256>,
    pub error: Option<String>,
}

/// Core authenticated request types for transport adapters.
#[derive(Clone, Debug)]
pub enum AuthRequest {
    Read {
        index: u64,
    },
    ReadSubtree {
        start_index: u64,
        leaf_count: usize,
    },
    ReadSubtreeUpperPath {
        start_index: u64,
        leaf_count: usize,
    },
    Write {
        req: AuthWriteRequest,
    },
    WriteBatch {
        req: AuthBatchWriteRequest,
    },
    WriteFreshSubtree {
        req: AuthFreshSubtreeWriteRequest,
    },
    /// Batch read by leaf indices; the server derives node indices.
    UnauthMultiRead {
        idxs: Vec<u64>,
    },
    Extend {
        new_leaves: u64,
    },
    GetLen,
}

/// Core authenticated responses for transport adapters.
#[derive(Clone, Debug)]
pub enum AuthResponse {
    Read {
        resp: AuthReadResponse,
    },
    ReadSubtree {
        resp: AuthSubtreeReadResponse,
    },
    ReadSubtreeUpperPath {
        resp: AuthSubtreeUpperPathReadResponse,
    },
    Write {
        resp: AuthWriteResponse,
    },
    WriteBatch {
        resp: AuthBatchWriteResponse,
    },
    WriteFreshSubtree {
        resp: AuthBatchWriteResponse,
    },
    UnauthMultiRead {
        shares: Vec<Vec<ShardShare>>,
    },
    Extend {
        ok: bool,
    },
    Len {
        len: u64,
    },
    Error {
        message: String,
    },
}

/// Authenticated service boundary used by transport adapters.
pub trait AuthService {
    fn handle_request(&mut self, req: AuthRequest) -> AuthResponse;
}

/// Single-server authenticated storage and root state.
#[derive(Clone)]
pub struct AuthServer {
    pub params: CodedArrayParams,
    pub auth: AuthParams,
    pub node_linearization: NodeLinearization,
    pub store: AuthStore,
    allocated_leaves: usize,
    assigned_indices: Option<Vec<usize>>,
    claims: Option<Vec<ClaimBundle>>,
}

impl AuthServer {
    fn read_nodes(&self, nodes: &[NodeKey]) -> Vec<AuthReadNodeReply> {
        let mut active_map = vec![None; nodes.len()];
        let mut idxs = Vec::new();
        for (pos, node) in nodes.iter().enumerate() {
            let idx = node_to_index_if_active(
                &self.auth,
                self.node_linearization,
                self.allocated_leaves,
                *node,
            );
            if let Some(idx) = idx {
                active_map[pos] = Some(idxs.len());
                idxs.push(idx);
            }
        }
        let shares_by_idx = self.store.unauth.handle_multi_read(&idxs);
        nodes
            .iter()
            .copied()
            .enumerate()
            .map(|(pos, node)| {
                let shares = match active_map[pos] {
                    Some(idx_pos) => shares_by_idx.get(idx_pos).cloned().unwrap_or_default(),
                    None => Vec::new(),
                };
                AuthReadNodeReply { node, shares }
            })
            .collect()
    }

    /// Create an authenticated server with full shard storage.
    pub fn new(
        params: CodedArrayParams,
        auth: AuthParams,
        node_linearization: NodeLinearization,
        allocated_leaves: usize,
    ) -> Result<Self, AuthWriteError> {
        Self::new_with_assignments(params, auth, node_linearization, allocated_leaves, None)
    }

    /// Create an authenticated server that stores only the assigned shard indices.
    pub fn new_with_assignments(
        params: CodedArrayParams,
        auth: AuthParams,
        node_linearization: NodeLinearization,
        allocated_leaves: usize,
        assigned_indices: Option<Vec<usize>>,
    ) -> Result<Self, AuthWriteError> {
        Self::new_with_assignments_and_claims(
            params,
            auth,
            node_linearization,
            allocated_leaves,
            assigned_indices,
            None,
        )
    }

    /// Create an authenticated server with optional claim-bound unauth storage underneath.
    pub fn new_with_assignments_and_claims(
        params: CodedArrayParams,
        auth: AuthParams,
        node_linearization: NodeLinearization,
        allocated_leaves: usize,
        assigned_indices: Option<Vec<usize>>,
        claims: Option<Vec<ClaimBundle>>,
    ) -> Result<Self, AuthWriteError> {
        let root0 = default_hash(&auth.defaults, 0);
        let store = match claims.as_ref() {
            Some(claims) => UnauthStore::new_with_claims(params.clone(), claims.clone()),
            None => UnauthStore::new(params.clone()),
        };
        let assigned_indices = assigned_indices.map(normalize_indices);
        let mut server = AuthServer {
            params,
            auth,
            node_linearization,
            store: AuthStore {
                root: root0,
                unauth: store,
            },
            allocated_leaves: 0,
            assigned_indices,
            claims,
        };
        server.ensure_allocated(allocated_leaves)?;
        Ok(server)
    }

    /// Ensure enough unauth array entries are allocated for the requested leaf count.
    pub fn ensure_allocated(&mut self, new_allocated: usize) -> Result<(), AuthWriteError> {
        if new_allocated <= self.allocated_leaves {
            self.allocated_leaves = self.allocated_leaves.max(new_allocated);
            return Ok(());
        }
        if new_allocated as u128 > self.auth.conceptual_l_max_u128() {
            return Err(AuthWriteError::InvalidIndex);
        }
        let height = self.auth.tree_height as u16;
        let cursor = mapping::cursor(self.node_linearization, height, new_allocated as u64);
        let auth = &self.auth;
        let node_linearization = self.node_linearization;
        let assigned_indices = self.assigned_indices.clone();
        let claims = self.claims.clone();
        match (claims.as_deref(), assigned_indices.as_deref()) {
            (Some(claims), _) => {
                self.store
                    .unauth
                    .extend_with_value_families_for_claims_timed(
                        cursor,
                        claims,
                        |range| default_family_key_for_range(auth, node_linearization, range),
                        |_range, family_key| default_values_for_family_key(auth, family_key),
                        None,
                    )
                    .map_err(|_| AuthWriteError::InvalidIndex)?;
            }
            (None, Some(indices)) => {
                self.store
                    .unauth
                    .extend_with_value_families_filtered_timed(
                        cursor,
                        indices,
                        |range| default_family_key_for_range(auth, node_linearization, range),
                        |_range, family_key| default_values_for_family_key(auth, family_key),
                        None,
                    )
                    .map_err(|_| AuthWriteError::InvalidIndex)?;
            }
            (None, None) => {
                self.store
                    .unauth
                    .extend_with_value_families_timed(
                        cursor,
                        |range| default_family_key_for_range(auth, node_linearization, range),
                        |_range, family_key| default_values_for_family_key(auth, family_key),
                        None,
                    )
                    .map_err(|_| AuthWriteError::InvalidIndex)?;
            }
        }
        self.allocated_leaves = new_allocated;
        Ok(())
    }

    /// Handle an authenticated read request by returning shard replies.
    pub fn handle_read(&self, index: u64) -> AuthReadResponse {
        let path_nodes = build_path_nodes(&self.auth, index);
        AuthReadResponse {
            nodes: self.read_nodes(&path_nodes),
        }
    }

    /// Handle an aligned subtree read request by returning shard replies.
    pub fn handle_read_subtree(
        &self,
        start_index: u64,
        leaf_count: usize,
    ) -> Result<AuthSubtreeReadResponse, String> {
        let plan = build_subtree_read_plan(&self.auth, start_index, leaf_count)
            .map_err(|err| format!("invalid subtree read: {:?}", err))?;
        let mut nodes = plan.leaf_nodes.clone();
        nodes.extend(plan.upper_path_nodes.iter().copied());
        let mut replies = self.read_nodes(&nodes);
        let upper_path_nodes = replies.split_off(plan.leaf_nodes.len());
        Ok(AuthSubtreeReadResponse {
            leaf_nodes: replies,
            upper_path_nodes,
        })
    }

    /// Handle an aligned subtree read request by returning only the upper-path replies.
    pub fn handle_read_subtree_upper_path(
        &self,
        start_index: u64,
        leaf_count: usize,
    ) -> Result<AuthSubtreeUpperPathReadResponse, String> {
        let plan = build_subtree_read_plan(&self.auth, start_index, leaf_count)
            .map_err(|err| format!("invalid subtree read: {:?}", err))?;
        let upper_path_nodes = self.read_nodes(&plan.upper_path_nodes);
        Ok(AuthSubtreeUpperPathReadResponse { upper_path_nodes })
    }

    /// Handle an unauthenticated batch read for auth paths.
    pub fn handle_unauth_multi_read(&self, indices: &[u64]) -> Vec<Vec<ShardShare>> {
        if indices.is_empty() {
            return Vec::new();
        }
        let plan = build_partial_tree_nodes(&self.auth, indices);
        let active_nodes = active_nodes_for_read(
            &self.auth,
            self.node_linearization,
            self.allocated_leaves,
            &plan.nodes,
        );
        if active_nodes.is_empty() {
            return Vec::new();
        }
        let mut idxs = Vec::with_capacity(active_nodes.len());
        for node in active_nodes {
            if let Some(idx) = node_to_index_if_active(
                &self.auth,
                self.node_linearization,
                self.allocated_leaves,
                node,
            ) {
                idxs.push(idx);
            }
        }
        self.store.unauth.handle_multi_read(&idxs)
    }

    /// Return the number of allocated authenticated leaves.
    pub fn allocated_leaves(&self) -> u64 {
        self.allocated_leaves as u64
    }

    /// Handle an authenticated write request by verifying and applying the update.
    pub fn handle_write(&mut self, req: AuthWriteRequest) -> AuthWriteResponse {
        if let Err(err) = self.ensure_allocated(self.allocated_leaves) {
            return AuthWriteResponse {
                ok: false,
                root_new: None,
                error: Some(format!("allocation failed: {:?}", err)),
            };
        }
        let path_index = match usize::try_from(req.path.index) {
            Ok(index) => index,
            Err(_) => {
                return AuthWriteResponse {
                    ok: false,
                    root_new: None,
                    error: Some("path index overflow".to_string()),
                };
            }
        };
        let path = AuthPath {
            index: path_index,
            siblings: req.path.siblings,
            root: req.path.root,
        };
        let update = match crate::auth::write::build_broadcast_update(
            &self.auth,
            &self.params,
            self.node_linearization,
            &self.store,
            self.allocated_leaves,
            path_index,
            req.old,
            req.new,
            path,
            DeltaMode::Aggregate,
        ) {
            Ok(update) => update,
            Err(err) => {
                return AuthWriteResponse {
                    ok: false,
                    root_new: None,
                    error: Some(format!("apply failed: {:?}", err)),
                };
            }
        };
        match crate::auth::write::apply_broadcast_update(
            &self.auth,
            &self.params,
            self.node_linearization,
            &mut self.store,
            &update,
        ) {
            Ok(()) => AuthWriteResponse {
                ok: true,
                root_new: Some(update.root_new),
                error: None,
            },
            Err(err) => AuthWriteResponse {
                ok: false,
                root_new: None,
                error: Some(format!("apply failed: {:?}", err)),
            },
        }
    }

    /// Handle an authenticated batch write request by verifying and applying the update.
    pub fn handle_write_batch(&mut self, req: AuthBatchWriteRequest) -> AuthBatchWriteResponse {
        if let Err(err) = self.ensure_allocated(self.allocated_leaves) {
            return AuthBatchWriteResponse {
                ok: false,
                root_new: None,
                error: Some(format!("allocation failed: {:?}", err)),
            };
        }
        let mut updates = Vec::with_capacity(req.updates.len());
        for update in req.updates {
            let index = match usize::try_from(update.index) {
                Ok(index) => index,
                Err(_) => {
                    return AuthBatchWriteResponse {
                        ok: false,
                        root_new: None,
                        error: Some("batch index overflow".to_string()),
                    };
                }
            };
            updates.push(LeafUpdateBatch {
                index,
                old: update.old,
                new: update.new,
            });
        }
        let opts = BatchWriteOpts {
            delta_mode: req.delta_mode,
        };
        let update = match crate::auth::write::build_broadcast_update_batch(
            &self.auth,
            &self.params,
            self.node_linearization,
            &self.store,
            self.allocated_leaves,
            updates,
            req.proof,
            opts,
        ) {
            Ok(update) => update,
            Err(err) => {
                return AuthBatchWriteResponse {
                    ok: false,
                    root_new: None,
                    error: Some(format!("apply failed: {:?}", err)),
                };
            }
        };
        match crate::auth::write::apply_prevalidated_broadcast_update_batch(
            &mut self.store,
            &update,
        ) {
            Ok(()) => AuthBatchWriteResponse {
                ok: true,
                root_new: Some(update.root_new),
                error: None,
            },
            Err(err) => AuthBatchWriteResponse {
                ok: false,
                root_new: None,
                error: Some(format!("apply failed: {:?}", err)),
            },
        }
    }

    /// Handle an authenticated write of one aligned fresh subtree.
    pub fn handle_write_fresh_subtree(
        &mut self,
        req: AuthFreshSubtreeWriteRequest,
    ) -> AuthBatchWriteResponse {
        if let Err(err) = self.ensure_allocated(self.allocated_leaves) {
            return AuthBatchWriteResponse {
                ok: false,
                root_new: None,
                error: Some(format!("allocation failed: {:?}", err)),
            };
        }
        let root_new = match crate::auth::write::apply_fresh_subtree_write_store(
            &self.auth,
            &self.params,
            self.node_linearization,
            &mut self.store,
            self.allocated_leaves,
            &req.leaves,
            &req.proof,
            req.delta_mode,
        ) {
            Ok(root_new) => root_new,
            Err(err) => {
                return AuthBatchWriteResponse {
                    ok: false,
                    root_new: None,
                    error: Some(format!("apply failed: {:?}", err)),
                };
            }
        };
        AuthBatchWriteResponse {
            ok: true,
            root_new: Some(root_new),
            error: None,
        }
    }
}

fn migrate_assigned_indices(
    params: &CodedArrayParams,
    assigned_indices: &[usize],
    target_s: usize,
) -> Vec<usize> {
    if target_s == params.s / 2 {
        let mut migrated = Vec::with_capacity(assigned_indices.len() * 2);
        for &registry_index in assigned_indices {
            let (left, right) = kappa_double_child_registry_indices(registry_index);
            migrated.push(left);
            migrated.push(right);
        }
        migrated.sort_unstable();
        migrated.dedup();
        migrated
    } else if target_s == params.s * 2 && registry_supports_sibling_kappa_halve(params) {
        let present: HashSet<_> = assigned_indices.iter().copied().collect();
        let mut migrated = Vec::new();
        for &registry_index in assigned_indices {
            if registry_index % 2 == 0 && present.contains(&(registry_index + 1)) {
                migrated.push(kappa_halve_parent_registry_index(registry_index));
            }
        }
        migrated.sort_unstable();
        migrated.dedup();
        migrated
    } else {
        assigned_indices.to_vec()
    }
}

impl AuthService for AuthServer {
    fn handle_request(&mut self, req: AuthRequest) -> AuthResponse {
        match req {
            AuthRequest::Read { index } => AuthResponse::Read {
                resp: self.handle_read(index),
            },
            AuthRequest::ReadSubtree {
                start_index,
                leaf_count,
            } => match self.handle_read_subtree(start_index, leaf_count) {
                Ok(resp) => AuthResponse::ReadSubtree { resp },
                Err(message) => AuthResponse::Error { message },
            },
            AuthRequest::ReadSubtreeUpperPath {
                start_index,
                leaf_count,
            } => match self.handle_read_subtree_upper_path(start_index, leaf_count) {
                Ok(resp) => AuthResponse::ReadSubtreeUpperPath { resp },
                Err(message) => AuthResponse::Error { message },
            },
            AuthRequest::Write { req } => AuthResponse::Write {
                resp: self.handle_write(req),
            },
            AuthRequest::WriteBatch { req } => AuthResponse::WriteBatch {
                resp: self.handle_write_batch(req),
            },
            AuthRequest::WriteFreshSubtree { req } => AuthResponse::WriteFreshSubtree {
                resp: self.handle_write_fresh_subtree(req),
            },
            AuthRequest::UnauthMultiRead { idxs } => AuthResponse::UnauthMultiRead {
                shares: self.handle_unauth_multi_read(&idxs),
            },
            AuthRequest::Extend { new_leaves } => {
                match self.ensure_allocated(new_leaves as usize) {
                    Ok(()) => AuthResponse::Extend { ok: true },
                    Err(err) => AuthResponse::Error {
                        message: format!("auth extend failed: {:?}", err),
                    },
                }
            }
            AuthRequest::GetLen => AuthResponse::Len {
                len: self.allocated_leaves as u64,
            },
        }
    }
}

impl UnauthService for AuthServer {
    fn handle_request(&mut self, req: UnauthRequest) -> UnauthResponse {
        match req {
            UnauthRequest::ReadSegmentRaw { .. } => UnauthResponse::Error {
                message: "raw unauth reads are not exposed by auth servers".to_string(),
            },
            UnauthRequest::ReadRowsAtZ { seg, z_eval } => UnauthResponse::ReadRowsAtZ {
                rows: self.store.unauth.handle_read_rows_at_z(seg, z_eval),
            },
            UnauthRequest::ReadRowsBatchAtZ { seg, z_evals } => UnauthResponse::ReadRowsBatchAtZ {
                rows: self.store.unauth.handle_read_rows_batch_at_z(seg, &z_evals),
            },
            UnauthRequest::ReadRowsMatrix { requests } => UnauthResponse::ReadRowsMatrix {
                rows: self.store.unauth.handle_read_rows_matrix(&requests),
            },
            UnauthRequest::RepairShard {
                seg,
                registry_index,
            } => match self.store.unauth.handle_repair_shard(seg, registry_index) {
                Ok(values) => UnauthResponse::RepairShard { values },
                Err(err) => UnauthResponse::Error {
                    message: format!("unauth repair failed: {:?}", err),
                },
            },
            UnauthRequest::RepairShardBatch {
                seg,
                registry_indices,
            } => match self
                .store
                .unauth
                .handle_repair_shard_batch(seg, &registry_indices)
            {
                Ok(values) => UnauthResponse::RepairShardBatch { values },
                Err(err) => UnauthResponse::Error {
                    message: format!("unauth batch repair failed: {:?}", err),
                },
            },
            UnauthRequest::EvalShardsAtRowLabels {
                seg,
                col_labels,
                row_labels,
            } => match self.store.unauth.handle_eval_shards_at_row_labels(
                seg,
                &col_labels,
                &row_labels,
            ) {
                Ok(values) => UnauthResponse::EvalShardsAtRowLabels { values },
                Err(err) => UnauthResponse::Error {
                    message: format!("unauth shard-eval failed: {:?}", err),
                },
            },
            UnauthRequest::EvalShardsAtRowLabelsMatrix {
                segs,
                col_labels,
                row_labels,
            } => match self.store.unauth.handle_eval_shards_at_row_labels_matrix(
                &segs,
                &col_labels,
                &row_labels,
            ) {
                Ok(values) => UnauthResponse::EvalShardsAtRowLabelsMatrix { values },
                Err(err) => UnauthResponse::Error {
                    message: format!("unauth shard-eval matrix failed: {:?}", err),
                },
            },
            UnauthRequest::Migrate {
                target_s,
                partner_shards,
                helper_rows,
            } => {
                let old_params = self.store.unauth.params.clone();
                match self.store.unauth.migrate_with_helper_rows(
                    target_s,
                    &partner_shards,
                    &helper_rows,
                ) {
                    Ok(()) => {
                        self.params = self.store.unauth.params.clone();
                        if let Some(assigned_indices) = self.assigned_indices.as_mut() {
                            *assigned_indices =
                                migrate_assigned_indices(&old_params, assigned_indices, target_s);
                        }
                        self.claims = self.store.unauth.claim_bundles();
                        UnauthResponse::Migrate { ok: true }
                    }
                    Err(err) => UnauthResponse::Error {
                        message: format!("unauth migrate failed: {:?}", err),
                    },
                }
            }
            UnauthRequest::SegmentDouble { helper_shards } => {
                let old_segments = self.store.unauth.segments();
                let old_n_grid = self.params.n_grid;
                match self
                    .store
                    .unauth
                    .segment_double_with_helpers(&helper_shards)
                {
                    Ok(()) => {
                        self.params = self.store.unauth.params.clone();
                        if old_segments % 2 == 1 {
                            let missing_start = (old_segments / 2) as u64
                                * self.params.n_grid as u64
                                + old_n_grid as u64;
                            let missing_values = (0..old_n_grid)
                                .map(|offset| {
                                    default_value_at_index(
                                        &self.auth,
                                        self.node_linearization,
                                        missing_start + offset as u64,
                                    )
                                })
                                .collect::<Vec<_>>();
                            if let Err(err) = self.store.unauth.hydrate_inactive_zero_range(
                                missing_start,
                                &missing_values,
                                DeltaMode::Aggregate,
                            ) {
                                return UnauthResponse::Error {
                                    message: format!(
                                        "unauth segment-double padding failed: {:?}",
                                        err
                                    ),
                                };
                            }
                        }
                        self.claims = self.store.unauth.claim_bundles();
                        self.assigned_indices = self.claims.as_ref().map(|claims| {
                            claims.iter().map(|claim| claim.registry_index).collect()
                        });
                        UnauthResponse::SegmentDouble { ok: true }
                    }
                    Err(err) => UnauthResponse::Error {
                        message: format!("unauth segment-double failed: {:?}", err),
                    },
                }
            }
            UnauthRequest::KappaDoubleOneClaimLeft {
                target_s,
                helper_evals,
            } => {
                match self
                    .store
                    .unauth
                    .migrate_kappa_double_one_claim_left(target_s, &helper_evals)
                {
                    Ok(()) => {
                        self.params = self.store.unauth.params.clone();
                        self.claims = self.store.unauth.claim_bundles();
                        self.assigned_indices = self.claims.as_ref().map(|claims| {
                            claims.iter().map(|claim| claim.registry_index).collect()
                        });
                        UnauthResponse::KappaDoubleOneClaimLeft { ok: true }
                    }
                    Err(err) => UnauthResponse::Error {
                        message: format!("unauth one-claim kappa-double failed: {:?}", err),
                    },
                }
            }
            UnauthRequest::EvalKappaDoubleTargetAtRowLabelsMatrix {
                target_s,
                segs,
                target_registry_indices,
                row_labels,
            } => match self
                .store
                .unauth
                .handle_eval_kappa_double_target_at_row_labels_matrix(
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
                match self.store.unauth.retain_registry_indices(&registry_indices) {
                    Ok(()) => {
                        self.assigned_indices = Some(registry_indices);
                        self.claims = self.store.unauth.claim_bundles();
                        UnauthResponse::RetainClaims { ok: true }
                    }
                    Err(err) => UnauthResponse::Error {
                        message: format!("unauth retain claims failed: {:?}", err),
                    },
                }
            }
            UnauthRequest::GetLen => UnauthResponse::Len {
                len: self.store.unauth.len(),
            },
            other => UnauthResponse::Error {
                message: format!("unsupported unauth message in auth mode: {:?}", other),
            },
        }
    }
}

fn normalize_indices(mut indices: Vec<usize>) -> Vec<usize> {
    indices.sort_unstable();
    indices.dedup();
    indices
}
