//! Authenticated client orchestration with transport abstraction.

use crate::auth::mapping::{self, NodeLinearization};
use crate::auth::merkle::{
    NodeKey, child_left, child_right, default_hash, label_from_content, node_hash,
};
#[cfg(feature = "bench")]
use crate::auth::read_path::reconstruct_path_from_shares_correcting_no_filter;
use crate::auth::read_path::{
    active_nodes_for_read, all_nodes_satisfied, build_merkle_range_plan_with_index_maps,
    build_partial_tree_nodes, build_path_nodes, build_subtree_read_plan, compute_range_root,
    reconstruct_path_from_shares, reconstruct_path_from_shares_correcting,
    reconstruct_subtree_from_leaves_and_upper_shares, reconstruct_subtree_from_shares,
    reconstruct_subtree_from_shares_correcting,
};
use crate::auth::service::{
    AuthBatchWriteRequest, AuthBatchWriteResponse, AuthBatchWriteUpdate,
    AuthFreshSubtreeWriteRequest, AuthPathRequest, AuthReadResponse, AuthSubtreeReadResponse,
    AuthSubtreeUpperPathReadResponse, AuthWriteRequest, AuthWriteResponse,
};
use crate::auth::{
    AuthParams, AuthPath, AuthRangeProof, AuthSubtreeProof, MerkleTreeProof, ReadError,
};
use crate::fanout::{
    broadcast_with_failures_parallel, collect_shares_by_pos_parallel, first_ok_parallel,
    spawn_parallel_fetches,
};
use crate::shares::ShardShare;
use crate::unauth::{
    CodedArrayParams, DecodeCache, DeltaMode, UnauthOptimizedReadPlan, UnauthOptimizedReadRequest,
    UnauthOptimizedReadResponse, UnauthOptimizedResponseIndex, UnauthReadBadFilter,
    decode_optimized_read_object_candidate_covering_index_indexed,
    decode_optimized_read_object_covering_index_indexed, decode_optimized_read_plan,
    error_correcting_share_target, optimized_read_satisfied, plan_optimized_read_indices,
};
use crate::value::Value256;
use std::collections::{HashMap, HashSet};

/// Errors returned by authenticated client operations.
#[derive(Clone, Debug)]
pub enum ClientError {
    Net(String),
    Registry(String),
    InvalidIndex,
    NotEnoughShards,
    InvalidProof,
    RootMismatch,
    InvalidResponse(String),
}

enum RangeCorrectionStep {
    Verified,
    Stuck,
}

struct DenseRangeLevel {
    first_pos: u64,
    slots: Vec<Option<usize>>,
}

struct RangeVerifyPlan {
    height: u32,
    start_index: u64,
    range_end: u64,
    requested_indices: Vec<u64>,
    nodes_by_pos: Vec<NodeKey>,
    leaf_slots: Vec<Option<usize>>,
    internal_levels: Vec<Option<DenseRangeLevel>>,
    boundary_slots: Vec<(NodeKey, usize)>,
}

impl RangeVerifyPlan {
    fn new(auth: &AuthParams, merkle_plan: &crate::auth::read_path::MerkleRangePlan) -> Self {
        let height = auth.tree_height as u32;
        let range_end = merkle_plan.requested_range_end();
        let mut nodes_by_pos =
            vec![NodeKey { level: 0, pos: 0 }; merkle_plan.requested_array_indices.len()];
        for (&node, &pos) in &merkle_plan.node_to_requested_pos {
            if let Some(slot) = nodes_by_pos.get_mut(pos) {
                *slot = node;
            }
        }

        let mut leaf_slots = vec![None; merkle_plan.leaf_count];
        for &node in &merkle_plan.requested_leaf_nodes {
            let Some(&requested_pos) = merkle_plan.node_to_requested_pos.get(&node) else {
                continue;
            };
            if node.level == height && node.pos >= merkle_plan.start_index {
                let offset = node.pos - merkle_plan.start_index;
                if let Some(slot) = usize::try_from(offset)
                    .ok()
                    .and_then(|offset| leaf_slots.get_mut(offset))
                {
                    *slot = Some(requested_pos);
                }
            }
        }

        let mut internal_levels = (0..=height)
            .map(|level| {
                if level == 0 || level >= height {
                    return None;
                }
                let subtree_height = height - level;
                let first_pos = merkle_plan.start_index >> subtree_height;
                let last_pos = (range_end - 1) >> subtree_height;
                let len = usize::try_from(last_pos - first_pos + 1).ok()?;
                Some(DenseRangeLevel {
                    first_pos,
                    slots: vec![None; len],
                })
            })
            .collect::<Vec<_>>();
        for &node in &merkle_plan.internal_checkpoint_nodes {
            let Some(&requested_pos) = merkle_plan.node_to_requested_pos.get(&node) else {
                continue;
            };
            let Some(Some(level_slots)) = internal_levels.get_mut(node.level as usize) else {
                continue;
            };
            if node.pos >= level_slots.first_pos {
                let offset = node.pos - level_slots.first_pos;
                if let Some(slot) = usize::try_from(offset)
                    .ok()
                    .and_then(|offset| level_slots.slots.get_mut(offset))
                {
                    *slot = Some(requested_pos);
                }
            }
        }

        let boundary_slots = merkle_plan
            .boundary_proof_nodes
            .iter()
            .filter_map(|&node| {
                merkle_plan
                    .node_to_requested_pos
                    .get(&node)
                    .copied()
                    .map(|pos| (node, pos))
            })
            .collect();

        Self {
            height,
            start_index: merkle_plan.start_index,
            range_end,
            requested_indices: merkle_plan.requested_array_indices.clone(),
            nodes_by_pos,
            leaf_slots,
            internal_levels,
            boundary_slots,
        }
    }

    fn requested_pos_for_node(&self, node: NodeKey) -> Option<usize> {
        if node.level == self.height {
            if node.pos >= self.start_index && node.pos < self.range_end {
                let offset = usize::try_from(node.pos - self.start_index).ok()?;
                if let Some(pos) = self.leaf_slots.get(offset).and_then(|slot| *slot) {
                    return Some(pos);
                }
            }
        }
        if let Some(Some(level_slots)) = self.internal_levels.get(node.level as usize) {
            if node.pos >= level_slots.first_pos {
                let offset = usize::try_from(node.pos - level_slots.first_pos).ok()?;
                if let Some(pos) = level_slots.slots.get(offset).and_then(|slot| *slot) {
                    return Some(pos);
                }
            }
        }
        self.boundary_slots
            .iter()
            .find_map(|&(candidate, pos)| (candidate == node).then_some(pos))
    }

    fn requested_pos_for_array_index(&self, array_index: u64) -> Option<usize> {
        self.requested_indices.binary_search(&array_index).ok()
    }

    fn node_for_requested_pos(&self, pos: usize) -> Option<NodeKey> {
        self.nodes_by_pos.get(pos).copied()
    }

    fn array_index_for_node(&self, node: NodeKey) -> Option<u64> {
        let pos = self.requested_pos_for_node(node)?;
        self.requested_indices.get(pos).copied()
    }

    fn value_for_node(
        &self,
        auth: &AuthParams,
        ordered_values: &[Value256],
        node: NodeKey,
    ) -> Result<Value256, ReadError> {
        if let Some(pos) = self.requested_pos_for_node(node) {
            return ordered_values
                .get(pos)
                .copied()
                .ok_or(ReadError::InvalidProof);
        }
        Ok(crate::auth::read_path::default_value_for_node(auth, node))
    }

    fn leaf_values(
        &self,
        auth: &AuthParams,
        ordered_values: &[Value256],
        merkle_plan: &crate::auth::read_path::MerkleRangePlan,
    ) -> Result<Vec<Value256>, ReadError> {
        merkle_plan
            .requested_leaf_nodes
            .iter()
            .copied()
            .map(|node| self.value_for_node(auth, ordered_values, node))
            .collect()
    }

    fn proof_nodes(
        &self,
        auth: &AuthParams,
        ordered_values: &[Value256],
        merkle_plan: &crate::auth::read_path::MerkleRangePlan,
    ) -> Result<Vec<(NodeKey, Value256)>, ReadError> {
        merkle_plan
            .boundary_proof_nodes
            .iter()
            .chain(merkle_plan.internal_checkpoint_nodes.iter())
            .copied()
            .map(|node| Ok((node, self.value_for_node(auth, ordered_values, node)?)))
            .collect()
    }
}

#[derive(Default)]
struct RangeDecodeCache {
    decode_cache: DecodeCache,
    values_by_pos: Vec<Option<Value256>>,
    labels_by_pos: Vec<Option<Value256>>,
    corrected_by_pos: Vec<bool>,
    certified_by_pos: Vec<bool>,
    candidate_provenance_by_pos: Vec<Option<usize>>,
    candidate_provenances: Vec<UnauthReadBadFilter>,
}

impl RangeDecodeCache {
    fn with_capacity(value_count: usize, object_count: usize) -> Self {
        Self {
            decode_cache: DecodeCache::new(),
            values_by_pos: vec![None; value_count],
            labels_by_pos: vec![None; value_count],
            corrected_by_pos: vec![false; value_count],
            certified_by_pos: vec![false; value_count],
            candidate_provenance_by_pos: vec![None; value_count],
            candidate_provenances: Vec::with_capacity(object_count),
        }
    }

    fn position_is_valid(&self, pos: usize, bad_filter: &UnauthReadBadFilter) -> bool {
        if self.corrected_by_pos.get(pos).copied().unwrap_or(false)
            || self.certified_by_pos.get(pos).copied().unwrap_or(false)
        {
            return true;
        }
        let Some(provenance_pos) = self
            .candidate_provenance_by_pos
            .get(pos)
            .and_then(|pos| *pos)
        else {
            return true;
        };
        self.candidate_provenances
            .get(provenance_pos)
            .is_some_and(|provenance| !provenance.overlaps(bad_filter))
    }

    fn label_for_node(
        &self,
        verify_plan: &RangeVerifyPlan,
        node: NodeKey,
        bad_filter: &UnauthReadBadFilter,
    ) -> Option<Value256> {
        let pos = verify_plan.requested_pos_for_node(node)?;
        if !self.position_is_valid(pos, bad_filter) {
            return None;
        }
        self.labels_by_pos.get(pos).and_then(|label| *label)
    }

    fn has_value_for_node(
        &self,
        verify_plan: &RangeVerifyPlan,
        node: NodeKey,
        bad_filter: &UnauthReadBadFilter,
    ) -> bool {
        let Some(pos) = verify_plan.requested_pos_for_node(node) else {
            return false;
        };
        if !self.position_is_valid(pos, bad_filter) {
            return false;
        }
        self.values_by_pos
            .get(pos)
            .is_some_and(|value| value.is_some())
    }

    fn ordered_values(&self, bad_filter: &UnauthReadBadFilter) -> Option<Vec<Value256>> {
        self.values_by_pos
            .iter()
            .enumerate()
            .map(|(pos, value)| {
                if !self.position_is_valid(pos, bad_filter) {
                    return None;
                }
                *value
            })
            .collect()
    }

    fn mark_node_certified(&mut self, verify_plan: &RangeVerifyPlan, node: NodeKey) {
        if let Some(pos) = verify_plan.requested_pos_for_node(node) {
            if let Some(slot) = self.certified_by_pos.get_mut(pos) {
                *slot = true;
            }
        }
    }

    fn node_is_corrected(&self, verify_plan: &RangeVerifyPlan, node: NodeKey) -> bool {
        let Some(pos) = verify_plan.requested_pos_for_node(node) else {
            return false;
        };
        self.corrected_by_pos.get(pos).copied().unwrap_or(false)
    }

    fn insert_candidate_values(
        &mut self,
        verify_plan: &RangeVerifyPlan,
        auth: &AuthParams,
        _object_pos: usize,
        provenance: UnauthReadBadFilter,
        decoded: Vec<(u64, Value256)>,
    ) {
        let provenance_pos = self.candidate_provenances.len();
        self.candidate_provenances.push(provenance);
        self.insert_values(verify_plan, auth, decoded, |cache, pos| {
            if !cache.corrected_by_pos.get(pos).copied().unwrap_or(false) {
                if let Some(slot) = cache.candidate_provenance_by_pos.get_mut(pos) {
                    *slot = Some(provenance_pos);
                }
            }
        });
    }

    fn insert_corrected_values(
        &mut self,
        verify_plan: &RangeVerifyPlan,
        auth: &AuthParams,
        decoded: Vec<(u64, Value256)>,
    ) {
        self.insert_values(verify_plan, auth, decoded, |cache, pos| {
            if let Some(slot) = cache.corrected_by_pos.get_mut(pos) {
                *slot = true;
            }
            if let Some(slot) = cache.candidate_provenance_by_pos.get_mut(pos) {
                *slot = None;
            }
        });
    }

    fn insert_values<F>(
        &mut self,
        verify_plan: &RangeVerifyPlan,
        auth: &AuthParams,
        decoded: Vec<(u64, Value256)>,
        mut after_insert: F,
    ) where
        F: FnMut(&mut Self, usize),
    {
        let height = auth.tree_height as u32;
        for (array_index, value) in decoded {
            if let Some(pos) = verify_plan.requested_pos_for_array_index(array_index) {
                if let Some(slot) = self.values_by_pos.get_mut(pos) {
                    *slot = Some(value);
                }
                if let Some(node) = verify_plan.node_for_requested_pos(pos) {
                    if let Some(slot) = self.labels_by_pos.get_mut(pos) {
                        *slot = Some(label_from_content(&auth.defaults, height, node, value));
                    }
                }
                after_insert(self, pos);
            }
        }
    }
}

/// Transport abstraction for authenticated client requests.
pub trait AuthTransport: Clone {
    type Server: Clone;

    fn supports_subtree_upper_path_reads(&self) -> bool {
        false
    }

    fn supports_fresh_subtree_writes(&self) -> bool {
        false
    }

    fn server_label(&self, server: &Self::Server) -> String;
    fn auth_get_len(&self, server: &Self::Server) -> Result<u64, ClientError>;
    fn auth_read(&self, server: &Self::Server, index: u64)
    -> Result<AuthReadResponse, ClientError>;
    fn auth_read_subtree(
        &self,
        server: &Self::Server,
        start_index: u64,
        leaf_count: usize,
    ) -> Result<AuthSubtreeReadResponse, ClientError>;
    fn auth_read_subtree_upper_path(
        &self,
        server: &Self::Server,
        start_index: u64,
        leaf_count: usize,
    ) -> Result<crate::auth::service::AuthSubtreeUpperPathReadResponse, ClientError> {
        let _ = (server, start_index, leaf_count);
        Err(ClientError::Net(
            "upper-path subtree reads unsupported by transport".to_string(),
        ))
    }
    fn unauth_multi_read(
        &self,
        server: &Self::Server,
        indices: &[u64],
    ) -> Result<Vec<Vec<ShardShare>>, ClientError>;
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
    fn auth_write(
        &self,
        server: &Self::Server,
        req: &AuthWriteRequest,
    ) -> Result<AuthWriteResponse, ClientError>;
    fn auth_write_batch(
        &self,
        server: &Self::Server,
        req: &AuthBatchWriteRequest,
    ) -> Result<AuthBatchWriteResponse, ClientError>;
    fn auth_write_fresh_subtree(
        &self,
        server: &Self::Server,
        req: &AuthFreshSubtreeWriteRequest,
    ) -> Result<AuthBatchWriteResponse, ClientError> {
        let _ = (server, req);
        Err(ClientError::Net(
            "fresh subtree writes unsupported by transport".to_string(),
        ))
    }
    fn auth_extend(&self, server: &Self::Server, new_len: u64) -> Result<bool, ClientError>;
}

/// Authenticated client that orchestrates reads/writes over a transport.
#[derive(Clone)]
pub struct AuthClient<T: AuthTransport> {
    pub params: CodedArrayParams,
    pub auth: AuthParams,
    pub server: T::Server,
    servers: Vec<T::Server>,
    node_linearization: NodeLinearization,
    allocated_leaves: usize,
    trusted_root: Option<Value256>,
    transport: T,
}

impl<T> AuthClient<T>
where
    T: AuthTransport + Send + Sync + 'static,
    T::Server: Clone + Send + Sync + 'static,
{
    pub fn new(
        transport: T,
        params: CodedArrayParams,
        auth: AuthParams,
        node_linearization: NodeLinearization,
        allocated_leaves: usize,
        server: T::Server,
        servers: Vec<T::Server>,
    ) -> Self {
        Self {
            params,
            auth,
            server,
            servers,
            node_linearization,
            allocated_leaves,
            trusted_root: None,
            transport,
        }
    }

    fn refresh_allocated_leaves(&mut self) -> Result<(), ClientError> {
        let transport = self.transport.clone();
        let len = first_ok_parallel(&self.servers, move |server| transport.auth_get_len(&server))
            .ok_or_else(|| {
            ClientError::Net("no auth servers responded to AuthGetLen".to_string())
        })?;
        let len_usize = usize::try_from(len)
            .map_err(|_| ClientError::InvalidResponse("auth len overflows usize".to_string()))?;
        self.allocated_leaves = len_usize;
        Ok(())
    }

    /// Return the currently trusted root, if one has been established.
    pub fn trusted_root(&self) -> Option<Value256> {
        self.trusted_root
    }

    /// Set the trusted root explicitly before the first authenticated read.
    pub fn set_trusted_root(&mut self, root: Value256) {
        self.trusted_root = Some(root);
    }

    /// Clear the trusted root so the next successful read establishes it.
    pub fn clear_trusted_root(&mut self) {
        self.trusted_root = None;
    }

    /// Return the current server-reported allocated length.
    ///
    /// This refreshes from any responsive auth server. The length is not
    /// independently threshold-verified here, so callers that treat
    /// `(trusted_root, length)` as a joint trust anchor must pin or validate
    /// the length out of band.
    pub fn current_len(&mut self) -> Result<u64, ClientError> {
        self.refresh_allocated_leaves()?;
        Ok(self.allocated_leaves as u64)
    }

    fn send_batch_write_request(&mut self, req: AuthBatchWriteRequest) -> Result<(), ClientError> {
        let mut root_new = None;
        let transport = self.transport.clone();
        let transport_for_label = transport.clone();
        let req_for_call = req.clone();
        let failed = broadcast_with_failures_parallel(
            &self.servers,
            |server| transport_for_label.server_label(server),
            move |server| transport.auth_write_batch(&server, &req_for_call),
            |resp| {
                if resp.ok {
                    let Some(resp_root) = resp.root_new else {
                        return Err("missing root_new in response".to_string());
                    };
                    if let Some(existing) = root_new {
                        if existing != resp_root {
                            return Err("root_new mismatch in response".to_string());
                        }
                    } else {
                        root_new = Some(resp_root);
                    }
                    Ok(())
                } else {
                    Err(resp
                        .error
                        .unwrap_or_else(|| "auth batch write failed".to_string()))
                }
            },
        );
        if !failed.is_empty() {
            return Err(ClientError::Net(format!(
                "auth batch write failed: {:?}",
                failed
            )));
        }
        let root_new = root_new.ok_or_else(|| {
            ClientError::InvalidResponse("missing root_new in responses".to_string())
        })?;
        self.trusted_root = Some(root_new);
        Ok(())
    }

    fn send_fresh_subtree_write_request(
        &mut self,
        req: AuthFreshSubtreeWriteRequest,
    ) -> Result<(), ClientError> {
        let mut root_new = None;
        let transport = self.transport.clone();
        let transport_for_label = transport.clone();
        let req_for_call = req.clone();
        let failed = broadcast_with_failures_parallel(
            &self.servers,
            |server| transport_for_label.server_label(server),
            move |server| transport.auth_write_fresh_subtree(&server, &req_for_call),
            |resp| {
                if resp.ok {
                    let Some(resp_root) = resp.root_new else {
                        return Err("missing root_new in response".to_string());
                    };
                    if let Some(existing) = root_new {
                        if existing != resp_root {
                            return Err("root_new mismatch in response".to_string());
                        }
                    } else {
                        root_new = Some(resp_root);
                    }
                    Ok(())
                } else {
                    Err(resp
                        .error
                        .unwrap_or_else(|| "auth fresh subtree write failed".to_string()))
                }
            },
        );
        if !failed.is_empty() {
            return Err(ClientError::Net(format!(
                "auth fresh subtree write failed: {:?}",
                failed
            )));
        }
        let root_new = root_new.ok_or_else(|| {
            ClientError::InvalidResponse("missing root_new in responses".to_string())
        })?;
        self.trusted_root = Some(root_new);
        Ok(())
    }

    fn read_default_subtree_proof_internal(
        &mut self,
        start_index: u64,
        leaf_count: usize,
    ) -> Result<AuthSubtreeProof, ClientError> {
        let plan =
            build_subtree_read_plan(&self.auth, start_index, leaf_count).map_err(map_read_error)?;
        let end = start_index
            .checked_add(usize_to_u64(leaf_count)?)
            .ok_or(ClientError::InvalidIndex)?;
        let end_usize = usize::try_from(end).map_err(|_| ClientError::InvalidIndex)?;
        if end_usize > self.allocated_leaves {
            return Err(ClientError::InvalidIndex);
        }

        let mut upper_map: HashMap<NodeKey, Vec<ShardShare>> = HashMap::new();
        let mut upper_seen: HashMap<NodeKey, HashSet<usize>> = HashMap::new();
        if !all_nodes_satisfied(
            &plan.upper_path_nodes,
            &upper_map,
            self.params.need,
            &self.auth,
            self.node_linearization,
            self.allocated_leaves,
        ) {
            let transport = self.transport.clone();
            let fetches = spawn_parallel_fetches(&self.servers, move |server| {
                transport.auth_read_subtree_upper_path(&server, start_index, leaf_count)
            });
            for _ in 0..self.servers.len() {
                let Ok((_, resp)) = fetches.recv() else {
                    break;
                };
                let AuthSubtreeUpperPathReadResponse { upper_path_nodes } = match resp {
                    Ok(resp) => resp,
                    Err(_) => continue,
                };
                for node_reply in upper_path_nodes {
                    let bucket = upper_map.entry(node_reply.node).or_insert_with(Vec::new);
                    let seen_bucket = upper_seen
                        .entry(node_reply.node)
                        .or_insert_with(HashSet::new);
                    for share in node_reply.shares {
                        if share.value.is_some() && seen_bucket.insert(share.registry_index) {
                            bucket.push(share);
                        }
                    }
                }
                if all_nodes_satisfied(
                    &plan.upper_path_nodes,
                    &upper_map,
                    self.params.need,
                    &self.auth,
                    self.node_linearization,
                    self.allocated_leaves,
                ) {
                    fetches.cancel();
                    break;
                }
            }
            if !all_nodes_satisfied(
                &plan.upper_path_nodes,
                &upper_map,
                self.params.need,
                &self.auth,
                self.node_linearization,
                self.allocated_leaves,
            ) {
                return Err(ClientError::NotEnoughShards);
            }
        }

        let default_leaves = vec![[0u8; 32]; leaf_count];
        let proof = reconstruct_subtree_from_leaves_and_upper_shares(
            &self.auth,
            &self.params,
            self.node_linearization,
            self.allocated_leaves,
            &plan,
            &default_leaves,
            &upper_map,
        )
        .map_err(map_read_error)?;
        let root = proof.root;
        if let Some(trusted) = self.trusted_root {
            if trusted != root {
                return Err(ClientError::RootMismatch);
            }
        }
        self.trusted_root = Some(root);
        Ok(proof)
    }

    fn build_leaf_batch_write_request(
        &mut self,
        updates: &[(u64, Value256)],
    ) -> Result<AuthBatchWriteRequest, ClientError> {
        if updates.is_empty() {
            return Ok(AuthBatchWriteRequest {
                updates: Vec::new(),
                proof: MerkleTreeProof {
                    root: self.trusted_root.unwrap_or([0u8; 32]),
                    nodes: Vec::new(),
                },
                delta_mode: DeltaMode::InterpolateFFT,
            });
        }
        for (index, _) in updates {
            let idx = usize::try_from(*index).map_err(|_| ClientError::InvalidIndex)?;
            if idx >= self.allocated_leaves {
                return Err(ClientError::InvalidIndex);
            }
        }
        let indices: Vec<u64> = updates.iter().map(|(idx, _)| *idx).collect();
        let read_results = self.read_batch_internal(&indices)?;

        let mut proof_nodes: HashMap<NodeKey, Value256> = HashMap::new();
        let mut root_old = None;
        let mut net_updates = Vec::with_capacity(updates.len());

        for ((index, new), (old, path)) in updates.iter().zip(read_results.into_iter()) {
            let idx = usize::try_from(*index).map_err(|_| ClientError::InvalidIndex)?;
            if path.index != idx {
                return Err(ClientError::InvalidResponse(
                    "path index mismatch in batch write".to_string(),
                ));
            }
            if let Some(existing) = root_old {
                if existing != path.root {
                    return Err(ClientError::RootMismatch);
                }
            } else {
                root_old = Some(path.root);
            }
            net_updates.push(AuthBatchWriteUpdate {
                index: *index,
                old,
                new: *new,
            });

            let nodes = build_path_nodes(&self.auth, *index);
            if nodes.len() != path.siblings.len() + 1 {
                return Err(ClientError::InvalidResponse(
                    "path height mismatch in batch write".to_string(),
                ));
            }
            insert_proof_node(&mut proof_nodes, nodes[0], old)?;
            for (node, sibling) in nodes.iter().skip(1).zip(path.siblings.iter()) {
                insert_proof_node(&mut proof_nodes, *node, *sibling)?;
            }
        }

        let root_old = root_old.ok_or_else(|| {
            ClientError::InvalidResponse("missing proof root in batch write".to_string())
        })?;
        let mut proof_nodes_vec: Vec<(NodeKey, Value256)> = proof_nodes.into_iter().collect();
        proof_nodes_vec.sort_by_key(|(node, _)| (node.level, node.pos));
        Ok(AuthBatchWriteRequest {
            updates: net_updates,
            proof: MerkleTreeProof {
                root: root_old,
                nodes: proof_nodes_vec,
            },
            delta_mode: DeltaMode::InterpolateFFT,
        })
    }

    fn build_subtree_batch_write_request(
        &mut self,
        start_index: u64,
        leaves: &[Value256],
    ) -> Result<AuthBatchWriteRequest, ClientError> {
        let leaf_count_u64 = usize_to_u64(leaves.len())?;
        let plan = build_subtree_read_plan(&self.auth, start_index, leaves.len())
            .map_err(map_read_error)?;
        let end = start_index
            .checked_add(leaf_count_u64)
            .ok_or(ClientError::InvalidIndex)?;
        let end_usize = usize::try_from(end).map_err(|_| ClientError::InvalidIndex)?;
        if end_usize > self.allocated_leaves {
            return Err(ClientError::InvalidIndex);
        }
        let (old_leaves, subtree_proof) = self.read_subtree_internal(start_index, leaves.len())?;
        let proof_nodes = plan
            .upper_path_nodes
            .iter()
            .copied()
            .zip(subtree_proof.upper_siblings.iter().copied())
            .collect();
        let updates = old_leaves
            .into_iter()
            .zip(leaves.iter().copied())
            .enumerate()
            .map(|(offset, (old, new))| {
                let offset = usize_to_u64(offset)?;
                let index = start_index
                    .checked_add(offset)
                    .ok_or(ClientError::InvalidIndex)?;
                Ok(AuthBatchWriteUpdate { index, old, new })
            })
            .collect::<Result<Vec<_>, ClientError>>()?;
        Ok(AuthBatchWriteRequest {
            updates,
            proof: MerkleTreeProof {
                root: subtree_proof.root,
                nodes: proof_nodes,
            },
            delta_mode: DeltaMode::InterpolateFFT,
        })
    }

    fn write_batch_unchecked(&mut self, updates: &[(u64, Value256)]) -> Result<(), ClientError> {
        if updates.is_empty() {
            return Ok(());
        }
        let req = self.build_leaf_batch_write_request(updates)?;
        self.submit_batch_write_request(&req)
    }

    fn write_subtree_unchecked(
        &mut self,
        start_index: u64,
        leaves: &[Value256],
    ) -> Result<(), ClientError> {
        if leaves.is_empty() {
            return Ok(());
        }
        let req = self.build_subtree_batch_write_request(start_index, leaves)?;
        self.send_batch_write_request(req)
    }

    fn extend_allocated_leaves(&mut self, new_len: u64) -> Result<(), ClientError> {
        if new_len as u128 > self.auth.conceptual_l_max_u128() {
            return Err(ClientError::InvalidIndex);
        }
        let new_len_usize = usize::try_from(new_len).map_err(|_| ClientError::InvalidIndex)?;
        if new_len_usize <= self.allocated_leaves {
            return Ok(());
        }
        let transport = self.transport.clone();
        let transport_for_label = transport.clone();
        let failed = broadcast_with_failures_parallel(
            &self.servers,
            |server| transport_for_label.server_label(server),
            move |server| transport.auth_extend(&server, new_len),
            |ok| {
                if ok {
                    Ok(())
                } else {
                    Err("auth extend failed".to_string())
                }
            },
        );
        if !failed.is_empty() {
            return Err(ClientError::Net(format!(
                "auth extend failed: {:?}",
                failed
            )));
        }
        self.allocated_leaves = new_len_usize;
        Ok(())
    }

    /// Read a contiguous leaf range.
    pub fn read_range(
        &mut self,
        start: u64,
        leaf_count: usize,
    ) -> Result<Vec<Value256>, ClientError> {
        self.refresh_allocated_leaves()?;
        if leaf_count == 0 {
            return Ok(Vec::new());
        }
        let leaf_count_u64 = usize_to_u64(leaf_count)?;
        if self.transport.supports_unauth_optimized_reads() {
            return self
                .read_range_optimized_target(start, leaf_count, self.params.need, None)
                .map(|(values, _)| values);
        }
        if leaf_count.is_power_of_two() && start % leaf_count_u64 == 0 {
            return self
                .read_subtree_internal(start, leaf_count)
                .map(|(values, _)| values);
        }
        let end = start
            .checked_add(leaf_count_u64)
            .ok_or(ClientError::InvalidIndex)?;
        if end as u128 > self.auth.conceptual_l_max_u128() {
            return Err(ClientError::InvalidIndex);
        }
        let indices: Vec<u64> = (start..end).collect();
        let values = self.read_batch_internal(&indices)?;
        Ok(values.into_iter().map(|(value, _)| value).collect())
    }

    /// Read a contiguous leaf range and return a canonical range proof.
    pub fn read_range_with_proof(
        &mut self,
        start: u64,
        leaf_count: usize,
    ) -> Result<(Vec<Value256>, AuthRangeProof), ClientError> {
        self.refresh_allocated_leaves()?;
        if leaf_count == 0 {
            return Err(ClientError::InvalidIndex);
        }
        if !self.transport.supports_unauth_optimized_reads() {
            return Err(ClientError::Net(
                "range proofs require unauth optimized reads on this transport".to_string(),
            ));
        }
        self.read_range_optimized_target(start, leaf_count, self.params.need, None)
    }

    /// Process pre-collected optimized unauth range responses.
    ///
    /// This mirrors [`Self::read_with_path_from_responses`] for contiguous
    /// range openings: helper replies have already arrived, so the method
    /// performs only client-side aggregation, unauth decoding, range proof
    /// assembly, Merkle-root computation, and trusted-root checking.
    pub fn read_range_with_proof_from_optimized_responses(
        &mut self,
        start: u64,
        leaf_count: usize,
        responses: &[UnauthOptimizedReadResponse],
    ) -> Result<(Vec<Value256>, AuthRangeProof), ClientError> {
        self.read_range_optimized_from_responses_internal(
            start,
            leaf_count,
            responses,
            self.params.need,
            None,
        )
    }

    /// Process pre-collected optimized unauth range responses while correcting
    /// up to `max_errors` bad helper replies.
    ///
    /// Correcting optimized range reads require a trusted root. The client uses
    /// that root to run Merkle-guided lazy object correction; there is no
    /// whole-range correcting decode pass for this API.
    pub fn read_range_with_proof_from_optimized_responses_correcting(
        &mut self,
        start: u64,
        leaf_count: usize,
        responses: &[UnauthOptimizedReadResponse],
        max_errors: usize,
    ) -> Result<(Vec<Value256>, AuthRangeProof), ClientError> {
        let share_target = error_correcting_share_target(self.params.need, max_errors);
        self.read_range_optimized_from_responses_internal(
            start,
            leaf_count,
            responses,
            share_target,
            Some(max_errors),
        )
    }

    /// Read a contiguous leaf range while correcting up to `max_errors` bad
    /// helper replies for every active decoded node.
    pub fn read_range_correcting(
        &mut self,
        start: u64,
        leaf_count: usize,
        max_errors: usize,
    ) -> Result<Vec<Value256>, ClientError> {
        self.refresh_allocated_leaves()?;
        if self.trusted_root.is_none() {
            return Err(ClientError::RootMismatch);
        }
        if leaf_count == 0 {
            return Ok(Vec::new());
        }
        let share_target = error_correcting_share_target(self.params.need, max_errors);
        let leaf_count_u64 = usize_to_u64(leaf_count)?;
        if self.transport.supports_unauth_optimized_reads() {
            return self
                .read_range_optimized_target(start, leaf_count, share_target, Some(max_errors))
                .map(|(values, _)| values);
        }
        if leaf_count.is_power_of_two() && start % leaf_count_u64 == 0 {
            return self
                .read_subtree_internal_target(start, leaf_count, share_target, Some(max_errors))
                .map(|(values, _)| values);
        }
        let end = start
            .checked_add(leaf_count_u64)
            .ok_or(ClientError::InvalidIndex)?;
        if end as u128 > self.auth.conceptual_l_max_u128() {
            return Err(ClientError::InvalidIndex);
        }
        let indices: Vec<u64> = (start..end).collect();
        let values = self.read_batch_internal_target(&indices, share_target, Some(max_errors))?;
        Ok(values.into_iter().map(|(value, _)| value).collect())
    }

    fn read_range_optimized_target(
        &mut self,
        start: u64,
        leaf_count: usize,
        share_target: usize,
        max_errors: Option<usize>,
    ) -> Result<(Vec<Value256>, AuthRangeProof), ClientError> {
        if leaf_count == 0 {
            return Err(ClientError::InvalidIndex);
        }
        let leaf_count_u64 = usize_to_u64(leaf_count)?;
        let end = start
            .checked_add(leaf_count_u64)
            .ok_or(ClientError::InvalidIndex)?;
        if end as u128 > self.auth.conceptual_l_max_u128() {
            return Err(ClientError::InvalidIndex);
        }
        let (merkle_plan, unauth_plan) =
            self.plan_range_optimized(start, leaf_count, share_target, max_errors)?;

        let mut responses = Vec::new();
        if !unauth_plan.request.objects.is_empty() {
            let transport = self.transport.clone();
            let request = unauth_plan.request.clone();
            let fetches = spawn_parallel_fetches(&self.servers, move |server| {
                transport.unauth_optimized_read(&server, &request)
            });
            for _ in 0..self.servers.len() {
                let Ok((helper_index, reply)) = fetches.recv() else {
                    break;
                };
                let Ok(mut response) = reply else {
                    continue;
                };
                response.helper_index = Some(helper_index);
                responses.push(response);
                if optimized_read_satisfied(&unauth_plan, &responses) {
                    fetches.cancel();
                    break;
                }
            }
            if !optimized_read_satisfied(&unauth_plan, &responses) {
                return Err(ClientError::NotEnoughShards);
            }
        }

        self.finish_range_optimized_from_responses(
            start,
            leaf_count,
            &merkle_plan,
            &unauth_plan,
            &responses,
        )
    }

    fn read_range_optimized_from_responses_internal(
        &mut self,
        start: u64,
        leaf_count: usize,
        responses: &[UnauthOptimizedReadResponse],
        share_target: usize,
        max_errors: Option<usize>,
    ) -> Result<(Vec<Value256>, AuthRangeProof), ClientError> {
        if leaf_count == 0 {
            return Err(ClientError::InvalidIndex);
        }
        let leaf_count_u64 = usize_to_u64(leaf_count)?;
        let end = start
            .checked_add(leaf_count_u64)
            .ok_or(ClientError::InvalidIndex)?;
        if end as u128 > self.auth.conceptual_l_max_u128() {
            return Err(ClientError::InvalidIndex);
        }
        let (merkle_plan, unauth_plan) =
            self.plan_range_optimized(start, leaf_count, share_target, max_errors)?;
        if !optimized_read_satisfied(&unauth_plan, responses) {
            return Err(ClientError::NotEnoughShards);
        }
        self.finish_range_optimized_from_responses(
            start,
            leaf_count,
            &merkle_plan,
            &unauth_plan,
            responses,
        )
    }

    fn plan_range_optimized(
        &self,
        start: u64,
        leaf_count: usize,
        share_target: usize,
        max_errors: Option<usize>,
    ) -> Result<
        (
            crate::auth::read_path::MerkleRangePlan,
            crate::unauth::UnauthOptimizedReadPlan,
        ),
        ClientError,
    > {
        let merkle_plan = build_merkle_range_plan_with_index_maps(
            &self.auth,
            self.node_linearization,
            self.allocated_leaves,
            start,
            leaf_count,
            max_errors.is_some(),
        )
        .map_err(map_read_error)?;
        let unauth_len = mapping::cursor(
            self.node_linearization,
            self.auth.tree_height as u16,
            self.allocated_leaves as u64,
        );
        let unauth_plan = plan_optimized_read_indices(
            &self.params,
            unauth_len,
            &merkle_plan.requested_array_indices,
            share_target,
            max_errors,
        );
        Ok((merkle_plan, unauth_plan))
    }

    fn finish_range_optimized_from_responses(
        &mut self,
        start: u64,
        leaf_count: usize,
        merkle_plan: &crate::auth::read_path::MerkleRangePlan,
        unauth_plan: &crate::unauth::UnauthOptimizedReadPlan,
        responses: &[UnauthOptimizedReadResponse],
    ) -> Result<(Vec<Value256>, AuthRangeProof), ClientError> {
        if unauth_plan.max_errors.is_some() {
            if self.trusted_root.is_none() {
                return Err(ClientError::RootMismatch);
            }
            self.finish_range_optimized_lazy_correcting(
                start,
                leaf_count,
                merkle_plan,
                unauth_plan,
                responses,
            )
        } else {
            self.finish_range_optimized_from_responses_with_decoder(
                start,
                leaf_count,
                merkle_plan,
                unauth_plan,
                responses,
                decode_optimized_read_plan,
            )
        }
    }

    fn finish_range_optimized_from_responses_with_decoder(
        &mut self,
        start: u64,
        leaf_count: usize,
        merkle_plan: &crate::auth::read_path::MerkleRangePlan,
        unauth_plan: &crate::unauth::UnauthOptimizedReadPlan,
        responses: &[UnauthOptimizedReadResponse],
        decoder: fn(
            &CodedArrayParams,
            &crate::unauth::UnauthOptimizedReadPlan,
            &[UnauthOptimizedReadResponse],
        ) -> Result<Vec<Value256>, crate::unauth::CodedReadError>,
    ) -> Result<(Vec<Value256>, AuthRangeProof), ClientError> {
        let decoded = self.decode_range_values_ordered(unauth_plan, responses, decoder)?;
        self.assemble_range_from_ordered_values(
            start,
            leaf_count,
            merkle_plan,
            unauth_plan,
            &decoded,
        )
    }

    fn finish_range_optimized_lazy_correcting(
        &mut self,
        start: u64,
        leaf_count: usize,
        merkle_plan: &crate::auth::read_path::MerkleRangePlan,
        unauth_plan: &UnauthOptimizedReadPlan,
        responses: &[UnauthOptimizedReadResponse],
    ) -> Result<(Vec<Value256>, AuthRangeProof), ClientError> {
        if !optimized_read_satisfied(unauth_plan, responses) {
            return Err(ClientError::NotEnoughShards);
        }
        let verify_plan = RangeVerifyPlan::new(&self.auth, merkle_plan);
        let mut cache = RangeDecodeCache::with_capacity(
            verify_plan.requested_indices.len(),
            unauth_plan.request.objects.len(),
        );
        let response_index =
            UnauthOptimizedResponseIndex::new(unauth_plan.request.objects.len(), responses);
        let mut corrected_object_positions = HashSet::new();
        let mut bad_filter = UnauthReadBadFilter::default();
        match self.verify_range_tree_or_correct(
            &verify_plan,
            unauth_plan,
            &response_index,
            &mut cache,
            &mut corrected_object_positions,
            &mut bad_filter,
        )? {
            RangeCorrectionStep::Verified => {
                let root = self.trusted_root.ok_or(ClientError::RootMismatch)?;
                let ordered_values = cache
                    .ordered_values(&bad_filter)
                    .ok_or(ClientError::NotEnoughShards)?;
                self.assemble_verified_range_from_ordered_values(
                    start,
                    leaf_count,
                    &verify_plan,
                    merkle_plan,
                    &ordered_values,
                    root,
                )
            }
            RangeCorrectionStep::Stuck => Err(ClientError::RootMismatch),
        }
    }

    fn verify_range_tree_or_correct(
        &self,
        verify_plan: &RangeVerifyPlan,
        correcting_plan: &UnauthOptimizedReadPlan,
        response_index: &UnauthOptimizedResponseIndex<'_>,
        cache: &mut RangeDecodeCache,
        corrected_object_positions: &mut HashSet<usize>,
        bad_filter: &mut UnauthReadBadFilter,
    ) -> Result<RangeCorrectionStep, ClientError> {
        let Some(trusted_root) = self.trusted_root else {
            return Ok(RangeCorrectionStep::Stuck);
        };
        self.verify_range_node_or_correct(
            NodeKey { level: 0, pos: 0 },
            trusted_root,
            verify_plan,
            correcting_plan,
            response_index,
            cache,
            corrected_object_positions,
            bad_filter,
        )
    }

    fn verify_range_node_or_correct(
        &self,
        node: NodeKey,
        trusted_label: Value256,
        verify_plan: &RangeVerifyPlan,
        correcting_plan: &UnauthOptimizedReadPlan,
        response_index: &UnauthOptimizedResponseIndex<'_>,
        cache: &mut RangeDecodeCache,
        corrected_object_positions: &mut HashSet<usize>,
        bad_filter: &mut UnauthReadBadFilter,
    ) -> Result<RangeCorrectionStep, ClientError> {
        loop {
            cache.mark_node_certified(verify_plan, node);
            if node.level >= verify_plan.height || !subtree_has_requested_leaf(node, verify_plan) {
                let label = self.ensure_range_node_label(
                    node,
                    verify_plan,
                    correcting_plan,
                    response_index,
                    cache,
                    bad_filter,
                )?;
                if label == trusted_label {
                    cache.mark_node_certified(verify_plan, node);
                    return Ok(RangeCorrectionStep::Verified);
                }
                if self.correct_range_node_object(
                    node,
                    verify_plan,
                    correcting_plan,
                    response_index,
                    cache,
                    corrected_object_positions,
                    bad_filter,
                )? {
                    continue;
                }
                return Ok(RangeCorrectionStep::Stuck);
            }

            let left = child_left(node);
            let right = child_right(node);
            let left_label = self.ensure_range_node_label(
                left,
                verify_plan,
                correcting_plan,
                response_index,
                cache,
                bad_filter,
            )?;
            let right_label = self.ensure_range_node_label(
                right,
                verify_plan,
                correcting_plan,
                response_index,
                cache,
                bad_filter,
            )?;
            if node_hash(left_label, right_label) != trusted_label {
                let suspect = if self.range_child_label_disagrees_with_children(
                    left,
                    left_label,
                    verify_plan,
                    correcting_plan,
                    response_index,
                    cache,
                    bad_filter,
                )? {
                    left
                } else if self.range_child_label_disagrees_with_children(
                    right,
                    right_label,
                    verify_plan,
                    correcting_plan,
                    response_index,
                    cache,
                    bad_filter,
                )? {
                    right
                } else {
                    choose_suspect_child(
                        left,
                        left_label,
                        right,
                        right_label,
                        cache,
                        verify_plan,
                        bad_filter,
                    )
                };
                let suspect_label = if suspect == left {
                    left_label
                } else {
                    right_label
                };
                if cache.node_is_corrected(verify_plan, suspect)
                    && subtree_has_requested_leaf(suspect, verify_plan)
                {
                    match self.verify_range_node_or_correct(
                        suspect,
                        suspect_label,
                        verify_plan,
                        correcting_plan,
                        response_index,
                        cache,
                        corrected_object_positions,
                        bad_filter,
                    )? {
                        RangeCorrectionStep::Verified => continue,
                        RangeCorrectionStep::Stuck => return Ok(RangeCorrectionStep::Stuck),
                    }
                }
                if self.correct_range_node_object(
                    suspect,
                    verify_plan,
                    correcting_plan,
                    response_index,
                    cache,
                    corrected_object_positions,
                    bad_filter,
                )? {
                    continue;
                }
                if subtree_has_requested_leaf(suspect, verify_plan) {
                    match self.verify_range_node_or_correct(
                        suspect,
                        suspect_label,
                        verify_plan,
                        correcting_plan,
                        response_index,
                        cache,
                        corrected_object_positions,
                        bad_filter,
                    )? {
                        RangeCorrectionStep::Verified => continue,
                        RangeCorrectionStep::Stuck => return Ok(RangeCorrectionStep::Stuck),
                    }
                }
                return Ok(RangeCorrectionStep::Stuck);
            }

            cache.mark_node_certified(verify_plan, left);
            cache.mark_node_certified(verify_plan, right);

            match self.verify_range_node_or_correct(
                left,
                left_label,
                verify_plan,
                correcting_plan,
                response_index,
                cache,
                corrected_object_positions,
                bad_filter,
            )? {
                RangeCorrectionStep::Verified => {}
                RangeCorrectionStep::Stuck => return Ok(RangeCorrectionStep::Stuck),
            }
            return self.verify_range_node_or_correct(
                right,
                right_label,
                verify_plan,
                correcting_plan,
                response_index,
                cache,
                corrected_object_positions,
                bad_filter,
            );
        }
    }

    fn range_child_label_disagrees_with_children(
        &self,
        node: NodeKey,
        label: Value256,
        verify_plan: &RangeVerifyPlan,
        correcting_plan: &UnauthOptimizedReadPlan,
        response_index: &UnauthOptimizedResponseIndex<'_>,
        cache: &mut RangeDecodeCache,
        bad_filter: &mut UnauthReadBadFilter,
    ) -> Result<bool, ClientError> {
        if node.level >= verify_plan.height || !subtree_has_requested_leaf(node, verify_plan) {
            return Ok(false);
        }
        let left = child_left(node);
        let right = child_right(node);
        let left_label = self.ensure_range_node_label(
            left,
            verify_plan,
            correcting_plan,
            response_index,
            cache,
            bad_filter,
        )?;
        let right_label = self.ensure_range_node_label(
            right,
            verify_plan,
            correcting_plan,
            response_index,
            cache,
            bad_filter,
        )?;
        Ok(node_hash(left_label, right_label) != label)
    }

    fn ensure_range_node_label(
        &self,
        node: NodeKey,
        verify_plan: &RangeVerifyPlan,
        correcting_plan: &UnauthOptimizedReadPlan,
        response_index: &UnauthOptimizedResponseIndex<'_>,
        cache: &mut RangeDecodeCache,
        bad_filter: &mut UnauthReadBadFilter,
    ) -> Result<Value256, ClientError> {
        if let Some(label) = cache.label_for_node(verify_plan, node, bad_filter) {
            return Ok(label);
        }
        let Some(array_index) = verify_plan.array_index_for_node(node) else {
            return Ok(default_hash(&self.auth.defaults, node.level as usize));
        };
        if !cache.has_value_for_node(verify_plan, node, bad_filter) {
            let (object_pos, decoded) =
                decode_optimized_read_object_candidate_covering_index_indexed(
                    &self.params,
                    correcting_plan,
                    response_index,
                    array_index,
                    Some(bad_filter),
                    &cache.decode_cache,
                )
                .map_err(|_| ClientError::NotEnoughShards)?;
            cache.insert_candidate_values(
                verify_plan,
                &self.auth,
                object_pos,
                decoded.provenance,
                decoded.values,
            );
        }
        cache
            .label_for_node(verify_plan, node, bad_filter)
            .ok_or(ClientError::NotEnoughShards)
    }

    fn correct_range_node_object(
        &self,
        node: NodeKey,
        verify_plan: &RangeVerifyPlan,
        correcting_plan: &UnauthOptimizedReadPlan,
        response_index: &UnauthOptimizedResponseIndex<'_>,
        cache: &mut RangeDecodeCache,
        corrected_object_positions: &mut HashSet<usize>,
        bad_filter: &mut UnauthReadBadFilter,
    ) -> Result<bool, ClientError> {
        let Some(array_index) = active_descendant_array_index(node, cache, verify_plan, bad_filter)
        else {
            return Ok(false);
        };
        let (object_pos, decoded) = decode_optimized_read_object_covering_index_indexed(
            &self.params,
            correcting_plan,
            response_index,
            array_index,
            corrected_object_positions,
            Some(bad_filter),
        )
        .map_err(|_| ClientError::NotEnoughShards)?;
        corrected_object_positions.insert(object_pos);
        cache.insert_corrected_values(verify_plan, &self.auth, decoded);
        Ok(true)
    }

    fn decode_range_values_ordered(
        &self,
        unauth_plan: &UnauthOptimizedReadPlan,
        responses: &[UnauthOptimizedReadResponse],
        decoder: fn(
            &CodedArrayParams,
            &crate::unauth::UnauthOptimizedReadPlan,
            &[UnauthOptimizedReadResponse],
        ) -> Result<Vec<Value256>, crate::unauth::CodedReadError>,
    ) -> Result<Vec<Value256>, ClientError> {
        let decoded = decoder(&self.params, unauth_plan, responses)
            .map_err(|_| ClientError::NotEnoughShards)?;
        if decoded.len() != unauth_plan.requested_indices.len() {
            return Err(ClientError::InvalidResponse(
                "decoded optimized read length mismatch".to_string(),
            ));
        }
        Ok(decoded)
    }

    fn assemble_range_from_ordered_values(
        &mut self,
        start: u64,
        leaf_count: usize,
        merkle_plan: &crate::auth::read_path::MerkleRangePlan,
        unauth_plan: &UnauthOptimizedReadPlan,
        ordered_values: &[Value256],
    ) -> Result<(Vec<Value256>, AuthRangeProof), ClientError> {
        if ordered_values.len() != unauth_plan.requested_indices.len() {
            return Err(ClientError::InvalidResponse(
                "decoded optimized read length mismatch".to_string(),
            ));
        }
        let leaves = merkle_plan
            .leaf_values_from_ordered_array_values(
                &self.auth,
                &unauth_plan.requested_indices,
                ordered_values,
            )
            .map_err(map_read_error)?;
        let proof_nodes = merkle_plan
            .proof_nodes_from_ordered_array_values(
                &self.auth,
                &unauth_plan.requested_indices,
                ordered_values,
            )
            .map_err(map_read_error)?;
        let root =
            compute_range_root(&self.auth, start, &leaves, &proof_nodes).map_err(map_read_error)?;
        let proof = AuthRangeProof {
            start_index: start,
            leaf_count,
            allocated_leaves: self.allocated_leaves as u64,
            proof_nodes,
            root,
        };
        if let Some(trusted) = self.trusted_root {
            if trusted != root {
                return Err(ClientError::RootMismatch);
            }
        }
        self.trusted_root = Some(root);
        Ok((leaves, proof))
    }

    fn assemble_verified_range_from_ordered_values(
        &mut self,
        start: u64,
        leaf_count: usize,
        verify_plan: &RangeVerifyPlan,
        merkle_plan: &crate::auth::read_path::MerkleRangePlan,
        ordered_values: &[Value256],
        root: Value256,
    ) -> Result<(Vec<Value256>, AuthRangeProof), ClientError> {
        let leaves = verify_plan
            .leaf_values(&self.auth, ordered_values, merkle_plan)
            .map_err(map_read_error)?;
        let proof_nodes = verify_plan
            .proof_nodes(&self.auth, ordered_values, merkle_plan)
            .map_err(map_read_error)?;
        let proof = AuthRangeProof {
            start_index: start,
            leaf_count,
            allocated_leaves: self.allocated_leaves as u64,
            proof_nodes,
            root,
        };
        self.trusted_root = Some(root);
        Ok((leaves, proof))
    }

    /// Append a contiguous slice of leaves at the current end of the array.
    ///
    /// This is a convenience helper over `AuthExtend` followed by a batched
    /// authenticated write. It is not an atomic append primitive: if the write
    /// fails after the extend succeeds, the allocated length may already have
    /// increased on the servers.
    pub fn append_leaves(&mut self, leaves: &[Value256]) -> Result<u64, ClientError> {
        self.refresh_allocated_leaves()?;
        if leaves.is_empty() {
            return Ok(self.allocated_leaves as u64);
        }
        let leaves_len_u64 = usize_to_u64(leaves.len())?;
        let start = self.allocated_leaves as u64;
        let new_len = start
            .checked_add(leaves_len_u64)
            .ok_or(ClientError::InvalidIndex)?;
        self.extend_allocated_leaves(new_len)?;
        let can_use_fresh_subtree_path = self.transport.supports_subtree_upper_path_reads()
            && self.transport.supports_fresh_subtree_writes();
        if can_use_fresh_subtree_path
            && leaves.len().is_power_of_two()
            && start % leaves_len_u64 == 0
        {
            let proof = self.read_default_subtree_proof_internal(start, leaves.len())?;
            self.write_fresh_subtree_with_proof(proof, leaves)?;
        } else {
            let updates: Vec<_> = leaves
                .iter()
                .copied()
                .enumerate()
                .map(|(offset, value)| {
                    let offset = usize_to_u64(offset)?;
                    let index = start.checked_add(offset).ok_or(ClientError::InvalidIndex)?;
                    Ok((index, value))
                })
                .collect::<Result<Vec<_>, ClientError>>()?;
            self.write_batch_unchecked(&updates)?;
        }
        Ok(start)
    }

    fn read_with_path_internal(&mut self, index: u64) -> Result<(Value256, AuthPath), ClientError> {
        self.read_with_path_internal_target(index, self.params.need, None)
    }

    fn read_with_path_internal_target(
        &mut self,
        index: u64,
        share_target: usize,
        max_errors: Option<usize>,
    ) -> Result<(Value256, AuthPath), ClientError> {
        if index as u128 >= self.auth.conceptual_l_max_u128() {
            return Err(ClientError::InvalidIndex);
        }
        let min_shares = max_errors
            .map(|max_errors| self.params.need + 2 * max_errors)
            .unwrap_or(share_target);
        let nodes = build_path_nodes(&self.auth, index);
        let node_set: HashSet<NodeKey> = nodes.iter().copied().collect();

        let mut shares: HashMap<NodeKey, Vec<ShardShare>> = HashMap::new();
        let mut seen: HashMap<NodeKey, HashSet<usize>> = HashMap::new();
        if !all_nodes_satisfied(
            &nodes,
            &shares,
            share_target,
            &self.auth,
            self.node_linearization,
            self.allocated_leaves,
        ) {
            let transport = self.transport.clone();
            let fetches = spawn_parallel_fetches(&self.servers, move |server| {
                transport.auth_read(&server, index)
            });
            for _ in 0..self.servers.len() {
                let Ok((_, resp)) = fetches.recv() else {
                    break;
                };
                let resp = match resp {
                    Ok(resp) => resp,
                    Err(_) => continue,
                };
                for node_reply in resp.nodes {
                    let key = node_reply.node;
                    if !node_set.contains(&key) {
                        continue;
                    }
                    let bucket = shares.entry(key).or_insert_with(Vec::new);
                    let seen_bucket = seen.entry(key).or_insert_with(HashSet::new);
                    for share in node_reply.shares {
                        if share.value.is_some() && seen_bucket.insert(share.registry_index) {
                            bucket.push(share);
                        }
                    }
                }
                if all_nodes_satisfied(
                    &nodes,
                    &shares,
                    share_target,
                    &self.auth,
                    self.node_linearization,
                    self.allocated_leaves,
                ) {
                    fetches.cancel();
                    break;
                }
            }
            if !all_nodes_satisfied(
                &nodes,
                &shares,
                min_shares,
                &self.auth,
                self.node_linearization,
                self.allocated_leaves,
            ) {
                return Err(ClientError::NotEnoughShards);
            }
        }
        let (value, path) = match max_errors {
            Some(max_errors) => reconstruct_path_from_shares_correcting(
                &self.auth,
                &self.params,
                self.node_linearization,
                self.allocated_leaves,
                index,
                &nodes,
                &shares,
                max_errors,
            ),
            None => reconstruct_path_from_shares(
                &self.auth,
                &self.params,
                self.node_linearization,
                self.allocated_leaves,
                index,
                &nodes,
                &shares,
            ),
        }
        .map_err(map_read_error)?;
        let root = path.root;
        if let Some(trusted) = self.trusted_root {
            if trusted != root {
                return Err(ClientError::RootMismatch);
            }
        }
        self.trusted_root = Some(root);
        Ok((value, path))
    }

    fn read_with_path_from_responses_internal(
        &mut self,
        index: u64,
        responses: &[AuthReadResponse],
        max_errors: Option<usize>,
    ) -> Result<(Value256, AuthPath), ClientError> {
        if index as u128 >= self.auth.conceptual_l_max_u128() {
            return Err(ClientError::InvalidIndex);
        }
        let min_shares = max_errors
            .map(|max_errors| self.params.need + 2 * max_errors)
            .unwrap_or(self.params.need);
        let nodes = build_path_nodes(&self.auth, index);
        let node_set: HashSet<NodeKey> = nodes.iter().copied().collect();

        let mut shares: HashMap<NodeKey, Vec<ShardShare>> = HashMap::new();
        let mut seen: HashMap<NodeKey, HashSet<usize>> = HashMap::new();
        for resp in responses {
            for node_reply in &resp.nodes {
                let key = node_reply.node;
                if !node_set.contains(&key) {
                    continue;
                }
                let bucket = shares.entry(key).or_insert_with(Vec::new);
                let seen_bucket = seen.entry(key).or_insert_with(HashSet::new);
                for share in &node_reply.shares {
                    if share.value.is_some() && seen_bucket.insert(share.registry_index) {
                        bucket.push(share.clone());
                    }
                }
            }
        }
        if !all_nodes_satisfied(
            &nodes,
            &shares,
            min_shares,
            &self.auth,
            self.node_linearization,
            self.allocated_leaves,
        ) {
            return Err(ClientError::NotEnoughShards);
        }

        let (value, path) = match max_errors {
            Some(max_errors) => reconstruct_path_from_shares_correcting(
                &self.auth,
                &self.params,
                self.node_linearization,
                self.allocated_leaves,
                index,
                &nodes,
                &shares,
                max_errors,
            ),
            None => reconstruct_path_from_shares(
                &self.auth,
                &self.params,
                self.node_linearization,
                self.allocated_leaves,
                index,
                &nodes,
                &shares,
            ),
        }
        .map_err(map_read_error)?;
        let root = path.root;
        if let Some(trusted) = self.trusted_root {
            if trusted != root {
                return Err(ClientError::RootMismatch);
            }
        }
        self.trusted_root = Some(root);
        Ok((value, path))
    }

    fn read_subtree_internal(
        &mut self,
        start_index: u64,
        leaf_count: usize,
    ) -> Result<(Vec<Value256>, AuthSubtreeProof), ClientError> {
        self.read_subtree_internal_target(start_index, leaf_count, self.params.need, None)
    }

    fn read_subtree_internal_target(
        &mut self,
        start_index: u64,
        leaf_count: usize,
        share_target: usize,
        max_errors: Option<usize>,
    ) -> Result<(Vec<Value256>, AuthSubtreeProof), ClientError> {
        let plan =
            build_subtree_read_plan(&self.auth, start_index, leaf_count).map_err(map_read_error)?;
        let min_shares = max_errors
            .map(|max_errors| self.params.need + 2 * max_errors)
            .unwrap_or(share_target);

        let mut nodes = plan.leaf_nodes.clone();
        nodes.extend(plan.upper_path_nodes.iter().copied());
        let node_set: HashSet<NodeKey> = nodes.iter().copied().collect();

        let mut shares: HashMap<NodeKey, Vec<ShardShare>> = HashMap::new();
        let mut seen: HashMap<NodeKey, HashSet<usize>> = HashMap::new();
        if !all_nodes_satisfied(
            &nodes,
            &shares,
            share_target,
            &self.auth,
            self.node_linearization,
            self.allocated_leaves,
        ) {
            let transport = self.transport.clone();
            let fetches = spawn_parallel_fetches(&self.servers, move |server| {
                transport.auth_read_subtree(&server, start_index, leaf_count)
            });
            for _ in 0..self.servers.len() {
                let Ok((_, resp)) = fetches.recv() else {
                    break;
                };
                let resp = match resp {
                    Ok(resp) => resp,
                    Err(_) => continue,
                };
                for node_reply in resp
                    .leaf_nodes
                    .into_iter()
                    .chain(resp.upper_path_nodes.into_iter())
                {
                    let key = node_reply.node;
                    if !node_set.contains(&key) {
                        continue;
                    }
                    let bucket = shares.entry(key).or_insert_with(Vec::new);
                    let seen_bucket = seen.entry(key).or_insert_with(HashSet::new);
                    for share in node_reply.shares {
                        if share.value.is_some() && seen_bucket.insert(share.registry_index) {
                            bucket.push(share);
                        }
                    }
                }
                if all_nodes_satisfied(
                    &nodes,
                    &shares,
                    share_target,
                    &self.auth,
                    self.node_linearization,
                    self.allocated_leaves,
                ) {
                    fetches.cancel();
                    break;
                }
            }
            if !all_nodes_satisfied(
                &nodes,
                &shares,
                min_shares,
                &self.auth,
                self.node_linearization,
                self.allocated_leaves,
            ) {
                return Err(ClientError::NotEnoughShards);
            }
        }
        let (values, proof) = match max_errors {
            Some(max_errors) => reconstruct_subtree_from_shares_correcting(
                &self.auth,
                &self.params,
                self.node_linearization,
                self.allocated_leaves,
                &plan,
                &shares,
                max_errors,
            ),
            None => reconstruct_subtree_from_shares(
                &self.auth,
                &self.params,
                self.node_linearization,
                self.allocated_leaves,
                &plan,
                &shares,
            ),
        }
        .map_err(map_read_error)?;
        let root = proof.root;
        if let Some(trusted) = self.trusted_root {
            if trusted != root {
                return Err(ClientError::RootMismatch);
            }
        }
        self.trusted_root = Some(root);
        Ok((values, proof))
    }

    fn read_batch_internal(
        &mut self,
        indices: &[u64],
    ) -> Result<Vec<(Value256, AuthPath)>, ClientError> {
        self.read_batch_internal_target(indices, self.params.need, None)
    }

    fn read_batch_internal_target(
        &mut self,
        indices: &[u64],
        share_target: usize,
        max_errors: Option<usize>,
    ) -> Result<Vec<(Value256, AuthPath)>, ClientError> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }
        let min_shares = max_errors
            .map(|max_errors| self.params.need + 2 * max_errors)
            .unwrap_or(share_target);
        let max = self.auth.conceptual_l_max_u128();
        if indices.iter().any(|idx| *idx as u128 >= max) {
            return Err(ClientError::InvalidIndex);
        }
        let plan = build_partial_tree_nodes(&self.auth, indices);
        let active_nodes = active_nodes_for_read(
            &self.auth,
            self.node_linearization,
            self.allocated_leaves,
            &plan.nodes,
        );

        let mut shares: HashMap<NodeKey, Vec<ShardShare>> = HashMap::new();
        if !active_nodes.is_empty() {
            let transport = self.transport.clone();
            let indices_owned = indices.to_vec();
            let (shares_by_pos, remaining) = collect_shares_by_pos_parallel(
                &self.servers,
                active_nodes.len(),
                share_target,
                move |server| transport.unauth_multi_read(&server, &indices_owned),
                |_, _| {
                    ClientError::InvalidResponse("unauth read batch length mismatch".to_string())
                },
            )?;
            if remaining != 0 && max_errors.is_none() {
                return Err(ClientError::NotEnoughShards);
            }
            if remaining != 0
                && shares_by_pos
                    .iter()
                    .any(|node_shares| node_shares.len() < min_shares)
            {
                return Err(ClientError::NotEnoughShards);
            }
            for (node, node_shares) in active_nodes.into_iter().zip(shares_by_pos.into_iter()) {
                shares.insert(node, node_shares);
            }
        }

        let mut root_seen = None;
        let mut results = Vec::with_capacity(indices.len());
        for (path_nodes, &index) in plan.paths.iter().zip(indices.iter()) {
            let (value, path) = match max_errors {
                Some(max_errors) => reconstruct_path_from_shares_correcting(
                    &self.auth,
                    &self.params,
                    self.node_linearization,
                    self.allocated_leaves,
                    index,
                    path_nodes,
                    &shares,
                    max_errors,
                ),
                None => reconstruct_path_from_shares(
                    &self.auth,
                    &self.params,
                    self.node_linearization,
                    self.allocated_leaves,
                    index,
                    path_nodes,
                    &shares,
                ),
            }
            .map_err(map_read_error)?;
            let root = path.root;
            if let Some(trusted) = self.trusted_root {
                if trusted != root {
                    return Err(ClientError::RootMismatch);
                }
            }
            if let Some(existing) = root_seen {
                if existing != root {
                    return Err(ClientError::RootMismatch);
                }
            } else {
                root_seen = Some(root);
            }
            results.push((value, path));
        }
        if let Some(root) = root_seen {
            self.trusted_root = Some(root);
        }
        Ok(results)
    }

    /// Read a leaf value and return its authentication path.
    pub fn read_with_path(&mut self, index: u64) -> Result<(Value256, AuthPath), ClientError> {
        self.refresh_allocated_leaves()?;
        self.read_with_path_internal(index)
    }

    /// Process pre-collected authenticated read responses for one leaf.
    ///
    /// This is intended for simulated-client benchmarks where helper replies
    /// have already arrived. It performs client-side share aggregation,
    /// Reed-Solomon decoding, path reconstruction, and trusted-root checking,
    /// but does not fetch from the transport.
    pub fn read_with_path_from_responses(
        &mut self,
        index: u64,
        responses: &[AuthReadResponse],
    ) -> Result<(Value256, AuthPath), ClientError> {
        self.read_with_path_from_responses_internal(index, responses, None)
    }

    /// Read a leaf value and path while correcting up to `max_errors` bad
    /// helper replies for every active Merkle node in the path.
    pub fn read_with_path_correcting(
        &mut self,
        index: u64,
        max_errors: usize,
    ) -> Result<(Value256, AuthPath), ClientError> {
        self.refresh_allocated_leaves()?;
        let share_target = error_correcting_share_target(self.params.need, max_errors);
        self.read_with_path_internal_target(index, share_target, Some(max_errors))
    }

    /// Process pre-collected authenticated read responses while correcting up
    /// to `max_errors` bad helper replies for every active Merkle node.
    pub fn read_with_path_from_responses_correcting(
        &mut self,
        index: u64,
        responses: &[AuthReadResponse],
        max_errors: usize,
    ) -> Result<(Value256, AuthPath), ClientError> {
        self.read_with_path_from_responses_internal(index, responses, Some(max_errors))
    }

    /// Benchmark-only variant of pre-collected correcting reads that does not
    /// carry detected bad-helper indices across Merkle-path nodes.
    #[cfg(feature = "bench")]
    pub fn read_with_path_from_responses_correcting_no_filter(
        &mut self,
        index: u64,
        responses: &[AuthReadResponse],
        max_errors: usize,
    ) -> Result<(Value256, AuthPath), ClientError> {
        if index as u128 >= self.auth.conceptual_l_max_u128() {
            return Err(ClientError::InvalidIndex);
        }
        let min_shares = self.params.need + 2 * max_errors;
        let nodes = build_path_nodes(&self.auth, index);
        let node_set: HashSet<NodeKey> = nodes.iter().copied().collect();

        let mut shares: HashMap<NodeKey, Vec<ShardShare>> = HashMap::new();
        let mut seen: HashMap<NodeKey, HashSet<usize>> = HashMap::new();
        for resp in responses {
            for node_reply in &resp.nodes {
                let key = node_reply.node;
                if !node_set.contains(&key) {
                    continue;
                }
                let bucket = shares.entry(key).or_insert_with(Vec::new);
                let seen_bucket = seen.entry(key).or_insert_with(HashSet::new);
                for share in &node_reply.shares {
                    if share.value.is_some() && seen_bucket.insert(share.registry_index) {
                        bucket.push(share.clone());
                    }
                }
            }
        }
        if !all_nodes_satisfied(
            &nodes,
            &shares,
            min_shares,
            &self.auth,
            self.node_linearization,
            self.allocated_leaves,
        ) {
            return Err(ClientError::NotEnoughShards);
        }

        let (value, path) = reconstruct_path_from_shares_correcting_no_filter(
            &self.auth,
            &self.params,
            self.node_linearization,
            self.allocated_leaves,
            index,
            &nodes,
            &shares,
            max_errors,
        )
        .map_err(map_read_error)?;
        let root = path.root;
        if let Some(trusted) = self.trusted_root {
            if trusted != root {
                return Err(ClientError::RootMismatch);
            }
        }
        self.trusted_root = Some(root);
        Ok((value, path))
    }

    /// Read a leaf value and verify it against the trusted root.
    pub fn read(&mut self, index: u64) -> Result<Value256, ClientError> {
        let (value, _) = self.read_with_path(index)?;
        Ok(value)
    }

    /// Read a leaf value while correcting up to `max_errors` bad helper replies
    /// for every active Merkle node in the path.
    pub fn read_correcting(
        &mut self,
        index: u64,
        max_errors: usize,
    ) -> Result<Value256, ClientError> {
        let (value, _) = self.read_with_path_correcting(index, max_errors)?;
        Ok(value)
    }

    /// Read multiple leaf values and verify them against the trusted root.
    pub fn read_batch(&mut self, indices: &[u64]) -> Result<Vec<Value256>, ClientError> {
        self.refresh_allocated_leaves()?;
        let values = self.read_batch_internal(indices)?;
        Ok(values.into_iter().map(|(value, _)| value).collect())
    }

    /// Read multiple leaf values while correcting up to `max_errors` bad helper
    /// replies for every active Merkle node in the batched partial tree.
    pub fn read_batch_correcting(
        &mut self,
        indices: &[u64],
        max_errors: usize,
    ) -> Result<Vec<Value256>, ClientError> {
        self.refresh_allocated_leaves()?;
        let share_target = error_correcting_share_target(self.params.need, max_errors);
        let values = self.read_batch_internal_target(indices, share_target, Some(max_errors))?;
        Ok(values.into_iter().map(|(value, _)| value).collect())
    }

    /// Read one aligned subtree and return its leaves plus the upper proof.
    pub fn read_subtree_with_proof(
        &mut self,
        start_index: u64,
        leaf_count: usize,
    ) -> Result<(Vec<Value256>, AuthSubtreeProof), ClientError> {
        self.refresh_allocated_leaves()?;
        self.read_subtree_internal(start_index, leaf_count)
    }

    /// Read one aligned subtree while correcting up to `max_errors` bad helper
    /// replies for every active node in the subtree proof.
    pub fn read_subtree_with_proof_correcting(
        &mut self,
        start_index: u64,
        leaf_count: usize,
        max_errors: usize,
    ) -> Result<(Vec<Value256>, AuthSubtreeProof), ClientError> {
        self.refresh_allocated_leaves()?;
        let share_target = error_correcting_share_target(self.params.need, max_errors);
        self.read_subtree_internal_target(start_index, leaf_count, share_target, Some(max_errors))
    }

    /// Read one aligned subtree and verify it against the trusted root.
    pub fn read_subtree(
        &mut self,
        start_index: u64,
        leaf_count: usize,
    ) -> Result<Vec<Value256>, ClientError> {
        let (values, _) = self.read_subtree_with_proof(start_index, leaf_count)?;
        Ok(values)
    }

    /// Read one aligned subtree while correcting up to `max_errors` bad helper
    /// replies for every active node in the subtree proof.
    pub fn read_subtree_correcting(
        &mut self,
        start_index: u64,
        leaf_count: usize,
        max_errors: usize,
    ) -> Result<Vec<Value256>, ClientError> {
        let (values, _) =
            self.read_subtree_with_proof_correcting(start_index, leaf_count, max_errors)?;
        Ok(values)
    }

    /// Write a new value to a leaf.
    pub fn write(&mut self, index: u64, new: Value256) -> Result<(), ClientError> {
        self.refresh_allocated_leaves()?;
        if index as usize >= self.allocated_leaves {
            return Err(ClientError::InvalidIndex);
        }
        let (old, path) = self.read_with_path_internal(index)?;
        let _ = self.apply_write_with_path(index, old, new, &path)?;
        Ok(())
    }

    /// Write a new value and return the old value, auth path, and new root.
    pub fn write_with_path(
        &mut self,
        index: u64,
        new: Value256,
    ) -> Result<(Value256, AuthPath, Value256), ClientError> {
        self.refresh_allocated_leaves()?;
        if index as usize >= self.allocated_leaves {
            return Err(ClientError::InvalidIndex);
        }
        let (old, path) = self.read_with_path_internal(index)?;
        let root_new = self.apply_write_with_path(index, old, new, &path)?;
        Ok((old, path, root_new))
    }

    fn apply_write_with_path(
        &mut self,
        index: u64,
        old: Value256,
        new: Value256,
        path: &AuthPath,
    ) -> Result<Value256, ClientError> {
        let idx = usize::try_from(index).map_err(|_| ClientError::InvalidIndex)?;
        if path.index != idx {
            return Err(ClientError::InvalidResponse(
                "path index mismatch in write".to_string(),
            ));
        }
        let req = AuthWriteRequest {
            old,
            new,
            path: AuthPathRequest {
                index: path.index as u64,
                siblings: path.siblings.clone(),
                root: path.root,
            },
        };
        let mut root_new = None;
        let transport = self.transport.clone();
        let transport_for_label = transport.clone();
        let req_for_call = req.clone();
        let failed = broadcast_with_failures_parallel(
            &self.servers,
            |server| transport_for_label.server_label(server),
            move |server| transport.auth_write(&server, &req_for_call),
            |resp| {
                if resp.ok {
                    let Some(resp_root) = resp.root_new else {
                        return Err("missing root_new in response".to_string());
                    };
                    if let Some(existing) = root_new {
                        if existing != resp_root {
                            return Err("root_new mismatch in response".to_string());
                        }
                    } else {
                        root_new = Some(resp_root);
                    }
                    Ok(())
                } else {
                    Err(resp
                        .error
                        .unwrap_or_else(|| "auth write failed".to_string()))
                }
            },
        );
        if !failed.is_empty() {
            return Err(ClientError::Net(format!("auth write failed: {:?}", failed)));
        }
        let root_new = root_new.ok_or_else(|| {
            ClientError::InvalidResponse("missing root_new in responses".to_string())
        })?;
        self.trusted_root = Some(root_new);
        Ok(root_new)
    }

    /// Write a batch of updates using a shared multiproof.
    pub fn write_batch(&mut self, updates: &[(u64, Value256)]) -> Result<(), ClientError> {
        let req = self.plan_write_batch_request(updates)?;
        self.submit_batch_write_request(&req)
    }

    /// Plan an authenticated batch-write request from leaf updates.
    ///
    /// The returned request is caller-visible and can be inspected, stored, or
    /// submitted later with [`Self::submit_batch_write_request`].
    pub fn plan_write_batch_request(
        &mut self,
        updates: &[(u64, Value256)],
    ) -> Result<AuthBatchWriteRequest, ClientError> {
        self.refresh_allocated_leaves()?;
        self.build_leaf_batch_write_request(updates)
    }

    /// Submit a previously planned authenticated batch-write request.
    pub fn submit_batch_write_request(
        &mut self,
        req: &AuthBatchWriteRequest,
    ) -> Result<(), ClientError> {
        if req.updates.is_empty() {
            return Ok(());
        }
        self.send_batch_write_request(req.clone())
    }

    /// Replace one aligned fresh subtree using a caller-supplied upper-path proof.
    ///
    /// The subtree must lie inside the currently allocated leaf range, and the
    /// covered old leaves are assumed to hold the default value. Callers that
    /// extend the active range should do so first via [`Self::extend`].
    pub fn write_fresh_subtree_with_proof(
        &mut self,
        proof: AuthSubtreeProof,
        leaves: &[Value256],
    ) -> Result<(), ClientError> {
        if leaves.is_empty() || !leaves.len().is_power_of_two() {
            return Err(ClientError::InvalidIndex);
        }
        if proof.leaf_count != leaves.len() {
            return Err(ClientError::InvalidProof);
        }
        if proof.start_index % (leaves.len() as u64) != 0 {
            return Err(ClientError::InvalidIndex);
        }
        let end = proof
            .start_index
            .checked_add(usize_to_u64(leaves.len())?)
            .ok_or(ClientError::InvalidIndex)?;
        if end as u128 > self.auth.conceptual_l_max_u128() || end > self.allocated_leaves as u64 {
            return Err(ClientError::InvalidIndex);
        }
        let req = AuthFreshSubtreeWriteRequest {
            leaves: leaves.to_vec(),
            proof,
            delta_mode: DeltaMode::Aggregate,
        };
        self.send_fresh_subtree_write_request(req)
    }

    /// Replace every leaf in one aligned subtree using a subtree proof.
    pub fn write_subtree(
        &mut self,
        start_index: u64,
        leaves: &[Value256],
    ) -> Result<(), ClientError> {
        self.refresh_allocated_leaves()?;
        self.write_subtree_unchecked(start_index, leaves)
    }

    /// Append one aligned subtree at the current frontier.
    ///
    /// This is still a convenience helper over `AuthExtend` followed by a
    /// subtree-shaped batch write. It is not atomic.
    pub fn append_subtree(&mut self, leaves: &[Value256]) -> Result<u64, ClientError> {
        self.refresh_allocated_leaves()?;
        if leaves.is_empty() {
            return Ok(self.allocated_leaves as u64);
        }
        if !leaves.len().is_power_of_two() {
            return Err(ClientError::InvalidIndex);
        }
        let leaves_len_u64 = usize_to_u64(leaves.len())?;
        let start = self.allocated_leaves as u64;
        if start % leaves_len_u64 != 0 {
            return Err(ClientError::InvalidIndex);
        }
        let new_len = start
            .checked_add(leaves_len_u64)
            .ok_or(ClientError::InvalidIndex)?;
        self.extend_allocated_leaves(new_len)?;
        if self.transport.supports_subtree_upper_path_reads()
            && self.transport.supports_fresh_subtree_writes()
        {
            let proof = self.read_default_subtree_proof_internal(start, leaves.len())?;
            self.write_fresh_subtree_with_proof(proof, leaves)?;
        } else {
            self.write_subtree_unchecked(start, leaves)?;
        }
        Ok(start)
    }

    /// Extend the allocated leaf range.
    pub fn extend(&mut self, new_len: u64) -> Result<(), ClientError> {
        self.refresh_allocated_leaves()?;
        self.extend_allocated_leaves(new_len)
    }
}

fn map_read_error(err: ReadError) -> ClientError {
    match err {
        ReadError::InvalidIndex => ClientError::InvalidIndex,
        ReadError::NotEnoughShards => ClientError::NotEnoughShards,
        ReadError::InvalidProof => ClientError::InvalidProof,
    }
}

fn usize_to_u64(value: usize) -> Result<u64, ClientError> {
    u64::try_from(value).map_err(|_| ClientError::InvalidIndex)
}

fn choose_suspect_child(
    left: NodeKey,
    left_label: Value256,
    right: NodeKey,
    right_label: Value256,
    cache: &RangeDecodeCache,
    verify_plan: &RangeVerifyPlan,
    bad_filter: &UnauthReadBadFilter,
) -> NodeKey {
    let left_corrected = cache.node_is_corrected(verify_plan, left);
    let right_corrected = cache.node_is_corrected(verify_plan, right);
    if left_corrected && !right_corrected {
        return right;
    }
    if right_corrected && !left_corrected {
        return left;
    }

    let left_ok = subtree_matches_label(left, left_label, cache, verify_plan, bad_filter);
    let right_ok = subtree_matches_label(right, right_label, cache, verify_plan, bad_filter);
    match (left_ok, right_ok) {
        (true, false) => right,
        (false, true) => left,
        _ => {
            let left_range = subtree_has_requested_leaf(left, verify_plan);
            let right_range = subtree_has_requested_leaf(right, verify_plan);
            match (left_range, right_range) {
                (true, false) => left,
                (false, true) => right,
                _ => left,
            }
        }
    }
}

fn subtree_matches_label(
    node: NodeKey,
    label: Value256,
    cache: &RangeDecodeCache,
    verify_plan: &RangeVerifyPlan,
    bad_filter: &UnauthReadBadFilter,
) -> bool {
    if node.level >= verify_plan.height {
        return true;
    }
    let left = child_left(node);
    let right = child_right(node);
    let (Some(left_label), Some(right_label)) = (
        cache.label_for_node(verify_plan, left, bad_filter),
        cache.label_for_node(verify_plan, right, bad_filter),
    ) else {
        return false;
    };
    node_hash(left_label, right_label) == label
        && subtree_matches_label(left, left_label, cache, verify_plan, bad_filter)
        && subtree_matches_label(right, right_label, cache, verify_plan, bad_filter)
}

fn subtree_has_requested_leaf(node: NodeKey, verify_plan: &RangeVerifyPlan) -> bool {
    let subtree_height = verify_plan.height.saturating_sub(node.level);
    let start = node.pos << subtree_height;
    let end = (node.pos + 1) << subtree_height;
    start < verify_plan.range_end && verify_plan.start_index < end
}

fn active_descendant_array_index(
    node: NodeKey,
    cache: &RangeDecodeCache,
    verify_plan: &RangeVerifyPlan,
    bad_filter: &UnauthReadBadFilter,
) -> Option<u64> {
    if let Some(array_index) = verify_plan.array_index_for_node(node) {
        return Some(array_index);
    }
    if node.level >= verify_plan.height {
        return None;
    }
    let left = child_left(node);
    let right = child_right(node);
    if cache
        .label_for_node(verify_plan, left, bad_filter)
        .is_some()
    {
        if let Some(array_index) =
            active_descendant_array_index(left, cache, verify_plan, bad_filter)
        {
            return Some(array_index);
        }
    }
    if cache
        .label_for_node(verify_plan, right, bad_filter)
        .is_some()
    {
        return active_descendant_array_index(right, cache, verify_plan, bad_filter);
    }
    None
}

fn insert_proof_node(
    proof_nodes: &mut HashMap<NodeKey, Value256>,
    node: NodeKey,
    value: Value256,
) -> Result<(), ClientError> {
    if let Some(existing) = proof_nodes.get(&node) {
        if *existing != value {
            return Err(ClientError::InvalidProof);
        }
    } else {
        proof_nodes.insert(node, value);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(level: u32, pos: u64) -> NodeKey {
        NodeKey { level, pos }
    }

    #[test]
    fn suspect_child_prefers_unchecked_boundary_over_verified_range_side() {
        let auth = AuthParams::new(2);
        let left = node(1, 0);
        let right = node(1, 1);
        let right_left = node(2, 2);
        let right_right = node(2, 3);
        let right_left_label = [2u8; 32];
        let right_right_label = [3u8; 32];
        let right_label = node_hash(right_left_label, right_right_label);
        let merkle_plan = crate::auth::read_path::MerkleRangePlan {
            node_linearization: mapping::NodeLinearization::Preorder,
            allocated_leaves: 0,
            start_index: 2,
            leaf_count: 2,
            requested_leaf_nodes: vec![right_left, right_right],
            boundary_proof_nodes: vec![left],
            internal_checkpoint_nodes: vec![right],
            node_to_array_index: HashMap::new(),
            node_to_requested_pos: HashMap::from([(right_left, 0), (right_right, 1)]),
            requested_array_indices: vec![0, 1],
        };
        let verify_plan = RangeVerifyPlan::new(&auth, &merkle_plan);
        let mut cache = RangeDecodeCache::with_capacity(2, 0);
        cache.labels_by_pos[0] = Some(right_left_label);
        cache.labels_by_pos[1] = Some(right_right_label);
        let bad_filter = UnauthReadBadFilter::default();

        let suspect = choose_suspect_child(
            left,
            [9u8; 32],
            right,
            right_label,
            &cache,
            &verify_plan,
            &bad_filter,
        );
        assert_eq!(suspect, left);
    }

    #[test]
    fn range_cache_keeps_certified_labels_when_bad_helper_is_discovered() {
        let auth = AuthParams::new(2);
        let certified_node = node(2, 0);
        let tentative_node = node(2, 1);
        let certified_index = 10;
        let tentative_index = 11;
        let merkle_plan = crate::auth::read_path::MerkleRangePlan {
            node_linearization: mapping::NodeLinearization::Preorder,
            allocated_leaves: 2,
            start_index: 0,
            leaf_count: 2,
            requested_leaf_nodes: vec![certified_node, tentative_node],
            boundary_proof_nodes: Vec::new(),
            internal_checkpoint_nodes: Vec::new(),
            node_to_array_index: HashMap::from([
                (certified_node, certified_index),
                (tentative_node, tentative_index),
            ]),
            node_to_requested_pos: HashMap::from([(certified_node, 0), (tentative_node, 1)]),
            requested_array_indices: vec![certified_index, tentative_index],
        };
        let verify_plan = RangeVerifyPlan::new(&auth, &merkle_plan);

        let mut cache = RangeDecodeCache::with_capacity(2, 4);
        cache.insert_candidate_values(
            &verify_plan,
            &auth,
            3,
            UnauthReadBadFilter {
                helper_indices: HashSet::from([0]),
                registry_indices: HashSet::from([0]),
            },
            vec![(certified_index, [1u8; 32]), (tentative_index, [2u8; 32])],
        );
        cache.mark_node_certified(&verify_plan, certified_node);

        let bad_filter = UnauthReadBadFilter {
            helper_indices: HashSet::from([0]),
            registry_indices: HashSet::new(),
        };

        assert_eq!(cache.values_by_pos[0], Some([1u8; 32]));
        assert!(
            cache
                .label_for_node(&verify_plan, certified_node, &bad_filter)
                .is_some()
        );
        assert_eq!(cache.values_by_pos[1], Some([2u8; 32]));
        assert!(
            cache
                .label_for_node(&verify_plan, tentative_node, &bad_filter)
                .is_none()
        );
        assert!(cache.ordered_values(&bad_filter).is_none());
    }
}
