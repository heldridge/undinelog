//! Authenticated SMT read-path helpers.
//!
//! This module provides shared utilities for translating nodes into coded-array
//! positions, decoding shard replies, and filling defaults. It does not apply
//! updates to parties.

use super::mapping;
use super::merkle::{
    NodeKey, default_hash, label_from_content, leaf_key, leaf_label, node_hash, parent,
    recompute_root, sibling,
};
use super::{AuthParams, AuthPath, AuthRangeProof, AuthSubtreeProof, ReadError};
use crate::field::{Fp, fp_zero};
use crate::shares::ShardShare;
use crate::unauth::{
    CodedReadError, DecodeCache, UnauthArrayParams, UnauthDecodeParams,
    decode_from_shares_at_z_with_cache,
};
use crate::value::{Value256, encode_value};
use std::collections::{BTreeSet, HashMap, HashSet};

/// Map coded-array read errors into auth read errors for the read path.
fn map_coded_read_error(err: CodedReadError) -> ReadError {
    // Keep error mapping narrow so unexpected errors are surfaced at compile time.
    match err {
        CodedReadError::NotEnoughShards => ReadError::NotEnoughShards,
    }
}

/// Build the leaf+siblings path nodes for a leaf index.
pub(crate) fn build_path_nodes(auth: &AuthParams, index: u64) -> Vec<NodeKey> {
    let height = auth.tree_height as u32;
    let mut key = leaf_key(height, index);
    let mut nodes = Vec::with_capacity(auth.tree_height + 1);
    nodes.push(key);
    for _ in 0..auth.tree_height {
        let sib = sibling(key);
        nodes.push(sib);
        key = parent(key);
    }
    nodes
}

/// Combined read paths for multiple leaves, deduplicated into a partial tree.
pub(crate) struct PartialTreeNodes {
    pub paths: Vec<Vec<NodeKey>>,
    pub nodes: Vec<NodeKey>,
}

/// One aligned-subtree read plan.
pub(crate) struct SubtreeReadPlan {
    pub start_index: u64,
    pub leaf_count: usize,
    pub subtree_root: NodeKey,
    pub leaf_nodes: Vec<NodeKey>,
    pub upper_path_nodes: Vec<NodeKey>,
}

/// Auth-side plan for opening one contiguous leaf range.
///
/// The plan names the Merkle leaves requested by the caller, the canonical
/// boundary proof nodes, the internal range-forest checkpoints, and the active
/// maintained array cells that must be fetched from the unauth layer.
pub(crate) struct MerkleRangePlan {
    pub node_linearization: mapping::NodeLinearization,
    pub allocated_leaves: usize,
    pub start_index: u64,
    pub leaf_count: usize,
    pub requested_leaf_nodes: Vec<NodeKey>,
    pub boundary_proof_nodes: Vec<NodeKey>,
    pub internal_checkpoint_nodes: Vec<NodeKey>,
    pub node_to_array_index: HashMap<NodeKey, u64>,
    pub node_to_requested_pos: HashMap<NodeKey, usize>,
    pub requested_array_indices: Vec<u64>,
}

impl MerkleRangePlan {
    pub(crate) fn new_with_index_maps(
        auth: &AuthParams,
        node_linearization: mapping::NodeLinearization,
        allocated_leaves: usize,
        start_index: u64,
        leaf_count: usize,
        build_index_maps: bool,
    ) -> Result<Self, ReadError> {
        let (requested_leaf_nodes, boundary_proof_nodes) =
            canonical_range_nodes(auth, start_index, leaf_count)?;
        let internal_checkpoint_nodes =
            canonical_range_internal_nodes(auth, start_index, leaf_count)?;

        let active_capacity = requested_leaf_nodes.len()
            + boundary_proof_nodes.len()
            + internal_checkpoint_nodes.len();
        let mut active_pairs = Vec::with_capacity(active_capacity);
        for node in requested_leaf_nodes
            .iter()
            .chain(boundary_proof_nodes.iter())
            .chain(internal_checkpoint_nodes.iter())
            .copied()
        {
            let Some(array_index) =
                node_to_index_if_active(auth, node_linearization, allocated_leaves, node)
            else {
                continue;
            };
            active_pairs.push((array_index, node));
        }
        active_pairs.sort_by(|(left_index, left_node), (right_index, right_node)| {
            left_index
                .cmp(right_index)
                .then_with(|| left_node.cmp(right_node))
        });

        let requested_array_indices = active_pairs
            .iter()
            .map(|(array_index, _)| *array_index)
            .collect::<Vec<_>>();
        let (node_to_array_index, node_to_requested_pos) = if build_index_maps {
            let mut node_to_array_index = HashMap::with_capacity(active_pairs.len());
            let mut node_to_requested_pos = HashMap::with_capacity(active_pairs.len());
            for (pos, &(array_index, node)) in active_pairs.iter().enumerate() {
                node_to_array_index.insert(node, array_index);
                node_to_requested_pos.insert(node, pos);
            }
            (node_to_array_index, node_to_requested_pos)
        } else {
            (HashMap::new(), HashMap::new())
        };

        Ok(MerkleRangePlan {
            node_linearization,
            allocated_leaves,
            start_index,
            leaf_count,
            requested_leaf_nodes,
            boundary_proof_nodes,
            internal_checkpoint_nodes,
            node_to_array_index,
            node_to_requested_pos,
            requested_array_indices,
        })
    }

    pub(crate) fn requested_range_end(&self) -> u64 {
        self.start_index + self.leaf_count as u64
    }

    /// Fill all non-returned range proof nodes from an ordered unauth decode.
    pub(crate) fn proof_nodes_from_ordered_array_values(
        &self,
        auth: &AuthParams,
        requested_indices: &[u64],
        values: &[Value256],
    ) -> Result<Vec<(NodeKey, Value256)>, ReadError> {
        self.boundary_proof_nodes
            .iter()
            .chain(self.internal_checkpoint_nodes.iter())
            .copied()
            .map(|node| {
                let value = self.node_value_from_ordered_array_values(
                    auth,
                    requested_indices,
                    values,
                    node,
                )?;
                Ok((node, value))
            })
            .collect()
    }

    /// Fill requested leaf values from an ordered unauth decode.
    pub(crate) fn leaf_values_from_ordered_array_values(
        &self,
        auth: &AuthParams,
        requested_indices: &[u64],
        values: &[Value256],
    ) -> Result<Vec<Value256>, ReadError> {
        self.requested_leaf_nodes
            .iter()
            .copied()
            .map(|node| {
                self.node_value_from_ordered_array_values(auth, requested_indices, values, node)
            })
            .collect()
    }

    fn node_value_from_ordered_array_values(
        &self,
        auth: &AuthParams,
        requested_indices: &[u64],
        values: &[Value256],
        node: NodeKey,
    ) -> Result<Value256, ReadError> {
        if let Some(&pos) = self.node_to_requested_pos.get(&node) {
            return values.get(pos).copied().ok_or(ReadError::InvalidProof);
        }
        if let Some(array_index) = self.active_array_index(auth, node) {
            let pos = requested_indices
                .binary_search(&array_index)
                .map_err(|_| ReadError::InvalidProof)?;
            return values.get(pos).copied().ok_or(ReadError::InvalidProof);
        }
        Ok(default_value_for_node(auth, node))
    }

    fn active_array_index(&self, auth: &AuthParams, node: NodeKey) -> Option<u64> {
        self.node_to_array_index.get(&node).copied().or_else(|| {
            node_to_index_if_active(auth, self.node_linearization, self.allocated_leaves, node)
        })
    }
}

pub(crate) fn build_merkle_range_plan_with_index_maps(
    auth: &AuthParams,
    node_linearization: mapping::NodeLinearization,
    allocated_leaves: usize,
    start_index: u64,
    leaf_count: usize,
    build_index_maps: bool,
) -> Result<MerkleRangePlan, ReadError> {
    MerkleRangePlan::new_with_index_maps(
        auth,
        node_linearization,
        allocated_leaves,
        start_index,
        leaf_count,
        build_index_maps,
    )
}

/// Build a partial tree for multiple leaves by combining path nodes.
pub(crate) fn build_partial_tree_nodes(auth: &AuthParams, indices: &[u64]) -> PartialTreeNodes {
    let mut nodes_set: HashSet<NodeKey> = HashSet::new();
    let mut paths = Vec::with_capacity(indices.len());
    for &index in indices {
        let path = build_path_nodes(auth, index);
        for node in &path {
            nodes_set.insert(*node);
        }
        paths.push(path);
    }
    let mut nodes: Vec<NodeKey> = nodes_set.into_iter().collect();
    nodes.sort();
    PartialTreeNodes { paths, nodes }
}

/// Build the node plan for one aligned subtree.
pub(crate) fn build_subtree_read_plan(
    auth: &AuthParams,
    start_index: u64,
    leaf_count: usize,
) -> Result<SubtreeReadPlan, ReadError> {
    if leaf_count == 0 || !leaf_count.is_power_of_two() {
        return Err(ReadError::InvalidIndex);
    }
    let leaf_count_u64 = u64::try_from(leaf_count).map_err(|_| ReadError::InvalidIndex)?;
    let end = (start_index as u128)
        .checked_add(leaf_count as u128)
        .ok_or(ReadError::InvalidIndex)?;
    if end > auth.conceptual_l_max_u128() {
        return Err(ReadError::InvalidIndex);
    }
    if start_index % leaf_count_u64 != 0 {
        return Err(ReadError::InvalidIndex);
    }
    let subtree_height = leaf_count.trailing_zeros() as usize;
    if subtree_height > auth.tree_height {
        return Err(ReadError::InvalidIndex);
    }
    let root_level = auth.tree_height - subtree_height;
    let subtree_root = NodeKey {
        level: root_level as u32,
        pos: start_index >> subtree_height,
    };
    let leaf_nodes = (0..leaf_count_u64)
        .map(|offset| leaf_key(auth.tree_height as u32, start_index + offset))
        .collect();
    let mut upper_path_nodes = Vec::with_capacity(root_level);
    let mut key = subtree_root;
    for _ in 0..root_level {
        upper_path_nodes.push(sibling(key));
        key = parent(key);
    }
    Ok(SubtreeReadPlan {
        start_index,
        leaf_count,
        subtree_root,
        leaf_nodes,
        upper_path_nodes,
    })
}

fn canonical_range_nodes(
    auth: &AuthParams,
    start_index: u64,
    leaf_count: usize,
) -> Result<(Vec<NodeKey>, Vec<NodeKey>), ReadError> {
    if leaf_count == 0 {
        return Err(ReadError::InvalidIndex);
    }
    let leaf_count_u64 = u64::try_from(leaf_count).map_err(|_| ReadError::InvalidIndex)?;
    let end = (start_index as u128)
        .checked_add(leaf_count as u128)
        .ok_or(ReadError::InvalidIndex)?;
    if end > auth.conceptual_l_max_u128() {
        return Err(ReadError::InvalidIndex);
    }

    if leaf_count.is_power_of_two() && start_index % leaf_count_u64 == 0 {
        let plan = build_subtree_read_plan(auth, start_index, leaf_count)?;
        return Ok((plan.leaf_nodes, plan.upper_path_nodes));
    }

    let height = auth.tree_height as u32;
    let requested_leaf_nodes = (0..leaf_count_u64)
        .map(|offset| leaf_key(height, start_index + offset))
        .collect::<Vec<_>>();
    let boundary_proof_nodes = canonical_boundary_proof_nodes(&requested_leaf_nodes);
    Ok((requested_leaf_nodes, boundary_proof_nodes))
}

fn canonical_range_boundary_proof_nodes(
    auth: &AuthParams,
    start_index: u64,
    leaf_count: usize,
) -> Result<Vec<NodeKey>, ReadError> {
    if leaf_count == 0 {
        return Err(ReadError::InvalidIndex);
    }
    let leaf_count_u64 = u64::try_from(leaf_count).map_err(|_| ReadError::InvalidIndex)?;
    let end = (start_index as u128)
        .checked_add(leaf_count as u128)
        .ok_or(ReadError::InvalidIndex)?;
    if end > auth.conceptual_l_max_u128() {
        return Err(ReadError::InvalidIndex);
    }

    if leaf_count.is_power_of_two() && start_index % leaf_count_u64 == 0 {
        let subtree_height = leaf_count.trailing_zeros() as usize;
        if subtree_height > auth.tree_height {
            return Err(ReadError::InvalidIndex);
        }
        let root_level = auth.tree_height - subtree_height;
        let mut key = NodeKey {
            level: root_level as u32,
            pos: start_index >> subtree_height,
        };
        let mut upper_path_nodes = Vec::with_capacity(root_level);
        for _ in 0..root_level {
            upper_path_nodes.push(sibling(key));
            key = parent(key);
        }
        return Ok(upper_path_nodes);
    }

    let height = auth.tree_height as u32;
    let requested_leaf_nodes = (0..leaf_count_u64)
        .map(|offset| leaf_key(height, start_index + offset))
        .collect::<Vec<_>>();
    Ok(canonical_boundary_proof_nodes(&requested_leaf_nodes))
}

fn canonical_range_internal_nodes(
    auth: &AuthParams,
    start_index: u64,
    leaf_count: usize,
) -> Result<Vec<NodeKey>, ReadError> {
    if leaf_count == 0 {
        return Err(ReadError::InvalidIndex);
    }
    let end = (start_index as u128)
        .checked_add(leaf_count as u128)
        .ok_or(ReadError::InvalidIndex)?;
    if end > auth.conceptual_l_max_u128() {
        return Err(ReadError::InvalidIndex);
    }

    let end_u64 = u64::try_from(end).map_err(|_| ReadError::InvalidIndex)?;
    let mut internal_nodes = Vec::new();
    let height = auth.tree_height as u32;
    for level in 1..height {
        let subtree_height = height - level;
        let first_pos = start_index >> subtree_height;
        let last_pos = (end_u64 - 1) >> subtree_height;
        for pos in first_pos..=last_pos {
            internal_nodes.push(NodeKey { level, pos });
        }
    }
    Ok(internal_nodes)
}

fn canonical_boundary_proof_nodes(leaf_nodes: &[NodeKey]) -> Vec<NodeKey> {
    let mut current = leaf_nodes.iter().copied().collect::<BTreeSet<_>>();
    let mut proof_nodes = Vec::new();
    let mut proof_seen = HashSet::new();

    while current
        .iter()
        .next()
        .map(|node| node.level > 0)
        .unwrap_or(false)
    {
        let current_set = current.iter().copied().collect::<HashSet<_>>();
        let mut next = BTreeSet::new();
        for &node in &current {
            let sibling = sibling(node);
            if !current_set.contains(&sibling) && proof_seen.insert(sibling) {
                proof_nodes.push(sibling);
            }
            next.insert(parent(node));
        }
        current = next;
    }

    proof_nodes
}

/// Filter to active nodes that require shard reads.
pub(crate) fn active_nodes_for_read(
    auth: &AuthParams,
    node_linearization: mapping::NodeLinearization,
    allocated_leaves: usize,
    nodes: &[NodeKey],
) -> Vec<NodeKey> {
    let mut active = Vec::new();
    for node in nodes {
        if node_to_index_if_active(auth, node_linearization, allocated_leaves, *node).is_some() {
            active.push(*node);
        }
    }
    active
}

/// Translate a node into its linearized index if active.
///
/// Returns `None` for nodes outside the allocated prefix.
pub(crate) fn node_to_index_if_active(
    auth: &AuthParams,
    node_linearization: mapping::NodeLinearization,
    allocated_leaves: usize,
    key: NodeKey,
) -> Option<u64> {
    // Only nodes inside the allocated prefix have backing storage.
    mapping::idx_if_active(
        node_linearization,
        auth.tree_height as u16,
        allocated_leaves as u64,
        key.level as u16,
        key.pos,
    )
}

/// Return the default value for an inactive node.
///
/// Leaves default to zero; internal nodes use the precomputed auth defaults.
pub(crate) fn default_value_for_node(auth: &AuthParams, key: NodeKey) -> Value256 {
    // Leaves default to zero values; internal nodes default to hash defaults.
    if key.level as usize == auth.tree_height {
        encode_value(fp_zero())
    } else {
        default_hash(&auth.defaults, key.level as usize)
    }
}

/// Decode a node label from shard replies.
///
/// The returned field element is the decoded label for logical index `idx`.
pub(crate) fn decode_node_from_shares<P: UnauthArrayParams + UnauthDecodeParams>(
    params: &P,
    idx: u64,
    shares: &[ShardShare],
) -> Result<Fp, ReadError> {
    params
        .decode_index_from_shares(idx, shares)
        .map_err(map_coded_read_error)
}

fn decode_node_from_shares_with_cache<P: UnauthArrayParams + UnauthDecodeParams>(
    params: &P,
    idx: u64,
    shares: &[ShardShare],
    cache: &DecodeCache,
) -> Result<Fp, ReadError> {
    decode_from_shares_at_z_with_cache(params, params.decode_index_eval(idx), shares, Some(cache))
        .map_err(map_coded_read_error)
}

fn decode_node_from_shares_with_policy<P: UnauthArrayParams + UnauthDecodeParams>(
    params: &P,
    idx: u64,
    shares: &[ShardShare],
    max_errors: Option<usize>,
    cache: Option<&DecodeCache>,
) -> Result<Fp, ReadError> {
    match max_errors {
        Some(max_errors) => params
            .decode_index_from_shares_with_error_correction(idx, shares, max_errors)
            .map(|decoded| decoded.value)
            .map_err(map_coded_read_error),
        None => match cache {
            Some(cache) => decode_node_from_shares_with_cache(params, idx, shares, cache),
            None => decode_node_from_shares(params, idx, shares),
        },
    }
}

fn decode_node_from_shares_with_global_bad_filter<P: UnauthArrayParams + UnauthDecodeParams>(
    params: &P,
    idx: u64,
    shares: &[ShardShare],
    max_errors: usize,
    bad_registry_indices: &mut HashSet<usize>,
) -> Result<Fp, ReadError> {
    let filtered = shares
        .iter()
        .filter(|share| !bad_registry_indices.contains(&share.registry_index()))
        .cloned()
        .collect::<Vec<_>>();
    let remaining_errors = max_errors.saturating_sub(bad_registry_indices.len());
    if remaining_errors == 0 {
        return decode_node_from_shares(params, idx, &filtered);
    }
    let decoded = params
        .decode_index_from_shares_with_error_correction(idx, &filtered, remaining_errors)
        .map_err(map_coded_read_error)?;
    for registry_index in decoded.bad_registry_indices {
        bad_registry_indices.insert(registry_index);
    }
    Ok(decoded.value)
}

/// Reconstruct an auth path (value + siblings + root) from shard shares.
pub(crate) fn reconstruct_path_from_shares<P: UnauthArrayParams + UnauthDecodeParams>(
    auth: &AuthParams,
    params: &P,
    node_linearization: mapping::NodeLinearization,
    allocated_leaves: usize,
    index: u64,
    nodes: &[NodeKey],
    shares: &HashMap<NodeKey, Vec<ShardShare>>,
) -> Result<(Value256, AuthPath), ReadError> {
    reconstruct_path_from_shares_with_policy(
        auth,
        params,
        node_linearization,
        allocated_leaves,
        index,
        nodes,
        shares,
        None,
    )
}

/// Reconstruct an auth path while correcting up to `max_errors` bad shard
/// replies per active node.
pub(crate) fn reconstruct_path_from_shares_correcting<P: UnauthArrayParams + UnauthDecodeParams>(
    auth: &AuthParams,
    params: &P,
    node_linearization: mapping::NodeLinearization,
    allocated_leaves: usize,
    index: u64,
    nodes: &[NodeKey],
    shares: &HashMap<NodeKey, Vec<ShardShare>>,
    max_errors: usize,
) -> Result<(Value256, AuthPath), ReadError> {
    reconstruct_path_from_shares_with_policy(
        auth,
        params,
        node_linearization,
        allocated_leaves,
        index,
        nodes,
        shares,
        Some(max_errors),
    )
}

/// Reconstruct an auth path while correcting each active node independently.
///
/// This is benchmark-only: production correcting reads use the global bad-helper
/// filter above, which drops helpers after they are detected on earlier nodes.
#[cfg(feature = "bench")]
pub(crate) fn reconstruct_path_from_shares_correcting_no_filter<
    P: UnauthArrayParams + UnauthDecodeParams,
>(
    auth: &AuthParams,
    params: &P,
    node_linearization: mapping::NodeLinearization,
    allocated_leaves: usize,
    index: u64,
    nodes: &[NodeKey],
    shares: &HashMap<NodeKey, Vec<ShardShare>>,
    max_errors: usize,
) -> Result<(Value256, AuthPath), ReadError> {
    if nodes.len() != auth.tree_height + 1 {
        return Err(ReadError::InvalidProof);
    }

    let mut values = HashMap::new();
    for node in nodes {
        let Some(idx) = node_to_index_if_active(auth, node_linearization, allocated_leaves, *node)
        else {
            values.insert(*node, default_value_for_node(auth, *node));
            continue;
        };
        let node_shares = shares.get(node).cloned().unwrap_or_default();
        let val =
            decode_node_from_shares_with_policy(params, idx, &node_shares, Some(max_errors), None)?;
        values.insert(*node, encode_value(val));
    }

    let Some(value) = values.get(&nodes[0]).copied() else {
        return Err(ReadError::InvalidProof);
    };
    let mut siblings = Vec::with_capacity(auth.tree_height);
    for node in nodes.iter().skip(1) {
        let Some(val) = values.get(node) else {
            return Err(ReadError::InvalidProof);
        };
        siblings.push(*val);
    }
    let root = recompute_root(
        auth.tree_height as u32,
        index,
        value,
        &siblings,
        &auth.defaults,
    );
    Ok((
        value,
        AuthPath {
            index: index as usize,
            siblings,
            root,
        },
    ))
}

fn reconstruct_path_from_shares_with_policy<P: UnauthArrayParams + UnauthDecodeParams>(
    auth: &AuthParams,
    params: &P,
    node_linearization: mapping::NodeLinearization,
    allocated_leaves: usize,
    index: u64,
    nodes: &[NodeKey],
    shares: &HashMap<NodeKey, Vec<ShardShare>>,
    max_errors: Option<usize>,
) -> Result<(Value256, AuthPath), ReadError> {
    if nodes.len() != auth.tree_height + 1 {
        return Err(ReadError::InvalidProof);
    }

    let mut values = HashMap::new();
    let mut bad_registry_indices = HashSet::new();
    for node in nodes {
        let Some(idx) = node_to_index_if_active(auth, node_linearization, allocated_leaves, *node)
        else {
            values.insert(*node, default_value_for_node(auth, *node));
            continue;
        };
        let node_shares = shares.get(node).cloned().unwrap_or_default();
        let val = match max_errors {
            Some(max_errors) => decode_node_from_shares_with_global_bad_filter(
                params,
                idx,
                &node_shares,
                max_errors,
                &mut bad_registry_indices,
            )?,
            None => decode_node_from_shares(params, idx, &node_shares)?,
        };
        values.insert(*node, encode_value(val));
    }

    let Some(value) = values.get(&nodes[0]).copied() else {
        return Err(ReadError::InvalidProof);
    };
    let mut siblings = Vec::with_capacity(auth.tree_height);
    for node in nodes.iter().skip(1) {
        let Some(val) = values.get(node) else {
            return Err(ReadError::InvalidProof);
        };
        siblings.push(*val);
    }
    let root = recompute_root(
        auth.tree_height as u32,
        index,
        value,
        &siblings,
        &auth.defaults,
    );
    Ok((
        value,
        AuthPath {
            index: index as usize,
            siblings,
            root,
        },
    ))
}

fn decode_node_value_from_shares_with_policy<P: UnauthArrayParams + UnauthDecodeParams>(
    auth: &AuthParams,
    params: &P,
    node_linearization: mapping::NodeLinearization,
    allocated_leaves: usize,
    node: NodeKey,
    shares: &HashMap<NodeKey, Vec<ShardShare>>,
    max_errors: Option<usize>,
    cache: Option<&DecodeCache>,
) -> Result<Value256, ReadError> {
    let Some(idx) = node_to_index_if_active(auth, node_linearization, allocated_leaves, node)
    else {
        return Ok(default_value_for_node(auth, node));
    };
    let node_shares = shares.get(&node).cloned().unwrap_or_default();
    let val = decode_node_from_shares_with_policy(params, idx, &node_shares, max_errors, cache)?;
    Ok(encode_value(val))
}

fn decode_node_value_from_shares_with_global_bad_filter<
    P: UnauthArrayParams + UnauthDecodeParams,
>(
    auth: &AuthParams,
    params: &P,
    node_linearization: mapping::NodeLinearization,
    allocated_leaves: usize,
    node: NodeKey,
    shares: &HashMap<NodeKey, Vec<ShardShare>>,
    max_errors: usize,
    bad_registry_indices: &mut HashSet<usize>,
) -> Result<Value256, ReadError> {
    let Some(idx) = node_to_index_if_active(auth, node_linearization, allocated_leaves, node)
    else {
        return Ok(default_value_for_node(auth, node));
    };
    let node_shares = shares.get(&node).cloned().unwrap_or_default();
    let val = decode_node_from_shares_with_global_bad_filter(
        params,
        idx,
        &node_shares,
        max_errors,
        bad_registry_indices,
    )?;
    Ok(encode_value(val))
}

fn recompute_subtree_root_from_leaves(
    auth: &AuthParams,
    start_index: u64,
    leaves: &[Value256],
) -> Value256 {
    debug_assert!(!leaves.is_empty());
    debug_assert!(leaves.len().is_power_of_two());
    let height = auth.tree_height as u32;
    let mut level: Vec<Value256> = leaves
        .iter()
        .copied()
        .enumerate()
        .map(|(offset, value)| {
            let offset = u64::try_from(offset).expect("leaf offset fits into u64");
            leaf_label(&auth.defaults, height, start_index + offset, value)
        })
        .collect();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks_exact(2) {
            next.push(node_hash(pair[0], pair[1]));
        }
        level = next;
    }
    level[0]
}

/// Reconstruct an upper-proof subtree using known leaf values and upper-path shares.
pub(crate) fn reconstruct_subtree_from_leaves_and_upper_shares<
    P: UnauthArrayParams + UnauthDecodeParams,
>(
    auth: &AuthParams,
    params: &P,
    node_linearization: mapping::NodeLinearization,
    allocated_leaves: usize,
    plan: &SubtreeReadPlan,
    raw_leaves: &[Value256],
    upper_path_shares: &HashMap<NodeKey, Vec<ShardShare>>,
) -> Result<AuthSubtreeProof, ReadError> {
    if raw_leaves.len() != plan.leaf_count {
        return Err(ReadError::InvalidProof);
    }
    let subtree_root_label = recompute_subtree_root_from_leaves(auth, plan.start_index, raw_leaves);
    let mut upper_siblings = Vec::with_capacity(plan.upper_path_nodes.len());
    for &node in &plan.upper_path_nodes {
        let Some(idx) = node_to_index_if_active(auth, node_linearization, allocated_leaves, node)
        else {
            upper_siblings.push(default_value_for_node(auth, node));
            continue;
        };
        let node_shares = upper_path_shares.get(&node).cloned().unwrap_or_default();
        let val = decode_node_from_shares(params, idx, &node_shares)?;
        upper_siblings.push(encode_value(val));
    }
    let root =
        recompute_root_from_subtree(auth, plan.subtree_root, subtree_root_label, &upper_siblings);
    Ok(AuthSubtreeProof {
        start_index: plan.start_index,
        leaf_count: plan.leaf_count,
        upper_siblings,
        root,
    })
}

fn recompute_root_from_subtree(
    auth: &AuthParams,
    subtree_root: NodeKey,
    subtree_root_label: Value256,
    upper_siblings: &[Value256],
) -> Value256 {
    let height = auth.tree_height as u32;
    let mut key = subtree_root;
    let mut cur = subtree_root_label;
    for sib_content in upper_siblings.iter() {
        let sib_key = sibling(key);
        let sib_label = label_from_content(&auth.defaults, height, sib_key, *sib_content);
        cur = if key.pos % 2 == 0 {
            node_hash(cur, sib_label)
        } else {
            node_hash(sib_label, cur)
        };
        key = parent(key);
    }
    cur
}

/// Reconstruct all leaves in one aligned subtree plus its upper authentication path.
pub(crate) fn reconstruct_subtree_from_shares<P: UnauthArrayParams + UnauthDecodeParams>(
    auth: &AuthParams,
    params: &P,
    node_linearization: mapping::NodeLinearization,
    allocated_leaves: usize,
    plan: &SubtreeReadPlan,
    shares: &HashMap<NodeKey, Vec<ShardShare>>,
) -> Result<(Vec<Value256>, AuthSubtreeProof), ReadError> {
    reconstruct_subtree_from_shares_with_policy(
        auth,
        params,
        node_linearization,
        allocated_leaves,
        plan,
        shares,
        None,
    )
}

/// Reconstruct one aligned subtree while correcting up to `max_errors` bad
/// shard replies per active node.
pub(crate) fn reconstruct_subtree_from_shares_correcting<
    P: UnauthArrayParams + UnauthDecodeParams,
>(
    auth: &AuthParams,
    params: &P,
    node_linearization: mapping::NodeLinearization,
    allocated_leaves: usize,
    plan: &SubtreeReadPlan,
    shares: &HashMap<NodeKey, Vec<ShardShare>>,
    max_errors: usize,
) -> Result<(Vec<Value256>, AuthSubtreeProof), ReadError> {
    reconstruct_subtree_from_shares_with_policy(
        auth,
        params,
        node_linearization,
        allocated_leaves,
        plan,
        shares,
        Some(max_errors),
    )
}

fn reconstruct_subtree_from_shares_with_policy<P: UnauthArrayParams + UnauthDecodeParams>(
    auth: &AuthParams,
    params: &P,
    node_linearization: mapping::NodeLinearization,
    allocated_leaves: usize,
    plan: &SubtreeReadPlan,
    shares: &HashMap<NodeKey, Vec<ShardShare>>,
    max_errors: Option<usize>,
) -> Result<(Vec<Value256>, AuthSubtreeProof), ReadError> {
    let mut leaves = Vec::with_capacity(plan.leaf_nodes.len());
    let mut bad_registry_indices = HashSet::new();
    let decode_cache = DecodeCache::new();
    for &node in &plan.leaf_nodes {
        let value = match max_errors {
            Some(max_errors) => decode_node_value_from_shares_with_global_bad_filter(
                auth,
                params,
                node_linearization,
                allocated_leaves,
                node,
                shares,
                max_errors,
                &mut bad_registry_indices,
            )?,
            None => decode_node_value_from_shares_with_policy(
                auth,
                params,
                node_linearization,
                allocated_leaves,
                node,
                shares,
                None,
                Some(&decode_cache),
            )?,
        };
        leaves.push(value);
    }
    let mut upper_siblings = Vec::with_capacity(plan.upper_path_nodes.len());
    for &node in &plan.upper_path_nodes {
        let value = match max_errors {
            Some(max_errors) => decode_node_value_from_shares_with_global_bad_filter(
                auth,
                params,
                node_linearization,
                allocated_leaves,
                node,
                shares,
                max_errors,
                &mut bad_registry_indices,
            )?,
            None => decode_node_value_from_shares_with_policy(
                auth,
                params,
                node_linearization,
                allocated_leaves,
                node,
                shares,
                None,
                Some(&decode_cache),
            )?,
        };
        upper_siblings.push(value);
    }
    let subtree_root_label = recompute_subtree_root_from_leaves(auth, plan.start_index, &leaves);
    let root =
        recompute_root_from_subtree(auth, plan.subtree_root, subtree_root_label, &upper_siblings);
    Ok((
        leaves,
        AuthSubtreeProof {
            start_index: plan.start_index,
            leaf_count: plan.leaf_count,
            upper_siblings,
            root,
        },
    ))
}

/// Return true if every active node has enough shares.
///
/// Inactive nodes are ignored.
pub(crate) fn all_nodes_satisfied(
    nodes: &[NodeKey],
    shares: &HashMap<NodeKey, Vec<ShardShare>>,
    need: usize,
    auth: &AuthParams,
    node_linearization: mapping::NodeLinearization,
    allocated_leaves: usize,
) -> bool {
    for node in nodes {
        // Ignore inactive nodes; they do not require shares.
        if node_to_index_if_active(auth, node_linearization, allocated_leaves, *node).is_none() {
            continue;
        }
        // Active nodes must meet the minimum shard threshold.
        let count = shares.get(node).map(|v| v.len()).unwrap_or(0);
        if count < need {
            return false;
        }
    }
    true
}

/// Verify a contiguous range proof and return the authenticated root.
pub(crate) fn compute_range_root(
    auth: &AuthParams,
    start_index: u64,
    leaves: &[Value256],
    proof_nodes: &[(NodeKey, Value256)],
) -> Result<Value256, ReadError> {
    if leaves.is_empty() {
        return Err(ReadError::InvalidProof);
    }
    let leaf_count_u64 = u64::try_from(leaves.len()).map_err(|_| ReadError::InvalidProof)?;
    let end = (start_index as u128)
        .checked_add(leaves.len() as u128)
        .ok_or(ReadError::InvalidProof)?;
    if end > auth.conceptual_l_max_u128() {
        return Err(ReadError::InvalidProof);
    }
    let expected_proof_nodes =
        canonical_range_boundary_proof_nodes(auth, start_index, leaves.len())?;
    let expected_internal_nodes = canonical_range_internal_nodes(auth, start_index, leaves.len())?;
    let expected_len = expected_proof_nodes
        .len()
        .checked_add(expected_internal_nodes.len())
        .ok_or(ReadError::InvalidProof)?;
    if proof_nodes.len() != expected_len {
        return Err(ReadError::InvalidProof);
    }

    let mut proof_values = HashMap::with_capacity(expected_proof_nodes.len());
    for (&expected_node, &(node, value)) in expected_proof_nodes.iter().zip(proof_nodes.iter()) {
        if node != expected_node {
            return Err(ReadError::InvalidProof);
        }
        if proof_values.insert(node, value).is_some() {
            return Err(ReadError::InvalidProof);
        }
    }

    let mut checkpoint_values_by_level = vec![Vec::new(); auth.tree_height + 1];
    for (&expected_node, &(node, value)) in expected_internal_nodes
        .iter()
        .zip(proof_nodes[expected_proof_nodes.len()..].iter())
    {
        if node != expected_node {
            return Err(ReadError::InvalidProof);
        }
        checkpoint_values_by_level[node.level as usize].push(value);
    }
    let mut checkpoint_offsets = vec![0usize; auth.tree_height + 1];

    let height = auth.tree_height as u32;
    let mut current_labels = Vec::with_capacity(leaves.len());
    for (offset, &value) in leaves.iter().enumerate() {
        let offset = u64::try_from(offset).map_err(|_| ReadError::InvalidProof)?;
        let pos = start_index
            .checked_add(offset)
            .ok_or(ReadError::InvalidProof)?;
        current_labels.push(leaf_label(&auth.defaults, height, pos, value));
    }
    let mut level = height;
    let mut start_pos = start_index;
    let mut end_pos = start_index
        .checked_add(leaf_count_u64)
        .ok_or(ReadError::InvalidProof)?;
    while level > 0 {
        let parent_level = level - 1;
        let parent_start = start_pos / 2;
        let parent_end = end_pos.div_ceil(2);
        let next_len =
            usize::try_from(parent_end - parent_start).map_err(|_| ReadError::InvalidProof)?;
        for (write_pos, parent_pos) in (parent_start..parent_end).enumerate() {
            let parent_node = NodeKey {
                level: parent_level,
                pos: parent_pos,
            };
            let left = NodeKey {
                level,
                pos: parent_pos * 2,
            };
            let right = NodeKey {
                level,
                pos: parent_pos * 2 + 1,
            };
            let left_label = if left.pos >= start_pos && left.pos < end_pos {
                let offset =
                    usize::try_from(left.pos - start_pos).map_err(|_| ReadError::InvalidProof)?;
                current_labels
                    .get(offset)
                    .copied()
                    .ok_or(ReadError::InvalidProof)?
            } else {
                let content = proof_values
                    .get(&left)
                    .copied()
                    .ok_or(ReadError::InvalidProof)?;
                label_from_content(&auth.defaults, height, left, content)
            };
            let right_label = if right.pos >= start_pos && right.pos < end_pos {
                let offset =
                    usize::try_from(right.pos - start_pos).map_err(|_| ReadError::InvalidProof)?;
                current_labels
                    .get(offset)
                    .copied()
                    .ok_or(ReadError::InvalidProof)?
            } else {
                let content = proof_values
                    .get(&right)
                    .copied()
                    .ok_or(ReadError::InvalidProof)?;
                label_from_content(&auth.defaults, height, right, content)
            };
            let computed_label = node_hash(left_label, right_label);
            if parent_level > 0 {
                let level_pos = parent_level as usize;
                let offset = checkpoint_offsets[level_pos];
                let Some(&content) = checkpoint_values_by_level[level_pos].get(offset) else {
                    return Err(ReadError::InvalidProof);
                };
                let expected_label =
                    label_from_content(&auth.defaults, height, parent_node, content);
                if expected_label != computed_label {
                    return Err(ReadError::InvalidProof);
                }
                checkpoint_offsets[level_pos] += 1;
            }
            let Some(slot) = current_labels.get_mut(write_pos) else {
                return Err(ReadError::InvalidProof);
            };
            *slot = computed_label;
        }
        current_labels.truncate(next_len);
        start_pos = parent_start;
        end_pos = parent_end;
        level = parent_level;
    }

    if start_pos != 0 || end_pos != 1 || current_labels.len() != 1 {
        return Err(ReadError::InvalidProof);
    }
    for (level, values) in checkpoint_values_by_level.iter().enumerate() {
        if checkpoint_offsets[level] != values.len() {
            return Err(ReadError::InvalidProof);
        }
    }
    Ok(current_labels[0])
}

/// Verify a contiguous range proof and return the authenticated root.
pub fn verify_range_proof(
    auth: &AuthParams,
    start_index: u64,
    leaves: &[Value256],
    proof: &AuthRangeProof,
) -> Result<Value256, ReadError> {
    if proof.start_index != start_index || proof.leaf_count != leaves.len() {
        return Err(ReadError::InvalidProof);
    }
    let end = (start_index as u128)
        .checked_add(leaves.len() as u128)
        .ok_or(ReadError::InvalidProof)?;
    if end > proof.allocated_leaves as u128
        || proof.allocated_leaves as u128 > auth.conceptual_l_max_u128()
    {
        return Err(ReadError::InvalidProof);
    }
    let root = compute_range_root(auth, start_index, leaves, &proof.proof_nodes)?;
    if root != proof.root {
        return Err(ReadError::InvalidProof);
    }
    Ok(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(level: u32, pos: u64) -> NodeKey {
        NodeKey { level, pos }
    }

    fn node_set(nodes: &[NodeKey]) -> HashSet<NodeKey> {
        nodes.iter().copied().collect()
    }

    #[test]
    fn aligned_range_plan_requests_internal_checkpoint_nodes() {
        let auth = AuthParams::new(4);
        let expected = [node(1, 0), node(2, 0), node(3, 0), node(3, 1)];

        let internal_nodes =
            canonical_range_internal_nodes(&auth, 0, 4).expect("aligned range internal nodes");
        assert_eq!(node_set(&internal_nodes), node_set(&expected));

        let plan = MerkleRangePlan::new_with_index_maps(
            &auth,
            mapping::NodeLinearization::Preorder,
            16,
            0,
            4,
            true,
        )
        .expect("range plan");
        assert_eq!(
            node_set(&plan.internal_checkpoint_nodes),
            node_set(&expected)
        );
        for expected_node in expected {
            let array_index = mapping::preorder_idx(
                auth.tree_height as u16,
                expected_node.level as u16,
                expected_node.pos,
            );
            assert!(plan.node_to_array_index.contains_key(&expected_node));
            assert!(plan.requested_array_indices.contains(&array_index));
        }
    }

    #[test]
    fn nonaligned_range_plan_requests_internal_checkpoint_ancestors() {
        let auth = AuthParams::new(4);
        let expected = [
            node(1, 0),
            node(2, 0),
            node(2, 1),
            node(3, 0),
            node(3, 1),
            node(3, 2),
            node(3, 3),
        ];

        let internal_nodes =
            canonical_range_internal_nodes(&auth, 1, 6).expect("nonaligned range internal nodes");
        assert_eq!(node_set(&internal_nodes), node_set(&expected));

        let plan = MerkleRangePlan::new_with_index_maps(
            &auth,
            mapping::NodeLinearization::Preorder,
            16,
            1,
            6,
            true,
        )
        .expect("range plan");
        assert_eq!(
            node_set(&plan.internal_checkpoint_nodes),
            node_set(&expected)
        );
        for expected_node in expected {
            assert!(plan.node_to_array_index.contains_key(&expected_node));
        }
    }

    #[test]
    fn singleton_range_requests_internal_checkpoint_ancestors() {
        let auth = AuthParams::new(4);
        let expected = [node(1, 0), node(2, 1), node(3, 2)];

        let internal_nodes =
            canonical_range_internal_nodes(&auth, 5, 1).expect("singleton range internal nodes");
        assert_eq!(node_set(&internal_nodes), node_set(&expected));

        let plan = MerkleRangePlan::new_with_index_maps(
            &auth,
            mapping::NodeLinearization::Preorder,
            16,
            5,
            1,
            true,
        )
        .expect("singleton range plan");
        assert_eq!(
            node_set(&plan.internal_checkpoint_nodes),
            node_set(&expected)
        );
    }
}
