//! Helpers for translating Merkle updates into per-node and coded-array deltas.

use super::mapping::NodeLinearization;
use super::merkle::{
    NodeKey, child_left, child_right, default_hash, is_valid_key, label_from_content, leaf_key,
    leaf_label, node_hash, parent, recompute_root, sibling,
};
use super::read_path::{build_subtree_read_plan, node_to_index_if_active};
use super::{
    AuthParams, AuthPath, AuthSubtreeProof, BatchWriteError, LeafUpdateBatch, MerkleTreeProof,
    WriteError,
};
use crate::field::{Fp, fp_zero};
use crate::value::{Value256, decode_value};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

fn auth_delta_trace_enabled() -> bool {
    std::env::var("AUTH_WRITE_TRACE")
        .map(|value| value == "1")
        .unwrap_or(false)
}

/// Translate a node into its logical unauth array index.
fn node_to_index(
    auth: &AuthParams,
    node_linearization: NodeLinearization,
    allocated_leaves: usize,
    key: NodeKey,
) -> Option<u64> {
    node_to_index_if_active(auth, node_linearization, allocated_leaves, key)
}

/// Compute per-node deltas needed to update a Merkle path.
pub fn compute_deltas_for_write(
    auth: &AuthParams,
    index: usize,
    old: Value256,
    new: Value256,
    siblings: &[Value256],
) -> Vec<(NodeKey, Fp)> {
    let height = auth.tree_height as u32;
    let mut deltas = Vec::with_capacity(auth.tree_height + 1);
    // Leaf delta is the difference between new and old leaf payloads.
    let mut key = leaf_key(height, index as u64);
    deltas.push((key, decode_value(new) - decode_value(old)));

    // Recompute labels up the path to derive parent deltas.
    let mut old_label = leaf_label(&auth.defaults, height, index as u64, old);
    let mut new_label = leaf_label(&auth.defaults, height, index as u64, new);
    for sib_content in siblings.iter() {
        let sib_key = sibling(key);
        let sib_label = label_from_content(&auth.defaults, height, sib_key, *sib_content);
        let parent_old = if key.pos % 2 == 0 {
            node_hash(old_label, sib_label)
        } else {
            node_hash(sib_label, old_label)
        };
        let parent_new = if key.pos % 2 == 0 {
            node_hash(new_label, sib_label)
        } else {
            node_hash(sib_label, new_label)
        };
        key = parent(key);
        // Delta is computed in the field for the parent label.
        deltas.push((key, decode_value(parent_new) - decode_value(parent_old)));
        old_label = parent_old;
        new_label = parent_new;
    }
    deltas
}

pub(crate) fn compute_single_update_deltas(
    auth: &AuthParams,
    node_linearization: NodeLinearization,
    allocated_leaves: usize,
    index: usize,
    old: Value256,
    new: Value256,
    path: &AuthPath,
) -> Result<(Value256, Vec<(u64, Fp)>), WriteError> {
    // Reject writes outside the allocated prefix.
    if index as u128 >= auth.conceptual_l_max_u128() || index >= allocated_leaves {
        return Err(WriteError::InvalidIndex);
    }
    // The path must match the requested index and have full height.
    if path.index != index || path.siblings.len() != auth.tree_height {
        return Err(WriteError::InvalidPath);
    }
    let height = auth.tree_height as u32;
    // Recompute the old root from the supplied path to validate it.
    let root_old = recompute_root(height, index as u64, old, &path.siblings, &auth.defaults);
    if root_old != path.root {
        return Err(WriteError::RootMismatch);
    }
    // Compute the new root after the leaf swap.
    let root_new = recompute_root(height, index as u64, new, &path.siblings, &auth.defaults);

    // Convert per-node deltas into logical unauth array deltas.
    let deltas = compute_deltas_for_write(auth, index, old, new, &path.siblings);
    let mut logical_deltas = Vec::new();
    for (key, delta) in deltas {
        if delta == fp_zero() {
            continue;
        }
        let Some(idx) = node_to_index(auth, node_linearization, allocated_leaves, key) else {
            return Err(WriteError::InvalidPath);
        };
        logical_deltas.push((idx, delta));
    }
    logical_deltas.sort_by_key(|(idx, _)| *idx);
    Ok((root_new, logical_deltas))
}

pub(crate) fn compute_batch_update_deltas(
    auth: &AuthParams,
    node_linearization: NodeLinearization,
    allocated_leaves: usize,
    updates: &[LeafUpdateBatch],
    proof: &MerkleTreeProof,
) -> Result<(Value256, Vec<(u64, Fp)>), BatchWriteError> {
    let trace = auth_delta_trace_enabled();
    let total_start = trace.then(Instant::now);
    // Empty batch is a no-op.
    if updates.is_empty() {
        return Ok((proof.root, Vec::new()));
    }

    let height = auth.tree_height as u32;
    let mut seen = HashSet::new();
    let mut updates_by_leaf: HashMap<NodeKey, (Value256, Value256)> = HashMap::new();
    for update in updates {
        // Ensure all updates target allocated leaves and are unique.
        if update.index as u128 >= auth.conceptual_l_max_u128() || update.index >= allocated_leaves
        {
            return Err(BatchWriteError::InvalidIndex);
        }
        if !seen.insert(update.index) {
            return Err(BatchWriteError::DuplicateIndex);
        }
        let key = leaf_key(height, update.index as u64);
        updates_by_leaf.insert(key, (update.old, update.new));
    }

    let root_old = proof.root;

    let mut proof_map: HashMap<NodeKey, Value256> = HashMap::new();
    for (node_key, content) in proof.nodes.iter().copied() {
        // Reject malformed keys or conflicting nodes in the proof.
        if !is_valid_key(height, node_key) {
            return Err(BatchWriteError::InvalidProof);
        }
        if let Some(existing) = proof_map.get(&node_key) {
            if *existing != content {
                return Err(BatchWriteError::ConflictingProofNode);
            }
        } else {
            proof_map.insert(node_key, content);
        }
    }

    let mut affected: HashSet<NodeKey> = HashSet::new();
    for &leaf in updates_by_leaf.keys() {
        // Track the leaf-to-root path for each updated leaf.
        let mut v = leaf;
        loop {
            affected.insert(v);
            if v.level == 0 {
                break;
            }
            v = parent(v);
        }
    }

    let mut nodes: Vec<NodeKey> = affected.iter().copied().collect();
    nodes.sort_by_key(|key| (std::cmp::Reverse(key.level), key.pos));

    let mut old_content: HashMap<NodeKey, Value256> = HashMap::new();
    let mut new_content: HashMap<NodeKey, Value256> = HashMap::new();
    for v in nodes {
        // Leaves: values come directly from the update batch.
        if v.level == height {
            let Some((old, new)) = updates_by_leaf.get(&v) else {
                return Err(BatchWriteError::InvalidProof);
            };
            old_content.insert(v, *old);
            new_content.insert(v, *new);
            continue;
        }
        // Internal nodes: derive labels from children (proof or updated values).
        let left = child_left(v);
        let right = child_right(v);
        let old_left = if affected.contains(&left) {
            *old_content
                .get(&left)
                .ok_or(BatchWriteError::InvalidProof)?
        } else {
            *proof_map.get(&left).ok_or(BatchWriteError::InvalidProof)?
        };
        let old_right = if affected.contains(&right) {
            *old_content
                .get(&right)
                .ok_or(BatchWriteError::InvalidProof)?
        } else {
            *proof_map.get(&right).ok_or(BatchWriteError::InvalidProof)?
        };
        let new_left = if affected.contains(&left) {
            *new_content
                .get(&left)
                .ok_or(BatchWriteError::InvalidProof)?
        } else {
            *proof_map.get(&left).ok_or(BatchWriteError::InvalidProof)?
        };
        let new_right = if affected.contains(&right) {
            *new_content
                .get(&right)
                .ok_or(BatchWriteError::InvalidProof)?
        } else {
            *proof_map.get(&right).ok_or(BatchWriteError::InvalidProof)?
        };
        // Recompute labels for both the old and new trees.
        let old_left_label = label_from_content(&auth.defaults, height, left, old_left);
        let old_right_label = label_from_content(&auth.defaults, height, right, old_right);
        let new_left_label = label_from_content(&auth.defaults, height, left, new_left);
        let new_right_label = label_from_content(&auth.defaults, height, right, new_right);
        let old_label = node_hash(old_left_label, old_right_label);
        let new_label = node_hash(new_left_label, new_right_label);
        old_content.insert(v, old_label);
        new_content.insert(v, new_label);
    }

    // Validate that the proof's old labels match the recomputed ones.
    for (node_key, content) in proof_map.iter() {
        if affected.contains(node_key) {
            let Some(old_label) = old_content.get(node_key) else {
                return Err(BatchWriteError::InvalidProof);
            };
            if old_label != content {
                return Err(BatchWriteError::InvalidProof);
            }
        }
    }

    // Root must match the proof root.
    let root_key = NodeKey { level: 0, pos: 0 };
    let root_recomputed = old_content
        .get(&root_key)
        .copied()
        .ok_or(BatchWriteError::InvalidProof)?;
    if root_recomputed != root_old {
        return Err(BatchWriteError::InvalidProof);
    }
    // New root is the recomputed root after applying updates.
    let root_new = *new_content
        .get(&root_key)
        .ok_or(BatchWriteError::InvalidProof)?;

    let mut deltas_by_index: HashMap<u64, Fp> = HashMap::new();
    for &node in affected.iter() {
        // Convert label deltas into logical unauth array deltas.
        let old_val = old_content
            .get(&node)
            .copied()
            .ok_or(BatchWriteError::InvalidProof)?;
        let new_val = new_content
            .get(&node)
            .copied()
            .ok_or(BatchWriteError::InvalidProof)?;
        let delta = decode_value(new_val) - decode_value(old_val);
        if delta == fp_zero() {
            continue;
        }
        let Some(idx) = node_to_index(auth, node_linearization, allocated_leaves, node) else {
            return Err(BatchWriteError::InvalidProof);
        };
        deltas_by_index
            .entry(idx)
            .and_modify(|d| *d += delta)
            .or_insert(delta);
    }

    // Normalize ordering for deterministic output and easy comparison.
    let mut logical_deltas: Vec<(u64, Fp)> = deltas_by_index.into_iter().collect();
    logical_deltas.sort_by_key(|(idx, _)| *idx);

    if trace {
        eprintln!(
            "[auth-write-trace] batch_delta_build leaves={} affected_nodes={} logical_deltas={} total_ms={:.3}",
            updates.len(),
            affected.len(),
            logical_deltas.len(),
            total_start.unwrap().elapsed().as_secs_f64() * 1_000.0,
        );
    }

    Ok((root_new, logical_deltas))
}

fn recompute_root_from_subtree_label(
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

pub(crate) fn compute_fresh_subtree_update_deltas(
    auth: &AuthParams,
    node_linearization: NodeLinearization,
    allocated_leaves: usize,
    leaves: &[Value256],
    proof: &AuthSubtreeProof,
) -> Result<(Value256, Vec<(u64, Fp)>), BatchWriteError> {
    if leaves.is_empty() {
        return Ok((proof.root, Vec::new()));
    }
    let plan = build_subtree_read_plan(auth, proof.start_index, leaves.len())
        .map_err(|_| BatchWriteError::InvalidProof)?;
    if proof.leaf_count != leaves.len() {
        return Err(BatchWriteError::InvalidProof);
    }
    if proof.upper_siblings.len() != plan.upper_path_nodes.len() {
        return Err(BatchWriteError::InvalidProof);
    }
    let leaf_count_u64 = u64::try_from(leaves.len()).map_err(|_| BatchWriteError::InvalidIndex)?;
    let end = proof
        .start_index
        .checked_add(leaf_count_u64)
        .ok_or(BatchWriteError::InvalidIndex)?;
    if end as u128 > auth.conceptual_l_max_u128() || end > allocated_leaves as u64 {
        return Err(BatchWriteError::InvalidIndex);
    }

    let old_subtree_root = default_hash(&auth.defaults, plan.subtree_root.level as usize);
    let root_old = recompute_root_from_subtree_label(
        auth,
        plan.subtree_root,
        old_subtree_root,
        &proof.upper_siblings,
    );
    if root_old != proof.root {
        return Err(BatchWriteError::InvalidProof);
    }

    let updates = leaves
        .iter()
        .copied()
        .enumerate()
        .map(|(offset, new)| {
            let offset = u64::try_from(offset).map_err(|_| BatchWriteError::InvalidIndex)?;
            let index = proof
                .start_index
                .checked_add(offset)
                .ok_or(BatchWriteError::InvalidIndex)?;
            let index = usize::try_from(index).map_err(|_| BatchWriteError::InvalidIndex)?;
            Ok(LeafUpdateBatch {
                index,
                old: [0u8; 32],
                new,
            })
        })
        .collect::<Result<Vec<_>, BatchWriteError>>()?;
    let proof_nodes = plan
        .upper_path_nodes
        .iter()
        .copied()
        .zip(proof.upper_siblings.iter().copied())
        .collect();
    compute_batch_update_deltas(
        auth,
        node_linearization,
        allocated_leaves,
        &updates,
        &MerkleTreeProof {
            root: proof.root,
            nodes: proof_nodes,
        },
    )
}
