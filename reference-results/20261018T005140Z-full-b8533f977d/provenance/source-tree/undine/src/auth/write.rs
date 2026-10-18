//! Authenticated SMT update helpers: broadcast construction and store application.
//!
//! This module constructs broadcast updates and applies them to store-local shard
//! vectors. Delta translation lives in `deltas`.

use super::allocation::default_value_at_index;
use super::deltas::{
    compute_batch_update_deltas, compute_fresh_subtree_update_deltas, compute_single_update_deltas,
};
use super::mapping::NodeLinearization;
use super::merkle::{
    NodeKey, child_left, child_right, default_hash, label_from_content, leaf_label, node_hash,
    parent, sibling,
};
use super::read_path::{build_subtree_read_plan, node_to_index_if_active};
use super::{
    AuthBroadcastBatchUpdate, AuthBroadcastUpdate, AuthParams, AuthPath, AuthStore,
    AuthSubtreeProof, BatchWriteError, BatchWriteOpts, LeafUpdateBatch, MerkleTreeProof,
    WriteError,
};
use crate::field::{Fp, fp_zero};
use crate::unauth::{CodedArrayParams, DeltaMode, UnauthArray, UnauthArrayParams, UnauthStore};
use crate::value::{Value256, decode_value};
use std::collections::HashMap;
use std::time::Instant;

fn auth_write_trace_enabled() -> bool {
    std::env::var("AUTH_WRITE_TRACE")
        .map(|value| value == "1")
        .unwrap_or(false)
}

/// Convenience wrapper: build and apply a broadcast update (aggregate mode).
pub fn write<U, P>(
    auth: &AuthParams,
    params: &P,
    node_linearization: NodeLinearization,
    store: &mut AuthStore<U>,
    allocated_leaves: usize,
    i: usize,
    old: Value256,
    new: Value256,
    path: AuthPath,
) -> Result<(), WriteError>
where
    U: UnauthArray<Params = P> + Clone,
    P: UnauthArrayParams,
{
    // Construct a self-contained update that carries all deltas and metadata.
    let update = build_broadcast_update(
        auth,
        params,
        node_linearization,
        store,
        allocated_leaves,
        i,
        old,
        new,
        path,
        DeltaMode::Aggregate,
    )?;
    // Apply against the local store (root/delta validation is repeated).
    apply_broadcast_update(auth, params, node_linearization, store, &update)
}

/// Build a broadcast update for a single-leaf write.
pub fn build_broadcast_update<U, P>(
    auth: &AuthParams,
    _params: &P,
    node_linearization: NodeLinearization,
    store: &AuthStore<U>,
    allocated_leaves: usize,
    i: usize,
    old: Value256,
    new: Value256,
    path: AuthPath,
    delta_mode: DeltaMode,
) -> Result<AuthBroadcastUpdate, WriteError>
where
    U: UnauthArray<Params = P> + Clone,
    P: UnauthArrayParams,
{
    // Compute the new root plus logical deltas for the leaf update.
    let (root_new, deltas) = compute_single_update_deltas(
        auth,
        node_linearization,
        allocated_leaves,
        i,
        old,
        new,
        &path,
    )?;
    // Reject updates that were computed from a stale root.
    let root_old = path.root;
    if store.root != root_old {
        return Err(WriteError::RootMismatch);
    }
    Ok(AuthBroadcastUpdate {
        index: i,
        old,
        new,
        path,
        allocated_leaves,
        root_new,
        delta_mode,
        deltas,
    })
}

/// Build a broadcast update for a batch write.
pub fn build_broadcast_update_batch<U, P>(
    auth: &AuthParams,
    _params: &P,
    node_linearization: NodeLinearization,
    store: &AuthStore<U>,
    allocated_leaves: usize,
    updates: Vec<LeafUpdateBatch>,
    proof: MerkleTreeProof,
    opts: BatchWriteOpts,
) -> Result<AuthBroadcastBatchUpdate, BatchWriteError>
where
    U: UnauthArray<Params = P> + Clone,
    P: UnauthArrayParams,
{
    // Ensure the batch proof is against the store's current root.
    let root_old = proof.root;
    if store.root != root_old {
        return Err(BatchWriteError::RootMismatch);
    }
    // Compute root+logical-delta summary for all leaves touched by the batch.
    let (root_new, deltas) =
        compute_batch_update_deltas(auth, node_linearization, allocated_leaves, &updates, &proof)?;
    // Package the result for broadcast to all parties.
    Ok(AuthBroadcastBatchUpdate {
        updates,
        proof,
        allocated_leaves,
        root_new,
        delta_mode: opts.delta_mode,
        deltas,
    })
}

/// Convenience wrapper: build and apply a broadcast batch update.
pub fn write_batch<U, P>(
    auth: &AuthParams,
    params: &P,
    node_linearization: NodeLinearization,
    store: &mut AuthStore<U>,
    allocated_leaves: usize,
    updates: Vec<LeafUpdateBatch>,
    proof: MerkleTreeProof,
    opts: BatchWriteOpts,
) -> Result<(), BatchWriteError>
where
    U: UnauthArray<Params = P> + Clone,
    P: UnauthArrayParams,
{
    // Build a batch update with all deltas and validation metadata.
    let update = build_broadcast_update_batch(
        auth,
        params,
        node_linearization,
        store,
        allocated_leaves,
        updates,
        proof,
        opts,
    )?;
    // Apply the update locally. The proof and delta set were just validated
    // while constructing `update`, so avoid recomputing the same multiproof.
    apply_prevalidated_broadcast_update_batch(store, &update)
}

/// Apply a broadcast update to the store.
pub fn apply_broadcast_update<U, P>(
    auth: &AuthParams,
    _coded: &P,
    node_linearization: NodeLinearization,
    store: &mut AuthStore<U>,
    update: &AuthBroadcastUpdate,
) -> Result<(), WriteError>
where
    U: UnauthArray<Params = P> + Clone,
    P: UnauthArrayParams,
{
    // The local root must match the update's path root.
    if store.root != update.path.root {
        return Err(WriteError::RootMismatch);
    }
    // Recompute deltas to validate the update before applying it.
    let (root_new, expected) = compute_single_update_deltas(
        auth,
        node_linearization,
        update.allocated_leaves,
        update.index,
        update.old,
        update.new,
        &update.path,
    )?;
    // Validate the update's claimed root and delta set.
    if root_new != update.root_new {
        return Err(WriteError::RootMismatch);
    }
    if expected != update.deltas {
        return Err(WriteError::InvalidPath);
    }
    // Apply logical deltas to the underlying array and update the root.
    store
        .unauth
        .apply_deltas(&update.deltas, update.delta_mode)
        .map_err(map_unauth_write_error)?;
    store.root = update.root_new;
    Ok(())
}

/// Apply a broadcast batch update to the store.
pub fn apply_broadcast_update_batch<U, P>(
    auth: &AuthParams,
    _coded: &P,
    node_linearization: NodeLinearization,
    store: &mut AuthStore<U>,
    update: &AuthBroadcastBatchUpdate,
) -> Result<(), BatchWriteError>
where
    U: UnauthArray<Params = P> + Clone,
    P: UnauthArrayParams,
{
    // Root must match the proof root before applying batch updates.
    if store.root != update.proof.root {
        return Err(BatchWriteError::RootMismatch);
    }
    // Recompute deltas and root to validate the update.
    let (root_new, expected) = compute_batch_update_deltas(
        auth,
        node_linearization,
        update.allocated_leaves,
        &update.updates,
        &update.proof,
    )?;
    // Validate the recomputed root and the full delta set.
    if root_new != update.root_new {
        return Err(BatchWriteError::InvalidProof);
    }
    if expected != update.deltas {
        return Err(BatchWriteError::InvalidProof);
    }
    // Apply the logical array deltas and update the stored root.
    store
        .unauth
        .apply_deltas(&update.deltas, update.delta_mode)
        .map_err(|_| BatchWriteError::InvalidIndex)?;
    store.root = update.root_new;
    Ok(())
}

/// Apply a batch update whose proof and delta set were already validated by
/// `build_broadcast_update_batch` against the same store root.
pub(crate) fn apply_prevalidated_broadcast_update_batch<U>(
    store: &mut AuthStore<U>,
    update: &AuthBroadcastBatchUpdate,
) -> Result<(), BatchWriteError>
where
    U: UnauthArray + Clone,
{
    if store.root != update.proof.root {
        return Err(BatchWriteError::RootMismatch);
    }
    store
        .unauth
        .apply_deltas(&update.deltas, update.delta_mode)
        .map_err(|_| BatchWriteError::InvalidIndex)?;
    store.root = update.root_new;
    Ok(())
}

/// Validate and apply one aligned fresh-subtree write.
///
/// The covered leaves are assumed to hold the default value under `proof.root`.
pub fn apply_fresh_subtree_write<U, P>(
    auth: &AuthParams,
    _coded: &P,
    node_linearization: NodeLinearization,
    store: &mut AuthStore<U>,
    allocated_leaves: usize,
    leaves: &[Value256],
    proof: &AuthSubtreeProof,
    delta_mode: DeltaMode,
) -> Result<Value256, BatchWriteError>
where
    U: UnauthArray<Params = P> + Clone,
    P: UnauthArrayParams,
{
    if store.root != proof.root {
        return Err(BatchWriteError::RootMismatch);
    }
    let trace = auth_write_trace_enabled();
    let compute_start = trace.then(Instant::now);
    let (root_new, deltas) = compute_fresh_subtree_update_deltas(
        auth,
        node_linearization,
        allocated_leaves,
        leaves,
        proof,
    )?;
    let compute_elapsed = compute_start.map(|start| start.elapsed());
    let apply_start = trace.then(Instant::now);
    store
        .unauth
        .apply_deltas(&deltas, delta_mode)
        .map_err(|_| BatchWriteError::InvalidIndex)?;
    let apply_elapsed = apply_start.map(|start| start.elapsed());
    store.root = root_new;
    if trace {
        eprintln!(
            "[auth-write-trace] fresh_subtree leaves={} logical_deltas={} compute_ms={:.3} apply_ms={:.3}",
            leaves.len(),
            deltas.len(),
            compute_elapsed.unwrap().as_secs_f64() * 1_000.0,
            apply_elapsed.unwrap().as_secs_f64() * 1_000.0,
        );
    }
    Ok(root_new)
}

/// Apply one aligned fresh-subtree write using a logical range-delta path.
pub fn apply_fresh_subtree_write_store(
    auth: &AuthParams,
    coded: &CodedArrayParams,
    node_linearization: NodeLinearization,
    store: &mut AuthStore<UnauthStore>,
    allocated_leaves: usize,
    leaves: &[Value256],
    proof: &AuthSubtreeProof,
    delta_mode: DeltaMode,
) -> Result<Value256, BatchWriteError> {
    if let Some(root_new) = try_apply_fresh_subtree_write_range_delta(
        auth,
        coded,
        node_linearization,
        store,
        allocated_leaves,
        leaves,
        proof,
        delta_mode,
    )? {
        return Ok(root_new);
    }
    apply_fresh_subtree_write(
        auth,
        coded,
        node_linearization,
        store,
        allocated_leaves,
        leaves,
        proof,
        delta_mode,
    )
}

fn try_apply_fresh_subtree_write_range_delta(
    auth: &AuthParams,
    _coded: &CodedArrayParams,
    node_linearization: NodeLinearization,
    store: &mut AuthStore<UnauthStore>,
    allocated_leaves: usize,
    leaves: &[Value256],
    proof: &AuthSubtreeProof,
    delta_mode: DeltaMode,
) -> Result<Option<Value256>, BatchWriteError> {
    if leaves.is_empty() {
        return Ok(Some(proof.root));
    }
    if store.root != proof.root {
        return Err(BatchWriteError::RootMismatch);
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

    let subtree_start = match node_to_index_if_active(
        auth,
        node_linearization,
        allocated_leaves,
        plan.subtree_root,
    ) {
        Some(idx) => idx,
        None => return Ok(None),
    };
    let subtree_node_count = leaves
        .len()
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(BatchWriteError::InvalidIndex)?;
    subtree_start
        .checked_add(subtree_node_count as u64)
        .ok_or(BatchWriteError::InvalidIndex)?;

    let trace = auth_write_trace_enabled();
    let total_start = trace.then(Instant::now);

    let old_subtree_root = default_hash(&auth.defaults, plan.subtree_root.level as usize);
    let build_start = trace.then(Instant::now);
    let (new_subtree_root, subtree_contents) =
        build_fresh_subtree_contents(auth, plan.subtree_root, proof.start_index, leaves);
    let build_elapsed = build_start.map(|start| start.elapsed());
    debug_assert_eq!(subtree_contents.len(), subtree_node_count);

    let upper_start = trace.then(Instant::now);
    let (root_old, root_new, upper_deltas) = compute_upper_path_deltas(
        auth,
        node_linearization,
        allocated_leaves,
        plan.subtree_root,
        old_subtree_root,
        new_subtree_root,
        &proof.upper_siblings,
    )?;
    let upper_elapsed = upper_start.map(|start| start.elapsed());
    if root_old != proof.root {
        return Err(BatchWriteError::InvalidProof);
    }

    let range_start = trace.then(Instant::now);
    store
        .unauth
        .write_fresh_range_from_defaults(subtree_start, &subtree_contents, delta_mode, |idx| {
            default_value_at_index(auth, node_linearization, idx)
        })
        .map_err(|_| BatchWriteError::InvalidIndex)?;
    let range_elapsed = range_start.map(|start| start.elapsed());

    let apply_start = trace.then(Instant::now);
    if !upper_deltas.is_empty() {
        store
            .unauth
            .apply_deltas(&upper_deltas, delta_mode)
            .map_err(|_| BatchWriteError::InvalidIndex)?;
    }
    let apply_elapsed = apply_start.map(|start| start.elapsed());
    store.root = root_new;

    if trace {
        eprintln!(
            "[auth-write-trace] fresh_subtree_fast leaves={} subtree_nodes={} ancestor_deltas={} build_ms={:.3} upper_ms={:.3} range_ms={:.3} apply_ms={:.3} total_ms={:.3}",
            leaves.len(),
            subtree_contents.len(),
            upper_deltas.len(),
            build_elapsed.unwrap().as_secs_f64() * 1_000.0,
            upper_elapsed.unwrap().as_secs_f64() * 1_000.0,
            range_elapsed.unwrap().as_secs_f64() * 1_000.0,
            apply_elapsed.unwrap().as_secs_f64() * 1_000.0,
            total_start.unwrap().elapsed().as_secs_f64() * 1_000.0,
        );
    }

    Ok(Some(root_new))
}

fn build_fresh_subtree_contents(
    auth: &AuthParams,
    subtree_root: NodeKey,
    start_index: u64,
    leaves: &[Value256],
) -> (Value256, Vec<Value256>) {
    let mut contents = Vec::with_capacity(leaves.len().saturating_mul(2).saturating_sub(1));
    let root_label =
        build_fresh_subtree_contents_rec(auth, subtree_root, start_index, leaves, &mut contents);
    (root_label, contents)
}

fn build_fresh_subtree_contents_rec(
    auth: &AuthParams,
    node: NodeKey,
    leaf_start: u64,
    leaves: &[Value256],
    out: &mut Vec<Value256>,
) -> Value256 {
    if node.level as usize == auth.tree_height {
        debug_assert_eq!(leaves.len(), 1);
        let value = leaves[0];
        out.push(value);
        return leaf_label(&auth.defaults, auth.tree_height as u32, leaf_start, value);
    }

    let slot = out.len();
    out.push([0u8; 32]);
    let mid = leaves.len() / 2;
    let left_label =
        build_fresh_subtree_contents_rec(auth, child_left(node), leaf_start, &leaves[..mid], out);
    let right_label = build_fresh_subtree_contents_rec(
        auth,
        child_right(node),
        leaf_start + mid as u64,
        &leaves[mid..],
        out,
    );
    let content = node_hash(left_label, right_label);
    out[slot] = content;
    content
}

fn compute_upper_path_deltas(
    auth: &AuthParams,
    node_linearization: NodeLinearization,
    allocated_leaves: usize,
    subtree_root: NodeKey,
    old_subtree_root: Value256,
    new_subtree_root: Value256,
    upper_siblings: &[Value256],
) -> Result<(Value256, Value256, Vec<(u64, Fp)>), BatchWriteError> {
    let mut by_index: HashMap<u64, Fp> = HashMap::new();
    let height = auth.tree_height as u32;
    let mut key = subtree_root;
    let mut old_label = old_subtree_root;
    let mut new_label = new_subtree_root;
    for sib_content in upper_siblings {
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
        let delta = decode_value(parent_new) - decode_value(parent_old);
        if delta != fp_zero() {
            let Some(idx) =
                node_to_index_if_active(auth, node_linearization, allocated_leaves, key)
            else {
                return Err(BatchWriteError::InvalidProof);
            };
            by_index
                .entry(idx)
                .and_modify(|existing| *existing += delta)
                .or_insert(delta);
        }
        old_label = parent_old;
        new_label = parent_new;
    }

    let mut deltas: Vec<(u64, Fp)> = by_index.into_iter().collect();
    deltas.sort_by_key(|(idx, _)| *idx);
    Ok((old_label, new_label, deltas))
}

fn map_unauth_write_error(_err: crate::unauth::WriteError) -> WriteError {
    // Auth writes only need the coarse error from the unauth store.
    WriteError::InvalidIndex
}
