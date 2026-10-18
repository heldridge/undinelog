#![allow(dead_code)]

use crate::Value256;
use crate::auth::mapping;
use crate::auth::merkle::{NodeKey, default_hash, leaf_key, parent, recompute_root, sibling};
use crate::auth::{
    AllocateError, AuthBroadcastBatchUpdate, AuthBroadcastUpdate, AuthParams, AuthPath, AuthStore,
    BatchWriteError, BatchWriteOpts, Config, LeafUpdateBatch, MerkleTreeProof, ReadError,
    WriteError,
};
use crate::unauth::{
    CodedArrayParams, DecodeCache, DeltaMode, ReadError as UnauthReadError, SetupTimings,
    UnauthStore, build_coded_params, read_batch_from_store, validate_coded_config,
};
use std::time::Instant;

/// Local, in-memory authenticated system harness for integration tests and dev tools.
pub struct AuthTestSystem {
    pub params: CodedArrayParams,
    pub auth: AuthParams,
    pub node_linearization: mapping::NodeLinearization,
    pub alloc_leaf_batch: usize,
    pub store: AuthStore,
    pub allocated_leaves: usize,
    segments: usize,
    decode_cache: DecodeCache,
}

/// Build a system without capturing setup timings.
pub fn setup(cfg: Config) -> AuthTestSystem {
    setup_internal(cfg, None)
}

/// Build a system and collect setup timing breakdowns.
pub fn setup_with_timing(cfg: Config) -> (AuthTestSystem, SetupTimings) {
    let mut timings = SetupTimings::default();
    let system = setup_internal(cfg, Some(&mut timings));
    (system, timings)
}

fn setup_internal(cfg: Config, mut timings: Option<&mut SetupTimings>) -> AuthTestSystem {
    validate_auth_config(&cfg);
    let params = build_coded_params(&cfg.coded);

    let tree_start = timings.as_ref().map(|_| Instant::now());
    let auth = AuthParams::new(cfg.auth.tree_height);
    let root0 = default_hash(&auth.defaults, 0);
    if let (Some(start), Some(timing)) = (tree_start, timings.as_deref_mut()) {
        timing.tree_build += start.elapsed();
    }

    let store = UnauthStore::new(params.clone());
    let store = AuthStore {
        root: root0,
        unauth: store,
    };

    let mut system = AuthTestSystem {
        params,
        auth,
        node_linearization: cfg.auth.node_linearization,
        alloc_leaf_batch: cfg.auth.alloc_leaf_batch,
        store,
        allocated_leaves: 0,
        segments: 0,
        decode_cache: DecodeCache::new(),
    };

    crate::auth::allocation::allocate_leaf_chunk(
        &system.auth,
        system.node_linearization,
        system.alloc_leaf_batch,
        &mut system.allocated_leaves,
        &mut system.store,
        timings.as_deref_mut(),
    )
    .expect("initial leaf allocation");
    system.segments = system.store.unauth.segments();

    system
}

impl AuthTestSystem {
    /// Read a leaf and return its Merkle authentication path.
    pub fn auth_read(&self, i: usize) -> Result<(Value256, AuthPath), ReadError> {
        // Reject indices outside the conceptual tree domain.
        if i as u128 >= self.auth.conceptual_l_max_u128() {
            return Err(ReadError::InvalidIndex);
        }
        let root = self.store.root;
        let height = self.auth.tree_height as u32;
        // Walk up the tree, collecting the leaf key and each sibling on the path.
        let mut nodes = Vec::with_capacity(self.auth.tree_height + 1);
        let mut key = leaf_key(height, i as u64);
        nodes.push(key);
        for _lvl in 0..self.auth.tree_height {
            let sib = sibling(key);
            nodes.push(sib);
            key = parent(key);
        }
        // Batch-read all active nodes; inactive nodes are filled with defaults.
        let node_values = self.read_nodes_by_key(&nodes)?;
        let value = node_values[0];
        let siblings = node_values[1..].to_vec();
        // Recompute the root to validate the read proof.
        let recomputed = recompute_root(height, i as u64, value, &siblings, &self.auth.defaults);
        if recomputed != root {
            return Err(ReadError::InvalidProof);
        }
        Ok((
            value,
            AuthPath {
                index: i,
                siblings,
                root,
            },
        ))
    }

    /// Alias for `auth_read`.
    pub fn read(&self, i: usize) -> Result<(Value256, AuthPath), ReadError> {
        self.auth_read(i)
    }

    /// Read node labels by key, returning defaults for inactive nodes.
    fn read_nodes_by_key(&self, nodes: &[NodeKey]) -> Result<Vec<Value256>, ReadError> {
        // Collect positions of active nodes and their linearized indices.
        let mut active_positions = Vec::new();
        let mut idxs = Vec::new();
        for (pos, node) in nodes.iter().enumerate() {
            let idx = crate::auth::read_path::node_to_index_if_active(
                &self.auth,
                self.node_linearization,
                self.allocated_leaves,
                *node,
            );
            if let Some(idx) = idx {
                active_positions.push(pos);
                idxs.push(idx);
            }
        }
        // Fill with read results for active nodes, leaving inactive as None.
        let mut values: Vec<Option<Value256>> = vec![None; nodes.len()];
        if !idxs.is_empty() {
            // One batched coded-array read, optionally assisted by decode cache.
            let read_values = read_batch_from_store(
                &self.params,
                &self.store.unauth.coded,
                &idxs,
                Some(&self.decode_cache),
            )
            .map_err(map_unauth_read_error)?;
            for (pos, value) in active_positions.into_iter().zip(read_values) {
                values[pos] = Some(value);
            }
        }
        // Materialize the full vector, substituting defaults for inactive nodes.
        let mut out = Vec::with_capacity(nodes.len());
        for (pos, node) in nodes.iter().enumerate() {
            let value = values[pos].unwrap_or_else(|| {
                crate::auth::read_path::default_value_for_node(&self.auth, *node)
            });
            out.push(value);
        }
        Ok(out)
    }

    /// Convenience wrapper: build and apply a broadcast update (aggregate mode).
    pub fn write(
        &mut self,
        i: usize,
        old: Value256,
        new: Value256,
        path: AuthPath,
    ) -> Result<(), WriteError> {
        crate::auth::write::write(
            &self.auth,
            &self.params,
            self.node_linearization,
            &mut self.store,
            self.allocated_leaves,
            i,
            old,
            new,
            path,
        )
    }

    /// Build a broadcast update for a single-leaf write.
    pub fn build_broadcast_update(
        &self,
        i: usize,
        old: Value256,
        new: Value256,
        path: AuthPath,
        delta_mode: DeltaMode,
    ) -> Result<AuthBroadcastUpdate, WriteError> {
        crate::auth::write::build_broadcast_update(
            &self.auth,
            &self.params,
            self.node_linearization,
            &self.store,
            self.allocated_leaves,
            i,
            old,
            new,
            path,
            delta_mode,
        )
    }

    /// Apply a broadcast update to the store.
    pub fn apply_broadcast_update(
        &mut self,
        update: &AuthBroadcastUpdate,
    ) -> Result<(), WriteError> {
        crate::auth::write::apply_broadcast_update(
            &self.auth,
            &self.params,
            self.node_linearization,
            &mut self.store,
            update,
        )
    }

    /// Build a broadcast update for a batch write.
    pub fn build_broadcast_update_batch(
        &self,
        updates: Vec<LeafUpdateBatch>,
        proof: MerkleTreeProof,
        opts: BatchWriteOpts,
    ) -> Result<AuthBroadcastBatchUpdate, BatchWriteError> {
        crate::auth::write::build_broadcast_update_batch(
            &self.auth,
            &self.params,
            self.node_linearization,
            &self.store,
            self.allocated_leaves,
            updates,
            proof,
            opts,
        )
    }

    /// Apply a broadcast batch update to the store.
    pub fn apply_broadcast_update_batch(
        &mut self,
        update: &AuthBroadcastBatchUpdate,
    ) -> Result<(), BatchWriteError> {
        crate::auth::write::apply_broadcast_update_batch(
            &self.auth,
            &self.params,
            self.node_linearization,
            &mut self.store,
            update,
        )
    }

    /// Convenience wrapper: build and apply a broadcast batch update.
    pub fn write_batch(
        &mut self,
        updates: Vec<LeafUpdateBatch>,
        proof: MerkleTreeProof,
        opts: BatchWriteOpts,
    ) -> Result<(), BatchWriteError> {
        crate::auth::write::write_batch(
            &self.auth,
            &self.params,
            self.node_linearization,
            &mut self.store,
            self.allocated_leaves,
            updates,
            proof,
            opts,
        )
    }

    /// Allocate the next leaf chunk without overriding labels.
    pub fn allocate_segment(&mut self) -> Result<(), AllocateError> {
        crate::auth::allocation::allocate_leaf_chunk(
            &self.auth,
            self.node_linearization,
            self.alloc_leaf_batch,
            &mut self.allocated_leaves,
            &mut self.store,
            None,
        )?;
        self.segments = self.store.unauth.segments();
        Ok(())
    }
}

fn map_unauth_read_error(err: UnauthReadError) -> ReadError {
    // Unauthenticated errors map into the auth read error space.
    match err {
        UnauthReadError::InvalidIndex => ReadError::InvalidIndex,
        UnauthReadError::NotEnoughShards => ReadError::NotEnoughShards,
    }
}

/// Emit a detailed mapping/allocation dump when SANITY=1 is set.
pub fn sanity_dump(system: &AuthTestSystem, sample_leaves: &[usize]) {
    if std::env::var("SANITY").ok().as_deref() != Some("1") {
        return;
    }
    let height = system.auth.tree_height as u16;
    let cursor = mapping::cursor(
        system.node_linearization,
        height,
        system.allocated_leaves as u64,
    );
    let segment_len = system.params.segment_len as u64;
    let needed_segments = cursor / segment_len + u64::from(cursor % segment_len != 0);
    eprintln!(
        "[sanity] tree_height={} alloc_leaf_batch={} segment_len={} N_grid={} s={} need={} shard_count={}",
        system.auth.tree_height,
        system.alloc_leaf_batch,
        system.params.segment_len,
        system.params.n_grid,
        system.params.s,
        system.params.need,
        system.params.shard_registry.len(),
    );
    eprintln!(
        "[sanity] allocated_leaves={} cursor={} needed_segments={} segments={}",
        system.allocated_leaves, cursor, needed_segments, system.segments,
    );
    if system.store.unauth.coded.shard_family_count() == 0 {
        eprintln!("[sanity] store_shard_keys=0");
    } else {
        eprintln!(
            "[sanity] store_shard_keys={}",
            system.store.unauth.coded.shard_family_count()
        );
        let mut keys: Vec<_> = system.store.unauth.coded.shard_keys().copied().collect();
        keys.sort_unstable();
        if let Some(key) = keys.first() {
            let seg_count = system
                .store
                .unauth
                .coded
                .shard_family_segment_count(key)
                .unwrap_or(0);
            let vec_len = system.params.s;
            eprintln!(
                "[sanity] store_sample_shard={:?} segments={} seg_vec_len={}",
                key, seg_count, vec_len,
            );
        }
    }

    let max_leaf = system.allocated_leaves.saturating_sub(1);
    let max_key = leaf_key(system.auth.tree_height as u32, max_leaf as u64);
    let max_idx = mapping::idx_if_active(
        system.node_linearization,
        height,
        system.allocated_leaves as u64,
        height,
        max_key.pos,
    )
    .unwrap_or(0);
    let max_seg = if cursor == 0 {
        0
    } else {
        max_idx / segment_len
    };
    eprintln!(
        "[sanity] alloc_max_leaf={} preorder_idx={} max_seg={}",
        max_leaf, max_idx, max_seg,
    );

    for &leaf in sample_leaves {
        if leaf >= system.allocated_leaves {
            eprintln!("[sanity] leaf={} skipped (unallocated)", leaf);
            continue;
        }
        eprintln!("[sanity] leaf={} path:", leaf);
        let mut key = leaf_key(system.auth.tree_height as u32, leaf as u64);
        loop {
            let idx = mapping::idx_if_active(
                system.node_linearization,
                height,
                system.allocated_leaves as u64,
                key.level as u16,
                key.pos,
            )
            .expect("inactive node in allocated prefix");
            let seg = idx / segment_len;
            let offset = idx % segment_len;
            let seg_base = seg * segment_len;
            eprintln!(
                "[sanity] level={} pos={} preorder_idx={} seg={} offset={} seg_base={}",
                key.level, key.pos, idx, seg, offset, seg_base,
            );
            if key.level == 0 {
                break;
            }
            key = parent(key);
        }
    }
}

/// Validate authentication-specific constraints and coded config consistency.
fn validate_auth_config(cfg: &Config) {
    assert!(cfg.auth.tree_height > 0);
    assert!(cfg.auth.alloc_leaf_batch > 0);
    assert!(cfg.auth.alloc_leaf_batch.is_power_of_two());
    assert!(matches!(
        cfg.auth.node_linearization,
        mapping::NodeLinearization::Preorder
    ));
    validate_coded_config(&cfg.coded);
    let total_nodes = conceptual_total_nodes_u128(cfg.auth.tree_height);
    assert!(cfg.coded.segment_len as u128 <= total_nodes.saturating_add(1));
    let max_leaves = conceptual_l_max_u128(cfg.auth.tree_height);
    assert!(cfg.auth.alloc_leaf_batch as u128 <= max_leaves);
}

/// Total nodes in a perfect binary tree of height H.
fn conceptual_total_nodes_u128(height: usize) -> u128 {
    let shift = height
        .checked_add(1)
        .expect("H too large for u128 node bound");
    let shift = u32::try_from(shift).expect("H too large for shift");
    let total = 1u128
        .checked_shl(shift)
        .expect("H too large for u128 node bound");
    total - 1
}

/// Conceptual max leaf count for a given height (2^H).
fn conceptual_l_max_u128(height: usize) -> u128 {
    let shift = u32::try_from(height).expect("H too large for shift");
    1u128
        .checked_shl(shift)
        .expect("H too large for u128 leaf bound")
}
