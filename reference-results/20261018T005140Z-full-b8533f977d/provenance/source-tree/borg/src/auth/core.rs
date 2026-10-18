//! Authenticated sparse Merkle tree layer built over a coded array.
//!
//! This module defines core data types for the authenticated sparse
//! Merkle tree wrapper. Read helpers live in `read_path.rs` and
//! update/application helpers live in `write.rs`.

use super::mapping::NodeLinearization;
use super::merkle::{DefaultHashes, NodeKey, build_default_hashes};
use crate::field::Fp;
use crate::unauth::{CodedArrayConfig, DeltaMode, UnauthArray, UnauthStore};
use crate::value::Value256;

/// Authentication-specific configuration for the Merkle wrapper.
#[derive(Clone)]
pub struct AuthConfig {
    /// Global tree height (root at level 0, leaves at level `tree_height`).
    pub tree_height: usize,
    /// Leaves appended per allocation chunk.
    pub alloc_leaf_batch: usize,
    /// Mapping order used to linearize nodes.
    pub node_linearization: NodeLinearization,
}

/// Configuration provided by callers to build an authenticated system.
#[derive(Clone)]
pub struct Config {
    /// Authentication (Merkle) configuration.
    pub auth: AuthConfig,
    /// Unauthenticated coded-array configuration.
    pub coded: CodedArrayConfig,
}

/// Authentication-specific parameters (Merkle height + defaults).
#[derive(Clone)]
pub struct AuthParams {
    /// Global tree height.
    pub tree_height: usize,
    /// Default labels for each level.
    pub defaults: DefaultHashes,
}

impl AuthParams {
    /// Construct authentication params with default hashes for the given height.
    pub fn new(height: usize) -> Self {
        AuthParams {
            tree_height: height,
            defaults: build_default_hashes(height),
        }
    }

    /// Conceptual max leaf count for the configured global height.
    pub fn conceptual_l_max_u128(&self) -> u128 {
        conceptual_l_max_u128(self.tree_height)
    }
}

/// Store state for the authenticated layer.
#[derive(Clone)]
pub struct AuthStore<U: UnauthArray + Clone = UnauthStore> {
    /// Current Merkle root commitment.
    pub root: Value256,
    /// Unauthenticated shard storage.
    pub unauth: U,
}

/// Allocation failure when extending the active leaf range.
#[derive(Clone, Debug)]
pub enum AllocateError {
    NoCapacity,
}

/// Authentication path returned by a read.
#[derive(Clone, Debug)]
pub struct AuthPath {
    pub index: usize,
    pub siblings: Vec<Value256>,
    pub root: Value256,
}

/// Authentication proof for a full aligned subtree read.
///
/// `upper_siblings` starts at the subtree root's sibling and continues upward
/// toward the global root.
#[derive(Clone, Debug)]
pub struct AuthSubtreeProof {
    pub start_index: u64,
    pub leaf_count: usize,
    pub upper_siblings: Vec<Value256>,
    pub root: Value256,
}

/// Authentication proof for a contiguous leaf range.
///
/// `proof_nodes` contains the canonical boundary proof nodes plus internal
/// checkpoint ancestors of the returned leaves. Leaf proof node values are raw
/// leaf values; internal proof node values are stored Merkle labels.
#[derive(Clone, Debug)]
pub struct AuthRangeProof {
    pub start_index: u64,
    pub leaf_count: usize,
    pub allocated_leaves: u64,
    pub proof_nodes: Vec<(NodeKey, Value256)>,
    pub root: Value256,
}

/// Batched leaf update with old/new values.
#[derive(Clone, Debug)]
pub struct LeafUpdateBatch {
    pub index: usize,
    pub old: Value256,
    pub new: Value256,
}

/// Frontier multiproof against the old root.
#[derive(Clone, Debug)]
pub struct MerkleTreeProof {
    pub root: Value256,
    pub nodes: Vec<(NodeKey, Value256)>,
}

/// Errors returned by batch write operations.
#[derive(Clone, Debug)]
pub enum BatchWriteError {
    InvalidIndex,
    DuplicateIndex,
    RootMismatch,
    InvalidProof,
    ConflictingProofNode,
}

/// Options that influence how delta batches are applied.
#[derive(Clone, Copy, Debug)]
pub struct BatchWriteOpts {
    pub delta_mode: DeltaMode,
}

/// Errors returned by read operations.
#[derive(Clone, Debug)]
pub enum ReadError {
    InvalidIndex,
    NotEnoughShards,
    InvalidProof,
}

/// Errors returned by write operations.
#[derive(Clone, Debug)]
pub enum WriteError {
    InvalidIndex,
    InvalidPath,
    RootMismatch,
}

/// Broadcast update for a single-leaf write.
#[derive(Clone, Debug)]
pub struct AuthBroadcastUpdate {
    pub index: usize,
    pub old: Value256,
    pub new: Value256,
    pub path: AuthPath,
    pub allocated_leaves: usize,
    pub root_new: Value256,
    pub delta_mode: DeltaMode,
    pub deltas: Vec<(u64, Fp)>,
}

/// Broadcast update for a batch write.
#[derive(Clone, Debug)]
pub struct AuthBroadcastBatchUpdate {
    pub updates: Vec<LeafUpdateBatch>,
    pub proof: MerkleTreeProof,
    pub allocated_leaves: usize,
    pub root_new: Value256,
    pub delta_mode: DeltaMode,
    pub deltas: Vec<(u64, Fp)>,
}

/// Total nodes in a perfect binary tree of height H.
pub(super) fn conceptual_total_nodes_u128(height: usize) -> u128 {
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
