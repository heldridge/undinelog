//! Authenticated sparse Merkle tree layer built over a coded array.
//!
//! This layer owns:
//! - Sparse Merkle node identity, hashing, and default labels.
//! - Preorder mapping and cursor-based append-only allocation.
//! - Translation from Merkle updates into unauth logical deltas.
//! - Broadcast-style update construction and store-local application.
//!
//! Transport adapters live in `crate::net::auth`.

pub mod allocation;
pub mod client;
mod core;
mod deltas;
pub mod mapping;
pub mod merkle;
pub mod read_path;
pub mod service;
pub mod write;

pub use core::{
    AllocateError, AuthBroadcastBatchUpdate, AuthBroadcastUpdate, AuthConfig, AuthParams, AuthPath,
    AuthRangeProof, AuthStore, AuthSubtreeProof, BatchWriteError, BatchWriteOpts, Config,
    LeafUpdateBatch, MerkleTreeProof, ReadError, WriteError,
};
pub use deltas::compute_deltas_for_write;
pub use mapping::NodeLinearization;
pub use read_path::verify_range_proof;
pub use write::{apply_broadcast_update, apply_broadcast_update_batch};
