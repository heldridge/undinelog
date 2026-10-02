//! Crate entry points and public API surface.
//!
//! # Overview
//! This crate separates a coded polynomial storage layer from a sparse Merkle
//! authentication layer:
//! - The **coded array** layer stores segment polynomials and shard vectors.
//! - The **auth layer** maps Merkle nodes into those segments and produces
//!   authentication paths and broadcast updates.

mod fanout;
mod field;
mod shares;
mod value;

pub mod auth;
pub mod coded_array;
pub mod core;
pub mod dev_tools;
pub mod net;
pub mod unauth;

// Export the base field type for callers/benchmarks.
pub use crate::field::Fp;
// Export field helpers for benchmarks and library callers.
pub use crate::field::{batch_invert, fp_one, fp_pow, fp_zero};
// Export encoding helpers for 256-bit values.
pub use crate::value::{Value256, decode_value, encode_value, random_value};
// Re-export unauth share types used by transport adapters and tests.
pub use crate::shares::{RowShare, SegmentPrimaryShard, ShardShare};

// Re-export Merkle helpers from the authenticated layer.
pub use crate::auth::merkle::{DefaultHashes, NodeKey, build_default_hashes, default_hash};
// Compatibility module path for existing Merkle helper callers.
pub mod merkle {
    pub use crate::auth::merkle::*;
}

// Re-export coded-array parameters and registry helpers.
pub use crate::coded_array::{
    CodedArrayConfig, CodedArrayParams, CodedStore, DecodeCache, DeltaMode, PaperGeometry,
    PaperGeometryError, ReadError as CodedReadError, RowKey, SegmentAxes, SetupTimings, ShardKey,
    ShardRegEntry, accept_candidate, build_coded_params, build_shard_registry_deterministic,
    collect_shard_points, compute_initial_rows, compute_initial_shards, decode_segment_value,
    decode_segment_value_with_cache, derive_paper_geometry, sample_random_registry,
    validate_coded_config,
};

// Expose low-level coded-array helpers for benchmarks/tests.
pub use crate::coded_array::shard_maths::{
    apply_deltas_to_one_row, apply_deltas_to_one_shard, shard_value,
};

// Re-export authenticated system types and errors.
pub use crate::auth::{
    AllocateError, AuthBroadcastBatchUpdate, AuthBroadcastUpdate, AuthConfig, AuthParams, AuthPath,
    AuthStore, AuthSubtreeProof, BatchWriteError, BatchWriteOpts, Config, LeafUpdateBatch,
    MerkleTreeProof, ReadError, WriteError, apply_broadcast_update, apply_broadcast_update_batch,
    compute_deltas_for_write,
};
