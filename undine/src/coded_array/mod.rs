//! Unauthenticated coded array storage layer.
//!
//! This module exposes a polynomial-coded array abstraction that supports:
//! - Append-only segment allocation with default (zero) values.
//! - Store-local shard updates from per-segment deltas.
//! - Reads by decoding from enough shard replies.
//!
//! It intentionally has no notion of Merkle trees, node keys, or authentication.

pub mod axis;
pub mod params;
pub mod reconstruct;
pub mod rs;
pub mod segment;
pub mod shard_maths;
pub mod store;

pub use params::{
    CodedArrayConfig, CodedArrayParams, PaperGeometry, PaperGeometryError, SegmentAxes,
    SetupTimings, ShardKey, ShardRegEntry, accept_candidate, build_coded_params,
    build_shard_registry_deterministic, compute_initial_rows, compute_initial_shards,
    derive_paper_geometry, sample_random_registry, validate_coded_config,
};
pub use reconstruct::{
    DecodeCache, ReadError, collect_shard_points, decode_segment_value,
    decode_segment_value_with_cache,
};
pub use shard_maths::{
    DeltaMode, apply_deltas_to_one_row, apply_deltas_to_one_shard, apply_deltas_to_segment,
    apply_deltas_to_segment_with_mode, apply_row_deltas_to_segment, shard_value,
    shard_value_from_store, shard_value_from_store_batch, shard_value_from_vector_batch,
};
pub use store::{CodedStore, RowKey};
