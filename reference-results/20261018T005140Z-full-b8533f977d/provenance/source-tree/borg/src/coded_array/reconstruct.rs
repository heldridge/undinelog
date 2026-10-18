//! Coded-array client helpers: shard-value collection and decode.
//!
//! This module gathers shard values from a local store and decodes the
//! requested value with Reed-Solomon interpolation. It has no knowledge of
//! Merkle trees or node keys.

use crate::coded_array::params::{CodedArrayParams, ShardKey, ShardRegEntry};
use crate::coded_array::rs::{BarycentricWeights, barycentric_eval, precompute_barycentric};
use crate::coded_array::shard_maths::shard_value_from_store;
use crate::coded_array::store::CodedStore;
use crate::field::Fp;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

/// Errors returned by coded-array reads.
#[derive(Clone, Debug)]
pub enum ReadError {
    NotEnoughShards,
}

/// Cached barycentric weights keyed by shard key sets.
#[derive(Default)]
pub struct DecodeCache {
    inner: RefCell<HashMap<Vec<ShardKey>, BarycentricWeights>>,
}

impl DecodeCache {
    /// Create an empty decode cache.
    pub fn new() -> Self {
        Self {
            inner: RefCell::new(HashMap::new()),
        }
    }

    pub(crate) fn get_weights_for_keys_and_z(
        &self,
        shard_keys: &[ShardKey],
        z_values: &[Fp],
    ) -> BarycentricWeights {
        debug_assert_eq!(shard_keys.len(), z_values.len());
        let mut cache = self.inner.borrow_mut();
        if let Some(weights) = cache.get(shard_keys) {
            return weights.clone();
        }
        let weights = precompute_barycentric(z_values);
        cache.insert(shard_keys.to_vec(), weights.clone());
        weights
    }
}

/// One shard value (`y`) paired with its registry entry.
#[derive(Clone, Debug)]
pub struct ShardPoint {
    /// Registry entry describing the shard's `(a, z)` parameters.
    pub entry: ShardRegEntry,
    /// Direct shard value at `alpha`.
    pub y: Fp,
}

/// Collect shard values for a specific segment/offset until `need` points exist.
///
/// Returns `alpha = x_j` and the decoded shard points.
pub fn collect_shard_points(
    params: &CodedArrayParams,
    store: &CodedStore,
    seg: usize,
    j: usize,
) -> Result<(Fp, Vec<ShardPoint>), ReadError> {
    let alpha = params.grid_point(j);
    let mut points = Vec::new();
    let mut seen = HashSet::new();
    for entry in params.shard_registry.iter().copied() {
        if !seen.insert(entry.key) {
            continue;
        }
        if let Some(y) = shard_value_from_store(params, store, entry, seg, alpha) {
            points.push(ShardPoint { entry, y });
            if points.len() >= params.need {
                break;
            }
        }
    }
    if points.len() < params.need {
        return Err(ReadError::NotEnoughShards);
    }
    Ok((alpha, points))
}

/// Decode a segment value at offset `j` without caching barycentric weights.
pub fn decode_segment_value(
    params: &CodedArrayParams,
    store: &CodedStore,
    seg: usize,
    j: usize,
) -> Result<Fp, ReadError> {
    decode_segment_value_with_cache(params, store, seg, j, None)
}

/// Decode a segment value at offset `j`, optionally using a weight cache.
pub fn decode_segment_value_with_cache(
    params: &CodedArrayParams,
    store: &CodedStore,
    seg: usize,
    j: usize,
    cache: Option<&DecodeCache>,
) -> Result<Fp, ReadError> {
    let (alpha, points) = collect_shard_points(params, store, seg, j)?;
    let z_eval = params.column_coordinate(alpha);
    let mut keys = Vec::with_capacity(points.len());
    let mut z_values = Vec::with_capacity(points.len());
    let mut r_values = Vec::with_capacity(points.len());
    for point in &points {
        keys.push(point.entry.key);
        z_values.push(point.entry.z);
        r_values.push(point.y);
    }
    let weights = match cache {
        Some(cache) => cache.get_weights_for_keys_and_z(&keys, &z_values),
        None => precompute_barycentric(&z_values),
    };
    Ok(barycentric_eval(&weights, &r_values, z_eval))
}
