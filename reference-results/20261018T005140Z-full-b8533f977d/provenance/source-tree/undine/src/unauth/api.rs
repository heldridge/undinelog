//! Minimal unauthenticated array interface used by higher layers.
//!
//! This trait is meant to be the narrow surface that the auth layer depends
//! on instead of the coded-array internals.

use super::{
    CorrectedShardDecode, Delta, DeltaMode, ExtendError, Index, ReadError, SetupTimings,
    ShardRegEntry, Value, WriteError,
};
use crate::field::{Fp, fp_zero};
use crate::shares::ShardShare;
use crate::unauth::{
    CodedArrayParams, CodedReadError, UnauthStore, read_batch_from_store, read_batch_from_store_raw,
};
use crate::value::encode_value;
use std::ops::Range;

/// Opaque key for a repeated logical value family.
///
/// Higher layers may choose any stable logical classification. The unauth
/// layer only uses this as a cache key for materialized coded state.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LogicalValueFamilyKey(Vec<u16>);

impl LogicalValueFamilyKey {
    pub fn from_codes(codes: Vec<u16>) -> Self {
        Self(codes)
    }

    pub fn codes(&self) -> &[u16] {
        &self.0
    }
}

/// Minimal array parameter interface needed by higher layers.
pub trait UnauthArrayParams {
    fn segment_len(&self) -> usize;
    fn n_grid(&self) -> usize;
    fn public_values_per_segment(&self) -> usize {
        self.n_grid()
    }
    fn logical_offset_to_grid_index(&self, logical_offset: usize) -> usize {
        logical_offset
    }

    fn zero_segment_values(&self) -> Vec<Value> {
        vec![encode_value(fp_zero()); self.n_grid()]
    }

    fn materialize_logical_segment<F>(&self, mut value_at_logical_offset: F) -> Vec<Value>
    where
        F: FnMut(usize) -> Value,
    {
        let mut segment = self.zero_segment_values();
        for logical_offset in 0..self.public_values_per_segment() {
            let grid_index = self.logical_offset_to_grid_index(logical_offset);
            segment[grid_index] = value_at_logical_offset(logical_offset);
        }
        segment
    }
}

/// Minimal decoding interface for reconstructing values from shard replies.
pub trait UnauthDecodeParams {
    /// Return the column coordinate of one public array index: the Z-axis
    /// point of its cell in its segment.
    fn decode_index_eval(&self, idx: u64) -> Fp;
    fn need(&self) -> usize;
    fn shard_registry(&self) -> &[ShardRegEntry];

    fn decode_index_from_shares(
        &self,
        idx: u64,
        shares: &[ShardShare],
    ) -> Result<Fp, CodedReadError>
    where
        Self: Sized,
    {
        crate::unauth::decode_from_shares_at_z(self, self.decode_index_eval(idx), shares)
    }

    fn decode_index_from_shares_with_error_correction(
        &self,
        idx: u64,
        shares: &[ShardShare],
        max_errors: usize,
    ) -> Result<CorrectedShardDecode, CodedReadError>
    where
        Self: Sized,
    {
        crate::unauth::decode_from_shares_at_z_with_error_correction(
            self,
            self.decode_index_eval(idx),
            shares,
            max_errors,
        )
    }
}

impl UnauthArrayParams for CodedArrayParams {
    fn segment_len(&self) -> usize {
        self.segment_len
    }

    fn n_grid(&self) -> usize {
        self.n_grid
    }

    fn logical_offset_to_grid_index(&self, logical_offset: usize) -> usize {
        crate::unauth::logical_offset_to_grid_index(self, logical_offset)
    }
}

impl UnauthDecodeParams for CodedArrayParams {
    fn decode_index_eval(&self, idx: u64) -> Fp {
        let (segment, logical_offset) = crate::unauth::public_index_to_segment(self, idx);
        crate::unauth::logical_offset_to_column_coordinate_for_segment(
            self,
            segment,
            logical_offset,
        )
    }

    fn need(&self) -> usize {
        self.need
    }

    fn shard_registry(&self) -> &[ShardRegEntry] {
        &self.shard_registry
    }
}

/// Minimal array-like interface for the unauthenticated layer.
pub trait UnauthArray {
    type Params: UnauthArrayParams;

    fn params(&self) -> &Self::Params;
    fn len(&self) -> Index;
    fn read(&self, i: Index) -> Result<Value, CodedReadError>;
    fn read_batch(&self, idxs: &[Index]) -> Result<Vec<Value>, ReadError>;
    fn read_batch_raw(&self, idxs: &[Index]) -> Result<Vec<Value>, ReadError> {
        self.read_batch(idxs)
    }

    fn extend(&mut self, new_len: Index) -> Result<(), ExtendError>;
    fn extend_with_values<F>(
        &mut self,
        new_len: Index,
        value_at_index: F,
    ) -> Result<(), ExtendError>
    where
        F: FnMut(Index) -> Value;
    fn extend_with_values_timed<F>(
        &mut self,
        new_len: Index,
        value_at_index: F,
        timings: Option<&mut SetupTimings>,
    ) -> Result<(), ExtendError>
    where
        F: FnMut(Index) -> Value;
    fn extend_with_value_families_timed<FK, FV>(
        &mut self,
        new_len: Index,
        family_key_for_range: FK,
        logical_values_for_range: FV,
        timings: Option<&mut SetupTimings>,
    ) -> Result<(), ExtendError>
    where
        FK: FnMut(Range<Index>) -> LogicalValueFamilyKey,
        FV: FnMut(Range<Index>, &LogicalValueFamilyKey) -> Vec<Value>;

    fn apply_deltas(&mut self, deltas: &[Delta], mode: DeltaMode) -> Result<(), WriteError>;
    fn add_range_deltas(&mut self, start: Index, deltas: &[Value]) -> Result<(), WriteError>;
    fn write_fresh_range_from_defaults<F>(
        &mut self,
        start: Index,
        values: &[Value],
        mode: DeltaMode,
        default_at_index: F,
    ) -> Result<(), WriteError>
    where
        F: FnMut(Index) -> Value;
    fn overwrite_range(&mut self, start: Index, values: &[Value]) -> Result<(), WriteError>;
    fn write_delta(&mut self, i: Index, delta: Fp) -> Result<(), WriteError>;
    fn apply_deltas_raw(&mut self, deltas: &[Delta], mode: DeltaMode) -> Result<(), WriteError> {
        self.apply_deltas(deltas, mode)
    }
    fn write_delta_raw(&mut self, i: Index, delta: Fp) -> Result<(), WriteError> {
        self.write_delta(i, delta)
    }
}

impl UnauthArray for UnauthStore {
    type Params = CodedArrayParams;

    fn params(&self) -> &Self::Params {
        &self.params
    }

    fn len(&self) -> Index {
        self.len()
    }

    fn read(&self, i: Index) -> Result<Value, CodedReadError> {
        UnauthStore::read(self, i)
    }

    fn read_batch(&self, idxs: &[Index]) -> Result<Vec<Value>, ReadError> {
        let mut out = vec![encode_value(fp_zero()); idxs.len()];
        let mut active_positions = Vec::new();
        let mut active_idxs = Vec::new();
        for (pos, &idx) in idxs.iter().enumerate() {
            if idx < self.len() {
                active_positions.push(pos);
                active_idxs.push(idx);
            }
        }
        let values = read_batch_from_store(&self.params, &self.coded, &active_idxs, None)?;
        for (pos, value) in active_positions.into_iter().zip(values.into_iter()) {
            out[pos] = value;
        }
        Ok(out)
    }

    fn read_batch_raw(&self, idxs: &[Index]) -> Result<Vec<Value>, ReadError> {
        let mut out = vec![encode_value(fp_zero()); idxs.len()];
        let span = crate::unauth::public_values_per_segment(&self.params) as u64;
        let mut active_positions = Vec::new();
        let mut active_idxs = Vec::new();
        for (pos, &idx) in idxs.iter().enumerate() {
            let seg = usize::try_from(idx / span).map_err(|_| ReadError::InvalidIndex)?;
            if seg < self.segments() {
                active_positions.push(pos);
                active_idxs.push(idx);
            }
        }
        let values = read_batch_from_store_raw(&self.params, &self.coded, &active_idxs, None)?;
        for (pos, value) in active_positions.into_iter().zip(values.into_iter()) {
            out[pos] = value;
        }
        Ok(out)
    }

    fn extend(&mut self, new_len: Index) -> Result<(), ExtendError> {
        UnauthStore::extend(self, new_len)
    }

    fn extend_with_values<F>(
        &mut self,
        new_len: Index,
        value_at_index: F,
    ) -> Result<(), ExtendError>
    where
        F: FnMut(Index) -> Value,
    {
        UnauthStore::extend_with_values(self, new_len, value_at_index)
    }

    fn extend_with_values_timed<F>(
        &mut self,
        new_len: Index,
        value_at_index: F,
        timings: Option<&mut SetupTimings>,
    ) -> Result<(), ExtendError>
    where
        F: FnMut(Index) -> Value,
    {
        UnauthStore::extend_with_values_timed(self, new_len, value_at_index, timings)
    }

    fn extend_with_value_families_timed<FK, FV>(
        &mut self,
        new_len: Index,
        family_key_for_range: FK,
        logical_values_for_range: FV,
        timings: Option<&mut SetupTimings>,
    ) -> Result<(), ExtendError>
    where
        FK: FnMut(Range<Index>) -> LogicalValueFamilyKey,
        FV: FnMut(Range<Index>, &LogicalValueFamilyKey) -> Vec<Value>,
    {
        UnauthStore::extend_with_value_families_timed(
            self,
            new_len,
            family_key_for_range,
            logical_values_for_range,
            timings,
        )
    }

    fn apply_deltas(&mut self, deltas: &[Delta], mode: DeltaMode) -> Result<(), WriteError> {
        UnauthStore::apply_deltas(self, deltas, mode)
    }

    fn add_range_deltas(&mut self, start: Index, deltas: &[Value]) -> Result<(), WriteError> {
        UnauthStore::add_range_deltas(self, start, deltas)
    }

    fn write_fresh_range_from_defaults<F>(
        &mut self,
        start: Index,
        values: &[Value],
        mode: DeltaMode,
        default_at_index: F,
    ) -> Result<(), WriteError>
    where
        F: FnMut(Index) -> Value,
    {
        UnauthStore::write_fresh_range_from_defaults(self, start, values, mode, default_at_index)
    }

    fn overwrite_range(&mut self, start: Index, values: &[Value]) -> Result<(), WriteError> {
        UnauthStore::overwrite_range(self, start, values)
    }

    fn write_delta(&mut self, i: Index, delta: Fp) -> Result<(), WriteError> {
        UnauthStore::write_delta(self, i, delta)
    }

    fn apply_deltas_raw(&mut self, deltas: &[Delta], mode: DeltaMode) -> Result<(), WriteError> {
        UnauthStore::apply_deltas_raw(self, deltas, mode)
    }

    fn write_delta_raw(&mut self, i: Index, delta: Fp) -> Result<(), WriteError> {
        UnauthStore::write_delta_raw(self, i, delta)
    }
}
