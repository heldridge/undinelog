//! Single-store shard storage for the unauthenticated coded array.
//!
//! # Overview
//! The store maintains:
//! - A subset of primary shard vectors, keyed by `(a, z)`.
//! - A subset of maintained auxiliary row units, keyed by `(owning shard, row
//!   label)`.
//!
//! The store does **not** keep any authentication data. It only stores coded
//! shard vectors that allow clients to reconstruct array values from any
//! sufficiently large subset of shard entries.

use crate::coded_array::params::{CodedArrayParams, ShardKey};
use crate::field::Fp;
use crate::value::{Value256, encode_value};
use std::collections::HashMap;

/// Identity of one maintained auxiliary row attached to an owning shard claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RowKey {
    pub owner: ShardKey,
    pub row_label_bytes: Value256,
}

impl RowKey {
    /// Build a row key from the owning shard claim and public row label.
    pub fn new(owner: ShardKey, row_label: Fp) -> Self {
        Self {
            owner,
            row_label_bytes: encode_value(row_label),
        }
    }
}

/// Shard store: a map of flat per-family vectors by key.
///
/// # Invariants
/// - For each `ShardKey`, the stored count is the number of allocated segments
///   and the flat vector has length `count * kappa`: one primary column per
///   segment.
/// - For each `RowKey`, the stored count is the number of allocated segments
///   and the flat vector has length `count * kappa`: one auxiliary row per
///   segment.
#[derive(Clone)]
pub struct CodedStore {
    /// Map from shard key to `(segment_count, flat segment-major values)`.
    shards: HashMap<ShardKey, (usize, Vec<Fp>)>,
    /// Map from maintained row key to `(segment_count, flat segment-major values)`.
    rows: HashMap<RowKey, (usize, Vec<Fp>)>,
}

impl Default for CodedStore {
    fn default() -> Self {
        Self::new()
    }
}

impl CodedStore {
    fn segment_range(
        width: usize,
        count: usize,
        seg: usize,
        len: usize,
    ) -> Option<std::ops::Range<usize>> {
        debug_assert_eq!(len, count * width, "flat family width/count mismatch");
        if seg >= count {
            return None;
        }
        let start = seg * width;
        Some(start..start + width)
    }

    fn flatten_segments(width: usize, segments: Vec<Vec<Fp>>, context: &str) -> (usize, Vec<Fp>) {
        assert!(
            segments.iter().all(|segment| segment.len() == width),
            "{context}",
        );
        let count = segments.len();
        let mut flat = Vec::with_capacity(count * width);
        for segment in segments {
            flat.extend(segment);
        }
        (count, flat)
    }

    fn clone_segments(width: usize, count: usize, values: &[Fp]) -> Vec<Vec<Fp>> {
        debug_assert_eq!(
            values.len(),
            count * width,
            "flat family width/count mismatch",
        );
        values
            .chunks_exact(width)
            .take(count)
            .map(|segment| segment.to_vec())
            .collect()
    }

    /// Create an empty coded store.
    pub fn new() -> Self {
        Self {
            shards: HashMap::new(),
            rows: HashMap::new(),
        }
    }

    /// Create a store from a nested segment representation used by tests/setup.
    pub fn with_nested(
        params: &CodedArrayParams,
        shards: HashMap<ShardKey, Vec<Vec<Fp>>>,
        rows: HashMap<RowKey, Vec<Vec<Fp>>>,
    ) -> Self {
        let row_width = params.row_len();
        Self {
            shards: shards
                .into_iter()
                .map(|(key, segments)| {
                    (
                        key,
                        Self::flatten_segments(
                            params.s,
                            segments,
                            "primary shard segment width mismatch",
                        ),
                    )
                })
                .collect(),
            rows: rows
                .into_iter()
                .map(|(key, segments)| {
                    (
                        key,
                        Self::flatten_segments(
                            row_width,
                            segments,
                            "row-label segment width mismatch",
                        ),
                    )
                })
                .collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.shards.is_empty() && self.rows.is_empty()
    }

    pub fn shard_family_count(&self) -> usize {
        self.shards.len()
    }

    pub fn row_family_count(&self) -> usize {
        self.rows.len()
    }

    pub fn shard_family_segment_count(&self, key: &ShardKey) -> Option<usize> {
        Some(self.shards.get(key)?.0)
    }

    pub fn row_family_segment_count(&self, key: &RowKey) -> Option<usize> {
        Some(self.rows.get(key)?.0)
    }

    pub fn contains_shard(&self, key: &ShardKey) -> bool {
        self.shards.contains_key(key)
    }

    pub fn contains_row(&self, key: &RowKey) -> bool {
        self.rows.contains_key(key)
    }

    pub fn reserve_shard_segments(
        &mut self,
        params: &CodedArrayParams,
        key: ShardKey,
        additional_segments: usize,
    ) {
        if additional_segments == 0 {
            return;
        }
        let (_, flat) = self.shards.entry(key).or_insert_with(|| (0, Vec::new()));
        flat.reserve(additional_segments * params.s);
    }

    pub fn reserve_row_segments(
        &mut self,
        params: &CodedArrayParams,
        key: RowKey,
        additional_segments: usize,
    ) {
        if additional_segments == 0 {
            return;
        }
        let row_width = params.row_len();
        let (_, flat) = self.rows.entry(key).or_insert_with(|| (0, Vec::new()));
        flat.reserve(additional_segments * row_width);
    }

    pub fn shard_keys(&self) -> impl Iterator<Item = &ShardKey> {
        self.shards.keys()
    }

    pub fn row_keys(&self) -> impl Iterator<Item = &RowKey> {
        self.rows.keys()
    }

    pub fn shard_segment<'a>(
        &'a self,
        params: &CodedArrayParams,
        key: &ShardKey,
        seg: usize,
    ) -> Option<&'a [Fp]> {
        let (count, values) = self.shards.get(key)?;
        let range = Self::segment_range(params.s, *count, seg, values.len())?;
        Some(&values[range])
    }

    pub fn shard_segment_mut<'a>(
        &'a mut self,
        params: &CodedArrayParams,
        key: &ShardKey,
        seg: usize,
    ) -> Option<&'a mut [Fp]> {
        let (count, values) = self.shards.get_mut(key)?;
        let range = Self::segment_range(params.s, *count, seg, values.len())?;
        Some(&mut values[range])
    }

    pub fn row_segment<'a>(
        &'a self,
        params: &CodedArrayParams,
        key: &RowKey,
        seg: usize,
    ) -> Option<&'a [Fp]> {
        let row_width = params.row_len();
        let (count, values) = self.rows.get(key)?;
        let range = Self::segment_range(row_width, *count, seg, values.len())?;
        Some(&values[range])
    }

    pub fn row_segment_mut<'a>(
        &'a mut self,
        params: &CodedArrayParams,
        key: &RowKey,
        seg: usize,
    ) -> Option<&'a mut [Fp]> {
        let row_width = params.row_len();
        let (count, values) = self.rows.get_mut(key)?;
        let range = Self::segment_range(row_width, *count, seg, values.len())?;
        Some(&mut values[range])
    }

    pub fn append_shard_segment(
        &mut self,
        params: &CodedArrayParams,
        key: ShardKey,
        seg: usize,
        values: &[Fp],
    ) {
        assert_eq!(
            values.len(),
            params.s,
            "primary shard segment width mismatch"
        );
        let (count, flat) = self.shards.entry(key).or_insert_with(|| (0, Vec::new()));
        assert_eq!(
            *count, seg,
            "append-only allocation expects contiguous primary segments",
        );
        flat.extend_from_slice(values);
        *count += 1;
    }

    pub fn append_row_segment(
        &mut self,
        params: &CodedArrayParams,
        key: RowKey,
        seg: usize,
        values: &[Fp],
    ) {
        assert_eq!(
            values.len(),
            params.row_len(),
            "row-label segment width mismatch",
        );
        let (count, flat) = self.rows.entry(key).or_insert_with(|| (0, Vec::new()));
        assert_eq!(
            *count, seg,
            "append-only allocation expects contiguous row-label segments",
        );
        flat.extend_from_slice(values);
        *count += 1;
    }

    pub fn overwrite_shard_segment(
        &mut self,
        params: &CodedArrayParams,
        key: ShardKey,
        seg: usize,
        values: &[Fp],
    ) {
        assert_eq!(
            values.len(),
            params.s,
            "primary shard segment width mismatch"
        );
        let (count, flat) = self
            .shards
            .get_mut(&key)
            .expect("overwrite expects allocated primary segment");
        let range = Self::segment_range(params.s, *count, seg, flat.len())
            .expect("overwrite expects allocated primary segment");
        flat[range].copy_from_slice(values);
    }

    pub fn overwrite_row_segment(
        &mut self,
        params: &CodedArrayParams,
        key: RowKey,
        seg: usize,
        values: &[Fp],
    ) {
        assert_eq!(
            values.len(),
            params.row_len(),
            "row-label segment width mismatch",
        );
        let (count, flat) = self
            .rows
            .get_mut(&key)
            .expect("overwrite expects allocated row-label segment");
        let range = Self::segment_range(params.row_len(), *count, seg, flat.len())
            .expect("overwrite expects allocated row-label segment");
        flat[range].copy_from_slice(values);
    }

    pub fn insert_shard_family_segments(
        &mut self,
        params: &CodedArrayParams,
        key: ShardKey,
        segments: Vec<Vec<Fp>>,
    ) {
        self.shards.insert(
            key,
            Self::flatten_segments(params.s, segments, "primary shard segment width mismatch"),
        );
    }

    pub fn insert_row_family_segments(
        &mut self,
        params: &CodedArrayParams,
        key: RowKey,
        segments: Vec<Vec<Fp>>,
    ) {
        self.rows.insert(
            key,
            Self::flatten_segments(
                params.row_len(),
                segments,
                "row-label segment width mismatch",
            ),
        );
    }

    pub fn remove_shard_family_segments(
        &mut self,
        params: &CodedArrayParams,
        key: &ShardKey,
    ) -> Option<Vec<Vec<Fp>>> {
        let (count, values) = self.shards.remove(key)?;
        Some(Self::clone_segments(params.s, count, &values))
    }

    pub fn remove_row_family_segments(
        &mut self,
        params: &CodedArrayParams,
        key: &RowKey,
    ) -> Option<Vec<Vec<Fp>>> {
        let row_width = params.row_len();
        let (count, values) = self.rows.remove(key)?;
        Some(Self::clone_segments(row_width, count, &values))
    }

    pub fn clear_shards(&mut self) {
        self.shards.clear();
    }

    pub fn clear_rows(&mut self) {
        self.rows.clear();
    }

    pub fn retain_shards(&mut self, mut keep: impl FnMut(&ShardKey) -> bool) {
        self.shards.retain(|key, _| keep(key));
    }

    pub fn retain_rows(&mut self, mut keep: impl FnMut(&RowKey) -> bool) {
        self.rows.retain(|key, _| keep(key));
    }

    pub fn clone_shard_family_segments(
        &self,
        params: &CodedArrayParams,
        key: &ShardKey,
    ) -> Option<Vec<Vec<Fp>>> {
        let (count, values) = self.shards.get(key)?;
        Some(Self::clone_segments(params.s, *count, values))
    }

    pub fn clone_row_family_segments(
        &self,
        params: &CodedArrayParams,
        key: &RowKey,
    ) -> Option<Vec<Vec<Fp>>> {
        let row_width = params.row_len();
        let (count, values) = self.rows.get(key)?;
        Some(Self::clone_segments(row_width, *count, values))
    }

    pub fn for_each_shard_segment(
        &self,
        params: &CodedArrayParams,
        seg: usize,
        mut f: impl FnMut(&ShardKey, &[Fp]),
    ) {
        for (key, (count, values)) in &self.shards {
            let Some(range) = Self::segment_range(params.s, *count, seg, values.len()) else {
                continue;
            };
            f(key, &values[range]);
        }
    }

    pub fn shard_segments_for_segment<'a>(
        &'a self,
        params: &CodedArrayParams,
        seg: usize,
    ) -> Vec<(&'a ShardKey, &'a [Fp])> {
        let mut out = Vec::new();
        for (key, (count, values)) in &self.shards {
            let Some(range) = Self::segment_range(params.s, *count, seg, values.len()) else {
                continue;
            };
            out.push((key, &values[range]));
        }
        out
    }

    pub fn for_each_row_segment(
        &self,
        params: &CodedArrayParams,
        seg: usize,
        mut f: impl FnMut(&RowKey, &[Fp]),
    ) {
        let row_width = params.row_len();
        for (key, (count, values)) in &self.rows {
            let Some(range) = Self::segment_range(row_width, *count, seg, values.len()) else {
                continue;
            };
            f(key, &values[range]);
        }
    }

    pub fn row_segments_for_segment<'a>(
        &'a self,
        params: &CodedArrayParams,
        seg: usize,
    ) -> Vec<(&'a RowKey, &'a [Fp])> {
        let row_width = params.row_len();
        let mut out = Vec::new();
        for (key, (count, values)) in &self.rows {
            let Some(range) = Self::segment_range(row_width, *count, seg, values.len()) else {
                continue;
            };
            out.push((key, &values[range]));
        }
        out
    }

    pub fn for_each_shard_segment_mut(
        &mut self,
        params: &CodedArrayParams,
        seg: usize,
        mut f: impl FnMut(&ShardKey, &mut [Fp]),
    ) {
        for (key, (count, values)) in &mut self.shards {
            let Some(range) = Self::segment_range(params.s, *count, seg, values.len()) else {
                continue;
            };
            f(key, &mut values[range]);
        }
    }

    pub fn for_each_row_segment_mut(
        &mut self,
        params: &CodedArrayParams,
        seg: usize,
        mut f: impl FnMut(&RowKey, &mut [Fp]),
    ) {
        let row_width = params.row_len();
        for (key, (count, values)) in &mut self.rows {
            let Some(range) = Self::segment_range(row_width, *count, seg, values.len()) else {
                continue;
            };
            f(key, &mut values[range]);
        }
    }

    pub fn row_segments_mut_for_segment<'a>(
        &'a mut self,
        params: &CodedArrayParams,
        seg: usize,
    ) -> Vec<(&'a RowKey, &'a mut [Fp])> {
        let row_width = params.row_len();
        let mut out = Vec::new();
        for (key, (count, values)) in &mut self.rows {
            let Some(range) = Self::segment_range(row_width, *count, seg, values.len()) else {
                continue;
            };
            out.push((key, &mut values[range]));
        }
        out
    }

    pub fn as_nested_shards_for_tests(
        &self,
        params: &CodedArrayParams,
    ) -> HashMap<ShardKey, Vec<Vec<Fp>>> {
        self.shards
            .iter()
            .map(|(key, (count, values))| (*key, Self::clone_segments(params.s, *count, values)))
            .collect()
    }

    pub fn as_nested_rows_for_tests(
        &self,
        params: &CodedArrayParams,
    ) -> HashMap<RowKey, Vec<Vec<Fp>>> {
        let row_width = params.row_len();
        self.rows
            .iter()
            .map(|(key, (count, values))| (*key, Self::clone_segments(row_width, *count, values)))
            .collect()
    }

    pub fn equals_for_tests(&self, params: &CodedArrayParams, other: &Self) -> bool {
        self.as_nested_shards_for_tests(params) == other.as_nested_shards_for_tests(params)
            && self.as_nested_rows_for_tests(params) == other.as_nested_rows_for_tests(params)
    }
}
