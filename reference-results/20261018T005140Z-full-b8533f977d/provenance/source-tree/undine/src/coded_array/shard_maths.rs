//! Server-side shard-value and shard update logic.
//!
//! # Overview
//! This module implements the math that servers (parties) use to:
//! - Answer shard queries for cells of a segment.
//! - Evaluate a stored column or row off its axis, as repair helpers do.
//! - Apply per-cell deltas to the stored column and row vectors.
//!
//! # Mathematical model
//! Let `P(Z, W)` be the segment polynomial. A claim with column identifier `C`
//! and row label `w` stores
//!
//! ```text
//! y[j]   = P(C, eta_j)    primary column, on the segment's W-axis
//! row[i] = P(z_i, w)      auxiliary row, on the segment's Z-axis
//! ```
//!
//! The shard value for the cell at `(z_i, eta_j)` is the stored entry `y[j]`.
//! No arithmetic is needed to answer a point query.
//!
//! # Updates
//! A delta `d` at the cell `(z_i, eta_j)` changes the segment polynomial by
//! `d * Lambda^Z_i(Z) * Lambda^W_j(W)`. Because the column is stored on the
//! W-axis and the row on the Z-axis, exactly one entry of each changes:
//!
//! ```text
//! y[j]   += d * Lambda^Z_i(C)
//! row[i] += d * Lambda^W_j(w)
//! ```
//!
//! A batch needs the basis values only for the distinct axis indices it
//! touches. They are computed once per claim with a single batch inversion.

use crate::coded_array::axis::{coset_lagrange_basis_entries_at, dot};
use crate::coded_array::params::{CodedArrayParams, SegmentAxes, ShardRegEntry};
use crate::coded_array::store::CodedStore;
use crate::field::{Fp, fp_zero};
use crate::value::decode_value;

/// Strategy hint passed with batches of deltas.
///
/// In the bivariate layout every delta is applied in closed form, so both
/// variants take the same path and give the same result at the same cost. The
/// type is kept so that the delta-application signatures stay stable.
#[derive(Clone, Copy, Debug)]
pub enum DeltaMode {
    /// Hint used by callers with small batches.
    Aggregate,
    /// Hint used by callers with large batches.
    InterpolateFFT,
}

/// Distinct axis indices touched by one batch of deltas, with the slot of
/// each index in the compact basis vectors.
struct DeltaAxisPlan {
    /// Distinct Z-axis indices, ascending.
    z_indices: Vec<usize>,
    /// Slot of each Z-axis index in `z_indices`; unused indices are stale.
    z_slot: Vec<usize>,
    /// Distinct W-axis indices, ascending.
    w_indices: Vec<usize>,
    /// Slot of each W-axis index in `w_indices`; unused indices are stale.
    w_slot: Vec<usize>,
    /// Compact basis values for the claim being updated.
    basis: Vec<Fp>,
}

impl DeltaAxisPlan {
    fn new() -> Self {
        Self {
            z_indices: Vec::new(),
            z_slot: Vec::new(),
            w_indices: Vec::new(),
            w_slot: Vec::new(),
            basis: Vec::new(),
        }
    }

    /// Record the distinct axis indices of `deltas`.
    fn prepare(&mut self, params: &CodedArrayParams, deltas: &[(usize, Fp)]) {
        let need = params.need;
        self.z_indices.clear();
        self.w_indices.clear();
        self.z_slot.clear();
        self.z_slot.resize(need, usize::MAX);
        self.w_slot.clear();
        self.w_slot.resize(need, usize::MAX);
        for &(grid_index, _) in deltas {
            let z_index = params.z_index_for_grid_index(grid_index);
            let w_index = params.w_index_for_grid_index(grid_index);
            if self.z_slot[z_index] == usize::MAX {
                self.z_slot[z_index] = 0;
                self.z_indices.push(z_index);
            }
            if self.w_slot[w_index] == usize::MAX {
                self.w_slot[w_index] = 0;
                self.w_indices.push(w_index);
            }
        }
        self.z_indices.sort_unstable();
        self.w_indices.sort_unstable();
        for (slot, &z_index) in self.z_indices.iter().enumerate() {
            self.z_slot[z_index] = slot;
        }
        for (slot, &w_index) in self.w_indices.iter().enumerate() {
            self.w_slot[w_index] = slot;
        }
    }
}

// Thread-local plan to keep the delta path allocation-free across segments.
thread_local! {
    static DELTA_AXIS_PLAN: std::cell::RefCell<DeltaAxisPlan> =
        std::cell::RefCell::new(DeltaAxisPlan::new());
}

/// Return the shard value of one cell from a stored column: the entry at the
/// cell's W-axis index.
#[inline]
pub fn shard_value_from_vector(y: &[Fp], w_index: usize) -> Option<Fp> {
    y.get(w_index).copied()
}

/// Return shard values for a batch of W-axis indices from a stored column.
pub fn shard_value_from_vector_batch(y: &[Fp], w_indices: &[usize]) -> Vec<Option<Fp>> {
    w_indices
        .iter()
        .map(|&w_index| shard_value_from_vector(y, w_index))
        .collect()
}

/// Fetch a shard vector for a store/segment and return its value for the cells
/// at one W-axis index.
pub fn shard_value_from_store(
    params: &CodedArrayParams,
    store: &CodedStore,
    entry: ShardRegEntry,
    seg: usize,
    w_index: usize,
) -> Option<Fp> {
    let y = store.shard_segment(params, &entry.key, seg)?;
    shard_value_from_vector(y, w_index)
}

/// Fetch a shard vector for a store/segment and return its values for a batch
/// of W-axis indices.
pub fn shard_value_from_store_batch(
    params: &CodedArrayParams,
    store: &CodedStore,
    entry: ShardRegEntry,
    seg: usize,
    w_indices: &[usize],
) -> Vec<Option<Fp>> {
    let Some(y) = store.shard_segment(params, &entry.key, seg) else {
        return vec![None; w_indices.len()];
    };
    shard_value_from_vector_batch(y, w_indices)
}

/// Shard value for segment 0 only.
pub fn shard_value(
    params: &CodedArrayParams,
    store: &CodedStore,
    entry: ShardRegEntry,
    w_index: usize,
) -> Option<Fp> {
    shard_value_from_store(params, store, entry, 0, w_index)
}

/// Evaluate a stored primary column at an off-axis W point.
///
/// `basis_w[j] = Lambda^W_j(w)` for the segment's W-axis. The result is
/// `P(C, w)`, the value a helper contributes to the repair of the row labeled
/// `w`.
#[inline]
pub fn shard_value_at_row_label(y: &[Fp], basis_w: &[Fp]) -> Fp {
    dot(y, basis_w)
}

/// Evaluate a stored auxiliary row at an off-axis Z point.
///
/// `basis_z[i] = Lambda^Z_i(C)` for the segment's Z-axis. The result is
/// `P(C, w)`, the value a helper contributes to the repair of the column `C`.
#[inline]
pub fn row_value_at_column(row: &[Fp], basis_z: &[Fp]) -> Fp {
    dot(row, basis_z)
}

/// Apply per-cell deltas to a single primary column.
///
/// `deltas` are `(grid_index, delta)` pairs for one segment with axes `axes`.
///
/// # Mathematical definition
/// `y[w_index] += delta * Lambda^Z_{z_index}(C)` with `C = entry.z`.
///
/// # Complexity
/// `O(m)` multiplications and one batch inversion for `m = deltas.len()`.
pub fn apply_deltas_to_one_shard(
    params: &CodedArrayParams,
    axes: &SegmentAxes,
    entry: ShardRegEntry,
    y_vec: &mut [Fp],
    deltas: &[(usize, Fp)],
) {
    if deltas.is_empty() {
        return;
    }
    DELTA_AXIS_PLAN.with(|plan_cell| {
        let mut plan = plan_cell.borrow_mut();
        plan.prepare(params, deltas);
        apply_deltas_to_one_shard_with_plan(params, axes, entry, y_vec, deltas, &mut plan);
    });
}

fn apply_deltas_to_one_shard_with_plan(
    params: &CodedArrayParams,
    axes: &SegmentAxes,
    entry: ShardRegEntry,
    y_vec: &mut [Fp],
    deltas: &[(usize, Fp)],
    plan: &mut DeltaAxisPlan,
) {
    let DeltaAxisPlan {
        z_indices,
        z_slot,
        basis,
        ..
    } = plan;
    basis.clear();
    basis.resize(z_indices.len(), fp_zero());
    coset_lagrange_basis_entries_at(
        &params.axis,
        params.inv_need,
        axes.inv_rho_z,
        entry.z,
        z_indices,
        basis,
    );
    for &(grid_index, delta) in deltas {
        let z_index = params.z_index_for_grid_index(grid_index);
        let w_index = params.w_index_for_grid_index(grid_index);
        y_vec[w_index] += delta * basis[z_slot[z_index]];
    }
}

/// Apply per-cell deltas to a single auxiliary row.
///
/// # Mathematical definition
/// `row[z_index] += delta * Lambda^W_{w_index}(w)` with `w = row_label`.
///
/// # Complexity
/// `O(m)` multiplications and one batch inversion for `m = deltas.len()`.
pub fn apply_deltas_to_one_row(
    params: &CodedArrayParams,
    axes: &SegmentAxes,
    row_label: Fp,
    row_vec: &mut [Fp],
    deltas: &[(usize, Fp)],
) {
    if deltas.is_empty() {
        return;
    }
    DELTA_AXIS_PLAN.with(|plan_cell| {
        let mut plan = plan_cell.borrow_mut();
        plan.prepare(params, deltas);
        apply_deltas_to_one_row_with_plan(params, axes, row_label, row_vec, deltas, &mut plan);
    });
}

fn apply_deltas_to_one_row_with_plan(
    params: &CodedArrayParams,
    axes: &SegmentAxes,
    row_label: Fp,
    row_vec: &mut [Fp],
    deltas: &[(usize, Fp)],
    plan: &mut DeltaAxisPlan,
) {
    let DeltaAxisPlan {
        w_indices,
        w_slot,
        basis,
        ..
    } = plan;
    basis.clear();
    basis.resize(w_indices.len(), fp_zero());
    coset_lagrange_basis_entries_at(
        &params.axis,
        params.inv_need,
        axes.inv_rho_w,
        row_label,
        w_indices,
        basis,
    );
    for &(grid_index, delta) in deltas {
        let z_index = params.z_index_for_grid_index(grid_index);
        let w_index = params.w_index_for_grid_index(grid_index);
        row_vec[z_index] += delta * basis[w_slot[w_index]];
    }
}

/// Apply deltas for a specific segment across all primary columns held by the
/// store.
pub fn apply_deltas_to_segment(
    params: &CodedArrayParams,
    store: &mut CodedStore,
    seg: usize,
    deltas: &[(usize, Fp)],
) {
    apply_deltas_to_segment_with_mode(params, store, seg, deltas, DeltaMode::Aggregate);
}

/// Apply deltas for a specific segment across all primary columns held by the
/// store.
///
/// `deltas` are `(grid_index, delta)` pairs. The mode is a caller hint; see
/// [`DeltaMode`].
///
/// # Complexity
/// `O(m)` multiplications and one batch inversion per stored column.
pub fn apply_deltas_to_segment_with_mode(
    params: &CodedArrayParams,
    store: &mut CodedStore,
    seg: usize,
    deltas: &[(usize, Fp)],
    _mode: DeltaMode,
) {
    if deltas.is_empty() {
        return;
    }
    let axes = params.segment_axes(seg);
    DELTA_AXIS_PLAN.with(|plan_cell| {
        let mut plan = plan_cell.borrow_mut();
        plan.prepare(params, deltas);
        store.for_each_shard_segment_mut(params, seg, |shard_key, y_vec| {
            let entry_idx = params
                .shard_registry_by_key
                .get(shard_key)
                .copied()
                .expect("missing shard entry");
            let entry = params.shard_registry[entry_idx];
            apply_deltas_to_one_shard_with_plan(params, &axes, entry, y_vec, deltas, &mut plan);
        });
    });
}

/// Apply deltas for a specific segment across all auxiliary rows held by the
/// store.
///
/// `deltas` are `(grid_index, delta)` pairs.
///
/// # Complexity
/// `O(m)` multiplications and one batch inversion per stored row.
pub fn apply_row_deltas_to_segment(
    params: &CodedArrayParams,
    store: &mut CodedStore,
    seg: usize,
    deltas: &[(usize, Fp)],
) {
    if deltas.is_empty() {
        return;
    }
    let axes = params.segment_axes(seg);
    DELTA_AXIS_PLAN.with(|plan_cell| {
        let mut plan = plan_cell.borrow_mut();
        plan.prepare(params, deltas);
        store.for_each_row_segment_mut(params, seg, |row_key, row_vec| {
            let row_label = decode_value(row_key.row_label_bytes);
            apply_deltas_to_one_row_with_plan(params, &axes, row_label, row_vec, deltas, &mut plan);
        });
    });
}

#[cfg(test)]
mod tests {
    //! Unit tests cover shard-value and delta-application correctness.
    use super::*;
    use crate::coded_array::params::{
        CodedArrayConfig, build_coded_params, build_shard_registry_deterministic,
        compute_initial_rows, compute_initial_shards,
    };
    use crate::coded_array::store::RowKey;
    use crate::value::encode_value;
    use ark_ff::UniformRand;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    const SEGMENT: usize = 6;

    /// Build params and materialize one segment for a store holding every
    /// registry entry's column and row.
    fn setup_one_store(
        kappa: usize,
        shard_count: usize,
        seed: u64,
    ) -> (CodedArrayParams, CodedStore, Vec<crate::Value256>) {
        let cfg = CodedArrayConfig {
            kappa,
            shard_count,
            shard_registry: build_shard_registry_deterministic(shard_count),
            party_registry_indices: vec![(0..shard_count).collect()],
            seed,
        };
        let params = build_coded_params(&cfg);
        let mut rng = ChaCha20Rng::seed_from_u64(seed ^ 0xBEEF_BEEF);
        let t_seg: Vec<_> = (0..params.n_grid)
            .map(|_| encode_value(Fp::rand(&mut rng)))
            .collect();
        let store = encode_store(&params, &t_seg);
        (params, store, t_seg)
    }

    /// Encode a store whose only populated segment is `SEGMENT`.
    fn encode_store(params: &CodedArrayParams, t_seg: &[crate::Value256]) -> CodedStore {
        let mut store = CodedStore::new();
        let zero = vec![encode_value(fp_zero()); params.n_grid];
        for seg in 0..=SEGMENT {
            let values = if seg == SEGMENT { t_seg } else { &zero[..] };
            let shards = compute_initial_shards(params, seg, values, None);
            let rows = compute_initial_rows(params, seg, values);
            for (idx, entry) in params.shard_registry.iter().enumerate() {
                store.append_shard_segment(params, entry.key, seg, &shards[idx]);
                store.append_row_segment(
                    params,
                    RowKey::new(entry.key, params.row_labels[idx]),
                    seg,
                    &rows[idx],
                );
            }
        }
        store
    }

    #[test]
    // Applying deltas equals re-encoding the updated segment.
    fn apply_deltas_matches_reencoding() {
        for (kappa, delta_count) in [(4usize, 3usize), (8, 12), (8, 96)] {
            let (params, mut store, mut t_seg) = setup_one_store(kappa, 5, kappa as u64);
            let mut rng = ChaCha20Rng::seed_from_u64(12345 + delta_count as u64);
            let mut deltas = Vec::with_capacity(delta_count);
            for _ in 0..delta_count {
                let grid_index = rng.gen_range(0..params.n_grid);
                let delta = Fp::rand(&mut rng);
                t_seg[grid_index] = encode_value(decode_value(t_seg[grid_index]) + delta);
                deltas.push((grid_index, delta));
            }

            apply_deltas_to_segment_with_mode(
                &params,
                &mut store,
                SEGMENT,
                &deltas,
                DeltaMode::InterpolateFFT,
            );
            apply_row_deltas_to_segment(&params, &mut store, SEGMENT, &deltas);

            let expected = encode_store(&params, &t_seg);
            assert!(store.equals_for_tests(&params, &expected));
        }
    }

    #[test]
    // A point query returns the stored entry at the cell's W-axis index.
    fn shard_value_is_the_stored_entry() {
        let (params, store, _) = setup_one_store(8, 4, 3);
        for entry in params.shard_registry.iter().copied() {
            let y = store
                .shard_segment(&params, &entry.key, SEGMENT)
                .expect("missing segment");
            for grid_index in 0..params.n_grid {
                let w_index = params.w_index_for_grid_index(grid_index);
                assert_eq!(
                    shard_value_from_store(&params, &store, entry, SEGMENT, w_index),
                    Some(y[w_index])
                );
            }
            let w_indices: Vec<usize> = (0..params.need).rev().collect();
            let batch = shard_value_from_store_batch(&params, &store, entry, SEGMENT, &w_indices);
            for (reply, &w_index) in batch.iter().zip(w_indices.iter()) {
                assert_eq!(*reply, Some(y[w_index]));
            }
        }
    }

    #[test]
    // A column evaluated at a row label and a row evaluated at a column
    // identifier agree: both are P(C, w).
    fn column_at_row_label_equals_row_at_column() {
        let (params, store, _) = setup_one_store(8, 6, 21);
        let axes = params.segment_axes(SEGMENT);
        let mut basis_z = vec![fp_zero(); params.need];
        let mut basis_w = vec![fp_zero(); params.need];
        for (col_idx, col_entry) in params.shard_registry.iter().enumerate() {
            let column = store
                .shard_segment(&params, &col_entry.key, SEGMENT)
                .expect("missing column");
            params.z_basis_at(&axes, col_entry.z, &mut basis_z);
            for (row_idx, row_entry) in params.shard_registry.iter().enumerate() {
                let row_label = params.row_labels[row_idx];
                let row = store
                    .row_segment(&params, &RowKey::new(row_entry.key, row_label), SEGMENT)
                    .expect("missing row");
                params.w_basis_at(&axes, row_label, &mut basis_w);
                assert_eq!(
                    shard_value_at_row_label(column, &basis_w),
                    row_value_at_column(row, &basis_z),
                    "column {col_idx} row {row_idx}"
                );
            }
        }
    }
}
