//! Parameter generation, shard registry setup, and segment encoding for the
//! coded array layer.
//!
//! # Overview
//! This module builds all public parameters used by the coded array:
//! - Segment geometry derived from the reconstruction threshold `kappa`
//! - The normalized coset axis shared by every segment
//! - Shard registry values and the per-claim auxiliary row labels
//! - The canonical public-logical to grid layout
//!
//! It also encodes the initial shard and row vectors for a supplied segment.
//!
//! # Mathematical model
//! - A segment holds `kappa^2` cells. They are the values of a polynomial
//!   `P(Z, W)` of degree `< kappa` in each variable on the product of the
//!   segment's two coset axes (see `axis.rs`).
//! - A claim with column identifier `C` and row label `w` stores the primary
//!   column `P(C, eta_j)` on the W-axis and the auxiliary row `P(z_i, w)` on
//!   the Z-axis, `kappa` values each.
//! - Cells are addressed by a flat grid index
//!   `g = z_index * kappa + w_index`.
//!
//! # Preconditions / invariants
//! - `kappa` is a power of two.
//! - `segment_len = n_grid = kappa^2` and `s = need = kappa`. The segment size
//!   is derived from `kappa`; it is never configured on its own.
//! - Every column identifier and row label is non-zero and lies outside the
//!   two-power roots of unity, so it is off every semantic axis at every
//!   level.

use crate::coded_array::axis::{
    self, axis_scale_from_index, canonical_logical_to_grid_index as axis_canonical_layout,
    coset_lagrange_basis_at, is_valid_assignment, segment_axis_indices,
};
use crate::coded_array::segment::{encode_auxiliary_row, encode_primary_column};
use crate::field::{Fp, fp_one, fp_pow, fp_zero};
use crate::value::{Value256, decode_value, encode_value};
use ark_ff::{FftField, Field, UniformRand};
use rand::Rng;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

/// Canonical shard identity derived from encoded `a` and `z` values.
///
/// The key is serialized so it can be used as a stable map key and safely
/// compared across processes or machines.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShardKey {
    pub a_bytes: [u8; 32],
    pub z_bytes: [u8; 32],
}

impl PartialOrd for ShardKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ShardKey {
    /// Order by z then a for deterministic iteration.
    fn cmp(&self, other: &Self) -> Ordering {
        match self.z_bytes.cmp(&other.z_bytes) {
            Ordering::Equal => self.a_bytes.cmp(&other.a_bytes),
            ordering => ordering,
        }
    }
}

/// Registry entry pairing numeric values with canonical key bytes.
///
/// # Mathematical meaning
/// - `a` is the claim's representative in the field.
/// - `z` is the claim's primary column identifier `C`, the Reed-Solomon
///   evaluation point of its shares. In the bivariate layout the column
///   identifier is the representative itself, so `z = a`, and it is the same
///   at every threshold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShardRegEntry {
    pub a: Fp,
    pub z: Fp,
    pub key: ShardKey,
}

impl ShardRegEntry {
    /// Build the registry entry of the claim whose column identifier is `a`.
    pub fn for_column_identifier(a: Fp) -> Self {
        let bytes = encode_value(a);
        Self {
            a,
            z: a,
            key: ShardKey {
                a_bytes: bytes,
                z_bytes: bytes,
            },
        }
    }
}

/// Configuration provided by callers to build coded array parameters.
///
/// This is the external API callers use to select the threshold and shard
/// registry values. Validation happens in `validate_coded_config`.
#[derive(Clone)]
pub struct CodedArrayConfig {
    /// Reconstruction threshold.
    ///
    /// Paper notation: `\kappa`. The segment size is derived from it,
    /// `\Seg = \kappa^2`, and so is the column length, `s = \kappa`.
    pub kappa: usize,
    /// Registry size (`shard_count`).
    pub shard_count: usize,
    /// Shard registry entries.
    pub shard_registry: Vec<ShardRegEntry>,
    /// Indices into `shard_registry` for benchmark and fixture assignment groups.
    pub party_registry_indices: Vec<Vec<usize>>,
    /// Deterministic seed, used by benchmarks and fixtures.
    pub seed: u64,
}

/// Errors returned when deriving the canonical logical layout ladder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CanonicalLayoutError {
    ZeroBaseGrid,
    BaseGridNotSquareOfPowerOfTwo,
    CurrentGridSmallerThanBase,
    CurrentGridNotMultipleOfBase,
    CurrentGridNotPowerOfFourMultiple,
}

/// Auxiliary rows maintained by one claim.
pub const ROWS_PER_CLAIM: usize = 1;

/// Paper-facing geometry derived from the advertised threshold.
///
/// The paper advertises only the reconstruction threshold `kappa`. The
/// coded-array implementation uses derived construction quantities:
/// `segment_len`, `n_grid`, column length `s`, and `need`. This helper keeps
/// the bridge explicit without renaming the internal construction fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PaperGeometry {
    /// Advertised reconstruction threshold (`\kappa`).
    pub kappa: usize,
    /// Derived field elements per segment (`\Seg = \kappa^2`), implementation
    /// `segment_len`.
    pub segment_len: usize,
    /// Derived number of grid cells per segment, implementation `n_grid`.
    pub n_grid: usize,
    /// Derived primary column length, implementation `s`.
    pub shard_len: usize,
    /// Auxiliary rows attached to one primary claim. Always one.
    pub rows_per_claim: usize,
}

impl PaperGeometry {
    /// Implementation decode threshold corresponding to paper `\kappa`.
    pub fn need(&self) -> usize {
        self.kappa
    }

    /// Implementation column length.
    pub fn s(&self) -> usize {
        self.shard_len
    }
}

/// Errors returned when deriving implementation geometry from paper parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaperGeometryError {
    ZeroKappa,
    KappaNotPowerOfTwo,
    GridOverflow,
}

/// Derive coded-array geometry from the advertised threshold.
///
/// The construction uses:
///
/// - `segment_len = n_grid = kappa^2`;
/// - `s = need = kappa`;
/// - one auxiliary row per claim.
pub fn derive_paper_geometry(kappa: usize) -> Result<PaperGeometry, PaperGeometryError> {
    if kappa == 0 {
        return Err(PaperGeometryError::ZeroKappa);
    }
    if !kappa.is_power_of_two() {
        return Err(PaperGeometryError::KappaNotPowerOfTwo);
    }
    let n_grid = kappa
        .checked_mul(kappa)
        .ok_or(PaperGeometryError::GridOverflow)?;
    Ok(PaperGeometry {
        kappa,
        segment_len: n_grid,
        n_grid,
        shard_len: kappa,
        rows_per_claim: 1,
    })
}

/// Public axis scales of one segment, with their inverses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentAxes {
    /// Z-axis scale `rho_Z`.
    pub rho_z: Fp,
    /// W-axis scale `rho_W`.
    pub rho_w: Fp,
    /// `1 / rho_Z`.
    pub inv_rho_z: Fp,
    /// `1 / rho_W`.
    pub inv_rho_w: Fp,
}

/// Derived public parameters and precomputed tables.
///
/// Most fields are immutable and computed once during setup. They are
/// intentionally duplicated rather than recomputed at runtime to keep the math
/// explicit and to avoid accidental mismatches.
#[derive(Clone)]
pub struct CodedArrayParams {
    /// Values per segment.
    ///
    /// Paper notation: `\Seg = \kappa^2`.
    pub segment_len: usize,
    /// Number of grid cells per segment. Equals `segment_len`.
    pub n_grid: usize,
    /// Primary column length. Equals `need`.
    pub s: usize,
    /// Minimum number of shards needed to decode.
    ///
    /// Paper notation: `\kappa`.
    pub need: usize,
    /// Primitive `need`-th root of unity; generator of the normalized axis.
    pub omega: Fp,
    /// `1 / need`.
    pub inv_need: Fp,
    /// Normalized axis points `omega^i` for `0 <= i < need`.
    pub axis: Vec<Fp>,
    /// Roots of the segment scale ladder: entry `t` is `zeta_{2^{t+1} need}`.
    pub scale_ladder_roots: Vec<Fp>,
    /// Inverses of `scale_ladder_roots`.
    pub inv_scale_ladder_roots: Vec<Fp>,
    /// Cells per segment at the base level of the canonical layout ladder.
    pub layout_base_n_grid: usize,
    /// Number of four-segment stitches from `layout_base_n_grid` to `n_grid`.
    pub layout_level: usize,
    /// Map public logical offsets to canonical grid indices.
    pub logical_to_grid_index: Vec<usize>,
    /// Reverse map from canonical grid indices to public logical offsets.
    pub grid_index_to_logical_offset: Vec<usize>,
    /// Registry of shard entries.
    pub shard_registry: Vec<ShardRegEntry>,
    /// Map from shard key to its index in the registry.
    pub shard_registry_by_key: HashMap<ShardKey, usize>,
    /// Per-group indices into `shard_registry` for benchmark and fixture assignments.
    pub party_registry_indices: Vec<Vec<usize>>,
    /// Public auxiliary row labels, one per registry entry. The claim at
    /// registry index `i` maintains the row labeled `row_labels[i]`.
    pub row_labels: Vec<Fp>,
    /// Encoded row labels keyed by public row-label index.
    pub row_label_bytes: Vec<Value256>,
    /// Reverse lookup from encoded row label to public row-label index.
    pub row_label_index_by_bytes: HashMap<Value256, usize>,
}

impl CodedArrayParams {
    /// Apply the canonical layout ladder for a vector family whose base level
    /// has `base_n_grid` cells per segment.
    pub fn apply_canonical_logical_layout(
        &mut self,
        base_n_grid: usize,
    ) -> Result<(), CanonicalLayoutError> {
        let layout_level = canonical_layout_level(self.n_grid, base_n_grid)?;
        self.logical_to_grid_index = canonical_logical_to_grid_index(self.n_grid, base_n_grid)?;
        self.grid_index_to_logical_offset =
            inverse_logical_to_grid_index(&self.logical_to_grid_index);
        self.layout_base_n_grid = base_n_grid;
        self.layout_level = layout_level;
        Ok(())
    }

    /// Rebuild the lookup tables derived from `row_labels`.
    pub fn refresh_row_metadata_caches(&mut self) {
        assert_eq!(
            self.row_labels.len(),
            self.shard_registry.len(),
            "every claim maintains exactly one auxiliary row"
        );
        self.row_label_bytes = self.row_labels.iter().copied().map(encode_value).collect();
        self.row_label_index_by_bytes = self
            .row_label_bytes
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, bytes)| (bytes, idx))
            .collect();
    }

    pub fn row_label_index_for_bytes(&self, row_label_bytes: Value256) -> Option<usize> {
        self.row_label_index_by_bytes.get(&row_label_bytes).copied()
    }

    /// Return the row-label index maintained by one claimed column.
    ///
    /// Every claim maintains exactly one auxiliary row, and its label index is
    /// the claim's registry index.
    pub fn row_label_index_for_registry_index(&self, registry_index: usize) -> Option<usize> {
        (registry_index < self.row_labels.len()).then_some(registry_index)
    }

    /// Return the row-label indices attached to one claimed column.
    pub fn row_label_indices_for_registry_index(&self, registry_index: usize) -> Vec<usize> {
        self.row_label_index_for_registry_index(registry_index)
            .into_iter()
            .collect()
    }

    /// Return the Z-axis index of one grid cell.
    #[inline]
    pub fn z_index_for_grid_index(&self, grid_index: usize) -> usize {
        debug_assert!(grid_index < self.n_grid);
        grid_index / self.need
    }

    /// Return the W-axis index of one grid cell.
    #[inline]
    pub fn w_index_for_grid_index(&self, grid_index: usize) -> usize {
        debug_assert!(grid_index < self.n_grid);
        grid_index % self.need
    }

    /// Return the grid index of the cell at `(z_index, w_index)`.
    #[inline]
    pub fn grid_index_for_axes(&self, z_index: usize, w_index: usize) -> usize {
        debug_assert!(z_index < self.need && w_index < self.need);
        z_index * self.need + w_index
    }

    /// Return the canonical grid index for one public logical offset.
    pub fn logical_grid_index(&self, logical_offset: usize) -> usize {
        self.logical_to_grid_index[logical_offset]
    }

    /// Return the public logical offset for one canonical grid index.
    pub fn logical_offset_for_grid_index(&self, grid_index: usize) -> usize {
        self.grid_index_to_logical_offset[grid_index]
    }

    /// Return the public axis scales of one segment.
    pub fn segment_axes(&self, segment: usize) -> SegmentAxes {
        let (z_scale_index, w_scale_index) = segment_axis_indices(segment);
        SegmentAxes {
            rho_z: axis_scale_from_index(&self.scale_ladder_roots, z_scale_index),
            rho_w: axis_scale_from_index(&self.scale_ladder_roots, w_scale_index),
            inv_rho_z: axis_scale_from_index(&self.inv_scale_ladder_roots, z_scale_index),
            inv_rho_w: axis_scale_from_index(&self.inv_scale_ladder_roots, w_scale_index),
        }
    }

    /// Return the index of `x` on the normalized axis, if it is one of its
    /// `need` points.
    pub fn axis_index_of(&self, x: Fp) -> Option<usize> {
        self.axis.iter().position(|&point| point == x)
    }

    /// Return the two field points of one cell of one segment: its Z-axis
    /// point and its W-axis point.
    pub fn cell_points(&self, axes: &SegmentAxes, grid_index: usize) -> (Fp, Fp) {
        (
            axes.rho_z * self.axis[self.z_index_for_grid_index(grid_index)],
            axes.rho_w * self.axis[self.w_index_for_grid_index(grid_index)],
        )
    }

    /// Lagrange basis of one segment's Z-axis at `x`: `out[i] = Lambda^Z_i(x)`.
    pub fn z_basis_at(&self, axes: &SegmentAxes, x: Fp, out: &mut [Fp]) {
        coset_lagrange_basis_at(&self.axis, self.inv_need, axes.inv_rho_z, x, out);
    }

    /// Lagrange basis of one segment's W-axis at `x`: `out[j] = Lambda^W_j(x)`.
    pub fn w_basis_at(&self, axes: &SegmentAxes, x: Fp, out: &mut [Fp]) {
        coset_lagrange_basis_at(&self.axis, self.inv_need, axes.inv_rho_w, x, out);
    }

    /// Return the maximum degree of one maintained row polynomial.
    pub fn row_degree(&self) -> usize {
        self.need - 1
    }

    /// Return the number of stored values in one auxiliary row segment.
    pub fn row_len(&self) -> usize {
        self.need
    }
}

/// Timings captured during setup for benchmarking.
#[derive(Clone, Debug, Default)]
pub struct SetupTimings {
    /// Time spent building Merkle default hashes.
    pub tree_build: Duration,
    /// Time spent encoding primary columns from segment cells.
    pub shard_eval: Duration,
    /// Time spent encoding maintained auxiliary rows from segment cells.
    pub row_eval: Duration,
}

/// Log2 for power-of-two values.
fn log2_pow2(n: usize) -> usize {
    assert!(n.is_power_of_two());
    usize::BITS as usize - 1 - n.leading_zeros() as usize
}

/// Return `kappa` for a grid of `n_grid = kappa^2` cells, if `kappa` is a
/// power of two.
fn kappa_for_grid(n_grid: usize) -> Option<usize> {
    if n_grid == 0 || !n_grid.is_power_of_two() {
        return None;
    }
    let log = log2_pow2(n_grid);
    (log % 2 == 0).then_some(1usize << (log / 2))
}

/// Return the canonical layout level for `(base_n_grid, n_grid)`: the number
/// of four-segment stitches between them.
pub fn canonical_layout_level(
    n_grid: usize,
    base_n_grid: usize,
) -> Result<usize, CanonicalLayoutError> {
    if base_n_grid == 0 {
        return Err(CanonicalLayoutError::ZeroBaseGrid);
    }
    let base_kappa =
        kappa_for_grid(base_n_grid).ok_or(CanonicalLayoutError::BaseGridNotSquareOfPowerOfTwo)?;
    if n_grid < base_n_grid {
        return Err(CanonicalLayoutError::CurrentGridSmallerThanBase);
    }
    if n_grid % base_n_grid != 0 {
        return Err(CanonicalLayoutError::CurrentGridNotMultipleOfBase);
    }
    let kappa =
        kappa_for_grid(n_grid).ok_or(CanonicalLayoutError::CurrentGridNotPowerOfFourMultiple)?;
    Ok(log2_pow2(kappa / base_kappa))
}

/// Return the canonical public-logical to grid permutation for the
/// four-segment layout ladder.
///
/// See `axis::canonical_logical_to_grid_index` for the rule.
pub fn canonical_logical_to_grid_index(
    n_grid: usize,
    base_n_grid: usize,
) -> Result<Vec<usize>, CanonicalLayoutError> {
    canonical_layout_level(n_grid, base_n_grid)?;
    let kappa = kappa_for_grid(n_grid).expect("validated grid size");
    let base_kappa = kappa_for_grid(base_n_grid).expect("validated base grid size");
    Ok(axis_canonical_layout(kappa, base_kappa))
}

/// Build the inverse of a known permutation from logical offsets to grid
/// indices.
pub fn inverse_logical_to_grid_index(logical_to_grid_index: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0usize; logical_to_grid_index.len()];
    for (logical_offset, &grid_index) in logical_to_grid_index.iter().enumerate() {
        inverse[grid_index] = logical_offset;
    }
    inverse
}

/// Bit-reversal permutation helper for FFT.
///
/// # Definition
/// Reverses the lowest `bits` of `value`, used to permute indices for
/// iterative radix-2 FFT.
fn bit_reverse(mut value: usize, bits: usize) -> usize {
    let mut rev = 0usize;
    for _ in 0..bits {
        rev = (rev << 1) | (value & 1);
        value >>= 1;
    }
    rev
}

/// In-place radix-2 FFT.
///
/// # Mathematical definition
/// Given coefficients of `A`, this evaluates `A` at `x_j = root^j` over the
/// subgroup generated by `root`.
///
/// # Preconditions
/// - `values.len()` is a power of two.
/// - `root` is a primitive `n`-th root of unity.
///
/// # Complexity
/// `O(n log n)` field operations.
pub fn fft_in_place(values: &mut [Fp], root: Fp) {
    let n = values.len();
    if n <= 1 {
        return;
    }
    assert!(n.is_power_of_two());
    let log_n = log2_pow2(n);
    for i in 0..n {
        let j = bit_reverse(i, log_n);
        if j > i {
            values.swap(i, j);
        }
    }
    let mut m = 2;
    while m <= n {
        let step = n / m;
        let w_m = fp_pow(root, step as u64);
        for k in (0..n).step_by(m) {
            let mut w = fp_one();
            for j in 0..(m / 2) {
                let t = w * values[k + j + (m / 2)];
                let u = values[k + j];
                values[k + j] = u + t;
                values[k + j + (m / 2)] = u - t;
                w *= w_m;
            }
        }
        m *= 2;
    }
}

/// In-place inverse FFT using the provided root of unity.
///
/// # Mathematical definition
/// If `values[j] = A(x_j)` for `x_j = root^j`, then the inverse FFT recovers
/// the coefficients of `A` (scaled by `1/n`).
///
/// # Preconditions
/// - `root` is a primitive `n`-th root of unity.
/// - `n` is a power of two.
pub fn inverse_fft_in_place(values: &mut [Fp], root: Fp) {
    let n = values.len();
    if n <= 1 {
        return;
    }
    let inv_root = root.inverse().unwrap();
    fft_in_place(values, inv_root);
    let inv_n = Fp::from(n as u64).inverse().unwrap();
    for v in values.iter_mut() {
        *v *= inv_n;
    }
}

/// First candidate scanned for auxiliary row labels.
///
/// Column identifiers are scanned from 2 upward. Row labels start far above
/// any realistic registry size, so the two public sequences do not interleave.
const ROW_LABEL_SCAN_START: u64 = 1 << 40;

/// Deterministically build one auxiliary row label per registry entry.
///
/// Labels are distinct and valid assignments: non-zero and outside the
/// two-power roots of unity, hence off every semantic W-axis at every level.
pub fn build_registry_row_labels(total: usize) -> Vec<Fp> {
    let mut labels = Vec::with_capacity(total);
    let mut cand = ROW_LABEL_SCAN_START;
    while labels.len() < total {
        let label = Fp::from(cand);
        if is_valid_assignment(label) {
            labels.push(label);
        }
        cand += 1;
    }
    labels
}

/// Decode one segment's public values into field cells, in grid order.
pub(crate) fn segment_cells_from_values(params: &CodedArrayParams, t_full: &[Value256]) -> Vec<Fp> {
    let mut cells = vec![fp_zero(); params.n_grid];
    for (cell, value) in cells.iter_mut().zip(t_full.iter()) {
        *cell = decode_value(*value);
    }
    cells
}

/// Encode the primary columns of selected claims from one segment's cells.
///
/// Returns `(registry_index, column)` pairs in ascending registry order.
///
/// # Complexity
/// `O(kappa^2)` field multiplications per claim.
pub(crate) fn compute_selected_shards_from_cells(
    params: &CodedArrayParams,
    axes: &SegmentAxes,
    cells: &[Fp],
    registry_indices: &[usize],
    timings: Option<&mut SetupTimings>,
) -> Vec<(usize, Vec<Fp>)> {
    if registry_indices.is_empty() {
        return Vec::new();
    }
    let start = timings.as_ref().map(|_| Instant::now());
    let mut selected = registry_indices.to_vec();
    selected.sort_unstable();
    selected.dedup();
    let mut basis_z = vec![fp_zero(); params.need];
    let mut shards = Vec::with_capacity(selected.len());
    for registry_index in selected {
        let entry = params.shard_registry[registry_index];
        params.z_basis_at(axes, entry.z, &mut basis_z);
        let mut column = vec![fp_zero(); params.s];
        encode_primary_column(cells, &basis_z, &mut column);
        shards.push((registry_index, column));
    }
    if let (Some(start), Some(timing)) = (start, timings) {
        timing.shard_eval += start.elapsed();
    }
    shards
}

/// Encode the auxiliary rows with the selected labels from one segment's
/// cells.
///
/// Returns `(row_label_index, row)` pairs in ascending label order.
///
/// # Complexity
/// `O(kappa^2)` field multiplications per row.
pub(crate) fn compute_selected_rows_from_cells(
    params: &CodedArrayParams,
    axes: &SegmentAxes,
    cells: &[Fp],
    row_label_indices: &[usize],
    timings: Option<&mut SetupTimings>,
) -> Vec<(usize, Vec<Fp>)> {
    if row_label_indices.is_empty() {
        return Vec::new();
    }
    let start = timings.as_ref().map(|_| Instant::now());
    let mut selected = row_label_indices.to_vec();
    selected.sort_unstable();
    selected.dedup();
    let mut basis_w = vec![fp_zero(); params.need];
    let mut rows = Vec::with_capacity(selected.len());
    for row_label_index in selected {
        assert!(
            row_label_index < params.row_labels.len(),
            "row label index {} out of range for {} labels",
            row_label_index,
            params.row_labels.len()
        );
        params.w_basis_at(axes, params.row_labels[row_label_index], &mut basis_w);
        let mut row = vec![fp_zero(); params.row_len()];
        encode_auxiliary_row(cells, &basis_w, &mut row);
        rows.push((row_label_index, row));
    }
    if let (Some(start), Some(timing)) = (start, timings) {
        timing.row_eval += start.elapsed();
    }
    rows
}

/// Encode the primary columns of every registry entry from one segment's
/// cells, in registry order.
pub(crate) fn compute_initial_shards_from_cells(
    params: &CodedArrayParams,
    axes: &SegmentAxes,
    cells: &[Fp],
    timings: Option<&mut SetupTimings>,
) -> Vec<Vec<Fp>> {
    let all: Vec<usize> = (0..params.shard_registry.len()).collect();
    compute_selected_shards_from_cells(params, axes, cells, &all, timings)
        .into_iter()
        .map(|(_, column)| column)
        .collect()
}

/// Encode the auxiliary rows of every registry entry from one segment's cells,
/// in row-label order.
pub(crate) fn compute_initial_rows_from_cells(
    params: &CodedArrayParams,
    axes: &SegmentAxes,
    cells: &[Fp],
    timings: Option<&mut SetupTimings>,
) -> Vec<Vec<Fp>> {
    let all: Vec<usize> = (0..params.row_labels.len()).collect();
    compute_selected_rows_from_cells(params, axes, cells, &all, timings)
        .into_iter()
        .map(|(_, row)| row)
        .collect()
}

/// Encode the primary columns of every registry entry for one segment.
///
/// `t_full` holds the segment's public values in grid order.
pub fn compute_initial_shards(
    params: &CodedArrayParams,
    segment: usize,
    t_full: &[Value256],
    timings: Option<&mut SetupTimings>,
) -> Vec<Vec<Fp>> {
    let cells = segment_cells_from_values(params, t_full);
    compute_initial_shards_from_cells(params, &params.segment_axes(segment), &cells, timings)
}

/// Encode the maintained auxiliary rows of every registry entry for one
/// segment.
///
/// `t_full` holds the segment's public values in grid order.
pub fn compute_initial_rows(
    params: &CodedArrayParams,
    segment: usize,
    t_full: &[Value256],
) -> Vec<Vec<Fp>> {
    let cells = segment_cells_from_values(params, t_full);
    compute_initial_rows_from_cells(params, &params.segment_axes(segment), &cells, None)
}

/// Validate a candidate column identifier and return its entry if acceptable.
///
/// # Preconditions enforced
/// - `a` is a valid assignment: non-zero and outside the two-power roots of
///   unity, so it is off every semantic Z-axis at every level
/// - `a` is unique within the registry
pub fn accept_candidate(a: Fp, seen_z: &HashSet<Value256>) -> Option<ShardRegEntry> {
    if !is_valid_assignment(a) {
        return None;
    }
    let entry = ShardRegEntry::for_column_identifier(a);
    if seen_z.contains(&entry.key.z_bytes) {
        return None;
    }
    Some(entry)
}

/// Deterministically build a shard registry by scanning field elements.
///
/// This is used for reproducible tests and benchmarks. The registry does not
/// depend on the threshold: a claim keeps its column identifier at every
/// level.
pub fn build_shard_registry_deterministic(target: usize) -> Vec<ShardRegEntry> {
    let mut shards = Vec::with_capacity(target);
    let mut seen_z: HashSet<Value256> = HashSet::new();
    let mut cand: u64 = 2;
    while shards.len() < target {
        let a = Fp::from(cand);
        if let Some(shard) = accept_candidate(a, &seen_z) {
            seen_z.insert(shard.key.z_bytes);
            shards.push(shard);
        }
        cand += 1;
    }
    shards
}

/// Sample a random shard registry.
///
/// Repeats until the target count is reached, rejecting invalid entries.
pub fn sample_random_registry<R: Rng + ?Sized>(rng: &mut R, target: usize) -> Vec<ShardRegEntry> {
    let mut shards = Vec::with_capacity(target);
    let mut seen_z: HashSet<Value256> = HashSet::new();
    while shards.len() < target {
        let a = Fp::rand(rng);
        if let Some(shard) = accept_candidate(a, &seen_z) {
            seen_z.insert(shard.key.z_bytes);
            shards.push(shard);
        }
    }
    shards
}

/// Validate configuration constraints and registry consistency.
///
/// # Invariants checked
/// - Power-of-two threshold within the field's two-adic root depth
/// - Registry key consistency, valid assignments, and unique identifiers
/// - Registry index groups are in range and unique per group
pub fn validate_coded_config(cfg: &CodedArrayConfig) {
    assert!(cfg.kappa > 0);
    assert!(cfg.kappa.is_power_of_two());
    assert!(cfg.shard_count > 0);
    assert_eq!(cfg.shard_count, cfg.shard_registry.len());
    let two_adicity = <Fp as FftField>::TWO_ADICITY as usize;
    assert!(log2_pow2(cfg.kappa) < two_adicity);
    let mut seen_z = HashSet::new();
    for entry in &cfg.shard_registry {
        assert_eq!(entry.key.a_bytes, encode_value(entry.a));
        assert_eq!(entry.key.z_bytes, encode_value(entry.z));
        assert_eq!(entry.z, entry.a);
        assert!(is_valid_assignment(entry.z));
        assert!(seen_z.insert(entry.key.z_bytes));
    }
    let registry_len = cfg.shard_registry.len();
    for indices in &cfg.party_registry_indices {
        let mut seen = HashSet::new();
        for &idx in indices {
            assert!(idx < registry_len);
            assert!(seen.insert(idx));
        }
    }
}

/// Build coded array parameters from a validated configuration.
///
/// # Algorithm outline
/// 1) Validate the threshold and shard registry constraints.
/// 2) Derive the segment geometry and the normalized axis.
/// 3) Derive one auxiliary row label per registry entry.
/// 4) Precompute registry and row-label lookup tables.
pub fn build_coded_params(cfg: &CodedArrayConfig) -> CodedArrayParams {
    validate_coded_config(cfg);
    let geometry = derive_paper_geometry(cfg.kappa).expect("validated threshold");
    let need = geometry.need();
    let n_grid = geometry.n_grid;

    let omega = axis::root_of_unity(need);
    let inv_need = Fp::from(need as u64).inverse().unwrap();
    let scale_ladder_roots = axis::scale_ladder_roots(need);
    let logical_to_grid_index = (0..n_grid).collect::<Vec<_>>();
    let grid_index_to_logical_offset = (0..n_grid).collect::<Vec<_>>();

    let mut params = CodedArrayParams {
        segment_len: geometry.segment_len,
        n_grid,
        s: geometry.s(),
        need,
        omega,
        inv_need,
        axis: axis::axis_points(need),
        scale_ladder_roots: scale_ladder_roots.clone(),
        inv_scale_ladder_roots: {
            let mut inverses = scale_ladder_roots;
            crate::field::batch_invert(&mut inverses);
            inverses
        },
        layout_base_n_grid: n_grid,
        layout_level: 0,
        logical_to_grid_index,
        grid_index_to_logical_offset,
        shard_registry: cfg.shard_registry.clone(),
        shard_registry_by_key: HashMap::new(),
        party_registry_indices: cfg.party_registry_indices.clone(),
        row_labels: build_registry_row_labels(cfg.shard_registry.len()),
        row_label_bytes: Vec::new(),
        row_label_index_by_bytes: HashMap::new(),
    };

    params.shard_registry_by_key = params
        .shard_registry
        .iter()
        .enumerate()
        .map(|(idx, entry)| (entry.key, idx))
        .collect();
    params.refresh_row_metadata_caches();

    params
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::encode_value;

    fn params_for(kappa: usize, shard_count: usize) -> CodedArrayParams {
        let cfg = CodedArrayConfig {
            kappa,
            shard_count,
            shard_registry: build_shard_registry_deterministic(shard_count),
            party_registry_indices: vec![(0..shard_count).collect()],
            seed: 7,
        };
        build_coded_params(&cfg)
    }

    fn deterministic_segment(params: &CodedArrayParams) -> Vec<crate::Value256> {
        (0..params.n_grid)
            .map(|idx| {
                let x = Fp::from((idx as u64 + 3) * 17);
                encode_value(x * x + Fp::from(idx as u64 + 11))
            })
            .collect()
    }

    #[test]
    fn geometry_is_derived_from_kappa() {
        for kappa in [1usize, 2, 4, 8, 16] {
            let geometry = derive_paper_geometry(kappa).expect("geometry");
            assert_eq!(geometry.segment_len, kappa * kappa);
            assert_eq!(geometry.n_grid, kappa * kappa);
            assert_eq!(geometry.s(), kappa);
            assert_eq!(geometry.need(), kappa);
            assert_eq!(geometry.rows_per_claim, 1);
        }
        assert_eq!(derive_paper_geometry(0), Err(PaperGeometryError::ZeroKappa));
        assert_eq!(
            derive_paper_geometry(6),
            Err(PaperGeometryError::KappaNotPowerOfTwo)
        );
    }

    #[test]
    fn canonical_layout_level_counts_four_segment_stitches() {
        assert_eq!(canonical_layout_level(16, 16), Ok(0));
        assert_eq!(canonical_layout_level(64, 16), Ok(1));
        assert_eq!(canonical_layout_level(256, 16), Ok(2));
        assert_eq!(
            canonical_layout_level(32, 16),
            Err(CanonicalLayoutError::CurrentGridNotPowerOfFourMultiple)
        );
        assert_eq!(
            canonical_layout_level(16, 8),
            Err(CanonicalLayoutError::BaseGridNotSquareOfPowerOfTwo)
        );
        assert_eq!(
            canonical_layout_level(4, 16),
            Err(CanonicalLayoutError::CurrentGridSmallerThanBase)
        );
        assert_eq!(
            canonical_layout_level(16, 0),
            Err(CanonicalLayoutError::ZeroBaseGrid)
        );
    }

    #[test]
    fn row_labels_are_valid_distinct_and_one_per_claim() {
        let params = params_for(4, 9);
        assert_eq!(params.row_labels.len(), params.shard_registry.len());
        let mut seen = HashSet::new();
        for (idx, label) in params.row_labels.iter().enumerate() {
            assert!(is_valid_assignment(*label));
            assert!(seen.insert(encode_value(*label)));
            assert_eq!(params.row_label_indices_for_registry_index(idx), vec![idx]);
        }
    }

    #[test]
    fn selected_encoding_matches_full_encoding() {
        let params = params_for(4, 6);
        let segment = 5usize;
        let axes = params.segment_axes(segment);
        let t_seg = deterministic_segment(&params);
        let cells = segment_cells_from_values(&params, &t_seg);

        let full_shards = compute_initial_shards(&params, segment, &t_seg, None);
        let full_rows = compute_initial_rows(&params, segment, &t_seg);
        let selected = [4usize, 1, 4];
        let shards = compute_selected_shards_from_cells(&params, &axes, &cells, &selected, None);
        let rows = compute_selected_rows_from_cells(&params, &axes, &cells, &selected, None);

        assert_eq!(shards.len(), 2);
        assert_eq!(rows.len(), 2);
        for (registry_index, column) in shards {
            assert_eq!(column, full_shards[registry_index]);
        }
        for (row_label_index, row) in rows {
            assert_eq!(row, full_rows[row_label_index]);
        }
    }

    #[test]
    fn encoded_column_interpolates_to_the_cells() {
        // Columns of kappa distinct claims determine every cell of the segment.
        let kappa = 4usize;
        let params = params_for(kappa, kappa);
        let segment = 2usize;
        let axes = params.segment_axes(segment);
        let t_seg = deterministic_segment(&params);
        let cells = segment_cells_from_values(&params, &t_seg);
        let shards = compute_initial_shards(&params, segment, &t_seg, None);
        let z: Vec<Fp> = params.shard_registry.iter().map(|entry| entry.z).collect();
        let weights = crate::coded_array::rs::precompute_barycentric(&z);
        for grid_index in 0..params.n_grid {
            let w_index = params.w_index_for_grid_index(grid_index);
            let y: Vec<Fp> = shards.iter().map(|column| column[w_index]).collect();
            let (cell_z, _) = params.cell_points(&axes, grid_index);
            assert_eq!(
                crate::coded_array::rs::barycentric_eval(&weights, &y, cell_z),
                cells[grid_index]
            );
        }
    }
}
