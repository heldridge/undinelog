//! Parameter generation and shard registry setup for the coded array layer.
//!
//! # Overview
//! This module builds all public parameters used by the coded array:
//! - Segment sizing (`segment_len`, `N_grid`)
//! - Reed-Solomon evaluation grid (`x_j`, `w_j`) and FFT roots
//! - Shard registry values `(a, z)` plus precomputed denominators
//!
//! It also constructs the initial shard vectors for a supplied segment.
//!
//! # Mathematical model
//! - We treat a segment as a polynomial `A(X)` of degree `< N_grid` where
//!   `N_grid` is a power-of-two FFT size (also the polynomial degree bound).
//! - Segment values `t_j` are evaluations of `A` at `x_j = omega^j` (roots of
//!   unity). The public segment capacity, FFT grid size, and polynomial degree
//!   bound are the same value.
//! - For each shard entry `(a, z)` with `z = a^s`, we evaluate `A` at
//!   `beta_t = a * h_t` where `h_t = omega^{t * (N_grid/s)}` are the `s`
//!   coset points used to assemble shard replies.
//!
//! # Preconditions / invariants
//! - `s` is a power of two.
//! - `N_grid = segment_len`.
//! - `s | N_grid` (since both are powers of two).
//! - For each shard entry, `a != 0` and `a^N_grid != 1`, so `k(a) = a^N_grid - 1`
//!   is non-zero.

use crate::coded_array::rs::{BarycentricWeights, precompute_barycentric};
use crate::field::batch_invert;
use crate::field::{Fp, fp_one, fp_pow, fp_zero};
use crate::value::{Value256, decode_value, encode_value};
use ark_ff::{FftField, Field, UniformRand};
use rand::Rng;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
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
/// - `a` is a shard identifier in the field.
/// - `z = a^s` is the RS evaluation point for the shard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShardRegEntry {
    pub a: Fp,
    pub z: Fp,
    pub key: ShardKey,
}

/// k(a) = a^N_grid - 1, used in shard evaluation.
///
/// # Why it appears
/// For the vanishing polynomial `P(X) = X^N_grid - 1` over the FFT grid,
/// `P(a)` appears in the barycentric interpolation identities used to compute
/// shard replies and updates.
pub fn k(a: Fp, n_grid: usize) -> Fp {
    fp_pow(a, n_grid as u64) - fp_one()
}

/// z(a) = a^s, used as the RS evaluation point.
///
/// # Preconditions
/// `s` must divide `N_grid` and be a power of two so that the `h_t` points are
/// well-defined for FFT-based evaluation.
pub fn z(a: Fp, s: usize) -> Fp {
    fp_pow(a, s as u64)
}

/// Configuration provided by callers to build coded array parameters.
///
/// This is the external API callers use to select sizes and shard registry
/// values. Validation happens in `validate_coded_config`.
#[derive(Clone)]
pub struct CodedArrayConfig {
    /// Values per segment (polynomial capacity).
    ///
    /// Paper notation: this is the implementation alias for `\Seg`.
    pub segment_len: usize,
    /// Shard vector length / number of coset points (power of two).
    ///
    /// Paper notation: this is the implementation alias for
    /// `\ShardLen = \Seg / \kappa`.
    pub s: usize,
    /// Registry size (`shard_count`).
    pub shard_count: usize,
    /// Shard registry entries (a, z).
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
    BaseGridNotPowerOfTwo,
    CurrentGridSmallerThanBase,
    CurrentGridNotMultipleOfBase,
    CurrentGridNotPowerOfTwoMultiple,
}

/// Paper-facing geometry derived from advertised EDVC parameters.
///
/// The paper advertises a reconstruction threshold `kappa` and an encoded
/// segment capacity `\Seg` in field elements. The coded-array implementation
/// uses derived construction quantities: `segment_len`, `n_grid`, shard length
/// `s`, and `need`. This helper keeps the bridge explicit without renaming the
/// internal construction fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PaperGeometry {
    /// Advertised reconstruction threshold (`\kappa`).
    pub kappa: usize,
    /// Advertised field elements per segment (`\Seg`), implementation `segment_len`.
    pub segment_len: usize,
    /// Derived evaluation-grid size (`N = \Seg`), implementation `n_grid`.
    pub n_grid: usize,
    /// Derived primary shard vector length, implementation `s`.
    pub shard_len: usize,
    /// Derived maintained-row budget per primary claim.
    pub rows_per_claim: usize,
}

impl PaperGeometry {
    /// Implementation decode threshold corresponding to paper `\kappa`.
    pub fn need(&self) -> usize {
        self.kappa
    }

    /// Implementation shard length corresponding to paper `\ShardLen`.
    pub fn s(&self) -> usize {
        self.shard_len
    }
}

/// Errors returned when deriving implementation geometry from paper parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaperGeometryError {
    ZeroKappa,
    ZeroSegmentLength,
    GridOverflow,
    NonPowerOfTwoGridSize,
    KappaDoesNotDivideGrid,
}

/// Derive coded-array geometry from paper parameters.
///
/// The paper-facing construction uses:
///
/// - `segment_len = \Seg`;
/// - `n_grid = segment_len`;
/// - `s = n_grid / kappa`;
/// - `need = kappa`;
/// - `rows_per_claim = max(1, ceil(s / need))`.
///
/// This is the paper-facing regime used by the operating points: the public
/// segment capacity, FFT grid size, and polynomial degree bound are all the
/// same power-of-two value.
pub fn derive_paper_geometry(
    segment_len: usize,
    kappa: usize,
) -> Result<PaperGeometry, PaperGeometryError> {
    if kappa == 0 {
        return Err(PaperGeometryError::ZeroKappa);
    }
    if segment_len == 0 {
        return Err(PaperGeometryError::ZeroSegmentLength);
    }
    let n_grid = segment_len;
    if !n_grid.is_power_of_two() {
        return Err(PaperGeometryError::NonPowerOfTwoGridSize);
    }
    if n_grid % kappa != 0 {
        return Err(PaperGeometryError::KappaDoesNotDivideGrid);
    }
    let shard_len = n_grid / kappa;
    let rows_per_claim = usize::max(1, shard_len.div_ceil(kappa));
    Ok(PaperGeometry {
        kappa,
        segment_len,
        n_grid,
        shard_len,
        rows_per_claim,
    })
}

/// Derived public parameters and precomputed tables.
///
/// Most fields are immutable and computed once during setup. They are
/// intentionally duplicated rather than recomputed at runtime to keep the math
/// explicit and to avoid accidental mismatches.
#[derive(Clone)]
pub struct CodedArrayParams {
    /// Values per segment (polynomial capacity).
    ///
    /// Paper notation: `\Seg`.
    pub segment_len: usize,
    /// FFT size / polynomial degree bound for segment representation.
    ///
    /// Paper notation in the paper-facing regime: `N = \Seg`.
    pub n_grid: usize,
    /// Shard vector length (number of coset points).
    ///
    /// Paper notation: `\ShardLen = N / \kappa`.
    pub s: usize,
    /// `delta = (N_grid - 1) / s` (Reed-Solomon error tolerance).
    pub delta: usize,
    /// Minimum number of shards needed to decode (`delta + 1`).
    ///
    /// Paper notation: this is `\kappa` in the paper-facing regime.
    pub need: usize,
    /// Primitive `N_grid`-th root of unity.
    pub omega: Fp,
    /// `1 / N_grid`.
    pub inv_n_grid: Fp,
    /// FFT grid points `x_j = omega^j`.
    pub x: Vec<Fp>,
    /// Base grid size for the canonical factor-two layout ladder.
    pub layout_base_n_grid: usize,
    /// Number of factor-two joins from `layout_base_n_grid` to `n_grid`.
    pub layout_level: usize,
    /// Map public logical offsets to canonical FFT-grid indices.
    pub logical_to_grid_index: Vec<usize>,
    /// Reverse map from canonical FFT-grid indices to public logical offsets.
    pub grid_index_to_logical_offset: Vec<usize>,
    /// Reverse lookup from encoded FFT-grid point to raw grid index.
    pub grid_index_by_x_bytes: HashMap<Value256, usize>,
    /// Barycentric weights `w_j = 1 / P'(x_j)` with `P(X)=X^N_grid-1`.
    pub w: Vec<Fp>,
    /// Coset points `h_t = omega^{t * (N_grid / s)}`.
    pub h: Vec<Fp>,
    /// Cached barycentric weights for the fixed shard-side subgroup `h_t`.
    pub h_bary_weights: BarycentricWeights,
    /// Registry of shard entries `(a, z)`.
    pub shard_registry: Vec<ShardRegEntry>,
    /// Map from shard key to its index in the registry.
    pub shard_registry_by_key: HashMap<ShardKey, usize>,
    /// Precomputed `1 / a` for each shard representative.
    pub inv_a: Vec<Fp>,
    /// Precomputed `1 / (s * a^{s-1})` for each shard, if defined.
    pub inv_denom: Vec<Option<Fp>>,
    /// Per-group indices into `shard_registry` for benchmark and fixture assignments.
    pub party_registry_indices: Vec<Vec<usize>>,
    /// Default maintained-row budget per claimed primary column.
    ///
    /// Paper notation:
    /// `\RowsPerClaim = max(1, ceil(\ShardLen / \kappa))`.
    pub rows_per_shard: usize,
    /// Public maintained row labels `W_s`.
    pub row_labels: Vec<Fp>,
    /// Total size of the maintained public row domain.
    pub row_domain_size: usize,
    /// Number of row labels in one structured row family.
    pub row_family_size: usize,
    /// Number of addressable exact row families in the current mode.
    pub row_family_count: usize,
    /// Exact row-label indices for each structured row family.
    pub row_family_row_label_indices: Arc<Vec<Box<[usize]>>>,
    /// Reverse lookup from row-label index to `(family_id, offset_in_family)`.
    pub row_family_location_by_row_label_index: Vec<Option<(RowFamilyId, usize)>>,
    /// Reverse lookup from one exact family row-index set to its family id.
    pub row_family_id_by_indices: HashMap<Vec<usize>, RowFamilyId>,
    /// Bases of the structured row-label cosets when the row domain is
    /// organized as full `omega`-cosets of size `n_grid`.
    pub row_label_coset_bases: Vec<Fp>,
    /// Encoded row labels keyed by public row-label index.
    pub row_label_bytes: Vec<Value256>,
    /// Reverse lookup from encoded row label to public row-label index.
    pub row_label_index_by_bytes: HashMap<Value256, usize>,
    /// Public off-curve row evaluation set `Z_s`.
    pub row_eval_points: Vec<Fp>,
    /// Cached barycentric weights for the fixed public row-evaluation set.
    pub row_eval_bary_weights: BarycentricWeights,
    /// Lookup for the induced `Z`-subgroup generated by `omega^s`.
    pub induced_z_subgroup_index_by_bytes: HashMap<Value256, usize>,
    /// Cached row-update basis vectors, indexed by induced column-coordinate
    /// class. There are only `need = N_grid / s` distinct bases, one for each
    /// induced subgroup element `A = x^s`, so this cache must scale with
    /// `need`, not with the raw FFT-grid size `n_grid`.
    pub row_update_basis_by_subgroup_index: Arc<Vec<Vec<Fp>>>,
}

impl CodedArrayParams {
    /// Apply the canonical layout ladder for a vector family with base grid
    /// `base_n_grid`.
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

    pub fn refresh_row_metadata_caches(&mut self) {
        self.row_domain_size = self.row_labels.len();
        self.row_family_size = self.rows_per_shard;
        self.row_family_count = if self.rows_per_shard == 0 {
            0
        } else {
            assert_eq!(
                self.row_labels.len(),
                self.shard_registry
                    .len()
                    .saturating_mul(self.rows_per_shard),
                "registry-sized row domain must match shard registry times rows_per_shard"
            );
            self.shard_registry.len()
        };
        let mut row_family_row_label_indices = Vec::with_capacity(self.row_family_count);
        let mut row_family_location_by_row_label_index = vec![None; self.row_labels.len()];
        let mut row_family_id_by_indices = HashMap::with_capacity(self.row_family_count);
        for family_slot in 0..self.row_family_count {
            let indices = build_row_family_row_label_indices(self, RowFamilyId(family_slot));
            let family_id = RowFamilyId(family_slot);
            for (offset, &row_label_index) in indices.iter().enumerate() {
                row_family_location_by_row_label_index[row_label_index] = Some((family_id, offset));
            }
            row_family_id_by_indices.insert(indices.clone(), family_id);
            row_family_row_label_indices.push(indices.into_boxed_slice());
        }
        self.row_family_row_label_indices = Arc::new(row_family_row_label_indices);
        self.row_family_location_by_row_label_index = row_family_location_by_row_label_index;
        self.row_family_id_by_indices = row_family_id_by_indices;
        self.row_label_coset_bases = detect_structured_row_label_coset_bases(
            &self.row_labels,
            row_label_root(self.omega, self.n_grid, self.s),
            usize::max(1, self.s),
        );

        let mut row_label_bytes = Vec::with_capacity(self.row_labels.len());
        for row_label in self.row_labels.iter().copied() {
            row_label_bytes.push(encode_value(row_label));
        }
        self.row_label_bytes = row_label_bytes;

        let mut row_label_index_by_bytes = HashMap::with_capacity(self.row_labels.len());
        for (idx, bytes) in self.row_label_bytes.iter().copied().enumerate() {
            row_label_index_by_bytes.insert(bytes, idx);
        }
        self.row_label_index_by_bytes = row_label_index_by_bytes;

        self.row_eval_bary_weights = precompute_barycentric(&self.row_eval_points);

        let subgroup_size = self.induced_z_grid_size();
        let z_root = fp_pow(self.omega, self.s as u64);
        let mut induced_lookup = HashMap::with_capacity(subgroup_size);
        let mut point = fp_one();
        for idx in 0..subgroup_size {
            induced_lookup.insert(encode_value(point), idx);
            point *= z_root;
        }
        self.induced_z_subgroup_index_by_bytes = induced_lookup;

        let subgroup_size = self.induced_z_grid_size();
        let mut row_update_basis_by_subgroup_index = Vec::with_capacity(subgroup_size);
        for subgroup_index in 0..subgroup_size {
            let mut basis = vec![fp_zero(); self.row_eval_points.len()];
            row_update_basis_for_grid_index(self, subgroup_index, &mut basis);
            row_update_basis_by_subgroup_index.push(basis);
        }
        self.row_update_basis_by_subgroup_index = Arc::new(row_update_basis_by_subgroup_index);
    }

    pub fn row_label_index_for_bytes(&self, row_label_bytes: Value256) -> Option<usize> {
        self.row_label_index_by_bytes.get(&row_label_bytes).copied()
    }

    pub fn row_family_id_for_registry_index(&self, registry_index: usize) -> Option<RowFamilyId> {
        (registry_index < self.row_family_count).then_some(RowFamilyId(registry_index))
    }

    pub fn row_label_indices_for_row_family(&self, family_id: RowFamilyId) -> Option<Vec<usize>> {
        self.row_label_indices_for_row_family_slice(family_id)
            .map(|indices| indices.to_vec())
    }

    pub fn row_label_indices_for_row_family_slice(
        &self,
        family_id: RowFamilyId,
    ) -> Option<&[usize]> {
        self.row_family_row_label_indices
            .get(family_id.0)
            .map(|indices| indices.as_ref())
    }

    pub fn row_family_id_for_row_label_indices(
        &self,
        row_label_indices: &[usize],
    ) -> Option<RowFamilyId> {
        let mut normalized = row_label_indices.to_vec();
        normalized.sort_unstable();
        normalized.dedup();
        if normalized.len() != self.row_family_size {
            return None;
        }
        self.row_family_id_by_indices.get(&normalized).copied()
    }

    pub fn row_family_location_for_row_label_index(
        &self,
        row_label_index: usize,
    ) -> Option<(RowFamilyId, usize)> {
        self.row_family_location_by_row_label_index
            .get(row_label_index)
            .copied()
            .flatten()
    }

    pub fn induced_z_subgroup_index_for_grid_index(&self, grid_index: usize) -> usize {
        let subgroup_size = self.induced_z_grid_size();
        debug_assert!(subgroup_size > 0);
        debug_assert!(grid_index < self.n_grid);
        grid_index % subgroup_size
    }

    pub fn row_update_basis_at_grid_index(&self, grid_index: usize) -> &[Fp] {
        &self.row_update_basis_by_subgroup_index
            [self.induced_z_subgroup_index_for_grid_index(grid_index)]
    }

    /// Return one semantic/coded grid point by raw FFT-grid index.
    pub fn grid_point(&self, grid_index: usize) -> Fp {
        self.x[grid_index]
    }

    /// Return the canonical FFT-grid index for one public logical offset.
    pub fn logical_grid_index(&self, logical_offset: usize) -> usize {
        self.logical_to_grid_index[logical_offset]
    }

    /// Return the public logical offset for one canonical FFT-grid index.
    pub fn logical_offset_for_grid_index(&self, grid_index: usize) -> usize {
        self.grid_index_to_logical_offset[grid_index]
    }

    /// Return the public column coordinate `A = alpha^s`.
    pub fn column_coordinate(&self, alpha: Fp) -> Fp {
        fp_pow(alpha, self.s as u64)
    }

    /// Return the public column coordinate for one raw FFT-grid index.
    pub fn column_coordinate_for_grid_index(&self, grid_index: usize) -> Fp {
        self.column_coordinate(self.grid_point(grid_index))
    }

    /// Return the maximum degree of one maintained row polynomial.
    pub fn row_degree(&self) -> usize {
        self.need - 1
    }

    /// Return the default row-label indices attached to one claimed column.
    pub fn row_label_indices_for_registry_index(&self, registry_index: usize) -> Vec<usize> {
        self.row_label_indices_for_row_family_slice(RowFamilyId(registry_index))
            .map(|indices| indices.to_vec())
            .unwrap_or_default()
    }

    /// Partition explicit rows across the two children of one adjacent kappa-double.
    ///
    /// The union of the returned sets matches the input, but each row label is
    /// assigned to exactly one child according to its descendant parity inside
    /// the current kappa-double orbit.
    pub fn partition_row_label_indices_for_kappa_double(
        &self,
        row_label_indices: &[usize],
    ) -> Option<(Vec<usize>, Vec<usize>)> {
        if self.s < 2 {
            return None;
        }
        let stride = self.induced_z_grid_size();
        if stride == 0 {
            return Some((Vec::new(), Vec::new()));
        }
        let mut left = Vec::new();
        let mut right = Vec::new();
        for &row_label_index in row_label_indices {
            if row_label_index >= self.row_labels.len() {
                return None;
            }
            if (row_label_index / stride) % 2 == 0 {
                left.push(row_label_index);
            } else {
                right.push(row_label_index);
            }
        }
        left.sort_unstable();
        left.dedup();
        right.sort_unstable();
        right.dedup();
        Some((left, right))
    }

    /// Return the induced `Z`-grid size `n = N_grid / s`.
    pub fn induced_z_grid_size(&self) -> usize {
        self.n_grid / self.s
    }

    /// Return the row-label indices in this mode that descend to one next kappa-double label.
    pub fn kappa_double_row_pair_indices(&self, target_row_index: usize) -> Option<(usize, usize)> {
        if self.s < 2 || target_row_index >= self.s / 2 {
            return None;
        }
        Some((2 * target_row_index, 2 * target_row_index + 1))
    }

    /// Return the current row-evaluation indices whose square lands on one kappa-halve target point.
    pub fn kappa_halve_row_eval_preimage_indices(
        &self,
        target_eval_index: usize,
    ) -> Option<(usize, usize)> {
        if self.need < 2 || target_eval_index >= self.need / 2 {
            return None;
        }
        Some((target_eval_index, target_eval_index + self.need / 2))
    }

    /// Return the current row-evaluation index reached by squaring one next kappa-double target point.
    pub fn kappa_double_row_eval_source_index(&self, target_eval_index: usize) -> Option<usize> {
        let target_need = self.need.checked_mul(2)?;
        if target_eval_index >= target_need {
            return None;
        }
        Some(target_eval_index % self.need)
    }
}

fn build_row_family_row_label_indices(
    params: &CodedArrayParams,
    family_id: RowFamilyId,
) -> Vec<usize> {
    if params.rows_per_shard == 0 || family_id.0 >= params.row_family_count {
        return Vec::new();
    }
    let start = family_id.0.saturating_mul(params.rows_per_shard);
    let end = start
        .saturating_add(params.rows_per_shard)
        .min(params.row_labels.len());
    (start..end).collect()
}

/// Timings captured during setup for benchmarking.
#[derive(Clone, Debug, Default)]
pub struct RowMaterializationTimings {
    /// Number of sparse row-materialization calls.
    pub sparse_calls: usize,
    /// Number of FFT-domain row-materialization calls.
    pub fft_domain_calls: usize,
    /// Total number of row families touched across all calls.
    pub row_family_count: usize,
    /// Total number of distinct row labels materialized across all calls.
    pub row_label_count: usize,
    /// Total number of selected rows `q` across all calls.
    pub selected_row_count: usize,
    /// Total FFT-domain size used across all calls.
    pub fft_domain_size: usize,
}

/// One explicit row-family descriptor in the current structured row domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RowFamilyId(pub usize);

#[derive(Clone, Debug, Default)]
pub struct SetupTimings {
    /// Time spent building Merkle default hashes.
    pub tree_build: Duration,
    /// Time spent in the initial inverse FFT.
    pub ifft: Duration,
    /// Time spent evaluating shards from coefficients.
    pub shard_eval: Duration,
    /// Time spent materializing maintained auxiliary rows from coefficients.
    pub row_eval: Duration,
    /// Aggregate row-materialization mode/counter breakdown.
    pub row_materialization: RowMaterializationTimings,
}

impl SetupTimings {
    pub(crate) fn record_sparse_row_materialization(
        &mut self,
        row_family_count: usize,
        row_label_count: usize,
        selected_row_count: usize,
    ) {
        self.row_materialization.sparse_calls += 1;
        self.row_materialization.row_family_count += row_family_count;
        self.row_materialization.row_label_count += row_label_count;
        self.row_materialization.selected_row_count += selected_row_count;
    }

    #[allow(dead_code)]
    pub(crate) fn record_fft_domain_row_materialization(
        &mut self,
        row_family_count: usize,
        row_label_count: usize,
        selected_row_count: usize,
        fft_domain_size: usize,
    ) {
        self.row_materialization.fft_domain_calls += 1;
        self.row_materialization.row_family_count += row_family_count;
        self.row_materialization.row_label_count += row_label_count;
        self.row_materialization.selected_row_count += selected_row_count;
        self.row_materialization.fft_domain_size += fft_domain_size;
    }
}

/// Log2 for power-of-two values.
fn log2_pow2(n: usize) -> usize {
    assert!(n.is_power_of_two());
    usize::BITS as usize - 1 - n.leading_zeros() as usize
}

/// Compute N_grid from a segment tree size.
///
/// `N_grid` is the FFT size / polynomial degree bound used to represent a
/// segment. In the paper-facing model it is exactly `segment_len`.
fn compute_seg_n_grid(segment_len: usize) -> usize {
    debug_assert!(segment_len.is_power_of_two());
    segment_len
}

/// Return the canonical factor-two layout level for `(base_n_grid, n_grid)`.
pub fn canonical_layout_level(
    n_grid: usize,
    base_n_grid: usize,
) -> Result<usize, CanonicalLayoutError> {
    if base_n_grid == 0 {
        return Err(CanonicalLayoutError::ZeroBaseGrid);
    }
    if !base_n_grid.is_power_of_two() {
        return Err(CanonicalLayoutError::BaseGridNotPowerOfTwo);
    }
    if n_grid < base_n_grid {
        return Err(CanonicalLayoutError::CurrentGridSmallerThanBase);
    }
    if n_grid % base_n_grid != 0 {
        return Err(CanonicalLayoutError::CurrentGridNotMultipleOfBase);
    }
    let ratio = n_grid / base_n_grid;
    if !ratio.is_power_of_two() {
        return Err(CanonicalLayoutError::CurrentGridNotPowerOfTwoMultiple);
    }
    Ok(log2_pow2(ratio))
}

fn bit_reverse_width(mut value: usize, width: usize) -> usize {
    let mut reversed = 0usize;
    for _ in 0..width {
        reversed = (reversed << 1) | (value & 1);
        value >>= 1;
    }
    reversed
}

/// Return the canonical public-logical to FFT-grid permutation for a factor-two
/// layout ladder.
///
/// If `n_grid = base_n_grid * 2^level` and `j = q * base_n_grid + r`, then:
///
/// ```text
/// grid(j) = r * 2^level + bit_reverse_level(q)
/// ```
pub fn canonical_logical_to_grid_index(
    n_grid: usize,
    base_n_grid: usize,
) -> Result<Vec<usize>, CanonicalLayoutError> {
    let level = canonical_layout_level(n_grid, base_n_grid)?;
    let stride = 1usize << level;
    let mut layout = Vec::with_capacity(n_grid);
    for logical_offset in 0..n_grid {
        let q = logical_offset / base_n_grid;
        let r = logical_offset % base_n_grid;
        layout.push(r * stride + bit_reverse_width(q, level));
    }
    Ok(layout)
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

#[cfg(test)]
#[allow(dead_code)]
/// Slow shard evaluation used in tests for cross-checks.
///
/// # Mathematical definition
/// Given segment evaluations `t_j = A(x_j)` on the FFT grid, compute shard
/// values at `beta_t = a * h_t` via barycentric interpolation:
/// `A(beta_t) = k(a) * sum_j (w_j * t_j / (beta_t - x_j))`
/// where `k(a) = a^N_grid - 1` and `w_j = 1 / P'(x_j)` for `P(X)=X^N_grid - 1`.
fn compute_initial_shards_slow(params: &CodedArrayParams, t_full: &[Value256]) -> Vec<Vec<Fp>> {
    let mut t_fp = Vec::with_capacity(params.n_grid);
    for &value in t_full.iter().take(params.n_grid) {
        t_fp.push(decode_value(value));
    }
    let mut wt_fp = Vec::with_capacity(params.n_grid);
    for (w_j, t_fp_j) in params.w.iter().zip(t_fp.iter()) {
        wt_fp.push(*w_j * *t_fp_j);
    }

    let mut shards = Vec::with_capacity(params.shard_registry.len());
    let mut denoms = vec![fp_zero(); params.n_grid];
    for entry in params.shard_registry.iter() {
        let a_u = entry.a;
        let k_u = k(a_u, params.n_grid);
        let mut y = Vec::with_capacity(params.s);
        for t in 0..params.s {
            let beta = a_u * params.h[t];
            for (j, x_j) in params.x.iter().enumerate().take(params.n_grid) {
                denoms[j] = beta - x_j;
            }
            batch_invert(&mut denoms);
            let mut sum = fp_zero();
            for (inv, wt_j) in denoms.iter().zip(wt_fp.iter()) {
                sum += *wt_j * *inv;
            }
            y.push(k_u * sum);
        }
        shards.push(y);
    }
    shards
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
/// Given `values[j] = A(x_j)` for `x_j = root^j`, this computes the DFT over
/// the subgroup generated by `root`, producing `A` evaluated at those points.
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

/// Evaluate the column-side lens coefficients `f_r(z)` for one column
/// coordinate `z = a^s`.
fn evaluate_column_polynomials_at_z(
    params: &CodedArrayParams,
    coeffs: &[Fp],
    z: Fp,
    out: &mut [Fp],
) {
    debug_assert_eq!(out.len(), params.s);
    let step = params.n_grid / params.s;
    for value in out.iter_mut() {
        *value = fp_zero();
    }
    for q in (0..step).rev() {
        let base = q * params.s;
        for r in 0..params.s {
            out[r] = out[r] * z + coeffs[base + r];
        }
    }
}

fn deterministic_off_grid_base(n_grid: usize, start: u64) -> Fp {
    let mut cand = start;
    loop {
        let point = Fp::from(cand);
        if point != fp_zero() && fp_pow(point, n_grid as u64) != fp_one() {
            return point;
        }
        cand += 1;
    }
}

fn row_label_root(omega: Fp, n_grid: usize, s: usize) -> Fp {
    if s <= 1 {
        fp_one()
    } else {
        fp_pow(omega, (n_grid / s) as u64)
    }
}

fn build_registry_row_labels(total: usize, omega: Fp, n_grid: usize, s: usize) -> Vec<Fp> {
    if total == 0 {
        return Vec::new();
    }

    let coset_size = usize::max(1, s);
    let root = row_label_root(omega, n_grid, coset_size);
    let coset_count = total.div_ceil(coset_size);
    let mut labels = Vec::with_capacity(total);
    let mut seen_orbits = HashSet::with_capacity(coset_count);
    let mut seed = n_grid as u64 + 2;
    while seen_orbits.len() < coset_count {
        let base = deterministic_off_grid_base(n_grid, seed);
        seed = seed.saturating_add(n_grid as u64 + 1);
        let orbit_id = encode_value(fp_pow(base, coset_size as u64));
        if !seen_orbits.insert(orbit_id) {
            continue;
        }
        let mut cur = fp_one();
        for _ in 0..coset_size {
            if labels.len() == total {
                break;
            }
            labels.push(base * cur);
            cur *= root;
        }
    }
    labels
}

fn detect_structured_row_label_coset_bases(
    row_labels: &[Fp],
    root: Fp,
    coset_size: usize,
) -> Vec<Fp> {
    if coset_size == 0 || row_labels.is_empty() {
        return Vec::new();
    }
    let coset_count = row_labels.len().div_ceil(coset_size);
    let mut bases = Vec::with_capacity(coset_count);
    for coset_idx in 0..coset_count {
        let start = coset_idx * coset_size;
        let end = usize::min(start + coset_size, row_labels.len());
        let base = row_labels[start];
        let mut cur = fp_one();
        for label in &row_labels[start..end] {
            if *label != base * cur {
                return Vec::new();
            }
            cur *= root;
        }
        bases.push(base);
    }
    bases
}

fn build_structured_row_eval_points(omega: Fp, n_grid: usize, s: usize, need: usize) -> Vec<Fp> {
    let z_root = fp_pow(omega, s as u64);
    let base = deterministic_off_grid_base(n_grid, (2 * n_grid) as u64 + 2);
    let base_z = fp_pow(base, s as u64);
    let mut points = Vec::with_capacity(need);
    let mut cur = fp_one();
    for _ in 0..need {
        points.push(base_z * cur);
        cur *= z_root;
    }
    points
}

fn entry_with_target_s(a: Fp, target_s: usize) -> ShardRegEntry {
    let z_val = z(a, target_s);
    ShardRegEntry {
        a,
        z: z_val,
        key: ShardKey {
            a_bytes: encode_value(a),
            z_bytes: encode_value(z_val),
        },
    }
}

pub fn kappa_double_split_twist(params: &CodedArrayParams) -> Fp {
    fp_pow(params.omega, (params.n_grid / params.s) as u64)
}

pub fn kappa_halve_merge_twist(params: &CodedArrayParams) -> Fp {
    fp_pow(params.omega, (params.n_grid / (2 * params.s)) as u64)
}

pub fn registry_supports_sibling_kappa_halve(params: &CodedArrayParams) -> bool {
    if params.shard_registry.len() % 2 != 0 {
        return false;
    }
    let twist = kappa_halve_merge_twist(params);
    for pair_idx in (0..params.shard_registry.len()).step_by(2) {
        let left = params.shard_registry[pair_idx];
        let right = params.shard_registry[pair_idx + 1];
        if right.a != left.a * twist || right.z != -left.z {
            return false;
        }
    }
    true
}

pub fn kappa_double_child_registry_indices(registry_index: usize) -> (usize, usize) {
    (2 * registry_index, 2 * registry_index + 1)
}

pub fn kappa_halve_parent_registry_index(registry_index: usize) -> usize {
    registry_index / 2
}

pub fn build_migrated_party_registry_indices(
    params: &CodedArrayParams,
    target_s: usize,
) -> Vec<Vec<usize>> {
    if target_s == params.s / 2 {
        params
            .party_registry_indices
            .iter()
            .map(|indices| {
                let mut migrated = Vec::with_capacity(indices.len() * 2);
                for &registry_index in indices {
                    let (left, right) = kappa_double_child_registry_indices(registry_index);
                    migrated.push(left);
                    migrated.push(right);
                }
                migrated.sort_unstable();
                migrated.dedup();
                migrated
            })
            .collect()
    } else if target_s == params.s * 2 && registry_supports_sibling_kappa_halve(params) {
        params
            .party_registry_indices
            .iter()
            .map(|indices| {
                let present: HashSet<_> = indices.iter().copied().collect();
                let mut migrated = Vec::new();
                for &registry_index in indices {
                    if registry_index % 2 == 0 && present.contains(&(registry_index + 1)) {
                        migrated.push(kappa_halve_parent_registry_index(registry_index));
                    }
                }
                migrated.sort_unstable();
                migrated.dedup();
                migrated
            })
            .collect()
    } else {
        params.party_registry_indices.clone()
    }
}

/// Rebuild a shard registry for a new mode while preserving the underlying representatives.
pub fn build_migrated_shard_registry(
    params: &CodedArrayParams,
    target_s: usize,
) -> Vec<ShardRegEntry> {
    let mut out = Vec::new();
    let mut seen_z = HashSet::new();
    if target_s == params.s / 2 {
        let twist = kappa_double_split_twist(params);
        out.reserve(params.shard_registry.len() * 2);
        for entry in &params.shard_registry {
            let left = entry_with_target_s(entry.a, target_s);
            let right = entry_with_target_s(entry.a * twist, target_s);
            for child in [left, right] {
                assert!(
                    seen_z.insert(child.key.z_bytes),
                    "migrated shard registry produced duplicate z coordinates"
                );
                out.push(child);
            }
        }
    } else if target_s == params.s * 2 && registry_supports_sibling_kappa_halve(params) {
        out.reserve(params.shard_registry.len() / 2);
        for pair_idx in (0..params.shard_registry.len()).step_by(2) {
            let merged = entry_with_target_s(params.shard_registry[pair_idx].a, target_s);
            assert!(
                seen_z.insert(merged.key.z_bytes),
                "migrated shard registry produced duplicate z coordinates"
            );
            out.push(merged);
        }
    } else {
        out.reserve(params.shard_registry.len());
        for entry in &params.shard_registry {
            let migrated = entry_with_target_s(entry.a, target_s);
            assert!(
                seen_z.insert(migrated.key.z_bytes),
                "migrated shard registry produced duplicate z coordinates"
            );
            out.push(migrated);
        }
    }
    out
}

pub(crate) fn compute_segment_coeffs_from_values(
    params: &CodedArrayParams,
    t_full: &[Value256],
    timings: Option<&mut SetupTimings>,
) -> Vec<Fp> {
    let mut timings = timings;
    let mut values = vec![fp_zero(); params.n_grid];
    for (idx, value) in t_full.iter().take(params.n_grid).enumerate() {
        values[idx] = decode_value(*value);
    }

    let ifft_start = timings.as_ref().map(|_| Instant::now());
    inverse_fft_in_place(&mut values, params.omega);
    if let (Some(start), Some(timing)) = (ifft_start, timings.as_deref_mut()) {
        timing.ifft += start.elapsed();
    }
    values
}

pub(crate) fn compute_segment_coeffs_from_values_on_coset(
    params: &CodedArrayParams,
    t_full: &[Value256],
    rho: Fp,
    timings: Option<&mut SetupTimings>,
) -> Vec<Fp> {
    let mut coeffs = compute_segment_coeffs_from_values(params, t_full, timings);
    if rho == fp_one() {
        return coeffs;
    }
    let inv_rho = rho
        .inverse()
        .expect("segment domain scale must be non-zero");
    let mut scale = fp_one();
    for coeff in &mut coeffs {
        *coeff *= scale;
        scale *= inv_rho;
    }
    coeffs
}

pub(crate) fn compute_initial_shards_from_coeffs(
    params: &CodedArrayParams,
    coeffs: &[Fp],
    timings: Option<&mut SetupTimings>,
) -> Vec<Vec<Fp>> {
    let mut timings = timings;
    let shard_start = timings.as_ref().map(|_| Instant::now());
    let step = params.n_grid / params.s;
    let root_s = fp_pow(params.omega, step as u64);
    let mut shards = Vec::with_capacity(params.shard_registry.len());
    let mut p_r = vec![fp_zero(); params.s];
    let mut c = vec![fp_zero(); params.s];
    for entry in params.shard_registry.iter() {
        let z_u = entry.z;
        evaluate_column_polynomials_at_z(params, coeffs, z_u, &mut p_r);

        let mut a_pow = fp_one();
        for r in 0..params.s {
            c[r] = a_pow * p_r[r];
            a_pow *= entry.a;
        }
        fft_in_place(&mut c, root_s);
        shards.push(c.clone());
    }
    if let (Some(start), Some(timing)) = (shard_start, timings.as_deref_mut()) {
        timing.shard_eval += start.elapsed();
    }
    shards
}

pub(crate) fn compute_selected_shards_from_coeffs(
    params: &CodedArrayParams,
    coeffs: &[Fp],
    registry_indices: &[usize],
    timings: Option<&mut SetupTimings>,
) -> Vec<(usize, Vec<Fp>)> {
    let mut timings = timings;
    if registry_indices.is_empty() {
        return Vec::new();
    }
    let shard_start = timings.as_ref().map(|_| Instant::now());
    let step = params.n_grid / params.s;
    let root_s = fp_pow(params.omega, step as u64);
    let mut p_r = vec![fp_zero(); params.s];
    let mut c = vec![fp_zero(); params.s];
    let mut selected = registry_indices.to_vec();
    selected.sort_unstable();
    selected.dedup();
    let mut shards = Vec::with_capacity(selected.len());
    for registry_index in selected {
        let entry = params.shard_registry[registry_index];
        evaluate_column_polynomials_at_z(params, coeffs, entry.z, &mut p_r);

        let mut a_pow = fp_one();
        for r in 0..params.s {
            c[r] = a_pow * p_r[r];
            a_pow *= entry.a;
        }
        fft_in_place(&mut c, root_s);
        shards.push((registry_index, c.clone()));
    }
    if let (Some(start), Some(timing)) = (shard_start, timings.as_deref_mut()) {
        timing.shard_eval += start.elapsed();
    }
    shards
}

pub(crate) fn compute_initial_rows_from_coeffs(
    params: &CodedArrayParams,
    coeffs: &[Fp],
    timings: Option<&mut SetupTimings>,
) -> Vec<Vec<Fp>> {
    let mut timings = timings;
    let row_start = timings.as_ref().map(|_| Instant::now());
    if params.row_labels.is_empty() || params.row_eval_points.is_empty() {
        return vec![vec![fp_zero(); params.row_eval_points.len()]; params.row_labels.len()];
    }

    let rows = if uses_fft_domain_for_full_row_materialization(params) {
        compute_initial_rows_from_coeffs_fft_domain(params, coeffs)
    } else {
        compute_initial_rows_from_coeffs_sparse(params, coeffs)
    };
    if let (Some(start), Some(timing)) = (row_start, timings.as_deref_mut()) {
        timing.row_eval += start.elapsed();
    }
    rows
}

fn compute_initial_rows_from_coeffs_sparse(
    params: &CodedArrayParams,
    coeffs: &[Fp],
) -> Vec<Vec<Fp>> {
    let mut rows = vec![vec![fp_zero(); params.row_eval_points.len()]; params.row_labels.len()];
    if rows.is_empty() || params.row_eval_points.is_empty() {
        return rows;
    }

    let mut p_r = vec![fp_zero(); params.s];
    for (z_idx, &z_eval) in params.row_eval_points.iter().enumerate() {
        evaluate_column_polynomials_at_z(params, coeffs, z_eval, &mut p_r);
        for (row_idx, &row_label) in params.row_labels.iter().enumerate() {
            let mut value = fp_zero();
            for coeff in p_r.iter().rev() {
                value = value * row_label + *coeff;
            }
            rows[row_idx][z_idx] = value;
        }
    }
    rows
}

pub(crate) fn compute_selected_rows_from_coeffs(
    params: &CodedArrayParams,
    coeffs: &[Fp],
    row_label_indices: &[usize],
    timings: Option<&mut SetupTimings>,
) -> Vec<(usize, Vec<Fp>)> {
    let mut timings = timings;
    if row_label_indices.is_empty() || params.row_eval_points.is_empty() {
        return row_label_indices
            .iter()
            .copied()
            .map(|row_label_index| (row_label_index, Vec::new()))
            .collect();
    }

    let row_start = timings.as_ref().map(|_| Instant::now());
    let mut selected = row_label_indices.to_vec();
    selected.sort_unstable();
    selected.dedup();
    validate_row_label_indices(params, &selected);
    let rows = if should_materialize_rows_via_fft_domain(params, &selected) {
        compute_selected_rows_from_coeffs_fft_domain(params, coeffs, &selected)
    } else {
        compute_selected_rows_from_coeffs_sparse(params, coeffs, &selected)
    };
    if let (Some(start), Some(timing)) = (row_start, timings.as_deref_mut()) {
        timing.row_eval += start.elapsed();
    }
    selected.into_iter().zip(rows).collect()
}

fn compute_selected_rows_from_coeffs_sparse(
    params: &CodedArrayParams,
    coeffs: &[Fp],
    selected: &[usize],
) -> Vec<Vec<Fp>> {
    let eval_count = params.row_eval_points.len();
    let mut rows = vec![vec![fp_zero(); eval_count]; selected.len()];
    if selected.is_empty() || eval_count == 0 {
        return rows;
    }

    let z_degree = params.n_grid / params.s;
    let z_root = fp_pow(params.omega, params.s as u64);
    let structured_eval_domain =
        eval_count == z_degree && row_eval_points_are_structured(params, z_root);

    for (row_pos, &row_label_index) in selected.iter().enumerate() {
        let row_label = params.row_labels[row_label_index];
        let mut z_coeffs = row_polynomial_coeffs_for_label(params, coeffs, row_label);
        if structured_eval_domain {
            evaluate_row_polynomial_on_structured_z_domain(
                params,
                z_root,
                &mut z_coeffs,
                &mut rows[row_pos],
            );
        } else {
            evaluate_row_polynomial_on_explicit_z_points(
                &z_coeffs,
                &params.row_eval_points,
                &mut rows[row_pos],
            );
        }
    }
    rows
}

fn row_polynomial_coeffs_for_label(
    params: &CodedArrayParams,
    coeffs: &[Fp],
    row_label: Fp,
) -> Vec<Fp> {
    let z_degree = params.n_grid / params.s;
    let mut z_coeffs = vec![fp_zero(); z_degree];
    for q in 0..z_degree {
        let base = q * params.s;
        let mut acc = fp_zero();
        for r in (0..params.s).rev() {
            acc = acc * row_label + coeffs[base + r];
        }
        z_coeffs[q] = acc;
    }
    z_coeffs
}

pub(crate) fn row_eval_points_are_structured(params: &CodedArrayParams, z_root: Fp) -> bool {
    let Some(&base_z) = params.row_eval_points.first() else {
        return false;
    };
    let mut cur = base_z;
    for &point in &params.row_eval_points {
        if point != cur {
            return false;
        }
        cur *= z_root;
    }
    true
}

fn evaluate_row_polynomial_on_structured_z_domain(
    params: &CodedArrayParams,
    z_root: Fp,
    z_coeffs: &mut [Fp],
    out: &mut [Fp],
) {
    if z_coeffs.is_empty() {
        return;
    }
    debug_assert_eq!(z_coeffs.len(), out.len());
    let base_z = params.row_eval_points[0];
    let mut base_pow = fp_one();
    for coeff in z_coeffs.iter_mut() {
        *coeff *= base_pow;
        base_pow *= base_z;
    }
    fft_in_place(z_coeffs, z_root);
    out.copy_from_slice(z_coeffs);
}

pub(crate) fn evaluate_row_coefficients_on_row_eval_points(
    params: &CodedArrayParams,
    coefficients: &[Fp],
) -> Vec<Fp> {
    let eval_count = params.row_eval_points.len();
    let mut out = vec![fp_zero(); eval_count];
    if eval_count == 0 {
        return out;
    }

    let z_degree = params.n_grid / params.s;
    let z_root = fp_pow(params.omega, params.s as u64);
    if coefficients.len() <= eval_count
        && eval_count == z_degree
        && row_eval_points_are_structured(params, z_root)
    {
        let mut coeffs = vec![fp_zero(); eval_count];
        coeffs[..coefficients.len()].copy_from_slice(coefficients);
        evaluate_row_polynomial_on_structured_z_domain(params, z_root, &mut coeffs, &mut out);
    } else {
        evaluate_row_polynomial_on_explicit_z_points(
            coefficients,
            &params.row_eval_points,
            &mut out,
        );
    }
    out
}

fn evaluate_row_polynomial_on_explicit_z_points(z_coeffs: &[Fp], z_points: &[Fp], out: &mut [Fp]) {
    debug_assert_eq!(z_points.len(), out.len());
    for (z_idx, &z_eval) in z_points.iter().enumerate() {
        let mut value = fp_zero();
        for coeff in z_coeffs.iter().rev() {
            value = value * z_eval + *coeff;
        }
        out[z_idx] = value;
    }
}

fn compute_initial_rows_from_coeffs_fft_domain(
    params: &CodedArrayParams,
    coeffs: &[Fp],
) -> Vec<Vec<Fp>> {
    let mut rows = vec![vec![fp_zero(); params.row_eval_points.len()]; params.row_labels.len()];
    let mut p_r = vec![fp_zero(); params.s];
    let mut domain_values = vec![fp_zero(); params.s];
    for (z_idx, &z_eval) in params.row_eval_points.iter().enumerate() {
        evaluate_column_polynomials_at_z(params, coeffs, z_eval, &mut p_r);
        for (coset_idx, &base) in params.row_label_coset_bases.iter().enumerate() {
            evaluate_row_polynomial_on_label_coset(params, &p_r, base, &mut domain_values);
            let row_base = coset_idx * params.s;
            for (offset, &value) in domain_values.iter().enumerate() {
                let row_idx = row_base + offset;
                if row_idx >= rows.len() {
                    break;
                }
                rows[row_idx][z_idx] = value;
            }
        }
    }
    rows
}

fn compute_selected_rows_from_coeffs_fft_domain(
    params: &CodedArrayParams,
    coeffs: &[Fp],
    selected: &[usize],
) -> Vec<Vec<Fp>> {
    let mut rows = vec![vec![fp_zero(); params.row_eval_points.len()]; selected.len()];
    let mut by_coset = vec![Vec::<(usize, usize)>::new(); params.row_label_coset_bases.len()];
    for (row_pos, &row_label_index) in selected.iter().enumerate() {
        let coset_idx = row_label_index / params.s;
        let offset = row_label_index % params.s;
        if let Some(coset_items) = by_coset.get_mut(coset_idx) {
            coset_items.push((row_pos, offset));
        }
    }
    let mut p_r = vec![fp_zero(); params.s];
    let mut domain_values = vec![fp_zero(); params.s];
    for (z_idx, &z_eval) in params.row_eval_points.iter().enumerate() {
        evaluate_column_polynomials_at_z(params, coeffs, z_eval, &mut p_r);
        for (coset_idx, positions) in by_coset.iter().enumerate() {
            if positions.is_empty() {
                continue;
            }
            evaluate_row_polynomial_on_label_coset(
                params,
                &p_r,
                params.row_label_coset_bases[coset_idx],
                &mut domain_values,
            );
            for &(row_pos, offset) in positions {
                rows[row_pos][z_idx] = domain_values[offset];
            }
        }
    }
    rows
}

fn evaluate_row_polynomial_on_label_coset(
    params: &CodedArrayParams,
    row_coeffs: &[Fp],
    coset_base: Fp,
    out: &mut [Fp],
) {
    debug_assert_eq!(row_coeffs.len(), params.s);
    debug_assert_eq!(out.len(), params.s);
    for value in out.iter_mut() {
        *value = fp_zero();
    }
    let mut base_pow = fp_one();
    for (idx, &coeff) in row_coeffs.iter().enumerate() {
        out[idx] = coeff * base_pow;
        base_pow *= coset_base;
    }
    fft_in_place(out, row_label_root(params.omega, params.n_grid, params.s));
}

fn validate_row_label_indices(params: &CodedArrayParams, row_label_indices: &[usize]) {
    for &row_label_index in row_label_indices {
        assert!(
            row_label_index < params.row_labels.len(),
            "row label index {} out of range for {} labels",
            row_label_index,
            params.row_labels.len()
        );
    }
}

pub(crate) fn full_row_fft_domain_size(params: &CodedArrayParams) -> usize {
    params.row_label_coset_bases.len().saturating_mul(params.s)
}

pub(crate) fn selected_row_fft_domain_size(
    params: &CodedArrayParams,
    row_label_indices: &[usize],
) -> usize {
    structured_row_label_cosets_touched(params, row_label_indices).saturating_mul(params.s)
}

pub(crate) fn uses_fft_domain_for_full_row_materialization(params: &CodedArrayParams) -> bool {
    !params.row_label_coset_bases.is_empty()
}

pub(crate) fn should_materialize_rows_via_fft_domain(
    params: &CodedArrayParams,
    row_label_indices: &[usize],
) -> bool {
    if row_label_indices.is_empty() || params.row_label_coset_bases.is_empty() {
        return false;
    }
    let touched_cosets = structured_row_label_cosets_touched(params, row_label_indices);
    if touched_cosets == 0 {
        return false;
    }
    let dense_work = touched_cosets
        .saturating_mul(params.s)
        .saturating_mul(log2_pow2(params.s).max(1));
    let sparse_work = row_label_indices.len().saturating_mul(params.s);
    sparse_work >= dense_work
}

fn structured_row_label_cosets_touched(
    params: &CodedArrayParams,
    row_label_indices: &[usize],
) -> usize {
    validate_row_label_indices(params, row_label_indices);
    if params.row_label_coset_bases.is_empty() {
        return 0;
    }
    let mut touched = vec![false; params.row_label_coset_bases.len()];
    let mut count = 0usize;
    for &row_label_index in row_label_indices {
        let coset_idx = row_label_index / params.s;
        if let Some(seen) = touched.get_mut(coset_idx) {
            if !*seen {
                *seen = true;
                count += 1;
            }
        }
    }
    count
}

/// Return the row-scaling factor `kappa_i(w)` for one grid point `x_i` and row
/// label `w`.
pub(crate) fn row_update_scale_for_grid_index(
    params: &CodedArrayParams,
    grid_index: usize,
    row_label: Fp,
) -> Fp {
    let x_i = params.grid_point(grid_index);
    if row_label == x_i {
        return fp_one();
    }

    let a_i = params.column_coordinate_for_grid_index(grid_index);
    let numerator = fp_pow(row_label, params.s as u64) - a_i;
    if numerator == fp_zero() {
        return fp_zero();
    }

    let x_i_pow = if params.s == 1 {
        fp_one()
    } else {
        fp_pow(x_i, (params.s - 1) as u64)
    };
    let denom = Fp::from(params.s as u64) * x_i_pow * (row_label - x_i);
    numerator * denom.inverse().expect("non-zero row update denominator")
}

/// Evaluate the shared `Z`-basis `Lambda_{A_i}(Z)` for one raw FFT-grid point.
pub(crate) fn row_update_basis_for_grid_index(
    params: &CodedArrayParams,
    grid_index: usize,
    out: &mut [Fp],
) {
    debug_assert_eq!(out.len(), params.row_eval_points.len());
    if out.is_empty() {
        return;
    }

    let a_i = params.column_coordinate_for_grid_index(grid_index);
    let n = params.induced_z_grid_size();
    debug_assert_eq!(n, params.need);
    let inv_n = Fp::from(n as u64)
        .inverse()
        .expect("induced Z-grid size is non-zero");

    for (value, &z_eval) in out.iter_mut().zip(params.row_eval_points.iter()) {
        *value = z_eval - a_i;
    }
    batch_invert(out);
    for (idx, value) in out.iter_mut().enumerate() {
        let z_eval = params.row_eval_points[idx];
        *value *= a_i * inv_n * (fp_pow(z_eval, n as u64) - fp_one());
    }
}

/// Materialize the maintained auxiliary rows for one segment.
pub fn compute_initial_rows_option_b(
    params: &CodedArrayParams,
    t_full: &[Value256],
) -> Vec<Vec<Fp>> {
    let coeffs = compute_segment_coeffs_from_values(params, t_full, None);
    compute_initial_rows_from_coeffs(params, &coeffs, None)
}

/// Fast shard evaluation using an IFFT + evaluation over roots of unity.
///
/// # Algorithm
/// 1) Interpret the segment values `t_j` as evaluations on the FFT grid
///    `x_j = omega^j` and apply an inverse FFT to recover coefficients of
///    `A(X) = sum_{i=0}^{N_grid-1} c_i X^i`.
/// 2) For each shard entry `(a, z)` with `z = a^s`, evaluate `A` on the coset
///    points `beta_t = a * h_t`, where `h_t = omega^{t * (N_grid/s)}`.
///    This is done by grouping coefficients by residue modulo `s` and using
///    a transposed Horner scheme followed by an `s`-point FFT.
///
/// # Complexity
/// - IFFT: `O(N_grid log N_grid)`
/// - Per shard: `O(N_grid) + O(s log s)`
pub fn compute_initial_shards_option_b(
    params: &CodedArrayParams,
    t_full: &[Value256],
    timings: Option<&mut SetupTimings>,
) -> Vec<Vec<Fp>> {
    let mut timings = timings;
    let coeffs = compute_segment_coeffs_from_values(params, t_full, timings.as_deref_mut());
    compute_initial_shards_from_coeffs(params, &coeffs, timings.as_deref_mut())
}

pub fn compute_initial_shards_option_b_on_coset(
    params: &CodedArrayParams,
    t_full: &[Value256],
    rho: Fp,
    timings: Option<&mut SetupTimings>,
) -> Vec<Vec<Fp>> {
    let mut timings = timings;
    let coeffs =
        compute_segment_coeffs_from_values_on_coset(params, t_full, rho, timings.as_deref_mut());
    compute_initial_shards_from_coeffs(params, &coeffs, timings.as_deref_mut())
}

/// Validate a candidate shard entry and return it if acceptable.
///
/// # Preconditions enforced
/// - `a != 0`
/// - `a^N_grid != 1` (so `k(a) != 0`)
/// - `z = a^s` is unique within the registry
pub fn accept_candidate(
    a: Fp,
    s: usize,
    n_grid: usize,
    seen_z: &HashSet<Value256>,
) -> Option<ShardRegEntry> {
    if a == fp_zero() {
        return None;
    }
    if fp_pow(a, n_grid as u64) == fp_one() {
        return None;
    }
    let z_val = z(a, s);
    let z_bytes = encode_value(z_val);
    if seen_z.contains(&z_bytes) {
        return None;
    }
    let a_bytes = encode_value(a);
    Some(ShardRegEntry {
        a,
        z: z_val,
        key: ShardKey { a_bytes, z_bytes },
    })
}

/// Deterministically build a shard registry by scanning field elements.
///
/// This is used for reproducible tests and benchmarks.
pub fn build_shard_registry_deterministic(
    target: usize,
    s: usize,
    n_grid: usize,
) -> Vec<ShardRegEntry> {
    let mut shards = Vec::with_capacity(target);
    let mut seen_z: HashSet<Value256> = HashSet::new();
    let mut cand: u64 = 2;
    while shards.len() < target {
        let a = Fp::from(cand);
        if let Some(shard) = accept_candidate(a, s, n_grid, &seen_z) {
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
pub fn sample_random_registry<R: Rng + ?Sized>(
    rng: &mut R,
    target: usize,
    s: usize,
    n_grid: usize,
) -> Vec<ShardRegEntry> {
    let mut shards = Vec::with_capacity(target);
    let mut seen_z: HashSet<Value256> = HashSet::new();
    while shards.len() < target {
        let a = Fp::rand(rng);
        if let Some(shard) = accept_candidate(a, s, n_grid, &seen_z) {
            seen_z.insert(shard.key.z_bytes);
            shards.push(shard);
        }
    }
    shards
}

/// Validate configuration constraints and registry consistency.
///
/// # Invariants checked
/// - Powers-of-two sizing and segment bounds
/// - Registry key consistency and unique `z` values
/// - Registry index groups are in range and unique per group
pub fn validate_coded_config(cfg: &CodedArrayConfig) {
    assert!(cfg.segment_len > 0);
    assert!(cfg.segment_len.is_power_of_two());
    assert!(cfg.s.is_power_of_two());
    assert!(cfg.shard_count > 0);
    assert_eq!(cfg.shard_count, cfg.shard_registry.len());
    let n_grid = compute_seg_n_grid(cfg.segment_len);
    assert!(n_grid.is_power_of_two());
    assert!(cfg.s <= n_grid);
    let mut seen_z = HashSet::new();
    for entry in &cfg.shard_registry {
        assert_eq!(entry.key.a_bytes, encode_value(entry.a));
        assert_eq!(entry.key.z_bytes, encode_value(entry.z));
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
/// 1) Validate sizing and shard registry constraints.
/// 2) Derive `N_grid`, FFT roots, and the evaluation grid (`x_j`, `w_j`).
/// 3) Precompute `h_t` points used for shard evaluation.
/// 4) Precompute registry lookup tables.
pub fn build_coded_params(cfg: &CodedArrayConfig) -> CodedArrayParams {
    validate_coded_config(cfg);
    let segment_len = cfg.segment_len;
    let n_grid = compute_seg_n_grid(segment_len);
    let two_adicity = <Fp as ark_ff::FftField>::TWO_ADICITY as usize;
    assert!(log2_pow2(n_grid) <= two_adicity);
    let s = cfg.s;
    let delta = (n_grid - 1) / s;
    let need = delta + 1;

    let omega = Fp::get_root_of_unity(n_grid as u64).expect("root of unity");
    assert!(fp_pow(omega, n_grid as u64) == fp_one());
    assert!(fp_pow(omega, (n_grid / 2) as u64) != fp_one());

    let inv_n_grid = Fp::from(n_grid as u64).inverse().unwrap();
    let mut x = Vec::with_capacity(n_grid);
    let mut grid_index_by_x_bytes = HashMap::with_capacity(n_grid);
    let mut w = Vec::with_capacity(n_grid);
    let mut cur = fp_one();
    for idx in 0..n_grid {
        x.push(cur);
        grid_index_by_x_bytes.insert(encode_value(cur), idx);
        w.push(cur * inv_n_grid);
        cur *= omega;
    }
    let logical_to_grid_index = (0..n_grid).collect::<Vec<_>>();
    let grid_index_to_logical_offset = (0..n_grid).collect::<Vec<_>>();

    let step = n_grid / s;
    let mut h = Vec::with_capacity(s);
    for t in 0..s {
        h.push(fp_pow(omega, (t * step) as u64));
    }
    let h_bary_weights = precompute_barycentric(&h);
    let rows_per_shard = usize::max(1, (s + need - 1) / need);
    let row_family_count = cfg.shard_registry.len();
    let row_labels = build_registry_row_labels(
        row_family_count.saturating_mul(rows_per_shard),
        omega,
        n_grid,
        s,
    );
    let row_eval_points = build_structured_row_eval_points(omega, n_grid, s, need);
    let row_domain_size = row_labels.len();
    let row_family_size = rows_per_shard;

    let mut params = CodedArrayParams {
        segment_len,
        n_grid: n_grid,
        s,
        delta,
        need,
        omega,
        inv_n_grid: inv_n_grid,
        x,
        layout_base_n_grid: n_grid,
        layout_level: 0,
        logical_to_grid_index,
        grid_index_to_logical_offset,
        grid_index_by_x_bytes,
        w,
        h,
        h_bary_weights,
        shard_registry: cfg.shard_registry.clone(),
        shard_registry_by_key: HashMap::new(),
        inv_a: Vec::new(),
        inv_denom: Vec::new(),
        party_registry_indices: cfg.party_registry_indices.clone(),
        rows_per_shard,
        row_labels,
        row_domain_size,
        row_family_size,
        row_family_count,
        row_family_row_label_indices: Arc::new(Vec::new()),
        row_family_location_by_row_label_index: Vec::new(),
        row_family_id_by_indices: HashMap::new(),
        row_label_coset_bases: Vec::new(),
        row_label_bytes: Vec::new(),
        row_label_index_by_bytes: HashMap::new(),
        row_eval_points,
        row_eval_bary_weights: BarycentricWeights {
            z: Vec::new(),
            w: Vec::new(),
        },
        induced_z_subgroup_index_by_bytes: HashMap::new(),
        row_update_basis_by_subgroup_index: Arc::new(Vec::new()),
    };

    let s_fp = Fp::from(params.s as u64);
    let mut inv_a = Vec::with_capacity(params.shard_registry.len());
    let mut inv_denom = Vec::with_capacity(params.shard_registry.len());
    let mut shard_registry_by_key = HashMap::with_capacity(params.shard_registry.len());
    for (idx, entry) in params.shard_registry.iter().enumerate() {
        shard_registry_by_key.insert(entry.key, idx);
        inv_a.push(
            entry
                .a
                .inverse()
                .expect("validated shard representative is non-zero"),
        );
        let a_pow = fp_pow(entry.a, (params.s - 1) as u64);
        let denom = s_fp * a_pow;
        inv_denom.push(denom.inverse());
    }
    params.inv_a = inv_a;
    params.inv_denom = inv_denom;
    params.shard_registry_by_key = shard_registry_by_key;
    params.refresh_row_metadata_caches();

    params
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::encode_value;

    fn params_for(segment_len: usize, s: usize) -> CodedArrayParams {
        let n_grid = compute_seg_n_grid(segment_len);
        let need = n_grid / s;
        build_coded_params(&CodedArrayConfig {
            segment_len,
            s,
            shard_count: need,
            shard_registry: build_shard_registry_deterministic(need, s, n_grid),
            party_registry_indices: (0..need).map(|idx| vec![idx]).collect(),
            seed: 11,
        })
    }

    fn deterministic_segment(params: &CodedArrayParams) -> Vec<crate::Value256> {
        (0..params.n_grid)
            .map(|idx| {
                let value = ((idx as u64 + 1) * 1_315_423_911u64) ^ 0x5eed_1234_u64;
                encode_value(Fp::from(value))
            })
            .collect()
    }

    #[test]
    fn canonical_layout_matches_repeated_even_odd_joins() {
        assert_eq!(
            canonical_logical_to_grid_index(4, 4).expect("level zero"),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            canonical_logical_to_grid_index(8, 4).expect("level one"),
            vec![0, 2, 4, 6, 1, 3, 5, 7]
        );
        assert_eq!(
            canonical_logical_to_grid_index(16, 4).expect("level two"),
            vec![0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15]
        );
    }

    #[test]
    fn canonical_layout_rejects_non_ladder_grid() {
        assert_eq!(
            canonical_layout_level(8, 0),
            Err(CanonicalLayoutError::ZeroBaseGrid)
        );
        assert_eq!(
            canonical_layout_level(8, 6),
            Err(CanonicalLayoutError::BaseGridNotPowerOfTwo)
        );
        assert_eq!(
            canonical_layout_level(8, 16),
            Err(CanonicalLayoutError::CurrentGridSmallerThanBase)
        );
        assert_eq!(
            canonical_layout_level(32, 12),
            Err(CanonicalLayoutError::BaseGridNotPowerOfTwo)
        );
    }

    #[test]
    fn selected_sparse_rows_match_full_materialization() {
        for (segment_len, s, selected) in [
            (4096usize, 64usize, vec![0usize, 1, 17, 63]),
            (4096usize, 512usize, vec![0usize, 7, 63, 127]),
            (128usize, 4usize, vec![0usize, 5, 17, 31]),
        ] {
            let params = params_for(segment_len, s);
            let segment = deterministic_segment(&params);
            let coeffs = compute_segment_coeffs_from_values(&params, &segment, None);
            let full_rows = compute_initial_rows_from_coeffs(&params, &coeffs, None);
            let selected_rows =
                compute_selected_rows_from_coeffs(&params, &coeffs, &selected, None);
            for (row_label_index, row) in selected_rows {
                assert_eq!(
                    row, full_rows[row_label_index],
                    "selected row {row_label_index} disagrees for segment_len={segment_len}, s={s}",
                );
            }
        }
    }

    #[test]
    fn row_coefficients_eval_matches_explicit_points() {
        for (segment_len, s) in [
            (128usize, 4usize),
            (1024usize, 64usize),
            (4096usize, 64usize),
        ] {
            let params = params_for(segment_len, s);
            let coeffs = (0..params.need)
                .map(|idx| Fp::from((idx as u64 + 3) * 17))
                .collect::<Vec<_>>();
            let fast = evaluate_row_coefficients_on_row_eval_points(&params, &coeffs);
            let mut explicit = vec![fp_zero(); params.row_eval_points.len()];
            evaluate_row_polynomial_on_explicit_z_points(
                &coeffs,
                &params.row_eval_points,
                &mut explicit,
            );
            assert_eq!(
                fast, explicit,
                "row coefficient evaluation mismatch for segment_len={segment_len}, s={s}",
            );
        }
    }
}
