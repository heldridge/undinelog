//! Server-side shard-value and shard update logic.
//!
//! # Overview
//! This module implements the math that servers (parties) use to:
//! - Answer shard queries at arbitrary evaluation points `alpha`.
//! - Apply per-node deltas to shard vectors, either incrementally or in batches.
//!
//! # Mathematical model
//! Let `A(X)` be the segment polynomial of degree `< N_grid` with evaluations
//! `A(x_j)` at FFT grid points `x_j = omega^j`. For a shard entry `(a, z)` with
//! `z = a^s`, define `beta_t = a * h_t` for `h_t = omega^{t * (N_grid / s)}`.
//! Each shard stores the vector `y_t = A(beta_t)` for `t = 0..s-1`.
//!
//! The shard value for a query `alpha` is the direct evaluation of the shard's
//! degree-`< s` column polynomial at `alpha`. The fast path below evaluates
//! that polynomial from the stored coset samples.
//!
//! # Batching
//! Applying deltas uses one of two strategies:
//! - Aggregate (loop-swapped) updates, good for small batches.
//! - FFT-based interpolation for large batches, amortizing costs across shards.

use crate::coded_array::params::{
    CodedArrayParams, ShardRegEntry, fft_in_place, inverse_fft_in_place, k,
};
use crate::coded_array::store::CodedStore;
use crate::field::{Fp, batch_invert, fp_one, fp_pow, fp_zero};
use crate::value::encode_value;
use ark_ff::Field;
#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicUsize, Ordering};

/// Strategy used when applying batches of deltas to shard vectors.
#[derive(Clone, Copy, Debug)]
pub enum DeltaMode {
    /// Strategy A: aggregate deltas using a loop-swapped method.
    Aggregate,
    /// Strategy B: FFT-based interpolate/eval for large batches.
    InterpolateFFT,
}

#[cfg(debug_assertions)]
static FFT_BRANCH_HITS: AtomicUsize = AtomicUsize::new(0);

#[cfg(debug_assertions)]
fn record_fft_branch_hit() {
    FFT_BRANCH_HITS.fetch_add(1, Ordering::Relaxed);
}

#[cfg(debug_assertions)]
/// Reset the debug counter for FFT-branch usage.
pub fn reset_fft_branch_hits() {
    FFT_BRANCH_HITS.store(0, Ordering::Relaxed);
}

#[cfg(debug_assertions)]
/// Return the number of times the FFT branch has executed.
pub fn fft_branch_hits() -> usize {
    FFT_BRANCH_HITS.load(Ordering::Relaxed)
}

/// Scratch space for shard-value evaluation to avoid reallocation.
struct ShardValueScratch {
    denoms: Vec<Fp>,
}

impl ShardValueScratch {
    /// Create an empty scratch buffer.
    fn new() -> Self {
        Self { denoms: Vec::new() }
    }
}

// Thread-local scratch to keep the fast path allocation-free.
thread_local! {
    static SHARD_VALUE_SCRATCH: std::cell::RefCell<ShardValueScratch> =
        std::cell::RefCell::new(ShardValueScratch::new());
}

/// Scratch space for aggregate shard updates.
struct ShardUpdateScratch {
    betas: Vec<Fp>,
    denoms: Vec<Fp>,
    xs: Vec<Fp>,
    weighted_deltas: Vec<Fp>,
}

impl ShardUpdateScratch {
    fn new() -> Self {
        Self {
            betas: Vec::new(),
            denoms: Vec::new(),
            xs: Vec::new(),
            weighted_deltas: Vec::new(),
        }
    }
}

thread_local! {
    static SHARD_UPDATE_SCRATCH: std::cell::RefCell<ShardUpdateScratch> =
        std::cell::RefCell::new(ShardUpdateScratch::new());
}

/// Minimum batch size before we switch to loop-swapped aggregation.
const AGGREGATE_THRESHOLD: usize = 32;
/// Minimum batch size before we switch to FFT-based interpolation.
///
/// The FFT path has higher constant overhead but asymptotically fewer
/// inversions. Below this threshold the aggregate method is faster.
const FFT_THRESHOLD: usize = 64;
/// Minimum batch size before we switch to FFT-based shard-value evaluation.
const SHARD_VALUE_FFT_THRESHOLD: usize = 64;

fn eval_poly_horner(coeffs: &[Fp], alpha: Fp) -> Fp {
    let mut acc = fp_zero();
    for coeff in coeffs.iter().rev() {
        acc = acc * alpha + *coeff;
    }
    acc
}

/// Compute the shard value from a shard vector `y` at evaluation point `alpha`.
///
/// This is the barycentric evaluation formula specialized to the shard's
/// `s`-point coset. If `alpha` is itself one of the stored coset points,
/// return the stored sample directly.
fn shard_value_from_y(
    params: &CodedArrayParams,
    entry: ShardRegEntry,
    y: &[Fp],
    alpha: Fp,
) -> Option<Fp> {
    let entry_idx = params.shard_registry_by_key.get(&entry.key).copied()?;
    let inv_denom = params.inv_denom.get(entry_idx).copied().flatten()?;
    let a_u = entry.a;
    for t in 0..params.s {
        if alpha == a_u * params.h[t] {
            return Some(y[t]);
        }
    }
    // Reuse thread-local buffers for per-call denominators.
    SHARD_VALUE_SCRATCH.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        if scratch.denoms.len() != params.s {
            scratch.denoms.resize(params.s, fp_zero());
        }
        for t in 0..params.s {
            scratch.denoms[t] = alpha - a_u * params.h[t];
        }
        batch_invert(&mut scratch.denoms);
        let mut numer = fp_zero();
        let mut denom = fp_zero();
        for t in 0..params.s {
            let inv = scratch.denoms[t];
            let c_t = params.h[t] * inv_denom;
            numer += c_t * y[t] * inv;
            denom += c_t * inv;
        }
        if denom == fp_zero() {
            None
        } else {
            Some(numer * denom.inverse().unwrap())
        }
    })
}

/// Compute shard values for a batch of evaluation points.
fn shard_value_from_y_batch(
    params: &CodedArrayParams,
    entry: ShardRegEntry,
    y: &[Fp],
    alphas: &[Fp],
) -> Vec<Option<Fp>> {
    if alphas.is_empty() {
        return Vec::new();
    }
    if alphas.len() < SHARD_VALUE_FFT_THRESHOLD {
        return alphas
            .iter()
            .copied()
            .map(|alpha| shard_value_from_y(params, entry, y, alpha))
            .collect();
    }
    let entry_idx = match params.shard_registry_by_key.get(&entry.key).copied() {
        Some(idx) => idx,
        None => return vec![None; alphas.len()],
    };
    if params.inv_denom.get(entry_idx).copied().flatten().is_none() {
        return vec![None; alphas.len()];
    }
    let a_inv = match entry.a.inverse() {
        Some(inv) => inv,
        None => return vec![None; alphas.len()],
    };
    let root_s = fp_pow(params.omega, (params.n_grid / params.s) as u64);
    let mut coeffs = y.to_vec();
    inverse_fft_in_place(&mut coeffs, root_s);
    let mut a_inv_pow = fp_one();
    for coeff in coeffs.iter_mut() {
        *coeff *= a_inv_pow;
        a_inv_pow *= a_inv;
    }

    let mut use_fft = alphas.len() >= SHARD_VALUE_FFT_THRESHOLD;
    let mut alpha_indices = Vec::new();
    if use_fft {
        // FFT evaluation is only valid for FFT-grid alphas.
        alpha_indices.reserve(alphas.len());
        for &alpha in alphas {
            if let Some(&j) = params.grid_index_by_x_bytes.get(&encode_value(alpha)) {
                alpha_indices.push(j);
            } else {
                use_fft = false;
                alpha_indices.clear();
                break;
            }
        }
    }

    let mut values = Vec::with_capacity(alphas.len());
    if use_fft {
        let mut evals = vec![fp_zero(); params.n_grid];
        evals[..params.s].copy_from_slice(&coeffs);
        fft_in_place(&mut evals, params.omega);
        for &idx in &alpha_indices {
            values.push(evals[idx]);
        }
    } else {
        for &alpha in alphas {
            values.push(eval_poly_horner(&coeffs, alpha));
        }
    }

    values.into_iter().map(Some).collect()
}

fn shard_value_from_y_batch_on_grid_coset(
    params: &CodedArrayParams,
    entry: ShardRegEntry,
    y: &[Fp],
    rho: Fp,
    grid_indices: &[usize],
) -> Vec<Option<Fp>> {
    if grid_indices.is_empty() {
        return Vec::new();
    }
    if grid_indices.len() < SHARD_VALUE_FFT_THRESHOLD {
        return grid_indices
            .iter()
            .copied()
            .map(|grid_index| {
                if grid_index >= params.n_grid {
                    None
                } else {
                    shard_value_from_y(params, entry, y, rho * params.grid_point(grid_index))
                }
            })
            .collect();
    }
    let entry_idx = match params.shard_registry_by_key.get(&entry.key).copied() {
        Some(idx) => idx,
        None => return vec![None; grid_indices.len()],
    };
    if params.inv_denom.get(entry_idx).copied().flatten().is_none() {
        return vec![None; grid_indices.len()];
    }
    if grid_indices
        .iter()
        .any(|&grid_index| grid_index >= params.n_grid)
    {
        return vec![None; grid_indices.len()];
    }

    let a_inv = match entry.a.inverse() {
        Some(inv) => inv,
        None => return vec![None; grid_indices.len()],
    };
    let root_s = fp_pow(params.omega, (params.n_grid / params.s) as u64);
    let mut coeffs = y.to_vec();
    inverse_fft_in_place(&mut coeffs, root_s);
    let mut a_inv_pow = fp_one();
    for coeff in coeffs.iter_mut() {
        *coeff *= a_inv_pow;
        a_inv_pow *= a_inv;
    }

    let mut evals = vec![fp_zero(); params.n_grid];
    let mut rho_pow = fp_one();
    for (eval_coeff, coeff) in evals.iter_mut().zip(coeffs.iter()) {
        *eval_coeff = *coeff * rho_pow;
        rho_pow *= rho;
    }
    fft_in_place(&mut evals, params.omega);

    grid_indices
        .iter()
        .map(|&grid_index| Some(evals[grid_index]))
        .collect()
}

/// Fetch a shard vector for a store/segment and compute its direct value.
pub fn shard_value_from_store(
    params: &CodedArrayParams,
    store: &CodedStore,
    entry: ShardRegEntry,
    seg: usize,
    alpha: Fp,
) -> Option<Fp> {
    let y = store.shard_segment(params, &entry.key, seg)?;
    shard_value_from_y(params, entry, y, alpha)
}

/// Fetch a shard vector for a store/segment and compute batched direct values.
pub fn shard_value_from_store_batch(
    params: &CodedArrayParams,
    store: &CodedStore,
    entry: ShardRegEntry,
    seg: usize,
    alphas: &[Fp],
) -> Vec<Option<Fp>> {
    let Some(y) = store.shard_segment(params, &entry.key, seg) else {
        return vec![None; alphas.len()];
    };
    shard_value_from_y_batch(params, entry, y, alphas)
}

/// Fetch a shard vector and evaluate it at `rho * x_j` for a batch of grid indices.
pub fn shard_value_from_store_batch_on_grid_coset(
    params: &CodedArrayParams,
    store: &CodedStore,
    entry: ShardRegEntry,
    seg: usize,
    rho: Fp,
    grid_indices: &[usize],
) -> Vec<Option<Fp>> {
    let Some(y) = store.shard_segment(params, &entry.key, seg) else {
        return vec![None; grid_indices.len()];
    };
    shard_value_from_y_batch_on_grid_coset(params, entry, y, rho, grid_indices)
}

/// Compute a direct shard value from a shard vector.
pub fn shard_value_from_vector(
    params: &CodedArrayParams,
    entry: ShardRegEntry,
    y: &[Fp],
    alpha: Fp,
) -> Option<Fp> {
    shard_value_from_y(params, entry, y, alpha)
}

/// Compute batched direct shard values from a shard vector.
pub fn shard_value_from_vector_batch(
    params: &CodedArrayParams,
    entry: ShardRegEntry,
    y: &[Fp],
    alphas: &[Fp],
) -> Vec<Option<Fp>> {
    shard_value_from_y_batch(params, entry, y, alphas)
}

/// Direct shard value for segment 0 only.
pub fn shard_value(
    params: &CodedArrayParams,
    store: &CodedStore,
    entry: ShardRegEntry,
    alpha: Fp,
) -> Option<Fp> {
    shard_value_from_store(params, store, entry, 0, alpha)
}

/// Apply per-node deltas to a single shard vector (core implementation).
///
/// # Mathematical definition
/// Each delta at grid point `x_j` updates `A(X)` by `delta * L_j(X)` where
/// `L_j` is the Lagrange basis polynomial. For a shard point `beta_t`:
/// ```text
/// A'(beta_t) = A(beta_t) + delta * w_j * k(a) / (beta_t - x_j)
/// ```
/// with `w_j = 1 / P'(x_j)` and `P(X) = X^N_grid - 1`.
///
/// # Complexity
/// `O(m * s)` per shard, where `m = deltas.len()`.
fn apply_deltas_to_one_shard_impl(
    params: &CodedArrayParams,
    entry: ShardRegEntry,
    y_vec: &mut [Fp],
    deltas: &[(usize, Fp)],
) {
    let a_u = entry.a;
    let k_u = k(a_u, params.n_grid);
    // Precompute beta values for this shard.
    let mut betas_u = Vec::with_capacity(params.s);
    for t in 0..params.s {
        betas_u.push(a_u * params.h[t]);
    }
    let mut denoms = vec![fp_zero(); params.s];
    for &(j, delta) in deltas {
        let x_j = params.x[j];
        let w_j = params.w[j];
        let scale = w_j * k_u;
        for t in 0..params.s {
            denoms[t] = betas_u[t] - x_j;
        }
        batch_invert(&mut denoms);
        for t in 0..params.s {
            y_vec[t] += delta * scale * denoms[t];
        }
    }
}

/// Apply per-node deltas to a single shard vector using loop-swapped aggregation.
///
/// This reorders loops to batch the inversions for each `beta_t`, reducing
/// constant factors for large `m` while keeping the same `O(m * s)` cost.
fn apply_deltas_to_one_shard_aggregate(
    params: &CodedArrayParams,
    entry: ShardRegEntry,
    y_vec: &mut [Fp],
    deltas: &[(usize, Fp)],
) {
    let m = deltas.len();
    if m == 0 {
        return;
    }
    if m < AGGREGATE_THRESHOLD {
        apply_deltas_to_one_shard_impl(params, entry, y_vec, deltas);
        return;
    }
    let a_u = entry.a;
    let k_u = k(a_u, params.n_grid);
    SHARD_UPDATE_SCRATCH.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        if scratch.betas.len() != params.s {
            scratch.betas.resize(params.s, fp_zero());
        }
        if scratch.denoms.len() < m {
            scratch.denoms.resize(m, fp_zero());
        }
        if scratch.xs.len() < m {
            scratch.xs.resize(m, fp_zero());
        }
        if scratch.weighted_deltas.len() < m {
            scratch.weighted_deltas.resize(m, fp_zero());
        }
        for t in 0..params.s {
            scratch.betas[t] = a_u * params.h[t];
        }
        for (idx, &(j, delta)) in deltas.iter().enumerate() {
            scratch.xs[idx] = params.x[j];
            scratch.weighted_deltas[idx] = delta * params.w[j];
        }
        for t in 0..params.s {
            let beta = scratch.betas[t];
            for r in 0..m {
                scratch.denoms[r] = beta - scratch.xs[r];
            }
            batch_invert(&mut scratch.denoms[..m]);
            let mut sum = fp_zero();
            for r in 0..m {
                sum += scratch.weighted_deltas[r] * scratch.denoms[r];
            }
            y_vec[t] += k_u * sum;
        }
    })
}

/// Apply deltas for a specific segment across all shards held by the store.
pub fn apply_deltas_to_segment(
    params: &CodedArrayParams,
    store: &mut CodedStore,
    seg: usize,
    deltas: &[(usize, Fp)],
) {
    apply_deltas_to_segment_with_mode(params, store, seg, deltas, DeltaMode::Aggregate);
}

/// Apply deltas for a specific segment across all shards held by the store with a chosen mode.
///
/// # FFT path (DeltaMode::InterpolateFFT)
/// For large batches, we first reconstruct the polynomial `D(X)` whose
/// evaluations on the FFT grid are the delta values. We exploit:
/// - `P(X) = X^N_grid - 1`
/// - `P'(x_j) = N_grid * x_j^{N_grid-1}`
/// - `w_j = 1 / P'(x_j)`
/// to convert the barycentric numerator into grid evaluations, then use an
/// inverse FFT to recover coefficients.
///
/// # Complexity
/// - Preprocessing: `O(N_grid log N_grid)` for the inverse FFT.
/// - Per shard: `O(N_grid + s log s)` to evaluate on the coset.
pub fn apply_deltas_to_segment_with_mode(
    params: &CodedArrayParams,
    store: &mut CodedStore,
    seg: usize,
    deltas: &[(usize, Fp)],
    mode: DeltaMode,
) {
    if deltas.is_empty() {
        return;
    }
    match mode {
        DeltaMode::Aggregate => {
            store.for_each_shard_segment_mut(params, seg, |shard_key, y_vec| {
                let entry_idx = params
                    .shard_registry_by_key
                    .get(shard_key)
                    .copied()
                    .expect("missing shard entry");
                let entry = params.shard_registry[entry_idx];
                apply_deltas_to_one_shard_aggregate(params, entry, y_vec, deltas);
            });
        }
        DeltaMode::InterpolateFFT => {
            if deltas.len() < FFT_THRESHOLD {
                apply_deltas_to_segment_with_mode(params, store, seg, deltas, DeltaMode::Aggregate);
                return;
            }
            if std::env::var("BENCH_DEBUG")
                .map(|value| value == "1")
                .unwrap_or(false)
            {
                eprintln!("[sanity] used_fft=true");
            }
            #[cfg(debug_assertions)]
            record_fft_branch_hit();
            // Build d_j = delta_j * w_j (barycentric numerator weights).
            // For roots of unity x_j:
            //   P(X) = X^N - 1,  P'(x_j) = N * x_j^{N-1}
            //   w_j = 1 / P'(x_j)
            // so d_j * N * x_j^{-1} = delta_j. We use inv_x = x_{N-j} since
            // x_j * x_{N-j} = 1 on the FFT grid.
            let n_grid = params.n_grid;
            let mut values = vec![fp_zero(); n_grid];
            for &(j, delta) in deltas {
                if j < n_grid {
                    values[j] += delta * params.w[j];
                }
            }
            let n_fp = Fp::from(n_grid as u64);
            for j in 0..n_grid {
                if values[j] != fp_zero() {
                    let inv_x = params.x[(n_grid - j) % n_grid];
                    values[j] *= n_fp * inv_x;
                }
            }
            // Now `values` are the FFT-grid evaluations of the delta polynomial.
            let mut coeffs = values;
            inverse_fft_in_place(&mut coeffs, params.omega);
            let step = n_grid / params.s;
            let root_s = crate::field::fp_pow(params.omega, step as u64);
            let mut p_r = vec![fp_zero(); params.s];
            let mut evals = vec![fp_zero(); params.s];
            store.for_each_shard_segment_mut(params, seg, |shard_key, y_vec| {
                let entry_idx = params
                    .shard_registry_by_key
                    .get(shard_key)
                    .copied()
                    .expect("missing shard entry");
                let entry = params.shard_registry[entry_idx];
                for v in p_r.iter_mut() {
                    *v = fp_zero();
                }
                for q in (0..step).rev() {
                    let base = q * params.s;
                    for r in 0..params.s {
                        p_r[r] = p_r[r] * entry.z + coeffs[base + r];
                    }
                }
                let mut a_pow = fp_one();
                for r in 0..params.s {
                    evals[r] = a_pow * p_r[r];
                    a_pow *= entry.a;
                }
                fft_in_place(&mut evals, root_s);
                let k_u = k(entry.a, n_grid);
                // For beta_t = a_u * h_t, P(beta_t)=beta_t^N-1=k_u, so
                // sum_j d_j/(beta_t-x_j) = A(beta_t)/k_u.
                let inv_k_u = k_u.inverse().expect("non-zero k_u");
                for t in 0..params.s {
                    let s_t = evals[t] * inv_k_u;
                    y_vec[t] += k_u * s_t;
                }
            });
        }
    }
}

/// Public wrapper for applying deltas to a single shard vector.
pub fn apply_deltas_to_one_shard(
    params: &CodedArrayParams,
    entry: ShardRegEntry,
    y_vec: &mut [Fp],
    deltas: &[(usize, Fp)],
) {
    apply_deltas_to_one_shard_impl(params, entry, y_vec, deltas);
}

#[cfg(test)]
mod tests {
    //! Unit tests cover shard-value and delta-application correctness.
    use super::{
        apply_deltas_to_segment, apply_deltas_to_segment_with_mode, shard_value_from_store,
        shard_value_from_vector_batch,
    };
    use crate::coded_array::params::{
        CodedArrayConfig, CodedArrayParams, ShardRegEntry, build_coded_params,
        build_shard_registry_deterministic, compute_initial_shards_option_b, k,
    };
    use crate::coded_array::store::CodedStore;
    use crate::field::{Fp, batch_invert, fp_pow, fp_zero};
    use crate::value::encode_value;
    use ark_ff::{Field, UniformRand};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    use super::DeltaMode;

    /// Build a deterministic registry for tests with matching `N_grid`.
    fn build_registry(g: usize, s: usize, segment_len: usize) -> Vec<ShardRegEntry> {
        let n_grid = segment_len;
        build_shard_registry_deterministic(g, s, n_grid)
    }

    /// Build params and materialize a single segment for one store.
    fn setup_one_store(
        segment_len: usize,
        s: usize,
        g: usize,
        seed: u64,
    ) -> (CodedArrayParams, CodedStore) {
        let shard_registry = build_registry(g, s, segment_len);
        let party_registry_indices = vec![(0..g).collect()];
        let cfg = CodedArrayConfig {
            s,
            segment_len,
            shard_count: g,
            shard_registry,
            party_registry_indices,
            seed,
        };
        let params = build_coded_params(&cfg);

        let mut store = CodedStore::new();

        let mut rng = ChaCha20Rng::seed_from_u64(seed ^ 0xBEEF_BEEF);
        let mut t_seg = Vec::with_capacity(params.n_grid);
        for _ in 0..params.n_grid {
            t_seg.push(encode_value(Fp::rand(&mut rng)));
        }
        let shards = compute_initial_shards_option_b(&params, &t_seg, None);
        for entry in params.shard_registry.iter().copied() {
            let idx = params
                .shard_registry_by_key
                .get(&entry.key)
                .copied()
                .expect("missing shard entry");
            store.append_shard_segment(&params, entry.key, 0, &shards[idx]);
        }

        (params, store)
    }

    /// Slow reference implementation used to validate the fast delta update.
    fn apply_deltas_to_store_slow(
        params: &CodedArrayParams,
        store: &mut CodedStore,
        seg: usize,
        deltas: &[(usize, Fp)],
    ) {
        store.for_each_shard_segment_mut(params, seg, |shard_key, shard| {
            let entry_idx = params
                .shard_registry_by_key
                .get(shard_key)
                .copied()
                .expect("missing shard entry");
            let a_u = params.shard_registry[entry_idx].a;
            let k_u = k(a_u, params.n_grid);
            for &(j, delta) in deltas {
                let x_j = params.x[j];
                let w_j = params.w[j];
                let mut denoms: Vec<Fp> = Vec::with_capacity(params.s);
                for t in 0..params.s {
                    let beta = a_u * params.h[t];
                    denoms.push(beta - x_j);
                }
                batch_invert(&mut denoms);
                for t in 0..params.s {
                    let mul = w_j * k_u * denoms[t];
                    shard[t] += delta * mul;
                }
            }
        });
    }

    /// Reference shard evaluation for correctness comparisons.
    fn shard_value_reference(
        params: &CodedArrayParams,
        entry: ShardRegEntry,
        y: &[Fp],
        alpha: Fp,
    ) -> Option<Fp> {
        let a_u = entry.a;
        for t in 0..params.s {
            if alpha == a_u * params.h[t] {
                return Some(y[t]);
            }
        }
        let a_pow = fp_pow(a_u, (params.s - 1) as u64);
        let s_fp = Fp::from(params.s as u64);
        let denom = s_fp * a_pow;
        let inv_denom = denom.inverse()?;
        let mut numer = fp_zero();
        let mut denom_acc = fp_zero();
        let mut denoms = Vec::with_capacity(params.s);
        for &h_t in params.h.iter().take(params.s) {
            let beta = a_u * h_t;
            denoms.push(alpha - beta);
        }
        batch_invert(&mut denoms);
        for ((&h_t, y_t), inv) in params
            .h
            .iter()
            .zip(y.iter())
            .zip(denoms.iter())
            .take(params.s)
        {
            let c_t = h_t * inv_denom;
            numer += c_t * *y_t * *inv;
            denom_acc += c_t * *inv;
        }
        if denom_acc == fp_zero() {
            None
        } else {
            Some(numer * denom_acc.inverse().unwrap())
        }
    }

    #[test]
    // Ensure fast delta application matches the reference implementation.
    fn apply_deltas_to_store_matches_reference() {
        let (params, mut store_fast) = setup_one_store(64, 8, 16, 0);
        let mut store_slow = store_fast.clone();

        let mut rng = ChaCha20Rng::seed_from_u64(12345);
        let mut deltas = Vec::with_capacity(12);
        for _ in 0..12 {
            let j = rng.gen_range(0..params.n_grid);
            let delta = Fp::rand(&mut rng);
            deltas.push((j, delta));
        }

        apply_deltas_to_segment(&params, &mut store_fast, 0, &deltas);
        apply_deltas_to_store_slow(&params, &mut store_slow, 0, &deltas);

        assert_eq!(
            store_fast.shard_family_count(),
            store_slow.shard_family_count()
        );
        for shard_key in store_fast.shard_keys() {
            let shard_fast = store_fast
                .shard_segment(&params, shard_key, 0)
                .expect("missing fast shard");
            let shard_slow = store_slow
                .shard_segment(&params, shard_key, 0)
                .expect("missing slow shard");
            assert_eq!(shard_fast.len(), shard_slow.len());
            for (idx, (a, b)) in shard_fast.iter().zip(shard_slow.iter()).enumerate() {
                assert_eq!(*a, *b, "shard {:?} idx {}", shard_key, idx);
            }
        }
    }

    #[test]
    // Ensure aggregate mode matches the reference implementation.
    fn apply_deltas_to_store_aggregate_matches_reference() {
        let (params, mut store_fast) = setup_one_store(64, 8, 16, 101);
        let mut store_slow = store_fast.clone();

        let mut rng = ChaCha20Rng::seed_from_u64(999);
        let mut deltas = Vec::with_capacity(48);
        for _ in 0..48 {
            let j = rng.gen_range(0..params.n_grid);
            let delta = Fp::rand(&mut rng);
            deltas.push((j, delta));
        }

        apply_deltas_to_segment_with_mode(
            &params,
            &mut store_fast,
            0,
            &deltas,
            DeltaMode::Aggregate,
        );
        apply_deltas_to_store_slow(&params, &mut store_slow, 0, &deltas);

        assert_eq!(
            store_fast.shard_family_count(),
            store_slow.shard_family_count()
        );
        for shard_key in store_fast.shard_keys() {
            let shard_fast = store_fast
                .shard_segment(&params, shard_key, 0)
                .expect("missing fast shard");
            let shard_slow = store_slow
                .shard_segment(&params, shard_key, 0)
                .expect("missing slow shard");
            assert_eq!(shard_fast.len(), shard_slow.len());
            for (idx, (a, b)) in shard_fast.iter().zip(shard_slow.iter()).enumerate() {
                assert_eq!(*a, *b, "shard {:?} idx {}", shard_key, idx);
            }
        }
    }

    #[test]
    // Ensure FFT mode matches the reference implementation for large batches.
    fn apply_deltas_to_store_fft_matches_reference() {
        let (params, mut store_fast) = setup_one_store(64, 8, 16, 202);
        let mut store_slow = store_fast.clone();

        let mut rng = ChaCha20Rng::seed_from_u64(424242);
        let mut deltas = Vec::with_capacity(96);
        for _ in 0..96 {
            let j = rng.gen_range(0..params.n_grid);
            let delta = Fp::rand(&mut rng);
            deltas.push((j, delta));
        }

        apply_deltas_to_segment_with_mode(
            &params,
            &mut store_fast,
            0,
            &deltas,
            DeltaMode::InterpolateFFT,
        );
        apply_deltas_to_store_slow(&params, &mut store_slow, 0, &deltas);

        assert_eq!(
            store_fast.shard_family_count(),
            store_slow.shard_family_count()
        );
        for shard_key in store_fast.shard_keys() {
            let shard_fast = store_fast
                .shard_segment(&params, shard_key, 0)
                .expect("missing fast shard");
            let shard_slow = store_slow
                .shard_segment(&params, shard_key, 0)
                .expect("missing slow shard");
            assert_eq!(shard_fast.len(), shard_slow.len());
            for (idx, (a, b)) in shard_fast.iter().zip(shard_slow.iter()).enumerate() {
                assert_eq!(*a, *b, "shard {:?} idx {}", shard_key, idx);
            }
        }
    }

    #[test]
    // Ensure fast shard-value evaluation matches the reference math.
    fn shard_value_matches_reference() {
        let (params, store) = setup_one_store(64, 64, 16, 0);

        let mut shard_keys: Vec<_> = store.shard_keys().copied().collect();
        shard_keys.sort_unstable();

        let mut rng = ChaCha20Rng::seed_from_u64(4242);
        for &shard_key in &shard_keys {
            let entry_idx = params
                .shard_registry_by_key
                .get(&shard_key)
                .copied()
                .expect("missing shard entry");
            let entry = params.shard_registry[entry_idx];
            let y = store
                .shard_segment(&params, &shard_key, 0)
                .expect("missing seg 0");
            for _ in 0..8 {
                let alpha = Fp::rand(&mut rng);
                let fast = shard_value_from_store(&params, &store, entry, 0, alpha);
                let slow = shard_value_reference(&params, entry, y, alpha);
                assert_eq!(fast, slow, "random alpha shard {:?}", shard_key);
            }

            let beta0 = entry.a * params.h[0];
            let fast = shard_value_from_store(&params, &store, entry, 0, beta0);
            let slow = shard_value_reference(&params, entry, y, beta0);
            assert_eq!(fast, slow, "beta alpha shard {:?}", shard_key);
        }
    }

    #[test]
    // Ensure batched shard-value evaluation matches the reference implementation (small batch).
    fn shard_value_batch_small_matches_reference() {
        let (params, store) = setup_one_store(64, 8, 16, 7);
        let entry = params.shard_registry[0];
        let y = store
            .shard_segment(&params, &entry.key, 0)
            .expect("missing seg 0");

        let mut rng = ChaCha20Rng::seed_from_u64(112233);
        let mut alphas = Vec::new();
        for _ in 0..8 {
            alphas.push(Fp::rand(&mut rng));
        }
        alphas.push(entry.a * params.h[0]);
        assert!(alphas.len() < super::SHARD_VALUE_FFT_THRESHOLD);

        let replies = shard_value_from_vector_batch(&params, entry, y, &alphas);
        for (alpha, reply) in alphas.iter().zip(replies.iter()) {
            let slow = shard_value_reference(&params, entry, y, *alpha);
            assert_eq!(reply, &slow);
        }
    }

    #[test]
    // Ensure batched shard-value evaluation matches the reference implementation (FFT batch).
    fn shard_value_batch_fft_matches_reference() {
        let (params, store) = setup_one_store(128, 8, 16, 9);
        let entry = params.shard_registry[0];
        let y = store
            .shard_segment(&params, &entry.key, 0)
            .expect("missing seg 0");

        let batch_size = super::SHARD_VALUE_FFT_THRESHOLD + 1;
        assert!(batch_size <= params.n_grid);
        let alphas: Vec<Fp> = params.x.iter().take(batch_size).copied().collect();

        let replies = shard_value_from_vector_batch(&params, entry, y, &alphas);
        for (alpha, reply) in alphas.iter().zip(replies.iter()) {
            let slow = shard_value_reference(&params, entry, y, *alpha);
            assert_eq!(reply, &slow);
        }
    }

    #[test]
    // Ensure direct coset-sample queries return the stored value.
    fn shard_value_none_case_matches_reference() {
        let (params, store) = setup_one_store(8, 1, 4, 1);

        for shard_key in store.shard_keys() {
            let entry_idx = params
                .shard_registry_by_key
                .get(shard_key)
                .copied()
                .expect("missing shard entry");
            let entry = params.shard_registry[entry_idx];
            let y = store
                .shard_segment(&params, shard_key, 0)
                .expect("missing seg 0");
            let alpha = entry.a * params.h[0];
            let fast = shard_value_from_store(&params, &store, entry, 0, alpha);
            let slow = shard_value_reference(&params, entry, y, alpha);
            assert_eq!(fast, slow, "coset sample shard {:?}", shard_key);
            assert_eq!(
                fast,
                Some(y[0]),
                "expected stored sample for shard {:?}",
                shard_key
            );
        }
    }
}
