//! Reed-Solomon helpers using barycentric interpolation.
//!
//! # Overview
//! This module provides a minimal Reed-Solomon toolkit used throughout the
//! system to decode values from shard points. The core primitive is
//! barycentric interpolation on arbitrary evaluation points.
//!
//! # Mathematical definition
//! Given distinct points `(z_i, y_i)`, the unique polynomial `A(X)` of degree
//! `< n` satisfying `A(z_i) = y_i` can be evaluated at `z` via:
//! ```text
//! A(z) = (sum_i w_i * y_i / (z - z_i)) / (sum_i w_i / (z - z_i))
//! ```
//! where `w_i = 1 / prod_{j != i} (z_i - z_j)` are barycentric weights.
//!
//! # Preconditions
//! - All `z_i` must be distinct (or interpolation is undefined).
//! - For `barycentric_eval`, if `z` equals some `z_i`, the function returns
//!   `y_i` directly to avoid division by zero.

use crate::field::{Fp, batch_invert, fp_one, fp_zero};
use ark_ff::Field;

/// Precomputed barycentric weights for a set of z-values.
///
/// `w[i] = 1 / prod_{j != i} (z[i] - z[j])`.
#[derive(Clone, Debug)]
pub struct BarycentricWeights {
    pub z: Vec<Fp>,
    pub w: Vec<Fp>,
}

/// Precompute barycentric weights for interpolation at arbitrary points.
///
/// # Complexity
/// `O(n^2)` field multiplications and one batch inversion.
pub fn precompute_barycentric(z: &[Fp]) -> BarycentricWeights {
    if z.is_empty() {
        return BarycentricWeights {
            z: Vec::new(),
            w: Vec::new(),
        };
    }
    let mut denom = vec![fp_one(); z.len()];
    for i in 0..z.len() {
        let mut acc = fp_one();
        for j in 0..z.len() {
            if i == j {
                continue;
            }
            acc *= z[i] - z[j];
        }
        denom[i] = acc;
    }
    batch_invert(&mut denom);
    BarycentricWeights {
        z: z.to_vec(),
        w: denom,
    }
}

/// Evaluate the interpolated polynomial at `z_eval` using barycentric weights.
///
/// # Mathematical definition
/// Computes:
/// `A(z_eval) = (sum_i w_i * y_i / (z_eval - z_i)) / (sum_i w_i / (z_eval - z_i))`
/// with a fast return if `z_eval` equals a sample point.
///
/// # Complexity
/// `O(n)` field operations plus a batch inversion.
pub fn barycentric_eval(weights: &BarycentricWeights, y: &[Fp], z_eval: Fp) -> Fp {
    assert_eq!(weights.z.len(), y.len());
    for (z_i, y_i) in weights.z.iter().zip(y.iter()) {
        if *z_i == z_eval {
            return *y_i;
        }
    }
    if weights.z.is_empty() {
        return fp_zero();
    }
    let mut denoms = Vec::with_capacity(weights.z.len());
    for z_i in weights.z.iter() {
        denoms.push(z_eval - *z_i);
    }
    batch_invert(&mut denoms);
    let mut num = fp_zero();
    let mut den = fp_zero();
    for i in 0..weights.z.len() {
        let inv = denoms[i];
        let w_i = weights.w[i];
        num += w_i * y[i] * inv;
        den += w_i * inv;
    }
    num * den.inverse().unwrap()
}

/// Evaluate two polynomials, represented on the same barycentric domain, at
/// many target points.
///
/// This is equivalent to calling [`barycentric_eval`] for both value vectors at
/// every target point, but it shares the denominator matrix and batch
/// inversion. It is useful when transport/kappa-rescale needs the same source
/// row sampled on a larger target row grid.
pub fn barycentric_eval_pair_many(
    weights: &BarycentricWeights,
    y0: &[Fp],
    y1: &[Fp],
    z_evals: &[Fp],
) -> (Vec<Fp>, Vec<Fp>) {
    assert_eq!(weights.z.len(), y0.len());
    assert_eq!(weights.z.len(), y1.len());
    if weights.z.is_empty() {
        return (
            vec![fp_zero(); z_evals.len()],
            vec![fp_zero(); z_evals.len()],
        );
    }

    let n = weights.z.len();
    let mut out0 = vec![fp_zero(); z_evals.len()];
    let mut out1 = vec![fp_zero(); z_evals.len()];
    let mut pending = Vec::new();
    let mut denoms = Vec::new();

    for (eval_pos, &z_eval) in z_evals.iter().enumerate() {
        if let Some(sample_pos) = weights.z.iter().position(|&z_i| z_i == z_eval) {
            out0[eval_pos] = y0[sample_pos];
            out1[eval_pos] = y1[sample_pos];
        } else {
            pending.push(eval_pos);
            denoms.extend(weights.z.iter().map(|&z_i| z_eval - z_i));
        }
    }

    batch_invert(&mut denoms);

    let mut nums0 = Vec::with_capacity(pending.len());
    let mut nums1 = Vec::with_capacity(pending.len());
    let mut final_denoms = Vec::with_capacity(pending.len());
    for (pending_pos, _) in pending.iter().enumerate() {
        let start = pending_pos * n;
        let mut num0 = fp_zero();
        let mut num1 = fp_zero();
        let mut den = fp_zero();
        for i in 0..n {
            let inv = denoms[start + i];
            let w_i = weights.w[i];
            let weighted = w_i * inv;
            num0 += weighted * y0[i];
            num1 += weighted * y1[i];
            den += weighted;
        }
        nums0.push(num0);
        nums1.push(num1);
        final_denoms.push(den);
    }

    batch_invert(&mut final_denoms);
    for (pending_pos, &eval_pos) in pending.iter().enumerate() {
        out0[eval_pos] = nums0[pending_pos] * final_denoms[pending_pos];
        out1[eval_pos] = nums1[pending_pos] * final_denoms[pending_pos];
    }

    (out0, out1)
}

/// Result of an oversampled Reed-Solomon decode.
#[derive(Clone, Debug)]
pub struct RsErrorCorrectionResult {
    /// Value of the recovered degree-`< degree_bound` polynomial at the
    /// requested evaluation point.
    pub value: Fp,
    /// Recovered polynomial coefficients in low-to-high order.
    pub coefficients: Vec<Fp>,
    /// Indices into the input point slice that disagreed with the recovered
    /// polynomial.
    pub bad_indices: Vec<usize>,
}

/// Errors returned by the Reed-Solomon error-correcting decoder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RsError {
    NotEnoughPoints,
    DuplicateEvaluationPoint,
    DecodeFailed,
}

/// Decode a degree-`< degree_bound` polynomial at `z_eval` from oversampled
/// Reed-Solomon points, correcting up to `max_errors` arbitrary bad points.
///
/// This uses the Berlekamp-Welch formulation. The fast non-adversarial path in
/// this crate remains barycentric interpolation; this routine is for explicit
/// oversampled reads where callers intentionally fetch at least
/// `degree_bound + 2 * max_errors` distinct points.
pub fn rs_decode_eval_with_errors(
    points: &[(Fp, Fp)],
    degree_bound: usize,
    z_eval: Fp,
    max_errors: usize,
) -> Result<RsErrorCorrectionResult, RsError> {
    if degree_bound == 0 || points.len() < degree_bound {
        return Err(RsError::NotEnoughPoints);
    }
    if has_duplicate_x(points) {
        return Err(RsError::DuplicateEvaluationPoint);
    }

    let capacity = points.len().saturating_sub(degree_bound) / 2;
    let max_errors = max_errors.min(capacity);
    // Most correcting reads should be clean. Try the cheap consistency check
    // first. If it fails, use the caller's configured correction budget directly:
    // the caller already paid for degree_bound + 2*max_errors replies.
    if let Some(decoded) = try_decode_with_budget(points, degree_bound, z_eval, 0) {
        return Ok(decoded);
    }
    if max_errors > 0 {
        if let Some(decoded) = try_decode_with_budget(points, degree_bound, z_eval, max_errors) {
            return Ok(decoded);
        }
    }
    Err(RsError::DecodeFailed)
}

fn try_decode_with_budget(
    points: &[(Fp, Fp)],
    degree_bound: usize,
    z_eval: Fp,
    errors: usize,
) -> Option<RsErrorCorrectionResult> {
    if points.len() < degree_bound + 2 * errors {
        return None;
    }
    let coefficients = berlekamp_welch(points, degree_bound, errors)?;
    let bad_indices = points
        .iter()
        .enumerate()
        .filter_map(|(idx, &(x, y))| (poly_eval(&coefficients, x) != y).then_some(idx))
        .collect::<Vec<_>>();
    if bad_indices.len() <= errors {
        Some(RsErrorCorrectionResult {
            value: poly_eval(&coefficients, z_eval),
            coefficients,
            bad_indices,
        })
    } else {
        None
    }
}

fn has_duplicate_x(points: &[(Fp, Fp)]) -> bool {
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            if points[i].0 == points[j].0 {
                return true;
            }
        }
    }
    false
}

fn berlekamp_welch(points: &[(Fp, Fp)], degree_bound: usize, errors: usize) -> Option<Vec<Fp>> {
    if errors == 0 {
        return interpolate_checked(points, degree_bound);
    }

    let q_len = degree_bound + errors;
    let vars = q_len + errors;
    let mut matrix = Vec::with_capacity(points.len());
    for &(x, y) in points {
        let mut row = vec![fp_zero(); vars + 1];
        let mut x_pow = fp_one();
        for slot in row.iter_mut().take(q_len) {
            *slot = x_pow;
            x_pow *= x;
        }

        x_pow = fp_one();
        for e_idx in 0..errors {
            row[q_len + e_idx] = -(y * x_pow);
            x_pow *= x;
        }
        row[vars] = y * fp_pow_usize(x, errors);
        matrix.push(row);
    }

    let solution = solve_linear_system(matrix, vars)?;
    let q_coeffs = trim_poly(solution[..q_len].to_vec());
    let mut e_coeffs = solution[q_len..].to_vec();
    e_coeffs.push(fp_one());
    let e_coeffs = trim_poly(e_coeffs);
    let (quotient, remainder) = poly_div_rem(&q_coeffs, &e_coeffs)?;
    if !poly_is_zero(&remainder) {
        return None;
    }
    if quotient
        .iter()
        .enumerate()
        .any(|(idx, coeff)| idx >= degree_bound && *coeff != fp_zero())
    {
        return None;
    }
    Some(pad_poly(quotient, degree_bound))
}

fn interpolate_checked(points: &[(Fp, Fp)], degree_bound: usize) -> Option<Vec<Fp>> {
    if points.len() < degree_bound {
        return None;
    }
    let coefficients =
        interpolate_coefficients_from_exact_points(&points[..degree_bound], degree_bound)?;
    if points[degree_bound..]
        .iter()
        .all(|&(x, y)| poly_eval(&coefficients, x) == y)
    {
        Some(coefficients)
    } else {
        None
    }
}

pub(crate) fn interpolate_coefficients_from_exact_points(
    points: &[(Fp, Fp)],
    degree_bound: usize,
) -> Option<Vec<Fp>> {
    if points.len() != degree_bound {
        return None;
    }
    if degree_bound == 0 {
        return Some(Vec::new());
    }

    let z = points.iter().map(|&(x, _)| x).collect::<Vec<_>>();
    let weights = precompute_barycentric(&z);

    let mut root_poly = vec![fp_one()];
    for &x in &z {
        let mut next = vec![fp_zero(); root_poly.len() + 1];
        for (idx, coeff) in root_poly.iter().copied().enumerate() {
            next[idx] -= coeff * x;
            next[idx + 1] += coeff;
        }
        root_poly = next;
    }

    let mut coefficients = vec![fp_zero(); degree_bound];
    for (point_idx, &(x, y)) in points.iter().enumerate() {
        let scale = y * weights.w[point_idx];
        let mut quotient = vec![fp_zero(); degree_bound];
        quotient[degree_bound - 1] = root_poly[degree_bound];
        for coeff_idx in (1..degree_bound).rev() {
            quotient[coeff_idx - 1] = root_poly[coeff_idx] + x * quotient[coeff_idx];
        }
        for (out, basis_coeff) in coefficients.iter_mut().zip(quotient.into_iter()) {
            *out += scale * basis_coeff;
        }
    }
    Some(coefficients)
}

fn solve_linear_system(mut matrix: Vec<Vec<Fp>>, vars: usize) -> Option<Vec<Fp>> {
    let rows = matrix.len();
    let mut pivot_row = 0usize;
    let mut pivot_for_col = vec![None; vars];

    for col in 0..vars {
        let pivot = (pivot_row..rows).find(|&row| matrix[row][col] != fp_zero());
        let Some(found) = pivot else {
            continue;
        };
        matrix.swap(pivot_row, found);
        let inv = matrix[pivot_row][col].inverse()?;
        for value in &mut matrix[pivot_row][col..=vars] {
            *value *= inv;
        }
        let pivot_values = matrix[pivot_row].clone();
        for row in 0..rows {
            if row == pivot_row {
                continue;
            }
            let factor = matrix[row][col];
            if factor == fp_zero() {
                continue;
            }
            for idx in col..=vars {
                matrix[row][idx] -= factor * pivot_values[idx];
            }
        }
        pivot_for_col[col] = Some(pivot_row);
        pivot_row += 1;
        if pivot_row == rows {
            break;
        }
    }

    for row in matrix.iter().take(rows).skip(pivot_row) {
        let lhs_zero = row[..vars].iter().all(|value| *value == fp_zero());
        if lhs_zero && row[vars] != fp_zero() {
            return None;
        }
    }

    let mut solution = vec![fp_zero(); vars];
    for (col, pivot) in pivot_for_col.into_iter().enumerate() {
        if let Some(row) = pivot {
            solution[col] = matrix[row][vars];
        }
    }
    Some(solution)
}

fn poly_eval(coeffs: &[Fp], x: Fp) -> Fp {
    let mut acc = fp_zero();
    for coeff in coeffs.iter().rev() {
        acc = acc * x + *coeff;
    }
    acc
}

fn fp_pow_usize(mut base: Fp, mut exp: usize) -> Fp {
    let mut acc = fp_one();
    while exp > 0 {
        if exp & 1 == 1 {
            acc *= base;
        }
        base *= base;
        exp >>= 1;
    }
    acc
}

fn trim_poly(mut coeffs: Vec<Fp>) -> Vec<Fp> {
    while coeffs.last().is_some_and(|value| *value == fp_zero()) {
        coeffs.pop();
    }
    coeffs
}

fn pad_poly(mut coeffs: Vec<Fp>, len: usize) -> Vec<Fp> {
    coeffs.resize(len, fp_zero());
    coeffs
}

fn poly_is_zero(coeffs: &[Fp]) -> bool {
    coeffs.iter().all(|value| *value == fp_zero())
}

fn poly_div_rem(numer: &[Fp], denom: &[Fp]) -> Option<(Vec<Fp>, Vec<Fp>)> {
    let denom = trim_poly(denom.to_vec());
    if denom.is_empty() {
        return None;
    }
    let mut rem = trim_poly(numer.to_vec());
    if rem.len() < denom.len() {
        return Some((Vec::new(), rem));
    }
    let mut quotient = vec![fp_zero(); rem.len() - denom.len() + 1];
    let denom_lead_inv = denom.last()?.inverse()?;
    while rem.len() >= denom.len() && !rem.is_empty() {
        let degree_shift = rem.len() - denom.len();
        let factor = *rem.last().unwrap() * denom_lead_inv;
        quotient[degree_shift] = factor;
        for (idx, denom_coeff) in denom.iter().enumerate() {
            let rem_idx = degree_shift + idx;
            rem[rem_idx] -= factor * *denom_coeff;
        }
        rem = trim_poly(rem);
    }
    Some((trim_poly(quotient), rem))
}

#[cfg(feature = "rs_bw")]
pub fn rs_decode_with_errors(points: &[(Fp, Fp)]) -> Result<(Vec<Fp>, Vec<usize>), RsError> {
    let degree_bound = points.len().div_ceil(3).max(1);
    let max_errors = points.len().saturating_sub(degree_bound) / 2;
    let decoded = rs_decode_eval_with_errors(points, degree_bound, fp_zero(), max_errors)?;
    Ok((decoded.coefficients, decoded.bad_indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn barycentric_eval_pair_many_matches_scalar_eval() {
        let z = vec![
            Fp::from(1u64),
            Fp::from(2u64),
            Fp::from(4u64),
            Fp::from(8u64),
        ];
        let y0 = vec![
            Fp::from(3u64),
            Fp::from(5u64),
            Fp::from(9u64),
            Fp::from(17u64),
        ];
        let y1 = vec![
            Fp::from(7u64),
            Fp::from(11u64),
            Fp::from(19u64),
            Fp::from(31u64),
        ];
        let evals = vec![
            Fp::from(2u64),
            Fp::from(3u64),
            Fp::from(6u64),
            Fp::from(8u64),
            Fp::from(13u64),
        ];
        let weights = precompute_barycentric(&z);

        let (many0, many1) = barycentric_eval_pair_many(&weights, &y0, &y1, &evals);

        for (pos, &z_eval) in evals.iter().enumerate() {
            assert_eq!(many0[pos], barycentric_eval(&weights, &y0, z_eval));
            assert_eq!(many1[pos], barycentric_eval(&weights, &y1, z_eval));
        }
    }
}
