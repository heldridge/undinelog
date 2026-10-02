//! Field helpers wrapping the BLS12-381 scalar field.
//!
//! # Overview
//! This module defines the base field used by the protocol and provides a few
//! small helpers that make the math-heavy code easier to read. The field is the
//! scalar field of BLS12-381 (a prime field of order ~2^255).
//!
//! # Mathematical model
//! We treat all values as elements of the finite field `Fp` with:
//! - Addition/multiplication modulo the field prime `p`.
//! - Multiplicative inverse defined for non-zero elements.
//!
//! # Notes / Pitfalls
//! - `batch_invert` assumes *all inputs are non-zero*. If any input is zero,
//!   the implementation returns zero for every entry. Callers should enforce
//!   this precondition to avoid masking errors.

use ark_bls12_381::Fr;
use ark_ff::{Field, One, Zero};

/// Base field element type used throughout the protocol.
pub type Fp = Fr;

/// Return the additive identity (0).
///
/// # Example
/// ```rust,ignore
/// use paper_a_optimized::fp_zero;
/// let z = fp_zero();
/// assert!(z.is_zero());
/// ```
pub fn fp_zero() -> Fp {
    Fp::zero()
}

/// Return the multiplicative identity (1).
///
/// # Example
/// ```rust,ignore
/// use paper_a_optimized::fp_one;
/// let one = fp_one();
/// assert!(one.is_one());
/// ```
pub fn fp_one() -> Fp {
    Fp::one()
}

/// Exponentiation helper for small integer exponents.
///
/// # Mathematical definition
/// Computes `base^exp` in the field, where `exp` is an unsigned integer.
///
/// # Complexity
/// `O(log exp)` field multiplications.
///
/// # Example
/// ```rust,ignore
/// use paper_a_optimized::{fp_one, fp_pow, Fp};
/// let base = Fp::from(7u64);
/// assert_eq!(fp_pow(base, 0), fp_one());
/// assert_eq!(fp_pow(base, 1), base);
/// ```
pub fn fp_pow(base: Fp, exp: u64) -> Fp {
    base.pow([exp])
}

/// Batch inversion using a prefix-product trick.
///
/// # Mathematical definition
/// For a vector `v = [v_0, ..., v_{n-1}]` (all non-zero), this computes:
/// `v_i <- 1 / v_i` for all `i` using one inversion and `O(n)` multiplications.
///
/// The method uses prefix products:
/// - `p_i = v_0 * v_1 * ... * v_{i-1}` (with `p_0 = 1`)
/// - `inv_all = 1 / (v_0 * v_1 * ... * v_{n-1})`
/// - then `v_i^{-1} = p_i * inv_all * (v_{i+1} * ... * v_{n-1})`
///
/// # Preconditions
/// - All entries in `values` must be non-zero. If any entry is zero, the
///   implementation sets **all outputs to zero**.
///
/// # Complexity
/// `O(n)` field multiplications + `O(1)` field inversion.
pub fn batch_invert(values: &mut [Fp]) {
    if values.is_empty() {
        return;
    }
    let mut prod = Vec::with_capacity(values.len());
    let mut acc = Fp::one();
    for v in values.iter() {
        prod.push(acc);
        acc *= v;
    }
    let inv_all = acc.inverse().unwrap_or_else(Fp::zero);
    let mut acc_inv = inv_all;
    for (v, prefix) in values.iter_mut().zip(prod.into_iter()).rev() {
        let tmp = *v;
        *v = acc_inv * prefix;
        acc_inv *= tmp;
    }
}
