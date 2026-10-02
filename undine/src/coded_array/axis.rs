//! Coset-axis arithmetic and public layout rules for the bivariate segment.
//!
//! # Mathematical model
//! One segment at threshold `kappa` holds `kappa^2` cells. They are the values
//! of a polynomial `P(Z, W)` of degree `< kappa` in each variable on a product
//! of two coset axes:
//!
//! ```text
//! Z-axis: z_i   = rho_Z * omega^i,   0 <= i < kappa
//! W-axis: eta_j = rho_W * omega^j,   0 <= j < kappa
//! ```
//!
//! where `omega` is a primitive `kappa`-th root of unity and `rho_Z`, `rho_W`
//! are the public scales of the segment. A claim with column identifier `C`
//! and row label `w`, both off every semantic axis, stores
//!
//! ```text
//! primary column: L_C[j] = P(C, eta_j)
//! auxiliary row:  R_w[i] = P(z_i, w)
//! ```
//!
//! Everything in this module is a function of public layout data only. The
//! segment kernels in `params.rs` and the store logic in `unauth` build on it.
//!
//! # Lagrange basis on a coset
//! For the axis `rho * H_n` and a point `x`, write `xh = x / rho`. The basis
//! polynomial of the axis point `rho * omega^i` is
//!
//! ```text
//! Lambda_i(x) = omega^i * (xh^n - 1) / (n * (xh - omega^i)).
//! ```
//!
//! # Levels
//! Thresholds form a ladder `kappa_l = 2^l * kappa_0`. Segment `b` at level
//! `l + 1` contains the level-`l` segments `4b + 2u + v` for `u, v in {0, 1}`.
//! Child `(u, v)` occupies the half of the parent's Z-axis selected by `u` and
//! the half of its W-axis selected by `v`:
//!
//! ```text
//! rho_Z(child) = rho_Z(parent) * zeta_{2 kappa_l}^u
//! rho_W(child) = rho_W(parent) * zeta_{2 kappa_l}^v
//! ```
//!
//! and a cell keeps the same pair of field points when its segment is merged.

use crate::field::{Fp, batch_invert, fp_one, fp_pow, fp_zero};
use ark_ff::{FftField, Field};
use std::sync::OnceLock;

/// Return `true` when `x` lies in the two-power roots-of-unity subgroup.
///
/// Every semantic axis at every level is a coset of roots of unity scaled by a
/// two-power root of unity, so all semantic axis points lie in this subgroup.
/// Claim assignments are taken outside it and are therefore off every axis at
/// every level.
pub fn is_two_power_root_of_unity(x: Fp) -> bool {
    if x == fp_zero() {
        return false;
    }
    let mut acc = x;
    for _ in 0..<Fp as FftField>::TWO_ADICITY {
        acc.square_in_place();
    }
    acc == fp_one()
}

/// Return `true` when `x` can serve as a column identifier or row label.
pub fn is_valid_assignment(x: Fp) -> bool {
    x != fp_zero() && !is_two_power_root_of_unity(x)
}

/// Primitive root of unity of the given power-of-two order.
pub fn root_of_unity(order: usize) -> Fp {
    <Fp as FftField>::get_root_of_unity(order as u64)
        .expect("field supplies the requested two-adic root")
}

/// The normalized axis `H_n`: `omega^i` for `0 <= i < n`.
pub fn axis_points(n: usize) -> Vec<Fp> {
    let omega = root_of_unity(n);
    let mut points = Vec::with_capacity(n);
    let mut cur = fp_one();
    for _ in 0..n {
        points.push(cur);
        cur *= omega;
    }
    points
}

/// Roots of the scale ladder at threshold `kappa`: entry `t` is
/// `zeta_{2^{t+1} kappa}`, for every depth the field's two-adicity supports.
pub fn scale_ladder_roots(kappa: usize) -> Vec<Fp> {
    debug_assert!(kappa.is_power_of_two());
    let two_adicity = <Fp as FftField>::TWO_ADICITY as usize;
    let kappa_log = kappa.trailing_zeros() as usize;
    (0..two_adicity.saturating_sub(kappa_log))
        .map(|depth| root_of_unity(kappa << (depth + 1)))
        .collect()
}

/// Split a segment index into its Z-scale and W-scale indices.
///
/// The segment index is read in base four from the least significant digit.
/// Digit `t` is `2u + v` for the grouping `t` levels above the current one.
/// The Z-scale index collects the `u` bits and the W-scale index the `v`
/// bits, so a segment's Z scale depends only on the first and its W scale
/// only on the second.
#[inline]
pub fn segment_axis_indices(segment: usize) -> (usize, usize) {
    let mut z_scale_index = 0usize;
    let mut w_scale_index = 0usize;
    let mut digits = segment;
    let mut depth = 0usize;
    while digits != 0 {
        z_scale_index |= ((digits >> 1) & 1) << depth;
        w_scale_index |= (digits & 1) << depth;
        digits >>= 2;
        depth += 1;
    }
    (z_scale_index, w_scale_index)
}

/// Axis scale for one scale index: bit `t` contributes `ladder_roots[t]`.
pub fn axis_scale_from_index(ladder_roots: &[Fp], scale_index: usize) -> Fp {
    let mut rho = fp_one();
    let mut bits = scale_index;
    let mut depth = 0usize;
    while bits != 0 {
        if bits & 1 != 0 {
            assert!(
                depth < ladder_roots.len(),
                "segment axis scale exceeds the field two-adic root depth"
            );
            rho *= ladder_roots[depth];
        }
        bits >>= 1;
        depth += 1;
    }
    rho
}

/// Public Z-axis and W-axis scales of one segment at threshold `kappa`.
///
/// Digit `t` of the segment index contributes `zeta_{2^{t+1} kappa}^u` to the
/// Z scale and `zeta_{2^{t+1} kappa}^v` to the W scale. This is the
/// two-variable form of the factor-two scale ladder: the four children of one
/// parent sit on the even and odd halves of the parent's two axes.
pub fn segment_axis_scales(kappa: usize, segment: usize) -> (Fp, Fp) {
    let ladder_roots = scale_ladder_roots(kappa);
    let (z_scale_index, w_scale_index) = segment_axis_indices(segment);
    (
        axis_scale_from_index(&ladder_roots, z_scale_index),
        axis_scale_from_index(&ladder_roots, w_scale_index),
    )
}

/// Split one flat grid index into its `(z_index, w_index)` pair.
#[inline]
pub fn grid_index_to_axes(kappa: usize, grid_index: usize) -> (usize, usize) {
    (grid_index / kappa, grid_index % kappa)
}

/// Combine a `(z_index, w_index)` pair into the flat grid index.
#[inline]
pub fn axes_to_grid_index(kappa: usize, z_index: usize, w_index: usize) -> usize {
    z_index * kappa + w_index
}

/// Canonical public-logical to grid permutation at threshold `kappa` for a
/// ladder whose base threshold is `base_kappa`.
///
/// Write a logical offset as `q = c * base_kappa^2 + r` with
/// `r = i0 * base_kappa + j0`, and write `c` in base four with digits
/// `2 u_t + v_t`, most significant first applied last. With
/// `level = log2(kappa / base_kappa)` the cell sits at
///
/// ```text
/// z_index = i0 * 2^level + sum_t u_t * 2^(level - 1 - t)
/// w_index = j0 * 2^level + sum_t v_t * 2^(level - 1 - t)
/// ```
///
/// which is the closed form of the recursion
/// `pi_{l+1}((2u + v) * Seg_l + q) = (2 pi^Z_l(q) + u, 2 pi^W_l(q) + v)`.
pub fn canonical_logical_to_grid_index(kappa: usize, base_kappa: usize) -> Vec<usize> {
    assert!(kappa.is_power_of_two());
    assert!(base_kappa.is_power_of_two());
    assert!(kappa >= base_kappa);
    let level = (kappa / base_kappa).trailing_zeros() as usize;
    let base_cells = base_kappa * base_kappa;
    let cells = kappa * kappa;
    let mut layout = Vec::with_capacity(cells);
    for logical_offset in 0..cells {
        let child = logical_offset / base_cells;
        let rem = logical_offset % base_cells;
        let mut z_index = rem / base_kappa;
        let mut w_index = rem % base_kappa;
        // Digit `t` of `child` is the grouping applied at level `t + 1`; each
        // application appends one low bit to both axis indices.
        for t in 0..level {
            let digit = (child >> (2 * t)) & 3;
            z_index = 2 * z_index + (digit >> 1);
            w_index = 2 * w_index + (digit & 1);
        }
        layout.push(axes_to_grid_index(kappa, z_index, w_index));
    }
    layout
}

/// Lagrange basis of the coset axis `rho * H_n` evaluated at `x`.
///
/// `axis` holds the normalized points `omega^i`, `inv_n = 1 / n`, and
/// `inv_rho = 1 / rho`. On return `out[i] = Lambda_i(x)`. If `x` is itself an
/// axis point the result is the corresponding unit vector.
///
/// # Complexity
/// `O(n)` multiplications and one batch inversion.
pub fn coset_lagrange_basis_at(axis: &[Fp], inv_n: Fp, inv_rho: Fp, x: Fp, out: &mut [Fp]) {
    let n = axis.len();
    debug_assert_eq!(out.len(), n);
    let xh = x * inv_rho;
    let vanishing = fp_pow(xh, n as u64) - fp_one();
    if vanishing == fp_zero() {
        for (slot, point) in out.iter_mut().zip(axis.iter()) {
            *slot = if *point == xh { fp_one() } else { fp_zero() };
        }
        return;
    }
    for (slot, point) in out.iter_mut().zip(axis.iter()) {
        *slot = xh - *point;
    }
    batch_invert(out);
    let scale = vanishing * inv_n;
    for (slot, point) in out.iter_mut().zip(axis.iter()) {
        *slot *= *point * scale;
    }
}

/// Selected entries of the coset Lagrange basis at an off-axis point `x`.
///
/// `out[k] = Lambda_{indices[k]}(x)`. This is the sparse form used when only a
/// few cells of a segment change.
///
/// # Preconditions
/// `x` is not a point of the axis.
pub fn coset_lagrange_basis_entries_at(
    axis: &[Fp],
    inv_n: Fp,
    inv_rho: Fp,
    x: Fp,
    indices: &[usize],
    out: &mut [Fp],
) {
    debug_assert_eq!(out.len(), indices.len());
    let n = axis.len();
    let xh = x * inv_rho;
    let scale = (fp_pow(xh, n as u64) - fp_one()) * inv_n;
    debug_assert!(scale != fp_zero(), "assignment lies on a semantic axis");
    for (slot, &index) in out.iter_mut().zip(indices.iter()) {
        *slot = xh - axis[index];
    }
    batch_invert(out);
    for (slot, &index) in out.iter_mut().zip(indices.iter()) {
        *slot *= axis[index] * scale;
    }
}

/// Evaluate at `x` the polynomial of degree `< n` given by its values on the
/// coset axis `rho * H_n`.
pub fn evaluate_from_axis(axis: &[Fp], inv_n: Fp, inv_rho: Fp, values: &[Fp], x: Fp) -> Fp {
    debug_assert_eq!(values.len(), axis.len());
    let mut basis = vec![fp_zero(); axis.len()];
    coset_lagrange_basis_at(axis, inv_n, inv_rho, x, &mut basis);
    dot(values, &basis)
}

/// Inner product of two equally long field vectors.
#[inline]
pub fn dot(left: &[Fp], right: &[Fp]) -> Fp {
    debug_assert_eq!(left.len(), right.len());
    let mut acc = fp_zero();
    for (a, b) in left.iter().zip(right.iter()) {
        acc += *a * *b;
    }
    acc
}

/// The two selector values of one axis at an assignment `x`.
///
/// For a parent axis `rho * H_{2n}` whose halves are the child axes
/// `rho * xi^u * H_n`, the selector of half `u` is
/// `(1 + (-1)^u (x / rho)^n) / 2`. It is the indicator of that half on the
/// parent axis and has degree `n`.
pub fn axis_selectors(n: usize, inv_rho: Fp, x: Fp) -> (Fp, Fp) {
    static HALF: OnceLock<Fp> = OnceLock::new();
    let half = *HALF.get_or_init(|| {
        Fp::from(2u64)
            .inverse()
            .expect("field characteristic is odd")
    });
    let power = fp_pow(x * inv_rho, n as u64);
    ((fp_one() + power) * half, (fp_one() - power) * half)
}

/// Inverses of the two selector values of an off-axis assignment, with one
/// field inversion.
///
/// # Preconditions
/// Both selectors are non-zero, which holds for every assignment off the
/// semantic axes.
pub fn invert_selectors(selectors: (Fp, Fp)) -> (Fp, Fp) {
    let (s0, s1) = selectors;
    let inv_product = (s0 * s1)
        .inverse()
        .expect("assignment lies off the semantic axes");
    (inv_product * s1, inv_product * s0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lagrange basis by the defining product formula, used as an independent
    /// check of the closed form.
    fn lagrange_basis_slow(points: &[Fp], x: Fp) -> Vec<Fp> {
        (0..points.len())
            .map(|i| {
                let mut num = fp_one();
                let mut den = fp_one();
                for k in 0..points.len() {
                    if k != i {
                        num *= x - points[k];
                        den *= points[i] - points[k];
                    }
                }
                num * den.inverse().unwrap()
            })
            .collect()
    }

    #[test]
    fn closed_form_basis_matches_product_formula() {
        for &n in &[2usize, 4, 8, 16] {
            let axis = axis_points(n);
            let inv_n = Fp::from(n as u64).inverse().unwrap();
            for &rho in &[fp_one(), root_of_unity(2 * n), root_of_unity(8 * n)] {
                let inv_rho = rho.inverse().unwrap();
                let scaled: Vec<Fp> = axis.iter().map(|point| rho * *point).collect();
                for &x in &[Fp::from(3u64), Fp::from(7u64), Fp::from(1234567u64)] {
                    assert!(is_valid_assignment(x));
                    let mut fast = vec![fp_zero(); n];
                    coset_lagrange_basis_at(&axis, inv_n, inv_rho, x, &mut fast);
                    assert_eq!(fast, lagrange_basis_slow(&scaled, x));

                    let indices: Vec<usize> = (0..n).step_by(2).collect();
                    let mut sparse = vec![fp_zero(); indices.len()];
                    coset_lagrange_basis_entries_at(
                        &axis,
                        inv_n,
                        inv_rho,
                        x,
                        &indices,
                        &mut sparse,
                    );
                    for (value, &index) in sparse.iter().zip(indices.iter()) {
                        assert_eq!(*value, fast[index]);
                    }
                }
            }
        }
    }

    #[test]
    fn basis_on_axis_point_is_unit_vector() {
        let n = 8usize;
        let axis = axis_points(n);
        let inv_n = Fp::from(n as u64).inverse().unwrap();
        let rho = root_of_unity(4 * n);
        let inv_rho = rho.inverse().unwrap();
        for k in 0..n {
            let mut basis = vec![fp_zero(); n];
            coset_lagrange_basis_at(&axis, inv_n, inv_rho, rho * axis[k], &mut basis);
            for (i, value) in basis.iter().enumerate() {
                assert_eq!(*value, if i == k { fp_one() } else { fp_zero() });
            }
        }
    }

    #[test]
    fn small_integers_are_valid_assignments() {
        for cand in 2u64..200 {
            assert!(is_valid_assignment(Fp::from(cand)), "candidate {cand}");
        }
        assert!(!is_valid_assignment(fp_zero()));
        assert!(!is_valid_assignment(fp_one()));
        assert!(!is_valid_assignment(root_of_unity(1 << 20)));
    }

    #[test]
    fn child_axes_are_halves_of_the_parent_axes() {
        for &kappa in &[2usize, 4, 8] {
            let child_axis = axis_points(kappa);
            let parent_axis = axis_points(2 * kappa);
            for parent in 0..6usize {
                let (parent_z, parent_w) = segment_axis_scales(2 * kappa, parent);
                for u in 0..2usize {
                    for v in 0..2usize {
                        let child = 4 * parent + 2 * u + v;
                        let (child_z, child_w) = segment_axis_scales(kappa, child);
                        for i in 0..kappa {
                            assert_eq!(child_z * child_axis[i], parent_z * parent_axis[2 * i + u]);
                            assert_eq!(child_w * child_axis[i], parent_w * parent_axis[2 * i + v]);
                        }
                    }
                }
            }
        }
    }

    /// The recursion from the layout definition, applied literally.
    fn layout_by_recursion(kappa: usize, base_kappa: usize) -> Vec<(usize, usize)> {
        if kappa == base_kappa {
            return (0..kappa * kappa)
                .map(|q| (q / base_kappa, q % base_kappa))
                .collect();
        }
        let child = layout_by_recursion(kappa / 2, base_kappa);
        let child_cells = (kappa / 2) * (kappa / 2);
        let mut layout = Vec::with_capacity(kappa * kappa);
        for digit in 0..4usize {
            let (u, v) = (digit >> 1, digit & 1);
            for q in 0..child_cells {
                layout.push((2 * child[q].0 + u, 2 * child[q].1 + v));
            }
        }
        layout
    }

    #[test]
    fn canonical_layout_matches_the_recursion_and_is_a_bijection() {
        for &base_kappa in &[2usize, 4] {
            for level in 0..4usize {
                let kappa = base_kappa << level;
                let layout = canonical_logical_to_grid_index(kappa, base_kappa);
                let expected = layout_by_recursion(kappa, base_kappa);
                assert_eq!(layout.len(), kappa * kappa);
                let mut seen = vec![false; kappa * kappa];
                for (grid, (z_index, w_index)) in layout.iter().zip(expected.iter()) {
                    assert_eq!(*grid, axes_to_grid_index(kappa, *z_index, *w_index));
                    assert!(!seen[*grid]);
                    seen[*grid] = true;
                }
            }
        }
    }

    #[test]
    fn merged_cells_keep_their_field_points() {
        let base_kappa = 2usize;
        for level in 0..3usize {
            let kappa = base_kappa << level;
            let child_layout = canonical_logical_to_grid_index(kappa, base_kappa);
            let parent_layout = canonical_logical_to_grid_index(2 * kappa, base_kappa);
            let child_axis = axis_points(kappa);
            let parent_axis = axis_points(2 * kappa);
            let cells = kappa * kappa;
            for parent in 0..3usize {
                let (parent_z, parent_w) = segment_axis_scales(2 * kappa, parent);
                for digit in 0..4usize {
                    let child = 4 * parent + digit;
                    let (child_z, child_w) = segment_axis_scales(kappa, child);
                    for q in 0..cells {
                        let (ci, cj) = grid_index_to_axes(kappa, child_layout[q]);
                        let (pi, pj) =
                            grid_index_to_axes(2 * kappa, parent_layout[digit * cells + q]);
                        assert_eq!(child_z * child_axis[ci], parent_z * parent_axis[pi]);
                        assert_eq!(child_w * child_axis[cj], parent_w * parent_axis[pj]);
                    }
                }
            }
        }
    }

    #[test]
    fn selectors_are_indicators_of_the_two_halves() {
        let kappa = 4usize;
        let parent_axis = axis_points(2 * kappa);
        let rho = root_of_unity(16 * kappa);
        let inv_rho = rho.inverse().unwrap();
        for (index, point) in parent_axis.iter().enumerate() {
            let (e0, e1) = axis_selectors(kappa, inv_rho, rho * *point);
            if index % 2 == 0 {
                assert_eq!((e0, e1), (fp_one(), fp_zero()));
            } else {
                assert_eq!((e0, e1), (fp_zero(), fp_one()));
            }
        }
        let (e0, e1) = axis_selectors(kappa, inv_rho, Fp::from(5u64));
        assert_eq!(e0 + e1, fp_one());
        assert!(e0 != fp_zero() && e1 != fp_zero());
    }
}
