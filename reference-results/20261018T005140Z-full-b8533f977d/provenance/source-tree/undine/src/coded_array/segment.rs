//! Segment kernels of the bivariate layout: encoding, the four-segment stitch,
//! and its inverse.
//!
//! All functions here are pure and work on one claim and one segment (or one
//! group of four segments). Cells of a segment are given in grid order,
//! `cells[z_index * kappa + w_index] = P(z_{z_index}, eta_{w_index})`.
//!
//! # Encoding
//! With `basis_z[i] = Lambda^Z_i(C)` and `basis_w[j] = Lambda^W_j(w)`:
//!
//! ```text
//! L_C[j] = P(C, eta_j) = sum_i cells[i, j] * basis_z[i]
//! R_w[i] = P(z_i, w)   = sum_j cells[i, j] * basis_w[j]
//! ```
//!
//! # Stitch
//! Four segments `P_uv` (segment `2u + v` of the group) merge into
//! `P_T = sum_{u,v} e_u(Z) f_v(W) P_uv`. For a claim at `(C, w)`:
//!
//! ```text
//! L_T[2i + v] = e_0(C) * L_0v[i] + e_1(C) * L_1v[i]
//! R_T[2i + u] = f_0(w) * R_u0[i] + f_1(w) * R_u1[i]
//! ```
//!
//! # Undoing the stitch
//! Given the stitched state and the source state of the two diagonal segments
//! `(0, 0)` and `(1, 1)`, the other two follow entry by entry:
//!
//! ```text
//! L_10[i] = (L_T[2i]     - e_0(C) * L_00[i]) / e_1(C)
//! L_01[i] = (L_T[2i + 1] - e_1(C) * L_11[i]) / e_0(C)
//! R_01[i] = (R_T[2i]     - f_0(w) * R_00[i]) / f_1(w)
//! R_10[i] = (R_T[2i + 1] - f_1(w) * R_11[i]) / f_0(w)
//! ```

//!
//! Given instead the two source columns with `u = 0` and the two source rows
//! with `v = 0` (the half that maintainers at the higher threshold can supply
//! through partial stitches), the remaining source state is:
//!
//! ```text
//! L_1v[i] = (L_T[2i + v] - e_0(C) * L_0v[i]) / e_1(C)
//! R_u1[i] = (R_T[2i + u] - f_0(w) * R_u0[i]) / f_1(w)
//! ```

use crate::field::{Fp, fp_zero};

/// Position of child `(u, v)` inside its group of four.
#[inline]
pub fn child_position(u: usize, v: usize) -> usize {
    2 * u + v
}

/// Encode one claim's primary column from the cells of one segment.
///
/// `out[j] = sum_i cells[i * kappa + j] * basis_z[i]`.
pub fn encode_primary_column(cells: &[Fp], basis_z: &[Fp], out: &mut [Fp]) {
    let kappa = basis_z.len();
    debug_assert_eq!(cells.len(), kappa * kappa);
    debug_assert_eq!(out.len(), kappa);
    for slot in out.iter_mut() {
        *slot = fp_zero();
    }
    for (row, weight) in cells.chunks_exact(kappa).zip(basis_z.iter()) {
        for (slot, cell) in out.iter_mut().zip(row.iter()) {
            *slot += *cell * *weight;
        }
    }
}

/// Encode one claim's auxiliary row from the cells of one segment.
///
/// `out[i] = sum_j cells[i * kappa + j] * basis_w[j]`.
pub fn encode_auxiliary_row(cells: &[Fp], basis_w: &[Fp], out: &mut [Fp]) {
    let kappa = basis_w.len();
    debug_assert_eq!(cells.len(), kappa * kappa);
    debug_assert_eq!(out.len(), kappa);
    for (slot, row) in out.iter_mut().zip(cells.chunks_exact(kappa)) {
        let mut acc = fp_zero();
        for (cell, weight) in row.iter().zip(basis_w.iter()) {
            acc += *cell * *weight;
        }
        *slot = acc;
    }
}

/// Stitch the primary columns of four source segments into the target column.
///
/// `children[2u + v]` is the source column of segment `(u, v)`, or `None` for
/// an entirely padded segment, whose column is zero. `selectors` are
/// `(e_0(C), e_1(C))` for the parent's Z-axis.
pub fn stitch_primary_column(selectors: (Fp, Fp), children: [Option<&[Fp]>; 4], out: &mut [Fp]) {
    let kappa = out.len() / 2;
    debug_assert_eq!(out.len(), 2 * kappa);
    let (e0, e1) = selectors;
    for v in 0..2 {
        let upper = children[child_position(0, v)];
        let lower = children[child_position(1, v)];
        for i in 0..kappa {
            let mut acc = fp_zero();
            if let Some(column) = upper {
                acc += e0 * column[i];
            }
            if let Some(column) = lower {
                acc += e1 * column[i];
            }
            out[2 * i + v] = acc;
        }
    }
}

/// Stitch the auxiliary rows of four source segments into the target row.
///
/// `children[2u + v]` is the source row of segment `(u, v)`, or `None` for an
/// entirely padded segment. `selectors` are `(f_0(w), f_1(w))` for the
/// parent's W-axis.
pub fn stitch_auxiliary_row(selectors: (Fp, Fp), children: [Option<&[Fp]>; 4], out: &mut [Fp]) {
    let kappa = out.len() / 2;
    debug_assert_eq!(out.len(), 2 * kappa);
    let (f0, f1) = selectors;
    for u in 0..2 {
        let left = children[child_position(u, 0)];
        let right = children[child_position(u, 1)];
        for i in 0..kappa {
            let mut acc = fp_zero();
            if let Some(row) = left {
                acc += f0 * row[i];
            }
            if let Some(row) = right {
                acc += f1 * row[i];
            }
            out[2 * i + u] = acc;
        }
    }
}

/// Recover the primary columns of segments `(0, 1)` and `(1, 0)` from the
/// stitched column and the columns of the diagonal segments `(0, 0)` and
/// `(1, 1)`.
///
/// `inverse_selectors` are `(1 / e_0(C), 1 / e_1(C))`; they exist because the
/// column identifier lies off the semantic Z-axes. Returns
/// `(column_01, column_10)`.
pub fn unstitch_primary_columns(
    selectors: (Fp, Fp),
    inverse_selectors: (Fp, Fp),
    stitched: &[Fp],
    column_00: &[Fp],
    column_11: &[Fp],
) -> (Vec<Fp>, Vec<Fp>) {
    let kappa = stitched.len() / 2;
    debug_assert_eq!(column_00.len(), kappa);
    debug_assert_eq!(column_11.len(), kappa);
    let (e0, e1) = selectors;
    let (inv_e0, inv_e1) = inverse_selectors;
    let mut column_01 = Vec::with_capacity(kappa);
    let mut column_10 = Vec::with_capacity(kappa);
    for i in 0..kappa {
        column_10.push((stitched[2 * i] - e0 * column_00[i]) * inv_e1);
        column_01.push((stitched[2 * i + 1] - e1 * column_11[i]) * inv_e0);
    }
    (column_01, column_10)
}

/// Recover the auxiliary rows of segments `(0, 1)` and `(1, 0)` from the
/// stitched row and the rows of the diagonal segments `(0, 0)` and `(1, 1)`.
///
/// `inverse_selectors` are `(1 / f_0(w), 1 / f_1(w))`; they exist because the
/// row label lies off the semantic W-axes. Returns `(row_01, row_10)`.
pub fn unstitch_auxiliary_rows(
    selectors: (Fp, Fp),
    inverse_selectors: (Fp, Fp),
    stitched: &[Fp],
    row_00: &[Fp],
    row_11: &[Fp],
) -> (Vec<Fp>, Vec<Fp>) {
    let kappa = stitched.len() / 2;
    debug_assert_eq!(row_00.len(), kappa);
    debug_assert_eq!(row_11.len(), kappa);
    let (f0, f1) = selectors;
    let (inv_f0, inv_f1) = inverse_selectors;
    let mut row_01 = Vec::with_capacity(kappa);
    let mut row_10 = Vec::with_capacity(kappa);
    for i in 0..kappa {
        row_01.push((stitched[2 * i] - f0 * row_00[i]) * inv_f1);
        row_10.push((stitched[2 * i + 1] - f1 * row_11[i]) * inv_f0);
    }
    (row_01, row_10)
}

/// Recover the primary columns of segments `(1, 0)` and `(1, 1)` from the
/// stitched column and the columns of the leading segments `(0, 0)` and
/// `(0, 1)`.
///
/// Returns `(column_10, column_11)`.
pub fn unstitch_primary_columns_from_leading(
    selectors: (Fp, Fp),
    inverse_selectors: (Fp, Fp),
    stitched: &[Fp],
    column_00: &[Fp],
    column_01: &[Fp],
) -> (Vec<Fp>, Vec<Fp>) {
    let kappa = stitched.len() / 2;
    debug_assert_eq!(column_00.len(), kappa);
    debug_assert_eq!(column_01.len(), kappa);
    let (e0, _) = selectors;
    let (_, inv_e1) = inverse_selectors;
    let mut column_10 = Vec::with_capacity(kappa);
    let mut column_11 = Vec::with_capacity(kappa);
    for i in 0..kappa {
        column_10.push((stitched[2 * i] - e0 * column_00[i]) * inv_e1);
        column_11.push((stitched[2 * i + 1] - e0 * column_01[i]) * inv_e1);
    }
    (column_10, column_11)
}

/// Recover the auxiliary rows of segments `(0, 1)` and `(1, 1)` from the
/// stitched row and the rows of the leading segments `(0, 0)` and `(1, 0)`.
///
/// Returns `(row_01, row_11)`.
pub fn unstitch_auxiliary_rows_from_leading(
    selectors: (Fp, Fp),
    inverse_selectors: (Fp, Fp),
    stitched: &[Fp],
    row_00: &[Fp],
    row_10: &[Fp],
) -> (Vec<Fp>, Vec<Fp>) {
    let kappa = stitched.len() / 2;
    debug_assert_eq!(row_00.len(), kappa);
    debug_assert_eq!(row_10.len(), kappa);
    let (f0, _) = selectors;
    let (_, inv_f1) = inverse_selectors;
    let mut row_01 = Vec::with_capacity(kappa);
    let mut row_11 = Vec::with_capacity(kappa);
    for i in 0..kappa {
        row_01.push((stitched[2 * i] - f0 * row_00[i]) * inv_f1);
        row_11.push((stitched[2 * i + 1] - f0 * row_10[i]) * inv_f1);
    }
    (row_01, row_11)
}

/// Evaluate a partial stitch held inside a stitched fragment.
///
/// The even-indexed entries of a stitched auxiliary row are the values of
/// `S_0(Z, w) = f_0(w) P_00(Z, w) + f_1(w) P_01(Z, w)` on the first half of the
/// parent's Z-axis, and the even-indexed entries of a stitched primary column
/// are the values of `S'_0(C, W) = e_0(C) P_00(C, W) + e_1(C) P_10(C, W)` on
/// the first half of the parent's W-axis. `half_basis` is the Lagrange basis
/// of that half axis at the requested point.
pub fn evaluate_partial_stitch(stitched: &[Fp], half_basis: &[Fp]) -> Fp {
    debug_assert_eq!(stitched.len(), 2 * half_basis.len());
    let mut acc = fp_zero();
    for (value, weight) in stitched.iter().step_by(2).zip(half_basis.iter()) {
        acc += *value * *weight;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coded_array::axis::{
        axis_points, axis_selectors, coset_lagrange_basis_at, invert_selectors, segment_axis_scales,
    };
    use crate::field::fp_one;
    use ark_ff::Field;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    struct Fragment {
        column: Vec<Fp>,
        row: Vec<Fp>,
    }

    fn random_cells(rng: &mut ChaCha20Rng, count: usize) -> Vec<Fp> {
        (0..count).map(|_| Fp::from(rng.r#gen::<u64>())).collect()
    }

    /// Slow reference: evaluate the bivariate interpolant of `cells` on the
    /// product grid at `(x, y)` with the product-formula Lagrange basis.
    fn reference_eval(cells: &[Fp], z_axis: &[Fp], w_axis: &[Fp], x: Fp, y: Fp) -> Fp {
        let basis = |points: &[Fp], at: Fp| -> Vec<Fp> {
            (0..points.len())
                .map(|i| {
                    let mut num = fp_one();
                    let mut den = fp_one();
                    for k in 0..points.len() {
                        if k != i {
                            num *= at - points[k];
                            den *= points[i] - points[k];
                        }
                    }
                    num * den.inverse().unwrap()
                })
                .collect()
        };
        let bz = basis(z_axis, x);
        let bw = basis(w_axis, y);
        let kappa = z_axis.len();
        let mut acc = fp_zero();
        for i in 0..kappa {
            for j in 0..kappa {
                acc += cells[i * kappa + j] * bz[i] * bw[j];
            }
        }
        acc
    }

    fn scaled_axes(kappa: usize, segment: usize) -> (Vec<Fp>, Vec<Fp>) {
        let (rho_z, rho_w) = segment_axis_scales(kappa, segment);
        let axis = axis_points(kappa);
        (
            axis.iter().map(|point| rho_z * *point).collect(),
            axis.iter().map(|point| rho_w * *point).collect(),
        )
    }

    fn encode(kappa: usize, segment: usize, cells: &[Fp], c: Fp, w: Fp) -> Fragment {
        let axis = axis_points(kappa);
        let inv_kappa = Fp::from(kappa as u64).inverse().unwrap();
        let (rho_z, rho_w) = segment_axis_scales(kappa, segment);
        let mut basis_z = vec![fp_zero(); kappa];
        let mut basis_w = vec![fp_zero(); kappa];
        coset_lagrange_basis_at(&axis, inv_kappa, rho_z.inverse().unwrap(), c, &mut basis_z);
        coset_lagrange_basis_at(&axis, inv_kappa, rho_w.inverse().unwrap(), w, &mut basis_w);
        let mut column = vec![fp_zero(); kappa];
        let mut row = vec![fp_zero(); kappa];
        encode_primary_column(cells, &basis_z, &mut column);
        encode_auxiliary_row(cells, &basis_w, &mut row);
        Fragment { column, row }
    }

    #[test]
    fn encoding_matches_the_reference_interpolant() {
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let (c, w) = (Fp::from(11u64), Fp::from(29u64));
        for &kappa in &[2usize, 4, 8] {
            for segment in [0usize, 1, 6, 13] {
                let cells = random_cells(&mut rng, kappa * kappa);
                let (z_axis, w_axis) = scaled_axes(kappa, segment);
                let fragment = encode(kappa, segment, &cells, c, w);
                for j in 0..kappa {
                    assert_eq!(
                        fragment.column[j],
                        reference_eval(&cells, &z_axis, &w_axis, c, w_axis[j])
                    );
                }
                for i in 0..kappa {
                    assert_eq!(
                        fragment.row[i],
                        reference_eval(&cells, &z_axis, &w_axis, z_axis[i], w)
                    );
                }
            }
        }
    }

    /// Cells of the parent segment in grid order, built from the four children
    /// by the rule that child `(u, v)` cell `(i, j)` sits at parent cell
    /// `(2i + u, 2j + v)`.
    fn merge_cells(kappa: usize, children: &[Vec<Fp>; 4]) -> Vec<Fp> {
        let parent_kappa = 2 * kappa;
        let mut parent = vec![fp_zero(); parent_kappa * parent_kappa];
        for u in 0..2 {
            for v in 0..2 {
                let child = &children[child_position(u, v)];
                for i in 0..kappa {
                    for j in 0..kappa {
                        parent[(2 * i + u) * parent_kappa + (2 * j + v)] = child[i * kappa + j];
                    }
                }
            }
        }
        parent
    }

    fn parent_selectors(kappa: usize, parent: usize, c: Fp, w: Fp) -> ((Fp, Fp), (Fp, Fp)) {
        let (rho_z, rho_w) = segment_axis_scales(2 * kappa, parent);
        (
            axis_selectors(kappa, rho_z.inverse().unwrap(), c),
            axis_selectors(kappa, rho_w.inverse().unwrap(), w),
        )
    }

    #[test]
    fn stitch_equals_direct_encoding_of_the_merged_segment() {
        let mut rng = ChaCha20Rng::seed_from_u64(11);
        let (c, w) = (Fp::from(5u64), Fp::from(101u64));
        for &kappa in &[2usize, 4, 8] {
            for parent in [0usize, 1, 3, 9] {
                let children: [Vec<Fp>; 4] =
                    std::array::from_fn(|_| random_cells(&mut rng, kappa * kappa));
                let fragments: Vec<Fragment> = (0..4)
                    .map(|pos| encode(kappa, 4 * parent + pos, &children[pos], c, w))
                    .collect();
                let (e, f) = parent_selectors(kappa, parent, c, w);

                let mut column = vec![fp_zero(); 2 * kappa];
                let mut row = vec![fp_zero(); 2 * kappa];
                stitch_primary_column(
                    e,
                    std::array::from_fn(|pos| Some(fragments[pos].column.as_slice())),
                    &mut column,
                );
                stitch_auxiliary_row(
                    f,
                    std::array::from_fn(|pos| Some(fragments[pos].row.as_slice())),
                    &mut row,
                );

                let merged = merge_cells(kappa, &children);
                let direct = encode(2 * kappa, parent, &merged, c, w);
                assert_eq!(column, direct.column);
                assert_eq!(row, direct.row);
            }
        }
    }

    #[test]
    fn stitch_treats_missing_children_as_zero_padding() {
        let mut rng = ChaCha20Rng::seed_from_u64(13);
        let (c, w) = (Fp::from(17u64), Fp::from(19u64));
        let kappa = 4usize;
        let parent = 2usize;
        let present = random_cells(&mut rng, kappa * kappa);
        let zero = vec![fp_zero(); kappa * kappa];
        let children = [present.clone(), zero.clone(), zero.clone(), zero];
        let fragment = encode(kappa, 4 * parent, &present, c, w);
        let (e, f) = parent_selectors(kappa, parent, c, w);

        let mut column = vec![fp_zero(); 2 * kappa];
        let mut row = vec![fp_zero(); 2 * kappa];
        stitch_primary_column(
            e,
            [Some(fragment.column.as_slice()), None, None, None],
            &mut column,
        );
        stitch_auxiliary_row(
            f,
            [Some(fragment.row.as_slice()), None, None, None],
            &mut row,
        );
        let direct = encode(2 * kappa, parent, &merge_cells(kappa, &children), c, w);
        assert_eq!(column, direct.column);
        assert_eq!(row, direct.row);
    }

    #[test]
    fn unstitch_recovers_the_off_diagonal_segments() {
        let mut rng = ChaCha20Rng::seed_from_u64(17);
        let (c, w) = (Fp::from(23u64), Fp::from(31u64));
        for &kappa in &[2usize, 4, 8] {
            let parent = 5usize;
            let children: [Vec<Fp>; 4] =
                std::array::from_fn(|_| random_cells(&mut rng, kappa * kappa));
            let fragments: Vec<Fragment> = (0..4)
                .map(|pos| encode(kappa, 4 * parent + pos, &children[pos], c, w))
                .collect();
            let (e, f) = parent_selectors(kappa, parent, c, w);
            let mut column = vec![fp_zero(); 2 * kappa];
            let mut row = vec![fp_zero(); 2 * kappa];
            stitch_primary_column(
                e,
                std::array::from_fn(|pos| Some(fragments[pos].column.as_slice())),
                &mut column,
            );
            stitch_auxiliary_row(
                f,
                std::array::from_fn(|pos| Some(fragments[pos].row.as_slice())),
                &mut row,
            );

            let (column_01, column_10) = unstitch_primary_columns(
                e,
                invert_selectors(e),
                &column,
                &fragments[child_position(0, 0)].column,
                &fragments[child_position(1, 1)].column,
            );
            let (row_01, row_10) = unstitch_auxiliary_rows(
                f,
                invert_selectors(f),
                &row,
                &fragments[child_position(0, 0)].row,
                &fragments[child_position(1, 1)].row,
            );
            assert_eq!(column_01, fragments[child_position(0, 1)].column);
            assert_eq!(column_10, fragments[child_position(1, 0)].column);
            assert_eq!(row_01, fragments[child_position(0, 1)].row);
            assert_eq!(row_10, fragments[child_position(1, 0)].row);
        }
    }

    #[test]
    fn unstitch_from_leading_recovers_the_trailing_segments() {
        let mut rng = ChaCha20Rng::seed_from_u64(23);
        let (c, w) = (Fp::from(29u64), Fp::from(37u64));
        for &kappa in &[2usize, 4, 8] {
            let parent = 6usize;
            let children: [Vec<Fp>; 4] =
                std::array::from_fn(|_| random_cells(&mut rng, kappa * kappa));
            let fragments: Vec<Fragment> = (0..4)
                .map(|pos| encode(kappa, 4 * parent + pos, &children[pos], c, w))
                .collect();
            let (e, f) = parent_selectors(kappa, parent, c, w);
            let mut column = vec![fp_zero(); 2 * kappa];
            let mut row = vec![fp_zero(); 2 * kappa];
            stitch_primary_column(
                e,
                std::array::from_fn(|pos| Some(fragments[pos].column.as_slice())),
                &mut column,
            );
            stitch_auxiliary_row(
                f,
                std::array::from_fn(|pos| Some(fragments[pos].row.as_slice())),
                &mut row,
            );

            let (column_10, column_11) = unstitch_primary_columns_from_leading(
                e,
                invert_selectors(e),
                &column,
                &fragments[child_position(0, 0)].column,
                &fragments[child_position(0, 1)].column,
            );
            let (row_01, row_11) = unstitch_auxiliary_rows_from_leading(
                f,
                invert_selectors(f),
                &row,
                &fragments[child_position(0, 0)].row,
                &fragments[child_position(1, 0)].row,
            );
            assert_eq!(column_10, fragments[child_position(1, 0)].column);
            assert_eq!(column_11, fragments[child_position(1, 1)].column);
            assert_eq!(row_01, fragments[child_position(0, 1)].row);
            assert_eq!(row_11, fragments[child_position(1, 1)].row);
        }
    }

    /// The appendix procedure: helpers at the higher threshold evaluate
    /// partial stitches, and the requester decodes its two leading source
    /// columns and its two leading source rows from `2 kappa` replies.
    #[test]
    fn partial_stitch_evaluations_decode_the_leading_source_half() {
        let mut rng = ChaCha20Rng::seed_from_u64(29);
        let (c, w) = (Fp::from(41u64), Fp::from(43u64));
        for &kappa in &[2usize, 4] {
            let parent = 3usize;
            let children: [Vec<Fp>; 4] =
                std::array::from_fn(|_| random_cells(&mut rng, kappa * kappa));
            let requester: Vec<Fragment> = (0..4)
                .map(|pos| encode(kappa, 4 * parent + pos, &children[pos], c, w))
                .collect();
            let (parent_rho_z, parent_rho_w) = segment_axis_scales(2 * kappa, parent);
            let half_axis = axis_points(kappa);
            let inv_kappa = Fp::from(kappa as u64).inverse().unwrap();

            // 2 kappa helpers, each holding one stitched column and one
            // stitched row, reply with (a_h, b_h).
            let mut samples_a = Vec::new();
            let mut samples_b = Vec::new();
            for helper in 0..2 * kappa {
                let c_h = Fp::from(100 + 7 * helper as u64);
                let w_h = Fp::from(1000 + 11 * helper as u64);
                let fragments: Vec<Fragment> = (0..4)
                    .map(|pos| encode(kappa, 4 * parent + pos, &children[pos], c_h, w_h))
                    .collect();
                let (e, f) = parent_selectors(kappa, parent, c_h, w_h);
                let mut column = vec![fp_zero(); 2 * kappa];
                let mut row = vec![fp_zero(); 2 * kappa];
                stitch_primary_column(
                    e,
                    std::array::from_fn(|pos| Some(fragments[pos].column.as_slice())),
                    &mut column,
                );
                stitch_auxiliary_row(
                    f,
                    std::array::from_fn(|pos| Some(fragments[pos].row.as_slice())),
                    &mut row,
                );
                let mut basis = vec![fp_zero(); kappa];
                coset_lagrange_basis_at(
                    &half_axis,
                    inv_kappa,
                    parent_rho_z.inverse().unwrap(),
                    c,
                    &mut basis,
                );
                samples_a.push((w_h, evaluate_partial_stitch(&row, &basis)));
                coset_lagrange_basis_at(
                    &half_axis,
                    inv_kappa,
                    parent_rho_w.inverse().unwrap(),
                    w,
                    &mut basis,
                );
                samples_b.push((c_h, evaluate_partial_stitch(&column, &basis)));
            }

            // The requester interpolates over the helpers' labels and reads
            // the two halves of the parent axes.
            let parent_axis = axis_points(2 * kappa);
            let interpolate_on = |samples: &[(Fp, Fp)], rho: Fp| -> Vec<Fp> {
                let points: Vec<Fp> = samples.iter().map(|(x, _)| *x).collect();
                parent_axis
                    .iter()
                    .map(|&point| {
                        let x = rho * point;
                        let mut acc = fp_zero();
                        for (i, &(x_i, y_i)) in samples.iter().enumerate() {
                            let mut num = Fp::from(1u64);
                            let mut den = Fp::from(1u64);
                            for (k, &x_k) in points.iter().enumerate() {
                                if k != i {
                                    num *= x - x_k;
                                    den *= x_i - x_k;
                                }
                            }
                            acc += y_i * num * den.inverse().unwrap();
                        }
                        acc
                    })
                    .collect()
            };
            let s0_on_w_axis = interpolate_on(&samples_a, parent_rho_w);
            let s0_prime_on_z_axis = interpolate_on(&samples_b, parent_rho_z);
            let even = |values: &[Fp]| values.iter().copied().step_by(2).collect::<Vec<_>>();
            let odd = |values: &[Fp]| {
                values
                    .iter()
                    .copied()
                    .skip(1)
                    .step_by(2)
                    .collect::<Vec<_>>()
            };
            assert_eq!(even(&s0_on_w_axis), requester[child_position(0, 0)].column);
            assert_eq!(odd(&s0_on_w_axis), requester[child_position(0, 1)].column);
            assert_eq!(
                even(&s0_prime_on_z_axis),
                requester[child_position(0, 0)].row
            );
            assert_eq!(
                odd(&s0_prime_on_z_axis),
                requester[child_position(1, 0)].row
            );
        }
    }

    #[test]
    fn update_changes_one_column_entry_and_one_row_entry() {
        let mut rng = ChaCha20Rng::seed_from_u64(19);
        let (c, w) = (Fp::from(37u64), Fp::from(41u64));
        let kappa = 8usize;
        let segment = 3usize;
        let mut cells = random_cells(&mut rng, kappa * kappa);
        let before = encode(kappa, segment, &cells, c, w);

        let (z_index, w_index) = (5usize, 2usize);
        let delta = Fp::from(rng.r#gen::<u64>());
        cells[z_index * kappa + w_index] += delta;
        let after = encode(kappa, segment, &cells, c, w);

        let axis = axis_points(kappa);
        let inv_kappa = Fp::from(kappa as u64).inverse().unwrap();
        let (rho_z, rho_w) = segment_axis_scales(kappa, segment);
        let mut basis_z = vec![fp_zero(); kappa];
        let mut basis_w = vec![fp_zero(); kappa];
        coset_lagrange_basis_at(&axis, inv_kappa, rho_z.inverse().unwrap(), c, &mut basis_z);
        coset_lagrange_basis_at(&axis, inv_kappa, rho_w.inverse().unwrap(), w, &mut basis_w);

        for j in 0..kappa {
            let expected = if j == w_index {
                before.column[j] + delta * basis_z[z_index]
            } else {
                before.column[j]
            };
            assert_eq!(after.column[j], expected);
        }
        for i in 0..kappa {
            let expected = if i == z_index {
                before.row[i] + delta * basis_w[w_index]
            } else {
                before.row[i]
            };
            assert_eq!(after.row[i], expected);
        }
    }
}
