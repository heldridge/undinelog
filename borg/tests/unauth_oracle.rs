use ark_ff::Field;
use paper_a_optimized::coded_array::params::CodedArrayConfig;
use paper_a_optimized::unauth::{
    block_grid_indices, build_migrated_params, registry_supports_sibling_kappa_halve,
    semantic_position_for_s,
};
use paper_a_optimized::{
    CodedArrayParams, Fp, build_coded_params, build_shard_registry_deterministic, fp_one, fp_pow,
    fp_zero,
};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
struct DirectClaim {
    representative: Fp,
    row_label_indices: Vec<usize>,
    column: Vec<Fp>,
    rows: BTreeMap<usize, Vec<Fp>>,
}

fn build_params(segment_len: usize, s: usize, shard_count: usize) -> CodedArrayParams {
    let n_grid = segment_len;
    let cfg = CodedArrayConfig {
        segment_len,
        s,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count, s, n_grid),
        party_registry_indices: vec![(0..shard_count).collect()],
        seed: 23,
    };
    build_coded_params(&cfg)
}

fn sample_segment_coeffs(n_grid: usize) -> Vec<Fp> {
    (0..n_grid)
        .map(|idx| Fp::from((7 * idx as u64) + ((idx % 5) as u64) + 3))
        .collect()
}

fn eval_poly(coeffs: &[Fp], x: Fp) -> Fp {
    let mut acc = fp_zero();
    for &coeff in coeffs.iter().rev() {
        acc *= x;
        acc += coeff;
    }
    acc
}

fn direct_row_value(coeffs: &[Fp], s: usize, w: Fp, z: Fp) -> Fp {
    assert_eq!(coeffs.len() % s, 0);
    let n = coeffs.len() / s;
    let mut out = fp_zero();
    let mut w_pow = fp_one();
    for r in 0..s {
        let mut f_r = fp_zero();
        let mut z_pow = fp_one();
        for q in 0..n {
            f_r += coeffs[q * s + r] * z_pow;
            z_pow *= z;
        }
        out += w_pow * f_r;
        w_pow *= w;
    }
    out
}

fn build_direct_claim(
    params: &CodedArrayParams,
    coeffs: &[Fp],
    representative: Fp,
    row_label_indices: &[usize],
) -> DirectClaim {
    let mut row_label_indices = row_label_indices.to_vec();
    row_label_indices.sort_unstable();
    row_label_indices.dedup();
    let column = params
        .h
        .iter()
        .map(|&h_t| eval_poly(coeffs, representative * h_t))
        .collect();
    let mut rows = BTreeMap::new();
    for &row_idx in &row_label_indices {
        let w = params.row_labels[row_idx];
        let values = params
            .row_eval_points
            .iter()
            .map(|&z| direct_row_value(coeffs, params.s, w, z))
            .collect();
        rows.insert(row_idx, values);
    }
    DirectClaim {
        representative,
        row_label_indices,
        column,
        rows,
    }
}

fn contract_row_label_indices_once(
    params: &CodedArrayParams,
    row_label_indices: &[usize],
) -> Vec<usize> {
    let stride = params.n_grid / params.s;
    let mut out: Vec<_> = row_label_indices
        .iter()
        .copied()
        .filter(|row_idx| ((row_idx / stride) % 2) == 0)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn find_eval_index(points: &[Fp], target: Fp) -> usize {
    points
        .iter()
        .position(|&point| point == target)
        .expect("target eval point exists in template")
}

fn kappa_double_claim_via_naive_paired_formula(
    source_params: &CodedArrayParams,
    target_params: &CodedArrayParams,
    source: &DirectClaim,
) -> DirectClaim {
    assert_eq!(source_params.s / 2, target_params.s);
    let stride = source_params.n_grid / source_params.s;
    let half_inv = Fp::from(2u64).inverse().expect("2 invertible");
    let target_s = target_params.s;
    let row_label_indices =
        contract_row_label_indices_once(source_params, &source.row_label_indices);
    let mut rows = BTreeMap::new();
    for &row_idx in &row_label_indices {
        let partner_idx = row_idx + stride;
        let source_row = source.rows.get(&row_idx).expect("source row present");
        let partner_row = source.rows.get(&partner_idx).expect("partner row present");
        let w = source_params.row_labels[row_idx];
        let denom_inv = (Fp::from(2u64) * fp_pow(w, target_s as u64))
            .inverse()
            .expect("nonzero kappa-double denominator");
        let values = target_params
            .row_eval_points
            .iter()
            .map(|&u| {
                let src_idx = find_eval_index(&source_params.row_eval_points, u * u);
                let left = source_row[src_idx];
                let right = partner_row[src_idx];
                (left + right) * half_inv + u * (left - right) * denom_inv
            })
            .collect();
        rows.insert(row_idx, values);
    }
    DirectClaim {
        representative: source.representative,
        row_label_indices,
        column: source.column.iter().copied().step_by(2).collect(),
        rows,
    }
}

fn kappa_double_claim_via_full_orbit_formula(
    source_params: &CodedArrayParams,
    target_params: &CodedArrayParams,
    source: &DirectClaim,
) -> DirectClaim {
    assert_eq!(source_params.s / 2, target_params.s);
    let stride = source_params.n_grid / source_params.s;
    let target_s = target_params.s;
    let target_s_inv = Fp::from(target_s as u64)
        .inverse()
        .expect("target s invertible");
    let eta = fp_pow(source_params.omega, stride as u64);
    let row_label_indices =
        contract_row_label_indices_once(source_params, &source.row_label_indices);
    let mut rows = BTreeMap::new();
    for &row_idx in &row_label_indices {
        let w = source_params.row_labels[row_idx];
        let w_m = fp_pow(w, target_s as u64);
        let values = target_params
            .row_eval_points
            .iter()
            .map(|&u| {
                let src_idx = find_eval_index(&source_params.row_eval_points, u * u);
                let base = source.rows.get(&row_idx).expect("base source row present")[src_idx];
                let base_coeff = (fp_one() + u * w_m.inverse().expect("w^m invertible"))
                    * Fp::from(2u64).inverse().expect("2 invertible");
                let mut out = base * base_coeff;
                let odd_coeff_scale =
                    (fp_one() - u * w_m.inverse().expect("w^m invertible")) * target_s_inv;
                for q in (1..source_params.s).step_by(2) {
                    let orbit_idx = (row_idx + q * stride) % source_params.n_grid;
                    let value = source
                        .rows
                        .get(&orbit_idx)
                        .expect("full orbit source row present")[src_idx];
                    let eta_q = fp_pow(eta, q as u64);
                    let denom_inv = (fp_one() - eta_q.inverse().expect("eta^q invertible"))
                        .inverse()
                        .expect("orbit denominator invertible");
                    out += value * odd_coeff_scale * denom_inv;
                }
                out
            })
            .collect();
        rows.insert(row_idx, values);
    }
    DirectClaim {
        representative: source.representative,
        row_label_indices,
        column: source.column.iter().copied().step_by(2).collect(),
        rows,
    }
}

fn kappa_halve_claim_via_formula(
    source_params: &CodedArrayParams,
    target_params: &CodedArrayParams,
    left: &DirectClaim,
    right: &DirectClaim,
) -> DirectClaim {
    assert_eq!(source_params.s * 2, target_params.s);
    assert_eq!(left.row_label_indices, right.row_label_indices);
    for row_idx in &left.row_label_indices {
        assert_eq!(left.rows.get(row_idx), right.rows.get(row_idx));
    }
    let half_inv = Fp::from(2u64).inverse().expect("2 invertible");
    let mut column = Vec::with_capacity(left.column.len() * 2);
    for (&left_value, &right_value) in left.column.iter().zip(right.column.iter()) {
        column.push(left_value);
        column.push(right_value);
    }
    let mut rows = BTreeMap::new();
    for &row_idx in &left.row_label_indices {
        let source_row = left.rows.get(&row_idx).expect("source row present");
        let w = source_params.row_labels[row_idx];
        let w_factor = fp_pow(w, source_params.s as u64);
        let values = target_params
            .row_eval_points
            .iter()
            .map(|&u| {
                let z_idx = source_params
                    .row_eval_points
                    .iter()
                    .position(|&z| z * z == u)
                    .expect("square-root in source template");
                let z = source_params.row_eval_points[z_idx];
                let neg_idx = find_eval_index(&source_params.row_eval_points, -z);
                let pos = source_row[z_idx];
                let neg = source_row[neg_idx];
                let denom_inv = (Fp::from(2u64) * z)
                    .inverse()
                    .expect("nonzero kappa-halve denominator");
                (pos + neg) * half_inv + w_factor * (pos - neg) * denom_inv
            })
            .collect();
        rows.insert(row_idx, values);
    }
    DirectClaim {
        representative: left.representative,
        row_label_indices: left.row_label_indices.clone(),
        column,
        rows,
    }
}

fn assert_claim_matches(expected: &DirectClaim, actual: &DirectClaim) {
    assert_eq!(actual.representative, expected.representative);
    assert_eq!(actual.row_label_indices, expected.row_label_indices);
    assert_eq!(actual.column, expected.column);
    assert_eq!(actual.rows, expected.rows);
}

#[test]
fn naive_two_row_kappa_double_formula_does_not_match_direct_target_claim() {
    let params = build_params(64, 8, 64);
    let kappa_doubled = build_migrated_params(&params, 4).expect("kappa-double params");
    let coeffs = sample_segment_coeffs(params.n_grid);
    let representative = params.shard_registry[0].a;
    let row_label_indices = vec![1, 9, 19, 27];
    let source = build_direct_claim(&params, &coeffs, representative, &row_label_indices);
    let actual = kappa_double_claim_via_naive_paired_formula(&params, &kappa_doubled, &source);
    let target_rows = contract_row_label_indices_once(&params, &row_label_indices);
    let expected = build_direct_claim(&kappa_doubled, &coeffs, representative, &target_rows);
    assert_eq!(actual.representative, expected.representative);
    assert_eq!(actual.row_label_indices, expected.row_label_indices);
    assert_eq!(actual.column, expected.column);
    assert_ne!(
        actual.rows, expected.rows,
        "the naive two-row kappa-double formula should fail the direct oracle under the current row definition"
    );
}

#[test]
#[ignore = "symbolic kappa-halve-row oracle needs rewrite for the current row-family model; concrete kappa-double/kappa-halve behavior is covered in unauth_migrate"]
fn kappa_halve_local_formula_matches_direct_target_claim() {
    let parent = build_params(64, 8, 32);
    let params = build_migrated_params(&parent, 4).expect("partner-paired current params");
    let kappa_halved = build_migrated_params(&params, 8).expect("kappa-halve params");
    assert!(registry_supports_sibling_kappa_halve(&params));
    let coeffs = sample_segment_coeffs(params.n_grid);
    let left_rep = params.shard_registry[0].a;
    let right_rep = params.shard_registry[1].a;
    assert_eq!(params.shard_registry[1].z, -params.shard_registry[0].z);
    let row_label_indices = vec![2, 7, 18];
    let left = build_direct_claim(&params, &coeffs, left_rep, &row_label_indices);
    let right = build_direct_claim(&params, &coeffs, right_rep, &row_label_indices);
    let actual = kappa_halve_claim_via_formula(&params, &kappa_halved, &left, &right);
    let expected = build_direct_claim(&kappa_halved, &coeffs, left_rep, &row_label_indices);
    assert_claim_matches(&expected, &actual);
}

#[test]
#[ignore = "symbolic full-orbit kappa-double oracle still assumes the old row-orbit indexing model; concrete kappa-double/kappa-halve behavior is covered in unauth_migrate"]
fn full_orbit_kappa_double_formula_matches_direct_target_claim() {
    let params = build_params(64, 8, 64);
    let kappa_doubled = build_migrated_params(&params, 4).expect("kappa-double params");
    let coeffs = sample_segment_coeffs(params.n_grid);
    let representative = params.shard_registry[0].a;
    let phase = 1usize;
    let stride = params.n_grid / params.s;
    let row_label_indices: Vec<_> = (0..params.s).map(|q| phase + q * stride).collect();
    let source = build_direct_claim(&params, &coeffs, representative, &row_label_indices);
    let actual = kappa_double_claim_via_full_orbit_formula(&params, &kappa_doubled, &source);
    let target_rows = contract_row_label_indices_once(&params, &row_label_indices);
    let expected = build_direct_claim(&kappa_doubled, &coeffs, representative, &target_rows);
    assert_claim_matches(&expected, &actual);
}

#[test]
fn array_mapping_changes_with_epoch_and_kappa_halve_restores_aligned_blocks() {
    let params = build_params(64, 8, 8);
    let kappa_doubled = build_migrated_params(&params, 4).expect("kappa-double params");
    let kappa_halved = build_migrated_params(&kappa_doubled, 8).expect("kappa-halve params");

    let mut changed = false;
    for logical_offset in 0..params.segment_len {
        let original = semantic_position_for_s(&params, params.s, logical_offset).grid_index;
        let migrated = semantic_position_for_s(&params, kappa_doubled.s, logical_offset).grid_index;
        let restored = semantic_position_for_s(&params, kappa_halved.s, logical_offset).grid_index;
        if original != migrated {
            changed = true;
        }
        assert_eq!(restored, original);
    }
    assert!(
        changed,
        "migration should change the current array-to-grid map"
    );

    for block in 0..(params.segment_len / params.s) {
        let current_coset = block_grid_indices(&params, block);
        let mapped: Vec<_> = (0..params.s)
            .map(|offset| {
                semantic_position_for_s(&params, params.s, block * params.s + offset).grid_index
            })
            .collect();
        assert_eq!(mapped, current_coset);
    }
}
