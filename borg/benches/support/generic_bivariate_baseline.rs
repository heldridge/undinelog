//! Benchmark-only generic bivariate baseline.
//!
//! This comparator intentionally does not enter the production API. It gives a
//! party the same kind of responsibility shape used by the optimized array--a
//! column plus auxiliary labels--but treats the maintained object as an ordinary
//! bivariate polynomial. Public point updates therefore use the generic product
//! patch `delta * L_a(Z) * L_b(W)`.

use crate::support::univariate_segment_baseline::{CompactLagrangeCache, lagrange_basis_naive};
use paper_a_optimized::coded_array::rs::{barycentric_eval, precompute_barycentric};
use paper_a_optimized::{CodedArrayParams, Fp, Value256, decode_value, fp_zero};

const FIELD_BYTES: u64 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BivariateUpdateMode {
    Naive,
    CachedSeparated,
}

#[derive(Clone, Debug)]
pub struct BivariateBasisCache {
    column_z: CompactLagrangeCache,
    column_w: CompactLagrangeCache,
    row_z: CompactLagrangeCache,
    row_w: CompactLagrangeCache,
}

#[derive(Clone, Debug)]
pub struct BivariateClaimState {
    pub registry_index: usize,
    pub z_column: Fp,
    pub w_column_points: Vec<Fp>,
    pub column_values: Vec<Fp>,
    pub row_labels: Vec<Fp>,
    pub z_row_points: Vec<Fp>,
    pub row_values: Vec<Vec<Fp>>,
    pub cache: Option<BivariateBasisCache>,
}

#[derive(Clone, Debug)]
pub struct GenericBivariateBaseline {
    pub segment_len: usize,
    pub source_z_domain: Vec<Fp>,
    pub source_w_domain: Vec<Fp>,
    pub claims: Vec<BivariateClaimState>,
}

impl GenericBivariateBaseline {
    pub fn new(
        params: &CodedArrayParams,
        logical_values: &[Value256],
        claim_count: usize,
        mode: BivariateUpdateMode,
    ) -> Self {
        assert!(logical_values.len() <= params.n_grid);
        assert!(claim_count + 1 < params.shard_registry.len());
        let source_z_domain = source_z_domain(params);
        let source_w_domain = params.h.clone();
        let matrix = source_matrix(params, logical_values);
        let mut claims = Vec::with_capacity(claim_count);
        for registry_index in 0..claim_count {
            let mut claim =
                claim_template(params, registry_index, &source_z_domain, &source_w_domain);
            hydrate_claim(&source_z_domain, &source_w_domain, &matrix, &mut claim);
            if mode == BivariateUpdateMode::CachedSeparated {
                claim.cache = Some(precompute_cache(&source_z_domain, &source_w_domain, &claim));
            }
            claims.push(claim);
        }
        Self {
            segment_len: logical_values.len(),
            source_z_domain,
            source_w_domain,
            claims,
        }
    }

    pub fn apply_updates(&mut self, updates: &[(u64, Fp)]) {
        for &(idx, delta) in updates {
            let (source_z_idx, source_w_idx) = self.source_position(idx as usize);
            for claim in &mut self.claims {
                match &claim.cache {
                    Some(cache) => {
                        let column_z = cache.column_z.lagrange_basis(
                            &self.source_z_domain,
                            std::slice::from_ref(&claim.z_column),
                            0,
                            source_z_idx,
                        );
                        for (cell_idx, cell) in claim.column_values.iter_mut().enumerate() {
                            let column_w = cache.column_w.lagrange_basis(
                                &self.source_w_domain,
                                &claim.w_column_points,
                                cell_idx,
                                source_w_idx,
                            );
                            *cell += delta * column_z * column_w;
                        }
                        for (row_idx, row) in claim.row_values.iter_mut().enumerate() {
                            let row_w = cache.row_w.lagrange_basis(
                                &self.source_w_domain,
                                &claim.row_labels,
                                row_idx,
                                source_w_idx,
                            );
                            for (cell_idx, cell) in row.iter_mut().enumerate() {
                                let row_z = cache.row_z.lagrange_basis(
                                    &self.source_z_domain,
                                    &claim.z_row_points,
                                    cell_idx,
                                    source_z_idx,
                                );
                                *cell += delta * row_z * row_w;
                            }
                        }
                    }
                    None => {
                        let column_z = lagrange_basis_naive(
                            &self.source_z_domain,
                            source_z_idx,
                            claim.z_column,
                        );
                        for (cell, &w_eval) in claim
                            .column_values
                            .iter_mut()
                            .zip(claim.w_column_points.iter())
                        {
                            let w_basis =
                                lagrange_basis_naive(&self.source_w_domain, source_w_idx, w_eval);
                            *cell += delta * column_z * w_basis;
                        }
                        for (row_idx, row) in claim.row_values.iter_mut().enumerate() {
                            let row_w = lagrange_basis_naive(
                                &self.source_w_domain,
                                source_w_idx,
                                claim.row_labels[row_idx],
                            );
                            for (cell, &z_eval) in row.iter_mut().zip(claim.z_row_points.iter()) {
                                let z_basis = lagrange_basis_naive(
                                    &self.source_z_domain,
                                    source_z_idx,
                                    z_eval,
                                );
                                *cell += delta * z_basis * row_w;
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn read_point_from_claims(&self, logical_index: usize, helper_count: usize) -> Fp {
        assert!(helper_count <= self.claims.len());
        let (source_z_idx, source_w_idx) = self.source_position(logical_index);
        let target_w = self.source_w_domain[source_w_idx];
        let mut z_points = Vec::with_capacity(helper_count);
        let mut shares = Vec::with_capacity(helper_count);
        for claim in self.claims.iter().take(helper_count) {
            z_points.push(claim.z_column);
            shares.push(eval_claim_column(claim, target_w));
        }
        let weights = precompute_barycentric(&z_points);
        barycentric_eval(&weights, &shares, self.source_z_domain[source_z_idx])
    }

    pub fn repair_claim_from_helpers(
        &self,
        target: &BivariateClaimState,
        helper_count: usize,
    ) -> BivariateClaimState {
        assert!(helper_count <= self.claims.len());
        let mut repaired = target.clone();
        let helper_claims = &self.claims[..helper_count];
        let z_points = helper_claims
            .iter()
            .map(|claim| claim.z_column)
            .collect::<Vec<_>>();
        let z_weights = precompute_barycentric(&z_points);

        for (cell_idx, &target_w) in repaired.w_column_points.iter().enumerate() {
            let shares = helper_claims
                .iter()
                .map(|claim| eval_claim_column(claim, target_w))
                .collect::<Vec<_>>();
            repaired.column_values[cell_idx] =
                barycentric_eval(&z_weights, &shares, repaired.z_column);
        }

        for (row_idx, &row_label) in repaired.row_labels.iter().enumerate() {
            for (cell_idx, &z_eval) in repaired.z_row_points.iter().enumerate() {
                let shares = helper_claims
                    .iter()
                    .map(|claim| eval_claim_column(claim, row_label))
                    .collect::<Vec<_>>();
                repaired.row_values[row_idx][cell_idx] =
                    barycentric_eval(&z_weights, &shares, z_eval);
            }
        }
        repaired.cache = None;
        repaired
    }

    pub fn repair_claim_same_geometry_from_helpers(
        &self,
        target: &BivariateClaimState,
        helper_count: usize,
    ) -> BivariateClaimState {
        assert!(helper_count <= self.claims.len());
        let mut repaired = target.clone();
        let helper_claims = &self.claims[..helper_count];

        let mut row_points = Vec::new();
        let mut row_values = Vec::new();
        for claim in helper_claims {
            for (row_idx, &row_label) in claim.row_labels.iter().enumerate() {
                row_points.push(row_label);
                row_values.push(eval_claim_row(claim, row_idx, repaired.z_column));
            }
        }
        assert!(
            row_points.len() >= self.source_w_domain.len(),
            "primary repair needs enough row-label evaluations"
        );
        let row_weights = precompute_barycentric(&row_points);
        for (cell_idx, &target_w) in repaired.w_column_points.iter().enumerate() {
            repaired.column_values[cell_idx] =
                barycentric_eval(&row_weights, &row_values, target_w);
        }

        let z_points = helper_claims
            .iter()
            .map(|claim| claim.z_column)
            .collect::<Vec<_>>();
        let z_weights = precompute_barycentric(&z_points);
        for (row_idx, &row_label) in repaired.row_labels.iter().enumerate() {
            let shares = helper_claims
                .iter()
                .map(|claim| eval_claim_column(claim, row_label))
                .collect::<Vec<_>>();
            for (cell_idx, &z_eval) in repaired.z_row_points.iter().enumerate() {
                repaired.row_values[row_idx][cell_idx] =
                    barycentric_eval(&z_weights, &shares, z_eval);
            }
        }
        repaired.cache = None;
        repaired
    }

    pub fn kappa_rescale_target_claim(
        &self,
        target_params: &CodedArrayParams,
        target_registry_index: usize,
        helper_count: usize,
    ) -> BivariateClaimState {
        let target_z_domain = source_z_domain(target_params);
        let target_w_domain = target_params.h.clone();
        let target = claim_template(
            target_params,
            target_registry_index,
            &target_z_domain,
            &target_w_domain,
        );
        self.repair_claim_from_helpers(&target, helper_count)
    }

    pub fn kappa_rescale_target_claim_by_decode_reencode(
        &self,
        target_params: &CodedArrayParams,
        target_registry_index: usize,
        helper_count: usize,
    ) -> BivariateClaimState {
        assert!(helper_count <= self.claims.len());
        assert_eq!(self.segment_len, target_params.segment_len);

        let helper_claims = &self.claims[..helper_count];
        let z_points = helper_claims
            .iter()
            .map(|claim| claim.z_column)
            .collect::<Vec<_>>();
        let z_weights = precompute_barycentric(&z_points);

        let mut decoded = vec![fp_zero(); self.source_z_domain.len() * self.source_w_domain.len()];
        for (w_idx, &w) in self.source_w_domain.iter().enumerate() {
            let shares = helper_claims
                .iter()
                .map(|claim| eval_claim_column(claim, w))
                .collect::<Vec<_>>();
            for (z_idx, &z) in self.source_z_domain.iter().enumerate() {
                decoded[z_idx * self.source_w_domain.len() + w_idx] =
                    barycentric_eval(&z_weights, &shares, z);
            }
        }

        let target_z_domain = source_z_domain(target_params);
        let target_w_domain = target_params.h.clone();
        let mut target_matrix = vec![vec![fp_zero(); target_params.s]; target_params.need];
        for idx in 0..target_params.segment_len {
            target_matrix[idx / target_params.s][idx % target_params.s] = decoded[idx];
        }
        let mut target = claim_template(
            target_params,
            target_registry_index,
            &target_z_domain,
            &target_w_domain,
        );
        hydrate_claim(
            &target_z_domain,
            &target_w_domain,
            &target_matrix,
            &mut target,
        );
        target
    }

    pub fn segment_double_target_claim_by_decode_reencode(
        &self,
        right: &Self,
        target_params: &CodedArrayParams,
        target_registry_index: usize,
        helper_count: usize,
    ) -> BivariateClaimState {
        assert_eq!(self.source_z_domain, right.source_z_domain);
        assert_eq!(self.source_w_domain, right.source_w_domain);
        assert_eq!(
            target_params.n_grid,
            self.source_z_domain.len() * self.source_w_domain.len() * 2
        );

        let left_decoded = self.decode_logical_values_from_claims(helper_count);
        let right_decoded = right.decode_logical_values_from_claims(helper_count);
        let mut target_values = Vec::with_capacity(target_params.n_grid);
        target_values.extend_from_slice(&left_decoded);
        target_values.extend_from_slice(&right_decoded);
        target_values.resize(target_params.n_grid, fp_zero());

        let target_z_domain = source_z_domain(target_params);
        let target_w_domain = target_params.h.clone();
        let mut target_matrix = vec![vec![fp_zero(); target_params.s]; target_params.need];
        for (idx, value) in target_values.into_iter().enumerate() {
            target_matrix[idx / target_params.s][idx % target_params.s] = value;
        }
        let mut target = claim_template(
            target_params,
            target_registry_index,
            &target_z_domain,
            &target_w_domain,
        );
        hydrate_claim(
            &target_z_domain,
            &target_w_domain,
            &target_matrix,
            &mut target,
        );
        target
    }

    pub fn decode_logical_values_from_claims(&self, helper_count: usize) -> Vec<Fp> {
        let matrix = self.decode_matrix_from_claims(helper_count);
        (0..self.segment_len)
            .map(|idx| matrix[idx / self.source_w_domain.len()][idx % self.source_w_domain.len()])
            .collect()
    }

    fn decode_matrix_from_claims(&self, helper_count: usize) -> Vec<Vec<Fp>> {
        assert!(helper_count <= self.claims.len());
        let helper_claims = &self.claims[..helper_count];
        let z_points = helper_claims
            .iter()
            .map(|claim| claim.z_column)
            .collect::<Vec<_>>();
        let z_weights = precompute_barycentric(&z_points);

        let mut decoded =
            vec![vec![fp_zero(); self.source_w_domain.len()]; self.source_z_domain.len()];
        for (w_idx, &w) in self.source_w_domain.iter().enumerate() {
            let shares = helper_claims
                .iter()
                .map(|claim| eval_claim_column(claim, w))
                .collect::<Vec<_>>();
            for (z_idx, &z) in self.source_z_domain.iter().enumerate() {
                decoded[z_idx][w_idx] = barycentric_eval(&z_weights, &shares, z);
            }
        }
        decoded
    }

    pub fn maintained_bytes_per_claim(&self) -> u64 {
        self.claims
            .first()
            .map(maintained_bytes_for_claim)
            .unwrap_or(0)
    }

    pub fn cache_bytes_per_claim(&self) -> u64 {
        self.claims
            .first()
            .and_then(|claim| claim.cache.as_ref())
            .map(cache_bytes)
            .unwrap_or(0)
    }

    fn source_position(&self, logical_index: usize) -> (usize, usize) {
        assert!(logical_index < self.segment_len);
        (
            logical_index / self.source_w_domain.len(),
            logical_index % self.source_w_domain.len(),
        )
    }
}

pub fn source_z_domain(params: &CodedArrayParams) -> Vec<Fp> {
    (0..params.need)
        .map(|idx| params.column_coordinate_for_grid_index(idx))
        .collect()
}

pub fn claim_template(
    params: &CodedArrayParams,
    registry_index: usize,
    source_z_domain: &[Fp],
    source_w_domain: &[Fp],
) -> BivariateClaimState {
    let z_column = params.shard_registry[registry_index].z;
    let w_base = params.shard_registry[registry_index + 1].a;
    let w_column_points = source_w_domain
        .iter()
        .copied()
        .map(|point| w_base * point)
        .collect::<Vec<_>>();
    let rows_per_claim = usize::max(1, params.s.div_ceil(params.need));
    let row_start = registry_index * rows_per_claim;
    let row_labels = (0..rows_per_claim)
        .map(|offset| params.row_labels[(row_start + offset) % params.row_labels.len()])
        .collect::<Vec<_>>();
    let z_row_points = if params.row_eval_points.len() >= source_z_domain.len() {
        params.row_eval_points[..source_z_domain.len()].to_vec()
    } else {
        source_z_domain.to_vec()
    };
    let column_values = vec![fp_zero(); w_column_points.len()];
    let row_values = vec![vec![fp_zero(); z_row_points.len()]; row_labels.len()];
    BivariateClaimState {
        registry_index,
        z_column,
        w_column_points,
        column_values,
        row_labels,
        z_row_points,
        row_values,
        cache: None,
    }
}

fn hydrate_claim(
    source_z_domain: &[Fp],
    source_w_domain: &[Fp],
    matrix: &[Vec<Fp>],
    claim: &mut BivariateClaimState,
) {
    let source_z_weights = precompute_barycentric(source_z_domain);
    let source_w_weights = precompute_barycentric(source_w_domain);
    for (cell, &w_eval) in claim
        .column_values
        .iter_mut()
        .zip(claim.w_column_points.iter())
    {
        *cell = eval_bivariate_with_weights(
            source_z_domain,
            &source_z_weights,
            &source_w_weights,
            matrix,
            claim.z_column,
            w_eval,
        );
    }
    for (row_idx, &row_label) in claim.row_labels.iter().enumerate() {
        for (cell, &z_eval) in claim.row_values[row_idx]
            .iter_mut()
            .zip(claim.z_row_points.iter())
        {
            *cell = eval_bivariate_with_weights(
                source_z_domain,
                &source_z_weights,
                &source_w_weights,
                matrix,
                z_eval,
                row_label,
            );
        }
    }
}

fn source_matrix(params: &CodedArrayParams, logical_values: &[Value256]) -> Vec<Vec<Fp>> {
    let mut matrix = vec![vec![fp_zero(); params.s]; params.need];
    for (idx, &value) in logical_values.iter().enumerate() {
        matrix[idx / params.s][idx % params.s] = decode_value(value);
    }
    matrix
}

fn eval_bivariate_with_weights(
    source_z_domain: &[Fp],
    source_z_weights: &paper_a_optimized::coded_array::rs::BarycentricWeights,
    source_w_weights: &paper_a_optimized::coded_array::rs::BarycentricWeights,
    matrix: &[Vec<Fp>],
    z: Fp,
    w: Fp,
) -> Fp {
    let z_slice = matrix
        .iter()
        .map(|row| barycentric_eval(source_w_weights, row, w))
        .collect::<Vec<_>>();
    debug_assert_eq!(source_z_domain.len(), z_slice.len());
    barycentric_eval(source_z_weights, &z_slice, z)
}

fn eval_claim_column(claim: &BivariateClaimState, w: Fp) -> Fp {
    let weights = precompute_barycentric(&claim.w_column_points);
    barycentric_eval(&weights, &claim.column_values, w)
}

fn eval_claim_row(claim: &BivariateClaimState, row_idx: usize, z: Fp) -> Fp {
    let weights = precompute_barycentric(&claim.z_row_points);
    barycentric_eval(&weights, &claim.row_values[row_idx], z)
}

fn precompute_cache(
    source_z_domain: &[Fp],
    source_w_domain: &[Fp],
    claim: &BivariateClaimState,
) -> BivariateBasisCache {
    let source_z_weights = precompute_barycentric(source_z_domain).w;
    let source_w_weights = precompute_barycentric(source_w_domain).w;
    BivariateBasisCache {
        column_z: CompactLagrangeCache::new(
            source_z_domain,
            &source_z_weights,
            std::slice::from_ref(&claim.z_column),
        ),
        column_w: CompactLagrangeCache::new(
            source_w_domain,
            &source_w_weights,
            &claim.w_column_points,
        ),
        row_z: CompactLagrangeCache::new(source_z_domain, &source_z_weights, &claim.z_row_points),
        row_w: CompactLagrangeCache::new(source_w_domain, &source_w_weights, &claim.row_labels),
    }
}

fn maintained_bytes_for_claim(claim: &BivariateClaimState) -> u64 {
    let column = claim.column_values.len() as u64;
    let rows = claim
        .row_values
        .iter()
        .map(|row| row.len() as u64)
        .sum::<u64>();
    (column + rows) * FIELD_BYTES
}

fn cache_bytes(cache: &BivariateBasisCache) -> u64 {
    let cells = cache.column_z.field_element_count() as u64
        + cache.column_w.field_element_count() as u64
        + cache.row_z.field_element_count() as u64
        + cache.row_w.field_element_count() as u64;
    cells * FIELD_BYTES
}
