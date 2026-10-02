//! Benchmark-only univariate segment baseline.
//!
//! This is not production code. It models a conventional univariate encoding
//! for one segment: a segment polynomial is evaluated at maintainer-owned
//! points, and a public point update adds `delta * L_i(x)` to every stored
//! evaluation.

use paper_a_optimized::coded_array::rs::{barycentric_eval, precompute_barycentric};
use paper_a_optimized::{CodedArrayParams, Fp, Value256, decode_value, encode_value, fp_zero};

const FIELD_BYTES: u64 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnivariateUpdateMode {
    Naive,
    Cached,
}

#[derive(Clone, Debug)]
pub struct CompactLagrangeCache {
    pub domain_weights: Vec<Fp>,
    pub eval_den_inv: Vec<Fp>,
    pub eval_domain_index: Vec<Option<usize>>,
}

#[derive(Clone, Debug)]
pub struct UnivariatePartyState {
    pub registry_index: usize,
    pub eval_points: Vec<Fp>,
    pub values: Vec<Fp>,
    pub basis_cache: Option<CompactLagrangeCache>,
}

#[derive(Clone, Debug)]
pub struct UnivariateSegmentBaseline {
    pub segment_len: usize,
    pub n_grid: usize,
    pub domain: Vec<Fp>,
    pub parties: Vec<UnivariatePartyState>,
}

impl UnivariateSegmentBaseline {
    pub fn new(
        params: &CodedArrayParams,
        logical_values: &[Value256],
        party_count: usize,
        mode: UnivariateUpdateMode,
    ) -> Self {
        assert!(logical_values.len() <= params.n_grid);
        assert!(party_count <= params.shard_registry.len());
        let domain = params.x.clone();
        let domain_values = padded_domain_values(params, logical_values);
        let mut parties = Vec::with_capacity(party_count);
        for registry_index in 0..party_count {
            let entry = params.shard_registry[registry_index];
            let eval_points = params
                .h
                .iter()
                .copied()
                .map(|h| entry.a * h)
                .collect::<Vec<_>>();
            let values = eval_points
                .iter()
                .copied()
                .map(|point| eval_on_domain(&domain, &domain_values, point))
                .collect::<Vec<_>>();
            let basis_cache = match mode {
                UnivariateUpdateMode::Naive => None,
                UnivariateUpdateMode::Cached => Some(CompactLagrangeCache::new(
                    &domain,
                    &precompute_barycentric(&domain).w,
                    &eval_points,
                )),
            };
            parties.push(UnivariatePartyState {
                registry_index,
                eval_points,
                values,
                basis_cache,
            });
        }
        Self {
            segment_len: logical_values.len(),
            n_grid: params.n_grid,
            domain,
            parties,
        }
    }

    pub fn apply_updates(&mut self, updates: &[(u64, Fp)]) {
        for &(idx, delta) in updates {
            let idx = idx as usize;
            assert!(idx < self.segment_len);
            for party in &mut self.parties {
                match &party.basis_cache {
                    Some(cache) => {
                        for (eval_idx, cell) in party.values.iter_mut().enumerate() {
                            *cell += delta
                                * cache.lagrange_basis(
                                    &self.domain,
                                    &party.eval_points,
                                    eval_idx,
                                    idx,
                                );
                        }
                    }
                    None => {
                        for (cell, &point) in party.values.iter_mut().zip(party.eval_points.iter())
                        {
                            *cell += delta * lagrange_basis_naive(&self.domain, idx, point);
                        }
                    }
                }
            }
        }
    }

    pub fn read_point_from_parties(&self, logical_index: usize, party_count: usize) -> Fp {
        assert!(logical_index < self.segment_len);
        assert!(party_count <= self.parties.len());
        let mut points = Vec::with_capacity(party_count * self.parties[0].values.len());
        let mut values = Vec::with_capacity(points.capacity());
        for party in self.parties.iter().take(party_count) {
            points.extend_from_slice(&party.eval_points);
            values.extend_from_slice(&party.values);
        }
        assert!(
            points.len() >= self.n_grid,
            "univariate point read needs enough scalar evaluations"
        );
        let points = &points[..self.n_grid];
        let values = &values[..self.n_grid];
        let weights = precompute_barycentric(points);
        barycentric_eval(&weights, values, self.domain[logical_index])
    }

    pub fn repair_party_from_helpers(
        &self,
        target: &UnivariatePartyState,
        helper_count: usize,
    ) -> Vec<Fp> {
        assert!(helper_count <= self.parties.len());
        let mut points = Vec::with_capacity(helper_count * self.parties[0].values.len());
        let mut values = Vec::with_capacity(points.capacity());
        for party in self.parties.iter().take(helper_count) {
            points.extend_from_slice(&party.eval_points);
            values.extend_from_slice(&party.values);
        }
        assert!(points.len() >= self.n_grid);
        let points = &points[..self.n_grid];
        let values = &values[..self.n_grid];
        let weights = precompute_barycentric(points);
        target
            .eval_points
            .iter()
            .copied()
            .map(|point| barycentric_eval(&weights, values, point))
            .collect()
    }

    pub fn kappa_rescale_target_party(
        &self,
        target_params: &CodedArrayParams,
        target_registry_index: usize,
        helper_count: usize,
    ) -> Vec<Fp> {
        let target = target_party_template(target_params, target_registry_index);
        self.repair_party_from_helpers(&target, helper_count)
    }

    pub fn maintained_bytes_per_party(&self) -> u64 {
        self.parties
            .first()
            .map(|party| party.values.len() as u64 * FIELD_BYTES)
            .unwrap_or(0)
    }

    pub fn cache_bytes_per_party(&self) -> u64 {
        self.parties
            .first()
            .and_then(|party| party.basis_cache.as_ref())
            .map(|cache| cache.field_element_count() as u64 * FIELD_BYTES)
            .unwrap_or(0)
    }
}

impl CompactLagrangeCache {
    pub fn new(domain: &[Fp], domain_weights: &[Fp], eval_points: &[Fp]) -> Self {
        assert_eq!(domain.len(), domain_weights.len());
        let mut eval_den_inv = Vec::with_capacity(eval_points.len());
        let mut eval_domain_index = Vec::with_capacity(eval_points.len());
        for &point in eval_points {
            let matching_index = domain
                .iter()
                .enumerate()
                .find_map(|(idx, &domain_point)| (domain_point == point).then_some(idx));
            eval_domain_index.push(matching_index);
            if matching_index.is_some() {
                eval_den_inv.push(Fp::from(0u64));
                continue;
            }
            let mut den = fp_zero();
            for (&weight, &domain_point) in domain_weights.iter().zip(domain.iter()) {
                den += weight / (point - domain_point);
            }
            eval_den_inv.push(Fp::from(1u64) / den);
        }
        Self {
            domain_weights: domain_weights.to_vec(),
            eval_den_inv,
            eval_domain_index,
        }
    }

    pub fn lagrange_basis(
        &self,
        domain: &[Fp],
        eval_points: &[Fp],
        eval_idx: usize,
        source_idx: usize,
    ) -> Fp {
        if let Some(domain_idx) = self.eval_domain_index[eval_idx] {
            return if domain_idx == source_idx {
                Fp::from(1u64)
            } else {
                fp_zero()
            };
        }
        self.domain_weights[source_idx] / (eval_points[eval_idx] - domain[source_idx])
            * self.eval_den_inv[eval_idx]
    }

    pub fn field_element_count(&self) -> usize {
        self.domain_weights.len() + self.eval_den_inv.len()
    }
}

pub fn target_party_template(
    params: &CodedArrayParams,
    registry_index: usize,
) -> UnivariatePartyState {
    let entry = params.shard_registry[registry_index];
    let eval_points = params
        .h
        .iter()
        .copied()
        .map(|h| entry.a * h)
        .collect::<Vec<_>>();
    UnivariatePartyState {
        registry_index,
        values: vec![fp_zero(); eval_points.len()],
        eval_points,
        basis_cache: None,
    }
}

pub fn padded_domain_values(params: &CodedArrayParams, logical_values: &[Value256]) -> Vec<Fp> {
    let mut values = vec![fp_zero(); params.n_grid];
    for (dst, value) in values.iter_mut().zip(logical_values.iter().copied()) {
        *dst = decode_value(value);
    }
    values
}

pub fn eval_on_domain(domain: &[Fp], domain_values: &[Fp], point: Fp) -> Fp {
    for (&domain_point, &value) in domain.iter().zip(domain_values.iter()) {
        if point == domain_point {
            return value;
        }
    }
    let weights = precompute_barycentric(domain);
    barycentric_eval(&weights, domain_values, point)
}

pub fn lagrange_basis_naive(domain: &[Fp], idx: usize, point: Fp) -> Fp {
    if point == domain[idx] {
        return Fp::from(1u64);
    }
    for (j, &domain_point) in domain.iter().enumerate() {
        if j != idx && point == domain_point {
            return fp_zero();
        }
    }
    let x_i = domain[idx];
    let mut acc = Fp::from(1u64);
    for (j, &x_j) in domain.iter().enumerate() {
        if j == idx {
            continue;
        }
        acc *= (point - x_j) / (x_i - x_j);
    }
    acc
}

pub fn encode_field_values(values: &[Fp]) -> Vec<Value256> {
    values.iter().copied().map(encode_value).collect()
}
