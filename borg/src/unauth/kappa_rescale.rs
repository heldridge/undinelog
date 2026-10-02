//! Fixed-segment kappa-rescale helpers for the unauthenticated coded array.
//!
//! The ordinary array operations stay in `unauth::mod`; this module contains
//! the adjacent `s` paths and benchmark preparation state used by historical
//! kappa-rescale experiments.

use super::{
    ClaimBundle, CodedArrayParams, CodedReadError, Fp, MigrateError, RowKey, ShardKey, UnauthStore,
    build_migrated_params, collect_target_row_owners_by_index, kappa_double_child_registry_indices,
    kappa_double_primary_shard_child, kappa_halve_parent_registry_index,
    materialize_row_on_eval_points_from_samples, normalize_claim_bundles, shard_value_from_vector,
    shard_value_from_vector_batch,
};
#[cfg(feature = "bench")]
use super::{kappa_halve_primary_shard, merge_claim_bundles_for_kappa_halve};
use crate::coded_array::rs::{BarycentricWeights, precompute_barycentric};
use std::collections::{HashMap, HashSet};

#[cfg(feature = "bench")]
use std::collections::BTreeMap;

/// Target-column evaluations supplied during compact one-claim kappa-double.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KappaDoubleRowEvalMigration {
    pub registry_index: usize,
    pub segments: Vec<Vec<Fp>>,
}

/// Target-column evaluations supplied during compact one-claim kappa-halve.
#[cfg(feature = "bench")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetRowEvalMigration {
    pub registry_index: usize,
    pub segments: Vec<Vec<Fp>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KappaDoubleHydrationSource {
    Local(usize),
    Helper(usize),
}

#[cfg(feature = "bench")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KappaHalveHydrationSource {
    Local(usize),
    Helper(usize),
}

/// Precomputed local plan for adjacent one-claim kappa-double.
///
/// The helper evaluations remain runtime data, but the target parameters,
/// deterministic target claims, row labels, sample columns, and interpolation
/// weights are stable for the current source state and target mode. This keeps
/// kappa-rescale benchmarks comparable to repair benchmarks, which also
/// prepare helper selection and interpolation metadata outside the timed
/// recoverer kernel.
#[derive(Clone)]
pub struct KappaDoubleOneClaimLeftPlan {
    target_params: CodedArrayParams,
    source_claims: Vec<ClaimBundle>,
    target_claims: Vec<ClaimBundle>,
    target_row_owners: Vec<(usize, Fp, ShardKey)>,
    sample_plan: Vec<KappaDoubleHydrationSource>,
    sample_z_values: Vec<Fp>,
    weights: BarycentricWeights,
}

/// Precomputed local plan for adjacent one-claim kappa-halve.
#[cfg(feature = "bench")]
#[derive(Clone)]
pub struct KappaHalveOneClaimWithPartnerPlan {
    target_params: CodedArrayParams,
    source_claims: Vec<ClaimBundle>,
    target_claims: Vec<ClaimBundle>,
    target_row_owners: Vec<(usize, Fp, ShardKey)>,
    sample_plan: Vec<KappaHalveHydrationSource>,
    sample_z_values: Vec<Fp>,
    weights: BarycentricWeights,
}

impl UnauthStore {
    fn build_kappa_double_one_claim_left_plan(
        &self,
        target_s: usize,
        helper_registry_indices: &[usize],
    ) -> Result<KappaDoubleOneClaimLeftPlan, MigrateError> {
        let source_claims = self.effective_claims_for_params(&self.params);
        let target_params = build_migrated_params(&self.params, target_s)?;
        if target_s >= self.params.s {
            return Err(MigrateError::InvalidTargetS);
        }

        let mut target_claims = Vec::with_capacity(source_claims.len());
        for claim in &source_claims {
            let (left, _) = kappa_double_child_registry_indices(claim.registry_index);
            let target_claim = ClaimBundle::exact_for_registry_index(&target_params, left)
                .ok_or(MigrateError::InvalidClaim)?;
            target_claims.push(target_claim);
        }
        let target_claims = normalize_claim_bundles(&target_params, target_claims);
        let target_row_owners = collect_target_row_owners_by_index(&target_params, &target_claims)?;

        let mut sample_plan = Vec::<KappaDoubleHydrationSource>::with_capacity(target_params.need);
        let mut sample_z_values = Vec::<Fp>::with_capacity(target_params.need);
        let mut seen_z = HashSet::new();
        for claim in &target_claims {
            let registry_index = claim.registry_index;
            let entry = *target_params
                .shard_registry
                .get(registry_index)
                .ok_or(MigrateError::InvalidPartnerShards)?;
            if !seen_z.insert(entry.key.z_bytes) {
                continue;
            }
            sample_plan.push(KappaDoubleHydrationSource::Local(registry_index));
            sample_z_values.push(entry.z);
            if sample_plan.len() >= target_params.need {
                break;
            }
        }
        if sample_plan.len() < target_params.need {
            for &registry_index in helper_registry_indices {
                let entry = *target_params
                    .shard_registry
                    .get(registry_index)
                    .ok_or(MigrateError::InvalidPartnerShards)?;
                if !seen_z.insert(entry.key.z_bytes) {
                    continue;
                }
                sample_plan.push(KappaDoubleHydrationSource::Helper(registry_index));
                sample_z_values.push(entry.z);
                if sample_plan.len() >= target_params.need {
                    break;
                }
            }
        }
        if sample_plan.len() < target_params.need {
            return Err(MigrateError::MissingPartnerShards);
        }
        let weights = precompute_barycentric(&sample_z_values);

        Ok(KappaDoubleOneClaimLeftPlan {
            target_params,
            source_claims,
            target_claims,
            target_row_owners,
            sample_plan,
            sample_z_values,
            weights,
        })
    }

    /// Precompute the stable local plan for adjacent one-claim kappa-double.
    ///
    /// `helper_registry_indices` are target-mode registry indices whose
    /// evaluations will be supplied during row hydration. The plan selects the
    /// first `need` distinct target-mode column labels from the retained local
    /// target claims and these helper targets.
    #[cfg(feature = "bench")]
    pub fn prepare_kappa_double_one_claim_left(
        &self,
        target_s: usize,
        helper_registry_indices: &[usize],
    ) -> Result<KappaDoubleOneClaimLeftPlan, MigrateError> {
        self.build_kappa_double_one_claim_left_plan(target_s, helper_registry_indices)
    }

    #[cfg(feature = "bench")]
    fn build_kappa_halve_one_claim_with_partner_plan(
        &self,
        target_s: usize,
        helper_registry_indices: &[usize],
    ) -> Result<KappaHalveOneClaimWithPartnerPlan, MigrateError> {
        let source_claims = self.effective_claims_for_params(&self.params);
        let target_params = build_migrated_params(&self.params, target_s)?;
        if target_s <= self.params.s {
            return Err(MigrateError::InvalidTargetS);
        }
        let target_claims =
            merge_claim_bundles_for_kappa_halve(&self.params, &target_params, &source_claims);
        let target_row_owners = collect_target_row_owners_by_index(&target_params, &target_claims)?;

        let mut sample_plan = Vec::<KappaHalveHydrationSource>::with_capacity(target_params.need);
        let mut sample_z_values = Vec::<Fp>::with_capacity(target_params.need);
        let mut seen_z = HashSet::new();
        for claim in &target_claims {
            let registry_index = claim.registry_index;
            let entry = *target_params
                .shard_registry
                .get(registry_index)
                .ok_or(MigrateError::InvalidPartnerShards)?;
            if !seen_z.insert(entry.key.z_bytes) {
                continue;
            }
            sample_plan.push(KappaHalveHydrationSource::Local(registry_index));
            sample_z_values.push(entry.z);
            if sample_plan.len() >= target_params.need {
                break;
            }
        }
        if sample_plan.len() < target_params.need {
            for &registry_index in helper_registry_indices {
                let entry = *target_params
                    .shard_registry
                    .get(registry_index)
                    .ok_or(MigrateError::InvalidPartnerShards)?;
                if !seen_z.insert(entry.key.z_bytes) {
                    continue;
                }
                sample_plan.push(KappaHalveHydrationSource::Helper(registry_index));
                sample_z_values.push(entry.z);
                if sample_plan.len() >= target_params.need {
                    break;
                }
            }
        }
        if sample_plan.len() < target_params.need {
            return Err(MigrateError::MissingPartnerShards);
        }
        let weights = precompute_barycentric(&sample_z_values);

        Ok(KappaHalveOneClaimWithPartnerPlan {
            target_params,
            source_claims,
            target_claims,
            target_row_owners,
            sample_plan,
            sample_z_values,
            weights,
        })
    }

    /// Precompute the stable local plan for adjacent one-claim kappa-halve.
    #[cfg(feature = "bench")]
    pub fn prepare_kappa_halve_one_claim_with_partner(
        &self,
        target_s: usize,
        helper_registry_indices: &[usize],
    ) -> Result<KappaHalveOneClaimWithPartnerPlan, MigrateError> {
        self.build_kappa_halve_one_claim_with_partner_plan(target_s, helper_registry_indices)
    }

    /// Adjacent one-claim kappa-double used by append-only log simulations.
    ///
    /// Each source claim keeps only its deterministic left child. The generic
    /// kappa-double path materializes both children and both row families because it
    /// does not know the final assignment policy; this path takes the policy as
    /// fixed and never allocates the discarded right child.
    pub fn migrate_kappa_double_one_claim_left(
        &mut self,
        target_s: usize,
        helper_evals: &[KappaDoubleRowEvalMigration],
    ) -> Result<(), MigrateError> {
        let helper_registry_indices = helper_evals
            .iter()
            .map(|helper| helper.registry_index)
            .collect::<Vec<_>>();
        let plan =
            self.build_kappa_double_one_claim_left_plan(target_s, &helper_registry_indices)?;
        self.apply_kappa_double_one_claim_left_plan(&plan, helper_evals)
    }

    fn apply_kappa_double_one_claim_left_plan(
        &mut self,
        plan: &KappaDoubleOneClaimLeftPlan,
        helper_evals: &[KappaDoubleRowEvalMigration],
    ) -> Result<(), MigrateError> {
        let mut migrated = self.kappa_double_encoded_state_retaining_left_children(
            plan.target_params.clone(),
            &plan.source_claims,
            &plan.target_claims,
        )?;
        migrated.hydrate_rows_from_target_evals_with_plan(helper_evals, plan)?;
        *self = migrated;
        Ok(())
    }

    /// Apply adjacent one-claim kappa-double using a precomputed local plan.
    #[cfg(feature = "bench")]
    pub fn migrate_kappa_double_one_claim_left_with_plan(
        &mut self,
        plan: &KappaDoubleOneClaimLeftPlan,
        helper_evals: &[KappaDoubleRowEvalMigration],
    ) -> Result<(), MigrateError> {
        self.apply_kappa_double_one_claim_left_plan(plan, helper_evals)
    }

    /// Bench-only split of `migrate_kappa_double_one_claim_left`.
    ///
    /// This exposes the local retained-child materialization phase without
    /// changing the production kappa-rescale API.
    #[cfg(feature = "bench")]
    pub fn bench_materialize_kappa_double_one_claim_left(
        &self,
        target_s: usize,
    ) -> Result<(UnauthStore, Vec<ClaimBundle>), MigrateError> {
        let source_claims = self.effective_claims_for_params(&self.params);
        let new_params = build_migrated_params(&self.params, target_s)?;
        if target_s >= self.params.s {
            return Err(MigrateError::InvalidTargetS);
        }
        let mut target_claims = Vec::with_capacity(source_claims.len());
        for claim in &source_claims {
            let (left, _) = kappa_double_child_registry_indices(claim.registry_index);
            let target_claim = ClaimBundle::exact_for_registry_index(&new_params, left)
                .ok_or(MigrateError::InvalidClaim)?;
            target_claims.push(target_claim);
        }
        let target_claims = normalize_claim_bundles(&new_params, target_claims);
        let migrated = self.kappa_double_encoded_state_retaining_left_children(
            new_params,
            &source_claims,
            &target_claims,
        )?;
        Ok((migrated, target_claims))
    }

    /// Bench-only retained-child materialization using a precomputed plan.
    #[cfg(feature = "bench")]
    pub fn bench_materialize_kappa_double_one_claim_left_with_plan(
        &self,
        plan: &KappaDoubleOneClaimLeftPlan,
    ) -> Result<(UnauthStore, Vec<ClaimBundle>), MigrateError> {
        let migrated = self.kappa_double_encoded_state_retaining_left_children(
            plan.target_params.clone(),
            &plan.source_claims,
            &plan.target_claims,
        )?;
        Ok((migrated, plan.target_claims.clone()))
    }

    /// Bench-only split of `migrate_kappa_double_one_claim_left`.
    ///
    /// Hydrates rows in a store already produced by
    /// `bench_materialize_kappa_double_one_claim_left`.
    #[cfg(feature = "bench")]
    pub fn bench_hydrate_kappa_double_one_claim_left_rows(
        &mut self,
        helper_evals: &[KappaDoubleRowEvalMigration],
        target_claims: &[ClaimBundle],
    ) -> Result<(), MigrateError> {
        self.hydrate_rows_from_target_evals_with_claims(helper_evals, target_claims)
    }

    /// Bench-only row hydration using a precomputed kappa-double plan.
    #[cfg(feature = "bench")]
    pub fn bench_hydrate_kappa_double_one_claim_left_rows_with_plan(
        &mut self,
        helper_evals: &[KappaDoubleRowEvalMigration],
        plan: &KappaDoubleOneClaimLeftPlan,
    ) -> Result<(), MigrateError> {
        self.hydrate_rows_from_target_evals_with_plan(helper_evals, plan)
    }

    /// Evaluate adjacent-kappa-double target children at explicit target row labels.
    pub fn handle_eval_kappa_double_target_at_row_labels_matrix(
        &self,
        target_s: usize,
        segs: &[usize],
        target_registry_indices: &[usize],
        row_labels: &[Fp],
    ) -> Result<Vec<Vec<Vec<Fp>>>, CodedReadError> {
        if segs.iter().any(|&seg| seg >= self.segments) {
            return Err(CodedReadError::NotEnoughShards);
        }
        let target_params = build_migrated_params(&self.params, target_s)
            .map_err(|_| CodedReadError::NotEnoughShards)?;
        let mut out = Vec::with_capacity(segs.len());
        for &seg in segs {
            let mut child_cache = HashMap::<usize, Vec<Fp>>::new();
            let mut by_target = Vec::with_capacity(target_registry_indices.len());
            for &target_registry_index in target_registry_indices {
                let source_registry_index =
                    kappa_halve_parent_registry_index(target_registry_index);
                if source_registry_index >= self.params.shard_registry.len()
                    || target_registry_index >= target_params.shard_registry.len()
                {
                    return Err(CodedReadError::NotEnoughShards);
                }
                if !child_cache.contains_key(&target_registry_index) {
                    if let Some(claims) = &self.claims {
                        let is_claimed = claims
                            .iter()
                            .any(|claim| claim.registry_index == source_registry_index);
                        if !is_claimed {
                            return Err(CodedReadError::NotEnoughShards);
                        }
                    }
                    let source_entry = self.params.shard_registry[source_registry_index];
                    let repaired_source;
                    let source_values = if let Some(values) =
                        self.coded
                            .shard_segment(&self.params, &source_entry.key, seg)
                    {
                        values
                    } else {
                        repaired_source = self.handle_repair_shard(seg, source_registry_index)?;
                        repaired_source.as_slice()
                    };
                    child_cache.insert(
                        target_registry_index,
                        kappa_double_primary_shard_child(
                            source_values,
                            target_registry_index % 2 == 0,
                        ),
                    );
                }
                let Some(child_values) = child_cache.get(&target_registry_index) else {
                    return Err(CodedReadError::NotEnoughShards);
                };
                let target_entry = target_params.shard_registry[target_registry_index];
                let evals = if row_labels.len() == 1 {
                    vec![shard_value_from_vector(
                        &target_params,
                        target_entry,
                        child_values,
                        row_labels[0],
                    )]
                } else {
                    shard_value_from_vector_batch(
                        &target_params,
                        target_entry,
                        child_values,
                        row_labels,
                    )
                };
                let mut row_values = Vec::with_capacity(row_labels.len());
                for value in evals {
                    let Some(value) = value else {
                        return Err(CodedReadError::NotEnoughShards);
                    };
                    row_values.push(value);
                }
                by_target.push(row_values);
            }
            out.push(by_target);
        }
        Ok(out)
    }

    /// Bench-only wrapper for compact primary-column evaluations at explicit
    /// row labels. This exposes the same value shape as the network
    /// `EvalShardsAtRowLabelsMatrix` request without widening the production
    /// unauthenticated-array API.
    #[cfg(feature = "bench")]
    pub fn bench_eval_shards_at_row_labels_matrix(
        &self,
        segs: &[usize],
        col_labels: &[Fp],
        row_labels: &[Fp],
    ) -> Result<Vec<Vec<Vec<Fp>>>, CodedReadError> {
        self.handle_eval_shards_at_row_labels_matrix(segs, col_labels, row_labels)
    }

    /// Bench-only adjacent kappa-halve primary materialization using direct partner
    /// primary material.
    #[cfg(feature = "bench")]
    pub fn bench_materialize_kappa_halve_one_claim_with_partner(
        &self,
        plan: &KappaHalveOneClaimWithPartnerPlan,
        partner_shards: &[super::PartnerShardMigration],
    ) -> Result<UnauthStore, MigrateError> {
        if plan.target_params.s <= self.params.s {
            return Err(MigrateError::InvalidTargetS);
        }
        let mut partner_by_index = HashMap::new();
        for partner in partner_shards {
            if partner.registry_index >= self.params.shard_registry.len()
                || partner.segments.len() != self.segments
            {
                return Err(MigrateError::InvalidPartnerShards);
            }
            if partner
                .segments
                .iter()
                .any(|segment_values| segment_values.len() != self.params.s)
            {
                return Err(MigrateError::InvalidPartnerShards);
            }
            partner_by_index.insert(partner.registry_index, partner);
        }

        let mut migrated =
            UnauthStore::new_with_claims(plan.target_params.clone(), plan.target_claims.clone());
        for target_claim in &plan.target_claims {
            let target_registry_index = target_claim.registry_index;
            let source_registry_index = 2 * target_registry_index;
            if !plan
                .source_claims
                .iter()
                .any(|claim| claim.registry_index == source_registry_index)
            {
                return Err(MigrateError::MissingPartnerShards);
            }
            let old_entry = self.params.shard_registry[source_registry_index];
            let new_entry = plan.target_params.shard_registry[target_registry_index];
            let partner = partner_by_index
                .get(&source_registry_index)
                .ok_or(MigrateError::MissingPartnerShards)?;
            let mut new_segments = Vec::with_capacity(self.segments);
            for seg in 0..self.segments {
                let old_values = self
                    .coded
                    .shard_segment(&self.params, &old_entry.key, seg)
                    .ok_or(MigrateError::MissingPartnerShards)?;
                let partner_values = partner
                    .segments
                    .get(seg)
                    .ok_or(MigrateError::InvalidPartnerShards)?;
                new_segments.push(kappa_halve_primary_shard(old_values, partner_values));
            }
            migrated.coded.insert_shard_family_segments(
                &plan.target_params,
                new_entry.key,
                new_segments,
            );
        }
        migrated.len = self.len;
        migrated.segments = self.segments;
        Ok(migrated)
    }

    /// Bench-only row hydration after direct partner-primary kappa-halve.
    #[cfg(feature = "bench")]
    pub fn bench_hydrate_kappa_halve_one_claim_rows_with_plan(
        &mut self,
        helper_evals: &[TargetRowEvalMigration],
        plan: &KappaHalveOneClaimWithPartnerPlan,
    ) -> Result<(), MigrateError> {
        if self.params.s != plan.target_params.s
            || self.params.need != plan.target_params.need
            || self.params.row_eval_points.len() != plan.target_params.row_eval_points.len()
        {
            return Err(MigrateError::InvalidTargetS);
        }
        self.coded.clear_rows();
        if plan.target_row_owners.is_empty() {
            return Ok(());
        }

        for helper in helper_evals {
            if helper.registry_index >= self.params.shard_registry.len()
                || helper.segments.len() != self.segments
            {
                return Err(MigrateError::InvalidPartnerShards);
            }
            if helper
                .segments
                .iter()
                .any(|values| values.len() != plan.target_row_owners.len())
            {
                return Err(MigrateError::InvalidPartnerShards);
            }
        }
        let mut helper_by_index = HashMap::<usize, &TargetRowEvalMigration>::new();
        for helper in helper_evals {
            if helper_by_index
                .insert(helper.registry_index, helper)
                .is_some()
            {
                return Err(MigrateError::InvalidPartnerShards);
            }
        }

        let target_row_labels = plan
            .target_row_owners
            .iter()
            .map(|(_, row_label, _)| *row_label)
            .collect::<Vec<_>>();
        let mut row_segments_by_pos =
            vec![Vec::with_capacity(self.segments); plan.target_row_owners.len()];
        for seg in 0..self.segments {
            let mut local_batches = HashMap::<usize, Vec<Fp>>::new();
            for source in &plan.sample_plan {
                let KappaHalveHydrationSource::Local(registry_index) = *source else {
                    continue;
                };
                if local_batches.contains_key(&registry_index) {
                    continue;
                }
                let entry = *self
                    .params
                    .shard_registry
                    .get(registry_index)
                    .ok_or(MigrateError::MissingLocalRows)?;
                let shard_values = self
                    .coded
                    .shard_segment(&self.params, &entry.key, seg)
                    .ok_or(MigrateError::MissingLocalRows)?;
                let evals = shard_value_from_vector_batch(
                    &self.params,
                    entry,
                    shard_values,
                    &target_row_labels,
                );
                let mut values = Vec::with_capacity(evals.len());
                for value in evals {
                    let Some(value) = value else {
                        return Err(MigrateError::MissingLocalRows);
                    };
                    values.push(value);
                }
                local_batches.insert(registry_index, values);
            }

            for row_pos in 0..plan.target_row_owners.len() {
                let mut row_values = Vec::with_capacity(self.params.need);
                for source in &plan.sample_plan {
                    match *source {
                        KappaHalveHydrationSource::Local(registry_index) => {
                            let Some(values) = local_batches.get(&registry_index) else {
                                return Err(MigrateError::MissingLocalRows);
                            };
                            let Some(&value) = values.get(row_pos) else {
                                return Err(MigrateError::MissingLocalRows);
                            };
                            row_values.push(value);
                        }
                        KappaHalveHydrationSource::Helper(registry_index) => {
                            let helper = helper_by_index
                                .get(&registry_index)
                                .ok_or(MigrateError::MissingPartnerShards)?;
                            let values = helper
                                .segments
                                .get(seg)
                                .ok_or(MigrateError::InvalidPartnerShards)?;
                            let Some(&value) = values.get(row_pos) else {
                                return Err(MigrateError::InvalidPartnerShards);
                            };
                            row_values.push(value);
                        }
                    }
                }

                if row_values.len() != self.params.need {
                    return Err(MigrateError::MissingPartnerShards);
                }
                let row_segment = materialize_row_on_eval_points_from_samples(
                    &self.params,
                    &plan.sample_z_values,
                    &row_values,
                    Some(&plan.weights),
                )
                .map_err(|_| MigrateError::MissingPartnerShards)?;
                row_segments_by_pos[row_pos].push(row_segment);
            }
        }
        for ((row_label_index, _, owner_key), row_segments) in plan
            .target_row_owners
            .iter()
            .copied()
            .zip(row_segments_by_pos.into_iter())
        {
            let row_key = RowKey::new(owner_key, self.params.row_labels[row_label_index]);
            self.coded
                .insert_row_family_segments(&self.params, row_key, row_segments);
        }
        Ok(())
    }

    #[cfg(feature = "bench")]
    fn hydrate_rows_from_target_evals_with_claims(
        &mut self,
        helper_evals: &[KappaDoubleRowEvalMigration],
        claims: &[ClaimBundle],
    ) -> Result<(), MigrateError> {
        let claims = normalize_claim_bundles(&self.params, claims.to_vec());
        self.coded.clear_rows();
        if claims.is_empty() {
            return Ok(());
        }

        let target_row_owners = collect_target_row_owners_by_index(&self.params, &claims)?;
        if target_row_owners.is_empty() {
            return Ok(());
        }
        for helper in helper_evals {
            if helper.registry_index >= self.params.shard_registry.len()
                || helper.segments.len() != self.segments
            {
                return Err(MigrateError::InvalidPartnerShards);
            }
            if helper
                .segments
                .iter()
                .any(|values| values.len() != target_row_owners.len())
            {
                return Err(MigrateError::InvalidPartnerShards);
            }
        }

        let mut helper_by_index = BTreeMap::<usize, &KappaDoubleRowEvalMigration>::new();
        for helper in helper_evals {
            if helper_by_index
                .insert(helper.registry_index, helper)
                .is_some()
            {
                return Err(MigrateError::InvalidPartnerShards);
            }
        }
        let mut local_registry_indices = Vec::new();
        for (registry_index, entry) in self.params.shard_registry.iter().copied().enumerate() {
            if self.coded.contains_shard(&entry.key) {
                if self.coded.shard_family_segment_count(&entry.key) != Some(self.segments) {
                    return Err(MigrateError::MissingLocalRows);
                }
                local_registry_indices.push(registry_index);
            }
        }
        #[derive(Clone, Copy)]
        enum HydrationSource {
            Local(usize),
            Helper(usize),
        }
        let mut sample_plan = Vec::<HydrationSource>::with_capacity(self.params.need);
        let mut sample_z_values = Vec::<Fp>::with_capacity(self.params.need);
        let mut seen_z = HashSet::new();
        for &registry_index in &local_registry_indices {
            let entry = self.params.shard_registry[registry_index];
            if !seen_z.insert(entry.key.z_bytes) {
                continue;
            }
            sample_plan.push(HydrationSource::Local(registry_index));
            sample_z_values.push(entry.z);
            if sample_plan.len() >= self.params.need {
                break;
            }
        }
        if sample_plan.len() < self.params.need {
            for &registry_index in helper_by_index.keys() {
                let entry = self.params.shard_registry[registry_index];
                if !seen_z.insert(entry.key.z_bytes) {
                    continue;
                }
                sample_plan.push(HydrationSource::Helper(registry_index));
                sample_z_values.push(entry.z);
                if sample_plan.len() >= self.params.need {
                    break;
                }
            }
        }
        if sample_plan.len() < self.params.need {
            return Err(MigrateError::MissingPartnerShards);
        }
        let weights = precompute_barycentric(&sample_z_values);

        let target_row_labels = target_row_owners
            .iter()
            .map(|(_, row_label, _)| *row_label)
            .collect::<Vec<_>>();
        let mut row_segments_by_pos =
            vec![Vec::with_capacity(self.segments); target_row_owners.len()];
        for seg in 0..self.segments {
            let mut local_batches = HashMap::<usize, Vec<Fp>>::new();
            for source in &sample_plan {
                let HydrationSource::Local(registry_index) = *source else {
                    continue;
                };
                if local_batches.contains_key(&registry_index) {
                    continue;
                }
                let entry = self.params.shard_registry[registry_index];
                let shard_values = self
                    .coded
                    .shard_segment(&self.params, &entry.key, seg)
                    .ok_or(MigrateError::MissingLocalRows)?;
                let evals = shard_value_from_vector_batch(
                    &self.params,
                    entry,
                    shard_values,
                    &target_row_labels,
                );
                let mut values = Vec::with_capacity(evals.len());
                for value in evals {
                    let Some(value) = value else {
                        return Err(MigrateError::MissingLocalRows);
                    };
                    values.push(value);
                }
                local_batches.insert(registry_index, values);
            }

            for row_pos in 0..target_row_owners.len() {
                let mut row_values = Vec::with_capacity(self.params.need);
                for source in &sample_plan {
                    match *source {
                        HydrationSource::Local(registry_index) => {
                            let Some(values) = local_batches.get(&registry_index) else {
                                return Err(MigrateError::MissingLocalRows);
                            };
                            let Some(&value) = values.get(row_pos) else {
                                return Err(MigrateError::MissingLocalRows);
                            };
                            row_values.push(value);
                        }
                        HydrationSource::Helper(registry_index) => {
                            let helper = helper_by_index
                                .get(&registry_index)
                                .ok_or(MigrateError::MissingPartnerShards)?;
                            let values = helper
                                .segments
                                .get(seg)
                                .ok_or(MigrateError::InvalidPartnerShards)?;
                            let Some(&value) = values.get(row_pos) else {
                                return Err(MigrateError::InvalidPartnerShards);
                            };
                            row_values.push(value);
                        }
                    }
                }

                if row_values.len() != self.params.need {
                    return Err(MigrateError::MissingPartnerShards);
                }
                let row_segment = materialize_row_on_eval_points_from_samples(
                    &self.params,
                    &sample_z_values,
                    &row_values,
                    Some(&weights),
                )
                .map_err(|_| MigrateError::MissingPartnerShards)?;
                row_segments_by_pos[row_pos].push(row_segment);
            }
        }
        for ((row_label_index, _, owner_key), row_segments) in target_row_owners
            .iter()
            .copied()
            .zip(row_segments_by_pos.into_iter())
        {
            let row_key = RowKey::new(owner_key, self.params.row_labels[row_label_index]);
            self.coded
                .insert_row_family_segments(&self.params, row_key, row_segments);
        }
        Ok(())
    }

    fn hydrate_rows_from_target_evals_with_plan(
        &mut self,
        helper_evals: &[KappaDoubleRowEvalMigration],
        plan: &KappaDoubleOneClaimLeftPlan,
    ) -> Result<(), MigrateError> {
        if self.params.s != plan.target_params.s
            || self.params.need != plan.target_params.need
            || self.params.row_eval_points.len() != plan.target_params.row_eval_points.len()
        {
            return Err(MigrateError::InvalidTargetS);
        }
        self.coded.clear_rows();
        if plan.target_row_owners.is_empty() {
            return Ok(());
        }

        for helper in helper_evals {
            if helper.registry_index >= self.params.shard_registry.len()
                || helper.segments.len() != self.segments
            {
                return Err(MigrateError::InvalidPartnerShards);
            }
            if helper
                .segments
                .iter()
                .any(|values| values.len() != plan.target_row_owners.len())
            {
                return Err(MigrateError::InvalidPartnerShards);
            }
        }

        let mut helper_by_index = HashMap::<usize, &KappaDoubleRowEvalMigration>::new();
        for helper in helper_evals {
            if helper_by_index
                .insert(helper.registry_index, helper)
                .is_some()
            {
                return Err(MigrateError::InvalidPartnerShards);
            }
        }

        let target_row_labels = plan
            .target_row_owners
            .iter()
            .map(|(_, row_label, _)| *row_label)
            .collect::<Vec<_>>();
        let mut row_segments_by_pos =
            vec![Vec::with_capacity(self.segments); plan.target_row_owners.len()];
        for seg in 0..self.segments {
            let mut local_batches = HashMap::<usize, Vec<Fp>>::new();
            for source in &plan.sample_plan {
                let KappaDoubleHydrationSource::Local(registry_index) = *source else {
                    continue;
                };
                if local_batches.contains_key(&registry_index) {
                    continue;
                }
                let entry = *self
                    .params
                    .shard_registry
                    .get(registry_index)
                    .ok_or(MigrateError::MissingLocalRows)?;
                let shard_values = self
                    .coded
                    .shard_segment(&self.params, &entry.key, seg)
                    .ok_or(MigrateError::MissingLocalRows)?;
                let evals = shard_value_from_vector_batch(
                    &self.params,
                    entry,
                    shard_values,
                    &target_row_labels,
                );
                let mut values = Vec::with_capacity(evals.len());
                for value in evals {
                    let Some(value) = value else {
                        return Err(MigrateError::MissingLocalRows);
                    };
                    values.push(value);
                }
                local_batches.insert(registry_index, values);
            }

            for row_pos in 0..plan.target_row_owners.len() {
                let mut row_values = Vec::with_capacity(self.params.need);
                for source in &plan.sample_plan {
                    match *source {
                        KappaDoubleHydrationSource::Local(registry_index) => {
                            let Some(values) = local_batches.get(&registry_index) else {
                                return Err(MigrateError::MissingLocalRows);
                            };
                            let Some(&value) = values.get(row_pos) else {
                                return Err(MigrateError::MissingLocalRows);
                            };
                            row_values.push(value);
                        }
                        KappaDoubleHydrationSource::Helper(registry_index) => {
                            let helper = helper_by_index
                                .get(&registry_index)
                                .ok_or(MigrateError::MissingPartnerShards)?;
                            let values = helper
                                .segments
                                .get(seg)
                                .ok_or(MigrateError::InvalidPartnerShards)?;
                            let Some(&value) = values.get(row_pos) else {
                                return Err(MigrateError::InvalidPartnerShards);
                            };
                            row_values.push(value);
                        }
                    }
                }

                if row_values.len() != self.params.need {
                    return Err(MigrateError::MissingPartnerShards);
                }
                let row_segment = materialize_row_on_eval_points_from_samples(
                    &self.params,
                    &plan.sample_z_values,
                    &row_values,
                    Some(&plan.weights),
                )
                .map_err(|_| MigrateError::MissingPartnerShards)?;
                row_segments_by_pos[row_pos].push(row_segment);
            }
        }
        for ((row_label_index, _, owner_key), row_segments) in plan
            .target_row_owners
            .iter()
            .copied()
            .zip(row_segments_by_pos.into_iter())
        {
            let row_key = RowKey::new(owner_key, self.params.row_labels[row_label_index]);
            self.coded
                .insert_row_family_segments(&self.params, row_key, row_segments);
        }
        Ok(())
    }
}
