//! Registry abstraction for party discovery and shard assignments.

use crate::coded_array::accept_candidate;
use crate::core::NodeLinearization;
use crate::field::Fp;
use crate::net::wire::NetFp;
use crate::unauth::{
    ClaimBundle, CodedArrayConfig, CodedArrayParams, RowFamilyId, ShardKey, ShardRegEntry,
    build_coded_params, build_migrated_params, build_segment_double_params,
    build_shard_registry_deterministic, kappa_double_child_registry_indices,
    kappa_halve_parent_registry_index, registry_supports_sibling_kappa_halve,
};
use crate::value::encode_value;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::Arc;

/// Network-facing unauthenticated (coded array) config.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetUnauthConfig {
    pub segment_len: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout_base_n_grid: Option<usize>,
    pub s: usize,
    pub shard_count: usize,
    pub seed: u64,
}

/// Network-facing authenticated (Merkle) config.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetAuthConfig {
    pub tree_height: usize,
    #[serde(default)]
    pub node_linearization: NodeLinearization,
    pub allocated_leaves: usize,
}

/// Network-facing system parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetSystemConfig {
    pub unauth: NetUnauthConfig,
    pub auth: NetAuthConfig,
    #[serde(default)]
    pub unauth_array_len: Option<u64>,
}

/// JSON-friendly shard registry entry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetShardRegEntry {
    pub a: NetFp,
    pub z: NetFp,
}

/// Convert JSON-friendly shard registry entries into core registry entries.
pub fn convert_shard_registry(entries: &[NetShardRegEntry]) -> Vec<ShardRegEntry> {
    entries
        .iter()
        .map(|entry| {
            let a = entry.a.0;
            let z = entry.z.0;
            let a_bytes = encode_value(a);
            let z_bytes = encode_value(z);
            ShardRegEntry {
                a,
                z,
                key: ShardKey { a_bytes, z_bytes },
            }
        })
        .collect()
}

/// Basic party metadata returned by a registry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartyInfo {
    pub party_id: u64,
    pub rpc_host: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pubkey: Option<String>,
    pub active: bool,
}

/// Assignment list for a party (registry indices).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartyAssignments {
    pub party_id: u64,
    pub indices: Vec<u32>,
}

/// Row-template metadata for one unauthenticated epoch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetUnauthEpoch {
    #[serde(default)]
    pub epoch_id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initialized_segments: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows_per_shard: Option<u32>,
    /// Deprecated compatibility field. Layout is derived from
    /// `system.unauth.layout_base_n_grid`; if this field is present it must
    /// match that derived layout.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logical_to_grid_index: Option<Vec<u64>>,
    pub row_labels: Vec<NetFp>,
    pub row_eval_points: Vec<NetFp>,
}

/// One registry-published claim bundle: one primary column plus an exact row-set descriptor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartyClaimBundle {
    pub party_id: u64,
    pub registry_index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row_family_id: Option<u32>,
    #[serde(default)]
    pub row_label_indices: Vec<u32>,
}

/// Errors returned by registry lookups.
#[derive(Clone, Debug)]
pub enum RegistryError {
    Io(String),
    Parse(String),
    Invalid(String),
    MissingParty(u64),
    NotSupported(String),
}

/// Registry abstraction for fetching parties and assignments.
pub trait RegistryClient: Send + Sync {
    fn fetch_parties(&self) -> Result<Vec<PartyInfo>, RegistryError>;
    fn fetch_assignments(&self, party_id: u64) -> Result<Vec<u32>, RegistryError>;
    fn fetch_unauth_epoch(&self) -> Result<Option<NetUnauthEpoch>, RegistryError> {
        Ok(None)
    }
    fn fetch_unauth_claims(&self) -> Result<Vec<PartyClaimBundle>, RegistryError> {
        Ok(Vec::new())
    }
}

/// JSON file format used by the mock registry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MockRegistryFile {
    pub system: NetSystemConfig,
    pub parties: Vec<PartyInfo>,
    pub assignments: Vec<PartyAssignments>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unauth_epoch: Option<NetUnauthEpoch>,
    #[serde(default)]
    pub unauth_claims: Vec<PartyClaimBundle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard_registry: Option<Vec<NetShardRegEntry>>,
}

impl MockRegistryFile {
    pub fn load(path: &Path) -> Result<Self, RegistryError> {
        let data = fs::read_to_string(path).map_err(|err| RegistryError::Io(err.to_string()))?;
        serde_json::from_str(&data).map_err(|err| RegistryError::Parse(err.to_string()))
    }
}

/// Build per-party registry indices aligned to parties sorted by party_id.
pub fn build_party_registry_indices(
    parties: &[PartyInfo],
    assignments: &[PartyAssignments],
) -> Vec<Vec<usize>> {
    let mut sorted = parties.to_vec();
    sorted.sort_by_key(|party| party.party_id);
    let mut index_by_id = HashMap::new();
    for (idx, party) in sorted.iter().enumerate() {
        index_by_id.insert(party.party_id, idx);
    }
    let mut out: Vec<Vec<usize>> = vec![Vec::new(); sorted.len()];
    for entry in assignments {
        if let Some(idx) = index_by_id.get(&entry.party_id) {
            let mut indices: Vec<usize> = entry.indices.iter().map(|v| *v as usize).collect();
            indices.sort_unstable();
            indices.dedup();
            out[*idx] = indices;
        }
    }
    out
}

/// Build per-party registry indices from explicit unauth claim bundles.
pub fn build_party_registry_indices_from_claims(
    parties: &[PartyInfo],
    claims: &[PartyClaimBundle],
) -> Vec<Vec<usize>> {
    let mut sorted = parties.to_vec();
    sorted.sort_by_key(|party| party.party_id);
    let mut index_by_id = HashMap::new();
    for (idx, party) in sorted.iter().enumerate() {
        index_by_id.insert(party.party_id, idx);
    }
    let mut out: Vec<Vec<usize>> = vec![Vec::new(); sorted.len()];
    for claim in claims {
        if let Some(idx) = index_by_id.get(&claim.party_id) {
            out[*idx].push(claim.registry_index as usize);
        }
    }
    for indices in &mut out {
        indices.sort_unstable();
        indices.dedup();
    }
    out
}

/// Resolve the shard registry, using the deterministic builder when absent.
pub fn resolve_shard_registry(file: &MockRegistryFile) -> Vec<ShardRegEntry> {
    match file.shard_registry.as_deref() {
        Some(entries) => convert_shard_registry(entries),
        None => {
            let n_grid = file.system.unauth.segment_len;
            build_shard_registry_deterministic(
                file.system.unauth.shard_count,
                file.system.unauth.s,
                n_grid,
            )
        }
    }
}

fn encode_shard_registry(entries: &[ShardRegEntry]) -> Vec<NetShardRegEntry> {
    entries
        .iter()
        .map(|entry| NetShardRegEntry {
            a: NetFp(entry.a),
            z: NetFp(entry.z),
        })
        .collect()
}

fn assignments_from_claims(
    parties: &[PartyInfo],
    claims: &[PartyClaimBundle],
) -> Vec<PartyAssignments> {
    let mut by_party: HashMap<u64, Vec<u32>> = HashMap::new();
    for claim in claims {
        by_party
            .entry(claim.party_id)
            .or_default()
            .push(claim.registry_index);
    }
    let mut out = Vec::new();
    for party in parties {
        let mut indices = by_party.remove(&party.party_id).unwrap_or_default();
        indices.sort_unstable();
        indices.dedup();
        out.push(PartyAssignments {
            party_id: party.party_id,
            indices,
        });
    }
    out
}

fn active_claim_count(file: &MockRegistryFile, claims: &[PartyClaimBundle]) -> usize {
    if file.parties.is_empty() {
        return claims.len();
    }
    let active_party_ids: HashSet<_> = file
        .parties
        .iter()
        .filter(|party| party.active)
        .map(|party| party.party_id)
        .collect();
    claims
        .iter()
        .filter(|claim| active_party_ids.contains(&claim.party_id))
        .count()
}

fn extend_shard_registry_preserving_existing(
    params: &CodedArrayParams,
    new_count: usize,
) -> Vec<ShardRegEntry> {
    let mut registry = params.shard_registry.clone();
    let mut seen_z: HashSet<_> = registry.iter().map(|entry| entry.key.z_bytes).collect();
    let mut cand = 2u64;
    while registry.len() < new_count {
        if let Some(entry) = accept_candidate(Fp::from(cand), params.s, params.n_grid, &seen_z) {
            seen_z.insert(entry.key.z_bytes);
            registry.push(entry);
        }
        cand += 1;
    }
    registry
}

/// Build the default epoch metadata implied by the current coded params.
pub fn default_unauth_epoch(params: &CodedArrayParams) -> NetUnauthEpoch {
    NetUnauthEpoch {
        epoch_id: 0,
        initialized_segments: None,
        rows_per_shard: Some(params.rows_per_shard as u32),
        logical_to_grid_index: None,
        row_labels: params.row_labels.iter().copied().map(NetFp).collect(),
        row_eval_points: params.row_eval_points.iter().copied().map(NetFp).collect(),
    }
}

fn base_party_registry_indices(file: &MockRegistryFile, registry_len: usize) -> Vec<Vec<usize>> {
    if file.assignments.is_empty() || file.parties.is_empty() {
        vec![(0..registry_len).collect()]
    } else {
        build_party_registry_indices(&file.parties, &file.assignments)
    }
}

fn decode_net_fps(values: &[NetFp]) -> Vec<crate::field::Fp> {
    values.iter().map(|value| value.0).collect()
}

/// Apply registry epoch metadata to coded params.
pub fn apply_unauth_epoch(
    params: &mut CodedArrayParams,
    epoch: &NetUnauthEpoch,
) -> Result<(), RegistryError> {
    let row_labels = decode_net_fps(&epoch.row_labels);
    if row_labels.len() != params.row_labels.len() {
        return Err(RegistryError::Invalid(format!(
            "expected {} row labels, got {}",
            params.row_labels.len(),
            row_labels.len()
        )));
    }
    let mut seen_labels = HashSet::new();
    for &row_label in &row_labels {
        if !seen_labels.insert(encode_value(row_label)) {
            return Err(RegistryError::Invalid(
                "duplicate row label in unauth_epoch".to_string(),
            ));
        }
    }

    let row_eval_points = decode_net_fps(&epoch.row_eval_points);
    if row_eval_points.len() != params.need {
        return Err(RegistryError::Invalid(format!(
            "expected {} row evaluation points, got {}",
            params.need,
            row_eval_points.len()
        )));
    }
    let mut seen_eval_points = HashSet::new();
    for &z_eval in &row_eval_points {
        if !seen_eval_points.insert(encode_value(z_eval)) {
            return Err(RegistryError::Invalid(
                "duplicate row evaluation point in unauth_epoch".to_string(),
            ));
        }
    }

    if let Some(rows_per_shard) = epoch.rows_per_shard {
        if rows_per_shard == 0 {
            return Err(RegistryError::Invalid(
                "rows_per_shard must be greater than zero".to_string(),
            ));
        }
        if row_labels.len() % rows_per_shard as usize != 0 {
            return Err(RegistryError::Invalid(format!(
                "rows_per_shard {} does not divide {} row labels",
                rows_per_shard,
                row_labels.len()
            )));
        }
        params.rows_per_shard = rows_per_shard as usize;
    }
    params.row_labels = row_labels;
    params.row_eval_points = row_eval_points;
    if let Some(legacy_logical_to_grid_index) = &epoch.logical_to_grid_index {
        if legacy_logical_to_grid_index.len() != params.n_grid {
            return Err(RegistryError::Invalid(format!(
                "expected {} logical grid indices, got {}",
                params.n_grid,
                legacy_logical_to_grid_index.len()
            )));
        }
        let mut seen = vec![false; params.n_grid];
        let mut decoded = vec![0usize; params.n_grid];
        for (logical_offset, &grid_index_u64) in legacy_logical_to_grid_index.iter().enumerate() {
            if grid_index_u64 >= params.n_grid as u64 {
                return Err(RegistryError::Invalid(format!(
                    "logical offset {} maps to out-of-range grid index {}",
                    logical_offset, grid_index_u64
                )));
            }
            let grid_index = grid_index_u64 as usize;
            if seen[grid_index] {
                return Err(RegistryError::Invalid(format!(
                    "logical grid layout repeats grid index {}",
                    grid_index
                )));
            }
            seen[grid_index] = true;
            decoded[logical_offset] = grid_index;
        }
        if decoded != params.logical_to_grid_index {
            return Err(RegistryError::Invalid(
                "unauth_epoch.logical_to_grid_index is deprecated; registry layout must match the canonical layout derived from system.unauth.layout_base_n_grid"
                    .to_string(),
            ));
        }
    }
    params.refresh_row_metadata_caches();
    Ok(())
}

/// Build default claim bundles from flat server assignments.
pub fn default_unauth_claims(
    params: &CodedArrayParams,
    assignments: &[PartyAssignments],
) -> Result<Vec<PartyClaimBundle>, RegistryError> {
    let mut claims = Vec::new();
    for assignment in assignments {
        let mut seen = HashSet::new();
        for &registry_index in &assignment.indices {
            if !seen.insert(registry_index) {
                continue;
            }
            let row_label_indices =
                params.row_label_indices_for_registry_index(registry_index as usize);
            if row_label_indices.len() != params.rows_per_shard {
                return Err(RegistryError::Invalid(format!(
                    "flat assignments cannot synthesize unique default rows for registry index {}: need {}, got {}",
                    registry_index,
                    params.rows_per_shard,
                    row_label_indices.len()
                )));
            }
            let row_family_id = params
                .row_family_id_for_registry_index(registry_index as usize)
                .map(|family_id| family_id.0 as u32);
            claims.push(PartyClaimBundle {
                party_id: assignment.party_id,
                registry_index,
                row_family_id,
                row_label_indices: if row_family_id.is_some() {
                    Vec::new()
                } else {
                    row_label_indices
                        .into_iter()
                        .map(|idx| idx as u32)
                        .collect()
                },
            });
        }
    }
    Ok(claims)
}

fn normalize_u32_indices(indices: &[u32]) -> Vec<u32> {
    let mut out = indices.to_vec();
    out.sort_unstable();
    out.dedup();
    out
}

fn row_family_rows_u32(
    params: &CodedArrayParams,
    row_family_id: u32,
) -> Result<Vec<u32>, RegistryError> {
    let family_id = RowFamilyId(row_family_id as usize);
    params
        .row_label_indices_for_row_family(family_id)
        .map(|indices| indices.into_iter().map(|idx| idx as u32).collect())
        .ok_or_else(|| {
            RegistryError::Invalid(format!(
                "claim references invalid row_family_id {}",
                row_family_id
            ))
        })
}

fn exact_row_family_id_for_rows(
    params: &CodedArrayParams,
    row_label_indices: &[u32],
) -> Option<u32> {
    let indices: Vec<_> = row_label_indices.iter().map(|idx| *idx as usize).collect();
    params
        .row_family_id_for_row_label_indices(&indices)
        .map(|family_id| family_id.0 as u32)
}

fn normalize_claim_rows_and_family(
    params: &CodedArrayParams,
    claim: &PartyClaimBundle,
) -> Result<(Option<u32>, Vec<u32>), RegistryError> {
    let explicit_rows = normalize_u32_indices(&claim.row_label_indices);
    match claim.row_family_id {
        Some(row_family_id) => {
            let family_rows = row_family_rows_u32(params, row_family_id)?;
            if !explicit_rows.is_empty() && explicit_rows != family_rows {
                return Err(RegistryError::Invalid(format!(
                    "claim for registry index {} has row_family_id {} but explicit rows {:?}",
                    claim.registry_index, row_family_id, explicit_rows
                )));
            }
            Ok((Some(row_family_id), family_rows))
        }
        None => Ok((
            exact_row_family_id_for_rows(params, &explicit_rows),
            explicit_rows,
        )),
    }
}

fn make_party_claim_bundle(
    params: &CodedArrayParams,
    party_id: u64,
    registry_index: u32,
    row_label_indices: Vec<u32>,
) -> PartyClaimBundle {
    let normalized_rows = normalize_u32_indices(&row_label_indices);
    let row_family_id = exact_row_family_id_for_rows(params, &normalized_rows);
    PartyClaimBundle {
        party_id,
        registry_index,
        row_family_id,
        row_label_indices: if row_family_id.is_some() {
            Vec::new()
        } else {
            normalized_rows
        },
    }
}

fn resolve_unauth_claims(
    params: &CodedArrayParams,
    claims: &[PartyClaimBundle],
) -> Result<Vec<PartyClaimBundle>, RegistryError> {
    let mut resolved = Vec::with_capacity(claims.len());
    for claim in claims {
        let registry_index = claim.registry_index as usize;
        if registry_index >= params.shard_registry.len() {
            return Err(RegistryError::Invalid(format!(
                "claim references out-of-range registry index {}",
                registry_index
            )));
        }
        let (row_family_id, row_label_indices) = normalize_claim_rows_and_family(params, claim)?;
        resolved.push(PartyClaimBundle {
            party_id: claim.party_id,
            registry_index: claim.registry_index,
            row_family_id,
            row_label_indices,
        });
    }
    Ok(resolved)
}

/// Validate explicit unauth claim bundles against the current coded params.
pub fn validate_unauth_claims(
    params: &CodedArrayParams,
    claims: &[PartyClaimBundle],
) -> Result<(), RegistryError> {
    let claims = resolve_unauth_claims(params, claims)?;
    let mut owners = HashMap::new();
    let mut row_owners = HashMap::new();
    for claim in &claims {
        let registry_index = claim.registry_index as usize;
        if let Some(existing_party) = owners.insert(registry_index, claim.party_id) {
            return Err(RegistryError::Invalid(format!(
                "registry index {} claimed by both party {} and party {}",
                registry_index, existing_party, claim.party_id
            )));
        }
        let mut seen_rows = HashSet::new();
        for &row_label_index in &claim.row_label_indices {
            let row_label_index = row_label_index as usize;
            if row_label_index >= params.row_labels.len() {
                return Err(RegistryError::Invalid(format!(
                    "claim references out-of-range row label index {}",
                    row_label_index
                )));
            }
            if !seen_rows.insert(row_label_index) {
                return Err(RegistryError::Invalid(format!(
                    "claim for registry index {} repeats row label index {}",
                    registry_index, row_label_index
                )));
            }
            if let Some((existing_registry_index, existing_party_id)) =
                row_owners.insert(row_label_index, (registry_index, claim.party_id))
            {
                return Err(RegistryError::Invalid(format!(
                    "row label index {} claimed by registry index {} (party {}) and registry index {} (party {})",
                    row_label_index,
                    existing_registry_index,
                    existing_party_id,
                    registry_index,
                    claim.party_id
                )));
            }
        }
    }
    Ok(())
}

/// Build coded params plus explicit unauth claims from a registry file.
pub fn build_unauth_registry_view(
    file: &MockRegistryFile,
) -> Result<(CodedArrayParams, Vec<PartyClaimBundle>), RegistryError> {
    let shard_registry = resolve_shard_registry(file);
    let cfg = CodedArrayConfig {
        segment_len: file.system.unauth.segment_len,
        s: file.system.unauth.s,
        shard_count: file.system.unauth.shard_count,
        shard_registry,
        party_registry_indices: base_party_registry_indices(file, file.system.unauth.shard_count),
        seed: file.system.unauth.seed,
    };
    let mut params = build_coded_params(&cfg);
    let layout_base_n_grid = file
        .system
        .unauth
        .layout_base_n_grid
        .unwrap_or(params.n_grid);
    params
        .apply_canonical_logical_layout(layout_base_n_grid)
        .map_err(|err| {
            RegistryError::Invalid(format!("invalid canonical unauth layout: {:?}", err))
        })?;
    let epoch = file
        .unauth_epoch
        .clone()
        .unwrap_or_else(|| default_unauth_epoch(&params));
    apply_unauth_epoch(&mut params, &epoch)?;

    let claims = if file.unauth_claims.is_empty() {
        default_unauth_claims(&params, &file.assignments)?
    } else {
        resolve_unauth_claims(&params, &file.unauth_claims)?
    };
    let claims = resolve_unauth_claims(&params, &claims)?;
    validate_unauth_claims(&params, &claims)?;

    params.party_registry_indices = if !claims.is_empty() && !file.parties.is_empty() {
        build_party_registry_indices_from_claims(&file.parties, &claims)
    } else {
        base_party_registry_indices(file, params.shard_registry.len())
    };
    Ok((params, claims))
}

/// Resolve one party's local claim bundles from registry claim metadata.
pub fn claim_bundles_for_party(
    params: &CodedArrayParams,
    claims: &[PartyClaimBundle],
    party_id: u64,
) -> Result<Vec<ClaimBundle>, RegistryError> {
    let claims = resolve_unauth_claims(params, claims)?;
    let mut local = Vec::new();
    for claim in &claims {
        if claim.party_id != party_id {
            continue;
        }
        let registry_index = claim.registry_index as usize;
        let row_family_id = claim
            .row_family_id
            .and_then(|family_id| {
                let family_id = RowFamilyId(family_id as usize);
                params
                    .row_label_indices_for_row_family(family_id)
                    .map(|_| family_id)
            })
            .ok_or_else(|| {
                RegistryError::Invalid(format!(
                    "claim for registry index {} did not resolve to one exact row family",
                    registry_index
                ))
            })?;
        local.push(ClaimBundle {
            registry_index,
            row_family_id,
        });
    }
    Ok(local)
}

/// Rewrite a registry file for a migrated unauth mode, keeping claim owners stable.
pub fn migrate_unauth_registry_file(
    file: &mut MockRegistryFile,
    target_s: usize,
) -> Result<(), RegistryError> {
    let (params, claims) = build_unauth_registry_view(file)?;
    let migrated = build_migrated_params(&params, target_s)
        .map_err(|err| RegistryError::Invalid(format!("{:?}", err)))?;
    let migrated_claims: Vec<_> = if target_s < params.s {
        claims
            .into_iter()
            .map(|claim| {
                let (left, right) =
                    kappa_double_child_registry_indices(claim.registry_index as usize);
                Ok(vec![
                    make_party_claim_bundle(
                        &migrated,
                        claim.party_id,
                        left as u32,
                        migrated
                            .row_label_indices_for_registry_index(left)
                            .into_iter()
                            .map(|idx| idx as u32)
                            .collect(),
                    ),
                    make_party_claim_bundle(
                        &migrated,
                        claim.party_id,
                        right as u32,
                        migrated
                            .row_label_indices_for_registry_index(right)
                            .into_iter()
                            .map(|idx| idx as u32)
                            .collect(),
                    ),
                ])
            })
            .collect::<Result<Vec<_>, RegistryError>>()?
            .into_iter()
            .flatten()
            .collect()
    } else if registry_supports_sibling_kappa_halve(&params) {
        let mut by_parent = HashMap::<(u64, u32), Vec<PartyClaimBundle>>::new();
        for claim in claims {
            let parent = kappa_halve_parent_registry_index(claim.registry_index as usize) as u32;
            by_parent
                .entry((claim.party_id, parent))
                .or_default()
                .push(claim);
        }
        let mut merged = Vec::with_capacity(by_parent.len());
        for ((party_id, parent), family) in by_parent {
            let _ = family;
            merged.push(make_party_claim_bundle(
                &migrated,
                party_id,
                parent,
                migrated
                    .row_label_indices_for_registry_index(parent as usize)
                    .into_iter()
                    .map(|idx| idx as u32)
                    .collect(),
            ));
        }
        merged.sort_by_key(|claim| (claim.party_id, claim.registry_index));
        merged
    } else {
        claims
            .into_iter()
            .map(|claim| {
                make_party_claim_bundle(
                    &migrated,
                    claim.party_id,
                    claim.registry_index,
                    claim.row_label_indices,
                )
            })
            .collect()
    };
    validate_unauth_claims(&migrated, &migrated_claims)?;
    file.system.unauth.s = target_s;
    file.system.unauth.shard_count = migrated.shard_registry.len();
    file.shard_registry = Some(encode_shard_registry(&migrated.shard_registry));
    file.unauth_epoch = Some(default_unauth_epoch(&migrated));
    file.unauth_claims = migrated_claims;
    file.assignments = assignments_from_claims(&file.parties, &file.unauth_claims);
    Ok(())
}

/// Rewrite a registry file after segment doubling.
///
/// This is the registry-side version of the segment-double note: `n_grid` and
/// `need` double, `s` stays fixed, and the canonical layout base is preserved
/// so every registry view derives the same even/odd physical layout.
pub fn segment_double_unauth_registry_file(
    file: &mut MockRegistryFile,
) -> Result<(), RegistryError> {
    let (params, claims) = build_unauth_registry_view(file)?;
    let merged = build_segment_double_params(&params)
        .map_err(|err| RegistryError::Invalid(format!("{:?}", err)))?;
    let live_claims = active_claim_count(file, &claims);
    if live_claims < merged.need {
        return Err(RegistryError::Invalid(format!(
            "segment-double requires at least {} active claims, got {}",
            merged.need, live_claims
        )));
    }
    let merged_claims: Vec<_> = claims
        .into_iter()
        .map(|claim| {
            make_party_claim_bundle(
                &merged,
                claim.party_id,
                claim.registry_index,
                merged
                    .row_label_indices_for_registry_index(claim.registry_index as usize)
                    .into_iter()
                    .map(|idx| idx as u32)
                    .collect(),
            )
        })
        .collect();
    validate_unauth_claims(&merged, &merged_claims)?;
    file.system.unauth.layout_base_n_grid = Some(params.layout_base_n_grid);
    file.system.unauth.segment_len = merged.segment_len;
    file.system.unauth.s = merged.s;
    file.system.unauth.shard_count = merged.shard_registry.len();
    file.shard_registry = Some(encode_shard_registry(&merged.shard_registry));
    let mut epoch = default_unauth_epoch(&merged);
    if let Some(old_epoch) = &file.unauth_epoch {
        epoch.epoch_id = old_epoch.epoch_id;
        epoch.initialized_segments = old_epoch.initialized_segments;
    }
    file.unauth_epoch = Some(epoch);
    file.unauth_claims = merged_claims;
    file.assignments = assignments_from_claims(&file.parties, &file.unauth_claims);
    Ok(())
}

/// Rewrite a registry for `kappa -> 2kappa` with fixed segment size.
pub fn kappa_double_unauth_registry_file(file: &mut MockRegistryFile) -> Result<(), RegistryError> {
    if file.system.unauth.s <= 1 {
        return Err(RegistryError::Invalid(
            "cannot kappa-double below s=1".to_string(),
        ));
    }
    migrate_unauth_registry_file(file, file.system.unauth.s / 2)
}

/// Rewrite a registry for `kappa -> kappa/2` with fixed segment size.
pub fn kappa_halve_unauth_registry_file(file: &mut MockRegistryFile) -> Result<(), RegistryError> {
    let target_s = file
        .system
        .unauth
        .s
        .checked_mul(2)
        .ok_or_else(|| RegistryError::Invalid("s overflow during kappa-halve".to_string()))?;
    migrate_unauth_registry_file(file, target_s)
}

/// Rewrite a registry for a one-claim-per-active-party kappa rescale.
///
/// Kappa-double keeps the left child of each source claim and deletes the sibling
/// after the migration prepare phase. Kappa-halve requires those retained left
/// children and maps them back to their parents. The coded shard registry still
/// contains the full target mode; only the maintained party claims are pruned.
pub fn migrate_unauth_registry_file_one_claim_left(
    file: &mut MockRegistryFile,
    target_s: usize,
) -> Result<(), RegistryError> {
    let (params, claims) = build_unauth_registry_view(file)?;
    validate_one_claim_per_active_party(file, &claims)?;
    let migrated = build_migrated_params(&params, target_s)
        .map_err(|err| RegistryError::Invalid(format!("{:?}", err)))?;
    let migrated_claims: Vec<_> = if target_s < params.s {
        claims
            .into_iter()
            .map(|claim| {
                let (left, _right) =
                    kappa_double_child_registry_indices(claim.registry_index as usize);
                make_party_claim_bundle(
                    &migrated,
                    claim.party_id,
                    left as u32,
                    migrated
                        .row_label_indices_for_registry_index(left)
                        .into_iter()
                        .map(|idx| idx as u32)
                        .collect(),
                )
            })
            .collect()
    } else if target_s > params.s {
        if !registry_supports_sibling_kappa_halve(&params) {
            return Err(RegistryError::Invalid(
                "one-claim kappa-halve requires sibling kappa-halve-compatible source registry"
                    .to_string(),
            ));
        }
        let mut seen_parents = HashSet::new();
        let mut out = Vec::with_capacity(claims.len());
        for claim in claims {
            let registry_index = claim.registry_index as usize;
            if registry_index % 2 != 0 {
                return Err(RegistryError::Invalid(format!(
                    "one-claim kappa-halve requires retained left child claims, got odd registry index {}",
                    registry_index
                )));
            }
            let parent = kappa_halve_parent_registry_index(registry_index);
            if !seen_parents.insert(parent) {
                return Err(RegistryError::Invalid(format!(
                    "multiple retained children map to parent registry index {}",
                    parent
                )));
            }
            out.push(make_party_claim_bundle(
                &migrated,
                claim.party_id,
                parent as u32,
                migrated
                    .row_label_indices_for_registry_index(parent)
                    .into_iter()
                    .map(|idx| idx as u32)
                    .collect(),
            ));
        }
        out
    } else {
        claims
    };
    if migrated_claims.len() < migrated.need {
        return Err(RegistryError::Invalid(format!(
            "one-claim kappa rescale leaves {} active claims, but target need is {}",
            migrated_claims.len(),
            migrated.need
        )));
    }
    validate_unauth_claims(&migrated, &migrated_claims)?;
    file.system.unauth.s = target_s;
    file.system.unauth.shard_count = migrated.shard_registry.len();
    file.shard_registry = Some(encode_shard_registry(&migrated.shard_registry));
    file.unauth_epoch = Some(default_unauth_epoch(&migrated));
    file.unauth_claims = migrated_claims;
    file.assignments = assignments_from_claims(&file.parties, &file.unauth_claims);
    Ok(())
}

/// Rewrite a one-claim registry for `kappa -> 2kappa`.
pub fn kappa_double_unauth_registry_file_one_claim_left(
    file: &mut MockRegistryFile,
) -> Result<(), RegistryError> {
    if file.system.unauth.s <= 1 {
        return Err(RegistryError::Invalid(
            "cannot kappa-double below s=1".to_string(),
        ));
    }
    migrate_unauth_registry_file_one_claim_left(file, file.system.unauth.s / 2)
}

/// Rewrite a one-claim registry for `kappa -> kappa/2`.
pub fn kappa_halve_unauth_registry_file_one_claim_left(
    file: &mut MockRegistryFile,
) -> Result<(), RegistryError> {
    let target_s = file
        .system
        .unauth
        .s
        .checked_mul(2)
        .ok_or_else(|| RegistryError::Invalid("s overflow during kappa-halve".to_string()))?;
    migrate_unauth_registry_file_one_claim_left(file, target_s)
}

fn validate_one_claim_per_active_party(
    file: &MockRegistryFile,
    claims: &[PartyClaimBundle],
) -> Result<(), RegistryError> {
    let active: HashSet<u64> = file
        .parties
        .iter()
        .filter(|party| party.active)
        .map(|party| party.party_id)
        .collect();
    let mut counts = HashMap::<u64, usize>::new();
    for claim in claims {
        if !active.contains(&claim.party_id) {
            return Err(RegistryError::Invalid(format!(
                "inactive party {} has a one-claim kappa-rescale claim",
                claim.party_id
            )));
        }
        *counts.entry(claim.party_id).or_default() += 1;
    }
    for party_id in active {
        match counts.get(&party_id).copied().unwrap_or(0) {
            1 => {}
            count => {
                return Err(RegistryError::Invalid(format!(
                    "active party {} has {} claims; one-claim kappa rescale requires exactly one",
                    party_id, count
                )));
            }
        }
    }
    Ok(())
}

/// Add new unauth shard claims for one joining party by extending the registry.
pub fn onboard_unauth_party_with_new_shards(
    file: &mut MockRegistryFile,
    party_id: u64,
    add_count: usize,
) -> Result<Vec<PartyClaimBundle>, RegistryError> {
    if add_count == 0 {
        return Ok(Vec::new());
    }
    if !file.parties.iter().any(|party| party.party_id == party_id) {
        return Err(RegistryError::MissingParty(party_id));
    }

    let (params, claims) = build_unauth_registry_view(file)?;
    let old_count = params.shard_registry.len();
    let new_count = old_count.checked_add(add_count).ok_or_else(|| {
        RegistryError::Invalid("shard_count overflow during onboarding".to_string())
    })?;
    let new_registry = extend_shard_registry_preserving_existing(&params, new_count);
    let mut new_params = build_coded_params(&CodedArrayConfig {
        segment_len: file.system.unauth.segment_len,
        s: params.s,
        shard_count: new_count,
        shard_registry: new_registry.clone(),
        party_registry_indices: base_party_registry_indices(file, new_count),
        seed: file.system.unauth.seed,
    });
    if new_params.n_grid != params.n_grid || new_params.rows_per_shard != params.rows_per_shard {
        return Err(RegistryError::Invalid(
            "onboarding must preserve the current unauth epoch geometry".to_string(),
        ));
    }
    let old_row_label_count = old_count
        .checked_mul(params.rows_per_shard)
        .ok_or_else(|| RegistryError::Invalid("row label count overflow".to_string()))?;
    let new_row_label_count = new_count
        .checked_mul(params.rows_per_shard)
        .ok_or_else(|| RegistryError::Invalid("row label count overflow".to_string()))?;
    if params.row_labels.len() != old_row_label_count
        || new_params.row_labels.len() != new_row_label_count
    {
        return Err(RegistryError::Invalid(
            "unexpected row label layout during onboarding".to_string(),
        ));
    }
    let mut row_labels = params.row_labels.clone();
    row_labels.extend_from_slice(&new_params.row_labels[old_row_label_count..]);
    new_params.row_labels = row_labels;
    new_params.row_eval_points = params.row_eval_points.clone();
    new_params.logical_to_grid_index = params.logical_to_grid_index.clone();
    new_params.grid_index_to_logical_offset = params.grid_index_to_logical_offset.clone();
    new_params.layout_base_n_grid = params.layout_base_n_grid;
    new_params.layout_level = params.layout_level;
    new_params.refresh_row_metadata_caches();

    let mut new_claims = Vec::with_capacity(add_count);
    for registry_index in old_count..new_count {
        let row_label_indices: Vec<u32> = new_params
            .row_label_indices_for_registry_index(registry_index)
            .into_iter()
            .map(|idx| idx as u32)
            .collect();
        if row_label_indices.len() != new_params.rows_per_shard {
            return Err(RegistryError::Invalid(format!(
                "not enough unique row labels to allocate onboarding claim for registry index {}",
                registry_index
            )));
        }
        new_claims.push(make_party_claim_bundle(
            &new_params,
            party_id,
            registry_index as u32,
            row_label_indices,
        ));
    }

    file.system.unauth.shard_count = new_count;
    file.shard_registry = Some(encode_shard_registry(&new_registry));
    let mut epoch = default_unauth_epoch(&new_params);
    if let Some(old_epoch) = &file.unauth_epoch {
        epoch.epoch_id = old_epoch.epoch_id;
        epoch.initialized_segments = old_epoch.initialized_segments;
    }
    file.unauth_epoch = Some(epoch);
    let mut all_claims = claims;
    all_claims.extend(new_claims.clone());
    file.unauth_claims = all_claims;
    file.assignments = assignments_from_claims(&file.parties, &file.unauth_claims);
    validate_unauth_claims(&build_unauth_registry_view(file)?.0, &file.unauth_claims)?;
    Ok(new_claims)
}

#[derive(Clone)]
struct MockRegistryData {
    parties: Vec<PartyInfo>,
    assignments: HashMap<u64, Vec<u32>>,
    unauth_epoch: Option<NetUnauthEpoch>,
    unauth_claims: Vec<PartyClaimBundle>,
}

/// In-memory registry for tests and local demos.
#[derive(Clone)]
pub struct MockRegistryClient {
    inner: Arc<MockRegistryData>,
}

impl MockRegistryClient {
    pub fn new(parties: Vec<PartyInfo>, assignments: HashMap<u64, Vec<u32>>) -> Self {
        Self::new_with_unauth_metadata(parties, assignments, None, Vec::new())
    }

    pub fn new_with_unauth_metadata(
        parties: Vec<PartyInfo>,
        assignments: HashMap<u64, Vec<u32>>,
        unauth_epoch: Option<NetUnauthEpoch>,
        unauth_claims: Vec<PartyClaimBundle>,
    ) -> Self {
        Self {
            inner: Arc::new(MockRegistryData {
                parties,
                assignments,
                unauth_epoch,
                unauth_claims,
            }),
        }
    }

    pub fn from_file(path: &Path) -> Result<(Self, MockRegistryFile), RegistryError> {
        let file = MockRegistryFile::load(path)?;
        let mut assignments = HashMap::new();
        for entry in &file.assignments {
            assignments.insert(entry.party_id, entry.indices.clone());
        }
        Ok((
            MockRegistryClient::new_with_unauth_metadata(
                file.parties.clone(),
                assignments,
                file.unauth_epoch.clone(),
                file.unauth_claims.clone(),
            ),
            file,
        ))
    }
}

impl RegistryClient for MockRegistryClient {
    fn fetch_parties(&self) -> Result<Vec<PartyInfo>, RegistryError> {
        Ok(self.inner.parties.clone())
    }

    fn fetch_assignments(&self, party_id: u64) -> Result<Vec<u32>, RegistryError> {
        self.inner
            .assignments
            .get(&party_id)
            .cloned()
            .ok_or(RegistryError::MissingParty(party_id))
    }

    fn fetch_unauth_epoch(&self) -> Result<Option<NetUnauthEpoch>, RegistryError> {
        Ok(self.inner.unauth_epoch.clone())
    }

    fn fetch_unauth_claims(&self) -> Result<Vec<PartyClaimBundle>, RegistryError> {
        Ok(self.inner.unauth_claims.clone())
    }
}

/// Registry client that queries an EVM chain (stub unless feature enabled).
pub struct EvmRegistryClient {
    pub rpc_url: String,
    pub contract: String,
}

impl EvmRegistryClient {
    pub fn new(rpc_url: String, contract: String) -> Self {
        Self { rpc_url, contract }
    }
}

#[cfg(feature = "evm")]
impl RegistryClient for EvmRegistryClient {
    fn fetch_parties(&self) -> Result<Vec<PartyInfo>, RegistryError> {
        Err(RegistryError::NotSupported(
            "evm registry client not implemented".to_string(),
        ))
    }

    fn fetch_assignments(&self, _party_id: u64) -> Result<Vec<u32>, RegistryError> {
        Err(RegistryError::NotSupported(
            "evm registry client not implemented".to_string(),
        ))
    }
}

#[cfg(not(feature = "evm"))]
impl RegistryClient for EvmRegistryClient {
    fn fetch_parties(&self) -> Result<Vec<PartyInfo>, RegistryError> {
        Err(RegistryError::NotSupported(
            "evm feature not enabled".to_string(),
        ))
    }

    fn fetch_assignments(&self, _party_id: u64) -> Result<Vec<u32>, RegistryError> {
        Err(RegistryError::NotSupported(
            "evm feature not enabled".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MockRegistryFile, NetAuthConfig, NetSystemConfig, NetUnauthConfig, PartyAssignments,
        PartyClaimBundle, PartyInfo, build_unauth_registry_view, default_unauth_epoch,
        encode_shard_registry, migrate_unauth_registry_file,
        migrate_unauth_registry_file_one_claim_left, onboard_unauth_party_with_new_shards,
        segment_double_unauth_registry_file,
    };
    use crate::unauth::{build_migrated_params, build_segment_double_params};
    use crate::{CodedArrayConfig, build_coded_params, build_shard_registry_deterministic};

    #[test]
    fn kappa_double_registry_reassigns_exact_child_row_families() {
        let segment_len = 8usize;
        let s = 4usize;
        let shard_count = 4usize;
        let n_grid = segment_len;
        let shard_registry = build_shard_registry_deterministic(shard_count, s, n_grid);
        let params = build_coded_params(&CodedArrayConfig {
            segment_len,
            s,
            shard_count,
            shard_registry: shard_registry.clone(),
            party_registry_indices: vec![vec![0, 1], vec![2, 3]],
            seed: 23,
        });
        let migrated_params = build_migrated_params(&params, s / 2).expect("migrated params");

        let mut claims = Vec::new();
        let mut expected_rows_by_child = std::collections::HashMap::new();
        for (party_id, indices) in [(0u64, [0usize, 1usize]), (1u64, [2usize, 3usize])] {
            for registry_index in indices {
                let row_label_indices = params.row_label_indices_for_registry_index(registry_index);
                let (left, right) =
                    crate::coded_array::params::kappa_double_child_registry_indices(registry_index);
                expected_rows_by_child.insert(
                    (party_id, left as u32),
                    migrated_params
                        .row_label_indices_for_registry_index(left)
                        .into_iter()
                        .map(|idx| idx as u32)
                        .collect::<Vec<_>>(),
                );
                expected_rows_by_child.insert(
                    (party_id, right as u32),
                    migrated_params
                        .row_label_indices_for_registry_index(right)
                        .into_iter()
                        .map(|idx| idx as u32)
                        .collect::<Vec<_>>(),
                );
                claims.push(PartyClaimBundle {
                    party_id,
                    registry_index: registry_index as u32,
                    row_family_id: params
                        .row_family_id_for_registry_index(registry_index)
                        .map(|family_id| family_id.0 as u32),
                    row_label_indices: row_label_indices
                        .into_iter()
                        .map(|idx| idx as u32)
                        .collect(),
                });
            }
        }

        let mut file = MockRegistryFile {
            system: NetSystemConfig {
                unauth: NetUnauthConfig {
                    segment_len,
                    layout_base_n_grid: None,
                    s,
                    shard_count,
                    seed: 23,
                },
                auth: NetAuthConfig {
                    tree_height: 3,
                    node_linearization: crate::core::NodeLinearization::Preorder,
                    allocated_leaves: 4,
                },
                unauth_array_len: Some(16),
            },
            parties: vec![
                PartyInfo {
                    party_id: 0,
                    rpc_host: "127.0.0.1:19000".to_string(),
                    pubkey: None,
                    active: true,
                },
                PartyInfo {
                    party_id: 1,
                    rpc_host: "127.0.0.1:19001".to_string(),
                    pubkey: None,
                    active: true,
                },
            ],
            assignments: vec![
                PartyAssignments {
                    party_id: 0,
                    indices: vec![0, 1],
                },
                PartyAssignments {
                    party_id: 1,
                    indices: vec![2, 3],
                },
            ],
            unauth_epoch: Some(default_unauth_epoch(&params)),
            unauth_claims: claims,
            shard_registry: Some(encode_shard_registry(&shard_registry)),
        };

        migrate_unauth_registry_file(&mut file, s / 2).expect("migrate registry");
        let (migrated_params, migrated_claims) =
            build_unauth_registry_view(&file).expect("resolved migrated registry view");

        assert_eq!(file.system.unauth.s, s / 2);
        assert_eq!(migrated_claims.len(), shard_count * 2);
        let mut seen_rows = std::collections::HashSet::new();
        for claim in &migrated_claims {
            assert_eq!(
                claim.row_label_indices,
                *expected_rows_by_child
                    .get(&(claim.party_id, claim.registry_index))
                    .expect("expected child rows for kappa-double claim")
            );
            for &row_label_index in &claim.row_label_indices {
                assert!(seen_rows.insert(row_label_index));
                assert!((row_label_index as usize) < migrated_params.row_labels.len());
            }
        }
    }

    #[test]
    fn one_claim_registry_kappa_double_prunes_to_left_children_and_kappa_halve_returns_to_parent() {
        let segment_len = 8usize;
        let s = 4usize;
        let shard_count = 4usize;
        let n_grid = segment_len;
        let shard_registry = build_shard_registry_deterministic(shard_count, s, n_grid);
        let params = build_coded_params(&CodedArrayConfig {
            segment_len,
            s,
            shard_count,
            shard_registry: shard_registry.clone(),
            party_registry_indices: (0..shard_count).map(|idx| vec![idx]).collect(),
            seed: 23,
        });
        let claims: Vec<_> = (0..shard_count)
            .map(|registry_index| PartyClaimBundle {
                party_id: registry_index as u64,
                registry_index: registry_index as u32,
                row_family_id: params
                    .row_family_id_for_registry_index(registry_index)
                    .map(|family_id| family_id.0 as u32),
                row_label_indices: Vec::new(),
            })
            .collect();
        let parties: Vec<_> = (0..shard_count)
            .map(|party_id| PartyInfo {
                party_id: party_id as u64,
                rpc_host: format!("127.0.0.1:{}", 19100 + party_id),
                pubkey: None,
                active: true,
            })
            .collect();
        let assignments: Vec<_> = (0..shard_count)
            .map(|registry_index| PartyAssignments {
                party_id: registry_index as u64,
                indices: vec![registry_index as u32],
            })
            .collect();
        let mut file = MockRegistryFile {
            system: NetSystemConfig {
                unauth: NetUnauthConfig {
                    segment_len,
                    layout_base_n_grid: None,
                    s,
                    shard_count,
                    seed: 23,
                },
                auth: NetAuthConfig {
                    tree_height: 3,
                    node_linearization: crate::core::NodeLinearization::Preorder,
                    allocated_leaves: 4,
                },
                unauth_array_len: Some(16),
            },
            parties,
            assignments,
            unauth_epoch: Some(default_unauth_epoch(&params)),
            unauth_claims: claims,
            shard_registry: Some(encode_shard_registry(&shard_registry)),
        };

        migrate_unauth_registry_file_one_claim_left(&mut file, s / 2)
            .expect("one-claim kappa-double");
        let (_kappa_doubled_params, kappa_doubled_claims) =
            build_unauth_registry_view(&file).expect("kappa-double registry view");
        assert_eq!(file.system.unauth.s, s / 2);
        assert_eq!(file.system.unauth.shard_count, shard_count * 2);
        assert_eq!(kappa_doubled_claims.len(), shard_count);
        for (party_id, claim) in kappa_doubled_claims.iter().enumerate() {
            assert_eq!(claim.party_id, party_id as u64);
            assert_eq!(claim.registry_index, (2 * party_id) as u32);
        }

        migrate_unauth_registry_file_one_claim_left(&mut file, s).expect("one-claim kappa-halve");
        let (_kappa_halved_params, kappa_halved_claims) =
            build_unauth_registry_view(&file).expect("kappa-halve registry view");
        assert_eq!(file.system.unauth.s, s);
        assert_eq!(file.system.unauth.shard_count, shard_count);
        assert_eq!(kappa_halved_claims.len(), shard_count);
        for (party_id, claim) in kappa_halved_claims.iter().enumerate() {
            assert_eq!(claim.party_id, party_id as u64);
            assert_eq!(claim.registry_index, party_id as u32);
        }
    }

    #[test]
    fn segment_double_registry_uses_canonical_layout_base() {
        let segment_len = 8usize;
        let s = 4usize;
        let shard_count = 8usize;
        let n_grid = segment_len;
        let shard_registry = build_shard_registry_deterministic(shard_count, s, n_grid);
        let params = build_coded_params(&CodedArrayConfig {
            segment_len,
            s,
            shard_count,
            shard_registry: shard_registry.clone(),
            party_registry_indices: vec![(0..shard_count).collect()],
            seed: 23,
        });
        let mut file = MockRegistryFile {
            system: NetSystemConfig {
                unauth: NetUnauthConfig {
                    segment_len,
                    layout_base_n_grid: None,
                    s,
                    shard_count,
                    seed: 23,
                },
                auth: NetAuthConfig {
                    tree_height: 3,
                    node_linearization: crate::core::NodeLinearization::Preorder,
                    allocated_leaves: 4,
                },
                unauth_array_len: Some(32),
            },
            parties: vec![PartyInfo {
                party_id: 0,
                rpc_host: "127.0.0.1:19000".to_string(),
                pubkey: None,
                active: true,
            }],
            assignments: vec![PartyAssignments {
                party_id: 0,
                indices: (0..shard_count as u32).collect(),
            }],
            unauth_epoch: Some(default_unauth_epoch(&params)),
            unauth_claims: Vec::new(),
            shard_registry: Some(encode_shard_registry(&shard_registry)),
        };

        let expected_once =
            build_segment_double_params(&params).expect("first segment-double params");
        segment_double_unauth_registry_file(&mut file).expect("first segment-double registry");
        let (view_once, _) = build_unauth_registry_view(&file).expect("first merged view");
        assert_eq!(
            view_once.logical_to_grid_index,
            expected_once.logical_to_grid_index
        );
        assert_eq!(file.system.unauth.layout_base_n_grid, Some(params.n_grid));
        assert_eq!(
            file.unauth_epoch
                .as_ref()
                .and_then(|epoch| epoch.logical_to_grid_index.as_ref()),
            None
        );

        let expected_twice =
            build_segment_double_params(&expected_once).expect("second segment-double params");
        segment_double_unauth_registry_file(&mut file).expect("second segment-double registry");
        let (view_twice, _) = build_unauth_registry_view(&file).expect("second merged view");
        assert_eq!(
            view_twice.logical_to_grid_index,
            expected_twice.logical_to_grid_index
        );

        let mut direct_file = file.clone();
        direct_file.system.unauth.segment_len = expected_twice.segment_len;
        direct_file.system.unauth.layout_base_n_grid = Some(params.n_grid);
        direct_file.unauth_epoch = Some(default_unauth_epoch(&expected_twice));
        let (direct_view, _) =
            build_unauth_registry_view(&direct_file).expect("direct canonical layout view");
        assert_eq!(
            direct_view.logical_to_grid_index,
            expected_twice.logical_to_grid_index
        );
        assert_eq!(direct_view.layout_level, 2);
    }

    #[test]
    fn segment_double_registry_requires_doubled_live_claims() {
        let segment_len = 8usize;
        let s = 4usize;
        let shard_count = 4usize;
        let n_grid = segment_len;
        let shard_registry = build_shard_registry_deterministic(shard_count, s, n_grid);
        let params = build_coded_params(&CodedArrayConfig {
            segment_len,
            s,
            shard_count,
            shard_registry: shard_registry.clone(),
            party_registry_indices: vec![vec![0], vec![1]],
            seed: 23,
        });
        let mut file = MockRegistryFile {
            system: NetSystemConfig {
                unauth: NetUnauthConfig {
                    segment_len,
                    layout_base_n_grid: None,
                    s,
                    shard_count,
                    seed: 23,
                },
                auth: NetAuthConfig {
                    tree_height: 3,
                    node_linearization: crate::core::NodeLinearization::Preorder,
                    allocated_leaves: 4,
                },
                unauth_array_len: Some(16),
            },
            parties: vec![
                PartyInfo {
                    party_id: 0,
                    rpc_host: "127.0.0.1:19000".to_string(),
                    pubkey: None,
                    active: true,
                },
                PartyInfo {
                    party_id: 1,
                    rpc_host: "127.0.0.1:19001".to_string(),
                    pubkey: None,
                    active: true,
                },
            ],
            assignments: vec![
                PartyAssignments {
                    party_id: 0,
                    indices: vec![0],
                },
                PartyAssignments {
                    party_id: 1,
                    indices: vec![1],
                },
            ],
            unauth_epoch: Some(default_unauth_epoch(&params)),
            unauth_claims: Vec::new(),
            shard_registry: Some(encode_shard_registry(&shard_registry)),
        };

        let err = segment_double_unauth_registry_file(&mut file)
            .expect_err("not enough active claims for doubled need");
        assert!(format!("{err:?}").contains("requires at least 4 active claims"));
    }

    #[test]
    fn onboarding_assignment_only_registry_materializes_existing_claims() {
        let segment_len = 8usize;
        let s = 4usize;
        let shard_count = 2usize;
        let n_grid = segment_len;
        let shard_registry = build_shard_registry_deterministic(shard_count, s, n_grid);
        let params = build_coded_params(&CodedArrayConfig {
            segment_len,
            s,
            shard_count,
            shard_registry: shard_registry.clone(),
            party_registry_indices: vec![vec![0], vec![1], Vec::new()],
            seed: 23,
        });
        let mut file = MockRegistryFile {
            system: NetSystemConfig {
                unauth: NetUnauthConfig {
                    segment_len,
                    layout_base_n_grid: None,
                    s,
                    shard_count,
                    seed: 23,
                },
                auth: NetAuthConfig {
                    tree_height: 3,
                    node_linearization: crate::core::NodeLinearization::Preorder,
                    allocated_leaves: 4,
                },
                unauth_array_len: Some(16),
            },
            parties: vec![
                PartyInfo {
                    party_id: 0,
                    rpc_host: "127.0.0.1:19000".to_string(),
                    pubkey: None,
                    active: true,
                },
                PartyInfo {
                    party_id: 1,
                    rpc_host: "127.0.0.1:19001".to_string(),
                    pubkey: None,
                    active: true,
                },
                PartyInfo {
                    party_id: 2,
                    rpc_host: "127.0.0.1:19002".to_string(),
                    pubkey: None,
                    active: false,
                },
            ],
            assignments: vec![
                PartyAssignments {
                    party_id: 0,
                    indices: vec![0],
                },
                PartyAssignments {
                    party_id: 1,
                    indices: vec![1],
                },
                PartyAssignments {
                    party_id: 2,
                    indices: Vec::new(),
                },
            ],
            unauth_epoch: Some(default_unauth_epoch(&params)),
            unauth_claims: Vec::new(),
            shard_registry: Some(encode_shard_registry(&shard_registry)),
        };

        onboard_unauth_party_with_new_shards(&mut file, 2, 1).expect("onboard");
        let (_params, claims) = build_unauth_registry_view(&file).expect("onboarded registry view");
        assert_eq!(claims.len(), 3);
        assert!(claims.iter().any(|claim| claim.party_id == 0));
        assert!(claims.iter().any(|claim| claim.party_id == 1));
        assert!(claims.iter().any(|claim| claim.party_id == 2));
    }

    #[test]
    fn onboarding_preserves_segment_double_epoch_metadata() {
        let segment_len = 8usize;
        let s = 4usize;
        let shard_count = 8usize;
        let n_grid = segment_len;
        let shard_registry = build_shard_registry_deterministic(shard_count, s, n_grid);
        let params = build_coded_params(&CodedArrayConfig {
            segment_len,
            s,
            shard_count,
            shard_registry: shard_registry.clone(),
            party_registry_indices: vec![(0..shard_count).collect(), Vec::new()],
            seed: 23,
        });
        let mut file = MockRegistryFile {
            system: NetSystemConfig {
                unauth: NetUnauthConfig {
                    segment_len,
                    layout_base_n_grid: None,
                    s,
                    shard_count,
                    seed: 23,
                },
                auth: NetAuthConfig {
                    tree_height: 3,
                    node_linearization: crate::core::NodeLinearization::Preorder,
                    allocated_leaves: 4,
                },
                unauth_array_len: Some(32),
            },
            parties: vec![
                PartyInfo {
                    party_id: 0,
                    rpc_host: "127.0.0.1:19000".to_string(),
                    pubkey: None,
                    active: true,
                },
                PartyInfo {
                    party_id: 1,
                    rpc_host: "127.0.0.1:19001".to_string(),
                    pubkey: None,
                    active: false,
                },
            ],
            assignments: vec![
                PartyAssignments {
                    party_id: 0,
                    indices: (0..shard_count as u32).collect(),
                },
                PartyAssignments {
                    party_id: 1,
                    indices: Vec::new(),
                },
            ],
            unauth_epoch: Some(default_unauth_epoch(&params)),
            unauth_claims: Vec::new(),
            shard_registry: Some(encode_shard_registry(&shard_registry)),
        };

        segment_double_unauth_registry_file(&mut file).expect("segment-double registry");
        let (merged, merged_claims) =
            build_unauth_registry_view(&file).expect("merged registry view");
        assert_eq!(merged_claims.len(), shard_count);
        let merged_registry = merged.shard_registry.clone();
        let merged_row_labels = merged.row_labels.clone();
        let merged_logical_to_grid = merged.logical_to_grid_index.clone();

        let new_claims = onboard_unauth_party_with_new_shards(&mut file, 1, 2)
            .expect("onboard new shard claims");
        assert_eq!(new_claims.len(), 2);
        let (onboarded, onboarded_claims) =
            build_unauth_registry_view(&file).expect("onboarded registry view");

        assert_eq!(onboarded.logical_to_grid_index, merged_logical_to_grid);
        assert_eq!(
            &onboarded.row_labels[..merged_row_labels.len()],
            merged_row_labels.as_slice()
        );
        assert_eq!(
            &onboarded.shard_registry[..merged_registry.len()],
            merged_registry.as_slice()
        );
        assert_eq!(onboarded_claims.len(), shard_count + 2);
        assert_eq!(onboarded_claims[shard_count].party_id, 1);
        assert_eq!(
            onboarded_claims[shard_count].registry_index,
            shard_count as u32
        );
        assert_eq!(onboarded_claims[shard_count + 1].party_id, 1);
        assert_eq!(
            onboarded_claims[shard_count + 1].registry_index,
            shard_count as u32 + 1
        );
    }
}
