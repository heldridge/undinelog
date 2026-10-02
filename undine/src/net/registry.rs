//! Registry abstraction for party discovery and shard assignments.

use crate::coded_array::accept_candidate;
use crate::coded_array::axis::is_valid_assignment;
use crate::coded_array::params::build_registry_row_labels;
use crate::core::NodeLinearization;
use crate::field::Fp;
use crate::net::wire::NetFp;
use crate::unauth::{
    ClaimBundle, CodedArrayConfig, CodedArrayParams, ShardKey, ShardRegEntry, build_coded_params,
    build_kappa_double_params, build_kappa_halve_params, build_shard_registry_deterministic,
};
use crate::value::encode_value;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::Arc;

/// Network-facing unauthenticated (coded array) config.
///
/// The advertisement carries only the threshold. The segment size is derived
/// from it, `Seg = kappa^2`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetUnauthConfig {
    /// Reconstruction threshold.
    pub kappa: usize,
    /// Threshold at the base level of the layout ladder. Defaults to `kappa`,
    /// and stays fixed while `kappa` is doubled and halved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout_base_kappa: Option<usize>,
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

/// Row-label metadata for one unauthenticated epoch.
///
/// `row_labels[i]` is the auxiliary row label of the claim at registry index
/// `i`. A claim keeps its label when `kappa` is doubled or halved.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetUnauthEpoch {
    #[serde(default)]
    pub epoch_id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initialized_segments: Option<u64>,
    pub row_labels: Vec<NetFp>,
}

/// One registry-published claim: the party that maintains the primary column
/// and the auxiliary row at one registry index.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartyClaimBundle {
    pub party_id: u64,
    pub registry_index: u32,
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
        None => build_shard_registry_deterministic(file.system.unauth.shard_count),
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
        if let Some(entry) = accept_candidate(Fp::from(cand), &seen_z) {
            seen_z.insert(entry.key.z_bytes);
            registry.push(entry);
        }
        cand += 1;
    }
    registry
}

/// Extend the row labels to `new_count` entries, keeping the existing ones.
fn extend_row_labels_preserving_existing(params: &CodedArrayParams, new_count: usize) -> Vec<Fp> {
    let mut row_labels = params.row_labels.clone();
    let mut seen: HashSet<_> = params.row_label_bytes.iter().copied().collect();
    // The deterministic scan is long enough to supply `new_count` labels even
    // if every existing label came from the same scan.
    for label in build_registry_row_labels(new_count + params.row_labels.len()) {
        if row_labels.len() >= new_count {
            break;
        }
        if seen.insert(encode_value(label)) {
            row_labels.push(label);
        }
    }
    row_labels
}

/// Build the default epoch metadata implied by the current coded params.
pub fn default_unauth_epoch(params: &CodedArrayParams) -> NetUnauthEpoch {
    NetUnauthEpoch {
        epoch_id: 0,
        initialized_segments: None,
        row_labels: params.row_labels.iter().copied().map(NetFp).collect(),
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
    if row_labels.len() != params.shard_registry.len() {
        return Err(RegistryError::Invalid(format!(
            "expected {} row labels, one per registry entry, got {}",
            params.shard_registry.len(),
            row_labels.len()
        )));
    }
    let mut seen_labels = HashSet::new();
    for &row_label in &row_labels {
        if !is_valid_assignment(row_label) {
            return Err(RegistryError::Invalid(
                "row label in unauth_epoch lies on a semantic axis".to_string(),
            ));
        }
        if !seen_labels.insert(encode_value(row_label)) {
            return Err(RegistryError::Invalid(
                "duplicate row label in unauth_epoch".to_string(),
            ));
        }
    }
    params.row_labels = row_labels;
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
            if registry_index as usize >= params.shard_registry.len() {
                return Err(RegistryError::Invalid(format!(
                    "assignment references out-of-range registry index {}",
                    registry_index
                )));
            }
            claims.push(PartyClaimBundle {
                party_id: assignment.party_id,
                registry_index,
            });
        }
    }
    Ok(claims)
}

/// Validate explicit unauth claim bundles against the current coded params.
///
/// Every claim names a registry entry, and no registry entry has two owners.
/// Column identifiers and row labels are distinct across the registry, so
/// distinct registry indices give distinct assignments on both axes.
pub fn validate_unauth_claims(
    params: &CodedArrayParams,
    claims: &[PartyClaimBundle],
) -> Result<(), RegistryError> {
    let mut owners = HashMap::new();
    for claim in claims {
        let registry_index = claim.registry_index as usize;
        if registry_index >= params.shard_registry.len() {
            return Err(RegistryError::Invalid(format!(
                "claim references out-of-range registry index {}",
                registry_index
            )));
        }
        if let Some(existing_party) = owners.insert(registry_index, claim.party_id) {
            return Err(RegistryError::Invalid(format!(
                "registry index {} claimed by both party {} and party {}",
                registry_index, existing_party, claim.party_id
            )));
        }
    }
    Ok(())
}

/// Build coded params plus explicit unauth claims from a registry file.
pub fn build_unauth_registry_view(
    file: &MockRegistryFile,
) -> Result<(CodedArrayParams, Vec<PartyClaimBundle>), RegistryError> {
    let kappa = file.system.unauth.kappa;
    if crate::unauth::derive_paper_geometry(kappa).is_err() {
        return Err(RegistryError::Invalid(format!(
            "unsupported threshold kappa = {}",
            kappa
        )));
    }
    let shard_registry = resolve_shard_registry(file);
    let cfg = CodedArrayConfig {
        kappa,
        shard_count: file.system.unauth.shard_count,
        shard_registry,
        party_registry_indices: base_party_registry_indices(file, file.system.unauth.shard_count),
        seed: file.system.unauth.seed,
    };
    let mut params = build_coded_params(&cfg);
    let layout_base_kappa = file.system.unauth.layout_base_kappa.unwrap_or(kappa);
    let layout_base_n_grid = layout_base_kappa
        .checked_mul(layout_base_kappa)
        .ok_or_else(|| RegistryError::Invalid("layout_base_kappa overflow".to_string()))?;
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
        file.unauth_claims.clone()
    };
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
    validate_unauth_claims(params, claims)?;
    Ok(claims
        .iter()
        .filter(|claim| claim.party_id == party_id)
        .map(|claim| ClaimBundle {
            registry_index: claim.registry_index as usize,
        })
        .collect())
}

/// Rewrite a registry file for another threshold on the same layout ladder.
///
/// Claims, column identifiers, and row labels are unchanged: only the
/// advertised threshold moves.
fn rescale_unauth_registry_file(
    file: &mut MockRegistryFile,
    params: &CodedArrayParams,
    claims: Vec<PartyClaimBundle>,
    rescaled: &CodedArrayParams,
) -> Result<(), RegistryError> {
    let live_claims = active_claim_count(file, &claims);
    if live_claims < rescaled.need {
        return Err(RegistryError::Invalid(format!(
            "threshold {} requires at least {} active claims, got {}",
            rescaled.need, rescaled.need, live_claims
        )));
    }
    let layout_base_kappa = file.system.unauth.layout_base_kappa.unwrap_or(params.need);
    file.system.unauth.layout_base_kappa = Some(layout_base_kappa);
    file.system.unauth.kappa = rescaled.need;
    file.shard_registry = Some(encode_shard_registry(&rescaled.shard_registry));
    let mut epoch = default_unauth_epoch(params);
    if let Some(old_epoch) = &file.unauth_epoch {
        epoch.epoch_id = old_epoch.epoch_id;
        epoch.initialized_segments = old_epoch.initialized_segments;
    }
    file.unauth_epoch = Some(epoch);
    file.unauth_claims = claims;
    file.assignments = assignments_from_claims(&file.parties, &file.unauth_claims);
    Ok(())
}

/// Rewrite a registry for `kappa -> 2 kappa`: the four-segment stitch.
///
/// The segment size goes from `kappa^2` to `4 kappa^2` with it. The layout
/// base is recorded so every registry view derives the same canonical layout.
pub fn kappa_double_unauth_registry_file(file: &mut MockRegistryFile) -> Result<(), RegistryError> {
    let (params, claims) = build_unauth_registry_view(file)?;
    let doubled = build_kappa_double_params(&params)
        .map_err(|err| RegistryError::Invalid(format!("{:?}", err)))?;
    rescale_unauth_registry_file(file, &params, claims, &doubled)
}

/// Rewrite a registry for `kappa -> kappa / 2`: the undo of the stitch.
pub fn kappa_halve_unauth_registry_file(file: &mut MockRegistryFile) -> Result<(), RegistryError> {
    let (params, claims) = build_unauth_registry_view(file)?;
    let halved = build_kappa_halve_params(&params)
        .map_err(|err| RegistryError::Invalid(format!("{:?}", err)))?;
    rescale_unauth_registry_file(file, &params, claims, &halved)
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
    let new_row_labels = extend_row_labels_preserving_existing(&params, new_count);
    if new_row_labels.len() != new_count {
        return Err(RegistryError::Invalid(
            "not enough unique row labels to allocate onboarding claims".to_string(),
        ));
    }

    let new_claims = (old_count..new_count)
        .map(|registry_index| PartyClaimBundle {
            party_id,
            registry_index: registry_index as u32,
        })
        .collect::<Vec<_>>();

    file.system.unauth.shard_count = new_count;
    file.shard_registry = Some(encode_shard_registry(&new_registry));
    let mut epoch = NetUnauthEpoch {
        epoch_id: 0,
        initialized_segments: None,
        row_labels: new_row_labels.into_iter().map(NetFp).collect(),
    };
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
        PartyInfo, build_unauth_registry_view, claim_bundles_for_party, default_unauth_epoch,
        encode_shard_registry, kappa_double_unauth_registry_file, kappa_halve_unauth_registry_file,
        onboard_unauth_party_with_new_shards,
    };
    use crate::unauth::build_kappa_double_params;
    use crate::{CodedArrayConfig, build_coded_params, build_shard_registry_deterministic};

    fn party(party_id: u64, active: bool) -> PartyInfo {
        PartyInfo {
            party_id,
            rpc_host: format!("127.0.0.1:{}", 19000 + party_id),
            pubkey: None,
            active,
        }
    }

    /// Registry file at threshold `kappa` with the given per-party claims.
    fn registry_file(
        kappa: usize,
        shard_count: usize,
        parties: Vec<PartyInfo>,
        assignments: Vec<Vec<u32>>,
    ) -> MockRegistryFile {
        let shard_registry = build_shard_registry_deterministic(shard_count);
        let params = build_coded_params(&CodedArrayConfig {
            kappa,
            shard_count,
            shard_registry: shard_registry.clone(),
            party_registry_indices: vec![(0..shard_count).collect()],
            seed: 23,
        });
        MockRegistryFile {
            system: NetSystemConfig {
                unauth: NetUnauthConfig {
                    kappa,
                    layout_base_kappa: None,
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
            assignments: parties
                .iter()
                .zip(assignments)
                .map(|(party, indices)| PartyAssignments {
                    party_id: party.party_id,
                    indices,
                })
                .collect(),
            parties,
            unauth_epoch: Some(default_unauth_epoch(&params)),
            unauth_claims: Vec::new(),
            shard_registry: Some(encode_shard_registry(&shard_registry)),
        }
    }

    #[test]
    fn kappa_double_registry_keeps_every_claim_and_derives_the_segment_size() {
        let shard_count = 8usize;
        let mut file = registry_file(
            2,
            shard_count,
            vec![party(0, true), party(1, true)],
            vec![(0..4).collect(), (4..8).collect()],
        );
        let (params, claims) = build_unauth_registry_view(&file).expect("source view");
        assert_eq!(params.need, 2);
        assert_eq!(params.n_grid, 4);

        kappa_double_unauth_registry_file(&mut file).expect("kappa-double registry");
        let (doubled, doubled_claims) = build_unauth_registry_view(&file).expect("doubled view");
        assert_eq!(file.system.unauth.kappa, 4);
        assert_eq!(file.system.unauth.layout_base_kappa, Some(2));
        assert_eq!(doubled.need, 4);
        assert_eq!(doubled.n_grid, 16);
        assert_eq!(doubled.layout_level, 1);
        assert_eq!(doubled_claims, claims);
        assert_eq!(doubled.shard_registry, params.shard_registry);
        assert_eq!(doubled.row_labels, params.row_labels);
        assert_eq!(
            doubled.party_registry_indices,
            params.party_registry_indices
        );
        for party_id in 0..2u64 {
            assert_eq!(
                claim_bundles_for_party(&doubled, &doubled_claims, party_id).expect("claims"),
                claim_bundles_for_party(&params, &claims, party_id).expect("claims"),
            );
        }
    }

    #[test]
    fn kappa_double_registry_uses_canonical_layout_base() {
        let shard_count = 16usize;
        let mut file = registry_file(
            2,
            shard_count,
            vec![party(0, true)],
            vec![(0..shard_count as u32).collect()],
        );
        let (params, _) = build_unauth_registry_view(&file).expect("source view");

        let expected_once = build_kappa_double_params(&params).expect("first double");
        kappa_double_unauth_registry_file(&mut file).expect("first kappa-double registry");
        let (view_once, _) = build_unauth_registry_view(&file).expect("first doubled view");
        assert_eq!(
            view_once.logical_to_grid_index,
            expected_once.logical_to_grid_index
        );

        let expected_twice = build_kappa_double_params(&expected_once).expect("second double");
        kappa_double_unauth_registry_file(&mut file).expect("second kappa-double registry");
        let (view_twice, _) = build_unauth_registry_view(&file).expect("second doubled view");
        assert_eq!(
            view_twice.logical_to_grid_index,
            expected_twice.logical_to_grid_index
        );
        assert_eq!(view_twice.layout_level, 2);
        assert_eq!(file.system.unauth.layout_base_kappa, Some(2));

        // A registry written directly at the doubled threshold with the same
        // base derives the same layout.
        let mut direct_file = registry_file(
            8,
            shard_count,
            vec![party(0, true)],
            vec![(0..shard_count as u32).collect()],
        );
        direct_file.system.unauth.layout_base_kappa = Some(2);
        let (direct_view, _) = build_unauth_registry_view(&direct_file).expect("direct view");
        assert_eq!(
            direct_view.logical_to_grid_index,
            expected_twice.logical_to_grid_index
        );
        assert_eq!(direct_view.layout_level, 2);
    }

    #[test]
    fn kappa_halve_registry_returns_to_the_source_threshold() {
        let shard_count = 8usize;
        let mut file = registry_file(
            2,
            shard_count,
            vec![party(0, true)],
            vec![(0..shard_count as u32).collect()],
        );
        let (params, claims) = build_unauth_registry_view(&file).expect("source view");

        let err = kappa_halve_unauth_registry_file(&mut file)
            .expect_err("the base level has no source segments");
        assert!(format!("{err:?}").contains("AtBaseLevel"));

        kappa_double_unauth_registry_file(&mut file).expect("kappa-double registry");
        kappa_halve_unauth_registry_file(&mut file).expect("kappa-halve registry");
        let (halved, halved_claims) = build_unauth_registry_view(&file).expect("halved view");
        assert_eq!(file.system.unauth.kappa, 2);
        assert_eq!(halved.need, params.need);
        assert_eq!(halved.n_grid, params.n_grid);
        assert_eq!(halved.layout_level, 0);
        assert_eq!(halved.logical_to_grid_index, params.logical_to_grid_index);
        assert_eq!(halved_claims, claims);
    }

    #[test]
    fn kappa_double_registry_requires_doubled_live_claims() {
        let mut file = registry_file(
            2,
            4,
            vec![party(0, true), party(1, true)],
            vec![vec![0], vec![1]],
        );
        let err = kappa_double_unauth_registry_file(&mut file)
            .expect_err("not enough active claims for doubled need");
        assert!(format!("{err:?}").contains("requires at least 4 active claims"));
    }

    #[test]
    fn onboarding_assignment_only_registry_materializes_existing_claims() {
        let mut file = registry_file(
            2,
            2,
            vec![party(0, true), party(1, true), party(2, false)],
            vec![vec![0], vec![1], Vec::new()],
        );

        onboard_unauth_party_with_new_shards(&mut file, 2, 1).expect("onboard");
        let (_params, claims) = build_unauth_registry_view(&file).expect("onboarded registry view");
        assert_eq!(claims.len(), 3);
        assert!(claims.iter().any(|claim| claim.party_id == 0));
        assert!(claims.iter().any(|claim| claim.party_id == 1));
        assert!(claims.iter().any(|claim| claim.party_id == 2));
    }

    #[test]
    fn onboarding_preserves_kappa_double_epoch_metadata() {
        let shard_count = 8usize;
        let mut file = registry_file(
            2,
            shard_count,
            vec![party(0, true), party(1, false)],
            vec![(0..shard_count as u32).collect(), Vec::new()],
        );

        kappa_double_unauth_registry_file(&mut file).expect("kappa-double registry");
        let (doubled, doubled_claims) =
            build_unauth_registry_view(&file).expect("doubled registry view");
        assert_eq!(doubled_claims.len(), shard_count);
        let doubled_registry = doubled.shard_registry.clone();
        let doubled_row_labels = doubled.row_labels.clone();
        let doubled_logical_to_grid = doubled.logical_to_grid_index.clone();

        let new_claims = onboard_unauth_party_with_new_shards(&mut file, 1, 2)
            .expect("onboard new shard claims");
        assert_eq!(new_claims.len(), 2);
        let (onboarded, onboarded_claims) =
            build_unauth_registry_view(&file).expect("onboarded registry view");

        assert_eq!(onboarded.need, doubled.need);
        assert_eq!(onboarded.logical_to_grid_index, doubled_logical_to_grid);
        assert_eq!(
            &onboarded.row_labels[..doubled_row_labels.len()],
            doubled_row_labels.as_slice()
        );
        assert_eq!(onboarded.row_labels.len(), shard_count + 2);
        assert_eq!(
            &onboarded.shard_registry[..doubled_registry.len()],
            doubled_registry.as_slice()
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
