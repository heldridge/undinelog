use crate::registry::parse_hex_32;
use paper_a_optimized::auth::AuthParams;
use paper_a_optimized::auth::service::AuthServer;
use paper_a_optimized::net;
use paper_a_optimized::net::registry::{
    MockRegistryFile, build_party_registry_indices, build_unauth_registry_view,
    claim_bundles_for_party, resolve_shard_registry,
};
use paper_a_optimized::unauth::{CodedArrayConfig, build_coded_params};
use serde::Deserialize;
use std::path::Path;
use std::sync::{Arc, RwLock};

#[derive(Debug, Deserialize)]
struct RecoveredShardState {
    registry_index: usize,
    values_hex: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RecoveredRowState {
    row_label_index: usize,
    values_hex: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RecoveredSegmentState {
    segment: usize,
    shards: Vec<RecoveredShardState>,
    rows: Vec<RecoveredRowState>,
}

#[derive(Debug, Deserialize)]
struct RecoveredPartyStateFile {
    party_id: u64,
    segments: Vec<RecoveredSegmentState>,
}

pub fn load_registry_file(path: &Path) -> Result<MockRegistryFile, String> {
    MockRegistryFile::load(path).map_err(|err| format!("failed to load registry: {err:?}"))
}

pub fn serve_auth_from_registry_file(
    path: &Path,
    listen_override: Option<&str>,
    party_id: Option<u64>,
    recovered_state_path: Option<&Path>,
) -> Result<(), String> {
    let file = load_registry_file(path)?;
    serve_auth_from_registry(&file, listen_override, party_id, recovered_state_path)
}

pub fn serve_auth_from_registry(
    file: &MockRegistryFile,
    listen_override: Option<&str>,
    party_id: Option<u64>,
    recovered_state_path: Option<&Path>,
) -> Result<(), String> {
    let (listen, effective_party_id) = resolve_listen(file, listen_override, party_id)?;
    let mut server = build_auth_server(file, effective_party_id)?;
    if let Some(path) = recovered_state_path {
        apply_recovered_party_state(&mut server, effective_party_id, path)?;
    }
    let server = Arc::new(RwLock::new(server));
    net::serve(&listen, move |msg| {
        if net::auth::is_read_only_message(&msg) {
            let server = server.read().expect("server lock");
            net::auth::handle_read_only_message(&server, msg).expect("read-only auth message")
        } else if net::unauth::is_auth_mode_read_only_message(&msg) {
            let server = server.read().expect("server lock");
            net::unauth::handle_auth_mode_read_only_message(&server, msg)
                .expect("read-only auth-mode unauth message")
        } else {
            let mut server = server.write().expect("server lock");
            match msg {
                net::Message::AuthRead { .. }
                | net::Message::AuthReadSubtree { .. }
                | net::Message::AuthReadSubtreeUpperPath { .. }
                | net::Message::AuthWrite { .. }
                | net::Message::AuthWriteBatch { .. }
                | net::Message::AuthWriteFreshSubtree { .. }
                | net::Message::AuthExtend { .. }
                | net::Message::AuthGetLen
                | net::Message::UnauthMultiRead { .. } => {
                    net::auth::handle_message(&mut *server, msg)
                }
                net::Message::UnauthRead { .. }
                | net::Message::UnauthReadBlock { .. }
                | net::Message::UnauthReadSegmentRaw { .. }
                | net::Message::UnauthReadRowsAtZ { .. }
                | net::Message::UnauthReadRowsBatchAtZ { .. }
                | net::Message::UnauthReadRowsMatrix { .. }
                | net::Message::UnauthOptimizedRead { .. }
                | net::Message::UnauthRepairShard { .. }
                | net::Message::UnauthRepairShardBatch { .. }
                | net::Message::UnauthEvalShardsAtRowLabels { .. }
                | net::Message::UnauthEvalShardsAtRowLabelsMatrix { .. }
                | net::Message::UnauthApplyDeltas { .. }
                | net::Message::UnauthApplyBlockDeltas { .. }
                | net::Message::UnauthExtend { .. }
                | net::Message::UnauthMigrate { .. }
                | net::Message::UnauthSegmentDouble { .. }
                | net::Message::UnauthKappaDoubleOneClaimLeft { .. }
                | net::Message::UnauthEvalKappaDoubleTargetAtRowLabelsMatrix { .. }
                | net::Message::UnauthRetainClaims { .. }
                | net::Message::UnauthGetLen => net::unauth::handle_message(&mut *server, msg),
            }
        }
    })
    .map_err(|err| err.to_string())
}

fn resolve_listen(
    file: &MockRegistryFile,
    listen_override: Option<&str>,
    party_id: Option<u64>,
) -> Result<(String, Option<u64>), String> {
    let listen = match listen_override {
        Some(listen) => listen.to_string(),
        None => {
            if let Some(party_id) = party_id {
                file.parties
                    .iter()
                    .find(|party| party.party_id == party_id)
                    .map(|party| party.rpc_host.clone())
                    .ok_or_else(|| format!("party {party_id} not found in registry"))?
            } else {
                file.parties
                    .iter()
                    .find(|party| party.active)
                    .map(|party| party.rpc_host.clone())
                    .ok_or_else(|| "no active servers in registry".to_string())?
            }
        }
    };

    let effective_party_id = party_id.or_else(|| {
        file.parties
            .iter()
            .find(|party| party.rpc_host == listen)
            .map(|party| party.party_id)
    });
    Ok((listen, effective_party_id))
}

fn build_auth_server(
    file: &MockRegistryFile,
    effective_party_id: Option<u64>,
) -> Result<AuthServer, String> {
    let shard_registry = resolve_shard_registry(file);
    let party_registry_indices = if file.assignments.is_empty() || file.parties.is_empty() {
        vec![(0..shard_registry.len()).collect()]
    } else {
        build_party_registry_indices(&file.parties, &file.assignments)
    };
    let cfg = CodedArrayConfig {
        segment_len: file.system.unauth.segment_len,
        s: file.system.unauth.s,
        shard_count: file.system.unauth.shard_count,
        shard_registry,
        party_registry_indices,
        seed: file.system.unauth.seed,
    };
    let params = build_coded_params(&cfg);
    let auth = AuthParams::new(file.system.auth.tree_height);
    let allocated = file.system.auth.allocated_leaves;
    let assigned_indices = effective_party_id.and_then(|party_id| {
        file.assignments
            .iter()
            .find(|assignment| assignment.party_id == party_id)
            .map(|assignment| {
                let mut indices: Vec<usize> =
                    assignment.indices.iter().map(|idx| *idx as usize).collect();
                indices.sort_unstable();
                indices.dedup();
                indices
            })
    });
    let claims = if let Some(party_id) = effective_party_id {
        let (claim_params, all_claims) = build_unauth_registry_view(file)
            .map_err(|err| format!("failed to build unauth registry view: {err:?}"))?;
        if all_claims.is_empty() {
            None
        } else {
            Some(
                claim_bundles_for_party(&claim_params, &all_claims, party_id).map_err(|err| {
                    format!("failed to resolve claim bundles for party {party_id}: {err:?}")
                })?,
            )
        }
    } else {
        None
    };

    AuthServer::new_with_assignments_and_claims(
        params,
        auth,
        file.system.auth.node_linearization,
        allocated,
        assigned_indices,
        claims,
    )
    .map_err(|err| format!("auth server init failed: {err:?}"))
}

fn apply_recovered_party_state(
    server: &mut AuthServer,
    effective_party_id: Option<u64>,
    path: &Path,
) -> Result<(), String> {
    let expected_party_id = effective_party_id
        .ok_or_else(|| "recovered state requires an explicit party id".to_string())?;
    let payload: RecoveredPartyStateFile = serde_json::from_str(
        &std::fs::read_to_string(path)
            .map_err(|err| format!("failed to read recovered state file: {err}"))?,
    )
    .map_err(|err| format!("failed to parse recovered state file: {err}"))?;
    if payload.party_id != expected_party_id {
        return Err(format!(
            "recovered state file is for party {}, but server is party {}",
            payload.party_id, expected_party_id
        ));
    }
    for segment in payload.segments {
        let shards = segment
            .shards
            .into_iter()
            .map(|shard| {
                let values = shard
                    .values_hex
                    .into_iter()
                    .map(|value| parse_hex_32(&value).map_err(|err| err.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((shard.registry_index, values))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let rows = segment
            .rows
            .into_iter()
            .map(|row| {
                let values = row
                    .values_hex
                    .into_iter()
                    .map(|value| parse_hex_32(&value).map_err(|err| err.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((row.row_label_index, values))
            })
            .collect::<Result<Vec<_>, String>>()?;
        server
            .store
            .unauth
            .install_claim_segment_state(segment.segment, &shards, &rows)
            .map_err(|err| {
                format!(
                    "failed to install recovered segment {}: {:?}",
                    segment.segment, err
                )
            })?;
    }
    Ok(())
}
