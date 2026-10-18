//! Shared system configuration helpers for networking demos.

use crate::auth::allocation::{
    default_family_key_for_range, default_value_at_index, default_values_for_family_key,
};
use crate::auth::mapping::{self, NodeLinearization};
use crate::auth::merkle::default_hash;
use crate::auth::{AuthParams, AuthStore, Config};
use crate::field::Fp;
use crate::merkle::build_default_hashes;
use crate::net::registry::{NetShardRegEntry, NetSystemConfig};
use crate::unauth::{
    ClaimBundle, CodedArrayConfig, CodedArrayParams, ShardKey, ShardRegEntry, UnauthArrayParams,
    UnauthStore, build_coded_params, build_shard_registry_deterministic, compute_initial_shards,
};
use crate::value::Value256;

/// Derived params used by the network client/server.
#[derive(Clone)]
pub struct NetSystem {
    pub auth: AuthParams,
    pub params: CodedArrayParams,
    pub node_linearization: NodeLinearization,
    pub allocated_leaves: usize,
}

/// Errors while building network system state.
#[derive(Clone, Debug)]
pub enum NetSystemError {
    InvalidRegistryIndex(usize),
    AllocationFailed(String),
}

/// Party-local shard storage plus allocation metadata.
#[derive(Clone)]
pub struct PartyStorage {
    pub store: AuthStore,
    pub segments: usize,
    pub allocated_leaves: usize,
}

impl NetSystem {
    /// Build network params from an authenticated config.
    pub fn from_config(cfg: Config) -> Self {
        let params = build_coded_params(&cfg.coded);
        let defaults = build_default_hashes(cfg.auth.tree_height);
        let allocated_leaves = cfg.auth.alloc_leaf_batch;
        NetSystem {
            auth: AuthParams {
                tree_height: cfg.auth.tree_height,
                defaults,
            },
            params,
            node_linearization: cfg.auth.node_linearization,
            allocated_leaves,
        }
    }

    /// Build network params from a JSON-friendly config and party assignments.
    pub fn from_parts(
        cfg: &NetSystemConfig,
        shard_registry: Option<Vec<ShardRegEntry>>,
        party_registry_indices: Vec<Vec<usize>>,
    ) -> Self {
        let shard_registry = shard_registry
            .unwrap_or_else(|| build_shard_registry_deterministic(cfg.unauth.shard_count));
        let coded_cfg = CodedArrayConfig {
            kappa: cfg.unauth.kappa,
            shard_count: cfg.unauth.shard_count,
            shard_registry,
            party_registry_indices,
            seed: cfg.unauth.seed,
        };
        let mut params = build_coded_params(&coded_cfg);
        if let Some(layout_base_kappa) = cfg.unauth.layout_base_kappa {
            params
                .apply_canonical_logical_layout(layout_base_kappa * layout_base_kappa)
                .expect("valid canonical unauth layout");
        }
        let defaults = build_default_hashes(cfg.auth.tree_height);
        let allocated_leaves = cfg.auth.allocated_leaves;
        NetSystem {
            auth: AuthParams {
                tree_height: cfg.auth.tree_height,
                defaults,
            },
            params,
            node_linearization: cfg.auth.node_linearization,
            allocated_leaves,
        }
    }
}

/// Convert JSON-friendly shard registry entries into core registry entries.
pub fn convert_shard_registry(entries: &[NetShardRegEntry]) -> Vec<ShardRegEntry> {
    entries
        .iter()
        .map(|entry| {
            let a = entry.a.0;
            let z = entry.z.0;
            let a_bytes = crate::value::encode_value(a);
            let z_bytes = crate::value::encode_value(z);
            ShardRegEntry {
                a,
                z,
                key: ShardKey { a_bytes, z_bytes },
            }
        })
        .collect()
}

/// Build party-local shard storage with default segment values.
pub fn build_party_storage(
    system: &NetSystem,
    assigned_indices: &[usize],
    claims: Option<&[ClaimBundle]>,
) -> Result<PartyStorage, NetSystemError> {
    let registry_len = system.params.shard_registry.len();
    for &idx in assigned_indices {
        if idx >= registry_len {
            return Err(NetSystemError::InvalidRegistryIndex(idx));
        }
    }
    let mut store = match claims {
        Some(local_claims) => {
            UnauthStore::new_with_claims(system.params.clone(), local_claims.to_vec())
        }
        None => UnauthStore::new(system.params.clone()),
    };
    append_segments(
        system,
        &mut store,
        system.allocated_leaves,
        assigned_indices,
        claims,
    )?;
    let root = default_hash(&system.auth.defaults, 0);
    let segments = store.segments();
    Ok(PartyStorage {
        store: AuthStore {
            root,
            unauth: store,
        },
        segments,
        allocated_leaves: system.allocated_leaves,
    })
}

/// Extend party storage up to the segments needed for `allocated_leaves`.
pub fn append_segments(
    system: &NetSystem,
    store: &mut UnauthStore,
    allocated_leaves: usize,
    assigned_indices: &[usize],
    claims: Option<&[ClaimBundle]>,
) -> Result<(), NetSystemError> {
    let height = system.auth.tree_height as u16;
    let cursor = mapping::cursor(system.node_linearization, height, allocated_leaves as u64);
    match claims {
        Some(local_claims) => store
            .extend_with_value_families_for_claims_timed(
                cursor,
                local_claims,
                |range| {
                    default_family_key_for_range(&system.auth, system.node_linearization, range)
                },
                |_range, family_key| default_values_for_family_key(&system.auth, family_key),
                None,
            )
            .map_err(|err| NetSystemError::AllocationFailed(format!("{:?}", err)))?,
        None => store
            .extend_with_value_families_filtered_timed(
                cursor,
                assigned_indices,
                |range| {
                    default_family_key_for_range(&system.auth, system.node_linearization, range)
                },
                |_range, family_key| default_values_for_family_key(&system.auth, family_key),
                None,
            )
            .map_err(|err| NetSystemError::AllocationFailed(format!("{:?}", err)))?,
    }
    Ok(())
}

pub(crate) fn default_segment_values(system: &NetSystem, seg: usize) -> Vec<Value256> {
    let seg_base = seg as u64 * system.params.public_values_per_segment() as u64;
    system.params.materialize_logical_segment(|logical_offset| {
        default_value_at_index(
            &system.auth,
            system.node_linearization,
            seg_base + logical_offset as u64,
        )
    })
}

pub(crate) fn default_segment_shards(system: &NetSystem, seg: usize) -> Vec<Vec<Fp>> {
    let t_seg = default_segment_values(system, seg);
    compute_initial_shards(&system.params, seg, &t_seg, None)
}
