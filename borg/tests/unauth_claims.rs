use paper_a_optimized::coded_array::params::CodedArrayConfig;
use paper_a_optimized::net::registry::{
    MockRegistryFile, NetAuthConfig, NetShardRegEntry, NetSystemConfig, NetUnauthConfig,
    NetUnauthEpoch, PartyAssignments, PartyClaimBundle, PartyInfo, build_unauth_registry_view,
    claim_bundles_for_party, migrate_unauth_registry_file, onboard_unauth_party_with_new_shards,
};
use paper_a_optimized::net::wire::NetFp;
use paper_a_optimized::unauth::{ClaimBundle, RowKey, UnauthStore};
use paper_a_optimized::{Fp, build_coded_params, build_shard_registry_deterministic, encode_value};

fn build_params() -> paper_a_optimized::CodedArrayParams {
    let segment_len = 8usize;
    let n_grid = segment_len;
    let s = 4usize;
    let shard_count = 4usize;
    let shard_registry = build_shard_registry_deterministic(shard_count, s, n_grid);
    let cfg = CodedArrayConfig {
        segment_len,
        s,
        shard_count,
        shard_registry,
        party_registry_indices: vec![(0..shard_count).collect()],
        seed: 42,
    };
    build_coded_params(&cfg)
}

fn encode_registry(entries: &[paper_a_optimized::ShardRegEntry]) -> Vec<NetShardRegEntry> {
    entries
        .iter()
        .map(|entry| NetShardRegEntry {
            a: NetFp(entry.a),
            z: NetFp(entry.z),
        })
        .collect()
}

fn epoch_from_params(params: &paper_a_optimized::CodedArrayParams) -> NetUnauthEpoch {
    NetUnauthEpoch {
        epoch_id: 7,
        initialized_segments: Some(0),
        rows_per_shard: Some(params.rows_per_shard as u32),
        logical_to_grid_index: None,
        row_labels: params.row_labels.iter().copied().map(NetFp).collect(),
        row_eval_points: params.row_eval_points.iter().copied().map(NetFp).collect(),
    }
}

fn exact_claim(params: &paper_a_optimized::CodedArrayParams, registry_index: usize) -> ClaimBundle {
    ClaimBundle::exact_for_registry_index(params, registry_index)
        .expect("exact claim should exist for registry index")
}

#[test]
fn claim_bound_extend_keeps_explicit_row_family_across_segments() {
    let params = build_params();
    let claim = exact_claim(&params, 1);
    let mut store = UnauthStore::new_with_claims(params.clone(), vec![claim.clone()]);

    store.extend(20).expect("initial extend");
    store.extend(28).expect("follow-up extend");

    let owner = params.shard_registry[claim.registry_index].key;
    assert_eq!(store.coded.shard_family_count(), 1);
    assert!(store.coded.contains_shard(&owner));
    assert_eq!(store.coded.shard_family_segment_count(&owner), Some(4));

    let claim_rows = claim
        .row_label_indices(&params)
        .expect("exact claim rows should resolve");
    let expected_rows = claim_rows
        .into_iter()
        .map(|idx| RowKey::new(owner, params.row_labels[idx]))
        .collect::<Vec<_>>();
    assert_eq!(store.coded.row_family_count(), expected_rows.len());
    for row_key in expected_rows {
        assert!(store.coded.contains_row(&row_key));
        assert_eq!(store.coded.row_family_segment_count(&row_key), Some(4));
    }
}

#[test]
fn claim_bound_extend_matches_generic_default_segment_materialization() {
    let params = build_params();
    let claim = exact_claim(&params, 1);
    let initial_len = 20u64;

    let mut optimized = UnauthStore::new_with_claims(params.clone(), vec![claim.clone()]);
    optimized.extend(initial_len).expect("optimized extend");

    let zero = encode_value(Fp::from(0u64));
    let mut generic = UnauthStore::new_with_claims(params.clone(), vec![claim.clone()]);
    generic
        .extend_with_segments_for_claims(initial_len, &[claim], |_| vec![zero; params.n_grid])
        .expect("generic extend");

    assert_eq!(optimized.len(), generic.len());
    assert_eq!(optimized.segments(), generic.segments());
    assert!(optimized.coded.equals_for_tests(&params, &generic.coded));
}

#[test]
fn registry_claim_view_builds_party_local_claim_bundles() {
    let params = build_params();
    let file = MockRegistryFile {
        system: NetSystemConfig {
            unauth: NetUnauthConfig {
                segment_len: 8,
                layout_base_n_grid: None,
                s: 4,
                shard_count: 4,
                seed: 42,
            },
            auth: NetAuthConfig {
                tree_height: 3,
                node_linearization: paper_a_optimized::core::NodeLinearization::Preorder,
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
                active: true,
            },
        ],
        assignments: Vec::new(),
        unauth_epoch: Some(epoch_from_params(&params)),
        unauth_claims: vec![
            PartyClaimBundle {
                party_id: 0,
                registry_index: 1,
                row_family_id: None,
                row_label_indices: params
                    .row_label_indices_for_registry_index(1)
                    .into_iter()
                    .map(|idx| idx as u32)
                    .collect(),
            },
            PartyClaimBundle {
                party_id: 1,
                registry_index: 2,
                row_family_id: None,
                row_label_indices: params
                    .row_label_indices_for_registry_index(2)
                    .into_iter()
                    .map(|idx| idx as u32)
                    .collect(),
            },
        ],
        shard_registry: Some(encode_registry(&params.shard_registry)),
    };

    let (view_params, claims) = build_unauth_registry_view(&file).expect("registry view");
    assert_eq!(view_params.row_labels, params.row_labels);
    assert_eq!(view_params.row_eval_points, params.row_eval_points);
    assert_eq!(
        view_params.party_registry_indices,
        vec![vec![1], vec![2], vec![]]
    );

    let local = claim_bundles_for_party(&view_params, &claims, 0).expect("party claims");
    assert_eq!(local, vec![exact_claim(&view_params, 1)]);

    let other = claim_bundles_for_party(&view_params, &claims, 2).expect("empty claims");
    assert!(other.is_empty());
}

#[test]
fn registry_claim_view_accepts_explicit_row_family_id_without_explicit_rows() {
    let params = build_params();
    let file = MockRegistryFile {
        system: NetSystemConfig {
            unauth: NetUnauthConfig {
                segment_len: 8,
                layout_base_n_grid: None,
                s: 4,
                shard_count: 4,
                seed: 42,
            },
            auth: NetAuthConfig {
                tree_height: 3,
                node_linearization: paper_a_optimized::core::NodeLinearization::Preorder,
                allocated_leaves: 4,
            },
            unauth_array_len: Some(16),
        },
        parties: vec![PartyInfo {
            party_id: 0,
            rpc_host: "127.0.0.1:19000".to_string(),
            pubkey: None,
            active: true,
        }],
        assignments: Vec::new(),
        unauth_epoch: Some(epoch_from_params(&params)),
        unauth_claims: vec![PartyClaimBundle {
            party_id: 0,
            registry_index: 1,
            row_family_id: Some(1),
            row_label_indices: Vec::new(),
        }],
        shard_registry: Some(encode_registry(&params.shard_registry)),
    };

    let (_, claims) = build_unauth_registry_view(&file).expect("registry view");
    assert_eq!(claims.len(), 1);
    assert_eq!(claims[0].registry_index, 1);
    assert_eq!(
        claims[0].row_label_indices,
        params
            .row_label_indices_for_registry_index(1)
            .into_iter()
            .map(|idx| idx as u32)
            .collect::<Vec<_>>()
    );
}

#[test]
fn registry_claim_view_rejects_conflicting_row_family_and_explicit_rows() {
    let params = build_params();
    let file = MockRegistryFile {
        system: NetSystemConfig {
            unauth: NetUnauthConfig {
                segment_len: 8,
                layout_base_n_grid: None,
                s: 4,
                shard_count: 4,
                seed: 42,
            },
            auth: NetAuthConfig {
                tree_height: 3,
                node_linearization: paper_a_optimized::core::NodeLinearization::Preorder,
                allocated_leaves: 4,
            },
            unauth_array_len: Some(16),
        },
        parties: vec![PartyInfo {
            party_id: 0,
            rpc_host: "127.0.0.1:19000".to_string(),
            pubkey: None,
            active: true,
        }],
        assignments: Vec::new(),
        unauth_epoch: Some(epoch_from_params(&params)),
        unauth_claims: vec![PartyClaimBundle {
            party_id: 0,
            registry_index: 1,
            row_family_id: Some(1),
            row_label_indices: vec![0, 1],
        }],
        shard_registry: Some(encode_registry(&params.shard_registry)),
    };

    let err = build_unauth_registry_view(&file)
        .err()
        .expect("conflicting row family metadata should reject");
    let msg = format!("{:?}", err);
    assert!(msg.contains("row_family_id"));
}

#[test]
fn exact_core_claim_carries_row_family_descriptor_and_expands_via_params() {
    let params = build_params();
    let claim = exact_claim(&params, 1);
    assert_eq!(claim.registry_index, 1);
    assert_eq!(
        claim.row_family_id,
        params
            .row_family_id_for_registry_index(1)
            .expect("default family id")
    );
    assert_eq!(
        claim.row_label_indices(&params),
        Some(params.row_label_indices_for_registry_index(1))
    );
}

#[test]
fn registry_rejects_global_duplicate_row_labels_across_claims() {
    let params = build_params();
    let file = MockRegistryFile {
        system: NetSystemConfig {
            unauth: NetUnauthConfig {
                segment_len: 8,
                layout_base_n_grid: None,
                s: 4,
                shard_count: 4,
                seed: 42,
            },
            auth: NetAuthConfig {
                tree_height: 3,
                node_linearization: paper_a_optimized::core::NodeLinearization::Preorder,
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
        assignments: Vec::new(),
        unauth_epoch: Some(epoch_from_params(&params)),
        unauth_claims: vec![
            PartyClaimBundle {
                party_id: 0,
                registry_index: 0,
                row_family_id: None,
                row_label_indices: vec![0, 2],
            },
            PartyClaimBundle {
                party_id: 1,
                registry_index: 1,
                row_family_id: None,
                row_label_indices: vec![2, 3],
            },
        ],
        shard_registry: Some(encode_registry(&params.shard_registry)),
    };

    let err = build_unauth_registry_view(&file)
        .err()
        .expect("duplicate row labels should reject");
    let msg = format!("{:?}", err);
    assert!(msg.contains("row label index 2"));
}

#[test]
fn registry_default_claims_from_legacy_assignments_use_unique_rows_when_pool_suffices() {
    let segment_len = 8usize;
    let s = 4usize;
    let shard_count = 5usize;
    let n_grid = segment_len;
    let shard_registry = build_shard_registry_deterministic(shard_count, s, n_grid);
    let params = build_coded_params(&CodedArrayConfig {
        segment_len,
        s,
        shard_count,
        shard_registry: shard_registry.clone(),
        party_registry_indices: vec![(0..shard_count).collect()],
        seed: 43,
    });

    let file = MockRegistryFile {
        system: NetSystemConfig {
            unauth: NetUnauthConfig {
                segment_len,
                layout_base_n_grid: None,
                s,
                shard_count,
                seed: 43,
            },
            auth: NetAuthConfig {
                tree_height: 3,
                node_linearization: paper_a_optimized::core::NodeLinearization::Preorder,
                allocated_leaves: 4,
            },
            unauth_array_len: Some(16),
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
        unauth_epoch: Some(epoch_from_params(&params)),
        unauth_claims: Vec::new(),
        shard_registry: Some(encode_registry(&shard_registry)),
    };

    let (_, claims) = build_unauth_registry_view(&file).expect("unique defaults should resolve");
    let mut seen = std::collections::HashSet::new();
    for claim in claims {
        for row_label_index in claim.row_label_indices {
            assert!(seen.insert(row_label_index));
        }
    }
}

#[test]
fn registry_default_claims_scale_with_registry_sized_row_domain() {
    let segment_len = 8usize;
    let s = 4usize;
    let shard_count = 9usize;
    let n_grid = segment_len;
    let shard_registry = build_shard_registry_deterministic(shard_count, s, n_grid);
    let params = build_coded_params(&CodedArrayConfig {
        segment_len,
        s,
        shard_count,
        shard_registry: shard_registry.clone(),
        party_registry_indices: vec![(0..shard_count).collect()],
        seed: 44,
    });

    let file = MockRegistryFile {
        system: NetSystemConfig {
            unauth: NetUnauthConfig {
                segment_len,
                layout_base_n_grid: None,
                s,
                shard_count,
                seed: 44,
            },
            auth: NetAuthConfig {
                tree_height: 3,
                node_linearization: paper_a_optimized::core::NodeLinearization::Preorder,
                allocated_leaves: 4,
            },
            unauth_array_len: Some(16),
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
        unauth_epoch: Some(epoch_from_params(&params)),
        unauth_claims: Vec::new(),
        shard_registry: Some(encode_registry(&shard_registry)),
    };

    let (_, claims) = build_unauth_registry_view(&file).expect("defaults should scale");
    assert_eq!(claims.len(), shard_count);
    let mut seen = std::collections::HashSet::new();
    for registry_index in 0..shard_count {
        let claim = claims
            .iter()
            .find(|claim| claim.registry_index == registry_index as u32)
            .expect("claim for registry index");
        let expected: Vec<u32> = params
            .row_label_indices_for_registry_index(registry_index)
            .into_iter()
            .map(|idx| idx as u32)
            .collect();
        assert_eq!(claim.row_label_indices, expected);
        for &row_label_index in &claim.row_label_indices {
            assert!(seen.insert(row_label_index));
        }
    }
}

#[test]
fn registry_unauth_epoch_refreshes_row_label_lookup_cache() {
    let params = build_params();
    let mut row_labels = params.row_labels.clone();
    row_labels.reverse();
    let mut row_eval_points = params.row_eval_points.clone();
    row_eval_points.reverse();

    let file = MockRegistryFile {
        system: NetSystemConfig {
            unauth: NetUnauthConfig {
                segment_len: 8,
                layout_base_n_grid: None,
                s: 4,
                shard_count: 4,
                seed: 42,
            },
            auth: NetAuthConfig {
                tree_height: 3,
                node_linearization: paper_a_optimized::core::NodeLinearization::Preorder,
                allocated_leaves: 4,
            },
            unauth_array_len: Some(16),
        },
        parties: vec![PartyInfo {
            party_id: 0,
            rpc_host: "127.0.0.1:19000".to_string(),
            pubkey: None,
            active: true,
        }],
        assignments: Vec::new(),
        unauth_epoch: Some(NetUnauthEpoch {
            epoch_id: 9,
            initialized_segments: Some(0),
            rows_per_shard: Some(params.rows_per_shard as u32),
            logical_to_grid_index: None,
            row_labels: row_labels.into_iter().map(NetFp).collect(),
            row_eval_points: row_eval_points.into_iter().map(NetFp).collect(),
        }),
        unauth_claims: vec![PartyClaimBundle {
            party_id: 0,
            registry_index: 1,
            row_family_id: None,
            row_label_indices: params
                .row_label_indices_for_registry_index(1)
                .into_iter()
                .rev()
                .map(|idx| idx as u32)
                .collect(),
        }],
        shard_registry: Some(encode_registry(&params.shard_registry)),
    };

    let (view_params, claims) = build_unauth_registry_view(&file).expect("registry view");
    assert_eq!(claims.len(), 1);
    for (idx, row_label) in view_params.row_labels.iter().copied().enumerate() {
        let row_label_bytes = encode_value(row_label);
        assert_eq!(view_params.row_label_bytes[idx], row_label_bytes);
        assert_eq!(
            view_params.row_label_index_for_bytes(row_label_bytes),
            Some(idx)
        );
    }
}

#[test]
fn registry_claim_view_normalizes_explicit_row_labels() {
    let params = build_params();
    let file = MockRegistryFile {
        system: NetSystemConfig {
            unauth: NetUnauthConfig {
                segment_len: 8,
                layout_base_n_grid: None,
                s: 4,
                shard_count: 4,
                seed: 42,
            },
            auth: NetAuthConfig {
                tree_height: 3,
                node_linearization: paper_a_optimized::core::NodeLinearization::Preorder,
                allocated_leaves: 4,
            },
            unauth_array_len: Some(16),
        },
        parties: vec![PartyInfo {
            party_id: 1,
            rpc_host: "127.0.0.1:19001".to_string(),
            pubkey: None,
            active: true,
        }],
        assignments: Vec::new(),
        unauth_epoch: Some(epoch_from_params(&params)),
        unauth_claims: vec![PartyClaimBundle {
            party_id: 1,
            registry_index: 1,
            row_family_id: None,
            row_label_indices: {
                let mut rows: Vec<u32> = params
                    .row_label_indices_for_registry_index(1)
                    .into_iter()
                    .map(|idx| idx as u32)
                    .collect();
                rows.push(rows[0]);
                rows.reverse();
                rows
            },
        }],
        shard_registry: Some(encode_registry(&params.shard_registry)),
    };

    let (_, claims) = build_unauth_registry_view(&file).expect("registry view");
    assert_eq!(claims.len(), 1);
    assert_eq!(
        claims[0].row_label_indices,
        params
            .row_label_indices_for_registry_index(1)
            .into_iter()
            .map(|idx| idx as u32)
            .collect::<Vec<_>>()
    );

    let local = claim_bundles_for_party(&params, &claims, 1).expect("party claims");
    assert_eq!(local, vec![exact_claim(&params, 1)]);
}

#[test]
fn registry_claim_view_expands_row_family_ids_into_local_legacy_rows() {
    let params = build_params();
    let family_id = params
        .row_family_id_for_registry_index(1)
        .expect("default family id");
    let file = MockRegistryFile {
        system: NetSystemConfig {
            unauth: NetUnauthConfig {
                segment_len: 8,
                layout_base_n_grid: None,
                s: 4,
                shard_count: 4,
                seed: 42,
            },
            auth: NetAuthConfig {
                tree_height: 3,
                node_linearization: paper_a_optimized::core::NodeLinearization::Preorder,
                allocated_leaves: 4,
            },
            unauth_array_len: Some(16),
        },
        parties: vec![PartyInfo {
            party_id: 1,
            rpc_host: "127.0.0.1:19001".to_string(),
            pubkey: None,
            active: true,
        }],
        assignments: Vec::new(),
        unauth_epoch: Some(epoch_from_params(&params)),
        unauth_claims: vec![PartyClaimBundle {
            party_id: 1,
            registry_index: 1,
            row_family_id: Some(family_id.0 as u32),
            row_label_indices: Vec::new(),
        }],
        shard_registry: Some(encode_registry(&params.shard_registry)),
    };

    let (_, claims) = build_unauth_registry_view(&file).expect("registry view");
    assert_eq!(claims.len(), 1);
    assert_eq!(claims[0].row_family_id, Some(family_id.0 as u32));
    assert_eq!(
        claims[0].row_label_indices,
        params
            .row_label_indices_for_registry_index(1)
            .into_iter()
            .map(|idx| idx as u32)
            .collect::<Vec<_>>()
    );

    let local = claim_bundles_for_party(&params, &claims, 1).expect("party claims");
    assert_eq!(local, vec![exact_claim(&params, 1)]);
}

#[test]
fn registry_claim_view_rejects_mismatched_row_family_id_and_explicit_rows() {
    let params = build_params();
    let family_id = params
        .row_family_id_for_registry_index(1)
        .expect("default family id");
    let file = MockRegistryFile {
        system: NetSystemConfig {
            unauth: NetUnauthConfig {
                segment_len: 8,
                layout_base_n_grid: None,
                s: 4,
                shard_count: 4,
                seed: 42,
            },
            auth: NetAuthConfig {
                tree_height: 3,
                node_linearization: paper_a_optimized::core::NodeLinearization::Preorder,
                allocated_leaves: 4,
            },
            unauth_array_len: Some(16),
        },
        parties: vec![PartyInfo {
            party_id: 1,
            rpc_host: "127.0.0.1:19001".to_string(),
            pubkey: None,
            active: true,
        }],
        assignments: Vec::new(),
        unauth_epoch: Some(epoch_from_params(&params)),
        unauth_claims: vec![PartyClaimBundle {
            party_id: 1,
            registry_index: 1,
            row_family_id: Some(family_id.0 as u32),
            row_label_indices: params
                .row_label_indices_for_registry_index(0)
                .into_iter()
                .map(|idx| idx as u32)
                .collect(),
        }],
        shard_registry: Some(encode_registry(&params.shard_registry)),
    };

    let err = build_unauth_registry_view(&file)
        .err()
        .expect("mismatched explicit rows should reject");
    let msg = format!("{:?}", err);
    assert!(msg.contains("row_family_id"));
}

#[test]
fn registry_migrate_kappa_double_keeps_claims_resolvable_in_target_lens() {
    let params = build_params();
    let explicit_rows = params.row_label_indices_for_registry_index(0);
    let mut file = MockRegistryFile {
        system: NetSystemConfig {
            unauth: NetUnauthConfig {
                segment_len: 8,
                layout_base_n_grid: None,
                s: 4,
                shard_count: 4,
                seed: 42,
            },
            auth: NetAuthConfig {
                tree_height: 3,
                node_linearization: paper_a_optimized::core::NodeLinearization::Preorder,
                allocated_leaves: 4,
            },
            unauth_array_len: Some(16),
        },
        parties: vec![PartyInfo {
            party_id: 0,
            rpc_host: "127.0.0.1:19000".to_string(),
            pubkey: None,
            active: true,
        }],
        assignments: Vec::new(),
        unauth_epoch: Some(epoch_from_params(&params)),
        unauth_claims: vec![PartyClaimBundle {
            party_id: 0,
            registry_index: 0,
            row_family_id: None,
            row_label_indices: explicit_rows.iter().map(|idx| *idx as u32).collect(),
        }],
        shard_registry: Some(encode_registry(&params.shard_registry)),
    };

    migrate_unauth_registry_file(&mut file, 2).expect("kappa-double registry migrate");
    assert_eq!(file.system.unauth.s, 2);
    assert_eq!(file.unauth_claims.len(), 2);
    assert_eq!(file.assignments.len(), 1);
    assert_eq!(file.assignments[0].indices, vec![0, 1]);

    let (target_params, claims) = build_unauth_registry_view(&file).expect("target registry view");
    let local = claim_bundles_for_party(&target_params, &claims, 0).expect("target party claims");
    assert_eq!(local.len(), 2);
    assert_eq!(
        local
            .iter()
            .map(|claim| claim.registry_index)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    let mut union_rows = Vec::new();
    for claim in &local {
        assert!(
            claim
                .row_label_indices(&target_params)
                .expect("local target claim rows")
                .iter()
                .all(|&idx| idx < target_params.row_labels.len())
        );
        union_rows.extend(
            claim
                .row_label_indices(&target_params)
                .expect("local target claim rows"),
        );
    }
    union_rows.sort_unstable();
    union_rows.dedup();
    assert_eq!(union_rows, explicit_rows);
}

#[test]
fn registry_onboarding_adds_new_shards_and_unique_rows_for_joining_party() {
    let params = build_params();
    let new_count = params.shard_registry.len() + 2;
    let new_registry = build_shard_registry_deterministic(new_count, params.s, params.n_grid);
    let new_params = build_coded_params(&CodedArrayConfig {
        segment_len: params.segment_len,
        s: params.s,
        shard_count: new_count,
        shard_registry: new_registry,
        party_registry_indices: vec![(0..new_count).collect()],
        seed: 42,
    });
    let mut file = MockRegistryFile {
        system: NetSystemConfig {
            unauth: NetUnauthConfig {
                segment_len: 8,
                layout_base_n_grid: None,
                s: 4,
                shard_count: 4,
                seed: 42,
            },
            auth: NetAuthConfig {
                tree_height: 3,
                node_linearization: paper_a_optimized::core::NodeLinearization::Preorder,
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
                party_id: 3,
                rpc_host: "127.0.0.1:19003".to_string(),
                pubkey: None,
                active: false,
            },
        ],
        assignments: Vec::new(),
        unauth_epoch: Some(epoch_from_params(&params)),
        unauth_claims: vec![
            PartyClaimBundle {
                party_id: 0,
                registry_index: 0,
                row_family_id: None,
                row_label_indices: vec![0],
            },
            PartyClaimBundle {
                party_id: 0,
                registry_index: 1,
                row_family_id: None,
                row_label_indices: vec![1],
            },
            PartyClaimBundle {
                party_id: 0,
                registry_index: 2,
                row_family_id: None,
                row_label_indices: vec![2],
            },
            PartyClaimBundle {
                party_id: 0,
                registry_index: 3,
                row_family_id: None,
                row_label_indices: vec![3],
            },
        ],
        shard_registry: Some(encode_registry(&params.shard_registry)),
    };

    let new_claims =
        onboard_unauth_party_with_new_shards(&mut file, 3, 2).expect("onboard joining party");
    assert_eq!(file.system.unauth.shard_count, 6);
    assert_eq!(new_claims.len(), 2);
    assert_eq!(new_claims[0].registry_index, 4);
    assert_eq!(new_claims[1].registry_index, 5);
    assert_eq!(new_claims[0].row_label_indices, Vec::<u32>::new());
    assert_eq!(new_claims[1].row_label_indices, Vec::<u32>::new());
    assert_eq!(
        new_claims[0].row_family_id,
        new_params
            .row_family_id_for_registry_index(4)
            .map(|family_id| family_id.0 as u32)
    );
    assert_eq!(
        new_claims[1].row_family_id,
        new_params
            .row_family_id_for_registry_index(5)
            .map(|family_id| family_id.0 as u32)
    );

    let (_, claims) = build_unauth_registry_view(&file).expect("updated registry view");
    let join_claims: Vec<_> = claims
        .into_iter()
        .filter(|claim| claim.party_id == 3)
        .collect();
    assert_eq!(join_claims.len(), 2);
    assert_eq!(
        join_claims[0].row_label_indices,
        new_params
            .row_label_indices_for_registry_index(4)
            .into_iter()
            .map(|idx| idx as u32)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        join_claims[1].row_label_indices,
        new_params
            .row_label_indices_for_registry_index(5)
            .into_iter()
            .map(|idx| idx as u32)
            .collect::<Vec<_>>()
    );
    assert_eq!(file.assignments.len(), 2);
    let join_assignment = file
        .assignments
        .iter()
        .find(|assignment| assignment.party_id == 3)
        .expect("joining assignment");
    assert_eq!(join_assignment.indices, vec![4, 5]);
}
