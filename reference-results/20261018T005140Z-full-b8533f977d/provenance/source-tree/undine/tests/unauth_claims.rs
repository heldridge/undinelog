use paper_a_optimized::coded_array::params::CodedArrayConfig;
use paper_a_optimized::net::registry::{
    MockRegistryFile, NetAuthConfig, NetShardRegEntry, NetSystemConfig, NetUnauthConfig,
    NetUnauthEpoch, PartyAssignments, PartyClaimBundle, PartyInfo, build_unauth_registry_view,
    claim_bundles_for_party, default_unauth_claims, kappa_double_unauth_registry_file,
    kappa_halve_unauth_registry_file, onboard_unauth_party_with_new_shards, validate_unauth_claims,
};
use paper_a_optimized::net::wire::NetFp;
use paper_a_optimized::unauth::{ClaimBundle, RowKey, UnauthStore};
use paper_a_optimized::{Fp, build_coded_params, build_shard_registry_deterministic, encode_value};
use std::collections::HashSet;

const KAPPA: usize = 2;

fn build_params_with(shard_count: usize, seed: u64) -> paper_a_optimized::CodedArrayParams {
    let cfg = CodedArrayConfig {
        kappa: KAPPA,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count),
        party_registry_indices: vec![(0..shard_count).collect()],
        seed,
    };
    build_coded_params(&cfg)
}

fn build_params() -> paper_a_optimized::CodedArrayParams {
    build_params_with(4, 42)
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
        row_labels: params.row_labels.iter().copied().map(NetFp).collect(),
    }
}

fn exact_claim(params: &paper_a_optimized::CodedArrayParams, registry_index: usize) -> ClaimBundle {
    ClaimBundle::exact_for_registry_index(params, registry_index)
        .expect("exact claim should exist for registry index")
}

fn party(party_id: u64, active: bool) -> PartyInfo {
    PartyInfo {
        party_id,
        rpc_host: format!("127.0.0.1:{}", 19000 + party_id),
        pubkey: None,
        active,
    }
}

fn party_claim(party_id: u64, registry_index: u32) -> PartyClaimBundle {
    PartyClaimBundle {
        party_id,
        registry_index,
    }
}

/// Registry file advertising `params` with explicit claims and no flat
/// assignments.
fn registry_file(
    params: &paper_a_optimized::CodedArrayParams,
    seed: u64,
    parties: Vec<PartyInfo>,
    unauth_claims: Vec<PartyClaimBundle>,
) -> MockRegistryFile {
    MockRegistryFile {
        system: NetSystemConfig {
            unauth: NetUnauthConfig {
                kappa: params.need,
                layout_base_kappa: None,
                shard_count: params.shard_registry.len(),
                seed,
            },
            auth: NetAuthConfig {
                tree_height: 3,
                node_linearization: paper_a_optimized::core::NodeLinearization::Preorder,
                allocated_leaves: 4,
            },
            unauth_array_len: Some(16),
        },
        parties,
        assignments: Vec::new(),
        unauth_epoch: Some(epoch_from_params(params)),
        unauth_claims,
        shard_registry: Some(encode_registry(&params.shard_registry)),
    }
}

fn registry_error(file: &MockRegistryFile, context: &str) -> String {
    let err = build_unauth_registry_view(file).err().expect(context);
    format!("{:?}", err)
}

#[test]
fn claim_bound_extend_keeps_claimed_column_and_row_across_segments() {
    let params = build_params();
    let claim = exact_claim(&params, 1);
    let mut store = UnauthStore::new_with_claims(params.clone(), vec![claim.clone()]);

    store.extend(10).expect("initial extend");
    store.extend(14).expect("follow-up extend");
    assert_eq!(store.segments(), 4);

    let owner = params.shard_registry[claim.registry_index].key;
    assert_eq!(store.coded.shard_family_count(), 1);
    assert!(store.coded.contains_shard(&owner));
    assert_eq!(store.coded.shard_family_segment_count(&owner), Some(4));

    let row_key = RowKey::new(owner, params.row_labels[claim.row_label_index()]);
    assert_eq!(store.coded.row_family_count(), 1);
    assert!(store.coded.contains_row(&row_key));
    assert_eq!(store.coded.row_family_segment_count(&row_key), Some(4));
    for seg in 0..store.segments() {
        assert_eq!(
            store
                .coded
                .shard_segment(&params, &owner, seg)
                .expect("claimed column")
                .len(),
            params.s
        );
        assert_eq!(
            store
                .coded
                .row_segment(&params, &row_key, seg)
                .expect("claimed row")
                .len(),
            params.row_len()
        );
    }
}

#[test]
fn claim_bound_extend_matches_generic_default_segment_materialization() {
    let params = build_params();
    let claim = exact_claim(&params, 1);
    let initial_len = 10u64;

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
    let file = registry_file(
        &params,
        42,
        vec![party(0, true), party(1, true), party(2, true)],
        vec![party_claim(0, 1), party_claim(1, 2)],
    );

    let (view_params, claims) = build_unauth_registry_view(&file).expect("registry view");
    assert_eq!(view_params.need, KAPPA);
    assert_eq!(view_params.n_grid, KAPPA * KAPPA);
    assert_eq!(view_params.row_labels, params.row_labels);
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
fn registry_claims_bind_stores_to_the_claimed_column_and_epoch_row() {
    let params = build_params();
    let mut epoch_labels = params.row_labels.clone();
    epoch_labels.reverse();
    let mut file = registry_file(
        &params,
        42,
        vec![party(0, true), party(1, true)],
        vec![party_claim(1, 3), party_claim(0, 0), party_claim(1, 1)],
    );
    file.unauth_epoch = Some(NetUnauthEpoch {
        epoch_id: 7,
        initialized_segments: Some(0),
        row_labels: epoch_labels.iter().copied().map(NetFp).collect(),
    });

    let (view_params, claims) = build_unauth_registry_view(&file).expect("registry view");
    assert_eq!(view_params.row_labels, epoch_labels);

    let local = claim_bundles_for_party(&view_params, &claims, 1).expect("party claims");
    assert_eq!(
        local,
        vec![exact_claim(&view_params, 3), exact_claim(&view_params, 1)]
    );

    let mut store = UnauthStore::new_with_claims(view_params.clone(), local.clone());
    store.extend(6).expect("extend");
    assert_eq!(store.segments(), 2);
    assert_eq!(store.coded.shard_family_count(), local.len());
    assert_eq!(store.coded.row_family_count(), local.len());
    for claim in &local {
        let owner = view_params.shard_registry[claim.registry_index].key;
        let row_key = RowKey::new(owner, epoch_labels[claim.registry_index]);
        assert_eq!(store.coded.shard_family_segment_count(&owner), Some(2));
        assert_eq!(store.coded.row_family_segment_count(&row_key), Some(2));
    }
    let unclaimed = view_params.shard_registry[0].key;
    assert!(!store.coded.contains_shard(&unclaimed));
    assert!(
        !store
            .coded
            .contains_row(&RowKey::new(unclaimed, epoch_labels[0]))
    );
}

#[test]
fn exact_core_claim_has_one_row_at_its_registry_index() {
    let params = build_params();
    let claim = exact_claim(&params, 1);
    assert_eq!(claim.registry_index, 1);
    assert_eq!(claim.row_label_index(), 1);
    assert_eq!(params.row_label_index_for_registry_index(1), Some(1));
    assert_eq!(
        claim.row_label_indices(&params),
        Some(params.row_label_indices_for_registry_index(1))
    );
    assert_eq!(claim.row_label_indices(&params), Some(vec![1]));
    assert!(ClaimBundle::exact_for_registry_index(&params, params.shard_registry.len()).is_none());
}

#[test]
fn registry_rejects_duplicate_row_labels_in_epoch() {
    let params = build_params();
    let mut file = registry_file(
        &params,
        42,
        vec![party(0, true), party(1, true)],
        vec![party_claim(0, 0), party_claim(1, 1)],
    );
    let mut row_labels = params.row_labels.clone();
    row_labels[3] = row_labels[2];
    file.unauth_epoch = Some(NetUnauthEpoch {
        epoch_id: 7,
        initialized_segments: Some(0),
        row_labels: row_labels.into_iter().map(NetFp).collect(),
    });

    let msg = registry_error(&file, "duplicate row labels should reject");
    assert!(msg.contains("duplicate row label"));
}

#[test]
fn registry_rejects_row_label_on_a_semantic_axis() {
    let params = build_params();
    for invalid in [Fp::from(0u64), Fp::from(1u64), params.axis[1]] {
        let mut file = registry_file(
            &params,
            42,
            vec![party(0, true)],
            vec![party_claim(0, 0), party_claim(0, 1)],
        );
        let mut row_labels = params.row_labels.clone();
        row_labels[1] = invalid;
        file.unauth_epoch = Some(NetUnauthEpoch {
            epoch_id: 7,
            initialized_segments: Some(0),
            row_labels: row_labels.into_iter().map(NetFp).collect(),
        });

        let msg = registry_error(&file, "row label on a semantic axis should reject");
        assert!(msg.contains("semantic axis"));
    }
}

#[test]
fn registry_rejects_epoch_without_one_row_label_per_registry_entry() {
    let params = build_params();
    let mut file = registry_file(&params, 42, vec![party(0, true)], vec![party_claim(0, 0)]);
    let mut epoch = epoch_from_params(&params);
    epoch.row_labels.pop();
    file.unauth_epoch = Some(epoch);

    let msg = registry_error(&file, "missing row label should reject");
    assert!(msg.contains("expected 4 row labels"));
}

#[test]
fn registry_rejects_registry_index_with_two_owners() {
    let params = build_params();
    let claims = vec![party_claim(0, 0), party_claim(1, 1), party_claim(1, 0)];
    let file = registry_file(
        &params,
        42,
        vec![party(0, true), party(1, true)],
        claims.clone(),
    );

    let msg = registry_error(&file, "two owners of one registry index should reject");
    assert!(msg.contains("registry index 0 claimed by both party 0 and party 1"));

    assert!(validate_unauth_claims(&params, &claims).is_err());
    assert!(claim_bundles_for_party(&params, &claims, 1).is_err());
    assert!(validate_unauth_claims(&params, &claims[..2]).is_ok());
}

#[test]
fn registry_rejects_out_of_range_claim_index() {
    let params = build_params();
    let claims = vec![party_claim(0, 0), party_claim(0, 4)];
    let file = registry_file(&params, 42, vec![party(0, true)], claims.clone());

    let msg = registry_error(&file, "out-of-range claim should reject");
    assert!(msg.contains("claim references out-of-range registry index 4"));
    assert!(validate_unauth_claims(&params, &claims).is_err());

    let assignments = vec![PartyAssignments {
        party_id: 0,
        indices: vec![0, 4],
    }];
    let err = default_unauth_claims(&params, &assignments)
        .err()
        .expect("out-of-range assignment should reject");
    assert!(format!("{:?}", err).contains("out-of-range registry index 4"));
}

#[test]
fn registry_default_claims_from_assignments_use_unique_rows() {
    let shard_count = 5usize;
    let params = build_params_with(shard_count, 43);
    let mut file = registry_file(&params, 43, vec![party(0, true)], Vec::new());
    file.assignments = vec![PartyAssignments {
        party_id: 0,
        indices: vec![0, 1, 1, 2, 3, 4, 0],
    }];

    let (view_params, claims) =
        build_unauth_registry_view(&file).expect("unique defaults should resolve");
    assert_eq!(claims.len(), shard_count);
    let mut seen = HashSet::new();
    for claim in &claims {
        assert_eq!(claim.party_id, 0);
        let row_label = view_params.row_labels[claim.registry_index as usize];
        assert!(seen.insert(encode_value(row_label)));
    }
}

#[test]
fn registry_default_claims_scale_with_registry_sized_row_domain() {
    let shard_count = 9usize;
    let params = build_params_with(shard_count, 44);
    let mut file = registry_file(&params, 44, vec![party(0, true)], Vec::new());
    file.assignments = vec![PartyAssignments {
        party_id: 0,
        indices: (0..shard_count as u32).collect(),
    }];

    let (view_params, claims) = build_unauth_registry_view(&file).expect("defaults should scale");
    assert_eq!(claims.len(), shard_count);
    assert_eq!(view_params.row_labels.len(), shard_count);
    let local = claim_bundles_for_party(&view_params, &claims, 0).expect("party claims");
    let mut seen = HashSet::new();
    for registry_index in 0..shard_count {
        let claim = local
            .iter()
            .find(|claim| claim.registry_index == registry_index)
            .expect("claim for registry index");
        assert_eq!(
            claim.row_label_indices(&view_params),
            Some(params.row_label_indices_for_registry_index(registry_index))
        );
        assert_eq!(claim.row_label_index(), registry_index);
        assert!(seen.insert(encode_value(view_params.row_labels[registry_index])));
    }
}

#[test]
fn registry_unauth_epoch_refreshes_row_label_lookup_cache() {
    let params = build_params();
    let mut row_labels = params.row_labels.clone();
    row_labels.reverse();

    let mut file = registry_file(&params, 42, vec![party(0, true)], vec![party_claim(0, 1)]);
    file.unauth_epoch = Some(NetUnauthEpoch {
        epoch_id: 9,
        initialized_segments: Some(0),
        row_labels: row_labels.iter().copied().map(NetFp).collect(),
    });

    let (view_params, claims) = build_unauth_registry_view(&file).expect("registry view");
    assert_eq!(claims.len(), 1);
    assert_eq!(view_params.row_labels, row_labels);
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
fn registry_kappa_double_keeps_claims_resolvable_at_the_doubled_threshold() {
    let params = build_params();
    let source_claims: Vec<_> = (0..4).map(|idx| party_claim(0, idx)).collect();
    let mut file = registry_file(&params, 42, vec![party(0, true)], source_claims.clone());

    kappa_double_unauth_registry_file(&mut file).expect("kappa-double registry");
    assert_eq!(file.system.unauth.kappa, 2 * KAPPA);
    assert_eq!(file.system.unauth.layout_base_kappa, Some(KAPPA));
    assert_eq!(file.unauth_claims, source_claims);
    assert_eq!(file.assignments.len(), 1);
    assert_eq!(file.assignments[0].indices, vec![0, 1, 2, 3]);

    let (target_params, claims) = build_unauth_registry_view(&file).expect("target registry view");
    assert_eq!(target_params.need, 2 * KAPPA);
    assert_eq!(target_params.n_grid, 4 * KAPPA * KAPPA);
    assert_eq!(target_params.shard_registry, params.shard_registry);
    assert_eq!(target_params.row_labels, params.row_labels);
    let local = claim_bundles_for_party(&target_params, &claims, 0).expect("target party claims");
    assert_eq!(
        local
            .iter()
            .map(|claim| claim.registry_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    for claim in &local {
        assert_eq!(
            claim.row_label_indices(&target_params),
            Some(vec![claim.registry_index])
        );
        assert_eq!(
            target_params.row_labels[claim.row_label_index()],
            params.row_labels[claim.registry_index]
        );
    }

    kappa_halve_unauth_registry_file(&mut file).expect("kappa-halve registry");
    assert_eq!(file.system.unauth.kappa, KAPPA);
    let (halved_params, halved_claims) =
        build_unauth_registry_view(&file).expect("halved registry view");
    assert_eq!(halved_params.need, KAPPA);
    assert_eq!(halved_params.n_grid, params.n_grid);
    assert_eq!(halved_claims, source_claims);
}

#[test]
fn registry_kappa_double_requires_doubled_active_claims() {
    let params = build_params();
    let mut file = registry_file(
        &params,
        42,
        vec![party(0, true), party(1, false)],
        vec![
            party_claim(0, 0),
            party_claim(0, 1),
            party_claim(0, 2),
            party_claim(1, 3),
        ],
    );

    let err = kappa_double_unauth_registry_file(&mut file)
        .err()
        .expect("three active claims cannot serve threshold four");
    assert!(format!("{:?}", err).contains("requires at least 4 active claims, got 3"));
    assert_eq!(file.system.unauth.kappa, KAPPA);
}

#[test]
fn registry_onboarding_adds_new_shards_and_unique_rows_for_joining_party() {
    let params = build_params();
    let new_count = params.shard_registry.len() + 2;
    let new_params = build_params_with(new_count, 42);
    let existing_claims: Vec<_> = (0..4).map(|idx| party_claim(0, idx)).collect();
    let mut file = registry_file(
        &params,
        42,
        vec![party(0, true), party(3, false)],
        existing_claims.clone(),
    );

    let new_claims =
        onboard_unauth_party_with_new_shards(&mut file, 3, 2).expect("onboard joining party");
    assert_eq!(file.system.unauth.shard_count, 6);
    assert_eq!(file.system.unauth.kappa, KAPPA);
    assert_eq!(new_claims, vec![party_claim(3, 4), party_claim(3, 5)]);

    let (view_params, claims) = build_unauth_registry_view(&file).expect("updated registry view");
    assert_eq!(&claims[..4], existing_claims.as_slice());
    assert_eq!(&view_params.row_labels[..4], params.row_labels.as_slice());
    assert_eq!(
        &view_params.shard_registry[..4],
        params.shard_registry.as_slice()
    );
    assert_eq!(view_params.row_labels, new_params.row_labels);
    assert_eq!(view_params.shard_registry, new_params.shard_registry);
    let unique_labels: HashSet<_> = view_params
        .row_labels
        .iter()
        .map(|row_label| encode_value(*row_label))
        .collect();
    assert_eq!(unique_labels.len(), new_count);

    let join_claims = claim_bundles_for_party(&view_params, &claims, 3).expect("joining claims");
    assert_eq!(
        join_claims,
        vec![exact_claim(&view_params, 4), exact_claim(&view_params, 5)]
    );
    for claim in &join_claims {
        assert_eq!(
            claim.row_label_indices(&view_params),
            Some(new_params.row_label_indices_for_registry_index(claim.registry_index))
        );
    }
    assert_eq!(file.assignments.len(), 2);
    let join_assignment = file
        .assignments
        .iter()
        .find(|assignment| assignment.party_id == 3)
        .expect("joining assignment");
    assert_eq!(join_assignment.indices, vec![4, 5]);
}
