use ark_ff::Field;
use paper_a_optimized::coded_array::params::CodedArrayConfig;
use paper_a_optimized::unauth::{
    ClaimBundle, HelperRowMigration, KappaDoubleRowEvalMigration, MigrateError,
    PartnerShardMigration, RowKey, UnauthStore, build_migrated_params,
};
use paper_a_optimized::{
    Fp, ShardKey, ShardRegEntry, build_coded_params, build_shard_registry_deterministic,
    decode_value, encode_value, fp_pow,
};

fn build_params(
    segment_len: usize,
    s: usize,
    shard_count: usize,
) -> paper_a_optimized::CodedArrayParams {
    let n_grid = segment_len;
    let cfg = CodedArrayConfig {
        segment_len,
        s,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count, s, n_grid),
        party_registry_indices: vec![(0..shard_count).collect()],
        seed: 11,
    };
    build_coded_params(&cfg)
}

fn collect_values(store: &UnauthStore) -> Vec<paper_a_optimized::Value256> {
    (0..store.len())
        .map(|idx| store.read(idx).expect("read"))
        .collect()
}

fn build_party_claims(
    params: &paper_a_optimized::CodedArrayParams,
    registry_indices: &[usize],
) -> Vec<ClaimBundle> {
    registry_indices
        .iter()
        .map(|&registry_index| {
            ClaimBundle::exact_for_registry_index(params, registry_index)
                .expect("exact claim should exist for registry index")
        })
        .collect()
}

fn collect_partner_shards(store: &UnauthStore) -> Vec<PartnerShardMigration> {
    store
        .params
        .shard_registry
        .iter()
        .enumerate()
        .filter_map(|(registry_index, entry)| {
            store
                .coded
                .clone_shard_family_segments(&store.params, &entry.key)
                .map(|segments| PartnerShardMigration {
                    registry_index,
                    segments,
                })
        })
        .collect()
}

fn collect_helper_rows(store: &UnauthStore, count: usize) -> Vec<HelperRowMigration> {
    let mut helper_rows = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (row_key, segments) in store.coded.as_nested_rows_for_tests(&store.params) {
        if !seen.insert(row_key.row_label_bytes) {
            continue;
        }
        helper_rows.push(HelperRowMigration {
            row_label: decode_value(row_key.row_label_bytes),
            segments,
        });
        if helper_rows.len() >= count {
            break;
        }
    }
    helper_rows.sort_by_key(|row| encode_value(row.row_label));
    helper_rows
}

fn assert_store_row_labels_are_unique(store: &UnauthStore) {
    let mut seen = std::collections::HashSet::new();
    for row_key in store.coded.row_keys() {
        assert!(
            seen.insert(row_key.row_label_bytes),
            "duplicate row label in store: {:?}",
            row_key
        );
    }
}

fn solve_linear_system_for_test(matrix: &[Vec<Fp>], rhs: &[Fp]) -> Option<Vec<Fp>> {
    if matrix.len() != rhs.len() {
        return None;
    }
    let vars = matrix.first().map_or(0, Vec::len);
    if matrix.iter().any(|row| row.len() != vars) {
        return None;
    }
    if vars == 0 {
        return rhs
            .iter()
            .all(|value| *value == Fp::from(0u64))
            .then(Vec::new);
    }

    let mut augmented = Vec::with_capacity(matrix.len());
    for (row, &value) in matrix.iter().zip(rhs.iter()) {
        let mut augmented_row = row.clone();
        augmented_row.push(value);
        augmented.push(augmented_row);
    }

    let mut pivot_rows = Vec::<(usize, usize)>::new();
    let mut pivot_row = 0usize;
    for col in 0..vars {
        let pivot = (pivot_row..augmented.len()).find(|&row| augmented[row][col] != Fp::from(0u64));
        let Some(pivot_idx) = pivot else {
            continue;
        };
        augmented.swap(pivot_row, pivot_idx);
        let inv = augmented[pivot_row][col].inverse()?;
        for value in &mut augmented[pivot_row][col..=vars] {
            *value *= inv;
        }
        let pivot_values = augmented[pivot_row].clone();
        for row in 0..augmented.len() {
            if row == pivot_row || augmented[row][col] == Fp::from(0u64) {
                continue;
            }
            let factor = augmented[row][col];
            for idx in col..=vars {
                augmented[row][idx] -= factor * pivot_values[idx];
            }
        }
        pivot_rows.push((pivot_row, col));
        pivot_row += 1;
    }

    for row in &augmented[pivot_row..] {
        if row[vars] != Fp::from(0u64) {
            return None;
        }
    }

    let mut solution = vec![Fp::from(0u64); vars];
    for (row, col) in pivot_rows {
        solution[col] = augmented[row][vars];
    }
    Some(solution)
}

fn assert_rows_match_fresh_target_oracle_and_target_columns(
    store: &UnauthStore,
    segment_values: &[Vec<paper_a_optimized::Value256>],
) {
    let target_claims = store.claim_bundles().expect("target claims");
    let target_params = store.params.clone();
    let mut fresh = UnauthStore::new_with_claims(target_params.clone(), target_claims.clone());
    fresh
        .extend_with_segments_for_claims(store.len(), &target_claims, |seg| {
            segment_values[seg as usize].clone()
        })
        .expect("fresh target extend");

    for (row_key, migrated_rows) in store.coded.as_nested_rows_for_tests(&store.params) {
        let fresh_rows = fresh
            .coded
            .clone_row_family_segments(&fresh.params, &row_key)
            .expect("fresh target row should exist");
        assert_eq!(
            migrated_rows, fresh_rows,
            "migrated row does not match fresh target oracle for {:?}",
            row_key
        );

        let row_label = decode_value(row_key.row_label_bytes);
        for (seg, migrated_row) in migrated_rows.iter().enumerate() {
            let repaired = store
                .repair_row_from_shards(seg, row_label)
                .expect("repair row from target columns");
            assert_eq!(
                migrated_row.as_slice(),
                repaired.as_slice(),
                "migrated row is incompatible with target columns for {:?} segment {}",
                row_key,
                seg
            );
        }
    }
}

#[test]
fn claim_bound_explicit_rows_match_fresh_kappa_doubled_oracle_and_kappa_doubled_columns() {
    let params = build_params(8, 4, 4);
    let source_claims = build_party_claims(&params, &[0, 1]);
    let mut segment_values: Vec<_> = (0..params.n_grid)
        .map(|idx| encode_value(Fp::from((idx + 1) as u64)))
        .collect();
    let raw_deltas = [(1u64, Fp::from(11u64)), (6u64, Fp::from(5u64))];

    let mut migrated = UnauthStore::new_with_claims(params.clone(), source_claims.clone());
    migrated
        .extend_with_segments_for_claims(params.segment_len as u64, &source_claims, |_| {
            segment_values.clone()
        })
        .expect("claim-bound extend");
    migrated
        .apply_deltas_raw(&raw_deltas, paper_a_optimized::DeltaMode::Aggregate)
        .expect("apply raw deltas before kappa-double");
    for &(raw_idx, delta) in &raw_deltas {
        let grid_index = raw_idx as usize;
        segment_values[grid_index] = encode_value(decode_value(segment_values[grid_index]) + delta);
    }
    migrated.migrate(2, &[]).expect("kappa-double");
    assert_rows_match_fresh_target_oracle_and_target_columns(&migrated, &[segment_values]);
}

#[test]
fn kappa_double_target_row_is_not_determined_by_a_single_projected_old_row_pair() {
    let source_params = build_params(8, 4, 4);
    let target_params =
        paper_a_optimized::unauth::build_migrated_params(&source_params, 2).expect("target params");
    let stride = source_params.induced_z_grid_size();
    let root = 0usize;
    let partner = root + stride;
    let source_dim = 2 * source_params.row_eval_points.len();

    let mut source_matrix = vec![vec![Fp::from(0u64); source_dim]; source_params.n_grid];
    let mut target_rhs =
        vec![vec![Fp::from(0u64); source_params.n_grid]; target_params.row_eval_points.len()];

    for basis_idx in 0..source_params.n_grid {
        let mut basis_values = vec![encode_value(Fp::from(0u64)); source_params.n_grid];
        basis_values[basis_idx] = encode_value(Fp::from(1u64));

        let source_rows =
            paper_a_optimized::compute_initial_rows_option_b(&source_params, &basis_values);
        let target_rows =
            paper_a_optimized::compute_initial_rows_option_b(&target_params, &basis_values);

        for (pos, value) in source_rows[root]
            .iter()
            .chain(source_rows[partner].iter())
            .enumerate()
        {
            source_matrix[basis_idx][pos] = *value;
        }
        for target_eval_idx in 0..target_params.row_eval_points.len() {
            target_rhs[target_eval_idx][basis_idx] = target_rows[root][target_eval_idx];
        }
    }

    let unsolved_target_evals: Vec<_> = target_rhs
        .iter()
        .enumerate()
        .filter_map(|(target_eval_idx, rhs)| {
            solve_linear_system_for_test(&source_matrix, rhs)
                .is_none()
                .then_some(target_eval_idx)
        })
        .collect();

    assert!(
        !unsolved_target_evals.is_empty(),
        "expected at least one kappa-double row evaluation to fall outside the span of a single projected old-row pair"
    );
}

#[test]
fn kappa_halve_migration_preserves_reads_and_repair() {
    let params = build_params(8, 4, 4);
    let mut store = UnauthStore::new(params.clone());
    store.extend(16).expect("extend");
    store
        .apply_deltas(
            &[
                (0, Fp::from(11u64)),
                (3, Fp::from(5u64)),
                (8, Fp::from(9u64)),
            ],
            paper_a_optimized::DeltaMode::Aggregate,
        )
        .expect("apply deltas");

    let source = store.clone();
    let expected = collect_values(&source);
    store.migrate(8, &[]).expect("migrate");

    assert_eq!(store.params.s, 8);
    assert_eq!(collect_values(&store), expected);

    let target_entry = store.params.shard_registry[0];
    let original = store
        .coded
        .clone_shard_family_segments(&store.params, &target_entry.key)
        .expect("target shard")
        .first()
        .cloned()
        .expect("segment shard");
    let removed = store
        .coded
        .remove_shard_family_segments(&store.params, &target_entry.key);
    assert!(removed.is_some(), "target shard missing");
    let repaired = store
        .repair_primary_shard_from_rows(0, target_entry)
        .expect("repair");
    assert_eq!(repaired, original);
}

#[test]
fn claim_bound_migration_reassigns_rows_for_target_mode() {
    let params = build_params(8, 4, 4);
    let claim = ClaimBundle::exact_for_registry_index(&params, 1)
        .expect("exact claim should exist for registry index");
    let mut store = UnauthStore::new_with_claims(params.clone(), vec![claim]);
    store.extend(14).expect("extend");

    let helper = {
        let mut full = UnauthStore::new(params.clone());
        full.extend(14).expect("helper extend");
        full
    };
    let source_entry = params.shard_registry[1];
    let twist = fp_pow(params.omega, (params.n_grid / (2 * params.s)) as u64);
    let partner_entry = ShardRegEntry {
        a: source_entry.a * twist,
        z: -source_entry.z,
        key: ShardKey {
            a_bytes: encode_value(source_entry.a * twist),
            z_bytes: encode_value(-source_entry.z),
        },
    };
    let partner = PartnerShardMigration {
        registry_index: 1,
        segments: (0..2)
            .map(|seg| {
                helper
                    .repair_primary_shard_from_rows(seg, partner_entry)
                    .expect("partner repair")
            })
            .collect(),
    };
    let helper_rows = collect_helper_rows(&helper, params.s);
    store
        .migrate_with_helper_rows(8, &[partner], &helper_rows)
        .expect("migrate");

    let owner = store.params.shard_registry[1].key;
    assert_eq!(store.params.s, 8);
    assert_eq!(store.coded.shard_family_count(), 1);
    assert!(store.coded.contains_shard(&owner));
    let migrated_claim = ClaimBundle::exact_for_registry_index(&store.params, 1)
        .expect("exact migrated claim should exist");
    let migrated_rows = migrated_claim
        .row_label_indices(&store.params)
        .expect("migrated exact claim rows");
    assert_eq!(store.coded.row_family_count(), migrated_rows.len());
    for row_label_index in migrated_rows {
        let row_label = store.params.row_labels[row_label_index];
        let row_key = RowKey::new(owner, row_label);
        assert!(store.coded.contains_row(&row_key));
    }
}

#[test]
fn kappa_double_migration_preserves_reads() {
    let params = build_params(8, 8, 4);
    let mut store = UnauthStore::new(params.clone());
    store.extend(8).expect("extend");
    store
        .apply_deltas(
            &[(0, Fp::from(3u64)), (5, Fp::from(13u64))],
            paper_a_optimized::DeltaMode::Aggregate,
        )
        .expect("apply deltas");

    let source = store.clone();
    let expected = collect_values(&source);
    store.migrate(4, &[]).expect("kappa-double migrate");

    assert_eq!(store.params.s, 4);
    assert_eq!(collect_values(&store), expected);
}

#[test]
fn claim_bound_kappa_double_rehydrates_rows_from_target_columns_for_depth_one_claims() {
    let params = build_params(8, 8, 4);
    let claims = build_party_claims(&params, &[1]);
    let mut segment_values: Vec<_> = (0..params.n_grid)
        .map(|idx| encode_value(Fp::from((idx + 3) as u64)))
        .collect();
    let raw_deltas = [(0u64, Fp::from(3u64)), (5u64, Fp::from(13u64))];
    let mut store = UnauthStore::new_with_claims(params.clone(), claims.clone());
    store
        .extend_with_segments_for_claims(params.segment_len as u64, &claims, |_| {
            segment_values.clone()
        })
        .expect("extend");
    store
        .apply_deltas_raw(&raw_deltas, paper_a_optimized::DeltaMode::Aggregate)
        .expect("apply deltas");
    for &(raw_idx, delta) in &raw_deltas {
        let grid_index = raw_idx as usize;
        segment_values[grid_index] = encode_value(decode_value(segment_values[grid_index]) + delta);
    }

    store.migrate(4, &[]).expect("one-step kappa-double");
    assert_rows_match_fresh_target_oracle_and_target_columns(&store, &[segment_values]);
}

#[test]
fn repeated_claim_bound_kappa_double_rehydrates_rows_after_each_step() {
    let params = build_params(8, 8, 4);
    let claims = build_party_claims(&params, &[0]);
    let mut segment_values: Vec<_> = (0..params.n_grid)
        .map(|idx| encode_value(Fp::from((idx + 17) as u64)))
        .collect();
    let raw_deltas = [(1u64, Fp::from(7u64)), (6u64, Fp::from(19u64))];
    let mut store = UnauthStore::new_with_claims(params, claims.clone());
    store
        .extend_with_segments_for_claims(8, &claims, |_| segment_values.clone())
        .expect("extend");
    store
        .apply_deltas_raw(&raw_deltas, paper_a_optimized::DeltaMode::Aggregate)
        .expect("apply deltas");
    for &(raw_idx, delta) in &raw_deltas {
        let grid_index = raw_idx as usize;
        segment_values[grid_index] = encode_value(decode_value(segment_values[grid_index]) + delta);
    }

    store.migrate(4, &[]).expect("first kappa-double");
    assert_rows_match_fresh_target_oracle_and_target_columns(&store, &[segment_values.clone()]);
    store.migrate(2, &[]).expect("second kappa-double");
    assert_rows_match_fresh_target_oracle_and_target_columns(&store, &[segment_values]);
}

#[test]
fn claim_bound_unique_rows_stay_unique_across_kappa_double_and_kappa_halve() {
    let params = build_params(8, 4, 4);
    let claims = build_party_claims(&params, &[0, 1]);
    let segment_values: Vec<_> = (0..params.n_grid)
        .map(|idx| encode_value(Fp::from((idx + 11) as u64)))
        .collect();

    let mut store = UnauthStore::new_with_claims(params, claims.clone());
    store
        .extend_with_segments_for_claims(8, &claims, |_| segment_values.clone())
        .expect("extend");
    assert_store_row_labels_are_unique(&store);

    store.migrate(2, &[]).expect("kappa-double");
    assert_store_row_labels_are_unique(&store);

    store.migrate(4, &[]).expect("kappa-halve");
    assert_store_row_labels_are_unique(&store);
}

#[test]
fn claim_bound_kappa_halve_matches_full_store_after_helper_hydrated_kappa_doubled_block_write() {
    use paper_a_optimized::coded_array::params::build_shard_registry_deterministic;
    use paper_a_optimized::unauth::DeltaMode;
    use paper_a_optimized::{Fp, build_coded_params, encode_value};

    let segment_len: usize = 32;
    let source_s: usize = 16;
    let kappa_doubled_s: usize = 8;
    let shard_count: usize = 8;
    let n_grid = segment_len;
    let params = build_coded_params(&CodedArrayConfig {
        segment_len,
        s: source_s,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count, source_s, n_grid),
        party_registry_indices: vec![vec![0, 1, 2, 3], vec![4, 5], vec![6, 7]],
        seed: 53,
    });
    let segment_values = vec![encode_value(Fp::from(0u64)); params.n_grid];

    let all_claims: Vec<_> = params
        .party_registry_indices
        .iter()
        .flat_map(|party_indices| build_party_claims(&params, party_indices))
        .collect();
    let mut full = UnauthStore::new_with_claims(params.clone(), all_claims.clone());
    full.extend_with_segments_for_claims(params.segment_len as u64, &all_claims, |_| {
        segment_values.clone()
    })
    .expect("full extend");

    let claims: Vec<_> = params.party_registry_indices[0]
        .iter()
        .map(|&registry_index| all_claims[registry_index].clone())
        .collect();
    let mut claim_bound = UnauthStore::new_with_claims(params.clone(), claims.clone());
    claim_bound
        .extend_with_segments_for_claims(params.segment_len as u64, &claims, |_| {
            segment_values.clone()
        })
        .expect("claim-bound extend");

    full.apply_block_deltas(
        1,
        &[
            (0usize, Fp::from(0x21u64)),
            (2usize, Fp::from(0x34u64)),
            (3usize, Fp::from(0x55u64)),
        ],
        DeltaMode::Aggregate,
    )
    .expect("full old-mode block write");
    claim_bound
        .apply_block_deltas(
            1,
            &[
                (0usize, Fp::from(0x21u64)),
                (2usize, Fp::from(0x34u64)),
                (3usize, Fp::from(0x55u64)),
            ],
            DeltaMode::Aggregate,
        )
        .expect("claim old-mode block write");

    full.migrate(kappa_doubled_s, &[])
        .expect("full kappa-double");
    let kappa_double_helpers = collect_partner_shards(&full);
    claim_bound
        .migrate(kappa_doubled_s, &kappa_double_helpers)
        .expect("claim kappa-double");

    let deltas = [
        (1usize, Fp::from(0x66u64)),
        (4usize, Fp::from(0x77u64)),
        (6usize, Fp::from(0x88u64)),
    ];
    full.apply_block_deltas(1, &deltas, DeltaMode::Aggregate)
        .expect("full kappa-double block write");
    claim_bound
        .apply_block_deltas(1, &deltas, DeltaMode::Aggregate)
        .expect("claim kappa-double block write");

    full.migrate(source_s, &[]).expect("full kappa-halve");
    claim_bound
        .migrate(source_s, &[])
        .expect("claim kappa-halve");

    for (key, claim_shards) in claim_bound
        .coded
        .as_nested_shards_for_tests(&claim_bound.params)
    {
        let full_shards = full
            .coded
            .clone_shard_family_segments(&full.params, &key)
            .expect("oracle shard should exist");
        assert_eq!(claim_shards, full_shards);
    }
    for (key, claim_rows) in claim_bound
        .coded
        .as_nested_rows_for_tests(&claim_bound.params)
    {
        let full_rows = full
            .coded
            .clone_row_family_segments(&full.params, &key)
            .expect("oracle row should exist");
        assert_eq!(claim_rows, full_rows);
    }
}

#[test]
fn kappa_halve_points_match_remapped_source_oracle_after_kappa_doubled_block_write() {
    use paper_a_optimized::coded_array::params::build_shard_registry_deterministic;
    use paper_a_optimized::unauth::DeltaMode;
    use paper_a_optimized::{Fp, build_coded_params, encode_value};

    let segment_len: usize = 32;
    let source_s: usize = 16;
    let kappa_doubled_s: usize = 8;
    let shard_count: usize = 8;
    let n_grid = segment_len;
    let params = build_coded_params(&CodedArrayConfig {
        segment_len,
        s: source_s,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count, source_s, n_grid),
        party_registry_indices: vec![vec![0, 1, 2, 3], vec![4, 5], vec![6, 7]],
        seed: 59,
    });
    let segment_values = vec![encode_value(Fp::from(0u64)); params.n_grid];

    let all_claims: Vec<_> = params
        .party_registry_indices
        .iter()
        .flat_map(|party_indices| build_party_claims(&params, party_indices))
        .collect();
    let mut migrated = UnauthStore::new_with_claims(params.clone(), all_claims.clone());
    migrated
        .extend_with_segments_for_claims(params.segment_len as u64, &all_claims, |_| {
            segment_values.clone()
        })
        .expect("full extend");
    migrated
        .apply_block_deltas(
            1,
            &[
                (0usize, Fp::from(0x21u64)),
                (2usize, Fp::from(0x34u64)),
                (3usize, Fp::from(0x55u64)),
            ],
            DeltaMode::Aggregate,
        )
        .expect("old-mode block write");
    let mut expected_values = collect_values(&migrated);
    migrated
        .migrate(kappa_doubled_s, &[])
        .expect("kappa-double");
    let deltas = [
        (1usize, Fp::from(0x66u64)),
        (4usize, Fp::from(0x77u64)),
        (6usize, Fp::from(0x88u64)),
    ];
    let kappa_doubled_pos = paper_a_optimized::unauth::semantic_block_position(&migrated.params, 1);
    let kappa_doubled_block_indices = paper_a_optimized::unauth::block_grid_indices(
        &migrated.params,
        kappa_doubled_pos.block_in_segment,
    );
    for &(offset, delta) in &deltas {
        let idx = kappa_doubled_pos.segment * migrated.params.segment_len
            + kappa_doubled_block_indices[offset];
        expected_values[idx] = encode_value(decode_value(expected_values[idx]) + delta);
    }
    migrated
        .apply_block_deltas(1, &deltas, DeltaMode::Aggregate)
        .expect("kappa-double block write");
    migrated.migrate(source_s, &[]).expect("kappa-halve");
    assert_eq!(collect_values(&migrated), expected_values);
}

#[test]
fn claim_bound_kappa_halve_rows_match_fresh_source_materialization_without_kappa_doubled_block_write()
 {
    use paper_a_optimized::coded_array::params::build_shard_registry_deterministic;
    use paper_a_optimized::unauth::DeltaMode;
    use paper_a_optimized::{Fp, build_coded_params, encode_value};

    let segment_len: usize = 32;
    let source_s: usize = 16;
    let kappa_doubled_s: usize = 8;
    let shard_count: usize = 8;
    let n_grid = segment_len;
    let params = build_coded_params(&CodedArrayConfig {
        segment_len,
        s: source_s,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count, source_s, n_grid),
        party_registry_indices: vec![vec![0, 1, 2, 3], vec![4, 5], vec![6, 7]],
        seed: 61,
    });
    let claims: Vec<_> = params
        .party_registry_indices
        .iter()
        .flat_map(|party_indices| build_party_claims(&params, party_indices))
        .collect();
    let segment_values = vec![encode_value(Fp::from(0u64)); params.n_grid];

    let mut migrated = UnauthStore::new_with_claims(params.clone(), claims.clone());
    migrated
        .extend_with_segments_for_claims(params.segment_len as u64, &claims, |_| {
            segment_values.clone()
        })
        .expect("claim-bound extend");
    migrated
        .apply_block_deltas(
            1,
            &[
                (0usize, Fp::from(0x21u64)),
                (2usize, Fp::from(0x34u64)),
                (3usize, Fp::from(0x55u64)),
            ],
            DeltaMode::Aggregate,
        )
        .expect("old-mode block write");
    migrated
        .migrate(kappa_doubled_s, &[])
        .expect("kappa-double");
    migrated.migrate(source_s, &[]).expect("kappa-halve");

    let kappa_halved_claims = migrated.claim_bundles().expect("kappa-halve claims");
    let mut fresh = UnauthStore::new_with_claims(params.clone(), kappa_halved_claims.clone());
    fresh
        .extend_with_segments_for_claims(params.segment_len as u64, &kappa_halved_claims, |_| {
            segment_values.clone()
        })
        .expect("fresh extend");
    fresh
        .apply_deltas(
            &[
                (1u64, Fp::from(0x21u64)),
                (5u64, Fp::from(0x34u64)),
                (7u64, Fp::from(0x55u64)),
            ],
            DeltaMode::Aggregate,
        )
        .expect("fresh logical writes");

    for (row_key, migrated_rows) in migrated.coded.as_nested_rows_for_tests(&migrated.params) {
        let fresh_rows = fresh
            .coded
            .clone_row_family_segments(&fresh.params, &row_key)
            .expect("fresh row should exist");
        assert_eq!(migrated_rows, fresh_rows);
    }
}

#[test]
fn claim_bound_kappa_halve_rows_match_full_row_oracle_after_helper_hydrated_kappa_doubled_block_write()
 {
    use paper_a_optimized::coded_array::params::build_shard_registry_deterministic;
    use paper_a_optimized::unauth::DeltaMode;
    use paper_a_optimized::{Fp, build_coded_params, encode_value};

    let segment_len: usize = 32;
    let source_s: usize = 16;
    let kappa_doubled_s: usize = 8;
    let shard_count: usize = 8;
    let n_grid = segment_len;
    let params = build_coded_params(&CodedArrayConfig {
        segment_len,
        s: source_s,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count, source_s, n_grid),
        party_registry_indices: vec![vec![0, 1, 2, 3], vec![4, 5], vec![6, 7]],
        seed: 67,
    });
    let segment_values = vec![encode_value(Fp::from(0u64)); params.n_grid];

    let projected_claims = build_party_claims(&params, &params.party_registry_indices[0]);
    let mut projected = UnauthStore::new_with_claims(params.clone(), projected_claims.clone());
    projected
        .extend_with_segments_for_claims(params.segment_len as u64, &projected_claims, |_| {
            segment_values.clone()
        })
        .expect("projected extend");

    let oracle_claims: Vec<_> = params
        .party_registry_indices
        .iter()
        .flat_map(|party_indices| build_party_claims(&params, party_indices))
        .collect();
    let mut oracle = UnauthStore::new_with_claims(params.clone(), oracle_claims.clone());
    oracle
        .extend_with_segments_for_claims(params.segment_len as u64, &oracle_claims, |_| {
            segment_values.clone()
        })
        .expect("oracle extend");

    projected
        .apply_block_deltas(
            1,
            &[
                (0usize, Fp::from(0x21u64)),
                (2usize, Fp::from(0x34u64)),
                (3usize, Fp::from(0x55u64)),
            ],
            DeltaMode::Aggregate,
        )
        .expect("projected old-mode block write");
    oracle
        .apply_block_deltas(
            1,
            &[
                (0usize, Fp::from(0x21u64)),
                (2usize, Fp::from(0x34u64)),
                (3usize, Fp::from(0x55u64)),
            ],
            DeltaMode::Aggregate,
        )
        .expect("oracle old-mode block write");
    oracle
        .migrate(kappa_doubled_s, &[])
        .expect("oracle kappa-double");
    let kappa_double_helpers = collect_partner_shards(&oracle);
    projected
        .migrate(kappa_doubled_s, &kappa_double_helpers)
        .expect("projected kappa-double");

    projected
        .apply_block_deltas(
            1,
            &[
                (1usize, Fp::from(0x66u64)),
                (4usize, Fp::from(0x77u64)),
                (6usize, Fp::from(0x88u64)),
            ],
            DeltaMode::Aggregate,
        )
        .expect("projected kappa-double block write");
    oracle
        .apply_block_deltas(
            1,
            &[
                (1usize, Fp::from(0x66u64)),
                (4usize, Fp::from(0x77u64)),
                (6usize, Fp::from(0x88u64)),
            ],
            DeltaMode::Aggregate,
        )
        .expect("oracle kappa-double block write");
    projected
        .migrate(source_s, &[])
        .expect("projected kappa-halve");
    oracle.migrate(source_s, &[]).expect("oracle kappa-halve");

    for (row_key, projected_rows) in projected.coded.as_nested_rows_for_tests(&projected.params) {
        let oracle_rows = oracle
            .coded
            .clone_row_family_segments(&oracle.params, &row_key)
            .expect("oracle row should exist");
        assert_eq!(
            projected_rows, oracle_rows,
            "row mismatch for {:?}",
            row_key
        );
    }
}

#[test]
fn kappa_halve_helper_rows_rebuild_cleared_kappa_doubled_rows_against_full_oracle() {
    let segment_len: usize = 32;
    let source_s: usize = 16;
    let kappa_doubled_s: usize = 8;
    let shard_count: usize = 8;
    let params = build_params(segment_len, source_s, shard_count);
    let segment_values = vec![encode_value(Fp::from(0u64)); params.n_grid];

    let mut healthy = UnauthStore::new(params.clone());
    healthy
        .extend_with_segments(params.segment_len as u64, |_| segment_values.clone())
        .expect("full extend");
    healthy
        .apply_block_deltas(
            1,
            &[
                (0usize, Fp::from(0x21u64)),
                (2usize, Fp::from(0x34u64)),
                (3usize, Fp::from(0x55u64)),
            ],
            paper_a_optimized::unauth::DeltaMode::Aggregate,
        )
        .expect("old-mode block write");
    healthy.migrate(kappa_doubled_s, &[]).expect("kappa-double");

    let helper_rows = collect_helper_rows(&healthy, kappa_doubled_s);
    assert_eq!(helper_rows.len(), kappa_doubled_s);

    let mut degraded = healthy.clone();
    degraded.coded.clear_rows();
    assert!(matches!(
        degraded.migrate(source_s, &[]),
        Err(MigrateError::MissingLocalRows)
    ));

    let mut oracle = healthy.clone();
    oracle.migrate(source_s, &[]).expect("oracle kappa-halve");

    degraded
        .migrate_with_helper_rows(source_s, &[], &helper_rows)
        .expect("kappa-halve with helper rows");

    assert_eq!(degraded.claim_bundles(), oracle.claim_bundles());
    assert!(
        degraded
            .coded
            .equals_for_tests(&degraded.params, &oracle.coded)
    );
}

#[test]
fn retained_claim_state_after_kappa_double_keeps_only_assigned_child() {
    let segment_len: usize = 8;
    let source_s: usize = 4;
    let target_s: usize = 2;
    let shard_count: usize = 4;
    let params = build_params(segment_len, source_s, shard_count);
    let segment_values = vec![encode_value(Fp::from(0u64)); params.n_grid];

    let all_claims = build_party_claims(&params, &[0, 1, 2, 3]);
    let mut oracle = UnauthStore::new_with_claims(params.clone(), all_claims.clone());
    oracle
        .extend_with_segments_for_claims(params.segment_len as u64, &all_claims, |_| {
            segment_values.clone()
        })
        .expect("oracle extend");
    oracle.migrate(target_s, &[]).expect("oracle kappa-double");
    let helper_columns = collect_partner_shards(&oracle);

    let source_claim = build_party_claims(&params, &[0]);
    let mut store = UnauthStore::new_with_claims(params.clone(), source_claim.clone());
    store
        .extend_with_segments_for_claims(params.segment_len as u64, &source_claim, |_| {
            segment_values.clone()
        })
        .expect("store extend");
    store
        .migrate(target_s, &helper_columns)
        .expect("claim-bound kappa-double");
    assert_eq!(store.claim_bundles().expect("claims").len(), 2);

    store
        .retain_registry_indices(&[0])
        .expect("retain assigned left child");
    let retained = store.claim_bundles().expect("retained claims");
    assert_eq!(retained.len(), 1);
    assert_eq!(retained[0].registry_index, 0);
    assert_eq!(store.coded.shard_family_count(), 1);
    assert_eq!(store.coded.row_family_count(), 1);
}

#[test]
fn one_claim_left_kappa_double_never_materializes_discarded_child() {
    let segment_len: usize = 8;
    let source_s: usize = 4;
    let target_s: usize = 2;
    let shard_count: usize = 4;
    let params = build_params(segment_len, source_s, shard_count);
    let segment_values = |seg: usize| {
        (0..params.n_grid)
            .map(|idx| encode_value(Fp::from((seg * 100 + idx + 1) as u64)))
            .collect::<Vec<_>>()
    };

    let all_claims = build_party_claims(&params, &[0, 1, 2, 3]);
    let mut oracle = UnauthStore::new_with_claims(params.clone(), all_claims.clone());
    oracle
        .extend_with_segments_for_claims((params.segment_len * 2) as u64, &all_claims, |seg| {
            segment_values(seg)
        })
        .expect("oracle extend");
    let target_params = build_migrated_params(&params, target_s).expect("target params");
    let retained_claim =
        ClaimBundle::exact_for_registry_index(&target_params, 0).expect("retained claim");
    let row_label_indices = retained_claim
        .row_label_indices_slice(&target_params)
        .expect("retained rows");
    let row_labels: Vec<_> = row_label_indices
        .iter()
        .map(|&idx| target_params.row_labels[idx])
        .collect();
    let helper_targets = [2usize, 4, 6];
    let segs = [0usize, 1];
    let evals = oracle
        .handle_eval_kappa_double_target_at_row_labels_matrix(
            target_s,
            &segs,
            &helper_targets,
            &row_labels,
        )
        .expect("helper evals");
    let helper_evals: Vec<_> = helper_targets
        .iter()
        .copied()
        .enumerate()
        .map(|(target_pos, registry_index)| KappaDoubleRowEvalMigration {
            registry_index,
            segments: evals
                .iter()
                .map(|values_for_seg| values_for_seg[target_pos].clone())
                .collect(),
        })
        .collect();
    oracle.migrate(target_s, &[]).expect("oracle kappa-double");

    let source_claim = build_party_claims(&params, &[0]);
    let mut store = UnauthStore::new_with_claims(params.clone(), source_claim.clone());
    store
        .extend_with_segments_for_claims((params.segment_len * 2) as u64, &source_claim, |seg| {
            segment_values(seg)
        })
        .expect("store extend");
    store
        .migrate_kappa_double_one_claim_left(target_s, &helper_evals)
        .expect("one-claim kappa-double");

    let retained = store.claim_bundles().expect("retained claims");
    assert_eq!(retained.len(), 1);
    assert_eq!(retained[0].registry_index, 0);
    assert_eq!(store.coded.shard_family_count(), 1);
    assert_eq!(store.coded.row_family_count(), 1);
    assert!(store.params.shard_registry.get(1).is_some());
    let discarded_key = store.params.shard_registry[1].key;
    assert!(!store.coded.contains_shard(&discarded_key));

    let retained_key = store.params.shard_registry[0].key;
    assert_eq!(
        store
            .coded
            .clone_shard_family_segments(&store.params, &retained_key),
        oracle
            .coded
            .clone_shard_family_segments(&oracle.params, &retained_key)
    );
    for &row_label_idx in retained[0]
        .row_label_indices_slice(&store.params)
        .expect("retained row labels")
    {
        let row_key = RowKey::new(retained_key, store.params.row_labels[row_label_idx]);
        assert_eq!(
            store
                .coded
                .clone_row_family_segments(&store.params, &row_key),
            oracle
                .coded
                .clone_row_family_segments(&oracle.params, &row_key)
        );
    }
}

#[test]
fn migrate_rejects_non_adjacent_power_of_two_jump() {
    let params = build_params(16, 4, 4);
    let mut store = UnauthStore::new(params);
    store.extend(16).expect("extend");

    let err = store
        .migrate(16, &[])
        .expect_err("expected invalid migrate");
    assert!(matches!(err, MigrateError::InvalidTargetS));
}
