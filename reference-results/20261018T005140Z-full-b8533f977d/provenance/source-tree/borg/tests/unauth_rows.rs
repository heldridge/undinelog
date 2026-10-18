use paper_a_optimized::coded_array::params::CodedArrayConfig;
use paper_a_optimized::unauth::service::{UnauthRequest, UnauthResponse, UnauthService};
use paper_a_optimized::unauth::{
    ClaimBundle, DeltaMode, RowKey, UnauthStore, block_grid_indices, logical_offset_to_grid_index,
};
use paper_a_optimized::{
    Fp, build_coded_params, build_shard_registry_deterministic, compute_initial_rows_option_b,
    decode_value, encode_value,
};
use std::collections::HashSet;

fn build_params() -> paper_a_optimized::CodedArrayParams {
    let segment_len: usize = 8;
    let s: usize = 4;
    let shard_count: usize = 4;
    let n_grid = segment_len;
    let shard_registry = build_shard_registry_deterministic(shard_count, s, n_grid);
    let party_registry_indices = vec![(0..shard_count).collect()];
    build_coded_params(&CodedArrayConfig {
        segment_len,
        s,
        shard_count,
        shard_registry,
        party_registry_indices,
        seed: 7,
    })
}

fn build_large_params() -> paper_a_optimized::CodedArrayParams {
    let segment_len: usize = 128;
    let s: usize = 16;
    let shard_count: usize = 8;
    let n_grid = segment_len;
    let shard_registry = build_shard_registry_deterministic(shard_count, s, n_grid);
    let party_registry_indices = vec![(0..shard_count).collect()];
    build_coded_params(&CodedArrayConfig {
        segment_len,
        s,
        shard_count,
        shard_registry,
        party_registry_indices,
        seed: 17,
    })
}

fn exact_claim(params: &paper_a_optimized::CodedArrayParams, registry_index: usize) -> ClaimBundle {
    ClaimBundle::exact_for_registry_index(params, registry_index)
        .expect("exact claim should exist for registry index")
}

fn build_segment_values(params: &paper_a_optimized::CodedArrayParams, seg: usize) -> Vec<[u8; 32]> {
    (0..params.n_grid)
        .map(|idx| encode_value(Fp::from((seg * 100 + idx + 1) as u64)))
        .collect()
}

fn collect_semantic_values(store: &UnauthStore) -> Vec<[u8; 32]> {
    (0..store.len())
        .map(|idx| store.read(idx).expect("read semantic value"))
        .collect()
}

fn semantic_values_to_raw_grid(
    n_grid: usize,
    _layout_s: usize,
    semantic_values: &[[u8; 32]],
) -> Vec<[u8; 32]> {
    let zero = encode_value(Fp::from(0u64));
    let mut raw = vec![zero; n_grid];
    for (logical_offset, &value) in semantic_values.iter().enumerate() {
        raw[logical_offset] = value;
    }
    raw
}

fn exact_store_matching_actual_rows(
    actual: &UnauthStore,
    semantic_values: &[[u8; 32]],
) -> UnauthStore {
    let params = actual.params.clone();
    assert_eq!(semantic_values.len(), actual.len() as usize);
    let row_label_index_by_bytes: std::collections::HashMap<_, _> = params
        .row_labels
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, row_label)| (encode_value(row_label), idx))
        .collect();
    let mut claims_by_registry = std::collections::BTreeMap::<usize, Vec<usize>>::new();
    for row_key in actual.coded.row_keys() {
        let registry_index = *params
            .shard_registry_by_key
            .get(&row_key.owner)
            .expect("row owner in registry");
        let row_label_index = *row_label_index_by_bytes
            .get(&row_key.row_label_bytes)
            .expect("row label in params");
        claims_by_registry
            .entry(registry_index)
            .or_default()
            .push(row_label_index);
    }
    for shard_key in actual.coded.shard_keys() {
        let registry_index = *params
            .shard_registry_by_key
            .get(shard_key)
            .expect("shard owner in registry");
        claims_by_registry.entry(registry_index).or_default();
    }
    let claims: Vec<_> = claims_by_registry
        .into_iter()
        .map(|(registry_index, mut row_label_indices)| {
            row_label_indices.sort_unstable();
            row_label_indices.dedup();
            assert_eq!(
                row_label_indices,
                params.row_label_indices_for_registry_index(registry_index)
            );
            exact_claim(&params, registry_index)
        })
        .collect();
    let mut store = if claims.is_empty() {
        UnauthStore::new(params.clone())
    } else {
        UnauthStore::new_with_claims(params.clone(), claims.clone())
    };
    if claims.is_empty() {
        store
            .extend_with_segments(actual.len(), |seg| {
                let start = seg * params.segment_len;
                let end = semantic_values.len().min(start + params.segment_len);
                semantic_values_to_raw_grid(params.n_grid, params.s, &semantic_values[start..end])
            })
            .expect("materialize exact store");
    } else {
        store
            .extend_with_segments_for_claims(actual.len(), &claims, |seg| {
                let start = seg * params.segment_len;
                let end = semantic_values.len().min(start + params.segment_len);
                semantic_values_to_raw_grid(params.n_grid, params.s, &semantic_values[start..end])
            })
            .expect("materialize exact claim-bound store");
    }
    store
}

fn row_mismatches(actual: &UnauthStore, exact: &UnauthStore) -> Vec<String> {
    let row_label_index_by_bytes: std::collections::HashMap<_, _> = actual
        .params
        .row_labels
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, row_label)| (encode_value(row_label), idx))
        .collect();
    let mut keys: Vec<_> = actual.coded.row_keys().copied().collect();
    for row_key in exact.coded.row_keys() {
        if !actual.coded.contains_row(row_key) {
            keys.push(*row_key);
        }
    }
    let mut mismatches = Vec::new();
    for row_key in keys {
        if actual
            .coded
            .clone_row_family_segments(&actual.params, &row_key)
            == exact
                .coded
                .clone_row_family_segments(&exact.params, &row_key)
        {
            continue;
        }
        let registry_index = actual
            .params
            .shard_registry_by_key
            .get(&row_key.owner)
            .copied()
            .map(|idx| idx.to_string())
            .unwrap_or_else(|| "?".to_string());
        let row_label_index = row_label_index_by_bytes
            .get(&row_key.row_label_bytes)
            .copied()
            .map(|idx| idx.to_string())
            .unwrap_or_else(|| "?".to_string());
        mismatches.push(format!(
            "registry {} row {}",
            registry_index, row_label_index
        ));
    }
    mismatches.sort();
    mismatches.dedup();
    mismatches
}

fn assert_rows_match_expected(
    params: &paper_a_optimized::CodedArrayParams,
    store: &UnauthStore,
    expected_rows: &[Vec<Fp>],
    claimed_indices: &[usize],
    expected_segments: usize,
) {
    assert_eq!(
        store.coded.row_family_count(),
        claimed_indices.len() * params.rows_per_shard
    );
    for &idx in claimed_indices {
        let entry = params.shard_registry[idx];
        for row_label_idx in params.row_label_indices_for_registry_index(idx) {
            let row_key = RowKey::new(entry.key, params.row_labels[row_label_idx]);
            let stored_segments = store
                .coded
                .clone_row_family_segments(params, &row_key)
                .expect("missing maintained row");
            assert_eq!(stored_segments.len(), expected_segments);
            assert_eq!(
                stored_segments[expected_segments - 1],
                expected_rows[row_label_idx]
            );
        }
    }
}

#[test]
fn extend_materializes_rows_consistent_with_same_segment_polynomial() {
    let params = build_params();
    let segment_values = build_segment_values(&params, 0);
    let expected_rows = compute_initial_rows_option_b(&params, &segment_values);
    let semantic_z_grid: HashSet<_> = (0..params.n_grid)
        .map(|idx| encode_value(params.column_coordinate_for_grid_index(idx)))
        .collect();

    assert_eq!(
        params.row_labels.len(),
        params.shard_registry.len() * params.rows_per_shard
    );
    assert_eq!(params.row_eval_points.len(), params.need);
    assert_eq!(params.rows_per_shard, 2);
    assert_eq!(params.row_degree(), params.need - 1);
    assert!(
        params
            .row_eval_points
            .iter()
            .all(|point| !semantic_z_grid.contains(&encode_value(*point)))
    );

    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments(params.segment_len as u64, |_| segment_values.clone())
        .expect("extend");

    assert_eq!(
        store.coded.shard_family_count(),
        params.shard_registry.len()
    );
    let claimed_indices: Vec<_> = (0..params.shard_registry.len()).collect();
    assert_rows_match_expected(&params, &store, &expected_rows, &claimed_indices, 1);
}

#[test]
fn filtered_extend_only_materializes_rows_for_claimed_columns() {
    let params = build_params();
    let assigned_indices = vec![1usize, 3usize];
    let segment_values = build_segment_values(&params, 0);
    let expected_rows = compute_initial_rows_option_b(&params, &segment_values);

    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments_filtered(params.segment_len as u64, &assigned_indices, |_| {
            segment_values.clone()
        })
        .expect("filtered extend");

    assert_eq!(store.coded.shard_family_count(), assigned_indices.len());
    assert_rows_match_expected(&params, &store, &expected_rows, &assigned_indices, 1);

    let claimed_keys: HashSet<_> = assigned_indices
        .iter()
        .map(|&idx| params.shard_registry[idx].key)
        .collect();

    for row_key in store.coded.row_keys() {
        let stored_segments = store
            .coded
            .clone_row_family_segments(&params, row_key)
            .expect("missing maintained row");
        assert!(claimed_keys.contains(&row_key.owner));
        assert_eq!(stored_segments.len(), 1);
    }
}

#[test]
fn claim_bound_extend_only_materializes_local_shards_and_rows() {
    let params = build_params();
    let claims = vec![exact_claim(&params, 1), exact_claim(&params, 3)];
    let segment_values = build_segment_values(&params, 0);
    let expected_rows = compute_initial_rows_option_b(&params, &segment_values);

    let mut store = UnauthStore::new_with_claims(params.clone(), claims.clone());
    store
        .extend_with_segments_for_claims(params.segment_len as u64, &claims, |_| {
            segment_values.clone()
        })
        .expect("claim-bound extend");

    let claimed_indices: Vec<_> = claims.iter().map(|claim| claim.registry_index).collect();
    assert_eq!(store.coded.shard_family_count(), claimed_indices.len());
    assert_rows_match_expected(&params, &store, &expected_rows, &claimed_indices, 1);

    let claimed_keys: HashSet<_> = claimed_indices
        .iter()
        .map(|&idx| params.shard_registry[idx].key)
        .collect();
    for shard_key in store.coded.shard_keys() {
        assert!(claimed_keys.contains(shard_key));
    }
    for row_key in store.coded.row_keys() {
        let stored_segments = store
            .coded
            .clone_row_family_segments(&params, row_key)
            .expect("missing maintained row");
        assert!(claimed_keys.contains(&row_key.owner));
        assert_eq!(stored_segments.len(), 1);
    }
}

#[test]
fn semantic_apply_deltas_updates_rows_consistent_with_rematerialization() {
    let params = build_params();
    let mut segment_values = build_segment_values(&params, 0);
    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments(params.segment_len as u64, |_| segment_values.clone())
        .expect("extend");

    let deltas = [(1u64, Fp::from(11u64)), (5u64, Fp::from(7u64))];
    store
        .apply_deltas(&deltas, DeltaMode::Aggregate)
        .expect("apply semantic deltas");

    for &(logical_idx, delta) in &deltas {
        let logical_offset = logical_idx as usize;
        let grid_index = logical_offset_to_grid_index(&params, logical_offset);
        let updated = decode_value(segment_values[grid_index]) + delta;
        segment_values[grid_index] = encode_value(updated);
        let read_value = decode_value(store.read(logical_idx).expect("read updated value"));
        assert_eq!(read_value, updated);
    }

    let expected_rows = compute_initial_rows_option_b(&params, &segment_values);
    let claimed_indices: Vec<_> = (0..params.shard_registry.len()).collect();
    assert_rows_match_expected(&params, &store, &expected_rows, &claimed_indices, 1);
}

#[test]
fn raw_apply_deltas_updates_rows_consistent_with_rematerialization() {
    let params = build_params();
    let mut segment_values = build_segment_values(&params, 0);
    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments(params.segment_len as u64, |_| segment_values.clone())
        .expect("extend");

    let deltas = [(0u64, Fp::from(3u64)), (6u64, Fp::from(13u64))];
    store
        .apply_deltas_raw(&deltas, DeltaMode::Aggregate)
        .expect("apply raw deltas");

    for &(raw_idx, delta) in &deltas {
        let grid_index = raw_idx as usize;
        let updated = decode_value(segment_values[grid_index]) + delta;
        segment_values[grid_index] = encode_value(updated);
        let read_value = decode_value(store.read_raw(raw_idx).expect("read updated raw value"));
        assert_eq!(read_value, updated);
    }

    let expected_rows = compute_initial_rows_option_b(&params, &segment_values);
    let claimed_indices: Vec<_> = (0..params.shard_registry.len()).collect();
    assert_rows_match_expected(&params, &store, &expected_rows, &claimed_indices, 1);
}

#[test]
fn raw_apply_many_deltas_updates_rows_consistent_with_rematerialization() {
    let params = build_large_params();
    let mut segment_values = build_segment_values(&params, 0);
    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments(params.segment_len as u64, |_| segment_values.clone())
        .expect("extend");

    let mut deltas = Vec::new();
    for round in 0..96u64 {
        let raw_idx = (round * 7) % params.segment_len as u64;
        deltas.push((raw_idx, Fp::from(round + 3)));
    }
    deltas.push((14, Fp::from(99u64)));
    store
        .apply_deltas_raw(&deltas, DeltaMode::Aggregate)
        .expect("apply raw deltas");

    for &(raw_idx, delta) in &deltas {
        let grid_index = raw_idx as usize;
        let updated = decode_value(segment_values[grid_index]) + delta;
        segment_values[grid_index] = encode_value(updated);
    }

    for raw_idx in [0u64, 7, 14, 63, 126] {
        let expected = decode_value(segment_values[raw_idx as usize]);
        let read_value = decode_value(store.read_raw(raw_idx).expect("read updated raw value"));
        assert_eq!(read_value, expected);
    }

    let expected_rows = compute_initial_rows_option_b(&params, &segment_values);
    let claimed_indices: Vec<_> = (0..params.shard_registry.len()).collect();
    assert_rows_match_expected(&params, &store, &expected_rows, &claimed_indices, 1);
}

#[test]
fn interpolate_fft_mode_updates_rows_consistent_with_rematerialization() {
    let params = build_params();
    let mut segment_values = build_segment_values(&params, 0);
    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments(params.segment_len as u64, |_| segment_values.clone())
        .expect("extend");

    let mut deltas = Vec::new();
    for round in 0..72u64 {
        let logical_idx = round % params.segment_len as u64;
        let delta = Fp::from(round + 1);
        deltas.push((logical_idx, delta));
    }
    store
        .apply_deltas(&deltas, DeltaMode::InterpolateFFT)
        .expect("apply fft-mode deltas");

    for &(logical_idx, delta) in &deltas {
        let logical_offset = logical_idx as usize;
        let grid_index = logical_offset_to_grid_index(&params, logical_offset);
        let updated = decode_value(segment_values[grid_index]) + delta;
        segment_values[grid_index] = encode_value(updated);
    }

    for logical_idx in 0..params.segment_len as u64 {
        let logical_offset = logical_idx as usize;
        let grid_index = logical_offset_to_grid_index(&params, logical_offset);
        let expected = decode_value(segment_values[grid_index]);
        let read_value = decode_value(store.read(logical_idx).expect("read updated value"));
        assert_eq!(read_value, expected);
    }

    let expected_rows = compute_initial_rows_option_b(&params, &segment_values);
    let claimed_indices: Vec<_> = (0..params.shard_registry.len()).collect();
    assert_rows_match_expected(&params, &store, &expected_rows, &claimed_indices, 1);
}

#[test]
fn aligned_block_update_updates_values_and_rows_consistent_with_rematerialization() {
    let params = build_params();
    let mut segment_values = build_segment_values(&params, 0);
    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments(params.segment_len as u64, |_| segment_values.clone())
        .expect("extend");

    let deltas = [
        (0usize, Fp::from(11u64)),
        (1usize, Fp::from(5u64)),
        (2usize, Fp::from(7u64)),
        (3usize, Fp::from(13u64)),
    ];
    store
        .apply_block_deltas(0, &deltas, DeltaMode::Aggregate)
        .expect("apply aligned block deltas");

    let grid_indices = block_grid_indices(&params, 0);
    for &(offset_in_block, delta) in &deltas {
        let grid_index = grid_indices[offset_in_block];
        let updated = decode_value(segment_values[grid_index]) + delta;
        segment_values[grid_index] = encode_value(updated);
        let read_value = decode_value(store.read(grid_index as u64).expect("read updated value"));
        assert_eq!(read_value, updated);
    }

    let expected_rows = compute_initial_rows_option_b(&params, &segment_values);
    let claimed_indices: Vec<_> = (0..params.shard_registry.len()).collect();
    assert_rows_match_expected(&params, &store, &expected_rows, &claimed_indices, 1);
}

#[test]
fn sparse_aligned_block_update_only_changes_selected_slots() {
    let params = build_params();
    let mut segment_values = build_segment_values(&params, 0);
    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments(params.segment_len as u64, |_| segment_values.clone())
        .expect("extend");

    let deltas = [(1usize, Fp::from(9u64)), (3usize, Fp::from(4u64))];
    store
        .apply_block_deltas(0, &deltas, DeltaMode::Aggregate)
        .expect("apply sparse block deltas");

    let grid_indices = block_grid_indices(&params, 0);
    for offset_in_block in 0..params.s {
        let grid_index = grid_indices[offset_in_block];
        let mut expected = decode_value(segment_values[grid_index]);
        for &(delta_offset, delta) in &deltas {
            if offset_in_block == delta_offset {
                expected += delta;
                segment_values[grid_index] = encode_value(expected);
            }
        }
        let read_value = decode_value(store.read(grid_index as u64).expect("read updated value"));
        assert_eq!(read_value, expected);
    }

    let expected_rows = compute_initial_rows_option_b(&params, &segment_values);
    let claimed_indices: Vec<_> = (0..params.shard_registry.len()).collect();
    assert_rows_match_expected(&params, &store, &expected_rows, &claimed_indices, 1);
}

#[test]
fn aligned_block_update_accepts_last_slot_when_segment_len_is_grid_size() {
    let params = build_params();
    let mut segment_values = build_segment_values(&params, 0);
    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments(params.segment_len as u64, |_| segment_values.clone())
        .expect("extend");

    let delta = Fp::from(1u64);
    store
        .apply_block_deltas(1, &[(3usize, Fp::from(1u64))], DeltaMode::Aggregate)
        .expect("last slot is a semantic value");

    let grid_index = block_grid_indices(&params, 1)[3];
    let expected = decode_value(segment_values[grid_index]) + delta;
    segment_values[grid_index] = encode_value(expected);
    let actual = decode_value(store.read(grid_index as u64).expect("read updated value"));
    assert_eq!(actual, expected);

    let expected_rows = compute_initial_rows_option_b(&params, &segment_values);
    let claimed_indices: Vec<_> = (0..params.shard_registry.len()).collect();
    assert_rows_match_expected(&params, &store, &expected_rows, &claimed_indices, 1);
}

#[test]
fn batched_row_reads_match_individual_row_reads_on_aligned_block_coordinates() {
    let params = build_params();
    let segment_values = build_segment_values(&params, 0);
    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments(params.segment_len as u64, |_| segment_values.clone())
        .expect("extend");

    let z_evals = vec![
        params.column_coordinate_for_grid_index(0),
        params.column_coordinate_for_grid_index(1),
    ];

    let batched = match store.handle_request(UnauthRequest::ReadRowsBatchAtZ {
        seg: 0,
        z_evals: z_evals.clone(),
    }) {
        UnauthResponse::ReadRowsBatchAtZ { rows } => rows,
        other => panic!("unexpected batch row response: {:?}", other),
    };

    assert_eq!(batched.len(), z_evals.len());
    for (batch_pos, z_eval) in z_evals.into_iter().enumerate() {
        let individual = match store.handle_request(UnauthRequest::ReadRowsAtZ { seg: 0, z_eval }) {
            UnauthResponse::ReadRowsAtZ { rows } => rows,
            other => panic!("unexpected individual row response: {:?}", other),
        };
        assert_eq!(batched[batch_pos], individual);
    }
}

#[test]
fn structured_block_update_matches_batched_point_reference() {
    let params = build_params();
    let segment_values = build_segment_values(&params, 0);
    let mut block_store = UnauthStore::new(params.clone());
    block_store
        .extend_with_segments(params.segment_len as u64, |_| segment_values.clone())
        .expect("extend block store");
    let mut point_store = UnauthStore::new(params.clone());
    point_store
        .extend_with_segments(params.segment_len as u64, |_| segment_values.clone())
        .expect("extend point store");

    let block = 0u64;
    let block_deltas = [
        (0usize, Fp::from(11u64)),
        (1usize, Fp::from(5u64)),
        (2usize, Fp::from(7u64)),
        (3usize, Fp::from(13u64)),
    ];
    let point_deltas: Vec<(u64, Fp)> = block_deltas
        .iter()
        .map(|&(offset, delta)| {
            let grid_index = block_grid_indices(&params, block as usize)[offset];
            (grid_index as u64, delta)
        })
        .collect();

    block_store
        .apply_block_deltas(block, &block_deltas, DeltaMode::Aggregate)
        .expect("structured block update");
    point_store
        .apply_deltas(&point_deltas, DeltaMode::Aggregate)
        .expect("batched point update");

    assert!(
        block_store
            .coded
            .equals_for_tests(&params, &point_store.coded)
    );
}

// This compares the array-wrapper block update against the corresponding set of
// stable point updates across a kappa-double/kappa-halve cycle.
#[test]
fn client_array_view_block_update_matches_batched_points_after_kappa_double_and_kappa_halve() {
    let segment_len: usize = 32;
    let source_s: usize = 16;
    let kappa_doubled_s: usize = 8;
    let shard_count: usize = 8;
    let n_grid = segment_len;
    let shard_registry = build_shard_registry_deterministic(shard_count, source_s, n_grid);
    let party_registry_indices = vec![(0..shard_count).collect()];
    let source_params = build_coded_params(&CodedArrayConfig {
        segment_len,
        s: source_s,
        shard_count,
        shard_registry,
        party_registry_indices,
        seed: 91,
    });
    let segment_values = vec![encode_value(Fp::from(0u64)); source_params.n_grid];

    let mut block_store = UnauthStore::new(source_params.clone());
    block_store
        .extend_with_segments(source_params.segment_len as u64, |_| segment_values.clone())
        .expect("extend block store");
    let mut point_store = UnauthStore::new(source_params.clone());
    point_store
        .extend_with_segments(source_params.segment_len as u64, |_| segment_values.clone())
        .expect("extend point store");

    block_store
        .migrate(kappa_doubled_s, &[])
        .expect("kappa-double block store");

    let block = 1u64;
    let block_deltas = [
        (1usize, Fp::from(0x66u64)),
        (4usize, Fp::from(0x77u64)),
        (6usize, Fp::from(0x88u64)),
    ];
    let kappa_doubled_params =
        paper_a_optimized::unauth::build_migrated_params(&source_params, kappa_doubled_s)
            .expect("kappa-double params");
    let kappa_doubled_block =
        paper_a_optimized::unauth::semantic_block_position(&kappa_doubled_params, block);
    let kappa_doubled_grid_indices = paper_a_optimized::unauth::block_grid_indices(
        &kappa_doubled_params,
        kappa_doubled_block.block_in_segment,
    );
    let point_deltas: Vec<(u64, Fp)> = block_deltas
        .iter()
        .map(|&(offset, delta)| {
            let grid_index = kappa_doubled_grid_indices[offset];
            (
                (kappa_doubled_block.segment * kappa_doubled_params.segment_len + grid_index)
                    as u64,
                delta,
            )
        })
        .collect();

    block_store
        .apply_block_deltas(block, &block_deltas, DeltaMode::Aggregate)
        .expect("structured block update");
    point_store
        .apply_deltas(&point_deltas, DeltaMode::Aggregate)
        .expect("batched point update");

    block_store
        .migrate(source_s, &[])
        .expect("kappa-halve block store");
    let block_values = collect_semantic_values(&block_store);
    let point_values = collect_semantic_values(&point_store);
    assert_eq!(block_values, point_values);
    assert_eq!(
        block_store.coded.as_nested_shards_for_tests(&source_params),
        point_store.coded.as_nested_shards_for_tests(&source_params)
    );

    let block_exact = exact_store_matching_actual_rows(&block_store, &block_values);
    let point_exact = exact_store_matching_actual_rows(&point_store, &point_values);
    assert!(
        row_mismatches(&block_store, &block_exact).is_empty(),
        "kappa-halve block store rows no longer match fresh materialization"
    );
    assert!(
        row_mismatches(&point_store, &point_exact).is_empty(),
        "point-reference store rows no longer match fresh materialization"
    );
}
