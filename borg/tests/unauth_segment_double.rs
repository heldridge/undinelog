use paper_a_optimized::unauth::{
    ClaimBundle, DeltaMode, UnauthDecodeParams, UnauthStore, aligned_block_indices,
    build_segment_double_params, logical_blocks_per_segment, logical_offset_to_alpha_for_segment,
    logical_offset_to_grid_index, public_index_to_segment, public_values_per_segment,
    read_batch_from_store, read_batch_from_store_raw, shard_value_from_store,
};
use paper_a_optimized::{
    CodedArrayConfig, CodedArrayParams, Fp, ShardShare, build_coded_params,
    build_shard_registry_deterministic, decode_value, encode_value, fp_zero,
};

fn build_params(n_grid: usize, s: usize, shard_count: usize) -> CodedArrayParams {
    let segment_len = n_grid;
    let shard_registry = build_shard_registry_deterministic(shard_count, s, n_grid);
    build_coded_params(&CodedArrayConfig {
        segment_len,
        s,
        shard_count,
        shard_registry,
        party_registry_indices: vec![(0..shard_count).collect()],
        seed: 17,
    })
}

fn segment_values(params: &CodedArrayParams, seg: usize) -> Vec<paper_a_optimized::Value256> {
    (0..params.n_grid)
        .map(|offset| encode_value(Fp::from((1 + seg * 100 + offset) as u64)))
        .collect()
}

fn raw_value(store: &UnauthStore, seg: usize, grid_index: usize) -> Fp {
    let idx = seg as u64 * public_values_per_segment(&store.params) as u64 + grid_index as u64;
    decode_value(store.read_raw(idx).expect("decode raw grid value"))
}

fn raw_value_from_coded(store: &UnauthStore, seg: usize, grid_index: usize) -> Fp {
    let idx = seg as u64 * public_values_per_segment(&store.params) as u64 + grid_index as u64;
    let values = read_batch_from_store_raw(&store.params, &store.coded, &[idx], None)
        .expect("decode raw grid value from coded state");
    decode_value(values[0])
}

fn collect_semantic_values(store: &UnauthStore) -> Vec<paper_a_optimized::Value256> {
    (0..store.len())
        .map(|idx| store.read(idx).expect("read semantic value"))
        .collect()
}

fn semantic_values_to_raw_grid(
    params: &CodedArrayParams,
    semantic_values: &[paper_a_optimized::Value256],
) -> Vec<paper_a_optimized::Value256> {
    let mut raw = vec![encode_value(fp_zero()); params.n_grid];
    for (logical_offset, &value) in semantic_values.iter().enumerate() {
        let grid_index = logical_offset_to_grid_index(params, logical_offset);
        raw[grid_index] = value;
    }
    raw
}

fn exact_store_matching_actual_rows(
    actual: &UnauthStore,
    semantic_values: &[paper_a_optimized::Value256],
) -> UnauthStore {
    let params = actual.params.clone();
    assert_eq!(semantic_values.len(), actual.len() as usize);
    let mut store = UnauthStore::new(params.clone());
    let values_per_segment = public_values_per_segment(&params);
    store
        .extend_with_segments(actual.len(), |seg| {
            let start = seg * values_per_segment;
            let end = semantic_values.len().min(start + values_per_segment);
            semantic_values_to_raw_grid(&params, &semantic_values[start..end])
        })
        .expect("materialize exact store");
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

#[test]
fn segment_double_params_keep_same_cosets() {
    let source = build_params(8, 4, 8);
    let target = build_segment_double_params(&source).expect("target params");

    assert_eq!(target.n_grid, 2 * source.n_grid);
    assert_eq!(target.segment_len, target.n_grid);
    assert_eq!(target.s, source.s);
    assert_eq!(target.need, 2 * source.need);
    assert_eq!(target.h, source.h);
    assert_eq!(target.shard_registry.len(), source.shard_registry.len());
    for (left, right) in source
        .shard_registry
        .iter()
        .zip(target.shard_registry.iter())
    {
        assert_eq!(left.a, right.a);
        assert_eq!(left.z, right.z);
        assert_eq!(left.key, right.key);
    }
}

#[test]
fn segment_double_preserves_public_reads_for_live_indices() {
    let params = build_params(8, 4, 8);
    let source_segments = 4usize;
    let len = (source_segments * public_values_per_segment(&params)) as u64;
    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments(len, |seg| segment_values(&params, seg))
        .expect("extend");
    let expected: Vec<_> = (0..len)
        .map(|idx| decode_value(store.read(idx).expect("source read")))
        .collect();

    store.segment_double().expect("segment-double");

    assert_eq!(store.len(), len);
    assert_eq!(store.params.n_grid, 2 * params.n_grid);
    assert_eq!(store.params.segment_len, store.params.n_grid);
    for (idx, expected_value) in expected.into_iter().enumerate() {
        let actual = decode_value(store.read(idx as u64).expect("target read"));
        assert_eq!(actual, expected_value, "public index {idx}");
    }
}

#[test]
fn segment_double_preserves_claim_bound_public_reads() {
    let n_grid = 32usize;
    let s = 16usize;
    let shard_count = 8usize;
    let shard_registry = build_shard_registry_deterministic(shard_count, s, n_grid);
    let params = build_coded_params(&CodedArrayConfig {
        segment_len: n_grid,
        s,
        shard_count,
        shard_registry,
        party_registry_indices: (0..shard_count).map(|idx| vec![idx]).collect(),
        seed: 23,
    });
    let len = 8u64;
    let mut full = UnauthStore::new(params.clone());
    full.extend_with_segments(len, |seg| {
        (0..params.n_grid)
            .map(|offset| encode_value(Fp::from((1 + seg * 100 + offset) as u64)))
            .collect()
    })
    .expect("extend");

    let expected: Vec<_> = (0..len)
        .map(|idx| decode_value(full.read(idx).expect("source read")))
        .collect();
    let mut claim_stores = Vec::new();
    for registry_index in 0..params.shard_registry.len() {
        let claim =
            ClaimBundle::exact_for_registry_index(&params, registry_index).expect("exact claim");
        let mut claim_store = UnauthStore::new_with_claims(params.clone(), vec![claim]);
        claim_store
            .extend_with_segments(len, |seg| {
                (0..params.n_grid)
                    .map(|offset| encode_value(Fp::from((1 + seg * 100 + offset) as u64)))
                    .collect()
            })
            .expect("extend claim store");
        claim_store.segment_double().expect("claim-bound merge");
        claim_stores.push(claim_store);
    }

    let target_params = claim_stores[0].params.clone();
    for idx in 0..len {
        let mut shares = Vec::new();
        let (seg, logical_offset) = public_index_to_segment(&target_params, idx);
        let alpha = logical_offset_to_alpha_for_segment(&target_params, seg, logical_offset);
        for claim_store in &claim_stores {
            for registry_index in 0..target_params.shard_registry.len() {
                let entry = target_params.shard_registry[registry_index];
                if let Some(value) =
                    shard_value_from_store(&target_params, &claim_store.coded, entry, seg, alpha)
                {
                    shares.push(ShardShare::new(registry_index, Some(value)));
                }
            }
        }
        let actual = target_params
            .decode_index_from_shares(idx, &shares)
            .expect("decode target public value");
        assert_eq!(
            actual, expected[idx as usize],
            "claim-bound public index {idx}"
        );
    }
}

#[test]
fn segment_double_concatenates_raw_grid_boundary() {
    let params = build_params(8, 4, 8);
    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments((2 * params.n_grid) as u64, |seg| {
            segment_values(&params, seg)
        })
        .expect("extend");
    let left_last = raw_value(&store, 0, params.n_grid - 1);
    let right_first = raw_value(&store, 1, 0);

    store.segment_double().expect("segment-double");

    let left_last_grid = logical_offset_to_grid_index(&store.params, params.n_grid - 1);
    let right_first_grid = logical_offset_to_grid_index(&store.params, params.n_grid);
    assert_eq!(raw_value(&store, 0, left_last_grid), left_last);
    assert_eq!(raw_value(&store, 0, right_first_grid), right_first);
}

#[test]
fn large_interpolate_deltas_on_merged_coset_update_shards_and_rows() {
    let params = build_params(128, 16, 16);
    let source_segments = 4usize;
    let len = (source_segments * public_values_per_segment(&params)) as u64;
    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments(len, |seg| segment_values(&params, seg))
        .expect("extend");
    store.segment_double().expect("segment-double");

    let target_segment = 1u64;
    let target_base = target_segment * public_values_per_segment(&store.params) as u64;
    let mut deltas = Vec::new();
    let mut expected = collect_semantic_values(&store);
    for logical_offset in 0..96u64 {
        let idx = target_base + logical_offset;
        let delta = Fp::from(1000 + logical_offset);
        deltas.push((idx, delta));
        let updated = decode_value(expected[idx as usize]) + delta;
        expected[idx as usize] = encode_value(updated);
    }

    store
        .apply_deltas(&deltas, DeltaMode::InterpolateFFT)
        .expect("large merged-coset delta batch");

    for (idx, expected_value) in expected.iter().copied().enumerate() {
        let actual = store.read(idx as u64).expect("read after coset update");
        assert_eq!(actual, expected_value, "public index {idx}");
    }

    let exact = exact_store_matching_actual_rows(&store, &expected);
    let mismatches = row_mismatches(&store, &exact);
    assert!(
        mismatches.is_empty(),
        "row state diverged after merged-coset FFT update: {mismatches:?}"
    );
}

#[test]
fn segment_double_defaults_missing_right_segment() {
    let params = build_params(8, 4, 8);
    let source_segments = 3usize;
    let len = (source_segments * public_values_per_segment(&params)) as u64;
    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments(len, |seg| segment_values(&params, seg))
        .expect("extend");
    let expected: Vec<_> = (0..len)
        .map(|idx| decode_value(store.read(idx).expect("source read")))
        .collect();

    store.segment_double().expect("segment-double");

    assert_eq!(store.segments(), 2);
    for (idx, expected_value) in expected.into_iter().enumerate() {
        let actual = decode_value(store.read(idx as u64).expect("target read"));
        assert_eq!(actual, expected_value, "public index {idx}");
    }
    for logical_offset in params.n_grid..store.params.n_grid {
        let grid_index = logical_offset_to_grid_index(&store.params, logical_offset);
        assert_eq!(raw_value(&store, 1, grid_index), fp_zero());
    }
}

#[test]
fn segment_double_twice_preserves_public_reads_and_tail_padding() {
    let params = build_params(8, 4, 8);
    let source_segments = 5usize;
    let source_values_per_segment = public_values_per_segment(&params);
    let len = (source_segments * source_values_per_segment) as u64;
    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments(len, |seg| segment_values(&params, seg))
        .expect("extend");

    let expected: Vec<_> = (0..len)
        .map(|idx| decode_value(store.read(idx).expect("source read")))
        .collect();
    let initial_by_source_segment: Vec<_> = (0..source_segments)
        .map(|seg| {
            (0..source_values_per_segment)
                .map(|offset| raw_value(&store, seg, params.logical_grid_index(offset)))
                .collect::<Vec<_>>()
        })
        .collect();

    store.segment_double().expect("first segment-double");
    assert_eq!(store.len(), len);
    assert_eq!(store.segments(), 3);
    for (idx, expected_value) in expected.iter().copied().enumerate() {
        let actual = decode_value(store.read(idx as u64).expect("read after first merge"));
        assert_eq!(
            actual, expected_value,
            "public index {idx} after first merge"
        );
    }

    store.segment_double().expect("second segment-double");

    assert_eq!(store.len(), len);
    assert_eq!(store.segments(), 2);
    assert_eq!(store.params.n_grid, 4 * params.n_grid);
    assert_eq!(store.params.segment_len, store.params.n_grid);
    for (idx, expected_value) in expected.iter().copied().enumerate() {
        let actual = decode_value(store.read(idx as u64).expect("read after second merge"));
        assert_eq!(
            actual, expected_value,
            "public index {idx} after second merge"
        );
    }

    for source_seg in 0..4 {
        let final_segment = 0usize;
        let final_logical_base = source_seg * source_values_per_segment;
        for source_offset in 0..source_values_per_segment {
            let final_logical_offset = final_logical_base + source_offset;
            let final_grid_index =
                logical_offset_to_grid_index(&store.params, final_logical_offset);
            assert_eq!(
                raw_value(&store, final_segment, final_grid_index),
                initial_by_source_segment[source_seg][source_offset],
                "source segment {source_seg} offset {source_offset} in first final segment",
            );
        }
    }

    assert_eq!(
        raw_value(
            &store,
            0,
            logical_offset_to_grid_index(&store.params, source_values_per_segment - 1)
        ),
        initial_by_source_segment[0][source_values_per_segment - 1],
    );
    assert_eq!(
        raw_value(
            &store,
            0,
            logical_offset_to_grid_index(&store.params, source_values_per_segment)
        ),
        initial_by_source_segment[1][0],
    );
    assert_eq!(
        raw_value(
            &store,
            0,
            logical_offset_to_grid_index(&store.params, 2 * source_values_per_segment - 1)
        ),
        initial_by_source_segment[1][source_values_per_segment - 1],
    );
    assert_eq!(
        raw_value(
            &store,
            0,
            logical_offset_to_grid_index(&store.params, 2 * source_values_per_segment)
        ),
        initial_by_source_segment[2][0],
    );

    for source_offset in 0..source_values_per_segment {
        let final_grid_index = logical_offset_to_grid_index(&store.params, source_offset);
        assert_eq!(
            raw_value_from_coded(&store, 1, final_grid_index),
            initial_by_source_segment[4][source_offset],
            "odd trailing source segment offset {source_offset}",
        );
    }
    for final_logical_offset in source_values_per_segment..store.params.n_grid {
        let final_grid_index = logical_offset_to_grid_index(&store.params, final_logical_offset);
        assert_eq!(
            raw_value_from_coded(&store, 1, final_grid_index),
            fp_zero(),
            "inactive tail logical offset {final_logical_offset}",
        );
    }
}

#[test]
fn block_write_after_merge_checks_liveness_by_public_logical_offset() {
    let params = build_params(8, 4, 8);
    let source_segments = 3usize;
    let len = (source_segments * public_values_per_segment(&params)) as u64;
    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments(len, |seg| segment_values(&params, seg))
        .expect("extend");
    store.segment_double().expect("segment-double");

    let target_segment = 1u64;
    let target_block = target_segment * logical_blocks_per_segment(&store.params) as u64;
    let block_grid_indices = paper_a_optimized::unauth::block_grid_indices(&store.params, 0);
    let logical_offset = store
        .params
        .logical_offset_for_grid_index(block_grid_indices[3]);
    let public_index =
        target_segment * public_values_per_segment(&store.params) as u64 + logical_offset as u64;
    assert!(public_index < len);
    let before = decode_value(store.read(public_index).expect("read before block write"));

    store
        .apply_block_deltas(target_block, &[(3, Fp::from(9u64))], DeltaMode::Aggregate)
        .expect("block write targets a live logical cell");

    let after = decode_value(store.read(public_index).expect("read after block write"));
    assert_eq!(after, before + Fp::from(9u64));
}

#[test]
fn segment_double_keeps_block_and_local_batch_reads_placement_aware() {
    let params = build_params(8, 4, 8);
    let source_segments = 4usize;
    let len = (source_segments * public_values_per_segment(&params)) as u64;
    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments(len, |seg| segment_values(&params, seg))
        .expect("extend");

    store.segment_double().expect("segment-double");

    let target_blocks_per_segment = logical_blocks_per_segment(&store.params) as u64;
    let block = target_blocks_per_segment;
    let logical_indices = aligned_block_indices(&store.params, block, store.len());
    let before_block = store.read_block(block).expect("read merged block");
    for (offset, idx) in logical_indices.iter().enumerate() {
        let Some(idx) = idx else {
            continue;
        };
        assert_eq!(
            decode_value(before_block[offset]),
            decode_value(store.read(*idx).expect("point read")),
            "block offset {offset} before write",
        );
    }

    store
        .write_block_batch_deltas(
            block,
            &[(0, Fp::from(77u64)), (3, Fp::from(88u64))],
            DeltaMode::Aggregate,
        )
        .expect("write merged block");

    let after_block = store.read_block(block).expect("read updated merged block");
    for (offset, idx) in logical_indices.iter().enumerate() {
        let Some(idx) = idx else {
            continue;
        };
        assert_eq!(
            decode_value(after_block[offset]),
            decode_value(store.read(*idx).expect("point read after block write")),
            "block offset {offset} after write",
        );
    }

    let idxs: Vec<_> = logical_indices.iter().filter_map(|idx| *idx).collect();
    let batch = read_batch_from_store(&store.params, &store.coded, &idxs, None)
        .expect("local batch read after merge");
    for (value, idx) in batch.iter().zip(idxs.iter()) {
        assert_eq!(
            decode_value(*value),
            decode_value(store.read(*idx).expect("point read for local batch")),
            "local batch index {idx}",
        );
    }

    let seg = 1usize;
    let raw_indices: Vec<_> = (0..store.params.s)
        .map(|j| seg as u64 * public_values_per_segment(&store.params) as u64 + j as u64)
        .collect();
    let raw_batch = read_batch_from_store_raw(&store.params, &store.coded, &raw_indices, None)
        .expect("local raw batch read after merge");
    for (value, idx) in raw_batch.iter().zip(raw_indices.iter()) {
        assert_eq!(
            decode_value(*value),
            decode_value(
                store
                    .read_raw(*idx)
                    .expect("raw point read for local batch")
            ),
            "local raw batch index {idx}",
        );
    }
}

#[test]
fn segment_double_primary_repair_uses_merged_rows() {
    let params = build_params(32, 16, 8);
    let source_segments = 4usize;
    let len = (source_segments * public_values_per_segment(&params)) as u64;
    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments(len, |seg| segment_values(&params, seg))
        .expect("extend");
    store
        .apply_deltas(
            &[
                (0, Fp::from(11u64)),
                (params.n_grid as u64 - 1, Fp::from(13u64)),
                (params.n_grid as u64, Fp::from(17u64)),
                (2 * params.n_grid as u64 + 3, Fp::from(19u64)),
            ],
            DeltaMode::Aggregate,
        )
        .expect("write before segment-double");

    store.segment_double().expect("segment-double");

    let target_seg = 1usize;
    let target_entry = store.params.shard_registry[3];
    let expected = store
        .coded
        .clone_shard_family_segments(&store.params, &target_entry.key)
        .expect("target shard family")
        .get(target_seg)
        .expect("target segment")
        .clone();

    let mut damaged = store.clone();
    let removed = damaged
        .coded
        .remove_shard_family_segments(&damaged.params, &target_entry.key);
    assert!(removed.is_some(), "target shard family was present");

    let repaired = damaged
        .repair_primary_shard_from_rows(target_seg, target_entry)
        .expect("repair primary from segment-doubled rows");
    assert_eq!(repaired, expected);
}
