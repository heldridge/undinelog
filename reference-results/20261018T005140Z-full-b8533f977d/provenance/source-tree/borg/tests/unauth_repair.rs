use paper_a_optimized::coded_array::params::CodedArrayConfig;
use paper_a_optimized::unauth::{DeltaMode, UnauthStore, barycentric_eval, precompute_barycentric};
use paper_a_optimized::{
    Fp, build_coded_params, build_shard_registry_deterministic, decode_value, encode_value,
};
use std::time::Instant;

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
        seed: 37,
    })
}

fn build_segment_values(params: &paper_a_optimized::CodedArrayParams, seg: usize) -> Vec<[u8; 32]> {
    (0..params.n_grid)
        .map(|idx| encode_value(Fp::from((seg * 100 + (idx + 1) * 3) as u64)))
        .collect()
}

fn build_updated_store() -> UnauthStore {
    let params = build_params();
    let segment_values = build_segment_values(&params, 0);
    let mut store = UnauthStore::new(params);
    store
        .extend_with_segments(store.params.segment_len as u64, |_| segment_values.clone())
        .expect("extend");
    store
        .apply_deltas(
            &[
                (0u64, Fp::from(11u64)),
                (3u64, Fp::from(5u64)),
                (5u64, Fp::from(7u64)),
            ],
            DeltaMode::Aggregate,
        )
        .expect("apply deltas");
    store
}

fn old_style_repair_row_from_shards(store: &UnauthStore, seg: usize, row_label: Fp) -> Vec<Fp> {
    let params = &store.params;
    let mut z_values = Vec::new();
    let mut row_values = Vec::new();
    for (_registry_index, entry) in params.shard_registry.iter().copied().enumerate() {
        let Some(shard_values) = store.coded.shard_segment(params, &entry.key, seg) else {
            continue;
        };
        let eval_points: Vec<_> = (0..params.s).map(|t| entry.a * params.h[t]).collect();
        let weights = precompute_barycentric(&eval_points);
        z_values.push(entry.z);
        row_values.push(barycentric_eval(&weights, shard_values, row_label));
        if row_values.len() >= params.need {
            break;
        }
    }
    let weights = precompute_barycentric(&z_values);
    params
        .row_eval_points
        .iter()
        .map(|&z_eval| barycentric_eval(&weights, &row_values, z_eval))
        .collect()
}

#[test]
fn primary_shard_repair_from_rows_matches_missing_column_after_updates() {
    let store = build_updated_store();
    let params = store.params.clone();
    let target_entry = params.shard_registry[1];
    let expected = store
        .coded
        .clone_shard_family_segments(&params, &target_entry.key)
        .expect("missing original shard")
        .get(0)
        .expect("missing original segment")
        .clone();

    let mut damaged = store.clone();
    let removed = damaged
        .coded
        .remove_shard_family_segments(&params, &target_entry.key);
    assert!(removed.is_some(), "target shard was not removed");

    let repaired = damaged
        .repair_primary_shard_from_rows(0, target_entry)
        .expect("repair primary shard from rows");
    assert_eq!(repaired, expected);
}

#[test]
fn auxiliary_row_repair_from_shards_matches_missing_row_after_updates() {
    let store = build_updated_store();
    let params = store.params.clone();
    let target_row_label = params.row_labels[0];
    let target_label_bytes = encode_value(target_row_label);
    let expected = store
        .coded
        .row_keys()
        .find_map(|row_key| {
            (row_key.row_label_bytes == target_label_bytes).then(|| {
                store
                    .coded
                    .row_segment(&params, row_key, 0)
                    .expect("missing row segment")
                    .to_vec()
            })
        })
        .expect("missing original row");

    let mut damaged = store.clone();
    damaged
        .coded
        .retain_rows(|row_key| row_key.row_label_bytes != target_label_bytes);
    assert!(
        !damaged
            .coded
            .row_keys()
            .any(|row_key| row_key.row_label_bytes == target_label_bytes)
    );

    let repaired = damaged
        .repair_row_from_shards(0, target_row_label)
        .expect("repair row from primary shards");
    assert_eq!(repaired, expected);

    let non_zero_entries = repaired
        .iter()
        .filter(|value| **value != Fp::from(0u64))
        .count();
    assert!(non_zero_entries > 0);
    assert_ne!(decode_value(encode_value(repaired[0])), Fp::from(0u64));
}

#[test]
fn optimized_row_repair_matches_old_direct_barycentric_reference() {
    let store = build_updated_store();
    let params = store.params.clone();
    let target_row_label = params.row_labels[0];

    let optimized = store
        .repair_row_from_shards(0, target_row_label)
        .expect("optimized row repair");
    let reference = old_style_repair_row_from_shards(&store, 0, target_row_label);

    assert_eq!(optimized, reference);
}

#[test]
#[ignore]
fn compare_optimized_and_old_row_repair_large_control_point() {
    let segment_len = 16_384usize;
    let s = 2_048usize;
    let shard_count = 8usize;
    let n_grid = segment_len;
    let shard_registry = build_shard_registry_deterministic(shard_count, s, n_grid);
    let params = build_coded_params(&CodedArrayConfig {
        segment_len,
        s,
        shard_count,
        shard_registry,
        party_registry_indices: vec![(0..shard_count).collect()],
        seed: 41,
    });

    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments(segment_len as u64, |seg| build_segment_values(&params, seg))
        .expect("extend");

    let target_row_label = params.row_labels[0];

    let optimized_start = Instant::now();
    let optimized = store
        .repair_row_from_shards(0, target_row_label)
        .expect("optimized row repair");
    let optimized_elapsed = optimized_start.elapsed();

    let reference_start = Instant::now();
    let reference = old_style_repair_row_from_shards(&store, 0, target_row_label);
    let reference_elapsed = reference_start.elapsed();

    assert_eq!(optimized, reference);
    println!("optimized row repair: {:?}", optimized_elapsed);
    println!("reference row repair: {:?}", reference_elapsed);
}
