use paper_a_optimized::coded_array::params::CodedArrayConfig;
use paper_a_optimized::unauth::{DeltaMode, UnauthStore, barycentric_eval, precompute_barycentric};
use paper_a_optimized::{
    Fp, build_coded_params, build_shard_registry_deterministic, decode_value, encode_value,
};
use std::time::Instant;

/// Segments in the fixture store. Segment 0 has unit axis scales; the others
/// exercise non-trivial Z-axis and W-axis scales.
const SEGMENTS: usize = 4;

fn build_params() -> paper_a_optimized::CodedArrayParams {
    let kappa: usize = 4;
    let shard_count: usize = 6;
    let shard_registry = build_shard_registry_deterministic(shard_count);
    let party_registry_indices = vec![(0..shard_count).collect()];
    build_coded_params(&CodedArrayConfig {
        kappa,
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
    let segment_len = params.segment_len as u64;
    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments(SEGMENTS as u64 * segment_len, |seg| {
            build_segment_values(&params, seg)
        })
        .expect("extend");
    let mut deltas = Vec::new();
    for seg in 0..SEGMENTS as u64 {
        deltas.push((seg * segment_len, Fp::from(11 + seg)));
        deltas.push((seg * segment_len + 3, Fp::from(5 + seg)));
        deltas.push((seg * segment_len + 5, Fp::from(7 + seg)));
    }
    store
        .apply_deltas(&deltas, DeltaMode::Aggregate)
        .expect("apply deltas");
    store
}

/// Repair the auxiliary row labeled `row_label` with plain barycentric
/// interpolation: evaluate `need` local primary columns at the row label over
/// the segment's W-axis, then interpolate those samples over the column
/// identifiers and evaluate on the segment's Z-axis.
fn reference_repair_row_from_shards(store: &UnauthStore, seg: usize, row_label: Fp) -> Vec<Fp> {
    let params = &store.params;
    let axes = params.segment_axes(seg);
    let w_axis_points: Vec<_> = params
        .axis
        .iter()
        .map(|&point| axes.rho_w * point)
        .collect();
    let column_weights = precompute_barycentric(&w_axis_points);
    let mut z_values = Vec::new();
    let mut row_values = Vec::new();
    for entry in params.shard_registry.iter() {
        let Some(shard_values) = store.coded.shard_segment(params, &entry.key, seg) else {
            continue;
        };
        z_values.push(entry.z);
        row_values.push(barycentric_eval(&column_weights, shard_values, row_label));
        if row_values.len() >= params.need {
            break;
        }
    }
    let weights = precompute_barycentric(&z_values);
    params
        .axis
        .iter()
        .map(|&point| barycentric_eval(&weights, &row_values, axes.rho_z * point))
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
        .expect("missing original shard");
    assert_eq!(expected.len(), SEGMENTS);

    let mut damaged = store.clone();
    let removed = damaged
        .coded
        .remove_shard_family_segments(&params, &target_entry.key);
    assert!(removed.is_some(), "target shard was not removed");

    for seg in 0..SEGMENTS {
        let repaired = damaged
            .repair_primary_shard_from_rows(seg, target_entry)
            .expect("repair primary shard from rows");
        assert_eq!(repaired, expected[seg]);
    }
}

#[test]
fn auxiliary_row_repair_from_shards_matches_missing_row_after_updates() {
    let store = build_updated_store();
    let params = store.params.clone();
    let target_row_label = params.row_labels[0];
    let target_label_bytes = encode_value(target_row_label);
    let target_row_key = store
        .coded
        .row_keys()
        .copied()
        .find(|row_key| row_key.row_label_bytes == target_label_bytes)
        .expect("missing original row");
    let expected = store
        .coded
        .clone_row_family_segments(&params, &target_row_key)
        .expect("missing original row segments");
    assert_eq!(expected.len(), SEGMENTS);

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

    for seg in 0..SEGMENTS {
        let repaired = damaged
            .repair_row_from_shards(seg, target_row_label)
            .expect("repair row from primary shards");
        assert_eq!(repaired, expected[seg]);

        let non_zero_entries = repaired
            .iter()
            .filter(|value| **value != Fp::from(0u64))
            .count();
        assert!(non_zero_entries > 0);
        assert_ne!(decode_value(encode_value(repaired[0])), Fp::from(0u64));
    }
}

#[test]
fn optimized_row_repair_matches_direct_barycentric_reference() {
    let store = build_updated_store();
    let params = store.params.clone();
    let target_row_label = params.row_labels[0];

    for seg in 0..SEGMENTS {
        let optimized = store
            .repair_row_from_shards(seg, target_row_label)
            .expect("optimized row repair");
        let reference = reference_repair_row_from_shards(&store, seg, target_row_label);

        assert_eq!(optimized, reference);
    }
}

#[test]
#[ignore]
fn compare_optimized_and_reference_row_repair_large_control_point() {
    let kappa = 128usize;
    let shard_count = 128usize;
    let shard_registry = build_shard_registry_deterministic(shard_count);
    let params = build_coded_params(&CodedArrayConfig {
        kappa,
        shard_count,
        shard_registry,
        party_registry_indices: vec![(0..shard_count).collect()],
        seed: 41,
    });

    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments(params.segment_len as u64, |seg| {
            build_segment_values(&params, seg)
        })
        .expect("extend");

    let target_row_label = params.row_labels[0];

    let optimized_start = Instant::now();
    let optimized = store
        .repair_row_from_shards(0, target_row_label)
        .expect("optimized row repair");
    let optimized_elapsed = optimized_start.elapsed();

    let reference_start = Instant::now();
    let reference = reference_repair_row_from_shards(&store, 0, target_row_label);
    let reference_elapsed = reference_start.elapsed();

    assert_eq!(optimized, reference);
    println!("optimized row repair: {:?}", optimized_elapsed);
    println!("reference row repair: {:?}", reference_elapsed);
}
