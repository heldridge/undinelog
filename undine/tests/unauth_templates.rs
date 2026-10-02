use paper_a_optimized::coded_array::params::{CodedArrayConfig, ROWS_PER_CLAIM};
use paper_a_optimized::unauth::{
    ClaimBundle, build_kappa_double_params, build_kappa_halve_params, logical_offset_to_grid_index,
    public_index_to_segment,
};
use paper_a_optimized::{Fp, build_coded_params, build_shard_registry_deterministic, encode_value};

fn build_params(kappa: usize, shard_count: usize) -> paper_a_optimized::CodedArrayParams {
    let cfg = CodedArrayConfig {
        kappa,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count),
        party_registry_indices: vec![(0..shard_count).collect()],
        seed: 17,
    };
    build_coded_params(&cfg)
}

/// Return the `(Z-axis point, W-axis point)` pair of the cell holding one
/// public array index.
fn cell_points_for_index(params: &paper_a_optimized::CodedArrayParams, idx: u64) -> (Fp, Fp) {
    let (segment, logical_offset) = public_index_to_segment(params, idx);
    let grid_index = logical_offset_to_grid_index(params, logical_offset);
    params.cell_points(&params.segment_axes(segment), grid_index)
}

#[test]
fn rescaled_params_preserve_representatives_and_column_identifiers() {
    let params = build_params(4, 8);
    let kappa_doubled = build_kappa_double_params(&params).expect("kappa-double params");
    let kappa_halved = build_kappa_halve_params(&kappa_doubled).expect("kappa-halve params");

    assert_eq!(kappa_doubled.need, 2 * params.need);
    assert_eq!(kappa_doubled.s, 2 * params.s);
    assert_eq!(kappa_doubled.n_grid, 4 * params.n_grid);
    assert_eq!(kappa_doubled.segment_len, 4 * params.segment_len);
    assert_eq!(kappa_doubled.layout_base_n_grid, params.n_grid);
    assert_eq!(kappa_doubled.layout_level, 1);
    assert_eq!(kappa_halved.need, params.need);
    assert_eq!(kappa_halved.s, params.s);
    assert_eq!(kappa_halved.n_grid, params.n_grid);
    assert_eq!(kappa_halved.layout_base_n_grid, params.n_grid);
    assert_eq!(kappa_halved.layout_level, 0);
    assert_eq!(
        kappa_halved.logical_to_grid_index,
        params.logical_to_grid_index
    );

    assert_eq!(
        kappa_doubled.shard_registry.len(),
        params.shard_registry.len()
    );
    assert_eq!(
        kappa_halved.shard_registry.len(),
        params.shard_registry.len()
    );

    for rescaled in [&kappa_doubled, &kappa_halved] {
        for (idx, (source, entry)) in params
            .shard_registry
            .iter()
            .zip(rescaled.shard_registry.iter())
            .enumerate()
        {
            assert_eq!(entry.a, source.a);
            assert_eq!(entry.z, source.a);
            assert_eq!(entry.key, source.key);
            assert_eq!(entry.key.a_bytes, encode_value(entry.a));
            assert_eq!(entry.key.z_bytes, encode_value(entry.z));
            assert_eq!(rescaled.shard_registry_by_key.get(&entry.key), Some(&idx));
        }
    }
}

#[test]
fn row_labels_track_registry_assignments_across_rescale() {
    let params = build_params(4, 8);
    let kappa_doubled = build_kappa_double_params(&params).expect("kappa-double params");
    let kappa_halved = build_kappa_halve_params(&kappa_doubled).expect("kappa-halve params");

    for level in [&params, &kappa_doubled, &kappa_halved] {
        assert_eq!(
            level.row_labels.len(),
            level.shard_registry.len() * ROWS_PER_CLAIM
        );
        assert_eq!(level.row_labels, params.row_labels);
        assert_eq!(level.row_label_bytes, params.row_label_bytes);
        assert_eq!(level.row_len(), level.need);
    }
}

#[test]
fn cells_keep_their_axis_points_across_rescale() {
    let params = build_params(4, 8);
    let kappa_doubled = build_kappa_double_params(&params).expect("kappa-double params");
    let kappa_halved = build_kappa_halve_params(&kappa_doubled).expect("kappa-halve params");

    // Three stitched segments, so the scales of both levels are non-trivial.
    let len = 3 * kappa_doubled.segment_len as u64;
    for idx in 0..len {
        let source_points = cell_points_for_index(&params, idx);
        assert_eq!(cell_points_for_index(&kappa_doubled, idx), source_points);
        assert_eq!(cell_points_for_index(&kappa_halved, idx), source_points);
    }

    // Source segment `4b + 2u + v` sits on the half of the stitched segment's
    // Z-axis selected by `u` and the half of its W-axis selected by `v`.
    for parent in 0..3usize {
        let parent_axes = kappa_doubled.segment_axes(parent);
        for u in 0..2usize {
            for v in 0..2usize {
                let child_axes = params.segment_axes(4 * parent + 2 * u + v);
                for i in 0..params.need {
                    assert_eq!(
                        child_axes.rho_z * params.axis[i],
                        parent_axes.rho_z * kappa_doubled.axis[2 * i + u]
                    );
                    assert_eq!(
                        child_axes.rho_w * params.axis[i],
                        parent_axes.rho_w * kappa_doubled.axis[2 * i + v]
                    );
                }
            }
        }
    }
}

#[test]
fn row_label_metadata_assigns_one_row_per_claim() {
    let params = build_params(8, 8);

    assert_eq!(params.row_labels.len(), params.shard_registry.len());
    assert_eq!(params.row_label_bytes.len(), params.row_labels.len());

    for registry_index in 0..params.shard_registry.len() {
        assert_eq!(
            params.row_label_index_for_registry_index(registry_index),
            Some(registry_index)
        );
        assert_eq!(
            params.row_label_indices_for_registry_index(registry_index),
            vec![registry_index]
        );
        let claim = ClaimBundle::exact_for_registry_index(&params, registry_index)
            .expect("claim for registry index");
        assert_eq!(claim.registry_index, registry_index);
        assert_eq!(claim.row_label_index(), registry_index);
        assert_eq!(claim.row_label_indices(&params), Some(vec![registry_index]));
    }
}

#[test]
fn row_label_metadata_rejects_out_of_range_claims() {
    let params = build_params(8, 8);
    let out_of_range = params.shard_registry.len();

    assert_eq!(
        params.row_label_index_for_registry_index(out_of_range),
        None
    );
    assert!(
        params
            .row_label_indices_for_registry_index(out_of_range)
            .is_empty()
    );
    assert!(ClaimBundle::exact_for_registry_index(&params, out_of_range).is_none());
}

#[test]
fn row_label_metadata_round_trips_label_bytes() {
    let params = build_params(8, 8);

    for (row_label_index, row_label) in params.row_labels.iter().copied().enumerate() {
        let label_bytes = encode_value(row_label);
        assert_eq!(params.row_label_bytes[row_label_index], label_bytes);
        assert_eq!(
            params.row_label_index_for_bytes(label_bytes),
            Some(row_label_index)
        );
    }

    let column_identifier_bytes = encode_value(params.shard_registry[0].z);
    assert_eq!(
        params.row_label_index_for_bytes(column_identifier_bytes),
        None
    );
}
