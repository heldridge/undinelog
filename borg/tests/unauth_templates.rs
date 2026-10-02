use paper_a_optimized::coded_array::params::CodedArrayConfig;
use paper_a_optimized::unauth::build_migrated_params;
use paper_a_optimized::{
    RowFamilyId, build_coded_params, build_shard_registry_deterministic, encode_value, fp_pow,
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
        seed: 17,
    };
    build_coded_params(&cfg)
}

#[test]
fn migrated_params_preserve_representatives_and_refresh_column_coordinates() {
    let params = build_params(32, 8, 8);
    let kappa_doubled = build_migrated_params(&params, 4).expect("kappa-double params");
    let kappa_halved = build_migrated_params(&kappa_doubled, 8).expect("kappa-halve params");

    assert_eq!(
        kappa_doubled.shard_registry.len(),
        params.shard_registry.len() * 2
    );
    assert_eq!(
        kappa_halved.shard_registry.len(),
        params.shard_registry.len()
    );

    for (idx, old) in params.shard_registry.iter().enumerate() {
        let left = kappa_doubled.shard_registry[2 * idx];
        let right = kappa_doubled.shard_registry[2 * idx + 1];
        assert_eq!(old.a, left.a);
        assert_eq!(left.z, fp_pow(old.a, 4));
        assert_eq!(left.key.z_bytes, encode_value(left.z));
        assert_eq!(right.z, -left.z);
    }

    for (old, new) in params
        .shard_registry
        .iter()
        .zip(kappa_halved.shard_registry.iter())
    {
        assert_eq!(old.a, new.a);
        assert_eq!(old.key.a_bytes, new.key.a_bytes);
        assert_eq!(new.z, fp_pow(old.a, 8));
        assert_eq!(new.key.z_bytes, encode_value(new.z));
    }
}

#[test]
fn row_domains_track_registry_assignments_across_migration() {
    let kappa_double_source_params = build_params(64, 8, 8);
    let kappa_doubled =
        build_migrated_params(&kappa_double_source_params, 4).expect("kappa-double params");
    let kappa_halve_source_params = build_params(64, 4, 4);
    let kappa_halved =
        build_migrated_params(&kappa_halve_source_params, 8).expect("kappa-halve params");

    assert_eq!(
        kappa_double_source_params.row_labels.len(),
        kappa_double_source_params.shard_registry.len() * kappa_double_source_params.rows_per_shard
    );
    assert_eq!(
        kappa_doubled.row_labels.len(),
        kappa_doubled.shard_registry.len() * kappa_doubled.rows_per_shard
    );
    assert_eq!(
        kappa_halve_source_params.row_labels.len(),
        kappa_halve_source_params.shard_registry.len() * kappa_halve_source_params.rows_per_shard
    );
    assert_eq!(
        kappa_halved.row_labels.len(),
        kappa_halved.shard_registry.len() * kappa_halved.rows_per_shard
    );
}

#[test]
fn row_eval_points_follow_square_and_sign_closure_across_epochs() {
    let kappa_double_source_params = build_params(64, 8, 8);
    let kappa_doubled =
        build_migrated_params(&kappa_double_source_params, 4).expect("kappa-double params");
    let kappa_halve_source_params = build_params(64, 4, 4);
    let kappa_halved =
        build_migrated_params(&kappa_halve_source_params, 8).expect("kappa-halve params");

    for idx in 0..kappa_halved.need {
        let z = kappa_halve_source_params.row_eval_points[idx];
        let z_neg = kappa_halve_source_params.row_eval_points[idx + kappa_halved.need];
        assert_eq!(z * z, kappa_halved.row_eval_points[idx]);
        assert_eq!(z_neg, -z);
    }

    for idx in 0..kappa_doubled.need {
        let u = kappa_doubled.row_eval_points[idx];
        assert_eq!(
            u * u,
            kappa_double_source_params.row_eval_points[idx % kappa_double_source_params.need]
        );
    }
}

#[test]
fn row_family_metadata_matches_legacy_structured_default_rows() {
    let params = build_params(64, 8, 8);

    assert_eq!(params.row_domain_size, params.row_labels.len());
    assert_eq!(params.row_family_size, params.rows_per_shard);
    assert_eq!(params.row_family_count, params.shard_registry.len());

    for registry_index in 0..params.shard_registry.len() {
        let family_id = RowFamilyId(registry_index);
        let explicit = params.row_label_indices_for_registry_index(registry_index);
        assert_eq!(
            params
                .row_label_indices_for_row_family(family_id)
                .expect("family rows"),
            explicit
        );
        assert_eq!(
            params.row_family_id_for_row_label_indices(&explicit),
            Some(family_id)
        );
    }
}

#[test]
fn row_family_adapter_rejects_nonfamily_row_shapes() {
    let params = build_params(64, 8, 8);
    assert_eq!(params.row_family_id_for_row_label_indices(&[0, 1]), None);
}

#[test]
fn structured_row_family_metadata_round_trips_default_claim_rows() {
    let params = build_params(64, 8, 8);

    assert_eq!(params.row_family_size, params.rows_per_shard);
    assert_eq!(
        params.row_family_count * params.row_family_size,
        params.row_labels.len()
    );

    for registry_index in 0..params.shard_registry.len() {
        let family_id = params
            .row_family_id_for_registry_index(registry_index)
            .expect("default family id");
        let family_rows = params
            .row_label_indices_for_row_family(family_id)
            .expect("family rows");
        let default_rows = params.row_label_indices_for_registry_index(registry_index);
        assert_eq!(family_rows, default_rows);
        assert_eq!(
            params.row_family_id_for_row_label_indices(&default_rows),
            Some(family_id)
        );
        for (offset, row_label_index) in family_rows.iter().copied().enumerate() {
            assert_eq!(
                params.row_family_location_for_row_label_index(row_label_index),
                Some((family_id, offset))
            );
        }
    }
}
