use paper_a_optimized::coded_array::params::CodedArrayConfig;
use paper_a_optimized::unauth::{ClaimBundle, SetupTimings, UnauthStore};
use paper_a_optimized::{Fp, build_coded_params, build_shard_registry_deterministic, encode_value};
use std::time::Instant;

fn build_params() -> paper_a_optimized::CodedArrayParams {
    let segment_len = 8usize;
    let s = 4usize;
    let shard_count = 4usize;
    let n_grid = segment_len;
    let shard_registry = build_shard_registry_deterministic(shard_count, s, n_grid);
    build_coded_params(&CodedArrayConfig {
        segment_len,
        s,
        shard_count,
        shard_registry,
        party_registry_indices: vec![(0..shard_count).collect()],
        seed: 11,
    })
}

fn build_segment_values(params: &paper_a_optimized::CodedArrayParams, seg: usize) -> Vec<[u8; 32]> {
    (0..params.n_grid)
        .map(|idx| encode_value(Fp::from((seg * 100 + idx + 1) as u64)))
        .collect()
}

fn exact_claim(params: &paper_a_optimized::CodedArrayParams, registry_index: usize) -> ClaimBundle {
    ClaimBundle::exact_for_registry_index(params, registry_index)
        .expect("exact claim should exist for registry index")
}

#[test]
fn timed_full_store_extend_matches_untimed_and_records_fft_row_stats() {
    let params = build_params();

    let mut untimed = UnauthStore::new(params.clone());
    untimed
        .extend_with_segments(14, |seg| build_segment_values(&params, seg))
        .expect("untimed extend");

    let mut timed = UnauthStore::new(params.clone());
    let mut timings = SetupTimings::default();
    timed
        .extend_with_segments_timed(
            14,
            |seg| build_segment_values(&params, seg),
            Some(&mut timings),
        )
        .expect("timed extend");

    assert_eq!(timed.len(), untimed.len());
    assert_eq!(timed.segments(), untimed.segments());
    assert!(timed.coded.equals_for_tests(&params, &untimed.coded));

    assert!(timings.row_eval > std::time::Duration::ZERO);
    assert_eq!(timings.row_materialization.sparse_calls, 0);
    assert_eq!(timings.row_materialization.fft_domain_calls, 2);
    assert_eq!(
        timings.row_materialization.row_family_count,
        2 * params.row_family_count
    );
    assert_eq!(
        timings.row_materialization.row_label_count,
        2 * params.row_labels.len()
    );
    assert_eq!(
        timings.row_materialization.selected_row_count,
        2 * params.row_labels.len()
    );
    assert_eq!(
        timings.row_materialization.fft_domain_size,
        2 * params.row_domain_size
    );
}

#[test]
fn timed_claim_bound_extend_matches_untimed_and_records_sparse_row_stats() {
    let mut params = build_params();
    params.row_label_coset_bases.clear();
    let claims = vec![exact_claim(&params, 0), exact_claim(&params, 1)];

    let mut untimed = UnauthStore::new_with_claims(params.clone(), claims.clone());
    untimed
        .extend_with_segments_timed(14, |seg| build_segment_values(&params, seg), None)
        .expect("untimed claim-bound extend");

    let mut timed = UnauthStore::new_with_claims(params.clone(), claims);
    let mut timings = SetupTimings::default();
    timed
        .extend_with_segments_timed(
            14,
            |seg| build_segment_values(&params, seg),
            Some(&mut timings),
        )
        .expect("timed claim-bound extend");

    assert_eq!(timed.len(), untimed.len());
    assert_eq!(timed.segments(), untimed.segments());
    assert!(timed.coded.equals_for_tests(&params, &untimed.coded));

    assert!(timings.row_eval > std::time::Duration::ZERO);
    assert_eq!(timings.row_materialization.sparse_calls, 2);
    assert_eq!(timings.row_materialization.fft_domain_calls, 0);
    assert_eq!(timings.row_materialization.row_family_count, 4);
    assert_eq!(timings.row_materialization.row_label_count, 8);
    assert_eq!(timings.row_materialization.selected_row_count, 8);
    assert_eq!(timings.row_materialization.fft_domain_size, 0);
}

#[test]
fn timed_claim_bound_extend_can_switch_to_fft_row_stats() {
    let params = build_params();
    let claims: Vec<_> = (0..params.shard_registry.len())
        .map(|registry_index| exact_claim(&params, registry_index))
        .collect();

    let mut untimed = UnauthStore::new_with_claims(params.clone(), claims.clone());
    untimed
        .extend_with_segments_timed(7, |seg| build_segment_values(&params, seg), None)
        .expect("untimed dense claim-bound extend");

    let mut timed = UnauthStore::new_with_claims(params.clone(), claims);
    let mut timings = SetupTimings::default();
    timed
        .extend_with_segments_timed(
            7,
            |seg| build_segment_values(&params, seg),
            Some(&mut timings),
        )
        .expect("timed dense claim-bound extend");

    assert_eq!(timed.len(), untimed.len());
    assert_eq!(timed.segments(), untimed.segments());
    assert!(timed.coded.equals_for_tests(&params, &untimed.coded));

    assert!(timings.row_eval > std::time::Duration::ZERO);
    assert_eq!(timings.row_materialization.sparse_calls, 0);
    assert_eq!(timings.row_materialization.fft_domain_calls, 1);
    assert_eq!(
        timings.row_materialization.row_family_count,
        params.shard_registry.len()
    );
    assert_eq!(
        timings.row_materialization.row_label_count,
        params.shard_registry.len() * params.rows_per_shard
    );
    assert_eq!(
        timings.row_materialization.selected_row_count,
        params.shard_registry.len() * params.rows_per_shard
    );
    assert_eq!(timings.row_materialization.fft_domain_size, params.n_grid);
}

#[test]
fn full_store_fft_materialization_matches_sparse_reference() {
    let params_fft = build_params();
    let mut store_fft = UnauthStore::new(params_fft.clone());
    store_fft
        .extend_with_segments(14, |seg| build_segment_values(&params_fft, seg))
        .expect("fft-path full extend");

    let mut params_sparse = params_fft.clone();
    params_sparse.row_label_coset_bases.clear();
    let mut store_sparse = UnauthStore::new(params_sparse.clone());
    store_sparse
        .extend_with_segments(14, |seg| build_segment_values(&params_sparse, seg))
        .expect("sparse full extend");

    assert_eq!(store_fft.len(), store_sparse.len());
    assert_eq!(store_fft.segments(), store_sparse.segments());
    assert!(
        store_fft
            .coded
            .equals_for_tests(&params_fft, &store_sparse.coded)
    );
}

#[test]
fn dense_claim_bound_fft_materialization_matches_sparse_reference() {
    let params_fft = build_params();
    let claims: Vec<_> = (0..params_fft.shard_registry.len())
        .map(|registry_index| exact_claim(&params_fft, registry_index))
        .collect();

    let mut store_fft = UnauthStore::new_with_claims(params_fft.clone(), claims.clone());
    store_fft
        .extend_with_segments(7, |seg| build_segment_values(&params_fft, seg))
        .expect("fft-path dense claim-bound extend");

    let mut params_sparse = params_fft.clone();
    params_sparse.row_label_coset_bases.clear();
    let mut store_sparse = UnauthStore::new_with_claims(params_sparse.clone(), claims);
    store_sparse
        .extend_with_segments(7, |seg| build_segment_values(&params_sparse, seg))
        .expect("sparse dense claim-bound extend");

    assert_eq!(store_fft.len(), store_sparse.len());
    assert_eq!(store_fft.segments(), store_sparse.segments());
    assert!(
        store_fft
            .coded
            .equals_for_tests(&params_fft, &store_sparse.coded)
    );
}

#[test]
#[ignore]
fn compare_sparse_and_fft_row_materialization_large_claim_bound_segment() {
    let segment_len = 16_384usize;
    let s = 2_048usize;
    let shard_count = 8usize;
    let n_grid = segment_len;
    let shard_registry = build_shard_registry_deterministic(shard_count, s, n_grid);
    let params_fft = build_coded_params(&CodedArrayConfig {
        segment_len,
        s,
        shard_count,
        shard_registry,
        party_registry_indices: vec![(0..shard_count).collect()],
        seed: 19,
    });
    let claims: Vec<_> = (0..params_fft.shard_registry.len())
        .map(|registry_index| exact_claim(&params_fft, registry_index))
        .collect();

    let mut store_fft = UnauthStore::new_with_claims(params_fft.clone(), claims.clone());
    let mut fft_timings = SetupTimings::default();
    let fft_start = Instant::now();
    store_fft
        .extend_with_segments_timed(
            segment_len as u64,
            |seg| build_segment_values(&params_fft, seg),
            Some(&mut fft_timings),
        )
        .expect("fft-domain extend");
    let fft_elapsed = fft_start.elapsed();

    let mut params_sparse = params_fft.clone();
    params_sparse.row_label_coset_bases.clear();
    let mut store_sparse = UnauthStore::new_with_claims(params_sparse.clone(), claims);
    let mut sparse_timings = SetupTimings::default();
    let sparse_start = Instant::now();
    store_sparse
        .extend_with_segments_timed(
            segment_len as u64,
            |seg| build_segment_values(&params_sparse, seg),
            Some(&mut sparse_timings),
        )
        .expect("sparse extend");
    let sparse_elapsed = sparse_start.elapsed();

    assert!(
        store_fft
            .coded
            .equals_for_tests(&params_fft, &store_sparse.coded)
    );
    assert_eq!(fft_timings.row_materialization.fft_domain_calls, 1);
    assert_eq!(sparse_timings.row_materialization.sparse_calls, 1);

    println!("fft elapsed: {:?}", fft_elapsed);
    println!("fft row eval: {:?}", fft_timings.row_eval);
    println!(
        "fft row stats: families={} labels={} selected={} fft_domain_size={}",
        fft_timings.row_materialization.row_family_count,
        fft_timings.row_materialization.row_label_count,
        fft_timings.row_materialization.selected_row_count,
        fft_timings.row_materialization.fft_domain_size
    );
    println!("sparse elapsed: {:?}", sparse_elapsed);
    println!("sparse row eval: {:?}", sparse_timings.row_eval);
    println!(
        "sparse row stats: families={} labels={} selected={} fft_domain_size={}",
        sparse_timings.row_materialization.row_family_count,
        sparse_timings.row_materialization.row_label_count,
        sparse_timings.row_materialization.selected_row_count,
        sparse_timings.row_materialization.fft_domain_size
    );
}
