use paper_a_optimized::coded_array::params::CodedArrayConfig;
use paper_a_optimized::unauth::{ClaimBundle, LogicalValueFamilyKey, SetupTimings, UnauthStore};
use paper_a_optimized::{Fp, build_coded_params, build_shard_registry_deterministic, encode_value};
use std::ops::Range;
use std::time::{Duration, Instant};

fn build_params() -> paper_a_optimized::CodedArrayParams {
    let kappa = 2usize;
    let shard_count = 4usize;
    let shard_registry = build_shard_registry_deterministic(shard_count);
    build_coded_params(&CodedArrayConfig {
        kappa,
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

/// Two value families alternate between consecutive segments.
fn family_key_for_range(
    params: &paper_a_optimized::CodedArrayParams,
    range: Range<u64>,
) -> LogicalValueFamilyKey {
    let seg = range.start / params.segment_len as u64;
    LogicalValueFamilyKey::from_codes(vec![(seg % 2) as u16])
}

/// Value of one family at one logical offset within a segment.
fn family_value(family_key: &LogicalValueFamilyKey, logical_offset: u64) -> [u8; 32] {
    let family = family_key.codes()[0] as u64;
    encode_value(Fp::from(family * 1_000 + logical_offset * 7 + 1))
}

fn family_values(
    params: &paper_a_optimized::CodedArrayParams,
    family_key: &LogicalValueFamilyKey,
) -> Vec<[u8; 32]> {
    (0..params.segment_len as u64)
        .map(|logical_offset| family_value(family_key, logical_offset))
        .collect()
}

/// Value at one public index of the array built from the alternating families.
fn family_value_at_index(params: &paper_a_optimized::CodedArrayParams, idx: u64) -> [u8; 32] {
    let segment_len = params.segment_len as u64;
    let seg_start = idx - idx % segment_len;
    let family_key = family_key_for_range(params, seg_start..seg_start + segment_len);
    family_value(&family_key, idx % segment_len)
}

#[test]
fn timed_full_store_extend_matches_untimed_and_records_encoding_time() {
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

    assert!(timings.shard_eval > Duration::ZERO);
    assert!(timings.row_eval > Duration::ZERO);
    assert_eq!(timings.tree_build, Duration::ZERO);
}

#[test]
fn timed_claim_bound_extend_matches_untimed_and_records_encoding_time() {
    let params = build_params();
    let claims = vec![exact_claim(&params, 0), exact_claim(&params, 1)];

    let mut untimed = UnauthStore::new_with_claims(params.clone(), claims.clone());
    untimed
        .extend_with_segments_timed(14, |seg| build_segment_values(&params, seg), None)
        .expect("untimed claim-bound extend");

    let mut timed = UnauthStore::new_with_claims(params.clone(), claims.clone());
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
    assert_eq!(timed.coded.shard_family_count(), claims.len());
    assert_eq!(timed.coded.row_family_count(), claims.len());

    assert!(timings.shard_eval > Duration::ZERO);
    assert!(timings.row_eval > Duration::ZERO);
    assert_eq!(timings.tree_build, Duration::ZERO);
}

#[test]
fn timed_extend_of_constant_segments_records_no_encoding_time() {
    let params = build_params();
    let claims: Vec<_> = (0..params.shard_registry.len())
        .map(|registry_index| exact_claim(&params, registry_index))
        .collect();
    let constant = Fp::from(29u64);
    let constant_segment = vec![encode_value(constant); params.n_grid];

    let mut untimed = UnauthStore::new_with_claims(params.clone(), claims.clone());
    untimed
        .extend_with_segments_timed(7, |_| constant_segment.clone(), None)
        .expect("untimed constant extend");

    let mut timed = UnauthStore::new_with_claims(params.clone(), claims);
    let mut timings = SetupTimings::default();
    timed
        .extend_with_segments_timed(7, |_| constant_segment.clone(), Some(&mut timings))
        .expect("timed constant extend");

    assert_eq!(timed.len(), untimed.len());
    assert_eq!(timed.segments(), untimed.segments());
    assert!(timed.coded.equals_for_tests(&params, &untimed.coded));

    // A constant segment is the constant polynomial: every stored column and
    // row entry is that constant, and no polynomial is evaluated.
    assert_eq!(timings.shard_eval, Duration::ZERO);
    assert_eq!(timings.row_eval, Duration::ZERO);
    for seg in 0..timed.segments() {
        for (_, column) in timed.coded.shard_segments_for_segment(&params, seg) {
            assert!(column.iter().all(|value| *value == constant));
        }
        for (_, row) in timed.coded.row_segments_for_segment(&params, seg) {
            assert!(row.iter().all(|value| *value == constant));
        }
    }
    for idx in 0..timed.len() {
        assert_eq!(timed.read(idx).expect("read"), encode_value(constant));
    }
}

#[test]
fn full_store_value_family_materialization_matches_per_segment_reference() {
    let params = build_params();
    let len = 16 * params.segment_len as u64;

    let mut store_families = UnauthStore::new(params.clone());
    let mut timings = SetupTimings::default();
    store_families
        .extend_with_value_families_timed(
            len,
            |range| family_key_for_range(&params, range),
            |_, family_key| family_values(&params, family_key),
            Some(&mut timings),
        )
        .expect("value-family full extend");

    let mut store_reference = UnauthStore::new(params.clone());
    store_reference
        .extend_with_values(len, |idx| family_value_at_index(&params, idx))
        .expect("per-segment full extend");

    assert_eq!(store_families.len(), store_reference.len());
    assert_eq!(store_families.segments(), store_reference.segments());
    assert!(
        store_families
            .coded
            .equals_for_tests(&params, &store_reference.coded)
    );
    assert!(timings.shard_eval > Duration::ZERO);
    assert!(timings.row_eval > Duration::ZERO);
}

#[test]
fn dense_claim_bound_value_family_materialization_matches_per_segment_reference() {
    let params = build_params();
    let len = 16 * params.segment_len as u64;
    let claims: Vec<_> = (0..params.shard_registry.len())
        .map(|registry_index| exact_claim(&params, registry_index))
        .collect();

    let mut store_families = UnauthStore::new_with_claims(params.clone(), claims.clone());
    store_families
        .extend_with_value_families_timed(
            len,
            |range| family_key_for_range(&params, range),
            |_, family_key| family_values(&params, family_key),
            None,
        )
        .expect("value-family dense claim-bound extend");

    let mut store_reference = UnauthStore::new_with_claims(params.clone(), claims);
    store_reference
        .extend_with_values(len, |idx| family_value_at_index(&params, idx))
        .expect("per-segment dense claim-bound extend");

    assert_eq!(store_families.len(), store_reference.len());
    assert_eq!(store_families.segments(), store_reference.segments());
    assert!(
        store_families
            .coded
            .equals_for_tests(&params, &store_reference.coded)
    );
}

#[test]
#[ignore]
fn compare_per_segment_and_value_family_materialization_large_claim_bound_segments() {
    let kappa = 128usize;
    let shard_count = 8usize;
    let segments = 16u64;
    let shard_registry = build_shard_registry_deterministic(shard_count);
    let params = build_coded_params(&CodedArrayConfig {
        kappa,
        shard_count,
        shard_registry,
        party_registry_indices: vec![(0..shard_count).collect()],
        seed: 19,
    });
    let len = segments * params.segment_len as u64;
    let claims: Vec<_> = (0..params.shard_registry.len())
        .map(|registry_index| exact_claim(&params, registry_index))
        .collect();

    let mut store_families = UnauthStore::new_with_claims(params.clone(), claims.clone());
    let mut family_timings = SetupTimings::default();
    let family_start = Instant::now();
    store_families
        .extend_with_value_families_timed(
            len,
            |range| family_key_for_range(&params, range),
            |_, family_key| family_values(&params, family_key),
            Some(&mut family_timings),
        )
        .expect("value-family extend");
    let family_elapsed = family_start.elapsed();

    let mut store_reference = UnauthStore::new_with_claims(params.clone(), claims);
    let mut reference_timings = SetupTimings::default();
    let reference_start = Instant::now();
    store_reference
        .extend_with_values_timed(
            len,
            |idx| family_value_at_index(&params, idx),
            Some(&mut reference_timings),
        )
        .expect("per-segment extend");
    let reference_elapsed = reference_start.elapsed();

    assert!(
        store_families
            .coded
            .equals_for_tests(&params, &store_reference.coded)
    );

    println!("value-family elapsed: {:?}", family_elapsed);
    println!("value-family shard eval: {:?}", family_timings.shard_eval);
    println!("value-family row eval: {:?}", family_timings.row_eval);
    println!("per-segment elapsed: {:?}", reference_elapsed);
    println!("per-segment shard eval: {:?}", reference_timings.shard_eval);
    println!("per-segment row eval: {:?}", reference_timings.row_eval);
}
