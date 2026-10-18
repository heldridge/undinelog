//! Basic unauthenticated party delta + read test.

use paper_a_optimized::coded_array::params::CodedArrayConfig;
use paper_a_optimized::decode_value;
use paper_a_optimized::unauth::{DeltaMode, UnauthStore, logical_offset_to_grid_index};
use paper_a_optimized::{Fp, build_shard_registry_deterministic, encode_value};

#[test]
fn unauth_store_applies_delta_and_reads() {
    let segment_len: usize = 8;
    let n_grid = segment_len;
    let initial_len: u64 = 4;
    let s = 8;
    let g = 1;
    let shard_registry = build_shard_registry_deterministic(g, s, n_grid);
    let party_registry_indices = vec![vec![0]];
    let cfg = CodedArrayConfig {
        segment_len,
        s,
        shard_count: g,
        shard_registry,
        party_registry_indices,
        seed: 7,
    };
    let params = paper_a_optimized::build_coded_params(&cfg);

    let mut store = UnauthStore::new(params.clone());
    store.extend(initial_len).expect("extend");
    store
        .apply_deltas(&[(2, Fp::from(5u64))], DeltaMode::Aggregate)
        .expect("apply delta");

    let value = decode_value(store.read(2).expect("read value"));
    assert_eq!(value, Fp::from(5u64));

    let raw_grid = logical_offset_to_grid_index(&params, 2);
    assert_eq!(raw_grid, 2);
}

#[test]
fn sorted_and_unsorted_delta_batches_match() {
    let segment_len: usize = 8;
    let n_grid = segment_len;
    let initial_len: u64 = 21;
    let s = 8;
    let g = 2;
    let shard_registry = build_shard_registry_deterministic(g, s, n_grid);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = CodedArrayConfig {
        segment_len,
        s,
        shard_count: g,
        shard_registry,
        party_registry_indices,
        seed: 11,
    };
    let params = paper_a_optimized::build_coded_params(&cfg);

    let sorted = vec![
        (0, Fp::from(1u64)),
        (3, Fp::from(2u64)),
        (8, Fp::from(3u64)),
        (13, Fp::from(4u64)),
        (20, Fp::from(5u64)),
    ];
    let unsorted = vec![sorted[3], sorted[1], sorted[4], sorted[0], sorted[2]];

    let mut sorted_store = UnauthStore::new(params.clone());
    sorted_store.extend(initial_len).expect("extend sorted");
    sorted_store
        .apply_deltas(&sorted, DeltaMode::Aggregate)
        .expect("apply sorted deltas");

    let mut unsorted_store = UnauthStore::new(params);
    unsorted_store.extend(initial_len).expect("extend unsorted");
    unsorted_store
        .apply_deltas(&unsorted, DeltaMode::Aggregate)
        .expect("apply unsorted deltas");

    for idx in 0..initial_len {
        assert_eq!(
            sorted_store.read(idx).expect("read sorted"),
            unsorted_store.read(idx).expect("read unsorted")
        );
    }
}

#[test]
fn unauth_extend_matches_generic_default_segment_materialization() {
    let segment_len: usize = 8;
    let n_grid = segment_len;
    let initial_len: u64 = 17;
    let s = 8;
    let g = 2;
    let shard_registry = build_shard_registry_deterministic(g, s, n_grid);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = CodedArrayConfig {
        segment_len,
        s,
        shard_count: g,
        shard_registry,
        party_registry_indices,
        seed: 9,
    };
    let params = paper_a_optimized::build_coded_params(&cfg);

    let mut optimized = UnauthStore::new(params.clone());
    optimized.extend(initial_len).expect("optimized extend");

    let zero = encode_value(Fp::from(0u64));
    let mut generic = UnauthStore::new(params.clone());
    generic
        .extend_with_segments(initial_len, |_| vec![zero; params.n_grid])
        .expect("generic extend");

    assert_eq!(optimized.len(), generic.len());
    assert_eq!(optimized.segments(), generic.segments());
    assert!(optimized.coded.equals_for_tests(&params, &generic.coded));
}
