//! Doubling and halving `kappa` on the unauthenticated store.
//!
//! Doubling `kappa` is the local four-segment stitch: four segments at
//! `(Seg, kappa)` become one segment at `(4 Seg, 2 kappa)`. Halving `kappa`
//! undoes it from the stitched fragment plus half of the source state.
//!
//! The oracle throughout is a store that encodes the same public values
//! directly at the target threshold.

use paper_a_optimized::coded_array::params::CodedArrayConfig;
use paper_a_optimized::unauth::service::{UnauthRequest, UnauthResponse, UnauthService};
use paper_a_optimized::unauth::{
    ClaimBundle, CodedArrayParams, DeltaMode, LeadingSourceDecoder, MigrateError,
    PartialStitchEvals, RecoveredClaimSegmentState, SourceHalfState, UnauthArray,
    UnauthArrayParams, UnauthOptimizedReadRequest, UnauthOptimizedReadResponse,
    UnauthReadObjectContribution, UnauthReadObjectId, UnauthStore, build_kappa_double_params,
    build_kappa_halve_params, build_params_for_kappa, decode_leading_source_half,
    decode_optimized_read_plan, handle_optimized_read_request,
    handle_optimized_read_request_with_defaults, plan_optimized_read_indices,
    public_values_per_segment,
};
use paper_a_optimized::{
    Fp, Value256, build_coded_params, build_shard_registry_deterministic, decode_value,
    encode_value,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn build_params(kappa: usize, shard_count: usize) -> CodedArrayParams {
    build_coded_params(&CodedArrayConfig {
        kappa,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count),
        party_registry_indices: vec![(0..shard_count).collect()],
        seed: 41,
    })
}

/// Deterministic public value at array index `idx`.
fn public_value(idx: u64) -> Value256 {
    encode_value(Fp::from(1_000 + 7 * idx))
}

/// Store holding `public_value(idx)` at every index below `len`, and zero in
/// the inactive tail of its last segment.
fn populated_store(
    params: &CodedArrayParams,
    len: u64,
    claims: Option<Vec<ClaimBundle>>,
) -> UnauthStore {
    let mut store = match claims {
        Some(claims) => UnauthStore::new_with_claims(params.clone(), claims),
        None => UnauthStore::new(params.clone()),
    };
    let zero = encode_value(Fp::from(0u64));
    store
        .extend_with_values(len, |idx| if idx < len { public_value(idx) } else { zero })
        .expect("extend");
    store
}

fn claims_for(params: &CodedArrayParams, registry_indices: &[usize]) -> Vec<ClaimBundle> {
    registry_indices
        .iter()
        .map(|&idx| ClaimBundle::exact_for_registry_index(params, idx).expect("claim"))
        .collect()
}

fn collect_values(store: &UnauthStore) -> Vec<Value256> {
    (0..store.len())
        .map(|idx| store.read(idx).expect("read"))
        .collect()
}

fn assert_same_coded_state(actual: &UnauthStore, expected: &UnauthStore) {
    assert_eq!(actual.len(), expected.len());
    assert_eq!(actual.segments(), expected.segments());
    assert_eq!(actual.params.need, expected.params.need);
    assert!(
        actual
            .coded
            .equals_for_tests(&expected.params, &expected.coded),
        "coded state differs from the direct encoding at kappa = {}",
        expected.params.need
    );
}

/// Source state of `claims` for one source segment, read from a store at the
/// lower threshold.
fn source_segment_state(
    source: &UnauthStore,
    claims: &[ClaimBundle],
    seg: usize,
) -> RecoveredClaimSegmentState {
    let params = &source.params;
    let mut shards = Vec::new();
    let mut rows = Vec::new();
    for claim in claims {
        let entry = params.shard_registry[claim.registry_index];
        let row_label = params.row_labels[claim.row_label_index()];
        let row_key = paper_a_optimized::RowKey::new(entry.key, row_label);
        // A source segment beyond the store is padding: its fragments are zero.
        let column = source
            .coded
            .shard_segment(params, &entry.key, seg)
            .map(|values| values.to_vec())
            .unwrap_or_else(|| vec![Fp::from(0u64); params.s]);
        let row = source
            .coded
            .row_segment(params, &row_key, seg)
            .map(|values| values.to_vec())
            .unwrap_or_else(|| vec![Fp::from(0u64); params.row_len()]);
        shards.push((claim.registry_index, column));
        rows.push((claim.row_label_index(), row));
    }
    RecoveredClaimSegmentState { shards, rows }
}

/// Diagonal source halves for every stitched segment, taken from a store at
/// the lower threshold that still holds the source state.
fn diagonal_halves_from_source(
    source: &UnauthStore,
    claims: &[ClaimBundle],
    stitched_segments: usize,
) -> Vec<SourceHalfState> {
    (0..stitched_segments)
        .map(|parent| SourceHalfState::Diagonal {
            segment_00: source_segment_state(source, claims, 4 * parent),
            segment_11: source_segment_state(source, claims, 4 * parent + 3),
        })
        .collect()
}

fn partial_stitch_evals(
    helper: &mut UnauthStore,
    segs: &[usize],
    col_labels: &[Fp],
    row_labels: &[Fp],
) -> Vec<Vec<PartialStitchEvals>> {
    match helper.handle_request(UnauthRequest::EvalPartialStitches {
        segs: segs.to_vec(),
        col_labels: col_labels.to_vec(),
        row_labels: row_labels.to_vec(),
    }) {
        UnauthResponse::EvalPartialStitches { evals } => evals,
        other => panic!("unexpected partial-stitch response: {:?}", other),
    }
}

/// Leading source halves for `claims`, decoded from the partial-stitch
/// evaluations of helpers at the stitched threshold.
fn leading_halves_from_helpers(
    stitched_params: &CodedArrayParams,
    claims: &[ClaimBundle],
    stitched_segments: usize,
    helpers: &mut [UnauthStore],
    max_errors: usize,
    mut corrupt: impl FnMut(&mut PartialStitchEvals),
) -> Vec<SourceHalfState> {
    let params = stitched_params;
    let segs = (0..stitched_segments).collect::<Vec<_>>();
    let col_labels = claims
        .iter()
        .map(|claim| params.shard_registry[claim.registry_index].z)
        .collect::<Vec<_>>();
    let row_labels = claims
        .iter()
        .map(|claim| params.row_labels[claim.row_label_index()])
        .collect::<Vec<_>>();
    let mut evals_by_seg = vec![Vec::new(); stitched_segments];
    for helper in helpers.iter_mut() {
        let evals = partial_stitch_evals(helper, &segs, &col_labels, &row_labels);
        for (seg, mut helper_evals) in evals.into_iter().enumerate() {
            for eval in helper_evals.iter_mut() {
                corrupt(eval);
            }
            evals_by_seg[seg].extend(helper_evals);
        }
    }
    evals_by_seg
        .into_iter()
        .enumerate()
        .map(|(seg, helper_evals)| {
            let mut columns_00 = Vec::new();
            let mut columns_01 = Vec::new();
            let mut rows_00 = Vec::new();
            let mut rows_10 = Vec::new();
            for (target_pos, claim) in claims.iter().enumerate() {
                let row_samples = helper_evals
                    .iter()
                    .map(|helper| {
                        (
                            params.row_labels[helper.registry_index],
                            helper.row_evals[target_pos],
                        )
                    })
                    .collect::<Vec<_>>();
                let column_samples = helper_evals
                    .iter()
                    .map(|helper| {
                        (
                            params.shard_registry[helper.registry_index].z,
                            helper.column_evals[target_pos],
                        )
                    })
                    .collect::<Vec<_>>();
                let fragments = decode_leading_source_half(
                    params,
                    seg,
                    &row_samples,
                    &column_samples,
                    max_errors,
                )
                .expect("decode leading source half");
                columns_00.push((claim.registry_index, fragments.column_00));
                columns_01.push((claim.registry_index, fragments.column_01));
                rows_00.push((claim.row_label_index(), fragments.row_00));
                rows_10.push((claim.row_label_index(), fragments.row_10));
            }
            SourceHalfState::Leading {
                columns_00,
                columns_01,
                rows_00,
                rows_10,
            }
        })
        .collect()
}

#[test]
fn kappa_double_params_derive_the_segment_size_and_keep_the_assignments() {
    let params = build_params(4, 12);
    let doubled = build_kappa_double_params(&params).expect("doubled params");
    assert_eq!(doubled.need, 8);
    assert_eq!(doubled.s, 8);
    assert_eq!(doubled.n_grid, 64);
    assert_eq!(doubled.segment_len, 64);
    assert_eq!(doubled.layout_level, 1);
    assert_eq!(doubled.layout_base_n_grid, params.n_grid);
    assert_eq!(doubled.shard_registry, params.shard_registry);
    assert_eq!(doubled.row_labels, params.row_labels);

    let halved = build_kappa_halve_params(&doubled).expect("halved params");
    assert_eq!(halved.need, params.need);
    assert_eq!(halved.n_grid, params.n_grid);
    assert_eq!(halved.layout_level, 0);
    assert_eq!(halved.logical_to_grid_index, params.logical_to_grid_index);

    assert!(matches!(
        build_kappa_halve_params(&params),
        Err(MigrateError::AtBaseLevel)
    ));
}

#[test]
fn kappa_double_preserves_public_reads_and_equals_direct_encoding() {
    for &(kappa, segments) in &[(2usize, 4usize), (2, 8), (4, 4), (4, 12)] {
        let params = build_params(kappa, 2 * kappa + 3);
        let len = (segments * params.n_grid) as u64;
        let mut store = populated_store(&params, len, None);
        let before = collect_values(&store);

        store.kappa_double().expect("kappa double");

        assert_eq!(store.params.need, 2 * kappa);
        assert_eq!(store.segments(), segments / 4);
        assert_eq!(collect_values(&store), before);
        let direct = populated_store(&store.params, len, None);
        assert_same_coded_state(&store, &direct);
    }
}

#[test]
fn kappa_double_preserves_claim_bound_state() {
    let params = build_params(4, 12);
    let claims = claims_for(&params, &[1, 4, 9]);
    let len = (8 * params.n_grid) as u64;
    let mut store = populated_store(&params, len, Some(claims.clone()));

    store.kappa_double().expect("kappa double");

    assert_eq!(store.claim_bundles(), Some(claims.clone()));
    let direct = populated_store(&store.params, len, Some(claims));
    assert_same_coded_state(&store, &direct);
    assert_eq!(store.coded.shard_family_count(), 3);
    assert_eq!(store.coded.row_family_count(), 3);
}

#[test]
fn kappa_double_halves_the_stored_field_elements_of_a_claim() {
    let params = build_params(4, 9);
    let claims = claims_for(&params, &[2]);
    let len = (8 * params.n_grid) as u64;
    let mut store = populated_store(&params, len, Some(claims));
    let stored_elements = |store: &UnauthStore| {
        let params = &store.params;
        let entry = params.shard_registry[2];
        let row_key = paper_a_optimized::RowKey::new(entry.key, params.row_labels[2]);
        let column_segments = store
            .coded
            .shard_family_segment_count(&entry.key)
            .expect("column family");
        let row_segments = store
            .coded
            .row_family_segment_count(&row_key)
            .expect("row family");
        column_segments * params.s + row_segments * params.row_len()
    };
    // One claim stores 2 kappa elements per segment: 8 segments of 8 elements.
    assert_eq!(stored_elements(&store), 64);
    store.kappa_double().expect("kappa double");
    // After the stitch the claim stores 4 kappa elements for every four
    // source segments, instead of 8 kappa: 2 segments of 16 elements.
    assert_eq!(stored_elements(&store), 32);
}

#[test]
fn kappa_double_pads_a_partial_last_group_with_zero_segments() {
    let params = build_params(2, 7);
    // Five full source segments plus a partial sixth: the second group of
    // four has two real segments and two padded ones.
    let len = (5 * params.n_grid + 3) as u64;
    let mut store = populated_store(&params, len, None);
    assert_eq!(store.segments(), 6);
    let before = collect_values(&store);

    store.kappa_double().expect("kappa double");

    assert_eq!(store.segments(), 2);
    assert_eq!(store.len(), len);
    assert_eq!(collect_values(&store), before);
    let direct = populated_store(&store.params, len, None);
    assert_same_coded_state(&store, &direct);
    // Beyond the logical length the store reads the default.
    assert_eq!(
        decode_value(store.read(len + 1).expect("read past len")),
        Fp::from(0u64)
    );
}

#[test]
fn kappa_double_twice_preserves_public_reads() {
    let params = build_params(2, 11);
    let len = (16 * params.n_grid + 5) as u64;
    let mut store = populated_store(&params, len, None);
    let before = collect_values(&store);

    store.kappa_double().expect("first kappa double");
    store.kappa_double().expect("second kappa double");

    assert_eq!(store.params.need, 8);
    assert_eq!(store.params.layout_level, 2);
    assert_eq!(store.segments(), 2);
    assert_eq!(collect_values(&store), before);
    let direct = populated_store(&store.params, len, None);
    assert_same_coded_state(&store, &direct);
}

#[test]
fn updates_after_kappa_double_match_direct_encoding() {
    let params = build_params(2, 7);
    let len = (8 * params.n_grid) as u64;
    let mut store = populated_store(&params, len, None);
    store.kappa_double().expect("kappa double");

    let mut rng = ChaCha20Rng::seed_from_u64(5);
    let deltas = (0..20)
        .map(|_| (rng.gen_range(0..len), Fp::from(rng.gen_range(1..1_000u64))))
        .collect::<Vec<_>>();
    store
        .apply_deltas(&deltas, DeltaMode::Aggregate)
        .expect("apply deltas");

    let mut expected = (0..len)
        .map(|idx| decode_value(public_value(idx)))
        .collect::<Vec<_>>();
    for &(idx, delta) in &deltas {
        expected[idx as usize] += delta;
    }
    let mut direct = UnauthStore::new(store.params.clone());
    direct
        .extend_with_values(len, |idx| encode_value(expected[idx as usize]))
        .expect("direct extend");
    assert_same_coded_state(&store, &direct);
    for (idx, value) in expected.iter().enumerate() {
        assert_eq!(decode_value(store.read(idx as u64).expect("read")), *value);
    }
}

#[test]
fn block_reads_and_repair_work_after_kappa_double() {
    let params = build_params(2, 7);
    let len = (8 * params.n_grid) as u64;
    let mut store = populated_store(&params, len, None);
    store.kappa_double().expect("kappa double");
    let params = store.params.clone();

    // An aligned block is the set of cells that share one Z-axis point; after
    // a stitch its public indices are not contiguous.
    for block in 0..(2 * params.need) as u64 {
        let values = store.read_block(block).expect("read block");
        let indices = paper_a_optimized::unauth::aligned_block_indices(&params, block, len);
        assert_eq!(values.len(), params.s);
        for (value, idx) in values.iter().zip(indices) {
            let idx = idx.expect("active index");
            assert_eq!(*value, public_value(idx));
        }
    }

    // Batch reads address cells by public index; raw reads by grid index.
    let idxs = [0u64, 3, 9, 17, 30];
    assert_eq!(
        UnauthArray::read_batch(&store, &idxs).expect("read batch"),
        idxs.map(public_value).to_vec()
    );
    let span = public_values_per_segment(&params) as u64;
    for seg in 0..store.segments() as u64 {
        for grid_index in [0usize, 5, 10, 15] {
            let raw_idx = seg * span + grid_index as u64;
            let public_idx = seg * span + params.logical_offset_for_grid_index(grid_index) as u64;
            assert_eq!(
                UnauthArray::read_batch_raw(&store, &[raw_idx]).expect("raw read batch"),
                vec![public_value(public_idx)]
            );
            assert_eq!(
                store.read_raw(raw_idx).expect("raw read"),
                public_value(public_idx)
            );
        }
    }

    for seg in 0..store.segments() {
        let entry = params.shard_registry[3];
        let expected = store
            .coded
            .shard_segment(&params, &entry.key, seg)
            .expect("stored column")
            .to_vec();
        let mut damaged = store.clone();
        damaged
            .coded
            .remove_shard_family_segments(&params, &entry.key);
        assert_eq!(
            damaged
                .repair_primary_shard_from_rows(seg, entry)
                .expect("repair column"),
            expected
        );

        let row_label = params.row_labels[5];
        let row_key = paper_a_optimized::RowKey::new(params.shard_registry[5].key, row_label);
        let expected_row = store
            .coded
            .row_segment(&params, &row_key, seg)
            .expect("stored row")
            .to_vec();
        assert_eq!(
            store
                .repair_row_from_shards(seg, row_label)
                .expect("repair row"),
            expected_row
        );
    }
}

#[test]
fn extending_after_kappa_double_appends_at_the_doubled_threshold() {
    let params = build_params(2, 7);
    let len = (8 * params.n_grid) as u64;
    let mut store = populated_store(&params, len, None);
    store.kappa_double().expect("kappa double");

    let span = public_values_per_segment(&store.params) as u64;
    let new_len = len + span + 5;
    store
        .extend_with_values(new_len, |idx| {
            if idx < new_len {
                public_value(idx)
            } else {
                encode_value(Fp::from(0u64))
            }
        })
        .expect("extend after double");
    assert_eq!(store.segments(), 4);
    for idx in 0..new_len {
        assert_eq!(store.read(idx).expect("read"), public_value(idx));
    }
    let direct = populated_store(&store.params, new_len, None);
    assert_same_coded_state(&store, &direct);
}

#[test]
fn kappa_halve_from_diagonal_source_state_restores_the_source_store() {
    for &(kappa, len_cells) in &[(2usize, 32u64), (2, 23), (4, 128), (4, 100)] {
        let params = build_params(kappa, 2 * kappa + 3);
        let claims = claims_for(&params, &[0, 2, 5]);
        let source = populated_store(&params, len_cells, Some(claims.clone()));
        let mut store = source.clone();
        store.kappa_double().expect("kappa double");

        // The diagonal source state is ordinary state at the lower
        // threshold; here it is still available from the source store.
        let halves = diagonal_halves_from_source(&source, &claims, store.segments());
        store.kappa_halve(&halves).expect("kappa halve");

        assert_same_coded_state(&store, &source);
        assert_eq!(store.params.layout_level, 0);
        assert_eq!(
            store.params.logical_to_grid_index,
            source.params.logical_to_grid_index
        );
    }
}

#[test]
fn kappa_halve_downloads_half_of_the_source_state() {
    let params = build_params(4, 11);
    let claims = claims_for(&params, &[6]);
    let source = populated_store(&params, (4 * params.n_grid) as u64, Some(claims.clone()));
    let mut store = source.clone();
    store.kappa_double().expect("kappa double");

    let halves = diagonal_halves_from_source(&source, &claims, store.segments());
    let downloaded: usize = halves
        .iter()
        .map(|half| match half {
            SourceHalfState::Diagonal {
                segment_00,
                segment_11,
            } => [segment_00, segment_11]
                .iter()
                .map(|state| {
                    state.shards.iter().map(|(_, v)| v.len()).sum::<usize>()
                        + state.rows.iter().map(|(_, v)| v.len()).sum::<usize>()
                })
                .sum::<usize>(),
            SourceHalfState::Leading { .. } => unreachable!(),
        })
        .sum();
    // The full source state of one claim for four segments is 8 kappa
    // elements; undoing the stitch needs 4 kappa of them.
    assert_eq!(downloaded, 4 * params.need);
    store.kappa_halve(&halves).expect("kappa halve");
    assert_same_coded_state(&store, &source);
}

#[test]
fn kappa_halve_with_helpers_at_the_doubled_threshold_restores_the_source_store() {
    for &(kappa, len_cells) in &[(2usize, 32u64), (2, 27), (4, 128)] {
        // Helpers hold 4 kappa claims in total; the receiver holds one more.
        let shard_count = 4 * kappa + 1;
        let params = build_params(kappa, shard_count);
        let receiver_claims = claims_for(&params, &[shard_count - 1]);
        let source = populated_store(&params, len_cells, Some(receiver_claims.clone()));
        let mut receiver = source.clone();
        receiver.kappa_double().expect("receiver kappa double");

        let mut helpers = (0..2)
            .map(|helper| {
                let indices = (helper * 2 * kappa..(helper + 1) * 2 * kappa).collect::<Vec<_>>();
                let mut store =
                    populated_store(&params, len_cells, Some(claims_for(&params, &indices)));
                store.kappa_double().expect("helper kappa double");
                store
            })
            .collect::<Vec<_>>();

        let stitched_params = receiver.params.clone();
        let halves = leading_halves_from_helpers(
            &stitched_params,
            &receiver_claims,
            receiver.segments(),
            &mut helpers,
            0,
            |_| {},
        );
        receiver.kappa_halve(&halves).expect("kappa halve");

        assert_same_coded_state(&receiver, &source);
    }
}

#[test]
fn shared_decoder_matches_per_segment_decoding() {
    let kappa = 4usize;
    let shard_count = 2 * kappa + 1;
    let params = build_params(kappa, shard_count);
    let len = (16 * params.n_grid) as u64;
    let helper_indices = (0..2 * kappa).collect::<Vec<_>>();
    let mut helper = populated_store(&params, len, Some(claims_for(&params, &helper_indices)));
    helper.kappa_double().expect("helper kappa double");
    let stitched_params = helper.params.clone();
    let segs = (0..helper.segments()).collect::<Vec<_>>();
    let col_labels = [stitched_params.shard_registry[shard_count - 1].z];
    let row_labels = [stitched_params.row_labels[shard_count - 1]];
    let evals = partial_stitch_evals(&mut helper, &segs, &col_labels, &row_labels);

    let decoder = LeadingSourceDecoder::new(&stitched_params, &helper_indices).expect("decoder");
    assert_eq!(decoder.registry_indices(), helper_indices.as_slice());
    for (seg, helper_evals) in evals.iter().enumerate() {
        let row_evals = helper_evals
            .iter()
            .map(|eval| eval.row_evals[0])
            .collect::<Vec<_>>();
        let column_evals = helper_evals
            .iter()
            .map(|eval| eval.column_evals[0])
            .collect::<Vec<_>>();
        let row_samples = helper_evals
            .iter()
            .map(|eval| {
                (
                    stitched_params.row_labels[eval.registry_index],
                    eval.row_evals[0],
                )
            })
            .collect::<Vec<_>>();
        let column_samples = helper_evals
            .iter()
            .map(|eval| {
                (
                    stitched_params.shard_registry[eval.registry_index].z,
                    eval.column_evals[0],
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            decoder
                .decode(&stitched_params, seg, &row_evals, &column_evals)
                .expect("shared decode"),
            decode_leading_source_half(&stitched_params, seg, &row_samples, &column_samples, 0)
                .expect("per-segment decode"),
        );
    }

    // The decoder needs exactly one helper claim per threshold unit, without
    // repeats.
    assert!(LeadingSourceDecoder::new(&stitched_params, &helper_indices[1..]).is_err());
    let mut repeated = helper_indices.clone();
    repeated[0] = repeated[1];
    assert!(LeadingSourceDecoder::new(&stitched_params, &repeated).is_err());
}

#[test]
fn kappa_halve_corrects_a_bad_helper_claim() {
    let kappa = 2usize;
    // Decoding at threshold 2 kappa with one error needs 2 kappa + 2 helper
    // claims.
    let helper_claim_count = 2 * kappa + 2;
    let shard_count = helper_claim_count + 1;
    let params = build_params(kappa, shard_count);
    let len = (8 * params.n_grid) as u64;
    let receiver_claims = claims_for(&params, &[shard_count - 1]);
    let source = populated_store(&params, len, Some(receiver_claims.clone()));
    let mut receiver = source.clone();
    receiver.kappa_double().expect("receiver kappa double");

    let helper_indices = (0..helper_claim_count).collect::<Vec<_>>();
    let mut helper = populated_store(&params, len, Some(claims_for(&params, &helper_indices)));
    helper.kappa_double().expect("helper kappa double");

    let stitched_params = receiver.params.clone();
    let halves = leading_halves_from_helpers(
        &stitched_params,
        &receiver_claims,
        receiver.segments(),
        std::slice::from_mut(&mut helper),
        1,
        |eval| {
            if eval.registry_index == 1 {
                for value in eval
                    .row_evals
                    .iter_mut()
                    .chain(eval.column_evals.iter_mut())
                {
                    *value += Fp::from(9u64);
                }
            }
        },
    );
    receiver.kappa_halve(&halves).expect("kappa halve");
    assert_same_coded_state(&receiver, &source);
}

#[test]
fn kappa_double_update_then_halve_matches_the_updated_source_store() {
    let kappa = 2usize;
    let params = build_params(kappa, 4 * kappa + 1);
    let len = (8 * params.n_grid) as u64;
    let receiver_claims = claims_for(&params, &[4 * kappa]);
    let helper_claims = claims_for(&params, &(0..4 * kappa).collect::<Vec<_>>());
    let mut source = populated_store(&params, len, Some(receiver_claims.clone()));
    let mut receiver = source.clone();
    let mut helper = populated_store(&params, len, Some(helper_claims));
    receiver.kappa_double().expect("receiver kappa double");
    helper.kappa_double().expect("helper kappa double");

    let deltas = vec![
        (0u64, Fp::from(3u64)),
        (9, Fp::from(5u64)),
        (17, Fp::from(7u64)),
        (len - 1, Fp::from(11u64)),
    ];
    for store in [&mut source, &mut receiver, &mut helper] {
        store
            .apply_deltas(&deltas, DeltaMode::Aggregate)
            .expect("apply deltas");
    }

    let stitched_params = receiver.params.clone();
    let halves = leading_halves_from_helpers(
        &stitched_params,
        &receiver_claims,
        receiver.segments(),
        std::slice::from_mut(&mut helper),
        0,
        |_| {},
    );
    receiver.kappa_halve(&halves).expect("kappa halve");
    assert_same_coded_state(&receiver, &source);
}

#[test]
fn kappa_halve_rejects_the_base_level_and_missing_source_state() {
    let params = build_params(2, 7);
    let len = (8 * params.n_grid) as u64;
    let claims = claims_for(&params, &[0]);
    let source = populated_store(&params, len, Some(claims.clone()));

    let mut base = source.clone();
    assert!(matches!(
        base.kappa_halve(&[]),
        Err(MigrateError::AtBaseLevel)
    ));

    let mut store = source.clone();
    store.kappa_double().expect("kappa double");
    let mut halves = diagonal_halves_from_source(&source, &claims, store.segments());
    halves.pop();
    assert!(matches!(
        store.kappa_halve(&halves),
        Err(MigrateError::MissingSourceFragments)
    ));

    // Fragments for a claim the store does not hold are rejected.
    let other_claims = claims_for(&params, &[1]);
    let other_source = populated_store(&params, len, Some(other_claims.clone()));
    let wrong = diagonal_halves_from_source(&other_source, &other_claims, store.segments());
    assert!(matches!(
        store.kappa_halve(&wrong),
        Err(MigrateError::InvalidSourceFragments)
    ));
    // A failed halve leaves the store unchanged.
    let mut stitched = source.clone();
    stitched.kappa_double().expect("kappa double");
    assert_same_coded_state(&store, &stitched);
}

#[test]
fn service_messages_double_and_halve_kappa() {
    let params = build_params(2, 9);
    let len = (4 * params.n_grid) as u64;
    let claims = claims_for(&params, &[8]);
    let source = populated_store(&params, len, Some(claims.clone()));
    let mut store = source.clone();

    assert!(matches!(
        store.handle_request(UnauthRequest::KappaDouble),
        UnauthResponse::KappaDouble { ok: true }
    ));
    assert_eq!(store.params.need, 4);
    assert_eq!(store.params.zero_segment_values().len(), 16);

    let source_halves = diagonal_halves_from_source(&source, &claims, store.segments());
    assert!(matches!(
        store.handle_request(UnauthRequest::KappaHalve { source_halves }),
        UnauthResponse::KappaHalve { ok: true }
    ));
    assert_same_coded_state(&store, &source);
}

mod client_orchestration {
    //! Doubling and halving `kappa` through the unauth client, over servers
    //! that each hold a few claims.

    use super::*;
    use paper_a_optimized::unauth::client::{ClientError, UnauthClient, UnauthTransport};
    use paper_a_optimized::{RowShare, ShardShare};
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct LocalServer {
        label: usize,
        store: Arc<Mutex<UnauthStore>>,
    }

    #[derive(Clone)]
    struct LocalTransport;

    fn call(server: &LocalServer, req: UnauthRequest) -> Result<UnauthResponse, ClientError> {
        let mut store = server.store.lock().expect("store lock");
        match store.handle_request(req) {
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            response => Ok(response),
        }
    }

    fn unexpected<T>(response: UnauthResponse) -> Result<T, ClientError> {
        Err(ClientError::InvalidResponse(format!("{:?}", response)))
    }

    impl UnauthTransport for LocalTransport {
        type Server = LocalServer;

        fn server_label(&self, server: &Self::Server) -> String {
            server.label.to_string()
        }

        fn unauth_get_len(&self, server: &Self::Server) -> Result<u64, ClientError> {
            match call(server, UnauthRequest::GetLen)? {
                UnauthResponse::Len { len } => Ok(len),
                other => unexpected(other),
            }
        }

        fn unauth_read(
            &self,
            server: &Self::Server,
            idx: u64,
        ) -> Result<Vec<ShardShare>, ClientError> {
            match call(server, UnauthRequest::Read { idx })? {
                UnauthResponse::Read { shares } => Ok(shares),
                other => unexpected(other),
            }
        }

        fn unauth_read_block(
            &self,
            server: &Self::Server,
            block: u64,
        ) -> Result<Vec<RowShare>, ClientError> {
            match call(server, UnauthRequest::ReadBlock { block })? {
                UnauthResponse::ReadBlock { rows } => Ok(rows),
                other => unexpected(other),
            }
        }

        fn unauth_read_rows_at_z(
            &self,
            server: &Self::Server,
            seg: usize,
            z_eval: Fp,
        ) -> Result<Vec<RowShare>, ClientError> {
            match call(server, UnauthRequest::ReadRowsAtZ { seg, z_eval })? {
                UnauthResponse::ReadRowsAtZ { rows } => Ok(rows),
                other => unexpected(other),
            }
        }

        fn unauth_read_rows_batch_at_z(
            &self,
            server: &Self::Server,
            seg: usize,
            z_evals: &[Fp],
        ) -> Result<Vec<Vec<RowShare>>, ClientError> {
            let req = UnauthRequest::ReadRowsBatchAtZ {
                seg,
                z_evals: z_evals.to_vec(),
            };
            match call(server, req)? {
                UnauthResponse::ReadRowsBatchAtZ { rows } => Ok(rows),
                other => unexpected(other),
            }
        }

        fn unauth_read_rows_matrix(
            &self,
            server: &Self::Server,
            requests: &[(usize, Vec<Fp>)],
        ) -> Result<Vec<Vec<Vec<RowShare>>>, ClientError> {
            let requests = requests
                .iter()
                .map(
                    |(seg, z_evals)| paper_a_optimized::unauth::service::RowReadRequest {
                        seg: *seg,
                        z_evals: z_evals.clone(),
                    },
                )
                .collect();
            match call(server, UnauthRequest::ReadRowsMatrix { requests })? {
                UnauthResponse::ReadRowsMatrix { rows } => Ok(rows),
                other => unexpected(other),
            }
        }

        fn unauth_multi_read(
            &self,
            server: &Self::Server,
            idxs: &[u64],
        ) -> Result<Vec<Vec<ShardShare>>, ClientError> {
            let req = UnauthRequest::MultiRead {
                idxs: idxs.to_vec(),
            };
            match call(server, req)? {
                UnauthResponse::MultiRead { shares } => Ok(shares),
                other => unexpected(other),
            }
        }

        fn unauth_repair_shard(
            &self,
            server: &Self::Server,
            seg: usize,
            registry_index: usize,
        ) -> Result<Vec<Fp>, ClientError> {
            let req = UnauthRequest::RepairShard {
                seg,
                registry_index,
            };
            match call(server, req)? {
                UnauthResponse::RepairShard { values } => Ok(values),
                other => unexpected(other),
            }
        }

        fn unauth_apply_deltas(
            &self,
            server: &Self::Server,
            deltas: &[(u64, Fp)],
            mode: DeltaMode,
        ) -> Result<bool, ClientError> {
            let req = UnauthRequest::ApplyDeltas {
                deltas: deltas.to_vec(),
                mode,
            };
            match call(server, req)? {
                UnauthResponse::ApplyDeltas { ok } => Ok(ok),
                other => unexpected(other),
            }
        }

        fn unauth_extend(&self, server: &Self::Server, new_len: u64) -> Result<bool, ClientError> {
            match call(server, UnauthRequest::Extend { new_len })? {
                UnauthResponse::Extend { ok } => Ok(ok),
                other => unexpected(other),
            }
        }

        fn unauth_kappa_double(&self, server: &Self::Server) -> Result<bool, ClientError> {
            match call(server, UnauthRequest::KappaDouble)? {
                UnauthResponse::KappaDouble { ok } => Ok(ok),
                other => unexpected(other),
            }
        }

        fn unauth_kappa_halve(
            &self,
            server: &Self::Server,
            source_halves: &[SourceHalfState],
        ) -> Result<bool, ClientError> {
            let req = UnauthRequest::KappaHalve {
                source_halves: source_halves.to_vec(),
            };
            match call(server, req)? {
                UnauthResponse::KappaHalve { ok } => Ok(ok),
                other => unexpected(other),
            }
        }

        fn unauth_eval_partial_stitches(
            &self,
            server: &Self::Server,
            segs: &[usize],
            col_labels: &[Fp],
            row_labels: &[Fp],
        ) -> Result<Vec<Vec<PartialStitchEvals>>, ClientError> {
            let req = UnauthRequest::EvalPartialStitches {
                segs: segs.to_vec(),
                col_labels: col_labels.to_vec(),
                row_labels: row_labels.to_vec(),
            };
            match call(server, req)? {
                UnauthResponse::EvalPartialStitches { evals } => Ok(evals),
                other => unexpected(other),
            }
        }
    }

    /// Servers that split the registry evenly, `claims_per_server` each.
    fn build_cluster(
        kappa: usize,
        server_count: usize,
        claims_per_server: usize,
        len: u64,
    ) -> (CodedArrayParams, Vec<LocalServer>) {
        let shard_count = server_count * claims_per_server;
        let owned = (0..server_count)
            .map(|server| {
                (server * claims_per_server..(server + 1) * claims_per_server).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let params = build_coded_params(&CodedArrayConfig {
            kappa,
            shard_count,
            shard_registry: build_shard_registry_deterministic(shard_count),
            party_registry_indices: owned.clone(),
            seed: 43,
        });
        let servers = owned
            .iter()
            .enumerate()
            .map(|(label, indices)| LocalServer {
                label,
                store: Arc::new(Mutex::new(populated_store(
                    &params,
                    len,
                    Some(claims_for(&params, indices)),
                ))),
            })
            .collect();
        (params, servers)
    }

    fn snapshot(servers: &[LocalServer]) -> Vec<UnauthStore> {
        servers
            .iter()
            .map(|server| server.store.lock().expect("store lock").clone())
            .collect()
    }

    #[test]
    fn client_kappa_double_is_local_and_keeps_reads() {
        let kappa = 2usize;
        let len = 37u64;
        let (params, servers) = build_cluster(kappa, 4, 2, len);
        let mut client = UnauthClient::new(
            LocalTransport,
            params.clone(),
            servers[0].clone(),
            servers.clone(),
            len,
        );
        let before = (0..len)
            .map(|idx| client.read(idx).expect("read before"))
            .collect::<Vec<_>>();

        client.kappa_double().expect("client kappa double");

        assert_eq!(client.params.need, 2 * kappa);
        for (server, source) in snapshot(&servers).iter().zip(0..) {
            assert_eq!(server.params.need, 2 * kappa, "server {}", source);
        }
        let after = (0..len)
            .map(|idx| client.read(idx).expect("read after"))
            .collect::<Vec<_>>();
        assert_eq!(after, before);
        assert_eq!(
            client.multi_read(&[0, 5, 17, 36]).expect("multi read"),
            vec![before[0], before[5], before[17], before[36]]
        );
        // Aligned blocks are read through the stitched auxiliary rows.
        let block = 3u64;
        let values = client.read_block(block).expect("read block");
        let indices = paper_a_optimized::unauth::aligned_block_indices(&client.params, block, len);
        for (value, idx) in values.iter().zip(indices) {
            if let Some(idx) = idx {
                assert_eq!(*value, before[idx as usize]);
            }
        }
    }

    #[test]
    fn client_kappa_halve_restores_every_server() {
        for &(kappa, len) in &[(2usize, 32u64), (2, 37), (4, 64)] {
            // Eight claims in total: enough helper claims at threshold
            // 2 kappa for kappa in {2, 4}.
            let (params, servers) = build_cluster(kappa, 4, 2, len);
            let sources = snapshot(&servers);
            let mut client = UnauthClient::new(
                LocalTransport,
                params.clone(),
                servers[0].clone(),
                servers.clone(),
                len,
            );
            client.kappa_double().expect("client kappa double");
            client
                .write_batch(
                    &[(1, Fp::from(3u64)), (len - 1, Fp::from(4u64))],
                    DeltaMode::Aggregate,
                )
                .expect("write at doubled threshold");

            client.kappa_halve().expect("client kappa halve");

            assert_eq!(client.params.need, kappa);
            for (server, mut source) in snapshot(&servers).into_iter().zip(sources) {
                source
                    .apply_deltas(
                        &[(1, Fp::from(3u64)), (len - 1, Fp::from(4u64))],
                        DeltaMode::Aggregate,
                    )
                    .expect("source deltas");
                assert_same_coded_state(&server, &source);
            }
            assert_eq!(
                decode_value(client.read(1).expect("read after halve")),
                decode_value(public_value(1)) + Fp::from(3u64)
            );
        }
    }

    #[test]
    fn client_kappa_halve_corrects_one_bad_helper_claim() {
        let kappa = 2usize;
        let len = 32u64;
        // Threshold 4 with one error needs six helper claims.
        let (params, servers) = build_cluster(kappa, 3, 2, len);
        let sources = snapshot(&servers);
        let mut client = UnauthClient::new(
            LocalTransport,
            params,
            servers[0].clone(),
            servers.clone(),
            len,
        );
        client.kappa_double().expect("client kappa double");

        // Corrupt the stitched state of one claim on one server, then put it
        // back after the halve so the comparison below is against honest
        // source state for the other servers.
        {
            let mut store = servers[2].store.lock().expect("store lock");
            let params = store.params.clone();
            let entry = params.shard_registry[5];
            let row_key = paper_a_optimized::RowKey::new(entry.key, params.row_labels[5]);
            for seg in 0..store.segments() {
                for value in store
                    .coded
                    .shard_segment_mut(&params, &entry.key, seg)
                    .expect("column")
                {
                    *value += Fp::from(1u64);
                }
                for value in store
                    .coded
                    .row_segment_mut(&params, &row_key, seg)
                    .expect("row")
                {
                    *value += Fp::from(1u64);
                }
            }
        }

        client
            .kappa_halve_correcting(1)
            .expect("client kappa halve correcting");

        // Servers 0 and 1 hold honest claims and are restored exactly.
        for (server, source) in snapshot(&servers).iter().zip(sources.iter()).take(2) {
            assert_same_coded_state(server, source);
        }
    }

    #[test]
    fn client_kappa_halve_needs_enough_helper_claims() {
        let kappa = 2usize;
        let len = 16u64;
        // Three claims cannot serve a decode at threshold four.
        let (params, servers) = build_cluster(kappa, 3, 1, len);
        let mut client = UnauthClient::new(
            LocalTransport,
            params,
            servers[0].clone(),
            servers.clone(),
            len,
        );
        client.kappa_double().expect("client kappa double");
        assert!(matches!(
            client.kappa_halve(),
            Err(ClientError::NotEnoughShards)
        ));
        // No server changed threshold.
        for server in snapshot(&servers) {
            assert_eq!(server.params.need, 2 * kappa);
        }
    }
}

#[test]
fn block_writes_after_kappa_double_check_liveness_by_public_index() {
    let params = build_params(2, 7);
    // Six source segments; the last stitched segment is half padding.
    let len = (6 * params.n_grid) as u64;
    let mut store = populated_store(&params, len, None);
    store.kappa_double().expect("kappa double");
    let params = store.params.clone();

    let blocks_per_segment = paper_a_optimized::unauth::logical_blocks_per_segment(&params) as u64;
    let mut wrote_live = false;
    let mut rejected_padding = false;
    for block in blocks_per_segment..2 * blocks_per_segment {
        let indices = paper_a_optimized::unauth::aligned_block_indices(&params, block, len);
        for (offset, idx) in indices.into_iter().enumerate() {
            let result = store.write_block_delta(block, offset, Fp::from(5u64));
            match idx {
                Some(idx) => {
                    result.expect("live block cell accepts a delta");
                    assert_eq!(
                        decode_value(store.read(idx).expect("read")),
                        decode_value(public_value(idx)) + Fp::from(5u64)
                    );
                    wrote_live = true;
                }
                None => {
                    assert!(result.is_err(), "padding cell must reject a delta");
                    rejected_padding = true;
                }
            }
        }
    }
    assert!(wrote_live && rejected_padding);
}

#[test]
fn large_delta_batches_after_kappa_double_match_direct_encoding_in_both_modes() {
    let params = build_params(4, 9);
    let len = (8 * params.n_grid) as u64;
    for mode in [DeltaMode::Aggregate, DeltaMode::InterpolateFFT] {
        let mut store = populated_store(&params, len, None);
        store.kappa_double().expect("kappa double");
        let deltas = (0..len)
            .filter(|idx| idx % 3 != 1)
            .map(|idx| (idx, Fp::from(2 * idx + 1)))
            .collect::<Vec<_>>();
        store.apply_deltas(&deltas, mode).expect("apply deltas");

        let mut expected = (0..len)
            .map(|idx| decode_value(public_value(idx)))
            .collect::<Vec<_>>();
        for &(idx, delta) in &deltas {
            expected[idx as usize] += delta;
        }
        let mut direct = UnauthStore::new(store.params.clone());
        direct
            .extend_with_values(len, |idx| encode_value(expected[idx as usize]))
            .expect("direct extend");
        assert_same_coded_state(&store, &direct);
    }
}

#[test]
fn extending_into_the_padding_of_a_stitched_segment_writes_the_new_values() {
    let params = build_params(2, 7);
    // An extension is given the value of every public index, also beyond the
    // new length inside the last segment.
    let extended = |params: &CodedArrayParams, len: u64| {
        let mut store = UnauthStore::new(params.clone());
        store.extend_with_values(len, public_value).expect("extend");
        store
    };
    // Six source segments: the second stitched segment is half padding.
    let len = (6 * params.n_grid) as u64;
    let mut store = extended(&params, len);
    store.kappa_double().expect("kappa double");
    assert_eq!(store.segments(), 2);

    // Grow inside the padded half, then past the stitched segment.
    for new_len in [len + 3, len + 8, len + 8 + 21] {
        store
            .extend_with_values(new_len, public_value)
            .expect("extend after double");
        for idx in 0..new_len {
            assert_eq!(
                store.read(idx).expect("read"),
                public_value(idx),
                "index {idx}"
            );
        }
        assert_same_coded_state(&store, &extended(&store.params, new_len));
    }
}

#[test]
fn zero_extension_after_kappa_double_keeps_the_padding() {
    let params = build_params(2, 7);
    let len = (6 * params.n_grid) as u64;
    let mut store = populated_store(&params, len, None);
    store.kappa_double().expect("kappa double");
    store.extend(len + 5).expect("zero extend");
    for idx in len..len + 5 {
        assert_eq!(decode_value(store.read(idx).expect("read")), Fp::from(0u64));
    }
    let mut direct = UnauthStore::new(store.params.clone());
    direct
        .extend_with_values(len + 5, |idx| {
            if idx < len {
                public_value(idx)
            } else {
                encode_value(Fp::from(0u64))
            }
        })
        .expect("direct extend");
    assert_same_coded_state(&store, &direct);
}

#[test]
fn extending_with_value_families_into_the_padding_writes_the_new_values() {
    use paper_a_optimized::unauth::LogicalValueFamilyKey;

    let params = build_params(2, 7);
    let len = (5 * params.n_grid) as u64;
    let claims = claims_for(&params, &[0, 3, 4]);
    let family_values = |range: std::ops::Range<u64>| range.map(public_value).collect::<Vec<_>>();
    let build = |params: &CodedArrayParams, len: u64| {
        let mut store = UnauthStore::new_with_claims(params.clone(), claims.clone());
        store
            .extend_with_value_families_timed(
                len,
                |range| LogicalValueFamilyKey::from_codes(vec![(range.start % 65_521) as u16]),
                |range, _| family_values(range),
                None,
            )
            .expect("family extend");
        store
    };

    let mut store = build(&params, len);
    store.kappa_double().expect("kappa double");
    let new_len = len + 9;
    store
        .extend_with_value_families_timed(
            new_len,
            |range| LogicalValueFamilyKey::from_codes(vec![(range.start % 65_521) as u16]),
            |range, _| family_values(range),
            None,
        )
        .expect("family extend after double");

    // Every stitched segment now holds the family values on all of its cells.
    let direct = build(&store.params, new_len);
    assert_same_coded_state(&store, &direct);
}

/// Every object of the layout at threshold `kappa` over `segments` segments:
/// each segment, each aligned block, and each point below `len`.
fn all_read_objects(kappa: usize, segments: usize, len: u64) -> UnauthOptimizedReadRequest {
    let mut objects = Vec::new();
    objects.extend((0..segments).map(|segment| UnauthReadObjectId::Segment { segment }));
    objects.extend((0..(segments * kappa) as u64).map(|block| UnauthReadObjectId::Block { block }));
    objects.extend((0..len).map(|array_index| UnauthReadObjectId::Point { array_index }));
    UnauthOptimizedReadRequest { kappa, objects }
}

/// A response as comparable data: per object, each claim's contribution.
fn response_fingerprint(
    response: &UnauthOptimizedReadResponse,
) -> Vec<(usize, Vec<Vec<Value256>>)> {
    response
        .objects
        .iter()
        .map(|object| match object {
            UnauthReadObjectContribution::Segment { object_pos, shards } => (
                *object_pos,
                shards
                    .iter()
                    .map(|shard| {
                        let mut entry = vec![encode_value(Fp::from(shard.registry_index as u64))];
                        entry.extend(shard.values.iter().copied().map(encode_value));
                        entry
                    })
                    .collect(),
            ),
            UnauthReadObjectContribution::Block { object_pos, rows } => (
                *object_pos,
                rows.iter()
                    .map(|row| vec![encode_value(row.row_label()), encode_value(row.value())])
                    .collect(),
            ),
            UnauthReadObjectContribution::Point { object_pos, shares } => (
                *object_pos,
                shares
                    .iter()
                    .map(|share| {
                        vec![
                            encode_value(Fp::from(share.registry_index() as u64)),
                            encode_value(share.value().expect("share value")),
                        ]
                    })
                    .collect(),
            ),
        })
        .collect()
}

#[test]
fn a_store_serves_reads_at_higher_thresholds_as_if_it_had_stitched() {
    // Full groups, a partial last group, and a group with one source segment.
    for &(kappa, segments, tail) in &[
        (2usize, 16usize, 0usize),
        (2, 64, 0),
        (2, 6, 3),
        (2, 21, 1),
        (4, 9, 5),
    ] {
        let shard_count = 16 * kappa;
        let params = build_params(kappa, shard_count);
        let len =
            ((segments - 1) * params.n_grid + if tail == 0 { params.n_grid } else { tail }) as u64;
        for claims in [None, Some(claims_for(&params, &[1, 4, shard_count - 1]))] {
            let source = populated_store(&params, len, claims);
            let mut stitched = source.clone();
            for levels in 1..=3 {
                stitched.kappa_double().expect("kappa double");
                let target_kappa = kappa << levels;
                assert_eq!(source.stitch_levels_to(target_kappa), Some(levels));
                let request = all_read_objects(target_kappa, stitched.segments(), len);

                let on_the_fly = handle_optimized_read_request(&source, &request);
                let expected = handle_optimized_read_request(&stitched, &request);
                assert_eq!(
                    response_fingerprint(&on_the_fly),
                    response_fingerprint(&expected),
                    "kappa = {kappa}, segments = {segments}, levels = {levels}"
                );
            }
        }
    }
}

#[test]
fn on_the_fly_reads_decode_at_the_higher_threshold() {
    let kappa = 4usize;
    let params = build_params(kappa, 20);
    let len = (10 * params.n_grid + 7) as u64;
    let source = populated_store(&params, len, None);
    let before = collect_values(&source);

    for levels in 1..=2 {
        let target_params = build_params_for_kappa(&params, kappa << levels).expect("params");
        let indices = (0..len).collect::<Vec<_>>();
        let plan = plan_optimized_read_indices(&target_params, len, &indices, 0, None);
        assert_eq!(plan.request.kappa, kappa << levels);
        assert!(!plan.cover.segments.is_empty());
        let response = handle_optimized_read_request(&source, &plan.request);
        let values =
            decode_optimized_read_plan(&target_params, &plan, &[response]).expect("decode");
        assert_eq!(values, before);
    }
}

#[test]
fn on_the_fly_reads_use_the_default_values_beyond_the_allocated_segments() {
    let kappa = 2usize;
    let params = build_params(kappa, 16);
    // Six source segments: the second stitched segment has two real source
    // segments and two default ones.
    let len = (5 * params.n_grid + 3) as u64;
    let source = populated_store(&params, len, None);
    let allocated = (source.segments() * params.n_grid) as u64;
    let default_value = |idx: u64| Fp::from(9_000 + 3 * idx);

    for levels in 1..=3 {
        let target_params = build_params_for_kappa(&params, kappa << levels).expect("params");
        let target_segments = source.segments().div_ceil(1 << (2 * levels));
        let capacity = (target_segments * target_params.n_grid) as u64;
        // The oracle encodes the same cells directly at the target threshold.
        let mut direct = UnauthStore::new(target_params.clone());
        direct
            .extend_with_values(capacity, |idx| {
                if idx < len {
                    public_value(idx)
                } else if idx < allocated {
                    encode_value(Fp::from(0u64))
                } else {
                    encode_value(default_value(idx))
                }
            })
            .expect("extend direct");

        let request = all_read_objects(kappa << levels, target_segments, capacity);
        let on_the_fly =
            handle_optimized_read_request_with_defaults(&source, &request, Some(&default_value));
        let expected = handle_optimized_read_request(&direct, &request);
        assert_eq!(
            response_fingerprint(&on_the_fly),
            response_fingerprint(&expected),
            "levels = {levels}"
        );
    }
}

#[test]
fn a_store_serves_only_thresholds_on_its_ladder_up_to_the_registry_size() {
    let params = build_params(4, 16);
    let len = (8 * params.n_grid) as u64;
    let mut store = populated_store(&params, len, None);

    assert_eq!(store.stitch_levels_to(4), Some(0));
    assert_eq!(store.stitch_levels_to(8), Some(1));
    assert_eq!(store.stitch_levels_to(16), Some(2));
    // Above the registry size no read can gather enough claims.
    assert_eq!(store.stitch_levels_to(32), None);
    assert_eq!(store.stitch_levels_to(2), None);
    assert_eq!(store.stitch_levels_to(12), None);
    assert_eq!(store.stitch_levels_to(0), None);

    let request = UnauthOptimizedReadRequest {
        kappa: 2,
        objects: vec![UnauthReadObjectId::Point { array_index: 0 }],
    };
    assert!(
        handle_optimized_read_request(&store, &request)
            .objects
            .is_empty()
    );

    // A stitched store no longer serves its source threshold.
    store.kappa_double().expect("kappa double");
    assert_eq!(store.stitch_levels_to(4), None);
    assert_eq!(store.stitch_levels_to(8), Some(0));
    assert_eq!(store.stitch_levels_to(16), Some(1));
}
