//! Message-level demo tests using the shared net protocol types.

use paper_a_optimized::coded_array::params::CodedArrayConfig;
use paper_a_optimized::coded_array::rs::{barycentric_eval, precompute_barycentric};
use paper_a_optimized::net::unauth;
use paper_a_optimized::net::wire::{
    NetFpBlob, NetIndexedFragments, NetMaybeFpBlob, NetPartialStitchEvals, NetRowBatchReply,
    NetSegmentPrimaryShardReply, NetSourceHalfState, UnauthOptimizedReadRequest,
    UnauthOptimizedReadResponse, UnauthReadObjectContribution, UnauthReadObjectId, decode_binary,
    encode_binary,
};
use paper_a_optimized::net::{Message, Response};
use paper_a_optimized::unauth::{
    DeltaMode, RowKey, UnauthStore, aligned_block_indices, block_alphas_for_segment,
    logical_offset_to_column_coordinate_for_segment, public_index_to_segment,
    semantic_block_position,
};
use paper_a_optimized::{Fp, build_coded_params, build_shard_registry_deterministic, encode_value};

fn shares_from_values(owned_indices: &[usize], values: &NetMaybeFpBlob) -> Vec<(usize, Fp)> {
    owned_indices
        .iter()
        .copied()
        .zip(values.0.iter().copied())
        .filter_map(|(registry_index, value)| value.map(|value| (registry_index, value)))
        .collect()
}

fn row_labels_for_owned_indices(
    params: &paper_a_optimized::CodedArrayParams,
    owned_indices: &[usize],
) -> Vec<Fp> {
    let mut labels: Vec<_> = owned_indices
        .iter()
        .flat_map(|&registry_index| params.row_label_indices_for_registry_index(registry_index))
        .filter_map(|row_label_index| params.row_labels.get(row_label_index).copied())
        .collect();
    labels.sort_by_key(|row_label| encode_value(*row_label));
    labels.dedup_by_key(|row_label| encode_value(*row_label));
    labels
}

/// Decode the value at public array index `idx` from primary-column shares:
/// interpolate over the helpers' column identifiers and evaluate at the
/// cell's Z-axis point.
fn decode_from_shares(
    params: &paper_a_optimized::CodedArrayParams,
    idx: u64,
    shares: &[(usize, Fp)],
) -> Fp {
    let (segment, logical_offset) = public_index_to_segment(params, idx);
    let z_eval = logical_offset_to_column_coordinate_for_segment(params, segment, logical_offset);
    let mut z_values = Vec::new();
    let mut r_values = Vec::new();
    for &(registry_index, y) in shares {
        let entry = params.shard_registry[registry_index];
        z_values.push(entry.z);
        r_values.push(y);
        if r_values.len() >= params.need {
            break;
        }
    }
    assert_eq!(r_values.len(), params.need);
    let weights = precompute_barycentric(&z_values);
    barycentric_eval(&weights, &r_values, z_eval)
}

fn decode_block_from_rows(
    params: &paper_a_optimized::CodedArrayParams,
    block: u64,
    owned_indices: &[usize],
    rows: &NetRowBatchReply,
) -> Vec<Fp> {
    let row_labels = row_labels_for_owned_indices(params, owned_indices);
    let row_values = rows.values.0.clone();
    let weights = precompute_barycentric(&row_labels);
    let pos = semantic_block_position(params, block);
    block_alphas_for_segment(params, pos.segment)
        .into_iter()
        .map(|alpha| barycentric_eval(&weights, &row_values, alpha))
        .collect()
}

fn row_batch_len(rows: &NetRowBatchReply) -> usize {
    rows.values.0.len()
}

#[test]
fn net_message_roundtrip_unauth_optimized_read() {
    let req = UnauthOptimizedReadRequest {
        kappa: 8,
        objects: vec![
            UnauthReadObjectId::Segment { segment: 2 },
            UnauthReadObjectId::Block { block: 5 },
            UnauthReadObjectId::Point { array_index: 17 },
        ],
    };
    let msg = Message::UnauthOptimizedRead { req: req.clone() };
    let payload = encode_binary(&msg).expect("encode optimized read request");
    let decoded: Message = decode_binary(&payload).expect("decode optimized read request");
    match decoded {
        Message::UnauthOptimizedRead { req: decoded_req } => assert_eq!(decoded_req, req),
        other => panic!("unexpected message: {:?}", other),
    }

    let resp = UnauthOptimizedReadResponse {
        objects: vec![
            UnauthReadObjectContribution::Segment {
                object_pos: 0,
                shards: vec![NetSegmentPrimaryShardReply {
                    values: NetFpBlob(vec![Fp::from(11u64), Fp::from(12u64)]),
                }],
            },
            UnauthReadObjectContribution::Block {
                object_pos: 1,
                rows: NetRowBatchReply {
                    values: NetFpBlob(vec![Fp::from(13u64), Fp::from(14u64)]),
                },
            },
            UnauthReadObjectContribution::Point {
                object_pos: 2,
                shares: NetMaybeFpBlob(vec![Some(Fp::from(15u64)), None]),
            },
        ],
    };
    let wire_resp = Response::UnauthOptimizedRead { resp: resp.clone() };
    let payload = encode_binary(&wire_resp).expect("encode optimized read response");
    let decoded: Response = decode_binary(&payload).expect("decode optimized read response");
    match decoded {
        Response::UnauthOptimizedRead { resp: decoded_resp } => assert_eq!(decoded_resp, resp),
        other => panic!("unexpected response: {:?}", other),
    }
}

#[test]
fn net_message_roundtrip_unauth() {
    let kappa: usize = 2;
    let initial_len: u64 = 4;
    let g = 2;
    let shard_registry = build_shard_registry_deterministic(g);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = CodedArrayConfig {
        kappa,
        shard_count: g,
        shard_registry,
        party_registry_indices,
        seed: 5,
    };
    let params = build_coded_params(&cfg);

    let mut store = UnauthStore::new(params.clone());
    store.extend(initial_len).expect("extend");

    let resp = unauth::handle_message(
        &mut store,
        Message::UnauthApplyDeltas {
            deltas: vec![(2, paper_a_optimized::net::wire::NetFp(Fp::from(11u64)))],
            mode: paper_a_optimized::net::wire::NetDeltaMode::from(DeltaMode::Aggregate),
        },
    );
    assert!(matches!(resp, Response::UnauthApplyDeltas { ok: true }));

    let resp = unauth::handle_message(&mut store, Message::UnauthRead { idx: 2 });
    let Response::UnauthRead { values } = resp else {
        panic!("unexpected response");
    };
    let shares = shares_from_values(&params.party_registry_indices[0], &values);
    let value = decode_from_shares(&params, 2, &shares);
    assert_eq!(value, Fp::from(11u64));
}

#[test]
fn net_message_roundtrip_unauth_kappa_rescale_wire_variants() {
    let msg = Message::UnauthKappaDouble;
    let payload = encode_binary(&msg).expect("encode kappa-double message");
    let decoded: Message = decode_binary(&payload).expect("decode kappa-double message");
    assert!(matches!(decoded, Message::UnauthKappaDouble));

    let fragments = |registry_index: u32, values: [u64; 2]| -> NetIndexedFragments {
        vec![(
            registry_index,
            NetFpBlob(values.iter().copied().map(Fp::from).collect()),
        )]
    };
    let msg = Message::UnauthKappaHalve {
        source_halves: vec![
            NetSourceHalfState::Diagonal {
                columns_00: fragments(2, [11, 7]),
                rows_00: fragments(2, [13, 17]),
                columns_11: fragments(2, [19, 23]),
                rows_11: fragments(2, [29, 31]),
            },
            NetSourceHalfState::Leading {
                columns_00: fragments(4, [37, 41]),
                columns_01: fragments(4, [43, 47]),
                rows_00: fragments(4, [53, 59]),
                rows_10: fragments(4, [61, 67]),
            },
        ],
    };
    let payload = encode_binary(&msg).expect("encode kappa-halve message");
    let decoded: Message = decode_binary(&payload).expect("decode kappa-halve message");
    let Message::UnauthKappaHalve { source_halves } = decoded else {
        panic!("unexpected message: {:?}", decoded);
    };
    assert_eq!(source_halves.len(), 2);
    match &source_halves[0] {
        NetSourceHalfState::Diagonal {
            columns_00,
            rows_00,
            columns_11,
            rows_11,
        } => {
            assert_eq!(columns_00, &fragments(2, [11, 7]));
            assert_eq!(rows_00, &fragments(2, [13, 17]));
            assert_eq!(columns_11, &fragments(2, [19, 23]));
            assert_eq!(rows_11, &fragments(2, [29, 31]));
        }
        other => panic!("unexpected source half: {:?}", other),
    }
    match &source_halves[1] {
        NetSourceHalfState::Leading {
            columns_00,
            columns_01,
            rows_00,
            rows_10,
        } => {
            assert_eq!(columns_00, &fragments(4, [37, 41]));
            assert_eq!(columns_01, &fragments(4, [43, 47]));
            assert_eq!(rows_00, &fragments(4, [53, 59]));
            assert_eq!(rows_10, &fragments(4, [61, 67]));
        }
        other => panic!("unexpected source half: {:?}", other),
    }

    let msg = Message::UnauthEvalPartialStitches {
        segs: vec![0, 1],
        col_labels: NetFpBlob(vec![Fp::from(2u64), Fp::from(4u64)]),
        row_labels: NetFpBlob(vec![Fp::from(3u64), Fp::from(5u64)]),
    };
    let payload = encode_binary(&msg).expect("encode partial-stitch-eval message");
    let decoded: Message = decode_binary(&payload).expect("decode partial-stitch-eval message");
    match decoded {
        Message::UnauthEvalPartialStitches {
            segs,
            col_labels,
            row_labels,
        } => {
            assert_eq!(segs, vec![0, 1]);
            assert_eq!(col_labels.0, vec![Fp::from(2u64), Fp::from(4u64)]);
            assert_eq!(row_labels.0, vec![Fp::from(3u64), Fp::from(5u64)]);
        }
        other => panic!("unexpected message: {:?}", other),
    }

    let msg = Message::UnauthRetainClaims {
        registry_indices: vec![0, 2, 4],
    };
    let payload = encode_binary(&msg).expect("encode retain message");
    let decoded: Message = decode_binary(&payload).expect("decode retain message");
    match decoded {
        Message::UnauthRetainClaims { registry_indices } => {
            assert_eq!(registry_indices, vec![0, 2, 4]);
        }
        other => panic!("unexpected message: {:?}", other),
    }

    let resp = Response::UnauthKappaDouble { ok: true };
    let payload = encode_binary(&resp).expect("encode kappa-double response");
    let decoded: Response = decode_binary(&payload).expect("decode kappa-double response");
    assert!(matches!(decoded, Response::UnauthKappaDouble { ok: true }));

    let resp = Response::UnauthKappaHalve { ok: true };
    let payload = encode_binary(&resp).expect("encode kappa-halve response");
    let decoded: Response = decode_binary(&payload).expect("decode kappa-halve response");
    assert!(matches!(decoded, Response::UnauthKappaHalve { ok: true }));

    let resp = Response::UnauthEvalPartialStitches {
        evals: vec![
            vec![NetPartialStitchEvals {
                registry_index: 1,
                row_evals: NetFpBlob(vec![Fp::from(13u64), Fp::from(17u64)]),
                column_evals: NetFpBlob(vec![Fp::from(19u64), Fp::from(23u64)]),
            }],
            Vec::new(),
        ],
    };
    let payload = encode_binary(&resp).expect("encode partial-stitch-eval response");
    let decoded: Response = decode_binary(&payload).expect("decode partial-stitch-eval response");
    match decoded {
        Response::UnauthEvalPartialStitches { evals } => {
            assert_eq!(evals.len(), 2);
            assert_eq!(evals[0].len(), 1);
            assert_eq!(evals[0][0].registry_index, 1);
            assert_eq!(
                evals[0][0].row_evals.0,
                vec![Fp::from(13u64), Fp::from(17u64)]
            );
            assert_eq!(
                evals[0][0].column_evals.0,
                vec![Fp::from(19u64), Fp::from(23u64)]
            );
            assert!(evals[1].is_empty());
        }
        other => panic!("unexpected response: {:?}", other),
    }

    let resp = Response::UnauthRetainClaims { ok: true };
    let payload = encode_binary(&resp).expect("encode retain response");
    let decoded: Response = decode_binary(&payload).expect("decode retain response");
    assert!(matches!(decoded, Response::UnauthRetainClaims { ok: true }));
}

#[test]
fn net_message_roundtrip_unauth_batch() {
    let kappa: usize = 2;
    let initial_len: u64 = 4;
    let g = 2;
    let shard_registry = build_shard_registry_deterministic(g);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = CodedArrayConfig {
        kappa,
        shard_count: g,
        shard_registry,
        party_registry_indices,
        seed: 9,
    };
    let params = build_coded_params(&cfg);

    let mut store = UnauthStore::new(params.clone());
    store.extend(initial_len).expect("extend");

    let resp = unauth::handle_message(
        &mut store,
        Message::UnauthApplyDeltas {
            deltas: vec![
                (2, paper_a_optimized::net::wire::NetFp(Fp::from(11u64))),
                (3, paper_a_optimized::net::wire::NetFp(Fp::from(7u64))),
            ],
            mode: paper_a_optimized::net::wire::NetDeltaMode::from(DeltaMode::Aggregate),
        },
    );
    assert!(matches!(resp, Response::UnauthApplyDeltas { ok: true }));

    let resp = unauth::handle_message(&mut store, Message::UnauthMultiRead { idxs: vec![2, 3] });
    let Response::UnauthMultiRead { values } = resp else {
        panic!("unexpected response");
    };
    let shares: Vec<_> = values
        .iter()
        .map(|values| shares_from_values(&params.party_registry_indices[0], values))
        .collect();
    assert_eq!(shares.len(), 2);
    let value_2 = decode_from_shares(&params, 2, &shares[0]);
    let value_3 = decode_from_shares(&params, 3, &shares[1]);
    assert_eq!(value_2, Fp::from(11u64));
    assert_eq!(value_3, Fp::from(7u64));
}

#[test]
fn net_message_roundtrip_unauth_block_update() {
    let kappa: usize = 4;
    let initial_len: u64 = 16;
    let g = 4;
    let shard_registry = build_shard_registry_deterministic(g);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = CodedArrayConfig {
        kappa,
        shard_count: g,
        shard_registry,
        party_registry_indices,
        seed: 13,
    };
    let params = build_coded_params(&cfg);

    let mut store = UnauthStore::new(params.clone());
    store.extend(initial_len).expect("extend");
    let block_indices: Vec<u64> = aligned_block_indices(&params, 1, initial_len)
        .into_iter()
        .map(|idx| idx.expect("active aligned index"))
        .collect();

    let resp = unauth::handle_message(
        &mut store,
        Message::UnauthApplyBlockDeltas {
            block: 1,
            deltas: vec![
                (0, paper_a_optimized::net::wire::NetFp(Fp::from(11u64))),
                (2, paper_a_optimized::net::wire::NetFp(Fp::from(7u64))),
                (3, paper_a_optimized::net::wire::NetFp(Fp::from(5u64))),
            ],
            mode: paper_a_optimized::net::wire::NetDeltaMode::from(DeltaMode::Aggregate),
        },
    );
    assert!(matches!(
        resp,
        Response::UnauthApplyBlockDeltas { ok: true }
    ));

    let resp = unauth::handle_message(&mut store, Message::UnauthReadBlock { block: 1 });
    let Response::UnauthReadBlock { rows } = resp else {
        panic!("unexpected response");
    };
    let block = decode_block_from_rows(&params, 1, &params.party_registry_indices[0], &rows);
    assert_eq!(block[0], Fp::from(11u64));
    assert_eq!(block[1], Fp::from(0u64));
    assert_eq!(block[2], Fp::from(7u64));
    assert_eq!(block[3], Fp::from(5u64));

    let resp = unauth::handle_message(
        &mut store,
        Message::UnauthMultiRead {
            idxs: block_indices,
        },
    );
    let Response::UnauthMultiRead { values } = resp else {
        panic!("unexpected response");
    };
    let shares: Vec<_> = values
        .iter()
        .map(|values| shares_from_values(&params.party_registry_indices[0], values))
        .collect();
    assert_eq!(shares.len(), 4);
    assert_eq!(decode_from_shares(&params, 4, &shares[0]), Fp::from(11u64));
    assert_eq!(decode_from_shares(&params, 5, &shares[1]), Fp::from(0u64));
    assert_eq!(decode_from_shares(&params, 6, &shares[2]), Fp::from(7u64));
    assert_eq!(decode_from_shares(&params, 7, &shares[3]), Fp::from(5u64));
}

#[test]
fn net_message_roundtrip_unauth_repair() {
    let kappa: usize = 4;
    let initial_len: u64 = 16;
    let g = 4;
    let shard_registry = build_shard_registry_deterministic(g);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = CodedArrayConfig {
        kappa,
        shard_count: g,
        shard_registry,
        party_registry_indices,
        seed: 17,
    };
    let params = build_coded_params(&cfg);

    let mut store = UnauthStore::new(params.clone());
    store.extend(initial_len).expect("extend");
    let resp = unauth::handle_message(
        &mut store,
        Message::UnauthApplyDeltas {
            deltas: vec![
                (0, paper_a_optimized::net::wire::NetFp(Fp::from(11u64))),
                (1, paper_a_optimized::net::wire::NetFp(Fp::from(7u64))),
            ],
            mode: paper_a_optimized::net::wire::NetDeltaMode::from(DeltaMode::Aggregate),
        },
    );
    assert!(matches!(resp, Response::UnauthApplyDeltas { ok: true }));

    let expected = store
        .repair_primary_shard_from_rows(0, params.shard_registry[0])
        .expect("repair shard");
    let resp = unauth::handle_message(
        &mut store,
        Message::UnauthRepairShard {
            seg: 0,
            registry_index: 0,
        },
    );
    let Response::UnauthRepairShard { values } = resp else {
        panic!("unexpected response");
    };
    let got: Vec<_> = values.into_iter().map(|value| value.0).collect();
    assert_eq!(got, expected);
}

#[test]
fn net_message_roundtrip_unauth_repair_batch() {
    let kappa: usize = 4;
    let initial_len: u64 = 16;
    let g = 4;
    let shard_registry = build_shard_registry_deterministic(g);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = CodedArrayConfig {
        kappa,
        shard_count: g,
        shard_registry,
        party_registry_indices,
        seed: 19,
    };
    let params = build_coded_params(&cfg);

    let mut store = UnauthStore::new(params.clone());
    store.extend(initial_len).expect("extend");
    let resp = unauth::handle_message(
        &mut store,
        Message::UnauthApplyDeltas {
            deltas: vec![
                (0, paper_a_optimized::net::wire::NetFp(Fp::from(11u64))),
                (1, paper_a_optimized::net::wire::NetFp(Fp::from(7u64))),
            ],
            mode: paper_a_optimized::net::wire::NetDeltaMode::from(DeltaMode::Aggregate),
        },
    );
    assert!(matches!(resp, Response::UnauthApplyDeltas { ok: true }));

    let expected0 = store
        .repair_primary_shard_from_rows(0, params.shard_registry[0])
        .expect("repair shard 0");
    let expected1 = store
        .repair_primary_shard_from_rows(0, params.shard_registry[1])
        .expect("repair shard 1");
    let resp = unauth::handle_message(
        &mut store,
        Message::UnauthRepairShardBatch {
            seg: 0,
            registry_indices: vec![0, 1],
        },
    );
    let Response::UnauthRepairShardBatch { values } = resp else {
        panic!("unexpected response");
    };
    let got: Vec<Vec<_>> = values
        .into_iter()
        .map(|batch| batch.into_iter().map(|value| value.0).collect())
        .collect();
    assert_eq!(got, vec![expected0.clone(), expected1]);

    let row_label_indices = vec![0u32, 1u32];
    let row_alphas: Vec<_> = row_label_indices
        .iter()
        .map(|&idx| params.row_labels[idx as usize])
        .collect();
    let w_axis_weights = precompute_barycentric(&block_alphas_for_segment(&params, 0));
    let expected_eval0 = row_alphas
        .iter()
        .map(|&row_alpha| barycentric_eval(&w_axis_weights, &expected0, row_alpha))
        .collect::<Vec<_>>();
    let resp = unauth::handle_message(
        &mut store,
        Message::UnauthEvalShardsAtRowLabels {
            seg: 0,
            col_labels: NetFpBlob(vec![params.shard_registry[0].z]),
            row_labels: NetFpBlob(row_alphas.clone()),
        },
    );
    let Response::UnauthEvalShardsAtRowLabels { values } = resp else {
        panic!("unexpected eval response");
    };
    let got_eval: Vec<Vec<_>> = values.into_iter().map(|batch| batch.0).collect();
    assert_eq!(got_eval, vec![expected_eval0.clone()]);

    let resp = unauth::handle_message(
        &mut store,
        Message::UnauthEvalShardsAtRowLabelsMatrix {
            segs: vec![0],
            col_labels: NetFpBlob(vec![params.shard_registry[0].z]),
            row_labels: NetFpBlob(row_alphas.clone()),
        },
    );
    let Response::UnauthEvalShardsAtRowLabelsMatrix { values } = resp else {
        panic!("unexpected eval-matrix response");
    };
    let got_eval_matrix: Vec<Vec<Vec<_>>> = values
        .0
        .chunks_exact(row_alphas.len())
        .map(|batch| vec![batch.to_vec()])
        .collect();
    assert_eq!(got_eval_matrix, vec![vec![expected_eval0]]);
}

#[test]
fn net_message_roundtrip_unauth_block() {
    let kappa: usize = 4;
    let initial_len: u64 = 16;
    let g = 4;
    let shard_registry = build_shard_registry_deterministic(g);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = CodedArrayConfig {
        kappa,
        shard_count: g,
        shard_registry,
        party_registry_indices,
        seed: 13,
    };
    let params = build_coded_params(&cfg);

    let mut store = UnauthStore::new(params.clone());
    store.extend(initial_len).expect("extend");
    let block_indices: Vec<u64> = aligned_block_indices(&params, 0, initial_len)
        .into_iter()
        .map(|idx| idx.expect("active aligned index"))
        .collect();
    let deltas = vec![
        (
            block_indices[0],
            paper_a_optimized::net::wire::NetFp(Fp::from(11u64)),
        ),
        (
            block_indices[1],
            paper_a_optimized::net::wire::NetFp(Fp::from(7u64)),
        ),
        (
            block_indices[2],
            paper_a_optimized::net::wire::NetFp(Fp::from(5u64)),
        ),
        (
            block_indices[3],
            paper_a_optimized::net::wire::NetFp(Fp::from(3u64)),
        ),
    ];
    let resp = unauth::handle_message(
        &mut store,
        Message::UnauthApplyDeltas {
            deltas,
            mode: paper_a_optimized::net::wire::NetDeltaMode::from(DeltaMode::Aggregate),
        },
    );
    assert!(matches!(resp, Response::UnauthApplyDeltas { ok: true }));

    let resp = unauth::handle_message(&mut store, Message::UnauthReadBlock { block: 0 });
    let Response::UnauthReadBlock { rows } = resp else {
        panic!("unexpected response");
    };
    assert!(row_batch_len(&rows) >= params.need);
    let values = decode_block_from_rows(&params, 0, &params.party_registry_indices[0], &rows);
    assert_eq!(
        values,
        vec![
            Fp::from(11u64),
            Fp::from(7u64),
            Fp::from(5u64),
            Fp::from(3u64),
        ]
    );
}

#[test]
fn net_message_roundtrip_unauth_read_rows_batch() {
    let kappa: usize = 4;
    let initial_len: u64 = 16;
    let g = 4;
    let shard_registry = build_shard_registry_deterministic(g);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = CodedArrayConfig {
        kappa,
        shard_count: g,
        shard_registry,
        party_registry_indices,
        seed: 29,
    };
    let params = build_coded_params(&cfg);

    let mut store = UnauthStore::new(params.clone());
    store.extend(initial_len).expect("extend");
    let resp = unauth::handle_message(
        &mut store,
        Message::UnauthApplyDeltas {
            deltas: vec![
                (0, paper_a_optimized::net::wire::NetFp(Fp::from(11u64))),
                (1, paper_a_optimized::net::wire::NetFp(Fp::from(7u64))),
            ],
            mode: paper_a_optimized::net::wire::NetDeltaMode::from(DeltaMode::Aggregate),
        },
    );
    assert!(matches!(resp, Response::UnauthApplyDeltas { ok: true }));

    let z_evals = vec![params.shard_registry[0].z, params.shard_registry[1].z];
    let resp = unauth::handle_message(
        &mut store,
        Message::UnauthReadRowsBatchAtZ {
            seg: 0,
            z_evals: NetFpBlob(z_evals.clone()),
        },
    );
    let Response::UnauthReadRowsBatchAtZ { rows } = resp else {
        panic!("unexpected response");
    };
    let row_count = store
        .coded
        .row_keys()
        .filter(|key| store.coded.row_segment(&params, key, 0).is_some())
        .count();
    assert_eq!(rows.values.0.len(), 2 * row_count);
    assert!(row_count >= params.need);
}

#[test]
fn net_message_roundtrip_unauth_read_rows_matrix() {
    let kappa: usize = 4;
    let initial_len: u64 = 16;
    let g = 4;
    let shard_registry = build_shard_registry_deterministic(g);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = CodedArrayConfig {
        kappa,
        shard_count: g,
        shard_registry,
        party_registry_indices,
        seed: 31,
    };
    let params = build_coded_params(&cfg);

    let mut store = UnauthStore::new(params.clone());
    store.extend(initial_len).expect("extend");
    let resp = unauth::handle_message(
        &mut store,
        Message::UnauthApplyDeltas {
            deltas: vec![(0, paper_a_optimized::net::wire::NetFp(Fp::from(11u64)))],
            mode: paper_a_optimized::net::wire::NetDeltaMode::from(DeltaMode::Aggregate),
        },
    );
    assert!(matches!(resp, Response::UnauthApplyDeltas { ok: true }));

    let resp = unauth::handle_message(
        &mut store,
        Message::UnauthReadRowsMatrix {
            requests: vec![
                paper_a_optimized::net::wire::NetRowReadRequest {
                    seg: 0,
                    z_evals: NetFpBlob(vec![params.shard_registry[0].z]),
                },
                paper_a_optimized::net::wire::NetRowReadRequest {
                    seg: 0,
                    z_evals: NetFpBlob(vec![
                        params.shard_registry[1].z,
                        params.shard_registry[2].z,
                    ]),
                },
            ],
        },
    );
    let Response::UnauthReadRowsMatrix { rows } = resp else {
        panic!("unexpected response");
    };
    let row_count = store
        .coded
        .row_keys()
        .filter(|key| store.coded.row_segment(&params, key, 0).is_some())
        .count();
    assert_eq!(rows.values.0.len(), 3 * row_count);
    assert!(row_count >= params.need);
}

#[test]
fn net_message_roundtrip_unauth_kappa_double_then_halve() {
    let kappa: usize = 2;
    let initial_len: u64 = 7;
    let g = 4;
    let shard_registry = build_shard_registry_deterministic(g);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = CodedArrayConfig {
        kappa,
        shard_count: g,
        shard_registry,
        party_registry_indices,
        seed: 21,
    };
    let params = build_coded_params(&cfg);

    let mut store = UnauthStore::new(params.clone());
    store.extend(initial_len).expect("extend");
    let resp = unauth::handle_message(
        &mut store,
        Message::UnauthApplyDeltas {
            deltas: vec![
                (0, paper_a_optimized::net::wire::NetFp(Fp::from(11u64))),
                (2, paper_a_optimized::net::wire::NetFp(Fp::from(5u64))),
                (4, paper_a_optimized::net::wire::NetFp(Fp::from(3u64))),
            ],
            mode: paper_a_optimized::net::wire::NetDeltaMode::from(DeltaMode::Aggregate),
        },
    );
    assert!(matches!(resp, Response::UnauthApplyDeltas { ok: true }));
    let source = store.clone();
    assert_eq!(source.segments(), 2);
    let expected: Vec<_> = (0..source.len())
        .map(|idx| {
            let bytes = source.read(idx).expect("read source");
            paper_a_optimized::decode_value(bytes)
        })
        .collect();
    let assert_reads = |store: &mut UnauthStore| {
        for idx in 0..initial_len {
            let resp = unauth::handle_message(store, Message::UnauthRead { idx });
            let Response::UnauthRead { values } = resp else {
                panic!("unexpected response");
            };
            let shares = shares_from_values(&store.params.party_registry_indices[0], &values);
            let value = decode_from_shares(&store.params, idx, &shares);
            assert_eq!(value, expected[idx as usize]);
        }
    };
    assert_reads(&mut store);
    assert_eq!(expected[2], Fp::from(5u64));
    assert_eq!(expected[4], Fp::from(3u64));

    let resp = unauth::handle_message(&mut store, Message::UnauthKappaDouble);
    assert!(matches!(resp, Response::UnauthKappaDouble { ok: true }));
    assert_eq!(store.params.need, 2 * kappa);
    assert_eq!(store.params.n_grid, 4 * params.n_grid);
    assert_eq!(store.segments(), 1);
    assert_eq!(store.len(), initial_len);
    assert_reads(&mut store);

    // The stitched auxiliary rows still repair the stitched primary columns.
    for registry_index in 0..g {
        let repaired = store
            .repair_primary_shard_from_rows(0, store.params.shard_registry[registry_index])
            .expect("repair stitched shard");
        assert_eq!(repaired.len(), 2 * kappa);
        let resp = unauth::handle_message(
            &mut store,
            Message::UnauthRepairShard {
                seg: 0,
                registry_index: registry_index as u32,
            },
        );
        let Response::UnauthRepairShard { values } = resp else {
            panic!("unexpected response");
        };
        let got: Vec<_> = values.into_iter().map(|value| value.0).collect();
        assert_eq!(got, repaired);
    }

    // The diagonal source segments of the single stitched segment are source
    // segments 0 and 3. The latter lies beyond the source array: its
    // fragments are zero.
    let source_fragments = |seg: usize| -> (NetIndexedFragments, NetIndexedFragments) {
        let mut columns = Vec::new();
        let mut rows = Vec::new();
        for registry_index in 0..g {
            let entry = params.shard_registry[registry_index];
            let row_key = RowKey::new(entry.key, params.row_labels[registry_index]);
            let column = source
                .coded
                .shard_segment(&params, &entry.key, seg)
                .map(|values| values.to_vec())
                .unwrap_or_else(|| vec![Fp::from(0u64); params.s]);
            let row = source
                .coded
                .row_segment(&params, &row_key, seg)
                .map(|values| values.to_vec())
                .unwrap_or_else(|| vec![Fp::from(0u64); params.row_len()]);
            columns.push((registry_index as u32, NetFpBlob(column)));
            rows.push((registry_index as u32, NetFpBlob(row)));
        }
        (columns, rows)
    };
    let (columns_00, rows_00) = source_fragments(0);
    let (columns_11, rows_11) = source_fragments(3);
    let resp = unauth::handle_message(
        &mut store,
        Message::UnauthKappaHalve {
            source_halves: vec![NetSourceHalfState::Diagonal {
                columns_00,
                rows_00,
                columns_11,
                rows_11,
            }],
        },
    );
    assert!(matches!(resp, Response::UnauthKappaHalve { ok: true }));
    assert_eq!(store.params.need, kappa);
    assert_eq!(store.params.n_grid, params.n_grid);
    assert_eq!(store.segments(), 2);
    assert_eq!(store.len(), initial_len);
    assert!(store.coded.equals_for_tests(&params, &source.coded));
    assert_reads(&mut store);
}

#[test]
fn net_message_roundtrip_unauth_kappa_double_preserves_boundaries_twice() {
    let kappa: usize = 2;
    let n_grid = kappa * kappa;
    let initial_len: u64 = (16 * n_grid) as u64;
    let g = 8;
    let shard_registry = build_shard_registry_deterministic(g);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = CodedArrayConfig {
        kappa,
        shard_count: g,
        shard_registry,
        party_registry_indices,
        seed: 22,
    };
    let params = build_coded_params(&cfg);
    assert_eq!(params.n_grid, n_grid);

    let mut store = UnauthStore::new(params.clone());
    store.extend(initial_len).expect("extend");
    let resp = unauth::handle_message(
        &mut store,
        Message::UnauthApplyDeltas {
            deltas: vec![
                (
                    (n_grid - 1) as u64,
                    paper_a_optimized::net::wire::NetFp(Fp::from(11u64)),
                ),
                (
                    n_grid as u64,
                    paper_a_optimized::net::wire::NetFp(Fp::from(13u64)),
                ),
                (
                    (2 * n_grid - 1) as u64,
                    paper_a_optimized::net::wire::NetFp(Fp::from(17u64)),
                ),
                (
                    (2 * n_grid) as u64,
                    paper_a_optimized::net::wire::NetFp(Fp::from(19u64)),
                ),
                (
                    (4 * n_grid) as u64,
                    paper_a_optimized::net::wire::NetFp(Fp::from(23u64)),
                ),
                (
                    (16 * n_grid - 1) as u64,
                    paper_a_optimized::net::wire::NetFp(Fp::from(29u64)),
                ),
            ],
            mode: paper_a_optimized::net::wire::NetDeltaMode::from(DeltaMode::Aggregate),
        },
    );
    assert!(matches!(resp, Response::UnauthApplyDeltas { ok: true }));
    let expected: Vec<_> = (0..store.len())
        .map(|idx| {
            let bytes = store.read(idx).expect("read source");
            paper_a_optimized::decode_value(bytes)
        })
        .collect();

    let boundary_indices = [
        (n_grid - 1) as u64,
        n_grid as u64,
        (2 * n_grid - 1) as u64,
        (2 * n_grid) as u64,
        (4 * n_grid - 1) as u64,
        (4 * n_grid) as u64,
        (16 * n_grid - 1) as u64,
    ];
    let assert_boundary_reads = |store: &mut UnauthStore, expected: &[Fp]| {
        for &idx in &boundary_indices {
            let resp = unauth::handle_message(store, Message::UnauthRead { idx });
            let Response::UnauthRead { values } = resp else {
                panic!("unexpected response");
            };
            let shares = shares_from_values(&store.params.party_registry_indices[0], &values);
            let value = decode_from_shares(&store.params, idx, &shares);
            assert_eq!(value, expected[idx as usize]);
        }
    };
    assert_boundary_reads(&mut store, &expected);

    let resp = unauth::handle_message(&mut store, Message::UnauthKappaDouble);
    assert!(matches!(resp, Response::UnauthKappaDouble { ok: true }));
    assert_eq!(store.params.n_grid, 4 * n_grid);
    assert_eq!(store.params.need, 2 * kappa);
    assert_eq!(store.segments(), 4);
    assert_boundary_reads(&mut store, &expected);

    let resp = unauth::handle_message(&mut store, Message::UnauthKappaDouble);
    assert!(matches!(resp, Response::UnauthKappaDouble { ok: true }));
    assert_eq!(store.params.n_grid, 16 * n_grid);
    assert_eq!(store.params.need, 4 * kappa);
    assert_eq!(store.segments(), 1);
    assert_boundary_reads(&mut store, &expected);
}
