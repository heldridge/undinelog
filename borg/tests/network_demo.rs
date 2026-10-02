//! Message-level demo tests using the shared net protocol types.

use paper_a_optimized::coded_array::params::CodedArrayConfig;
use paper_a_optimized::coded_array::rs::{barycentric_eval, precompute_barycentric};
use paper_a_optimized::net::unauth;
use paper_a_optimized::net::wire::{
    NetFp, NetFpBlob, NetKappaDoubleRowEvalMigration, NetMaybeFpBlob, NetPartnerShardMigration,
    NetRowBatchReply, NetSegmentPrimaryShardReply, UnauthOptimizedReadRequest,
    UnauthOptimizedReadResponse, UnauthReadObjectContribution, UnauthReadObjectId, decode_binary,
    encode_binary,
};
use paper_a_optimized::net::{Message, Response};
use paper_a_optimized::unauth::{
    DeltaMode, UnauthStore, aligned_block_indices, block_alphas,
    logical_offset_to_column_coordinate_for_segment, logical_offset_to_grid_index_for_s,
    semantic_block_position, shard_value_from_vector_batch,
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

fn decode_from_shares_for_layout(
    params: &paper_a_optimized::CodedArrayParams,
    layout_s: usize,
    logical_offset: usize,
    shares: &[(usize, Fp)],
) -> Fp {
    let j = logical_offset_to_grid_index_for_s(params, layout_s, logical_offset);
    let z_eval = params.column_coordinate_for_grid_index(j);
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
    let weights = precompute_barycentric(&z_values);
    barycentric_eval(&weights, &r_values, z_eval)
}

fn decode_from_shares(
    params: &paper_a_optimized::CodedArrayParams,
    logical_offset: usize,
    shares: &[(usize, Fp)],
) -> Fp {
    decode_from_shares_for_layout(params, params.s, logical_offset, shares)
}

fn decode_from_public_shares(
    params: &paper_a_optimized::CodedArrayParams,
    segment: usize,
    logical_offset: usize,
    shares: &[(usize, Fp)],
) -> Fp {
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
    block_alphas(params, pos.block_in_segment)
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
fn net_message_roundtrip_unauth_segment_double_wire_variants() {
    let helper_evals = vec![NetKappaDoubleRowEvalMigration {
        registry_index: 2,
        segments: vec![vec![NetFp(Fp::from(11u64)), NetFp(Fp::from(7u64))]],
    }];
    let msg = Message::UnauthKappaDoubleOneClaimLeft {
        target_s: 8,
        helper_evals: helper_evals.clone(),
    };
    let payload = encode_binary(&msg).expect("encode kappa-double message");
    let decoded: Message = decode_binary(&payload).expect("decode kappa-double message");
    match decoded {
        Message::UnauthKappaDoubleOneClaimLeft {
            target_s,
            helper_evals: decoded_evals,
        } => {
            assert_eq!(target_s, 8);
            assert_eq!(decoded_evals.len(), 1);
            assert_eq!(
                decoded_evals[0].registry_index,
                helper_evals[0].registry_index
            );
            assert_eq!(decoded_evals[0].segments, helper_evals[0].segments);
        }
        other => panic!("unexpected message: {:?}", other),
    }

    let msg = Message::UnauthEvalKappaDoubleTargetAtRowLabelsMatrix {
        target_s: 8,
        segs: vec![0, 1],
        target_registry_indices: vec![2, 4],
        row_labels: NetFpBlob(vec![Fp::from(3u64), Fp::from(5u64)]),
    };
    let payload = encode_binary(&msg).expect("encode kappa-double-eval message");
    let decoded: Message = decode_binary(&payload).expect("decode kappa-double-eval message");
    match decoded {
        Message::UnauthEvalKappaDoubleTargetAtRowLabelsMatrix {
            target_s,
            segs,
            target_registry_indices,
            row_labels,
        } => {
            assert_eq!(target_s, 8);
            assert_eq!(segs, vec![0, 1]);
            assert_eq!(target_registry_indices, vec![2, 4]);
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

    let helper_shards = vec![NetPartnerShardMigration {
        registry_index: 1,
        segments: vec![vec![NetFp(Fp::from(19u64)), NetFp(Fp::from(23u64))]],
    }];
    let msg = Message::UnauthSegmentDouble {
        helper_shards: helper_shards.clone(),
    };
    let payload = encode_binary(&msg).expect("encode segment-double message");
    let decoded: Message = decode_binary(&payload).expect("decode segment-double message");
    match decoded {
        Message::UnauthSegmentDouble {
            helper_shards: decoded_shards,
        } => {
            assert_eq!(decoded_shards.len(), 1);
            assert_eq!(decoded_shards[0].registry_index, 1);
            assert_eq!(decoded_shards[0].segments, helper_shards[0].segments);
        }
        other => panic!("unexpected message: {:?}", other),
    }

    let resp = Response::UnauthKappaDoubleOneClaimLeft { ok: true };
    let payload = encode_binary(&resp).expect("encode kappa-double response");
    let decoded: Response = decode_binary(&payload).expect("decode kappa-double response");
    assert!(matches!(
        decoded,
        Response::UnauthKappaDoubleOneClaimLeft { ok: true }
    ));

    let resp = Response::UnauthEvalKappaDoubleTargetAtRowLabelsMatrix {
        values: NetFpBlob(vec![Fp::from(13u64), Fp::from(17u64)]),
    };
    let payload = encode_binary(&resp).expect("encode kappa-double-eval response");
    let decoded: Response = decode_binary(&payload).expect("decode kappa-double-eval response");
    match decoded {
        Response::UnauthEvalKappaDoubleTargetAtRowLabelsMatrix { values } => {
            assert_eq!(values.0, vec![Fp::from(13u64), Fp::from(17u64)]);
        }
        other => panic!("unexpected response: {:?}", other),
    }

    let resp = Response::UnauthRetainClaims { ok: true };
    let payload = encode_binary(&resp).expect("encode retain response");
    let decoded: Response = decode_binary(&payload).expect("decode retain response");
    assert!(matches!(decoded, Response::UnauthRetainClaims { ok: true }));

    let resp = Response::UnauthSegmentDouble { ok: true };
    let payload = encode_binary(&resp).expect("encode segment-double response");
    let decoded: Response = decode_binary(&payload).expect("decode segment-double response");
    assert!(matches!(
        decoded,
        Response::UnauthSegmentDouble { ok: true }
    ));
}

#[test]
fn net_message_roundtrip_unauth_batch() {
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
    let segment_len: usize = 8;
    let n_grid = segment_len;
    let initial_len: u64 = 8;
    let s = 4;
    let g = 4;
    let shard_registry = build_shard_registry_deterministic(g, s, n_grid);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = CodedArrayConfig {
        segment_len,
        s,
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
    let segment_len: usize = 8;
    let n_grid = segment_len;
    let initial_len: u64 = 8;
    let s = 4;
    let g = 4;
    let shard_registry = build_shard_registry_deterministic(g, s, n_grid);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = CodedArrayConfig {
        segment_len,
        s,
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
    let segment_len: usize = 8;
    let n_grid = segment_len;
    let initial_len: u64 = 8;
    let s = 4;
    let g = 4;
    let shard_registry = build_shard_registry_deterministic(g, s, n_grid);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = CodedArrayConfig {
        segment_len,
        s,
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
    let expected_eval0 =
        shard_value_from_vector_batch(&params, params.shard_registry[0], &expected0, &row_alphas)
            .into_iter()
            .map(|value| value.expect("row-label eval"))
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
    let segment_len: usize = 8;
    let n_grid = segment_len;
    let initial_len: u64 = 8;
    let s = 4;
    let g = 4;
    let shard_registry = build_shard_registry_deterministic(g, s, n_grid);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = CodedArrayConfig {
        segment_len,
        s,
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
    assert!(row_batch_len(&rows) >= params.s);
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
    let segment_len: usize = 8;
    let n_grid = segment_len;
    let initial_len: u64 = 8;
    let s = 4;
    let g = 4;
    let shard_registry = build_shard_registry_deterministic(g, s, n_grid);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = CodedArrayConfig {
        segment_len,
        s,
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
    assert!(row_count >= params.s);
}

#[test]
fn net_message_roundtrip_unauth_read_rows_matrix() {
    let segment_len: usize = 8;
    let n_grid = segment_len;
    let initial_len: u64 = 8;
    let s = 4;
    let g = 4;
    let shard_registry = build_shard_registry_deterministic(g, s, n_grid);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = CodedArrayConfig {
        segment_len,
        s,
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
    assert!(row_count >= params.s);
}

#[test]
fn net_message_roundtrip_unauth_migrate() {
    let segment_len: usize = 8;
    let n_grid = segment_len;
    let initial_len: u64 = 7;
    let s = 4;
    let g = 4;
    let shard_registry = build_shard_registry_deterministic(g, s, n_grid);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = CodedArrayConfig {
        segment_len,
        s,
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
            ],
            mode: paper_a_optimized::net::wire::NetDeltaMode::from(DeltaMode::Aggregate),
        },
    );
    assert!(matches!(resp, Response::UnauthApplyDeltas { ok: true }));
    let source = store.clone();
    let expected: Vec<_> = (0..source.len())
        .map(|idx| {
            let bytes = source.read(idx).expect("read source");
            paper_a_optimized::decode_value(bytes)
        })
        .collect();

    let resp = unauth::handle_message(
        &mut store,
        Message::UnauthMigrate {
            target_s: 8,
            partner_shards: Vec::new(),
            helper_rows: Vec::new(),
        },
    );
    assert!(matches!(resp, Response::UnauthMigrate { ok: true }));

    let resp = unauth::handle_message(&mut store, Message::UnauthRead { idx: 4 });
    let Response::UnauthRead { values } = resp else {
        panic!("unexpected response");
    };
    let shares = shares_from_values(&store.params.party_registry_indices[0], &values);
    let value = decode_from_shares(&store.params, 4, &shares);
    assert_eq!(value, expected[4]);

    let resp = unauth::handle_message(&mut store, Message::UnauthRead { idx: 2 });
    let Response::UnauthRead { values } = resp else {
        panic!("unexpected response");
    };
    let shares = shares_from_values(&store.params.party_registry_indices[0], &values);
    let value = decode_from_shares(&store.params, 2, &shares);
    assert_eq!(value, expected[2]);
    assert_eq!(value, Fp::from(5u64));
}

#[test]
fn net_message_roundtrip_unauth_segment_double_preserves_boundaries_twice() {
    let segment_len: usize = 8;
    let n_grid = segment_len;
    let initial_len: u64 = (4 * n_grid) as u64;
    let s = 4;
    let g = 8;
    let shard_registry = build_shard_registry_deterministic(g, s, n_grid);
    let party_registry_indices = vec![(0..g).collect()];
    let cfg = CodedArrayConfig {
        segment_len,
        s,
        shard_count: g,
        shard_registry,
        party_registry_indices,
        seed: 22,
    };
    let params = build_coded_params(&cfg);

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
    ];
    let assert_boundary_reads = |store: &mut UnauthStore, expected: &[Fp]| {
        for &idx in &boundary_indices {
            let resp = unauth::handle_message(store, Message::UnauthRead { idx });
            let Response::UnauthRead { values } = resp else {
                panic!("unexpected response");
            };
            let shares = shares_from_values(&store.params.party_registry_indices[0], &values);
            let logical_offset = (idx % store.params.n_grid as u64) as usize;
            let segment = (idx / store.params.n_grid as u64) as usize;
            let value = decode_from_public_shares(&store.params, segment, logical_offset, &shares);
            assert_eq!(value, expected[idx as usize]);
        }
    };

    let resp = unauth::handle_message(
        &mut store,
        Message::UnauthSegmentDouble {
            helper_shards: Vec::new(),
        },
    );
    assert!(matches!(resp, Response::UnauthSegmentDouble { ok: true }));
    assert_eq!(store.params.n_grid, 2 * n_grid);
    assert_eq!(store.params.s, s);
    assert_boundary_reads(&mut store, &expected);

    let resp = unauth::handle_message(
        &mut store,
        Message::UnauthSegmentDouble {
            helper_shards: Vec::new(),
        },
    );
    assert!(matches!(resp, Response::UnauthSegmentDouble { ok: true }));
    assert_eq!(store.params.n_grid, 4 * n_grid);
    assert_eq!(store.params.s, s);
    assert_boundary_reads(&mut store, &expected);
}
