use paper_a_optimized::coded_array::params::ROWS_PER_CLAIM;
use paper_a_optimized::unauth::client::{ClientError, UnauthClient, UnauthTransport};
use paper_a_optimized::unauth::service::{UnauthRequest, UnauthResponse, UnauthService};
use paper_a_optimized::unauth::{
    ClaimBundle, DeltaMode, PartialStitchEvals, SourceHalfState, UnauthStore,
    decode_from_shares_with_error_correction, logical_offset_to_grid_index, shard_value_from_store,
};
use paper_a_optimized::{
    CodedArrayConfig, Fp, RowShare, ShardShare, Value256, build_coded_params,
    build_shard_registry_deterministic, decode_value, encode_value,
};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct LocalServer {
    party_id: u64,
    corrupt: bool,
    store: Arc<Mutex<UnauthStore>>,
}

#[derive(Clone)]
struct CorruptingTransport;

fn corrupt_share(share: &mut ShardShare) {
    if let Some(mut value) = share.value() {
        value += Fp::from(10_000u64 + share.registry_index() as u64);
        *share = ShardShare::new(share.registry_index(), Some(value));
    }
}

fn corrupt_row(row: &mut RowShare, party_id: u64) {
    *row = RowShare::new(
        row.row_label(),
        row.value() + Fp::from(20_000u64 + party_id),
    );
}

impl UnauthTransport for CorruptingTransport {
    type Server = LocalServer;

    fn server_label(&self, server: &Self::Server) -> String {
        server.party_id.to_string()
    }

    fn unauth_get_len(&self, server: &Self::Server) -> Result<u64, ClientError> {
        match server
            .store
            .lock()
            .expect("store lock")
            .handle_request(UnauthRequest::GetLen)
        {
            UnauthResponse::Len { len } => Ok(len),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "unexpected len response: {other:?}"
            ))),
        }
    }

    fn unauth_read(&self, server: &Self::Server, idx: u64) -> Result<Vec<ShardShare>, ClientError> {
        match server
            .store
            .lock()
            .expect("store lock")
            .handle_request(UnauthRequest::Read { idx })
        {
            UnauthResponse::Read { mut shares } => {
                if server.corrupt {
                    for share in &mut shares {
                        corrupt_share(share);
                    }
                }
                Ok(shares)
            }
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "unexpected read response: {other:?}"
            ))),
        }
    }

    fn unauth_multi_read(
        &self,
        server: &Self::Server,
        idxs: &[u64],
    ) -> Result<Vec<Vec<ShardShare>>, ClientError> {
        match server
            .store
            .lock()
            .expect("store lock")
            .handle_request(UnauthRequest::MultiRead {
                idxs: idxs.to_vec(),
            }) {
            UnauthResponse::MultiRead { mut shares } => {
                if server.corrupt {
                    for batch in &mut shares {
                        for share in batch {
                            corrupt_share(share);
                        }
                    }
                }
                Ok(shares)
            }
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "unexpected multi-read response: {other:?}"
            ))),
        }
    }

    fn unauth_read_block(
        &self,
        _server: &Self::Server,
        _block: u64,
    ) -> Result<Vec<RowShare>, ClientError> {
        Err(ClientError::Net("unused in this test".to_string()))
    }

    fn unauth_read_rows_at_z(
        &self,
        _server: &Self::Server,
        _seg: usize,
        _z_eval: Fp,
    ) -> Result<Vec<RowShare>, ClientError> {
        Err(ClientError::Net("unused in this test".to_string()))
    }

    fn unauth_read_rows_batch_at_z(
        &self,
        _server: &Self::Server,
        _seg: usize,
        _z_evals: &[Fp],
    ) -> Result<Vec<Vec<RowShare>>, ClientError> {
        Err(ClientError::Net("unused in this test".to_string()))
    }

    fn unauth_read_rows_matrix(
        &self,
        server: &Self::Server,
        requests: &[(usize, Vec<Fp>)],
    ) -> Result<Vec<Vec<Vec<RowShare>>>, ClientError> {
        match server.store.lock().expect("store lock").handle_request(
            UnauthRequest::ReadRowsMatrix {
                requests: requests
                    .iter()
                    .map(
                        |(seg, z_evals)| paper_a_optimized::unauth::service::RowReadRequest {
                            seg: *seg,
                            z_evals: z_evals.clone(),
                        },
                    )
                    .collect(),
            },
        ) {
            UnauthResponse::ReadRowsMatrix { mut rows } => {
                if server.corrupt {
                    for batches in &mut rows {
                        for batch in batches {
                            for row in batch {
                                corrupt_row(row, server.party_id);
                            }
                        }
                    }
                }
                Ok(rows)
            }
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "unexpected row-matrix response: {other:?}"
            ))),
        }
    }

    fn unauth_repair_shard(
        &self,
        _server: &Self::Server,
        _seg: usize,
        _registry_index: usize,
    ) -> Result<Vec<Fp>, ClientError> {
        Err(ClientError::Net("unused in this test".to_string()))
    }

    fn unauth_eval_shards_at_row_labels(
        &self,
        server: &Self::Server,
        seg: usize,
        col_labels: &[Fp],
        row_labels: &[Fp],
    ) -> Result<Vec<Vec<Fp>>, ClientError> {
        match server.store.lock().expect("store lock").handle_request(
            UnauthRequest::EvalShardsAtRowLabels {
                seg,
                col_labels: col_labels.to_vec(),
                row_labels: row_labels.to_vec(),
            },
        ) {
            UnauthResponse::EvalShardsAtRowLabels { mut values } => {
                if server.corrupt {
                    for batch in &mut values {
                        for value in batch {
                            *value += Fp::from(30_000u64 + server.party_id);
                        }
                    }
                }
                Ok(values)
            }
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "unexpected shard-eval response: {other:?}"
            ))),
        }
    }

    fn unauth_apply_deltas(
        &self,
        _server: &Self::Server,
        _deltas: &[(u64, Fp)],
        _mode: DeltaMode,
    ) -> Result<bool, ClientError> {
        Err(ClientError::Net("unused in this test".to_string()))
    }

    fn unauth_extend(&self, _server: &Self::Server, _new_len: u64) -> Result<bool, ClientError> {
        Err(ClientError::Net("unused in this test".to_string()))
    }

    fn unauth_kappa_double(&self, server: &Self::Server) -> Result<bool, ClientError> {
        match server
            .store
            .lock()
            .expect("store lock")
            .handle_request(UnauthRequest::KappaDouble)
        {
            UnauthResponse::KappaDouble { ok } => Ok(ok),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "unexpected kappa-double response: {other:?}"
            ))),
        }
    }

    fn unauth_kappa_halve(
        &self,
        server: &Self::Server,
        source_halves: &[SourceHalfState],
    ) -> Result<bool, ClientError> {
        match server
            .store
            .lock()
            .expect("store lock")
            .handle_request(UnauthRequest::KappaHalve {
                source_halves: source_halves.to_vec(),
            }) {
            UnauthResponse::KappaHalve { ok } => Ok(ok),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "unexpected kappa-halve response: {other:?}"
            ))),
        }
    }

    fn unauth_eval_partial_stitches(
        &self,
        server: &Self::Server,
        segs: &[usize],
        col_labels: &[Fp],
        row_labels: &[Fp],
    ) -> Result<Vec<Vec<PartialStitchEvals>>, ClientError> {
        match server.store.lock().expect("store lock").handle_request(
            UnauthRequest::EvalPartialStitches {
                segs: segs.to_vec(),
                col_labels: col_labels.to_vec(),
                row_labels: row_labels.to_vec(),
            },
        ) {
            UnauthResponse::EvalPartialStitches { mut evals } => {
                if server.corrupt {
                    for helper in evals.iter_mut().flatten() {
                        for value in helper
                            .row_evals
                            .iter_mut()
                            .chain(helper.column_evals.iter_mut())
                        {
                            *value += Fp::from(40_000u64 + server.party_id);
                        }
                    }
                }
                Ok(evals)
            }
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "unexpected partial-stitch response: {other:?}"
            ))),
        }
    }
}

fn params_for_need_two() -> paper_a_optimized::CodedArrayParams {
    let kappa = 2usize;
    let shard_count = 6usize;
    let cfg = CodedArrayConfig {
        kappa,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count),
        party_registry_indices: (0..shard_count).map(|idx| vec![idx]).collect(),
        seed: 77,
    };
    let params = build_coded_params(&cfg);
    assert_eq!(params.need, 2);
    assert_eq!(params.shard_registry.len(), 3 * params.need);
    params
}

/// Base-level parameters with threshold 2 and the given claim assignment.
fn params_for_parties(
    party_registry_indices: Vec<Vec<usize>>,
    seed: u64,
) -> paper_a_optimized::CodedArrayParams {
    let kappa = 2usize;
    let shard_count = party_registry_indices.iter().map(Vec::len).sum();
    let cfg = CodedArrayConfig {
        kappa,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count),
        party_registry_indices,
        seed,
    };
    let params = build_coded_params(&cfg);
    assert_eq!(params.need, 2);
    params
}

/// Values of segment `seg` such that public index `idx` holds `base + idx`.
fn segment_with_indexed_values(
    params: &paper_a_optimized::CodedArrayParams,
    seg: usize,
    base: u64,
) -> Vec<Value256> {
    (0..params.n_grid)
        .map(|offset| encode_value(Fp::from(base + (seg * params.n_grid + offset) as u64)))
        .collect()
}

/// One server per party, each holding the party's claims. Parties listed in
/// `corrupt_parties` corrupt every helper reply they send.
fn servers_with_indexed_values(
    params: &paper_a_optimized::CodedArrayParams,
    len: u64,
    base: u64,
    corrupt_parties: &[usize],
) -> Vec<LocalServer> {
    params
        .party_registry_indices
        .iter()
        .enumerate()
        .map(|(party_id, owned_indices)| {
            let mut store = UnauthStore::new(params.clone());
            store
                .extend_with_segments_filtered(len, owned_indices, |seg| {
                    segment_with_indexed_values(params, seg, base)
                })
                .expect("extend filtered store");
            LocalServer {
                party_id: party_id as u64,
                corrupt: corrupt_parties.contains(&party_id),
                store: Arc::new(Mutex::new(store)),
            }
        })
        .collect()
}

fn honest_views(servers: &[LocalServer]) -> Vec<LocalServer> {
    servers
        .iter()
        .map(|server| LocalServer {
            party_id: server.party_id,
            corrupt: false,
            store: Arc::clone(&server.store),
        })
        .collect()
}

fn clone_stores(servers: &[LocalServer]) -> Vec<UnauthStore> {
    servers
        .iter()
        .map(|server| server.store.lock().expect("store lock").clone())
        .collect()
}

fn assert_stores_match(servers: &[LocalServer], expected: &[UnauthStore]) {
    assert_eq!(servers.len(), expected.len());
    for (server, expected) in servers.iter().zip(expected.iter()) {
        let store = server.store.lock().expect("store lock");
        assert_eq!(store.params.need, expected.params.need);
        assert_eq!(store.len(), expected.len());
        assert_eq!(store.segments(), expected.segments());
        assert!(
            store
                .coded
                .equals_for_tests(&expected.params, &expected.coded),
            "party {} coded state differs",
            server.party_id
        );
    }
}

fn segment_with_raw_value(raw_index: usize, value: Fp, segment_len: usize) -> Vec<Value256> {
    let mut segment = vec![encode_value(Fp::from(0u64)); segment_len];
    segment[raw_index] = encode_value(value);
    segment
}

fn shares_for_raw_value(
    params: &paper_a_optimized::CodedArrayParams,
    raw_index: usize,
    value: Fp,
) -> Vec<ShardShare> {
    let segment = segment_with_raw_value(raw_index, value, params.segment_len);
    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments(params.segment_len as u64, |_| segment.clone())
        .expect("extend full store");

    let w_index = params.w_index_for_grid_index(raw_index);
    let mut shares = Vec::new();
    for (registry_index, entry) in params.shard_registry.iter().copied().enumerate() {
        let value =
            shard_value_from_store(params, &store.coded, entry, 0, w_index).expect("shard value");
        shares.push(ShardShare::new(registry_index, Some(value)));
    }
    shares
}

#[test]
fn correcting_decode_handles_zero_partial_and_max_bad_helpers_with_max_budget() {
    let params = params_for_need_two();
    let idx = 2u64;
    let j = logical_offset_to_grid_index(&params, idx as usize);
    let expected = Fp::from(42u64);
    let base_shares = shares_for_raw_value(&params, j, expected);

    for bad_registry_indices in [vec![], vec![1usize], vec![1usize, 5usize]] {
        let mut shares = base_shares.clone();
        for share in &mut shares {
            if bad_registry_indices.contains(&share.registry_index()) {
                corrupt_share(share);
            }
        }

        let decoded = decode_from_shares_with_error_correction(&params, idx, &shares, params.need)
            .expect("decode");
        assert_eq!(decoded.share_count, 3 * params.need);
        assert_eq!(decoded.value, expected);
        assert_eq!(decoded.bad_registry_indices, bad_registry_indices);
    }
}

#[test]
fn correcting_decode_filters_need_bad_helpers_from_three_need() {
    let params = params_for_need_two();
    let idx = 2u64;
    let j = logical_offset_to_grid_index(&params, idx as usize);
    let expected = Fp::from(42u64);
    let segment = segment_with_raw_value(j, expected, params.segment_len);
    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments(params.segment_len as u64, |_| segment.clone())
        .expect("extend full store");

    let w_index = params.w_index_for_grid_index(j);
    let mut shares = Vec::new();
    for (registry_index, entry) in params.shard_registry.iter().copied().enumerate() {
        let value =
            shard_value_from_store(&params, &store.coded, entry, 0, w_index).expect("shard value");
        shares.push(ShardShare::new(registry_index, Some(value)));
    }

    let bad_registry_indices = vec![1usize, 5usize];
    for share in &mut shares {
        if bad_registry_indices.contains(&share.registry_index()) {
            corrupt_share(share);
        }
    }

    let decoded = decode_from_shares_with_error_correction(&params, idx, &shares, params.need)
        .expect("decode");
    assert_eq!(decoded.share_count, 3 * params.need);
    assert_eq!(decoded.value, expected);
    assert_eq!(decoded.bad_registry_indices, bad_registry_indices);
}

#[test]
fn unauth_client_correcting_read_queries_three_need_and_recovers() {
    let params = params_for_need_two();
    let idx = 2u64;
    let j = logical_offset_to_grid_index(&params, idx as usize);
    let expected = Fp::from(123u64);
    let segment = segment_with_raw_value(j, expected, params.segment_len);

    let mut servers = Vec::new();
    for registry_index in 0..params.shard_registry.len() {
        let mut store = UnauthStore::new(params.clone());
        store
            .extend_with_segments_filtered(params.segment_len as u64, &[registry_index], |_| {
                segment.clone()
            })
            .expect("extend filtered store");
        servers.push(LocalServer {
            party_id: registry_index as u64,
            corrupt: registry_index < params.need,
            store: Arc::new(Mutex::new(store)),
        });
    }

    let mut client = UnauthClient::new_with_owned_indices(
        CorruptingTransport,
        params.clone(),
        servers[0].clone(),
        servers.clone(),
        (0..params.shard_registry.len())
            .map(|idx| vec![idx])
            .collect(),
        params.segment_len as u64,
    );

    let value = client
        .read_correcting(idx, params.need)
        .expect("correcting read");
    assert_eq!(decode_value(value), expected);

    let values = client
        .multi_read_correcting(&[idx], params.need)
        .expect("correcting multi-read");
    assert_eq!(decode_value(values[0]), expected);
}

#[test]
fn unauth_client_correcting_repairs_filter_bad_helpers() {
    let params = params_for_need_two();
    let j = logical_offset_to_grid_index(&params, 2usize);
    let segment = segment_with_raw_value(j, Fp::from(777u64), params.segment_len);

    let mut servers = Vec::new();
    for registry_index in 0..params.shard_registry.len() {
        let mut store = UnauthStore::new(params.clone());
        store
            .extend_with_segments_filtered(params.segment_len as u64, &[registry_index], |_| {
                segment.clone()
            })
            .expect("extend filtered store");
        servers.push(LocalServer {
            party_id: registry_index as u64,
            corrupt: registry_index < params.need,
            store: Arc::new(Mutex::new(store)),
        });
    }

    let mut full_store = UnauthStore::new(params.clone());
    full_store
        .extend_with_segments_filtered(
            params.segment_len as u64,
            &(0..params.shard_registry.len()).collect::<Vec<_>>(),
            |_| segment.clone(),
        )
        .expect("full extend");
    let expected_shard = full_store
        .repair_primary_shard_from_rows(0, params.shard_registry[5])
        .expect("expected shard repair")
        .into_iter()
        .map(encode_value)
        .collect::<Vec<_>>();
    let expected_row = full_store
        .repair_row_from_shards(0, params.row_labels[0])
        .expect("expected row repair")
        .into_iter()
        .map(encode_value)
        .collect::<Vec<_>>();

    let client = UnauthClient::new_with_owned_indices(
        CorruptingTransport,
        params.clone(),
        servers[0].clone(),
        servers.clone(),
        (0..params.shard_registry.len())
            .map(|idx| vec![idx])
            .collect(),
        params.segment_len as u64,
    );

    let repaired_shard = client
        .repair_shard_from_network_correcting(0, 5, params.need * ROWS_PER_CLAIM)
        .expect("correcting shard repair");
    assert_eq!(repaired_shard, expected_shard);

    let repaired_row = client
        .repair_row_from_network_correcting(0, 0, params.need)
        .expect("correcting row repair");
    assert_eq!(repaired_row, expected_row);
}

#[test]
fn unauth_client_correcting_kappa_halve_filters_bad_helper_data() {
    let params = params_for_parties((0..6).map(|idx| vec![idx]).collect(), 77);
    let idx = 2u64;
    let base = 999u64;
    // Six source segments: the second group of four is stitched with padding.
    let len = (6 * params.segment_len) as u64;
    let servers = servers_with_indexed_values(&params, len, base, &[0]);
    let source_stores = clone_stores(&servers);

    let mut client = UnauthClient::new_with_owned_indices(
        CorruptingTransport,
        params.clone(),
        servers[0].clone(),
        servers.clone(),
        params.party_registry_indices.clone(),
        len,
    );

    client.kappa_double().expect("local kappa-double");
    assert_eq!(client.params.need, 2 * params.need);
    let kappa_doubled_stores = clone_stores(&servers);

    assert!(
        matches!(
            client.kappa_halve_correcting(2),
            Err(ClientError::NotEnoughShards)
        ),
        "correcting two bad helper claims needs kappa + 4 helper claims, the cluster has kappa + 2",
    );
    assert_stores_match(&servers, &kappa_doubled_stores);

    client
        .kappa_halve_correcting(1)
        .expect("correcting kappa-halve");
    assert_eq!(client.params.need, params.need);
    assert_stores_match(&servers, &source_stores);

    let clean_servers = honest_views(&servers);
    let mut clean_client = UnauthClient::new_with_owned_indices(
        CorruptingTransport,
        client.params.clone(),
        clean_servers[0].clone(),
        clean_servers,
        client.params.party_registry_indices.clone(),
        len,
    );
    let kappa_halved_value = clean_client.read(idx).expect("read after kappa-halve");
    assert_eq!(decode_value(kappa_halved_value), Fp::from(base + idx));
}

#[test]
fn unauth_client_correcting_read_after_kappa_double_filters_bad_helper_data() {
    let params = params_for_parties((0..6).map(|idx| vec![idx]).collect(), 77);
    let base = 1001u64;
    let len = (4 * params.segment_len) as u64;
    let servers = servers_with_indexed_values(&params, len, base, &[0]);

    let mut client = UnauthClient::new_with_owned_indices(
        CorruptingTransport,
        params.clone(),
        servers[0].clone(),
        servers.clone(),
        params.party_registry_indices.clone(),
        len,
    );

    client.kappa_double().expect("local kappa-double");
    assert_eq!(client.params.need, 2 * params.need);

    assert!(
        client.read_correcting(2, 2).is_err(),
        "correcting two bad shard points needs kappa + 4 helper claims, the cluster has kappa + 2",
    );
    let idxs = (0..len).collect::<Vec<_>>();
    for &idx in &idxs {
        let value = client
            .read_correcting(idx, 1)
            .expect("correcting read after kappa-double");
        assert_eq!(decode_value(value), Fp::from(base + idx));
    }
    let values = client
        .multi_read_correcting(&idxs, 1)
        .expect("correcting multi-read after kappa-double");
    for (&idx, value) in idxs.iter().zip(values) {
        assert_eq!(decode_value(value), Fp::from(base + idx));
    }
}

#[test]
fn unauth_client_correcting_kappa_halve_after_kappa_double_uses_multi_claim_servers() {
    let params = params_for_parties(
        vec![
            vec![0, 1],
            vec![2],
            vec![3],
            vec![4],
            vec![5],
            vec![6],
            vec![7],
        ],
        42,
    );

    let idx = 5u64;
    let base = 4242u64;
    let len = (8 * params.segment_len) as u64;
    let servers = servers_with_indexed_values(&params, len, base, &[0]);
    let source_stores = clone_stores(&servers);

    let mut client = UnauthClient::new_with_owned_indices(
        CorruptingTransport,
        params.clone(),
        servers[0].clone(),
        servers.clone(),
        params.party_registry_indices.clone(),
        len,
    );

    client.kappa_double().expect("local kappa-double");
    assert_eq!(client.params.need, 4);
    let kappa_doubled_stores = clone_stores(&servers);

    // The first five servers hold exactly kappa + 2 claims, two of them on
    // the corrupt server.
    let mut short_client = UnauthClient::new_with_owned_indices(
        CorruptingTransport,
        client.params.clone(),
        servers[0].clone(),
        servers[..5].to_vec(),
        client.params.party_registry_indices[..5].to_vec(),
        len,
    );
    assert!(
        short_client.kappa_halve_correcting(1).is_err(),
        "the corrupt helper contributes two helper claims, so one RS-point error is insufficient",
    );
    assert_stores_match(&servers, &kappa_doubled_stores);

    client
        .kappa_halve_correcting(2)
        .expect("correcting kappa-halve with two bad helper claims from one helper server");
    assert_eq!(client.params.s, params.s);
    assert_stores_match(&servers, &source_stores);

    let clean_servers = honest_views(&servers);
    let mut clean_client = UnauthClient::new_with_owned_indices(
        CorruptingTransport,
        client.params.clone(),
        clean_servers[0].clone(),
        clean_servers,
        client.params.party_registry_indices.clone(),
        len,
    );
    let kappa_halved_value = clean_client
        .read(idx)
        .expect("read after kappa-double and kappa-halve");
    assert_eq!(decode_value(kappa_halved_value), Fp::from(base + idx));
}

#[test]
fn correcting_claim_repair_handles_multi_claim_servers_with_capped_row_provenance() {
    let params = params_for_parties(
        vec![
            vec![0, 1, 2, 3, 4],
            vec![5],
            vec![6],
            vec![7],
            vec![8],
            vec![9],
        ],
        91,
    );
    let j = logical_offset_to_grid_index(&params, 2usize);
    let segment = segment_with_raw_value(j, Fp::from(2024u64), params.segment_len);
    let mut servers = Vec::new();
    for (party_id, owned_indices) in params.party_registry_indices.iter().enumerate() {
        let mut store = UnauthStore::new(params.clone());
        store
            .extend_with_segments_filtered(params.segment_len as u64, owned_indices, |_| {
                segment.clone()
            })
            .expect("extend filtered store");
        servers.push(LocalServer {
            party_id: party_id as u64,
            corrupt: false,
            store: Arc::new(Mutex::new(store)),
        });
    }

    let client = UnauthClient::new_with_owned_indices(
        CorruptingTransport,
        params.clone(),
        servers[0].clone(),
        servers.clone(),
        params.party_registry_indices.clone(),
        params.segment_len as u64,
    );
    let claim = ClaimBundle::exact_for_registry_index(&params, 9).expect("claim");
    let repaired = client
        .repair_claim_segment_fp_correcting(0, &claim, 1)
        .expect("claim repair with capped primary-row planning");
    assert_eq!(repaired.shards.len(), 1);
    assert_eq!(repaired.shards[0].0, 9);
    assert_eq!(repaired.shards[0].1.len(), params.s);
    assert_eq!(repaired.rows.len(), ROWS_PER_CLAIM);
    assert_eq!(repaired.rows[0].0, claim.row_label_index());

    let owner = servers[5].store.lock().expect("store lock");
    let entry = params.shard_registry[9];
    let row_key = paper_a_optimized::RowKey::new(entry.key, params.row_labels[9]);
    assert_eq!(
        repaired.shards[0].1.as_slice(),
        owner
            .coded
            .shard_segment(&params, &entry.key, 0)
            .expect("stored shard")
    );
    assert_eq!(
        repaired.rows[0].1.as_slice(),
        owner
            .coded
            .row_segment(&params, &row_key, 0)
            .expect("stored row")
    );
}
