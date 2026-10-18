use paper_a_optimized::unauth::client::{ClientError, UnauthClient, UnauthTransport};
use paper_a_optimized::unauth::service::{UnauthRequest, UnauthResponse, UnauthService};
use paper_a_optimized::unauth::{
    ClaimBundle, DeltaMode, HelperRowMigration, PartnerShardMigration, UnauthStore,
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

    fn unauth_migrate(
        &self,
        server: &Self::Server,
        target_s: usize,
        partner_shards: &[PartnerShardMigration],
        helper_rows: &[HelperRowMigration],
    ) -> Result<bool, ClientError> {
        match server
            .store
            .lock()
            .expect("store lock")
            .handle_request(UnauthRequest::Migrate {
                target_s,
                partner_shards: partner_shards.to_vec(),
                helper_rows: helper_rows.to_vec(),
            }) {
            UnauthResponse::Migrate { ok } => Ok(ok),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "unexpected migrate response: {other:?}"
            ))),
        }
    }
}

fn params_for_need_two() -> paper_a_optimized::CodedArrayParams {
    let segment_len = 8usize;
    let n_grid = segment_len;
    let s = 4usize;
    let shard_count = 6usize;
    let cfg = CodedArrayConfig {
        segment_len,
        s,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count, s, n_grid),
        party_registry_indices: (0..shard_count).map(|idx| vec![idx]).collect(),
        seed: 77,
    };
    let params = build_coded_params(&cfg);
    assert_eq!(params.need, 2);
    assert_eq!(params.shard_registry.len(), 3 * params.need);
    params
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

    let alpha = params.grid_point(raw_index);
    let mut shares = Vec::new();
    for (registry_index, entry) in params.shard_registry.iter().copied().enumerate() {
        let value =
            shard_value_from_store(params, &store.coded, entry, 0, alpha).expect("shard value");
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

        let decoded = decode_from_shares_with_error_correction(&params, j, &shares, params.need)
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

    let alpha = params.grid_point(j);
    let mut shares = Vec::new();
    for (registry_index, entry) in params.shard_registry.iter().copied().enumerate() {
        let value =
            shard_value_from_store(&params, &store.coded, entry, 0, alpha).expect("shard value");
        shares.push(ShardShare::new(registry_index, Some(value)));
    }

    let bad_registry_indices = vec![1usize, 5usize];
    for share in &mut shares {
        if bad_registry_indices.contains(&share.registry_index()) {
            corrupt_share(share);
        }
    }

    let decoded =
        decode_from_shares_with_error_correction(&params, j, &shares, params.need).expect("decode");
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
        .repair_shard_from_network_correcting(0, 5, params.need * params.rows_per_shard)
        .expect("correcting shard repair");
    assert_eq!(repaired_shard, expected_shard);

    let repaired_row = client
        .repair_row_from_network_correcting(0, 0, params.need)
        .expect("correcting row repair");
    assert_eq!(repaired_row, expected_row);
}

#[test]
fn unauth_client_correcting_migration_filters_bad_helper_data() {
    let params = params_for_need_two();
    let idx = 2u64;
    let expected = Fp::from(999u64);
    let segment = vec![encode_value(expected); params.segment_len];

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
            corrupt: registry_index == 0,
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

    assert!(
        client.migrate_correcting(params.s * 2, 1).is_err(),
        "the corrupt helper contributes two row points, so one RS-point error is insufficient",
    );
    client
        .migrate_correcting(params.s * 2, 2)
        .expect("correcting kappa-halve migration");
    let clean_servers = servers
        .iter()
        .map(|server| LocalServer {
            party_id: server.party_id,
            corrupt: false,
            store: Arc::clone(&server.store),
        })
        .collect::<Vec<_>>();
    let mut clean_client = UnauthClient::new_with_owned_indices(
        CorruptingTransport,
        client.params.clone(),
        clean_servers[0].clone(),
        clean_servers,
        client.params.party_registry_indices.clone(),
        params.segment_len as u64,
    );
    let migrated_value = clean_client
        .read(idx)
        .expect("correcting read after migration");
    assert_eq!(decode_value(migrated_value), expected);
}

#[test]
fn unauth_client_correcting_kappa_double_filters_bad_helper_data() {
    let params = params_for_need_two();
    let idx = 2u64;
    let expected = Fp::from(1001u64);
    let segment = vec![encode_value(expected); params.segment_len];

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
            corrupt: registry_index == 0,
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

    assert!(
        client.migrate_correcting(params.s / 2, 1).is_err(),
        "the corrupt helper contributes two row points, so one RS-point error is insufficient",
    );
    client
        .migrate_correcting(params.s / 2, 2)
        .expect("correcting kappa-double migration");
    let clean_servers = servers
        .iter()
        .map(|server| LocalServer {
            party_id: server.party_id,
            corrupt: false,
            store: Arc::clone(&server.store),
        })
        .collect::<Vec<_>>();
    let mut clean_client = UnauthClient::new_with_owned_indices(
        CorruptingTransport,
        client.params.clone(),
        clean_servers[0].clone(),
        clean_servers,
        client.params.party_registry_indices.clone(),
        params.segment_len as u64,
    );
    let migrated_value = clean_client
        .read(idx)
        .expect("correcting read after kappa-double");
    assert_eq!(decode_value(migrated_value), expected);
}

#[test]
fn unauth_client_correcting_kappa_halve_after_kappa_double_uses_multi_shard_servers() {
    let segment_len = 128usize;
    let n_grid = segment_len;
    let s = 16usize;
    let shard_count = 10usize;
    let cfg = CodedArrayConfig {
        segment_len,
        s,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count, s, n_grid),
        party_registry_indices: (0..shard_count).map(|idx| vec![idx]).collect(),
        seed: 42,
    };
    let params = build_coded_params(&cfg);
    assert_eq!(params.need, 8);

    let idx = 5u64;
    let expected = Fp::from(4242u64);
    let segment = vec![encode_value(expected); params.segment_len];
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
            corrupt: registry_index == 0,
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

    client
        .migrate_correcting(params.s / 2, 2)
        .expect("correcting kappa-double migration");
    assert_eq!(client.params.need, 16);
    assert_eq!(client.params.party_registry_indices[0].len(), 2);

    client
        .kappa_halve_correcting(2)
        .expect("correcting kappa-halve with two bad shard points from one helper server");
    assert_eq!(client.params.s, params.s);

    let clean_servers = servers
        .iter()
        .map(|server| LocalServer {
            party_id: server.party_id,
            corrupt: false,
            store: Arc::clone(&server.store),
        })
        .collect::<Vec<_>>();
    let mut clean_client = UnauthClient::new_with_owned_indices(
        CorruptingTransport,
        client.params.clone(),
        clean_servers[0].clone(),
        clean_servers,
        client.params.party_registry_indices.clone(),
        params.segment_len as u64,
    );
    let migrated_value = clean_client
        .read(idx)
        .expect("correcting read after kappa-double and kappa-halve");
    assert_eq!(decode_value(migrated_value), expected);
}

#[test]
fn correcting_claim_repair_handles_multi_claim_servers_with_capped_row_provenance() {
    let segment_len = 8usize;
    let n_grid = segment_len;
    let s = 4usize;
    let shard_count = 10usize;
    let cfg = CodedArrayConfig {
        segment_len,
        s,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count, s, n_grid),
        party_registry_indices: vec![
            vec![0, 1, 2, 3, 4],
            vec![5],
            vec![6],
            vec![7],
            vec![8],
            vec![9],
        ],
        seed: 91,
    };
    let params = build_coded_params(&cfg);
    assert_eq!(params.need, 2);
    assert_eq!(params.rows_per_shard, 2);

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
        servers,
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
    assert_eq!(repaired.rows.len(), params.rows_per_shard);
}
