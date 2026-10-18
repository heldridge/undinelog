use paper_a_optimized::coded_array::params::CodedArrayConfig;
use paper_a_optimized::unauth::client::{ClientError, UnauthClient, UnauthTransport};
use paper_a_optimized::unauth::service::{UnauthRequest, UnauthResponse, UnauthService};
use paper_a_optimized::unauth::{
    ClaimBundle, DeltaMode, MigrateError, UnauthStore, aligned_block_indices,
};
use paper_a_optimized::{
    Fp, RowShare, ShardShare, build_coded_params, build_shard_registry_deterministic, decode_value,
    encode_value,
};
use std::sync::{Arc, Mutex};

fn apply_expected_block_deltas(
    mut values: Vec<paper_a_optimized::Value256>,
    deltas: &[(usize, Fp)],
) -> Vec<paper_a_optimized::Value256> {
    for &(offset, delta) in deltas {
        values[offset] = encode_value(decode_value(values[offset]) + delta);
    }
    values
}

fn active_block_indices(
    params: &paper_a_optimized::CodedArrayParams,
    block: u64,
    len: u64,
) -> Vec<u64> {
    aligned_block_indices(params, block, len)
        .into_iter()
        .flatten()
        .collect()
}

fn remap_block_indices_between_layouts(
    segment_len: usize,
    n_grid: usize,
    from_s: usize,
    _to_s: usize,
    block: u64,
) -> Vec<u64> {
    let blocks_per_segment = segment_len.div_ceil(from_s) as u64;
    let seg = block / blocks_per_segment;
    let block_in_segment = block % blocks_per_segment;
    let step = n_grid / from_s;
    (0..from_s)
        .filter_map(|offset| {
            let grid_index = block_in_segment as usize + offset * step;
            (grid_index < segment_len).then_some(seg * segment_len as u64 + grid_index as u64)
        })
        .collect()
}

fn collect_semantic_array_values<T>(
    client: &mut UnauthClient<T>,
    len: u64,
) -> Vec<paper_a_optimized::Value256>
where
    T: UnauthTransport + Send + Sync + 'static,
    T::Server: Clone + Send + Sync + 'static,
{
    let idxs: Vec<u64> = (0..len).collect();
    client
        .multi_read(&idxs)
        .expect("collect current semantic array values")
}

fn semantic_values_to_raw_grid(
    n_grid: usize,
    _layout_s: usize,
    semantic_values: &[paper_a_optimized::Value256],
) -> Vec<paper_a_optimized::Value256> {
    let zero = encode_value(Fp::from(0u64));
    let mut raw = vec![zero; n_grid];
    for (logical_offset, &value) in semantic_values.iter().enumerate() {
        raw[logical_offset] = value;
    }
    raw
}

fn exact_store_matching_actual_rows(
    actual: &UnauthStore,
    semantic_values: &[paper_a_optimized::Value256],
) -> UnauthStore {
    let params = actual.params.clone();
    assert_eq!(semantic_values.len(), actual.len() as usize);
    let row_label_index_by_bytes: std::collections::HashMap<_, _> = params
        .row_labels
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, row_label)| (encode_value(row_label), idx))
        .collect();
    let mut claims_by_registry = std::collections::BTreeMap::<usize, Vec<usize>>::new();
    for row_key in actual.coded.row_keys() {
        let registry_index = *params
            .shard_registry_by_key
            .get(&row_key.owner)
            .expect("row owner in registry");
        let row_label_index = *row_label_index_by_bytes
            .get(&row_key.row_label_bytes)
            .expect("row label in params");
        claims_by_registry
            .entry(registry_index)
            .or_default()
            .push(row_label_index);
    }
    for shard_key in actual.coded.shard_keys() {
        let registry_index = *params
            .shard_registry_by_key
            .get(shard_key)
            .expect("shard owner in registry");
        claims_by_registry.entry(registry_index).or_default();
    }
    let claims: Vec<_> = claims_by_registry
        .into_iter()
        .map(|(registry_index, mut row_label_indices)| {
            row_label_indices.sort_unstable();
            row_label_indices.dedup();
            assert_eq!(
                row_label_indices,
                params.row_label_indices_for_registry_index(registry_index)
            );
            ClaimBundle::exact_for_registry_index(&params, registry_index)
                .expect("exact claim should exist for registry index")
        })
        .collect();
    let mut store = if claims.is_empty() {
        UnauthStore::new(params.clone())
    } else {
        UnauthStore::new_with_claims(params.clone(), claims.clone())
    };
    if claims.is_empty() {
        store
            .extend_with_segments(actual.len(), |seg| {
                let start = seg * params.segment_len;
                let end = semantic_values.len().min(start + params.segment_len);
                semantic_values_to_raw_grid(params.n_grid, params.s, &semantic_values[start..end])
            })
            .expect("materialize exact store");
    } else {
        store
            .extend_with_segments_for_claims(actual.len(), &claims, |seg| {
                let start = seg * params.segment_len;
                let end = semantic_values.len().min(start + params.segment_len);
                semantic_values_to_raw_grid(params.n_grid, params.s, &semantic_values[start..end])
            })
            .expect("materialize exact claim-bound store");
    }
    store
}

fn row_mismatches(actual: &UnauthStore, exact: &UnauthStore) -> Vec<String> {
    let row_label_index_by_bytes: std::collections::HashMap<_, _> = actual
        .params
        .row_labels
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, row_label)| (encode_value(row_label), idx))
        .collect();
    let mut keys: Vec<_> = actual.coded.row_keys().copied().collect();
    for row_key in exact.coded.row_keys() {
        if !actual.coded.contains_row(row_key) {
            keys.push(*row_key);
        }
    }
    let mut mismatches = Vec::new();
    for row_key in keys {
        if actual
            .coded
            .clone_row_family_segments(&actual.params, &row_key)
            == exact
                .coded
                .clone_row_family_segments(&exact.params, &row_key)
        {
            continue;
        }
        let registry_index = actual
            .params
            .shard_registry_by_key
            .get(&row_key.owner)
            .copied()
            .map(|idx| idx.to_string())
            .unwrap_or_else(|| "?".to_string());
        let row_label_index = row_label_index_by_bytes
            .get(&row_key.row_label_bytes)
            .copied()
            .map(|idx| idx.to_string())
            .unwrap_or_else(|| "?".to_string());
        mismatches.push(format!(
            "registry {} row {}",
            registry_index, row_label_index
        ));
    }
    mismatches.sort();
    mismatches.dedup();
    mismatches
}

fn assert_rows_match_column_reconstruction<T>(
    client: &mut UnauthClient<T>,
    server: &MockServer,
    context: &str,
) where
    T: UnauthTransport + Send + Sync + 'static,
    T::Server: Clone + Send + Sync + 'static,
{
    let len = server.store.lock().expect("store lock").len();
    let semantic_values = collect_semantic_array_values(client, len);
    let store = server.store.lock().expect("store lock");
    let exact = exact_store_matching_actual_rows(&store, &semantic_values);
    let mismatches = row_mismatches(&store, &exact);
    assert!(
        mismatches.is_empty(),
        "{}: {} mismatched rows: {}",
        context,
        mismatches.len(),
        mismatches.join(", ")
    );
}

fn build_party_claims(
    params: &paper_a_optimized::CodedArrayParams,
    registry_indices: &[usize],
) -> Vec<ClaimBundle> {
    registry_indices
        .iter()
        .map(|&registry_index| {
            ClaimBundle::exact_for_registry_index(params, registry_index)
                .expect("exact claim should exist for registry index")
        })
        .collect()
}

#[derive(Clone)]
struct MockServer {
    label: String,
    store: Arc<Mutex<UnauthStore>>,
}

#[derive(Clone)]
struct MockTransport;

#[derive(Clone)]
struct ShardOnlyTransport;

#[derive(Clone)]
struct NoBlockUpdateTransport;

#[derive(Clone)]
struct RowOnlyTransport;

#[derive(Clone)]
struct NoGrowFetchTransport;

impl UnauthTransport for MockTransport {
    type Server = MockServer;

    fn server_label(&self, server: &Self::Server) -> String {
        server.label.clone()
    }

    fn unauth_get_len(&self, server: &Self::Server) -> Result<u64, ClientError> {
        let mut store = server.store.lock().expect("store lock");
        match store.handle_request(UnauthRequest::GetLen) {
            UnauthResponse::Len { len } => Ok(len),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn unauth_read(&self, server: &Self::Server, idx: u64) -> Result<Vec<ShardShare>, ClientError> {
        let mut store = server.store.lock().expect("store lock");
        match store.handle_request(UnauthRequest::Read { idx }) {
            UnauthResponse::Read { shares } => Ok(shares),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn unauth_read_block(
        &self,
        server: &Self::Server,
        block: u64,
    ) -> Result<Vec<RowShare>, ClientError> {
        let mut store = server.store.lock().expect("store lock");
        match store.handle_request(UnauthRequest::ReadBlock { block }) {
            UnauthResponse::ReadBlock { rows } => Ok(rows),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn unauth_read_rows_at_z(
        &self,
        server: &Self::Server,
        seg: usize,
        z_eval: Fp,
    ) -> Result<Vec<RowShare>, ClientError> {
        let mut store = server.store.lock().expect("store lock");
        match store.handle_request(UnauthRequest::ReadRowsAtZ { seg, z_eval }) {
            UnauthResponse::ReadRowsAtZ { rows } => Ok(rows),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn unauth_read_rows_batch_at_z(
        &self,
        server: &Self::Server,
        seg: usize,
        z_evals: &[Fp],
    ) -> Result<Vec<Vec<RowShare>>, ClientError> {
        let mut store = server.store.lock().expect("store lock");
        match store.handle_request(UnauthRequest::ReadRowsBatchAtZ {
            seg,
            z_evals: z_evals.to_vec(),
        }) {
            UnauthResponse::ReadRowsBatchAtZ { rows } => Ok(rows),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn unauth_read_rows_matrix(
        &self,
        server: &Self::Server,
        requests: &[(usize, Vec<Fp>)],
    ) -> Result<Vec<Vec<Vec<RowShare>>>, ClientError> {
        let mut store = server.store.lock().expect("store lock");
        match store.handle_request(UnauthRequest::ReadRowsMatrix {
            requests: requests
                .iter()
                .map(
                    |(seg, z_evals)| paper_a_optimized::unauth::service::RowReadRequest {
                        seg: *seg,
                        z_evals: z_evals.clone(),
                    },
                )
                .collect(),
        }) {
            UnauthResponse::ReadRowsMatrix { rows } => Ok(rows),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn unauth_multi_read(
        &self,
        server: &Self::Server,
        idxs: &[u64],
    ) -> Result<Vec<Vec<ShardShare>>, ClientError> {
        let mut store = server.store.lock().expect("store lock");
        match store.handle_request(UnauthRequest::MultiRead {
            idxs: idxs.to_vec(),
        }) {
            UnauthResponse::MultiRead { shares } => Ok(shares),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn unauth_repair_shard(
        &self,
        server: &Self::Server,
        seg: usize,
        registry_index: usize,
    ) -> Result<Vec<Fp>, ClientError> {
        let mut store = server.store.lock().expect("store lock");
        match store.handle_request(UnauthRequest::RepairShard {
            seg,
            registry_index,
        }) {
            UnauthResponse::RepairShard { values } => Ok(values),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn unauth_eval_shards_at_row_labels(
        &self,
        server: &Self::Server,
        seg: usize,
        col_labels: &[Fp],
        row_labels: &[Fp],
    ) -> Result<Vec<Vec<Fp>>, ClientError> {
        let mut store = server.store.lock().expect("store lock");
        match store.handle_request(UnauthRequest::EvalShardsAtRowLabels {
            seg,
            col_labels: col_labels.to_vec(),
            row_labels: row_labels.to_vec(),
        }) {
            UnauthResponse::EvalShardsAtRowLabels { values } => Ok(values),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn unauth_apply_deltas(
        &self,
        server: &Self::Server,
        deltas: &[(u64, Fp)],
        mode: DeltaMode,
    ) -> Result<bool, ClientError> {
        let mut store = server.store.lock().expect("store lock");
        match store.handle_request(UnauthRequest::ApplyDeltas {
            deltas: deltas.to_vec(),
            mode,
        }) {
            UnauthResponse::ApplyDeltas { ok } => Ok(ok),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn unauth_apply_block_deltas(
        &self,
        server: &Self::Server,
        block: u64,
        deltas: &[(usize, Fp)],
        mode: DeltaMode,
    ) -> Result<bool, ClientError> {
        let mut store = server.store.lock().expect("store lock");
        match store.handle_request(UnauthRequest::ApplyBlockDeltas {
            block,
            deltas: deltas.to_vec(),
            mode,
        }) {
            UnauthResponse::ApplyBlockDeltas { ok } => Ok(ok),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn unauth_extend(&self, server: &Self::Server, new_len: u64) -> Result<bool, ClientError> {
        let mut store = server.store.lock().expect("store lock");
        match store.handle_request(UnauthRequest::Extend { new_len }) {
            UnauthResponse::Extend { ok } => Ok(ok),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn unauth_migrate(
        &self,
        server: &Self::Server,
        target_s: usize,
        partner_shards: &[paper_a_optimized::unauth::PartnerShardMigration],
        helper_rows: &[paper_a_optimized::unauth::HelperRowMigration],
    ) -> Result<bool, ClientError> {
        let mut store = server.store.lock().expect("store lock");
        match store.handle_request(UnauthRequest::Migrate {
            target_s,
            partner_shards: partner_shards.to_vec(),
            helper_rows: helper_rows.to_vec(),
        }) {
            UnauthResponse::Migrate { ok } => Ok(ok),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!("{:?}", other))),
        }
    }
}

impl UnauthTransport for NoGrowFetchTransport {
    type Server = MockServer;

    fn server_label(&self, server: &Self::Server) -> String {
        MockTransport.server_label(server)
    }

    fn unauth_get_len(&self, server: &Self::Server) -> Result<u64, ClientError> {
        MockTransport.unauth_get_len(server)
    }

    fn unauth_read(&self, server: &Self::Server, idx: u64) -> Result<Vec<ShardShare>, ClientError> {
        MockTransport.unauth_read(server, idx)
    }

    fn unauth_read_block(
        &self,
        server: &Self::Server,
        block: u64,
    ) -> Result<Vec<RowShare>, ClientError> {
        MockTransport.unauth_read_block(server, block)
    }

    fn unauth_read_rows_at_z(
        &self,
        _server: &Self::Server,
        _seg: usize,
        _z_eval: Fp,
    ) -> Result<Vec<RowShare>, ClientError> {
        Err(ClientError::Net(
            "unexpected kappa-halve network row fetch".to_string(),
        ))
    }

    fn unauth_read_rows_batch_at_z(
        &self,
        _server: &Self::Server,
        _seg: usize,
        _z_evals: &[Fp],
    ) -> Result<Vec<Vec<RowShare>>, ClientError> {
        Err(ClientError::Net(
            "unexpected kappa-halve network row batch fetch".to_string(),
        ))
    }

    fn unauth_read_rows_matrix(
        &self,
        _server: &Self::Server,
        _requests: &[(usize, Vec<Fp>)],
    ) -> Result<Vec<Vec<Vec<RowShare>>>, ClientError> {
        Err(ClientError::Net(
            "unexpected kappa-halve row matrix fetch".to_string(),
        ))
    }

    fn unauth_multi_read(
        &self,
        server: &Self::Server,
        idxs: &[u64],
    ) -> Result<Vec<Vec<ShardShare>>, ClientError> {
        MockTransport.unauth_multi_read(server, idxs)
    }

    fn unauth_repair_shard(
        &self,
        _server: &Self::Server,
        _seg: usize,
        _registry_index: usize,
    ) -> Result<Vec<Fp>, ClientError> {
        Err(ClientError::Net(
            "unexpected kappa-halve shard repair".to_string(),
        ))
    }

    fn unauth_apply_deltas(
        &self,
        server: &Self::Server,
        deltas: &[(u64, Fp)],
        mode: DeltaMode,
    ) -> Result<bool, ClientError> {
        MockTransport.unauth_apply_deltas(server, deltas, mode)
    }

    fn unauth_apply_block_deltas(
        &self,
        server: &Self::Server,
        block: u64,
        deltas: &[(usize, Fp)],
        mode: DeltaMode,
    ) -> Result<bool, ClientError> {
        MockTransport.unauth_apply_block_deltas(server, block, deltas, mode)
    }

    fn unauth_extend(&self, server: &Self::Server, new_len: u64) -> Result<bool, ClientError> {
        MockTransport.unauth_extend(server, new_len)
    }

    fn unauth_migrate(
        &self,
        server: &Self::Server,
        target_s: usize,
        partner_shards: &[paper_a_optimized::unauth::PartnerShardMigration],
        helper_rows: &[paper_a_optimized::unauth::HelperRowMigration],
    ) -> Result<bool, ClientError> {
        MockTransport.unauth_migrate(server, target_s, partner_shards, helper_rows)
    }
}

impl UnauthTransport for ShardOnlyTransport {
    type Server = MockServer;

    fn server_label(&self, server: &Self::Server) -> String {
        MockTransport.server_label(server)
    }

    fn unauth_get_len(&self, server: &Self::Server) -> Result<u64, ClientError> {
        MockTransport.unauth_get_len(server)
    }

    fn unauth_read(&self, server: &Self::Server, idx: u64) -> Result<Vec<ShardShare>, ClientError> {
        MockTransport.unauth_read(server, idx)
    }

    fn unauth_read_block(
        &self,
        _server: &Self::Server,
        _block: u64,
    ) -> Result<Vec<RowShare>, ClientError> {
        Err(ClientError::Net("block row reads disabled".to_string()))
    }

    fn unauth_read_rows_at_z(
        &self,
        _server: &Self::Server,
        _seg: usize,
        _z_eval: Fp,
    ) -> Result<Vec<RowShare>, ClientError> {
        Err(ClientError::Net("row reads disabled".to_string()))
    }

    fn unauth_read_rows_batch_at_z(
        &self,
        _server: &Self::Server,
        _seg: usize,
        _z_evals: &[Fp],
    ) -> Result<Vec<Vec<RowShare>>, ClientError> {
        Err(ClientError::Net("row reads disabled".to_string()))
    }

    fn unauth_read_rows_matrix(
        &self,
        _server: &Self::Server,
        _requests: &[(usize, Vec<Fp>)],
    ) -> Result<Vec<Vec<Vec<RowShare>>>, ClientError> {
        Err(ClientError::Net("row reads disabled".to_string()))
    }

    fn unauth_multi_read(
        &self,
        server: &Self::Server,
        idxs: &[u64],
    ) -> Result<Vec<Vec<ShardShare>>, ClientError> {
        MockTransport.unauth_multi_read(server, idxs)
    }

    fn unauth_repair_shard(
        &self,
        server: &Self::Server,
        seg: usize,
        registry_index: usize,
    ) -> Result<Vec<Fp>, ClientError> {
        MockTransport.unauth_repair_shard(server, seg, registry_index)
    }

    fn unauth_repair_shard_batch(
        &self,
        server: &Self::Server,
        seg: usize,
        registry_indices: &[usize],
    ) -> Result<Vec<Vec<Fp>>, ClientError> {
        let mut store = server.store.lock().expect("store lock");
        match store.handle_request(UnauthRequest::RepairShardBatch {
            seg,
            registry_indices: registry_indices.to_vec(),
        }) {
            UnauthResponse::RepairShardBatch { values } => Ok(values),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn unauth_eval_shards_at_row_labels(
        &self,
        server: &Self::Server,
        seg: usize,
        col_labels: &[Fp],
        row_labels: &[Fp],
    ) -> Result<Vec<Vec<Fp>>, ClientError> {
        let mut store = server.store.lock().expect("store lock");
        match store.handle_request(UnauthRequest::EvalShardsAtRowLabels {
            seg,
            col_labels: col_labels.to_vec(),
            row_labels: row_labels.to_vec(),
        }) {
            UnauthResponse::EvalShardsAtRowLabels { values } => Ok(values),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn unauth_apply_deltas(
        &self,
        server: &Self::Server,
        deltas: &[(u64, Fp)],
        mode: DeltaMode,
    ) -> Result<bool, ClientError> {
        MockTransport.unauth_apply_deltas(server, deltas, mode)
    }

    fn unauth_extend(&self, server: &Self::Server, new_len: u64) -> Result<bool, ClientError> {
        MockTransport.unauth_extend(server, new_len)
    }

    fn unauth_migrate(
        &self,
        server: &Self::Server,
        target_s: usize,
        partner_shards: &[paper_a_optimized::unauth::PartnerShardMigration],
        helper_rows: &[paper_a_optimized::unauth::HelperRowMigration],
    ) -> Result<bool, ClientError> {
        MockTransport.unauth_migrate(server, target_s, partner_shards, helper_rows)
    }
}

impl UnauthTransport for RowOnlyTransport {
    type Server = MockServer;

    fn server_label(&self, server: &Self::Server) -> String {
        MockTransport.server_label(server)
    }

    fn unauth_get_len(&self, server: &Self::Server) -> Result<u64, ClientError> {
        MockTransport.unauth_get_len(server)
    }

    fn unauth_read(
        &self,
        _server: &Self::Server,
        _idx: u64,
    ) -> Result<Vec<ShardShare>, ClientError> {
        Err(ClientError::Net("point reads disabled".to_string()))
    }

    fn unauth_read_block(
        &self,
        server: &Self::Server,
        block: u64,
    ) -> Result<Vec<RowShare>, ClientError> {
        MockTransport.unauth_read_block(server, block)
    }

    fn unauth_read_rows_at_z(
        &self,
        server: &Self::Server,
        seg: usize,
        z_eval: Fp,
    ) -> Result<Vec<RowShare>, ClientError> {
        MockTransport.unauth_read_rows_at_z(server, seg, z_eval)
    }

    fn unauth_read_rows_batch_at_z(
        &self,
        server: &Self::Server,
        seg: usize,
        z_evals: &[Fp],
    ) -> Result<Vec<Vec<RowShare>>, ClientError> {
        MockTransport.unauth_read_rows_batch_at_z(server, seg, z_evals)
    }

    fn unauth_read_rows_matrix(
        &self,
        server: &Self::Server,
        requests: &[(usize, Vec<Fp>)],
    ) -> Result<Vec<Vec<Vec<RowShare>>>, ClientError> {
        MockTransport.unauth_read_rows_matrix(server, requests)
    }

    fn unauth_multi_read(
        &self,
        _server: &Self::Server,
        _idxs: &[u64],
    ) -> Result<Vec<Vec<ShardShare>>, ClientError> {
        Err(ClientError::Net("multi-read disabled".to_string()))
    }

    fn unauth_repair_shard(
        &self,
        _server: &Self::Server,
        _seg: usize,
        _registry_index: usize,
    ) -> Result<Vec<Fp>, ClientError> {
        Err(ClientError::Net("shard repair rpc disabled".to_string()))
    }

    fn unauth_repair_shard_batch(
        &self,
        _server: &Self::Server,
        _seg: usize,
        _registry_indices: &[usize],
    ) -> Result<Vec<Vec<Fp>>, ClientError> {
        Err(ClientError::Net("shard repair rpc disabled".to_string()))
    }

    fn unauth_apply_deltas(
        &self,
        server: &Self::Server,
        deltas: &[(u64, Fp)],
        mode: DeltaMode,
    ) -> Result<bool, ClientError> {
        MockTransport.unauth_apply_deltas(server, deltas, mode)
    }

    fn unauth_extend(&self, server: &Self::Server, new_len: u64) -> Result<bool, ClientError> {
        MockTransport.unauth_extend(server, new_len)
    }

    fn unauth_migrate(
        &self,
        server: &Self::Server,
        target_s: usize,
        partner_shards: &[paper_a_optimized::unauth::PartnerShardMigration],
        helper_rows: &[paper_a_optimized::unauth::HelperRowMigration],
    ) -> Result<bool, ClientError> {
        MockTransport.unauth_migrate(server, target_s, partner_shards, helper_rows)
    }
}

impl UnauthTransport for NoBlockUpdateTransport {
    type Server = MockServer;

    fn server_label(&self, server: &Self::Server) -> String {
        MockTransport.server_label(server)
    }

    fn unauth_get_len(&self, server: &Self::Server) -> Result<u64, ClientError> {
        MockTransport.unauth_get_len(server)
    }

    fn unauth_read(&self, server: &Self::Server, idx: u64) -> Result<Vec<ShardShare>, ClientError> {
        MockTransport.unauth_read(server, idx)
    }

    fn unauth_read_block(
        &self,
        server: &Self::Server,
        block: u64,
    ) -> Result<Vec<RowShare>, ClientError> {
        MockTransport.unauth_read_block(server, block)
    }

    fn unauth_read_rows_at_z(
        &self,
        server: &Self::Server,
        seg: usize,
        z_eval: Fp,
    ) -> Result<Vec<RowShare>, ClientError> {
        MockTransport.unauth_read_rows_at_z(server, seg, z_eval)
    }

    fn unauth_read_rows_batch_at_z(
        &self,
        server: &Self::Server,
        seg: usize,
        z_evals: &[Fp],
    ) -> Result<Vec<Vec<RowShare>>, ClientError> {
        MockTransport.unauth_read_rows_batch_at_z(server, seg, z_evals)
    }

    fn unauth_read_rows_matrix(
        &self,
        server: &Self::Server,
        requests: &[(usize, Vec<Fp>)],
    ) -> Result<Vec<Vec<Vec<RowShare>>>, ClientError> {
        MockTransport.unauth_read_rows_matrix(server, requests)
    }

    fn unauth_multi_read(
        &self,
        server: &Self::Server,
        idxs: &[u64],
    ) -> Result<Vec<Vec<ShardShare>>, ClientError> {
        MockTransport.unauth_multi_read(server, idxs)
    }

    fn unauth_repair_shard(
        &self,
        server: &Self::Server,
        seg: usize,
        registry_index: usize,
    ) -> Result<Vec<Fp>, ClientError> {
        MockTransport.unauth_repair_shard(server, seg, registry_index)
    }

    fn unauth_repair_shard_batch(
        &self,
        server: &Self::Server,
        seg: usize,
        registry_indices: &[usize],
    ) -> Result<Vec<Vec<Fp>>, ClientError> {
        MockTransport.unauth_repair_shard_batch(server, seg, registry_indices)
    }

    fn unauth_apply_deltas(
        &self,
        server: &Self::Server,
        deltas: &[(u64, Fp)],
        mode: DeltaMode,
    ) -> Result<bool, ClientError> {
        MockTransport.unauth_apply_deltas(server, deltas, mode)
    }

    fn unauth_extend(&self, server: &Self::Server, new_len: u64) -> Result<bool, ClientError> {
        MockTransport.unauth_extend(server, new_len)
    }

    fn unauth_migrate(
        &self,
        server: &Self::Server,
        target_s: usize,
        partner_shards: &[paper_a_optimized::unauth::PartnerShardMigration],
        helper_rows: &[paper_a_optimized::unauth::HelperRowMigration],
    ) -> Result<bool, ClientError> {
        MockTransport.unauth_migrate(server, target_s, partner_shards, helper_rows)
    }
}

fn build_params() -> paper_a_optimized::CodedArrayParams {
    let segment_len: usize = 8;
    let s: usize = 4;
    let shard_count: usize = 4;
    let n_grid = segment_len;
    let shard_registry = build_shard_registry_deterministic(shard_count, s, n_grid);
    let party_registry_indices = vec![(0..shard_count).collect()];
    build_coded_params(&CodedArrayConfig {
        segment_len,
        s,
        shard_count,
        shard_registry,
        party_registry_indices,
        seed: 21,
    })
}

fn build_segment_values(params: &paper_a_optimized::CodedArrayParams) -> Vec<[u8; 32]> {
    (0..params.n_grid)
        .map(|idx| encode_value(Fp::from((idx + 1) as u64)))
        .collect()
}

#[test]
fn client_block_read_matches_point_reads_on_same_semantic_block() {
    let params = build_params();
    let segment_values = build_segment_values(&params);
    let owned_indices = vec![vec![0usize], vec![1usize], vec![2usize]];
    let mut servers = Vec::new();
    for (idx, claimed) in owned_indices.iter().cloned().enumerate() {
        let mut store = UnauthStore::new(params.clone());
        store
            .extend_with_segments_filtered(params.segment_len as u64, &claimed, |_| {
                segment_values.clone()
            })
            .expect("filtered extend");
        servers.push(MockServer {
            label: idx.to_string(),
            store: Arc::new(Mutex::new(store)),
        });
    }

    let mut client = UnauthClient::new_with_owned_indices(
        MockTransport,
        params.clone(),
        servers[0].clone(),
        servers.clone(),
        owned_indices,
        8,
    );

    let block0 = client.read_block(0).expect("read block 0");
    let points0: Vec<_> = active_block_indices(&params, 0, params.segment_len as u64)
        .into_iter()
        .map(|idx| client.read(idx).expect("read point"))
        .collect();
    assert_eq!(block0, points0);

    let block1 = client.read_block(1).expect("read block 1");
    let points1: Vec<_> = active_block_indices(&params, 1, params.segment_len as u64)
        .into_iter()
        .map(|idx| client.read(idx).expect("read point"))
        .collect();
    assert_eq!(block1, points1);
}

#[test]
fn client_block_read_uses_row_path_when_point_reads_are_disabled() {
    let params = build_params();
    let segment_values = build_segment_values(&params);
    let owned_indices = vec![vec![0usize], vec![1usize], vec![2usize]];
    let mut servers = Vec::new();
    for (idx, claimed) in owned_indices.iter().cloned().enumerate() {
        let mut store = UnauthStore::new(params.clone());
        store
            .extend_with_segments_filtered(params.segment_len as u64, &claimed, |_| {
                segment_values.clone()
            })
            .expect("filtered extend");
        servers.push(MockServer {
            label: idx.to_string(),
            store: Arc::new(Mutex::new(store)),
        });
    }

    let mut client = UnauthClient::new_with_owned_indices(
        RowOnlyTransport,
        params.clone(),
        servers[0].clone(),
        servers.clone(),
        owned_indices,
        8,
    );

    let block0 = client.read_block(0).expect("row-path block read");
    assert_eq!(
        block0,
        vec![
            encode_value(Fp::from(1u64)),
            encode_value(Fp::from(3u64)),
            encode_value(Fp::from(5u64)),
            encode_value(Fp::from(7u64)),
        ]
    );
}

#[test]
fn client_block_batch_read_matches_repeated_row_path_reads() {
    let params = build_params();
    let segment_values = build_segment_values(&params);
    let owned_indices = vec![vec![0usize], vec![1usize], vec![2usize]];
    let mut servers = Vec::new();
    for (idx, claimed) in owned_indices.iter().cloned().enumerate() {
        let mut store = UnauthStore::new(params.clone());
        store
            .extend_with_segments_filtered(params.segment_len as u64, &claimed, |_| {
                segment_values.clone()
            })
            .expect("filtered extend");
        servers.push(MockServer {
            label: idx.to_string(),
            store: Arc::new(Mutex::new(store)),
        });
    }

    let mut client = UnauthClient::new_with_owned_indices(
        RowOnlyTransport,
        params.clone(),
        servers[0].clone(),
        servers.clone(),
        owned_indices,
        8,
    );

    let expected = vec![
        client.read_block(0).expect("block 0"),
        client.read_block(1).expect("block 1"),
    ];
    let batched = client
        .read_block_batch(&[0, 1])
        .expect("batched block read");
    assert_eq!(batched, expected);
}

#[test]
fn client_block_write_updates_block_read_and_point_reads() {
    let params = build_params();
    let segment_values = build_segment_values(&params);
    let owned_indices = vec![vec![0usize], vec![1usize], vec![2usize]];
    let mut servers = Vec::new();
    for (idx, claimed) in owned_indices.iter().cloned().enumerate() {
        let mut store = UnauthStore::new(params.clone());
        store
            .extend_with_segments_filtered(params.segment_len as u64, &claimed, |_| {
                segment_values.clone()
            })
            .expect("filtered extend");
        servers.push(MockServer {
            label: idx.to_string(),
            store: Arc::new(Mutex::new(store)),
        });
    }

    let mut client = UnauthClient::new_with_owned_indices(
        MockTransport,
        params.clone(),
        servers[0].clone(),
        servers.clone(),
        owned_indices,
        8,
    );
    client
        .write_block(
            0,
            &[
                (0usize, Fp::from(11u64)),
                (2usize, Fp::from(7u64)),
                (3usize, Fp::from(5u64)),
            ],
            DeltaMode::Aggregate,
        )
        .expect("block write");

    let block0 = client.read_block(0).expect("read block 0");
    let points0: Vec<_> = active_block_indices(&params, 0, params.segment_len as u64)
        .into_iter()
        .map(|idx| client.read(idx).expect("read point"))
        .collect();
    assert_eq!(block0, points0);
    assert_ne!(block0[0], encode_value(Fp::from(1u64)));
    assert_eq!(block0[1], encode_value(Fp::from(3u64)));
    assert_ne!(block0[2], encode_value(Fp::from(5u64)));
    assert_ne!(block0[3], encode_value(Fp::from(7u64)));
}

#[test]
fn client_array_view_kappa_doubled_block_write_survives_kappa_halve_on_same_logical_points() {
    let segment_len: usize = 32;
    let source_s: usize = 16;
    let kappa_doubled_s: usize = 8;
    let shard_count: usize = 8;
    let n_grid = segment_len;
    let params = build_coded_params(&CodedArrayConfig {
        segment_len,
        s: source_s,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count, source_s, n_grid),
        party_registry_indices: vec![(0..shard_count).collect()],
        seed: 31,
    });
    let segment_values = vec![encode_value(Fp::from(0u64)); params.n_grid];
    let owned_indices = vec![(0..shard_count).collect::<Vec<_>>()];

    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments_filtered(params.segment_len as u64, &owned_indices[0], |_| {
            segment_values.clone()
        })
        .expect("full extend");

    let servers = vec![MockServer {
        label: "0".to_string(),
        store: Arc::new(Mutex::new(store)),
    }];
    let mut client = UnauthClient::new_with_owned_indices(
        MockTransport,
        params,
        servers[0].clone(),
        servers,
        owned_indices,
        8,
    );

    client.migrate(kappa_doubled_s).expect("kappa-double");

    let block = 1u64;
    let before_kappa_doubled_block = client
        .read_block(block)
        .expect("read kappa-double block before write");
    let deltas = [
        (1usize, Fp::from(0x66u64)),
        (4usize, Fp::from(0x77u64)),
        (6usize, Fp::from(0x88u64)),
    ];
    client
        .write_block(block, &deltas, DeltaMode::Aggregate)
        .expect("kappa-double block write");

    let kappa_doubled_block = client.read_block(block).expect("read kappa-double block");
    let expected_kappa_doubled_block =
        apply_expected_block_deltas(before_kappa_doubled_block, &deltas);
    assert_eq!(kappa_doubled_block, expected_kappa_doubled_block);

    client.migrate(source_s).expect("kappa-halve");

    let stable_indices =
        remap_block_indices_between_layouts(segment_len, n_grid, kappa_doubled_s, source_s, block);
    let kappa_halved_points = client
        .multi_read(&stable_indices)
        .expect("read kappa-halve logical points");
    assert_eq!(kappa_halved_points, expected_kappa_doubled_block);
}

#[test]
fn client_array_view_migrated_block_write_kappa_halve_to_readable_block() {
    let segment_len: usize = 32;
    let source_s: usize = 16;
    let kappa_doubled_s: usize = 8;
    let shard_count: usize = 8;
    let n_grid = segment_len;
    let params = build_coded_params(&CodedArrayConfig {
        segment_len,
        s: source_s,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count, source_s, n_grid),
        party_registry_indices: vec![(0..shard_count).collect()],
        seed: 39,
    });
    let segment_values = vec![encode_value(Fp::from(0u64)); params.n_grid];
    let owned_indices = vec![(0..shard_count).collect::<Vec<_>>()];

    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments_filtered(params.segment_len as u64, &owned_indices[0], |_| {
            segment_values.clone()
        })
        .expect("full extend");

    let servers = vec![MockServer {
        label: "0".to_string(),
        store: Arc::new(Mutex::new(store)),
    }];

    let block = 1u64;
    let source_layout_indices =
        remap_block_indices_between_layouts(segment_len, n_grid, kappa_doubled_s, source_s, block);
    let mut source_client = UnauthClient::new_with_owned_indices(
        MockTransport,
        params.clone(),
        servers[0].clone(),
        servers.clone(),
        owned_indices.clone(),
        8,
    );
    source_client
        .write_block(
            block,
            &[
                (0usize, Fp::from(0x21u64)),
                (2usize, Fp::from(0x34u64)),
                (3usize, Fp::from(0x55u64)),
            ],
            DeltaMode::Aggregate,
        )
        .expect("old-mode block write");
    assert_rows_match_column_reconstruction(
        &mut source_client,
        &servers[0],
        "full-store source rows before kappa-double",
    );
    let source_layout_points_before_kappa_double = source_client
        .multi_read(&source_layout_indices)
        .expect("read source-layout points before kappa-double");

    source_client
        .migrate(kappa_doubled_s)
        .expect("kappa-double");

    let kappa_doubled_params = servers[0].store.lock().expect("store lock").params.clone();
    let kappa_doubled_owned_indices =
        vec![(0..kappa_doubled_params.shard_registry.len()).collect::<Vec<_>>()];
    let mut migrated_client = UnauthClient::new_with_owned_indices(
        NoBlockUpdateTransport,
        kappa_doubled_params,
        servers[0].clone(),
        servers.clone(),
        kappa_doubled_owned_indices,
        8,
    );
    assert_rows_match_column_reconstruction(
        &mut migrated_client,
        &servers[0],
        "full-store kappa-double rows before update",
    );

    let before_kappa_doubled_block = migrated_client
        .read_block(block)
        .expect("read kappa-double block before write");
    let deltas = [
        (1usize, Fp::from(0x66u64)),
        (4usize, Fp::from(0x77u64)),
        (6usize, Fp::from(0x88u64)),
    ];
    migrated_client
        .write_block(block, &deltas, DeltaMode::Aggregate)
        .expect("migrated block write");
    let expected_kappa_doubled_block =
        apply_expected_block_deltas(before_kappa_doubled_block, &deltas);
    assert_eq!(
        migrated_client
            .read_block(block)
            .expect("read kappa-double block after write"),
        expected_kappa_doubled_block
    );
    assert_rows_match_column_reconstruction(
        &mut migrated_client,
        &servers[0],
        "full-store kappa-double rows after update",
    );

    migrated_client.migrate(source_s).expect("kappa-halve");

    let kappa_halved_params = servers[0].store.lock().expect("store lock").params.clone();
    let kappa_halved_owned_indices =
        vec![(0..kappa_halved_params.shard_registry.len()).collect::<Vec<_>>()];
    let mut kappa_halved_client = UnauthClient::new_with_owned_indices(
        MockTransport,
        kappa_halved_params.clone(),
        servers[0].clone(),
        servers.clone(),
        kappa_halved_owned_indices.clone(),
        8,
    );
    assert_rows_match_column_reconstruction(
        &mut kappa_halved_client,
        &servers[0],
        "full-store kappa-halve rows",
    );

    let kappa_halved_points = kappa_halved_client
        .multi_read(&source_layout_indices)
        .expect("read kappa-halve remapped points");
    assert_eq!(
        kappa_halved_points,
        apply_expected_block_deltas(source_layout_points_before_kappa_double.clone(), &deltas)
    );

    let kappa_halved_block = kappa_halved_client
        .read_block(block)
        .expect("read kappa-halve block");
    let full_kappa_halved_block_indices =
        remap_block_indices_between_layouts(segment_len, n_grid, source_s, kappa_doubled_s, block);
    let kappa_halved_full_block_points = kappa_halved_client
        .multi_read(&full_kappa_halved_block_indices)
        .expect("read kappa-halve full block points");
    assert_eq!(kappa_halved_block, kappa_halved_full_block_points);
    let mut row_only_kappa_halved = UnauthClient::new_with_owned_indices(
        RowOnlyTransport,
        kappa_halved_params,
        servers[0].clone(),
        servers.clone(),
        kappa_halved_owned_indices,
        8,
    );
    assert_eq!(
        row_only_kappa_halved
            .read_block(block)
            .expect("row-path kappa-halve block"),
        kappa_halved_block
    );
}

#[test]
fn client_array_view_kappa_doubled_block_write_survives_kappa_halve_on_nonzero_full_store() {
    let segment_len: usize = 32;
    let source_s: usize = 16;
    let kappa_doubled_s: usize = 8;
    let shard_count: usize = 8;
    let n_grid = segment_len;
    let params = build_coded_params(&CodedArrayConfig {
        segment_len,
        s: source_s,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count, source_s, n_grid),
        party_registry_indices: vec![(0..shard_count).collect()],
        seed: 41,
    });
    let segment_values: Vec<_> = (0..params.n_grid)
        .map(|idx| encode_value(Fp::from((idx + 1) as u64)))
        .collect();
    let owned_indices = vec![(0..shard_count).collect::<Vec<_>>()];

    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments_filtered(params.segment_len as u64, &owned_indices[0], |_| {
            segment_values.clone()
        })
        .expect("full extend");

    let servers = vec![MockServer {
        label: "0".to_string(),
        store: Arc::new(Mutex::new(store)),
    }];
    let mut client = UnauthClient::new_with_owned_indices(
        MockTransport,
        params,
        servers[0].clone(),
        servers,
        owned_indices,
        8,
    );

    client
        .write_block(
            1,
            &[
                (0usize, Fp::from(0x21u64)),
                (2usize, Fp::from(0x34u64)),
                (3usize, Fp::from(0x55u64)),
            ],
            DeltaMode::Aggregate,
        )
        .expect("old-mode block write");
    client
        .read_block(1)
        .expect("source block before kappa-double");
    let source_layout_indices =
        remap_block_indices_between_layouts(segment_len, n_grid, kappa_doubled_s, source_s, 1);
    let source_layout_points_before_kappa_double = client
        .multi_read(&source_layout_indices)
        .expect("read source-layout points before kappa-double");

    client.migrate(kappa_doubled_s).expect("kappa-double");

    let block = 1u64;
    let before_kappa_doubled_block = client
        .read_block(block)
        .expect("read kappa-double block before write");
    let deltas = [
        (1usize, Fp::from(0x66u64)),
        (4usize, Fp::from(0x77u64)),
        (6usize, Fp::from(0x88u64)),
    ];
    client
        .write_block(block, &deltas, DeltaMode::Aggregate)
        .expect("kappa-double block write");

    let kappa_doubled_block = client.read_block(block).expect("read kappa-double block");
    let expected_kappa_doubled_block =
        apply_expected_block_deltas(before_kappa_doubled_block, &deltas);
    assert_eq!(kappa_doubled_block, expected_kappa_doubled_block);
    client.migrate(source_s).expect("kappa-halve");

    let kappa_halved_points = client
        .multi_read(&source_layout_indices)
        .expect("read kappa-halve remapped points");
    assert_eq!(
        kappa_halved_points,
        apply_expected_block_deltas(source_layout_points_before_kappa_double, &deltas)
    );
}

#[test]
fn client_array_view_kappa_double_keeps_old_block_fetchable_across_two_new_block_ids() {
    let segment_len: usize = 32;
    let source_s: usize = 16;
    let kappa_doubled_s: usize = 8;
    let shard_count: usize = 8;
    let n_grid = segment_len;
    let params = build_coded_params(&CodedArrayConfig {
        segment_len,
        s: source_s,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count, source_s, n_grid),
        party_registry_indices: vec![(0..shard_count).collect()],
        seed: 51,
    });
    let segment_values: Vec<_> = (0..params.n_grid)
        .map(|idx| encode_value(Fp::from((idx + 1) as u64)))
        .collect();
    let owned_indices = vec![(0..shard_count).collect::<Vec<_>>()];

    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments_filtered(params.segment_len as u64, &owned_indices[0], |_| {
            segment_values.clone()
        })
        .expect("full extend");

    let servers = vec![MockServer {
        label: "0".to_string(),
        store: Arc::new(Mutex::new(store)),
    }];
    let mut client = UnauthClient::new_with_owned_indices(
        MockTransport,
        params,
        servers[0].clone(),
        servers,
        owned_indices,
        8,
    );

    let old_block = client.read_block(1).expect("old block");
    client.migrate(kappa_doubled_s).expect("kappa-double");

    let remapped_old_block_indices =
        remap_block_indices_between_layouts(segment_len, n_grid, source_s, kappa_doubled_s, 1);
    assert_eq!(
        client
            .multi_read(&remapped_old_block_indices)
            .expect("read remapped old block"),
        old_block
    );
    assert_ne!(
        client.read_block(1).expect("current kappa-double block"),
        old_block
    );
}

#[test]
fn client_array_view_block_reads_follow_demo_relation_without_row_rpcs() {
    let segment_len: usize = 32;
    let source_s: usize = 16;
    let kappa_doubled_s: usize = 8;
    let shard_count: usize = 8;
    let n_grid = segment_len;
    let params = build_coded_params(&CodedArrayConfig {
        segment_len,
        s: source_s,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count, source_s, n_grid),
        party_registry_indices: vec![(0..shard_count).collect()],
        seed: 61,
    });
    let segment_values: Vec<_> = (0..params.n_grid)
        .map(|idx| encode_value(Fp::from((idx + 1) as u64)))
        .collect();
    let owned_indices = vec![(0..shard_count).collect::<Vec<_>>()];
    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments_filtered(params.segment_len as u64, &owned_indices[0], |_| {
            segment_values.clone()
        })
        .expect("full extend");
    let servers = vec![MockServer {
        label: "0".to_string(),
        store: Arc::new(Mutex::new(store)),
    }];

    {
        let mut client = UnauthClient::new_with_owned_indices(
            MockTransport,
            params.clone(),
            servers[0].clone(),
            servers.clone(),
            owned_indices.clone(),
            8,
        );
        client.migrate(kappa_doubled_s).expect("kappa-double");
    }

    let kappa_doubled_params = servers[0].store.lock().expect("store lock").params.clone();
    let kappa_doubled_owned_indices =
        vec![(0..kappa_doubled_params.shard_registry.len()).collect::<Vec<_>>()];
    let mut point_client = UnauthClient::new_with_owned_indices(
        ShardOnlyTransport,
        kappa_doubled_params,
        servers[0].clone(),
        servers.clone(),
        kappa_doubled_owned_indices,
        8,
    );

    assert!(matches!(
        point_client.read_block(1),
        Err(ClientError::NotEnoughShards)
    ));
}

#[test]
fn client_array_view_block_batch_reads_follow_demo_relation_without_row_rpcs() {
    let segment_len: usize = 32;
    let source_s: usize = 16;
    let kappa_doubled_s: usize = 8;
    let shard_count: usize = 8;
    let n_grid = segment_len;
    let params = build_coded_params(&CodedArrayConfig {
        segment_len,
        s: source_s,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count, source_s, n_grid),
        party_registry_indices: vec![(0..shard_count).collect()],
        seed: 71,
    });
    let segment_values: Vec<_> = (0..params.n_grid)
        .map(|idx| encode_value(Fp::from((idx + 1) as u64)))
        .collect();

    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments_filtered(
            params.segment_len as u64,
            &(0..shard_count).collect::<Vec<_>>(),
            |_| segment_values.clone(),
        )
        .expect("full extend");
    store.migrate(kappa_doubled_s, &[]).expect("kappa-double");

    let kappa_doubled_params = store.params.clone();
    let owned_indices = vec![(0..kappa_doubled_params.shard_registry.len()).collect::<Vec<_>>()];
    let servers = vec![MockServer {
        label: "0".to_string(),
        store: Arc::new(Mutex::new(store)),
    }];

    let mut point_client = UnauthClient::new_with_owned_indices(
        ShardOnlyTransport,
        kappa_doubled_params.clone(),
        servers[0].clone(),
        servers.clone(),
        owned_indices,
        8,
    );

    let blocks = [1u64, 3u64];
    assert!(matches!(
        point_client.read_block_batch(&blocks),
        Err(ClientError::NotEnoughShards)
    ));
}

#[test]
fn client_array_view_kappa_halve_keeps_old_blocks_fetchable_inside_larger_block() {
    let segment_len: usize = 32;
    let source_s: usize = 16;
    let kappa_doubled_s: usize = 8;
    let shard_count: usize = 8;
    let n_grid = segment_len;
    let params = build_coded_params(&CodedArrayConfig {
        segment_len,
        s: source_s,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count, source_s, n_grid),
        party_registry_indices: vec![(0..shard_count).collect()],
        seed: 81,
    });
    let segment_values: Vec<_> = (0..params.n_grid)
        .map(|idx| encode_value(Fp::from((idx + 1) as u64)))
        .collect();
    let owned_indices = vec![(0..shard_count).collect::<Vec<_>>()];
    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments_filtered(params.segment_len as u64, &owned_indices[0], |_| {
            segment_values.clone()
        })
        .expect("full extend");
    let servers = vec![MockServer {
        label: "0".to_string(),
        store: Arc::new(Mutex::new(store)),
    }];

    {
        let mut client = UnauthClient::new_with_owned_indices(
            MockTransport,
            params.clone(),
            servers[0].clone(),
            servers.clone(),
            owned_indices.clone(),
            8,
        );
        client.migrate(kappa_doubled_s).expect("kappa-double");
    }

    let kappa_doubled_params = servers[0].store.lock().expect("store lock").params.clone();
    let kappa_doubled_owned_indices =
        vec![(0..kappa_doubled_params.shard_registry.len()).collect::<Vec<_>>()];
    let mut client = UnauthClient::new_with_owned_indices(
        MockTransport,
        kappa_doubled_params,
        servers[0].clone(),
        servers.clone(),
        kappa_doubled_owned_indices,
        8,
    );

    let kappa_halved_block = 1u64;
    let old_block_indices = remap_block_indices_between_layouts(
        segment_len,
        n_grid,
        source_s,
        kappa_doubled_s,
        kappa_halved_block,
    );
    let old_block_under_shrink = client
        .multi_read(&old_block_indices)
        .expect("read remapped old block under kappa-double");

    client.migrate(source_s).expect("kappa-halve");

    assert_eq!(
        client
            .read_block(kappa_halved_block)
            .expect("kappa-halve block under new lens"),
        old_block_under_shrink
    );
}

#[test]
fn client_array_view_kappa_doubled_block_write_survives_kappa_halve_with_claim_bound_servers() {
    let segment_len: usize = 128;
    let source_s: usize = 16;
    let kappa_doubled_s: usize = 8;
    let shard_count: usize = 8;
    let n_grid = segment_len;
    let params = build_coded_params(&CodedArrayConfig {
        segment_len,
        s: source_s,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count, source_s, n_grid),
        party_registry_indices: vec![vec![0, 1, 2, 3], vec![4, 5], vec![6, 7]],
        seed: 37,
    });
    let segment_values: Vec<_> = (0..params.n_grid)
        .map(|idx| encode_value(Fp::from((idx + 1) as u64)))
        .collect();
    let owned_indices = vec![vec![0usize, 1, 2, 3], vec![4usize, 5], vec![6usize, 7]];

    let mut servers = Vec::new();
    for (idx, claimed) in owned_indices.iter().cloned().enumerate() {
        let claims = build_party_claims(&params, &claimed);
        let mut store = UnauthStore::new_with_claims(params.clone(), claims.clone());
        store
            .extend_with_segments_for_claims(params.segment_len as u64, &claims, |_| {
                segment_values.clone()
            })
            .expect("claim-bound extend");
        if idx == 0 {
            let mut local_probe = store.clone();
            assert!(matches!(
                local_probe.migrate(kappa_doubled_s, &[]),
                Err(MigrateError::MissingLocalRows)
            ));
        }
        servers.push(MockServer {
            label: idx.to_string(),
            store: Arc::new(Mutex::new(store)),
        });
    }

    let mut client = UnauthClient::new_with_owned_indices(
        MockTransport,
        params,
        servers[0].clone(),
        servers,
        owned_indices,
        8,
    );

    client
        .write_block(
            1,
            &[
                (0usize, Fp::from(0x21u64)),
                (2usize, Fp::from(0x34u64)),
                (3usize, Fp::from(0x55u64)),
            ],
            DeltaMode::Aggregate,
        )
        .expect("old-mode block write");
    let source_layout_indices =
        remap_block_indices_between_layouts(segment_len, n_grid, kappa_doubled_s, source_s, 1);
    let source_layout_points_before_kappa_double = client
        .multi_read(&source_layout_indices)
        .expect("read source-layout points before kappa-double");

    client.migrate(kappa_doubled_s).expect("kappa-double");

    let block = 1u64;
    let before_kappa_doubled_block = client
        .read_block(block)
        .expect("read kappa-double block before write");
    let deltas = [
        (1usize, Fp::from(0x66u64)),
        (4usize, Fp::from(0x77u64)),
        (6usize, Fp::from(0x88u64)),
    ];
    client
        .write_block(block, &deltas, DeltaMode::Aggregate)
        .expect("kappa-double block write");

    let kappa_doubled_block = client.read_block(block).expect("read kappa-double block");
    let expected_kappa_doubled_block =
        apply_expected_block_deltas(before_kappa_doubled_block, &deltas);
    assert_eq!(kappa_doubled_block, expected_kappa_doubled_block);
    client.migrate(source_s).expect("kappa-halve");

    let kappa_halved_points = client
        .multi_read(&source_layout_indices)
        .expect("read kappa-halve remapped points");
    assert_eq!(
        kappa_halved_points,
        apply_expected_block_deltas(source_layout_points_before_kappa_double, &deltas)
    );
}

#[test]
fn client_kappa_halve_requires_helper_rows_after_kappa_double() {
    let segment_len: usize = 128;
    let source_s: usize = 16;
    let kappa_doubled_s: usize = 8;
    let shard_count: usize = 8;
    let n_grid = segment_len;
    let params = build_coded_params(&CodedArrayConfig {
        segment_len,
        s: source_s,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count, source_s, n_grid),
        party_registry_indices: vec![vec![0, 1, 2, 3], vec![4, 5], vec![6, 7]],
        seed: 41,
    });
    let segment_values: Vec<_> = (0..params.n_grid)
        .map(|idx| encode_value(Fp::from((idx + 1) as u64)))
        .collect();
    let owned_indices = vec![vec![0usize, 1, 2, 3], vec![4usize, 5], vec![6usize, 7]];

    let mut servers = Vec::new();
    for claimed in owned_indices.iter().cloned() {
        let mut store = UnauthStore::new(params.clone());
        store
            .extend_with_segments_filtered(params.segment_len as u64, &claimed, |_| {
                segment_values.clone()
            })
            .expect("filtered extend");
        servers.push(MockServer {
            label: servers.len().to_string(),
            store: Arc::new(Mutex::new(store)),
        });
    }

    let mut client = UnauthClient::new_with_owned_indices(
        MockTransport,
        params,
        servers[0].clone(),
        servers.clone(),
        owned_indices,
        8,
    );

    let _source_block = client.read_block(1).expect("source block");
    client
        .migrate(kappa_doubled_s)
        .expect("network kappa-double");

    let kappa_doubled_params = servers[0].store.lock().expect("store lock").params.clone();
    let kappa_doubled_owned_indices = kappa_doubled_params.party_registry_indices.clone();
    let mut client = UnauthClient::new_with_owned_indices(
        NoGrowFetchTransport,
        kappa_doubled_params,
        servers[0].clone(),
        servers.clone(),
        kappa_doubled_owned_indices,
        8,
    );

    assert!(matches!(
        client.migrate(source_s),
        Err(ClientError::NotEnoughShards)
    ));
    assert!(
        !client
            .read_block(1)
            .expect("kappa-double block remains readable")
            .is_empty()
    );
}

#[test]
fn client_array_view_migrated_block_write_kappa_halve_to_readable_block_with_claim_bound_zero_store()
 {
    let segment_len: usize = 128;
    let source_s: usize = 16;
    let kappa_doubled_s: usize = 8;
    let shard_count: usize = 8;
    let n_grid = segment_len;
    let params = build_coded_params(&CodedArrayConfig {
        segment_len,
        s: source_s,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count, source_s, n_grid),
        party_registry_indices: vec![vec![0, 1, 2, 3], vec![4, 5], vec![6, 7]],
        seed: 43,
    });
    let segment_values = vec![encode_value(Fp::from(0u64)); params.n_grid];
    let owned_indices = vec![vec![0usize, 1, 2, 3], vec![4usize, 5], vec![6usize, 7]];

    let mut servers = Vec::new();
    for claimed in owned_indices.iter().cloned() {
        let claims = build_party_claims(&params, &claimed);
        let mut store = UnauthStore::new_with_claims(params.clone(), claims.clone());
        store
            .extend_with_segments_for_claims(params.segment_len as u64, &claims, |_| {
                segment_values.clone()
            })
            .expect("claim-bound extend");
        servers.push(MockServer {
            label: servers.len().to_string(),
            store: Arc::new(Mutex::new(store)),
        });
    }

    let block = 1u64;
    let source_layout_indices =
        remap_block_indices_between_layouts(segment_len, n_grid, kappa_doubled_s, source_s, block);
    let mut client = UnauthClient::new_with_owned_indices(
        MockTransport,
        params,
        servers[0].clone(),
        servers.clone(),
        owned_indices,
        8,
    );

    client
        .write_block(
            block,
            &[
                (0usize, Fp::from(0x21u64)),
                (2usize, Fp::from(0x34u64)),
                (3usize, Fp::from(0x55u64)),
            ],
            DeltaMode::Aggregate,
        )
        .expect("old-mode block write");
    assert_rows_match_column_reconstruction(
        &mut client,
        &servers[0],
        "claim-bound source rows before kappa-double",
    );
    let source_block_before_shrink = client
        .read_block(block)
        .expect("source block before kappa-double");
    let source_layout_points_before_kappa_double = client
        .multi_read(&source_layout_indices)
        .expect("read source-layout points before kappa-double");

    client.migrate(kappa_doubled_s).expect("kappa-double");
    assert_rows_match_column_reconstruction(
        &mut client,
        &servers[0],
        "claim-bound kappa-double rows before update",
    );

    let before_kappa_doubled_block = client
        .read_block(block)
        .expect("read kappa-double block before write");
    let deltas = [
        (1usize, Fp::from(0x66u64)),
        (4usize, Fp::from(0x77u64)),
        (6usize, Fp::from(0x88u64)),
    ];
    client
        .write_block(block, &deltas, DeltaMode::Aggregate)
        .expect("kappa-double block write");
    assert_rows_match_column_reconstruction(
        &mut client,
        &servers[0],
        "claim-bound kappa-double rows after update",
    );

    let kappa_doubled_block = client.read_block(block).expect("read kappa-double block");
    let expected_kappa_doubled_block =
        apply_expected_block_deltas(before_kappa_doubled_block, &deltas);
    assert_eq!(kappa_doubled_block, expected_kappa_doubled_block);

    client.migrate(source_s).expect("kappa-halve");
    assert_rows_match_column_reconstruction(
        &mut client,
        &servers[0],
        "claim-bound kappa-halve rows",
    );

    assert_eq!(
        client
            .multi_read(&source_layout_indices)
            .expect("read kappa-halve remapped points"),
        apply_expected_block_deltas(source_layout_points_before_kappa_double, &deltas)
    );

    let kappa_halved_block = client.read_block(block).expect("read kappa-halve block");
    let full_kappa_halved_block_indices =
        remap_block_indices_between_layouts(segment_len, n_grid, source_s, kappa_doubled_s, block);
    assert_eq!(
        client
            .multi_read(&full_kappa_halved_block_indices)
            .expect("read kappa-halve aligned block by points"),
        kappa_halved_block
    );

    let kappa_halved_owned_indices = vec![vec![0usize, 1, 2, 3], vec![4usize, 5], vec![6usize, 7]];
    let mut row_only_single = UnauthClient::new_with_owned_indices(
        RowOnlyTransport,
        client.params.clone(),
        servers[0].clone(),
        vec![servers[0].clone()],
        vec![kappa_halved_owned_indices[0].clone()],
        8,
    );
    let mut row_only_all = UnauthClient::new_with_owned_indices(
        RowOnlyTransport,
        client.params.clone(),
        servers[0].clone(),
        servers.clone(),
        kappa_halved_owned_indices,
        8,
    );
    assert!(matches!(
        row_only_single.read_block(block),
        Err(ClientError::NotEnoughShards)
    ));
    assert_eq!(
        row_only_all.read_block(block).expect("all-party row path"),
        kappa_halved_block
    );
    assert_ne!(kappa_halved_block, source_block_before_shrink);
}

#[test]
fn client_network_repair_reconstructs_joining_party_shard_from_live_rows() {
    let params = build_params();
    let segment_values = build_segment_values(&params);
    let owned_indices = vec![vec![0usize], vec![1usize], vec![2usize]];
    let mut servers = Vec::new();
    for (idx, claimed) in owned_indices.iter().cloned().enumerate() {
        let mut store = UnauthStore::new(params.clone());
        store
            .extend_with_segments_filtered(params.segment_len as u64, &claimed, |_| {
                segment_values.clone()
            })
            .expect("filtered extend");
        servers.push(MockServer {
            label: idx.to_string(),
            store: Arc::new(Mutex::new(store)),
        });
    }

    let mut full_store = UnauthStore::new(params.clone());
    full_store
        .extend_with_segments_filtered(
            params.segment_len as u64,
            &(0..params.shard_registry.len()).collect::<Vec<_>>(),
            |_| segment_values.clone(),
        )
        .expect("full extend");
    let expected = full_store
        .repair_primary_shard_from_rows(0, params.shard_registry[1])
        .expect("expected repair");

    let client = UnauthClient::new_with_owned_indices(
        RowOnlyTransport,
        params.clone(),
        servers[0].clone(),
        servers.clone(),
        owned_indices,
        8,
    );
    let repaired = client
        .repair_shard_from_network(0, 1)
        .expect("network repair");

    let expected_values: Vec<_> = expected.into_iter().map(encode_value).collect();
    assert_eq!(repaired, expected_values);
}

#[test]
fn client_network_row_repair_reconstructs_joining_party_rows_from_live_cluster() {
    let params = build_params();
    let segment_values = build_segment_values(&params);
    let owned_indices = vec![vec![0usize], vec![1usize], vec![2usize]];
    let mut servers = Vec::new();
    for (idx, claimed) in owned_indices.iter().cloned().enumerate() {
        let mut store = UnauthStore::new(params.clone());
        store
            .extend_with_segments_filtered(params.segment_len as u64, &claimed, |_| {
                segment_values.clone()
            })
            .expect("filtered extend");
        servers.push(MockServer {
            label: idx.to_string(),
            store: Arc::new(Mutex::new(store)),
        });
    }

    let mut full_store = UnauthStore::new(params.clone());
    full_store
        .extend_with_segments_filtered(
            params.segment_len as u64,
            &(0..params.shard_registry.len()).collect::<Vec<_>>(),
            |_| segment_values.clone(),
        )
        .expect("full extend");
    let expected = full_store
        .repair_row_from_shards(0, params.row_labels[0])
        .expect("expected row repair");

    let client = UnauthClient::new_with_owned_indices(
        MockTransport,
        params.clone(),
        servers[0].clone(),
        servers.clone(),
        owned_indices,
        8,
    );
    let repaired = client
        .repair_row_from_network(0, 0)
        .expect("network row repair");

    let expected_values: Vec<_> = expected.into_iter().map(encode_value).collect();
    assert_eq!(repaired, expected_values);
}

#[test]
fn client_network_row_repair_reconstructs_multiple_joining_party_rows_from_live_cluster() {
    let params = build_params();
    let segment_values = build_segment_values(&params);
    let owned_indices = vec![vec![0usize], vec![1usize], vec![2usize]];
    let mut servers = Vec::new();
    for (idx, claimed) in owned_indices.iter().cloned().enumerate() {
        let mut store = UnauthStore::new(params.clone());
        store
            .extend_with_segments_filtered(params.segment_len as u64, &claimed, |_| {
                segment_values.clone()
            })
            .expect("filtered extend");
        servers.push(MockServer {
            label: idx.to_string(),
            store: Arc::new(Mutex::new(store)),
        });
    }

    let mut full_store = UnauthStore::new(params.clone());
    full_store
        .extend_with_segments_filtered(
            params.segment_len as u64,
            &(0..params.shard_registry.len()).collect::<Vec<_>>(),
            |_| segment_values.clone(),
        )
        .expect("full extend");
    let expected_row_2 = full_store
        .repair_row_from_shards(0, params.row_labels[2])
        .expect("expected row 2");
    let expected_row_0 = full_store
        .repair_row_from_shards(0, params.row_labels[0])
        .expect("expected row 0");

    let client = UnauthClient::new_with_owned_indices(
        ShardOnlyTransport,
        params.clone(),
        servers[0].clone(),
        servers.clone(),
        owned_indices,
        8,
    );
    let repaired = client
        .repair_rows_from_network(0, &[2, 0])
        .expect("network row repair batch");

    let repaired_values: Vec<_> = repaired
        .into_iter()
        .map(|(row_label_index, values)| (row_label_index, values))
        .collect();
    assert_eq!(
        repaired_values,
        vec![
            (2, expected_row_2.into_iter().map(encode_value).collect()),
            (0, expected_row_0.into_iter().map(encode_value).collect())
        ]
    );
}

#[test]
fn client_network_claim_repair_matches_split_shard_and_row_repair() {
    let params = build_params();
    let segment_values = build_segment_values(&params);
    let owned_indices = vec![vec![0usize], vec![1usize], vec![2usize], vec![3usize]];
    let mut servers = Vec::new();
    for (idx, claimed) in owned_indices.iter().cloned().enumerate() {
        let mut store =
            UnauthStore::new_with_claims(params.clone(), build_party_claims(&params, &claimed));
        store
            .extend_with_segments_filtered(params.segment_len as u64, &claimed, |_| {
                segment_values.clone()
            })
            .expect("filtered extend");
        servers.push(MockServer {
            label: idx.to_string(),
            store: Arc::new(Mutex::new(store)),
        });
    }

    let target_claim = ClaimBundle::exact_for_registry_index(&params, 0).expect("target claim");
    let helper_servers = vec![servers[1].clone(), servers[2].clone(), servers[3].clone()];
    let helper_owned_indices = vec![vec![1usize], vec![2usize], vec![3usize]];
    let client = UnauthClient::new_with_owned_indices(
        MockTransport,
        params.clone(),
        helper_servers[0].clone(),
        helper_servers,
        helper_owned_indices,
        params.segment_len as u64,
    );

    let plan = client
        .prepare_claim_recovery(&target_claim)
        .expect("claim recovery plan");
    let repaired = client
        .repair_claim_segment_fp_with_plan(0, &plan)
        .expect("fused claim repair");

    let target_rows = target_claim
        .row_label_indices(&params)
        .expect("target claim rows");
    let legacy_shards = client
        .repair_shards_from_network(0, &[target_claim.registry_index])
        .expect("legacy shard repair");
    let legacy_rows = client
        .repair_rows_from_network(0, &target_rows)
        .expect("legacy row repair");

    let expected_shards: Vec<_> = legacy_shards
        .into_iter()
        .map(|(registry_index, values)| {
            (
                registry_index,
                values.into_iter().map(decode_value).collect::<Vec<_>>(),
            )
        })
        .collect();
    let expected_rows: Vec<_> = legacy_rows
        .into_iter()
        .map(|(row_label_index, values)| {
            (
                row_label_index,
                values.into_iter().map(decode_value).collect::<Vec<_>>(),
            )
        })
        .collect();

    assert_eq!(repaired.shards, expected_shards);
    assert_eq!(repaired.rows, expected_rows);

    let mut recovered = UnauthStore::new_with_claims(params.clone(), vec![target_claim.clone()]);
    recovered
        .extend(params.segment_len as u64)
        .expect("extend recovered target");
    let install_plan = recovered
        .claim_segment_install_plan()
        .expect("claim install plan");
    recovered
        .install_recovered_claim_segment_state_owned_with_plan(0, &install_plan, repaired)
        .expect("install recovered claim");

    let expected_target = servers[0].store.lock().expect("target store lock").clone();
    assert!(
        recovered
            .coded
            .equals_for_tests(&params, &expected_target.coded)
    );
}

#[test]
fn client_network_row_repair_prefers_live_shards_over_row_reads() {
    let params = build_params();
    let segment_values = build_segment_values(&params);
    let owned_indices = vec![vec![0usize], vec![1usize], vec![2usize]];
    let mut servers = Vec::new();
    for (idx, claimed) in owned_indices.iter().cloned().enumerate() {
        let mut store = UnauthStore::new(params.clone());
        store
            .extend_with_segments_filtered(params.segment_len as u64, &claimed, |_| {
                segment_values.clone()
            })
            .expect("filtered extend");
        servers.push(MockServer {
            label: idx.to_string(),
            store: Arc::new(Mutex::new(store)),
        });
    }

    let mut full_store = UnauthStore::new(params.clone());
    full_store
        .extend_with_segments_filtered(
            params.segment_len as u64,
            &(0..params.shard_registry.len()).collect::<Vec<_>>(),
            |_| segment_values.clone(),
        )
        .expect("full extend");
    let expected = full_store
        .repair_row_from_shards(0, params.row_labels[0])
        .expect("expected row repair");

    let client = UnauthClient::new_with_owned_indices(
        ShardOnlyTransport,
        params.clone(),
        servers[0].clone(),
        servers.clone(),
        owned_indices,
        8,
    );
    let repaired = client
        .repair_row_from_network(0, 0)
        .expect("network row repair");

    let expected_values: Vec<_> = expected.into_iter().map(encode_value).collect();
    assert_eq!(repaired, expected_values);
}

#[test]
fn client_network_row_repair_requires_live_shards() {
    let params = build_params();
    let segment_values = build_segment_values(&params);
    let owned_indices = vec![vec![0usize], vec![1usize], vec![2usize]];
    let mut servers = Vec::new();
    for (idx, claimed) in owned_indices.iter().cloned().enumerate() {
        let mut store = UnauthStore::new(params.clone());
        store
            .extend_with_segments_filtered(params.segment_len as u64, &claimed, |_| {
                segment_values.clone()
            })
            .expect("filtered extend");
        servers.push(MockServer {
            label: idx.to_string(),
            store: Arc::new(Mutex::new(store)),
        });
    }

    let client = UnauthClient::new_with_owned_indices(
        RowOnlyTransport,
        params.clone(),
        servers[0].clone(),
        servers.clone(),
        owned_indices,
        8,
    );
    let err = client
        .repair_row_from_network(0, 0)
        .expect_err("row repair should require shard rpc");
    assert!(matches!(err, ClientError::NotEnoughShards));
}
