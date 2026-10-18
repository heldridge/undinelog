use paper_a_optimized::coded_array::params::CodedArrayConfig;
use paper_a_optimized::unauth::client::{ClientError, UnauthClient, UnauthTransport};
use paper_a_optimized::unauth::service::{UnauthRequest, UnauthResponse, UnauthService};
use paper_a_optimized::unauth::{
    ClaimBundle, DeltaMode, MigrateError, PartialStitchEvals, SourceHalfState, UnauthStore,
    aligned_block_indices, build_kappa_double_params,
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

/// Store with the params and claims of `actual`, encoded directly from the
/// public array values.
fn exact_store_matching_actual_claims(
    actual: &UnauthStore,
    semantic_values: &[paper_a_optimized::Value256],
) -> UnauthStore {
    let params = actual.params.clone();
    assert_eq!(semantic_values.len(), actual.len() as usize);
    let mut registry_indices: Vec<_> = actual
        .coded
        .shard_keys()
        .map(|shard_key| {
            *params
                .shard_registry_by_key
                .get(shard_key)
                .expect("shard owner in registry")
        })
        .collect();
    registry_indices.sort_unstable();
    let claims = build_party_claims(&params, &registry_indices);
    let zero = encode_value(Fp::from(0u64));
    let mut store = UnauthStore::new_with_claims(params, claims);
    store
        .extend_with_values(actual.len(), |idx| {
            semantic_values.get(idx as usize).copied().unwrap_or(zero)
        })
        .expect("materialize exact store");
    store
}

fn row_mismatches(actual: &UnauthStore, exact: &UnauthStore) -> Vec<String> {
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
        mismatches.push(format!("registry {}", registry_index));
    }
    mismatches.sort();
    mismatches.dedup();
    mismatches
}

/// Check that the server's primary columns and auxiliary rows are the direct
/// encoding of the public values the client currently reads.
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
    let exact = exact_store_matching_actual_claims(&store, &semantic_values);
    let mismatches = row_mismatches(&store, &exact);
    assert!(
        mismatches.is_empty(),
        "{}: {} mismatched rows: {}",
        context,
        mismatches.len(),
        mismatches.join(", ")
    );
    assert!(
        store.coded.equals_for_tests(&store.params, &exact.coded),
        "{}: coded state differs from the direct encoding",
        context
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
struct NoStitchEvalTransport;

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

    fn unauth_kappa_double(&self, server: &Self::Server) -> Result<bool, ClientError> {
        let mut store = server.store.lock().expect("store lock");
        match store.handle_request(UnauthRequest::KappaDouble) {
            UnauthResponse::KappaDouble { ok } => Ok(ok),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn unauth_kappa_halve(
        &self,
        server: &Self::Server,
        source_halves: &[SourceHalfState],
    ) -> Result<bool, ClientError> {
        let mut store = server.store.lock().expect("store lock");
        match store.handle_request(UnauthRequest::KappaHalve {
            source_halves: source_halves.to_vec(),
        }) {
            UnauthResponse::KappaHalve { ok } => Ok(ok),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn unauth_eval_partial_stitches(
        &self,
        server: &Self::Server,
        segs: &[usize],
        col_labels: &[Fp],
        row_labels: &[Fp],
    ) -> Result<Vec<Vec<PartialStitchEvals>>, ClientError> {
        let mut store = server.store.lock().expect("store lock");
        match store.handle_request(UnauthRequest::EvalPartialStitches {
            segs: segs.to_vec(),
            col_labels: col_labels.to_vec(),
            row_labels: row_labels.to_vec(),
        }) {
            UnauthResponse::EvalPartialStitches { evals } => Ok(evals),
            UnauthResponse::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!("{:?}", other))),
        }
    }
}

impl UnauthTransport for NoStitchEvalTransport {
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

    fn unauth_kappa_double(&self, server: &Self::Server) -> Result<bool, ClientError> {
        MockTransport.unauth_kappa_double(server)
    }

    fn unauth_kappa_halve(
        &self,
        server: &Self::Server,
        source_halves: &[SourceHalfState],
    ) -> Result<bool, ClientError> {
        MockTransport.unauth_kappa_halve(server, source_halves)
    }

    fn unauth_eval_partial_stitches(
        &self,
        _server: &Self::Server,
        _segs: &[usize],
        _col_labels: &[Fp],
        _row_labels: &[Fp],
    ) -> Result<Vec<Vec<PartialStitchEvals>>, ClientError> {
        Err(ClientError::Net(
            "partial-stitch evaluations disabled".to_string(),
        ))
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

    fn unauth_kappa_double(&self, server: &Self::Server) -> Result<bool, ClientError> {
        MockTransport.unauth_kappa_double(server)
    }

    fn unauth_kappa_halve(
        &self,
        server: &Self::Server,
        source_halves: &[SourceHalfState],
    ) -> Result<bool, ClientError> {
        MockTransport.unauth_kappa_halve(server, source_halves)
    }

    fn unauth_eval_partial_stitches(
        &self,
        server: &Self::Server,
        segs: &[usize],
        col_labels: &[Fp],
        row_labels: &[Fp],
    ) -> Result<Vec<Vec<PartialStitchEvals>>, ClientError> {
        MockTransport.unauth_eval_partial_stitches(server, segs, col_labels, row_labels)
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

    fn unauth_kappa_double(&self, server: &Self::Server) -> Result<bool, ClientError> {
        MockTransport.unauth_kappa_double(server)
    }

    fn unauth_kappa_halve(
        &self,
        server: &Self::Server,
        source_halves: &[SourceHalfState],
    ) -> Result<bool, ClientError> {
        MockTransport.unauth_kappa_halve(server, source_halves)
    }

    fn unauth_eval_partial_stitches(
        &self,
        server: &Self::Server,
        segs: &[usize],
        col_labels: &[Fp],
        row_labels: &[Fp],
    ) -> Result<Vec<Vec<PartialStitchEvals>>, ClientError> {
        MockTransport.unauth_eval_partial_stitches(server, segs, col_labels, row_labels)
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

    fn unauth_kappa_double(&self, server: &Self::Server) -> Result<bool, ClientError> {
        MockTransport.unauth_kappa_double(server)
    }

    fn unauth_kappa_halve(
        &self,
        server: &Self::Server,
        source_halves: &[SourceHalfState],
    ) -> Result<bool, ClientError> {
        MockTransport.unauth_kappa_halve(server, source_halves)
    }

    fn unauth_eval_partial_stitches(
        &self,
        server: &Self::Server,
        segs: &[usize],
        col_labels: &[Fp],
        row_labels: &[Fp],
    ) -> Result<Vec<Vec<PartialStitchEvals>>, ClientError> {
        MockTransport.unauth_eval_partial_stitches(server, segs, col_labels, row_labels)
    }
}
fn build_params() -> paper_a_optimized::CodedArrayParams {
    let kappa: usize = 4;
    let shard_count: usize = 6;
    let shard_registry = build_shard_registry_deterministic(shard_count);
    let party_registry_indices = vec![(0..shard_count).collect()];
    build_coded_params(&CodedArrayConfig {
        kappa,
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

/// One claim per live server; registry index 5 is not held by any of them.
fn build_owned_indices() -> Vec<Vec<usize>> {
    vec![
        vec![0usize],
        vec![1usize],
        vec![2usize],
        vec![3usize],
        vec![4usize],
    ]
}

/// Parameters for the kappa-doubling tests: threshold 4 at the base level and
/// 8 after one stitch, with eight claims split among the given parties.
fn build_stitch_params(
    party_registry_indices: Vec<Vec<usize>>,
    seed: u64,
) -> paper_a_optimized::CodedArrayParams {
    let kappa: usize = 4;
    let shard_count: usize = 8;
    build_coded_params(&CodedArrayConfig {
        kappa,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count),
        party_registry_indices,
        seed,
    })
}

/// Values of segment `seg` such that public index `idx` holds `idx + 1`.
fn build_indexed_segment_values(
    params: &paper_a_optimized::CodedArrayParams,
    seg: usize,
) -> Vec<[u8; 32]> {
    (0..params.n_grid)
        .map(|offset| encode_value(Fp::from((seg * params.n_grid + offset + 1) as u64)))
        .collect()
}

#[test]
fn client_block_read_matches_point_reads_on_same_semantic_block() {
    let params = build_params();
    let segment_values = build_segment_values(&params);
    let owned_indices = build_owned_indices();
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
        params.segment_len as u64,
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
    assert_ne!(block0, block1);
}

#[test]
fn client_block_read_uses_row_path_when_point_reads_are_disabled() {
    let params = build_params();
    let segment_values = build_segment_values(&params);
    let owned_indices = build_owned_indices();
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
        params.segment_len as u64,
    );

    let block0 = client.read_block(0).expect("row-path block read");
    assert_eq!(
        block0,
        vec![
            encode_value(Fp::from(1u64)),
            encode_value(Fp::from(2u64)),
            encode_value(Fp::from(3u64)),
            encode_value(Fp::from(4u64)),
        ]
    );
    let block2 = client.read_block(2).expect("row-path block read");
    assert_eq!(
        block2,
        vec![
            encode_value(Fp::from(9u64)),
            encode_value(Fp::from(10u64)),
            encode_value(Fp::from(11u64)),
            encode_value(Fp::from(12u64)),
        ]
    );
}

#[test]
fn client_block_batch_read_matches_repeated_row_path_reads() {
    let params = build_params();
    let segment_values = build_segment_values(&params);
    let owned_indices = build_owned_indices();
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
        params.segment_len as u64,
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
    let owned_indices = build_owned_indices();
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
        params.segment_len as u64,
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
    assert_eq!(
        block0,
        vec![
            encode_value(Fp::from(12u64)),
            encode_value(Fp::from(2u64)),
            encode_value(Fp::from(10u64)),
            encode_value(Fp::from(9u64)),
        ]
    );
    assert_eq!(
        client.read_block(1).expect("read block 1"),
        segment_values[params.s..2 * params.s].to_vec()
    );
}

#[test]
fn client_array_view_kappa_doubled_block_write_survives_kappa_halve_on_same_logical_points() {
    let shard_count: usize = 8;
    let params = build_stitch_params(vec![(0..shard_count).collect()], 31);
    let len = (4 * params.segment_len) as u64;
    let segment_values = vec![encode_value(Fp::from(0u64)); params.n_grid];
    let owned_indices = vec![(0..shard_count).collect::<Vec<_>>()];

    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments_filtered(len, &owned_indices[0], |_| segment_values.clone())
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
        len,
    );

    client.kappa_double().expect("kappa-double");

    let block = 1u64;
    let stable_indices = active_block_indices(&client.params, block, len);
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

    client.kappa_halve().expect("kappa-halve");

    let kappa_halved_points = client
        .multi_read(&stable_indices)
        .expect("read kappa-halve logical points");
    assert_eq!(kappa_halved_points, expected_kappa_doubled_block);
}

#[test]
fn client_array_view_kappa_doubled_block_write_kappa_halve_to_readable_block() {
    let shard_count: usize = 8;
    let params = build_stitch_params(vec![(0..shard_count).collect()], 39);
    let len = (4 * params.segment_len) as u64;
    let segment_values = vec![encode_value(Fp::from(0u64)); params.n_grid];
    let owned_indices = vec![(0..shard_count).collect::<Vec<_>>()];

    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments_filtered(len, &owned_indices[0], |_| segment_values.clone())
        .expect("full extend");

    let servers = vec![MockServer {
        label: "0".to_string(),
        store: Arc::new(Mutex::new(store)),
    }];

    let source_block = 1u64;
    // After the stitch, block 2 interleaves source blocks 1 and 5.
    let kappa_doubled_block = 2u64;
    let kappa_doubled_block_indices = active_block_indices(
        &build_kappa_double_params(&params).expect("kappa-double params"),
        kappa_doubled_block,
        len,
    );
    let mut source_client = UnauthClient::new_with_owned_indices(
        MockTransport,
        params.clone(),
        servers[0].clone(),
        servers.clone(),
        owned_indices.clone(),
        len,
    );
    source_client
        .write_block(
            source_block,
            &[
                (0usize, Fp::from(0x21u64)),
                (2usize, Fp::from(0x34u64)),
                (3usize, Fp::from(0x55u64)),
            ],
            DeltaMode::Aggregate,
        )
        .expect("source block write");
    assert_rows_match_column_reconstruction(
        &mut source_client,
        &servers[0],
        "full-store source rows before kappa-double",
    );
    let points_before_kappa_double = source_client
        .multi_read(&kappa_doubled_block_indices)
        .expect("read points before kappa-double");

    source_client.kappa_double().expect("kappa-double");

    let kappa_doubled_params = servers[0].store.lock().expect("store lock").params.clone();
    assert_eq!(kappa_doubled_params.need, 2 * params.need);
    let kappa_doubled_owned_indices =
        vec![(0..kappa_doubled_params.shard_registry.len()).collect::<Vec<_>>()];
    let mut kappa_doubled_client = UnauthClient::new_with_owned_indices(
        NoBlockUpdateTransport,
        kappa_doubled_params,
        servers[0].clone(),
        servers.clone(),
        kappa_doubled_owned_indices,
        len,
    );
    assert_rows_match_column_reconstruction(
        &mut kappa_doubled_client,
        &servers[0],
        "full-store kappa-double rows before update",
    );

    let before_kappa_doubled_block = kappa_doubled_client
        .read_block(kappa_doubled_block)
        .expect("read kappa-double block before write");
    assert_eq!(before_kappa_doubled_block, points_before_kappa_double);
    let deltas = [
        (1usize, Fp::from(0x66u64)),
        (4usize, Fp::from(0x77u64)),
        (6usize, Fp::from(0x88u64)),
    ];
    kappa_doubled_client
        .write_block(kappa_doubled_block, &deltas, DeltaMode::Aggregate)
        .expect("kappa-double block write");
    let expected_kappa_doubled_block =
        apply_expected_block_deltas(before_kappa_doubled_block, &deltas);
    assert_eq!(
        kappa_doubled_client
            .read_block(kappa_doubled_block)
            .expect("read kappa-double block after write"),
        expected_kappa_doubled_block
    );
    assert_rows_match_column_reconstruction(
        &mut kappa_doubled_client,
        &servers[0],
        "full-store kappa-double rows after update",
    );

    kappa_doubled_client.kappa_halve().expect("kappa-halve");

    let kappa_halved_params = servers[0].store.lock().expect("store lock").params.clone();
    assert_eq!(kappa_halved_params.need, params.need);
    assert_eq!(kappa_halved_params.layout_level, 0);
    let kappa_halved_owned_indices =
        vec![(0..kappa_halved_params.shard_registry.len()).collect::<Vec<_>>()];
    let mut kappa_halved_client = UnauthClient::new_with_owned_indices(
        MockTransport,
        kappa_halved_params.clone(),
        servers[0].clone(),
        servers.clone(),
        kappa_halved_owned_indices.clone(),
        len,
    );
    assert_rows_match_column_reconstruction(
        &mut kappa_halved_client,
        &servers[0],
        "full-store kappa-halve rows",
    );

    let kappa_halved_points = kappa_halved_client
        .multi_read(&kappa_doubled_block_indices)
        .expect("read kappa-halve points");
    assert_eq!(kappa_halved_points, expected_kappa_doubled_block);

    let kappa_halved_block = kappa_halved_client
        .read_block(source_block)
        .expect("read kappa-halve block");
    let kappa_halved_block_points = kappa_halved_client
        .multi_read(&active_block_indices(
            &kappa_halved_params,
            source_block,
            len,
        ))
        .expect("read kappa-halve block points");
    assert_eq!(kappa_halved_block, kappa_halved_block_points);
    let mut row_only_kappa_halved = UnauthClient::new_with_owned_indices(
        RowOnlyTransport,
        kappa_halved_params,
        servers[0].clone(),
        servers.clone(),
        kappa_halved_owned_indices,
        len,
    );
    assert_eq!(
        row_only_kappa_halved
            .read_block(source_block)
            .expect("row-path kappa-halve block"),
        kappa_halved_block
    );
}

#[test]
fn client_array_view_kappa_doubled_block_write_survives_kappa_halve_on_nonzero_full_store() {
    let shard_count: usize = 8;
    let params = build_stitch_params(vec![(0..shard_count).collect()], 41);
    let len = (4 * params.segment_len) as u64;
    let owned_indices = vec![(0..shard_count).collect::<Vec<_>>()];

    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments_filtered(len, &owned_indices[0], |seg| {
            build_indexed_segment_values(&params, seg)
        })
        .expect("full extend");

    let servers = vec![MockServer {
        label: "0".to_string(),
        store: Arc::new(Mutex::new(store)),
    }];
    let block = 2u64;
    let kappa_doubled_block_indices = active_block_indices(
        &build_kappa_double_params(&params).expect("kappa-double params"),
        block,
        len,
    );
    let mut client = UnauthClient::new_with_owned_indices(
        MockTransport,
        params,
        servers[0].clone(),
        servers,
        owned_indices,
        len,
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
        .expect("source block write");
    client
        .read_block(1)
        .expect("source block before kappa-double");
    let points_before_kappa_double = client
        .multi_read(&kappa_doubled_block_indices)
        .expect("read points before kappa-double");

    client.kappa_double().expect("kappa-double");

    let before_kappa_doubled_block = client
        .read_block(block)
        .expect("read kappa-double block before write");
    assert_eq!(before_kappa_doubled_block, points_before_kappa_double);
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
    client.kappa_halve().expect("kappa-halve");

    let kappa_halved_points = client
        .multi_read(&kappa_doubled_block_indices)
        .expect("read kappa-halve points");
    assert_eq!(
        kappa_halved_points,
        apply_expected_block_deltas(points_before_kappa_double, &deltas)
    );
}

#[test]
fn client_array_view_kappa_double_keeps_old_blocks_fetchable_inside_one_doubled_block() {
    let shard_count: usize = 8;
    let params = build_stitch_params(vec![(0..shard_count).collect()], 51);
    let len = (4 * params.segment_len) as u64;
    let owned_indices = vec![(0..shard_count).collect::<Vec<_>>()];

    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments_filtered(len, &owned_indices[0], |seg| {
            build_indexed_segment_values(&params, seg)
        })
        .expect("full extend");

    let servers = vec![MockServer {
        label: "0".to_string(),
        store: Arc::new(Mutex::new(store)),
    }];
    let old_block_indices = active_block_indices(&params, 1, len);
    let mut client = UnauthClient::new_with_owned_indices(
        MockTransport,
        params,
        servers[0].clone(),
        servers,
        owned_indices,
        len,
    );

    let old_block = client.read_block(1).expect("old block");
    let old_partner_block = client.read_block(5).expect("old partner block");
    client.kappa_double().expect("kappa-double");

    assert_eq!(
        client
            .multi_read(&old_block_indices)
            .expect("read old block points"),
        old_block
    );
    // Old block 1 of source segment 0 and old block 1 of source segment 1
    // share a Z-axis point after the stitch: they interleave in block 2.
    let interleaved: Vec<_> = old_block
        .iter()
        .zip(old_partner_block.iter())
        .flat_map(|(even, odd)| [*even, *odd])
        .collect();
    assert_eq!(
        client.read_block(2).expect("current kappa-double block"),
        interleaved
    );
    assert_ne!(
        client.read_block(1).expect("current kappa-double block"),
        interleaved
    );
}

#[test]
fn client_array_view_block_reads_need_row_rpcs_after_kappa_double() {
    let shard_count: usize = 8;
    let params = build_stitch_params(vec![(0..shard_count).collect()], 61);
    let len = (4 * params.segment_len) as u64;
    let owned_indices = vec![(0..shard_count).collect::<Vec<_>>()];
    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments_filtered(len, &owned_indices[0], |seg| {
            build_indexed_segment_values(&params, seg)
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
            len,
        );
        client.kappa_double().expect("kappa-double");
    }

    let kappa_doubled_params = servers[0].store.lock().expect("store lock").params.clone();
    let kappa_doubled_owned_indices =
        vec![(0..kappa_doubled_params.shard_registry.len()).collect::<Vec<_>>()];
    let mut point_client = UnauthClient::new_with_owned_indices(
        ShardOnlyTransport,
        kappa_doubled_params.clone(),
        servers[0].clone(),
        servers.clone(),
        kappa_doubled_owned_indices,
        len,
    );

    assert!(matches!(
        point_client.read_block(1),
        Err(ClientError::NotEnoughShards)
    ));
    let block_indices = active_block_indices(&kappa_doubled_params, 1, len);
    assert_eq!(
        point_client
            .multi_read(&block_indices)
            .expect("point reads of the block"),
        block_indices
            .iter()
            .map(|idx| encode_value(Fp::from(idx + 1)))
            .collect::<Vec<_>>()
    );
}

#[test]
fn client_array_view_block_batch_reads_need_row_rpcs_after_kappa_double() {
    let shard_count: usize = 8;
    let params = build_stitch_params(vec![(0..shard_count).collect()], 71);
    let len = (4 * params.segment_len) as u64;

    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments_filtered(len, &(0..shard_count).collect::<Vec<_>>(), |seg| {
            build_indexed_segment_values(&params, seg)
        })
        .expect("full extend");
    store.kappa_double().expect("kappa-double");

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
        len,
    );

    let blocks = [1u64, 3u64];
    assert!(matches!(
        point_client.read_block_batch(&blocks),
        Err(ClientError::NotEnoughShards)
    ));
}

#[test]
fn client_array_view_kappa_halve_keeps_doubled_block_fetchable_across_two_halved_blocks() {
    let shard_count: usize = 8;
    let params = build_stitch_params(vec![(0..shard_count).collect()], 81);
    let len = (4 * params.segment_len) as u64;
    let owned_indices = vec![(0..shard_count).collect::<Vec<_>>()];
    let mut store = UnauthStore::new(params.clone());
    store
        .extend_with_segments_filtered(len, &owned_indices[0], |seg| {
            build_indexed_segment_values(&params, seg)
        })
        .expect("full extend");
    let source_store = store.clone();
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
            len,
        );
        client.kappa_double().expect("kappa-double");
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
        len,
    );

    let kappa_halved_block = 1u64;
    let kappa_halved_partner_block = 5u64;
    let kappa_halved_block_indices = active_block_indices(&params, kappa_halved_block, len);
    let kappa_halved_block_under_kappa_double = client
        .multi_read(&kappa_halved_block_indices)
        .expect("read kappa-halve block points under kappa-double");
    let kappa_doubled_block = client.read_block(2).expect("kappa-double block");

    client.kappa_halve().expect("kappa-halve");

    let kappa_halved = client
        .read_block(kappa_halved_block)
        .expect("kappa-halve block");
    assert_eq!(kappa_halved, kappa_halved_block_under_kappa_double);
    let kappa_halved_partner = client
        .read_block(kappa_halved_partner_block)
        .expect("kappa-halve partner block");
    let interleaved: Vec<_> = kappa_halved
        .iter()
        .zip(kappa_halved_partner.iter())
        .flat_map(|(even, odd)| [*even, *odd])
        .collect();
    assert_eq!(interleaved, kappa_doubled_block);

    let store = servers[0].store.lock().expect("store lock");
    assert_eq!(store.segments(), source_store.segments());
    assert!(
        store
            .coded
            .equals_for_tests(&source_store.params, &source_store.coded)
    );
}

#[test]
fn client_array_view_kappa_doubled_block_write_survives_kappa_halve_with_claim_bound_servers() {
    let owned_indices = vec![vec![0usize, 1, 2, 3], vec![4usize, 5], vec![6usize, 7]];
    let params = build_stitch_params(owned_indices.clone(), 37);
    let len = (8 * params.segment_len) as u64;

    let mut servers = Vec::new();
    for (idx, claimed) in owned_indices.iter().cloned().enumerate() {
        let claims = build_party_claims(&params, &claimed);
        let mut store = UnauthStore::new_with_claims(params.clone(), claims.clone());
        store
            .extend_with_segments_for_claims(len, &claims, |seg| {
                build_indexed_segment_values(&params, seg)
            })
            .expect("claim-bound extend");
        if idx == 0 {
            let mut local_probe = store.clone();
            local_probe.kappa_double().expect("local kappa-double");
            assert!(matches!(
                local_probe.kappa_halve(&[]),
                Err(MigrateError::MissingSourceFragments)
            ));
        }
        servers.push(MockServer {
            label: idx.to_string(),
            store: Arc::new(Mutex::new(store)),
        });
    }

    let block = 2u64;
    let kappa_doubled_block_indices = active_block_indices(
        &build_kappa_double_params(&params).expect("kappa-double params"),
        block,
        len,
    );
    let mut client = UnauthClient::new_with_owned_indices(
        MockTransport,
        params,
        servers[0].clone(),
        servers,
        owned_indices,
        len,
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
        .expect("source block write");
    let points_before_kappa_double = client
        .multi_read(&kappa_doubled_block_indices)
        .expect("read points before kappa-double");

    client.kappa_double().expect("kappa-double");

    let before_kappa_doubled_block = client
        .read_block(block)
        .expect("read kappa-double block before write");
    assert_eq!(before_kappa_doubled_block, points_before_kappa_double);
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
    client.kappa_halve().expect("kappa-halve");

    let kappa_halved_points = client
        .multi_read(&kappa_doubled_block_indices)
        .expect("read kappa-halve points");
    assert_eq!(
        kappa_halved_points,
        apply_expected_block_deltas(points_before_kappa_double, &deltas)
    );
}

#[test]
fn client_kappa_halve_requires_partial_stitch_helpers_after_kappa_double() {
    let owned_indices = vec![vec![0usize, 1, 2, 3], vec![4usize, 5], vec![6usize, 7]];
    let params = build_stitch_params(owned_indices.clone(), 41);
    // Six source segments: the second group of four is stitched with padding.
    let len = (6 * params.segment_len) as u64;

    let mut servers = Vec::new();
    for claimed in owned_indices.iter().cloned() {
        let mut store = UnauthStore::new(params.clone());
        store
            .extend_with_segments_filtered(len, &claimed, |seg| {
                build_indexed_segment_values(&params, seg)
            })
            .expect("filtered extend");
        servers.push(MockServer {
            label: servers.len().to_string(),
            store: Arc::new(Mutex::new(store)),
        });
    }

    let mut client = UnauthClient::new_with_owned_indices(
        MockTransport,
        params.clone(),
        servers[0].clone(),
        servers.clone(),
        owned_indices,
        len,
    );

    let _source_block = client.read_block(1).expect("source block");
    let source_stores: Vec<_> = servers
        .iter()
        .map(|server| server.store.lock().expect("store lock").clone())
        .collect();
    client.kappa_double().expect("network kappa-double");

    let kappa_doubled_params = servers[0].store.lock().expect("store lock").params.clone();
    let kappa_doubled_owned_indices = kappa_doubled_params.party_registry_indices.clone();
    let mut client = UnauthClient::new_with_owned_indices(
        NoStitchEvalTransport,
        kappa_doubled_params.clone(),
        servers[0].clone(),
        servers.clone(),
        kappa_doubled_owned_indices.clone(),
        len,
    );

    assert!(matches!(
        client.kappa_halve(),
        Err(ClientError::NotEnoughShards)
    ));
    for server in &servers {
        assert_eq!(
            server.store.lock().expect("store lock").params.need,
            2 * params.need
        );
    }
    assert!(
        !client
            .read_block(1)
            .expect("kappa-double block remains readable")
            .is_empty()
    );

    let mut client = UnauthClient::new_with_owned_indices(
        MockTransport,
        kappa_doubled_params,
        servers[0].clone(),
        servers.clone(),
        kappa_doubled_owned_indices,
        len,
    );
    client
        .kappa_halve()
        .expect("kappa-halve with partial-stitch helpers");
    for (server, source_store) in servers.iter().zip(source_stores.iter()) {
        let store = server.store.lock().expect("store lock");
        assert_eq!(store.params.need, params.need);
        assert_eq!(store.segments(), source_store.segments());
        assert!(
            store
                .coded
                .equals_for_tests(&source_store.params, &source_store.coded)
        );
    }
}

#[test]
fn client_array_view_kappa_doubled_block_write_kappa_halve_to_readable_block_with_claim_bound_zero_store()
 {
    let owned_indices = vec![vec![0usize, 1, 2, 3], vec![4usize, 5], vec![6usize, 7]];
    let params = build_stitch_params(owned_indices.clone(), 43);
    let len = (8 * params.segment_len) as u64;
    let segment_values = vec![encode_value(Fp::from(0u64)); params.n_grid];

    let mut servers = Vec::new();
    for claimed in owned_indices.iter().cloned() {
        let claims = build_party_claims(&params, &claimed);
        let mut store = UnauthStore::new_with_claims(params.clone(), claims.clone());
        store
            .extend_with_segments_for_claims(len, &claims, |_| segment_values.clone())
            .expect("claim-bound extend");
        servers.push(MockServer {
            label: servers.len().to_string(),
            store: Arc::new(Mutex::new(store)),
        });
    }

    let source_block = 1u64;
    // After the stitch, block 2 interleaves source blocks 1 and 5.
    let kappa_doubled_block = 2u64;
    let kappa_doubled_block_indices = active_block_indices(
        &build_kappa_double_params(&params).expect("kappa-double params"),
        kappa_doubled_block,
        len,
    );
    let source_block_indices = active_block_indices(&params, source_block, len);
    let mut client = UnauthClient::new_with_owned_indices(
        MockTransport,
        params,
        servers[0].clone(),
        servers.clone(),
        owned_indices.clone(),
        len,
    );

    client
        .write_block(
            source_block,
            &[
                (0usize, Fp::from(0x21u64)),
                (2usize, Fp::from(0x34u64)),
                (3usize, Fp::from(0x55u64)),
            ],
            DeltaMode::Aggregate,
        )
        .expect("source block write");
    for server in &servers {
        assert_rows_match_column_reconstruction(
            &mut client,
            server,
            "claim-bound source rows before kappa-double",
        );
    }
    let source_block_before_kappa_double = client
        .read_block(source_block)
        .expect("source block before kappa-double");
    let points_before_kappa_double = client
        .multi_read(&kappa_doubled_block_indices)
        .expect("read points before kappa-double");

    client.kappa_double().expect("kappa-double");
    for server in &servers {
        assert_rows_match_column_reconstruction(
            &mut client,
            server,
            "claim-bound kappa-double rows before update",
        );
    }

    let before_kappa_doubled_block = client
        .read_block(kappa_doubled_block)
        .expect("read kappa-double block before write");
    assert_eq!(before_kappa_doubled_block, points_before_kappa_double);
    let deltas = [
        (1usize, Fp::from(0x66u64)),
        (4usize, Fp::from(0x77u64)),
        (6usize, Fp::from(0x88u64)),
    ];
    client
        .write_block(kappa_doubled_block, &deltas, DeltaMode::Aggregate)
        .expect("kappa-double block write");
    for server in &servers {
        assert_rows_match_column_reconstruction(
            &mut client,
            server,
            "claim-bound kappa-double rows after update",
        );
    }

    let expected_kappa_doubled_block =
        apply_expected_block_deltas(before_kappa_doubled_block, &deltas);
    assert_eq!(
        client
            .read_block(kappa_doubled_block)
            .expect("read kappa-double block"),
        expected_kappa_doubled_block
    );

    client.kappa_halve().expect("kappa-halve");
    for server in &servers {
        assert_rows_match_column_reconstruction(
            &mut client,
            server,
            "claim-bound kappa-halve rows",
        );
    }

    assert_eq!(
        client
            .multi_read(&kappa_doubled_block_indices)
            .expect("read kappa-halve points"),
        expected_kappa_doubled_block
    );

    let kappa_halved_block = client
        .read_block(source_block)
        .expect("read kappa-halve block");
    assert_eq!(
        client
            .multi_read(&source_block_indices)
            .expect("read kappa-halve aligned block by points"),
        kappa_halved_block
    );

    // One party with fewer than `kappa` claims cannot serve a block alone.
    let mut row_only_single = UnauthClient::new_with_owned_indices(
        RowOnlyTransport,
        client.params.clone(),
        servers[1].clone(),
        vec![servers[1].clone()],
        vec![owned_indices[1].clone()],
        len,
    );
    let mut row_only_all = UnauthClient::new_with_owned_indices(
        RowOnlyTransport,
        client.params.clone(),
        servers[0].clone(),
        servers.clone(),
        owned_indices,
        len,
    );
    assert!(matches!(
        row_only_single.read_block(source_block),
        Err(ClientError::NotEnoughShards)
    ));
    assert_eq!(
        row_only_all
            .read_block(source_block)
            .expect("all-party row path"),
        kappa_halved_block
    );
    assert_ne!(kappa_halved_block, source_block_before_kappa_double);
}

#[test]
fn client_network_repair_reconstructs_joining_party_shard_from_live_rows() {
    let params = build_params();
    let segment_values = build_segment_values(&params);
    let owned_indices = build_owned_indices();
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
        .repair_primary_shard_from_rows(0, params.shard_registry[5])
        .expect("expected repair");
    assert_eq!(
        full_store
            .coded
            .shard_segment(&params, &params.shard_registry[5].key, 0)
            .expect("stored shard"),
        expected.as_slice()
    );

    let client = UnauthClient::new_with_owned_indices(
        RowOnlyTransport,
        params.clone(),
        servers[0].clone(),
        servers.clone(),
        owned_indices,
        params.segment_len as u64,
    );
    let repaired = client
        .repair_shard_from_network(0, 5)
        .expect("network repair");

    let expected_values: Vec<_> = expected.into_iter().map(encode_value).collect();
    assert_eq!(repaired, expected_values);
}

#[test]
fn client_network_row_repair_reconstructs_joining_party_rows_from_live_cluster() {
    let params = build_params();
    let segment_values = build_segment_values(&params);
    let owned_indices = build_owned_indices();
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
        .repair_row_from_shards(0, params.row_labels[5])
        .expect("expected row repair");

    let client = UnauthClient::new_with_owned_indices(
        MockTransport,
        params.clone(),
        servers[0].clone(),
        servers.clone(),
        owned_indices,
        params.segment_len as u64,
    );
    let repaired = client
        .repair_row_from_network(0, 5)
        .expect("network row repair");

    let expected_values: Vec<_> = expected.into_iter().map(encode_value).collect();
    assert_eq!(repaired, expected_values);
}

#[test]
fn client_network_row_repair_reconstructs_multiple_joining_party_rows_from_live_cluster() {
    let params = build_params();
    let segment_values = build_segment_values(&params);
    let owned_indices = build_owned_indices();
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
    let expected_row_5 = full_store
        .repair_row_from_shards(0, params.row_labels[5])
        .expect("expected row 5");
    let expected_row_0 = full_store
        .repair_row_from_shards(0, params.row_labels[0])
        .expect("expected row 0");

    let client = UnauthClient::new_with_owned_indices(
        ShardOnlyTransport,
        params.clone(),
        servers[0].clone(),
        servers.clone(),
        owned_indices,
        params.segment_len as u64,
    );
    let repaired = client
        .repair_rows_from_network(0, &[5, 0])
        .expect("network row repair batch");

    let repaired_values: Vec<_> = repaired
        .into_iter()
        .map(|(row_label_index, values)| (row_label_index, values))
        .collect();
    assert_eq!(
        repaired_values,
        vec![
            (5, expected_row_5.into_iter().map(encode_value).collect()),
            (0, expected_row_0.into_iter().map(encode_value).collect())
        ]
    );
}

#[test]
fn client_network_claim_repair_matches_split_shard_and_row_repair() {
    let params = build_params();
    let segment_values = build_segment_values(&params);
    let owned_indices: Vec<_> = (0..params.shard_registry.len())
        .map(|registry_index| vec![registry_index])
        .collect();
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
    let helper_servers = servers[1..].to_vec();
    let helper_owned_indices = owned_indices[1..].to_vec();
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
    assert_eq!(target_rows, vec![target_claim.row_label_index()]);
    let split_shards = client
        .repair_shards_from_network(0, &[target_claim.registry_index])
        .expect("split shard repair");
    let split_rows = client
        .repair_rows_from_network(0, &target_rows)
        .expect("split row repair");

    let expected_shards: Vec<_> = split_shards
        .into_iter()
        .map(|(registry_index, values)| {
            (
                registry_index,
                values.into_iter().map(decode_value).collect::<Vec<_>>(),
            )
        })
        .collect();
    let expected_rows: Vec<_> = split_rows
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
    let owned_indices = build_owned_indices();
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
        params.segment_len as u64,
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
    let owned_indices = build_owned_indices();
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
        params.segment_len as u64,
    );
    let err = client
        .repair_row_from_network(0, 0)
        .expect_err("row repair should require shard rpc");
    assert!(matches!(err, ClientError::NotEnoughShards));
}
