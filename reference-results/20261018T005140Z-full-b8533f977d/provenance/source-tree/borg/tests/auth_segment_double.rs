use append_only_log_demo::frontier::append_aligned_subtree_from_frontier;
use paper_a_optimized::auth::client::{AuthClient, AuthTransport, ClientError as AuthClientError};
use paper_a_optimized::auth::service::{
    AuthBatchWriteRequest, AuthBatchWriteResponse, AuthFreshSubtreeWriteRequest, AuthReadResponse,
    AuthRequest, AuthResponse, AuthServer, AuthService, AuthSubtreeReadResponse,
    AuthSubtreeUpperPathReadResponse, AuthWriteRequest, AuthWriteResponse,
};
use paper_a_optimized::core::NodeLinearization;
use paper_a_optimized::unauth::service::{UnauthRequest, UnauthResponse, UnauthService};
use paper_a_optimized::unauth::{ClaimBundle, build_segment_double_params};
use paper_a_optimized::{
    AuthParams, CodedArrayConfig, Fp, ShardShare, build_coded_params,
    build_shard_registry_deterministic, encode_value,
};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct LocalServer {
    label: usize,
    inner: Arc<Mutex<AuthServer>>,
}

#[derive(Clone)]
struct LocalTransport;

impl AuthTransport for LocalTransport {
    type Server = LocalServer;

    fn server_label(&self, server: &Self::Server) -> String {
        server.label.to_string()
    }

    fn auth_get_len(&self, server: &Self::Server) -> Result<u64, AuthClientError> {
        let mut guard = server.inner.lock().expect("server lock");
        match AuthService::handle_request(&mut *guard, AuthRequest::GetLen) {
            AuthResponse::Len { len } => Ok(len),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn auth_read(
        &self,
        server: &Self::Server,
        index: u64,
    ) -> Result<AuthReadResponse, AuthClientError> {
        let mut guard = server.inner.lock().expect("server lock");
        match AuthService::handle_request(&mut *guard, AuthRequest::Read { index }) {
            AuthResponse::Read { resp } => Ok(resp),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn auth_read_subtree(
        &self,
        server: &Self::Server,
        start_index: u64,
        leaf_count: usize,
    ) -> Result<AuthSubtreeReadResponse, AuthClientError> {
        let mut guard = server.inner.lock().expect("server lock");
        match AuthService::handle_request(
            &mut *guard,
            AuthRequest::ReadSubtree {
                start_index,
                leaf_count,
            },
        ) {
            AuthResponse::ReadSubtree { resp } => Ok(resp),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn auth_read_subtree_upper_path(
        &self,
        server: &Self::Server,
        start_index: u64,
        leaf_count: usize,
    ) -> Result<AuthSubtreeUpperPathReadResponse, AuthClientError> {
        let mut guard = server.inner.lock().expect("server lock");
        match AuthService::handle_request(
            &mut *guard,
            AuthRequest::ReadSubtreeUpperPath {
                start_index,
                leaf_count,
            },
        ) {
            AuthResponse::ReadSubtreeUpperPath { resp } => Ok(resp),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn unauth_multi_read(
        &self,
        server: &Self::Server,
        indices: &[u64],
    ) -> Result<Vec<Vec<ShardShare>>, AuthClientError> {
        let mut guard = server.inner.lock().expect("server lock");
        match AuthService::handle_request(
            &mut *guard,
            AuthRequest::UnauthMultiRead {
                idxs: indices.to_vec(),
            },
        ) {
            AuthResponse::UnauthMultiRead { shares } => Ok(shares),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn auth_write(
        &self,
        server: &Self::Server,
        req: &AuthWriteRequest,
    ) -> Result<AuthWriteResponse, AuthClientError> {
        let mut guard = server.inner.lock().expect("server lock");
        match AuthService::handle_request(&mut *guard, AuthRequest::Write { req: req.clone() }) {
            AuthResponse::Write { resp } => Ok(resp),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn auth_write_batch(
        &self,
        server: &Self::Server,
        req: &AuthBatchWriteRequest,
    ) -> Result<AuthBatchWriteResponse, AuthClientError> {
        let mut guard = server.inner.lock().expect("server lock");
        match AuthService::handle_request(&mut *guard, AuthRequest::WriteBatch { req: req.clone() })
        {
            AuthResponse::WriteBatch { resp } => Ok(resp),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn auth_write_fresh_subtree(
        &self,
        server: &Self::Server,
        req: &AuthFreshSubtreeWriteRequest,
    ) -> Result<AuthBatchWriteResponse, AuthClientError> {
        let mut guard = server.inner.lock().expect("server lock");
        match AuthService::handle_request(
            &mut *guard,
            AuthRequest::WriteFreshSubtree { req: req.clone() },
        ) {
            AuthResponse::WriteFreshSubtree { resp } => Ok(resp),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!("{:?}", other))),
        }
    }

    fn auth_extend(&self, server: &Self::Server, new_len: u64) -> Result<bool, AuthClientError> {
        let mut guard = server.inner.lock().expect("server lock");
        match AuthService::handle_request(
            &mut *guard,
            AuthRequest::Extend {
                new_leaves: new_len,
            },
        ) {
            AuthResponse::Extend { ok } => Ok(ok),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!("{:?}", other))),
        }
    }
}

#[test]
fn auth_batch_reads_survive_local_unauth_segment_double() {
    let n_grid = 32usize;
    let s = 16usize;
    let shard_count = 8usize;
    let shard_registry = build_shard_registry_deterministic(shard_count, s, n_grid);
    let params = build_coded_params(&CodedArrayConfig {
        segment_len: n_grid,
        s,
        shard_count,
        shard_registry,
        party_registry_indices: (0..shard_count).map(|idx| vec![idx]).collect(),
        seed: 42,
    });
    let auth = AuthParams::new(3);
    let allocated_leaves = 4usize;
    let mut servers = Vec::new();
    for registry_index in 0..shard_count {
        let claim =
            ClaimBundle::exact_for_registry_index(&params, registry_index).expect("exact claim");
        let server = AuthServer::new_with_assignments_and_claims(
            params.clone(),
            auth.clone(),
            NodeLinearization::Preorder,
            allocated_leaves,
            Some(vec![registry_index]),
            Some(vec![claim]),
        )
        .expect("auth server");
        servers.push(LocalServer {
            label: registry_index,
            inner: Arc::new(Mutex::new(server)),
        });
    }

    let mut client = AuthClient::new(
        LocalTransport,
        params.clone(),
        auth,
        NodeLinearization::Preorder,
        allocated_leaves,
        servers[0].clone(),
        servers.clone(),
    );
    let updates: Vec<_> = (0..allocated_leaves as u64)
        .map(|idx| (idx, encode_value(Fp::from(idx + 1))))
        .collect();
    client.write_batch(&updates).expect("batch write");
    let before = client
        .read_batch(&[0, 1, 2, 3])
        .expect("read before segment-double");

    for server in &servers {
        let mut guard = server.inner.lock().expect("server lock");
        match UnauthService::handle_request(
            &mut *guard,
            UnauthRequest::SegmentDouble {
                helper_shards: Vec::new(),
            },
        ) {
            UnauthResponse::SegmentDouble { ok: true } => {}
            other => panic!("unexpected segment-double response: {:?}", other),
        }
    }
    client.params = build_segment_double_params(&client.params).expect("target params");
    client.clear_trusted_root();

    let after = client
        .read_batch(&[0, 1, 2, 3])
        .expect("read after segment-double");
    assert_eq!(after, before);
}

#[test]
fn auth_append_only_frontier_survives_segment_double_then_next_append() {
    let n_grid = 8usize;
    let s = 4usize;
    let shard_count = 8usize;
    let shard_registry = build_shard_registry_deterministic(shard_count, s, n_grid);
    let params = build_coded_params(&CodedArrayConfig {
        segment_len: n_grid,
        s,
        shard_count,
        shard_registry,
        party_registry_indices: (0..shard_count).map(|idx| vec![idx]).collect(),
        seed: 43,
    });
    let auth = AuthParams::new(4);
    let mut servers = Vec::new();
    for registry_index in 0..shard_count {
        let claim =
            ClaimBundle::exact_for_registry_index(&params, registry_index).expect("exact claim");
        let server = AuthServer::new_with_assignments_and_claims(
            params.clone(),
            auth.clone(),
            NodeLinearization::Preorder,
            0,
            Some(vec![registry_index]),
            Some(vec![claim]),
        )
        .expect("auth server");
        servers.push(LocalServer {
            label: registry_index,
            inner: Arc::new(Mutex::new(server)),
        });
    }

    let mut client = AuthClient::new(
        LocalTransport,
        params.clone(),
        auth,
        NodeLinearization::Preorder,
        0,
        servers[0].clone(),
        servers.clone(),
    );
    let first = (0..4)
        .map(|idx| encode_value(Fp::from(10 + idx as u64)))
        .collect::<Vec<_>>();
    let frontier =
        append_aligned_subtree_from_frontier(&mut client, 0, None, &first).expect("first append");
    assert_eq!(
        client.read(0).expect("read first before segment-double"),
        first[0]
    );

    for server in &servers {
        let mut guard = server.inner.lock().expect("server lock");
        match UnauthService::handle_request(
            &mut *guard,
            UnauthRequest::SegmentDouble {
                helper_shards: Vec::new(),
            },
        ) {
            UnauthResponse::SegmentDouble { ok: true } => {}
            other => panic!("unexpected segment-double response: {:?}", other),
        }
    }
    client.params = build_segment_double_params(&client.params).expect("target params");
    assert_eq!(
        client.read(0).expect("read first after segment-double"),
        first[0]
    );

    let second = (0..4)
        .map(|idx| encode_value(Fp::from(100 + idx as u64)))
        .collect::<Vec<_>>();
    append_aligned_subtree_from_frontier(&mut client, 4, Some(&frontier), &second)
        .expect("second append after segment-double");
    assert_eq!(
        client.read(0).expect("read first after second append"),
        first[0]
    );
    assert_eq!(
        client.read(4).expect("read second after segment-double"),
        second[0]
    );
}
