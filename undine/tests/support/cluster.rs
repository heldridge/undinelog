//! A local cluster of authenticated servers, one claim each, for tests.
#![allow(dead_code)]
#![allow(unused_imports)]

use paper_a_optimized::auth::client::{AuthClient, AuthTransport, ClientError as AuthClientError};
use paper_a_optimized::auth::service::{
    AuthBatchWriteRequest, AuthBatchWriteResponse, AuthFreshSubtreeWriteRequest, AuthReadResponse,
    AuthRequest, AuthResponse, AuthServer, AuthService, AuthSubtreeReadResponse,
    AuthSubtreeUpperPathReadResponse, AuthWriteRequest, AuthWriteResponse,
};
use paper_a_optimized::core::NodeLinearization;
use paper_a_optimized::unauth::service::{UnauthRequest, UnauthResponse, UnauthService};
use paper_a_optimized::unauth::{
    ClaimBundle, CodedArrayParams, PartialStitchEvals, SourceHalfState, UnauthOptimizedReadRequest,
    UnauthOptimizedReadResponse, UnauthReadObjectId, build_kappa_double_params,
    build_kappa_halve_params, build_params_for_kappa, decode_leading_source_half,
};
use paper_a_optimized::{
    AuthParams, CodedArrayConfig, Fp, ShardShare, Value256, build_coded_params,
    build_shard_registry_deterministic, encode_value,
};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct LocalServer {
    pub label: usize,
    pub inner: Arc<Mutex<AuthServer>>,
    /// Return wrong shares on point reads.
    pub corrupt_reads: bool,
}

#[derive(Clone)]
pub struct LocalTransport;

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
        kappa: usize,
    ) -> Result<AuthReadResponse, AuthClientError> {
        let mut guard = server.inner.lock().expect("server lock");
        match AuthService::handle_request(&mut *guard, AuthRequest::Read { index, kappa }) {
            AuthResponse::Read { mut resp } => {
                if server.corrupt_reads {
                    for node in &mut resp.nodes {
                        for share in &mut node.shares {
                            let value = share.value().expect("share value") + Fp::from(1u64);
                            *share = ShardShare::new(share.registry_index(), Some(value));
                        }
                    }
                }
                Ok(resp)
            }
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

    fn supports_unauth_optimized_reads(&self) -> bool {
        true
    }

    fn unauth_optimized_read(
        &self,
        server: &Self::Server,
        request: &UnauthOptimizedReadRequest,
    ) -> Result<UnauthOptimizedReadResponse, AuthClientError> {
        let guard = server.inner.lock().expect("server lock");
        Ok(guard.handle_optimized_read(request))
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

    #[cfg(feature = "private")]
    fn auth_read_private(
        &self,
        server: &Self::Server,
        request: &paper_a_optimized::private::PrivateReadRequest,
    ) -> Result<paper_a_optimized::private::PrivateReadResponse, AuthClientError> {
        let mut guard = server.inner.lock().expect("server lock");
        match AuthService::handle_request(
            &mut *guard,
            AuthRequest::PrivateRead {
                req: request.clone(),
            },
        ) {
            AuthResponse::PrivateRead { mut resp } => {
                if server.corrupt_reads {
                    for reply in &mut resp.replies {
                        for ciphertext in &mut reply.response {
                            for byte in ciphertext.iter_mut().step_by(7) {
                                *byte ^= 0x5a;
                            }
                        }
                    }
                }
                Ok(resp)
            }
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

/// One auth server per registry entry, each bound to its one claim.
pub fn build_servers(
    params: &CodedArrayParams,
    auth: &AuthParams,
    allocated_leaves: usize,
) -> Vec<LocalServer> {
    (0..params.shard_registry.len())
        .map(|registry_index| {
            let claim =
                ClaimBundle::exact_for_registry_index(params, registry_index).expect("exact claim");
            let server = AuthServer::new_with_assignments_and_claims(
                params.clone(),
                auth.clone(),
                NodeLinearization::Preorder,
                allocated_leaves,
                Some(vec![registry_index]),
                Some(vec![claim]),
            )
            .expect("auth server");
            LocalServer {
                label: registry_index,
                inner: Arc::new(Mutex::new(server)),
                corrupt_reads: false,
            }
        })
        .collect()
}

pub fn build_params(kappa: usize, shard_count: usize, seed: u64) -> CodedArrayParams {
    build_coded_params(&CodedArrayConfig {
        kappa,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count),
        party_registry_indices: (0..shard_count).map(|idx| vec![idx]).collect(),
        seed,
    })
}

pub fn written_cluster(
    kappa: usize,
    shard_count: usize,
    tree_height: usize,
    leaves: usize,
    seed: u64,
) -> (
    CodedArrayParams,
    AuthParams,
    Vec<LocalServer>,
    Vec<Value256>,
    Value256,
) {
    let params = build_params(kappa, shard_count, seed);
    let auth = AuthParams::new(tree_height);
    let servers = build_servers(&params, &auth, leaves);
    let mut client = reader(&params, &auth, leaves, &servers);
    let values = (0..leaves as u64)
        .map(|idx| encode_value(Fp::from(500 + 11 * idx)))
        .collect::<Vec<_>>();
    let updates = values
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx as u64, *value))
        .collect::<Vec<_>>();
    client.write_batch(&updates).expect("batch write");
    let root = client.trusted_root().expect("root after write");
    (params, auth, servers, values, root)
}

/// Client reading at the threshold of `params`, with no trusted root yet.
pub fn reader(
    params: &CodedArrayParams,
    auth: &AuthParams,
    leaves: usize,
    servers: &[LocalServer],
) -> AuthClient<LocalTransport> {
    AuthClient::new(
        LocalTransport,
        params.clone(),
        auth.clone(),
        NodeLinearization::Preorder,
        leaves,
        servers[0].clone(),
        servers.to_vec(),
    )
}

