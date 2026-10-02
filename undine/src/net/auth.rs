//! RPC helpers for the authenticated layer.
//!
//! This module provides network-facing servers and clients for authenticated
//! reads/writes. Core Merkle logic and data types live in `crate::auth`.

pub use crate::auth::client::ClientError;
use crate::auth::client::{AuthClient as CoreAuthClient, AuthTransport};
use crate::auth::merkle::NodeKey;
use crate::auth::service::{
    AuthBatchWriteRequest as ServiceAuthBatchWriteRequest,
    AuthBatchWriteResponse as ServiceAuthBatchWriteResponse,
    AuthBatchWriteUpdate as ServiceAuthBatchWriteUpdate,
    AuthFreshSubtreeWriteRequest as ServiceAuthFreshSubtreeWriteRequest,
    AuthPathRequest as ServiceAuthPathRequest, AuthReadNodeReply as ServiceAuthReadNodeReply,
    AuthReadResponse as ServiceAuthReadResponse, AuthRequest as ServiceAuthRequest,
    AuthResponse as ServiceAuthResponse, AuthService,
    AuthSubtreeReadResponse as ServiceAuthSubtreeReadResponse,
    AuthSubtreeUpperPathReadResponse as ServiceAuthSubtreeUpperPathReadResponse,
    AuthWriteRequest as ServiceAuthWriteRequest, AuthWriteResponse as ServiceAuthWriteResponse,
};
use crate::auth::{AuthParams, AuthSubtreeProof, MerkleTreeProof};
use crate::field::Fp;
use crate::net::registry::{MockRegistryFile, PartyInfo, build_unauth_registry_view};
use crate::net::unauth::{
    core_optimized_request_to_net, core_optimized_response_to_net,
    derived_row_labels_for_owned_indices, net_optimized_request_to_core,
    net_optimized_response_to_core,
};
use crate::net::wire::{
    AuthBatchWriteRequest as NetAuthBatchWriteRequest,
    AuthBatchWriteResponse as NetAuthBatchWriteResponse,
    AuthFreshSubtreeWriteRequest as NetAuthFreshSubtreeWriteRequest,
    AuthReadNodeReply as NetAuthReadNodeReply, AuthReadRequest as NetAuthReadRequest,
    AuthReadResponse as NetAuthReadResponse, AuthSubtreeReadRequest as NetAuthSubtreeReadRequest,
    AuthSubtreeReadResponse as NetAuthSubtreeReadResponse,
    AuthSubtreeUpperPathReadRequest as NetAuthSubtreeUpperPathReadRequest,
    AuthSubtreeUpperPathReadResponse as NetAuthSubtreeUpperPathReadResponse,
    AuthWriteRequest as NetAuthWriteRequest, AuthWriteResponse as NetAuthWriteResponse,
    NetAuthPath, NetAuthSubtreeProof, NetBatchWriteUpdate, NetDeltaMode, NetMerkleProof,
    NetNodeKey, NetValue, NetValueBlob,
};
use crate::net::{Message, Response};
use crate::shares::ShardShare;
use crate::unauth::DeltaMode;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

fn normalize_owned_indices(indices: &[usize]) -> Vec<usize> {
    let mut out = indices.to_vec();
    out.sort_unstable();
    out.dedup();
    out
}

fn map_unauth_client_error(err: crate::unauth::client::ClientError) -> ClientError {
    match err {
        crate::unauth::client::ClientError::Net(message) => ClientError::Net(message),
        crate::unauth::client::ClientError::Registry(message) => ClientError::Registry(message),
        crate::unauth::client::ClientError::InvalidConfig(message) => {
            ClientError::InvalidResponse(message)
        }
        crate::unauth::client::ClientError::InvalidIndex => ClientError::InvalidIndex,
        crate::unauth::client::ClientError::NotEnoughShards => ClientError::NotEnoughShards,
        crate::unauth::client::ClientError::InvalidResponse(message) => {
            ClientError::InvalidResponse(message)
        }
    }
}

fn net_leaf_count_to_usize(leaf_count: u64) -> Result<usize, Response> {
    usize::try_from(leaf_count).map_err(|_| Response::Error {
        message: "auth subtree leaf_count overflows usize".to_string(),
    })
}

fn pack_ordered_shard_values(shares: &[ShardShare]) -> crate::net::wire::NetMaybeFpBlob {
    let mut sorted = shares.to_vec();
    sorted.sort_unstable_by_key(|share| share.registry_index);
    crate::net::wire::NetMaybeFpBlob(sorted.into_iter().map(|share| share.value).collect())
}

fn decode_ordered_shard_values(
    owned_indices: &[usize],
    values: crate::net::wire::NetMaybeFpBlob,
) -> Result<Vec<ShardShare>, ClientError> {
    if values.0.is_empty() {
        return Ok(Vec::new());
    }
    if values.0.len() != owned_indices.len() {
        return Err(ClientError::InvalidResponse(format!(
            "ordered shard length mismatch: expected {} values, got {}",
            owned_indices.len(),
            values.0.len()
        )));
    }
    Ok(owned_indices
        .iter()
        .copied()
        .zip(values.0)
        .map(|(registry_index, value)| ShardShare {
            registry_index,
            value,
        })
        .collect())
}

/// Handle a generic network message for the authenticated mode.
pub fn handle_message<S: AuthService>(service: &mut S, msg: Message) -> Response {
    let req = match msg {
        Message::AuthRead { req } => ServiceAuthRequest::Read {
            index: req.leaf,
            kappa: crate::net::unauth::net_kappa_to_usize(req.kappa),
        },
        Message::AuthReadSubtree { req } => {
            let leaf_count = match net_leaf_count_to_usize(req.leaf_count) {
                Ok(leaf_count) => leaf_count,
                Err(resp) => return resp,
            };
            ServiceAuthRequest::ReadSubtree {
                start_index: req.start_index,
                leaf_count,
            }
        }
        Message::AuthReadSubtreeUpperPath { req } => {
            let leaf_count = match net_leaf_count_to_usize(req.leaf_count) {
                Ok(leaf_count) => leaf_count,
                Err(resp) => return resp,
            };
            ServiceAuthRequest::ReadSubtreeUpperPath {
                start_index: req.start_index,
                leaf_count,
            }
        }
        Message::AuthWrite { req } => ServiceAuthRequest::Write {
            req: net_write_request_to_service(req),
        },
        Message::AuthWriteBatch { req } => ServiceAuthRequest::WriteBatch {
            req: net_batch_write_request_to_service(req),
        },
        Message::AuthWriteFreshSubtree { req } => ServiceAuthRequest::WriteFreshSubtree {
            req: net_fresh_subtree_write_request_to_service(req),
        },
        Message::UnauthMultiRead { idxs } => ServiceAuthRequest::UnauthMultiRead { idxs },
        Message::AuthExtend { new_leaves } => ServiceAuthRequest::Extend { new_leaves },
        Message::AuthGetLen => ServiceAuthRequest::GetLen,
        #[cfg(feature = "private")]
        Message::AuthReadPrivate { req } => match crate::private::wire::request_from_net(&req) {
            Ok(req) => ServiceAuthRequest::PrivateRead { req },
            Err(message) => return Response::Error { message },
        },
        #[cfg(not(feature = "private"))]
        Message::AuthReadPrivate { .. } => {
            return Response::Error {
                message: "this server is built without private openings".to_string(),
            };
        }
        other => {
            return Response::Error {
                message: format!("unsupported auth message: {:?}", other),
            };
        }
    };

    match service.handle_request(req) {
        ServiceAuthResponse::Read { resp } => Response::AuthRead {
            resp: service_read_response_to_net(resp),
        },
        ServiceAuthResponse::ReadSubtree { resp } => Response::AuthReadSubtree {
            resp: service_subtree_read_response_to_net(resp),
        },
        ServiceAuthResponse::ReadSubtreeUpperPath { resp } => Response::AuthReadSubtreeUpperPath {
            resp: service_subtree_upper_path_response_to_net(resp),
        },
        ServiceAuthResponse::Write { resp } => Response::AuthWrite {
            resp: service_write_response_to_net(resp),
        },
        ServiceAuthResponse::WriteBatch { resp } => Response::AuthWriteBatch {
            resp: service_batch_write_response_to_net(resp),
        },
        ServiceAuthResponse::WriteFreshSubtree { resp } => Response::AuthWriteFreshSubtree {
            resp: service_batch_write_response_to_net(resp),
        },
        ServiceAuthResponse::UnauthMultiRead { shares } => Response::UnauthMultiRead {
            values: shares
                .into_iter()
                .map(|batch| pack_ordered_shard_values(&batch))
                .collect(),
        },
        ServiceAuthResponse::Extend { ok } => Response::AuthExtend { ok },
        ServiceAuthResponse::Len { len } => Response::AuthLen { len },
        #[cfg(feature = "private")]
        ServiceAuthResponse::PrivateRead { resp } => Response::AuthReadPrivate {
            resp: crate::private::wire::response_to_net(resp),
        },
        ServiceAuthResponse::Error { message } => Response::Error { message },
    }
}

fn service_read_response_to_net(resp: ServiceAuthReadResponse) -> NetAuthReadResponse {
    NetAuthReadResponse {
        nodes: resp
            .nodes
            .into_iter()
            .map(|node| NetAuthReadNodeReply {
                node: node_key_to_net(node.node),
                values: pack_ordered_shard_values(&node.shares),
            })
            .collect(),
    }
}

fn service_subtree_read_response_to_net(
    resp: ServiceAuthSubtreeReadResponse,
) -> NetAuthSubtreeReadResponse {
    NetAuthSubtreeReadResponse {
        leaf_nodes: resp
            .leaf_nodes
            .into_iter()
            .map(|node| NetAuthReadNodeReply {
                node: node_key_to_net(node.node),
                values: pack_ordered_shard_values(&node.shares),
            })
            .collect(),
        upper_path_nodes: resp
            .upper_path_nodes
            .into_iter()
            .map(|node| NetAuthReadNodeReply {
                node: node_key_to_net(node.node),
                values: pack_ordered_shard_values(&node.shares),
            })
            .collect(),
        raw_segment_shards: Vec::new(),
    }
}

fn service_subtree_upper_path_response_to_net(
    resp: ServiceAuthSubtreeUpperPathReadResponse,
) -> NetAuthSubtreeUpperPathReadResponse {
    NetAuthSubtreeUpperPathReadResponse {
        upper_path_nodes: resp
            .upper_path_nodes
            .into_iter()
            .map(|node| NetAuthReadNodeReply {
                node: node_key_to_net(node.node),
                values: pack_ordered_shard_values(&node.shares),
            })
            .collect(),
    }
}

fn net_write_request_to_service(req: NetAuthWriteRequest) -> ServiceAuthWriteRequest {
    ServiceAuthWriteRequest {
        old: req.old.0,
        new: req.new.0,
        path: ServiceAuthPathRequest {
            index: req.path.index,
            siblings: req.path.siblings.into_iter().map(|v| v.0).collect(),
            root: req.path.root.0,
        },
    }
}

fn service_write_response_to_net(resp: ServiceAuthWriteResponse) -> NetAuthWriteResponse {
    NetAuthWriteResponse {
        ok: resp.ok,
        root_new: resp.root_new.map(NetValue),
        error: resp.error,
    }
}

fn net_batch_write_request_to_service(
    req: NetAuthBatchWriteRequest,
) -> ServiceAuthBatchWriteRequest {
    let updates = req
        .updates
        .into_iter()
        .map(|update| ServiceAuthBatchWriteUpdate {
            index: update.index,
            old: update.old.0,
            new: update.new.0,
        })
        .collect();
    let proof = MerkleTreeProof {
        root: req.proof.root.0,
        nodes: req
            .proof
            .nodes
            .into_iter()
            .map(|(node, value)| (net_to_node_key(node), value.0))
            .collect(),
    };
    ServiceAuthBatchWriteRequest {
        updates,
        proof,
        delta_mode: DeltaMode::from(req.delta_mode),
    }
}

fn service_batch_write_response_to_net(
    resp: ServiceAuthBatchWriteResponse,
) -> NetAuthBatchWriteResponse {
    NetAuthBatchWriteResponse {
        ok: resp.ok,
        root_new: resp.root_new.map(NetValue),
        error: resp.error,
    }
}

fn net_read_response_to_service(
    owned_indices: &[usize],
    resp: NetAuthReadResponse,
) -> Result<ServiceAuthReadResponse, ClientError> {
    let mut nodes = Vec::with_capacity(resp.nodes.len());
    for node in resp.nodes {
        nodes.push(ServiceAuthReadNodeReply {
            node: net_to_node_key(node.node),
            shares: decode_ordered_shard_values(owned_indices, node.values)?,
        });
    }
    Ok(ServiceAuthReadResponse { nodes })
}

fn net_subtree_read_response_to_service(
    owned_indices: &[usize],
    resp: NetAuthSubtreeReadResponse,
) -> Result<ServiceAuthSubtreeReadResponse, ClientError> {
    let mut leaf_nodes = Vec::with_capacity(resp.leaf_nodes.len());
    for node in resp.leaf_nodes {
        leaf_nodes.push(ServiceAuthReadNodeReply {
            node: net_to_node_key(node.node),
            shares: decode_ordered_shard_values(owned_indices, node.values)?,
        });
    }
    let mut upper_path_nodes = Vec::with_capacity(resp.upper_path_nodes.len());
    for node in resp.upper_path_nodes {
        upper_path_nodes.push(ServiceAuthReadNodeReply {
            node: net_to_node_key(node.node),
            shares: decode_ordered_shard_values(owned_indices, node.values)?,
        });
    }
    Ok(ServiceAuthSubtreeReadResponse {
        leaf_nodes,
        upper_path_nodes,
    })
}

fn net_subtree_upper_path_response_to_service(
    owned_indices: &[usize],
    resp: NetAuthSubtreeUpperPathReadResponse,
) -> Result<ServiceAuthSubtreeUpperPathReadResponse, ClientError> {
    let mut upper_path_nodes = Vec::with_capacity(resp.upper_path_nodes.len());
    for node in resp.upper_path_nodes {
        upper_path_nodes.push(ServiceAuthReadNodeReply {
            node: net_to_node_key(node.node),
            shares: decode_ordered_shard_values(owned_indices, node.values)?,
        });
    }
    Ok(ServiceAuthSubtreeUpperPathReadResponse { upper_path_nodes })
}

fn service_write_request_to_net(req: &ServiceAuthWriteRequest) -> NetAuthWriteRequest {
    NetAuthWriteRequest {
        old: NetValue(req.old),
        new: NetValue(req.new),
        path: NetAuthPath {
            index: req.path.index,
            siblings: req.path.siblings.iter().copied().map(NetValue).collect(),
            root: NetValue(req.path.root),
        },
    }
}

fn net_write_response_to_service(resp: NetAuthWriteResponse) -> ServiceAuthWriteResponse {
    ServiceAuthWriteResponse {
        ok: resp.ok,
        root_new: resp.root_new.map(|value| value.0),
        error: resp.error,
    }
}

fn service_batch_write_request_to_net(
    req: &ServiceAuthBatchWriteRequest,
) -> NetAuthBatchWriteRequest {
    let updates = req
        .updates
        .iter()
        .map(|update| NetBatchWriteUpdate {
            index: update.index,
            old: NetValue(update.old),
            new: NetValue(update.new),
        })
        .collect();
    let proof_nodes = req
        .proof
        .nodes
        .iter()
        .map(|(node, value)| (node_key_to_net(*node), NetValue(*value)))
        .collect();
    NetAuthBatchWriteRequest {
        updates,
        proof: NetMerkleProof {
            root: NetValue(req.proof.root),
            nodes: proof_nodes,
        },
        delta_mode: NetDeltaMode::from(req.delta_mode),
    }
}

fn net_fresh_subtree_write_request_to_service(
    req: NetAuthFreshSubtreeWriteRequest,
) -> ServiceAuthFreshSubtreeWriteRequest {
    ServiceAuthFreshSubtreeWriteRequest {
        leaves: req.leaves.0,
        proof: AuthSubtreeProof {
            start_index: req.proof.start_index,
            leaf_count: usize::try_from(req.proof.leaf_count)
                .expect("fresh subtree proof leaf_count overflows usize"),
            upper_siblings: req
                .proof
                .upper_siblings
                .into_iter()
                .map(|value| value.0)
                .collect(),
            root: req.proof.root.0,
        },
        delta_mode: DeltaMode::from(req.delta_mode),
    }
}

fn service_fresh_subtree_write_request_to_net(
    req: &ServiceAuthFreshSubtreeWriteRequest,
) -> NetAuthFreshSubtreeWriteRequest {
    NetAuthFreshSubtreeWriteRequest {
        leaves: NetValueBlob(req.leaves.clone()),
        proof: NetAuthSubtreeProof {
            start_index: req.proof.start_index,
            leaf_count: u64::try_from(req.proof.leaf_count)
                .expect("fresh subtree proof leaf_count overflows u64"),
            upper_siblings: req
                .proof
                .upper_siblings
                .iter()
                .copied()
                .map(NetValue)
                .collect(),
            root: NetValue(req.proof.root),
        },
        delta_mode: NetDeltaMode::from(req.delta_mode),
    }
}

fn net_batch_write_response_to_service(
    resp: NetAuthBatchWriteResponse,
) -> ServiceAuthBatchWriteResponse {
    ServiceAuthBatchWriteResponse {
        ok: resp.ok,
        root_new: resp.root_new.map(|value| value.0),
        error: resp.error,
    }
}

#[derive(Clone)]
pub struct NetAuthTransport {
    owned_indices_by_party_id: Arc<HashMap<u64, Vec<usize>>>,
    row_labels_by_party_id: Arc<HashMap<u64, Vec<Fp>>>,
}

pub type AuthClient = CoreAuthClient<NetAuthTransport>;

impl NetAuthTransport {
    pub fn with_owned_indices_by_party_id(
        owned_indices_by_party_id: HashMap<u64, Vec<usize>>,
    ) -> Self {
        Self::with_party_metadata(owned_indices_by_party_id, HashMap::new())
    }

    pub fn with_party_metadata(
        owned_indices_by_party_id: HashMap<u64, Vec<usize>>,
        row_labels_by_party_id: HashMap<u64, Vec<Fp>>,
    ) -> Self {
        Self {
            owned_indices_by_party_id: Arc::new(
                owned_indices_by_party_id
                    .into_iter()
                    .map(|(party_id, indices)| (party_id, normalize_owned_indices(&indices)))
                    .collect(),
            ),
            row_labels_by_party_id: Arc::new(row_labels_by_party_id),
        }
    }

    pub fn for_single_party(party_id: u64, owned_indices: Vec<usize>) -> Self {
        let mut map = HashMap::new();
        map.insert(party_id, owned_indices);
        Self::with_owned_indices_by_party_id(map)
    }

    fn owned_indices_for_server(&self, server: &PartyInfo) -> Result<&[usize], ClientError> {
        self.owned_indices_by_party_id
            .get(&server.party_id)
            .map(|indices| indices.as_slice())
            .ok_or_else(|| {
                ClientError::InvalidResponse(format!(
                    "missing derived shard ownership for auth server {}",
                    server.party_id
                ))
            })
    }

    fn row_labels_for_server(&self, server: &PartyInfo) -> Result<&[Fp], ClientError> {
        self.row_labels_by_party_id
            .get(&server.party_id)
            .map(|labels| labels.as_slice())
            .ok_or_else(|| {
                ClientError::InvalidResponse(format!(
                    "missing derived row labels for auth server {}",
                    server.party_id
                ))
            })
    }
}

pub fn is_read_only_message(msg: &Message) -> bool {
    matches!(
        msg,
        Message::AuthRead { .. }
            | Message::AuthReadSubtree { .. }
            | Message::AuthReadSubtreeUpperPath { .. }
            | Message::UnauthMultiRead { .. }
            | Message::AuthGetLen
    )
}

pub fn handle_read_only_message(
    service: &crate::auth::service::AuthServer,
    msg: Message,
) -> Option<Response> {
    let response = match msg {
        Message::AuthRead { req } => Response::AuthRead {
            resp: service_read_response_to_net(
                service.handle_read_at(req.leaf, crate::net::unauth::net_kappa_to_usize(req.kappa)),
            ),
        },
        Message::AuthReadSubtree { req } => match net_leaf_count_to_usize(req.leaf_count) {
            Ok(leaf_count) => match service.handle_read_subtree(req.start_index, leaf_count) {
                Ok(resp) => Response::AuthReadSubtree {
                    resp: service_subtree_read_response_to_net(resp),
                },
                Err(message) => Response::Error { message },
            },
            Err(resp) => resp,
        },
        Message::AuthReadSubtreeUpperPath { req } => {
            match net_leaf_count_to_usize(req.leaf_count) {
                Ok(leaf_count) => {
                    match service.handle_read_subtree_upper_path(req.start_index, leaf_count) {
                        Ok(resp) => Response::AuthReadSubtreeUpperPath {
                            resp: service_subtree_upper_path_response_to_net(resp),
                        },
                        Err(message) => Response::Error { message },
                    }
                }
                Err(resp) => resp,
            }
        }
        Message::UnauthMultiRead { idxs } => Response::UnauthMultiRead {
            values: service
                .handle_unauth_multi_read(&idxs)
                .into_iter()
                .map(|batch| pack_ordered_shard_values(&batch))
                .collect(),
        },
        Message::UnauthOptimizedRead { req } => {
            let resp = net_optimized_request_to_core(&req)
                .map(|core_req| service.handle_optimized_read(&core_req))
                .and_then(core_optimized_response_to_net);
            match resp {
                Ok(resp) => Response::UnauthOptimizedRead { resp },
                Err(err) => Response::Error {
                    message: format!("unauth optimized read failed: {:?}", err),
                },
            }
        }
        Message::AuthGetLen => Response::AuthLen {
            len: service.allocated_leaves(),
        },
        _ => return None,
    };
    Some(response)
}

impl AuthTransport for NetAuthTransport {
    type Server = PartyInfo;

    fn supports_subtree_upper_path_reads(&self) -> bool {
        true
    }

    fn supports_fresh_subtree_writes(&self) -> bool {
        true
    }

    fn server_label(&self, server: &Self::Server) -> String {
        server.party_id.to_string()
    }

    fn auth_get_len(&self, server: &Self::Server) -> Result<u64, ClientError> {
        let resp = crate::net::rpc_call(&server.rpc_host, &Message::AuthGetLen)
            .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::AuthLen { len } => Ok(len),
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn auth_read(
        &self,
        server: &Self::Server,
        index: u64,
        kappa: usize,
    ) -> Result<ServiceAuthReadResponse, ClientError> {
        let req = Message::AuthRead {
            req: NetAuthReadRequest {
                leaf: index,
                kappa: kappa as u64,
            },
        };
        let resp = crate::net::rpc_call(&server.rpc_host, &req)
            .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::AuthRead { resp } => {
                let owned_indices = self.owned_indices_for_server(server)?;
                net_read_response_to_service(owned_indices, resp)
            }
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    #[cfg(feature = "private")]
    fn auth_read_private(
        &self,
        server: &Self::Server,
        request: &crate::private::PrivateReadRequest,
    ) -> Result<crate::private::PrivateReadResponse, ClientError> {
        let req = Message::AuthReadPrivate {
            req: crate::private::wire::request_to_net(request),
        };
        let resp = crate::net::rpc_call(&server.rpc_host, &req)
            .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::AuthReadPrivate { resp } => {
                let owned_indices = self.owned_indices_for_server(server)?;
                let resp = crate::private::wire::response_from_net(resp);
                // A server answers for the claims the registry assigns it.
                if resp
                    .replies
                    .iter()
                    .any(|reply| !owned_indices.contains(&reply.registry_index))
                {
                    return Err(ClientError::InvalidResponse(
                        "private reply for a claim outside the server's registry entry"
                            .to_string(),
                    ));
                }
                Ok(resp)
            }
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn auth_read_subtree(
        &self,
        server: &Self::Server,
        start_index: u64,
        leaf_count: usize,
    ) -> Result<ServiceAuthSubtreeReadResponse, ClientError> {
        let leaf_count = u64::try_from(leaf_count).map_err(|_| ClientError::InvalidIndex)?;
        let req = Message::AuthReadSubtree {
            req: NetAuthSubtreeReadRequest {
                start_index,
                leaf_count,
            },
        };
        let resp = crate::net::rpc_call(&server.rpc_host, &req)
            .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::AuthReadSubtree { resp } => {
                let owned_indices = self.owned_indices_for_server(server)?;
                net_subtree_read_response_to_service(owned_indices, resp)
            }
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn auth_read_subtree_upper_path(
        &self,
        server: &Self::Server,
        start_index: u64,
        leaf_count: usize,
    ) -> Result<ServiceAuthSubtreeUpperPathReadResponse, ClientError> {
        let leaf_count = u64::try_from(leaf_count).map_err(|_| ClientError::InvalidIndex)?;
        let req = Message::AuthReadSubtreeUpperPath {
            req: NetAuthSubtreeUpperPathReadRequest {
                start_index,
                leaf_count,
            },
        };
        let resp = crate::net::rpc_call(&server.rpc_host, &req)
            .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::AuthReadSubtreeUpperPath { resp } => {
                let owned_indices = self.owned_indices_for_server(server)?;
                net_subtree_upper_path_response_to_service(owned_indices, resp)
            }
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn unauth_multi_read(
        &self,
        server: &Self::Server,
        indices: &[u64],
    ) -> Result<Vec<Vec<ShardShare>>, ClientError> {
        let resp = crate::net::rpc_call(
            &server.rpc_host,
            &Message::UnauthMultiRead {
                idxs: indices.to_vec(),
            },
        )
        .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::UnauthMultiRead { values } => {
                let owned_indices = self.owned_indices_for_server(server)?;
                values
                    .into_iter()
                    .map(|batch| decode_ordered_shard_values(owned_indices, batch))
                    .collect()
            }
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn supports_unauth_optimized_reads(&self) -> bool {
        !self.owned_indices_by_party_id.is_empty() || !self.row_labels_by_party_id.is_empty()
    }

    fn unauth_optimized_read(
        &self,
        server: &Self::Server,
        request: &crate::unauth::UnauthOptimizedReadRequest,
    ) -> Result<crate::unauth::UnauthOptimizedReadResponse, ClientError> {
        let req = core_optimized_request_to_net(request).map_err(map_unauth_client_error)?;
        let resp = crate::net::rpc_call(&server.rpc_host, &Message::UnauthOptimizedRead { req })
            .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::UnauthOptimizedRead { resp } => {
                let owned_indices = self.owned_indices_for_server(server).ok();
                let row_labels = self.row_labels_for_server(server).ok();
                net_optimized_response_to_core(owned_indices, row_labels, resp)
                    .map_err(map_unauth_client_error)
            }
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn auth_write(
        &self,
        server: &Self::Server,
        req: &ServiceAuthWriteRequest,
    ) -> Result<ServiceAuthWriteResponse, ClientError> {
        let resp = crate::net::rpc_call(
            &server.rpc_host,
            &Message::AuthWrite {
                req: service_write_request_to_net(req),
            },
        )
        .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::AuthWrite { resp } => Ok(net_write_response_to_service(resp)),
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn auth_write_batch(
        &self,
        server: &Self::Server,
        req: &ServiceAuthBatchWriteRequest,
    ) -> Result<ServiceAuthBatchWriteResponse, ClientError> {
        let resp = crate::net::rpc_call(
            &server.rpc_host,
            &Message::AuthWriteBatch {
                req: service_batch_write_request_to_net(req),
            },
        )
        .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::AuthWriteBatch { resp } => Ok(net_batch_write_response_to_service(resp)),
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn auth_write_fresh_subtree(
        &self,
        server: &Self::Server,
        req: &ServiceAuthFreshSubtreeWriteRequest,
    ) -> Result<ServiceAuthBatchWriteResponse, ClientError> {
        let resp = crate::net::rpc_call(
            &server.rpc_host,
            &Message::AuthWriteFreshSubtree {
                req: service_fresh_subtree_write_request_to_net(req),
            },
        )
        .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::AuthWriteFreshSubtree { resp } => {
                Ok(net_batch_write_response_to_service(resp))
            }
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn auth_extend(&self, server: &Self::Server, new_len: u64) -> Result<bool, ClientError> {
        let resp = crate::net::rpc_call(
            &server.rpc_host,
            &Message::AuthExtend {
                new_leaves: new_len,
            },
        )
        .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::AuthExtend { ok } => Ok(ok),
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }
}

impl CoreAuthClient<NetAuthTransport> {
    /// Construct a client from a registry JSON file.
    pub fn from_registry_file(path: &Path) -> Result<Self, ClientError> {
        let file = MockRegistryFile::load(path)
            .map_err(|err| ClientError::Registry(format!("{:?}", err)))?;
        Self::from_registry(&file)
    }

    /// Construct a client from an in-memory registry file.
    pub fn from_registry(file: &MockRegistryFile) -> Result<Self, ClientError> {
        let (params, claims) = build_unauth_registry_view(file)
            .map_err(|err| ClientError::Registry(format!("{:?}", err)))?;
        let mut servers = file.parties.clone();
        servers.retain(|p| p.active);
        servers.sort_by_key(|p| p.party_id);
        let server = servers
            .first()
            .cloned()
            .ok_or_else(|| ClientError::Registry("no active auth servers".to_string()))?;
        let allocated = file.system.auth.allocated_leaves;
        let auth = AuthParams {
            tree_height: file.system.auth.tree_height,
            defaults: crate::auth::merkle::build_default_hashes(file.system.auth.tree_height),
        };
        let mut owned_indices_by_party_id: HashMap<u64, Vec<usize>> = HashMap::new();
        for claim in claims {
            owned_indices_by_party_id
                .entry(claim.party_id)
                .or_default()
                .push(claim.registry_index as usize);
        }
        for indices in owned_indices_by_party_id.values_mut() {
            *indices = normalize_owned_indices(indices);
        }
        let row_labels_by_party_id = owned_indices_by_party_id
            .iter()
            .map(|(&party_id, indices)| {
                (
                    party_id,
                    derived_row_labels_for_owned_indices(&params, indices),
                )
            })
            .collect();
        Ok(Self::new(
            NetAuthTransport::with_party_metadata(
                owned_indices_by_party_id,
                row_labels_by_party_id,
            ),
            params,
            auth,
            file.system.auth.node_linearization,
            allocated,
            server,
            servers,
        ))
    }
}

fn net_to_node_key(node: NetNodeKey) -> NodeKey {
    NodeKey {
        level: node.level,
        pos: node.pos,
    }
}

fn node_key_to_net(node: NodeKey) -> NetNodeKey {
    NetNodeKey {
        level: node.level,
        pos: node.pos,
    }
}
