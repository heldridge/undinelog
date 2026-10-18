//! Network client helpers for party RPCs and multi-party reads/writes.

use crate::auth::merkle::{NodeKey, leaf_key, parent, recompute_root, sibling};
use crate::auth::AuthPath;
use crate::field::Fp;
use crate::net::registry::{PartyInfo, RegistryClient, RegistryError};
use crate::net::system::NetSystem;
use crate::net::wire::{
    AuthReadRequest, AuthReadResponse, AuthWriteRequest, AuthWriteResponse, MAX_FRAME_LEN,
    NetAuthPath, NetValue, ReadShardRequest, ReadShardResponse, RpcRequest, RpcResponse,
    decode_binary, encode_binary, read_frame, write_frame,
};
use crate::unauth::{
    barycentric_eval, logical_offset_to_column_coordinate_for_segment, precompute_barycentric,
    public_values_per_segment,
};
use crate::value::{Value256, encode_value};
use ark_ff::Field;
use std::collections::HashMap;
use std::net::TcpStream;
use std::time::Duration;

/// Errors produced by network clients.
#[derive(Debug)]
pub enum NetError {
    Io(String),
    Codec(String),
    Registry(RegistryError),
    InvalidIndex,
    NotEnoughShards,
    RootMismatch,
    NoParties,
    AllPartiesFailed,
    InvalidResponse(String),
}

/// Single-party RPC client.
#[derive(Clone)]
pub struct PartyClient {
    host: String,
    max_frame_len: usize,
    timeout: Option<Duration>,
}

impl PartyClient {
    pub fn new(host: String) -> Self {
        Self {
            host,
            max_frame_len: MAX_FRAME_LEN,
            timeout: Some(Duration::from_secs(2)),
        }
    }

    pub fn with_max_frame_len(mut self, max_frame_len: usize) -> Self {
        self.max_frame_len = max_frame_len;
        self
    }

    pub fn with_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn read_shard(&self, req: ReadShardRequest) -> Result<ReadShardResponse, NetError> {
        match self.send_request(RpcRequest::ReadShard(req))? {
            RpcResponse::ReadShard(resp) => Ok(resp),
            RpcResponse::Error(err) => Err(NetError::InvalidResponse(err.message)),
            other => Err(NetError::InvalidResponse(format!(
                "unexpected response: {:?}",
                other
            ))),
        }
    }

    pub fn auth_read(&self, req: AuthReadRequest) -> Result<AuthReadResponse, NetError> {
        match self.send_request(RpcRequest::AuthRead(req))? {
            RpcResponse::AuthRead(resp) => Ok(resp),
            RpcResponse::Error(err) => Err(NetError::InvalidResponse(err.message)),
            other => Err(NetError::InvalidResponse(format!(
                "unexpected response: {:?}",
                other
            ))),
        }
    }

    pub fn auth_write(&self, req: AuthWriteRequest) -> Result<AuthWriteResponse, NetError> {
        match self.send_request(RpcRequest::AuthWrite(req))? {
            RpcResponse::AuthWrite(resp) => Ok(resp),
            RpcResponse::Error(err) => Err(NetError::InvalidResponse(err.message)),
            other => Err(NetError::InvalidResponse(format!(
                "unexpected response: {:?}",
                other
            ))),
        }
    }

    fn send_request(&self, req: RpcRequest) -> Result<RpcResponse, NetError> {
        let mut stream =
            TcpStream::connect(&self.host).map_err(|err| NetError::Io(err.to_string()))?;
        if let Some(timeout) = self.timeout {
            let _ = stream.set_read_timeout(Some(timeout));
            let _ = stream.set_write_timeout(Some(timeout));
        }
        let payload = encode_binary(&req).map_err(NetError::Codec)?;
        write_frame(&mut stream, &payload).map_err(|err| NetError::Io(err.to_string()))?;
        let frame = read_frame(&mut stream, self.max_frame_len)
            .map_err(|err| NetError::Io(err.to_string()))?
            .ok_or_else(|| NetError::InvalidResponse("eof".to_string()))?;
        decode_binary(&frame).map_err(NetError::Codec)
    }
}

#[derive(Clone)]
struct PartyAssignment {
    info: PartyInfo,
    owned_indices: Vec<usize>,
}

/// Summary of a broadcast attempt.
#[derive(Clone, Debug)]
pub struct BroadcastSummary {
    pub ok: Vec<u64>,
    pub failed: Vec<(u64, String)>,
}

/// Multi-party network client that performs reads and broadcast writes.
pub struct NetworkClient {
    system: NetSystem,
    registry: Box<dyn RegistryClient>,
    parties: Vec<PartyAssignment>,
    max_frame_len: usize,
    trusted_root: Option<Value256>,
}

impl NetworkClient {
    pub fn new(system: NetSystem, registry: Box<dyn RegistryClient>) -> Self {
        Self {
            system,
            registry,
            parties: Vec::new(),
            max_frame_len: MAX_FRAME_LEN,
            trusted_root: None,
        }
    }

    /// Override the trusted root used for authenticated reads.
    pub fn set_trusted_root(&mut self, root: Value256) {
        self.trusted_root = Some(root);
    }

    /// Clear the trusted root (next read will accept any root and update it).
    pub fn clear_trusted_root(&mut self) {
        self.trusted_root = None;
    }

    /// Return the currently trusted root, if any.
    pub fn trusted_root(&self) -> Option<Value256> {
        self.trusted_root
    }

    pub fn with_max_frame_len(mut self, max_frame_len: usize) -> Self {
        self.max_frame_len = max_frame_len;
        self
    }

    pub fn refresh_registry(&mut self) -> Result<(), NetError> {
        let mut parties = self.registry.fetch_parties().map_err(NetError::Registry)?;
        parties.retain(|p| p.active);
        parties.sort_by_key(|p| p.party_id);
        let mut out = Vec::with_capacity(parties.len());
        for (position, info) in parties.into_iter().enumerate() {
            let owned_indices = self
                .system
                .params
                .party_registry_indices
                .get(position)
                .cloned()
                .unwrap_or_default();
            out.push(PartyAssignment { info, owned_indices });
        }
        self.parties = out;
        Ok(())
    }

    pub fn read_leaf(&mut self, index: usize) -> Result<Value256, NetError> {
        let (value, _) = self.read_leaf_with_path(index)?;
        Ok(value)
    }

    pub fn read_leaf_with_path(&mut self, index: usize) -> Result<(Value256, AuthPath), NetError> {
        if index as u128 >= self.system.auth.conceptual_l_max_u128() {
            return Err(NetError::InvalidIndex);
        }
        let height = self.system.auth.tree_height as u32;
        let mut key = leaf_key(height, index as u64);
        let mut nodes = Vec::with_capacity(self.system.auth.tree_height + 1);
        let mut siblings_keys = Vec::with_capacity(self.system.auth.tree_height);
        nodes.push(key);
        for _ in 0..self.system.auth.tree_height {
            let sib = sibling(key);
            nodes.push(sib);
            siblings_keys.push(sib);
            key = parent(key);
        }
        let values = self.read_nodes(index as u64, &nodes)?;
        let value = *values.get(&nodes[0]).ok_or(NetError::NotEnoughShards)?;
        let mut siblings = Vec::with_capacity(self.system.auth.tree_height);
        for sib in &siblings_keys {
            let sib_value = *values.get(sib).ok_or(NetError::NotEnoughShards)?;
            siblings.push(sib_value);
        }
        let root = recompute_root(
            height,
            index as u64,
            value,
            &siblings,
            &self.system.auth.defaults,
        );
        if let Some(expected) = self.trusted_root {
            if expected != root {
                return Err(NetError::RootMismatch);
            }
        }
        self.trusted_root = Some(root);
        Ok((
            value,
            AuthPath {
                index,
                siblings,
                root,
            },
        ))
    }

    pub fn write_leaf(
        &mut self,
        index: usize,
        new: Value256,
    ) -> Result<BroadcastSummary, NetError> {
        if index >= self.system.allocated_leaves {
            return Err(NetError::InvalidIndex);
        }
        let (old, path) = self.read_leaf_with_path(index)?;
        let update = AuthWriteRequest {
            old: NetValue(old),
            new: NetValue(new),
            path: NetAuthPath {
                index: path.index as u64,
                siblings: path.siblings.iter().map(|v| NetValue(*v)).collect(),
                root: NetValue(path.root),
            },
        };
        let (summary, root_new) = self.broadcast_auth_write(update)?;
        self.trusted_root = Some(root_new);
        Ok(summary)
    }

    pub fn broadcast_auth_write(
        &mut self,
        update: AuthWriteRequest,
    ) -> Result<(BroadcastSummary, Value256), NetError> {
        if self.parties.is_empty() {
            self.refresh_registry()?;
        }
        if self.parties.is_empty() {
            return Err(NetError::NoParties);
        }
        let mut ok = Vec::new();
        let mut failed = Vec::new();
        let mut root_new = None;
        for party in &self.parties {
            let client = PartyClient::new(party.info.rpc_host.clone())
                .with_max_frame_len(self.max_frame_len);
            match client.auth_write(update.clone()) {
                Ok(resp) if resp.ok => {
                    let Some(resp_root) = resp.root_new else {
                        failed.push((
                            party.info.party_id,
                            "missing root_new in response".to_string(),
                        ));
                        continue;
                    };
                    if let Some(existing) = root_new {
                        if existing != resp_root.0 {
                            failed.push((
                                party.info.party_id,
                                "root_new mismatch in response".to_string(),
                            ));
                            continue;
                        }
                    } else {
                        root_new = Some(resp_root.0);
                    }
                    ok.push(party.info.party_id);
                }
                Ok(resp) => failed.push((
                    party.info.party_id,
                    resp.error.unwrap_or_else(|| "unknown error".to_string()),
                )),
                Err(err) => failed.push((party.info.party_id, format!("{:?}", err))),
            }
        }
        if ok.is_empty() {
            return Err(NetError::AllPartiesFailed);
        }
        let root_new = root_new.ok_or_else(|| {
            NetError::InvalidResponse("missing root_new in responses".to_string())
        })?;
        Ok((BroadcastSummary { ok, failed }, root_new))
    }

    fn read_nodes(
        &mut self,
        leaf: u64,
        nodes: &[NodeKey],
    ) -> Result<HashMap<NodeKey, Value256>, NetError> {
        if self.parties.is_empty() {
            self.refresh_registry()?;
        }
        if self.parties.is_empty() {
            return Err(NetError::NoParties);
        }
        let req = AuthReadRequest {
            leaf,
            kappa: self.system.params.need as u64,
        };

        let mut shares_by_node: HashMap<NodeKey, HashMap<usize, Fp>> = HashMap::new();
        for node in nodes {
            shares_by_node.insert(*node, HashMap::new());
        }

        for party in &self.parties {
            let client = PartyClient::new(party.info.rpc_host.clone())
                .with_max_frame_len(self.max_frame_len);
            let resp = match client.auth_read(req.clone()) {
                Ok(resp) => resp,
                Err(_) => continue,
            };
            apply_shares(&mut shares_by_node, &resp.nodes, &party.owned_indices);
        }

        let mut values = HashMap::new();
        let height = self.system.auth.tree_height as u16;
        let segment_len = public_values_per_segment(&self.system.params) as u64;
        for node in nodes {
            let Some(shares) = shares_by_node.get(node) else {
                return Err(NetError::NotEnoughShards);
            };
            let idx = mapping::idx_if_active(
                self.system.node_linearization,
                height,
                self.system.allocated_leaves as u64,
                node.level as u16,
                node.pos,
            )
            .ok_or(NetError::InvalidIndex)?;
            let seg = (idx / segment_len) as usize;
            let logical_offset = (idx % segment_len) as usize;
            let z_eval = logical_offset_to_column_coordinate_for_segment(
                &self.system.params,
                seg,
                logical_offset,
            );
            let mut z_values = Vec::new();
            let mut r_values = Vec::new();
            for (idx, entry) in self.system.params.shard_registry.iter().enumerate() {
                if let Some(value) = shares.get(&idx) {
                    z_values.push(entry.z);
                    r_values.push(*value);
                    if r_values.len() >= self.system.params.need {
                        break;
                    }
                }
            }
            if r_values.len() < self.system.params.need {
                return Err(NetError::NotEnoughShards);
            }
            let weights = precompute_barycentric(&z_values);
            let decoded = barycentric_eval(&weights, &r_values, z_eval);
            values.insert(*node, encode_value(decoded));
        }

        Ok(values)
    }
}

fn apply_shares(
    shares_by_node: &mut HashMap<NodeKey, HashMap<usize, Fp>>,
    nodes: &[crate::net::wire::AuthReadNodeReply],
    owned_indices: &[usize],
) {
    for node_reply in nodes {
        let node = NodeKey {
            level: node_reply.node.level,
            pos: node_reply.node.pos,
        };
        let Some(entries) = shares_by_node.get_mut(&node) else {
            continue;
        };
        for (idx, value) in owned_indices
            .iter()
            .copied()
            .zip(node_reply.values.0.iter().copied())
        {
            let Some(value) = value else {
                continue;
            };
            if entries.contains_key(&idx) {
                continue;
            }
            entries.insert(idx, value);
        }
    }
}
