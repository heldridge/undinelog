//! Party server implementation for TCP RPCs.
//!
//! This module owns per-party state and handles a small set of RPCs:
//! - Read a shard share from the unauth store.
//! - Read Merkle nodes (auth reads).
//! - Apply authenticated updates (auth writes).

use crate::auth::mapping;
use crate::auth::read_path::build_path_nodes;
use crate::auth::{AuthPath, apply_broadcast_update};
use crate::net::registry::{
    RegistryClient, RegistryError, apply_unauth_epoch, build_party_registry_indices_from_claims,
    claim_bundles_for_party, validate_unauth_claims,
};
use crate::net::system::{
    NetSystem, NetSystemError, PartyStorage, append_segments, build_party_storage,
    default_segment_shards,
};
use crate::net::wire::{
    AuthReadNodeReply, AuthReadRequest, AuthReadResponse, AuthWriteRequest, AuthWriteResponse,
    MAX_FRAME_LEN, NetMaybeFpBlob, NetNodeKey, NetValue, ReadShardResponse, RpcError, RpcRequest,
    RpcResponse, decode_binary, encode_binary, read_frame, write_frame,
};
use crate::unauth::{
    ClaimBundle, DeltaMode, logical_offset_to_alpha_for_segment, public_values_per_segment,
    shard_value_from_store,
};
use crate::value::Value256;
use std::collections::HashSet;
use std::io;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Server errors during initialization and setup.
#[derive(Debug)]
pub enum PartyServerError {
    /// Registry RPC or lookup failures.
    Registry(RegistryError),
    /// System parameter or storage initialization failures.
    System(NetSystemError),
    /// IO failures when starting the server.
    Io(String),
}

/// In-memory party server state.
///
/// This is shared across per-connection threads and guarded by a mutex.
pub struct PartyServerState {
    /// Logical party identifier (matches registry assignments).
    pub party_id: u64,
    /// Registry indices assigned to this party (deduped, sorted).
    pub assigned_indices: Vec<usize>,
    /// Quick membership test for assigned indices.
    assigned_set: HashSet<usize>,
    /// Explicit local unauth claim bundles when the registry publishes them.
    claims: Option<Vec<ClaimBundle>>,
    /// Network system parameters (auth + unauth settings).
    pub system: NetSystem,
    /// Party-local storage and derived metadata.
    pub storage: PartyStorage,
}

impl PartyServerState {
    pub fn new(
        party_id: u64,
        system: NetSystem,
        mut assigned_indices: Vec<usize>,
        claims: Option<Vec<ClaimBundle>>,
    ) -> Result<Self, NetSystemError> {
        // Normalize the index list so we can safely iterate and dedupe.
        assigned_indices.sort_unstable();
        assigned_indices.dedup();
        let assigned_set: HashSet<usize> = assigned_indices.iter().copied().collect();
        // Build storage only for the shards this party owns.
        let storage = build_party_storage(&system, &assigned_indices, claims.as_deref())?;
        Ok(Self {
            party_id,
            assigned_indices,
            assigned_set,
            claims,
            system,
            storage,
        })
    }

    fn ensure_segments(&mut self, allocated_leaves: usize) -> Result<(), NetSystemError> {
        // Ensure the unauth store has enough segments to cover `allocated_leaves`.
        append_segments(
            &self.system,
            &mut self.storage.store.unauth,
            allocated_leaves,
            &self.assigned_indices,
            self.claims.as_deref(),
        )?;
        // Mirror storage metadata for auth reads that depend on segment counts.
        self.storage.segments = self.storage.store.unauth.segments();
        self.storage.allocated_leaves = allocated_leaves;
        Ok(())
    }
}

/// Party server instance that owns shared state.
///
/// This is a thin wrapper around shared state + per-connection worker threads.
pub struct PartyServer {
    /// Shared mutable state for all incoming connections.
    state: Arc<Mutex<PartyServerState>>,
    /// Maximum allowed RPC frame length.
    max_frame_len: usize,
}

impl PartyServer {
    pub fn from_registry(
        party_id: u64,
        mut system: NetSystem,
        registry: &dyn RegistryClient,
    ) -> Result<Self, PartyServerError> {
        let parties = registry
            .fetch_parties()
            .map_err(PartyServerError::Registry)?;
        if let Some(epoch) = registry
            .fetch_unauth_epoch()
            .map_err(PartyServerError::Registry)?
        {
            apply_unauth_epoch(&mut system.params, &epoch).map_err(PartyServerError::Registry)?;
        }
        let claims = registry
            .fetch_unauth_claims()
            .map_err(PartyServerError::Registry)?;
        let (assigned_indices, local_claims) = if claims.is_empty() {
            let indices = registry
                .fetch_assignments(party_id)
                .map_err(PartyServerError::Registry)?;
            (indices.into_iter().map(|v| v as usize).collect(), None)
        } else {
            validate_unauth_claims(&system.params, &claims).map_err(PartyServerError::Registry)?;
            if !parties.is_empty() {
                system.params.party_registry_indices =
                    build_party_registry_indices_from_claims(&parties, &claims);
            }
            let local_claims = claim_bundles_for_party(&system.params, &claims, party_id)
                .map_err(PartyServerError::Registry)?;
            let assigned_indices = local_claims
                .iter()
                .map(|claim| claim.registry_index)
                .collect();
            (assigned_indices, Some(local_claims))
        };
        // Build the per-party state from system parameters + assignments.
        let state = PartyServerState::new(party_id, system, assigned_indices, local_claims)
            .map_err(PartyServerError::System)?;
        Ok(Self {
            state: Arc::new(Mutex::new(state)),
            max_frame_len: MAX_FRAME_LEN,
        })
    }

    pub fn with_max_frame_len(mut self, max_frame_len: usize) -> Self {
        // Allow tests or demos to tune framing limits.
        self.max_frame_len = max_frame_len;
        self
    }

    pub fn serve(self, listener: TcpListener, stop: Arc<AtomicBool>) -> io::Result<()> {
        // Non-blocking accept loop so we can poll `stop`.
        listener.set_nonblocking(true)?;
        while !stop.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((stream, _addr)) => {
                    // Spawn a worker thread per connection.
                    let state = self.state.clone();
                    let max_len = self.max_frame_len;
                    thread::spawn(move || {
                        let _ = handle_connection(stream, state, max_len);
                    });
                }
                Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                    // Idle a little to avoid a busy loop.
                    thread::sleep(Duration::from_millis(5));
                }
                Err(err) => return Err(err),
            }
        }
        Ok(())
    }

    pub fn spawn(self, listener: TcpListener) -> PartyServerHandle {
        // Run the server in a background thread and return a handle to stop it.
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();
        let addr = listener.local_addr().ok();
        let thread = thread::spawn(move || {
            let _ = self.serve(listener, stop_clone);
        });
        PartyServerHandle { stop, thread, addr }
    }
}

/// Handle for a running server thread.
pub struct PartyServerHandle {
    /// Signal for a graceful shutdown.
    stop: Arc<AtomicBool>,
    /// Thread join handle for the server loop.
    thread: thread::JoinHandle<()>,
    /// Bound address of the listener (if available).
    addr: Option<SocketAddr>,
}

impl PartyServerHandle {
    pub fn stop(self) {
        // Trigger shutdown and wait for the thread to exit.
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.thread.join();
    }

    pub fn addr(&self) -> Option<SocketAddr> {
        // Expose the bound address for tests/demos.
        self.addr
    }
}

fn handle_connection(
    mut stream: TcpStream,
    state: Arc<Mutex<PartyServerState>>,
    max_frame_len: usize,
) -> io::Result<()> {
    loop {
        // Read a length-prefixed binary frame. EOF => graceful connection close.
        let Some(frame) = read_frame(&mut stream, max_frame_len)? else {
            return Ok(());
        };
        // Parse the RPC request or return a structured error response.
        let req: RpcRequest = match decode_binary(&frame) {
            Ok(req) => req,
            Err(err) => {
                let err_msg = RpcResponse::Error(RpcError { message: err });
                let payload = encode_binary(&err_msg)
                    .map_err(|codec_err| io::Error::new(io::ErrorKind::InvalidData, codec_err))?;
                write_frame(&mut stream, &payload)?;
                return Ok(());
            }
        };

        // Dispatch to request-specific handlers.
        let response = match req {
            RpcRequest::ReadShard(req) => {
                // Fast unauth shard read: only answer if this party owns the shard.
                let state = state.lock().expect("state lock");
                let seg = usize::try_from(req.seg).ok();
                let idx = req.registry_index as usize;
                let reply = match seg {
                    Some(seg)
                        if state.assigned_set.contains(&idx) && seg < state.storage.segments =>
                    {
                        // Read an existing shard share from local storage.
                        let entry = state.system.params.shard_registry[idx];
                        shard_value_from_store(
                            &state.system.params,
                            &state.storage.store.unauth.coded,
                            entry,
                            seg,
                            req.alpha.0,
                        )
                        .map(crate::net::wire::NetFp)
                    }
                    // Unknown segment or shard not owned by this party.
                    _ => None,
                };
                RpcResponse::ReadShard(ReadShardResponse { value: reply })
            }
            RpcRequest::AuthRead(req) => {
                // Auth read: return shares for requested Merkle nodes.
                let state = state.lock().expect("state lock");
                let response = auth_read_nodes(&state, req);
                RpcResponse::AuthRead(response)
            }
            RpcRequest::AuthWrite(req) => {
                // Auth write: validate, extend, and apply updates.
                let mut state = state.lock().expect("state lock");
                let response = apply_auth_write(&mut state, req);
                RpcResponse::AuthWrite(response)
            }
        };

        // Encode the response back to the client.
        let payload = encode_binary(&response)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        write_frame(&mut stream, &payload)?;
    }
}

fn auth_read_nodes(state: &PartyServerState, req: AuthReadRequest) -> AuthReadResponse {
    if req.leaf as u128 >= state.system.auth.conceptual_l_max_u128() {
        return AuthReadResponse { nodes: Vec::new() };
    }
    let request_nodes: Vec<NetNodeKey> = build_path_nodes(&state.system.auth, req.leaf)
        .into_iter()
        .map(|node| NetNodeKey {
            level: node.level,
            pos: node.pos,
        })
        .collect();
    // Short-hands for params and derived sizes.
    let params = &state.system.params;
    let height = state.system.auth.tree_height as u16;
    let segment_len = public_values_per_segment(params) as u64;
    // Cache default shards per segment to avoid recomputing them repeatedly.
    let mut default_cache: std::collections::HashMap<u64, Vec<Vec<crate::field::Fp>>> =
        std::collections::HashMap::new();
    // Preserve input order; one reply per requested node.
    let mut nodes = Vec::with_capacity(request_nodes.len());
    for node in request_nodes {
        // Reject impossible nodes early (level too high or position out of range).
        if node.level > height as u32 || (node.level < 64 && node.pos >= (1u64 << node.level)) {
            nodes.push(AuthReadNodeReply {
                node,
                values: NetMaybeFpBlob(Vec::new()),
            });
            continue;
        }
        let Some(idx) = mapping::idx_if_active(
            state.system.node_linearization,
            height,
            state.storage.allocated_leaves as u64,
            node.level as u16,
            node.pos,
        ) else {
            // Node is outside the active tree for the current allocation.
            nodes.push(AuthReadNodeReply {
                node,
                values: NetMaybeFpBlob(Vec::new()),
            });
            continue;
        };
        // Translate node index to segment + within-segment position.
        let seg = idx / segment_len;
        let j = (idx % segment_len) as usize;
        let seg_usize = seg as usize;
        let alpha = logical_offset_to_alpha_for_segment(params, seg_usize, j);
        // One share per assigned registry index.
        let mut values = Vec::with_capacity(state.assigned_indices.len());
        if seg_usize < state.storage.segments {
            // Active segment: read from local storage.
            for &reg_idx in &state.assigned_indices {
                let entry = params.shard_registry[reg_idx];
                let value = shard_value_from_store(
                    params,
                    &state.storage.store.unauth.coded,
                    entry,
                    seg_usize,
                    alpha,
                );
                values.push(value);
            }
        } else {
            // Future segment: serve deterministic default shards.
            let shards = default_cache
                .entry(seg)
                .or_insert_with(|| default_segment_shards(&state.system, seg_usize));
            for &reg_idx in &state.assigned_indices {
                let entry = params.shard_registry[reg_idx];
                let y = &shards[reg_idx];
                let value = crate::unauth::shard_value_from_vector(params, entry, y, alpha)
                    .unwrap_or_else(crate::field::fp_zero);
                values.push(Some(value));
            }
        }
        nodes.push(AuthReadNodeReply {
            node,
            values: NetMaybeFpBlob(values),
        });
    }
    AuthReadResponse { nodes }
}

fn apply_auth_write(state: &mut PartyServerState, req: AuthWriteRequest) -> AuthWriteResponse {
    // Ensure all required segments are present before applying deltas.
    if let Err(err) = state.ensure_segments(state.storage.allocated_leaves) {
        return AuthWriteResponse {
            ok: false,
            root_new: None,
            error: Some(format!("segment allocation failed: {:?}", err)),
        };
    }
    let path_index = match usize::try_from(req.path.index) {
        Ok(index) => index,
        Err(_) => {
            return AuthWriteResponse {
                ok: false,
                root_new: None,
                error: Some("path index overflow".to_string()),
            };
        }
    };
    let siblings: Vec<Value256> = req.path.siblings.iter().map(|v| v.0).collect();
    let path = AuthPath {
        index: path_index,
        siblings,
        root: req.path.root.0,
    };
    let update = match crate::auth::write::build_broadcast_update(
        &state.system.auth,
        &state.system.params,
        state.system.node_linearization,
        &state.storage.store,
        state.storage.allocated_leaves,
        path_index,
        req.old.0,
        req.new.0,
        path,
        DeltaMode::Aggregate,
    ) {
        Ok(update) => update,
        Err(err) => {
            return AuthWriteResponse {
                ok: false,
                root_new: None,
                error: Some(format!("apply failed: {:?}", err)),
            };
        }
    };
    // Apply the authenticated broadcast update to local storage.
    match apply_broadcast_update(
        &state.system.auth,
        &state.system.params,
        state.system.node_linearization,
        &mut state.storage.store,
        &update,
    ) {
        Ok(()) => AuthWriteResponse {
            ok: true,
            root_new: Some(NetValue(update.root_new)),
            error: None,
        },
        Err(err) => AuthWriteResponse {
            ok: false,
            root_new: None,
            error: Some(format!("{:?}", err)),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::PartyServer;
    use crate::core::NodeLinearization;
    use crate::net::registry::{
        MockRegistryClient, NetAuthConfig, NetSystemConfig, NetUnauthConfig, PartyClaimBundle,
        PartyInfo, default_unauth_epoch,
    };
    use crate::net::system::NetSystem;
    use crate::unauth::RowKey;
    use crate::{CodedArrayConfig, build_coded_params, build_shard_registry_deterministic};
    use std::collections::HashMap;

    #[test]
    fn party_server_from_registry_uses_claim_bundles_for_local_rows() {
        let segment_len = 8usize;
        let s = 4usize;
        let shard_count = 4usize;
        let n_grid = segment_len;
        let shard_registry = build_shard_registry_deterministic(shard_count, s, n_grid);
        let coded_cfg = CodedArrayConfig {
            segment_len,
            s,
            shard_count,
            shard_registry,
            party_registry_indices: vec![(0..shard_count).collect()],
            seed: 42,
        };
        let params = build_coded_params(&coded_cfg);
        let system_cfg = NetSystemConfig {
            unauth: NetUnauthConfig {
                segment_len,
                layout_base_n_grid: None,
                s,
                shard_count,
                seed: 42,
            },
            auth: NetAuthConfig {
                tree_height: 3,
                node_linearization: NodeLinearization::Preorder,
                allocated_leaves: 4,
            },
            unauth_array_len: Some(16),
        };
        let system = NetSystem::from_parts(
            &system_cfg,
            Some(params.shard_registry.clone()),
            vec![(0..shard_count).collect()],
        );
        let registry = MockRegistryClient::new_with_unauth_metadata(
            vec![PartyInfo {
                party_id: 0,
                rpc_host: "127.0.0.1:19000".to_string(),
                pubkey: None,
                active: true,
            }],
            HashMap::new(),
            Some(default_unauth_epoch(&params)),
            vec![PartyClaimBundle {
                party_id: 0,
                registry_index: 1,
                row_family_id: params
                    .row_family_id_for_registry_index(1)
                    .map(|family_id| family_id.0 as u32),
                row_label_indices: Vec::new(),
            }],
        );

        let server = PartyServer::from_registry(0, system, &registry).expect("party server");
        let state = server.state.lock().expect("state lock");
        assert_eq!(state.assigned_indices, vec![1]);
        assert_eq!(
            state
                .claims
                .as_ref()
                .expect("local claims")
                .iter()
                .map(|claim| claim.registry_index)
                .collect::<Vec<_>>(),
            vec![1]
        );
        let owner = state.system.params.shard_registry[1].key;
        let expected_rows = state.system.params.row_label_indices_for_registry_index(1);
        assert_eq!(state.storage.store.unauth.coded.shard_family_count(), 1);
        assert!(state.storage.store.unauth.coded.contains_shard(&owner));
        assert_eq!(
            state.storage.store.unauth.coded.row_family_count(),
            expected_rows.len()
        );
        for row_label_index in expected_rows {
            let row_key = RowKey::new(owner, state.system.params.row_labels[row_label_index]);
            assert!(state.storage.store.unauth.coded.contains_row(&row_key));
        }
    }
}
