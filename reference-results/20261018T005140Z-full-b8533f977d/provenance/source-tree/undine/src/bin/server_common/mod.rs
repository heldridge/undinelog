use paper_a_optimized::auth::service::AuthServer;
use paper_a_optimized::auth::{AuthParams, NodeLinearization};
use paper_a_optimized::net;
use paper_a_optimized::net::registry::{
    MockRegistryFile, build_unauth_registry_view, claim_bundles_for_party,
};
use paper_a_optimized::unauth::{UnauthStore, public_values_per_segment};
use std::env;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

#[derive(Default)]
struct Args {
    mode: Option<String>,
    listen: Option<String>,
    registry: Option<PathBuf>,
    party_id: Option<u64>,
    kappa: Option<usize>,
    node_linearization: Option<String>,
    shard_count: Option<usize>,
    tree_height: Option<usize>,
    unauth_array_len: Option<u64>,
    allocated_leaves: Option<usize>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--mode" => args.mode = it.next(),
            "--listen" => args.listen = it.next(),
            "--registry" => args.registry = it.next().map(PathBuf::from),
            "--party-id" => {
                let v = it.next().ok_or("--party-id missing value")?;
                args.party_id = Some(v.parse().map_err(|_| "invalid --party-id")?);
            }
            "--kappa" => {
                let v = it.next().ok_or("--kappa missing value")?;
                args.kappa = Some(v.parse().map_err(|_| "invalid --kappa")?);
            }
            "--node-linearization" => {
                args.node_linearization = it.next();
            }
            "--shard-count" => {
                let v = it.next().ok_or("--shard-count missing value")?;
                args.shard_count = Some(v.parse().map_err(|_| "invalid --shard-count")?);
            }
            "--tree-height" => {
                let v = it.next().ok_or("--tree-height missing value")?;
                args.tree_height = Some(v.parse().map_err(|_| "invalid --tree-height")?);
            }
            "--unauth-array-len" => {
                let v = it.next().ok_or("--unauth-array-len missing value")?;
                args.unauth_array_len = Some(v.parse().map_err(|_| "invalid --unauth-array-len")?);
            }
            "--allocated-leaves" => {
                let v = it.next().ok_or("--allocated-leaves missing value")?;
                args.allocated_leaves = Some(v.parse().map_err(|_| "invalid --allocated-leaves")?);
            }
            _ => return Err(format!("unknown arg {arg}")),
        }
    }
    Ok(args)
}

fn apply_overrides(args: &Args, file: &mut MockRegistryFile) {
    if let Some(tree_height) = args.tree_height {
        file.system.auth.tree_height = tree_height;
    }
    if let Some(kappa) = args.kappa {
        file.system.unauth.kappa = kappa;
    }
    if let Some(node_linearization) = args.node_linearization.as_deref() {
        let parsed = match node_linearization {
            "preorder" => NodeLinearization::Preorder,
            _ => {
                eprintln!("invalid --node-linearization (expected preorder)");
                std::process::exit(1);
            }
        };
        file.system.auth.node_linearization = parsed;
    }
    if let Some(shard_count) = args.shard_count {
        file.system.unauth.shard_count = shard_count;
    }
    if let Some(len) = args.unauth_array_len {
        file.system.unauth_array_len = Some(len);
    }
    if let Some(alloc) = args.allocated_leaves {
        file.system.auth.allocated_leaves = alloc;
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub enum ResponseFault {
    Honest,
    CorruptReadResponses { after_file: Option<PathBuf> },
}

#[allow(dead_code)]
impl ResponseFault {
    pub fn corrupt_reads_from_env() -> Self {
        Self::CorruptReadResponses {
            after_file: env::var_os("NET_DEMO_CORRUPT_AFTER_FILE").map(PathBuf::from),
        }
    }

    fn should_corrupt(&self) -> bool {
        match self {
            ResponseFault::Honest => false,
            ResponseFault::CorruptReadResponses { after_file } => after_file
                .as_ref()
                .map(|path| path.exists())
                .unwrap_or(true),
        }
    }

    fn maybe_apply(&self, response: net::Response) -> net::Response {
        if self.should_corrupt() {
            corrupt_demo_response(response)
        } else {
            response
        }
    }
}

fn corrupt_fp(value: &mut paper_a_optimized::Fp) {
    *value += paper_a_optimized::fp_one();
}

fn corrupt_net_fp(value: &mut net::wire::NetFp) {
    corrupt_fp(&mut value.0);
}

fn corrupt_fp_blob(blob: &mut net::wire::NetFpBlob) {
    for value in &mut blob.0 {
        corrupt_fp(value);
    }
}

fn corrupt_maybe_fp_blob(blob: &mut net::wire::NetMaybeFpBlob) {
    for value in &mut blob.0 {
        if let Some(value) = value {
            corrupt_fp(value);
        }
    }
}

fn corrupt_segment_shards(shards: &mut [net::wire::NetSegmentPrimaryShardReply]) {
    for shard in shards {
        corrupt_fp_blob(&mut shard.values);
    }
}

fn corrupt_auth_nodes(nodes: &mut [net::wire::AuthReadNodeReply]) {
    for node in nodes {
        corrupt_maybe_fp_blob(&mut node.values);
    }
}

fn corrupt_demo_response(mut response: net::Response) -> net::Response {
    match &mut response {
        net::Response::UnauthRead { values } => corrupt_maybe_fp_blob(values),
        net::Response::UnauthReadBlock { rows }
        | net::Response::UnauthReadRowsAtZ { rows }
        | net::Response::UnauthReadRowsBatchAtZ { rows } => corrupt_fp_blob(&mut rows.values),
        net::Response::UnauthReadRowsMatrix { rows } => corrupt_fp_blob(&mut rows.values),
        net::Response::UnauthReadSegmentRaw { shards } => corrupt_segment_shards(shards),
        net::Response::UnauthMultiRead { values } => {
            for values in values {
                corrupt_maybe_fp_blob(values);
            }
        }
        net::Response::UnauthRepairShard { values } => {
            for value in values {
                corrupt_net_fp(value);
            }
        }
        net::Response::UnauthRepairShardBatch { values } => {
            for batch in values {
                for value in batch {
                    corrupt_net_fp(value);
                }
            }
        }
        net::Response::UnauthEvalShardsAtRowLabels { values } => {
            for values in values {
                corrupt_fp_blob(values);
            }
        }
        net::Response::UnauthEvalShardsAtRowLabelsMatrix { values } => {
            corrupt_fp_blob(values);
        }
        net::Response::UnauthEvalPartialStitches { evals } => {
            for helper in evals.iter_mut().flatten() {
                corrupt_fp_blob(&mut helper.row_evals);
                corrupt_fp_blob(&mut helper.column_evals);
            }
        }
        net::Response::AuthRead { resp } => corrupt_auth_nodes(&mut resp.nodes),
        net::Response::AuthReadPrivate { resp } => {
            // Flip bits of every ciphertext: the client decrypts wrong
            // entries for every node of the opening.
            for reply in &mut resp.replies {
                for ciphertext in &mut reply.ciphertexts {
                    for byte in ciphertext.0.iter_mut().step_by(7) {
                        *byte ^= 0x5a;
                    }
                }
            }
        }
        net::Response::AuthReadSubtree { resp } => {
            corrupt_auth_nodes(&mut resp.leaf_nodes);
            corrupt_auth_nodes(&mut resp.upper_path_nodes);
            corrupt_segment_shards(&mut resp.raw_segment_shards);
        }
        net::Response::AuthReadSubtreeUpperPath { resp } => {
            corrupt_auth_nodes(&mut resp.upper_path_nodes);
        }
        _ => {}
    }
    response
}

pub fn run(response_fault: ResponseFault) {
    let args = parse_args().unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(1);
    });

    let mode = args.mode.clone().unwrap_or_else(|| {
        eprintln!("--mode is required");
        std::process::exit(1);
    });
    let registry_path = args.registry.clone().unwrap_or_else(|| {
        eprintln!("--registry is required");
        std::process::exit(1);
    });
    let mut file = MockRegistryFile::load(&registry_path).unwrap_or_else(|err| {
        eprintln!("failed to load registry: {:?}", err);
        std::process::exit(1);
    });
    apply_overrides(&args, &mut file);
    let listen = args.listen.clone().unwrap_or_else(|| {
        if let Some(party_id) = args.party_id {
            file.parties
                .iter()
                .find(|p| p.party_id == party_id)
                .map(|p| p.rpc_host.clone())
                .unwrap_or_else(|| {
                    eprintln!("party {party_id} not found in registry");
                    std::process::exit(1);
                })
        } else {
            file.parties
                .iter()
                .find(|p| p.active)
                .map(|p| p.rpc_host.clone())
                .unwrap_or_else(|| {
                    eprintln!("no active servers in registry");
                    std::process::exit(1);
                })
        }
    });
    let effective_party_id = args.party_id.or_else(|| {
        file.parties
            .iter()
            .find(|party| party.rpc_host == listen)
            .map(|party| party.party_id)
    });

    match mode.as_str() {
        "unauth" => {
            let (params, claims) = build_unauth_registry_view(&file).unwrap_or_else(|err| {
                eprintln!("failed to build unauth registry view: {:?}", err);
                std::process::exit(1);
            });
            let mut store = if claims.is_empty() {
                UnauthStore::new(params)
            } else if let Some(party_id) = effective_party_id {
                let local_claims = claim_bundles_for_party(&params, &claims, party_id)
                    .unwrap_or_else(|err| {
                        eprintln!(
                            "failed to resolve claim bundles for party {party_id}: {:?}",
                            err
                        );
                        std::process::exit(1);
                    });
                UnauthStore::new_with_claims(params, local_claims)
            } else {
                UnauthStore::new(params)
            };
            let initial_len = file
                .system
                .unauth_array_len
                .unwrap_or(public_values_per_segment(&store.params) as u64);
            store.extend(initial_len).unwrap_or_else(|err| {
                eprintln!("unauth extend failed: {:?}", err);
                std::process::exit(1);
            });
            let store = Arc::new(RwLock::new(store));
            let response_fault = response_fault.clone();
            net::serve(&listen, move |msg| {
                let response = if net::unauth::is_read_only_message(&msg) {
                    let store = store.read().expect("store lock");
                    net::unauth::handle_read_only_message(&store, msg)
                        .expect("read-only unauth message")
                } else {
                    let mut store = store.write().expect("store lock");
                    net::unauth::handle_message(&mut *store, msg)
                };
                response_fault.maybe_apply(response)
            })
            .unwrap();
        }
        "auth" => {
            let (params, all_claims) = build_unauth_registry_view(&file).unwrap_or_else(|err| {
                eprintln!(
                    "failed to build unauth registry view for auth server: {:?}",
                    err
                );
                std::process::exit(1);
            });
            let auth = AuthParams::new(file.system.auth.tree_height);
            let allocated = file.system.auth.allocated_leaves;
            let assigned_indices = args.party_id.map(|party_id| {
                let mut indices: Vec<usize> = if all_claims.is_empty() {
                    let entry = file
                        .assignments
                        .iter()
                        .find(|assignment| assignment.party_id == party_id)
                        .unwrap_or_else(|| {
                            eprintln!("no assignments found for party {party_id}");
                            std::process::exit(1);
                        });
                    entry.indices.iter().map(|idx| *idx as usize).collect()
                } else {
                    claim_bundles_for_party(&params, &all_claims, party_id)
                        .unwrap_or_else(|err| {
                            eprintln!(
                                "failed to resolve auth-server assigned indices for party {party_id}: {:?}",
                                err
                            );
                            std::process::exit(1);
                        })
                        .into_iter()
                        .map(|claim| claim.registry_index)
                        .collect()
                };
                indices.sort_unstable();
                indices.dedup();
                indices
            });
            let claims = if let Some(party_id) = effective_party_id {
                if all_claims.is_empty() {
                    None
                } else {
                    Some(
                        claim_bundles_for_party(&params, &all_claims, party_id).unwrap_or_else(
                            |err| {
                                eprintln!(
                                    "failed to resolve auth-server claim bundles for party {party_id}: {:?}",
                                    err
                                );
                                std::process::exit(1);
                            },
                        ),
                    )
                }
            } else {
                None
            };
            let server = AuthServer::new_with_assignments_and_claims(
                params,
                auth,
                file.system.auth.node_linearization,
                allocated,
                assigned_indices,
                claims,
            )
            .unwrap_or_else(|err| {
                eprintln!("auth server init failed: {:?}", err);
                std::process::exit(1);
            });
            let server = Arc::new(RwLock::new(server));
            let response_fault = response_fault.clone();
            // After a batch the private state is updated off the lock, while
            // the server keeps answering from the old one.
            #[cfg(feature = "private")]
            let keeper = paper_a_optimized::private::PrivateKeeper::new(Arc::clone(&server));
            net::serve(&listen, move |msg| {
                let response = if net::auth::is_read_only_message(&msg) {
                    let server = server.read().expect("server lock");
                    net::auth::handle_read_only_message(&server, msg)
                        .expect("read-only auth message")
                } else if net::unauth::is_auth_mode_read_only_message(&msg) {
                    let server = server.read().expect("server lock");
                    net::unauth::handle_auth_mode_read_only_message(&server, msg)
                        .expect("read-only auth-mode unauth message")
                } else {
                    let is_batch = matches!(
                        msg,
                        net::Message::AuthWrite { .. }
                            | net::Message::AuthWriteBatch { .. }
                            | net::Message::AuthWriteFreshSubtree { .. }
                    );
                    let response = handle_auth_locked(&server, msg);
                    #[cfg(feature = "private")]
                    if is_batch {
                        keeper.after_batch();
                    }
                    #[cfg(not(feature = "private"))]
                    let _ = is_batch;
                    response
                };
                response_fault.maybe_apply(response)
            })
            .unwrap();
        }
        other => {
            eprintln!("unknown mode {other}");
            std::process::exit(1);
        }
    }
}

/// Handle a message that changes the auth server, under its write lock.
fn handle_auth_locked(server: &RwLock<AuthServer>, msg: net::Message) -> net::Response {
    let mut server = server.write().expect("server lock");
    match msg {
        net::Message::AuthRead { .. }
        | net::Message::AuthReadSubtree { .. }
        | net::Message::AuthReadSubtreeUpperPath { .. }
        | net::Message::AuthWrite { .. }
        | net::Message::AuthWriteBatch { .. }
        | net::Message::AuthWriteFreshSubtree { .. }
        | net::Message::AuthExtend { .. }
        | net::Message::AuthGetLen
        | net::Message::AuthReadPrivate { .. }
        | net::Message::UnauthMultiRead { .. } => net::auth::handle_message(&mut *server, msg),
        net::Message::UnauthRead { .. }
        | net::Message::UnauthReadBlock { .. }
        | net::Message::UnauthReadSegmentRaw { .. }
        | net::Message::UnauthReadRowsAtZ { .. }
        | net::Message::UnauthReadRowsBatchAtZ { .. }
        | net::Message::UnauthReadRowsMatrix { .. }
        | net::Message::UnauthOptimizedRead { .. }
        | net::Message::UnauthRepairShard { .. }
        | net::Message::UnauthRepairShardBatch { .. }
        | net::Message::UnauthEvalShardsAtRowLabels { .. }
        | net::Message::UnauthEvalShardsAtRowLabelsMatrix { .. }
        | net::Message::UnauthApplyDeltas { .. }
        | net::Message::UnauthApplyBlockDeltas { .. }
        | net::Message::UnauthExtend { .. }
        | net::Message::UnauthKappaDouble
        | net::Message::UnauthKappaHalve { .. }
        | net::Message::UnauthEvalPartialStitches { .. }
        | net::Message::UnauthRetainClaims { .. }
        | net::Message::UnauthGetLen => net::unauth::handle_message(&mut *server, msg),
    }
}
