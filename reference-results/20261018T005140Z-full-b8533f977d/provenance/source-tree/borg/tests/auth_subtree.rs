use append_only_log_demo::frontier::{
    PreparedBulkUpdatesAppend, append_aligned_subtree_from_frontier,
};
use append_only_log_demo::layout::BULK_UPDATES_SIZE;
use paper_a_optimized::auth::client::{AuthClient, AuthTransport, ClientError as AuthClientError};
use paper_a_optimized::auth::service::{
    AuthBatchWriteRequest, AuthBatchWriteResponse, AuthFreshSubtreeWriteRequest, AuthReadResponse,
    AuthRequest, AuthResponse, AuthServer, AuthService, AuthSubtreeReadResponse,
    AuthSubtreeUpperPathReadResponse, AuthWriteRequest, AuthWriteResponse,
};
use paper_a_optimized::merkle::recompute_root;
use paper_a_optimized::unauth::{
    ClaimBundle, UnauthOptimizedReadRequest, UnauthOptimizedReadResponse,
    UnauthReadObjectContribution, error_correcting_share_target, handle_optimized_read_request,
};
use paper_a_optimized::{
    AuthConfig, AuthParams, AuthSubtreeProof, CodedArrayConfig, Config, RowShare, ShardShare,
    Value256, build_coded_params, build_default_hashes, build_shard_registry_deterministic,
    default_hash, fp_one, random_value,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct LocalServer {
    label: String,
    corrupt_reads: bool,
    inner: Arc<Mutex<AuthServer>>,
}

#[derive(Clone)]
struct LocalTransport {
    optimized: bool,
    captured_request: Option<Arc<Mutex<Option<UnauthOptimizedReadRequest>>>>,
}

struct BuiltClaimBoundClient {
    client: AuthClient<LocalTransport>,
    servers: Vec<LocalServer>,
}

impl LocalTransport {
    fn corrupt_shares(shares: &mut [paper_a_optimized::ShardShare]) {
        for share in shares {
            if let Some(mut value) = share.value() {
                value += paper_a_optimized::Fp::from(100_000u64 + share.registry_index() as u64);
                *share = paper_a_optimized::ShardShare::new(share.registry_index(), Some(value));
            }
        }
    }

    fn read_subtree_upper_path_impl(
        &self,
        server: &LocalServer,
        start_index: u64,
        leaf_count: usize,
    ) -> Result<AuthSubtreeUpperPathReadResponse, AuthClientError> {
        let mut guard = server.inner.lock().expect("auth server lock");
        match AuthService::handle_request(
            &mut *guard,
            AuthRequest::ReadSubtreeUpperPath {
                start_index,
                leaf_count,
            },
        ) {
            AuthResponse::ReadSubtreeUpperPath { resp } => Ok(resp),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!(
                "invalid auth subtree upper-path response: {:?}",
                other
            ))),
        }
    }
}

impl AuthTransport for LocalTransport {
    type Server = LocalServer;

    fn supports_subtree_upper_path_reads(&self) -> bool {
        true
    }

    fn supports_fresh_subtree_writes(&self) -> bool {
        true
    }

    fn server_label(&self, server: &Self::Server) -> String {
        server.label.clone()
    }

    fn auth_get_len(&self, server: &Self::Server) -> Result<u64, AuthClientError> {
        let mut guard = server.inner.lock().expect("auth server lock");
        match AuthService::handle_request(&mut *guard, AuthRequest::GetLen) {
            AuthResponse::Len { len } => Ok(len),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!(
                "invalid auth len response: {:?}",
                other
            ))),
        }
    }

    fn auth_read(
        &self,
        server: &Self::Server,
        index: u64,
    ) -> Result<AuthReadResponse, AuthClientError> {
        let mut guard = server.inner.lock().expect("auth server lock");
        match AuthService::handle_request(&mut *guard, AuthRequest::Read { index }) {
            AuthResponse::Read { mut resp } => {
                if server.corrupt_reads {
                    for node in &mut resp.nodes {
                        Self::corrupt_shares(&mut node.shares);
                    }
                }
                Ok(resp)
            }
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!(
                "invalid auth read response: {:?}",
                other
            ))),
        }
    }

    fn auth_read_subtree(
        &self,
        server: &Self::Server,
        start_index: u64,
        leaf_count: usize,
    ) -> Result<AuthSubtreeReadResponse, AuthClientError> {
        let mut guard = server.inner.lock().expect("auth server lock");
        match AuthService::handle_request(
            &mut *guard,
            AuthRequest::ReadSubtree {
                start_index,
                leaf_count,
            },
        ) {
            AuthResponse::ReadSubtree { mut resp } => {
                if server.corrupt_reads {
                    for node in &mut resp.leaf_nodes {
                        Self::corrupt_shares(&mut node.shares);
                    }
                    for node in &mut resp.upper_path_nodes {
                        Self::corrupt_shares(&mut node.shares);
                    }
                }
                Ok(resp)
            }
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!(
                "invalid auth subtree read response: {:?}",
                other
            ))),
        }
    }

    fn auth_read_subtree_upper_path(
        &self,
        server: &Self::Server,
        start_index: u64,
        leaf_count: usize,
    ) -> Result<AuthSubtreeUpperPathReadResponse, AuthClientError> {
        self.read_subtree_upper_path_impl(server, start_index, leaf_count)
    }

    fn unauth_multi_read(
        &self,
        server: &Self::Server,
        indices: &[u64],
    ) -> Result<Vec<Vec<paper_a_optimized::ShardShare>>, AuthClientError> {
        let mut guard = server.inner.lock().expect("auth server lock");
        match AuthService::handle_request(
            &mut *guard,
            AuthRequest::UnauthMultiRead {
                idxs: indices.to_vec(),
            },
        ) {
            AuthResponse::UnauthMultiRead { mut shares } => {
                if server.corrupt_reads {
                    for batch in &mut shares {
                        Self::corrupt_shares(batch);
                    }
                }
                Ok(shares)
            }
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!(
                "invalid unauth multi-read response: {:?}",
                other
            ))),
        }
    }

    fn supports_unauth_optimized_reads(&self) -> bool {
        self.optimized
    }

    fn unauth_optimized_read(
        &self,
        server: &Self::Server,
        request: &paper_a_optimized::unauth::UnauthOptimizedReadRequest,
    ) -> Result<paper_a_optimized::unauth::UnauthOptimizedReadResponse, AuthClientError> {
        if let Some(captured) = &self.captured_request {
            *captured.lock().expect("capture optimized request lock") = Some(request.clone());
        }
        let guard = server.inner.lock().expect("auth server lock");
        let mut response =
            paper_a_optimized::unauth::handle_optimized_read_request(&guard.store.unauth, request);
        if server.corrupt_reads {
            for contribution in &mut response.objects {
                match contribution {
                    paper_a_optimized::unauth::UnauthReadObjectContribution::Segment {
                        shards,
                        ..
                    } => {
                        for shard in shards {
                            for value in &mut shard.values {
                                *value += paper_a_optimized::Fp::from(
                                    100_000u64 + shard.registry_index as u64,
                                );
                            }
                        }
                    }
                    paper_a_optimized::unauth::UnauthReadObjectContribution::Block {
                        rows, ..
                    } => {
                        for row in rows {
                            let value = row.value() + paper_a_optimized::Fp::from(100_000u64);
                            *row = paper_a_optimized::RowShare::new(row.row_label(), value);
                        }
                    }
                    paper_a_optimized::unauth::UnauthReadObjectContribution::Point {
                        shares,
                        ..
                    } => Self::corrupt_shares(shares),
                }
            }
        }
        Ok(response)
    }

    fn auth_write(
        &self,
        server: &Self::Server,
        req: &AuthWriteRequest,
    ) -> Result<AuthWriteResponse, AuthClientError> {
        let mut guard = server.inner.lock().expect("auth server lock");
        match AuthService::handle_request(&mut *guard, AuthRequest::Write { req: req.clone() }) {
            AuthResponse::Write { resp } => Ok(resp),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!(
                "invalid auth write response: {:?}",
                other
            ))),
        }
    }

    fn auth_write_batch(
        &self,
        server: &Self::Server,
        req: &AuthBatchWriteRequest,
    ) -> Result<AuthBatchWriteResponse, AuthClientError> {
        let mut guard = server.inner.lock().expect("auth server lock");
        match AuthService::handle_request(&mut *guard, AuthRequest::WriteBatch { req: req.clone() })
        {
            AuthResponse::WriteBatch { resp } => Ok(resp),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!(
                "invalid auth batch write response: {:?}",
                other
            ))),
        }
    }

    fn auth_write_fresh_subtree(
        &self,
        server: &Self::Server,
        req: &AuthFreshSubtreeWriteRequest,
    ) -> Result<AuthBatchWriteResponse, AuthClientError> {
        let mut guard = server.inner.lock().expect("auth server lock");
        match AuthService::handle_request(
            &mut *guard,
            AuthRequest::WriteFreshSubtree { req: req.clone() },
        ) {
            AuthResponse::WriteFreshSubtree { resp } => Ok(resp),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!(
                "invalid auth fresh subtree write response: {:?}",
                other
            ))),
        }
    }

    fn auth_extend(&self, server: &Self::Server, new_len: u64) -> Result<bool, AuthClientError> {
        let mut guard = server.inner.lock().expect("auth server lock");
        match AuthService::handle_request(
            &mut *guard,
            AuthRequest::Extend {
                new_leaves: new_len,
            },
        ) {
            AuthResponse::Extend { ok } => Ok(ok),
            AuthResponse::Error { message } => Err(AuthClientError::Net(message)),
            other => Err(AuthClientError::InvalidResponse(format!(
                "invalid auth extend response: {:?}",
                other
            ))),
        }
    }
}

fn build_registry(g: usize, s: usize, segment_len: usize) -> Vec<paper_a_optimized::ShardRegEntry> {
    let n_grid = segment_len;
    build_shard_registry_deterministic(g, s, n_grid)
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

fn make_client() -> AuthClient<LocalTransport> {
    let cfg = Config {
        auth: AuthConfig {
            tree_height: 4,
            alloc_leaf_batch: 4,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            segment_len: 8,
            s: 4,
            shard_count: 8,
            shard_registry: build_registry(8, 4, 8),
            party_registry_indices: vec![(0..8).collect()],
            seed: 7,
        },
    };
    let params = build_coded_params(&cfg.coded);
    let auth = AuthParams::new(cfg.auth.tree_height);
    let server = LocalServer {
        label: "local".to_string(),
        corrupt_reads: false,
        inner: Arc::new(Mutex::new(
            AuthServer::new(params.clone(), auth.clone(), cfg.auth.node_linearization, 0)
                .expect("build auth server"),
        )),
    };
    AuthClient::new(
        LocalTransport {
            optimized: false,
            captured_request: None,
        },
        params,
        auth,
        cfg.auth.node_linearization,
        0,
        server.clone(),
        vec![server],
    )
}

fn make_claim_bound_client(
    tree_height: usize,
    alloc_leaf_batch: usize,
) -> AuthClient<LocalTransport> {
    let shard_splits = [vec![0, 1, 2, 3], vec![4, 5], vec![6, 7]];
    make_claim_bound_client_with_layout(tree_height, alloc_leaf_batch, 128, 16, &shard_splits)
}

fn make_claim_bound_client_with_layout(
    tree_height: usize,
    alloc_leaf_batch: usize,
    segment_len: usize,
    s: usize,
    shard_splits: &[Vec<usize>],
) -> AuthClient<LocalTransport> {
    make_claim_bound_client_with_layout_and_options(
        tree_height,
        alloc_leaf_batch,
        segment_len,
        s,
        shard_splits,
        0,
        false,
    )
}

fn make_claim_bound_client_with_layout_and_corrupt_prefix(
    tree_height: usize,
    alloc_leaf_batch: usize,
    segment_len: usize,
    s: usize,
    shard_splits: &[Vec<usize>],
    corrupt_prefix: usize,
) -> AuthClient<LocalTransport> {
    make_claim_bound_client_with_layout_and_options(
        tree_height,
        alloc_leaf_batch,
        segment_len,
        s,
        shard_splits,
        corrupt_prefix,
        false,
    )
}

fn make_claim_bound_client_with_layout_and_options(
    tree_height: usize,
    alloc_leaf_batch: usize,
    segment_len: usize,
    s: usize,
    shard_splits: &[Vec<usize>],
    corrupt_prefix: usize,
    optimized: bool,
) -> AuthClient<LocalTransport> {
    build_claim_bound_client_with_layout_and_options(
        tree_height,
        alloc_leaf_batch,
        segment_len,
        s,
        shard_splits,
        corrupt_prefix,
        optimized,
        None,
    )
    .client
}

fn build_claim_bound_client_with_layout_and_options(
    tree_height: usize,
    alloc_leaf_batch: usize,
    segment_len: usize,
    s: usize,
    shard_splits: &[Vec<usize>],
    corrupt_prefix: usize,
    optimized: bool,
    captured_request: Option<Arc<Mutex<Option<UnauthOptimizedReadRequest>>>>,
) -> BuiltClaimBoundClient {
    let shard_count = shard_splits
        .iter()
        .flatten()
        .copied()
        .max()
        .map(|idx| idx + 1)
        .unwrap_or(0);
    let cfg = Config {
        auth: AuthConfig {
            tree_height,
            alloc_leaf_batch,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            segment_len,
            s,
            shard_count,
            shard_registry: build_registry(shard_count, s, segment_len),
            party_registry_indices: shard_splits.to_vec(),
            seed: 42,
        },
    };
    let params = build_coded_params(&cfg.coded);
    let auth = AuthParams::new(cfg.auth.tree_height);
    let mut servers = Vec::new();
    for (party_id, indices) in shard_splits.iter().enumerate() {
        let server = LocalServer {
            label: format!("party-{party_id}"),
            corrupt_reads: party_id < corrupt_prefix,
            inner: Arc::new(Mutex::new(
                AuthServer::new_with_assignments_and_claims(
                    params.clone(),
                    auth.clone(),
                    cfg.auth.node_linearization,
                    0,
                    Some(indices.clone()),
                    Some(build_party_claims(&params, indices)),
                )
                .expect("build auth server"),
            )),
        };
        servers.push(server);
    }
    let server = servers[0].clone();
    let client = AuthClient::new(
        LocalTransport {
            optimized,
            captured_request,
        },
        params,
        auth,
        cfg.auth.node_linearization,
        0,
        server,
        servers.clone(),
    );
    BuiltClaimBoundClient { client, servers }
}

fn sample_leaves(seed: u64, len: usize) -> Vec<Value256> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..len).map(|_| random_value(&mut rng)).collect()
}

fn capture_optimized_responses(
    servers: &[LocalServer],
    request: &UnauthOptimizedReadRequest,
    helper_count: usize,
) -> Vec<UnauthOptimizedReadResponse> {
    servers
        .iter()
        .take(helper_count)
        .enumerate()
        .map(|(helper_index, server)| {
            let guard = server.inner.lock().expect("auth server lock");
            let mut response = handle_optimized_read_request(&guard.store.unauth, request);
            response.helper_index = Some(helper_index);
            response
        })
        .collect()
}

fn corrupt_optimized_responses_distributed(
    responses: &[UnauthOptimizedReadResponse],
    bad_helper_count: usize,
) -> Vec<UnauthOptimizedReadResponse> {
    let mut corrupted = responses.to_vec();
    for (bad_pos, response) in corrupted.iter_mut().take(bad_helper_count).enumerate() {
        if response.objects.is_empty() {
            continue;
        }
        let object_pos_to_corrupt = bad_pos % response.objects.len();
        for (object_pos, contribution) in response.objects.iter_mut().enumerate() {
            if object_pos != object_pos_to_corrupt {
                continue;
            }
            match contribution {
                UnauthReadObjectContribution::Segment { shards, .. } => {
                    for shard in shards {
                        let mut delta = fp_one();
                        for value in &mut shard.values {
                            *value += delta;
                            delta += fp_one();
                        }
                    }
                }
                UnauthReadObjectContribution::Block { rows, .. } => {
                    for row in rows {
                        *row = RowShare::new(row.row_label(), row.value() + fp_one());
                    }
                }
                UnauthReadObjectContribution::Point { shares, .. } => {
                    for share in shares {
                        let corrupted_value = share.value().map(|value| value + fp_one());
                        *share = ShardShare::new(share.registry_index(), corrupted_value);
                    }
                }
            }
        }
    }
    corrupted
}

#[test]
fn append_only_log_correcting_leaf_read_returns_valid_proof_after_bad_helpers() {
    let need = 2usize;
    let shard_splits: Vec<Vec<usize>> = (0..(3 * need)).map(|idx| vec![idx]).collect();
    let mut client =
        make_claim_bound_client_with_layout_and_corrupt_prefix(4, 1, 8, 4, &shard_splits, need);
    assert_eq!(client.params.need, need);

    let leaves = sample_leaves(101, 1);
    let frontier = append_aligned_subtree_from_frontier(&mut client, 0, None, &leaves)
        .expect("append through appendOnlyLog frontier helper");
    let trusted_root = client.trusted_root().expect("trusted root after append");
    assert_eq!(trusted_root, frontier.path.root);

    let (value, path) = client
        .read_with_path_correcting(0, need)
        .expect("correcting appendOnlyLog read");
    assert_eq!(value, leaves[0]);

    let recomputed = recompute_root(
        client.auth.tree_height as u32,
        path.index as u64,
        value,
        &path.siblings,
        &client.auth.defaults,
    );
    assert_eq!(recomputed, path.root);
    assert_eq!(path.root, trusted_root);
}

#[test]
fn optimized_range_precollected_corrects_distributed_kappa_bad_helpers() {
    let need = 4usize;
    let shard_splits: Vec<Vec<usize>> = (0..(3 * need)).map(|idx| vec![idx]).collect();
    let captured_request = Arc::new(Mutex::new(None));
    let built = build_claim_bound_client_with_layout_and_options(
        7,
        4,
        16,
        4,
        &shard_splits,
        0,
        true,
        Some(Arc::clone(&captured_request)),
    );
    let mut client = built.client;
    assert_eq!(client.params.need, need);

    let leaves = sample_leaves(701, 48);
    client.append_leaves(&leaves).expect("append leaves");
    let trusted_root = client.trusted_root().expect("trusted root after append");

    let start = 5u64;
    let leaf_count = 32usize;
    let warm_range = client
        .read_range_correcting(start, leaf_count, need)
        .expect("warm correcting optimized range read");
    assert_eq!(
        warm_range,
        leaves[start as usize..start as usize + leaf_count]
    );

    let request = captured_request
        .lock()
        .expect("captured optimized request lock")
        .clone()
        .expect("captured optimized request");
    assert!(
        request.objects.len() >= need,
        "small benchmark-shaped range must have enough objects for distributed corrupted helpers: objects={}, need={}",
        request.objects.len(),
        need
    );

    let correcting_helper_replies = error_correcting_share_target(need, need);
    let clean_responses =
        capture_optimized_responses(&built.servers, &request, correcting_helper_replies);
    assert_eq!(clean_responses.len(), correcting_helper_replies);

    let mut clean_client = client.clone();
    let (clean_range, clean_proof) = clean_client
        .read_range_with_proof_from_optimized_responses_correcting(
            start,
            leaf_count,
            &clean_responses,
            need,
        )
        .expect("clean pre-collected correcting range open");
    assert_eq!(
        clean_range,
        leaves[start as usize..start as usize + leaf_count]
    );
    let clean_root = paper_a_optimized::auth::verify_range_proof(
        &client.auth,
        start,
        &clean_range,
        &clean_proof,
    )
    .expect("verify clean range proof");
    assert_eq!(clean_root, trusted_root);

    let bad_responses = corrupt_optimized_responses_distributed(&clean_responses, need);
    let (range, proof) = client
        .read_range_with_proof_from_optimized_responses_correcting(
            start,
            leaf_count,
            &bad_responses,
            need,
        )
        .expect("distributed bad-helper pre-collected correcting range open");
    assert_eq!(range, leaves[start as usize..start as usize + leaf_count]);
    let root = paper_a_optimized::auth::verify_range_proof(&client.auth, start, &range, &proof)
        .expect("verify corrected range proof");
    assert_eq!(root, trusted_root);
}

#[test]
fn subtree_append_and_read_round_trip() {
    let mut client = make_client();
    let leaves = sample_leaves(1, 4);

    let start = client.append_subtree(&leaves).expect("append subtree");
    assert_eq!(start, 0);

    let (read_back, proof) = client
        .read_subtree_with_proof(0, 4)
        .expect("read subtree with proof");
    assert_eq!(read_back, leaves);
    assert_eq!(proof.start_index, 0);
    assert_eq!(proof.leaf_count, 4);
    assert_eq!(proof.upper_siblings.len(), 2);
    assert_eq!(client.trusted_root(), Some(proof.root));

    let range = client.read_range(0, 4).expect("read range");
    assert_eq!(range, leaves);
}

#[test]
fn nonaligned_range_with_proof_uses_optimized_unauth_read() {
    let shard_splits = [
        vec![0],
        vec![1],
        vec![2],
        vec![3],
        vec![4],
        vec![5],
        vec![6],
        vec![7],
    ];
    let mut client =
        make_claim_bound_client_with_layout_and_options(4, 4, 16, 4, &shard_splits, 0, true);
    let leaves = sample_leaves(21, 8);
    client.append_leaves(&leaves).expect("append leaves");

    let (range, proof) = client
        .read_range_with_proof(1, 6)
        .expect("read range with proof");
    assert_eq!(range, leaves[1..7].to_vec());

    let verified_root =
        paper_a_optimized::auth::verify_range_proof(&client.auth, 1, &range, &proof)
            .expect("verify range proof");
    assert_eq!(client.trusted_root(), Some(verified_root));

    let repeated = client
        .read_batch(&(1u64..7).collect::<Vec<_>>())
        .expect("repeated reads");
    assert_eq!(repeated, range);

    let mut tampered = range.clone();
    tampered[2] = random_value(&mut ChaCha20Rng::seed_from_u64(999));
    assert!(matches!(
        paper_a_optimized::auth::verify_range_proof(&client.auth, 1, &tampered, &proof),
        Err(paper_a_optimized::ReadError::InvalidProof)
    ));

    let mut tampered_proof = proof.clone();
    assert!(!tampered_proof.proof_nodes.is_empty());
    tampered_proof.proof_nodes[0].1 = random_value(&mut ChaCha20Rng::seed_from_u64(1001));
    assert!(matches!(
        paper_a_optimized::auth::verify_range_proof(&client.auth, 1, &range, &tampered_proof),
        Err(paper_a_optimized::ReadError::InvalidProof)
    ));

    let checkpoint_pos = proof
        .proof_nodes
        .iter()
        .position(|(node, _)| {
            node.level == client.auth.tree_height as u32 - 1 && matches!(node.pos, 1 | 2)
        })
        .expect("range proof carries an internal checkpoint node");
    let mut tampered_checkpoint = proof.clone();
    tampered_checkpoint.proof_nodes[checkpoint_pos].1 =
        random_value(&mut ChaCha20Rng::seed_from_u64(1002));
    assert!(matches!(
        paper_a_optimized::auth::verify_range_proof(&client.auth, 1, &range, &tampered_checkpoint),
        Err(paper_a_optimized::ReadError::InvalidProof)
    ));

    let mut incomplete_proof = proof.clone();
    incomplete_proof.proof_nodes.pop();
    assert!(matches!(
        paper_a_optimized::auth::verify_range_proof(&client.auth, 1, &range, &incomplete_proof),
        Err(paper_a_optimized::ReadError::InvalidProof)
    ));
}

#[test]
fn nonaligned_range_correcting_uses_optimized_unauth_read() {
    let shard_splits = [
        vec![0],
        vec![1],
        vec![2],
        vec![3],
        vec![4],
        vec![5],
        vec![6],
        vec![7],
    ];
    let mut client =
        make_claim_bound_client_with_layout_and_options(4, 4, 16, 4, &shard_splits, 1, true);
    let leaves = sample_leaves(31, 8);
    client.append_leaves(&leaves).expect("append leaves");

    let range = client
        .read_range_correcting(1, 6, 1)
        .expect("correcting range read");
    assert_eq!(range, leaves[1..7].to_vec());
}

#[test]
fn subtree_write_replaces_aligned_interval() {
    let mut client = make_client();
    let initial = sample_leaves(2, 4);
    client
        .append_subtree(&initial)
        .expect("append initial subtree");

    let replacement = sample_leaves(3, 4);
    client
        .write_subtree(0, &replacement)
        .expect("write subtree");

    let read_back = client.read_subtree(0, 4).expect("read replaced subtree");
    assert_eq!(read_back, replacement);
}

#[test]
fn subtree_ops_reject_invalid_shapes() {
    let mut client = make_client();
    let bad = sample_leaves(4, 3);
    assert!(matches!(
        client.append_subtree(&bad),
        Err(AuthClientError::InvalidIndex)
    ));

    let good = sample_leaves(5, 4);
    client.append_subtree(&good).expect("append subtree");

    let replacement = sample_leaves(6, 2);
    assert!(matches!(
        client.read_subtree_with_proof(1, 2),
        Err(AuthClientError::InvalidIndex)
    ));
    assert!(matches!(
        client.write_subtree(1, &replacement),
        Err(AuthClientError::InvalidIndex)
    ));
}

#[test]
fn fresh_subtree_write_with_default_proof_round_trips() {
    let mut client = make_client();
    let leaves = sample_leaves(11, 4);
    client.extend(4).expect("extend fresh subtree range");
    let defaults = build_default_hashes(client.auth.tree_height);
    let proof = AuthSubtreeProof {
        start_index: 0,
        leaf_count: 4,
        upper_siblings: vec![default_hash(&defaults, 2), default_hash(&defaults, 1)],
        root: default_hash(&defaults, 0),
    };
    client
        .write_fresh_subtree_with_proof(proof, &leaves)
        .expect("write fresh subtree");
    let read_back = client.read_subtree(0, 4).expect("read fresh subtree");
    assert_eq!(read_back, leaves);
}

#[test]
fn fresh_subtree_write_rejects_nondefault_old_contents() {
    let mut client = make_client();
    let initial = sample_leaves(12, 4);
    client.append_subtree(&initial).expect("append initial");
    let replacement = sample_leaves(13, 4);
    let defaults = build_default_hashes(client.auth.tree_height);
    let proof = AuthSubtreeProof {
        start_index: 0,
        leaf_count: 4,
        upper_siblings: vec![default_hash(&defaults, 2), default_hash(&defaults, 1)],
        root: client.trusted_root().expect("trusted root"),
    };
    assert!(matches!(
        client.write_fresh_subtree_with_proof(proof, &replacement),
        Err(AuthClientError::Net(_))
    ));
}

#[test]
#[ignore = "debug-scale regression for claim-bound archive-style subtree appends"]
fn claim_bound_frontier_bulk_updates_subtree_appends_scale_past_twenty_batches() {
    const TREE_HEIGHT: usize = 22;
    const BULK_UPDATE_LEAVES: usize = 131_072;
    const BATCH_COUNT: usize = 20;
    let mut client = make_claim_bound_client(TREE_HEIGHT, BULK_UPDATE_LEAVES);
    let updates = sample_leaves(77, BULK_UPDATE_LEAVES);
    for batch_index in 0..BATCH_COUNT {
        let start = client
            .append_subtree(&updates)
            .unwrap_or_else(|err| panic!("batch {batch_index} append failed: {err:?}"));
        assert_eq!(start, (batch_index * BULK_UPDATE_LEAVES) as u64);
    }
}

#[test]
#[ignore = "debug reproducer for one high-prefix archive-style subtree write"]
fn claim_bound_high_prefix_bulk_updates_subtree_write_after_large_extend() {
    const TREE_HEIGHT: usize = 22;
    const BULK_UPDATE_LEAVES: usize = 131_072;
    const TARGET_BATCH: u64 = 19;
    let mut client = make_claim_bound_client(TREE_HEIGHT, BULK_UPDATE_LEAVES);
    let start = TARGET_BATCH * BULK_UPDATE_LEAVES as u64;
    let end = start + BULK_UPDATE_LEAVES as u64;
    client
        .extend(end)
        .unwrap_or_else(|err| panic!("extend to {end} failed: {err:?}"));
    let updates = sample_leaves(91, BULK_UPDATE_LEAVES);
    client
        .write_subtree(start, &updates)
        .unwrap_or_else(|err| panic!("high-prefix write at {start} failed: {err:?}"));
}

#[test]
fn claim_bound_large_segment_subtree_reads_cover_late_aligned_chunks() {
    const TREE_HEIGHT: usize = 16;
    const SEGMENT_LEN: usize = 4096;
    const S: usize = 64;
    let shard_splits: Vec<Vec<usize>> = (0..8)
        .map(|group| ((group * 8)..((group + 1) * 8)).collect())
        .collect();
    let mut client = make_claim_bound_client_with_layout(
        TREE_HEIGHT,
        SEGMENT_LEN,
        SEGMENT_LEN,
        S,
        &shard_splits,
    );
    let segment_values = sample_leaves(1234, SEGMENT_LEN);

    let start = client
        .append_subtree(&segment_values)
        .expect("append large subtree");
    assert_eq!(start, 0);

    let first_half = client
        .read_subtree(0, SEGMENT_LEN / 2)
        .expect("read first aligned half");
    assert_eq!(first_half, segment_values[..SEGMENT_LEN / 2].to_vec());

    let second_half = client
        .read_subtree((SEGMENT_LEN / 2) as u64, SEGMENT_LEN / 2)
        .expect("read second aligned half");
    assert_eq!(second_half, segment_values[SEGMENT_LEN / 2..].to_vec());

    let full = client
        .read_subtree(0, SEGMENT_LEN)
        .expect("read full aligned subtree");
    assert_eq!(full, segment_values);
}

#[test]
fn claim_bound_large_segment_subtree_reads_after_high_prefix_write() {
    const TREE_HEIGHT: usize = 20;
    const SEGMENT_LEN: usize = 4096;
    const S: usize = 64;
    const TARGET_SEGMENT: u64 = 16;
    let shard_splits: Vec<Vec<usize>> = (0..8)
        .map(|group| ((group * 8)..((group + 1) * 8)).collect())
        .collect();
    let mut client = make_claim_bound_client_with_layout(
        TREE_HEIGHT,
        SEGMENT_LEN,
        SEGMENT_LEN,
        S,
        &shard_splits,
    );
    let start = TARGET_SEGMENT * SEGMENT_LEN as u64;
    let end = start + SEGMENT_LEN as u64;
    client.extend(end).expect("extend high prefix");
    let segment_values = sample_leaves(5678, SEGMENT_LEN);
    client
        .write_subtree(start, &segment_values)
        .expect("write aligned high-prefix subtree");

    let first_half = client
        .read_subtree(start, SEGMENT_LEN / 2)
        .expect("read first aligned high-prefix half");
    assert_eq!(first_half, segment_values[..SEGMENT_LEN / 2].to_vec());

    let second_half = client
        .read_subtree(start + (SEGMENT_LEN / 2) as u64, SEGMENT_LEN / 2)
        .expect("read second aligned high-prefix half");
    assert_eq!(second_half, segment_values[SEGMENT_LEN / 2..].to_vec());

    let full = client
        .read_subtree(start, SEGMENT_LEN)
        .expect("read full aligned high-prefix subtree");
    assert_eq!(full, segment_values);
}

#[test]
fn claim_bound_bulk_update_geometry_subtree_reads_match_app_offsets() {
    const TREE_HEIGHT: usize = 22;
    const SEGMENT_LEN: usize = 4096;
    const S: usize = 64;
    const BULK_UPDATE_LEAVES: usize = 131_072;
    const BATCH_START: u64 = BULK_UPDATE_LEAVES as u64;
    let shard_splits: Vec<Vec<usize>> = (0..8)
        .map(|group| ((group * 8)..((group + 1) * 8)).collect())
        .collect();
    let mut client = make_claim_bound_client_with_layout(
        TREE_HEIGHT,
        BULK_UPDATE_LEAVES,
        SEGMENT_LEN,
        S,
        &shard_splits,
    );
    let end = BATCH_START + SEGMENT_LEN as u64;
    client
        .extend(end)
        .expect("extend through first segment-sized chunk of batch 1");
    let chunk = sample_leaves(91011, SEGMENT_LEN);
    client
        .write_subtree(BATCH_START, &chunk)
        .expect("write first segment-sized chunk inside batch 1");

    let first_half = client
        .read_subtree(BATCH_START, SEGMENT_LEN / 2)
        .expect("read first aligned batch chunk half");
    assert_eq!(first_half, chunk[..SEGMENT_LEN / 2].to_vec());

    let second_half = client
        .read_subtree(BATCH_START + (SEGMENT_LEN / 2) as u64, SEGMENT_LEN / 2)
        .expect("read second aligned batch chunk half");
    assert_eq!(second_half, chunk[SEGMENT_LEN / 2..].to_vec());

    let range = client
        .read_range(BATCH_START, SEGMENT_LEN)
        .expect("read exact segment-sized range at batch start");
    assert_eq!(range, chunk);
}

#[test]
#[ignore = "debug reproducer for prepared bulk-update frontier appends on the repair-benchmark layout"]
fn prepared_bulk_updates_frontier_appends_survive_sixteen_batches_on_repair_layout() {
    const TREE_HEIGHT: usize = 50;
    const SEGMENT_LEN: usize = 4096;
    const S: usize = 64;
    const SHARD_COUNT: usize = 65;
    const BATCH_COUNT: usize = 16;

    let shard_splits = vec![(0..SHARD_COUNT).collect::<Vec<_>>()];
    let mut client = make_claim_bound_client_with_layout(
        TREE_HEIGHT,
        BULK_UPDATES_SIZE,
        SEGMENT_LEN,
        S,
        &shard_splits,
    );
    let mut planner = PreparedBulkUpdatesAppend::new(TREE_HEIGHT).expect("planner");
    let updates = sample_leaves(20260420, BULK_UPDATES_SIZE);
    let mut frontier = None;
    let mut frontier_len = 0u64;

    for batch_index in 0..BATCH_COUNT {
        let next_frontier = planner
            .append_from_frontier(&mut client, frontier_len, frontier.as_ref(), &updates)
            .unwrap_or_else(|err| {
                panic!(
                    "prepared bulk-updates append failed at batch {batch_index}, frontier_len={frontier_len}: {err:?}"
                )
            });
        frontier_len += BULK_UPDATES_SIZE as u64;
        assert_eq!(
            client.trusted_root(),
            Some(next_frontier.path.root),
            "trusted root diverged after batch {batch_index}"
        );
        let (live_value, live_path) =
            client
                .read_with_path(frontier_len - 1)
                .unwrap_or_else(|err| {
                    panic!("read live frontier after batch {batch_index} failed: {err:?}")
                });
        assert_eq!(
            live_value, next_frontier.value,
            "cached frontier value diverged after batch {batch_index}"
        );
        assert_eq!(
            live_path.index, next_frontier.path.index,
            "cached frontier index diverged after batch {batch_index}"
        );
        assert_eq!(
            live_path.siblings, next_frontier.path.siblings,
            "cached frontier siblings diverged after batch {batch_index}"
        );
        assert_eq!(
            live_path.root, next_frontier.path.root,
            "cached frontier root diverged after batch {batch_index}"
        );
        frontier = Some(next_frontier);
    }
}

#[test]
#[ignore = "debug control for generic bulk-update subtree appends on the repair-benchmark layout"]
fn generic_bulk_updates_frontier_appends_survive_four_batches_on_repair_layout() {
    const TREE_HEIGHT: usize = 50;
    const SEGMENT_LEN: usize = 4096;
    const S: usize = 64;
    const SHARD_COUNT: usize = 65;
    const BATCH_COUNT: usize = 4;

    let shard_splits = vec![(0..SHARD_COUNT).collect::<Vec<_>>()];
    let mut client = make_claim_bound_client_with_layout(
        TREE_HEIGHT,
        BULK_UPDATES_SIZE,
        SEGMENT_LEN,
        S,
        &shard_splits,
    );
    let updates = sample_leaves(20260421, BULK_UPDATES_SIZE);

    for batch_index in 0..BATCH_COUNT {
        let start = client
            .append_subtree(&updates)
            .unwrap_or_else(|err| panic!("generic append failed at batch {batch_index}: {err:?}"));
        assert_eq!(start, (batch_index * BULK_UPDATES_SIZE) as u64);
        let (live_value, live_path) = client
            .read_with_path(start + BULK_UPDATES_SIZE as u64 - 1)
            .unwrap_or_else(|err| {
                panic!("read live frontier after generic batch {batch_index} failed: {err:?}")
            });
        assert_eq!(client.trusted_root(), Some(live_path.root));
        assert_eq!(live_value, *updates.last().expect("last update leaf"));
    }
}
