use append_only_log_demo::frontier::append_aligned_subtree_from_frontier;
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
struct LocalServer {
    label: usize,
    inner: Arc<Mutex<AuthServer>>,
    /// Return wrong shares on point reads.
    corrupt_reads: bool,
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
fn build_servers(
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

fn build_params(kappa: usize, shard_count: usize, seed: u64) -> CodedArrayParams {
    build_coded_params(&CodedArrayConfig {
        kappa,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count),
        party_registry_indices: (0..shard_count).map(|idx| vec![idx]).collect(),
        seed,
    })
}

/// Double `kappa` on every server: each one stitches its own fragments.
fn kappa_double_all(servers: &[LocalServer]) {
    for server in servers {
        let mut guard = server.inner.lock().expect("server lock");
        match UnauthService::handle_request(&mut *guard, UnauthRequest::KappaDouble) {
            UnauthResponse::KappaDouble { ok: true } => {}
            other => panic!("unexpected kappa-double response: {:?}", other),
        }
    }
}

/// Halve `kappa` on every server. All servers are at the doubled threshold,
/// so each one's missing source half is decoded from the partial-stitch
/// evaluations of the others.
fn kappa_halve_all(params: &CodedArrayParams, servers: &[LocalServer]) {
    let col_labels = params
        .shard_registry
        .iter()
        .map(|entry| entry.z)
        .collect::<Vec<_>>();
    let row_labels = params.row_labels.clone();
    let segment_count = {
        let guard = servers[0].inner.lock().expect("server lock");
        guard.store.unauth.segments()
    };
    let segs = (0..segment_count).collect::<Vec<_>>();

    // Gather every server's evaluations while all are still at the doubled
    // threshold.
    let mut evals_by_seg: Vec<Vec<PartialStitchEvals>> = vec![Vec::new(); segment_count];
    for server in servers {
        let mut guard = server.inner.lock().expect("server lock");
        match UnauthService::handle_request(
            &mut *guard,
            UnauthRequest::EvalPartialStitches {
                segs: segs.clone(),
                col_labels: col_labels.clone(),
                row_labels: row_labels.clone(),
            },
        ) {
            UnauthResponse::EvalPartialStitches { evals } => {
                for (seg, helper_evals) in evals.into_iter().enumerate() {
                    evals_by_seg[seg].extend(helper_evals);
                }
            }
            other => panic!("unexpected partial-stitch response: {:?}", other),
        }
    }

    for (registry_index, server) in servers.iter().enumerate() {
        let source_halves = evals_by_seg
            .iter()
            .enumerate()
            .map(|(seg, helper_evals)| {
                let row_samples = helper_evals
                    .iter()
                    .map(|helper| {
                        (
                            params.row_labels[helper.registry_index],
                            helper.row_evals[registry_index],
                        )
                    })
                    .collect::<Vec<_>>();
                let column_samples = helper_evals
                    .iter()
                    .map(|helper| {
                        (
                            params.shard_registry[helper.registry_index].z,
                            helper.column_evals[registry_index],
                        )
                    })
                    .collect::<Vec<_>>();
                let fragments =
                    decode_leading_source_half(params, seg, &row_samples, &column_samples, 0)
                        .expect("decode leading source half");
                SourceHalfState::Leading {
                    columns_00: vec![(registry_index, fragments.column_00)],
                    columns_01: vec![(registry_index, fragments.column_01)],
                    rows_00: vec![(registry_index, fragments.row_00)],
                    rows_10: vec![(registry_index, fragments.row_10)],
                }
            })
            .collect::<Vec<_>>();
        let mut guard = server.inner.lock().expect("server lock");
        match UnauthService::handle_request(
            &mut *guard,
            UnauthRequest::KappaHalve { source_halves },
        ) {
            UnauthResponse::KappaHalve { ok: true } => {}
            other => panic!("unexpected kappa-halve response: {:?}", other),
        }
    }
}

#[test]
fn auth_batch_reads_survive_kappa_double_and_kappa_halve() {
    let params = build_params(2, 8, 42);
    let auth = AuthParams::new(3);
    let allocated_leaves = 4usize;
    let servers = build_servers(&params, &auth, allocated_leaves);

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
        .expect("read before kappa-double");

    kappa_double_all(&servers);
    let doubled = build_kappa_double_params(&client.params).expect("doubled params");
    client.params = doubled.clone();
    client.clear_trusted_root();
    let after_double = client
        .read_batch(&[0, 1, 2, 3])
        .expect("read after kappa-double");
    assert_eq!(after_double, before);

    kappa_halve_all(&doubled, &servers);
    client.params = build_kappa_halve_params(&doubled).expect("halved params");
    client.clear_trusted_root();
    let after_halve = client
        .read_batch(&[0, 1, 2, 3])
        .expect("read after kappa-halve");
    assert_eq!(after_halve, before);
}

#[test]
fn auth_append_only_frontier_survives_kappa_double_then_next_append() {
    let params = build_params(2, 8, 43);
    let auth = AuthParams::new(4);
    let servers = build_servers(&params, &auth, 0);

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
        client.read(0).expect("read first before kappa-double"),
        first[0]
    );

    kappa_double_all(&servers);
    let doubled = build_kappa_double_params(&client.params).expect("doubled params");
    client.params = doubled.clone();
    assert_eq!(
        client.read(0).expect("read first after kappa-double"),
        first[0]
    );

    let second = (0..4)
        .map(|idx| encode_value(Fp::from(100 + idx as u64)))
        .collect::<Vec<_>>();
    let frontier = append_aligned_subtree_from_frontier(&mut client, 4, Some(&frontier), &second)
        .expect("second append after kappa-double");
    assert_eq!(
        client.read(0).expect("read first after second append"),
        first[0]
    );
    assert_eq!(
        client.read(4).expect("read second after kappa-double"),
        second[0]
    );

    // Undo the stitch and keep appending at the source threshold.
    kappa_halve_all(&doubled, &servers);
    client.params = build_kappa_halve_params(&doubled).expect("halved params");
    for (idx, value) in first.iter().chain(second.iter()).enumerate() {
        assert_eq!(
            client.read(idx as u64).expect("read after kappa-halve"),
            *value
        );
    }
    let third = (0..8)
        .map(|idx| encode_value(Fp::from(1_000 + idx as u64)))
        .collect::<Vec<_>>();
    append_aligned_subtree_from_frontier(&mut client, 8, Some(&frontier), &third)
        .expect("third append after kappa-halve");
    for (idx, value) in first
        .iter()
        .chain(second.iter())
        .chain(third.iter())
        .enumerate()
    {
        assert_eq!(
            client.read(idx as u64).expect("read after third append"),
            *value
        );
    }
}

/// Cluster at threshold `kappa` holding `leaves` written leaves, with the
/// written values and the root they commit to.
fn written_cluster(
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
fn reader(
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

fn stored_kappa(server: &LocalServer) -> usize {
    server.inner.lock().expect("server lock").params.need
}

#[test]
fn clients_open_at_higher_thresholds_from_servers_at_the_stored_threshold() {
    let leaves = 11usize;
    let (params, auth, servers, values, root) = written_cluster(2, 16, 5, leaves, 51);

    for levels in 1..=3 {
        let target = build_params_for_kappa(&params, 2 << levels).expect("target params");
        let mut client = reader(&target, &auth, leaves, &servers);
        for (idx, value) in values.iter().enumerate() {
            assert_eq!(
                client.read(idx as u64).expect("point opening"),
                *value,
                "leaf {idx} at kappa = {}",
                target.need
            );
        }
        // The opening verifies against the commitment made at the stored
        // threshold.
        assert_eq!(client.trusted_root(), Some(root));
        assert_eq!(client.read_range(0, leaves).expect("range opening"), values);
        assert_eq!(
            client.read_range(3, 6).expect("inner range opening"),
            values[3..9]
        );
    }
    // Serving at a higher threshold leaves every server's state in place.
    assert!(servers.iter().all(|server| stored_kappa(server) == 2));
}

#[test]
fn servers_at_different_thresholds_answer_the_same_opening() {
    let leaves = 13usize;
    let (params, auth, servers, values, root) = written_cluster(2, 16, 5, leaves, 52);
    let doubled = build_kappa_double_params(&params).expect("doubled params");
    let target = build_kappa_double_params(&doubled).expect("target params");

    // A second cluster written directly at the target threshold.
    let direct_servers = build_servers(&target, &auth, leaves);
    let mut direct_client = reader(&target, &auth, leaves, &direct_servers);
    let updates = values
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx as u64, *value))
        .collect::<Vec<_>>();
    direct_client.write_batch(&updates).expect("direct write");
    assert_eq!(direct_client.trusted_root(), Some(root));

    // Claims 0..4 stay at kappa, 4..8 stitch once, 8..12 stitch twice, and
    // 12..16 come from the cluster written at the target threshold.
    kappa_double_all(&servers[4..12]);
    kappa_double_all(&servers[8..12]);
    let mut mixed = servers[..12].to_vec();
    mixed.extend_from_slice(&direct_servers[12..]);
    assert_eq!(
        mixed.iter().map(stored_kappa).collect::<Vec<_>>(),
        [2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8]
    );

    // The target threshold is 8: any opening needs claims from at least two
    // of the four groups.
    for skip in 0..4 {
        let mut helpers = mixed.clone();
        helpers.drain(4 * skip..4 * skip + 4);
        let mut client = reader(&target, &auth, leaves, &helpers);
        client.set_trusted_root(root);
        for (idx, value) in values.iter().enumerate() {
            assert_eq!(client.read(idx as u64).expect("point opening"), *value);
        }
        assert_eq!(client.read_range(0, leaves).expect("range opening"), values);
    }
}

#[test]
fn on_the_fly_openings_correct_bad_helpers() {
    let leaves = 9usize;
    let (params, auth, mut servers, values, root) = written_cluster(2, 16, 4, leaves, 53);
    let target = build_params_for_kappa(&params, 4).expect("target params");
    servers[1].corrupt_reads = true;
    servers[6].corrupt_reads = true;

    let mut client = reader(&target, &auth, leaves, &servers);
    client.set_trusted_root(root);
    for (idx, value) in values.iter().enumerate() {
        assert_eq!(
            client
                .read_correcting(idx as u64, 2)
                .expect("correcting point opening"),
            *value
        );
    }
}

#[test]
fn a_server_answers_only_thresholds_it_serves() {
    let leaves = 6usize;
    let (_, _, servers, _, _) = written_cluster(4, 16, 4, leaves, 54);
    let mut guard = servers[0].inner.lock().expect("server lock");

    for (kappa, served) in [
        (4usize, true),
        (8, true),
        (16, true),
        (2, false),
        (12, false),
        (32, false),
    ] {
        let nodes =
            match AuthService::handle_request(&mut *guard, AuthRequest::Read { index: 0, kappa }) {
                AuthResponse::Read { resp } => resp.nodes,
                other => panic!("unexpected read response: {:?}", other),
            };
        assert_eq!(!nodes.is_empty(), served, "kappa = {kappa}");

        let request = UnauthOptimizedReadRequest {
            kappa,
            objects: vec![UnauthReadObjectId::Point { array_index: 0 }],
        };
        let response = guard.handle_optimized_read(&request);
        assert_eq!(!response.objects.is_empty(), served, "kappa = {kappa}");
        // The service message gives the same answer.
        match UnauthService::handle_request(&mut *guard, UnauthRequest::OptimizedRead { request }) {
            UnauthResponse::OptimizedRead { response } => {
                assert_eq!(!response.objects.is_empty(), served, "kappa = {kappa}")
            }
            other => panic!("unexpected optimized read response: {:?}", other),
        }
    }
}
