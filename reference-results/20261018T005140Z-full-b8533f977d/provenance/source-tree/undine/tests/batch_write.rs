//! Batch write tests for multiproof updates.

mod support;

use paper_a_optimized::auth::AuthParams;
use paper_a_optimized::auth::client::{AuthClient, AuthTransport, ClientError as AuthClientError};
use paper_a_optimized::auth::service::{
    AuthBatchWriteRequest, AuthBatchWriteResponse, AuthReadResponse, AuthSubtreeReadResponse,
    AuthWriteRequest, AuthWriteResponse,
};
use paper_a_optimized::merkle::{NodeKey, leaf_key, leaf_label, node_hash, parent, sibling};
use paper_a_optimized::{
    AuthConfig, AuthPath, BatchWriteError, BatchWriteOpts, CodedArrayConfig, Config, DeltaMode,
    LeafUpdateBatch, MerkleTreeProof, build_coded_params, build_shard_registry_deterministic,
    random_value,
};
use paper_a_optimized::{DefaultHashes, Value256};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::collections::{HashMap, HashSet};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;
use support::setup;

/// Compute a Merkle root from explicit leaf values (full tree).
fn compute_root_from_leaves(
    height: u32,
    defaults: &DefaultHashes,
    leaves: &[Value256],
) -> Value256 {
    let mut level = Vec::with_capacity(leaves.len());
    for (i, value) in leaves.iter().enumerate() {
        level.push(leaf_label(defaults, height, i as u64, *value));
    }
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks_exact(2) {
            next.push(node_hash(pair[0], pair[1]));
        }
        level = next;
    }
    level[0]
}

/// Build a frontier multiproof from multiple authentication paths.
fn build_proof_from_paths(height: u32, root: Value256, paths: &[AuthPath]) -> MerkleTreeProof {
    let mut nodes_map: HashMap<NodeKey, Value256> = HashMap::new();
    for path in paths {
        let mut v = leaf_key(height, path.index as u64);
        for &sib_content in &path.siblings {
            let sib = sibling(v);
            if let Some(existing) = nodes_map.insert(sib, sib_content) {
                assert_eq!(existing, sib_content, "conflicting proof node {:?}", sib);
            }
            v = parent(v);
        }
    }
    let mut nodes: Vec<(NodeKey, Value256)> = nodes_map.into_iter().collect();
    nodes.sort_by_key(|(node_id, _)| *node_id);
    MerkleTreeProof { root, nodes }
}

#[derive(Clone, Copy)]
enum GateRole {
    Slow,
    Fast,
}

#[derive(Clone)]
struct GateServer {
    label: &'static str,
    role: GateRole,
}

#[derive(Clone)]
struct ParallelGateTransport {
    len_gate: Arc<(Mutex<bool>, Condvar)>,
    extend_gate: Arc<(Mutex<bool>, Condvar)>,
}

impl ParallelGateTransport {
    fn wait_for_gate(gate: &Arc<(Mutex<bool>, Condvar)>) {
        let (lock, cvar) = &**gate;
        let mut opened = lock.lock().expect("gate lock");
        while !*opened {
            opened = cvar.wait(opened).expect("gate wait");
        }
    }

    fn open_gate(gate: &Arc<(Mutex<bool>, Condvar)>) {
        let (lock, cvar) = &**gate;
        let mut opened = lock.lock().expect("gate lock");
        *opened = true;
        cvar.notify_all();
    }
}

impl AuthTransport for ParallelGateTransport {
    type Server = GateServer;

    fn server_label(&self, server: &Self::Server) -> String {
        server.label.to_string()
    }

    fn auth_get_len(&self, server: &Self::Server) -> Result<u64, AuthClientError> {
        match server.role {
            GateRole::Slow => Self::wait_for_gate(&self.len_gate),
            GateRole::Fast => Self::open_gate(&self.len_gate),
        }
        Ok(4)
    }

    fn auth_read(
        &self,
        _server: &Self::Server,
        _index: u64,
        _kappa: usize,
    ) -> Result<AuthReadResponse, AuthClientError> {
        Err(AuthClientError::Net("unused auth read".to_string()))
    }

    fn auth_read_subtree(
        &self,
        _server: &Self::Server,
        _start_index: u64,
        _leaf_count: usize,
    ) -> Result<AuthSubtreeReadResponse, AuthClientError> {
        Err(AuthClientError::Net("unused auth subtree read".to_string()))
    }

    fn unauth_multi_read(
        &self,
        _server: &Self::Server,
        _indices: &[u64],
    ) -> Result<Vec<Vec<paper_a_optimized::ShardShare>>, AuthClientError> {
        Err(AuthClientError::Net("unused unauth multi read".to_string()))
    }

    fn auth_write(
        &self,
        _server: &Self::Server,
        _req: &AuthWriteRequest,
    ) -> Result<AuthWriteResponse, AuthClientError> {
        Err(AuthClientError::Net("unused auth write".to_string()))
    }

    fn auth_write_batch(
        &self,
        _server: &Self::Server,
        _req: &AuthBatchWriteRequest,
    ) -> Result<AuthBatchWriteResponse, AuthClientError> {
        Err(AuthClientError::Net("unused auth write batch".to_string()))
    }

    fn auth_extend(&self, server: &Self::Server, _new_len: u64) -> Result<bool, AuthClientError> {
        match server.role {
            GateRole::Slow => Self::wait_for_gate(&self.extend_gate),
            GateRole::Fast => Self::open_gate(&self.extend_gate),
        }
        Ok(true)
    }
}

#[test]
/// Batch updates should yield the same root as recomputing from leaves.
fn batch_root_transition_correctness() {
    let cfg = Config {
        auth: AuthConfig {
            tree_height: 3,
            alloc_leaf_batch: 8,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            kappa: 4,
            shard_count: 8,
            shard_registry: build_shard_registry_deterministic(8),
            party_registry_indices: vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
            seed: 5,
        },
    };
    let mut system = setup(cfg);

    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let mut updates = Vec::new();
    let mut leaves = Vec::with_capacity(system.allocated_leaves);
    let mut paths_all = Vec::with_capacity(system.allocated_leaves);
    for i in 0..system.allocated_leaves {
        let (value, path) = system.read(i).expect("read leaf");
        leaves.push(value);
        paths_all.push(path);
    }
    let mut picked = std::collections::HashSet::new();
    let mut paths = Vec::new();
    while updates.len() < 3 {
        let idx = rng.gen_range(0..system.allocated_leaves);
        if !picked.insert(idx) {
            continue;
        }
        let old = leaves[idx];
        let new = random_value(&mut rng);
        leaves[idx] = new;
        updates.push(LeafUpdateBatch {
            index: idx,
            old,
            new,
        });
        paths.push(paths_all[idx].clone());
    }

    let proof = build_proof_from_paths(system.auth.tree_height as u32, system.store.root, &paths);
    let opts = BatchWriteOpts {
        delta_mode: DeltaMode::Aggregate,
    };
    system
        .write_batch(updates, proof, opts)
        .expect("batch write");

    let expected_root = compute_root_from_leaves(
        system.auth.tree_height as u32,
        &system.auth.defaults,
        &leaves,
    );
    assert_eq!(system.store.root, expected_root);
}

#[test]
/// Batch updates must reject duplicate indices.
fn batch_rejects_duplicate_indices() {
    let cfg = Config {
        auth: AuthConfig {
            tree_height: 2,
            alloc_leaf_batch: 4,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            kappa: 2,
            shard_count: 4,
            shard_registry: build_shard_registry_deterministic(4),
            party_registry_indices: vec![vec![0, 1, 2, 3]],
            seed: 6,
        },
    };
    let mut system = setup(cfg);
    let (old, _) = system.read(1).expect("read");
    let (_, path) = system.read(1).expect("read path");
    let proof = build_proof_from_paths(system.auth.tree_height as u32, system.store.root, &[path]);
    let updates = vec![
        LeafUpdateBatch {
            index: 1,
            old,
            new: old,
        },
        LeafUpdateBatch {
            index: 1,
            old,
            new: old,
        },
    ];
    let err = system
        .write_batch(
            updates,
            proof,
            BatchWriteOpts {
                delta_mode: DeltaMode::Aggregate,
            },
        )
        .expect_err("duplicate should fail");
    assert!(matches!(err, BatchWriteError::DuplicateIndex));
}

#[test]
/// Batch updates must reject mismatched roots.
fn batch_rejects_root_mismatch() {
    let cfg = Config {
        auth: AuthConfig {
            tree_height: 2,
            alloc_leaf_batch: 4,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            kappa: 2,
            shard_count: 4,
            shard_registry: build_shard_registry_deterministic(4),
            party_registry_indices: vec![vec![0, 1, 2, 3]],
            seed: 7,
        },
    };
    let mut system = setup(cfg);
    let (old, _) = system.read(2).expect("read");
    let (_, path) = system.read(2).expect("read path");
    let mut proof =
        build_proof_from_paths(system.auth.tree_height as u32, system.store.root, &[path]);
    proof.root[0] ^= 0x01;
    let updates = vec![LeafUpdateBatch {
        index: 2,
        old,
        new: old,
    }];
    let err = system
        .write_batch(
            updates,
            proof,
            BatchWriteOpts {
                delta_mode: DeltaMode::Aggregate,
            },
        )
        .expect_err("root mismatch should fail");
    assert!(matches!(err, BatchWriteError::RootMismatch));
}

#[test]
/// Batch updates must reject conflicting proof nodes.
fn batch_rejects_conflicting_proof_nodes() {
    let cfg = Config {
        auth: AuthConfig {
            tree_height: 2,
            alloc_leaf_batch: 4,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            kappa: 2,
            shard_count: 4,
            shard_registry: build_shard_registry_deterministic(4),
            party_registry_indices: vec![vec![0, 1, 2, 3]],
            seed: 8,
        },
    };
    let mut system = setup(cfg);
    let (old, _) = system.read(0).expect("read");
    let (_, path) = system.read(0).expect("read path");
    let mut proof =
        build_proof_from_paths(system.auth.tree_height as u32, system.store.root, &[path]);
    if let Some((node, value)) = proof.nodes.first().copied() {
        let mut flipped = value;
        flipped[0] ^= 0x01;
        proof.nodes.push((node, flipped));
    }
    let updates = vec![LeafUpdateBatch {
        index: 0,
        old,
        new: old,
    }];
    let err = system
        .write_batch(
            updates,
            proof,
            BatchWriteOpts {
                delta_mode: DeltaMode::Aggregate,
            },
        )
        .expect_err("conflict should fail");
    assert!(matches!(err, BatchWriteError::ConflictingProofNode));
}

#[test]
/// Batch updates must reject missing frontier nodes.
fn batch_rejects_missing_frontier_node() {
    let cfg = Config {
        auth: AuthConfig {
            tree_height: 3,
            alloc_leaf_batch: 8,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            kappa: 4,
            shard_count: 4,
            shard_registry: build_shard_registry_deterministic(4),
            party_registry_indices: vec![vec![0, 1, 2, 3]],
            seed: 9,
        },
    };
    let mut system = setup(cfg);
    let (old, _) = system.read(3).expect("read");
    let proof = MerkleTreeProof {
        root: system.store.root,
        nodes: Vec::new(),
    };
    let updates = vec![LeafUpdateBatch {
        index: 3,
        old,
        new: old,
    }];
    let err = system
        .write_batch(
            updates,
            proof,
            BatchWriteOpts {
                delta_mode: DeltaMode::Aggregate,
            },
        )
        .expect_err("missing proof should fail");
    assert!(matches!(err, BatchWriteError::InvalidProof));
}

/// Compare sequential writes to batched writes for the same updates.
fn run_batch_vs_sequential(delta_mode: DeltaMode) {
    let cfg = Config {
        auth: AuthConfig {
            tree_height: 7,
            alloc_leaf_batch: 64,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            kappa: 8,
            shard_count: 16,
            shard_registry: build_shard_registry_deterministic(16),
            party_registry_indices: vec![
                vec![0, 1, 2, 3, 4, 5, 6, 7],
                vec![8, 9, 10, 11, 12, 13, 14, 15],
            ],
            seed: 11,
        },
    };
    let mut system_seq = setup(cfg.clone());
    let mut system_batch = setup(cfg);

    system_seq.allocate_segment().expect("allocate segment");
    system_batch.allocate_segment().expect("allocate segment");

    let mut rng = ChaCha20Rng::seed_from_u64(2024);
    let mut updates = Vec::new();
    let mut paths = Vec::new();

    for idx in 0..system_batch.alloc_leaf_batch {
        let (old, path) = system_batch.read(idx).expect("read old");
        let mut new = random_value(&mut rng);
        if new == old {
            new[0] ^= 0x01;
        }
        updates.push(LeafUpdateBatch {
            index: idx,
            old,
            new,
        });
        paths.push(path);
    }

    let mut picked: HashSet<usize> = HashSet::new();
    while picked.len() < 4 {
        let idx = rng.gen_range(system_batch.alloc_leaf_batch..system_batch.allocated_leaves);
        if !picked.insert(idx) {
            continue;
        }
        let (old, path) = system_batch.read(idx).expect("read old");
        let mut new = random_value(&mut rng);
        if new == old {
            new[0] ^= 0x01;
        }
        updates.push(LeafUpdateBatch {
            index: idx,
            old,
            new,
        });
        paths.push(path);
    }

    let proof = build_proof_from_paths(
        system_batch.auth.tree_height as u32,
        system_batch.store.root,
        &paths,
    );

    for update in &updates {
        let (old_current, path) = system_seq.read(update.index).expect("read seq");
        assert_eq!(
            old_current, update.old,
            "sequential old mismatch at {}",
            update.index
        );
        system_seq
            .write(update.index, update.old, update.new, path)
            .expect("sequential write");
    }

    system_batch
        .write_batch(updates.clone(), proof, BatchWriteOpts { delta_mode })
        .expect("batch write");

    assert_eq!(system_seq.store.root, system_batch.store.root, "root");
    assert_eq!(
        system_seq.store.unauth.coded.shard_family_count(),
        system_batch.store.unauth.coded.shard_family_count(),
        "shard count"
    );
    for shard_key in system_seq.store.unauth.coded.shard_keys() {
        let seq_segments = system_seq
            .store
            .unauth
            .coded
            .clone_shard_family_segments(&system_seq.store.unauth.params, shard_key)
            .expect("missing sequential shard");
        let batch_segments = system_batch
            .store
            .unauth
            .coded
            .clone_shard_family_segments(&system_batch.store.unauth.params, shard_key)
            .expect("missing shard");
        assert_eq!(
            seq_segments.len(),
            batch_segments.len(),
            "shard {:?}",
            shard_key
        );
        for (seg_idx, seq_vec) in seq_segments.iter().enumerate() {
            let batch_vec = batch_segments.get(seg_idx).expect("missing segment");
            assert_eq!(seq_vec, batch_vec, "shard {:?} seg {}", shard_key, seg_idx);
        }
    }
    assert_eq!(
        system_seq
            .store
            .unauth
            .coded
            .as_nested_rows_for_tests(&system_seq.store.unauth.params),
        system_batch
            .store
            .unauth
            .coded
            .as_nested_rows_for_tests(&system_batch.store.unauth.params),
        "auxiliary rows"
    );

    for i in 0..system_seq.allocated_leaves {
        let (seq_value, _) = system_seq.read(i).expect("read seq value");
        let (batch_value, _) = system_batch.read(i).expect("read batch value");
        assert_eq!(seq_value, batch_value, "leaf {}", i);
    }
}

#[test]
/// Batch writes should match the equivalent sequence of individual writes.
fn batch_write_matches_sequential_writes() {
    run_batch_vs_sequential(DeltaMode::Aggregate);
    run_batch_vs_sequential(DeltaMode::InterpolateFFT);
}

#[test]
fn auth_client_extend_refreshes_len_and_broadcasts_in_parallel() {
    let len_gate = Arc::new((Mutex::new(false), Condvar::new()));
    let extend_gate = Arc::new((Mutex::new(false), Condvar::new()));
    let transport = ParallelGateTransport {
        len_gate,
        extend_gate,
    };
    let params = build_coded_params(&CodedArrayConfig {
        kappa: 2,
        shard_count: 2,
        shard_registry: build_shard_registry_deterministic(2),
        party_registry_indices: vec![vec![0, 1]],
        seed: 101,
    });
    let auth = AuthParams::new(3);
    let slow = GateServer {
        label: "slow",
        role: GateRole::Slow,
    };
    let fast = GateServer {
        label: "fast",
        role: GateRole::Fast,
    };
    let mut client = AuthClient::new(
        transport,
        params,
        auth,
        paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        0,
        slow.clone(),
        vec![slow, fast],
    );

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(client.extend(5));
    });

    let result = rx
        .recv_timeout(Duration::from_secs(2))
        .expect("auth extend should finish quickly")
        .expect("auth extend");
    assert_eq!(result, ());
}
