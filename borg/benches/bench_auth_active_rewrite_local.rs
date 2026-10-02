//! Local authenticated-array active-rewrite benchmark.
//!
//! This benchmark uses the auth layer directly. It first materializes an active
//! prefix, then measures one maintainer applying a general `WriteBatch` request
//! that rewrites distinct, non-adjacent active positions.

use paper_a_optimized::auth::mapping::NodeLinearization;
use paper_a_optimized::auth::merkle::{
    NodeKey, child_left, child_right, default_hash, leaf_label, node_hash, parent, sibling,
};
use paper_a_optimized::auth::service::{
    AuthBatchWriteRequest, AuthBatchWriteUpdate, AuthFreshSubtreeWriteRequest, AuthRequest,
    AuthResponse, AuthServer, AuthService,
};
use paper_a_optimized::auth::{AuthParams, AuthSubtreeProof, MerkleTreeProof};
use paper_a_optimized::unauth::{ClaimBundle, DeltaMode};
use paper_a_optimized::{
    AuthConfig, CodedArrayConfig, Config, Value256, build_coded_params,
    build_shard_registry_deterministic, random_value,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const BENCH_SEED: u64 = 42;
const DEFAULT_TRIALS: usize = 5;
const TREE_HEIGHT: usize = 50;
const DEFAULT_SEGMENT_LENS: &[usize] = &[128];
const DEFAULT_KAPPAS: &[usize] = &[8];
const DEFAULT_ACTIVE_FIELD_ELEMENTS: usize = 262_144;
const DEFAULT_REWRITE_FIELD_ELEMENTS: &[usize] = &[1_024, 4_096, 32_768, 131_072];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InitMode {
    Dense,
    DefaultActive,
}

impl InitMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::DefaultActive => "default-active",
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct MetricStats {
    mean: u64,
    p50: u64,
    p95: u64,
    trials: u64,
    samples: Vec<u64>,
}

#[derive(Clone, Debug, Serialize)]
struct BenchMeta {
    seed: u64,
    trials: usize,
    timestamp_unix_ms: u64,
    git: Option<String>,
    notes: String,
    init_mode: String,
    active_field_elements: usize,
    segment_lens: Vec<usize>,
    kappas: Vec<usize>,
    rewrite_field_elements: Vec<usize>,
}

#[derive(Clone, Debug, Serialize)]
struct BenchEntry {
    tree_height: usize,
    segment_len: usize,
    n_grid: usize,
    s: usize,
    need: usize,
    active_field_elements: usize,
    rewrite_field_elements: usize,
    min_index_gap: u64,
    changed_merkle_nodes: usize,
    proof_nodes: usize,
    update_ns: MetricStats,
    field_elements_per_sec_mean: f64,
}

#[derive(Clone, Debug, Serialize)]
struct BenchResults {
    meta: BenchMeta,
    entries: Vec<BenchEntry>,
}

fn percentile(sorted: &[u64], pct: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (pct * (sorted.len() - 1)) / 100;
    sorted[idx]
}

fn stats_from_samples(mut samples: Vec<u64>) -> MetricStats {
    if samples.is_empty() {
        return MetricStats {
            mean: 0,
            p50: 0,
            p95: 0,
            trials: 0,
            samples: Vec::new(),
        };
    }
    let sum: u128 = samples.iter().map(|&v| v as u128).sum();
    let mean = (sum / samples.len() as u128) as u64;
    let raw_samples = samples.clone();
    samples.sort_unstable();
    MetricStats {
        mean,
        p50: percentile(&samples, 50),
        p95: percentile(&samples, 95),
        trials: samples.len() as u64,
        samples: raw_samples,
    }
}

fn field_elements_per_sec(count: usize, stats: &MetricStats) -> f64 {
    if stats.mean == 0 {
        return 0.0;
    }
    count as f64 / (stats.mean as f64 / 1_000_000_000.0)
}

fn git_sha() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let sha = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if sha.is_empty() { None } else { Some(sha) }
}

fn parse_trials_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn parse_list_env(name: &str, default: &[usize]) -> Vec<usize> {
    let Ok(raw) = std::env::var(name) else {
        return default.to_vec();
    };
    let values = raw
        .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .unwrap_or_else(|_| panic!("invalid {name} value `{part}`"))
        })
        .collect::<Vec<_>>();
    if values.is_empty() {
        default.to_vec()
    } else {
        values
    }
}

fn parse_usize_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn parse_init_mode_env() -> InitMode {
    match std::env::var("BENCH_AUTH_ACTIVE_REWRITE_INIT_MODE")
        .unwrap_or_else(|_| "dense".to_string())
        .trim()
    {
        "dense" => InitMode::Dense,
        "default-active" | "default_active" => InitMode::DefaultActive,
        other => panic!(
            "invalid BENCH_AUTH_ACTIVE_REWRITE_INIT_MODE `{other}`; use dense or default-active"
        ),
    }
}

fn build_config(segment_len: usize, kappa: usize) -> Config {
    let n_grid = segment_len;
    assert!(segment_len.is_power_of_two(), "segment_len must be N");
    assert_eq!(n_grid % kappa, 0, "kappa must divide N");
    let s = n_grid / kappa;
    let need = (n_grid - 1) / s + 1;
    assert_eq!(need, kappa, "derived helper threshold mismatch");
    Config {
        auth: AuthConfig {
            tree_height: TREE_HEIGHT,
            alloc_leaf_batch: segment_len / 2,
            node_linearization: NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            segment_len,
            s,
            shard_count: need,
            shard_registry: build_shard_registry_deterministic(need, s, n_grid),
            party_registry_indices: (0..need).map(|idx| vec![idx]).collect(),
            seed: BENCH_SEED,
        },
    }
}

fn claim_for_registry_index(
    params: &paper_a_optimized::CodedArrayParams,
    registry_index: usize,
) -> ClaimBundle {
    ClaimBundle::exact_for_registry_index(params, registry_index)
        .expect("exact claim should exist for registry index")
}

fn build_claim_server(cfg: &Config, registry_index: usize) -> AuthServer {
    let params = build_coded_params(&cfg.coded);
    let claim = claim_for_registry_index(&params, registry_index);
    AuthServer::new_with_assignments_and_claims(
        params,
        AuthParams::new(cfg.auth.tree_height),
        cfg.auth.node_linearization,
        0,
        None,
        Some(vec![claim]),
    )
    .expect("build claim server")
}

fn build_values(seed: u64, count: usize) -> Vec<Value256> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..count).map(|_| random_value(&mut rng)).collect()
}

fn build_default_subtree_proof(auth: &AuthParams, leaf_count: usize) -> AuthSubtreeProof {
    assert!(leaf_count.is_power_of_two());
    let subtree_height = leaf_count.trailing_zeros() as usize;
    let root_level = auth.tree_height - subtree_height;
    let mut upper_siblings = Vec::with_capacity(root_level);
    let mut key = NodeKey {
        level: root_level as u32,
        pos: 0,
    };
    for _ in 0..root_level {
        upper_siblings.push(default_hash(&auth.defaults, key.level as usize));
        key = parent(key);
    }
    AuthSubtreeProof {
        start_index: 0,
        leaf_count,
        upper_siblings,
        root: default_hash(&auth.defaults, 0),
    }
}

fn warm_active_prefix(server: &mut AuthServer, auth: &AuthParams, values: &[Value256]) {
    match server.handle_request(AuthRequest::Extend {
        new_leaves: values.len() as u64,
    }) {
        AuthResponse::Extend { ok } => assert!(ok),
        AuthResponse::Error { message } => panic!("extend failed: {message}"),
        other => panic!("unexpected extend response: {other:?}"),
    }
    let req = AuthFreshSubtreeWriteRequest {
        leaves: values.to_vec(),
        proof: build_default_subtree_proof(auth, values.len()),
        delta_mode: DeltaMode::InterpolateFFT,
    };
    match server.handle_request(AuthRequest::WriteFreshSubtree { req }) {
        AuthResponse::WriteFreshSubtree { resp } => {
            assert!(resp.ok, "warm write failed: {:?}", resp.error)
        }
        AuthResponse::Error { message } => panic!("warm write failed: {message}"),
        other => panic!("unexpected warm write response: {other:?}"),
    }
}

fn rewrite_indices(active_field_elements: usize, count: usize) -> Vec<u64> {
    assert!(count > 0);
    assert!(
        active_field_elements >= 2 * count,
        "active prefix must allow non-adjacent rewrite indices"
    );
    let stride = active_field_elements / count;
    assert!(stride >= 2);
    (0..count)
        .map(|idx| (idx * stride + stride / 2) as u64)
        .collect()
}

fn min_gap(indices: &[u64]) -> u64 {
    indices
        .windows(2)
        .map(|window| window[1] - window[0])
        .min()
        .unwrap_or(0)
}

fn build_content_map(
    auth: &AuthParams,
    values: &[Value256],
) -> (Value256, HashMap<NodeKey, Value256>) {
    assert!(values.len().is_power_of_two());
    let height = auth.tree_height as u32;
    let mut content = HashMap::with_capacity(values.len() * 2);
    let mut labels = Vec::with_capacity(values.len());
    for (idx, &value) in values.iter().enumerate() {
        let key = NodeKey {
            level: height,
            pos: idx as u64,
        };
        content.insert(key, value);
        labels.push(leaf_label(&auth.defaults, height, idx as u64, value));
    }

    let mut level = height;
    while labels.len() > 1 {
        let parent_level = level - 1;
        let mut next = Vec::with_capacity(labels.len() / 2);
        for (pos, pair) in labels.chunks_exact(2).enumerate() {
            let label = node_hash(pair[0], pair[1]);
            content.insert(
                NodeKey {
                    level: parent_level,
                    pos: pos as u64,
                },
                label,
            );
            next.push(label);
        }
        labels = next;
        level = parent_level;
    }

    let mut key = NodeKey { level, pos: 0 };
    let mut cur = labels[0];
    while key.level > 0 {
        let sib = sibling(key);
        let sib_label = default_hash(&auth.defaults, sib.level as usize);
        content.insert(sib, sib_label);
        cur = if key.pos % 2 == 0 {
            node_hash(cur, sib_label)
        } else {
            node_hash(sib_label, cur)
        };
        key = parent(key);
        content.insert(key, cur);
    }
    (cur, content)
}

fn content_for(auth: &AuthParams, content: &HashMap<NodeKey, Value256>, key: NodeKey) -> Value256 {
    content.get(&key).copied().unwrap_or_else(|| {
        if key.level == auth.tree_height as u32 {
            [0u8; 32]
        } else {
            default_hash(&auth.defaults, key.level as usize)
        }
    })
}

fn default_content_for(auth: &AuthParams, key: NodeKey) -> Value256 {
    if key.level == auth.tree_height as u32 {
        [0u8; 32]
    } else {
        default_hash(&auth.defaults, key.level as usize)
    }
}

fn build_rewrite_request(
    auth: &AuthParams,
    old_values: &[Value256],
    indices: &[u64],
    new_values: &[Value256],
) -> (AuthBatchWriteRequest, usize) {
    assert_eq!(indices.len(), new_values.len());
    let (root, content) = build_content_map(auth, old_values);
    let height = auth.tree_height as u32;
    let mut affected = HashSet::new();
    for &idx in indices {
        let mut key = NodeKey {
            level: height,
            pos: idx,
        };
        loop {
            affected.insert(key);
            if key.level == 0 {
                break;
            }
            key = parent(key);
        }
    }

    let mut proof_nodes = HashMap::new();
    for &key in &affected {
        if key.level == height {
            continue;
        }
        for child in [child_left(key), child_right(key)] {
            if !affected.contains(&child) {
                proof_nodes.insert(child, content_for(auth, &content, child));
            }
        }
    }
    let mut proof_nodes_vec: Vec<_> = proof_nodes.into_iter().collect();
    proof_nodes_vec.sort_by_key(|(node, _)| *node);

    let updates = indices
        .iter()
        .copied()
        .zip(new_values.iter().copied())
        .map(|(index, new)| AuthBatchWriteUpdate {
            index,
            old: old_values[index as usize],
            new,
        })
        .collect();
    let changed_merkle_nodes = affected.len();
    (
        AuthBatchWriteRequest {
            updates,
            proof: MerkleTreeProof {
                root,
                nodes: proof_nodes_vec,
            },
            delta_mode: DeltaMode::InterpolateFFT,
        },
        changed_merkle_nodes,
    )
}

fn build_default_active_rewrite_request(
    auth: &AuthParams,
    indices: &[u64],
    new_values: &[Value256],
) -> (AuthBatchWriteRequest, usize) {
    assert_eq!(indices.len(), new_values.len());
    let height = auth.tree_height as u32;
    let mut affected = HashSet::new();
    for &idx in indices {
        let mut key = NodeKey {
            level: height,
            pos: idx,
        };
        loop {
            affected.insert(key);
            if key.level == 0 {
                break;
            }
            key = parent(key);
        }
    }

    let mut proof_nodes = HashMap::new();
    for &key in &affected {
        if key.level == height {
            continue;
        }
        for child in [child_left(key), child_right(key)] {
            if !affected.contains(&child) {
                proof_nodes.insert(child, default_content_for(auth, child));
            }
        }
    }
    let mut proof_nodes_vec: Vec<_> = proof_nodes.into_iter().collect();
    proof_nodes_vec.sort_by_key(|(node, _)| *node);

    let updates = indices
        .iter()
        .copied()
        .zip(new_values.iter().copied())
        .map(|(index, new)| AuthBatchWriteUpdate {
            index,
            old: [0u8; 32],
            new,
        })
        .collect();
    let changed_merkle_nodes = affected.len();
    (
        AuthBatchWriteRequest {
            updates,
            proof: MerkleTreeProof {
                root: default_hash(&auth.defaults, 0),
                nodes: proof_nodes_vec,
            },
            delta_mode: DeltaMode::InterpolateFFT,
        },
        changed_merkle_nodes,
    )
}

fn apply_rewrite(server: &mut AuthServer, req: &AuthBatchWriteRequest) -> Value256 {
    match server.handle_request(AuthRequest::WriteBatch { req: req.clone() }) {
        AuthResponse::WriteBatch { resp } => {
            assert!(resp.ok, "rewrite failed: {:?}", resp.error);
            resp.root_new.expect("missing root_new")
        }
        AuthResponse::Error { message } => panic!("rewrite failed: {message}"),
        other => panic!("unexpected rewrite response: {other:?}"),
    }
}

fn bench_entry_from_template(
    cfg: &Config,
    params: &paper_a_optimized::CodedArrayParams,
    template: &AuthServer,
    trials: usize,
    active_field_elements: usize,
    rewrite_count: usize,
    req: AuthBatchWriteRequest,
    changed_merkle_nodes: usize,
    min_index_gap: u64,
) -> BenchEntry {
    assert_eq!(template.store.root, req.proof.root);

    let mut samples = Vec::with_capacity(trials);
    let mut expected_root = None;
    for _ in 0..trials {
        let mut server = template.clone();
        let start = Instant::now();
        let root = black_box(apply_rewrite(&mut server, &req));
        samples.push(start.elapsed().as_nanos() as u64);
        let expected = if let Some(expected_root) = expected_root {
            assert_eq!(root, expected_root);
            expected_root
        } else {
            expected_root = Some(root);
            root
        };
        assert_eq!(server.store.root, expected);
    }
    let update_ns = stats_from_samples(samples);
    BenchEntry {
        tree_height: TREE_HEIGHT,
        segment_len: cfg.coded.segment_len,
        n_grid: params.n_grid,
        s: cfg.coded.s,
        need: params.need,
        active_field_elements,
        rewrite_field_elements: rewrite_count,
        min_index_gap,
        changed_merkle_nodes,
        proof_nodes: req.proof.nodes.len(),
        field_elements_per_sec_mean: field_elements_per_sec(rewrite_count, &update_ns),
        update_ns,
    }
}

fn bench_entry(
    cfg: &Config,
    trials: usize,
    init_mode: InitMode,
    active_field_elements: usize,
    rewrite_count: usize,
) -> BenchEntry {
    let auth = AuthParams::new(cfg.auth.tree_height);
    let params = build_coded_params(&cfg.coded);
    let indices = rewrite_indices(active_field_elements, rewrite_count);
    let new_values = build_values(
        BENCH_SEED ^ 0xA171_0002 ^ rewrite_count as u64,
        rewrite_count,
    );

    let mut template = build_claim_server(cfg, 0);
    let (req, changed_merkle_nodes) = match init_mode {
        InitMode::Dense => {
            let old_values = build_values(BENCH_SEED ^ 0xA171_0001, active_field_elements);
            warm_active_prefix(&mut template, &auth, &old_values);
            build_rewrite_request(&auth, &old_values, &indices, &new_values)
        }
        InitMode::DefaultActive => {
            match template.handle_request(AuthRequest::Extend {
                new_leaves: active_field_elements as u64,
            }) {
                AuthResponse::Extend { ok } => assert!(ok),
                AuthResponse::Error { message } => panic!("extend failed: {message}"),
                other => panic!("unexpected extend response: {other:?}"),
            }
            build_default_active_rewrite_request(&auth, &indices, &new_values)
        }
    };
    bench_entry_from_template(
        cfg,
        &params,
        &template,
        trials,
        active_field_elements,
        rewrite_count,
        req,
        changed_merkle_nodes,
        min_gap(&indices),
    )
}

fn push_default_active_entries(
    entries: &mut Vec<BenchEntry>,
    cfg: &Config,
    trials: usize,
    active_field_elements: usize,
    rewrite_counts: &[usize],
) {
    let auth = AuthParams::new(cfg.auth.tree_height);
    let params = build_coded_params(&cfg.coded);
    let mut template = build_claim_server(cfg, 0);
    match template.handle_request(AuthRequest::Extend {
        new_leaves: active_field_elements as u64,
    }) {
        AuthResponse::Extend { ok } => assert!(ok),
        AuthResponse::Error { message } => panic!("extend failed: {message}"),
        other => panic!("unexpected extend response: {other:?}"),
    }

    for &rewrite_count in rewrite_counts {
        let indices = rewrite_indices(active_field_elements, rewrite_count);
        let new_values = build_values(
            BENCH_SEED ^ 0xA171_0002 ^ rewrite_count as u64,
            rewrite_count,
        );
        let (req, changed_merkle_nodes) =
            build_default_active_rewrite_request(&auth, &indices, &new_values);
        entries.push(bench_entry_from_template(
            cfg,
            &params,
            &template,
            trials,
            active_field_elements,
            rewrite_count,
            req,
            changed_merkle_nodes,
            min_gap(&indices),
        ));
    }
}

fn write_results(path: &Path, results: &BenchResults) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let data =
        serde_json::to_vec_pretty(results).expect("serialize bench_auth_active_rewrite_local");
    fs::write(path, data).expect("write bench_auth_active_rewrite_local.json");
}

fn build_results_meta(
    trials: usize,
    init_mode: InitMode,
    active_field_elements: usize,
    segment_lens: Vec<usize>,
    kappas: Vec<usize>,
    rewrite_counts: Vec<usize>,
) -> BenchMeta {
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    BenchMeta {
        seed: BENCH_SEED,
        trials,
        timestamp_unix_ms: timestamp_ms,
        git: git_sha(),
        notes: "auth-layer-only active-rewrite benchmark: materialize an active prefix, then time one claim-bound maintainer applying a general WriteBatch request over distinct non-adjacent active positions; request construction is outside the timed loop".to_string(),
        init_mode: init_mode.as_str().to_string(),
        active_field_elements,
        segment_lens,
        kappas,
        rewrite_field_elements: rewrite_counts,
    }
}

fn main() {
    let trials = parse_trials_env("BENCH_AUTH_ACTIVE_REWRITE_TRIALS", DEFAULT_TRIALS);
    let init_mode = parse_init_mode_env();
    let active_field_elements = parse_usize_env(
        "BENCH_AUTH_ACTIVE_REWRITE_ACTIVE_FIELD_ELEMENTS",
        DEFAULT_ACTIVE_FIELD_ELEMENTS,
    );
    let segment_lens = parse_list_env(
        "BENCH_AUTH_ACTIVE_REWRITE_SEGMENT_LENS",
        DEFAULT_SEGMENT_LENS,
    );
    let kappas = parse_list_env("BENCH_AUTH_ACTIVE_REWRITE_KAPPAS", DEFAULT_KAPPAS);
    let rewrite_counts = parse_list_env(
        "BENCH_AUTH_ACTIVE_REWRITE_FIELD_ELEMENTS",
        DEFAULT_REWRITE_FIELD_ELEMENTS,
    );
    let out_path = std::env::var("BENCH_AUTH_ACTIVE_REWRITE_OUT")
        .unwrap_or_else(|_| "results/bench_auth_active_rewrite_local.json".to_string());
    let path = Path::new(&out_path);
    let meta = build_results_meta(
        trials,
        init_mode,
        active_field_elements,
        segment_lens.clone(),
        kappas.clone(),
        rewrite_counts.clone(),
    );
    let mut results = BenchResults {
        meta,
        entries: Vec::new(),
    };
    for &segment_len in &segment_lens {
        for &kappa in &kappas {
            let cfg = build_config(segment_len, kappa);
            match init_mode {
                InitMode::Dense => {
                    for &count in &rewrite_counts {
                        results.entries.push(bench_entry(
                            &cfg,
                            trials,
                            init_mode,
                            active_field_elements,
                            count,
                        ));
                        write_results(path, &results);
                    }
                }
                InitMode::DefaultActive => {
                    push_default_active_entries(
                        &mut results.entries,
                        &cfg,
                        trials,
                        active_field_elements,
                        &rewrite_counts,
                    );
                    write_results(path, &results);
                }
            }
        }
    }
    write_results(path, &results);
}
