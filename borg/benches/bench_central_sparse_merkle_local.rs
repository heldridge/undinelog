//! Local centralized sparse-Merkle baseline for authenticated-array comparison.
//!
//! This benchmark intentionally measures only operations that a centralized
//! full-state sparse-Merkle maintainer naturally supports: applying a transition
//! that inserts fresh field elements and producing/verifying a point opening. It
//! is not a repair, onboarding, or kappa-rescale baseline.

use paper_a_optimized::auth::merkle::{
    DefaultHashes, NodeKey, build_default_hashes, child_left, child_right, default_hash,
    leaf_label, node_hash, parent, sibling,
};
use paper_a_optimized::{Value256, random_value};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const BENCH_SEED: u64 = 42;
const DEFAULT_TRIALS: usize = 10;
const APPEND_TREE_HEIGHT: usize = 50;
const READ_TREE_HEIGHT: usize = 50;
const DEFAULT_REWRITE_FIELD_ELEMENTS: &[usize] = &[1_024, 4_096, 32_768, 131_072];
const ACTIVE_FIELD_ELEMENTS: usize = 262_144;
const BULK_UPDATE_FIELD_ELEMENTS: usize = ACTIVE_FIELD_ELEMENTS;
const READ_COMMITTED_BATCHES: usize = 1;
const READ_QUERY_BATCH: usize = 0;
const READ_QUERY_OFFSET: usize = BULK_UPDATE_FIELD_ELEMENTS / 2;

#[derive(Clone)]
struct CentralSparseMerkle {
    height: usize,
    defaults: DefaultHashes,
    nodes: HashMap<NodeKey, Value256>,
    values: HashMap<u64, Value256>,
    len: u64,
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
    rewrite_tree_height: usize,
    read_tree_height: usize,
    rewrite_field_elements: Vec<usize>,
    read_committed_field_elements: usize,
    read_query_index: u64,
    include_combined_open: bool,
}

#[derive(Clone, Debug, Serialize)]
struct BenchEntry {
    operation: String,
    tree_height: usize,
    field_elements: Option<usize>,
    active_field_elements: Option<usize>,
    query_index: Option<u64>,
    proof_siblings: Option<usize>,
    min_index_gap: Option<u64>,
    changed_merkle_nodes: Option<usize>,
    latency_ns: MetricStats,
    open_produce_ns: Option<MetricStats>,
    open_verify_ns: Option<MetricStats>,
    field_elements_per_sec_mean: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
struct BenchResults {
    meta: BenchMeta,
    entries: Vec<BenchEntry>,
}

impl CentralSparseMerkle {
    fn new(height: usize) -> Self {
        Self {
            height,
            defaults: build_default_hashes(height),
            nodes: HashMap::new(),
            values: HashMap::new(),
            len: 0,
        }
    }

    fn default_for(&self, key: NodeKey) -> Value256 {
        default_hash(&self.defaults, key.level as usize)
    }

    fn label_for(&self, key: NodeKey, changed: &BTreeMap<NodeKey, Value256>) -> Value256 {
        changed
            .get(&key)
            .copied()
            .or_else(|| self.nodes.get(&key).copied())
            .unwrap_or_else(|| self.default_for(key))
    }

    fn append_fresh(&mut self, leaves: &[Value256]) {
        let start = self.len;
        let end = start + leaves.len() as u64;
        assert!(
            end <= (1u64 << self.height),
            "central sparse Merkle append exceeds capacity"
        );

        let mut changed = BTreeMap::new();
        for (offset, &value) in leaves.iter().enumerate() {
            let index = start + offset as u64;
            let key = NodeKey {
                level: self.height as u32,
                pos: index,
            };
            let label = leaf_label(&self.defaults, self.height as u32, index, value);
            self.values.insert(index, value);
            self.nodes.insert(key, label);
            changed.insert(key, label);
        }
        self.len = end;

        while let Some(first_key) = changed.keys().next().copied() {
            if first_key.level == 0 {
                break;
            }
            let mut parents = BTreeSet::new();
            for &key in changed.keys() {
                parents.insert(parent(key));
            }

            let mut next = BTreeMap::new();
            for key in parents {
                let left = child_left(key);
                let right = child_right(key);
                let label = node_hash(
                    self.label_for(left, &changed),
                    self.label_for(right, &changed),
                );
                if label == self.default_for(key) {
                    self.nodes.remove(&key);
                } else {
                    self.nodes.insert(key, label);
                }
                next.insert(key, label);
            }
            changed = next;
        }
    }

    fn rewrite_active(&mut self, updates: &[(u64, Value256)]) {
        let mut changed = BTreeMap::new();
        for &(index, value) in updates {
            assert!(index < self.len, "rewrite index outside active prefix");
            let key = NodeKey {
                level: self.height as u32,
                pos: index,
            };
            let label = leaf_label(&self.defaults, self.height as u32, index, value);
            self.values.insert(index, value);
            self.nodes.insert(key, label);
            changed.insert(key, label);
        }

        while let Some(first_key) = changed.keys().next().copied() {
            if first_key.level == 0 {
                break;
            }
            let mut parents = BTreeSet::new();
            for &key in changed.keys() {
                parents.insert(parent(key));
            }

            let mut next = BTreeMap::new();
            for key in parents {
                let left = child_left(key);
                let right = child_right(key);
                let label = node_hash(
                    self.label_for(left, &changed),
                    self.label_for(right, &changed),
                );
                if label == self.default_for(key) {
                    self.nodes.remove(&key);
                } else {
                    self.nodes.insert(key, label);
                }
                next.insert(key, label);
            }
            changed = next;
        }
    }

    fn open(&self, index: u64) -> (Value256, Vec<Value256>, Value256) {
        assert!(index < (1u64 << self.height), "open index out of range");
        let value = self.values.get(&index).copied().unwrap_or([0u8; 32]);
        let mut key = NodeKey {
            level: self.height as u32,
            pos: index,
        };
        let mut siblings = Vec::with_capacity(self.height);
        while key.level > 0 {
            let sib = sibling(key);
            siblings.push(
                self.nodes
                    .get(&sib)
                    .copied()
                    .unwrap_or_else(|| self.default_for(sib)),
            );
            key = parent(key);
        }
        let root_key = NodeKey { level: 0, pos: 0 };
        let root = self
            .nodes
            .get(&root_key)
            .copied()
            .unwrap_or_else(|| self.default_for(root_key));
        (value, siblings, root)
    }
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

fn parse_bool_env(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => panic!("invalid {name} boolean `{raw}`"),
        },
        Err(_) => default,
    }
}

fn random_values(seed: u64, count: usize) -> Vec<Value256> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..count).map(|_| random_value(&mut rng)).collect()
}

fn field_elements_per_sec(count: usize, stats: &MetricStats) -> f64 {
    if stats.mean == 0 {
        return 0.0;
    }
    count as f64 / (stats.mean as f64 / 1_000_000_000.0)
}

fn rewrite_indices(count: usize) -> Vec<u64> {
    assert!(count > 0);
    assert!(
        ACTIVE_FIELD_ELEMENTS >= 2 * count,
        "active prefix must allow non-adjacent rewrite indices"
    );
    let stride = ACTIVE_FIELD_ELEMENTS / count;
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

fn changed_merkle_node_count(indices: &[u64]) -> usize {
    let mut affected = BTreeSet::new();
    for &idx in indices {
        let mut key = NodeKey {
            level: APPEND_TREE_HEIGHT as u32,
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
    affected.len()
}

fn build_active_tree() -> CentralSparseMerkle {
    let leaves = random_values(BENCH_SEED ^ 0xC371_0000, ACTIVE_FIELD_ELEMENTS);
    let mut tree = CentralSparseMerkle::new(APPEND_TREE_HEIGHT);
    tree.append_fresh(&leaves);
    tree
}

fn bench_rewrite(trials: usize, count: usize) -> BenchEntry {
    let base = build_active_tree();
    let indices = rewrite_indices(count);
    let new_values = random_values(BENCH_SEED ^ 0xC371_1000 ^ count as u64, count);
    let updates = indices
        .iter()
        .copied()
        .zip(new_values.iter().copied())
        .collect::<Vec<_>>();
    let mut check = base.clone();
    check.rewrite_active(&updates);
    let expected_root = check.open(0).2;

    let mut samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        let mut tree = base.clone();
        let start = Instant::now();
        tree.rewrite_active(black_box(&updates));
        samples.push(start.elapsed().as_nanos() as u64);
        assert_eq!(tree.len, ACTIVE_FIELD_ELEMENTS as u64);
        assert_eq!(tree.open(0).1.len(), APPEND_TREE_HEIGHT);
        assert_eq!(tree.open(0).2, expected_root);
    }
    let latency_ns = stats_from_samples(samples);
    BenchEntry {
        operation: "rewrite_active_field_elements".to_string(),
        tree_height: APPEND_TREE_HEIGHT,
        field_elements: Some(count),
        active_field_elements: Some(ACTIVE_FIELD_ELEMENTS),
        query_index: None,
        proof_siblings: None,
        min_index_gap: Some(min_gap(&indices)),
        changed_merkle_nodes: Some(changed_merkle_node_count(&indices)),
        field_elements_per_sec_mean: Some(field_elements_per_sec(count, &latency_ns)),
        latency_ns,
        open_produce_ns: None,
        open_verify_ns: None,
    }
}

fn build_read_tree() -> CentralSparseMerkle {
    build_active_tree()
}

fn verify_open_labels(
    height: usize,
    defaults: &DefaultHashes,
    index: u64,
    value: Value256,
    siblings: &[Value256],
    root: Value256,
) -> bool {
    let mut key = NodeKey {
        level: height as u32,
        pos: index,
    };
    let mut cur = leaf_label(defaults, height as u32, index, value);
    for &sib_label in siblings {
        cur = if key.pos % 2 == 0 {
            node_hash(cur, sib_label)
        } else {
            node_hash(sib_label, cur)
        };
        key = parent(key);
    }
    cur == root
}

fn bench_open(trials: usize, include_combined: bool) -> BenchEntry {
    let tree = build_read_tree();
    let query_index = (READ_QUERY_BATCH * BULK_UPDATE_FIELD_ELEMENTS + READ_QUERY_OFFSET) as u64;
    let expected = tree.open(query_index);
    assert_eq!(expected.1.len(), READ_TREE_HEIGHT);
    assert!(verify_open_labels(
        READ_TREE_HEIGHT,
        &tree.defaults,
        query_index,
        expected.0,
        &expected.1,
        expected.2,
    ));

    let mut produce_samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        let start = Instant::now();
        let opened = black_box(tree.open(query_index));
        produce_samples.push(start.elapsed().as_nanos() as u64);
        assert_eq!(opened.1.len(), expected.1.len());
        assert_eq!(opened.2, expected.2);
    }

    let mut verify_samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        let start = Instant::now();
        let verified = black_box(verify_open_labels(
            READ_TREE_HEIGHT,
            &tree.defaults,
            query_index,
            expected.0,
            &expected.1,
            expected.2,
        ));
        verify_samples.push(start.elapsed().as_nanos() as u64);
        assert!(verified);
    }

    let mut combined_samples = Vec::with_capacity(trials);
    if include_combined {
        for _ in 0..trials {
            let start = Instant::now();
            let opened = black_box(tree.open(query_index));
            let verified = verify_open_labels(
                READ_TREE_HEIGHT,
                &tree.defaults,
                query_index,
                opened.0,
                &opened.1,
                expected.2,
            );
            combined_samples.push(start.elapsed().as_nanos() as u64);
            assert!(verified);
            assert_eq!(opened.1.len(), expected.1.len());
            assert_eq!(opened.2, expected.2);
        }
    }
    BenchEntry {
        operation: "point_open".to_string(),
        tree_height: READ_TREE_HEIGHT,
        field_elements: None,
        active_field_elements: Some(READ_COMMITTED_BATCHES * BULK_UPDATE_FIELD_ELEMENTS),
        query_index: Some(query_index),
        proof_siblings: Some(READ_TREE_HEIGHT),
        min_index_gap: None,
        changed_merkle_nodes: None,
        latency_ns: stats_from_samples(combined_samples),
        open_produce_ns: Some(stats_from_samples(produce_samples)),
        open_verify_ns: Some(stats_from_samples(verify_samples)),
        field_elements_per_sec_mean: None,
    }
}

fn main() {
    let trials = parse_trials_env("BENCH_CENTRAL_SPARSE_MERKLE_TRIALS", DEFAULT_TRIALS);
    let rewrite_counts = parse_list_env(
        "BENCH_CENTRAL_SPARSE_MERKLE_REWRITE_FIELD_ELEMENTS",
        DEFAULT_REWRITE_FIELD_ELEMENTS,
    );
    let include_combined_open =
        parse_bool_env("BENCH_CENTRAL_SPARSE_MERKLE_INCLUDE_COMBINED_OPEN", true);

    let mut entries = Vec::new();
    for count in rewrite_counts.iter().copied() {
        entries.push(bench_rewrite(trials, count));
    }
    entries.push(bench_open(trials, include_combined_open));

    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let results = BenchResults {
        meta: BenchMeta {
            seed: BENCH_SEED,
            trials,
            timestamp_unix_ms: timestamp_ms,
            git: git_sha(),
            notes: "centralized full-state sparse-Merkle local baseline: rewrite distinct non-adjacent active field elements and produce and verify one point opening; no coded repair, onboarding, or kappa-rescale operation is modeled".to_string(),
            rewrite_tree_height: APPEND_TREE_HEIGHT,
            read_tree_height: READ_TREE_HEIGHT,
            rewrite_field_elements: rewrite_counts,
            read_committed_field_elements: READ_COMMITTED_BATCHES * BULK_UPDATE_FIELD_ELEMENTS,
            read_query_index: (READ_QUERY_BATCH * BULK_UPDATE_FIELD_ELEMENTS + READ_QUERY_OFFSET)
                as u64,
            include_combined_open,
        },
        entries,
    };

    let out_dir = Path::new("results");
    let _ = fs::create_dir_all(out_dir);
    let path = out_dir.join("bench_central_sparse_merkle_local.json");
    let data =
        serde_json::to_vec_pretty(&results).expect("serialize bench_central_sparse_merkle_local");
    fs::write(path, data).expect("write bench_central_sparse_merkle_local.json");
}
