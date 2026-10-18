//! Helpers for the unauth benchmark suites and compile-time bench tests.
//!
//! # Overview
//! This module provides bench-only configuration utilities plus deterministic,
//! reproducible sequences used by the current unauth benchmark surface. The
//! helpers expose three conceptually different knobs:
//! - Segment / polynomial sizing: `alloc_leaf_batch` (leaf count per allocation call)
//!   and `segment_len` (values per segment, which defines the polynomial domain
//!   for updates).
//! - FFT grid sizing: `N_grid` (power-of-two grid used for FFT-based
//!   interpolation), equal to `segment_len`.
//! - Shards per party: `shard_count` and `party_registry_indices` control how many shard
//!   vectors each party owns and therefore how much work per-party loops do.
//!
//! # Bench pitfalls
//! - Changing `shard_count` only changes timed work if the benchmark loops over all
//!   shards in `party_registry_indices`. If a benchmark pins work to a single
//!   shard, `shard_count` mostly affects setup and memory, not the timed loop.
//! - FFT timing may appear flat in `s` when `shard_count` is chosen inversely with `s`
//!   (a common bench configuration), because total work per party can stay
//!   roughly constant while `s` changes.
//! - To isolate "one shard" timings, set `party_registry_indices` to a single
//!   index in the bench. To measure "all shards", leave
//!   `party_registry_indices` as the full `0..shard_count` range.
//!
//! When `SANITY=1`, benches emit `[sanity]` lines with `segment_len`, `N_grid`,
//! `s`, `need`, `shard_count`, and party shard counts. These values are derived from the
//! coded-array sizing formulas below and are intended to help cross-check
//! which segment size and shard registry the benchmark is actually timing.
//!
//! # Important distinction
//! `alloc_leaf_batch` is the number of leaves appended per allocation call (the
//! allocation-batch size). `segment_len` is the unauth segment capacity. In the default
//! mapping used by these benches, we pack the induced subtree for each allocation batch
//! into a single segment, so `segment_len = 2 * alloc_leaf_batch`. Other
//! layouts can choose a different `segment_len`, but the helpers here assume
//! this default mapping when deriving sizes.

use paper_a_optimized::merkle::NodeKey;
use paper_a_optimized::{
    AuthConfig, CodedArrayConfig, Config, Fp, build_shard_registry_deterministic,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

pub mod append_only_log_repair;
pub mod generic_bivariate_baseline;
pub mod rs_striped_baseline;
pub mod state_metrics;
pub mod unauth_update_common;
pub mod univariate_segment_baseline;

/// Derived parameter values used by the benchmark harness.
///
/// # Invariants
/// - `segment_len = 2 * alloc_leaf_batch` for the default mapping used here.
/// - `n_grid == segment_len`.
/// - `need = delta + 1`.
///
/// # Complexity
/// O(1) to compute.
///
/// # Bench usage
/// The bench suites primarily call `make_config`, but the derived values here
/// mirror the parameters reported by the current unauth benchmarks.
#[derive(Clone, Debug)]
pub struct DerivedParams {
    /// Values per segment (`2 * alloc_leaf_batch`).
    pub segment_len: usize,
    /// FFT size / polynomial degree bound.
    pub n_grid: usize,
    /// RS decoding radius `(n_grid - 1) / s`.
    pub delta: usize,
    /// Minimum shards needed to decode (`delta + 1`).
    pub need: usize,
}

/// Compute tree/grid sizes and decoding thresholds for a given segment size and s.
///
/// # Inputs
/// - `alloc_leaf_batch`: number of leaves per allocation batch.
/// - `s`: shard vector length (number of ratio points per shard).
///
/// # Preconditions
/// - `alloc_leaf_batch` is a power of two.
/// - `s` is a power of two and divides `n_grid`.
///
/// # Outputs
/// Returns `DerivedParams` with `segment_len`, `n_grid`, `delta`, and `need`.
///
/// # Complexity
/// O(1).
///
/// # Bench usage
/// This mirrors the parameter math used by the current unauth benchmark
/// helpers via `make_config`.
pub fn derive_segment_params(alloc_leaf_batch: usize, s: usize) -> DerivedParams {
    // Segment capacity is the next power-of-two capacity over the induced subtree.
    let segment_len = 2 * alloc_leaf_batch;
    let n_grid = segment_len;
    // RS decoding radius for one shard vector of length `s`.
    let delta = (n_grid - 1) / s;
    // Minimum number of shards needed to decode the segment polynomial.
    let need = delta + 1;
    DerivedParams {
        segment_len,
        n_grid,
        delta,
        need,
    }
}

/// Compute tree/grid sizes and decoding thresholds for a given leaf count and s.
///
/// # Inputs
/// - `l`: number of leaves per allocation batch.
/// - `s`: shard vector length (number of ratio points per shard).
///
/// # Preconditions
/// Same as `derive_segment_params`.
///
/// # Outputs
/// Returns `DerivedParams` with `segment_len`, `n_grid`, `delta`, and `need`.
///
/// # Complexity
/// O(1).
///
/// # Bench usage
/// Provided for convenience and parity with older bench helpers; current
/// benches call `make_config` directly.
pub fn derive_params(l: usize, s: usize) -> DerivedParams {
    derive_segment_params(l, s)
}

/// Choose a shard registry size based on the decoding threshold.
///
/// # Inputs
/// - `need`: minimum shards required to decode.
///
/// # Outputs
/// Returns `g`, the registry size (`shard_count`), as a power of two.
///
/// # Preconditions
/// - `need > 0`.
///
/// # Complexity
/// O(1).
///
/// # Bench usage
/// Used by `make_config`, which is invoked by the current unauth benches.
/// The result is a power of two, clamped to reasonable bounds.
pub fn choose_g(need: usize) -> usize {
    // Lower bound avoids tiny registries that under-state indexing overhead.
    let min_g = 64usize;
    // Upper bound keeps memory/cache effects stable across sweeps.
    let cap_g = 4096usize;
    // Round up to a power of two for deterministic iteration and alignment.
    let mut g = need.max(min_g).next_power_of_two();
    if g > cap_g && need <= cap_g {
        g = cap_g;
    }
    // Ensure `g` still satisfies the decoding requirement if `need` was large.
    if g < need {
        g = need;
    }
    g
}

/// Bench-only wrapper around `fp_zero`.
pub fn fp_zero() -> Fp {
    paper_a_optimized::fp_zero()
}

/// Bench-only wrapper around `fp_one`.
pub fn fp_one() -> Fp {
    paper_a_optimized::fp_one()
}

/// Bench-only wrapper around `fp_pow`.
pub fn fp_pow(base: Fp, exp: u64) -> Fp {
    paper_a_optimized::fp_pow(base, exp)
}

/// Bench-only wrapper around `batch_invert`.
pub fn batch_invert(values: &mut [Fp]) {
    paper_a_optimized::batch_invert(values);
}

/// Bench-only wrapper around `coded_array::params::fft_in_place`.
pub fn fft_in_place(values: &mut [Fp], root: Fp) {
    paper_a_optimized::coded_array::params::fft_in_place(values, root);
}

/// Bench-only wrapper around `coded_array::params::inverse_fft_in_place`.
pub fn inverse_fft_in_place(values: &mut [Fp], root: Fp) {
    paper_a_optimized::coded_array::params::inverse_fft_in_place(values, root);
}

/// Build a deterministic config used by the benchmark suites with explicit segment sizing.
///
/// # Inputs
/// - `h`: total tree height (`tree_height`).
/// - `alloc_leaf_batch`: leaf count per allocation.
/// - `s`: shard vector length (ratio points per shard).
/// - `parties`: number of parties to instantiate.
/// - `seed`: RNG seed for deterministic setup.
///
/// # Outputs
/// Returns a `Config` with derived segment sizes, FFT grid size, shard registry,
/// and per-party shard assignments.
///
/// # Preconditions
/// - `alloc_leaf_batch` is a power of two.
/// - `s` is a power of two and divides `N_grid`.
///
/// # Complexity
/// O(shard_count + parties * shard_count) for registry and per-party index lists.
///
/// # Bench usage
/// The current unauth benchmarks call this helper to keep their parameter math
/// consistent.
/// All parties receive identical shard registry indices to avoid skew in
/// per-party performance measurements.
pub fn make_config(
    h: usize,
    alloc_leaf_batch: usize,
    s: usize,
    parties: usize,
    seed: u64,
) -> Config {
    // Segment capacity is the next power-of-two capacity over the induced subtree.
    let segment_len = 2 * alloc_leaf_batch;
    let n_grid = segment_len;
    // Minimum shards required to decode: delta + 1.
    let need = (n_grid - 1) / s + 1;
    let g = choose_g(need);
    let mut party_registry_indices = Vec::with_capacity(parties);
    let all_indices: Vec<usize> = (0..g).collect();
    for _ in 0..parties {
        // Give each party all shard indices so per-party loops scale with shard_count.
        // Benchmarks that want "one shard" timing should override this list.
        party_registry_indices.push(all_indices.clone());
    }
    let shard_registry = build_shard_registry_deterministic(g, s, n_grid);
    Config {
        auth: AuthConfig {
            tree_height: h,
            alloc_leaf_batch,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            segment_len,
            s,
            shard_count: g,
            shard_registry,
            party_registry_indices,
            seed,
        },
    }
}

/// Build a deterministic sequence of node indices for timing loops.
///
/// # Inputs
/// - `height`: inclusive tree height (`tree_height`).
/// - `leaf_count`: number of allocated leaves.
/// - `len`: number of nodes to produce.
/// - `seed`: RNG seed for repeatability.
///
/// # Outputs
/// Returns a vector of `NodeKey` with `(level, pos)` pairs.
///
/// # Preconditions
/// - `leaf_count > 0`.
/// - `height` fits in `u32` (caller responsibility).
///
/// # Complexity
/// O(len).
///
/// # Bench usage
/// Used by benchmarks that need deterministic node samples for read-path or
/// maintenance-path timing.
///
/// Each node is drawn by sampling a leaf index and a random level, then
/// projecting the leaf index up to that level.
pub fn build_node_sequence(height: u32, leaf_count: usize, len: usize, seed: u64) -> Vec<NodeKey> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut seq = Vec::with_capacity(len);
    for _ in 0..len {
        let leaf = rng.gen_range(0..leaf_count) as u64;
        let level = rng.gen_range(0..=height);
        // Move a leaf index to an ancestor by shifting away `height - level`
        // bits. Guard against shifts >= 64 for a u64 index.
        let shift = height - level;
        let pos = if shift == 0 {
            leaf
        } else if shift >= 64 {
            0
        } else {
            leaf >> shift
        };
        seq.push(NodeKey { level, pos });
    }
    seq
}

/// Build a deterministic sequence of leaf indices for timing loops.
///
/// # Inputs
/// - `leaf_count`: number of allocated leaves.
/// - `len`: number of leaves to produce.
/// - `seed`: RNG seed for repeatability.
///
/// # Outputs
/// Returns a vector of leaf indices in `[0, leaf_count)`.
///
/// # Preconditions
/// - `leaf_count > 0`.
///
/// # Complexity
/// O(len).
///
/// # Bench usage
/// Used by benchmarks that need deterministic logical read/write samples.
pub fn build_leaf_sequence(leaf_count: usize, len: usize, seed: u64) -> Vec<usize> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed ^ 0xA5A5_5A5A);
    let mut seq = Vec::with_capacity(len);
    for _ in 0..len {
        seq.push(rng.gen_range(0..leaf_count));
    }
    seq
}

/// Build a deterministic sequence of segment offsets for timing loops.
///
/// # Inputs
/// - `max`: exclusive upper bound for offsets.
/// - `len`: number of offsets to produce.
/// - `seed`: RNG seed for repeatability.
pub fn build_offset_sequence(max: usize, len: usize, seed: u64) -> Vec<usize> {
    assert!(max > 0);
    let mut rng = ChaCha20Rng::seed_from_u64(seed ^ 0x5A5A_A5A5);
    let mut seq = Vec::with_capacity(len);
    for _ in 0..len {
        seq.push(rng.gen_range(0..max));
    }
    seq
}
