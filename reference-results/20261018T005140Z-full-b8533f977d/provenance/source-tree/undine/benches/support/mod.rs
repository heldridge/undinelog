//! Helpers for the unauth benchmark suites and compile-time bench tests.
//!
//! # Overview
//! This module provides bench-only configuration utilities plus deterministic,
//! reproducible sequences used by the current unauth benchmark surface. The
//! helpers expose two conceptually different knobs:
//! - Threshold: `kappa`. The segment size `kappa^2` and the column length
//!   `kappa` are derived from it.
//! - Shards per party: `shard_count` and `party_registry_indices` control how many shard
//!   vectors each party owns and therefore how much work per-party loops do.
//!
//! # Bench pitfalls
//! - Changing `shard_count` only changes timed work if the benchmark loops over all
//!   shards in `party_registry_indices`. If a benchmark pins work to a single
//!   shard, `shard_count` mostly affects setup and memory, not the timed loop.
//! - To isolate "one shard" timings, set `party_registry_indices` to a single
//!   index in the bench. To measure "all shards", leave
//!   `party_registry_indices` as the full `0..shard_count` range.
//!
//! # Important distinction
//! `alloc_leaf_batch` is the number of leaves appended per allocation call (the
//! allocation-batch size). The segment capacity is `kappa^2` and is independent
//! of it.

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
#[derive(Clone, Debug)]
pub struct DerivedParams {
    /// Values per segment (`kappa^2`).
    pub segment_len: usize,
    /// Cells per segment; equals `segment_len`.
    pub n_grid: usize,
    /// Minimum shards needed to decode (`kappa`).
    pub need: usize,
}

/// Derive the segment geometry from the threshold.
pub fn derive_params(kappa: usize) -> DerivedParams {
    DerivedParams {
        segment_len: kappa * kappa,
        n_grid: kappa * kappa,
        need: kappa,
    }
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

/// Build a deterministic config used by the benchmark suites.
///
/// # Inputs
/// - `h`: total tree height (`tree_height`).
/// - `alloc_leaf_batch`: leaf count per allocation.
/// - `kappa`: reconstruction threshold; the segment has `kappa^2` cells.
/// - `parties`: number of parties to instantiate.
/// - `seed`: RNG seed for deterministic setup.
///
/// All parties receive identical shard registry indices to avoid skew in
/// per-party performance measurements.
pub fn make_config(
    h: usize,
    alloc_leaf_batch: usize,
    kappa: usize,
    parties: usize,
    seed: u64,
) -> Config {
    let g = choose_g(kappa);
    let all_indices: Vec<usize> = (0..g).collect();
    // Give each party all shard indices so per-party loops scale with shard_count.
    // Benchmarks that want "one shard" timing should override this list.
    let party_registry_indices = vec![all_indices; parties];
    Config {
        auth: AuthConfig {
            tree_height: h,
            alloc_leaf_batch,
            node_linearization: paper_a_optimized::auth::mapping::NodeLinearization::Preorder,
        },
        coded: CodedArrayConfig {
            kappa,
            shard_count: g,
            shard_registry: build_shard_registry_deterministic(g),
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
