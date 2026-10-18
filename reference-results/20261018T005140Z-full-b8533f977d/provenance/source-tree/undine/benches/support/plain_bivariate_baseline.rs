//! Benchmark-only plain-bivariate / full-state baseline.
//!
//! This baseline reuses the array's bivariate encoding but materializes one
//! full-state maintainer with no claim partitioning: it holds the primary
//! column and the auxiliary row of every registry entry. It is therefore a
//! same-substrate, claim-free comparator: useful for measuring what the
//! bivariate encoding gives before the paper's distributed claim discipline
//! and maintainer semantics are added.

use crate::support::state_metrics::{StoreFootprint, measure_store_footprint};
use paper_a_optimized::unauth::{CodedArrayParams, DeltaMode, Index, UnauthStore};
use paper_a_optimized::{Fp, Value256, decode_value};

#[derive(Clone)]
pub struct PlainBivariateMaintainer {
    pub store: UnauthStore,
}

impl PlainBivariateMaintainer {
    pub fn new(params: CodedArrayParams, len: u64, deltas: &[(u64, Fp)]) -> Self {
        let mut store = UnauthStore::new(params);
        store.extend(len).expect("extend full-state store");
        store
            .apply_deltas(deltas, DeltaMode::Aggregate)
            .expect("seed full-state store");
        Self { store }
    }

    pub fn refresh(&mut self, deltas: &[(u64, Fp)]) {
        self.store
            .apply_deltas(deltas, DeltaMode::Aggregate)
            .expect("refresh full-state store");
    }

    pub fn apply_public_rounds(&mut self, rounds: usize, deltas: &[(u64, Fp)]) {
        for _ in 0..rounds {
            self.refresh(deltas);
        }
    }

    pub fn read_point(&self, idx: Index) -> Value256 {
        self.store.read(idx).expect("plain-bivariate point read")
    }

    pub fn read_block(&self, block: Index) -> Vec<Value256> {
        self.store
            .read_block(block)
            .expect("plain-bivariate block read")
    }

    pub fn footprint(&self) -> StoreFootprint {
        measure_store_footprint(&self.store)
    }
}

pub fn decode_point(value: Value256) -> Fp {
    decode_value(value)
}

pub fn coded_store_bytes(store: &UnauthStore) -> u64 {
    measure_store_footprint(store).total_bytes
}
