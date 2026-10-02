use paper_a_optimized::net::registry::MockRegistryFile;
use paper_a_optimized::unauth::{ClaimBundle, UnauthStore};
use serde::Serialize;

const FIELD_BYTES: u64 = 32;
const INDEX_BYTES: u64 = 4;

#[derive(Clone, Debug, Default, Serialize)]
pub struct ClaimFootprint {
    pub claim_count: u64,
    pub row_label_count: u64,
    pub metadata_bytes: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct StoreFootprint {
    pub primary_scalars: u64,
    pub auxiliary_scalars: u64,
    pub primary_bytes: u64,
    pub auxiliary_bytes: u64,
    pub claim_count: u64,
    pub row_label_count: u64,
    pub claim_metadata_bytes: u64,
    pub total_bytes: u64,
}

impl StoreFootprint {
    pub fn plus(&self, other: &Self) -> Self {
        Self {
            primary_scalars: self.primary_scalars + other.primary_scalars,
            auxiliary_scalars: self.auxiliary_scalars + other.auxiliary_scalars,
            primary_bytes: self.primary_bytes + other.primary_bytes,
            auxiliary_bytes: self.auxiliary_bytes + other.auxiliary_bytes,
            claim_count: self.claim_count + other.claim_count,
            row_label_count: self.row_label_count + other.row_label_count,
            claim_metadata_bytes: self.claim_metadata_bytes + other.claim_metadata_bytes,
            total_bytes: self.total_bytes + other.total_bytes,
        }
    }
}

pub fn measure_claim_bundles(claims: &[ClaimBundle]) -> ClaimFootprint {
    let claim_count = claims.len() as u64;
    let row_label_count = 0;
    let metadata_bytes = claim_count * (2 * INDEX_BYTES);
    ClaimFootprint {
        claim_count,
        row_label_count,
        metadata_bytes,
    }
}

pub fn measure_store_footprint(store: &UnauthStore) -> StoreFootprint {
    let primary_scalars = store
        .coded
        .as_nested_shards_for_tests(&store.params)
        .values()
        .flat_map(|segments| segments.iter())
        .map(|segment| segment.len() as u64)
        .sum::<u64>();
    let auxiliary_scalars = store
        .coded
        .as_nested_rows_for_tests(&store.params)
        .values()
        .flat_map(|segments| segments.iter())
        .map(|segment| segment.len() as u64)
        .sum::<u64>();
    let primary_bytes = primary_scalars * FIELD_BYTES;
    let auxiliary_bytes = auxiliary_scalars * FIELD_BYTES;
    let claims = store.claim_bundles().unwrap_or_default();
    let claim_fp = measure_claim_bundles(&claims);
    StoreFootprint {
        primary_scalars,
        auxiliary_scalars,
        primary_bytes,
        auxiliary_bytes,
        claim_count: claim_fp.claim_count,
        row_label_count: claim_fp.row_label_count,
        claim_metadata_bytes: claim_fp.metadata_bytes,
        total_bytes: primary_bytes + auxiliary_bytes + claim_fp.metadata_bytes,
    }
}

pub fn measure_store_set(stores: &[UnauthStore]) -> StoreFootprint {
    stores
        .iter()
        .map(measure_store_footprint)
        .fold(StoreFootprint::default(), |acc, item| acc.plus(&item))
}

pub fn registry_json_bytes(file: &MockRegistryFile) -> u64 {
    serde_json::to_vec(file)
        .map(|bytes| bytes.len() as u64)
        .unwrap_or(0)
}
