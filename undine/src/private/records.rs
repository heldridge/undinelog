//! The stored entries of a record.

use crate::Fp;
use crate::field::fp_zero;
use crate::private::layout::PrivateLayout;
use crate::unauth::UnauthStore;

/// Registry indices of the claims whose primary columns this store holds.
pub fn local_claims(store: &UnauthStore) -> Vec<usize> {
    if store.segments() == 0 {
        return Vec::new();
    }
    let mut claims = store
        .handle_read_segment_raw(0)
        .into_iter()
        .map(|column| column.registry_index)
        .collect::<Vec<_>>();
    claims.sort_unstable();
    claims
}

/// The entries of one record of one claim: the primary columns of its
/// segments, in order. Segments beyond the stored ones contribute zeros.
pub fn record_entries(
    store: &UnauthStore,
    layout: &PrivateLayout,
    registry_index: usize,
    record: usize,
) -> Vec<Fp> {
    let kappa = layout.kappa();
    let mut entries = vec![fp_zero(); layout.entries_per_record()];
    for k in 0..layout.segments_per_record() {
        let segment = record * layout.segments_per_record() + k;
        if segment >= store.segments() {
            break;
        }
        if let Some(column) = store
            .handle_read_segment_raw(segment)
            .into_iter()
            .find(|column| column.registry_index == registry_index)
        {
            entries[k * kappa..(k + 1) * kappa].copy_from_slice(&column.values);
        }
    }
    entries
}
