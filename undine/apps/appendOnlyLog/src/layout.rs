use std::ops::Range;

#[cfg(all(feature = "bulk-updates-1mb", feature = "bulk-updates-2mb"))]
compile_error!("append-only-log-demo bulk-update-size features are mutually exclusive");
#[cfg(all(feature = "bulk-updates-1mb", feature = "bulk-updates-8mb"))]
compile_error!("append-only-log-demo bulk-update-size features are mutually exclusive");
#[cfg(all(feature = "bulk-updates-2mb", feature = "bulk-updates-8mb"))]
compile_error!("append-only-log-demo bulk-update-size features are mutually exclusive");

/// Width of one authenticated core value.
pub const LEAF_BYTES: usize = 32;

/// Number of field elements in one application entry.
pub const FIELD_ELEMENTS_PER_CODEC_GROUP: usize = 1;

/// Number of bytes in one application entry.
///
/// AppendOnlyLog payloads are sequences of canonical scalar-field encodings,
/// not arbitrary byte strings. Each entry is therefore exactly one field
/// element.
pub const PAYLOAD_BYTES_PER_CODEC_GROUP: usize = LEAF_BYTES;

/// Number of field-element updates in one bulk append.
#[cfg(feature = "bulk-updates-1mb")]
pub const BULK_UPDATES_SIZE: usize = 32_768;
#[cfg(all(not(feature = "bulk-updates-1mb"), feature = "bulk-updates-2mb"))]
pub const BULK_UPDATES_SIZE: usize = 65_536;
#[cfg(all(
    not(feature = "bulk-updates-1mb"),
    not(feature = "bulk-updates-2mb"),
    feature = "bulk-updates-8mb"
))]
pub const BULK_UPDATES_SIZE: usize = 262_144;
#[cfg(all(
    not(feature = "bulk-updates-1mb"),
    not(feature = "bulk-updates-2mb"),
    not(feature = "bulk-updates-8mb")
))]
pub const BULK_UPDATES_SIZE: usize = 131_072;

/// Number of codec groups in one bulk append.
pub const BULK_UPDATE_GROUPS: usize = BULK_UPDATES_SIZE / FIELD_ELEMENTS_PER_CODEC_GROUP;

/// Fixed-size append batch used by the demo application.
///
/// This is an application-layer policy, not a core-storage invariant. Feature
/// variants are named by their approximate MiB class; exact byte capacity is
/// `BULK_UPDATES_SIZE * LEAF_BYTES`.
pub const BULK_UPDATES_BYTES: usize = BULK_UPDATE_GROUPS * PAYLOAD_BYTES_PER_CODEC_GROUP;

/// The bundled fixture uses six field elements as a synthetic header prefix.
pub const BATCH_HEADER_BYTES: usize = 6 * LEAF_BYTES;

/// Return the logical bulk-append index for one source-log index.
pub fn bulk_updates_index_for_height(height: u64) -> u64 {
    height
}

/// Return the byte range covered by one source-log index.
pub fn bulk_updates_byte_range_for_height(height: u64) -> Range<usize> {
    let start = height as usize * BULK_UPDATES_BYTES;
    start..start + BULK_UPDATES_BYTES
}

/// Return the leaf range covered by one source-log index.
pub fn leaf_range_for_height(height: u64) -> Range<usize> {
    let start = height as usize * BULK_UPDATES_SIZE;
    start..start + BULK_UPDATES_SIZE
}

/// Return the starting leaf index for one source-log index.
pub fn bulk_updates_start_leaf_index(height: u64) -> u64 {
    height * BULK_UPDATES_SIZE as u64
}

/// Return the number of committed bulk appends implied by a leaf count.
pub fn committed_bulk_update_batches(committed_leaves: u64) -> u64 {
    committed_leaves / BULK_UPDATES_SIZE as u64
}

/// Return the tip height implied by a committed leaf count.
pub fn tip_height(committed_leaves: u64) -> Option<u64> {
    let bulk_update_batch_count = committed_bulk_update_batches(committed_leaves);
    if bulk_update_batch_count == 0 {
        None
    } else {
        Some(bulk_update_batch_count - 1)
    }
}

/// Check whether one height is fully committed under the given leaf count.
pub fn bulk_updates_are_committed(committed_leaves: u64, height: u64) -> bool {
    committed_leaves >= (height + 1) * BULK_UPDATES_SIZE as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bulk_updates_geometry_is_consistent() {
        assert_eq!(LEAF_BYTES, 32);
        assert_eq!(BULK_UPDATES_SIZE % FIELD_ELEMENTS_PER_CODEC_GROUP, 0);
        assert_eq!(
            BULK_UPDATES_BYTES,
            BULK_UPDATE_GROUPS * PAYLOAD_BYTES_PER_CODEC_GROUP
        );
        assert_eq!(bulk_updates_index_for_height(7), 7);
        assert_eq!(
            bulk_updates_start_leaf_index(3),
            3 * BULK_UPDATES_SIZE as u64
        );
        assert_eq!(
            bulk_updates_byte_range_for_height(2),
            2 * BULK_UPDATES_BYTES..3 * BULK_UPDATES_BYTES
        );
        assert_eq!(
            leaf_range_for_height(2),
            2 * BULK_UPDATES_SIZE..3 * BULK_UPDATES_SIZE
        );
    }

    #[test]
    fn tip_tracking_uses_last_full_bulk_updates() {
        assert_eq!(tip_height(0), None);
        assert_eq!(tip_height(1), None);
        assert_eq!(committed_bulk_update_batches(0), 0);
        assert_eq!(committed_bulk_update_batches(1), 0);
        assert_eq!(committed_bulk_update_batches(BULK_UPDATES_SIZE as u64), 1);
        assert!(bulk_updates_are_committed(2 * BULK_UPDATES_SIZE as u64, 1));
        assert!(!bulk_updates_are_committed(BULK_UPDATES_SIZE as u64, 1));
    }
}
