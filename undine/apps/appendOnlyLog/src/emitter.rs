use crate::archive::{ArchiveError, LoadedArchive, load_archive};
use crate::codec::{bulk_updates_bytes_as_leaves, encode_bulk_updates};
use crate::frontier::PreparedBulkUpdatesAppend;
use crate::layout::{
    BULK_UPDATES_BYTES, BULK_UPDATES_SIZE, bulk_updates_start_leaf_index,
    committed_bulk_update_batches,
};
use crate::mock::MockAppendOnlyLogBatch;
use crate::registry::AppendOnlyLogAnchorFile;
use paper_a_optimized::Value256;
use std::path::Path;

pub struct AppendOnlyLogEmitter {
    archive: LoadedArchive,
    prepared_append: PreparedBulkUpdatesAppend,
}

impl AppendOnlyLogEmitter {
    pub fn from_anchor_file(path: &Path) -> Result<Self, ArchiveError> {
        let archive = load_archive(path)?;
        let prepared_append = PreparedBulkUpdatesAppend::new(archive.inner.auth.tree_height)?;
        Ok(Self {
            archive,
            prepared_append,
        })
    }

    pub fn anchor(&self) -> &AppendOnlyLogAnchorFile {
        &self.archive.anchor
    }

    pub fn emit_batch_bytes_at_height(
        &mut self,
        height: u64,
        batch_bytes: &[u8],
    ) -> Result<Value256, ArchiveError> {
        if batch_bytes.len() == BULK_UPDATES_BYTES {
            return self.emit_bulk_updates_bytes_at_height(height, batch_bytes);
        }
        let bulk_updates = encode_bulk_updates(batch_bytes).map_err(ArchiveError::Codec)?;
        self.emit_bulk_updates_bytes_at_height(height, &bulk_updates)
    }

    pub fn emit_bulk_updates_bytes_at_height(
        &mut self,
        height: u64,
        bulk_updates_bytes: &[u8],
    ) -> Result<Value256, ArchiveError> {
        let leaves =
            bulk_updates_bytes_as_leaves(bulk_updates_bytes).map_err(ArchiveError::Codec)?;
        Self::emit_preencoded_bulk_updates_leaves_impl(
            &mut self.archive,
            &mut self.prepared_append,
            height,
            &leaves,
        )
    }

    fn emit_preencoded_bulk_updates_leaves_impl(
        archive: &mut LoadedArchive,
        prepared_append: &mut PreparedBulkUpdatesAppend,
        height: u64,
        leaves: &[Value256],
    ) -> Result<Value256, ArchiveError> {
        if archive.anchor.committed_leaves != 0
            && archive
                .anchor
                .trusted_root()
                .map_err(ArchiveError::Anchor)?
                .is_none()
        {
            return Err(ArchiveError::MissingTrustedRoot);
        }
        if leaves.len() != BULK_UPDATES_SIZE {
            return Err(ArchiveError::InvalidFrontierState(format!(
                "bulk updates append requires {} leaves, got {}",
                BULK_UPDATES_SIZE,
                leaves.len()
            )));
        }

        let expected_height = committed_bulk_update_batches(archive.anchor.committed_leaves);
        if height != expected_height {
            return Err(ArchiveError::HeightMismatch {
                expected: expected_height,
                actual: height,
            });
        }

        let bulk_updates_start = bulk_updates_start_leaf_index(height);
        let bulk_updates_end = bulk_updates_start + BULK_UPDATES_SIZE as u64;
        if archive.anchor.committed_leaves < bulk_updates_start
            || archive.anchor.committed_leaves > bulk_updates_end
        {
            return Err(ArchiveError::HeightMismatch {
                expected: expected_height,
                actual: height,
            });
        }

        let server_len = archive.inner.current_len().map_err(ArchiveError::Auth)?;
        if server_len < archive.anchor.committed_leaves {
            return Err(ArchiveError::CommittedLengthMismatch {
                anchor_committed: archive.anchor.committed_leaves,
                server_reported: server_len,
            });
        }
        if server_len < bulk_updates_end {
            archive
                .inner
                .extend(bulk_updates_end)
                .map_err(ArchiveError::Auth)?;
            let server_len = archive.inner.current_len().map_err(ArchiveError::Auth)?;
            if server_len < bulk_updates_end {
                return Err(ArchiveError::CommittedLengthMismatch {
                    anchor_committed: bulk_updates_end,
                    server_reported: server_len,
                });
            }
        }

        let frontier_before = archive.anchor.committed_leaves;
        let cached_frontier = archive
            .anchor
            .cached_frontier_leaf()
            .map_err(ArchiveError::Anchor)?;
        let new_frontier = prepared_append.append_from_prepared_frontier(
            &mut archive.inner,
            frontier_before,
            cached_frontier.as_ref(),
            leaves,
        )?;
        debug_assert_eq!(frontier_before, bulk_updates_start_leaf_index(height));
        let target_len = frontier_before
            .checked_add(BULK_UPDATES_SIZE as u64)
            .ok_or(ArchiveError::CommittedLengthOverflow)?;
        let root = archive
            .inner
            .trusted_root()
            .ok_or(ArchiveError::MissingTrustedRoot)?;
        archive.anchor.committed_leaves = target_len;
        archive.anchor.set_trusted_root(root);
        archive.anchor.set_cached_frontier_leaf(Some(new_frontier));
        archive
            .anchor
            .save(&archive.anchor_path)
            .map_err(ArchiveError::Anchor)?;

        archive
            .inner
            .trusted_root()
            .ok_or(ArchiveError::MissingTrustedRoot)
    }

    pub fn emit_mock_fixture(
        &mut self,
    ) -> Result<(MockAppendOnlyLogBatch, Value256), ArchiveError> {
        let batch = MockAppendOnlyLogBatch::load_fixture().map_err(ArchiveError::MockBatch)?;
        let bulk_updates = batch.bulk_updates_bytes().map_err(ArchiveError::Codec)?;
        let root = self.emit_bulk_updates_bytes_at_height(batch.height, &bulk_updates)?;
        Ok((batch, root))
    }
}
