use crate::archive::{ArchiveError, LoadedArchive, load_archive};
use crate::codec::{leaves_to_bulk_updates, leaves_to_bulk_updates_bytes};
use crate::layout::{
    BULK_UPDATES_BYTES, BULK_UPDATES_SIZE, FIELD_ELEMENTS_PER_CODEC_GROUP,
    PAYLOAD_BYTES_PER_CODEC_GROUP, bulk_updates_are_committed, bulk_updates_start_leaf_index,
};
use crate::registry::AppendOnlyLogAnchorFile;
use paper_a_optimized::{AuthPath, Value256};
use std::path::Path;
use std::thread;

const MAX_BULK_UPDATES_READ_SLICE_LEAVES: usize = 2048;
const MAX_BULK_UPDATES_READ_WORKERS: usize = 1;

pub struct AppendOnlyLogClient {
    archive: LoadedArchive,
}

impl AppendOnlyLogClient {
    pub fn from_anchor_file(path: &Path) -> Result<Self, ArchiveError> {
        Ok(Self {
            archive: load_archive(path)?,
        })
    }

    pub fn anchor(&self) -> &AppendOnlyLogAnchorFile {
        &self.archive.anchor
    }

    pub fn trusted_root(&self) -> Result<Option<[u8; 32]>, ArchiveError> {
        self.archive
            .anchor
            .trusted_root()
            .map_err(ArchiveError::Anchor)
    }

    pub fn server_current_len(&mut self) -> Result<u64, ArchiveError> {
        self.archive.inner.current_len().map_err(ArchiveError::Auth)
    }

    pub fn tip_height(&self) -> Option<u64> {
        self.archive.anchor.tip_height()
    }

    pub fn read_leaf_with_path(
        &mut self,
        index: u64,
    ) -> Result<(Value256, AuthPath), ArchiveError> {
        self.ensure_trusted_root_for_read()?;
        if index >= self.archive.anchor.committed_leaves {
            return Err(ArchiveError::LeafIndexNotCommitted {
                index,
                committed_leaves: self.archive.anchor.committed_leaves,
            });
        }
        self.archive
            .inner
            .read_with_path(index)
            .map_err(ArchiveError::Auth)
    }

    pub fn read_leaf_with_path_correcting(
        &mut self,
        index: u64,
        max_errors: usize,
    ) -> Result<(Value256, AuthPath), ArchiveError> {
        self.ensure_trusted_root_for_read()?;
        if index >= self.archive.anchor.committed_leaves {
            return Err(ArchiveError::LeafIndexNotCommitted {
                index,
                committed_leaves: self.archive.anchor.committed_leaves,
            });
        }
        self.archive
            .inner
            .read_with_path_correcting(index, max_errors)
            .map_err(ArchiveError::Auth)
    }

    pub fn read_bulk_updates(&mut self, height: u64) -> Result<Vec<u8>, ArchiveError> {
        self.ensure_trusted_root_for_read()?;
        if !bulk_updates_are_committed(self.archive.anchor.committed_leaves, height) {
            return Err(ArchiveError::HeightNotCommitted {
                height,
                committed_leaves: self.archive.anchor.committed_leaves,
            });
        }
        let slice_leaves = self.bulk_updates_read_slice_leaves();
        let worker_count = self.bulk_updates_read_worker_count(slice_leaves);
        if worker_count <= 1 {
            return self.read_bulk_updates_serial(height, slice_leaves);
        }
        self.read_bulk_updates_parallel(height, slice_leaves, worker_count)
    }

    pub fn read_bulk_updates_correcting(
        &mut self,
        height: u64,
        max_errors: usize,
    ) -> Result<Vec<u8>, ArchiveError> {
        self.ensure_trusted_root_for_read()?;
        if !bulk_updates_are_committed(self.archive.anchor.committed_leaves, height) {
            return Err(ArchiveError::HeightNotCommitted {
                height,
                committed_leaves: self.archive.anchor.committed_leaves,
            });
        }
        self.read_bulk_updates_serial_correcting(
            height,
            self.bulk_updates_read_slice_leaves(),
            max_errors,
        )
    }

    pub fn read_bulk_updates_range(
        &mut self,
        height: u64,
        offset: usize,
        len: usize,
    ) -> Result<Vec<u8>, ArchiveError> {
        self.ensure_trusted_root_for_read()?;
        if !bulk_updates_are_committed(self.archive.anchor.committed_leaves, height) {
            return Err(ArchiveError::HeightNotCommitted {
                height,
                committed_leaves: self.archive.anchor.committed_leaves,
            });
        }
        let end = offset
            .checked_add(len)
            .ok_or(ArchiveError::InvalidByteRange { offset, len })?;
        if end > BULK_UPDATES_BYTES {
            return Err(ArchiveError::InvalidByteRange { offset, len });
        }
        if len == 0 {
            return Ok(Vec::new());
        }
        if offset == 0 && len == BULK_UPDATES_BYTES {
            return self.read_bulk_updates(height);
        }

        let start_group = offset / PAYLOAD_BYTES_PER_CODEC_GROUP;
        let end_group = end.div_ceil(PAYLOAD_BYTES_PER_CODEC_GROUP);
        let start_leaf = start_group * FIELD_ELEMENTS_PER_CODEC_GROUP;
        let end_leaf = end_group * FIELD_ELEMENTS_PER_CODEC_GROUP;
        let start = bulk_updates_start_leaf_index(height) + start_leaf as u64;
        let leaf_count = end_leaf - start_leaf;
        let leaves = self
            .archive
            .inner
            .read_range(start, leaf_count)
            .map_err(ArchiveError::Auth)?;
        let bytes = leaves_to_bulk_updates_bytes(&leaves).map_err(ArchiveError::Codec)?;
        let local_start = offset % PAYLOAD_BYTES_PER_CODEC_GROUP;
        Ok(bytes[local_start..local_start + len].to_vec())
    }

    pub fn read_bulk_updates_range_correcting(
        &mut self,
        height: u64,
        offset: usize,
        len: usize,
        max_errors: usize,
    ) -> Result<Vec<u8>, ArchiveError> {
        self.ensure_trusted_root_for_read()?;
        if !bulk_updates_are_committed(self.archive.anchor.committed_leaves, height) {
            return Err(ArchiveError::HeightNotCommitted {
                height,
                committed_leaves: self.archive.anchor.committed_leaves,
            });
        }
        let end = offset
            .checked_add(len)
            .ok_or(ArchiveError::InvalidByteRange { offset, len })?;
        if end > BULK_UPDATES_BYTES {
            return Err(ArchiveError::InvalidByteRange { offset, len });
        }
        if len == 0 {
            return Ok(Vec::new());
        }
        if offset == 0 && len == BULK_UPDATES_BYTES {
            return self.read_bulk_updates_correcting(height, max_errors);
        }

        let start_group = offset / PAYLOAD_BYTES_PER_CODEC_GROUP;
        let end_group = end.div_ceil(PAYLOAD_BYTES_PER_CODEC_GROUP);
        let start_leaf = start_group * FIELD_ELEMENTS_PER_CODEC_GROUP;
        let end_leaf = end_group * FIELD_ELEMENTS_PER_CODEC_GROUP;
        let start = bulk_updates_start_leaf_index(height) + start_leaf as u64;
        let leaf_count = end_leaf - start_leaf;
        let leaves = self
            .archive
            .inner
            .read_range_correcting(start, leaf_count, max_errors)
            .map_err(ArchiveError::Auth)?;
        let bytes = leaves_to_bulk_updates_bytes(&leaves).map_err(ArchiveError::Codec)?;
        let local_start = offset % PAYLOAD_BYTES_PER_CODEC_GROUP;
        Ok(bytes[local_start..local_start + len].to_vec())
    }

    fn ensure_trusted_root_for_read(&self) -> Result<(), ArchiveError> {
        if self.archive.anchor.committed_leaves != 0
            && self
                .archive
                .anchor
                .trusted_root()
                .map_err(ArchiveError::Anchor)?
                .is_none()
        {
            return Err(ArchiveError::MissingTrustedRoot);
        }
        Ok(())
    }

    fn bulk_updates_read_slice_leaves(&self) -> usize {
        let segment_len = self.archive.inner.params.segment_len;
        let candidate =
            if segment_len.is_power_of_two() && segment_len <= MAX_BULK_UPDATES_READ_SLICE_LEAVES {
                segment_len
            } else {
                MAX_BULK_UPDATES_READ_SLICE_LEAVES
            };
        let grouped = candidate - (candidate % FIELD_ELEMENTS_PER_CODEC_GROUP);
        grouped.max(FIELD_ELEMENTS_PER_CODEC_GROUP)
    }

    fn bulk_updates_read_worker_count(&self, slice_leaves: usize) -> usize {
        let slice_count = BULK_UPDATES_SIZE.div_ceil(slice_leaves);
        let host_limit = thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(1);
        slice_count
            .min(host_limit)
            .min(MAX_BULK_UPDATES_READ_WORKERS)
            .max(1)
    }

    fn read_bulk_updates_serial(
        &mut self,
        height: u64,
        slice_leaves: usize,
    ) -> Result<Vec<u8>, ArchiveError> {
        let start = bulk_updates_start_leaf_index(height);
        let mut leaves = Vec::with_capacity(BULK_UPDATES_SIZE);
        let mut offset = 0usize;
        while offset < BULK_UPDATES_SIZE {
            let slice_start = start + offset as u64;
            let slice_len = slice_leaves.min(BULK_UPDATES_SIZE - offset);
            let slice_values = self
                .archive
                .inner
                .read_range(slice_start, slice_len)
                .map_err(ArchiveError::Auth)?;
            leaves.extend(slice_values);
            offset += slice_len;
        }
        leaves_to_bulk_updates(&leaves).map_err(ArchiveError::Codec)
    }

    fn read_bulk_updates_serial_correcting(
        &mut self,
        height: u64,
        slice_leaves: usize,
        max_errors: usize,
    ) -> Result<Vec<u8>, ArchiveError> {
        let start = bulk_updates_start_leaf_index(height);
        let mut leaves = Vec::with_capacity(BULK_UPDATES_SIZE);
        let mut offset = 0usize;
        while offset < BULK_UPDATES_SIZE {
            let slice_start = start + offset as u64;
            let slice_len = slice_leaves.min(BULK_UPDATES_SIZE - offset);
            let slice_values = self
                .archive
                .inner
                .read_range_correcting(slice_start, slice_len, max_errors)
                .map_err(ArchiveError::Auth)?;
            leaves.extend(slice_values);
            offset += slice_len;
        }
        leaves_to_bulk_updates(&leaves).map_err(ArchiveError::Codec)
    }

    fn read_bulk_updates_parallel(
        &self,
        height: u64,
        slice_leaves: usize,
        worker_count: usize,
    ) -> Result<Vec<u8>, ArchiveError> {
        let slice_count = BULK_UPDATES_SIZE.div_ceil(slice_leaves);
        let mut handles = Vec::with_capacity(worker_count);
        for worker in 0..worker_count {
            let anchor_path = self.archive.anchor_path.clone();
            handles.push(thread::spawn(
                move || -> Result<Vec<(usize, Vec<u8>)>, ArchiveError> {
                    let mut client = AppendOnlyLogClient::from_anchor_file(&anchor_path)?;
                    let mut slices = Vec::new();
                    let mut slice_index = worker;
                    let slice_bytes = slice_leaves / FIELD_ELEMENTS_PER_CODEC_GROUP
                        * PAYLOAD_BYTES_PER_CODEC_GROUP;
                    while slice_index < slice_count {
                        let offset = slice_index * slice_bytes;
                        let remaining = BULK_UPDATES_BYTES - offset;
                        let len = remaining.min(slice_bytes);
                        let bytes = client.read_bulk_updates_range(height, offset, len)?;
                        slices.push((slice_index, bytes));
                        slice_index += worker_count;
                    }
                    Ok(slices)
                },
            ));
        }

        let mut ordered: Vec<Option<Vec<u8>>> = vec![None; slice_count];
        for handle in handles {
            let slices = handle.join().map_err(|_| {
                ArchiveError::Internal("bulk-updates read worker panicked".to_string())
            })??;
            for (slice_index, bytes) in slices {
                ordered[slice_index] = Some(bytes);
            }
        }

        let mut bulk_updates = Vec::with_capacity(BULK_UPDATES_BYTES);
        for maybe_bytes in ordered {
            let bytes = maybe_bytes.ok_or_else(|| {
                ArchiveError::Internal(
                    "bulk-updates read worker returned an incomplete result".to_string(),
                )
            })?;
            bulk_updates.extend_from_slice(&bytes);
        }
        Ok(bulk_updates)
    }
}
