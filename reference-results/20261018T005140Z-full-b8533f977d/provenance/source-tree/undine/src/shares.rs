//! Core shard share types used across auth/unauth helpers.

use crate::field::Fp;

/// Per-node shard value tied to a registry index.
#[derive(Clone, Debug)]
pub struct ShardShare {
    pub(crate) registry_index: usize,
    pub(crate) value: Option<Fp>,
}

/// Per-row aligned-block reply tied to a maintained row label.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowShare {
    pub(crate) row_label: Fp,
    pub(crate) value: Fp,
}

/// One locally stored primary shard vector for a segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentPrimaryShard {
    pub registry_index: usize,
    pub values: Vec<Fp>,
}

impl RowShare {
    pub fn new(row_label: Fp, value: Fp) -> Self {
        Self { row_label, value }
    }

    pub fn row_label(&self) -> Fp {
        self.row_label
    }

    pub fn value(&self) -> Fp {
        self.value
    }
}

impl ShardShare {
    pub fn new(registry_index: usize, value: Option<Fp>) -> Self {
        Self {
            registry_index,
            value,
        }
    }

    pub fn registry_index(&self) -> usize {
        self.registry_index
    }

    pub fn value(&self) -> Option<Fp> {
        self.value
    }
}
