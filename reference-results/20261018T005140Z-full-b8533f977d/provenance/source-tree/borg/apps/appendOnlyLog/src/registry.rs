use crate::layout::{BULK_UPDATES_SIZE, tip_height};
use paper_a_optimized::net::registry::MockRegistryFile;
use paper_a_optimized::{AuthPath, Value256};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// App-local source of registry/anchor state for the append-only log demo.
///
/// The demo currently supports a materialized local registry JSON file. The
/// `chain_transactions` variant reserves the application-layer shape for a
/// future source that derives that local registry from a public update stream.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AppendOnlyLogRegistrySource {
    LocalFile {
        registry_path: PathBuf,
    },
    ChainTransactions {
        materialized_registry_path: PathBuf,
        network: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        start_height: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tx_marker: Option<String>,
    },
}

impl AppendOnlyLogRegistrySource {
    pub fn materialized_registry_path(&self) -> &Path {
        match self {
            Self::LocalFile { registry_path } => registry_path.as_path(),
            Self::ChainTransactions {
                materialized_registry_path,
                ..
            } => materialized_registry_path.as_path(),
        }
    }
}

/// Application-local trust anchor for the append-only log demo.
///
/// This is intentionally outside the core crate. The anchor binds:
/// - the materialized registry view used to find servers
/// - the committed leaf count used as the application frontier
/// - the trusted authenticated root used for reads
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AppendOnlyLogAnchorFile {
    pub source: AppendOnlyLogRegistrySource,
    #[serde(default)]
    pub committed_leaves: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trusted_root_hex: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frontier_leaf: Option<SerializedFrontierLeafState>,
}

impl AppendOnlyLogAnchorFile {
    pub fn load(path: &Path) -> Result<Self, AnchorError> {
        let data = fs::read_to_string(path).map_err(|err| AnchorError::Io(err.to_string()))?;
        serde_json::from_str(&data).map_err(|err| AnchorError::Parse(err.to_string()))
    }

    pub fn save(&self, path: &Path) -> Result<(), AnchorError> {
        let data = serde_json::to_string_pretty(self)
            .map_err(|err| AnchorError::Parse(err.to_string()))?;
        fs::write(path, data).map_err(|err| AnchorError::Io(err.to_string()))
    }

    pub fn trusted_root(&self) -> Result<Option<Value256>, AnchorError> {
        self.trusted_root_hex
            .as_deref()
            .map(parse_hex_32)
            .transpose()
    }

    pub fn set_trusted_root(&mut self, root: Value256) {
        self.trusted_root_hex = Some(format_hex_32(root));
    }

    pub fn clear_trusted_root(&mut self) {
        self.trusted_root_hex = None;
    }

    pub fn cached_frontier_leaf(
        &self,
    ) -> Result<Option<AppendOnlyLogFrontierLeafState>, AnchorError> {
        self.frontier_leaf
            .as_ref()
            .map(SerializedFrontierLeafState::decode)
            .transpose()
    }

    pub fn set_cached_frontier_leaf(&mut self, frontier: Option<AppendOnlyLogFrontierLeafState>) {
        self.frontier_leaf = frontier.map(SerializedFrontierLeafState::encode);
    }

    pub fn tip_height(&self) -> Option<u64> {
        tip_height(self.committed_leaves)
    }

    pub fn committed_bulk_update_batches(&self) -> u64 {
        self.committed_leaves / BULK_UPDATES_SIZE as u64
    }

    pub fn resolved_registry_path(&self, anchor_path: &Path) -> PathBuf {
        let candidate = self.source.materialized_registry_path();
        if candidate.is_absolute() {
            candidate.to_path_buf()
        } else {
            anchor_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(candidate)
        }
    }

    pub fn load_registry(&self, anchor_path: &Path) -> Result<MockRegistryFile, AnchorError> {
        MockRegistryFile::load(&self.resolved_registry_path(anchor_path))
            .map_err(|err| AnchorError::Registry(format!("{err:?}")))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnchorError {
    Io(String),
    Parse(String),
    InvalidHex(String),
    Registry(String),
}

#[derive(Clone, Debug)]
pub struct AppendOnlyLogFrontierLeafState {
    pub value: Value256,
    pub path: AuthPath,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SerializedFrontierLeafState {
    pub index: u64,
    pub value_hex: String,
    pub siblings_hex: Vec<String>,
    pub root_hex: String,
}

impl SerializedFrontierLeafState {
    fn encode(frontier: AppendOnlyLogFrontierLeafState) -> Self {
        Self {
            index: frontier.path.index as u64,
            value_hex: format_hex_32(frontier.value),
            siblings_hex: frontier
                .path
                .siblings
                .into_iter()
                .map(format_hex_32)
                .collect(),
            root_hex: format_hex_32(frontier.path.root),
        }
    }

    fn decode(&self) -> Result<AppendOnlyLogFrontierLeafState, AnchorError> {
        let value = parse_hex_32(&self.value_hex)?;
        let siblings = self
            .siblings_hex
            .iter()
            .map(|value| parse_hex_32(value))
            .collect::<Result<Vec<_>, _>>()?;
        let root = parse_hex_32(&self.root_hex)?;
        Ok(AppendOnlyLogFrontierLeafState {
            value,
            path: AuthPath {
                index: usize::try_from(self.index).map_err(|_| {
                    AnchorError::Parse("frontier index overflows usize".to_string())
                })?,
                siblings,
                root,
            },
        })
    }
}

impl core::fmt::Display for AnchorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io(message) => write!(f, "{message}"),
            Self::Parse(message) => write!(f, "{message}"),
            Self::InvalidHex(message) => write!(f, "{message}"),
            Self::Registry(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for AnchorError {}

pub fn parse_hex_32(input: &str) -> Result<Value256, AnchorError> {
    let trimmed = input.strip_prefix("0x").unwrap_or(input);
    if trimmed.len() != 64 {
        return Err(AnchorError::InvalidHex(format!(
            "expected 64 hex chars, got {}",
            trimmed.len()
        )));
    }

    let bytes = trimmed.as_bytes();
    let mut out = [0u8; 32];
    for index in 0..32 {
        let hi = hex_value(bytes[2 * index]).ok_or_else(|| {
            AnchorError::InvalidHex(format!("invalid hex digit {}", bytes[2 * index] as char))
        })?;
        let lo = hex_value(bytes[2 * index + 1]).ok_or_else(|| {
            AnchorError::InvalidHex(format!(
                "invalid hex digit {}",
                bytes[2 * index + 1] as char
            ))
        })?;
        out[31 - index] = (hi << 4) | lo;
    }
    Ok(out)
}

pub fn format_hex_32(value: Value256) -> String {
    let mut out = String::with_capacity(66);
    out.push_str("0x");
    for byte in value.into_iter().rev() {
        out.push(nibble_to_hex(byte >> 4));
        out.push(nibble_to_hex(byte & 0x0f));
    }
    out
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn nibble_to_hex(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        10..=15 => (b'a' + (nibble - 10)) as char,
        _ => '?',
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use paper_a_optimized::AuthPath;

    #[test]
    fn root_hex_roundtrip_matches_client_format() {
        let mut value = [0u8; 32];
        value[0] = 0x34;
        value[31] = 0x12;
        let rendered = format_hex_32(value);
        assert_eq!(
            rendered,
            "0x1200000000000000000000000000000000000000000000000000000000000034"
        );
        assert_eq!(parse_hex_32(&rendered).unwrap(), value);
    }

    #[test]
    fn resolves_relative_registry_path_against_anchor_file() {
        let anchor = AppendOnlyLogAnchorFile {
            source: AppendOnlyLogRegistrySource::LocalFile {
                registry_path: PathBuf::from("registry/demo.json"),
            },
            committed_leaves: BULK_UPDATES_SIZE as u64,
            trusted_root_hex: None,
            frontier_leaf: None,
        };
        let resolved = anchor.resolved_registry_path(Path::new("/tmp/anchor/demo-anchor.json"));
        assert_eq!(resolved, PathBuf::from("/tmp/anchor/registry/demo.json"));
        assert_eq!(anchor.committed_bulk_update_batches(), 1);
        assert_eq!(anchor.tip_height(), Some(0));
    }

    #[test]
    fn cached_frontier_leaf_roundtrip() {
        let frontier = AppendOnlyLogFrontierLeafState {
            value: [9u8; 32],
            path: AuthPath {
                index: 17,
                siblings: vec![[1u8; 32], [2u8; 32], [3u8; 32]],
                root: [4u8; 32],
            },
        };
        let encoded = SerializedFrontierLeafState::encode(frontier.clone());
        let decoded = encoded.decode().expect("decode frontier");
        assert_eq!(decoded.value, frontier.value);
        assert_eq!(decoded.path.index, frontier.path.index);
        assert_eq!(decoded.path.siblings, frontier.path.siblings);
        assert_eq!(decoded.path.root, frontier.path.root);
    }
}
