use crate::codec::{CodecError, encode_bulk_updates};
use crate::layout::{BATCH_HEADER_BYTES, LEAF_BYTES};
use paper_a_optimized::{Fp, encode_value};

/// Fixture text embedded in the app crate for the demo batch.
pub const MOCK_BATCH_FIXTURE: &str = include_str!("../fixtures/mock-append-only-log-batch.txt");

/// Errors while parsing the mock append-only log fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MockBatchError {
    MissingField(&'static str),
    InvalidInteger {
        key: &'static str,
        value: String,
    },
    InvalidHex {
        key: &'static str,
        value: String,
    },
    UnknownField(String),
    WrongHexWidth {
        key: &'static str,
        expected_bytes: usize,
        actual_bytes: usize,
    },
}

impl core::fmt::Display for MockBatchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingField(key) => write!(f, "missing fixture field {key}"),
            Self::InvalidInteger { key, value } => {
                write!(f, "invalid integer for {key}: {value}")
            }
            Self::InvalidHex { key, value } => {
                write!(f, "invalid hex for {key}: {value}")
            }
            Self::UnknownField(key) => write!(f, "unknown fixture field {key}"),
            Self::WrongHexWidth {
                key,
                expected_bytes,
                actual_bytes,
            } => write!(
                f,
                "expected {expected_bytes} bytes for {key}, got {actual_bytes}"
            ),
        }
    }
}

impl std::error::Error for MockBatchError {}

/// A deterministic mock append-only log batch for the app-layer demo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockAppendOnlyLogBatch {
    pub height: u64,
    pub format_version: u32,
    pub previous_digest: [u8; 32],
    pub payload_digest: [u8; 32],
    pub timestamp: u32,
    pub policy: u32,
    pub nonce: u32,
    pub payload: Vec<u8>,
}

impl MockAppendOnlyLogBatch {
    /// Parse a text fixture into a synthetic log-like batch.
    pub fn from_fixture_text(input: &str) -> Result<Self, MockBatchError> {
        let mut height = None;
        let mut format_version = None;
        let mut previous_digest = None;
        let mut payload_digest = None;
        let mut timestamp = None;
        let mut policy = None;
        let mut nonce = None;
        let mut payload_text = None;

        for raw_line in input.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| MockBatchError::UnknownField(line.to_owned()))?;
            let key = key.trim();
            let value = value.trim();

            match key {
                "height" => {
                    height = Some(value.parse().map_err(|_| MockBatchError::InvalidInteger {
                        key: "height",
                        value: value.to_owned(),
                    })?)
                }
                "format_version" => {
                    format_version =
                        Some(value.parse().map_err(|_| MockBatchError::InvalidInteger {
                            key: "format_version",
                            value: value.to_owned(),
                        })?)
                }
                "previous_digest_hex" => {
                    previous_digest = Some(parse_hex_32("previous_digest_hex", value)?);
                }
                "payload_digest_hex" => {
                    payload_digest = Some(parse_hex_32("payload_digest_hex", value)?);
                }
                "timestamp" => {
                    timestamp = Some(value.parse().map_err(|_| MockBatchError::InvalidInteger {
                        key: "timestamp",
                        value: value.to_owned(),
                    })?)
                }
                "policy" => {
                    policy = Some(value.parse().map_err(|_| MockBatchError::InvalidInteger {
                        key: "policy",
                        value: value.to_owned(),
                    })?)
                }
                "nonce" => {
                    nonce = Some(value.parse().map_err(|_| MockBatchError::InvalidInteger {
                        key: "nonce",
                        value: value.to_owned(),
                    })?)
                }
                "payload_text" => payload_text = Some(value.to_owned()),
                other => return Err(MockBatchError::UnknownField(other.to_owned())),
            }
        }

        Ok(Self {
            height: height.ok_or(MockBatchError::MissingField("height"))?,
            format_version: format_version.ok_or(MockBatchError::MissingField("format_version"))?,
            previous_digest: previous_digest
                .ok_or(MockBatchError::MissingField("previous_digest_hex"))?,
            payload_digest: payload_digest
                .ok_or(MockBatchError::MissingField("payload_digest_hex"))?,
            timestamp: timestamp.ok_or(MockBatchError::MissingField("timestamp"))?,
            policy: policy.ok_or(MockBatchError::MissingField("policy"))?,
            nonce: nonce.ok_or(MockBatchError::MissingField("nonce"))?,
            payload: payload_text
                .ok_or(MockBatchError::MissingField("payload_text"))?
                .into_bytes(),
        })
    }

    /// Load the bundled mock fixture.
    pub fn load_fixture() -> Result<Self, MockBatchError> {
        Self::from_fixture_text(MOCK_BATCH_FIXTURE)
    }

    /// Build the synthetic field-entry header.
    pub fn header_bytes(&self) -> [u8; BATCH_HEADER_BYTES] {
        let mut header = [0u8; BATCH_HEADER_BYTES];
        header[0..32].copy_from_slice(&encode_value(Fp::from(self.format_version as u64)));
        header[32..64].copy_from_slice(&self.previous_digest);
        header[64..96].copy_from_slice(&self.payload_digest);
        header[96..128].copy_from_slice(&encode_value(Fp::from(self.timestamp as u64)));
        header[128..160].copy_from_slice(&encode_value(Fp::from(self.policy as u64)));
        header[160..192].copy_from_slice(&encode_value(Fp::from(self.nonce as u64)));
        header
    }

    /// Return the synthetic field-entry sequence used by the demo app.
    pub fn batch_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(BATCH_HEADER_BYTES + self.payload.len() * LEAF_BYTES);
        bytes.extend_from_slice(&self.header_bytes());
        for byte in &self.payload {
            bytes.extend_from_slice(&encode_value(Fp::from(*byte as u64)));
        }
        bytes
    }

    /// Return the batch as a fixed-size archival bulk_updates.
    pub fn bulk_updates_bytes(&self) -> Result<Vec<u8>, CodecError> {
        encode_bulk_updates(&self.batch_bytes())
    }
}

fn parse_hex_32(key: &'static str, value: &str) -> Result<[u8; 32], MockBatchError> {
    let bytes = parse_hex_bytes(key, value)?;
    if bytes.len() != 32 {
        return Err(MockBatchError::WrongHexWidth {
            key,
            expected_bytes: 32,
            actual_bytes: bytes.len(),
        });
    }

    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

fn parse_hex_bytes(key: &'static str, value: &str) -> Result<Vec<u8>, MockBatchError> {
    let trimmed = value.strip_prefix("0x").unwrap_or(value);
    if trimmed.len() % 2 != 0 {
        return Err(MockBatchError::InvalidHex {
            key,
            value: value.to_owned(),
        });
    }

    let mut bytes = Vec::with_capacity(trimmed.len() / 2);
    let chars = trimmed.as_bytes();
    for index in (0..chars.len()).step_by(2) {
        let hi = hex_value(chars[index]).ok_or_else(|| MockBatchError::InvalidHex {
            key,
            value: value.to_owned(),
        })?;
        let lo = hex_value(chars[index + 1]).ok_or_else(|| MockBatchError::InvalidHex {
            key,
            value: value.to_owned(),
        })?;
        bytes.push((hi << 4) | lo);
    }
    Ok(bytes)
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{bulk_updates_to_leaves, decode_bulk_updates};
    use crate::layout::{BATCH_HEADER_BYTES, BULK_UPDATES_BYTES, BULK_UPDATES_SIZE};

    #[test]
    fn loads_fixture_and_encodes_bulk_updates() {
        let batch = MockAppendOnlyLogBatch::load_fixture().unwrap();
        assert_eq!(batch.height, 0);
        assert_eq!(batch.header_bytes().len(), BATCH_HEADER_BYTES);

        let bulk_updates = batch.bulk_updates_bytes().unwrap();
        assert_eq!(bulk_updates.len(), BULK_UPDATES_BYTES);
        assert_eq!(&bulk_updates[..BATCH_HEADER_BYTES], &batch.header_bytes());
        assert!(
            bulk_updates[batch.batch_bytes().len()..]
                .iter()
                .all(|b| *b == 0)
        );

        let leaves = bulk_updates_to_leaves(&bulk_updates).unwrap();
        assert_eq!(leaves.len(), BULK_UPDATES_SIZE);
        let rebuilt = crate::codec::leaves_to_bulk_updates(&leaves).unwrap();
        assert_eq!(decode_bulk_updates(&rebuilt).unwrap(), bulk_updates);
    }
}
