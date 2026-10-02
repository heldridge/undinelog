use crate::layout::{BULK_UPDATES_BYTES, BULK_UPDATES_SIZE, LEAF_BYTES};
use num_bigint::BigUint;
use paper_a_optimized::Value256;
use std::sync::OnceLock;

const BLS12_381_SCALAR_MODULUS_HEX: &str =
    "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001";

/// Errors returned by the append-only log bulk-update codec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecError {
    /// The supplied update batch exceeds the fixed bulk-update size.
    WrongBulkUpdatesLength { expected: usize, actual: usize },
    /// The supplied payload is not a whole number of field elements.
    WrongEntryAlignment { entry_bytes: usize, actual: usize },
    /// The supplied leaf slice does not match the fixed bulk updates geometry.
    WrongLeafCount { expected: usize, actual: usize },
    /// A decoded field element was not canonical for the scalar field.
    NonCanonicalLeaf { index: usize },
}

impl core::fmt::Display for CodecError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WrongBulkUpdatesLength { expected, actual } => {
                write!(f, "expected bulk-updates length {expected}, got {actual}")
            }
            Self::WrongEntryAlignment {
                entry_bytes,
                actual,
            } => write!(
                f,
                "payload length {actual} is not a multiple of field-entry width {entry_bytes}"
            ),
            Self::WrongLeafCount { expected, actual } => {
                write!(f, "expected {expected} leaves, got {actual}")
            }
            Self::NonCanonicalLeaf { index } => {
                write!(f, "leaf {index} is not a canonical scalar encoding")
            }
        }
    }
}

impl std::error::Error for CodecError {}

/// Encode a batch payload into a fixed-size append-log bulk update.
///
/// The application layer owns the bulk-update policy here; the core library
/// only sees a generic fixed-width leaf array after validation. AppendOnlyLog
/// payloads are sequences of canonical field-element encodings, not arbitrary
/// bytes. Short payloads are padded with zero field elements.
pub fn encode_bulk_updates(entry_bytes: &[u8]) -> Result<Vec<u8>, CodecError> {
    if entry_bytes.len() > BULK_UPDATES_BYTES {
        return Err(CodecError::WrongBulkUpdatesLength {
            expected: BULK_UPDATES_BYTES,
            actual: entry_bytes.len(),
        });
    }
    if entry_bytes.len() % LEAF_BYTES != 0 {
        return Err(CodecError::WrongEntryAlignment {
            entry_bytes: LEAF_BYTES,
            actual: entry_bytes.len(),
        });
    }

    let mut bulk_updates = vec![0u8; BULK_UPDATES_BYTES];
    bulk_updates[..entry_bytes.len()].copy_from_slice(entry_bytes);
    bulk_updates_bytes_as_leaves(&bulk_updates)?;
    Ok(bulk_updates)
}

fn field_modulus() -> &'static BigUint {
    static MODULUS: OnceLock<BigUint> = OnceLock::new();
    MODULUS.get_or_init(|| {
        BigUint::parse_bytes(BLS12_381_SCALAR_MODULUS_HEX.as_bytes(), 16)
            .expect("valid BLS12-381 scalar modulus")
    })
}

fn leaf_to_biguint(leaf: &Value256, index: usize) -> Result<BigUint, CodecError> {
    let value = BigUint::from_bytes_le(leaf);
    if &value >= field_modulus() {
        return Err(CodecError::NonCanonicalLeaf { index });
    }
    Ok(value)
}

/// Encode an exact-size bulk update into field-canonical leaves.
pub fn bulk_updates_bytes_as_leaves(
    bulk_updates_bytes: &[u8],
) -> Result<Vec<Value256>, CodecError> {
    if bulk_updates_bytes.len() != BULK_UPDATES_BYTES {
        return Err(CodecError::WrongBulkUpdatesLength {
            expected: BULK_UPDATES_BYTES,
            actual: bulk_updates_bytes.len(),
        });
    }
    let mut leaves = Vec::with_capacity(BULK_UPDATES_SIZE);
    for (index, chunk) in bulk_updates_bytes.chunks_exact(LEAF_BYTES).enumerate() {
        let mut leaf = [0u8; LEAF_BYTES];
        leaf.copy_from_slice(chunk);
        leaf_to_biguint(&leaf, index)?;
        leaves.push(leaf);
    }
    Ok(leaves)
}

/// Validate and return a fixed-size bulk update.
///
/// This does not strip padding, because the application may want to preserve the
/// exact fixed field-entry batch.
pub fn decode_bulk_updates(bulk_updates_bytes: &[u8]) -> Result<Vec<u8>, CodecError> {
    if bulk_updates_bytes.len() != BULK_UPDATES_BYTES {
        return Err(CodecError::WrongBulkUpdatesLength {
            expected: BULK_UPDATES_BYTES,
            actual: bulk_updates_bytes.len(),
        });
    }

    bulk_updates_bytes_as_leaves(bulk_updates_bytes)?;
    Ok(bulk_updates_bytes.to_vec())
}

/// Split a fixed-size bulk update into field-canonical leaves.
pub fn bulk_updates_to_leaves(bulk_updates_bytes: &[u8]) -> Result<Vec<Value256>, CodecError> {
    if bulk_updates_bytes.len() != BULK_UPDATES_BYTES {
        return Err(CodecError::WrongBulkUpdatesLength {
            expected: BULK_UPDATES_BYTES,
            actual: bulk_updates_bytes.len(),
        });
    }

    bulk_updates_bytes_as_leaves(bulk_updates_bytes)
}

/// Reassemble field-canonical leaves into payload bytes.
pub fn leaves_to_bulk_updates_bytes(leaves: &[Value256]) -> Result<Vec<u8>, CodecError> {
    let mut bulk_updates = Vec::with_capacity(leaves.len() * LEAF_BYTES);
    for (index, leaf) in leaves.iter().enumerate() {
        leaf_to_biguint(leaf, index)?;
        bulk_updates.extend_from_slice(leaf);
    }
    Ok(bulk_updates)
}

/// Reassemble a fixed-size bulk update from field-canonical leaves.
pub fn leaves_to_bulk_updates(leaves: &[Value256]) -> Result<Vec<u8>, CodecError> {
    if leaves.len() != BULK_UPDATES_SIZE {
        return Err(CodecError::WrongLeafCount {
            expected: BULK_UPDATES_SIZE,
            actual: leaves.len(),
        });
    }
    leaves_to_bulk_updates_bytes(leaves)
}

/// Encode field-entry bytes directly into leaves.
pub fn encode_bulk_updates_leaves(entry_bytes: &[u8]) -> Result<Vec<Value256>, CodecError> {
    let mut leaves = Vec::with_capacity(BULK_UPDATES_SIZE);
    encode_bulk_updates_leaves_into(entry_bytes, &mut leaves)?;
    Ok(leaves)
}

/// Encode field-entry bytes directly into a reusable leaf buffer.
pub fn encode_bulk_updates_leaves_into(
    entry_bytes: &[u8],
    out: &mut Vec<Value256>,
) -> Result<(), CodecError> {
    if entry_bytes.len() > BULK_UPDATES_BYTES {
        return Err(CodecError::WrongBulkUpdatesLength {
            expected: BULK_UPDATES_BYTES,
            actual: entry_bytes.len(),
        });
    }
    if entry_bytes.len() % LEAF_BYTES != 0 {
        return Err(CodecError::WrongEntryAlignment {
            entry_bytes: LEAF_BYTES,
            actual: entry_bytes.len(),
        });
    }

    out.clear();
    out.reserve(BULK_UPDATES_SIZE);
    for (index, chunk) in entry_bytes.chunks_exact(LEAF_BYTES).enumerate() {
        let mut leaf = [0u8; LEAF_BYTES];
        leaf.copy_from_slice(chunk);
        leaf_to_biguint(&leaf, index)?;
        out.push(leaf);
    }
    out.resize(BULK_UPDATES_SIZE, [0u8; LEAF_BYTES]);
    Ok(())
}

/// Reassemble leaves and validate the fixed bulk-update geometry.
pub fn decode_bulk_updates_leaves(leaves: &[Value256]) -> Result<Vec<u8>, CodecError> {
    let bulk_updates = leaves_to_bulk_updates(leaves)?;
    decode_bulk_updates(&bulk_updates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use paper_a_optimized::{Fp, encode_value};

    #[test]
    fn bulk_updates_roundtrip_preserves_field_entries_inside_padding() {
        let mut block = Vec::new();
        block.extend_from_slice(&encode_value(Fp::from(7u64)));
        block.extend_from_slice(&encode_value(Fp::from(11u64)));
        let bulk_updates = encode_bulk_updates(&block).unwrap();
        assert_eq!(&bulk_updates[..block.len()], block);
        assert!(bulk_updates[block.len()..].iter().all(|b| *b == 0));
        assert_eq!(decode_bulk_updates(&bulk_updates).unwrap(), bulk_updates);
    }

    #[test]
    fn leaves_roundtrip_bulk_update_entries() {
        let mut block = Vec::new();
        block.extend_from_slice(&encode_value(Fp::from(13u64)));
        block.extend_from_slice(&encode_value(Fp::from(17u64)));
        let bulk_updates = encode_bulk_updates(&block).unwrap();
        let leaves = bulk_updates_to_leaves(&bulk_updates).unwrap();
        assert_eq!(leaves.len(), BULK_UPDATES_SIZE);
        let rebuilt = leaves_to_bulk_updates(&leaves).unwrap();
        assert_eq!(rebuilt, bulk_updates);
    }

    #[test]
    fn canonical_bulk_updates_bytes_roundtrip_through_leaves() {
        let mut bulk_updates = Vec::with_capacity(BULK_UPDATES_BYTES);
        for idx in 0..BULK_UPDATES_SIZE {
            bulk_updates.extend_from_slice(&encode_value(Fp::from(idx as u64)));
        }
        let leaves = bulk_updates_bytes_as_leaves(&bulk_updates).unwrap();
        assert_eq!(leaves.len(), BULK_UPDATES_SIZE);
        for (idx, leaf) in leaves.iter().enumerate() {
            let value = BigUint::from_bytes_le(leaf);
            assert!(&value < field_modulus(), "leaf {idx} is not canonical");
        }
        let rebuilt = leaves_to_bulk_updates(&leaves).unwrap();
        assert_eq!(rebuilt, bulk_updates);
    }

    #[test]
    fn rejects_noncanonical_field_entry() {
        let mut bulk_updates = vec![0u8; BULK_UPDATES_BYTES];
        bulk_updates[..LEAF_BYTES].fill(0xff);
        assert_eq!(
            bulk_updates_bytes_as_leaves(&bulk_updates).unwrap_err(),
            CodecError::NonCanonicalLeaf { index: 0 }
        );
    }

    #[test]
    fn rejects_partial_field_entry() {
        assert_eq!(
            encode_bulk_updates(&[1u8; LEAF_BYTES - 1]).unwrap_err(),
            CodecError::WrongEntryAlignment {
                entry_bytes: LEAF_BYTES,
                actual: LEAF_BYTES - 1
            }
        );
    }
}
