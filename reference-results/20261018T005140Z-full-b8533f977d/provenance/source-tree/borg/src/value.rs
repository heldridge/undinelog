//! Encoding helpers for fixed-width 256-bit values.
//!
//! # Overview
//! The protocol uses 32-byte payloads as the external representation of field
//! elements. This module defines the canonical encoding/decoding rules:
//! - `Value256` is a 32-byte little-endian array.
//! - Decoding uses modular reduction into the field.
//! - Encoding uses the field's canonical integer representation.
//!
//! # Notes / Pitfalls
//! - `decode_value` accepts any 32-byte array and reduces it modulo the field
//!   prime. It does **not** enforce canonical encoding.
//! - `encode_value` is canonical: decoding the output yields the same element.
//! - Random sampling uses 64 bytes of entropy to avoid bias from truncation.

use crate::field::Fp;
use ark_ff::{BigInteger, PrimeField};
use rand::RngCore;

/// Canonical byte representation for field elements.
pub type Value256 = [u8; 32];

/// Decode a 32-byte little-endian value into the field.
///
/// # Mathematical definition
/// Interprets `bytes` as a little-endian integer `x` and returns `x mod p`.
///
/// # Example
/// ```rust
/// use paper_a_optimized::{decode_value, encode_value, Fp};
/// let fp = Fp::from(42u64);
/// let enc = encode_value(fp);
/// let dec = decode_value(enc);
/// assert_eq!(fp, dec);
/// ```
pub fn decode_value(bytes: Value256) -> Fp {
    Fp::from_le_bytes_mod_order(&bytes)
}

/// Encode a field element into a 32-byte little-endian value.
///
/// # Mathematical definition
/// Converts the field element to its canonical integer representation and
/// serializes it in little-endian order.
///
/// # Invariants
/// - `decode_value(encode_value(x)) == x` for all field elements `x`.
pub fn encode_value(fp: Fp) -> Value256 {
    let mut out = [0u8; 32];
    let mut bytes = fp.into_bigint().to_bytes_le();
    assert!(bytes.len() <= 32);
    bytes.resize(32, 0u8);
    out.copy_from_slice(&bytes);
    out
}

/// Sample a random field element and encode it to 32 bytes.
///
/// # Algorithm
/// Draw 64 random bytes, reduce modulo the field prime, and then encode.
/// Using 64 bytes rather than 32 avoids bias from truncation.
pub fn random_value<R: RngCore>(rng: &mut R) -> Value256 {
    let mut buf = [0u8; 64];
    rng.fill_bytes(&mut buf);
    let fp = Fp::from_le_bytes_mod_order(&buf);
    encode_value(fp)
}
