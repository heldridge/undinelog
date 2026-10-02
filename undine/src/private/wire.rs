//! Wire form of a private opening.
//!
//! A PIR query coefficient and a packing-key word both hold two residues of
//! 28 bits, one in each half of a 64-bit word. On the wire a word is seven
//! bytes.

use crate::net::wire::{
    AuthPrivateReadReply, AuthPrivateReadRequest, AuthPrivateReadResponse, NetByteBlob,
};
use crate::private::opening::{PrivateReadReply, PrivateReadRequest, PrivateReadResponse};
use crate::private::pir::PackKeys;

/// Bytes of one word on the wire.
pub const WORD_BYTES: usize = 7;
const RESIDUE_BITS: u32 = 28;
const RESIDUE_MASK: u64 = (1 << RESIDUE_BITS) - 1;

/// Pack words whose two halves are below `2^28` into seven bytes each.
pub fn pack_words(words: &[u64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * WORD_BYTES);
    for &word in words {
        let lo = word & 0xffff_ffff;
        let hi = word >> 32;
        debug_assert!(lo <= RESIDUE_MASK && hi <= RESIDUE_MASK);
        let packed = (lo & RESIDUE_MASK) | ((hi & RESIDUE_MASK) << RESIDUE_BITS);
        bytes.extend_from_slice(&packed.to_le_bytes()[..WORD_BYTES]);
    }
    bytes
}

/// Inverse of [`pack_words`].
pub fn unpack_words(bytes: &[u8]) -> Result<Vec<u64>, String> {
    if bytes.len() % WORD_BYTES != 0 {
        return Err(format!(
            "word blob has {} trailing bytes",
            bytes.len() % WORD_BYTES
        ));
    }
    Ok(bytes
        .chunks_exact(WORD_BYTES)
        .map(|chunk| {
            let mut buf = [0u8; 8];
            buf[..WORD_BYTES].copy_from_slice(chunk);
            let packed = u64::from_le_bytes(buf);
            (packed & RESIDUE_MASK) | ((packed >> RESIDUE_BITS) << 32)
        })
        .collect())
}

/// Wire form of a request.
pub fn request_to_net(request: &PrivateReadRequest) -> AuthPrivateReadRequest {
    AuthPrivateReadRequest {
        leaves: request.leaves,
        kappa: request.kappa as u64,
        segments_per_record: request.segments_per_record as u64,
        query: NetByteBlob(pack_words(&request.query)),
        pack_keys: NetByteBlob(pack_words(&request.pack_keys.words)),
    }
}

/// A request from its wire form.
pub fn request_from_net(request: &AuthPrivateReadRequest) -> Result<PrivateReadRequest, String> {
    Ok(PrivateReadRequest {
        leaves: request.leaves,
        kappa: usize::try_from(request.kappa).map_err(|_| "kappa overflows usize".to_string())?,
        segments_per_record: usize::try_from(request.segments_per_record)
            .map_err(|_| "segments per record overflows usize".to_string())?,
        query: unpack_words(&request.query.0)?,
        pack_keys: PackKeys {
            words: unpack_words(&request.pack_keys.0)?,
        },
    })
}

/// Wire form of a response.
pub fn response_to_net(response: PrivateReadResponse) -> AuthPrivateReadResponse {
    AuthPrivateReadResponse {
        replies: response
            .replies
            .into_iter()
            .map(|reply| AuthPrivateReadReply {
                registry_index: reply.registry_index as u32,
                ciphertexts: reply.response.into_iter().map(NetByteBlob).collect(),
            })
            .collect(),
    }
}

/// A response from its wire form.
pub fn response_from_net(response: AuthPrivateReadResponse) -> PrivateReadResponse {
    PrivateReadResponse {
        replies: response
            .replies
            .into_iter()
            .map(|reply| PrivateReadReply {
                registry_index: reply.registry_index as usize,
                response: reply.ciphertexts.into_iter().map(|blob| blob.0).collect(),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_round_trip_in_seven_bytes() {
        let words = vec![
            0u64,
            RESIDUE_MASK,
            RESIDUE_MASK << 32,
            RESIDUE_MASK | (RESIDUE_MASK << 32),
            0x0123_4567 | (0x0abc_def0 << 32),
        ];
        let bytes = pack_words(&words);
        assert_eq!(bytes.len(), words.len() * WORD_BYTES);
        assert_eq!(unpack_words(&bytes).expect("unpack"), words);
        assert!(unpack_words(&bytes[1..]).is_err());
    }
}
