//! Wire types and framing helpers for binary TCP RPCs.

use crate::DeltaMode;
use crate::field::Fp;
use crate::value::{Value256, decode_value, encode_value};
use base64::Engine as _;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::io::{self, Read, Write};
use std::net::TcpStream;

/// Maximum allowed frame length in bytes.
///
/// Authenticated subtree replies can exceed 8 MiB once one server owns many
/// claims and returns one large chunk in a single RPC, so the default budget
/// must leave headroom for that legitimate case.
pub const MAX_FRAME_LEN: usize = 16 * 1024 * 1024;

/// Field element wrapper encoded as canonical human-readable hex for JSON and
/// raw bytes for binary transport.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NetFp(pub Fp);

impl NetFp {
    /// Convert from a 32-byte little-endian encoding.
    pub fn from_bytes_le(bytes: [u8; 32]) -> Self {
        NetFp(decode_value(bytes))
    }

    /// Convert to a 32-byte little-endian encoding.
    pub fn to_bytes_le(self) -> [u8; 32] {
        encode_value(self.0)
    }
}

impl Serialize for NetFp {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bytes = self.to_bytes_le();
        if serializer.is_human_readable() {
            let mut out = String::with_capacity(64);
            for byte in bytes.into_iter().rev() {
                out.push(hex_digit(byte >> 4));
                out.push(hex_digit(byte & 0x0f));
            }
            serializer.serialize_str(&out)
        } else {
            bytes.serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for NetFp {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            let s = String::deserialize(deserializer)?;
            let bytes = parse_hex_32(&s).map_err(serde::de::Error::custom)?;
            Ok(NetFp::from_bytes_le(bytes))
        } else {
            let bytes = <[u8; 32]>::deserialize(deserializer)?;
            Ok(NetFp::from_bytes_le(bytes))
        }
    }
}

/// Encoded 32-byte value wrapper for human-readable JSON and raw bytes for
/// binary transport.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NetValue(pub Value256);

impl NetValue {
    /// Convert from a 32-byte little-endian encoding.
    pub fn from_bytes_le(bytes: [u8; 32]) -> Self {
        NetValue(bytes)
    }

    /// Convert to a 32-byte little-endian encoding.
    pub fn to_bytes_le(self) -> [u8; 32] {
        self.0
    }
}

impl Serialize for NetValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bytes = self.to_bytes_le();
        if serializer.is_human_readable() {
            let mut out = String::with_capacity(64);
            for byte in bytes.into_iter().rev() {
                out.push(hex_digit(byte >> 4));
                out.push(hex_digit(byte & 0x0f));
            }
            serializer.serialize_str(&out)
        } else {
            bytes.serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for NetValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            let s = String::deserialize(deserializer)?;
            let bytes = parse_hex_32(&s).map_err(serde::de::Error::custom)?;
            Ok(NetValue::from_bytes_le(bytes))
        } else {
            let bytes = <[u8; 32]>::deserialize(deserializer)?;
            Ok(NetValue::from_bytes_le(bytes))
        }
    }
}

/// Compact transport wrapper for a vector of 32-byte values.
///
/// Human-readable serializers use base64 so large blobs stay compact in JSON;
/// binary serializers emit the raw packed bytes directly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetValueBlob(pub Vec<Value256>);

impl Serialize for NetValueBlob {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bytes = pack_value_blob_bytes(&self.0);
        if serializer.is_human_readable() {
            let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
            serializer.serialize_str(&encoded)
        } else {
            serializer.serialize_bytes(&bytes)
        }
    }
}

impl<'de> Deserialize<'de> for NetValueBlob {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes = if deserializer.is_human_readable() {
            let s = String::deserialize(deserializer)?;
            base64::engine::general_purpose::STANDARD
                .decode(s.as_bytes())
                .map_err(serde::de::Error::custom)?
        } else {
            Vec::<u8>::deserialize(deserializer)?
        };
        unpack_value_blob_bytes(&bytes)
            .map(NetValueBlob)
            .map_err(serde::de::Error::custom)
    }
}

fn pack_value_blob_bytes(values: &[Value256]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 32);
    for value in values {
        bytes.extend_from_slice(value);
    }
    bytes
}

fn unpack_value_blob_bytes(bytes: &[u8]) -> Result<Vec<Value256>, String> {
    if bytes.len() % 32 != 0 {
        return Err(format!(
            "decoded value blob has {} trailing bytes",
            bytes.len() % 32
        ));
    }
    let mut values = Vec::with_capacity(bytes.len() / 32);
    for chunk in bytes.chunks_exact(32) {
        let mut value = [0u8; 32];
        value.copy_from_slice(chunk);
        values.push(value);
    }
    Ok(values)
}

/// Compact transport wrapper for raw bytes.
///
/// Human-readable serializers use base64; binary serializers emit the bytes
/// directly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetByteBlob(pub Vec<u8>);

impl Serialize for NetByteBlob {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            let encoded = base64::engine::general_purpose::STANDARD.encode(&self.0);
            serializer.serialize_str(&encoded)
        } else {
            serializer.serialize_bytes(&self.0)
        }
    }
}

impl<'de> Deserialize<'de> for NetByteBlob {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes = if deserializer.is_human_readable() {
            let s = String::deserialize(deserializer)?;
            base64::engine::general_purpose::STANDARD
                .decode(s.as_bytes())
                .map_err(serde::de::Error::custom)?
        } else {
            Vec::<u8>::deserialize(deserializer)?
        };
        Ok(NetByteBlob(bytes))
    }
}

/// Compact transport wrapper for a vector of field elements.
///
/// Human-readable serializers use base64 so large blobs stay compact in JSON;
/// binary serializers emit the raw packed bytes directly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetFpBlob(pub Vec<Fp>);

impl Serialize for NetFpBlob {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bytes = pack_fp_blob_bytes(&self.0);
        if serializer.is_human_readable() {
            let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
            serializer.serialize_str(&encoded)
        } else {
            serializer.serialize_bytes(&bytes)
        }
    }
}

impl<'de> Deserialize<'de> for NetFpBlob {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes = if deserializer.is_human_readable() {
            let s = String::deserialize(deserializer)?;
            base64::engine::general_purpose::STANDARD
                .decode(s.as_bytes())
                .map_err(serde::de::Error::custom)?
        } else {
            Vec::<u8>::deserialize(deserializer)?
        };
        unpack_fp_blob_bytes(&bytes)
            .map(NetFpBlob)
            .map_err(serde::de::Error::custom)
    }
}

fn pack_fp_blob_bytes(values: &[Fp]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 32);
    for value in values {
        bytes.extend_from_slice(&encode_value(*value));
    }
    bytes
}

fn unpack_fp_blob_bytes(bytes: &[u8]) -> Result<Vec<Fp>, String> {
    if bytes.len() % 32 != 0 {
        return Err(format!(
            "decoded field blob has {} trailing bytes",
            bytes.len() % 32
        ));
    }
    let mut values = Vec::with_capacity(bytes.len() / 32);
    for chunk in bytes.chunks_exact(32) {
        let mut value = [0u8; 32];
        value.copy_from_slice(chunk);
        values.push(decode_value(value));
    }
    Ok(values)
}

/// Compact transport wrapper for optional field elements in canonical order.
///
/// The binary layout is:
///
/// - 8 little-endian bytes: slot count
/// - `ceil(count / 8)` bitmap bytes, least-significant bit first
/// - one 32-byte field element for each present slot, in slot order
///
/// This preserves sparse/absent local replies without repeating registry
/// indices or row labels in every response.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetMaybeFpBlob(pub Vec<Option<Fp>>);

impl Serialize for NetMaybeFpBlob {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bytes = pack_maybe_fp_blob_bytes(&self.0);
        if serializer.is_human_readable() {
            let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
            serializer.serialize_str(&encoded)
        } else {
            serializer.serialize_bytes(&bytes)
        }
    }
}

impl<'de> Deserialize<'de> for NetMaybeFpBlob {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes = if deserializer.is_human_readable() {
            let s = String::deserialize(deserializer)?;
            base64::engine::general_purpose::STANDARD
                .decode(s.as_bytes())
                .map_err(serde::de::Error::custom)?
        } else {
            Vec::<u8>::deserialize(deserializer)?
        };
        unpack_maybe_fp_blob_bytes(&bytes)
            .map(NetMaybeFpBlob)
            .map_err(serde::de::Error::custom)
    }
}

fn pack_maybe_fp_blob_bytes(values: &[Option<Fp>]) -> Vec<u8> {
    let bitmap_len = values.len().div_ceil(8);
    let present_count = values.iter().filter(|value| value.is_some()).count();
    let mut bytes = Vec::with_capacity(8 + bitmap_len + present_count * 32);
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    let bitmap_start = bytes.len();
    bytes.resize(bitmap_start + bitmap_len, 0);
    for (slot, value) in values.iter().enumerate() {
        if let Some(value) = value {
            bytes[bitmap_start + slot / 8] |= 1u8 << (slot % 8);
            bytes.extend_from_slice(&encode_value(*value));
        }
    }
    bytes
}

fn unpack_maybe_fp_blob_bytes(bytes: &[u8]) -> Result<Vec<Option<Fp>>, String> {
    if bytes.len() < 8 {
        return Err("optional field blob is missing its slot count".to_string());
    }
    let mut count_bytes = [0u8; 8];
    count_bytes.copy_from_slice(&bytes[..8]);
    let count = u64::from_le_bytes(count_bytes) as usize;
    let bitmap_len = count.div_ceil(8);
    if bytes.len() < 8 + bitmap_len {
        return Err("optional field blob is missing bitmap bytes".to_string());
    }
    let bitmap = &bytes[8..8 + bitmap_len];
    let mut offset = 8 + bitmap_len;
    let mut values = Vec::with_capacity(count);
    for slot in 0..count {
        if bitmap[slot / 8] & (1u8 << (slot % 8)) == 0 {
            values.push(None);
            continue;
        }
        if offset + 32 > bytes.len() {
            return Err("optional field blob ended inside a field element".to_string());
        }
        let mut value = [0u8; 32];
        value.copy_from_slice(&bytes[offset..offset + 32]);
        values.push(Some(decode_value(value)));
        offset += 32;
    }
    if offset != bytes.len() {
        return Err(format!(
            "optional field blob has {} trailing bytes",
            bytes.len() - offset
        ));
    }
    Ok(values)
}

fn hex_digit(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        10..=15 => (b'a' + (nibble - 10)) as char,
        _ => '?',
    }
}

fn parse_hex_32(input: &str) -> Result<[u8; 32], String> {
    let s = input.strip_prefix("0x").unwrap_or(input);
    if s.len() != 64 {
        return Err(format!("expected 64 hex chars for NetFp, got {}", s.len()));
    }
    let mut out = [0u8; 32];
    let bytes = s.as_bytes();
    for i in 0..32 {
        let hi = from_hex_digit(bytes[2 * i])?;
        let lo = from_hex_digit(bytes[2 * i + 1])?;
        out[31 - i] = (hi << 4) | lo;
    }
    Ok(out)
}

fn from_hex_digit(b: u8) -> Result<u8, String> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(format!("invalid hex digit {}", b as char)),
    }
}

/// Network-friendly delta mode enum.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NetDeltaMode {
    Aggregate,
    InterpolateFft,
}

/// Network-friendly node identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NetNodeKey {
    pub level: u32,
    pub pos: u64,
}

/// Network-friendly authentication path.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetAuthPath {
    pub index: u64,
    pub siblings: Vec<NetValue>,
    pub root: NetValue,
}

impl From<DeltaMode> for NetDeltaMode {
    fn from(mode: DeltaMode) -> Self {
        match mode {
            DeltaMode::Aggregate => NetDeltaMode::Aggregate,
            DeltaMode::InterpolateFFT => NetDeltaMode::InterpolateFft,
        }
    }
}

impl From<NetDeltaMode> for DeltaMode {
    fn from(mode: NetDeltaMode) -> Self {
        match mode {
            NetDeltaMode::Aggregate => DeltaMode::Aggregate,
            NetDeltaMode::InterpolateFft => DeltaMode::InterpolateFFT,
        }
    }
}

/// Read request for one direct shard value.
///
/// `alpha` is the W-axis point of the requested cell in segment `seg`; the
/// reply is the stored primary-column entry at that point.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReadShardRequest {
    pub seg: u64,
    pub alpha: NetFp,
    pub registry_index: u32,
}

/// Read response for one direct shard value.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReadShardResponse {
    pub value: Option<NetFp>,
}

/// Ordered shard values in the server's canonical owned-index order.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetShardValueReply {
    pub values: NetMaybeFpBlob,
}

/// One multi-`z` row reply sharing one row-label vector across all batches.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetRowBatchReply {
    /// Flattened batch-major row values. The client reconstructs batch
    /// boundaries from the original request and the helper's claimed row set.
    pub values: NetFpBlob,
}

/// One multi-request row reply sharing one row-label vector across all batches.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetRowMatrixReply {
    /// Flattened request-major, batch-major row values. The client reconstructs
    /// batch boundaries from the original request and the helper's claimed row
    /// set.
    pub values: NetFpBlob,
}

/// One locally stored primary shard vector for a segment.
///
/// This is a raw storage dump: the server returns the vector exactly as it is
/// maintained, with no evaluation or repair fallback.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetSegmentPrimaryShardReply {
    pub values: NetFpBlob,
}

/// One batched row-read request for one segment and multiple column coordinates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetRowReadRequest {
    pub seg: u64,
    pub z_evals: NetFpBlob,
}

/// Compact planned unauthenticated object id for optimized reads.
///
/// The client constructs this ordered object list from its unauth optimized read
/// plan. Servers only serve these objects; they do not re-plan from auth range
/// metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnauthReadObjectId {
    Segment { segment: u64 },
    Block { block: u64 },
    Point { array_index: u64 },
}

/// Server-facing optimized unauthenticated object read request.
///
/// `kappa` is the threshold the client reads at; object identifiers are
/// positions in the layout of that threshold.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnauthOptimizedReadRequest {
    pub kappa: u64,
    pub objects: Vec<UnauthReadObjectId>,
}

/// One helper contribution for a planned unauthenticated object.
///
/// `object_pos` indexes the request's ordered `objects` vector, avoiding a
/// repeated object id in each response contribution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnauthReadObjectContribution {
    Segment {
        object_pos: u32,
        shards: Vec<NetSegmentPrimaryShardReply>,
    },
    Block {
        object_pos: u32,
        rows: NetRowBatchReply,
    },
    Point {
        object_pos: u32,
        shares: NetMaybeFpBlob,
    },
}

/// Position-keyed optimized unauthenticated object read response.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnauthOptimizedReadResponse {
    pub objects: Vec<UnauthReadObjectContribution>,
}

/// Fragments of several claims on the wire: `(index, values)` pairs, where
/// the index is a registry index for columns and a row-label index for rows.
pub type NetIndexedFragments = Vec<(u32, NetFpBlob)>;

/// Half of a store's source state for one stitched segment, supplied to undo
/// the stitch.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NetSourceHalfState {
    /// Source state of the diagonal segments `(0, 0)` and `(1, 1)`.
    Diagonal {
        columns_00: NetIndexedFragments,
        rows_00: NetIndexedFragments,
        columns_11: NetIndexedFragments,
        rows_11: NetIndexedFragments,
    },
    /// Primary columns of segments `(0, 0)`, `(0, 1)` and auxiliary rows of
    /// segments `(0, 0)`, `(1, 0)`.
    Leading {
        columns_00: NetIndexedFragments,
        columns_01: NetIndexedFragments,
        rows_00: NetIndexedFragments,
        rows_10: NetIndexedFragments,
    },
}

/// One helper claim's partial-stitch evaluations for one stitched segment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetPartialStitchEvals {
    pub registry_index: u32,
    pub row_evals: NetFpBlob,
    pub column_evals: NetFpBlob,
}

/// Authenticated read request for a leaf index, at the threshold `kappa` the
/// client reads at.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthReadRequest {
    pub leaf: u64,
    pub kappa: u64,
}

/// Authenticated read request for one aligned subtree.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthSubtreeReadRequest {
    pub start_index: u64,
    pub leaf_count: u64,
}

/// Authenticated read request for only the upper path of one aligned subtree.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthSubtreeUpperPathReadRequest {
    pub start_index: u64,
    pub leaf_count: u64,
}

/// Private opening request: one PIR query over the primary columns, the
/// same for every position of a log of `leaves` filled leaves.
///
/// `query` and `pack_keys` hold 56-bit words, seven bytes each.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthPrivateReadRequest {
    pub leaves: u64,
    pub kappa: u64,
    pub pattern_bits: u64,
    pub query: NetByteBlob,
    pub pack_keys: NetByteBlob,
}

/// The PIR response of one claim to a private opening.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthPrivateReadReply {
    pub registry_index: u32,
    pub ciphertexts: Vec<NetByteBlob>,
}

/// Private opening response: one reply per claim the server holds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthPrivateReadResponse {
    pub replies: Vec<AuthPrivateReadReply>,
}

/// Authenticated read response with shard replies for each node.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthReadNodeReply {
    pub node: NetNodeKey,
    pub values: NetMaybeFpBlob,
}

/// Authenticated read response containing per-node shard replies.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthReadResponse {
    pub nodes: Vec<AuthReadNodeReply>,
}

/// Authenticated subtree read response containing leaf and upper-path replies.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthSubtreeReadResponse {
    pub leaf_nodes: Vec<AuthReadNodeReply>,
    pub upper_path_nodes: Vec<AuthReadNodeReply>,
    #[serde(default)]
    pub raw_segment_shards: Vec<NetSegmentPrimaryShardReply>,
}

/// Authenticated subtree read response containing only the upper-path replies.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthSubtreeUpperPathReadResponse {
    pub upper_path_nodes: Vec<AuthReadNodeReply>,
}

/// Network-friendly upper-path proof for one aligned subtree.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetAuthSubtreeProof {
    pub start_index: u64,
    pub leaf_count: u64,
    pub upper_siblings: Vec<NetValue>,
    pub root: NetValue,
}

/// Per-segment delta bundle for broadcast updates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetSegmentDeltas {
    pub seg: u64,
    pub deltas: Vec<(u32, NetFp)>,
}

/// Batch update entry for authenticated writes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetBatchWriteUpdate {
    pub index: u64,
    pub old: NetValue,
    pub new: NetValue,
}

/// Network-friendly multiproof for batch writes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetMerkleProof {
    pub root: NetValue,
    pub nodes: Vec<(NetNodeKey, NetValue)>,
}

/// Authenticated write request (client supplies old/new/path only).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthWriteRequest {
    pub old: NetValue,
    pub new: NetValue,
    pub path: NetAuthPath,
}

/// Authenticated batch write request (client supplies old/new/proof only).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthBatchWriteRequest {
    pub updates: Vec<NetBatchWriteUpdate>,
    pub proof: NetMerkleProof,
    pub delta_mode: NetDeltaMode,
}

/// Authenticated write request for one aligned fresh subtree.
///
/// The old contents are implicitly the default values for the covered leaves.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthFreshSubtreeWriteRequest {
    pub leaves: NetValueBlob,
    pub proof: NetAuthSubtreeProof,
    pub delta_mode: NetDeltaMode,
}

/// Authenticated write response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthWriteResponse {
    pub ok: bool,
    #[serde(default)]
    pub root_new: Option<NetValue>,
    #[serde(default)]
    pub error: Option<String>,
}

/// Authenticated batch write response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthBatchWriteResponse {
    pub ok: bool,
    #[serde(default)]
    pub root_new: Option<NetValue>,
    #[serde(default)]
    pub error: Option<String>,
}

/// Legacy RPC request envelope.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RpcRequest {
    ReadShard(ReadShardRequest),
    AuthRead(AuthReadRequest),
    AuthWrite(AuthWriteRequest),
}

/// Legacy RPC response envelope.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RpcResponse {
    ReadShard(ReadShardResponse),
    AuthRead(AuthReadResponse),
    AuthWrite(AuthWriteResponse),
    Error(RpcError),
}

/// RPC error response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RpcError {
    pub message: String,
}

impl fmt::Display for RpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Read a single length-prefixed frame.
pub fn read_frame(stream: &mut TcpStream, max_len: usize) -> io::Result<Option<Vec<u8>>> {
    let mut len_buf = [0u8; 4];
    match stream.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(err) => return Err(err),
    }
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > max_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame too large: {}", len),
        ));
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf)?;
    Ok(Some(buf))
}

/// Write a single length-prefixed frame.
pub fn write_frame(stream: &mut TcpStream, payload: &[u8]) -> io::Result<()> {
    let len = payload.len();
    let len_u32 = u32::try_from(len).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("payload too large: {}", len),
        )
    })?;
    stream.write_all(&len_u32.to_le_bytes())?;
    stream.write_all(payload)?;
    Ok(())
}

/// Encode one typed message into the binary RPC payload format.
pub fn encode_binary<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    bincode::serde::encode_to_vec(value, bincode::config::standard()).map_err(|err| err.to_string())
}

/// Decode one typed message from the binary RPC payload format.
pub fn decode_binary<T: DeserializeOwned>(payload: &[u8]) -> Result<T, String> {
    let (value, consumed): (T, usize) =
        bincode::serde::decode_from_slice(payload, bincode::config::standard())
            .map_err(|err| err.to_string())?;
    if consumed != payload.len() {
        return Err(format!(
            "binary payload has {} trailing bytes",
            payload.len().saturating_sub(consumed)
        ));
    }
    Ok(value)
}

/// Invert a denominator safely for network replies.
#[cfg(test)]
mod tests {
    use super::{
        NetFp, NetFpBlob, NetMaybeFpBlob, NetValue, NetValueBlob, decode_binary, encode_binary,
    };
    use crate::field::Fp;

    #[test]
    fn value_blob_round_trips_compact_json() {
        let values = vec![[0x11u8; 32], [0x22u8; 32], [0x33u8; 32]];
        let blob = NetValueBlob(values.clone());
        let encoded = serde_json::to_string(&blob).expect("serialize blob");
        // Base64 blob should be materially smaller than per-value JSON hex strings.
        assert!(encoded.len() < 3 * 70);
        let decoded: NetValueBlob = serde_json::from_str(&encoded).expect("deserialize blob");
        assert_eq!(decoded.0, values);
    }

    #[test]
    fn net_scalars_round_trip_compact_binary() {
        let fp = NetFp(Fp::from(42u64));
        let encoded = encode_binary(&fp).expect("encode fp");
        assert!(encoded.len() <= 40);
        let decoded: NetFp = decode_binary(&encoded).expect("decode fp");
        assert_eq!(decoded, fp);

        let value = NetValue([0xabu8; 32]);
        let encoded = encode_binary(&value).expect("encode value");
        assert!(encoded.len() <= 40);
        let decoded: NetValue = decode_binary(&encoded).expect("decode value");
        assert_eq!(decoded, value);
    }

    #[test]
    fn blobs_round_trip_compact_binary() {
        let values = NetValueBlob(vec![[0x11u8; 32], [0x22u8; 32], [0x33u8; 32]]);
        let encoded = encode_binary(&values).expect("encode value blob");
        assert!(encoded.len() < 3 * 40);
        let decoded: NetValueBlob = decode_binary(&encoded).expect("decode value blob");
        assert_eq!(decoded, values);

        let fblob = NetFpBlob(vec![Fp::from(1u64), Fp::from(2u64), Fp::from(3u64)]);
        let encoded = encode_binary(&fblob).expect("encode fp blob");
        assert!(encoded.len() < 3 * 40);
        let decoded: NetFpBlob = decode_binary(&encoded).expect("decode fp blob");
        assert_eq!(decoded, fblob);

        let maybe = NetMaybeFpBlob(vec![Some(Fp::from(1u64)), None, Some(Fp::from(3u64))]);
        let encoded = encode_binary(&maybe).expect("encode maybe fp blob");
        assert!(encoded.len() < 3 * 40);
        let decoded: NetMaybeFpBlob = decode_binary(&encoded).expect("decode maybe fp blob");
        assert_eq!(decoded, maybe);
    }
}
