//! Merkle tree helpers for leaf and internal node hashing.
//!
//! # Overview
//! The system maintains a conceptual full binary Merkle tree of height `H`.
//! Leaves are at level `H`, the root is at level `0`. Each node is identified
//! by `(level, pos)` where `pos` is the index at that level.
//!
//! # Node identity and intervals
//! For a fixed height `H`, node `(level, pos)` covers the leaf interval:
//! `range = [pos * 2^(H - level), (pos + 1) * 2^(H - level))`.
//! This definition is used consistently for Merkle hashing and logical array
//! mapping elsewhere in the codebase.
//!
//! ASCII picture for `H = 2`:
//! ```text
//! level 0:          (0,0)
//!                 /       \
//! level 1:     (1,0)     (1,1)
//!             /   \     /   \
//! level 2:  (2,0)(2,1)(2,2)(2,3)
//! ```
//!
//! # Hashing model
//! - Leaves: `leaf_hash(i, v)` hashes the leaf index `i` and value `v`.
//! - Internal nodes: `node_hash(left, right)` hashes the two child labels.
//! - Default hashes are level-specific and are precomputed from a default leaf.
//!   When a leaf's stored value is all-zero, we *replace* it with the default
//!   leaf label to preserve a compact representation.
//!
//! # Notes / Pitfalls
//! - The default leaf is **position independent**; it is domain separated from
//!   regular leaf hashing. This means the "empty" tree root is deterministic.
//! - `label_from_content` must be used when the caller has *stored content*
//!   (raw leaf value or internal label) and needs the label.

use crate::field::Fp;
use crate::value::{Value256, encode_value};
use ark_ff::PrimeField;
use blake3::Hasher;
use std::cmp::Ordering;

/// Canonical identity for a node in the conceptual tree.
///
/// # Invariants
/// - `level` is in `[0, H]`.
/// - `pos < 2^level`.
///
/// # Ordering
/// The derived ordering is `(level, pos)`, which is **not** tree order. It is
/// used only for deterministic maps/sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeKey {
    pub level: u32,
    pub pos: u64,
}

impl PartialOrd for NodeKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NodeKey {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.level.cmp(&other.level) {
            Ordering::Equal => self.pos.cmp(&other.pos),
            ordering => ordering,
        }
    }
}

/// Precomputed default hashes by level (0=root .. H=leaf).
///
/// `by_level[L]` is the label for a default subtree rooted at level `L`.
#[derive(Clone, Debug)]
pub struct DefaultHashes {
    pub by_level: Vec<Value256>,
}

/// Hash tagged inputs into a field element and return its byte encoding.
///
/// # Algorithm
/// Uses BLAKE3 with a domain-separating `tag`, then reduces the digest into
/// the field and encodes it to 32 bytes.
///
/// # Why a tag?
/// Distinct tags prevent collisions between different hash domains, e.g.
/// leaf vs internal nodes vs default leaves.
fn hash_to_value(tag: u8, parts: &[&[u8]]) -> Value256 {
    let mut hasher = Hasher::new();
    // Tag is included as the first byte to domain-separate callers.
    hasher.update(&[tag]);
    for p in parts {
        hasher.update(p);
    }
    let digest = hasher.finalize();
    let fp = Fp::from_le_bytes_mod_order(digest.as_bytes());
    encode_value(fp)
}

/// Leaf hash includes the leaf index and its stored value.
///
/// # Mathematical definition
/// `H_leaf(i, v) = H(tag=0x00 || i || v)` where `i` is little-endian.
pub fn leaf_hash(i: u64, value: Value256) -> Value256 {
    let idx = i.to_le_bytes();
    hash_to_value(0x00, &[&idx, &value])
}

/// Internal node hash includes left and right child labels.
///
/// # Mathematical definition
/// `H_node(L, R) = H(tag=0x01 || L || R)`.
pub fn node_hash(left: Value256, right: Value256) -> Value256 {
    hash_to_value(0x01, &[&left, &right])
}

/// Domain-separated default leaf hash (position independent).
///
/// This is the hash of an abstract "empty leaf". It is independent of index.
fn default_leaf_hash() -> Value256 {
    hash_to_value(0x02, &[b"default_leaf"])
}

/// Build the level-only default hash vector for a tree of height H.
///
/// # Algorithm
/// - Level `H` (leaves) is `default_leaf_hash()`.
/// - Level `L` is `node_hash(default[L+1], default[L+1])`.
pub fn build_default_hashes(height: usize) -> DefaultHashes {
    let mut by_level = vec![[0u8; 32]; height + 1];
    by_level[height] = default_leaf_hash();
    if height > 0 {
        for level in (0..height).rev() {
            let child = by_level[level + 1];
            by_level[level] = node_hash(child, child);
        }
    }
    DefaultHashes { by_level }
}

/// Return the default hash for a specific level.
pub fn default_hash(defaults: &DefaultHashes, level: usize) -> Value256 {
    defaults.by_level[level]
}

/// Create the leaf node key for a given index and tree height.
///
/// # Preconditions
/// - `index < 2^height` for a perfect tree.
pub fn leaf_key(height: u32, index: u64) -> NodeKey {
    NodeKey {
        level: height,
        pos: index,
    }
}

/// Return the parent key for a node.
///
/// # Preconditions
/// - `key.level > 0` (root has no parent).
pub fn parent(key: NodeKey) -> NodeKey {
    assert!(key.level > 0, "root has no parent");
    NodeKey {
        level: key.level - 1,
        pos: key.pos / 2,
    }
}

/// Return the sibling key for a node.
///
/// # Definition
/// Siblings differ in the least significant bit of `pos` at the same level.
pub fn sibling(key: NodeKey) -> NodeKey {
    NodeKey {
        level: key.level,
        pos: key.pos ^ 1,
    }
}

/// Return the left child key for a node.
pub fn child_left(key: NodeKey) -> NodeKey {
    NodeKey {
        level: key.level + 1,
        pos: key.pos * 2,
    }
}

/// Return the right child key for a node.
pub fn child_right(key: NodeKey) -> NodeKey {
    NodeKey {
        level: key.level + 1,
        pos: key.pos * 2 + 1,
    }
}

/// Return true if the key is within a tree of the given height.
///
/// # Preconditions
/// - `height` is the global tree height.
pub fn is_valid_key(height: u32, key: NodeKey) -> bool {
    if key.level > height {
        return false;
    }
    let max_pos = 1u128 << key.level;
    (key.pos as u128) < max_pos
}

/// Return the label for a leaf, honoring the level-only default hash.
///
/// # Behavior
/// If `value` is all-zero, we use the default leaf label for that level rather
/// than hashing the zero value.
pub fn leaf_label(defaults: &DefaultHashes, height: u32, index: u64, value: Value256) -> Value256 {
    if value == [0u8; 32] {
        default_hash(defaults, height as usize)
    } else {
        leaf_hash(index, value)
    }
}

/// Return the label for a node based on its stored content.
///
/// # Rationale
/// - For leaves, stored content is the *raw value* and must be hashed.
/// - For internal nodes, stored content is already a label.
pub fn label_from_content(
    defaults: &DefaultHashes,
    height: u32,
    key: NodeKey,
    content: Value256,
) -> Value256 {
    if key.level == height {
        leaf_label(defaults, height, key.pos, content)
    } else {
        content
    }
}

/// Recompute a Merkle root from a value and its authentication path.
///
/// # Algorithm
/// Starting from the leaf label, fold in sibling labels bottom-up using
/// `node_hash`. At each step we decide whether the current node is a left or
/// right child based on `pos`.
///
/// # Complexity
/// `O(H)` hash evaluations.
pub fn recompute_root(
    height: u32,
    index: u64,
    value: Value256,
    siblings: &[Value256],
    defaults: &DefaultHashes,
) -> Value256 {
    let mut key = leaf_key(height, index);
    let mut cur = leaf_label(defaults, height, index, value);
    for sib_content in siblings.iter() {
        let sib_key = sibling(key);
        let sib_label = label_from_content(defaults, height, sib_key, *sib_content);
        cur = if key.pos % 2 == 0 {
            node_hash(cur, sib_label)
        } else {
            node_hash(sib_label, cur)
        };
        key = parent(key);
    }
    cur
}
