//! Preorder mapping for nodes in a prefix-allocated tree.
//!
//! # Overview
//! The system conceptually allocates leaves in a prefix `[0..k)` of a perfect
//! binary tree of height `H`. This module provides the mapping between:
//! - A node `(level, pos)` in the full tree.
//! - Its preorder index in a depth-first traversal (root, then left subtree,
//!   then right subtree).
//!
//! This preorder index is used as a stable, append-only logical array
//! linearization: once a node is "active", its preorder index never changes
//! even as more leaves are allocated.
//!
//! # Definitions
//! - Tree height `H` uses the same convention as `merkle.rs`: root at level 0,
//!   leaves at level `H`.
//! - Preorder traversal order is:
//!   ```text
//!   visit(node)
//!   traverse(left subtree)
//!   traverse(right subtree)
//!   ```
//! - Active prefix: only nodes whose leaf interval intersects `[0..k)` are
//!   considered active. `cursor_preorder(H, k)` returns the number of nodes
//!   visited when traversing *only active nodes*.
//!
//! # Invariants
//! - `cursor_preorder` is monotone in `k`.
//! - `preorder_idx_if_active` returns `None` if a node does not intersect the
//!   allocated leaf prefix.
//! - If `idx < cursor_preorder(H, k)`, then `node_from_preorder_idx(H, idx)`
//!   returns an active node for that prefix.

pub use crate::core::NodeLinearization;

/// Total nodes in a perfect binary subtree of the given height.
fn total_nodes_for_height(height: u16) -> u128 {
    let shift = u32::from(height)
        .checked_add(1)
        .expect("height too large for node shift");
    if shift >= 128 {
        return u128::MAX;
    }
    (1u128 << shift) - 1
}

/// Leaf capacity for a perfect subtree of height `height`.
///
/// Returns `None` if the count does not fit into `u64`.
fn leaf_capacity(height: u16) -> Option<u64> {
    if height >= 64 {
        None
    } else {
        Some(1u64 << height)
    }
}

/// Size of the subtree rooted at `level` in a height-`H` tree.
///
/// # Mathematical definition
/// A perfect subtree of height `h` has `(2^(h+1) - 1)` nodes. Here
/// `h = H - level` for a node at level `level` in a height-`H` tree.
pub fn subtree_size(level: u16, height: u16) -> u128 {
    assert!(level <= height, "level out of range");
    let subtree_height = height - level;
    total_nodes_for_height(subtree_height)
}

/// Preorder index for a node at `(level, pos)` in a height-`H` tree.
///
/// # Algorithm
/// Walk the path from the root to `(level, pos)` and count how many nodes would
/// be visited before this node in preorder traversal. For each step:
/// - If we go left, add `1` (the parent itself).
/// - If we go right, add `1 + size(left_subtree)`.
///
/// This corresponds to the recurrence:
/// ```text
/// idx(root) = 0
/// idx(right child) = 1 + size(left_subtree)
/// idx(left child)  = 1
/// ```
///
/// # Preconditions
/// - `level <= height`.
/// - `pos < 2^level`.
pub fn preorder_idx(height: u16, level: u16, pos: u64) -> u64 {
    assert!(level <= height, "level out of range");
    if level < 64 {
        let max_pos = 1u64 << level;
        assert!(pos < max_pos, "pos out of range for level");
    }
    let mut idx: u128 = 0;
    for i in 1..=level {
        let shift = level - i;
        let bit = (pos >> shift) & 1;
        if bit == 0 {
            // Move left: only the parent node is skipped.
            idx = idx.saturating_add(1);
        } else {
            // Move right: skip parent plus the entire left subtree.
            let left_size = subtree_size(i, height);
            idx = idx.saturating_add(1).saturating_add(left_size);
        }
    }
    u64::try_from(idx).expect("preorder index overflow")
}

/// Determine whether a node intersects the prefix `[0..k)`.
///
/// # Mathematical definition
/// A node `(level, pos)` covers the leaf range
/// `[pos * 2^(H - level), (pos + 1) * 2^(H - level))`. This node is active
/// if and only if that interval intersects `[0..k)`.
fn is_active(height: u16, k: u64, level: u16, pos: u64) -> bool {
    if level > height || k == 0 {
        return false;
    }
    let shift = height - level;
    if shift >= 64 {
        return pos == 0;
    }
    if shift == 0 {
        return pos < k;
    }
    let q = k >> shift;
    let r = k & ((1u64 << shift) - 1);
    pos < q || (pos == q && r != 0)
}

/// Preorder index for an active node, or None if inactive for prefix size `k`.
///
/// # Invariant
/// The returned index is guaranteed to be `< cursor_preorder(height, k)`.
pub fn preorder_idx_if_active(height: u16, k: u64, level: u16, pos: u64) -> Option<u64> {
    if !is_active(height, k, level, pos) {
        return None;
    }
    let idx = preorder_idx(height, level, pos);
    let cursor = cursor_preorder(height, k);
    if idx < cursor { Some(idx) } else { None }
}

/// Cursor (active preorder prefix length) for prefix size `k`.
///
/// # Intuition
/// Think of traversing the tree in preorder, but pruning any subtree that is
/// completely outside `[0..k)`. The cursor is the number of nodes visited.
///
/// # Monotonicity
/// Increasing `k` can only increase the cursor.
pub fn cursor_preorder(height: u16, k: u64) -> u64 {
    if k == 0 {
        return 0;
    }
    let mut cursor: u128 = 0;
    let mut remaining = k;
    let mut h = height;
    loop {
        if remaining == 0 {
            break;
        }
        cursor = cursor.saturating_add(1);
        if h == 0 {
            break;
        }
        let left_height = h - 1;
        match leaf_capacity(left_height) {
            None => {
                h = left_height;
            }
            Some(left_leaves) => {
                if remaining <= left_leaves {
                    h = left_height;
                } else {
                    let left_nodes = total_nodes_for_height(left_height);
                    cursor = cursor.saturating_add(left_nodes);
                    remaining -= left_leaves;
                    h = left_height;
                }
            }
        }
    }
    if cursor > u64::MAX as u128 {
        u64::MAX
    } else {
        cursor as u64
    }
}

/// Invert a preorder index into `(level, pos)` for a height-`H` tree.
///
/// # Algorithm
/// Starting from the root, decide whether the index lies in the left subtree
/// or right subtree by comparing with `size(left_subtree)`. This is the inverse
/// of `preorder_idx`.
///
/// # Preconditions
/// - `idx` must be `< total_nodes_for_height(H)`.
pub fn node_from_preorder_idx(height: u16, idx: u64) -> (u16, u64) {
    let mut level: u16 = 0;
    let mut pos: u64 = 0;
    let mut offset: u128 = idx as u128;
    let mut h = height;
    loop {
        if offset == 0 {
            return (level, pos);
        }
        if h == 0 {
            assert!(offset == 0, "preorder index out of range");
            return (level, pos);
        }
        let left_size = total_nodes_for_height(h - 1);
        if offset <= left_size {
            offset -= 1;
            level += 1;
            pos <<= 1;
            h -= 1;
        } else {
            offset -= 1 + left_size;
            level += 1;
            pos = (pos << 1) | 1;
            h -= 1;
        }
    }
}

/// Cursor (active prefix length) under the selected mapping order.
pub fn cursor(order: NodeLinearization, height: u16, k: u64) -> u64 {
    match order {
        NodeLinearization::Preorder => cursor_preorder(height, k),
    }
}

/// Mapping order-aware lookup for active node indices.
pub fn idx_if_active(
    order: NodeLinearization,
    height: u16,
    k: u64,
    level: u16,
    pos: u64,
) -> Option<u64> {
    match order {
        NodeLinearization::Preorder => preorder_idx_if_active(height, k, level, pos),
    }
}

/// Invert an index into `(level, pos)` under the selected mapping order.
pub fn node_from_idx(order: NodeLinearization, height: u16, idx: u64) -> (u16, u64) {
    match order {
        NodeLinearization::Preorder => node_from_preorder_idx(height, idx),
    }
}
