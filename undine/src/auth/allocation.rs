//! Allocation helpers for authenticated storage.
//!
//! These routines extend the coded array with default Merkle values and track
//! the allocated leaf prefix.

use super::core::conceptual_total_nodes_u128;
use super::mapping;
use super::mapping::NodeLinearization;
use super::merkle::default_hash;
use super::{AllocateError, AuthParams, AuthStore};
use crate::field::fp_zero;
use crate::unauth::{LogicalValueFamilyKey, SetupTimings, UnauthArray};
use crate::value::{Value256, encode_value};
use std::ops::Range;

const ZERO_FAMILY_CODE: u16 = u16::MAX;

/// Allocate the next leaf chunk without overriding labels.
pub fn allocate_leaf_chunk<U>(
    auth: &AuthParams,
    node_linearization: NodeLinearization,
    alloc_leaf_batch: usize,
    allocated_leaves: &mut usize,
    store: &mut AuthStore<U>,
    mut timings: Option<&mut SetupTimings>,
) -> Result<(), AllocateError>
where
    U: UnauthArray + Clone,
{
    // Compute the new leaf allocation boundary and validate capacity.
    let old = *allocated_leaves as u64;
    let new = old
        .checked_add(alloc_leaf_batch as u64)
        .ok_or(AllocateError::NoCapacity)?;
    if new as u128 > auth.conceptual_l_max_u128() {
        return Err(AllocateError::NoCapacity);
    }
    let new_usize = usize::try_from(new).map_err(|_| AllocateError::NoCapacity)?;

    // Extend the underlying array with default node values.
    let height = auth.tree_height as u16;
    let cursor = mapping::cursor(node_linearization, height, new);
    store
        .unauth
        .extend_with_value_families_timed(
            cursor,
            |range| default_family_key_for_range(auth, node_linearization, range),
            |_range, family_key| default_values_for_family_key(auth, family_key),
            timings.as_deref_mut(),
        )
        .map_err(|_| AllocateError::NoCapacity)?;
    // Sync allocation length after a successful extend.
    *allocated_leaves = new_usize;
    Ok(())
}

pub(crate) fn default_family_key_for_range(
    auth: &AuthParams,
    node_linearization: NodeLinearization,
    range: Range<u64>,
) -> LogicalValueFamilyKey {
    let mut codes = Vec::with_capacity((range.end - range.start) as usize);
    let height = auth.tree_height as u16;
    let total_nodes = conceptual_total_nodes_u128(auth.tree_height);
    for idx in range {
        if idx as u128 >= total_nodes {
            codes.push(ZERO_FAMILY_CODE);
            continue;
        }
        let (level, _pos) = mapping::node_from_idx(node_linearization, height, idx);
        if level == height {
            codes.push(ZERO_FAMILY_CODE);
        } else {
            codes.push(level);
        }
    }
    LogicalValueFamilyKey::from_codes(codes)
}

pub(crate) fn default_values_for_family_key(
    auth: &AuthParams,
    family_key: &LogicalValueFamilyKey,
) -> Vec<Value256> {
    family_key
        .codes()
        .iter()
        .map(|&code| value_from_family_code(auth, code))
        .collect()
}

pub(crate) fn default_value_at_index(
    auth: &AuthParams,
    node_linearization: NodeLinearization,
    idx: u64,
) -> Value256 {
    let total_nodes = conceptual_total_nodes_u128(auth.tree_height);
    if idx as u128 >= total_nodes {
        return encode_value(fp_zero());
    }
    let (level, _pos) = mapping::node_from_idx(node_linearization, auth.tree_height as u16, idx);
    if level as usize == auth.tree_height {
        encode_value(fp_zero())
    } else {
        default_hash(&auth.defaults, level as usize)
    }
}

fn value_from_family_code(auth: &AuthParams, code: u16) -> Value256 {
    if code == ZERO_FAMILY_CODE {
        encode_value(fp_zero())
    } else {
        default_hash(&auth.defaults, code as usize)
    }
}
