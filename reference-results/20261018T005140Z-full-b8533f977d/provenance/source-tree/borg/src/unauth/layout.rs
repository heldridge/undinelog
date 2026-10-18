//! Semantic layout helpers for the unauthenticated paper path.
//!
//! Ordinary array reads and writes use a stable public order inside each
//! segment: logical offset `j` is mapped through the canonical
//! `logical_to_grid_index` permutation derived from the layout base and join
//! level. All `segment_len == n_grid` stored grid values are public semantic
//! values. Aligned reads use the current `H_s` coset layout and therefore still
//! depend on `s`.

use crate::field::{Fp, fp_one};
use crate::unauth::CodedArrayParams;
use ark_ff::FftField;

/// One logical point in the paper's segment layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticPosition {
    /// Logical offset within the segment.
    pub logical_offset: usize,
    /// Semantic block id within the segment.
    pub block: usize,
    /// Offset within the aligned block.
    pub offset_in_block: usize,
    /// Underlying FFT-grid index used by the coded-array core.
    pub grid_index: usize,
}

/// One logical aligned block in the paper's segment layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticBlockPosition {
    /// Global logical block index.
    pub logical_block: u64,
    /// Segment containing the block.
    pub segment: usize,
    /// Block id within the segment.
    pub block_in_segment: usize,
    /// Underlying representative FFT-grid index for the block's column.
    pub representative_grid_index: usize,
}

/// Return the number of public semantic values carried by one segment.
pub fn public_values_per_segment(params: &CodedArrayParams) -> usize {
    params.n_grid
}

/// Map a global public array index into `(segment, logical_offset)`.
pub fn public_index_to_segment(params: &CodedArrayParams, idx: u64) -> (usize, usize) {
    let span = public_values_per_segment(params) as u64;
    ((idx / span) as usize, (idx % span) as usize)
}

/// Return the deterministic physical-grid scale for one logical segment.
///
/// The scale follows the factor-two merge tree described in
/// `documentation/segment-double.md`: for any adjacent pair at the
/// current grid size, the right segment's scale is the left scale multiplied by
/// a primitive `2 * n_grid` root. This keeps adjacent segments on sibling
/// even/odd cosets without storing per-segment placement metadata locally.
pub fn segment_domain_rho(params: &CodedArrayParams, segment: usize) -> Fp {
    let two_adicity = <Fp as FftField>::TWO_ADICITY as usize;
    let n_log = params.n_grid.trailing_zeros() as usize;
    let mut rho = fp_one();
    let mut bits = segment;
    let mut level = 0usize;
    while bits != 0 && n_log + level + 1 <= two_adicity {
        if bits & 1 == 1 {
            let order = (params.n_grid as u64) << (level + 1);
            let root = <Fp as FftField>::get_root_of_unity(order)
                .expect("field supplies the requested two-adic root");
            rho *= root;
        }
        bits >>= 1;
        level += 1;
    }
    debug_assert!(
        bits == 0,
        "segment domain scale exceeds the field two-adic root depth"
    );
    rho
}

/// Return the semantic layout position for one logical segment offset.
pub fn semantic_position(params: &CodedArrayParams, logical_offset: usize) -> SemanticPosition {
    semantic_position_for_s(params, params.s, logical_offset)
}

/// Return the semantic layout position for one logical segment offset under an explicit layout `s`.
pub fn semantic_position_for_s(
    params: &CodedArrayParams,
    layout_s: usize,
    logical_offset: usize,
) -> SemanticPosition {
    assert!(logical_offset < public_values_per_segment(params));
    assert!(layout_s > 0);
    assert!(layout_s.is_power_of_two());
    assert!(layout_s <= params.n_grid);
    let block = logical_offset / layout_s;
    let offset_in_block = logical_offset % layout_s;
    let blocks_per_segment = params.n_grid / layout_s;
    let grid_index = block + offset_in_block * blocks_per_segment;
    debug_assert!(grid_index < params.n_grid);
    SemanticPosition {
        logical_offset,
        block,
        offset_in_block,
        grid_index,
    }
}

/// Map a public logical payload offset to the underlying FFT-grid index.
///
/// This mapping is intentionally independent of `s`: changing the helper
/// threshold changes the aligned/coset view, but it must not rename public
/// array cells for `read(i)` or point updates.
pub fn logical_offset_to_grid_index(params: &CodedArrayParams, logical_offset: usize) -> usize {
    assert!(logical_offset < public_values_per_segment(params));
    params.logical_grid_index(logical_offset)
}

/// Map an aligned/coset-layout offset to the underlying FFT-grid index under
/// an explicit layout `s`.
///
/// This helper is for aligned block operations and migration tests; it is not
/// the public `read(i)` layout.
pub fn logical_offset_to_grid_index_for_s(
    params: &CodedArrayParams,
    layout_s: usize,
    logical_offset: usize,
) -> usize {
    semantic_position_for_s(params, layout_s, logical_offset).grid_index
}

/// Return the semantic-grid evaluation point for a logical offset.
pub fn logical_offset_to_alpha(params: &CodedArrayParams, logical_offset: usize) -> Fp {
    params.grid_point(logical_offset_to_grid_index(params, logical_offset))
}

/// Return the scaled physical-grid point for one segment/logical offset.
pub fn logical_offset_to_alpha_for_segment(
    params: &CodedArrayParams,
    segment: usize,
    logical_offset: usize,
) -> Fp {
    segment_domain_rho(params, segment)
        * params.grid_point(logical_offset_to_grid_index(params, logical_offset))
}

/// Return the scaled physical-grid point for one segment/raw grid index.
pub fn grid_index_to_alpha_for_segment(
    params: &CodedArrayParams,
    segment: usize,
    grid_index: usize,
) -> Fp {
    assert!(grid_index < params.n_grid);
    segment_domain_rho(params, segment) * params.grid_point(grid_index)
}

/// Return the public column coordinate `A = alpha^s` for a logical offset.
pub fn logical_offset_to_column_coordinate(params: &CodedArrayParams, logical_offset: usize) -> Fp {
    params.column_coordinate_for_grid_index(logical_offset_to_grid_index(params, logical_offset))
}

/// Return `alpha^s` for one scaled segment/logical offset.
pub fn logical_offset_to_column_coordinate_for_segment(
    params: &CodedArrayParams,
    segment: usize,
    logical_offset: usize,
) -> Fp {
    params.column_coordinate(logical_offset_to_alpha_for_segment(
        params,
        segment,
        logical_offset,
    ))
}

/// Return the number of logical aligned blocks per segment.
pub fn logical_blocks_per_segment(params: &CodedArrayParams) -> usize {
    public_values_per_segment(params).div_ceil(params.s)
}

/// Return the semantic block position for one global logical block index.
pub fn semantic_block_position(
    params: &CodedArrayParams,
    logical_block: u64,
) -> SemanticBlockPosition {
    let blocks_per_segment = logical_blocks_per_segment(params) as u64;
    let segment = (logical_block / blocks_per_segment) as usize;
    let block_in_segment = (logical_block % blocks_per_segment) as usize;
    debug_assert!(block_in_segment < logical_blocks_per_segment(params));
    SemanticBlockPosition {
        logical_block,
        segment,
        block_in_segment,
        representative_grid_index: block_in_segment,
    }
}

/// Return the raw FFT-grid indices belonging to one aligned semantic block.
pub fn block_grid_indices(params: &CodedArrayParams, block_in_segment: usize) -> Vec<usize> {
    assert!(block_in_segment < logical_blocks_per_segment(params));
    let step = params.n_grid / params.s;
    (0..params.s)
        .map(|offset| block_in_segment + offset * step)
        .collect()
}

/// Return the public array indices covered by one aligned block.
///
/// The aligned block is a structured subset of ordinary public indices: each
/// returned index can also be read with `get(i)`. A `None` entry denotes the
/// reserved grid point or a not-yet-active array position.
pub fn aligned_block_indices(
    params: &CodedArrayParams,
    logical_block: u64,
    len: u64,
) -> Vec<Option<u64>> {
    let pos = semantic_block_position(params, logical_block);
    let seg_base = pos.segment as u64 * public_values_per_segment(params) as u64;
    block_grid_indices(params, pos.block_in_segment)
        .into_iter()
        .map(|grid_index| {
            let logical_offset = params.logical_offset_for_grid_index(grid_index);
            let idx = seg_base + logical_offset as u64;
            (idx < len).then_some(idx)
        })
        .collect()
}

/// Return the semantic-grid evaluation points for one aligned semantic block.
pub fn block_alphas(params: &CodedArrayParams, block_in_segment: usize) -> Vec<Fp> {
    block_grid_indices(params, block_in_segment)
        .into_iter()
        .map(|grid_index| params.grid_point(grid_index))
        .collect()
}

/// Return the scaled physical-grid evaluation points for one segment/block.
pub fn block_alphas_for_segment(
    params: &CodedArrayParams,
    segment: usize,
    block_in_segment: usize,
) -> Vec<Fp> {
    let rho = segment_domain_rho(params, segment);
    block_grid_indices(params, block_in_segment)
        .into_iter()
        .map(|grid_index| rho * params.grid_point(grid_index))
        .collect()
}

/// Return the public column coordinate `A = x^s` for one aligned block.
pub fn logical_block_to_column_coordinate(params: &CodedArrayParams, logical_block: u64) -> Fp {
    let pos = semantic_block_position(params, logical_block);
    params.column_coordinate_for_grid_index(pos.representative_grid_index)
}

/// Return `A = alpha^s` for one aligned block in its scaled physical segment.
pub fn logical_block_to_column_coordinate_for_segment(
    params: &CodedArrayParams,
    logical_block: u64,
) -> Fp {
    let pos = semantic_block_position(params, logical_block);
    params.column_coordinate(
        segment_domain_rho(params, pos.segment) * params.grid_point(pos.representative_grid_index),
    )
}

/// Return the current-mode block id within a segment for one induced-grid column coordinate.
pub fn block_in_segment_from_column_coordinate(
    params: &CodedArrayParams,
    z_eval: Fp,
) -> Option<usize> {
    let z_root = crate::fp_pow(params.omega, params.s as u64);
    let mut cur = crate::fp_one();
    for idx in 0..params.induced_z_grid_size() {
        if cur == z_eval {
            return Some(idx);
        }
        cur *= z_root;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{
        aligned_block_indices, block_grid_indices, block_in_segment_from_column_coordinate,
        logical_block_to_column_coordinate, logical_blocks_per_segment,
        logical_offset_to_grid_index, public_values_per_segment, semantic_block_position,
        semantic_position,
    };
    use crate::coded_array::params::{
        CodedArrayConfig, build_coded_params, build_shard_registry_deterministic,
    };

    #[test]
    fn stable_payload_layout_is_separate_from_aligned_cosets() {
        let segment_len: usize = 8;
        let s: usize = 4;
        let n_grid = segment_len;
        let cfg = CodedArrayConfig {
            segment_len,
            s,
            shard_count: 1,
            shard_registry: build_shard_registry_deterministic(1, s, n_grid),
            party_registry_indices: vec![vec![0]],
            seed: 1,
        };
        let params = build_coded_params(&cfg);
        let expected = [0usize, 2, 4, 6, 1, 3, 5, 7];
        for (logical, grid) in expected.into_iter().enumerate() {
            let pos = semantic_position(&params, logical);
            assert_eq!(pos.grid_index, grid);
            assert_eq!(logical_offset_to_grid_index(&params, logical), logical);
        }
        assert_eq!(logical_blocks_per_segment(&params), 2);
        let block0 = semantic_block_position(&params, 0);
        assert_eq!(block0.segment, 0);
        assert_eq!(block0.block_in_segment, 0);
        assert_eq!(block_grid_indices(&params, 0), vec![0, 2, 4, 6]);
        let block1 = semantic_block_position(&params, 1);
        assert_eq!(block1.segment, 0);
        assert_eq!(block1.block_in_segment, 1);
        assert_eq!(block_grid_indices(&params, 1), vec![1, 3, 5, 7]);
        assert_eq!(
            aligned_block_indices(&params, 1, public_values_per_segment(&params) as u64),
            vec![Some(1), Some(3), Some(5), Some(7)]
        );
    }

    #[test]
    fn column_coordinate_roundtrips_to_block_in_segment() {
        let segment_len: usize = 32;
        let s: usize = 8;
        let n_grid = segment_len;
        let cfg = CodedArrayConfig {
            segment_len,
            s,
            shard_count: 1,
            shard_registry: build_shard_registry_deterministic(1, s, n_grid),
            party_registry_indices: vec![vec![0]],
            seed: 1,
        };
        let params = build_coded_params(&cfg);
        for block_in_segment in 0..logical_blocks_per_segment(&params) {
            let block = block_in_segment as u64;
            let z_eval = logical_block_to_column_coordinate(&params, block);
            assert_eq!(
                block_in_segment_from_column_coordinate(&params, z_eval),
                Some(block_in_segment)
            );
        }
    }
}
