//! Semantic layout helpers for the unauthenticated paper path.
//!
//! Ordinary array reads and writes use a stable public order inside each
//! segment: logical offset `j` is mapped through the canonical
//! `logical_to_grid_index` permutation derived from the layout base and the
//! stitch level. All `segment_len == n_grid` stored grid values are public
//! semantic values.
//!
//! A segment's cells form a `kappa x kappa` grid on the product of its two
//! coset axes. An aligned block is the set of `kappa` cells that share one
//! Z-axis point; it is the unit read through auxiliary rows. After a stitch
//! the cells of one block are not contiguous in public order, so block
//! membership always goes through the grid index.

use crate::field::Fp;
use crate::unauth::CodedArrayParams;

/// One logical point in the paper's segment layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticPosition {
    /// Logical offset within the segment.
    pub logical_offset: usize,
    /// Aligned block id within the segment: the cell's Z-axis index.
    pub block: usize,
    /// Offset within the aligned block: the cell's W-axis index.
    pub offset_in_block: usize,
    /// Underlying grid index used by the coded-array core.
    pub grid_index: usize,
}

/// One logical aligned block in the paper's segment layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticBlockPosition {
    /// Global logical block index.
    pub logical_block: u64,
    /// Segment containing the block.
    pub segment: usize,
    /// Block id within the segment: the Z-axis index shared by its cells.
    pub block_in_segment: usize,
    /// Grid index of the block's first cell.
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

/// Return the semantic layout position for one logical segment offset.
pub fn semantic_position(params: &CodedArrayParams, logical_offset: usize) -> SemanticPosition {
    let grid_index = logical_offset_to_grid_index(params, logical_offset);
    SemanticPosition {
        logical_offset,
        block: params.z_index_for_grid_index(grid_index),
        offset_in_block: params.w_index_for_grid_index(grid_index),
        grid_index,
    }
}

/// Map a public logical payload offset to the underlying grid index.
///
/// This mapping depends only on the layout base and the stitch level, never
/// on the segment.
pub fn logical_offset_to_grid_index(params: &CodedArrayParams, logical_offset: usize) -> usize {
    assert!(logical_offset < public_values_per_segment(params));
    params.logical_grid_index(logical_offset)
}

/// Return the column coordinate of one cell: its Z-axis point in its segment.
///
/// Point reads interpolate primary-column entries over the column identifiers
/// and evaluate at this point.
pub fn logical_offset_to_column_coordinate_for_segment(
    params: &CodedArrayParams,
    segment: usize,
    logical_offset: usize,
) -> Fp {
    let grid_index = logical_offset_to_grid_index(params, logical_offset);
    params.segment_axes(segment).rho_z * params.axis[params.z_index_for_grid_index(grid_index)]
}

/// Return the number of logical aligned blocks per segment.
pub fn logical_blocks_per_segment(params: &CodedArrayParams) -> usize {
    params.need
}

/// Return the semantic block position for one global logical block index.
pub fn semantic_block_position(
    params: &CodedArrayParams,
    logical_block: u64,
) -> SemanticBlockPosition {
    let blocks_per_segment = logical_blocks_per_segment(params) as u64;
    let segment = (logical_block / blocks_per_segment) as usize;
    let block_in_segment = (logical_block % blocks_per_segment) as usize;
    SemanticBlockPosition {
        logical_block,
        segment,
        block_in_segment,
        representative_grid_index: params.grid_index_for_axes(block_in_segment, 0),
    }
}

/// Return the grid indices belonging to one aligned semantic block, ordered
/// by W-axis index.
pub fn block_grid_indices(params: &CodedArrayParams, block_in_segment: usize) -> Vec<usize> {
    assert!(block_in_segment < logical_blocks_per_segment(params));
    (0..params.s)
        .map(|w_index| params.grid_index_for_axes(block_in_segment, w_index))
        .collect()
}

/// Return the public array indices covered by one aligned block.
///
/// The aligned block is a structured subset of ordinary public indices: each
/// returned index can also be read with `get(i)`. A `None` entry denotes a
/// not-yet-active array position.
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

/// Return the W-axis points of one segment: the evaluation points of every
/// aligned block and of every primary column in that segment.
pub fn block_alphas_for_segment(params: &CodedArrayParams, segment: usize) -> Vec<Fp> {
    let rho_w = params.segment_axes(segment).rho_w;
    params.axis.iter().map(|&point| rho_w * point).collect()
}

/// Return the column coordinate of one aligned block: the Z-axis point shared
/// by its cells.
pub fn logical_block_to_column_coordinate_for_segment(
    params: &CodedArrayParams,
    logical_block: u64,
) -> Fp {
    let pos = semantic_block_position(params, logical_block);
    params.segment_axes(pos.segment).rho_z * params.axis[pos.block_in_segment]
}

/// Return the block id within `segment` whose cells share the Z-axis point
/// `z_eval`, if `z_eval` lies on that segment's Z-axis.
pub fn block_in_segment_from_column_coordinate(
    params: &CodedArrayParams,
    segment: usize,
    z_eval: Fp,
) -> Option<usize> {
    params.axis_index_of(z_eval * params.segment_axes(segment).inv_rho_z)
}

#[cfg(test)]
mod tests {
    use super::{
        aligned_block_indices, block_grid_indices, block_in_segment_from_column_coordinate,
        logical_block_to_column_coordinate_for_segment, logical_blocks_per_segment,
        logical_offset_to_grid_index, public_values_per_segment, semantic_block_position,
        semantic_position,
    };
    use crate::coded_array::params::{
        CodedArrayConfig, CodedArrayParams, build_coded_params, build_shard_registry_deterministic,
    };

    fn params_for(kappa: usize) -> CodedArrayParams {
        build_coded_params(&CodedArrayConfig {
            kappa,
            shard_count: 1,
            shard_registry: build_shard_registry_deterministic(1),
            party_registry_indices: vec![vec![0]],
            seed: 1,
        })
    }

    #[test]
    fn base_layout_is_row_major_on_the_cell_grid() {
        let params = params_for(4);
        assert_eq!(public_values_per_segment(&params), 16);
        assert_eq!(logical_blocks_per_segment(&params), 4);
        for logical in 0..16 {
            let pos = semantic_position(&params, logical);
            assert_eq!(pos.grid_index, logical);
            assert_eq!(pos.block, logical / 4);
            assert_eq!(pos.offset_in_block, logical % 4);
            assert_eq!(logical_offset_to_grid_index(&params, logical), logical);
        }
        let block5 = semantic_block_position(&params, 5);
        assert_eq!(block5.segment, 1);
        assert_eq!(block5.block_in_segment, 1);
        assert_eq!(block_grid_indices(&params, 1), vec![4, 5, 6, 7]);
        assert_eq!(
            aligned_block_indices(&params, 5, 2 * 16),
            vec![Some(20), Some(21), Some(22), Some(23)]
        );
        assert_eq!(
            aligned_block_indices(&params, 5, 22),
            vec![Some(20), Some(21), None, None]
        );
    }

    #[test]
    fn blocks_of_a_stitched_layout_partition_the_segment() {
        let mut params = params_for(4);
        params.apply_canonical_logical_layout(4).expect("layout");
        let span = public_values_per_segment(&params) as u64;
        let mut seen = vec![false; span as usize];
        for block in 0..logical_blocks_per_segment(&params) as u64 {
            for idx in aligned_block_indices(&params, block, span) {
                let idx = idx.expect("active index") as usize;
                assert!(!seen[idx]);
                seen[idx] = true;
                assert_eq!(semantic_position(&params, idx).block, block as usize);
            }
        }
        assert!(seen.into_iter().all(|covered| covered));
    }

    #[test]
    fn column_coordinate_roundtrips_to_block_in_segment() {
        let params = params_for(8);
        for segment in [0usize, 1, 2, 7] {
            for block_in_segment in 0..logical_blocks_per_segment(&params) {
                let block =
                    (segment * logical_blocks_per_segment(&params) + block_in_segment) as u64;
                let z_eval = logical_block_to_column_coordinate_for_segment(&params, block);
                assert_eq!(
                    block_in_segment_from_column_coordinate(&params, segment, z_eval),
                    Some(block_in_segment)
                );
            }
        }
    }
}
