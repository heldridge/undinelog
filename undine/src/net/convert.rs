//! Shared conversions between wire and core reconfiguration types.

use crate::field::Fp;
use crate::net::wire::{NetFpBlob, NetIndexedFragments, NetPartialStitchEvals, NetSourceHalfState};
use crate::unauth::{PartialStitchEvals, RecoveredClaimSegmentState, SourceHalfState};

fn net_fragments_to_core(fragments: NetIndexedFragments) -> Vec<(usize, Vec<Fp>)> {
    fragments
        .into_iter()
        .map(|(index, values)| (index as usize, values.0))
        .collect()
}

fn core_fragments_to_net(fragments: &[(usize, Vec<Fp>)]) -> NetIndexedFragments {
    fragments
        .iter()
        .map(|(index, values)| (*index as u32, NetFpBlob(values.clone())))
        .collect()
}

pub(crate) fn net_source_halves_to_core(halves: Vec<NetSourceHalfState>) -> Vec<SourceHalfState> {
    halves
        .into_iter()
        .map(|half| match half {
            NetSourceHalfState::Diagonal {
                columns_00,
                rows_00,
                columns_11,
                rows_11,
            } => SourceHalfState::Diagonal {
                segment_00: RecoveredClaimSegmentState {
                    shards: net_fragments_to_core(columns_00),
                    rows: net_fragments_to_core(rows_00),
                },
                segment_11: RecoveredClaimSegmentState {
                    shards: net_fragments_to_core(columns_11),
                    rows: net_fragments_to_core(rows_11),
                },
            },
            NetSourceHalfState::Leading {
                columns_00,
                columns_01,
                rows_00,
                rows_10,
            } => SourceHalfState::Leading {
                columns_00: net_fragments_to_core(columns_00),
                columns_01: net_fragments_to_core(columns_01),
                rows_00: net_fragments_to_core(rows_00),
                rows_10: net_fragments_to_core(rows_10),
            },
        })
        .collect()
}

pub(crate) fn core_source_halves_to_net(halves: &[SourceHalfState]) -> Vec<NetSourceHalfState> {
    halves
        .iter()
        .map(|half| match half {
            SourceHalfState::Diagonal {
                segment_00,
                segment_11,
            } => NetSourceHalfState::Diagonal {
                columns_00: core_fragments_to_net(&segment_00.shards),
                rows_00: core_fragments_to_net(&segment_00.rows),
                columns_11: core_fragments_to_net(&segment_11.shards),
                rows_11: core_fragments_to_net(&segment_11.rows),
            },
            SourceHalfState::Leading {
                columns_00,
                columns_01,
                rows_00,
                rows_10,
            } => NetSourceHalfState::Leading {
                columns_00: core_fragments_to_net(columns_00),
                columns_01: core_fragments_to_net(columns_01),
                rows_00: core_fragments_to_net(rows_00),
                rows_10: core_fragments_to_net(rows_10),
            },
        })
        .collect()
}

pub(crate) fn net_partial_stitch_evals_to_core(
    evals: Vec<Vec<NetPartialStitchEvals>>,
) -> Vec<Vec<PartialStitchEvals>> {
    evals
        .into_iter()
        .map(|segment| {
            segment
                .into_iter()
                .map(|helper| PartialStitchEvals {
                    registry_index: helper.registry_index as usize,
                    row_evals: helper.row_evals.0,
                    column_evals: helper.column_evals.0,
                })
                .collect()
        })
        .collect()
}

pub(crate) fn core_partial_stitch_evals_to_net(
    evals: Vec<Vec<PartialStitchEvals>>,
) -> Vec<Vec<NetPartialStitchEvals>> {
    evals
        .into_iter()
        .map(|segment| {
            segment
                .into_iter()
                .map(|helper| NetPartialStitchEvals {
                    registry_index: helper.registry_index as u32,
                    row_evals: NetFpBlob(helper.row_evals),
                    column_evals: NetFpBlob(helper.column_evals),
                })
                .collect()
        })
        .collect()
}
