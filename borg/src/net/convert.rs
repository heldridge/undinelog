//! Shared conversions between wire and core shard share types.

use crate::net::wire::{
    NetFp, NetHelperRowMigration, NetKappaDoubleRowEvalMigration, NetPartnerShardMigration,
};
use crate::unauth::{HelperRowMigration, KappaDoubleRowEvalMigration, PartnerShardMigration};

pub(crate) fn net_partner_shards_to_core(
    partners: &[NetPartnerShardMigration],
) -> Vec<PartnerShardMigration> {
    partners
        .iter()
        .map(|partner| PartnerShardMigration {
            registry_index: partner.registry_index as usize,
            segments: partner
                .segments
                .iter()
                .map(|segment| segment.iter().map(|value| value.0).collect())
                .collect(),
        })
        .collect()
}

pub(crate) fn core_partner_shards_to_net(
    partners: &[PartnerShardMigration],
) -> Vec<NetPartnerShardMigration> {
    partners
        .iter()
        .map(|partner| NetPartnerShardMigration {
            registry_index: partner.registry_index as u32,
            segments: partner
                .segments
                .iter()
                .map(|segment| segment.iter().copied().map(NetFp).collect())
                .collect(),
        })
        .collect()
}

pub(crate) fn net_helper_rows_to_core(rows: &[NetHelperRowMigration]) -> Vec<HelperRowMigration> {
    rows.iter()
        .map(|row| HelperRowMigration {
            row_label: row.row_label.0,
            segments: row
                .segments
                .iter()
                .map(|segment| segment.iter().map(|value| value.0).collect())
                .collect(),
        })
        .collect()
}

pub(crate) fn core_helper_rows_to_net(rows: &[HelperRowMigration]) -> Vec<NetHelperRowMigration> {
    rows.iter()
        .map(|row| NetHelperRowMigration {
            row_label: NetFp(row.row_label),
            segments: row
                .segments
                .iter()
                .map(|segment| segment.iter().copied().map(NetFp).collect())
                .collect(),
        })
        .collect()
}

pub(crate) fn net_kappa_double_row_evals_to_core(
    helpers: &[NetKappaDoubleRowEvalMigration],
) -> Vec<KappaDoubleRowEvalMigration> {
    helpers
        .iter()
        .map(|helper| KappaDoubleRowEvalMigration {
            registry_index: helper.registry_index as usize,
            segments: helper
                .segments
                .iter()
                .map(|segment| segment.iter().map(|value| value.0).collect())
                .collect(),
        })
        .collect()
}

pub(crate) fn core_kappa_double_row_evals_to_net(
    helpers: &[KappaDoubleRowEvalMigration],
) -> Vec<NetKappaDoubleRowEvalMigration> {
    helpers
        .iter()
        .map(|helper| NetKappaDoubleRowEvalMigration {
            registry_index: helper.registry_index as u32,
            segments: helper
                .segments
                .iter()
                .map(|segment| segment.iter().copied().map(NetFp).collect())
                .collect(),
        })
        .collect()
}
