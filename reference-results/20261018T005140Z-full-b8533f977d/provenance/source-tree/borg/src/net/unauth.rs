//! RPC helpers for the unauthenticated layer.
//!
//! This module provides network-facing clients and RPC handlers for the
//! unauthenticated coded-array storage. Core data types live in `crate::unauth`.

use crate::field::Fp;
use crate::net::convert::{
    core_helper_rows_to_net, core_kappa_double_row_evals_to_net, core_partner_shards_to_net,
    net_helper_rows_to_core, net_kappa_double_row_evals_to_core, net_partner_shards_to_core,
};
use crate::net::registry::{
    MockRegistryFile, PartyInfo, build_unauth_registry_view, claim_bundles_for_party,
};
use crate::net::wire::{
    NetDeltaMode, NetFp, NetFpBlob, NetRowReadRequest,
    UnauthOptimizedReadRequest as NetUnauthOptimizedReadRequest,
    UnauthOptimizedReadResponse as NetUnauthOptimizedReadResponse,
    UnauthReadObjectContribution as NetUnauthReadObjectContribution,
    UnauthReadObjectId as NetUnauthReadObjectId,
};
use crate::net::{Message, Response};
use crate::shares::{RowShare, SegmentPrimaryShard, ShardShare};
pub use crate::unauth::client::ClientError;
use crate::unauth::client::{UnauthClient as CoreUnauthClient, UnauthTransport};
use crate::unauth::service::{UnauthRequest, UnauthResponse, UnauthService};
use crate::unauth::{DeltaMode, Index, public_values_per_segment};
use crate::value::encode_value;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone)]
pub struct NetUnauthTransport {
    row_labels_by_party_id: Arc<std::collections::HashMap<u64, Vec<Fp>>>,
    owned_indices_by_party_id: Arc<std::collections::HashMap<u64, Vec<usize>>>,
}

impl Default for NetUnauthTransport {
    fn default() -> Self {
        Self {
            row_labels_by_party_id: Arc::new(std::collections::HashMap::new()),
            owned_indices_by_party_id: Arc::new(std::collections::HashMap::new()),
        }
    }
}

pub type UnauthClient = CoreUnauthClient<NetUnauthTransport>;

pub fn is_read_only_message(msg: &Message) -> bool {
    matches!(
        msg,
        Message::UnauthRead { .. }
            | Message::UnauthReadBlock { .. }
            | Message::UnauthReadSegmentRaw { .. }
            | Message::UnauthReadRowsAtZ { .. }
            | Message::UnauthReadRowsBatchAtZ { .. }
            | Message::UnauthReadRowsMatrix { .. }
            | Message::UnauthOptimizedRead { .. }
            | Message::UnauthMultiRead { .. }
            | Message::UnauthRepairShard { .. }
            | Message::UnauthRepairShardBatch { .. }
            | Message::UnauthEvalShardsAtRowLabels { .. }
            | Message::UnauthEvalShardsAtRowLabelsMatrix { .. }
            | Message::UnauthEvalKappaDoubleTargetAtRowLabelsMatrix { .. }
            | Message::UnauthGetLen
    )
}

pub fn is_auth_mode_read_only_message(msg: &Message) -> bool {
    matches!(
        msg,
        Message::UnauthReadRowsAtZ { .. }
            | Message::UnauthReadSegmentRaw { .. }
            | Message::UnauthReadRowsBatchAtZ { .. }
            | Message::UnauthReadRowsMatrix { .. }
            | Message::UnauthOptimizedRead { .. }
            | Message::UnauthRepairShard { .. }
            | Message::UnauthRepairShardBatch { .. }
            | Message::UnauthEvalShardsAtRowLabels { .. }
            | Message::UnauthEvalShardsAtRowLabelsMatrix { .. }
            | Message::UnauthEvalKappaDoubleTargetAtRowLabelsMatrix { .. }
            | Message::UnauthGetLen
    )
}

fn pack_row_batch_reply(batches: Vec<Vec<RowShare>>) -> crate::net::wire::NetRowBatchReply {
    let mut flat_values = Vec::new();
    for batch in batches {
        flat_values.extend(batch.into_iter().map(|row| row.value()));
    }
    crate::net::wire::NetRowBatchReply {
        values: NetFpBlob(flat_values),
    }
}

fn pack_row_matrix_reply(rows: Vec<Vec<Vec<RowShare>>>) -> crate::net::wire::NetRowMatrixReply {
    let mut flat_values = Vec::new();
    for req_rows in rows {
        for batch in req_rows {
            flat_values.extend(batch.into_iter().map(|row| row.value()));
        }
    }
    crate::net::wire::NetRowMatrixReply {
        values: NetFpBlob(flat_values),
    }
}

pub(crate) fn derived_row_labels_for_owned_indices(
    params: &crate::unauth::CodedArrayParams,
    owned_indices: &[usize],
) -> Vec<Fp> {
    let mut labels: Vec<_> = owned_indices
        .iter()
        .flat_map(|&registry_index| params.row_label_indices_for_registry_index(registry_index))
        .filter_map(|row_label_index| params.row_labels.get(row_label_index).copied())
        .collect();
    labels.sort_by_key(|row_label| encode_value(*row_label));
    labels.dedup_by_key(|row_label| encode_value(*row_label));
    labels
}

fn decode_row_values_batch(
    row_labels: &[Fp],
    batch_count: usize,
    values: NetFpBlob,
) -> Result<Vec<Vec<RowShare>>, ClientError> {
    let row_count = row_labels.len();
    let expected = row_count * batch_count;
    if values.0.len() != expected {
        return Err(ClientError::InvalidResponse(format!(
            "row batch length mismatch: expected {} values, got {}",
            expected,
            values.0.len()
        )));
    }
    Ok(values
        .0
        .chunks_exact(row_count)
        .map(|chunk| {
            row_labels
                .iter()
                .copied()
                .zip(chunk.iter().copied())
                .map(|(row_label, value)| RowShare { row_label, value })
                .collect()
        })
        .collect())
}

fn decode_row_values_matrix(
    row_labels: &[Fp],
    batch_counts: &[usize],
    values: NetFpBlob,
) -> Result<Vec<Vec<Vec<RowShare>>>, ClientError> {
    let row_count = row_labels.len();
    let expected_batches: usize = batch_counts.iter().sum();
    let expected_values = row_count * expected_batches;
    if values.0.len() != expected_values {
        return Err(ClientError::InvalidResponse(format!(
            "row matrix length mismatch: expected {} values, got {}",
            expected_values,
            values.0.len()
        )));
    }
    let mut offset = 0usize;
    let mut out = Vec::with_capacity(batch_counts.len());
    for &batch_count in batch_counts {
        let mut req_rows = Vec::with_capacity(batch_count);
        for _ in 0..batch_count {
            let end = offset + row_count;
            let chunk = &values.0[offset..end];
            req_rows.push(
                row_labels
                    .iter()
                    .copied()
                    .zip(chunk.iter().copied())
                    .map(|(row_label, value)| RowShare { row_label, value })
                    .collect(),
            );
            offset = end;
        }
        out.push(req_rows);
    }
    Ok(out)
}

fn pack_eval_matrix_reply(values: Vec<Vec<Vec<Fp>>>) -> NetFpBlob {
    let mut flat_values = Vec::new();
    for per_seg in values {
        for batch in per_seg {
            flat_values.extend(batch);
        }
    }
    NetFpBlob(flat_values)
}

fn decode_eval_matrix_reply(
    seg_count: usize,
    col_count: usize,
    row_count: usize,
    values: NetFpBlob,
) -> Result<Vec<Vec<Vec<Fp>>>, ClientError> {
    let expected = seg_count
        .checked_mul(col_count)
        .and_then(|v| v.checked_mul(row_count))
        .ok_or_else(|| ClientError::InvalidResponse("eval matrix size overflow".to_string()))?;
    if values.0.len() != expected {
        return Err(ClientError::InvalidResponse(format!(
            "eval matrix length mismatch: expected {} values, got {}",
            expected,
            values.0.len()
        )));
    }
    let mut offset = 0usize;
    let mut out = Vec::with_capacity(seg_count);
    for _ in 0..seg_count {
        let mut per_seg = Vec::with_capacity(col_count);
        for _ in 0..col_count {
            let end = offset + row_count;
            per_seg.push(values.0[offset..end].to_vec());
            offset = end;
        }
        out.push(per_seg);
    }
    Ok(out)
}

impl NetUnauthTransport {
    pub fn with_party_metadata(
        row_labels_by_party_id: std::collections::HashMap<u64, Vec<Fp>>,
        owned_indices_by_party_id: std::collections::HashMap<u64, Vec<usize>>,
    ) -> Self {
        Self {
            row_labels_by_party_id: Arc::new(row_labels_by_party_id),
            owned_indices_by_party_id: Arc::new(
                owned_indices_by_party_id
                    .into_iter()
                    .map(|(party_id, mut indices)| {
                        indices.sort_unstable();
                        indices.dedup();
                        (party_id, indices)
                    })
                    .collect(),
            ),
        }
    }

    pub fn for_single_party(party_id: u64, row_labels: Vec<Fp>, owned_indices: Vec<usize>) -> Self {
        let mut row_labels_by_party_id = std::collections::HashMap::new();
        row_labels_by_party_id.insert(party_id, row_labels);
        let mut owned_indices_by_party_id = std::collections::HashMap::new();
        owned_indices_by_party_id.insert(party_id, owned_indices);
        Self::with_party_metadata(row_labels_by_party_id, owned_indices_by_party_id)
    }

    fn owned_indices_for_server(&self, server: &PartyInfo) -> Result<&[usize], ClientError> {
        self.owned_indices_by_party_id
            .get(&server.party_id)
            .map(|indices| indices.as_slice())
            .ok_or_else(|| {
                ClientError::InvalidResponse(format!(
                    "missing derived shard ownership for helper party {}",
                    server.party_id
                ))
            })
    }

    fn row_labels_for_server(&self, server: &PartyInfo) -> Result<&[Fp], ClientError> {
        self.row_labels_by_party_id
            .get(&server.party_id)
            .map(|labels| labels.as_slice())
            .ok_or_else(|| {
                ClientError::InvalidResponse(format!(
                    "missing derived row labels for helper party {}",
                    server.party_id
                ))
            })
    }
}

fn pack_ordered_shard_values(shares: &[ShardShare]) -> crate::net::wire::NetMaybeFpBlob {
    let mut sorted = shares.to_vec();
    sorted.sort_unstable_by_key(|share| share.registry_index);
    crate::net::wire::NetMaybeFpBlob(sorted.into_iter().map(|share| share.value).collect())
}

fn decode_ordered_shard_values(
    owned_indices: &[usize],
    values: crate::net::wire::NetMaybeFpBlob,
) -> Result<Vec<ShardShare>, ClientError> {
    if values.0.len() != owned_indices.len() {
        return Err(ClientError::InvalidResponse(format!(
            "ordered shard length mismatch: expected {} values, got {}",
            owned_indices.len(),
            values.0.len()
        )));
    }
    Ok(owned_indices
        .iter()
        .copied()
        .zip(values.0)
        .map(|(registry_index, value)| ShardShare {
            registry_index,
            value,
        })
        .collect())
}

fn pack_single_row_reply(rows: Vec<RowShare>) -> crate::net::wire::NetRowBatchReply {
    pack_row_batch_reply(vec![rows])
}

fn decode_single_row_values(
    row_labels: &[Fp],
    values: NetFpBlob,
) -> Result<Vec<RowShare>, ClientError> {
    let mut batches = decode_row_values_batch(row_labels, 1, values)?;
    Ok(batches.pop().unwrap_or_default())
}

fn pack_ordered_segment_primary_shards(
    shards: &[SegmentPrimaryShard],
) -> Vec<crate::net::wire::NetSegmentPrimaryShardReply> {
    let mut sorted = shards.to_vec();
    sorted.sort_unstable_by_key(|shard| shard.registry_index);
    sorted
        .into_iter()
        .map(|shard| crate::net::wire::NetSegmentPrimaryShardReply {
            values: NetFpBlob(shard.values),
        })
        .collect()
}

fn decode_ordered_segment_primary_shards(
    owned_indices: &[usize],
    shards: Vec<crate::net::wire::NetSegmentPrimaryShardReply>,
) -> Result<Vec<SegmentPrimaryShard>, ClientError> {
    if shards.len() != owned_indices.len() {
        return Err(ClientError::InvalidResponse(format!(
            "ordered raw-shard length mismatch: expected {} shards, got {}",
            owned_indices.len(),
            shards.len()
        )));
    }
    Ok(owned_indices
        .iter()
        .copied()
        .zip(shards)
        .map(|(registry_index, shard)| SegmentPrimaryShard {
            registry_index,
            values: shard.values.0,
        })
        .collect())
}

pub(crate) fn core_optimized_request_to_net(
    req: &crate::unauth::UnauthOptimizedReadRequest,
) -> Result<NetUnauthOptimizedReadRequest, ClientError> {
    let mut objects = Vec::with_capacity(req.objects.len());
    for object in &req.objects {
        let object = match *object {
            crate::unauth::UnauthReadObjectId::Segment { segment } => {
                NetUnauthReadObjectId::Segment {
                    segment: u64::try_from(segment).map_err(|_| {
                        ClientError::InvalidConfig(
                            "optimized read segment id overflows u64".to_string(),
                        )
                    })?,
                }
            }
            crate::unauth::UnauthReadObjectId::Block { block } => {
                NetUnauthReadObjectId::Block { block }
            }
            crate::unauth::UnauthReadObjectId::Point { array_index } => {
                NetUnauthReadObjectId::Point { array_index }
            }
        };
        objects.push(object);
    }
    Ok(NetUnauthOptimizedReadRequest { objects })
}

pub(crate) fn net_optimized_request_to_core(
    req: &NetUnauthOptimizedReadRequest,
) -> Result<crate::unauth::UnauthOptimizedReadRequest, ClientError> {
    let mut objects = Vec::with_capacity(req.objects.len());
    for object in &req.objects {
        let object = match *object {
            NetUnauthReadObjectId::Segment { segment } => {
                crate::unauth::UnauthReadObjectId::Segment {
                    segment: usize::try_from(segment).map_err(|_| {
                        ClientError::InvalidResponse(
                            "optimized read segment id overflows usize".to_string(),
                        )
                    })?,
                }
            }
            NetUnauthReadObjectId::Block { block } => {
                crate::unauth::UnauthReadObjectId::Block { block }
            }
            NetUnauthReadObjectId::Point { array_index } => {
                crate::unauth::UnauthReadObjectId::Point { array_index }
            }
        };
        objects.push(object);
    }
    Ok(crate::unauth::UnauthOptimizedReadRequest { objects })
}

pub fn core_optimized_response_to_net(
    response: crate::unauth::UnauthOptimizedReadResponse,
) -> Result<NetUnauthOptimizedReadResponse, ClientError> {
    let mut objects = Vec::with_capacity(response.objects.len());
    for contribution in response.objects {
        let contribution = match contribution {
            crate::unauth::UnauthReadObjectContribution::Segment { object_pos, shards } => {
                NetUnauthReadObjectContribution::Segment {
                    object_pos: u32::try_from(object_pos).map_err(|_| {
                        ClientError::InvalidResponse(
                            "optimized read object position overflows u32".to_string(),
                        )
                    })?,
                    shards: pack_ordered_segment_primary_shards(&shards),
                }
            }
            crate::unauth::UnauthReadObjectContribution::Block { object_pos, rows } => {
                NetUnauthReadObjectContribution::Block {
                    object_pos: u32::try_from(object_pos).map_err(|_| {
                        ClientError::InvalidResponse(
                            "optimized read object position overflows u32".to_string(),
                        )
                    })?,
                    rows: pack_single_row_reply(rows),
                }
            }
            crate::unauth::UnauthReadObjectContribution::Point { object_pos, shares } => {
                NetUnauthReadObjectContribution::Point {
                    object_pos: u32::try_from(object_pos).map_err(|_| {
                        ClientError::InvalidResponse(
                            "optimized read object position overflows u32".to_string(),
                        )
                    })?,
                    shares: pack_ordered_shard_values(&shares),
                }
            }
        };
        objects.push(contribution);
    }
    Ok(NetUnauthOptimizedReadResponse { objects })
}

pub(crate) fn net_optimized_response_to_core(
    owned_indices: Option<&[usize]>,
    row_labels: Option<&[Fp]>,
    response: NetUnauthOptimizedReadResponse,
) -> Result<crate::unauth::UnauthOptimizedReadResponse, ClientError> {
    let mut objects = Vec::with_capacity(response.objects.len());
    for contribution in response.objects {
        let contribution = match contribution {
            NetUnauthReadObjectContribution::Segment { object_pos, shards } => {
                let Some(owned_indices) = owned_indices else {
                    continue;
                };
                crate::unauth::UnauthReadObjectContribution::Segment {
                    object_pos: object_pos as usize,
                    shards: decode_ordered_segment_primary_shards(owned_indices, shards)?,
                }
            }
            NetUnauthReadObjectContribution::Block { object_pos, rows } => {
                let Some(row_labels) = row_labels else {
                    continue;
                };
                crate::unauth::UnauthReadObjectContribution::Block {
                    object_pos: object_pos as usize,
                    rows: decode_single_row_values(row_labels, rows.values)?,
                }
            }
            NetUnauthReadObjectContribution::Point { object_pos, shares } => {
                let Some(owned_indices) = owned_indices else {
                    continue;
                };
                crate::unauth::UnauthReadObjectContribution::Point {
                    object_pos: object_pos as usize,
                    shares: decode_ordered_shard_values(owned_indices, shares)?,
                }
            }
        };
        objects.push(contribution);
    }
    Ok(crate::unauth::UnauthOptimizedReadResponse {
        helper_index: None,
        objects,
    })
}

pub fn handle_read_only_message(
    service: &crate::unauth::UnauthStore,
    msg: Message,
) -> Option<Response> {
    let response = match msg {
        Message::UnauthRead { idx } => Response::UnauthRead {
            values: pack_ordered_shard_values(&service.handle_read(idx)),
        },
        Message::UnauthReadBlock { block } => Response::UnauthReadBlock {
            rows: pack_single_row_reply(service.handle_read_block(block)),
        },
        Message::UnauthReadSegmentRaw { seg } => Response::UnauthReadSegmentRaw {
            shards: pack_ordered_segment_primary_shards(
                &service.handle_read_segment_raw(seg as usize),
            ),
        },
        Message::UnauthReadRowsAtZ { seg, z_eval } => Response::UnauthReadRowsAtZ {
            rows: pack_single_row_reply(service.handle_read_rows_at_z(seg as usize, z_eval.0)),
        },
        Message::UnauthReadRowsBatchAtZ { seg, z_evals } => Response::UnauthReadRowsBatchAtZ {
            rows: pack_row_batch_reply(
                service.handle_read_rows_batch_at_z(seg as usize, &z_evals.0),
            ),
        },
        Message::UnauthReadRowsMatrix { requests } => Response::UnauthReadRowsMatrix {
            rows: pack_row_matrix_reply(
                service.handle_read_rows_matrix(
                    &requests
                        .into_iter()
                        .map(|request| crate::unauth::service::RowReadRequest {
                            seg: request.seg as usize,
                            z_evals: request.z_evals.0,
                        })
                        .collect::<Vec<_>>(),
                ),
            ),
        },
        Message::UnauthOptimizedRead { req } => {
            let resp = net_optimized_request_to_core(&req)
                .map(|core_req| crate::unauth::handle_optimized_read_request(service, &core_req))
                .and_then(core_optimized_response_to_net);
            match resp {
                Ok(resp) => Response::UnauthOptimizedRead { resp },
                Err(err) => Response::Error {
                    message: format!("unauth optimized read failed: {:?}", err),
                },
            }
        }
        Message::UnauthMultiRead { idxs } => Response::UnauthMultiRead {
            values: service
                .handle_multi_read(&idxs)
                .into_iter()
                .map(|batch| pack_ordered_shard_values(&batch))
                .collect(),
        },
        Message::UnauthRepairShard {
            seg,
            registry_index,
        } => match service.handle_repair_shard(seg as usize, registry_index as usize) {
            Ok(values) => Response::UnauthRepairShard {
                values: values.into_iter().map(NetFp).collect(),
            },
            Err(err) => Response::Error {
                message: format!("unauth repair failed: {:?}", err),
            },
        },
        Message::UnauthRepairShardBatch {
            seg,
            registry_indices,
        } => match service.handle_repair_shard_batch(
            seg as usize,
            &registry_indices
                .into_iter()
                .map(|value| value as usize)
                .collect::<Vec<_>>(),
        ) {
            Ok(values) => Response::UnauthRepairShardBatch {
                values: values
                    .into_iter()
                    .map(|batch| batch.into_iter().map(NetFp).collect())
                    .collect(),
            },
            Err(err) => Response::Error {
                message: format!("unauth batch repair failed: {:?}", err),
            },
        },
        Message::UnauthEvalShardsAtRowLabels {
            seg,
            col_labels,
            row_labels,
        } => match service.handle_eval_shards_at_row_labels(
            seg as usize,
            &col_labels.0,
            &row_labels.0,
        ) {
            Ok(values) => Response::UnauthEvalShardsAtRowLabels {
                values: values.into_iter().map(NetFpBlob).collect(),
            },
            Err(err) => Response::Error {
                message: format!("unauth shard-eval failed: {:?}", err),
            },
        },
        Message::UnauthEvalShardsAtRowLabelsMatrix {
            segs,
            col_labels,
            row_labels,
        } => match service.handle_eval_shards_at_row_labels_matrix(
            &segs.into_iter().map(|seg| seg as usize).collect::<Vec<_>>(),
            &col_labels.0,
            &row_labels.0,
        ) {
            Ok(values) => Response::UnauthEvalShardsAtRowLabelsMatrix {
                values: pack_eval_matrix_reply(values),
            },
            Err(err) => Response::Error {
                message: format!("unauth shard-eval matrix failed: {:?}", err),
            },
        },
        Message::UnauthEvalKappaDoubleTargetAtRowLabelsMatrix {
            target_s,
            segs,
            target_registry_indices,
            row_labels,
        } => match service.handle_eval_kappa_double_target_at_row_labels_matrix(
            target_s as usize,
            &segs.into_iter().map(|seg| seg as usize).collect::<Vec<_>>(),
            &target_registry_indices
                .into_iter()
                .map(|idx| idx as usize)
                .collect::<Vec<_>>(),
            &row_labels.0,
        ) {
            Ok(values) => Response::UnauthEvalKappaDoubleTargetAtRowLabelsMatrix {
                values: pack_eval_matrix_reply(values),
            },
            Err(err) => Response::Error {
                message: format!("unauth kappa-double target eval failed: {:?}", err),
            },
        },
        Message::UnauthGetLen => Response::UnauthLen { len: service.len() },
        _ => return None,
    };
    Some(response)
}

pub fn handle_auth_mode_read_only_message(
    service: &crate::auth::service::AuthServer,
    msg: Message,
) -> Option<Response> {
    let response = match msg {
        Message::UnauthReadRowsAtZ { seg, z_eval } => Response::UnauthReadRowsAtZ {
            rows: pack_single_row_reply(
                service
                    .store
                    .unauth
                    .handle_read_rows_at_z(seg as usize, z_eval.0),
            ),
        },
        Message::UnauthReadSegmentRaw { seg } => Response::UnauthReadSegmentRaw {
            shards: pack_ordered_segment_primary_shards(
                &service.store.unauth.handle_read_segment_raw(seg as usize),
            ),
        },
        Message::UnauthReadRowsBatchAtZ { seg, z_evals } => Response::UnauthReadRowsBatchAtZ {
            rows: pack_row_batch_reply(
                service
                    .store
                    .unauth
                    .handle_read_rows_batch_at_z(seg as usize, &z_evals.0),
            ),
        },
        Message::UnauthReadRowsMatrix { requests } => Response::UnauthReadRowsMatrix {
            rows: pack_row_matrix_reply(
                service.store.unauth.handle_read_rows_matrix(
                    &requests
                        .into_iter()
                        .map(|request| crate::unauth::service::RowReadRequest {
                            seg: request.seg as usize,
                            z_evals: request.z_evals.0,
                        })
                        .collect::<Vec<_>>(),
                ),
            ),
        },
        Message::UnauthOptimizedRead { req } => {
            let resp = net_optimized_request_to_core(&req)
                .map(|core_req| {
                    crate::unauth::handle_optimized_read_request(&service.store.unauth, &core_req)
                })
                .and_then(core_optimized_response_to_net);
            match resp {
                Ok(resp) => Response::UnauthOptimizedRead { resp },
                Err(err) => Response::Error {
                    message: format!("unauth optimized read failed: {:?}", err),
                },
            }
        }
        Message::UnauthRepairShard {
            seg,
            registry_index,
        } => match service
            .store
            .unauth
            .handle_repair_shard(seg as usize, registry_index as usize)
        {
            Ok(values) => Response::UnauthRepairShard {
                values: values.into_iter().map(NetFp).collect(),
            },
            Err(err) => Response::Error {
                message: format!("unauth repair failed: {:?}", err),
            },
        },
        Message::UnauthRepairShardBatch {
            seg,
            registry_indices,
        } => match service.store.unauth.handle_repair_shard_batch(
            seg as usize,
            &registry_indices
                .into_iter()
                .map(|value| value as usize)
                .collect::<Vec<_>>(),
        ) {
            Ok(values) => Response::UnauthRepairShardBatch {
                values: values
                    .into_iter()
                    .map(|batch| batch.into_iter().map(NetFp).collect())
                    .collect(),
            },
            Err(err) => Response::Error {
                message: format!("unauth batch repair failed: {:?}", err),
            },
        },
        Message::UnauthEvalShardsAtRowLabels {
            seg,
            col_labels,
            row_labels,
        } => match service.store.unauth.handle_eval_shards_at_row_labels(
            seg as usize,
            &col_labels.0,
            &row_labels.0,
        ) {
            Ok(values) => Response::UnauthEvalShardsAtRowLabels {
                values: values.into_iter().map(NetFpBlob).collect(),
            },
            Err(err) => Response::Error {
                message: format!("unauth shard-eval failed: {:?}", err),
            },
        },
        Message::UnauthEvalShardsAtRowLabelsMatrix {
            segs,
            col_labels,
            row_labels,
        } => match service
            .store
            .unauth
            .handle_eval_shards_at_row_labels_matrix(
                &segs.into_iter().map(|seg| seg as usize).collect::<Vec<_>>(),
                &col_labels.0,
                &row_labels.0,
            ) {
            Ok(values) => Response::UnauthEvalShardsAtRowLabelsMatrix {
                values: pack_eval_matrix_reply(values),
            },
            Err(err) => Response::Error {
                message: format!("unauth shard-eval matrix failed: {:?}", err),
            },
        },
        Message::UnauthEvalKappaDoubleTargetAtRowLabelsMatrix {
            target_s,
            segs,
            target_registry_indices,
            row_labels,
        } => match service
            .store
            .unauth
            .handle_eval_kappa_double_target_at_row_labels_matrix(
                target_s as usize,
                &segs.into_iter().map(|seg| seg as usize).collect::<Vec<_>>(),
                &target_registry_indices
                    .into_iter()
                    .map(|idx| idx as usize)
                    .collect::<Vec<_>>(),
                &row_labels.0,
            ) {
            Ok(values) => Response::UnauthEvalKappaDoubleTargetAtRowLabelsMatrix {
                values: pack_eval_matrix_reply(values),
            },
            Err(err) => Response::Error {
                message: format!("unauth kappa-double target eval failed: {:?}", err),
            },
        },
        Message::UnauthGetLen => Response::UnauthLen {
            len: service.store.unauth.len(),
        },
        _ => return None,
    };
    Some(response)
}

impl UnauthTransport for NetUnauthTransport {
    type Server = PartyInfo;

    fn server_label(&self, server: &Self::Server) -> String {
        server.party_id.to_string()
    }

    fn unauth_get_len(&self, server: &Self::Server) -> Result<u64, ClientError> {
        let resp = crate::net::rpc_call(&server.rpc_host, &Message::UnauthGetLen)
            .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::UnauthLen { len } => Ok(len),
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn unauth_read(
        &self,
        server: &Self::Server,
        idx: Index,
    ) -> Result<Vec<ShardShare>, ClientError> {
        let resp = crate::net::rpc_call(&server.rpc_host, &Message::UnauthRead { idx })
            .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::UnauthRead { values } => {
                let owned_indices = self.owned_indices_for_server(server)?;
                decode_ordered_shard_values(owned_indices, values)
            }
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn unauth_read_block(
        &self,
        server: &Self::Server,
        block: Index,
    ) -> Result<Vec<RowShare>, ClientError> {
        let resp = crate::net::rpc_call(&server.rpc_host, &Message::UnauthReadBlock { block })
            .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::UnauthReadBlock { rows } => {
                let row_labels = self.row_labels_for_server(server)?;
                decode_single_row_values(row_labels, rows.values)
            }
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn unauth_read_segment_raw(
        &self,
        server: &Self::Server,
        seg: usize,
    ) -> Result<Vec<SegmentPrimaryShard>, ClientError> {
        let resp = crate::net::rpc_call(
            &server.rpc_host,
            &Message::UnauthReadSegmentRaw { seg: seg as u64 },
        )
        .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::UnauthReadSegmentRaw { shards } => {
                let owned_indices = self.owned_indices_for_server(server)?;
                decode_ordered_segment_primary_shards(owned_indices, shards)
            }
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn supports_unauth_optimized_reads(&self) -> bool {
        !self.owned_indices_by_party_id.is_empty() || !self.row_labels_by_party_id.is_empty()
    }

    fn unauth_optimized_read(
        &self,
        server: &Self::Server,
        request: &crate::unauth::UnauthOptimizedReadRequest,
    ) -> Result<crate::unauth::UnauthOptimizedReadResponse, ClientError> {
        let req = core_optimized_request_to_net(request)?;
        let resp = crate::net::rpc_call(&server.rpc_host, &Message::UnauthOptimizedRead { req })
            .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::UnauthOptimizedRead { resp } => {
                let owned_indices = self.owned_indices_for_server(server).ok();
                let row_labels = self.row_labels_for_server(server).ok();
                net_optimized_response_to_core(owned_indices, row_labels, resp)
            }
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn unauth_read_rows_at_z(
        &self,
        server: &Self::Server,
        seg: usize,
        z_eval: Fp,
    ) -> Result<Vec<RowShare>, ClientError> {
        let resp = crate::net::rpc_call(
            &server.rpc_host,
            &Message::UnauthReadRowsAtZ {
                seg: seg as u64,
                z_eval: NetFp(z_eval),
            },
        )
        .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::UnauthReadRowsAtZ { rows } => {
                let row_labels = self.row_labels_for_server(server)?;
                decode_single_row_values(row_labels, rows.values)
            }
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn unauth_read_rows_batch_at_z(
        &self,
        server: &Self::Server,
        seg: usize,
        z_evals: &[Fp],
    ) -> Result<Vec<Vec<RowShare>>, ClientError> {
        let resp = crate::net::rpc_call(
            &server.rpc_host,
            &Message::UnauthReadRowsBatchAtZ {
                seg: seg as u64,
                z_evals: NetFpBlob(z_evals.to_vec()),
            },
        )
        .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::UnauthReadRowsBatchAtZ { rows } => {
                let row_labels = self.row_labels_for_server(server)?;
                decode_row_values_batch(row_labels, z_evals.len(), rows.values)
            }
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn unauth_read_rows_matrix(
        &self,
        server: &Self::Server,
        requests: &[(usize, Vec<Fp>)],
    ) -> Result<Vec<Vec<Vec<RowShare>>>, ClientError> {
        let resp = crate::net::rpc_call(
            &server.rpc_host,
            &Message::UnauthReadRowsMatrix {
                requests: requests
                    .iter()
                    .map(|(seg, z_evals)| NetRowReadRequest {
                        seg: *seg as u64,
                        z_evals: NetFpBlob(z_evals.clone()),
                    })
                    .collect(),
            },
        )
        .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::UnauthReadRowsMatrix { rows } => {
                let row_labels = self.row_labels_for_server(server)?;
                decode_row_values_matrix(
                    row_labels,
                    &requests
                        .iter()
                        .map(|(_, z_evals)| z_evals.len())
                        .collect::<Vec<_>>(),
                    rows.values,
                )
            }
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn unauth_multi_read(
        &self,
        server: &Self::Server,
        idxs: &[Index],
    ) -> Result<Vec<Vec<ShardShare>>, ClientError> {
        let resp = crate::net::rpc_call(
            &server.rpc_host,
            &Message::UnauthMultiRead {
                idxs: idxs.to_vec(),
            },
        )
        .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::UnauthMultiRead { values } => {
                let owned_indices = self.owned_indices_for_server(server)?;
                values
                    .into_iter()
                    .map(|batch| decode_ordered_shard_values(owned_indices, batch))
                    .collect()
            }
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn unauth_repair_shard(
        &self,
        server: &Self::Server,
        seg: usize,
        registry_index: usize,
    ) -> Result<Vec<Fp>, ClientError> {
        let resp = crate::net::rpc_call(
            &server.rpc_host,
            &Message::UnauthRepairShard {
                seg: seg as u64,
                registry_index: registry_index as u32,
            },
        )
        .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::UnauthRepairShard { values } => {
                Ok(values.into_iter().map(|value| value.0).collect())
            }
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn unauth_repair_shard_batch(
        &self,
        server: &Self::Server,
        seg: usize,
        registry_indices: &[usize],
    ) -> Result<Vec<Vec<Fp>>, ClientError> {
        let resp = crate::net::rpc_call(
            &server.rpc_host,
            &Message::UnauthRepairShardBatch {
                seg: seg as u64,
                registry_indices: registry_indices.iter().map(|&value| value as u32).collect(),
            },
        )
        .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::UnauthRepairShardBatch { values } => Ok(values
                .into_iter()
                .map(|batch| batch.into_iter().map(|value| value.0).collect())
                .collect()),
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn unauth_eval_shards_at_row_labels(
        &self,
        server: &Self::Server,
        seg: usize,
        col_labels: &[Fp],
        row_labels: &[Fp],
    ) -> Result<Vec<Vec<Fp>>, ClientError> {
        let resp = crate::net::rpc_call(
            &server.rpc_host,
            &Message::UnauthEvalShardsAtRowLabels {
                seg: seg as u64,
                col_labels: NetFpBlob(col_labels.to_vec()),
                row_labels: NetFpBlob(row_labels.to_vec()),
            },
        )
        .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::UnauthEvalShardsAtRowLabels { values } => {
                Ok(values.into_iter().map(|batch| batch.0).collect())
            }
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn unauth_eval_shards_at_row_labels_matrix(
        &self,
        server: &Self::Server,
        segs: &[usize],
        col_labels: &[Fp],
        row_labels: &[Fp],
    ) -> Result<Vec<Vec<Vec<Fp>>>, ClientError> {
        let resp = crate::net::rpc_call(
            &server.rpc_host,
            &Message::UnauthEvalShardsAtRowLabelsMatrix {
                segs: segs.iter().map(|&seg| seg as u64).collect(),
                col_labels: NetFpBlob(col_labels.to_vec()),
                row_labels: NetFpBlob(row_labels.to_vec()),
            },
        )
        .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::UnauthEvalShardsAtRowLabelsMatrix { values } => {
                decode_eval_matrix_reply(segs.len(), col_labels.len(), row_labels.len(), values)
            }
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn unauth_apply_deltas(
        &self,
        server: &Self::Server,
        deltas: &[(Index, Fp)],
        mode: DeltaMode,
    ) -> Result<bool, ClientError> {
        let payload: Vec<(u64, NetFp)> = deltas.iter().map(|(i, d)| (*i, NetFp(*d))).collect();
        let req = Message::UnauthApplyDeltas {
            deltas: payload,
            mode: NetDeltaMode::from(mode),
        };
        let resp = crate::net::rpc_call(&server.rpc_host, &req)
            .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::UnauthApplyDeltas { ok } => Ok(ok),
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn unauth_apply_block_deltas(
        &self,
        server: &Self::Server,
        block: Index,
        deltas: &[(usize, Fp)],
        mode: DeltaMode,
    ) -> Result<bool, ClientError> {
        let payload: Vec<(u32, NetFp)> = deltas
            .iter()
            .map(|(offset, delta)| (*offset as u32, NetFp(*delta)))
            .collect();
        let req = Message::UnauthApplyBlockDeltas {
            block,
            deltas: payload,
            mode: NetDeltaMode::from(mode),
        };
        let resp = crate::net::rpc_call(&server.rpc_host, &req)
            .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::UnauthApplyBlockDeltas { ok } => Ok(ok),
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn unauth_extend(&self, server: &Self::Server, new_len: Index) -> Result<bool, ClientError> {
        let resp = crate::net::rpc_call(&server.rpc_host, &Message::UnauthExtend { new_len })
            .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::UnauthExtend { ok } => Ok(ok),
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn unauth_migrate(
        &self,
        server: &Self::Server,
        target_s: usize,
        partner_shards: &[crate::unauth::PartnerShardMigration],
        helper_rows: &[crate::unauth::HelperRowMigration],
    ) -> Result<bool, ClientError> {
        // The wire shape stays the same across migration directions:
        // kappa-halve sends source-indexed partner columns plus helper rows,
        // kappa-double sends target-indexed helper columns for eager row hydration.
        let resp = crate::net::rpc_call(
            &server.rpc_host,
            &Message::UnauthMigrate {
                target_s: target_s as u64,
                partner_shards: core_partner_shards_to_net(partner_shards),
                helper_rows: core_helper_rows_to_net(helper_rows),
            },
        )
        .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::UnauthMigrate { ok } => Ok(ok),
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn unauth_segment_double(
        &self,
        server: &Self::Server,
        helper_shards: &[crate::unauth::PartnerShardMigration],
    ) -> Result<bool, ClientError> {
        let resp = crate::net::rpc_call(
            &server.rpc_host,
            &Message::UnauthSegmentDouble {
                helper_shards: core_partner_shards_to_net(helper_shards),
            },
        )
        .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::UnauthSegmentDouble { ok } => Ok(ok),
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn unauth_kappa_double_one_claim_left(
        &self,
        server: &Self::Server,
        target_s: usize,
        helper_evals: &[crate::unauth::KappaDoubleRowEvalMigration],
    ) -> Result<bool, ClientError> {
        let resp = crate::net::rpc_call(
            &server.rpc_host,
            &Message::UnauthKappaDoubleOneClaimLeft {
                target_s: target_s as u64,
                helper_evals: core_kappa_double_row_evals_to_net(helper_evals),
            },
        )
        .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::UnauthKappaDoubleOneClaimLeft { ok } => Ok(ok),
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn unauth_eval_kappa_double_target_at_row_labels_matrix(
        &self,
        server: &Self::Server,
        target_s: usize,
        segs: &[usize],
        target_registry_indices: &[usize],
        row_labels: &[Fp],
    ) -> Result<Vec<Vec<Vec<Fp>>>, ClientError> {
        let resp = crate::net::rpc_call(
            &server.rpc_host,
            &Message::UnauthEvalKappaDoubleTargetAtRowLabelsMatrix {
                target_s: target_s as u64,
                segs: segs.iter().map(|&seg| seg as u64).collect(),
                target_registry_indices: target_registry_indices
                    .iter()
                    .map(|&idx| idx as u32)
                    .collect(),
                row_labels: NetFpBlob(row_labels.to_vec()),
            },
        )
        .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::UnauthEvalKappaDoubleTargetAtRowLabelsMatrix { values } => {
                decode_eval_matrix_reply(
                    segs.len(),
                    target_registry_indices.len(),
                    row_labels.len(),
                    values,
                )
            }
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }

    fn unauth_retain_claims(
        &self,
        server: &Self::Server,
        registry_indices: &[usize],
    ) -> Result<bool, ClientError> {
        let registry_indices: Vec<u32> = registry_indices
            .iter()
            .map(|&idx| {
                u32::try_from(idx).map_err(|_| {
                    ClientError::InvalidConfig("registry index overflows u32".to_string())
                })
            })
            .collect::<Result<_, _>>()?;
        let resp = crate::net::rpc_call(
            &server.rpc_host,
            &Message::UnauthRetainClaims { registry_indices },
        )
        .map_err(|err| ClientError::Net(format!("{:?}", err)))?;
        match resp {
            Response::UnauthRetainClaims { ok } => Ok(ok),
            Response::Error { message } => Err(ClientError::Net(message)),
            other => Err(ClientError::InvalidResponse(format!(
                "invalid response: {:?}",
                other
            ))),
        }
    }
}

impl CoreUnauthClient<NetUnauthTransport> {
    /// Construct a client from a registry JSON file.
    pub fn from_registry_file(path: &Path) -> Result<Self, ClientError> {
        let file = MockRegistryFile::load(path)
            .map_err(|err| ClientError::Registry(format!("{:?}", err)))?;
        Self::from_registry(&file)
    }

    /// Construct a client from an in-memory registry file.
    pub fn from_registry(file: &MockRegistryFile) -> Result<Self, ClientError> {
        let (params, claims) = build_unauth_registry_view(file)
            .map_err(|err| ClientError::Registry(format!("{:?}", err)))?;
        let mut servers = file.parties.clone();
        servers.retain(|p| p.active);
        servers.sort_by_key(|p| p.party_id);
        let server = servers
            .first()
            .cloned()
            .ok_or_else(|| ClientError::Registry("no active unauth servers".to_string()))?;
        let len = file
            .system
            .unauth_array_len
            .unwrap_or(public_values_per_segment(&params) as u64);
        let mut server_owned_indices = Vec::with_capacity(servers.len());
        let mut row_labels_by_party_id = std::collections::HashMap::new();
        let mut owned_indices_by_party_id = std::collections::HashMap::new();
        for party in &servers {
            let claims = claim_bundles_for_party(&params, &claims, party.party_id)
                .map_err(|err| ClientError::Registry(format!("{:?}", err)))?;
            let owned_indices: Vec<_> = claims
                .into_iter()
                .map(|claim| claim.registry_index)
                .collect();
            owned_indices_by_party_id.insert(party.party_id, owned_indices.clone());
            row_labels_by_party_id.insert(
                party.party_id,
                derived_row_labels_for_owned_indices(&params, &owned_indices),
            );
            server_owned_indices.push(owned_indices);
        }
        Ok(Self::new_with_owned_indices(
            NetUnauthTransport::with_party_metadata(
                row_labels_by_party_id,
                owned_indices_by_party_id,
            ),
            params,
            server,
            servers,
            server_owned_indices,
            len,
        ))
    }
}

/// Handle a generic network message for the unauthenticated mode.
pub fn handle_message<S: UnauthService>(service: &mut S, msg: Message) -> Response {
    let req = match msg {
        Message::UnauthRead { idx } => UnauthRequest::Read { idx },
        Message::UnauthReadBlock { block } => UnauthRequest::ReadBlock { block },
        Message::UnauthReadSegmentRaw { seg } => {
            UnauthRequest::ReadSegmentRaw { seg: seg as usize }
        }
        Message::UnauthReadRowsAtZ { seg, z_eval } => UnauthRequest::ReadRowsAtZ {
            seg: seg as usize,
            z_eval: z_eval.0,
        },
        Message::UnauthReadRowsBatchAtZ { seg, z_evals } => UnauthRequest::ReadRowsBatchAtZ {
            seg: seg as usize,
            z_evals: z_evals.0,
        },
        Message::UnauthReadRowsMatrix { requests } => UnauthRequest::ReadRowsMatrix {
            requests: requests
                .into_iter()
                .map(|request| crate::unauth::service::RowReadRequest {
                    seg: request.seg as usize,
                    z_evals: request.z_evals.0,
                })
                .collect(),
        },
        Message::UnauthMultiRead { idxs } => UnauthRequest::MultiRead { idxs },
        Message::UnauthRepairShard {
            seg,
            registry_index,
        } => UnauthRequest::RepairShard {
            seg: seg as usize,
            registry_index: registry_index as usize,
        },
        Message::UnauthRepairShardBatch {
            seg,
            registry_indices,
        } => UnauthRequest::RepairShardBatch {
            seg: seg as usize,
            registry_indices: registry_indices
                .into_iter()
                .map(|registry_index| registry_index as usize)
                .collect(),
        },
        Message::UnauthEvalShardsAtRowLabels {
            seg,
            col_labels,
            row_labels,
        } => UnauthRequest::EvalShardsAtRowLabels {
            seg: seg as usize,
            col_labels: col_labels.0,
            row_labels: row_labels.0,
        },
        Message::UnauthEvalShardsAtRowLabelsMatrix {
            segs,
            col_labels,
            row_labels,
        } => UnauthRequest::EvalShardsAtRowLabelsMatrix {
            segs: segs.into_iter().map(|seg| seg as usize).collect(),
            col_labels: col_labels.0,
            row_labels: row_labels.0,
        },
        Message::UnauthApplyDeltas { deltas, mode } => {
            let deltas_fp: Vec<(u64, Fp)> =
                deltas.into_iter().map(|(idx, fp)| (idx, fp.0)).collect();
            UnauthRequest::ApplyDeltas {
                deltas: deltas_fp,
                mode: DeltaMode::from(mode),
            }
        }
        Message::UnauthApplyBlockDeltas {
            block,
            deltas,
            mode,
        } => {
            let deltas_fp: Vec<(usize, Fp)> = deltas
                .into_iter()
                .map(|(offset, fp)| (offset as usize, fp.0))
                .collect();
            UnauthRequest::ApplyBlockDeltas {
                block,
                deltas: deltas_fp,
                mode: DeltaMode::from(mode),
            }
        }
        Message::UnauthExtend { new_len } => UnauthRequest::Extend { new_len },
        Message::UnauthMigrate {
            target_s,
            partner_shards,
            helper_rows,
        } => UnauthRequest::Migrate {
            target_s: target_s as usize,
            partner_shards: net_partner_shards_to_core(&partner_shards),
            helper_rows: net_helper_rows_to_core(&helper_rows),
        },
        Message::UnauthSegmentDouble { helper_shards } => UnauthRequest::SegmentDouble {
            helper_shards: net_partner_shards_to_core(&helper_shards),
        },
        Message::UnauthKappaDoubleOneClaimLeft {
            target_s,
            helper_evals,
        } => UnauthRequest::KappaDoubleOneClaimLeft {
            target_s: target_s as usize,
            helper_evals: net_kappa_double_row_evals_to_core(&helper_evals),
        },
        Message::UnauthEvalKappaDoubleTargetAtRowLabelsMatrix {
            target_s,
            segs,
            target_registry_indices,
            row_labels,
        } => UnauthRequest::EvalKappaDoubleTargetAtRowLabelsMatrix {
            target_s: target_s as usize,
            segs: segs.into_iter().map(|seg| seg as usize).collect(),
            target_registry_indices: target_registry_indices
                .into_iter()
                .map(|idx| idx as usize)
                .collect(),
            row_labels: row_labels.0,
        },
        Message::UnauthRetainClaims { registry_indices } => UnauthRequest::RetainClaims {
            registry_indices: registry_indices
                .into_iter()
                .map(|idx| idx as usize)
                .collect(),
        },
        Message::UnauthGetLen => UnauthRequest::GetLen,
        other => {
            return Response::Error {
                message: format!("unsupported unauth message: {:?}", other),
            };
        }
    };

    match service.handle_request(req) {
        UnauthResponse::Read { shares } => Response::UnauthRead {
            values: pack_ordered_shard_values(&shares),
        },
        UnauthResponse::ReadBlock { rows } => Response::UnauthReadBlock {
            rows: pack_single_row_reply(rows),
        },
        UnauthResponse::ReadSegmentRaw { shards } => Response::UnauthReadSegmentRaw {
            shards: pack_ordered_segment_primary_shards(&shards),
        },
        UnauthResponse::OptimizedRead { response } => {
            match core_optimized_response_to_net(response) {
                Ok(resp) => Response::UnauthOptimizedRead { resp },
                Err(err) => Response::Error {
                    message: format!("unauth optimized read failed: {:?}", err),
                },
            }
        }
        UnauthResponse::ReadRowsAtZ { rows } => Response::UnauthReadRowsAtZ {
            rows: pack_single_row_reply(rows),
        },
        UnauthResponse::ReadRowsBatchAtZ { rows } => Response::UnauthReadRowsBatchAtZ {
            rows: pack_row_batch_reply(rows),
        },
        UnauthResponse::ReadRowsMatrix { rows } => Response::UnauthReadRowsMatrix {
            rows: pack_row_matrix_reply(rows),
        },
        UnauthResponse::MultiRead { shares } => Response::UnauthMultiRead {
            values: shares
                .into_iter()
                .map(|batch| pack_ordered_shard_values(&batch))
                .collect(),
        },
        UnauthResponse::RepairShard { values } => Response::UnauthRepairShard {
            values: values.into_iter().map(NetFp).collect(),
        },
        UnauthResponse::RepairShardBatch { values } => Response::UnauthRepairShardBatch {
            values: values
                .into_iter()
                .map(|batch| batch.into_iter().map(NetFp).collect())
                .collect(),
        },
        UnauthResponse::EvalShardsAtRowLabels { values } => Response::UnauthEvalShardsAtRowLabels {
            values: values.into_iter().map(NetFpBlob).collect(),
        },
        UnauthResponse::EvalShardsAtRowLabelsMatrix { values } => {
            Response::UnauthEvalShardsAtRowLabelsMatrix {
                values: pack_eval_matrix_reply(values),
            }
        }
        UnauthResponse::EvalKappaDoubleTargetAtRowLabelsMatrix { values } => {
            Response::UnauthEvalKappaDoubleTargetAtRowLabelsMatrix {
                values: pack_eval_matrix_reply(values),
            }
        }
        UnauthResponse::ApplyDeltas { ok } => Response::UnauthApplyDeltas { ok },
        UnauthResponse::ApplyBlockDeltas { ok } => Response::UnauthApplyBlockDeltas { ok },
        UnauthResponse::Extend { ok } => Response::UnauthExtend { ok },
        UnauthResponse::Migrate { ok } => Response::UnauthMigrate { ok },
        UnauthResponse::SegmentDouble { ok } => Response::UnauthSegmentDouble { ok },
        UnauthResponse::KappaDoubleOneClaimLeft { ok } => {
            Response::UnauthKappaDoubleOneClaimLeft { ok }
        }
        UnauthResponse::RetainClaims { ok } => Response::UnauthRetainClaims { ok },
        UnauthResponse::Len { len } => Response::UnauthLen { len },
        UnauthResponse::Error { message } => Response::Error { message },
    }
}
