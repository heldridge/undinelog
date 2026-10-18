//! Exact snapshots of the append-built state used by public read benchmarks.
//! Mutable reconfiguration and update trials must restore their own baseline.

use paper_a_optimized::auth::service::AuthServer;
use paper_a_optimized::core::NodeLinearization;
use paper_a_optimized::unauth::{ClaimBundle, CodedStore, RowKey, ShardKey};
use paper_a_optimized::{AuthParams, CodedArrayParams, Value256, decode_value, encode_value};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct AuthFixture {
    leaves: usize,
    root: Value256,
    array_len: u64,
    segments: usize,
    shards: Vec<(Value256, Value256, Vec<Vec<Value256>>)>,
    rows: Vec<(Value256, Value256, Value256, Vec<Vec<Value256>>)>,
}

impl AuthFixture {
    pub fn capture(server: &AuthServer) -> Self {
        let store = &server.store.unauth;
        let mut shard_keys = store.coded.shard_keys().copied().collect::<Vec<_>>();
        shard_keys.sort();
        let mut row_keys = store.coded.row_keys().copied().collect::<Vec<_>>();
        row_keys.sort_by_key(|key| (key.owner, key.row_label_bytes));
        let encode = |values: Vec<Vec<_>>| {
            values
                .into_iter()
                .map(|segment| segment.into_iter().map(encode_value).collect())
                .collect()
        };
        Self {
            leaves: server.allocated_leaves() as usize,
            root: server.store.root,
            array_len: store.len(),
            segments: store.segments(),
            shards: shard_keys
                .iter()
                .map(|key| {
                    (
                        key.a_bytes,
                        key.z_bytes,
                        encode(
                            store
                                .coded
                                .clone_shard_family_segments(&store.params, key)
                                .unwrap(),
                        ),
                    )
                })
                .collect(),
            rows: row_keys
                .iter()
                .map(|key| {
                    (
                        key.owner.a_bytes,
                        key.owner.z_bytes,
                        key.row_label_bytes,
                        encode(
                            store
                                .coded
                                .clone_row_family_segments(&store.params, key)
                                .unwrap(),
                        ),
                    )
                })
                .collect(),
        }
    }

    pub fn restore(
        self,
        params: &CodedArrayParams,
        auth: &AuthParams,
        order: NodeLinearization,
        claims: Option<Vec<ClaimBundle>>,
    ) -> AuthServer {
        let mut server = AuthServer::new_with_assignments_and_claims(
            params.clone(),
            auth.clone(),
            order,
            self.leaves,
            None,
            claims,
        )
        .expect("restore allocated benchmark state");
        assert_eq!(server.store.unauth.len(), self.array_len);
        assert_eq!(server.store.unauth.segments(), self.segments);
        let mut coded = CodedStore::new();
        for (a_bytes, z_bytes, segments) in self.shards {
            assert_eq!(segments.len(), self.segments);
            coded.insert_shard_family_segments(
                params,
                ShardKey { a_bytes, z_bytes },
                segments
                    .into_iter()
                    .map(|values| values.into_iter().map(decode_value).collect())
                    .collect(),
            );
        }
        for (a_bytes, z_bytes, row_label_bytes, segments) in self.rows {
            assert_eq!(segments.len(), self.segments);
            coded.insert_row_family_segments(
                params,
                RowKey {
                    owner: ShardKey { a_bytes, z_bytes },
                    row_label_bytes,
                },
                segments
                    .into_iter()
                    .map(|values| values.into_iter().map(decode_value).collect())
                    .collect(),
            );
        }
        assert_eq!(
            coded.shard_family_count(),
            server.store.unauth.coded.shard_family_count()
        );
        assert_eq!(
            coded.row_family_count(),
            server.store.unauth.coded.row_family_count()
        );
        server.store.unauth.coded = coded;
        server.store.root = self.root;
        server
    }
}
