use paper_a_optimized::net;
use paper_a_optimized::net::registry::PartyInfo;
use paper_a_optimized::net::unauth::{self, NetUnauthTransport};
use paper_a_optimized::unauth::client::UnauthTransport;
use paper_a_optimized::unauth::service::{UnauthRequest, UnauthResponse, UnauthService};
use paper_a_optimized::unauth::{
    CodedArrayConfig, UnauthStore, build_coded_params, build_shard_registry_deterministic,
};
use paper_a_optimized::{Fp, encode_value};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

fn build_params() -> paper_a_optimized::CodedArrayParams {
    let segment_len: usize = 8;
    let s: usize = 4;
    let shard_count: usize = 4;
    let n_grid = segment_len;
    let shard_registry = build_shard_registry_deterministic(shard_count, s, n_grid);
    build_coded_params(&CodedArrayConfig {
        segment_len,
        s,
        shard_count,
        shard_registry,
        party_registry_indices: vec![(0..shard_count).collect()],
        seed: 42,
    })
}

fn raw_segment_dump(store: &UnauthStore, seg: usize) -> Vec<(usize, Vec<Fp>)> {
    let mut out = Vec::new();
    for (shard_key, values) in store.coded.shard_segments_for_segment(&store.params, seg) {
        let registry_index = *store
            .params
            .shard_registry_by_key
            .get(shard_key)
            .expect("registry index");
        out.push((registry_index, values.to_vec()));
    }
    out.sort_by_key(|(registry_index, _)| *registry_index);
    out
}

fn row_labels_for_owned_indices(
    params: &paper_a_optimized::CodedArrayParams,
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

#[test]
fn unauth_service_raw_segment_dump_returns_exact_local_vectors() {
    let params = build_params();
    let mut store = UnauthStore::new(params.clone());
    store.extend(params.segment_len as u64).expect("extend");

    let removed_key = params.shard_registry[1].key;
    let removed_values = store
        .coded
        .remove_shard_family_segments(&params, &removed_key)
        .expect("remove one local shard");
    assert!(
        store.coded.row_family_count() > 0,
        "rows remain materialized, so repair would be possible if the API tried to compute"
    );
    assert_eq!(removed_values.len(), 1);

    let response = store.handle_request(UnauthRequest::ReadSegmentRaw { seg: 0 });
    let UnauthResponse::ReadSegmentRaw { shards } = response else {
        panic!("unexpected response: {:?}", response);
    };
    let got: Vec<(usize, Vec<Fp>)> = shards
        .into_iter()
        .map(|shard| (shard.registry_index, shard.values))
        .collect();
    let expected = raw_segment_dump(&store, 0);
    assert_eq!(got, expected);
    assert!(!got.iter().any(|(registry_index, _)| *registry_index == 1));
}

#[test]
fn net_transport_raw_segment_dump_roundtrips_verbatim() {
    let params = build_params();
    let mut store = UnauthStore::new(params.clone());
    store.extend(params.segment_len as u64).expect("extend");
    let expected = raw_segment_dump(&store, 0);

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind listener");
    let addr = listener.local_addr().expect("local addr");
    let shared_store = Arc::new(Mutex::new(store));
    let server_store = shared_store.clone();
    thread::spawn(move || {
        let _ = net::serve_with_listener(listener, move |msg| {
            let mut store = server_store.lock().expect("store lock");
            unauth::handle_message(&mut *store, msg)
        });
    });
    thread::sleep(Duration::from_millis(50));

    let owned_indices: Vec<usize> = (0..params.shard_registry.len()).collect();
    let transport = NetUnauthTransport::for_single_party(
        0,
        row_labels_for_owned_indices(&params, &owned_indices),
        owned_indices,
    );
    let party = PartyInfo {
        party_id: 0,
        rpc_host: addr.to_string(),
        pubkey: None,
        active: true,
    };
    let shards = transport
        .unauth_read_segment_raw(&party, 0)
        .expect("raw segment rpc");
    let got: Vec<(usize, Vec<Fp>)> = shards
        .into_iter()
        .map(|shard| (shard.registry_index, shard.values))
        .collect();
    assert_eq!(got, expected);
}
