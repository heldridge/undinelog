//! A private opening returns the value and the path of the public opening.
#![cfg(feature = "private")]

#[path = "support/cluster.rs"]
mod cluster;

use cluster::{LocalServer, reader, written_cluster};
use paper_a_optimized::auth::client::ClientError;
use paper_a_optimized::core::NodeLinearization;
use paper_a_optimized::private::PrivateOpening;

fn corrupt(servers: &[LocalServer], labels: &[usize]) -> Vec<LocalServer> {
    servers
        .iter()
        .cloned()
        .map(|mut server| {
            server.corrupt_reads = labels.contains(&server.label);
            server
        })
        .collect()
}

#[test]
fn private_openings_equal_public_openings() {
    for &(kappa, tree_height, leaves, segments_per_record) in &[
        (2usize, 7usize, 1usize, 1usize),
        (2, 7, 37, 1),
        (2, 7, 64, 2),
        (4, 8, 150, 1),
        (2, 50, 29, 1),
    ] {
        let (params, auth, servers, values, root) =
            written_cluster(kappa, 3 * kappa, tree_height, leaves, 101);
        let mut client = reader(&params, &auth, leaves, &servers);
        let indices = if leaves <= 40 {
            (0..leaves as u64).collect::<Vec<_>>()
        } else {
            vec![0, 1, 7, leaves as u64 / 2, leaves as u64 - 2, leaves as u64 - 1]
        };
        for index in indices {
            let (public_value, public_path) = client.read_with_path(index).expect("public opening");
            let (value, path) = client
                .read_private_with_path(index, segments_per_record, None)
                .expect("private opening");
            assert_eq!(value, values[index as usize]);
            assert_eq!(value, public_value);
            assert_eq!(path.index, public_path.index);
            assert_eq!(path.siblings, public_path.siblings);
            assert_eq!(path.root, public_path.root);
            assert_eq!(path.root, root);
        }
        assert_eq!(client.trusted_root(), Some(root));
    }
}

#[test]
fn private_openings_correct_bad_replies() {
    let kappa = 2usize;
    let leaves = 45usize;
    let (params, auth, servers, values, root) = written_cluster(kappa, 3 * kappa, 7, leaves, 102);
    // kappa of the 3 kappa servers return corrupted responses.
    let servers = corrupt(&servers, &[1, 4]);
    let mut client = reader(&params, &auth, leaves, &servers);
    for index in [0u64, 13, 44] {
        let (value, path) = client
            .read_private_with_path(index, 1, Some(kappa))
            .expect("private opening with corrupted replies");
        assert_eq!(value, values[index as usize]);
        assert_eq!(path.root, root);
        let (public_value, public_path) = reader(&params, &auth, leaves, &servers)
            .read_with_path_correcting(index, kappa)
            .expect("public opening with corrupted replies");
        assert_eq!(value, public_value);
        assert_eq!(path.siblings, public_path.siblings);
    }
    // The corrupted replies alone fail verification.
    let corrupted_only = [servers[1].clone(), servers[4].clone()];
    let mut plain = reader(&params, &auth, leaves, &corrupted_only);
    plain.set_trusted_root(root);
    assert!(plain.read_private_with_path(13, 1, None).is_err());
}

#[test]
fn a_private_request_has_one_shape_for_every_position() {
    let leaves = 100u64;
    let (params, auth, _servers, _values, _root) = written_cluster(2, 6, 8, leaves as usize, 103);
    let shape_of = |index: u64| {
        let opening = PrivateOpening::new(
            &params,
            &auth,
            NodeLinearization::Preorder,
            leaves,
            1,
            index,
        )
        .expect("opening");
        let request = opening.request();
        (
            request.leaves,
            request.kappa,
            request.segments_per_record,
            request.query.len(),
            request.pack_keys.words.len(),
        )
    };
    let first = shape_of(0);
    for index in [1u64, 2, 50, 99] {
        assert_eq!(shape_of(index), first);
    }
}

#[test]
fn servers_answer_only_their_current_log() {
    let kappa = 2usize;
    let leaves = 20usize;
    let (params, auth, servers, values, _root) = written_cluster(kappa, 3 * kappa, 6, leaves, 104);
    let mut client = reader(&params, &auth, leaves, &servers);
    assert!(matches!(
        client.read_private_with_path(leaves as u64, 1, None),
        Err(ClientError::InvalidIndex)
    ));

    // A request for another log length is refused by every server, so the
    // client gathers no share.
    let opening = PrivateOpening::new(
        &params,
        &auth,
        NodeLinearization::Preorder,
        leaves as u64 - 3,
        1,
        0,
    )
    .expect("opening");
    let request = opening.request();
    let mut guard = servers[0].inner.lock().expect("server lock");
    assert!(guard.handle_private_read(&request).is_err());
    drop(guard);

    // The preprocessing is reused across requests and still answers.
    for index in [3u64, 19] {
        assert_eq!(
            client.read_private(index, 1).expect("private opening"),
            values[index as usize]
        );
    }
}
