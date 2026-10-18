//! A private opening returns the value and the path of the public opening.
#![cfg(feature = "private")]

#[path = "support/cluster.rs"]
mod cluster;

use cluster::{LocalServer, reader, written_cluster};
use paper_a_optimized::auth::client::ClientError;
use paper_a_optimized::core::NodeLinearization;
use paper_a_optimized::private::{Grouping, PrivateKeeper, PrivateOpening, PrivateState};
use paper_a_optimized::{Fp, encode_value};
use std::sync::{Arc, RwLock};

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
    for &(kappa, tree_height, leaves) in &[
        (2usize, 7usize, 1usize),
        (2, 7, 37),
        (2, 7, 64),
        (4, 8, 150),
        (2, 50, 29),
        // A log long enough for the public pattern to span several segments.
        (2, 10, 600),
    ] {
        let (params, auth, servers, values, root) =
            written_cluster(kappa, 3 * kappa, tree_height, leaves, 101);
        if leaves == 600 {
            let layout = paper_a_optimized::private::PrivateLayout::for_log(
                &params,
                &auth,
                NodeLinearization::Preorder,
                leaves as u64,
            )
            .expect("layout");
            assert!(layout.group_slots() > 2, "the public pattern spans several segments");
        }
        let mut client = reader(&params, &auth, leaves, &servers);
        let indices = if leaves <= 40 {
            (0..leaves as u64).collect::<Vec<_>>()
        } else {
            vec![0, 1, 7, leaves as u64 / 2, leaves as u64 - 2, leaves as u64 - 1]
        };
        for index in indices {
            let (public_value, public_path) = client.read_with_path(index).expect("public opening");
            let (value, path) = client
                .read_private_with_path(index, None)
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
            .read_private_with_path(index, Some(kappa))
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
    assert!(plain.read_private_with_path(13, None).is_err());
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
            request.pattern_bits,
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
        client.read_private_with_path(leaves as u64, None),
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

    // A request with another grouping than the public one is refused too.
    let other_grouping = PrivateOpening::new(
        &params,
        &auth,
        NodeLinearization::Preorder,
        leaves as u64,
        2,
        0,
    )
    .expect("opening");
    let request = other_grouping.request();
    assert_eq!(request.leaves, leaves as u64);
    assert!(guard.handle_private_read(&request).is_err());
    drop(guard);

    // The preprocessing is reused across requests and still answers.
    for index in [3u64, 19] {
        assert_eq!(
            client.read_private(index).expect("private opening"),
            values[index as usize]
        );
    }
}

/// After an append the keeper updates a server's private state off the
/// lock and installs it; the installed state answers the longer log as a
/// fresh one does.
#[test]
fn the_keeper_updates_the_state_off_the_lock() {
    let kappa = 2usize;
    let leaves = 20usize;
    let (params, auth, servers, values, _root) = written_cluster(kappa, 3 * kappa, 6, leaves, 105);
    let mut client = reader(&params, &auth, leaves, &servers);
    assert_eq!(client.read_private(3).expect("private opening"), values[3]);

    let appended = (0..7u64)
        .map(|k| encode_value(Fp::from(900 + k)))
        .collect::<Vec<_>>();
    assert_eq!(client.append_leaves(&appended).expect("append"), leaves as u64);
    let longer = (leaves + appended.len()) as u64;

    // One server as the binaries hold it, with the state of the old log.
    let shared = Arc::new(RwLock::new(servers[1].inner.lock().expect("server lock").clone()));
    let grouping = Grouping::public(longer, kappa);
    {
        let server = shared.read().expect("server lock");
        let state = server.private_state().expect("prepared state");
        assert_eq!(state.updates(), 0);
        assert!(!state.is_current(&server, grouping));
    }
    let keeper = PrivateKeeper::new(Arc::clone(&shared));
    keeper.after_batch();
    keeper.wait();

    let mut server = shared.write().expect("server lock");
    let state = server.private_state().expect("installed state");
    assert_eq!(state.updates(), 1);
    assert_eq!(state.leaves(), longer);
    assert!(state.is_current(&server, grouping));
    let fresh = PrivateState::build(&server, grouping).expect("fresh state");
    for index in [3u64, longer - 1] {
        let opening = PrivateOpening::for_log(&params, &auth, NodeLinearization::Preorder, longer, index)
            .expect("opening");
        let request = opening.request();
        let updated = opening.decode(&server.handle_private_read(&request).expect("reply"));
        let expected = opening.decode(&fresh.answer(&request).expect("fresh reply"));
        assert_eq!(updated.nodes.len(), expected.nodes.len());
        for (a, b) in updated.nodes.iter().zip(&expected.nodes) {
            assert_eq!(a.shares.len(), b.shares.len());
            for (x, y) in a.shares.iter().zip(&b.shares) {
                assert_eq!(x.registry_index(), y.registry_index());
                assert!(x.value() == y.value(), "share of leaf {index} after the keeper's update");
            }
        }
    }
    // A second call finds the state current and does nothing.
    drop(server);
    keeper.after_batch();
    keeper.wait();
    assert_eq!(shared.read().expect("server lock").private_state().expect("state").updates(), 1);
}

/// The demo's shape: a tree of height 3 with 4 leaves at threshold 8,
/// where the public pattern of 2^5 leaves is cut to the tree. The client
/// and the server agree on it.
#[test]
fn a_pattern_larger_than_the_tree_is_the_tree() {
    let (params, auth, servers, values, _root) = written_cluster(8, 32, 3, 4, 7);
    let opening = PrivateOpening::for_log(&params, &auth, NodeLinearization::Preorder, 4, 0)
        .expect("opening");
    let request = opening.request();
    assert_eq!(request.pattern_bits, 3);
    servers[0]
        .inner
        .lock()
        .expect("server lock")
        .handle_private_read(&request)
        .expect("the server accepts the request");
    let mut client = reader(&params, &auth, 4, &servers);
    for index in 0..4u64 {
        assert_eq!(client.read_private(index).expect("private opening"), values[index as usize]);
    }
}
