#![cfg(feature = "private")]

use paper_a_optimized::auth::merkle::{leaf_hash, node_hash};
use paper_a_optimized::net::wire::{decode_binary, encode_binary};
use paper_a_optimized::private::bulk::bulk_leaf_value;
use paper_a_optimized::private::pir::{CELL_BITS, PirShape};
use paper_a_optimized::private::treepir::{
    BalancedColoring, TreePirLayout, TreePirRequest, TreePirServer,
};

#[test]
fn encrypted_treepir_reconstructs_real_merkle_paths() {
    let height = 4;
    let leaves = 1usize << height;
    let values = (0..leaves)
        .map(|i| bulk_leaf_value(42, i as u64))
        .collect::<Vec<_>>();
    let mut tree = vec![[0u8; 32]; 2 * leaves];
    for i in 0..leaves {
        tree[leaves + i] = leaf_hash(i as u64, values[i]);
    }
    for i in (1..leaves).rev() {
        tree[i] = node_hash(tree[2 * i], tree[2 * i + 1]);
    }
    let coloring = BalancedColoring::new(height);
    // Exercise both interior entries and partially filled last records,
    // despite the minimum YPIR dimensions in this small correctness test.
    let nodes_per_record = 3;
    let layout = TreePirLayout {
        height,
        nodes_per_record,
        shape: PirShape {
            patterns: 8usize.div_ceil(nodes_per_record),
            slots: 1,
            group_slots: 1,
            record_cells: (nodes_per_record * 256).div_ceil(CELL_BITS),
            record_bits: nodes_per_record * 256,
        },
    };
    let servers = (0..usize::from(height))
        .map(|color| {
            let nodes = coloring.class_nodes(color);
            TreePirServer::new(layout, color, &nodes, |node| tree[node as usize]).0
        })
        .collect::<Vec<_>>();
    for leaf in [0, 6, 15] {
        let mut siblings = Vec::new();
        for location in coloring.locate(leaf) {
            let (query, bytes) = layout.request(location).unwrap();
            let reply = servers[location.color].answer(&bytes).unwrap();
            let recovered = query.decode(&reply).unwrap();
            assert_eq!(recovered, tree[location.sibling() as usize]);
            siblings.push(recovered);

            let mut wrong: TreePirRequest = decode_binary(&bytes).unwrap();
            wrong.color = ((location.color + 1) % usize::from(height)) as u8;
            assert!(
                servers[location.color]
                    .answer(&encode_binary(&wrong).unwrap())
                    .is_err()
            );
            assert!(query.decode(&reply[..reply.len() - 1]).is_err());
        }
        let mut label = leaf_hash(leaf, values[leaf as usize]);
        let mut node = leaves as u64 + leaf;
        for sibling in siblings.into_iter().rev() {
            label = if node % 2 == 0 {
                node_hash(label, sibling)
            } else {
                node_hash(sibling, label)
            };
            node /= 2;
        }
        assert_eq!(label, tree[1]);
    }
}

#[test]
fn exact_height_threshold_comparison_shapes() {
    for height in [8, 16, 32] {
        let layout = TreePirLayout::new(height);
        let max_nodes = *BalancedColoring::new(height).class_sizes().last().unwrap() as usize;
        assert!(layout.shape.patterns * layout.nodes_per_record >= max_nodes);
        assert!(layout.shape.db_rows() <= 1 << 18);
        assert!(layout.shape.db_cols() <= 1 << 18);
        assert!(layout.shape.record_cells * CELL_BITS >= layout.nodes_per_record * 256);
    }
}
