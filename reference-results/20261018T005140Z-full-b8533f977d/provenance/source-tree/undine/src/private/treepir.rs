//! TreePIR's balanced color splitting and client index selection.
//!
//! Ported from the authors' `CSA.java` and `SubCSA.java` at
//! <https://github.com/PIR-PIXR/TreePIR/tree/930063c5aefc441244abb4890fdf35383f3aa956>.
//! See `third_party/treepir/LICENSE` for the upstream MIT license. The port
//! preserves stable size sorting and the authors' database insertion order,
//! but uses integer arithmetic and 64-bit node IDs and offsets.
//!
//! Heap node 1 is the root; leaves start at `2^height`. The root is not
//! colored. Both children are emitted before descending left then right,
//! exactly as in CSA. A database stores the *sibling's* label at each
//! colored node: following the ordinary leaf path then retrieves a Merkle
//! authentication path from the sibling-swapped tree.

use serde::{Deserialize, Serialize};

use crate::auth::merkle::{DefaultHashes, leaf_label, node_hash};
use crate::net::wire::{NetByteBlob, decode_binary, encode_binary};
use crate::value::Value256;
use crate::private::pir::{
    CELL_BITS, PackKeys, PirClient, PirShape, PreprocessTimes, TableDatabase, bytes_from_cells,
    bytes_to_cells,
};
use crate::private::wire::{WORD_BYTES, pack_words, unpack_words};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ColorSize {
    color: usize,
    size: u64,
}

/// Where one authentication-path node is stored. Offsets are zero based.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeLocation {
    pub color: usize,
    pub offset: u64,
    /// Node on the root-to-leaf path in the sibling-swapped tree.
    pub node: u64,
}

impl NodeLocation {
    /// Heap ID in the original Merkle tree of the value to retrieve.
    pub fn sibling(&self) -> u64 {
        self.node ^ 1
    }
}

/// Public balanced class sizes, sufficient for a client to locate all proof
/// nodes without receiving an index proportional to the tree size.
#[derive(Clone, Debug)]
pub struct BalancedColoring {
    height: u8,
    sequence: Vec<ColorSize>,
}

impl BalancedColoring {
    pub fn new(height: u8) -> Self {
        assert!((1..=62).contains(&height), "height must be in 1..=62");
        let nodes = (1u64 << (height + 1)) - 2;
        let quotient = nodes / u64::from(height);
        let small = u64::from(height) - nodes % u64::from(height);
        let sequence = (0..usize::from(height))
            .map(|color| ColorSize {
                color,
                size: quotient + u64::from(color as u64 >= small),
            })
            .collect();
        Self { height, sequence }
    }

    pub fn height(&self) -> u8 {
        self.height
    }

    pub fn class_sizes(&self) -> Vec<u64> {
        self.sequence.iter().map(|c| c.size).collect()
    }

    /// Run the complete color-splitting traversal. The callback receives a
    /// color, its next database offset, and the colored heap node ID.
    /// A streaming consumer needs only O(height²) layout working memory.
    pub fn visit(&self, mut emit: impl FnMut(NodeLocation)) {
        let mut counts = vec![0; usize::from(self.height)];
        visit_subtree(1, &self.sequence, &mut counts, &mut emit);
        debug_assert_eq!(counts, self.class_sizes());
    }

    /// Build the authors' per-color lists of colored node IDs.
    pub fn materialize(&self) -> Vec<Vec<u64>> {
        let mut classes = self
            .sequence
            .iter()
            .map(|c| Vec::with_capacity(usize::try_from(c.size).expect("class fits address space")))
            .collect::<Vec<_>>();
        self.visit(|entry| classes[entry.color].push(entry.node));
        classes
    }

    /// Materialize just one server's class. Unneeded subtrees are skipped
    /// using the public size sequences. This avoids holding the entire
    /// tree's node-ID lists for large per-server experiments.
    pub fn class_nodes(&self, color: usize) -> Vec<u64> {
        assert!(color < self.sequence.len());
        let mut nodes = Vec::with_capacity(self.sequence[color].size as usize);
        visit_one_color(1, &self.sequence, color, &mut nodes);
        assert_eq!(nodes.len() as u64, self.sequence[color].size);
        nodes
    }

    /// The nodes of one server's class at the database offsets `offsets`, in
    /// CSA insertion order: `class_nodes(color)[offsets]`. Subtrees before
    /// the range are skipped by their public class sizes, so reading one PIR
    /// row does not enumerate the class.
    pub fn class_range(&self, color: usize, offsets: std::ops::Range<u64>) -> Vec<u64> {
        assert!(color < self.sequence.len());
        assert!(offsets.end <= self.sequence[color].size, "offsets beyond the class");
        let mut skip = offsets.start;
        let mut wanted = offsets.end.saturating_sub(offsets.start);
        let mut nodes = Vec::with_capacity(wanted as usize);
        range_of_color(1, &self.sequence, color, &mut skip, &mut wanted, &mut nodes);
        nodes
    }

    /// Locate every proof node, in root-to-leaf order. This follows just one
    /// branch of CSA, accumulating counts of earlier nodes and skipped left
    /// subtrees as in SubCSA. It never constructs the complete coloring.
    pub fn locate(&self, leaf: u64) -> Vec<NodeLocation> {
        assert!(leaf < 1u64 << self.height, "leaf is outside the tree");
        let mut sequence = self.sequence.clone();
        let mut counts = vec![0u64; usize::from(self.height)];
        let mut result = Vec::with_capacity(usize::from(self.height));
        let mut root = 1u64;
        for depth in 0..self.height {
            let right = (leaf >> (self.height - depth - 1)) & 1 != 0;
            let left_color = sequence[0].color;
            let right_color = if sequence[0].size == 2 {
                left_color
            } else {
                sequence[1].color
            };
            let color = if right { right_color } else { left_color };
            let offset = counts[color] + u64::from(right && left_color == right_color);
            root = 2 * root + u64::from(right);
            result.push(NodeLocation {
                color,
                offset,
                node: root,
            });
            // CSA emits both children before either child's descendants.
            counts[left_color] += 1;
            counts[right_color] += 1;
            if sequence.len() > 1 {
                let (left, right_sequence) = feasible_split(&sequence);
                if right {
                    for entry in &left {
                        counts[entry.color] += entry.size;
                    }
                    sequence = right_sequence;
                } else {
                    sequence = left;
                }
            }
        }
        result
    }
}

fn visit_one_color(root: u64, sequence: &[ColorSize], color: usize, nodes: &mut Vec<u64>) {
    if !sequence.iter().any(|c| c.color == color) {
        return;
    }
    let left_color = sequence[0].color;
    let right_color = if sequence[0].size == 2 {
        left_color
    } else {
        sequence[1].color
    };
    if left_color == color {
        nodes.push(2 * root);
    }
    if right_color == color {
        nodes.push(2 * root + 1);
    }
    if sequence.len() > 1 {
        let (left, right) = feasible_split(sequence);
        visit_one_color(2 * root, &left, color, nodes);
        visit_one_color(2 * root + 1, &right, color, nodes);
    }
}

/// [`visit_one_color`] restricted to the class nodes after the first
/// `skip`, up to `wanted` of them. `sequence` lists the class sizes among
/// the descendants of `root`, so a subtree with no wanted node is skipped
/// whole.
fn range_of_color(
    root: u64,
    sequence: &[ColorSize],
    color: usize,
    skip: &mut u64,
    wanted: &mut u64,
    nodes: &mut Vec<u64>,
) {
    if *wanted == 0 {
        return;
    }
    let Some(total) = sequence.iter().find(|c| c.color == color).map(|c| c.size) else {
        return;
    };
    if *skip >= total {
        *skip -= total;
        return;
    }
    let left_color = sequence[0].color;
    let right_color = if sequence[0].size == 2 {
        left_color
    } else {
        sequence[1].color
    };
    for (child_color, node) in [(left_color, 2 * root), (right_color, 2 * root + 1)] {
        if child_color != color || *wanted == 0 {
            continue;
        }
        if *skip > 0 {
            *skip -= 1;
        } else {
            nodes.push(node);
            *wanted -= 1;
        }
    }
    if sequence.len() > 1 {
        let (left, right) = feasible_split(sequence);
        range_of_color(2 * root, &left, color, skip, wanted, nodes);
        range_of_color(2 * root + 1, &right, color, skip, wanted, nodes);
    }
}

/// Pack each color class into a conventional rectangular YPIR database.
/// A PIR row holds several 32-byte node labels; the client keeps the
/// within-row offset private locally. All classes use a common public shape.
#[derive(Clone, Copy, Debug)]
pub struct TreePirLayout {
    pub height: u8,
    pub nodes_per_record: usize,
    pub shape: PirShape,
}

impl TreePirLayout {
    pub fn new(height: u8) -> Self {
        let coloring = BalancedColoring::new(height);
        let max_nodes = *coloring.class_sizes().last().unwrap() as usize;
        // Minimize the actual query-coefficient + ciphertext bytes over
        // power-of-two row counts, subject to YPIR's supported dimensions.
        // The packing key is a shape-independent additive cost.
        let mut best = None;
        for row_bits in 11..=18 {
            let nodes_per_record = max_nodes.div_ceil(1usize << row_bits).max(1);
            let record_cells = (nodes_per_record * 256).div_ceil(CELL_BITS);
            let shape = PirShape {
                patterns: max_nodes.div_ceil(nodes_per_record),
                slots: 1,
                group_slots: 1,
                record_cells,
                record_bits: nodes_per_record * 256,
            };
            // Avoid constructing large NTT parameter objects for candidates
            // outside the documented YPIR correctness bound.
            if record_cells.div_ceil(2048) * 2048 > 1 << 18 {
                continue;
            }
            let bytes = shape.db_rows() * WORD_BYTES + shape.outputs() * shape.ciphertext_bytes();
            if best.as_ref().is_none_or(|(cost, _)| bytes < *cost) {
                best = Some((
                    bytes,
                    Self {
                        height,
                        nodes_per_record,
                        shape,
                    },
                ));
            }
        }
        best.expect("color database exceeds supported YPIR dimensions")
            .1
    }

    /// Fresh key and query for this class. Distinct class queries use
    /// independent secrets: the backend reuses a public query matrix, so
    /// reusing one secret across queries for different rows would expose
    /// their differences to colluding servers.
    pub fn request(&self, location: NodeLocation) -> Result<(TreePirQuery, Vec<u8>), String> {
        let query = self.query_with(location, PirClient::generate(self.shape))?;
        let (words, keys) = query.encrypted_query();
        let bytes = query.serialize_request(&words, &keys)?;
        Ok((query, bytes))
    }

    /// The query of `location` under the secret key `client`, before any
    /// encryption. [`Self::request`] is this with a fresh key, then
    /// [`TreePirQuery::encrypted_query`] and
    /// [`TreePirQuery::serialize_request`].
    pub fn query_with(&self, location: NodeLocation, client: PirClient) -> Result<TreePirQuery, String> {
        if location.color >= usize::from(self.height) {
            return Err("invalid color".into());
        }
        Ok(TreePirQuery {
            layout: *self,
            location,
            client,
        })
    }

    /// The PIR row of the class that holds database offset `offset`.
    pub fn row_of(&self, offset: u64) -> usize {
        offset as usize / self.nodes_per_record
    }

    /// The class offsets stored in PIR row `row` of a class of `size` nodes.
    pub fn row_offsets(&self, row: usize, size: u64) -> std::ops::Range<u64> {
        let start = ((row * self.nodes_per_record) as u64).min(size);
        start..(start + self.nodes_per_record as u64).min(size)
    }

    /// The cells of a PIR row that stores the class nodes `nodes`: the
    /// label of each node's sibling in the original tree, read by `labels`
    /// from its heap number.
    pub fn row_cells(&self, nodes: &[u64], mut labels: impl FnMut(u64) -> [u8; 32], cells: &mut [u16]) {
        bytes_to_cells(nodes.iter().flat_map(|&node| labels(node ^ 1)), cells);
    }
}

/// On-wire requests contain no node ID or selected database offset.
#[derive(Serialize, Deserialize)]
pub struct TreePirRequest {
    pub height: u8,
    pub color: u8,
    pub query: NetByteBlob,
    pub pack_keys: NetByteBlob,
}

#[derive(Serialize, Deserialize)]
pub struct TreePirReply {
    pub height: u8,
    pub color: u8,
    pub ciphertexts: Vec<NetByteBlob>,
}

pub struct TreePirQuery {
    layout: TreePirLayout,
    pub location: NodeLocation,
    client: PirClient,
}

impl TreePirQuery {
    /// The encrypted query of the row that holds the node, and the packing
    /// keys of the secret key.
    pub fn encrypted_query(&self) -> (Vec<u64>, PackKeys) {
        (
            self.client.query(self.layout.row_of(self.location.offset)),
            self.client.pack_keys(),
        )
    }

    /// The serialized request that carries `query` and `keys`.
    pub fn serialize_request(&self, query: &[u64], keys: &PackKeys) -> Result<Vec<u8>, String> {
        encode_binary(&TreePirRequest {
            height: self.layout.height,
            color: self.location.color as u8,
            query: NetByteBlob(pack_words(query)),
            pack_keys: NetByteBlob(pack_words(&keys.words)),
        })
    }

    /// A serialized reply whose row holds `cells`, encrypted directly under
    /// this query's secret key in the format of a server's reply
    /// ([`PirClient::encrypt_slots`]); for benchmarks of the client.
    #[cfg(feature = "bench")]
    pub fn encrypt_row(&self, cells: &[u16]) -> Result<Vec<u8>, String> {
        let slots = [cells.iter().map(|&cell| u64::from(cell)).collect::<Vec<_>>()];
        encode_binary(&TreePirReply {
            height: self.layout.height,
            color: self.location.color as u8,
            ciphertexts: self
                .client
                .encrypt_slots(&slots)
                .into_iter()
                .map(NetByteBlob)
                .collect(),
        })
    }

    /// Parse and decrypt a reply, returning the selected original-tree
    /// sibling label rather than the rest of the fetched PIR row.
    pub fn decode(&self, bytes: &[u8]) -> Result<[u8; 32], String> {
        let row = self.decode_row(bytes)?;
        Ok(bytes_from_cells(
            &row,
            self.location.offset as usize % self.layout.nodes_per_record,
        ))
    }

    /// Parse and decrypt a reply into the cells of the whole fetched row.
    pub fn decode_row(&self, bytes: &[u8]) -> Result<Vec<u64>, String> {
        let reply: TreePirReply = decode_binary(bytes)?;
        if reply.height != self.layout.height || reply.color as usize != self.location.color {
            return Err("reply belongs to a different tree or color".into());
        }
        let ciphertexts = reply
            .ciphertexts
            .into_iter()
            .map(|c| c.0)
            .collect::<Vec<_>>();
        let mut slots = self
            .client
            .decode(&ciphertexts)
            .map_err(|e| e.to_string())?;
        Ok(slots.swap_remove(0))
    }
}

/// A real YPIR server for one TreePIR color class.
pub struct TreePirServer {
    pub layout: TreePirLayout,
    pub color: usize,
    pub database: TableDatabase,
}

impl TreePirServer {
    /// `labels(node)` reads the label of a node in the *original* tree.
    /// `nodes` must be the class in CSA insertion order. Coloring and label
    /// generation are deliberately separate from the measured YPIR phases.
    pub fn new(
        layout: TreePirLayout,
        color: usize,
        nodes: &[u64],
        mut labels: impl FnMut(u64) -> [u8; 32],
    ) -> (Self, std::time::Duration, PreprocessTimes) {
        assert_eq!(
            nodes.len() as u64,
            BalancedColoring::new(layout.height).class_sizes()[color]
        );
        let (database, build, preprocessing) =
            TableDatabase::from_rows_timed(layout.shape, |row, cells| {
                let offsets = layout.row_offsets(row, nodes.len() as u64);
                layout.row_cells(
                    &nodes[offsets.start as usize..offsets.end as usize],
                    &mut labels,
                    cells,
                );
            });
        (
            Self {
                layout,
                color,
                database,
            },
            build,
            preprocessing,
        )
    }

    /// The full local server pipeline, from serialized request to reply.
    pub fn answer(&self, bytes: &[u8]) -> Result<Vec<u8>, String> {
        let request: TreePirRequest = decode_binary(bytes)?;
        if request.height != self.layout.height || request.color as usize != self.color {
            return Err("request belongs to a different tree or color".into());
        }
        let query = unpack_words(&request.query.0)?;
        let keys = PackKeys {
            words: unpack_words(&request.pack_keys.0)?,
        };
        let ciphertexts = self
            .database
            .answer(&query, &keys)
            .map_err(|e| e.to_string())?;
        encode_binary(&TreePirReply {
            height: self.layout.height,
            color: self.color as u8,
            ciphertexts: ciphertexts.into_iter().map(NetByteBlob).collect(),
        })
    }
}

/// The root that a leaf's value and the sibling labels TreePIR retrieves
/// for it lead to: `siblings` in the root-to-leaf order of
/// [`BalancedColoring::locate`]. A client checks it against its trusted root.
pub fn root_from_labels(
    defaults: &DefaultHashes,
    height: u8,
    leaf: u64,
    value: Value256,
    siblings: &[[u8; 32]],
) -> Value256 {
    assert_eq!(siblings.len(), usize::from(height), "one sibling per level");
    let mut node = (1u64 << height) + leaf;
    let mut label = leaf_label(defaults, u32::from(height), leaf, value);
    for &sibling in siblings.iter().rev() {
        label = if node % 2 == 0 {
            node_hash(label, sibling)
        } else {
            node_hash(sibling, label)
        };
        node /= 2;
    }
    label
}

fn visit_subtree(
    root: u64,
    c: &[ColorSize],
    counts: &mut [u64],
    emit: &mut impl FnMut(NodeLocation),
) {
    let left_color = c[0].color;
    let right_color = if c[0].size == 2 {
        left_color
    } else {
        c[1].color
    };
    for (color, node) in [(left_color, 2 * root), (right_color, 2 * root + 1)] {
        emit(NodeLocation {
            color,
            offset: counts[color],
            node,
        });
        counts[color] += 1;
    }
    if c.len() > 1 {
        let (left, right) = feasible_split(c);
        visit_subtree(2 * root, &left, counts, emit);
        visit_subtree(2 * root + 1, &right, counts, emit);
    }
}

/// The authors' FeasibleSplit procedure. Stable sorting is significant:
/// color ties must be resolved identically by the server and the client.
fn feasible_split(c: &[ColorSize]) -> (Vec<ColorSize>, Vec<ColorSize>) {
    assert!(c.len() >= 2);
    let mut left = Vec::with_capacity(c.len() - 1);
    let mut right = Vec::with_capacity(c.len() - 1);
    let mut left_sum;
    let mut right_sum;
    let start;
    if c[0].size == 2 {
        left_sum = c[1].size / 2;
        right_sum = c[1].size.div_ceil(2);
        left.push(ColorSize {
            color: c[1].color,
            size: left_sum,
        });
        right.push(ColorSize {
            color: c[1].color,
            size: right_sum,
        });
        start = 2;
    } else {
        left_sum = c[1].size - 1;
        right_sum = c[0].size - 1;
        left.push(ColorSize {
            color: c[1].color,
            size: left_sum,
        });
        right.push(ColorSize {
            color: c[0].color,
            size: right_sum,
        });
        if c.len() > 2 {
            let half = c[2].size + c[0].size - c[1].size;
            let a = half.div_ceil(2);
            let b = c[1].size - c[0].size + half / 2;
            left.push(ColorSize {
                color: c[2].color,
                size: a,
            });
            right.push(ColorSize {
                color: c[2].color,
                size: b,
            });
            left_sum += a;
            right_sum += b;
        }
        start = 3;
    }
    for entry in c.iter().skip(start) {
        let a = if left_sum < right_sum {
            entry.size.div_ceil(2)
        } else {
            entry.size / 2
        };
        let b = entry.size - a;
        left.push(ColorSize {
            color: entry.color,
            size: a,
        });
        right.push(ColorSize {
            color: entry.color,
            size: b,
        });
        left_sum += a;
        right_sum += b;
    }
    left.sort_by_key(|entry| entry.size);
    right.sort_by_key(|entry| entry.size);
    (left, right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_authors_java_reference() {
        // Regenerated independently by scripts/check_treepir_reference.py.
        let reference = include_str!("../../tests/fixtures/treepir-reference.csv");
        for line in reference.lines() {
            let mut fields = line.split(',');
            let kind = fields.next().unwrap();
            let height: u8 = fields.next().unwrap().parse().unwrap();
            let argument: u64 = fields.next().unwrap().parse().unwrap();
            let expected = fields
                .map(|field| field.parse::<u64>().unwrap())
                .collect::<Vec<_>>();
            let coloring = BalancedColoring::new(height);
            let actual = if kind == "C" {
                coloring.class_nodes(argument as usize)
            } else {
                let mut offsets = vec![0; height as usize];
                for item in coloring.locate(argument) {
                    offsets[item.color] = item.offset;
                }
                offsets
            };
            assert_eq!(
                actual, expected,
                "{kind}, height {height}, argument {argument}"
            );
        }
    }

    #[test]
    fn exhaustive_coloring_and_index_selection() {
        for height in 1..=12 {
            let coloring = BalancedColoring::new(height);
            let classes = coloring.materialize();
            for (color, nodes) in classes.iter().enumerate() {
                assert_eq!(&coloring.class_nodes(color), nodes);
            }
            let sizes = coloring.class_sizes();
            assert_eq!(
                sizes.iter().max().unwrap() - sizes.iter().min().unwrap(),
                u64::from(((1u64 << (height + 1)) - 2) % u64::from(height) != 0)
            );
            let mut seen = vec![false; 1 << (height + 1)];
            for (nodes, &size) in classes.iter().zip(&sizes) {
                assert_eq!(nodes.len() as u64, size);
                for &node in nodes {
                    assert!(!seen[node as usize]);
                    seen[node as usize] = true;
                }
            }
            assert!(seen[2..].iter().all(|&b| b));
            for leaf in 0..1u64 << height {
                let locations = coloring.locate(leaf);
                let mut colors = vec![false; height as usize];
                for (depth, location) in locations.iter().enumerate() {
                    assert!(!colors[location.color]);
                    colors[location.color] = true;
                    assert_eq!(
                        classes[location.color][location.offset as usize],
                        location.node
                    );
                    assert_eq!(
                        location.node,
                        ((1u64 << height) + leaf) >> (height as usize - depth - 1)
                    );
                }
                assert!(colors.iter().all(|&b| b));
            }
        }
    }

    #[test]
    fn class_ranges_are_slices_of_the_classes() {
        for height in 1..=12u8 {
            let coloring = BalancedColoring::new(height);
            for (color, nodes) in coloring.materialize().iter().enumerate() {
                let size = nodes.len() as u64;
                let lengths = [0u64, 1, 2, 3, 7, 64, size];
                for start in 0..=size {
                    // Every start for small classes, a spread for larger ones.
                    if size > 64 && start % 37 != 0 && start + 1 < size {
                        continue;
                    }
                    for &length in &lengths {
                        let end = (start + length).min(size);
                        assert_eq!(
                            coloring.class_range(color, start..end),
                            nodes[start as usize..end as usize],
                            "height {height}, color {color}, offsets {start}..{end}"
                        );
                    }
                }
            }
        }
        // A row of a large class, without the class.
        let coloring = BalancedColoring::new(32);
        for leaf in [0u64, (1 << 32) - 1, 0x1234_5678] {
            for location in coloring.locate(leaf) {
                let nodes = coloring.class_range(location.color, location.offset..location.offset + 1);
                assert_eq!(nodes, [location.node]);
            }
        }
    }

    #[test]
    fn large_indices_do_not_truncate_to_32_bits() {
        for height in [31, 32, 36, 50, 62] {
            let coloring = BalancedColoring::new(height);
            let sizes = coloring.class_sizes();
            for leaf in [
                0,
                1,
                (1u64 << height) / 3,
                (1u64 << height) - 2,
                (1u64 << height) - 1,
            ] {
                let locations = coloring.locate(leaf);
                let mut colors = vec![false; height as usize];
                for item in &locations {
                    assert!(!colors[item.color]);
                    colors[item.color] = true;
                    assert!(item.offset < sizes[item.color]);
                }
                assert_eq!(locations.last().unwrap().node, (1u64 << height) + leaf);
                assert!(colors.iter().all(|&b| b));
            }
        }
    }
}
