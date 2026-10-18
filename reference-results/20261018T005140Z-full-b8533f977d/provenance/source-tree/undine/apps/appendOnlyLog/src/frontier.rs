use crate::archive::ArchiveError;
use crate::layout::BULK_UPDATES_SIZE;
use crate::registry::AppendOnlyLogFrontierLeafState;
use paper_a_optimized::auth::client::{AuthClient as CoreAuthClient, AuthTransport};
use paper_a_optimized::merkle::{
    NodeKey, default_hash, label_from_content, leaf_key, leaf_label, node_hash, parent,
    recompute_root, sibling,
};
use paper_a_optimized::{AuthPath, AuthSubtreeProof, Value256};
use std::collections::BTreeMap;

fn default_node_content(tree_height: usize, node: NodeKey) -> Value256 {
    if node.level as usize == tree_height {
        [0u8; 32]
    } else {
        default_hash(
            &paper_a_optimized::build_default_hashes(tree_height),
            node.level as usize,
        )
    }
}

fn build_upper_path_nodes(tree_height: usize, start_index: u64, leaf_count: usize) -> Vec<NodeKey> {
    let subtree_height = leaf_count.trailing_zeros() as usize;
    let root_level = tree_height - subtree_height;
    let mut key = NodeKey {
        level: root_level as u32,
        pos: start_index >> subtree_height,
    };
    let mut upper = Vec::with_capacity(root_level);
    for _ in 0..root_level {
        upper.push(sibling(key));
        key = parent(key);
    }
    upper
}

fn frontier_known_contents(
    tree_height: usize,
    frontier_len: u64,
    frontier: &AppendOnlyLogFrontierLeafState,
) -> Result<BTreeMap<NodeKey, Value256>, ArchiveError> {
    if frontier_len == 0 {
        return Err(ArchiveError::InvalidFrontierState(
            "frontier witness requested for empty archive".to_string(),
        ));
    }
    if frontier.path.index + 1 != frontier_len as usize {
        return Err(ArchiveError::InvalidFrontierState(
            "cached frontier witness index does not match committed frontier".to_string(),
        ));
    }
    if frontier.path.siblings.len() != tree_height {
        return Err(ArchiveError::InvalidFrontierState(
            "cached frontier witness has wrong path height".to_string(),
        ));
    }
    let defaults = paper_a_optimized::build_default_hashes(tree_height);
    let height = tree_height as u32;
    let index = frontier.path.index as u64;
    let mut contents = BTreeMap::new();
    let mut current_content = frontier.value;
    let mut current_label = leaf_label(&defaults, height, index, frontier.value);
    let mut key = leaf_key(height, index);
    contents.insert(key, current_content);
    for sibling_content in &frontier.path.siblings {
        let sibling_key = sibling(key);
        contents.insert(sibling_key, *sibling_content);
        let sibling_label = label_from_content(&defaults, height, sibling_key, *sibling_content);
        current_label = if key.pos % 2 == 0 {
            node_hash(current_label, sibling_label)
        } else {
            node_hash(sibling_label, current_label)
        };
        key = parent(key);
        current_content = current_label;
        contents.insert(key, current_content);
    }
    if current_label != frontier.path.root {
        return Err(ArchiveError::InvalidFrontierState(
            "cached frontier witness does not reconstruct to the trusted root".to_string(),
        ));
    }
    Ok(contents)
}

fn node_range(tree_height: usize, node: NodeKey) -> Result<(u64, u64), ArchiveError> {
    let leaf_span_shift = tree_height
        .checked_sub(node.level as usize)
        .ok_or_else(|| {
            ArchiveError::InvalidFrontierState("node level exceeds tree height".to_string())
        })?;
    let width = 1u64
        .checked_shl(leaf_span_shift as u32)
        .ok_or(ArchiveError::CommittedLengthOverflow)?;
    let start = node
        .pos
        .checked_mul(width)
        .ok_or(ArchiveError::CommittedLengthOverflow)?;
    let end = start
        .checked_add(width)
        .ok_or(ArchiveError::CommittedLengthOverflow)?;
    Ok((start, end))
}

fn build_upper_siblings(
    tree_height: usize,
    frontier_len: u64,
    leaf_count: usize,
    frontier: Option<&AppendOnlyLogFrontierLeafState>,
) -> Result<Vec<Value256>, ArchiveError> {
    let upper_nodes = build_upper_path_nodes(tree_height, frontier_len, leaf_count);
    if frontier_len == 0 {
        return Ok(upper_nodes
            .into_iter()
            .map(|node| default_node_content(tree_height, node))
            .collect());
    }
    let frontier = frontier.ok_or(ArchiveError::MissingFrontierWitness)?;
    let known_contents = frontier_known_contents(tree_height, frontier_len, frontier)?;
    let mut upper = Vec::with_capacity(upper_nodes.len());
    for sibling_node in upper_nodes.into_iter() {
        if let Some(content) = known_contents.get(&sibling_node) {
            upper.push(*content);
            continue;
        }
        let (start, end) = node_range(tree_height, sibling_node)?;
        if start >= frontier_len {
            upper.push(default_node_content(tree_height, sibling_node));
            continue;
        }
        if end <= frontier_len {
            return Err(ArchiveError::InvalidFrontierState(format!(
                "cached frontier witness cannot derive committed upper sibling at level {} pos {}",
                sibling_node.level, sibling_node.pos
            )));
        }
        return Err(ArchiveError::InvalidFrontierState(format!(
            "upper sibling at level {} pos {} straddles the committed frontier",
            sibling_node.level, sibling_node.pos
        )));
    }
    Ok(upper)
}

fn build_rightmost_path_inside_subtree(
    tree_height: usize,
    start_index: u64,
    leaves: &[Value256],
) -> Vec<Value256> {
    let defaults = paper_a_optimized::build_default_hashes(tree_height);
    let mut level_contents = leaves.to_vec();
    let mut level_labels: Vec<Value256> = leaves
        .iter()
        .copied()
        .enumerate()
        .map(|(offset, value)| {
            leaf_label(
                &defaults,
                tree_height as u32,
                start_index + offset as u64,
                value,
            )
        })
        .collect();
    let mut siblings = Vec::with_capacity(leaves.len().trailing_zeros() as usize);
    let mut idx = leaves.len() - 1;
    while level_contents.len() > 1 {
        siblings.push(level_contents[idx ^ 1]);
        let mut next_contents = Vec::with_capacity(level_contents.len() / 2);
        let mut next_labels = Vec::with_capacity(level_labels.len() / 2);
        for pair in 0..(level_labels.len() / 2) {
            let parent_label = node_hash(level_labels[2 * pair], level_labels[2 * pair + 1]);
            next_contents.push(parent_label);
            next_labels.push(parent_label);
        }
        idx >>= 1;
        level_contents = next_contents;
        level_labels = next_labels;
    }
    siblings
}

pub struct PreparedBulkUpdatesAppend {
    tree_height: usize,
    subtree_height: usize,
    empty_upper_siblings: Vec<Value256>,
    defaults: paper_a_optimized::DefaultHashes,
    label_scratch: Vec<Value256>,
}

impl PreparedBulkUpdatesAppend {
    pub fn new(tree_height: usize) -> Result<Self, ArchiveError> {
        if !BULK_UPDATES_SIZE.is_power_of_two() {
            return Err(ArchiveError::InvalidFrontierState(
                "bulk updates leaf count must be a power of two".to_string(),
            ));
        }
        let subtree_height = BULK_UPDATES_SIZE.trailing_zeros() as usize;
        if subtree_height > tree_height {
            return Err(ArchiveError::InvalidFrontierState(
                "bulk updates subtree exceeds configured authentication tree height".to_string(),
            ));
        }
        let defaults = paper_a_optimized::build_default_hashes(tree_height);
        let root_level = tree_height - subtree_height;
        let empty_upper_siblings = (0..root_level)
            .map(|offset| default_hash(&defaults, root_level - offset))
            .collect();
        Ok(Self {
            tree_height,
            subtree_height,
            empty_upper_siblings,
            defaults,
            label_scratch: vec![[0u8; 32]; BULK_UPDATES_SIZE],
        })
    }

    fn build_upper_siblings(
        &self,
        frontier_len: u64,
        frontier: Option<&AppendOnlyLogFrontierLeafState>,
    ) -> Result<Vec<Value256>, ArchiveError> {
        if frontier_len == 0 {
            return Ok(self.empty_upper_siblings.clone());
        }
        let upper =
            build_upper_siblings(self.tree_height, frontier_len, BULK_UPDATES_SIZE, frontier)?;
        debug_assert_eq!(upper.len(), self.empty_upper_siblings.len());
        Ok(upper)
    }

    fn rightmost_path_inside_subtree_in_place(
        &mut self,
        start_index: u64,
        leaves: &[Value256],
    ) -> Result<Vec<Value256>, ArchiveError> {
        if leaves.len() != BULK_UPDATES_SIZE {
            return Err(ArchiveError::InvalidFrontierState(format!(
                "prepared bulk updates append expects exactly {} leaves, got {}",
                BULK_UPDATES_SIZE,
                leaves.len()
            )));
        }
        for (offset, leaf) in leaves.iter().copied().enumerate() {
            self.label_scratch[offset] = leaf_label(
                &self.defaults,
                self.tree_height as u32,
                start_index + offset as u64,
                leaf,
            );
        }
        let mut siblings = Vec::with_capacity(self.subtree_height);
        let mut idx = leaves.len() - 1;
        let mut width = leaves.len();
        siblings.push(leaves[idx ^ 1]);
        while width > 1 {
            let next_width = width / 2;
            for pair in 0..next_width {
                self.label_scratch[pair] = node_hash(
                    self.label_scratch[2 * pair],
                    self.label_scratch[2 * pair + 1],
                );
            }
            idx >>= 1;
            width = next_width;
            if width > 1 {
                siblings.push(self.label_scratch[idx ^ 1]);
            }
        }
        Ok(siblings)
    }

    fn frontier_after_bulk_updates_append(
        &mut self,
        start_index: u64,
        leaves: &[Value256],
        upper_siblings: &[Value256],
    ) -> Result<AppendOnlyLogFrontierLeafState, ArchiveError> {
        let new_last_index = start_index
            .checked_add(leaves.len() as u64)
            .and_then(|end| end.checked_sub(1))
            .ok_or(ArchiveError::CommittedLengthOverflow)?;
        let value = *leaves.last().ok_or_else(|| {
            ArchiveError::InvalidFrontierState("missing appended leaf".to_string())
        })?;
        let mut siblings = self.rightmost_path_inside_subtree_in_place(start_index, leaves)?;
        siblings.extend_from_slice(upper_siblings);
        if siblings.len() != self.tree_height {
            return Err(ArchiveError::InvalidFrontierState(
                "aligned append produced a frontier path with wrong height".to_string(),
            ));
        }
        let root = recompute_root(
            self.tree_height as u32,
            new_last_index,
            value,
            &siblings,
            &self.defaults,
        );
        Ok(AppendOnlyLogFrontierLeafState {
            value,
            path: AuthPath {
                index: usize::try_from(new_last_index)
                    .map_err(|_| ArchiveError::CommittedLengthOverflow)?,
                siblings,
                root,
            },
        })
    }

    fn append_prepared_with_upper_siblings<T>(
        &mut self,
        client: &mut CoreAuthClient<T>,
        frontier_len: u64,
        leaves: &[Value256],
        upper_siblings: Vec<Value256>,
        prior_root: Value256,
    ) -> Result<AppendOnlyLogFrontierLeafState, ArchiveError>
    where
        T: AuthTransport + Send + Sync + 'static,
        T::Server: Clone + Send + Sync + 'static,
    {
        let proof = AuthSubtreeProof {
            start_index: frontier_len,
            leaf_count: leaves.len(),
            root: prior_root,
            upper_siblings: upper_siblings.clone(),
        };
        client
            .write_fresh_subtree_with_proof(proof, leaves)
            .map_err(ArchiveError::Auth)?;

        let new_frontier =
            self.frontier_after_bulk_updates_append(frontier_len, leaves, &upper_siblings)?;
        if client.trusted_root() != Some(new_frontier.path.root) {
            return Err(ArchiveError::InvalidFrontierState(
                "trusted root after append does not match the app-local frontier witness"
                    .to_string(),
            ));
        }
        Ok(new_frontier)
    }

    pub fn append_from_prepared_frontier<T>(
        &mut self,
        client: &mut CoreAuthClient<T>,
        frontier_len: u64,
        frontier: Option<&AppendOnlyLogFrontierLeafState>,
        leaves: &[Value256],
    ) -> Result<AppendOnlyLogFrontierLeafState, ArchiveError>
    where
        T: AuthTransport + Send + Sync + 'static,
        T::Server: Clone + Send + Sync + 'static,
    {
        if leaves.len() != BULK_UPDATES_SIZE {
            return Err(ArchiveError::InvalidFrontierState(format!(
                "bulk updates append requires exactly {} leaves, got {}",
                BULK_UPDATES_SIZE,
                leaves.len()
            )));
        }
        let leaf_count = leaves.len() as u64;
        if frontier_len % leaf_count != 0 {
            return Err(ArchiveError::InvalidFrontierState(
                "aligned append start is not aligned to the appended subtree size".to_string(),
            ));
        }
        if self.subtree_height > client.auth.tree_height {
            return Err(ArchiveError::InvalidFrontierState(
                "append subtree is deeper than the configured authentication tree".to_string(),
            ));
        }

        let upper_siblings = self.build_upper_siblings(frontier_len, frontier)?;
        let prior_root = if frontier_len == 0 {
            default_hash(&self.defaults, 0)
        } else {
            frontier
                .ok_or(ArchiveError::MissingFrontierWitness)?
                .path
                .root
        };
        self.append_prepared_with_upper_siblings(
            client,
            frontier_len,
            leaves,
            upper_siblings,
            prior_root,
        )
    }

    pub fn append_from_frontier<T>(
        &mut self,
        client: &mut CoreAuthClient<T>,
        frontier_len: u64,
        frontier: Option<&AppendOnlyLogFrontierLeafState>,
        leaves: &[Value256],
    ) -> Result<AppendOnlyLogFrontierLeafState, ArchiveError>
    where
        T: AuthTransport + Send + Sync + 'static,
        T::Server: Clone + Send + Sync + 'static,
    {
        let leaf_count = leaves.len() as u64;
        let new_len = frontier_len
            .checked_add(leaf_count)
            .ok_or(ArchiveError::CommittedLengthOverflow)?;
        client.extend(new_len).map_err(ArchiveError::Auth)?;
        self.append_from_prepared_frontier(client, frontier_len, frontier, leaves)
    }
}

fn frontier_after_aligned_append(
    tree_height: usize,
    start_index: u64,
    leaves: &[Value256],
    upper_siblings: &[Value256],
) -> Result<AppendOnlyLogFrontierLeafState, ArchiveError> {
    let new_last_index = start_index
        .checked_add(leaves.len() as u64)
        .and_then(|end| end.checked_sub(1))
        .ok_or(ArchiveError::CommittedLengthOverflow)?;
    let value = *leaves
        .last()
        .ok_or_else(|| ArchiveError::InvalidFrontierState("missing appended leaf".to_string()))?;
    let defaults = paper_a_optimized::build_default_hashes(tree_height);
    let mut siblings = build_rightmost_path_inside_subtree(tree_height, start_index, leaves);
    siblings.extend_from_slice(upper_siblings);
    if siblings.len() != tree_height {
        return Err(ArchiveError::InvalidFrontierState(
            "aligned append produced a frontier path with wrong height".to_string(),
        ));
    }
    let root = recompute_root(
        tree_height as u32,
        new_last_index,
        value,
        &siblings,
        &defaults,
    );
    Ok(AppendOnlyLogFrontierLeafState {
        value,
        path: AuthPath {
            index: usize::try_from(new_last_index)
                .map_err(|_| ArchiveError::CommittedLengthOverflow)?,
            siblings,
            root,
        },
    })
}

pub fn append_aligned_subtree_from_frontier<T>(
    client: &mut CoreAuthClient<T>,
    frontier_len: u64,
    frontier: Option<&AppendOnlyLogFrontierLeafState>,
    leaves: &[Value256],
) -> Result<AppendOnlyLogFrontierLeafState, ArchiveError>
where
    T: AuthTransport + Send + Sync + 'static,
    T::Server: Clone + Send + Sync + 'static,
{
    if leaves.is_empty() || !leaves.len().is_power_of_two() {
        return Err(ArchiveError::InvalidFrontierState(
            "aligned append requires a non-empty power-of-two leaf slice".to_string(),
        ));
    }
    let leaf_count = leaves.len() as u64;
    if frontier_len % leaf_count != 0 {
        return Err(ArchiveError::InvalidFrontierState(
            "aligned append start is not aligned to the appended subtree size".to_string(),
        ));
    }
    let tree_height = client.auth.tree_height;
    if leaves.len().trailing_zeros() as usize > tree_height {
        return Err(ArchiveError::InvalidFrontierState(
            "append subtree is deeper than the configured authentication tree".to_string(),
        ));
    }

    let upper_siblings = build_upper_siblings(tree_height, frontier_len, leaves.len(), frontier)?;
    let new_len = frontier_len
        .checked_add(leaf_count)
        .ok_or(ArchiveError::CommittedLengthOverflow)?;
    client.extend(new_len).map_err(ArchiveError::Auth)?;
    let proof = AuthSubtreeProof {
        start_index: frontier_len,
        leaf_count: leaves.len(),
        root: if frontier_len == 0 {
            default_hash(&paper_a_optimized::build_default_hashes(tree_height), 0)
        } else {
            frontier
                .ok_or(ArchiveError::MissingFrontierWitness)?
                .path
                .root
        },
        upper_siblings: upper_siblings.clone(),
    };
    client
        .write_fresh_subtree_with_proof(proof, leaves)
        .map_err(ArchiveError::Auth)?;

    let new_frontier =
        frontier_after_aligned_append(tree_height, frontier_len, leaves, &upper_siblings)?;
    if client.trusted_root() != Some(new_frontier.path.root) {
        return Err(ArchiveError::InvalidFrontierState(
            "trusted root after append does not match the app-local frontier witness".to_string(),
        ));
    }
    Ok(new_frontier)
}

#[cfg(test)]
mod tests {
    use super::*;
    use paper_a_optimized::build_default_hashes;

    #[test]
    fn empty_frontier_uses_default_root() {
        let defaults = build_default_hashes(4);
        let upper = build_upper_siblings(4, 0, 4, None).expect("upper siblings");
        assert_eq!(upper.len(), 2);
        assert_eq!(upper[0], default_hash(&defaults, 2));
        assert_eq!(upper[1], default_hash(&defaults, 1));
    }
}
