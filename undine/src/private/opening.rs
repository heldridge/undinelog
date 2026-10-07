//! One private opening: the request a client sends to every selected
//! maintainer, a maintainer's reply, and the client's decoding of a reply
//! into the shares of a public opening.

use std::sync::Arc;

use crate::auth::AuthParams;
use crate::auth::mapping::NodeLinearization;
use crate::auth::merkle::{NodeKey, leaf_key, parent, sibling};
use crate::auth::service::{AuthReadNodeReply, AuthReadResponse, AuthServer};
use crate::private::layout::{Grouping, LayoutError, PlannedEntry, PrivateLayout};
use crate::private::pir::{PackKeys, PirClient, PirDatabase, PirShape, entry_from_cells};
use crate::private::records::local_claims;
use crate::shares::ShardShare;
use crate::unauth::CodedArrayParams;
use crate::value::Value256;

/// The request of a private opening. It is the same for every maintainer
/// and for every position of a log. The groups and the pattern size are the
/// public grouping of the log length and the threshold, which the client
/// and the maintainer both derive from these two values.
#[derive(Clone, Debug)]
pub struct PrivateReadRequest {
    /// Filled leaves of the log the client opens: the anchor's length.
    pub leaves: u64,
    /// Threshold at which the client combines the replies.
    pub kappa: usize,
    /// Leaves per pattern, as a power of two: the public pattern size of
    /// `leaves` and `kappa`, repeated so that a maintainer refuses a request
    /// built for another one.
    pub pattern_bits: u32,
    /// The PIR query: one coefficient per database row.
    pub query: Vec<u64>,
    /// The packing keys of the client's secret key.
    pub pack_keys: PackKeys,
}

/// The reply of one claim.
#[derive(Clone, Debug)]
pub struct PrivateReadReply {
    pub registry_index: usize,
    pub response: Vec<Vec<u8>>,
}

/// The reply of one maintainer: one PIR response per claim it holds.
#[derive(Clone, Debug)]
pub struct PrivateReadResponse {
    pub replies: Vec<PrivateReadReply>,
}

/// The client side of one private opening.
pub struct PrivateOpening {
    index: u64,
    tree_height: usize,
    layout: PrivateLayout,
    plan: Vec<PlannedEntry>,
    client: PirClient,
}

impl PrivateOpening {
    /// Prepare the opening of leaf `index` of a log of `leaves` filled
    /// leaves under the public grouping, with a fresh secret key.
    pub fn for_log(
        params: &CodedArrayParams,
        auth: &AuthParams,
        node_linearization: NodeLinearization,
        leaves: u64,
        index: u64,
    ) -> Result<Self, LayoutError> {
        let grouping = Grouping::public(leaves, params.s);
        Self::with_grouping(params, auth, node_linearization, leaves, grouping, index)
    }

    /// Prepare the opening of leaf `index` of a log of `leaves` filled
    /// leaves with groups of `segments_per_record` segments and the largest
    /// pattern whose subtree fits in one group, with a fresh secret key.
    pub fn new(
        params: &CodedArrayParams,
        auth: &AuthParams,
        node_linearization: NodeLinearization,
        leaves: u64,
        segments_per_record: usize,
        index: u64,
    ) -> Result<Self, LayoutError> {
        if segments_per_record == 0 || !segments_per_record.is_power_of_two() {
            return Err(LayoutError::SegmentsPerRecord(segments_per_record));
        }
        let grouping = Grouping::one_group(segments_per_record, params.n_grid);
        Self::with_grouping(params, auth, node_linearization, leaves, grouping, index)
    }

    /// Prepare the opening of leaf `index` of a log of `leaves` filled
    /// leaves with the groups and the pattern size of `grouping`, with a
    /// fresh secret key. A maintainer answers only the public grouping of
    /// the log length.
    pub fn with_grouping(
        params: &CodedArrayParams,
        auth: &AuthParams,
        node_linearization: NodeLinearization,
        leaves: u64,
        grouping: Grouping,
        index: u64,
    ) -> Result<Self, LayoutError> {
        let layout =
            PrivateLayout::with_grouping(params, auth, node_linearization, leaves, grouping)?;
        let plan = layout.opening_plan(index);
        let client = PirClient::generate(PirShape::of(&layout));
        Ok(Self {
            index,
            tree_height: auth.tree_height,
            layout,
            plan,
            client,
        })
    }

    /// The layout this opening reads.
    pub fn layout(&self) -> &PrivateLayout {
        &self.layout
    }

    /// The request to send to every selected maintainer.
    pub fn request(&self) -> PrivateReadRequest {
        PrivateReadRequest {
            leaves: self.layout.leaves(),
            kappa: self.layout.kappa(),
            pattern_bits: self.layout.group_bits(),
            query: self.client.query(self.layout.pattern_of(self.index)),
            pack_keys: self.client.pack_keys(),
        }
    }

    /// Decode a maintainer's reply into the shares of a public opening of
    /// the same leaf. A reply of one claim that does not decode contributes
    /// no share.
    pub fn decode(&self, response: &PrivateReadResponse) -> AuthReadResponse {
        let mut nodes = Vec::with_capacity(self.tree_height + 1);
        let mut key = leaf_key(self.tree_height as u32, self.index);
        nodes.push(key);
        for _ in 0..self.tree_height {
            nodes.push(sibling(key));
            key = parent(key);
        }
        let mut replies = nodes
            .into_iter()
            .map(|node| AuthReadNodeReply {
                node,
                shares: Vec::new(),
            })
            .collect::<Vec<_>>();
        let position_of = |node: NodeKey| -> usize {
            if node.level as usize == self.tree_height && node.pos == self.index {
                0
            } else {
                self.tree_height - node.level as usize + 1
            }
        };
        for reply in &response.replies {
            let Ok(slots) = self.client.decode(&reply.response) else {
                continue;
            };
            for entry in &self.plan {
                let value = entry_from_cells(&slots[entry.slot], entry.entry_in_record);
                replies[position_of(entry.node)]
                    .shares
                    .push(ShardShare::new(reply.registry_index, Some(value)));
            }
        }
        AuthReadResponse { nodes: replies }
    }
}

/// Run `work` and return its result at `release`, not before. The protocol
/// of a private opening fixes the time of the client's next action in
/// advance, so that it does not depend on the replies; a client that acts
/// on an opening as soon as it is decoded would let a maintainer time a
/// lie, since correcting replies takes longer than checking them. The
/// release must lie past the slowest decoding: a result that is ready
/// only after it is returned then.
pub fn release_at<T>(release: std::time::Instant, work: impl FnOnce() -> T) -> T {
    let result = work();
    if let Some(remaining) = release.checked_duration_since(std::time::Instant::now()) {
        std::thread::sleep(remaining);
    }
    result
}

/// What [`PrivateState::apply_update`] needs: the layout of the new log
/// length, the records whose cells changed, and those cells per database,
/// read from the server by [`PrivateState::plan_update`].
pub struct PlannedUpdate {
    leaves: u64,
    root: Value256,
    layout: PrivateLayout,
    changed: Vec<usize>,
    cells: Vec<Vec<u16>>,
}

impl PlannedUpdate {
    /// The anchor the update leads to.
    pub fn anchor(&self) -> (u64, Value256) {
        (self.leaves, self.root)
    }
}

/// A maintainer's preprocessed state for private openings: one PIR database
/// per claim, for one log length and one grouping.
#[derive(Clone)]
pub struct PrivateState {
    leaves: usize,
    root: Value256,
    kappa: usize,
    grouping: Grouping,
    databases: Vec<(usize, PirDatabase)>,
    /// Updates applied since the state was built.
    updates: usize,
}

impl PrivateState {
    /// Preprocess the claims of `server` for the groups and the pattern
    /// size of `grouping`.
    pub fn build(server: &AuthServer, grouping: Grouping) -> Result<Self, String> {
        let leaves = server.allocated_leaves();
        let layout = PrivateLayout::with_grouping(
            &server.params,
            &server.auth,
            server.node_linearization,
            leaves,
            grouping,
        )
        .map_err(|err| err.to_string())?;
        let store = &server.store.unauth;
        let databases = local_claims(store)
            .into_iter()
            .map(|registry_index| {
                (registry_index, PirDatabase::build(store, &layout, registry_index))
            })
            .collect();
        Ok(Self {
            leaves: leaves as usize,
            root: server.store.root,
            kappa: server.params.need,
            grouping,
            databases,
            updates: 0,
        })
    }

    /// Log length this state serves.
    pub fn leaves(&self) -> u64 {
        self.leaves as u64
    }

    /// Updates applied since the state was built.
    pub fn updates(&self) -> usize {
        self.updates
    }

    /// Whether this state can be brought to the current fragment of `server`
    /// by an update: the threshold and the grouping are the same and the
    /// log is at least as long.
    pub fn is_updatable(&self, server: &AuthServer, grouping: Grouping) -> bool {
        self.kappa == server.params.need
            && self.grouping == grouping
            && self.leaves as u64 <= server.allocated_leaves()
    }

    /// Bring every database to the current fragment of `server`.
    ///
    /// Every update of the log is an append, so when the log grew only the
    /// records from the old end on and those on the path of the first new
    /// leaf are read; when it did not, every record is compared with its
    /// stored cells. A maintainer that rewrote leaves below the old end
    /// builds the state again instead.
    pub fn update(&mut self, server: &AuthServer) -> Result<(), String> {
        let leaves = server.allocated_leaves();
        let layout = PrivateLayout::with_grouping(
            &server.params,
            &server.auth,
            server.node_linearization,
            leaves,
            self.grouping,
        )
        .map_err(|err| err.to_string())?;
        let appended =
            (leaves > self.leaves as u64).then(|| layout.records_changed_by_append(self.leaves as u64));
        for (registry_index, database) in &mut self.databases {
            let _ = database.update(
                &server.store.unauth,
                &layout,
                *registry_index,
                appended.as_deref(),
            );
        }
        self.leaves = leaves as usize;
        self.root = server.store.root;
        self.updates += 1;
        Ok(())
    }

    /// What an update to the current fragment of `server` needs, read under
    /// the server's lock: the layout of the new length and, per database,
    /// the records that changed with their cells. The update itself,
    /// [`Self::apply_update`], then runs without the server. The records
    /// are those of an append when the log grew, and every record when it
    /// did not.
    pub fn plan_update(&self, server: &AuthServer) -> Result<PlannedUpdate, String> {
        let leaves = server.allocated_leaves();
        let layout = PrivateLayout::with_grouping(
            &server.params,
            &server.auth,
            server.node_linearization,
            leaves,
            self.grouping,
        )
        .map_err(|err| err.to_string())?;
        let changed = if leaves > self.leaves as u64 {
            layout.records_changed_by_append(self.leaves as u64)
        } else {
            (0..layout.records()).collect()
        };
        let cells = self
            .databases
            .iter()
            .map(|(registry_index, _)| {
                PirDatabase::cells_of_records(&server.store.unauth, &layout, *registry_index, &changed)
            })
            .collect();
        Ok(PlannedUpdate {
            leaves,
            root: server.store.root,
            layout,
            changed,
            cells,
        })
    }

    /// Bring every database to the fragment that `plan` describes.
    pub fn apply_update(&mut self, plan: &PlannedUpdate) {
        for ((_, database), cells) in self.databases.iter_mut().zip(&plan.cells) {
            let _ = database.update_from_cells(&plan.layout, &plan.changed, cells);
        }
        self.leaves = plan.leaves as usize;
        self.root = plan.root;
        self.updates += 1;
    }

    /// Whether this state was built from the current fragment of `server`
    /// with the given grouping.
    pub fn is_current(&self, server: &AuthServer, grouping: Grouping) -> bool {
        self.leaves as u64 == server.allocated_leaves()
            && self.root == server.store.root
            && self.kappa == server.params.need
            && self.grouping == grouping
    }

    /// Answer a request against every claim.
    pub fn answer(&self, request: &PrivateReadRequest) -> Result<PrivateReadResponse, String> {
        let replies = self
            .databases
            .iter()
            .map(|(registry_index, database)| {
                database
                    .answer(&request.query, &request.pack_keys)
                    .map(|response| PrivateReadReply {
                        registry_index: *registry_index,
                        response,
                    })
                    .map_err(|err| err.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PrivateReadResponse { replies })
    }
}

impl AuthServer {
    /// Preprocess this server's claims for private openings with the groups
    /// and the pattern size of `grouping`. The state is reused while the
    /// fragment is unchanged and updated after an append; a change of
    /// threshold or grouping builds it again.
    pub fn prepare_private(&mut self, grouping: Grouping) -> Result<(), String> {
        if let Some(state) = self.private_state.take() {
            if state.is_current(self, grouping) {
                self.private_state = Some(state);
                return Ok(());
            }
            // After an append, preprocess only the difference.
            if state.is_updatable(self, grouping) {
                if let Ok(mut state) = Arc::try_unwrap(state) {
                    state.update(self)?;
                    self.private_state = Some(Arc::new(state));
                    return Ok(());
                }
            }
        }
        self.private_state = Some(Arc::new(PrivateState::build(self, grouping)?));
        Ok(())
    }

    /// The preprocessed state for private openings, once prepared.
    pub fn private_state(&self) -> Option<&PrivateState> {
        self.private_state.as_deref()
    }

    /// Install a state prepared elsewhere, if it is the state of this
    /// server's current fragment; returns whether it was.
    pub fn install_private_state(&mut self, state: PrivateState) -> bool {
        let grouping = Grouping::public(self.allocated_leaves(), self.params.need);
        if !state.is_current(self, grouping) {
            return false;
        }
        self.private_state = Some(Arc::new(state));
        true
    }

    /// Answer a private opening at the stored threshold, the current log
    /// length, and the public grouping of that length.
    pub fn handle_private_read(
        &mut self,
        request: &PrivateReadRequest,
    ) -> Result<PrivateReadResponse, String> {
        if request.kappa != self.params.need {
            return Err(format!(
                "private openings are served at the stored threshold {}",
                self.params.need
            ));
        }
        if request.leaves != self.allocated_leaves() {
            return Err(format!(
                "private opening for a log of {} leaves; this log has {}",
                request.leaves,
                self.allocated_leaves()
            ));
        }
        // The public grouping changes with the log length, so a state of
        // an earlier grouping is built again here. A pattern is at most the
        // tree, as the layout has it, so the request names that size.
        let grouping = Grouping::public(request.leaves, self.params.need);
        let pattern_bits = grouping.pattern_bits.min(self.auth.tree_height as u32);
        if request.pattern_bits != pattern_bits {
            return Err(format!(
                "private openings of a log of {} leaves at threshold {} use patterns of 2^{} leaves",
                request.leaves, self.params.need, pattern_bits
            ));
        }
        self.prepare_private(grouping)?;
        self.private_state
            .as_ref()
            .expect("private state was just prepared")
            .answer(request)
    }
}
