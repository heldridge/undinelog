//! Keeping a maintainer's private-opening state current while it serves.
//!
//! After an accepted batch the server's fragment is that of the new anchor
//! and its private state is one update behind. [`PrivateKeeper::after_batch`]
//! takes what the update needs under the server's lock, a copy of the state
//! and the cells of the changed records, and runs the update of
//! Section 6.2 on the copy on its own thread, with the packing tables
//! derived on every core; the server keeps answering from the old state
//! meanwhile and installs the new one when it is ready. While the update
//! runs the maintainer holds two states.

use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;

use crate::auth::service::AuthServer;
use crate::private::layout::Grouping;

/// Updates the private state of a shared server off its lock.
pub struct PrivateKeeper {
    server: Arc<RwLock<AuthServer>>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl PrivateKeeper {
    pub fn new(server: Arc<RwLock<AuthServer>>) -> Self {
        Self {
            server,
            worker: Mutex::new(None),
        }
    }

    /// Start the update of the private state to the server's current
    /// fragment, and return at once. Nothing happens when the server holds
    /// no private state, when that state is current, or when an update is
    /// still running; in the last case the next call, or the next private
    /// read, brings the state to the fragment of its time.
    pub fn after_batch(&self) {
        let mut worker = self.worker.lock().expect("keeper worker");
        if worker.as_ref().is_some_and(|handle| !handle.is_finished()) {
            return;
        }
        let planned = {
            let server = self.server.read().expect("server lock");
            let grouping = Grouping::public(server.allocated_leaves(), server.params.need);
            let Some(state) = server.private_state() else {
                return;
            };
            if state.is_current(&server, grouping) || !state.is_updatable(&server, grouping) {
                return;
            }
            match state.plan_update(&server) {
                Ok(plan) => (state.clone(), plan),
                Err(_) => return,
            }
        };
        let server = Arc::clone(&self.server);
        *worker = Some(std::thread::spawn(move || {
            let (mut state, plan) = planned;
            state.apply_update(&plan);
            let mut server = server.write().expect("server lock");
            // The state is installed only if the fragment is still the one
            // the update led to; otherwise a later call starts over.
            let _ = server.install_private_state(state);
        }));
    }

    /// Wait for a running update to finish.
    pub fn wait(&self) {
        if let Some(handle) = self.worker.lock().expect("keeper worker").take() {
            let _ = handle.join();
        }
    }
}
