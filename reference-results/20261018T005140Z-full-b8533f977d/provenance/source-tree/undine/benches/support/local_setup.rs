//! Finish local setup before timing operations on its server or captured replies.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Reclaim the setup server after all background fetches release their copies.
///
/// Drop the setup client and any other local owners before calling this. A
/// fetch can send its reply before releasing its server reference. Keeping our
/// own reference until it is unique prevents a worker from becoming the last
/// owner and destroying the setup server during a timed client operation.
/// The caller can retain the returned server or drop it before starting timers.
pub fn take_local_server<T>(server: Arc<Mutex<T>>) -> T {
    take_local_server_with_timeout(server, Duration::from_secs(30))
}

fn take_local_server_with_timeout<T>(mut server: Arc<Mutex<T>>, timeout: Duration) -> T {
    let start = Instant::now();
    loop {
        match Arc::try_unwrap(server) {
            Ok(server) => {
                return server
                    .into_inner()
                    .unwrap_or_else(|_| panic!("local setup server lock poisoned"));
            }
            Err(shared) => server = shared,
        }
        assert!(
            start.elapsed() < timeout,
            "local setup still has {} server owners; drop the setup client before timing",
            Arc::strong_count(&server),
        );
        std::thread::yield_now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn unique_server_is_returned_without_changing_its_state() {
        let server = Arc::new(Mutex::new(vec![1, 2, 3]));
        assert_eq!(take_local_server(server), vec![1, 2, 3]);
    }

    #[test]
    fn worker_cannot_destroy_the_server_during_client_processing() {
        struct Server(Arc<Mutex<Option<std::thread::ThreadId>>>);
        impl Drop for Server {
            fn drop(&mut self) {
                *self.0.lock().unwrap() = Some(std::thread::current().id());
            }
        }

        let destroyed_by = Arc::new(Mutex::new(None));
        let server = Arc::new(Mutex::new(Server(Arc::clone(&destroyed_by))));
        let worker_server = Arc::clone(&server);
        let (release_tx, release_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            release_rx.recv().unwrap();
            drop(worker_server);
        });
        let (started_tx, started_rx) = mpsc::channel();
        let (server_tx, server_rx) = mpsc::channel();
        let owner = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            server_tx.send(take_local_server(server)).unwrap();
        });
        started_rx.recv().unwrap();
        assert!(matches!(
            server_rx.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
        release_tx.send(()).unwrap();
        let server = server_rx.recv().unwrap();
        worker.join().unwrap();
        owner.join().unwrap();
        assert!(destroyed_by.lock().unwrap().is_none());
        drop(server);
        assert_eq!(
            *destroyed_by.lock().unwrap(),
            Some(std::thread::current().id())
        );
    }

    #[test]
    #[should_panic(expected = "drop the setup client before timing")]
    fn retained_local_owner_fails_instead_of_waiting_forever() {
        let server = Arc::new(Mutex::new(()));
        let _retained = Arc::clone(&server);
        take_local_server_with_timeout(server, Duration::ZERO);
    }
}
