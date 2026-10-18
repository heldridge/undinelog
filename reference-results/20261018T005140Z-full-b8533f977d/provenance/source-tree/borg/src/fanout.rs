//! Shared fan-out helpers for client orchestration.

use crate::shares::ShardShare;
use std::collections::HashSet;
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc,
};
use std::thread;

type FanoutJob = Box<dyn FnOnce() + Send + 'static>;

pub(crate) fn first_ok<S, T, E, F>(servers: &[S], mut fetch: F) -> Option<T>
where
    F: FnMut(&S) -> Result<T, E>,
{
    for server in servers {
        if let Ok(value) = fetch(server) {
            return Some(value);
        }
    }
    None
}

pub(crate) struct ParallelFetches<T, E> {
    rx: mpsc::Receiver<(usize, Result<T, E>)>,
    cancel: Arc<AtomicBool>,
}

impl<T, E> ParallelFetches<T, E> {
    pub(crate) fn recv(&self) -> Result<(usize, Result<T, E>), mpsc::RecvError> {
        self.rx.recv()
    }

    pub(crate) fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl<T, E> Drop for ParallelFetches<T, E> {
    fn drop(&mut self) {
        self.cancel();
    }
}

fn fanout_parallelism_cap() -> usize {
    static FANOUT_PARALLELISM: OnceLock<usize> = OnceLock::new();
    *FANOUT_PARALLELISM.get_or_init(|| {
        let configured = std::env::var("NET_FANOUT_PARALLELISM")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or_else(|| {
                thread::available_parallelism()
                    .map(|parallelism| parallelism.get().saturating_mul(4))
                    .unwrap_or(16)
            });
        configured.clamp(1, 256)
    })
}

struct FanoutPool {
    tx: mpsc::Sender<FanoutJob>,
}

fn fanout_pool() -> &'static FanoutPool {
    static POOL: OnceLock<FanoutPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let worker_count = fanout_parallelism_cap();
        let (tx, rx) = mpsc::channel::<FanoutJob>();
        let rx = Arc::new(Mutex::new(rx));
        for worker_idx in 0..worker_count {
            let rx = Arc::clone(&rx);
            thread::Builder::new()
                .name(format!("fanout-{worker_idx}"))
                .spawn(move || {
                    loop {
                        let job = {
                            let lock = rx.lock().expect("fanout pool receiver poisoned");
                            lock.recv()
                        };
                        match job {
                            Ok(job) => job(),
                            Err(_) => break,
                        }
                    }
                })
                .expect("failed to spawn fanout worker");
        }
        FanoutPool { tx }
    })
}

pub(crate) fn spawn_parallel_fetches<S, T, E, F>(servers: &[S], fetch: F) -> ParallelFetches<T, E>
where
    S: Clone + Send + Sync + 'static,
    T: Send + 'static,
    E: Send + 'static,
    F: Fn(S) -> Result<T, E> + Send + Sync + 'static,
{
    let (tx, rx) = mpsc::channel();
    if servers.is_empty() {
        return ParallelFetches {
            rx,
            cancel: Arc::new(AtomicBool::new(true)),
        };
    }

    let fetch = Arc::new(fetch);
    let servers = Arc::new(servers.iter().cloned().collect::<Vec<_>>());
    let cancel = Arc::new(AtomicBool::new(false));
    let next_idx = Arc::new(AtomicUsize::new(0));
    let worker_count = servers.len().min(fanout_parallelism_cap());
    let pool = fanout_pool();
    for _ in 0..worker_count {
        let tx = tx.clone();
        let fetch = Arc::clone(&fetch);
        let servers = Arc::clone(&servers);
        let cancel_flag = Arc::clone(&cancel);
        let next_idx = Arc::clone(&next_idx);
        // The shared pool bounds in-flight RPC fan-out without paying thread
        // startup costs on every request. Timeout-backed transports still cap
        // any work that is already in flight when cancellation arrives.
        pool.tx
            .send(Box::new(move || {
                loop {
                    if cancel_flag.load(Ordering::Relaxed) {
                        return;
                    }
                    let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                    if idx >= servers.len() {
                        return;
                    }
                    let reply = fetch(servers[idx].clone());
                    if cancel_flag.load(Ordering::Relaxed) {
                        return;
                    }
                    if tx.send((idx, reply)).is_err() {
                        return;
                    }
                }
            }))
            .expect("fanout pool unavailable");
    }
    drop(tx);
    ParallelFetches { rx, cancel }
}

pub(crate) fn first_ok_parallel<S, T, E, F>(servers: &[S], fetch: F) -> Option<T>
where
    S: Clone + Send + Sync + 'static,
    T: Send + 'static,
    E: Send + 'static,
    F: Fn(S) -> Result<T, E> + Send + Sync + 'static,
{
    if servers.len() <= 1 {
        return first_ok(servers, |server| fetch(server.clone()));
    }
    let fetches = spawn_parallel_fetches(servers, fetch);
    for _ in 0..servers.len() {
        let Ok((_, reply)) = fetches.recv() else {
            break;
        };
        if let Ok(value) = reply {
            fetches.cancel();
            return Some(value);
        }
    }
    None
}

pub(crate) fn collect_shares_by_pos<S, E, F, G>(
    servers: &[S],
    expected_len: usize,
    need: usize,
    mut fetch: F,
    mut invalid_len: G,
) -> Result<(Vec<Vec<ShardShare>>, usize), E>
where
    F: FnMut(&S) -> Result<Vec<Vec<ShardShare>>, E>,
    G: FnMut(usize, usize) -> E,
{
    let mut shares_by_pos = vec![Vec::new(); expected_len];
    let mut seen_by_pos: Vec<HashSet<usize>> = (0..expected_len).map(|_| HashSet::new()).collect();
    let mut remaining = expected_len;
    let mut satisfied = vec![false; expected_len];

    for server in servers {
        if remaining == 0 {
            break;
        }
        let reply = match fetch(server) {
            Ok(reply) => reply,
            Err(_) => continue,
        };
        if reply.len() != expected_len {
            return Err(invalid_len(expected_len, reply.len()));
        }
        for (i, shard_shares) in reply.into_iter().enumerate() {
            if satisfied[i] {
                continue;
            }
            for share in shard_shares {
                if share.value.is_some() && seen_by_pos[i].insert(share.registry_index) {
                    shares_by_pos[i].push(share);
                    if shares_by_pos[i].len() >= need {
                        satisfied[i] = true;
                        remaining = remaining.saturating_sub(1);
                        break;
                    }
                }
            }
        }
    }

    Ok((shares_by_pos, remaining))
}

pub(crate) fn collect_shares_by_pos_parallel<S, E, F, G>(
    servers: &[S],
    expected_len: usize,
    need: usize,
    fetch: F,
    mut invalid_len: G,
) -> Result<(Vec<Vec<ShardShare>>, usize), E>
where
    S: Clone + Send + Sync + 'static,
    E: Send + 'static,
    F: Fn(S) -> Result<Vec<Vec<ShardShare>>, E> + Send + Sync + 'static,
    G: FnMut(usize, usize) -> E,
{
    if servers.len() <= 1 {
        return collect_shares_by_pos(
            servers,
            expected_len,
            need,
            |server| fetch(server.clone()),
            invalid_len,
        );
    }

    let mut shares_by_pos = vec![Vec::new(); expected_len];
    let mut seen_by_pos: Vec<HashSet<usize>> = (0..expected_len).map(|_| HashSet::new()).collect();
    let mut remaining = expected_len;
    let mut satisfied = vec![false; expected_len];
    let fetches = spawn_parallel_fetches(servers, fetch);

    for _ in 0..servers.len() {
        if remaining == 0 {
            fetches.cancel();
            break;
        }
        let Ok((_, reply)) = fetches.recv() else {
            break;
        };
        let reply = match reply {
            Ok(reply) => reply,
            Err(_) => continue,
        };
        if reply.len() != expected_len {
            fetches.cancel();
            return Err(invalid_len(expected_len, reply.len()));
        }
        for (i, shard_shares) in reply.into_iter().enumerate() {
            if satisfied[i] {
                continue;
            }
            for share in shard_shares {
                if share.value.is_some() && seen_by_pos[i].insert(share.registry_index) {
                    shares_by_pos[i].push(share);
                    if shares_by_pos[i].len() >= need {
                        satisfied[i] = true;
                        remaining = remaining.saturating_sub(1);
                        break;
                    }
                }
            }
        }
    }

    Ok((shares_by_pos, remaining))
}

pub(crate) fn broadcast_with_failures<S, R, E, L, F, H>(
    servers: &[S],
    mut label: L,
    mut call: F,
    mut handle: H,
) -> Vec<(String, String)>
where
    E: std::fmt::Debug,
    L: FnMut(&S) -> String,
    F: FnMut(&S) -> Result<R, E>,
    H: FnMut(R) -> Result<(), String>,
{
    let mut failed = Vec::new();
    for server in servers {
        let name = label(server);
        match call(server) {
            Ok(resp) => {
                if let Err(msg) = handle(resp) {
                    failed.push((name, msg));
                }
            }
            Err(err) => failed.push((name, format!("{:?}", err))),
        }
    }
    failed
}

pub(crate) fn broadcast_with_failures_parallel<S, R, E, L, F, H>(
    servers: &[S],
    mut label: L,
    call: F,
    mut handle: H,
) -> Vec<(String, String)>
where
    S: Clone + Send + Sync + 'static,
    R: Send + 'static,
    E: std::fmt::Debug + Send + 'static,
    L: FnMut(&S) -> String,
    F: Fn(S) -> Result<R, E> + Send + Sync + 'static,
    H: FnMut(R) -> Result<(), String>,
{
    if servers.len() <= 1 {
        return broadcast_with_failures(servers, label, |server| call(server.clone()), handle);
    }

    let labels: Vec<String> = servers.iter().map(&mut label).collect();
    let fetches = spawn_parallel_fetches(servers, call);
    let mut failed = Vec::new();
    for _ in 0..servers.len() {
        let Ok((idx, reply)) = fetches.recv() else {
            break;
        };
        let name = labels[idx].clone();
        match reply {
            Ok(resp) => {
                if let Err(msg) = handle(resp) {
                    failed.push((idx, name, msg));
                }
            }
            Err(err) => failed.push((idx, name, format!("{:?}", err))),
        }
    }
    failed.sort_by_key(|(idx, _, _)| *idx);
    failed
        .into_iter()
        .map(|(_, name, msg)| (name, msg))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::fp_one;
    use crate::shares::ShardShare;
    use std::time::Duration;

    fn make_share(registry_index: usize) -> ShardShare {
        ShardShare {
            registry_index,
            value: Some(fp_one()),
        }
    }

    fn make_none_share(registry_index: usize) -> ShardShare {
        ShardShare {
            registry_index,
            value: None,
        }
    }

    #[test]
    fn first_ok_returns_first_success() {
        let servers = vec![1, 2, 3];
        let mut hits = Vec::new();
        let result = first_ok(&servers, |server| {
            hits.push(*server);
            if *server == 2 { Ok("ok") } else { Err("err") }
        });
        assert_eq!(result, Some("ok"));
        assert_eq!(hits, vec![1, 2]);
    }

    #[test]
    fn first_ok_none_on_all_err() {
        let servers = vec![1, 2];
        let result: Option<i32> = first_ok(&servers, |_server| Err("err"));
        assert_eq!(result, None);
    }

    #[test]
    fn first_ok_parallel_returns_first_success_by_arrival() {
        let servers = vec![1, 2, 3];
        let result = first_ok_parallel(&servers, |server| {
            if server == 1 {
                thread::sleep(Duration::from_millis(20));
                Err("late err")
            } else if server == 2 {
                Ok("ok")
            } else {
                thread::sleep(Duration::from_millis(10));
                Err("err")
            }
        });
        assert_eq!(result, Some("ok"));
    }

    #[test]
    fn first_ok_parallel_cancels_queued_work() {
        let servers: Vec<_> = (0..fanout_parallelism_cap() * 4).collect();
        let started = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let started_for_fetch = std::sync::Arc::clone(&started);
        let result = first_ok_parallel(&servers, move |server| {
            started_for_fetch.fetch_add(1, Ordering::Relaxed);
            if server == 0 {
                Ok("ok")
            } else {
                thread::sleep(Duration::from_millis(20));
                Err("late err")
            }
        });
        assert_eq!(result, Some("ok"));
        assert!(started.load(Ordering::Relaxed) < servers.len());
    }

    #[test]
    fn collect_shares_by_pos_aggregates_until_need() {
        let servers = vec![0, 1];
        let (shares_by_pos, remaining) = collect_shares_by_pos(
            &servers,
            2,
            2,
            |server| match server {
                0 => Ok(vec![
                    vec![make_share(0), make_share(1)],
                    vec![make_share(0), make_none_share(2)],
                ]),
                1 => Ok(vec![vec![make_share(0)], vec![make_share(1)]]),
                _ => unreachable!(),
            },
            |expected, got| format!("expected {expected} got {got}"),
        )
        .expect("collect_shares_by_pos should succeed");
        assert_eq!(remaining, 0);
        assert_eq!(shares_by_pos.len(), 2);
        assert_eq!(shares_by_pos[0].len(), 2);
        assert_eq!(shares_by_pos[1].len(), 2);
    }

    #[test]
    fn collect_shares_by_pos_reports_invalid_len() {
        let servers = vec![0];
        let err = collect_shares_by_pos(
            &servers,
            1,
            1,
            |_server| Ok(vec![vec![make_share(0)], vec![make_share(1)]]),
            |expected, got| format!("expected {expected} got {got}"),
        )
        .expect_err("length mismatch should error");
        assert_eq!(err, "expected 1 got 2");
    }

    #[test]
    fn collect_shares_by_pos_parallel_aggregates_until_need() {
        let servers = vec![0, 1];
        let (shares_by_pos, remaining) = collect_shares_by_pos_parallel(
            &servers,
            2,
            2,
            |server| match server {
                0 => Ok(vec![
                    vec![make_share(0), make_share(1)],
                    vec![make_share(0), make_none_share(2)],
                ]),
                1 => Ok(vec![vec![make_share(0)], vec![make_share(1)]]),
                _ => unreachable!(),
            },
            |expected, got| format!("expected {expected} got {got}"),
        )
        .expect("collect_shares_by_pos_parallel should succeed");
        assert_eq!(remaining, 0);
        assert_eq!(shares_by_pos[0].len(), 2);
        assert_eq!(shares_by_pos[1].len(), 2);
    }

    #[derive(Debug)]
    struct TestErr(&'static str);

    #[test]
    fn broadcast_with_failures_collects_errors() {
        let servers = vec![1, 2, 3];
        let mut calls = Vec::new();
        let failed = broadcast_with_failures(
            &servers,
            |server| format!("srv-{server}"),
            |server| {
                calls.push(*server);
                if *server == 2 {
                    let err = TestErr("boom");
                    let _ = err.0;
                    Err(err)
                } else {
                    Ok(*server)
                }
            },
            |resp| {
                if resp == 3 {
                    Err("bad response".to_string())
                } else {
                    Ok(())
                }
            },
        );
        assert_eq!(calls, vec![1, 2, 3]);
        assert_eq!(
            failed,
            vec![
                ("srv-2".to_string(), "TestErr(\"boom\")".to_string()),
                ("srv-3".to_string(), "bad response".to_string()),
            ]
        );
    }

    #[test]
    fn broadcast_with_failures_parallel_collects_errors() {
        let servers = vec![1, 2, 3];
        let failed = broadcast_with_failures_parallel(
            &servers,
            |server| format!("srv-{server}"),
            |server| {
                if server == 2 {
                    let err = TestErr("boom");
                    let _ = err.0;
                    Err(err)
                } else {
                    Ok(server)
                }
            },
            |resp| {
                if resp == 3 {
                    Err("bad response".to_string())
                } else {
                    Ok(())
                }
            },
        );
        assert_eq!(
            failed,
            vec![
                ("srv-2".to_string(), "TestErr(\"boom\")".to_string()),
                ("srv-3".to_string(), "bad response".to_string()),
            ]
        );
    }
}
