use append_only_log_demo::registry::{AppendOnlyLogAnchorFile, AppendOnlyLogRegistrySource};
use paper_a_optimized::core::NodeLinearization;
use paper_a_optimized::net::registry::{
    MockRegistryFile, NetAuthConfig, NetSystemConfig, NetUnauthConfig, PartyAssignments, PartyInfo,
    default_unauth_claims, default_unauth_epoch,
};
use paper_a_optimized::net::{self, Message, Response};
use paper_a_optimized::{CodedArrayConfig, build_coded_params, build_shard_registry_deterministic};
use serde_json::Value;
use std::fs;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const KAPPA: usize = 2;
const TREE_HEIGHT: usize = 18;
const SEED: u64 = 23;

struct TempWorkspace {
    path: PathBuf,
}

impl TempWorkspace {
    fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "append_log_cli_correcting_{}_{}",
            std::process::id(),
            now
        ));
        fs::create_dir_all(&path).expect("create temp workspace");
        Self { path }
    }

    fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TempWorkspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

struct ServerCluster {
    children: Vec<Child>,
}

impl Drop for ServerCluster {
    fn drop(&mut self) {
        for child in &mut self.children {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

struct ProxyGuard {
    handle: Option<net::ServerHandle>,
    corruptions: Arc<AtomicUsize>,
}

impl ProxyGuard {
    fn corruption_count(&self) -> usize {
        self.corruptions.load(Ordering::SeqCst)
    }
}

impl Drop for ProxyGuard {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.stop();
        }
    }
}

#[test]
fn client_binary_correcting_read_filters_bad_helpers_and_verifies_proof() {
    let tmp = TempWorkspace::new();
    let registry_path = tmp.join("registry.json");
    let anchor_path = tmp.join("anchor.json");

    let party_count = 3 * 2 + 1;
    let real_addrs = reserve_loopback_addrs(party_count);
    let mut registry = build_registry(&real_addrs);
    write_json(&registry_path, &registry);
    write_anchor(&anchor_path, &registry_path);

    let server_bin = env!("CARGO_BIN_EXE_append_only_log_demo_server");
    let client_bin = env!("CARGO_BIN_EXE_append_only_log_demo_client");
    let _cluster = start_servers(server_bin, &registry_path, &real_addrs);

    let emit = Command::new(server_bin)
        .env("NET_RPC_TIMEOUT_MS", "60000")
        .args([
            "--op",
            "emit-mock",
            "--anchor",
            path_str(&anchor_path),
            "--height",
            "0",
        ])
        .output()
        .expect("run emit-mock");
    assert_success("emit-mock", &emit);

    let proxies = install_corrupting_proxies(&mut registry, &[0, 1], &real_addrs);
    write_json(&registry_path, &registry);

    let read = Command::new(client_bin)
        .env("NET_FANOUT_PARALLELISM", "1")
        .env("NET_RPC_TIMEOUT_MS", "60000")
        .args([
            "--op",
            "read-leaf",
            "--anchor",
            path_str(&anchor_path),
            "--index",
            "0",
            "--max-errors",
            "2",
        ])
        .output()
        .expect("run correcting read-leaf");
    let stdout = assert_success("correcting read-leaf", &read);
    let value: Value = serde_json::from_str(&stdout).expect("parse read-leaf json");

    assert_eq!(value["correcting"].as_bool(), Some(true));
    assert_eq!(value["max_errors"].as_u64(), Some(2));
    assert_eq!(value["proof_root_matches_trusted"].as_bool(), Some(true));
    assert_eq!(value["sibling_count"].as_u64(), Some(TREE_HEIGHT as u64));
    assert_eq!(proxies.len(), 2);
    for proxy in &proxies {
        assert!(
            proxy.corruption_count() > 0,
            "proxy did not corrupt any AuthRead replies"
        );
    }
}

#[test]
fn client_binary_opens_at_a_higher_threshold_from_servers_at_the_registry_threshold() {
    let tmp = TempWorkspace::new();
    let registry_path = tmp.join("registry.json");
    let anchor_path = tmp.join("anchor.json");

    let addrs = reserve_loopback_addrs(4 * KAPPA);
    write_json(&registry_path, &build_registry(&addrs));
    write_anchor(&anchor_path, &registry_path);

    let server_bin = env!("CARGO_BIN_EXE_append_only_log_demo_server");
    let client_bin = env!("CARGO_BIN_EXE_append_only_log_demo_client");
    let _cluster = start_servers(server_bin, &registry_path, &addrs);

    let emit = Command::new(server_bin)
        .env("NET_RPC_TIMEOUT_MS", "60000")
        .args([
            "--op",
            "emit-mock",
            "--anchor",
            path_str(&anchor_path),
            "--height",
            "0",
        ])
        .output()
        .expect("run emit-mock");
    assert_success("emit-mock", &emit);

    let client = |args: &[&str]| {
        Command::new(client_bin)
            .env("NET_RPC_TIMEOUT_MS", "60000")
            .args(["--anchor", path_str(&anchor_path)])
            .args(args)
            .output()
            .expect("run client binary")
    };
    let read_leaf = |extra: &[&str]| -> Value {
        let mut args = vec!["--op", "read-leaf", "--index", "5"];
        args.extend_from_slice(extra);
        let stdout = assert_success("read-leaf", &client(&args));
        serde_json::from_str(&stdout).expect("parse read-leaf json")
    };
    let read_range = |extra: &[&str]| -> String {
        let mut args = vec![
            "--op",
            "read-range",
            "--height",
            "0",
            "--offset",
            "100",
            "--len",
            "500",
        ];
        args.extend_from_slice(extra);
        assert_success("read-range", &client(&args))
    };

    let stored = read_leaf(&[]);
    assert_eq!(stored["open_kappa"].as_u64(), Some(KAPPA as u64));
    let stored_range = read_range(&[]);
    for open_kappa in [2 * KAPPA, 4 * KAPPA] {
        let open_kappa = open_kappa.to_string();
        let opened = read_leaf(&["--open-kappa", &open_kappa]);
        assert_eq!(opened["open_kappa"].to_string(), open_kappa);
        assert_eq!(opened["value_hex"], stored["value_hex"]);
        assert_eq!(opened["siblings_hex"], stored["siblings_hex"]);
        assert_eq!(opened["proof_root_matches_trusted"].as_bool(), Some(true));
        assert_eq!(read_range(&["--open-kappa", &open_kappa]), stored_range);
    }

    // The registry has `4 * KAPPA` claims.
    let too_large = (8 * KAPPA).to_string();
    let rejected = client(&[
        "--op",
        "read-leaf",
        "--index",
        "5",
        "--open-kappa",
        &too_large,
    ]);
    assert!(!rejected.status.success());
    assert!(
        String::from_utf8_lossy(&rejected.stderr).contains("claims of the registry"),
        "stderr did not explain the rejected threshold"
    );
    let rejected = client(&["--op", "tip", "--open-kappa", "4"]);
    assert!(!rejected.status.success());
    assert!(
        String::from_utf8_lossy(&rejected.stderr)
            .contains("--open-kappa applies to read-leaf, read-bulk-updates, and read-range")
    );
}

fn build_registry(addrs: &[String]) -> MockRegistryFile {
    let shard_count = addrs.len();
    let shard_registry = build_shard_registry_deterministic(shard_count);
    let party_registry_indices = (0..shard_count).map(|idx| vec![idx]).collect::<Vec<_>>();
    let cfg = CodedArrayConfig {
        kappa: KAPPA,
        shard_count,
        shard_registry,
        party_registry_indices,
        seed: SEED,
    };
    let params = build_coded_params(&cfg);
    assert_eq!(params.need, KAPPA);

    let parties = addrs
        .iter()
        .enumerate()
        .map(|(party_id, addr)| PartyInfo {
            party_id: party_id as u64,
            rpc_host: addr.clone(),
            pubkey: None,
            active: true,
        })
        .collect::<Vec<_>>();
    let assignments = (0..shard_count)
        .map(|idx| PartyAssignments {
            party_id: idx as u64,
            indices: vec![idx as u32],
        })
        .collect::<Vec<_>>();
    let unauth_claims =
        default_unauth_claims(&params, &assignments).expect("default unauth claims");

    MockRegistryFile {
        system: NetSystemConfig {
            unauth: NetUnauthConfig {
                kappa: KAPPA,
                layout_base_kappa: None,
                shard_count,
                seed: SEED,
            },
            auth: NetAuthConfig {
                tree_height: TREE_HEIGHT,
                node_linearization: NodeLinearization::Preorder,
                allocated_leaves: 0,
            },
            unauth_array_len: Some(0),
        },
        parties,
        assignments,
        unauth_epoch: Some(default_unauth_epoch(&params)),
        unauth_claims,
        shard_registry: None,
    }
}

fn write_anchor(anchor_path: &Path, registry_path: &Path) {
    let anchor = AppendOnlyLogAnchorFile {
        source: AppendOnlyLogRegistrySource::LocalFile {
            registry_path: registry_path.to_path_buf(),
        },
        committed_leaves: 0,
        trusted_root_hex: None,
        frontier_leaf: None,
    };
    anchor.save(anchor_path).expect("write anchor");
}

fn start_servers(server_bin: &str, registry_path: &Path, addrs: &[String]) -> ServerCluster {
    let mut children = Vec::with_capacity(addrs.len());
    for (party_id, addr) in addrs.iter().enumerate() {
        let child = Command::new(server_bin)
            .args([
                "--op",
                "serve",
                "--registry",
                path_str(registry_path),
                "--party-id",
                &party_id.to_string(),
                "--listen",
                addr,
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn appendOnlyLog server");
        children.push(child);
    }

    for addr in addrs {
        wait_for_server(addr);
    }

    ServerCluster { children }
}

fn install_corrupting_proxies(
    registry: &mut MockRegistryFile,
    party_ids: &[usize],
    real_addrs: &[String],
) -> Vec<ProxyGuard> {
    let mut proxies = Vec::with_capacity(party_ids.len());
    for &party_id in party_ids {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind proxy listener");
        let proxy_addr = listener.local_addr().expect("proxy local addr").to_string();
        let real_addr = real_addrs[party_id].clone();
        let corruptions = Arc::new(AtomicUsize::new(0));
        let proxy_corruptions = Arc::clone(&corruptions);
        let handle =
            net::spawn_with_listener(listener, move |msg| match net::rpc_call(&real_addr, &msg) {
                Ok(mut resp) => {
                    let changed = corrupt_auth_read_response(&mut resp);
                    if changed != 0 {
                        proxy_corruptions.fetch_add(changed, Ordering::SeqCst);
                    }
                    resp
                }
                Err(err) => Response::Error {
                    message: format!("{err:?}"),
                },
            });

        registry.parties[party_id].rpc_host = proxy_addr;
        proxies.push(ProxyGuard {
            handle: Some(handle),
            corruptions,
        });
    }
    proxies
}

fn corrupt_auth_read_response(resp: &mut Response) -> usize {
    let Response::AuthRead { resp } = resp else {
        return 0;
    };

    let mut changed = 0;
    for node in &mut resp.nodes {
        if let Some(value) = node.values.0.iter_mut().find(|value| value.is_some()) {
            let Some(current) = *value else {
                unreachable!("checked Some")
            };
            *value = Some(current + paper_a_optimized::fp_one());
            changed += 1;
        }
    }
    changed
}

fn wait_for_server(addr: &str) {
    let mut last_err = None;
    for _ in 0..200 {
        match net::rpc_call(addr, &Message::AuthGetLen) {
            Ok(Response::AuthLen { .. }) => return,
            Ok(other) => panic!("unexpected server response from {addr}: {other:?}"),
            Err(err) => {
                last_err = Some(err);
                thread::sleep(Duration::from_millis(25));
            }
        }
    }
    panic!("server {addr} did not become ready: {last_err:?}");
}

fn reserve_loopback_addrs(count: usize) -> Vec<String> {
    (0..count)
        .map(|_| {
            let listener = TcpListener::bind("127.0.0.1:0").expect("reserve loopback port");
            listener.local_addr().expect("reserved addr").to_string()
        })
        .collect()
}

fn write_json(path: &Path, value: &impl serde::Serialize) {
    let data = serde_json::to_vec_pretty(value).expect("serialize json");
    fs::write(path, data).expect("write json");
}

fn assert_success(context: &str, output: &Output) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "{context} failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout,
        stderr
    );
    stdout.into_owned()
}

fn path_str(path: &Path) -> &str {
    path.to_str().expect("path is valid utf-8")
}
