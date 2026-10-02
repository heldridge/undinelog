use paper_a_optimized::core::NodeLinearization;
use paper_a_optimized::net::registry::{
    MockRegistryFile, NetAuthConfig, NetSystemConfig, NetUnauthConfig, PartyAssignments, PartyInfo,
    default_unauth_claims, default_unauth_epoch,
};
use paper_a_optimized::net::{self, Message, Response};
use paper_a_optimized::{CodedArrayConfig, build_coded_params, build_shard_registry_deterministic};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::Duration;

#[test]
fn client_cli_accepts_max_errors_flag_before_loading_registry() {
    let output = Command::new(env!("CARGO_BIN_EXE_client"))
        .args([
            "--mode",
            "unauth",
            "--op",
            "repair",
            "--registry",
            "/definitely/missing/registry.json",
            "--segment",
            "0",
            "--registry-index",
            "0",
            "--max-errors",
            "0",
        ])
        .output()
        .expect("run client binary");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("failed to load registry"),
        "stderr did not show registry loading after parsing --max-errors:\n{stderr}"
    );
}

#[test]
fn client_cli_rejects_invalid_max_errors() {
    let output = Command::new(env!("CARGO_BIN_EXE_client"))
        .args(["--mode", "unauth", "--op", "repair", "--max-errors", "nope"])
        .output()
        .expect("run client binary");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("invalid --max-errors"),
        "stderr did not reject invalid --max-errors:\n{stderr}"
    );
}

#[test]
fn client_cli_rejects_invalid_open_kappa() {
    let output = Command::new(env!("CARGO_BIN_EXE_client"))
        .args(["--mode", "auth", "--op", "read", "--open-kappa", "nope"])
        .output()
        .expect("run client binary");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("invalid --open-kappa"),
        "stderr did not reject invalid --open-kappa:\n{stderr}"
    );
}

#[test]
fn client_cli_limits_open_kappa_to_auth_openings() {
    for args in [
        ["--mode", "unauth", "--op", "read", "--open-kappa", "8"],
        ["--mode", "auth", "--op", "multi-read", "--open-kappa", "8"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_client"))
            .args(args)
            .args(["--registry", "/definitely/missing/registry.json"])
            .output()
            .expect("run client binary");
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("--open-kappa applies to auth-mode read and read-range"),
            "stderr did not reject --open-kappa for {args:?}:\n{stderr}"
        );
    }
}

/// Auth servers spawned from the server binary, killed on drop.
struct ServerCluster {
    children: Vec<Child>,
    dir: PathBuf,
}

impl Drop for ServerCluster {
    fn drop(&mut self) {
        for child in &mut self.children {
            let _ = child.kill();
            let _ = child.wait();
        }
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Registry with one claim per party at threshold `kappa`.
fn one_claim_registry(kappa: usize, addrs: &[String], allocated_leaves: usize) -> MockRegistryFile {
    let shard_count = addrs.len();
    let seed = 29;
    let params = build_coded_params(&CodedArrayConfig {
        kappa,
        shard_count,
        shard_registry: build_shard_registry_deterministic(shard_count),
        party_registry_indices: (0..shard_count).map(|idx| vec![idx]).collect(),
        seed,
    });
    let assignments = (0..shard_count)
        .map(|idx| PartyAssignments {
            party_id: idx as u64,
            indices: vec![idx as u32],
        })
        .collect::<Vec<_>>();
    let unauth_claims = default_unauth_claims(&params, &assignments).expect("unauth claims");
    MockRegistryFile {
        system: NetSystemConfig {
            unauth: NetUnauthConfig {
                kappa,
                layout_base_kappa: None,
                shard_count,
                seed,
            },
            auth: NetAuthConfig {
                tree_height: 6,
                node_linearization: NodeLinearization::Preorder,
                allocated_leaves,
            },
            unauth_array_len: Some(0),
        },
        parties: addrs
            .iter()
            .enumerate()
            .map(|(party_id, addr)| PartyInfo {
                party_id: party_id as u64,
                rpc_host: addr.clone(),
                pubkey: None,
                active: true,
            })
            .collect(),
        assignments,
        unauth_epoch: Some(default_unauth_epoch(&params)),
        unauth_claims,
        shard_registry: None,
    }
}

fn start_auth_servers(registry_path: &Path, addrs: &[String], dir: PathBuf) -> ServerCluster {
    let children = addrs
        .iter()
        .enumerate()
        .map(|(party_id, addr)| {
            Command::new(env!("CARGO_BIN_EXE_server"))
                .args(["--mode", "auth", "--registry"])
                .arg(registry_path)
                .args(["--party-id", &party_id.to_string(), "--listen", addr])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("spawn server")
        })
        .collect();
    let cluster = ServerCluster { children, dir };
    for addr in addrs {
        let ready = (0..200).any(|_| {
            let up = matches!(
                net::rpc_call(addr, &Message::AuthGetLen),
                Ok(Response::AuthLen { .. })
            );
            if !up {
                thread::sleep(Duration::from_millis(25));
            }
            up
        });
        assert!(ready, "server {addr} did not become ready");
    }
    cluster
}

/// Run the client binary in auth mode and return its standard output lines.
fn auth_client(registry_path: &Path, args: &[&str]) -> Vec<String> {
    let output = Command::new(env!("CARGO_BIN_EXE_client"))
        .env("NET_RPC_TIMEOUT_MS", "60000")
        .args(["--mode", "auth", "--registry"])
        .arg(registry_path)
        .args(args)
        .output()
        .expect("run client binary");
    assert!(
        output.status.success(),
        "client {args:?} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::to_string)
        .collect()
}

#[test]
fn client_cli_opens_at_a_higher_threshold_from_servers_at_the_registry_threshold() {
    let dir = std::env::temp_dir().join(format!("client_cli_open_kappa_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let registry_path = dir.join("registry.json");
    let leaves = 6usize;
    let addrs = (0..8)
        .map(|_| {
            let listener = TcpListener::bind("127.0.0.1:0").expect("reserve loopback port");
            listener.local_addr().expect("reserved addr").to_string()
        })
        .collect::<Vec<_>>();
    let registry = one_claim_registry(2, &addrs, leaves);
    std::fs::write(
        &registry_path,
        serde_json::to_vec_pretty(&registry).expect("serialize registry"),
    )
    .expect("write registry");
    let _cluster = start_auth_servers(&registry_path, &addrs, dir.clone());

    let values = (0..leaves)
        .map(|idx| format!("0x{:02x}{}", idx + 1, "00".repeat(31)))
        .collect::<Vec<_>>();
    for (idx, value) in values.iter().enumerate() {
        auth_client(
            &registry_path,
            &[
                "--op",
                "write",
                "--index",
                &idx.to_string(),
                "--value",
                value,
            ],
        );
    }

    for open_kappa in ["4", "8"] {
        for (idx, value) in values.iter().enumerate() {
            let read = auth_client(
                &registry_path,
                &[
                    "--op",
                    "read",
                    "--index",
                    &idx.to_string(),
                    "--open-kappa",
                    open_kappa,
                ],
            );
            assert_eq!(read, [value.clone()], "leaf {idx} at kappa = {open_kappa}");
        }
        let range = auth_client(
            &registry_path,
            &[
                "--op",
                "read-range",
                "--index",
                "1",
                "--count",
                "4",
                "--open-kappa",
                open_kappa,
            ],
        );
        assert_eq!(range, values[1..5]);
    }
    // The registry has eight claims: a threshold of sixteen cannot be opened.
    let output = Command::new(env!("CARGO_BIN_EXE_client"))
        .args(["--mode", "auth", "--registry"])
        .arg(&registry_path)
        .args(["--op", "read", "--index", "0", "--open-kappa", "16"])
        .output()
        .expect("run client binary");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("threshold 16 exceeds the 8 claims of the registry"),
        "stderr did not explain the rejected threshold:\n{stderr}"
    );
}
