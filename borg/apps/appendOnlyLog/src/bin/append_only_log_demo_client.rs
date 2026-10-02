use append_only_log_demo::client::AppendOnlyLogClient;
use append_only_log_demo::layout::tip_height;
use append_only_log_demo::mock::MockAppendOnlyLogBatch;
use append_only_log_demo::registry::{
    AppendOnlyLogAnchorFile, AppendOnlyLogRegistrySource, format_hex_32, parse_hex_32,
};
use serde_json::json;
use std::env;
use std::fs;
use std::path::PathBuf;

#[derive(Default)]
struct Args {
    anchor: Option<PathBuf>,
    op: Option<String>,
    registry: Option<PathBuf>,
    height: Option<u64>,
    index: Option<u64>,
    offset: Option<usize>,
    len: Option<usize>,
    file: Option<PathBuf>,
    out: Option<PathBuf>,
    committed_leaves: Option<u64>,
    trusted_root: Option<[u8; 32]>,
    max_errors: Option<usize>,
    source_kind: Option<String>,
    network: Option<String>,
    start_height: Option<u64>,
    tx_marker: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--anchor" => args.anchor = it.next().map(PathBuf::from),
            "--op" => args.op = it.next(),
            "--registry" => args.registry = it.next().map(PathBuf::from),
            "--height" => {
                let value = it.next().ok_or("--height missing value")?;
                args.height = Some(value.parse().map_err(|_| "invalid --height")?);
            }
            "--index" => {
                let value = it.next().ok_or("--index missing value")?;
                args.index = Some(value.parse().map_err(|_| "invalid --index")?);
            }
            "--offset" => {
                let value = it.next().ok_or("--offset missing value")?;
                args.offset = Some(value.parse().map_err(|_| "invalid --offset")?);
            }
            "--len" => {
                let value = it.next().ok_or("--len missing value")?;
                args.len = Some(value.parse().map_err(|_| "invalid --len")?);
            }
            "--file" => args.file = it.next().map(PathBuf::from),
            "--out" => args.out = it.next().map(PathBuf::from),
            "--committed-leaves" => {
                let value = it.next().ok_or("--committed-leaves missing value")?;
                args.committed_leaves =
                    Some(value.parse().map_err(|_| "invalid --committed-leaves")?);
            }
            "--trusted-root" => {
                let value = it.next().ok_or("--trusted-root missing value")?;
                args.trusted_root = Some(parse_hex_32(&value).map_err(|err| err.to_string())?);
            }
            "--max-errors" => {
                let value = it.next().ok_or("--max-errors missing value")?;
                args.max_errors = Some(value.parse().map_err(|_| "invalid --max-errors")?);
            }
            "--source-kind" => args.source_kind = it.next(),
            "--network" => args.network = it.next(),
            "--start-height" => {
                let value = it.next().ok_or("--start-height missing value")?;
                args.start_height = Some(value.parse().map_err(|_| "invalid --start-height")?);
            }
            "--tx-marker" => args.tx_marker = it.next(),
            _ => return Err(format!("unknown arg {arg}")),
        }
    }
    Ok(args)
}

fn usage() -> &'static str {
    "ops:
  init-anchor --anchor <path> --registry <path> [--source-kind local_file|chain_transactions]
              [--trusted-root <hex32>] [--committed-leaves <n>] [--network <name>]
              [--start-height <n>] [--tx-marker <string>]
  show-anchor --anchor <path>
  state --anchor <path>
  tip --anchor <path>
  read-leaf --anchor <path> --index <n> [--max-errors <n>]
  read-bulk-updates --anchor <path> --height <n> [--out <path>]
  read-range --anchor <path> --height <n> --offset <n> --len <n> [--out <path>]
  dump-mock [--out <path>]"
}

fn main() {
    let args = parse_args().unwrap_or_else(|err| {
        eprintln!("{err}\n\n{}", usage());
        std::process::exit(1);
    });

    let op = args.op.clone().unwrap_or_else(|| {
        eprintln!("--op is required\n\n{}", usage());
        std::process::exit(1);
    });

    match op.as_str() {
        "init-anchor" => init_anchor(&args),
        "show-anchor" => show_anchor(&args),
        "state" => show_state(&args),
        "tip" => show_tip(&args),
        "read-leaf" => read_leaf(&args),
        "read-bulk-updates" => read_bulk_updates(&args),
        "read-range" => read_range(&args),
        "dump-mock" => dump_mock(&args),
        other => {
            eprintln!("unknown op {other}\n\n{}", usage());
            std::process::exit(1);
        }
    }
}

fn init_anchor(args: &Args) {
    let anchor_path = args.anchor.clone().unwrap_or_else(|| {
        eprintln!("--anchor is required for init-anchor");
        std::process::exit(1);
    });
    let registry_path = args.registry.clone().unwrap_or_else(|| {
        eprintln!("--registry is required for init-anchor");
        std::process::exit(1);
    });
    let source = match args.source_kind.as_deref() {
        None | Some("local_file") => AppendOnlyLogRegistrySource::LocalFile { registry_path },
        Some("chain_transactions") => AppendOnlyLogRegistrySource::ChainTransactions {
            materialized_registry_path: registry_path,
            network: args.network.clone().unwrap_or_else(|| "chain".to_string()),
            start_height: args.start_height,
            tx_marker: args.tx_marker.clone(),
        },
        Some(other) => {
            eprintln!("invalid --source-kind {other}");
            std::process::exit(1);
        }
    };
    let mut anchor = AppendOnlyLogAnchorFile {
        source,
        committed_leaves: args.committed_leaves.unwrap_or(0),
        trusted_root_hex: None,
        frontier_leaf: None,
    };
    if let Some(root) = args.trusted_root {
        anchor.set_trusted_root(root);
    }
    anchor.save(&anchor_path).unwrap_or_else(|err| {
        eprintln!("failed to write anchor: {err}");
        std::process::exit(1);
    });
    println!("{}", serde_json::to_string_pretty(&anchor).unwrap());
}

fn show_anchor(args: &Args) {
    let anchor_path = args.anchor.clone().unwrap_or_else(|| {
        eprintln!("--anchor is required for show-anchor");
        std::process::exit(1);
    });
    let anchor = AppendOnlyLogAnchorFile::load(&anchor_path).unwrap_or_else(|err| {
        eprintln!("failed to load anchor: {err}");
        std::process::exit(1);
    });
    println!("{}", serde_json::to_string_pretty(&anchor).unwrap());
}

fn show_state(args: &Args) {
    let anchor_path = args.anchor.clone().unwrap_or_else(|| {
        eprintln!("--anchor is required for state");
        std::process::exit(1);
    });
    let mut client = AppendOnlyLogClient::from_anchor_file(&anchor_path).unwrap_or_else(|err| {
        eprintln!("client init failed: {err}");
        std::process::exit(1);
    });
    let anchor = client.anchor().clone();
    let registry = anchor.load_registry(&anchor_path).unwrap_or_else(|err| {
        eprintln!("failed to load materialized registry: {err}");
        std::process::exit(1);
    });
    let server_len = client.server_current_len().unwrap_or_else(|err| {
        eprintln!("failed to query server length: {err}");
        std::process::exit(1);
    });
    let active_servers = registry.parties.iter().filter(|party| party.active).count();
    let claim_count = registry.unauth_claims.len();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "source": anchor.source,
            "trusted_root": anchor.trusted_root_hex,
            "anchor_committed_leaves": anchor.committed_leaves,
            "anchor_tip_height": anchor.tip_height(),
            "server_current_len": server_len,
            "server_tip_height": tip_height(server_len),
            "party_count": registry.parties.len(),
            "active_server_count": active_servers,
            "unauth_segment_len": registry.system.unauth.segment_len,
            "unauth_s": registry.system.unauth.s,
            "unauth_shard_count": registry.system.unauth.shard_count,
            "unauth_claim_count": claim_count,
        }))
        .unwrap()
    );
}

fn show_tip(args: &Args) {
    let anchor_path = args.anchor.clone().unwrap_or_else(|| {
        eprintln!("--anchor is required for tip");
        std::process::exit(1);
    });
    let client = AppendOnlyLogClient::from_anchor_file(&anchor_path).unwrap_or_else(|err| {
        eprintln!("client init failed: {err}");
        std::process::exit(1);
    });
    match client.tip_height() {
        Some(height) => println!("{height}"),
        None => println!("none"),
    }
}

fn read_leaf(args: &Args) {
    let anchor_path = args.anchor.clone().unwrap_or_else(|| {
        eprintln!("--anchor is required for read-leaf");
        std::process::exit(1);
    });
    let index = args.index.unwrap_or_else(|| {
        eprintln!("--index is required for read-leaf");
        std::process::exit(1);
    });
    let mut client = AppendOnlyLogClient::from_anchor_file(&anchor_path).unwrap_or_else(|err| {
        eprintln!("client init failed: {err}");
        std::process::exit(1);
    });
    let trusted_root = client
        .trusted_root()
        .unwrap_or_else(|err| {
            eprintln!("failed to load trusted root: {err}");
            std::process::exit(1);
        })
        .map(format_hex_32);
    let max_errors = args.max_errors.unwrap_or(0);
    let correcting = max_errors > 0;
    let read_result = if correcting {
        client.read_leaf_with_path_correcting(index, max_errors)
    } else {
        client.read_leaf_with_path(index)
    };
    let (value, path) = read_result.unwrap_or_else(|err| {
        eprintln!("read-leaf failed: {err}");
        std::process::exit(1);
    });
    let proof_root = format_hex_32(path.root);
    let proof_root_matches_trusted = trusted_root.as_deref() == Some(proof_root.as_str());
    let siblings_hex = path
        .siblings
        .iter()
        .copied()
        .map(format_hex_32)
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "index": index,
            "value_hex": format_hex_32(value),
            "max_errors": max_errors,
            "correcting": correcting,
            "proof_root": proof_root,
            "trusted_root": trusted_root,
            "proof_root_matches_trusted": proof_root_matches_trusted,
            "path_index": path.index,
            "sibling_count": path.siblings.len(),
            "siblings_hex": siblings_hex,
        }))
        .unwrap()
    );
}

fn read_bulk_updates(args: &Args) {
    let anchor_path = args.anchor.clone().unwrap_or_else(|| {
        eprintln!("--anchor is required for read-bulk-updates");
        std::process::exit(1);
    });
    let height = args.height.unwrap_or_else(|| {
        eprintln!("--height is required for read-bulk-updates");
        std::process::exit(1);
    });
    let mut client = AppendOnlyLogClient::from_anchor_file(&anchor_path).unwrap_or_else(|err| {
        eprintln!("client init failed: {err}");
        std::process::exit(1);
    });
    let bulk_updates = client.read_bulk_updates(height).unwrap_or_else(|err| {
        eprintln!("read-bulk-updates failed: {err}");
        std::process::exit(1);
    });
    if let Some(out_path) = args.out.as_ref() {
        fs::write(out_path, &bulk_updates).unwrap_or_else(|err| {
            eprintln!("failed to write bulk updates bytes: {err}");
            std::process::exit(1);
        });
    }
    let header_hex = hex_bytes(&bulk_updates[..80]);
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "height": height,
            "bulk_updates_bytes": bulk_updates.len(),
            "header_hex": header_hex,
            "out": args.out.as_ref().map(|path| path.display().to_string()),
        }))
        .unwrap()
    );
}

fn read_range(args: &Args) {
    let anchor_path = args.anchor.clone().unwrap_or_else(|| {
        eprintln!("--anchor is required for read-range");
        std::process::exit(1);
    });
    let height = args.height.unwrap_or_else(|| {
        eprintln!("--height is required for read-range");
        std::process::exit(1);
    });
    let offset = args.offset.unwrap_or_else(|| {
        eprintln!("--offset is required for read-range");
        std::process::exit(1);
    });
    let len = args.len.unwrap_or_else(|| {
        eprintln!("--len is required for read-range");
        std::process::exit(1);
    });
    let mut client = AppendOnlyLogClient::from_anchor_file(&anchor_path).unwrap_or_else(|err| {
        eprintln!("client init failed: {err}");
        std::process::exit(1);
    });
    let bytes = client
        .read_bulk_updates_range(height, offset, len)
        .unwrap_or_else(|err| {
            eprintln!("read-range failed: {err}");
            std::process::exit(1);
        });
    if let Some(out_path) = args.out.as_ref() {
        fs::write(out_path, &bytes).unwrap_or_else(|err| {
            eprintln!("failed to write range bytes: {err}");
            std::process::exit(1);
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "height": height,
                "offset": offset,
                "len": len,
                "out": out_path.display().to_string(),
            }))
            .unwrap()
        );
    } else {
        println!("{}", hex_bytes(&bytes));
    }
}

fn dump_mock(args: &Args) {
    let batch = MockAppendOnlyLogBatch::load_fixture().unwrap_or_else(|err| {
        eprintln!("failed to load mock fixture: {err}");
        std::process::exit(1);
    });
    let batch_bytes = batch.batch_bytes();
    if let Some(out_path) = args.out.as_ref() {
        fs::write(out_path, &batch_bytes).unwrap_or_else(|err| {
            eprintln!("failed to write mock batch: {err}");
            std::process::exit(1);
        });
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "height": batch.height,
            "payload_bytes": batch_bytes.len(),
            "batch_bytes": batch_bytes.len(),
            "out": args.out.as_ref().map(|path| path.display().to_string()),
        }))
        .unwrap()
    );
}

fn hex_bytes(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(2 * bytes.len() + 2);
    out.push_str("0x");
    for byte in bytes {
        out.push(nibble_to_hex(byte >> 4));
        out.push(nibble_to_hex(byte & 0x0f));
    }
    out
}

fn nibble_to_hex(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        10..=15 => (b'a' + (nibble - 10)) as char,
        _ => '?',
    }
}
