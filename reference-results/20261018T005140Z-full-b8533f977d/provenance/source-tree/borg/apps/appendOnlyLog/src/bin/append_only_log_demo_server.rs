use append_only_log_demo::codec::encode_bulk_updates;
use append_only_log_demo::emitter::AppendOnlyLogEmitter;
use append_only_log_demo::layout::committed_bulk_update_batches;
use append_only_log_demo::mock::MockAppendOnlyLogBatch;
use append_only_log_demo::registry::format_hex_32;
use append_only_log_demo::server::serve_auth_from_registry_file;
use serde_json::json;
use std::env;
use std::fs;
use std::path::PathBuf;

#[derive(Default)]
struct Args {
    op: Option<String>,
    anchor: Option<PathBuf>,
    registry: Option<PathBuf>,
    recovered_state: Option<PathBuf>,
    listen: Option<String>,
    party_id: Option<u64>,
    height: Option<u64>,
    file: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--op" => args.op = it.next(),
            "--anchor" => args.anchor = it.next().map(PathBuf::from),
            "--registry" => args.registry = it.next().map(PathBuf::from),
            "--recovered-state" => args.recovered_state = it.next().map(PathBuf::from),
            "--listen" => args.listen = it.next(),
            "--party-id" => {
                let value = it.next().ok_or("--party-id missing value")?;
                args.party_id = Some(value.parse().map_err(|_| "invalid --party-id")?);
            }
            "--height" => {
                let value = it.next().ok_or("--height missing value")?;
                args.height = Some(value.parse().map_err(|_| "invalid --height")?);
            }
            "--file" => args.file = it.next().map(PathBuf::from),
            _ => return Err(format!("unknown arg {arg}")),
        }
    }
    Ok(args)
}

fn usage() -> &'static str {
    "ops:
  serve --registry <path> [--listen <host:port>] [--party-id <id>] [--recovered-state <path>]
  emit-mock --anchor <path> [--height <n>]
  emit-file --anchor <path> [--height <n>] --file <path>"
}

fn main() {
    let args = parse_args().unwrap_or_else(|err| {
        eprintln!("{err}\n\n{}", usage());
        std::process::exit(1);
    });
    let op = args.op.clone().unwrap_or_else(|| "serve".to_string());
    match op.as_str() {
        "serve" => serve(&args),
        "emit-mock" => emit_mock(&args),
        "emit-file" => emit_file(&args),
        other => {
            eprintln!("unknown op {other}\n\n{}", usage());
            std::process::exit(1);
        }
    }
}

fn serve(args: &Args) {
    let registry_path = args.registry.clone().unwrap_or_else(|| {
        eprintln!("--registry is required for serve\n\n{}", usage());
        std::process::exit(1);
    });
    serve_auth_from_registry_file(
        &registry_path,
        args.listen.as_deref(),
        args.party_id,
        args.recovered_state.as_deref(),
    )
    .unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(1);
    });
}

fn emit_mock(args: &Args) {
    let anchor_path = args.anchor.clone().unwrap_or_else(|| {
        eprintln!("--anchor is required for emit-mock");
        std::process::exit(1);
    });
    let mut emitter = AppendOnlyLogEmitter::from_anchor_file(&anchor_path).unwrap_or_else(|err| {
        eprintln!("server emitter init failed: {err}");
        std::process::exit(1);
    });
    let batch = MockAppendOnlyLogBatch::load_fixture().unwrap_or_else(|err| {
        eprintln!("failed to load mock fixture: {err}");
        std::process::exit(1);
    });
    let bulk_updates_bytes = batch.bulk_updates_bytes().unwrap_or_else(|err| {
        eprintln!("failed to encode mock batch bulk_updates: {err}");
        std::process::exit(1);
    });
    let height = args
        .height
        .unwrap_or_else(|| committed_bulk_update_batches(emitter.anchor().committed_leaves));
    let root = emitter
        .emit_bulk_updates_bytes_at_height(height, &bulk_updates_bytes)
        .unwrap_or_else(|err| {
            eprintln!("emit-mock failed: {err}");
            std::process::exit(1);
        });
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "emitter": "server",
            "height": height,
            "payload_bytes": batch.batch_bytes().len(),
            "batch_bytes": batch.batch_bytes().len(),
            "bulk_updates_bytes": bulk_updates_bytes.len(),
            "committed_leaves": emitter.anchor().committed_leaves,
            "trusted_root": format_hex_32(root),
        }))
        .unwrap()
    );
}

fn emit_file(args: &Args) {
    let anchor_path = args.anchor.clone().unwrap_or_else(|| {
        eprintln!("--anchor is required for emit-file");
        std::process::exit(1);
    });
    let file_path = args.file.clone().unwrap_or_else(|| {
        eprintln!("--file is required for emit-file");
        std::process::exit(1);
    });
    let batch_bytes = fs::read(&file_path).unwrap_or_else(|err| {
        eprintln!("failed to read batch file: {err}");
        std::process::exit(1);
    });
    let bulk_updates_bytes = encode_bulk_updates(&batch_bytes).unwrap_or_else(|err| {
        eprintln!("failed to encode batch bulk_updates: {err}");
        std::process::exit(1);
    });
    let mut emitter = AppendOnlyLogEmitter::from_anchor_file(&anchor_path).unwrap_or_else(|err| {
        eprintln!("server emitter init failed: {err}");
        std::process::exit(1);
    });
    let height = args
        .height
        .unwrap_or_else(|| committed_bulk_update_batches(emitter.anchor().committed_leaves));
    let root = emitter
        .emit_bulk_updates_bytes_at_height(height, &bulk_updates_bytes)
        .unwrap_or_else(|err| {
            eprintln!("emit-file failed: {err}");
            std::process::exit(1);
        });
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "emitter": "server",
            "height": height,
            "payload_bytes": batch_bytes.len(),
            "batch_bytes": batch_bytes.len(),
            "bulk_updates_bytes": bulk_updates_bytes.len(),
            "committed_leaves": emitter.anchor().committed_leaves,
            "trusted_root": format_hex_32(root),
        }))
        .unwrap()
    );
}
