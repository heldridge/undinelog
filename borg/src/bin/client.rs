use paper_a_optimized::auth::{AuthPath, NodeLinearization};
use paper_a_optimized::decode_value;
use paper_a_optimized::net::auth::AuthClient;
use paper_a_optimized::net::registry::{
    MockRegistryFile, build_unauth_registry_view, claim_bundles_for_party,
    kappa_double_unauth_registry_file, kappa_halve_unauth_registry_file,
    migrate_unauth_registry_file, migrate_unauth_registry_file_one_claim_left,
    onboard_unauth_party_with_new_shards, segment_double_unauth_registry_file,
};
use paper_a_optimized::net::unauth::UnauthClient;
use paper_a_optimized::unauth::{DeltaMode, public_values_per_segment};
use serde::Deserialize;
use serde_json::json;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Default)]
struct Args {
    mode: Option<String>,
    registry: Option<PathBuf>,
    op: Option<String>,
    index: Option<u64>,
    party_id: Option<u64>,
    count: Option<usize>,
    segment: Option<usize>,
    registry_index: Option<usize>,
    row_label_index: Option<usize>,
    value: Option<String>,
    batch_file: Option<PathBuf>,
    new_len: Option<u64>,
    target_s: Option<usize>,
    delta_mode: Option<String>,
    segment_len: Option<usize>,
    node_linearization: Option<String>,
    s: Option<usize>,
    shard_count: Option<usize>,
    tree_height: Option<usize>,
    unauth_array_len: Option<u64>,
    allocated_leaves: Option<usize>,
    trusted_root: Option<[u8; 32]>,
    max_errors: usize,
    verbose: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--mode" => args.mode = it.next(),
            "--registry" => args.registry = it.next().map(PathBuf::from),
            "--op" => args.op = it.next(),
            "--index" => {
                let v = it.next().ok_or("--index missing value")?;
                args.index = Some(v.parse().map_err(|_| "invalid --index")?);
            }
            "--party-id" => {
                let v = it.next().ok_or("--party-id missing value")?;
                args.party_id = Some(v.parse().map_err(|_| "invalid --party-id")?);
            }
            "--count" => {
                let v = it.next().ok_or("--count missing value")?;
                args.count = Some(v.parse().map_err(|_| "invalid --count")?);
            }
            "--segment" => {
                let v = it.next().ok_or("--segment missing value")?;
                args.segment = Some(v.parse().map_err(|_| "invalid --segment")?);
            }
            "--registry-index" => {
                let v = it.next().ok_or("--registry-index missing value")?;
                args.registry_index = Some(v.parse().map_err(|_| "invalid --registry-index")?);
            }
            "--row-label-index" => {
                let v = it.next().ok_or("--row-label-index missing value")?;
                args.row_label_index = Some(v.parse().map_err(|_| "invalid --row-label-index")?);
            }
            "--value" => args.value = it.next(),
            "--batch-file" => args.batch_file = it.next().map(PathBuf::from),
            "--new-len" => {
                let v = it.next().ok_or("--new-len missing value")?;
                args.new_len = Some(v.parse().map_err(|_| "invalid --new-len")?);
            }
            "--target-s" => {
                let v = it.next().ok_or("--target-s missing value")?;
                args.target_s = Some(v.parse().map_err(|_| "invalid --target-s")?);
            }
            "--delta-mode" => args.delta_mode = it.next(),
            "--segment-len" => {
                let v = it.next().ok_or("--segment-len missing value")?;
                args.segment_len = Some(v.parse().map_err(|_| "invalid --segment-len")?);
            }
            "--node-linearization" => {
                args.node_linearization = it.next();
            }
            "--s" => {
                let v = it.next().ok_or("--s missing value")?;
                args.s = Some(v.parse().map_err(|_| "invalid --s")?);
            }
            "--shard-count" => {
                let v = it.next().ok_or("--shard-count missing value")?;
                args.shard_count = Some(v.parse().map_err(|_| "invalid --shard-count")?);
            }
            "--tree-height" => {
                let v = it.next().ok_or("--tree-height missing value")?;
                args.tree_height = Some(v.parse().map_err(|_| "invalid --tree-height")?);
            }
            "--unauth-array-len" => {
                let v = it.next().ok_or("--unauth-array-len missing value")?;
                args.unauth_array_len = Some(v.parse().map_err(|_| "invalid --unauth-array-len")?);
            }
            "--allocated-leaves" => {
                let v = it.next().ok_or("--allocated-leaves missing value")?;
                args.allocated_leaves = Some(v.parse().map_err(|_| "invalid --allocated-leaves")?);
            }
            "--trusted-root" => {
                let v = it.next().ok_or("--trusted-root missing value")?;
                args.trusted_root = Some(parse_hex_32(&v)?);
            }
            "--max-errors" => {
                let v = it.next().ok_or("--max-errors missing value")?;
                args.max_errors = v.parse().map_err(|_| "invalid --max-errors")?;
            }
            "--verbose" => args.verbose = true,
            _ => return Err(format!("unknown arg {arg}")),
        }
    }
    Ok(args)
}

fn apply_overrides(args: &Args, file: &mut MockRegistryFile) {
    if let Some(tree_height) = args.tree_height {
        file.system.auth.tree_height = tree_height;
    }
    if let Some(segment_len) = args.segment_len {
        file.system.unauth.segment_len = segment_len;
    }
    if let Some(node_linearization) = args.node_linearization.as_deref() {
        let parsed = match node_linearization {
            "preorder" => NodeLinearization::Preorder,
            _ => {
                eprintln!("invalid --node-linearization (expected preorder)");
                std::process::exit(1);
            }
        };
        file.system.auth.node_linearization = parsed;
    }
    if let Some(s) = args.s {
        file.system.unauth.s = s;
    }
    if let Some(shard_count) = args.shard_count {
        file.system.unauth.shard_count = shard_count;
    }
    if let Some(len) = args.unauth_array_len {
        file.system.unauth_array_len = Some(len);
    }
    if let Some(alloc) = args.allocated_leaves {
        file.system.auth.allocated_leaves = alloc;
    }
}

fn parse_hex_32(input: &str) -> Result<[u8; 32], String> {
    let s = input.strip_prefix("0x").unwrap_or(input);
    if s.len() != 64 {
        return Err(format!("expected 64 hex chars, got {}", s.len()));
    }
    let mut out = [0u8; 32];
    let bytes = s.as_bytes();
    for i in 0..32 {
        let hi = from_hex_digit(bytes[2 * i])?;
        let lo = from_hex_digit(bytes[2 * i + 1])?;
        out[31 - i] = (hi << 4) | lo;
    }
    Ok(out)
}

fn from_hex_digit(b: u8) -> Result<u8, String> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(format!("invalid hex digit {}", b as char)),
    }
}

fn to_hex(bytes: [u8; 32]) -> String {
    let mut out = String::with_capacity(66);
    out.push_str("0x");
    for byte in bytes.into_iter().rev() {
        out.push(hex_digit(byte >> 4));
        out.push(hex_digit(byte & 0x0f));
    }
    out
}

fn hex_digit(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        10..=15 => (b'a' + (nibble - 10)) as char,
        _ => '?',
    }
}

fn print_auth_path(path: &AuthPath) {
    println!("auth_path:");
    println!("  index: {}", path.index);
    println!("  root: {}", to_hex(path.root));
    for (i, sibling) in path.siblings.iter().enumerate() {
        println!("  sibling[{}]: {}", i, to_hex(*sibling));
    }
}

#[derive(Deserialize)]
struct BatchDelta {
    index: u64,
    delta: String,
}

#[derive(Deserialize)]
struct BatchBlockDelta {
    offset: usize,
    delta: String,
}

#[derive(Deserialize)]
struct BatchWrite {
    index: u64,
    value: String,
}

fn write_registry_file(registry_path: &Path, file: &MockRegistryFile) {
    let updated = serde_json::to_string_pretty(file).unwrap_or_else(|err| {
        eprintln!("failed to encode updated registry: {err}");
        std::process::exit(1);
    });
    fs::write(registry_path, updated).unwrap_or_else(|err| {
        eprintln!("failed to write updated registry: {err}");
        std::process::exit(1);
    });
}

fn active_owned_indices(file: &MockRegistryFile) -> Vec<Vec<usize>> {
    let (params, claims) = build_unauth_registry_view(file).unwrap_or_else(|err| {
        eprintln!("failed to build target unauth registry view: {:?}", err);
        std::process::exit(1);
    });
    let mut parties = file.parties.clone();
    parties.retain(|party| party.active);
    parties.sort_by_key(|party| party.party_id);
    parties
        .iter()
        .map(|party| {
            claim_bundles_for_party(&params, &claims, party.party_id)
                .unwrap_or_else(|err| {
                    eprintln!("failed to resolve target claims: {:?}", err);
                    std::process::exit(1);
                })
                .into_iter()
                .map(|claim| claim.registry_index)
                .collect()
        })
        .collect()
}

fn migrate_one_claim_left(
    client: &mut UnauthClient,
    file: &mut MockRegistryFile,
    registry_path: &Path,
    target_s: usize,
    max_errors: usize,
) {
    let mut target_file = file.clone();
    migrate_unauth_registry_file_one_claim_left(&mut target_file, target_s).unwrap_or_else(|err| {
        eprintln!("failed to compute one-claim target registry: {:?}", err);
        std::process::exit(1);
    });
    let retain_plan = active_owned_indices(&target_file);
    let result = if max_errors == 0 {
        client.migrate_one_claim_left(target_s, retain_plan)
    } else {
        client.migrate_one_claim_left_correcting(target_s, retain_plan, max_errors)
    };
    result.unwrap_or_else(|err| {
        eprintln!("one-claim migration failed: {:?}", err);
        std::process::exit(1);
    });
    *file = target_file;
    write_registry_file(registry_path, file);
}

fn run_unauth_maintenance_op(
    op: &str,
    args: &Args,
    file: &mut MockRegistryFile,
    registry_path: &Path,
) -> bool {
    match op {
        "migrate"
        | "migrate-one-claim-left"
        | "kappa-halve"
        | "kappa-halve-one-claim-left"
        | "kappa-double"
        | "kappa-double-one-claim-left"
        | "segment-double"
        | "repair"
        | "repair-row"
        | "repair-party-state"
        | "repair-party-state-json"
        | "repair-party-all-json" => {
            let mut client = UnauthClient::from_registry(file).unwrap_or_else(|err| {
                eprintln!("unauth client init failed: {:?}", err);
                std::process::exit(1);
            });
            match op {
                "migrate" => {
                    let target_s = args.target_s.unwrap_or_else(|| {
                        eprintln!("--target-s is required for migrate");
                        std::process::exit(1);
                    });
                    let result = if args.max_errors == 0 {
                        client.migrate(target_s)
                    } else {
                        client.migrate_correcting(target_s, args.max_errors)
                    };
                    result.unwrap_or_else(|err| {
                        eprintln!("migrate failed: {:?}", err);
                        std::process::exit(1);
                    });
                    migrate_unauth_registry_file(file, target_s).unwrap_or_else(|err| {
                        eprintln!("failed to update registry after migrate: {:?}", err);
                        std::process::exit(1);
                    });
                    write_registry_file(registry_path, file);
                }
                "migrate-one-claim-left" => {
                    let target_s = args.target_s.unwrap_or_else(|| {
                        eprintln!("--target-s is required for migrate-one-claim-left");
                        std::process::exit(1);
                    });
                    migrate_one_claim_left(
                        &mut client,
                        file,
                        registry_path,
                        target_s,
                        args.max_errors,
                    );
                }
                "kappa-halve" => {
                    let target_s = file.system.unauth.s.checked_mul(2).unwrap_or_else(|| {
                        eprintln!("kappa-halve would overflow s");
                        std::process::exit(1);
                    });
                    let result = if args.max_errors == 0 {
                        client.kappa_halve()
                    } else {
                        client.kappa_halve_correcting(args.max_errors)
                    };
                    result.unwrap_or_else(|err| {
                        eprintln!("kappa-halve failed: {:?}", err);
                        std::process::exit(1);
                    });
                    kappa_halve_unauth_registry_file(file).unwrap_or_else(|err| {
                        eprintln!("failed to update registry after kappa-halve: {:?}", err);
                        std::process::exit(1);
                    });
                    debug_assert_eq!(file.system.unauth.s, target_s);
                    write_registry_file(registry_path, file);
                }
                "kappa-halve-one-claim-left" => {
                    let target_s = file.system.unauth.s.checked_mul(2).unwrap_or_else(|| {
                        eprintln!("kappa-halve would overflow s");
                        std::process::exit(1);
                    });
                    migrate_one_claim_left(
                        &mut client,
                        file,
                        registry_path,
                        target_s,
                        args.max_errors,
                    );
                }
                "kappa-double" => {
                    if file.system.unauth.s <= 1 {
                        eprintln!("cannot kappa-double below s=1");
                        std::process::exit(1);
                    }
                    let result = if args.max_errors == 0 {
                        client.kappa_double()
                    } else {
                        client.kappa_double_correcting(args.max_errors)
                    };
                    result.unwrap_or_else(|err| {
                        eprintln!("kappa-double failed: {:?}", err);
                        std::process::exit(1);
                    });
                    kappa_double_unauth_registry_file(file).unwrap_or_else(|err| {
                        eprintln!("failed to update registry after kappa-double: {:?}", err);
                        std::process::exit(1);
                    });
                    write_registry_file(registry_path, file);
                }
                "kappa-double-one-claim-left" => {
                    if file.system.unauth.s <= 1 {
                        eprintln!("cannot kappa-double below s=1");
                        std::process::exit(1);
                    }
                    let target_s = file.system.unauth.s / 2;
                    migrate_one_claim_left(
                        &mut client,
                        file,
                        registry_path,
                        target_s,
                        args.max_errors,
                    );
                }
                "segment-double" => {
                    let mut updated_file = file.clone();
                    segment_double_unauth_registry_file(&mut updated_file).unwrap_or_else(|err| {
                        eprintln!("failed to prepare segment-double registry: {:?}", err);
                        std::process::exit(1);
                    });
                    client.segment_double().unwrap_or_else(|err| {
                        eprintln!("segment-double failed: {:?}", err);
                        std::process::exit(1);
                    });
                    *file = updated_file;
                    write_registry_file(registry_path, file);
                }
                "repair" => {
                    let seg = args.segment.unwrap_or_else(|| {
                        eprintln!("--segment is required for repair");
                        std::process::exit(1);
                    });
                    let registry_index = args.registry_index.unwrap_or_else(|| {
                        eprintln!("--registry-index is required for repair");
                        std::process::exit(1);
                    });
                    let values = match args.party_id {
                        Some(party_id) => client
                            .repair_shard(party_id, seg, registry_index)
                            .unwrap_or_else(|err| {
                                eprintln!("repair failed: {:?}", err);
                                std::process::exit(1);
                            }),
                        None => client
                            .repair_shard_from_network_correcting(
                                seg,
                                registry_index,
                                args.max_errors,
                            )
                            .unwrap_or_else(|err| {
                                eprintln!("network repair failed: {:?}", err);
                                std::process::exit(1);
                            }),
                    };
                    for value in values {
                        println!("{}", to_hex(value));
                    }
                }
                "repair-row" => {
                    let seg = args.segment.unwrap_or_else(|| {
                        eprintln!("--segment is required for repair-row");
                        std::process::exit(1);
                    });
                    let row_label_index = args.row_label_index.unwrap_or_else(|| {
                        eprintln!("--row-label-index is required for repair-row");
                        std::process::exit(1);
                    });
                    let values = client
                        .repair_row_from_network_correcting(seg, row_label_index, args.max_errors)
                        .unwrap_or_else(|err| {
                            eprintln!("row repair failed: {:?}", err);
                            std::process::exit(1);
                        });
                    for value in values {
                        println!("{}", to_hex(value));
                    }
                }
                "repair-party-state" | "repair-party-state-json" | "repair-party-all-json" => {
                    let party_id = args.party_id.unwrap_or_else(|| {
                        eprintln!("--party-id is required for repair-party-state");
                        std::process::exit(1);
                    });
                    let (params, claims) = build_unauth_registry_view(file).unwrap_or_else(|err| {
                        eprintln!("failed to build unauth registry view: {:?}", err);
                        std::process::exit(1);
                    });
                    let claim_bundles = claim_bundles_for_party(&params, &claims, party_id)
                        .unwrap_or_else(|err| {
                            eprintln!("failed to resolve party claims: {:?}", err);
                            std::process::exit(1);
                        });
                    let registry_indices: Vec<_> = claim_bundles
                        .iter()
                        .map(|claim| claim.registry_index)
                        .collect();
                    let mut row_label_indices = Vec::new();
                    for claim in &claim_bundles {
                        row_label_indices.extend(claim.row_label_indices(&params).unwrap_or_else(
                            || {
                                eprintln!(
                                    "claim {} has invalid row family {:?}",
                                    claim.registry_index, claim.row_family_id
                                );
                                std::process::exit(1);
                            },
                        ));
                    }
                    row_label_indices.sort_unstable();
                    row_label_indices.dedup();
                    let segment_len = public_values_per_segment(&params) as u64;
                    let total_len = file
                        .system
                        .unauth_array_len
                        .unwrap_or(public_values_per_segment(&params) as u64);
                    let segment_count = total_len.div_ceil(segment_len) as usize;
                    let segment_range: Vec<usize> = if op == "repair-party-all-json" {
                        (0..segment_count).collect()
                    } else {
                        let seg = args.segment.unwrap_or_else(|| {
                            eprintln!("--segment is required for repair-party-state");
                            std::process::exit(1);
                        });
                        vec![seg]
                    };

                    let all_segments = client
                        .repair_party_state_segments_from_network_correcting(
                            &segment_range,
                            &registry_indices,
                            &row_label_indices,
                            args.max_errors,
                        )
                        .unwrap_or_else(|err| {
                            eprintln!("party state repair failed: {:?}", err);
                            std::process::exit(1);
                        })
                        .into_iter()
                        .map(|state| (state.seg, state.shards, state.rows))
                        .collect::<Vec<_>>();

                    if op == "repair-party-state-json" || op == "repair-party-all-json" {
                        let segments_json: Vec<_> = all_segments
                            .into_iter()
                            .map(|(seg, repaired_shards, repaired_rows)| {
                                let shards_json: Vec<_> = repaired_shards
                                    .into_iter()
                                    .map(|(registry_index, values)| {
                                        json!({
                                            "registry_index": registry_index,
                                            "values_hex": values.into_iter().map(to_hex).collect::<Vec<_>>(),
                                        })
                                    })
                                    .collect();
                                let rows_json: Vec<_> = repaired_rows
                                    .into_iter()
                                    .map(|(row_label_index, values)| {
                                        json!({
                                            "row_label_index": row_label_index,
                                            "values_hex": values.into_iter().map(to_hex).collect::<Vec<_>>(),
                                        })
                                    })
                                    .collect();
                                json!({
                                    "segment": seg,
                                    "shards": shards_json,
                                    "rows": rows_json,
                                })
                            })
                            .collect();
                        let payload = if op == "repair-party-all-json" {
                            json!({
                                "party_id": party_id,
                                "segment_count": segment_count,
                                "segments": segments_json,
                            })
                        } else {
                            json!({
                                "party_id": party_id,
                                "segment": segments_json[0]["segment"],
                                "shards": segments_json[0]["shards"],
                                "rows": segments_json[0]["rows"],
                            })
                        };
                        println!("{}", serde_json::to_string_pretty(&payload).unwrap());
                    } else {
                        let (seg, repaired_shards, repaired_rows) = all_segments
                            .into_iter()
                            .next()
                            .expect("single-segment repair output");
                        for (registry_index, values) in repaired_shards {
                            println!("shard {}", registry_index);
                            for value in values {
                                println!("{}", to_hex(value));
                            }
                        }
                        for (row_label_index, values) in repaired_rows {
                            println!("row {}", row_label_index);
                            for value in values {
                                println!("{}", to_hex(value));
                            }
                        }
                        let _ = seg;
                    }
                }
                _ => unreachable!(),
            }
            true
        }
        "onboard-party" => {
            let party_id = args.party_id.unwrap_or_else(|| {
                eprintln!("--party-id is required for onboard-party");
                std::process::exit(1);
            });
            let count = args.count.unwrap_or_else(|| {
                eprintln!("--count is required for onboard-party");
                std::process::exit(1);
            });
            let new_claims = onboard_unauth_party_with_new_shards(file, party_id, count)
                .unwrap_or_else(|err| {
                    eprintln!("failed to onboard party: {:?}", err);
                    std::process::exit(1);
                });
            write_registry_file(registry_path, file);
            for claim in new_claims {
                let descriptor = if !claim.row_label_indices.is_empty() {
                    claim
                        .row_label_indices
                        .iter()
                        .map(|idx| idx.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                } else if let Some(row_family_id) = claim.row_family_id {
                    format!("family:{}", row_family_id)
                } else {
                    "-".to_string()
                };
                println!(
                    "party {} <- shard {} rows {}",
                    claim.party_id, claim.registry_index, descriptor
                );
            }
            true
        }
        _ => false,
    }
}

fn main() {
    let args = parse_args().unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(1);
    });

    let mode = args.mode.clone().unwrap_or_else(|| {
        eprintln!("--mode is required");
        std::process::exit(1);
    });
    let op = args.op.clone().unwrap_or_else(|| {
        eprintln!("--op is required");
        std::process::exit(1);
    });
    let registry_path = args.registry.clone().unwrap_or_else(|| {
        eprintln!("--registry is required");
        std::process::exit(1);
    });
    let verbose = args.verbose;

    let mut file = MockRegistryFile::load(&registry_path).unwrap_or_else(|err| {
        eprintln!("failed to load registry: {:?}", err);
        std::process::exit(1);
    });
    apply_overrides(&args, &mut file);

    match mode.as_str() {
        "unauth" => {
            if run_unauth_maintenance_op(&op, &args, &mut file, &registry_path) {
                return;
            }
            let mut client = UnauthClient::from_registry(&file).unwrap_or_else(|err| {
                eprintln!("unauth client init failed: {:?}", err);
                std::process::exit(1);
            });
            match op.as_str() {
                "read" => {
                    let index = args.index.unwrap_or_else(|| {
                        eprintln!("--index is required for read");
                        std::process::exit(1);
                    });
                    let value = if args.max_errors == 0 {
                        client.read(index)
                    } else {
                        client.read_correcting(index, args.max_errors)
                    }
                    .unwrap_or_else(|err| {
                        eprintln!("read failed: {:?}", err);
                        std::process::exit(1);
                    });
                    if verbose {
                        println!("value: {}", to_hex(value));
                    } else {
                        println!("{}", to_hex(value));
                    }
                }
                "read-block" => {
                    let index = args.index.unwrap_or_else(|| {
                        eprintln!("--index is required for read-block");
                        std::process::exit(1);
                    });
                    let values = client.read_block(index).unwrap_or_else(|err| {
                        eprintln!("block read failed: {:?}", err);
                        std::process::exit(1);
                    });
                    for value in values {
                        println!("{}", to_hex(value));
                    }
                }
                "write" => {
                    let index = args.index.unwrap_or_else(|| {
                        eprintln!("--index is required for write");
                        std::process::exit(1);
                    });
                    let value = args.value.clone().unwrap_or_else(|| {
                        eprintln!("--value is required for write");
                        std::process::exit(1);
                    });
                    let bytes = parse_hex_32(&value).unwrap_or_else(|err| {
                        eprintln!("invalid value: {err}");
                        std::process::exit(1);
                    });
                    let delta = decode_value(bytes);
                    client.write_delta(index, delta).unwrap_or_else(|err| {
                        eprintln!("write failed: {:?}", err);
                        std::process::exit(1);
                    });
                }
                "block-write" => {
                    let block = args.index.unwrap_or_else(|| {
                        eprintln!("--index is required for block-write");
                        std::process::exit(1);
                    });
                    let batch_path = args.batch_file.clone().unwrap_or_else(|| {
                        eprintln!("--batch-file is required for block-write");
                        std::process::exit(1);
                    });
                    let data = fs::read_to_string(&batch_path).unwrap_or_else(|err| {
                        eprintln!("failed to read block batch file: {err}");
                        std::process::exit(1);
                    });
                    let items: Vec<BatchBlockDelta> =
                        serde_json::from_str(&data).unwrap_or_else(|err| {
                            eprintln!("invalid block batch json: {err}");
                            std::process::exit(1);
                        });
                    let mut deltas = Vec::with_capacity(items.len());
                    for item in items {
                        let bytes = parse_hex_32(&item.delta).unwrap_or_else(|err| {
                            eprintln!("invalid delta: {err}");
                            std::process::exit(1);
                        });
                        deltas.push((item.offset, decode_value(bytes)));
                    }
                    let mode = match args.delta_mode.as_deref() {
                        Some("fft") => DeltaMode::InterpolateFFT,
                        _ => DeltaMode::Aggregate,
                    };
                    client
                        .write_block(block, &deltas, mode)
                        .unwrap_or_else(|err| {
                            eprintln!("block write failed: {:?}", err);
                            std::process::exit(1);
                        });
                }
                "batch-write" => {
                    let batch_path = args.batch_file.clone().unwrap_or_else(|| {
                        eprintln!("--batch-file is required for batch-write");
                        std::process::exit(1);
                    });
                    let data = fs::read_to_string(&batch_path).unwrap_or_else(|err| {
                        eprintln!("failed to read batch file: {err}");
                        std::process::exit(1);
                    });
                    let items: Vec<BatchDelta> =
                        serde_json::from_str(&data).unwrap_or_else(|err| {
                            eprintln!("invalid batch json: {err}");
                            std::process::exit(1);
                        });
                    let mut deltas = Vec::with_capacity(items.len());
                    for item in items {
                        let bytes = parse_hex_32(&item.delta).unwrap_or_else(|err| {
                            eprintln!("invalid delta: {err}");
                            std::process::exit(1);
                        });
                        deltas.push((item.index, decode_value(bytes)));
                    }
                    let mode = match args.delta_mode.as_deref() {
                        Some("fft") => DeltaMode::InterpolateFFT,
                        _ => DeltaMode::Aggregate,
                    };
                    client.write_batch(&deltas, mode).unwrap_or_else(|err| {
                        eprintln!("batch write failed: {:?}", err);
                        std::process::exit(1);
                    });
                }
                "multi-read" | "batch-read" => {
                    let batch_path = args.batch_file.clone().unwrap_or_else(|| {
                        eprintln!("--batch-file is required for multi-read");
                        std::process::exit(1);
                    });
                    let data = fs::read_to_string(&batch_path).unwrap_or_else(|err| {
                        eprintln!("failed to read batch file: {err}");
                        std::process::exit(1);
                    });
                    let items: Vec<BatchDelta> =
                        serde_json::from_str(&data).unwrap_or_else(|err| {
                            eprintln!("invalid batch json: {err}");
                            std::process::exit(1);
                        });
                    let idxs: Vec<u64> = items.into_iter().map(|item| item.index).collect();
                    let values = if args.max_errors == 0 {
                        client.multi_read(&idxs)
                    } else {
                        client.multi_read_correcting(&idxs, args.max_errors)
                    }
                    .unwrap_or_else(|err| {
                        eprintln!("multi-read failed: {:?}", err);
                        std::process::exit(1);
                    });
                    for value in values {
                        println!("{}", to_hex(value));
                    }
                }
                "extend" => {
                    let new_len = args.new_len.unwrap_or_else(|| {
                        eprintln!("--new-len is required for extend");
                        std::process::exit(1);
                    });
                    client.extend(new_len).unwrap_or_else(|err| {
                        eprintln!("extend failed: {:?}", err);
                        std::process::exit(1);
                    });
                }
                other => {
                    eprintln!("unknown op {other}");
                    std::process::exit(1);
                }
            }
        }
        "auth" => {
            if run_unauth_maintenance_op(&op, &args, &mut file, &registry_path) {
                return;
            }
            let mut client = AuthClient::from_registry(&file).unwrap_or_else(|err| {
                eprintln!("auth client init failed: {:?}", err);
                std::process::exit(1);
            });
            if let Some(trusted_root) = args.trusted_root {
                client.set_trusted_root(trusted_root);
            }
            match op.as_str() {
                "read" => {
                    let index = args.index.unwrap_or_else(|| {
                        eprintln!("--index is required for read");
                        std::process::exit(1);
                    });
                    if verbose {
                        let (value, path) = if args.max_errors == 0 {
                            client.read_with_path(index)
                        } else {
                            client.read_with_path_correcting(index, args.max_errors)
                        }
                        .unwrap_or_else(|err| {
                            eprintln!("read failed: {:?}", err);
                            std::process::exit(1);
                        });
                        println!("value: {}", to_hex(value));
                        print_auth_path(&path);
                    } else {
                        let value = if args.max_errors == 0 {
                            client.read(index)
                        } else {
                            client.read_correcting(index, args.max_errors)
                        }
                        .unwrap_or_else(|err| {
                            eprintln!("read failed: {:?}", err);
                            std::process::exit(1);
                        });
                        println!("{}", to_hex(value));
                    }
                }
                "multi-read" | "batch-read" => {
                    let batch_path = args.batch_file.clone().unwrap_or_else(|| {
                        eprintln!("--batch-file is required for multi-read");
                        std::process::exit(1);
                    });
                    let data = fs::read_to_string(&batch_path).unwrap_or_else(|err| {
                        eprintln!("failed to read batch file: {err}");
                        std::process::exit(1);
                    });
                    let items: Vec<BatchDelta> =
                        serde_json::from_str(&data).unwrap_or_else(|err| {
                            eprintln!("invalid batch json: {err}");
                            std::process::exit(1);
                        });
                    let idxs: Vec<u64> = items.into_iter().map(|item| item.index).collect();
                    let values = if args.max_errors == 0 {
                        client.read_batch(&idxs)
                    } else {
                        client.read_batch_correcting(&idxs, args.max_errors)
                    }
                    .unwrap_or_else(|err| {
                        eprintln!("multi-read failed: {:?}", err);
                        std::process::exit(1);
                    });
                    for value in values {
                        println!("{}", to_hex(value));
                    }
                }
                "write" => {
                    let index = args.index.unwrap_or_else(|| {
                        eprintln!("--index is required for write");
                        std::process::exit(1);
                    });
                    let value = args.value.clone().unwrap_or_else(|| {
                        eprintln!("--value is required for write");
                        std::process::exit(1);
                    });
                    let bytes = parse_hex_32(&value).unwrap_or_else(|err| {
                        eprintln!("invalid value: {err}");
                        std::process::exit(1);
                    });
                    if verbose {
                        let (old, path, root_new) =
                            client.write_with_path(index, bytes).unwrap_or_else(|err| {
                                eprintln!("write failed: {:?}", err);
                                std::process::exit(1);
                            });
                        println!("pre-read auth path:");
                        println!("  old: {}", to_hex(old));
                        println!("  index: {}", path.index);
                        println!("  root: {}", to_hex(path.root));
                        for (i, sibling) in path.siblings.iter().enumerate() {
                            println!("  sibling[{}]: {}", i, to_hex(*sibling));
                        }
                        println!("root_new: {}", to_hex(root_new));
                    } else {
                        client.write(index, bytes).unwrap_or_else(|err| {
                            eprintln!("write failed: {:?}", err);
                            std::process::exit(1);
                        });
                    }
                }
                "batch-write" => {
                    let batch_path = args.batch_file.clone().unwrap_or_else(|| {
                        eprintln!("--batch-file is required for batch-write");
                        std::process::exit(1);
                    });
                    let data = fs::read_to_string(&batch_path).unwrap_or_else(|err| {
                        eprintln!("failed to read batch file: {err}");
                        std::process::exit(1);
                    });
                    let items: Vec<BatchWrite> =
                        serde_json::from_str(&data).unwrap_or_else(|err| {
                            eprintln!("invalid batch json: {err}");
                            std::process::exit(1);
                        });
                    let mut updates = Vec::with_capacity(items.len());
                    for item in items {
                        let bytes = parse_hex_32(&item.value).unwrap_or_else(|err| {
                            eprintln!("invalid value: {err}");
                            std::process::exit(1);
                        });
                        updates.push((item.index, bytes));
                    }
                    client.write_batch(&updates).unwrap_or_else(|err| {
                        eprintln!("batch write failed: {:?}", err);
                        std::process::exit(1);
                    });
                }
                "read-range" => {
                    let index = args.index.unwrap_or_else(|| {
                        eprintln!("--index is required for read-range");
                        std::process::exit(1);
                    });
                    let count = args.count.unwrap_or_else(|| {
                        eprintln!("--count is required for read-range");
                        std::process::exit(1);
                    });
                    let values = if args.max_errors == 0 {
                        client.read_range(index, count)
                    } else {
                        client.read_range_correcting(index, count, args.max_errors)
                    }
                    .unwrap_or_else(|err| {
                        eprintln!("read-range failed: {:?}", err);
                        std::process::exit(1);
                    });
                    for value in values {
                        println!("{}", to_hex(value));
                    }
                }
                "extend" => {
                    let new_len = args.new_len.unwrap_or_else(|| {
                        eprintln!("--new-len is required for extend");
                        std::process::exit(1);
                    });
                    let max_leaves = client.auth.conceptual_l_max_u128();
                    if new_len as u128 > max_leaves {
                        eprintln!(
                            "extend failed: new_len {} exceeds max leaves {} for tree_height {}",
                            new_len, max_leaves, client.auth.tree_height
                        );
                        std::process::exit(1);
                    }
                    client.extend(new_len).unwrap_or_else(|err| {
                        eprintln!("extend failed: {:?}", err);
                        std::process::exit(1);
                    });
                }
                other => {
                    eprintln!("unknown op {other}");
                    std::process::exit(1);
                }
            }
        }
        other => {
            eprintln!("unknown mode {other}");
            std::process::exit(1);
        }
    }
}
