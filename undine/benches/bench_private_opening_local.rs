//! Private openings of one position: one server's answer, its preprocessing,
//! the preprocessing update after an append, and the client's work.
//!
//! One process measures one configuration, chosen by `ROLE`:
//! - `maintainer`: one claim of a maintainer. Its PIR database is the claim's
//!   primary columns, built by the bulk builder for a seeded log. The process
//!   times the client's request, the maintainer's first pass and complete
//!   answer, the client's decoding of the reply, the hint of a sample of
//!   database columns, and the preprocessing update after each of a sequence
//!   of appends. With `DEFAULT_PREFIX=1` the leaves of the initial log are
//!   allocated and unwritten, as after an allocation, so the columns are
//!   built without hashing; the appended leaves are written. The PIR work
//!   does not depend on the cell values, so the times are those of a
//!   written log of the same length.
//! - `single`: a server that stores the whole tree in records of
//!   `CELLS_PER_RECORD` cells. Same request, answer, decoding, and hint
//!   sample; it also reports the database shape at the log lengths of
//!   `SHAPE_LEAVES_LIST`.
//! - `table`: YPIR over the table of complete openings of one claim, one row
//!   per pattern. First pass and complete answer.
//! - `client`: a client that opens one position from `kappa` replies and from
//!   `3 kappa` replies of which `kappa` lie, privately and publicly. Every
//!   reply comes from the PIR database of its own claim.
//!
//! Setup runs on `SETUP_THREADS` threads over all online CPUs; the timed
//! operations run on the calling thread with the CPU set the process was
//! started with.

use paper_a_optimized::auth::client::{AuthClient, AuthTransport, ClientError as AuthClientError};
use paper_a_optimized::auth::mapping::NodeLinearization;
use paper_a_optimized::auth::service::{
    AuthBatchWriteRequest, AuthBatchWriteResponse, AuthReadNodeReply, AuthReadResponse,
    AuthSubtreeReadResponse, AuthWriteRequest, AuthWriteResponse,
};
use paper_a_optimized::net;
use paper_a_optimized::net::wire::{
    AuthPrivateReadReply, AuthPrivateReadRequest, AuthPrivateReadResponse, NetByteBlob,
    decode_binary, encode_binary,
};
use paper_a_optimized::private::bulk::CHUNK_BITS;
use paper_a_optimized::private::pir::{bytes_from_cells, bytes_to_cells, UPPER_CELLS};
use paper_a_optimized::private::wire::{
    WORD_BYTES, pack_words, request_to_net, response_from_net, unpack_words,
};
use paper_a_optimized::private::{
    BulkLog, PackKeys, PirClient, PirDatabase, PirError, PirShape, PlannedEntry, PrivateLayout,
    PrivateOpening, PrivateReadResponse, TableDatabase, build_primary_columns,
    build_tree_cells, bulk_leaf_value, entries_to_cells, entry_from_cells, pattern_bits,
    Grouping,
};
use paper_a_optimized::unauth::error_correcting_share_target;
use paper_a_optimized::merkle::{default_hash, leaf_key, parent, recompute_root, sibling};
use paper_a_optimized::{
    AuthParams, CodedArrayConfig, CodedArrayParams, Fp, ShardShare, Value256, build_coded_params,
    build_shard_registry_deterministic, decode_value, encode_value, fp_one, fp_zero,
};
use serde::Serialize;
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const ENV_PREFIX: &str = "BENCH_PRIVATE_OPENING";
const BENCH_SEED: u64 = 42;
const DEFAULT_TRIALS: usize = 3;
const DEFAULT_KAPPA: usize = 8;
const DEFAULT_LEAVES: usize = 1 << 15;
const DEFAULT_TREE_HEIGHT: usize = 50;
const DEFAULT_CELLS_PER_RECORD: usize = 64;
const DEFAULT_APPEND_LEAVES: usize = 1 << 10;
const DEFAULT_HINT_SAMPLE_COLUMNS: usize = 64;
const DEFAULT_CLIENT_SETUP_MEMORY_MIB: usize = 12 * 1024;
const FIELD_BYTES: u64 = 32;
const ORDER: NodeLinearization = NodeLinearization::Preorder;

#[derive(Clone, Debug, Serialize)]
struct MetricStats {
    mean: u64,
    p50: u64,
    p95: u64,
    trials: u64,
    samples: Vec<u64>,
}

#[derive(Clone, Debug, Serialize)]
struct BenchResults {
    meta: BenchMeta,
    entries: Vec<BenchEntry>,
}

#[derive(Clone, Debug, Serialize)]
struct BenchMeta {
    seed: u64,
    trials: usize,
    role: String,
    kappa: usize,
    leaves: u64,
    tree_height: usize,
    segments_per_record: usize,
    cells_per_record: usize,
    append_leaves: u64,
    hint_sample_columns: usize,
    /// Instruction set of the PIR backend's kernels: AVX-512 when the build
    /// targets it, otherwise AVX2 when the host has it, otherwise scalar.
    pir_path: &'static str,
    setup_threads: usize,
    setup_cpus: Option<String>,
    timed_cpus: Option<String>,
    setup_seconds: f64,
    peak_rss_bytes: Option<u64>,
    timestamp_unix_ms: u64,
    git: Option<String>,
    notes: String,
}

/// Dimensions and sizes of one PIR database.
#[derive(Clone, Debug, Serialize)]
struct ShapeFields {
    /// Leaves of the log.
    leaves: u64,
    /// Bytes of the log: 32 per leaf.
    log_bytes: u64,
    /// Bytes of one record: 32 per entry.
    record_bytes: u64,
    records: u64,
    patterns: u64,
    /// Leaves per pattern, as a power of two.
    pattern_bits: u64,
    slots: u64,
    /// Slots of a record that hold groups.
    group_slots: u64,
    db_rows: u64,
    db_cols: u64,
    /// Database columns that hold stored data: the plaintext cells of a
    /// record, its groups and one entry per upper slot.
    record_columns: u64,
    /// Ciphertexts of one reply.
    outputs: u64,
    /// Bytes of the records the server answers from: 32 per entry.
    database_bytes: u64,
    /// Serialized bytes of the client's request.
    request_bytes: u64,
    /// Serialized bytes of the reply of one claim.
    reply_bytes: u64,
}

/// One claim of a maintainer, or a server that stores the whole tree.
#[derive(Clone, Debug, Serialize)]
struct ServerEntry {
    role: &'static str,
    kappa: usize,
    segments_per_record: usize,
    cells_per_record: usize,
    #[serde(flatten)]
    shape: ShapeFields,
    /// Bytes of the hint and the response-packing tables.
    preprocessed_bytes: u64,
    /// Bytes of the stored records, the row table, and the preprocessing.
    serving_state_bytes: u64,
    /// Key generation, the query, and the serialized request.
    client_request_ns: MetricStats,
    /// The query's inner product with every database column.
    server_first_pass_ns: MetricStats,
    /// From the serialized request to the serialized reply.
    server_answer_ns: MetricStats,
    /// From the serialized reply to the entries of the opening.
    client_decode_ns: MetricStats,
    /// The hint of `hint_sample_columns` database columns.
    hint_sample_ns: MetricStats,
    hint_sample_columns: u64,
    /// After an append of `append_leaves` leaves: the changed records and
    /// the patched hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    update_hint_ns: Option<MetricStats>,
    /// After the same append: the response-packing tables of every reply
    /// ciphertext, from the new hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    update_packing_ns: Option<MetricStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    append_leaves: Option<u64>,
    /// The response-packing tables of one reply ciphertext.
    packing_table_ns: MetricStats,
}

/// The database shape of a server at another log length.
#[derive(Clone, Debug, Serialize)]
struct ShapeEntry {
    role: &'static str,
    kappa: usize,
    segments_per_record: usize,
    cells_per_record: usize,
    #[serde(flatten)]
    shape: ShapeFields,
}

/// YPIR over the table of complete openings of one claim.
#[derive(Clone, Debug, Serialize)]
struct TableEntry {
    role: &'static str,
    kappa: usize,
    segments_per_record: usize,
    #[serde(flatten)]
    shape: ShapeFields,
    /// Bytes of the table: one complete opening per pattern, 32 per entry.
    table_bytes: u64,
    server_first_pass_ns: MetricStats,
    server_answer_ns: MetricStats,
}

/// One client that opens one position.
#[derive(Clone, Debug, Serialize)]
struct ClientEntry {
    role: &'static str,
    kappa: usize,
    segments_per_record: usize,
    #[serde(flatten)]
    shape: ShapeFields,
    zero_bad_replies: usize,
    correcting_replies: usize,
    bad_replies: usize,
    /// Key generation, the query, and the serialized request.
    client_request_ns: MetricStats,
    /// Decoding `kappa` replies, then Combine and Verify.
    client_private_zero_bad_ns: MetricStats,
    /// Decoding `3 kappa` honest replies, then Combine with error
    /// correction and Verify: the same registry as the lying case.
    client_private_honest_ns: MetricStats,
    /// Decoding `3 kappa` replies of which `kappa` lie, then Combine and
    /// Verify.
    client_private_kappa_bad_ns: MetricStats,
    /// Combine and Verify from the `kappa` public replies.
    client_public_zero_bad_ns: MetricStats,
    /// Combine with error correction and Verify from the `3 kappa` honest
    /// public replies.
    client_public_honest_ns: MetricStats,
    /// Combine and Verify from the `3 kappa` public replies.
    client_public_kappa_bad_ns: MetricStats,
}

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
enum BenchEntry {
    Server(Box<ServerEntry>),
    Shape(ShapeEntry),
    Table(TableEntry),
    Client(ClientEntry),
}

fn percentile(sorted: &[u64], pct: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (pct * (sorted.len() - 1)) / 100;
    sorted[idx]
}

fn stats_from_samples(mut samples: Vec<u64>) -> MetricStats {
    if samples.is_empty() {
        return MetricStats {
            mean: 0,
            p50: 0,
            p95: 0,
            trials: 0,
            samples: Vec::new(),
        };
    }
    let sum: u128 = samples.iter().map(|&v| v as u128).sum();
    let mean = (sum / samples.len() as u128) as u64;
    let raw_samples = samples.clone();
    samples.sort_unstable();
    MetricStats {
        mean,
        p50: percentile(&samples, 50),
        p95: percentile(&samples, 95),
        trials: samples.len() as u64,
        samples: raw_samples,
    }
}

/// Run `f` once and return its result and duration in nanoseconds.
fn timed<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let start = Instant::now();
    let out = black_box(f());
    (out, start.elapsed().as_nanos() as u64)
}

fn git_sha() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let sha = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if sha.is_empty() { None } else { Some(sha) }
}

fn parse_usize_env(suffix: &str, default: usize) -> usize {
    std::env::var(format!("{ENV_PREFIX}_{suffix}"))
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn parse_usize_list_env(suffix: &str) -> Vec<usize> {
    let Some(raw) = std::env::var(format!("{ENV_PREFIX}_{suffix}")).ok() else {
        return Vec::new();
    };
    raw.split(|c: char| c == ',' || c == ';' || c.is_whitespace())
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse::<usize>()
                .expect("list entry must be an integer")
        })
        .collect()
}

// ---------------------------------------------------------------------------
// CPU set of the calling thread.

/// A line of `/proc/thread-self/status`, without its label.
fn thread_status(label: &str) -> Option<String> {
    let status = fs::read_to_string("/proc/thread-self/status").ok()?;
    status
        .lines()
        .find_map(|line| line.strip_prefix(label))
        .map(|rest| rest.trim().to_string())
}

/// CPUs the calling thread may run on, as the kernel lists them.
fn allowed_cpus() -> Option<String> {
    thread_status("Cpus_allowed_list:")
}

/// Peak resident memory of the process.
fn peak_rss_bytes() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find_map(|line| line.strip_prefix("VmHWM:"))?;
    let kib = line.trim().trim_end_matches("kB").trim().parse::<u64>().ok()?;
    Some(kib * 1024)
}

/// Move the calling thread to the CPUs of `list`. Threads it spawns
/// afterwards start on the same CPUs.
fn set_thread_cpus(list: &str) -> bool {
    let Ok(link) = fs::read_link("/proc/thread-self") else {
        return false;
    };
    let Some(tid) = link.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    Command::new("taskset")
        .args(["-cp", list, tid])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// The CPU set of the timed operations and the widened set of the setup.
struct CpuPlan {
    timed: Option<String>,
    setup: Option<String>,
    setup_threads: usize,
}

impl CpuPlan {
    /// Widen the calling thread to all online CPUs for the setup.
    fn widen() -> Self {
        let timed = allowed_cpus();
        let online = fs::read_to_string("/sys/devices/system/cpu/online")
            .ok()
            .map(|raw| raw.trim().to_string());
        let setup = match (&timed, online) {
            (Some(_), Some(online)) if set_thread_cpus(&online) => allowed_cpus(),
            _ => timed.clone(),
        };
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        CpuPlan {
            timed,
            setup,
            setup_threads: parse_usize_env("SETUP_THREADS", available),
        }
    }

    /// Return the calling thread to the CPU set it started with.
    fn restore(&self) {
        if let (Some(timed), Some(setup)) = (&self.timed, &self.setup) {
            if timed != setup {
                assert!(set_thread_cpus(timed), "restore the CPU set of the timed operations");
                assert_eq!(allowed_cpus().as_ref(), Some(timed), "CPU set of the timed operations");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Parameters and shapes.

fn build_params(kappa: usize, claims: usize) -> CodedArrayParams {
    assert!(kappa.is_power_of_two() && kappa >= 2, "kappa must be a power of two");
    build_coded_params(&CodedArrayConfig {
        kappa,
        shard_count: claims,
        shard_registry: build_shard_registry_deterministic(claims),
        party_registry_indices: (0..claims).map(|idx| vec![idx]).collect(),
        seed: BENCH_SEED,
    })
}

/// Bytes of a bincode variable-length integer.
fn varint_bytes(value: u64) -> u64 {
    match value {
        0..=250 => 1,
        251..=0xffff => 3,
        0x1_0000..=0xffff_ffff => 5,
        _ => 9,
    }
}

/// Bytes of a serialized byte blob of `len` bytes.
fn blob_bytes(len: u64) -> u64 {
    varint_bytes(len) + len
}

/// Serialized bytes of the request message of a database shape.
fn request_message_bytes(shape: &PirShape, leaves: u64, kappa: u64, pattern_bits: u64) -> u64 {
    // Message tag, three integers, and the two blobs of seven-byte words.
    1 + varint_bytes(leaves)
        + varint_bytes(kappa)
        + varint_bytes(pattern_bits)
        + blob_bytes(shape.db_rows() as u64 * WORD_BYTES as u64)
        + blob_bytes((shape.pack_key_words() * WORD_BYTES) as u64)
}

/// Serialized bytes of the reply message of one claim.
fn reply_message_bytes(shape: &PirShape, registry_index: u64) -> u64 {
    let outputs = shape.outputs() as u64;
    // Message tag, one reply, its claim, and its ciphertexts.
    1 + varint_bytes(1)
        + varint_bytes(registry_index)
        + varint_bytes(outputs)
        + outputs * blob_bytes(shape.ciphertext_bytes() as u64)
}

fn shape_fields(layout: &PrivateLayout, kappa: usize, registry_index: usize) -> ShapeFields {
    let shape = PirShape::of(layout);
    let record_bytes = layout.entries_per_record() as u64 * FIELD_BYTES;
    ShapeFields {
        leaves: layout.leaves(),
        log_bytes: layout.leaves() * FIELD_BYTES,
        record_bytes,
        records: layout.records() as u64,
        patterns: shape.patterns as u64,
        pattern_bits: u64::from(layout.group_bits()),
        slots: shape.slots as u64,
        group_slots: shape.group_slots as u64,
        db_rows: shape.db_rows() as u64,
        db_cols: shape.db_cols() as u64,
        record_columns: shape.row_cells() as u64,
        outputs: shape.outputs() as u64,
        database_bytes: layout.records() as u64 * record_bytes,
        request_bytes: request_message_bytes(
            &shape,
            layout.leaves(),
            kappa as u64,
            u64::from(layout.group_bits()),
        ),
        reply_bytes: reply_message_bytes(&shape, registry_index as u64),
    }
}

// ---------------------------------------------------------------------------
// One request, answer, and decoding on the wire.

/// The serialized request of a client with a fresh key for `pattern`.
fn client_request(layout: &PrivateLayout, kappa: usize, pattern: usize) -> (PirClient, Vec<u8>) {
    let client = PirClient::generate(PirShape::of(layout));
    let query = client.query(pattern);
    let pack_keys = client.pack_keys();
    let message = net::Message::AuthReadPrivate {
        req: AuthPrivateReadRequest {
            leaves: layout.leaves(),
            kappa: kappa as u64,
            pattern_bits: u64::from(layout.group_bits()),
            query: NetByteBlob(pack_words(&query)),
            pack_keys: NetByteBlob(pack_words(&pack_keys.words)),
        },
    };
    let bytes = encode_binary(&message).expect("serialize the request");
    (client, bytes)
}

/// The query and the packing keys of a serialized request.
fn parse_request(request: &[u8]) -> (Vec<u64>, PackKeys) {
    let net::Message::AuthReadPrivate { req } =
        decode_binary::<net::Message>(request).expect("parse the request")
    else {
        panic!("the request is a private opening");
    };
    let query = unpack_words(&req.query.0).expect("query words");
    let pack_keys = PackKeys {
        words: unpack_words(&req.pack_keys.0).expect("packing-key words"),
    };
    (query, pack_keys)
}

/// From the serialized request to the serialized reply of one claim.
fn serve(
    answer: &dyn Fn(&[u64], &PackKeys) -> Result<Vec<Vec<u8>>, PirError>,
    registry_index: usize,
    request: &[u8],
) -> Vec<u8> {
    let (query, pack_keys) = parse_request(request);
    let response = answer(&query, &pack_keys).expect("answer");
    let message = net::Response::AuthReadPrivate {
        resp: AuthPrivateReadResponse {
            replies: vec![AuthPrivateReadReply {
                registry_index: registry_index as u32,
                ciphertexts: response.into_iter().map(NetByteBlob).collect(),
            }],
        },
    };
    encode_binary(&message).expect("serialize the reply")
}

/// The replies of a serialized reply message.
fn parse_reply(reply: &[u8]) -> PrivateReadResponse {
    let net::Response::AuthReadPrivate { resp } =
        decode_binary::<net::Response>(reply).expect("parse the reply")
    else {
        panic!("the reply is a private opening");
    };
    response_from_net(resp)
}

/// From the serialized reply of one claim to the entries of the opening, in
/// the order of `plan`.
fn client_decode(client: &PirClient, plan: &[PlannedEntry], reply: &[u8]) -> Vec<Fp> {
    let response = parse_reply(reply);
    let slots = client
        .decode(&response.replies[0].response)
        .expect("decode the reply");
    plan.iter()
        .map(|entry| entry_from_cells(&slots[entry.slot], entry.entry_in_record))
        .collect()
}

/// The entries of the opening of `leaf` as the database stores them.
fn stored_opening(database: &PirDatabase, layout: &PrivateLayout, leaf: u64) -> Vec<Fp> {
    let row = database.row(layout.pattern_of(leaf));
    layout
        .opening_plan(leaf)
        .iter()
        .map(|entry| {
            let id = row[entry.slot].expect("a planned entry has an item");
            let cells = database
                .slot_cells(entry.slot, id)
                .iter()
                .map(|&cell| cell as u64)
                .collect::<Vec<_>>();
            entry_from_cells(&cells, entry.entry_in_record)
        })
        .collect()
}

/// The leaf opened in trial `trial`: the first leaf of a pattern, spread
/// over the log.
fn trial_leaf(layout: &PrivateLayout, trial: usize) -> u64 {
    let patterns = layout.patterns() as u64;
    let pattern = (trial as u64).wrapping_mul(2_654_435_761).wrapping_add(patterns / 3) % patterns;
    (pattern << layout.group_bits()).min(layout.leaves() - 1)
}

/// `count` database columns that hold records, evenly spread.
fn sample_columns(shape: &PirShape, count: usize) -> Vec<usize> {
    let record_columns = shape.row_cells();
    let count = count.min(record_columns);
    (0..count).map(|i| i * record_columns / count).collect()
}

/// The answer trials shared by the server roles.
struct AnswerSamples {
    client_request: Vec<u64>,
    first_pass: Vec<u64>,
    answer: Vec<u64>,
    client_decode: Vec<u64>,
}

/// Time `trials` openings against one database. Every decoded opening is
/// compared with `expected(leaf)`.
#[allow(clippy::too_many_arguments)]
fn answer_trials(
    trials: usize,
    layout: &PrivateLayout,
    kappa: usize,
    registry_index: usize,
    fields: &ShapeFields,
    first_pass: &dyn Fn(&[u64]) -> Result<Vec<u64>, PirError>,
    answer: &dyn Fn(&[u64], &PackKeys) -> Result<Vec<Vec<u8>>, PirError>,
    expected: &dyn Fn(u64) -> Vec<Fp>,
) -> AnswerSamples {
    let mut samples = AnswerSamples {
        client_request: Vec::with_capacity(trials),
        first_pass: Vec::with_capacity(trials),
        answer: Vec::with_capacity(trials),
        client_decode: Vec::with_capacity(trials),
    };
    for trial in 0..trials {
        let leaf = trial_leaf(layout, trial);
        let pattern = layout.pattern_of(leaf);
        let plan = layout.opening_plan(leaf);

        let ((client, request), request_ns) =
            timed(|| client_request(layout, kappa, pattern));
        assert_eq!(request.len() as u64, fields.request_bytes, "request bytes");
        samples.client_request.push(request_ns);

        let (query, _) = parse_request(&request);
        let (_, first_pass_ns) = timed(|| first_pass(&query).expect("first pass"));
        samples.first_pass.push(first_pass_ns);
        drop(query);

        let (reply, answer_ns) = timed(|| serve(answer, registry_index, &request));
        assert_eq!(reply.len() as u64, fields.reply_bytes, "reply bytes");
        samples.answer.push(answer_ns);
        drop(request);

        let (entries, decode_ns) = timed(|| client_decode(&client, &plan, &reply));
        samples.client_decode.push(decode_ns);
        assert!(entries == expected(leaf), "trial {trial}: the opening of leaf {leaf}");
    }
    samples
}

// ---------------------------------------------------------------------------
// Roles.

struct Common {
    trials: usize,
    kappa: usize,
    leaves: u64,
    tree_height: usize,
    hint_sample_columns: usize,
}

/// Time the hint of the sample columns, once per trial, and compare it with
/// the stored hint.
fn hint_sample_trials(common: &Common, database: &PirDatabase) -> (MetricStats, u64) {
    let cols = sample_columns(&database.shape(), common.hint_sample_columns);
    let mut samples = Vec::with_capacity(common.trials);
    for trial in 0..common.trials {
        let (hint, ns) = timed(|| database.hint_of_columns(&cols));
        samples.push(ns);
        if trial == 0 {
            for (col, values) in cols.iter().zip(&hint) {
                assert!(*values == database.stored_hint_column(*col), "hint of column {col}");
            }
        }
    }
    (stats_from_samples(samples), cols.len() as u64)
}

/// Time the response-packing tables of one reply ciphertext, once per trial.
fn packing_table_trials(common: &Common, database: &PirDatabase) -> MetricStats {
    let outputs = database.shape().outputs();
    let samples = (0..common.trials)
        .map(|trial| timed(|| database.packing_table_of_output(trial % outputs)).1)
        .collect();
    stats_from_samples(samples)
}

/// The grouping of a run: the public one by default. `PATTERN_BITS` sets
/// the pattern size and `SEGMENTS_PER_RECORD` the group; with only
/// `SEGMENTS_PER_RECORD`, the pattern is the largest whose subtree fits in
/// one group, so a record holds two groups.
fn grouping_of_env(common: &Common) -> Grouping {
    let segments = std::env::var(format!("{ENV_PREFIX}_SEGMENTS_PER_RECORD")).ok();
    let bits = std::env::var(format!("{ENV_PREFIX}_PATTERN_BITS")).ok();
    let segments_per_group = parse_usize_env("SEGMENTS_PER_RECORD", 1);
    match (segments, bits) {
        (Some(_), None) => Grouping::one_group(segments_per_group, common.kappa * common.kappa),
        _ => Grouping {
            segments_per_group,
            pattern_bits: parse_usize_env(
                "PATTERN_BITS",
                pattern_bits(common.leaves, common.kappa) as usize,
            ) as u32,
        },
    }
}

/// The pattern size of a server that stores the whole tree in groups of
/// `cells_per_record` cells: the public one for threshold 1 by default, and
/// with only `CELLS_PER_RECORD` the largest that fits in one group.
fn single_pattern_bits(common: &Common, cells_per_record: usize) -> u32 {
    let cells = std::env::var(format!("{ENV_PREFIX}_CELLS_PER_RECORD")).ok();
    let bits = std::env::var(format!("{ENV_PREFIX}_PATTERN_BITS")).ok();
    match (cells, bits) {
        (Some(_), None) => Grouping::one_group(1, cells_per_record).pattern_bits,
        _ => parse_usize_env("PATTERN_BITS", pattern_bits(common.leaves, 1) as usize) as u32,
    }
}

fn run_maintainer(common: &Common, cpus: &CpuPlan) -> Vec<BenchEntry> {
    let grouping = grouping_of_env(common);
    let segments_per_record = grouping.segments_per_group;
    let append_leaves = parse_usize_env("APPEND_LEAVES", DEFAULT_APPEND_LEAVES) as u64;
    let registry_index = 0usize;
    let params = build_params(common.kappa, common.kappa);
    let auth = AuthParams::new(common.tree_height);
    let layout_at = |leaves: u64| {
        PrivateLayout::with_grouping(&params, &auth, ORDER, leaves, grouping).expect("layout")
    };

    // Setup: the primary columns of the seeded log and their preprocessing.
    let default_leaves = if parse_usize_env("DEFAULT_PREFIX", 0) == 1 { common.leaves } else { 0 };
    let mut bulk = BulkLog::build_with_default_prefix(
        &params,
        &auth,
        ORDER,
        common.leaves,
        default_leaves,
        BENCH_SEED,
        registry_index,
        cpus.setup_threads,
        CHUNK_BITS,
    );
    let layout = layout_at(common.leaves);
    let cells = PirDatabase::cells_from_entries(&layout, bulk.entries());
    let (mut database, _) = PirDatabase::from_cells_timed(&layout, cells, cpus.setup_threads);
    cpus.restore();

    let fields = shape_fields(&layout, common.kappa, registry_index);
    let preprocessed_bytes = database.preprocessed_bytes() as u64;
    let serving_state_bytes = preprocessed_bytes + database.record_state_bytes() as u64;

    // Openings against the log of `leaves` leaves.
    let answers = {
        let expected = |leaf: u64| stored_opening(&database, &layout, leaf);
        // The stored cells are the entries of the bulk log.
        let leaf = trial_leaf(&layout, 0);
        for (entry, value) in layout.opening_plan(leaf).iter().zip(expected(leaf)) {
            let index = layout.entry_index(entry.array_index);
            assert!(value == bulk.entries()[index], "stored entry of the bulk log");
        }
        answer_trials(
            common.trials,
            &layout,
            common.kappa,
            registry_index,
            &fields,
            &|query| database.first_pass(query),
            &|query, pack_keys| database.answer(query, pack_keys),
            &expected,
        )
    };
    let (hint_sample_ns, hint_sample_columns) = hint_sample_trials(common, &database);
    let packing_table_ns = packing_table_trials(common, &database);

    // A sequence of appends, each followed by the preprocessing update.
    let mut update_hint = Vec::with_capacity(common.trials);
    let mut update_packing = Vec::with_capacity(common.trials);
    let mut leaves = common.leaves;
    for _ in 0..common.trials {
        let old_leaves = leaves;
        leaves += append_leaves;
        bulk.extend(leaves, 1);
        let start = Instant::now();
        let new_layout = layout_at(leaves);
        let changed = new_layout.records_changed_by_append(old_leaves);
        let planning = start.elapsed();
        let times = database.update_from_entries(&new_layout, bulk.entries(), Some(&changed));
        update_hint.push((planning + times.hint).as_nanos() as u64);
        update_packing.push(times.packing.as_nanos() as u64);
    }

    // The updated database equals the database of the longer log: its cells,
    // the hint of the sample columns, and openings across the log.
    let final_layout = layout_at(leaves);
    {
        // Record by record, so that the fresh cells are never held whole
        // next to the database's.
        let record_cells = database.shape().record_cells;
        let per_record = final_layout.entries_per_record();
        let entries = bulk.entries();
        let mut padded = vec![fp_zero(); per_record];
        let mut fresh = vec![0u16; record_cells];
        for record in 0..final_layout.records() {
            let start = (record * per_record).min(entries.len());
            let end = ((record + 1) * per_record).min(entries.len());
            padded[..end - start].copy_from_slice(&entries[start..end]);
            padded[end - start..].fill(fp_zero());
            entries_to_cells(&padded, &mut fresh);
            assert!(database.record_cells(record) == fresh.as_slice(), "block {record} after the updates");
        }
        // The records equal the fresh ones, so the upper nodes read from
        // them are the fresh upper nodes.
        let fresh_upper = PirDatabase::upper_from_cells(&final_layout, database.cells());
        for (level, cells) in fresh_upper.iter().enumerate() {
            for pos in 0..final_layout.upper_count(level) {
                assert!(
                    database.upper_cells(level, pos)
                        == &cells[pos * UPPER_CELLS..(pos + 1) * UPPER_CELLS],
                    "upper node {pos} at level {level} after the updates"
                );
            }
        }
    }
    let cols = sample_columns(&database.shape(), common.hint_sample_columns);
    for (col, values) in cols.iter().zip(database.hint_of_columns(&cols)) {
        assert!(values == database.stored_hint_column(*col), "updated hint of column {col}");
    }
    for leaf in [0, common.leaves - 1, common.leaves, leaves - 1] {
        let plan = final_layout.opening_plan(leaf);
        let (client, request) =
            client_request(&final_layout, common.kappa, final_layout.pattern_of(leaf));
        let reply = serve(
            &|query, pack_keys| database.answer(query, pack_keys),
            registry_index,
            &request,
        );
        assert!(
            client_decode(&client, &plan, &reply) == stored_opening(&database, &final_layout, leaf),
            "opening of leaf {leaf} after the updates"
        );
    }

    vec![BenchEntry::Server(Box::new(ServerEntry {
        role: "maintainer",
        kappa: common.kappa,
        segments_per_record,
        cells_per_record: layout.cells_per_record(),
        shape: fields,
        preprocessed_bytes,
        serving_state_bytes,
        client_request_ns: stats_from_samples(answers.client_request),
        server_first_pass_ns: stats_from_samples(answers.first_pass),
        server_answer_ns: stats_from_samples(answers.answer),
        client_decode_ns: stats_from_samples(answers.client_decode),
        hint_sample_ns,
        hint_sample_columns,
        update_hint_ns: Some(stats_from_samples(update_hint)),
        update_packing_ns: Some(stats_from_samples(update_packing)),
        append_leaves: Some(append_leaves),
        packing_table_ns,
    }))]
}

fn run_single(common: &Common, cpus: &CpuPlan) -> Vec<BenchEntry> {
    let cells_per_record = parse_usize_env("CELLS_PER_RECORD", DEFAULT_CELLS_PER_RECORD);
    let pattern = single_pattern_bits(common, cells_per_record);
    let shape_leaves = parse_usize_list_env("SHAPE_LEAVES_LIST");
    let auth = AuthParams::new(common.tree_height);
    let layout_at = |leaves: u64| {
        PrivateLayout::whole_tree_with_pattern(&auth, ORDER, leaves, cells_per_record, pattern)
            .expect("layout")
    };

    let layout = layout_at(common.leaves);
    let (cells, root) = build_tree_cells(&auth, ORDER, BENCH_SEED, &layout, cpus.setup_threads);
    let (database, _) = PirDatabase::from_cells_timed(&layout, cells, cpus.setup_threads);
    cpus.restore();

    // The single server serves one database: kappa is one on the wire.
    let fields = shape_fields(&layout, 1, 0);
    let preprocessed_bytes = database.preprocessed_bytes() as u64;
    let serving_state_bytes = preprocessed_bytes + database.record_state_bytes() as u64;

    let answers = {
        // The decoded cells are the leaf and the siblings of its path: they
        // hash to the root of the log.
        let expected = |leaf: u64| {
            let opening = stored_opening(&database, &layout, leaf);
            let plan = layout.opening_plan(leaf);
            let value = bulk_leaf_value(BENCH_SEED, leaf);
            assert!(opening[0] == decode_value(value), "leaf {leaf}");
            let height = common.tree_height;
            // A sibling outside the stored tree has the default content of
            // its level: no value for a leaf, the default label above.
            let mut siblings = (0..height)
                .map(|step| match step {
                    0 => [0u8; 32],
                    _ => default_hash(&auth.defaults, height - step),
                })
                .collect::<Vec<Value256>>();
            for (entry, cell) in plan.iter().zip(&opening).skip(1) {
                siblings[height - entry.node.level as usize] = encode_value(*cell);
            }
            assert!(
                recompute_root(height as u32, leaf, value, &siblings, &auth.defaults) == root,
                "path of leaf {leaf}"
            );
            opening
        };
        answer_trials(
            common.trials,
            &layout,
            1,
            0,
            &fields,
            &|query| database.first_pass(query),
            &|query, pack_keys| database.answer(query, pack_keys),
            &expected,
        )
    };
    let (hint_sample_ns, hint_sample_columns) = hint_sample_trials(common, &database);
    let packing_table_ns = packing_table_trials(common, &database);

    let mut entries = vec![BenchEntry::Server(Box::new(ServerEntry {
        role: "single",
        kappa: 1,
        segments_per_record: 0,
        cells_per_record,
        shape: fields,
        preprocessed_bytes,
        serving_state_bytes,
        client_request_ns: stats_from_samples(answers.client_request),
        server_first_pass_ns: stats_from_samples(answers.first_pass),
        server_answer_ns: stats_from_samples(answers.answer),
        client_decode_ns: stats_from_samples(answers.client_decode),
        hint_sample_ns,
        hint_sample_columns,
        update_hint_ns: None,
        update_packing_ns: None,
        append_leaves: None,
        packing_table_ns,
    }))];
    // A longer log has the pattern size of its own length: the measured
    // log's distance from the public size is kept.
    let offset = i64::from(pattern) - i64::from(pattern_bits(common.leaves, 1));
    for leaves in shape_leaves {
        let shape_pattern = (i64::from(pattern_bits(leaves as u64, 1)) + offset) as u32;
        let layout = PrivateLayout::whole_tree_with_pattern(
            &auth,
            ORDER,
            leaves as u64,
            cells_per_record,
            shape_pattern,
        )
        .expect("layout");
        entries.push(BenchEntry::Shape(ShapeEntry {
            role: "single-shape",
            kappa: 1,
            segments_per_record: 0,
            cells_per_record,
            shape: shape_fields(&layout, 1, 0),
        }));
    }
    entries
}

fn run_table(common: &Common, cpus: &CpuPlan) -> Vec<BenchEntry> {
    let grouping = grouping_of_env(common);
    let segments_per_record = grouping.segments_per_group;
    let registry_index = 0usize;
    let params = build_params(common.kappa, common.kappa);
    let auth = AuthParams::new(common.tree_height);
    let layout = PrivateLayout::with_grouping(&params, &auth, ORDER, common.leaves, grouping)
        .expect("layout");

    let bulk = build_primary_columns(
        &params,
        &auth,
        ORDER,
        common.leaves,
        BENCH_SEED,
        registry_index,
        cpus.setup_threads,
    );
    let cells = PirDatabase::cells_from_entries(&layout, &bulk.entries);
    drop(bulk);
    let (database, _) = PirDatabase::from_cells_timed(&layout, cells, cpus.setup_threads);
    let table = TableDatabase::from_database(&database);
    cpus.restore();

    let fields = shape_fields(&layout, common.kappa, registry_index);
    let table_bytes = fields.patterns * fields.slots * fields.record_bytes;
    // The table answers with the bytes of the record-based database.
    {
        let (_, request) = client_request(&layout, common.kappa, 0);
        let (query, pack_keys) = parse_request(&request);
        assert!(
            table.answer(&query, &pack_keys).expect("table answer")
                == database.answer(&query, &pack_keys).expect("database answer"),
            "the table and the records give one answer"
        );
    }

    let mut first_pass = Vec::with_capacity(common.trials);
    let mut answer = Vec::with_capacity(common.trials);
    for trial in 0..common.trials {
        let leaf = trial_leaf(&layout, trial);
        let plan = layout.opening_plan(leaf);
        let (client, request) =
            client_request(&layout, common.kappa, layout.pattern_of(leaf));
        let (query, _) = parse_request(&request);
        first_pass.push(timed(|| table.first_pass(&query).expect("first pass")).1);
        let (reply, answer_ns) = timed(|| {
            serve(
                &|query, pack_keys| table.answer(query, pack_keys),
                registry_index,
                &request,
            )
        });
        answer.push(answer_ns);
        assert!(
            client_decode(&client, &plan, &reply) == stored_opening(&database, &layout, leaf),
            "trial {trial}: the opening of leaf {leaf}"
        );
    }

    vec![BenchEntry::Table(TableEntry {
        role: "table",
        kappa: common.kappa,
        segments_per_record,
        shape: fields,
        table_bytes,
        server_first_pass_ns: stats_from_samples(first_pass),
        server_answer_ns: stats_from_samples(answer),
    })]
}

#[derive(Clone)]
struct NoopTransport;

impl AuthTransport for NoopTransport {
    type Server = usize;

    fn server_label(&self, server: &Self::Server) -> String {
        format!("replay-{server}")
    }

    fn auth_get_len(&self, _server: &Self::Server) -> Result<u64, AuthClientError> {
        Err(AuthClientError::Net("collected replies carry no length".to_string()))
    }

    fn auth_read(
        &self,
        _server: &Self::Server,
        _index: u64,
        _kappa: usize,
    ) -> Result<AuthReadResponse, AuthClientError> {
        Err(AuthClientError::Net("replies are collected in the setup".to_string()))
    }

    fn auth_read_subtree(
        &self,
        _server: &Self::Server,
        _start_index: u64,
        _leaf_count: usize,
    ) -> Result<AuthSubtreeReadResponse, AuthClientError> {
        Err(AuthClientError::Net("replies are collected in the setup".to_string()))
    }

    fn unauth_multi_read(
        &self,
        _server: &Self::Server,
        _indices: &[u64],
    ) -> Result<Vec<Vec<ShardShare>>, AuthClientError> {
        Err(AuthClientError::Net("replies are collected in the setup".to_string()))
    }

    fn auth_write(
        &self,
        _server: &Self::Server,
        _req: &AuthWriteRequest,
    ) -> Result<AuthWriteResponse, AuthClientError> {
        Err(AuthClientError::Net("the client only reads".to_string()))
    }

    fn auth_write_batch(
        &self,
        _server: &Self::Server,
        _req: &AuthBatchWriteRequest,
    ) -> Result<AuthBatchWriteResponse, AuthClientError> {
        Err(AuthClientError::Net("the client only reads".to_string()))
    }

    fn auth_extend(&self, _server: &Self::Server, _new_len: u64) -> Result<bool, AuthClientError> {
        Err(AuthClientError::Net("the client only reads".to_string()))
    }
}

/// The public reply of one claim to the opening of `index`: one share per
/// stored node of the path.
fn public_reply(
    tree_height: usize,
    index: u64,
    registry_index: usize,
    plan: &[PlannedEntry],
    opening: &[Fp],
) -> AuthReadResponse {
    let mut key = leaf_key(tree_height as u32, index);
    let mut nodes = vec![AuthReadNodeReply {
        node: key,
        shares: Vec::new(),
    }];
    for _ in 0..tree_height {
        nodes.push(AuthReadNodeReply {
            node: sibling(key),
            shares: Vec::new(),
        });
        key = parent(key);
    }
    for (entry, value) in plan.iter().zip(opening) {
        let position = if entry.node.level as usize == tree_height && entry.node.pos == index {
            0
        } else {
            tree_height - entry.node.level as usize + 1
        };
        nodes[position]
            .shares
            .push(ShardShare::new(registry_index, Some(*value)));
    }
    AuthReadResponse { nodes }
}

fn same_replies(left: &AuthReadResponse, right: &AuthReadResponse) -> bool {
    left.nodes.len() == right.nodes.len()
        && left.nodes.iter().zip(&right.nodes).all(|(a, b)| {
            a.node == b.node
                && a.shares.len() == b.shares.len()
                && a.shares.iter().zip(&b.shares).all(|(x, y)| {
                    x.registry_index() == y.registry_index() && x.value() == y.value()
                })
        })
}

fn run_client(common: &Common, cpus: &CpuPlan) -> Vec<BenchEntry> {
    let grouping = grouping_of_env(common);
    let segments_per_record = grouping.segments_per_group;
    let memory_mib = parse_usize_env("CLIENT_SETUP_MEMORY_MIB", DEFAULT_CLIENT_SETUP_MEMORY_MIB);
    let kappa = common.kappa;
    let claims = error_correcting_share_target(kappa, kappa);
    let params = build_params(kappa, claims);
    let auth = AuthParams::new(common.tree_height);
    let index = common.leaves / 2 + 1;

    // One opening whose replies the setup collects.
    let opening = PrivateOpening::with_grouping(&params, &auth, ORDER, common.leaves, grouping, index)
        .expect("opening");
    let layout = opening.layout().clone();
    let plan = layout.opening_plan(index);
    let fields = shape_fields(&layout, kappa, claims - 1);

    // Each claim answers the request honestly, and the first `kappa` claims
    // answer a second time with a lie about one node of the path each, so
    // that the same registry is timed with and without lies. The client's
    // work does not depend on the stored values or on how a reply was
    // formed, so the claims hold the columns of a log of allocated, unwritten
    // leaves, which the bulk builder writes without hashing the tree for as
    // many claims per walk as two thirds of the memory hold, and a claim's
    // reply is the row the query selects encrypted under the client's key,
    // in place of the server's preprocessing and answer.
    let column_mib = (fields.database_bytes as usize >> 20).max(1);
    let claims_per_walk = (memory_mib * 2 / 3 / column_mib).clamp(1, claims);
    let workers = cpus.setup_threads.clamp(1, claims);
    let shape = PirShape::of(&layout);
    let pattern = layout.pattern_of(index);
    let row = layout.row(pattern);
    let per_record = layout.entries_per_record();
    let honest: Mutex<Vec<Option<(Vec<u8>, AuthReadResponse, Value256)>>> =
        Mutex::new(vec![None; claims]);
    let lying: Mutex<Vec<Option<(Vec<u8>, AuthReadResponse, Value256)>>> =
        Mutex::new(vec![None; kappa]);
    // The cells of one record of `entries`, as the database holds them.
    let record_cells = |entries: &[Fp], record: usize| -> Vec<u16> {
        let start = (record * per_record).min(entries.len());
        let end = ((record + 1) * per_record).min(entries.len());
        let mut padded = vec![fp_zero(); per_record];
        padded[..end - start].copy_from_slice(&entries[start..end]);
        let mut cells = vec![0u16; shape.record_cells];
        entries_to_cells(&padded, &mut cells);
        cells
    };
    let answer = |claim: usize, entries: &[Fp], root: Value256| {
        let stored = plan
            .iter()
            .map(|entry| entries[layout.entry_index(entry.array_index)])
            .collect::<Vec<_>>();
        // The row the query selects: the record of each group slot and the
        // entry of each upper slot, as `PirDatabase` lays them out.
        let slots = row
            .iter()
            .enumerate()
            .map(|(slot, id)| {
                let mut cells = vec![0u16; shape.slot_width(slot)];
                match id {
                    Some(record) if slot < shape.group_slots => {
                        cells = record_cells(entries, *record);
                    }
                    Some(pos) => {
                        let entry_index = layout.upper_entry_index(slot - shape.group_slots, *pos);
                        let block = record_cells(entries, entry_index / per_record)
                            .into_iter()
                            .map(u64::from)
                            .collect::<Vec<_>>();
                        bytes_to_cells(bytes_from_cells(&block, entry_index % per_record), &mut cells);
                    }
                    None => {}
                }
                cells.into_iter().map(u64::from).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let reply = encode_binary(&net::Response::AuthReadPrivate {
            resp: AuthPrivateReadResponse {
                replies: vec![AuthPrivateReadReply {
                    registry_index: claim as u32,
                    ciphertexts: opening.encrypt_slots(&slots).into_iter().map(NetByteBlob).collect(),
                }],
            },
        })
        .expect("serialize the reply");
        let public = public_reply(common.tree_height, index, claim, &plan, &stored);
        (reply, public, root)
    };
    for first in (0..claims).step_by(claims_per_walk) {
        let group = (first..(first + claims_per_walk).min(claims)).collect::<Vec<_>>();
        let columns = BulkLog::build_for_claims(
            &params,
            &auth,
            ORDER,
            common.leaves,
            common.leaves,
            BENCH_SEED,
            &group,
            cpus.setup_threads,
            CHUNK_BITS,
        )
        .into_claim_columns();
        let columns = columns.into_iter().map(Mutex::new).collect::<Vec<_>>();
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| {
                    loop {
                        let position = next.fetch_add(1, Ordering::Relaxed);
                        if position >= group.len() {
                            break;
                        }
                        let claim = group[position];
                        let mut bulk = columns[position].lock().expect("columns");
                        let bulk = &mut *bulk;
                        honest.lock().expect("honest replies")[claim] =
                            Some(answer(claim, &bulk.entries, bulk.root));
                        if claim < kappa {
                            let entry = &plan[claim % plan.len()];
                            bulk.entries[layout.entry_index(entry.array_index)] += fp_one();
                            lying.lock().expect("lying replies")[claim] =
                                Some(answer(claim, &bulk.entries, bulk.root));
                        }
                        // The columns are not needed again.
                        bulk.entries = Vec::new();
                    }
                });
            }
        });
    }
    cpus.restore();
    let collect = |replies: Mutex<Vec<Option<(Vec<u8>, AuthReadResponse, Value256)>>>| {
        replies
            .into_inner()
            .expect("collected replies")
            .into_iter()
            .map(|reply| reply.expect("every claim replied"))
            .collect::<Vec<_>>()
    };
    let (honest, lying) = (collect(honest), collect(lying));
    let root = honest[0].2;
    // The opened leaf is unwritten.
    let expected_value = encode_value(fp_zero());
    assert_eq!(honest[claims - 1].0.len() as u64, fields.reply_bytes, "reply bytes");

    // The honest replies of every claim, and the replies of the registry
    // in which the first `kappa` claims lie.
    let registry_lying = lying.iter().chain(&honest[kappa..]).collect::<Vec<_>>();
    let private_zero_bad = honest[..kappa]
        .iter()
        .map(|(reply, _, _)| reply.as_slice())
        .collect::<Vec<_>>();
    let private_honest_replies =
        honest.iter().map(|(reply, _, _)| reply.as_slice()).collect::<Vec<_>>();
    let private_kappa_bad = registry_lying
        .iter()
        .map(|(reply, _, _)| reply.as_slice())
        .collect::<Vec<_>>();
    let public_zero_bad = honest[..kappa]
        .iter()
        .map(|(_, public, _)| public.clone())
        .collect::<Vec<_>>();
    let public_honest_replies =
        honest.iter().map(|(_, public, _)| public.clone()).collect::<Vec<_>>();
    let public_kappa_bad = registry_lying
        .iter()
        .map(|(_, public, _)| public.clone())
        .collect::<Vec<_>>();

    let decode_all = |replies: &[&[u8]]| -> Vec<AuthReadResponse> {
        replies
            .iter()
            .map(|reply| opening.decode(&parse_reply(reply)))
            .collect()
    };
    // A decoded private reply is the public reply of its claim.
    for (decoded, public) in decode_all(&private_kappa_bad).iter().zip(&public_kappa_bad) {
        assert!(same_replies(decoded, public), "a decoded reply is the public reply");
    }

    let processing_client = || {
        let mut client = AuthClient::new(
            NoopTransport,
            params.clone(),
            auth.clone(),
            ORDER,
            common.leaves as usize,
            0usize,
            Vec::new(),
        );
        client.set_trusted_root(root);
        client
    };
    let mut client = processing_client();
    let check = |result: (Value256, paper_a_optimized::AuthPath)| {
        assert_eq!(result.0, expected_value);
        assert_eq!(result.1.root, root);
    };

    let mut request_ns = Vec::with_capacity(common.trials);
    let mut private_zero = Vec::with_capacity(common.trials);
    let mut private_honest = Vec::with_capacity(common.trials);
    let mut private_bad = Vec::with_capacity(common.trials);
    let mut public_zero = Vec::with_capacity(common.trials);
    let mut public_honest = Vec::with_capacity(common.trials);
    let mut public_bad = Vec::with_capacity(common.trials);
    for _ in 0..common.trials {
        let (bytes, ns) = timed(|| {
            let opening = PrivateOpening::with_grouping(
                &params,
                &auth,
                ORDER,
                common.leaves,
                grouping,
                index,
            )
            .expect("opening");
            let message = net::Message::AuthReadPrivate {
                req: request_to_net(&opening.request()),
            };
            encode_binary(&message).expect("serialize the request")
        });
        assert_eq!(bytes.len() as u64, fields.request_bytes, "request bytes");
        request_ns.push(ns);

        let (result, ns) = timed(|| {
            let responses = decode_all(&private_zero_bad);
            client.read_with_path_from_responses(index, &responses)
        });
        private_zero.push(ns);
        check(result.expect("private opening from kappa replies"));

        let (result, ns) = timed(|| {
            let responses = decode_all(&private_honest_replies);
            client.read_with_path_from_responses_correcting(index, &responses, kappa)
        });
        private_honest.push(ns);
        check(result.expect("private opening from 3 kappa honest replies"));

        let (result, ns) = timed(|| {
            let responses = decode_all(&private_kappa_bad);
            client.read_with_path_from_responses_correcting(index, &responses, kappa)
        });
        private_bad.push(ns);
        check(result.expect("private opening with kappa lying replies"));

        let (result, ns) = timed(|| client.read_with_path_from_responses(index, &public_zero_bad));
        public_zero.push(ns);
        check(result.expect("public opening from kappa replies"));

        let (result, ns) = timed(|| {
            client.read_with_path_from_responses_correcting(index, &public_honest_replies, kappa)
        });
        public_honest.push(ns);
        check(result.expect("public opening from 3 kappa honest replies"));

        let (result, ns) = timed(|| {
            client.read_with_path_from_responses_correcting(index, &public_kappa_bad, kappa)
        });
        public_bad.push(ns);
        check(result.expect("public opening with kappa lying replies"));
    }

    vec![BenchEntry::Client(ClientEntry {
        role: "client",
        kappa,
        segments_per_record,
        shape: fields,
        zero_bad_replies: kappa,
        correcting_replies: claims,
        bad_replies: kappa,
        client_request_ns: stats_from_samples(request_ns),
        client_private_zero_bad_ns: stats_from_samples(private_zero),
        client_private_honest_ns: stats_from_samples(private_honest),
        client_private_kappa_bad_ns: stats_from_samples(private_bad),
        client_public_zero_bad_ns: stats_from_samples(public_zero),
        client_public_honest_ns: stats_from_samples(public_honest),
        client_public_kappa_bad_ns: stats_from_samples(public_bad),
    })]
}

fn run_benchmark() {
    let role = std::env::var(format!("{ENV_PREFIX}_ROLE")).unwrap_or_else(|_| "maintainer".to_string());
    let common = Common {
        trials: parse_usize_env("TRIALS", DEFAULT_TRIALS),
        kappa: parse_usize_env("KAPPA", DEFAULT_KAPPA),
        leaves: parse_usize_env("LEAVES", DEFAULT_LEAVES) as u64,
        tree_height: parse_usize_env("TREE_HEIGHT", DEFAULT_TREE_HEIGHT),
        hint_sample_columns: parse_usize_env("HINT_SAMPLE_COLUMNS", DEFAULT_HINT_SAMPLE_COLUMNS),
    };

    let start = Instant::now();
    let cpus = CpuPlan::widen();
    let entries = match role.as_str() {
        "maintainer" => run_maintainer(&common, &cpus),
        "single" => run_single(&common, &cpus),
        "table" => run_table(&common, &cpus),
        "client" => run_client(&common, &cpus),
        other => panic!("{ENV_PREFIX}_ROLE must be maintainer, single, table, or client; got {other}"),
    };
    let elapsed = start.elapsed().as_secs_f64();

    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let results = BenchResults {
        meta: BenchMeta {
            seed: BENCH_SEED,
            trials: common.trials,
            role,
            kappa: common.kappa,
            leaves: common.leaves,
            tree_height: common.tree_height,
            segments_per_record: parse_usize_env("SEGMENTS_PER_RECORD", 1),
            cells_per_record: parse_usize_env("CELLS_PER_RECORD", DEFAULT_CELLS_PER_RECORD),
            append_leaves: parse_usize_env("APPEND_LEAVES", DEFAULT_APPEND_LEAVES) as u64,
            hint_sample_columns: common.hint_sample_columns,
            pir_path: if cfg!(target_feature = "avx512f") {
                "avx512f"
            } else if cfg!(target_arch = "x86_64") && std::arch::is_x86_feature_detected!("avx2") {
                "avx2"
            } else {
                "scalar"
            },
            setup_threads: cpus.setup_threads,
            setup_cpus: cpus.setup.clone(),
            timed_cpus: cpus.timed.clone(),
            setup_seconds: elapsed,
            peak_rss_bytes: peak_rss_bytes(),
            timestamp_unix_ms: timestamp_ms,
            git: git_sha(),
            notes: "private openings of one position: one server's first pass and answer to a serialized request, the client's request and decoding, the hint of a sample of database columns, the response-packing tables, and the preprocessing update after appends; or one client's Combine and Verify from private and from public replies".to_string(),
        },
        entries,
    };

    let out_dir = Path::new("results");
    let _ = fs::create_dir_all(out_dir);
    let path = out_dir.join("bench_private_opening_local.json");
    let data = serde_json::to_vec_pretty(&results).expect("serialize bench_private_opening_local");
    fs::write(path, data).expect("write bench_private_opening_local.json");
}

fn main() {
    run_benchmark();
}

#[test]
#[ignore]
fn bench_private_opening_local() {
    run_benchmark();
}
