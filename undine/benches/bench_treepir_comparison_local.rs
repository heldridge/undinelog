//! UndineLog against TreePIR at `h = kappa`, both over the same YPIR backend.
//!
//! One process measures one configuration, chosen by `ROLE`:
//! - `server`: the preparation and the answer of one UndineLog maintainer
//!   (claim `MAINTAINER`) and one TreePIR server (color `COLOR`), alternated
//!   trial by trial. The maintainer's fragment is built once per process
//!   and kept in a temporary file; every trial reloads it untimed and
//!   rebuilds both PIR servers from the start: UndineLog gathers its primary
//!   columns into PIR cells, reads the row table and the upper nodes, and
//!   builds the hint and the packing tables; TreePIR enumerates its class,
//!   writes its database columns, and builds the hint and the packing
//!   tables. One fresh query is answered per rebuild. Both servers use the client's [`SuffixTree`]: an allocated,
//!   unwritten prefix followed by seeded leaves, with exactly `2^h` leaves.
//!   UndineLog uses `BulkLog::build_with_default_prefix` and checks its root
//!   against that tree. The PIR stages run on the calling CPU; fragment
//!   construction, including hashing the seeded suffix, runs once on
//!   `SETUP_THREADS` threads over all online CPUs; its wall time is recorded
//!   in the result, outside the timed series.
//! - `client`: the client's work for one opening in each system, alternated
//!   trial by trial: preparing the requests (UndineLog one request for its
//!   `kappa` maintainers, TreePIR `h` requests), then processing the replies
//!   into a verified opening (UndineLog decrypts `kappa` replies, combines
//!   their shares and verifies the path; TreePIR decrypts `h` replies and
//!   verifies the path). The tree has `2^h - SEEDED_LEAVES` allocated,
//!   unwritten leaves followed by `SEEDED_LEAVES` seeded ones, and every
//!   opening is of a seeded leaf. With `REPLIES=real` every reply comes from
//!   a real server of each system. With `REPLIES=generated` every reply is
//!   the row its server would return, read from the tree and encrypted under
//!   the client's key; the process then also builds one real server of each
//!   system and times its replies and the generated ones in pairs.
//!
//! Each component timing is its own measurement of the same opening, and
//! each total is timed directly on a second opening.

use paper_a_optimized::auth::client::{AuthClient, AuthTransport, ClientError as AuthClientError};
use paper_a_optimized::auth::mapping::{self, NodeLinearization};
use paper_a_optimized::auth::service::{
    AuthBatchWriteRequest, AuthBatchWriteResponse, AuthReadResponse, AuthSubtreeReadResponse,
    AuthWriteRequest, AuthWriteResponse,
};
use paper_a_optimized::merkle::recompute_root;
use paper_a_optimized::net;
use paper_a_optimized::net::wire::{
    AuthPrivateReadReply, AuthPrivateReadRequest, AuthPrivateReadResponse, NetByteBlob,
    decode_binary, encode_binary,
};
use paper_a_optimized::private::bulk::CHUNK_BITS;
use paper_a_optimized::private::pir::{bytes_from_cells, bytes_to_cells};
use paper_a_optimized::private::treepir::{
    BalancedColoring, NodeLocation, TreePirLayout, TreePirQuery, TreePirRequest, TreePirServer,
    root_from_labels,
};
use paper_a_optimized::private::wire::{WORD_BYTES, pack_words, response_from_net, unpack_words};
use paper_a_optimized::private::{
    BulkLog, PackKeys, PirClient, PirDatabase, PirShape, PlannedEntry, PrivateLayout,
    PrivateOpening, SuffixTree, entries_to_cells, entry_from_cells,
    suffix_segment_entries,
};
use paper_a_optimized::{
    AuthParams, CodedArrayConfig, CodedArrayParams, Fp, ShardShare, Value256, build_coded_params,
    build_shard_registry_deterministic, decode_value, encode_value, fp_zero,
};
use serde::Serialize;
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const ENV_PREFIX: &str = "BENCH_TREEPIR_COMPARISON";
const BENCH_SEED: u64 = 42;
const DEFAULT_TRIALS: usize = 3;
const DEFAULT_HEIGHT: usize = 8;
const DEFAULT_SEEDED_LEAVES: u64 = 1 << 20;
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

fn percentile(sorted: &[u64], pct: usize) -> u64 {
    sorted[(pct * (sorted.len() - 1)) / 100]
}

fn stats(samples: Vec<u64>) -> MetricStats {
    assert!(!samples.is_empty(), "a metric without samples");
    let sum: u128 = samples.iter().map(|&v| v as u128).sum();
    let mut sorted = samples.clone();
    sorted.sort_unstable();
    MetricStats {
        mean: (sum / samples.len() as u128) as u64,
        p50: percentile(&sorted, 50),
        p95: percentile(&sorted, 95),
        trials: samples.len() as u64,
        samples,
    }
}

/// Run `f` once and return its result and duration in nanoseconds.
fn timed<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let start = Instant::now();
    let out = black_box(f());
    (out, (start.elapsed().as_nanos() as u64).max(1))
}

fn env_string(suffix: &str) -> Option<String> {
    std::env::var(format!("{ENV_PREFIX}_{suffix}")).ok()
}

fn env_usize(suffix: &str, default: usize) -> usize {
    env_string(suffix)
        .map(|raw| raw.trim().parse::<usize>().expect("integer benchmark option"))
        .unwrap_or(default)
}

fn git_sha() -> Option<String> {
    let output = Command::new("git").args(["rev-parse", "HEAD"]).output().ok()?;
    let sha = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (output.status.success() && !sha.is_empty()).then_some(sha)
}

fn thread_status(label: &str) -> Option<String> {
    let status = fs::read_to_string("/proc/thread-self/status").ok()?;
    status
        .lines()
        .find_map(|line| line.strip_prefix(label))
        .map(|rest| rest.trim().to_string())
}

fn allowed_cpus() -> Option<String> {
    thread_status("Cpus_allowed_list:")
}

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

/// The CPU set of the timed operations and the set of the setup, which
/// spans every online CPU.
struct CpuPlan {
    timed: Option<String>,
    setup: Option<String>,
    setup_threads: usize,
}

impl CpuPlan {
    fn new() -> Self {
        let timed = allowed_cpus();
        let online = fs::read_to_string("/sys/devices/system/cpu/online")
            .ok()
            .map(|raw| raw.trim().to_string());
        let setup = match (&timed, online) {
            (Some(timed), Some(online)) => {
                if set_thread_cpus(&online) {
                    let setup = allowed_cpus();
                    assert!(set_thread_cpus(timed), "return to the timed CPU set");
                    setup
                } else {
                    Some(timed.clone())
                }
            }
            _ => timed.clone(),
        };
        let available = setup
            .as_deref()
            .map(cpu_count)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
        CpuPlan {
            timed,
            setup,
            setup_threads: env_usize("SETUP_THREADS", available).max(1),
        }
    }

    /// Run `work` on the setup CPUs, then return to the timed CPU set.
    fn on_setup_cpus<T>(&self, work: impl FnOnce() -> T) -> T {
        if let (Some(timed), Some(setup)) = (&self.timed, &self.setup) {
            if timed != setup {
                assert!(set_thread_cpus(setup), "widen to the setup CPUs");
                let result = work();
                assert!(set_thread_cpus(timed), "return to the timed CPU set");
                assert_eq!(allowed_cpus().as_ref(), Some(timed), "timed CPU set");
                return result;
            }
        }
        work()
    }
}

/// CPUs of a kernel CPU list such as `0-3,8`.
fn cpu_count(list: &str) -> usize {
    list.split(',')
        .filter(|part| !part.is_empty())
        .map(|part| match part.split_once('-') {
            Some((low, high)) => {
                high.trim().parse::<usize>().unwrap() - low.trim().parse::<usize>().unwrap() + 1
            }
            None => 1,
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Parameters.

struct Common {
    trials: usize,
    height: u8,
    kappa: usize,
    leaves: u64,
}

impl Common {
    fn from_env() -> Self {
        let height = env_usize("HEIGHT", DEFAULT_HEIGHT);
        assert!(
            (2..=62).contains(&height) && height.is_power_of_two(),
            "h = kappa needs a power-of-two height"
        );
        Common {
            trials: env_usize("TRIALS", DEFAULT_TRIALS).max(1),
            height: height as u8,
            kappa: height,
            leaves: 1u64 << height,
        }
    }

    fn auth(&self) -> AuthParams {
        AuthParams::new(usize::from(self.height))
    }

    fn seeded_leaves(&self) -> u64 {
        let seeded = (env_usize("SEEDED_LEAVES", DEFAULT_SEEDED_LEAVES as usize) as u64)
            .min(self.leaves);
        assert!(seeded > 0, "at least one seeded leaf");
        seeded
    }

    /// Exactly `kappa` claims: every reply is needed and none is faulty.
    fn params(&self) -> CodedArrayParams {
        build_coded_params(&CodedArrayConfig {
            kappa: self.kappa,
            shard_count: self.kappa,
            shard_registry: build_shard_registry_deterministic(self.kappa),
            party_registry_indices: (0..self.kappa).map(|idx| vec![idx]).collect(),
            seed: BENCH_SEED,
        })
    }
}

/// Leaf `trial` of `count` leaves starting at `first`, spread over them.
fn spread_leaf(first: u64, count: u64, trial: usize) -> u64 {
    first + (trial as u64).wrapping_mul(2_654_435_761).wrapping_add(count / 3) % count
}

/// Serialized sizes of one request: the blob of the query, the blob of the
/// packing keys, and the whole message.
#[derive(Clone, Copy, Debug, Serialize)]
struct RequestBytes {
    request_bytes: u64,
    query_bytes: u64,
    key_bytes: u64,
}

fn shape_entry(shape: &PirShape) -> ShapeFields {
    ShapeFields {
        db_rows: shape.db_rows() as u64,
        db_cols: shape.db_cols() as u64,
        record_columns: shape.row_cells() as u64,
        outputs: shape.outputs() as u64,
        ciphertext_payload_bytes: (shape.outputs() * shape.ciphertext_bytes()) as u64,
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
struct ShapeFields {
    db_rows: u64,
    db_cols: u64,
    record_columns: u64,
    outputs: u64,
    /// Ciphertext bytes of one reply, before serialization.
    ciphertext_payload_bytes: u64,
}

// ---------------------------------------------------------------------------
// UndineLog on the wire.

fn undine_request_message(layout: &PrivateLayout, query: &[u64], keys: &PackKeys) -> Vec<u8> {
    encode_binary(&net::Message::AuthReadPrivate {
        req: AuthPrivateReadRequest {
            leaves: layout.leaves(),
            kappa: layout.kappa() as u64,
            pattern_bits: u64::from(layout.group_bits()),
            query: NetByteBlob(pack_words(query)),
            pack_keys: NetByteBlob(pack_words(&keys.words)),
        },
    })
    .expect("serialize the request")
}

fn undine_reply_message(registry_index: usize, ciphertexts: Vec<Vec<u8>>) -> Vec<u8> {
    encode_binary(&net::Response::AuthReadPrivate {
        resp: AuthPrivateReadResponse {
            replies: vec![AuthPrivateReadReply {
                registry_index: registry_index as u32,
                ciphertexts: ciphertexts.into_iter().map(NetByteBlob).collect(),
            }],
        },
    })
    .expect("serialize the reply")
}

fn undine_serve(database: &PirDatabase, registry_index: usize, request: &[u8]) -> Vec<u8> {
    let net::Message::AuthReadPrivate { req } =
        decode_binary::<net::Message>(request).expect("parse the request")
    else {
        panic!("the request is a private opening");
    };
    let query = unpack_words(&req.query.0).expect("query words");
    let keys = PackKeys {
        words: unpack_words(&req.pack_keys.0).expect("packing-key words"),
    };
    undine_reply_message(registry_index, database.answer(&query, &keys).expect("answer"))
}

/// Parse one maintainer's reply and decode it into the shares of the path.
fn undine_decode(opening: &PrivateOpening, reply: &[u8]) -> AuthReadResponse {
    let net::Response::AuthReadPrivate { resp } =
        decode_binary::<net::Response>(reply).expect("parse the reply")
    else {
        panic!("the reply is a private opening");
    };
    opening.decode(&response_from_net(resp))
}

/// The slots of a decoded reply, for exact comparisons of two replies.
fn undine_slots(opening: &PrivateOpening, reply: &[u8]) -> Vec<Vec<u64>> {
    let net::Response::AuthReadPrivate { resp } =
        decode_binary::<net::Response>(reply).expect("parse the reply")
    else {
        panic!("the reply is a private opening");
    };
    let response = response_from_net(resp);
    opening.client().decode(&response.replies[0].response).expect("decode the reply")
}

fn request_bytes_of(query_words: usize, key_words: usize, request: &[u8]) -> RequestBytes {
    RequestBytes {
        request_bytes: request.len() as u64,
        query_bytes: (query_words * WORD_BYTES) as u64,
        key_bytes: (key_words * WORD_BYTES) as u64,
    }
}

// ---------------------------------------------------------------------------
// The server role.

#[derive(Clone, Debug, Serialize)]
struct UndineServerEntry {
    role: &'static str,
    system: &'static str,
    height: u8,
    kappa: usize,
    leaves: u64,
    log_bytes: u64,
    seeded_leaves: u64,
    tree_root: Value256,
    registry_index: usize,
    #[serde(flatten)]
    shape: ShapeFields,
    pattern_bits: u64,
    /// Bytes of the claim's primary columns: its PIR database before
    /// conversion, 32 per entry.
    database_bytes: u64,
    /// Bytes of the claim's coded fragment: the primary columns and the
    /// equally sized auxiliary rows that repair uses.
    fragment_bytes: u64,
    /// Bytes of the PIR cells, the upper nodes' cells, and the row table.
    pir_bytes: u64,
    /// Bytes of the hint and the response-packing tables.
    preprocessed_bytes: u64,
    /// Bytes of the hint.
    hint_bytes: u64,
    /// Bytes of the response-packing tables.
    packing_bytes: u64,
    /// `pir_bytes + preprocessed_bytes`.
    serving_state_bytes: u64,
    #[serde(flatten)]
    blocks: HintBlocks,
    #[serde(flatten)]
    request: RequestBytes,
    reply_bytes: u64,
    encoding_threads: usize,
    /// Building the claim's fragment once per process, including encoding
    /// its primary columns and hashing the seeded suffix, on
    /// `encoding_threads` threads (wall seconds). Every repetition starts
    /// from this fragment, reloaded untimed.
    fragment_build_seconds: f64,
    /// The primary columns into PIR cells, record by record.
    gather_cells_ns: MetricStats,
    /// The row table and the upper nodes' cells.
    gather_upper_ns: MetricStats,
    /// The hint, which gathers its columns from the cells as it goes.
    hint_ns: MetricStats,
    /// The response-packing tables from the hint.
    packing_ns: MetricStats,
    /// The four stages above, per rebuild.
    prepare_ns: MetricStats,
    /// The query's inner product with every database column.
    first_pass_ns: MetricStats,
    /// From the serialized request to the serialized reply.
    answer_ns: MetricStats,
}

#[derive(Clone, Debug, Serialize)]
struct TreePirServerEntry {
    role: &'static str,
    system: &'static str,
    height: u8,
    leaves: u64,
    log_bytes: u64,
    seeded_leaves: u64,
    tree_root: Value256,
    color: usize,
    class_nodes: u64,
    nodes_per_record: u64,
    #[serde(flatten)]
    shape: ShapeFields,
    /// Bytes of the class's labels, 32 per node.
    database_bytes: u64,
    /// Bytes of the padded 14-bit-symbol database.
    pir_bytes: u64,
    preprocessed_bytes: u64,
    hint_bytes: u64,
    packing_bytes: u64,
    serving_state_bytes: u64,
    #[serde(flatten)]
    blocks: HintBlocks,
    #[serde(flatten)]
    request: RequestBytes,
    reply_bytes: u64,
    /// The class's nodes in database order.
    enumerate_ns: MetricStats,
    /// The database columns from the class's labels.
    columns_ns: MetricStats,
    hint_ns: MetricStats,
    packing_ns: MetricStats,
    /// The four stages above, per rebuild.
    prepare_ns: MetricStats,
    first_pass_ns: MetricStats,
    answer_ns: MetricStats,
}

/// The hint's blocks of `poly_len` rows of one column, and those that are
/// all zero, for which the hint's kernel does no arithmetic.
#[derive(Clone, Copy, Debug, Serialize)]
struct HintBlocks {
    hint_blocks: u64,
    zero_hint_blocks: u64,
    /// The blocks of the padding columns after the record columns, which
    /// are zero in every database.
    padding_hint_blocks: u64,
}

impl HintBlocks {
    fn of(shape: &PirShape, (zero, all): (u64, u64)) -> Self {
        let per_column = all / shape.db_cols() as u64;
        HintBlocks {
            hint_blocks: all,
            zero_hint_blocks: zero,
            padding_hint_blocks: (shape.db_cols() - shape.row_cells()) as u64 * per_column,
        }
    }
}

/// A fragment's primary-column entries in a temporary file, 32 bytes each,
/// removed when dropped.
struct FragmentFile {
    path: std::path::PathBuf,
    entries: usize,
}

impl FragmentFile {
    fn write(entries: &[Fp]) -> Self {
        use std::io::Write;
        let path = std::env::temp_dir().join(format!(
            "bench-treepir-comparison-fragment-{}.bin",
            std::process::id()
        ));
        let mut out = std::io::BufWriter::with_capacity(
            1 << 20,
            fs::File::create(&path).expect("create the fragment file"),
        );
        for entry in entries {
            out.write_all(&encode_value(*entry)).expect("write the fragment");
        }
        out.flush().expect("write the fragment");
        Self {
            path,
            entries: entries.len(),
        }
    }

    fn read(&self) -> Vec<Fp> {
        use std::io::Read;
        let mut input = std::io::BufReader::with_capacity(
            1 << 20,
            fs::File::open(&self.path).expect("open the fragment file"),
        );
        let mut entries = Vec::with_capacity(self.entries);
        let mut bytes = [0u8; 32];
        for _ in 0..self.entries {
            input.read_exact(&mut bytes).expect("read the fragment");
            entries.push(decode_value(bytes));
        }
        entries
    }
}

impl Drop for FragmentFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Default)]
struct Samples(HashMap<&'static str, Vec<u64>>);

impl Samples {
    fn push(&mut self, name: &'static str, value: u64) {
        self.0.entry(name).or_default().push(value);
    }

    fn take(&mut self, name: &'static str) -> MetricStats {
        stats(self.0.remove(name).unwrap_or_else(|| panic!("no samples of {name}")))
    }
}

struct ServerSizes {
    database_bytes: u64,
    pir_bytes: u64,
    hint_bytes: u64,
    packing_bytes: u64,
    blocks: Option<HintBlocks>,
    request: RequestBytes,
    reply_bytes: u64,
}

fn run_server(common: &Common, cpus: &CpuPlan) -> Vec<serde_json::Value> {
    let registry_index = env_usize("MAINTAINER", 0);
    let color = env_usize("COLOR", 0);
    assert!(registry_index < common.kappa, "MAINTAINER is a claim of the threshold");
    assert!(color < usize::from(common.height), "COLOR is a class of the tree");
    let params = common.params();
    let auth = common.auth();
    let layout = PrivateLayout::for_log(&params, &auth, ORDER, common.leaves).expect("layout");
    let shape = PirShape::of(&layout);
    let seeded = common.seeded_leaves();
    let default_leaves = common.leaves - seeded;
    let tree = cpus.on_setup_cpus(|| SuffixTree::new(&auth, default_leaves, BENCH_SEED));
    let coloring = BalancedColoring::new(common.height);
    let tree_layout = TreePirLayout::new(common.height);
    let class_size = coloring.class_sizes()[color];

    let mut undine = Samples::default();
    let mut treepir = Samples::default();
    let mut undine_sizes = None;
    let mut treepir_sizes = None;

    // The maintainer's fragment is built once. It is kept on disk between
    // repetitions, so that neither server's preparation shares the memory
    // with it, and every repetition reloads it untimed.
    let (bulk, fragment_build_ns) = cpus.on_setup_cpus(|| {
        timed(|| {
            BulkLog::build_with_default_prefix(
                &params,
                &auth,
                ORDER,
                common.leaves,
                default_leaves,
                BENCH_SEED,
                registry_index,
                cpus.setup_threads,
                CHUNK_BITS,
            )
        })
    });
    assert_eq!(bulk.root(), tree.root(), "the maintainer and TreePIR tree roots");
    let fragment = FragmentFile::write(bulk.entries());
    {
        // The reloaded fragment is the built one. Both are held here, before
        // any PIR state exists, which stays below the later peaks.
        let reloaded = fragment.read();
        assert!(reloaded == bulk.entries(), "the reloaded fragment");
    }
    drop(bulk);

    let undine_trial = |trial: usize, samples: &mut Samples| -> ServerSizes {
        let entries = fragment.read();
        let leaf = spread_leaf(default_leaves, seeded, trial);
        let plan = layout.opening_plan(leaf);
        let expected = plan
            .iter()
            .map(|entry| entries[layout.entry_index(entry.array_index)])
            .collect::<Vec<_>>();
        let database_bytes = entries.len() as u64 * FIELD_BYTES;
        let (cells, gather_cells_ns) = timed(|| PirDatabase::cells_from_entries(&layout, &entries));
        drop(entries);
        let (database, times) = PirDatabase::from_cells_timed(&layout, cells, 1);
        let stages = [
            gather_cells_ns,
            times.gather.as_nanos() as u64,
            times.hint.as_nanos() as u64,
            times.packing.as_nanos() as u64,
        ];
        samples.push("gather_cells", stages[0]);
        samples.push("gather_upper", stages[1].max(1));
        samples.push("hint", stages[2]);
        samples.push("packing", stages[3]);
        samples.push("prepare", stages.iter().sum());

        let client = PirClient::generate(shape);
        let query = client.query(layout.pattern_of(leaf));
        let keys = client.pack_keys();
        let request = undine_request_message(&layout, &query, &keys);
        let (_, first_pass_ns) = timed(|| database.first_pass(&query).expect("first pass"));
        samples.push("first_pass", first_pass_ns);
        let (reply, answer_ns) = timed(|| undine_serve(&database, registry_index, &request));
        samples.push("answer", answer_ns);
        let net::Response::AuthReadPrivate { resp } =
            decode_binary::<net::Response>(&reply).expect("parse the reply")
        else {
            panic!("the reply is a private opening");
        };
        let slots = client
            .decode(&response_from_net(resp).replies[0].response)
            .expect("decode the reply");
        let decoded = plan
            .iter()
            .map(|entry| entry_from_cells(&slots[entry.slot], entry.entry_in_record))
            .collect::<Vec<_>>();
        assert!(decoded == expected, "trial {trial}: the opening of leaf {leaf}");
        // Counted once, after every timer of the trial.
        let blocks = (trial == 0).then(|| HintBlocks::of(&shape, database.zero_hint_blocks()));
        ServerSizes {
            database_bytes,
            pir_bytes: database.record_state_bytes() as u64,
            hint_bytes: database.hint_bytes() as u64,
            packing_bytes: database.packing_bytes() as u64,
            blocks,
            request: request_bytes_of(query.len(), keys.words.len(), &request),
            reply_bytes: reply.len() as u64,
        }
    };

    let treepir_trial = |trial: usize, samples: &mut Samples| -> ServerSizes {
        let (nodes, enumerate_ns) = timed(|| coloring.class_nodes(color));
        samples.push("enumerate", enumerate_ns);
        let (server, columns, times) =
            TreePirServer::new(tree_layout, color, &nodes, |node| tree.label_of_heap(node));
        drop(nodes);
        let stages = [
            enumerate_ns,
            columns.as_nanos() as u64,
            times.hint.as_nanos() as u64,
            times.packing.as_nanos() as u64,
        ];
        samples.push("columns", stages[1]);
        samples.push("hint", stages[2]);
        samples.push("packing", stages[3]);
        samples.push("prepare", stages.iter().sum());

        // A leaf whose path has a node of this class.
        let leaf = spread_leaf(default_leaves, seeded, trial);
        let location = *coloring
            .locate(leaf)
            .iter()
            .find(|item| item.color == color)
            .expect("every path has one node of every class");
        let (query, request) = tree_layout.request(location).expect("request");
        let parsed: TreePirRequest = decode_binary(&request).expect("parse the request");
        let words = unpack_words(&parsed.query.0).expect("query words");
        let key_words = unpack_words(&parsed.pack_keys.0).expect("key words").len();
        let (_, first_pass_ns) = timed(|| server.database.first_pass(&words).expect("first pass"));
        samples.push("first_pass", first_pass_ns);
        let (reply, answer_ns) = timed(|| server.answer(&request).expect("answer"));
        samples.push("answer", answer_ns);
        assert_eq!(
            query.decode(&reply).expect("decode"),
            tree.label_of_heap(location.sibling()),
            "trial {trial}: the node of leaf {leaf}"
        );
        let blocks = (trial == 0)
            .then(|| HintBlocks::of(&tree_layout.shape, server.database.zero_hint_blocks()));
        ServerSizes {
            database_bytes: class_size * FIELD_BYTES,
            pir_bytes: server.database.database_bytes() as u64,
            hint_bytes: server.database.hint_bytes() as u64,
            packing_bytes: server.database.packing_bytes() as u64,
            blocks,
            request: request_bytes_of(words.len(), key_words, &request),
            reply_bytes: reply.len() as u64,
        }
    };

    for trial in 0..common.trials {
        // The order alternates, so that a drift of the machine affects both.
        for system in if trial % 2 == 0 { [0, 1] } else { [1, 0] } {
            if system == 0 {
                let sizes = undine_trial(trial, &mut undine);
                undine_sizes.get_or_insert(sizes);
            } else {
                let sizes = treepir_trial(trial, &mut treepir);
                treepir_sizes.get_or_insert(sizes);
            }
        }
        eprintln!("trial {}/{} complete", trial + 1, common.trials);
    }

    let sizes = undine_sizes.expect("one trial");
    let undine_entry = UndineServerEntry {
        role: "server",
        system: "undine",
        height: common.height,
        kappa: common.kappa,
        leaves: common.leaves,
        log_bytes: common.leaves * FIELD_BYTES,
        seeded_leaves: seeded,
        tree_root: tree.root(),
        registry_index,
        shape: shape_entry(&shape),
        pattern_bits: u64::from(layout.group_bits()),
        database_bytes: sizes.database_bytes,
        fragment_bytes: 2 * sizes.database_bytes,
        pir_bytes: sizes.pir_bytes,
        preprocessed_bytes: sizes.hint_bytes + sizes.packing_bytes,
        hint_bytes: sizes.hint_bytes,
        packing_bytes: sizes.packing_bytes,
        serving_state_bytes: sizes.pir_bytes + sizes.hint_bytes + sizes.packing_bytes,
        blocks: sizes.blocks.expect("counted in the first trial"),
        request: sizes.request,
        reply_bytes: sizes.reply_bytes,
        encoding_threads: cpus.setup_threads,
        fragment_build_seconds: fragment_build_ns as f64 / 1e9,
        gather_cells_ns: undine.take("gather_cells"),
        gather_upper_ns: undine.take("gather_upper"),
        hint_ns: undine.take("hint"),
        packing_ns: undine.take("packing"),
        prepare_ns: undine.take("prepare"),
        first_pass_ns: undine.take("first_pass"),
        answer_ns: undine.take("answer"),
    };
    let sizes = treepir_sizes.expect("one trial");
    let treepir_entry = TreePirServerEntry {
        role: "server",
        system: "treepir",
        height: common.height,
        leaves: common.leaves,
        log_bytes: common.leaves * FIELD_BYTES,
        seeded_leaves: seeded,
        tree_root: tree.root(),
        color,
        class_nodes: class_size,
        nodes_per_record: tree_layout.nodes_per_record as u64,
        shape: shape_entry(&tree_layout.shape),
        database_bytes: sizes.database_bytes,
        pir_bytes: sizes.pir_bytes,
        preprocessed_bytes: sizes.hint_bytes + sizes.packing_bytes,
        hint_bytes: sizes.hint_bytes,
        packing_bytes: sizes.packing_bytes,
        serving_state_bytes: sizes.pir_bytes + sizes.hint_bytes + sizes.packing_bytes,
        blocks: sizes.blocks.expect("counted in the first trial"),
        request: sizes.request,
        reply_bytes: sizes.reply_bytes,
        enumerate_ns: treepir.take("enumerate"),
        columns_ns: treepir.take("columns"),
        hint_ns: treepir.take("hint"),
        packing_ns: treepir.take("packing"),
        prepare_ns: treepir.take("prepare"),
        first_pass_ns: treepir.take("first_pass"),
        answer_ns: treepir.take("answer"),
    };
    vec![
        serde_json::to_value(undine_entry).unwrap(),
        serde_json::to_value(treepir_entry).unwrap(),
    ]
}

// ---------------------------------------------------------------------------
// The client role.

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

/// The records of every claim that the openings of a set of leaves read,
/// encoded from the tree without the claims' columns.
struct ClaimRecords {
    per_record: usize,
    /// `records[record][claim]`: the claim's entries of the record.
    records: HashMap<usize, Vec<Vec<Fp>>>,
}

impl ClaimRecords {
    fn new(
        params: &CodedArrayParams,
        tree: &SuffixTree<'_>,
        layout: &PrivateLayout,
        claims: &[usize],
        leaves: impl IntoIterator<Item = u64>,
    ) -> Self {
        let per_record = layout.entries_per_record();
        let kappa = layout.kappa();
        let shape = PirShape::of(layout);
        let mut wanted = BTreeSet::new();
        for leaf in leaves {
            for (slot, id) in layout.row(layout.pattern_of(leaf)).into_iter().enumerate() {
                match id {
                    Some(record) if slot < shape.group_slots => {
                        wanted.insert(record);
                    }
                    Some(pos) => {
                        let index = layout.upper_entry_index(slot - shape.group_slots, pos);
                        wanted.insert(index / per_record);
                    }
                    None => {}
                }
            }
        }
        let height = tree.height() as u16;
        let segments_total = mapping::cursor(ORDER, height, 1u64 << height)
            .div_ceil(params.n_grid as u64) as usize;
        let segments_per_record = layout.segments_per_record();
        let segments = wanted
            .iter()
            .flat_map(|&record| record * segments_per_record..(record + 1) * segments_per_record)
            .filter(|&segment| segment < segments_total)
            .collect::<Vec<_>>();
        let encoded = suffix_segment_entries(params, tree, claims, &segments);
        let by_segment = segments.iter().copied().zip(encoded).collect::<HashMap<_, _>>();
        let records = wanted
            .iter()
            .map(|&record| {
                let entries = (0..claims.len())
                    .map(|claim| {
                        let mut values = vec![fp_zero(); per_record];
                        for k in 0..segments_per_record {
                            if let Some(segment) = by_segment.get(&(record * segments_per_record + k)) {
                                values[k * kappa..(k + 1) * kappa].copy_from_slice(&segment[claim]);
                            }
                        }
                        values
                    })
                    .collect();
                (record, entries)
            })
            .collect();
        Self { per_record, records }
    }

    fn record_cells(&self, shape: &PirShape, claim: usize, record: usize) -> Vec<u16> {
        let mut cells = vec![0u16; shape.record_cells];
        entries_to_cells(&self.records[&record][claim], &mut cells);
        cells
    }

    /// The row the query of `pattern` selects from claim `claim`: the
    /// record of each group slot and the entry of each upper slot, as
    /// `PirDatabase` lays them out.
    fn slots(&self, layout: &PrivateLayout, claim: usize, pattern: usize) -> Vec<Vec<u64>> {
        let shape = PirShape::of(layout);
        layout
            .row(pattern)
            .iter()
            .enumerate()
            .map(|(slot, id)| {
                let mut cells = vec![0u16; shape.slot_width(slot)];
                match id {
                    Some(record) if slot < shape.group_slots => {
                        cells = self.record_cells(&shape, claim, *record);
                    }
                    Some(pos) => {
                        let index = layout.upper_entry_index(slot - shape.group_slots, *pos);
                        let block = self
                            .record_cells(&shape, claim, index / self.per_record)
                            .into_iter()
                            .map(u64::from)
                            .collect::<Vec<_>>();
                        bytes_to_cells(bytes_from_cells(&block, index % self.per_record), &mut cells);
                    }
                    None => {}
                }
                assert_eq!(cells.len(), shape.slot_width(slot), "cells of slot {slot}");
                cells.into_iter().map(u64::from).collect()
            })
            .collect()
    }
}

/// The servers whose replies a client job uses.
enum Replies {
    /// Every reply from a real server of each system.
    Real {
        maintainers: Vec<PirDatabase>,
        colors: Vec<TreePirServer>,
    },
    /// Every reply generated from the tree.
    Generated { records: ClaimRecords },
}

struct ClientSetup<'a> {
    common: &'a Common,
    params: &'a CodedArrayParams,
    auth: &'a AuthParams,
    tree: &'a SuffixTree<'a>,
    layout: &'a PrivateLayout,
    shape: PirShape,
    coloring: &'a BalancedColoring,
    tree_layout: TreePirLayout,
    class_sizes: Vec<u64>,
}

impl ClientSetup<'_> {
    /// The cells of the TreePIR row that holds `location`.
    fn treepir_row(&self, location: NodeLocation) -> Vec<u16> {
        let row = self.tree_layout.row_of(location.offset);
        let offsets = self.tree_layout.row_offsets(row, self.class_sizes[location.color]);
        let nodes = self.coloring.class_range(location.color, offsets);
        let mut cells = vec![0u16; self.tree_layout.shape.record_cells];
        self.tree_layout
            .row_cells(&nodes, |node| self.tree.label_of_heap(node), &mut cells);
        cells
    }

    fn undine_generated_reply(&self, records: &ClaimRecords, opening: &PrivateOpening, leaf: u64, claim: usize) -> Vec<u8> {
        let slots = records.slots(self.layout, claim, self.layout.pattern_of(leaf));
        undine_reply_message(claim, opening.encrypt_slots(&slots))
    }

    fn undine_replies(&self, replies: &Replies, opening: &PrivateOpening, request: &[u8], leaf: u64) -> Vec<Vec<u8>> {
        match replies {
            Replies::Real { maintainers, .. } => maintainers
                .iter()
                .enumerate()
                .map(|(claim, database)| undine_serve(database, claim, request))
                .collect(),
            Replies::Generated { records, .. } => (0..self.common.kappa)
                .map(|claim| self.undine_generated_reply(records, opening, leaf, claim))
                .collect(),
        }
    }

    fn treepir_replies(&self, replies: &Replies, queries: &[TreePirQuery], requests: &[Vec<u8>]) -> Vec<Vec<u8>> {
        match replies {
            Replies::Real { colors, .. } => queries
                .iter()
                .zip(requests)
                .map(|(query, request)| colors[query.location.color].answer(request).expect("answer"))
                .collect(),
            Replies::Generated { .. } => queries
                .iter()
                .map(|query| query.encrypt_row(&self.treepir_row(query.location)).expect("reply"))
                .collect(),
        }
    }

    fn auth_client(&self) -> AuthClient<NoopTransport> {
        let mut client = AuthClient::new(
            NoopTransport,
            self.params.clone(),
            self.auth.clone(),
            ORDER,
            self.common.leaves as usize,
            0usize,
            Vec::new(),
        );
        client.set_trusted_root(self.tree.root());
        client
    }

    /// UndineLog's request for `leaf` with a fresh key, in its parts.
    fn undine_request(&self, leaf: u64, samples: Option<&mut Samples>) -> (PrivateOpening, Vec<u8>, RequestBytes) {
        let layout = self.layout;
        let ((pattern, plan), index_ns) = timed(|| (layout.pattern_of(leaf), layout.opening_plan(leaf)));
        let (client, keygen_ns) = timed(|| PirClient::generate(self.shape));
        let ((query, keys), query_ns) = timed(|| (client.query(pattern), client.pack_keys()));
        let (request, serialize_ns) = timed(|| undine_request_message(layout, &query, &keys));
        if let Some(samples) = samples {
            samples.push("index", index_ns);
            samples.push("keygen", keygen_ns);
            samples.push("query", query_ns);
            samples.push("serialize", serialize_ns);
        }
        let bytes = request_bytes_of(query.len(), keys.words.len(), &request);
        let opening = PrivateOpening::from_parts(
            layout.clone(),
            usize::from(self.common.height),
            leaf,
            plan,
            client,
        );
        (opening, request, bytes)
    }

    fn undine_request_direct(&self, leaf: u64) -> ((PrivateOpening, Vec<u8>), u64) {
        let layout = self.layout;
        let (parts, ns) = timed(|| {
            let pattern = layout.pattern_of(leaf);
            let plan: Vec<PlannedEntry> = layout.opening_plan(leaf);
            let client = PirClient::generate(self.shape);
            let query = client.query(pattern);
            let keys = client.pack_keys();
            let request = undine_request_message(layout, &query, &keys);
            (plan, client, request)
        });
        let (plan, client, request) = parts;
        let opening = PrivateOpening::from_parts(
            layout.clone(),
            usize::from(self.common.height),
            leaf,
            plan,
            client,
        );
        ((opening, request), ns)
    }

    fn check_undine(&self, leaf: u64, result: (Value256, paper_a_optimized::AuthPath)) -> (Value256, Vec<Value256>) {
        assert_eq!(result.0, self.tree.leaf_value(leaf), "the value of leaf {leaf}");
        assert_eq!(result.1.root, self.tree.root(), "the root of leaf {leaf}");
        (result.0, result.1.siblings)
    }

    /// TreePIR's `h` requests for `leaf`, with a fresh key each, in parts.
    fn treepir_request(&self, leaf: u64, samples: Option<&mut Samples>) -> (Vec<TreePirQuery>, Vec<Vec<u8>>, RequestBytes) {
        let h = usize::from(self.common.height);
        let (locations, index_ns) = timed(|| self.coloring.locate(leaf));
        let (clients, keygen_ns) =
            timed(|| (0..h).map(|_| PirClient::generate(self.tree_layout.shape)).collect::<Vec<_>>());
        let ((queries, encrypted), query_ns) = timed(|| {
            let queries = locations
                .iter()
                .zip(clients)
                .map(|(&location, client)| self.tree_layout.query_with(location, client).expect("query"))
                .collect::<Vec<_>>();
            let encrypted = queries.iter().map(|query| query.encrypted_query()).collect::<Vec<_>>();
            (queries, encrypted)
        });
        let (requests, serialize_ns) = timed(|| {
            queries
                .iter()
                .zip(&encrypted)
                .map(|(query, (words, keys))| query.serialize_request(words, keys).expect("request"))
                .collect::<Vec<_>>()
        });
        if let Some(samples) = samples {
            samples.push("index", index_ns);
            samples.push("keygen", keygen_ns);
            samples.push("query", query_ns);
            samples.push("serialize", serialize_ns);
        }
        let (words, keys) = &encrypted[0];
        let bytes = request_bytes_of(words.len(), keys.words.len(), &requests[0]);
        assert!(requests.iter().all(|request| request.len() == requests[0].len()));
        (queries, requests, bytes)
    }

    fn treepir_request_direct(&self, leaf: u64) -> ((Vec<TreePirQuery>, Vec<Vec<u8>>), u64) {
        timed(|| {
            let locations = self.coloring.locate(leaf);
            let mut queries = Vec::with_capacity(locations.len());
            let mut requests = Vec::with_capacity(locations.len());
            for location in locations {
                let query = self
                    .tree_layout
                    .query_with(location, PirClient::generate(self.tree_layout.shape))
                    .expect("query");
                let (words, keys) = query.encrypted_query();
                requests.push(query.serialize_request(&words, &keys).expect("request"));
                queries.push(query);
            }
            (queries, requests)
        })
    }

    fn treepir_decrypt(&self, queries: &[TreePirQuery], replies: &[Vec<u8>]) -> Vec<[u8; 32]> {
        queries
            .iter()
            .zip(replies)
            .map(|(query, reply)| query.decode(reply).expect("decode"))
            .collect()
    }

    fn treepir_verify(&self, leaf: u64, labels: &[[u8; 32]]) -> Value256 {
        root_from_labels(
            &self.auth.defaults,
            self.common.height,
            leaf,
            self.tree.leaf_value(leaf),
            labels,
        )
    }
}

#[derive(Clone, Debug, Serialize)]
struct ClientEntry {
    role: &'static str,
    system: &'static str,
    replies_source: &'static str,
    height: u8,
    kappa: usize,
    leaves: u64,
    log_bytes: u64,
    seeded_leaves: u64,
    /// Servers contacted per opening: `kappa` maintainers, or `h` TreePIR
    /// servers.
    servers: u64,
    /// Distinct queries per opening.
    queries: u64,
    #[serde(flatten)]
    shape: ShapeFields,
    #[serde(flatten)]
    request: RequestBytes,
    reply_bytes: u64,
    /// Every transmitted request of one opening.
    upload_bytes: u64,
    /// Every reply of one opening.
    download_bytes: u64,
    /// The pattern and the plan of the opening, or the `h` colors and
    /// offsets of the path.
    index_ns: MetricStats,
    /// Fresh secret keys: one, or one per TreePIR server.
    keygen_ns: MetricStats,
    /// The encrypted queries and their packing keys.
    query_ns: MetricStats,
    /// The serialized requests.
    serialize_ns: MetricStats,
    /// Parsing and decrypting every reply and extracting the opening's
    /// shares or proof nodes.
    decrypt_ns: MetricStats,
    /// UndineLog: combining the shares and verifying the path.
    #[serde(skip_serializing_if = "Option::is_none")]
    combine_verify_ns: Option<MetricStats>,
    /// The root from the recovered value and path.
    verify_ns: MetricStats,
    /// Preparing the requests, timed as a whole on a second opening.
    request_total_ns: MetricStats,
    /// Processing the replies into a verified opening, timed as a whole on
    /// the second opening.
    process_total_ns: MetricStats,
    /// `request_total_ns + process_total_ns` of the second opening.
    client_total_ns: MetricStats,
}

#[derive(Clone, Debug, Serialize)]
struct PairedEntry {
    role: &'static str,
    system: &'static str,
    height: u8,
    /// The claim or the class of the real server.
    server: usize,
    /// Decoding the real server's reply.
    real_decode_ns: MetricStats,
    /// Decoding the generated reply for the same server and query.
    generated_decode_ns: MetricStats,
}

fn run_client(common: &Common, cpus: &CpuPlan) -> Vec<serde_json::Value> {
    let replies_source = env_string("REPLIES").unwrap_or_else(|| "generated".to_string());
    assert!(
        replies_source == "real" || replies_source == "generated",
        "REPLIES is real or generated"
    );
    let seeded = common.seeded_leaves();
    let default_leaves = common.leaves - seeded;
    let params = common.params();
    let auth = common.auth();
    let layout = PrivateLayout::for_log(&params, &auth, ORDER, common.leaves).expect("layout");
    let shape = PirShape::of(&layout);
    let coloring = BalancedColoring::new(common.height);
    let tree_layout = TreePirLayout::new(common.height);
    let claims = (0..common.kappa).collect::<Vec<_>>();
    let (tree, tree_ns) = cpus.on_setup_cpus(|| timed(|| SuffixTree::new(&auth, default_leaves, BENCH_SEED)));
    eprintln!("tree of {seeded} seeded leaves in {:.1} s", tree_ns as f64 / 1e9);
    let setup = ClientSetup {
        common,
        params: &params,
        auth: &auth,
        tree: &tree,
        layout: &layout,
        shape,
        coloring: &coloring,
        tree_layout,
        class_sizes: coloring.class_sizes(),
    };
    // Two openings per trial and system, then one pair per trial and system
    // for the real-reply check.
    let leaf_of = |k: usize| spread_leaf(default_leaves, seeded, k);
    let component_leaf = |trial: usize| leaf_of(4 * trial);
    let direct_leaf = |trial: usize| leaf_of(4 * trial + 1);
    let paired_leaf = |trial: usize| leaf_of(4 * trial + 2);

    let replies = if replies_source == "real" {
        let (maintainers, colors) = cpus.on_setup_cpus(|| {
            let bulk = BulkLog::build_for_claims(
                &params,
                &auth,
                ORDER,
                common.leaves,
                default_leaves,
                BENCH_SEED,
                &claims,
                cpus.setup_threads,
                CHUNK_BITS,
            );
            assert_eq!(bulk.root(), tree.root(), "the bulk log and the tree");
            let maintainers = (0..claims.len())
                .map(|claim| {
                    let cells = PirDatabase::cells_from_entries(&layout, bulk.claim_entries(claim));
                    PirDatabase::from_cells_timed(&layout, cells, cpus.setup_threads).0
                })
                .collect::<Vec<_>>();
            let colors = (0..usize::from(common.height))
                .map(|color| {
                    let nodes = coloring.class_nodes(color);
                    TreePirServer::new(tree_layout, color, &nodes, |node| tree.label_of_heap(node)).0
                })
                .collect::<Vec<_>>();
            (maintainers, colors)
        });
        Replies::Real { maintainers, colors }
    } else {
        let leaves = (0..common.trials)
            .flat_map(|trial| [component_leaf(trial), direct_leaf(trial), paired_leaf(trial)]);
        let (records, ns) = cpus.on_setup_cpus(|| {
            timed(|| ClaimRecords::new(&params, &tree, &layout, &claims, leaves))
        });
        eprintln!("{} records of {} claims in {:.1} s", records.records.len(), claims.len(), ns as f64 / 1e9);
        Replies::Generated { records }
    };
    // With real replies, the generated ones are built as well, to compare.
    let check_records = (replies_source == "real").then(|| {
        ClaimRecords::new(
            &params,
            &tree,
            &layout,
            &claims,
            (0..common.trials).map(component_leaf),
        )
    });

    let mut undine = Samples::default();
    let mut treepir = Samples::default();
    let mut undine_bytes = None;
    let mut treepir_bytes = None;
    let mut undine_reply_bytes = 0;
    let mut treepir_reply_bytes = 0;
    let mut auth_client = setup.auth_client();

    for trial in 0..common.trials {
        for system in if trial % 2 == 0 { [0, 1] } else { [1, 0] } {
            if system == 0 {
                // Components, on one opening.
                let leaf = component_leaf(trial);
                let (opening, request, bytes) = setup.undine_request(leaf, Some(&mut undine));
                undine_bytes.get_or_insert(bytes);
                let replies_now = setup.undine_replies(&replies, &opening, &request, leaf);
                if let Replies::Real { .. } = &replies {
                    // Every real reply decodes to the generated one.
                    let records = check_records.as_ref().expect("records to compare");
                    for (claim, reply) in replies_now.iter().enumerate() {
                        let generated = setup.undine_generated_reply(records, &opening, leaf, claim);
                        assert_eq!(reply.len(), generated.len(), "reply bytes of claim {claim}");
                        assert!(
                            undine_slots(&opening, reply) == undine_slots(&opening, &generated),
                            "trial {trial}: the generated reply of claim {claim}"
                        );
                    }
                }
                undine_reply_bytes = replies_now[0].len() as u64;
                assert!(replies_now.iter().all(|reply| reply.len() as u64 == undine_reply_bytes));
                let (responses, decrypt_ns) = timed(|| {
                    replies_now
                        .iter()
                        .map(|reply| undine_decode(&opening, reply))
                        .collect::<Vec<_>>()
                });
                undine.push("decrypt", decrypt_ns);
                let (result, combine_ns) =
                    timed(|| auth_client.read_with_path_from_responses(leaf, &responses));
                undine.push("combine_verify", combine_ns);
                let (value, siblings) = setup.check_undine(leaf, result.expect("UndineLog opening"));
                let (root, verify_ns) = timed(|| {
                    recompute_root(u32::from(common.height), leaf, value, &siblings, &auth.defaults)
                });
                undine.push("verify", verify_ns);
                assert_eq!(root, tree.root(), "the recomputed root");

                // The whole client, on a second opening.
                let leaf = direct_leaf(trial);
                let ((opening, request), request_ns) = setup.undine_request_direct(leaf);
                let replies_now = setup.undine_replies(&replies, &opening, &request, leaf);
                let (result, process_ns) = timed(|| {
                    let responses = replies_now
                        .iter()
                        .map(|reply| undine_decode(&opening, reply))
                        .collect::<Vec<_>>();
                    auth_client.read_with_path_from_responses(leaf, &responses)
                });
                setup.check_undine(leaf, result.expect("UndineLog opening"));
                undine.push("request_total", request_ns);
                undine.push("process_total", process_ns);
                undine.push("client_total", request_ns + process_ns);
            } else {
                let leaf = component_leaf(trial);
                let (queries, requests, bytes) = setup.treepir_request(leaf, Some(&mut treepir));
                treepir_bytes.get_or_insert(bytes);
                let replies_now = setup.treepir_replies(&replies, &queries, &requests);
                if let Replies::Real { .. } = &replies {
                    for (query, reply) in queries.iter().zip(&replies_now) {
                        let generated = query.encrypt_row(&setup.treepir_row(query.location)).expect("reply");
                        assert_eq!(reply.len(), generated.len(), "reply bytes");
                        assert_eq!(
                            query.decode_row(reply).expect("decode"),
                            query.decode_row(&generated).expect("decode"),
                            "trial {trial}: the generated row of color {}",
                            query.location.color
                        );
                    }
                }
                treepir_reply_bytes = replies_now[0].len() as u64;
                assert!(replies_now.iter().all(|reply| reply.len() as u64 == treepir_reply_bytes));
                let (labels, decrypt_ns) = timed(|| setup.treepir_decrypt(&queries, &replies_now));
                treepir.push("decrypt", decrypt_ns);
                let (root, verify_ns) = timed(|| setup.treepir_verify(leaf, &labels));
                treepir.push("verify", verify_ns);
                assert_eq!(root, tree.root(), "TreePIR proof of leaf {leaf}");

                let leaf = direct_leaf(trial);
                let ((queries, requests), request_ns) = setup.treepir_request_direct(leaf);
                let replies_now = setup.treepir_replies(&replies, &queries, &requests);
                let (root, process_ns) = timed(|| {
                    let labels = setup.treepir_decrypt(&queries, &replies_now);
                    setup.treepir_verify(leaf, &labels)
                });
                assert_eq!(root, tree.root(), "TreePIR proof of leaf {leaf}");
                treepir.push("request_total", request_ns);
                treepir.push("process_total", process_ns);
                treepir.push("client_total", request_ns + process_ns);
            }
        }
        eprintln!("client trial {}/{} complete", trial + 1, common.trials);
    }

    let h = u64::from(common.height);
    let kappa = common.kappa as u64;
    let undine_request = undine_bytes.expect("one trial");
    let treepir_request = treepir_bytes.expect("one trial");
    let mut entries = vec![
        serde_json::to_value(ClientEntry {
            role: "client",
            system: "undine",
            replies_source: if replies_source == "real" { "real" } else { "generated" },
            height: common.height,
            kappa: common.kappa,
            leaves: common.leaves,
            log_bytes: common.leaves * FIELD_BYTES,
            seeded_leaves: seeded,
            servers: kappa,
            queries: 1,
            shape: shape_entry(&shape),
            request: undine_request,
            reply_bytes: undine_reply_bytes,
            upload_bytes: kappa * undine_request.request_bytes,
            download_bytes: kappa * undine_reply_bytes,
            index_ns: undine.take("index"),
            keygen_ns: undine.take("keygen"),
            query_ns: undine.take("query"),
            serialize_ns: undine.take("serialize"),
            decrypt_ns: undine.take("decrypt"),
            combine_verify_ns: Some(undine.take("combine_verify")),
            verify_ns: undine.take("verify"),
            request_total_ns: undine.take("request_total"),
            process_total_ns: undine.take("process_total"),
            client_total_ns: undine.take("client_total"),
        })
        .unwrap(),
        serde_json::to_value(ClientEntry {
            role: "client",
            system: "treepir",
            replies_source: if replies_source == "real" { "real" } else { "generated" },
            height: common.height,
            kappa: common.kappa,
            leaves: common.leaves,
            log_bytes: common.leaves * FIELD_BYTES,
            seeded_leaves: seeded,
            servers: h,
            queries: h,
            shape: shape_entry(&tree_layout.shape),
            request: treepir_request,
            reply_bytes: treepir_reply_bytes,
            upload_bytes: h * treepir_request.request_bytes,
            download_bytes: h * treepir_reply_bytes,
            index_ns: treepir.take("index"),
            keygen_ns: treepir.take("keygen"),
            query_ns: treepir.take("query"),
            serialize_ns: treepir.take("serialize"),
            decrypt_ns: treepir.take("decrypt"),
            combine_verify_ns: None,
            verify_ns: treepir.take("verify"),
            request_total_ns: treepir.take("request_total"),
            process_total_ns: treepir.take("process_total"),
            client_total_ns: treepir.take("client_total"),
        })
        .unwrap(),
    ];

    if let Replies::Generated { records, .. } = &replies {
        drop(auth_client);
        entries.extend(paired_check(&setup, records, cpus, &paired_leaf));
    }
    entries
}

/// One real server of each system, on the opening tree: its reply and the
/// generated reply for the same server and query are decoded in turn, in an
/// order that alternates. The two decode to the same row.
fn paired_check(
    setup: &ClientSetup<'_>,
    records: &ClaimRecords,
    cpus: &CpuPlan,
    paired_leaf: &dyn Fn(usize) -> u64,
) -> Vec<serde_json::Value> {
    let common = setup.common;
    let claim = 0usize;
    let mut real = Vec::with_capacity(common.trials);
    let mut generated = Vec::with_capacity(common.trials);
    {
        let database = cpus.on_setup_cpus(|| {
            let bulk = BulkLog::build_with_default_prefix(
                setup.params,
                setup.auth,
                ORDER,
                common.leaves,
                setup.tree.default_leaves(),
                BENCH_SEED,
                claim,
                cpus.setup_threads,
                CHUNK_BITS,
            );
            assert_eq!(bulk.root(), setup.tree.root(), "the bulk log and the tree");
            let cells = PirDatabase::cells_from_entries(setup.layout, bulk.entries());
            drop(bulk);
            PirDatabase::from_cells_timed(setup.layout, cells, cpus.setup_threads).0
        });
        for trial in 0..common.trials {
            let leaf = paired_leaf(trial);
            let (opening, request, _) = setup.undine_request(leaf, None);
            let real_reply = undine_serve(&database, claim, &request);
            let generated_reply = setup.undine_generated_reply(records, &opening, leaf, claim);
            assert_eq!(real_reply.len(), generated_reply.len(), "reply bytes");
            assert!(
                undine_slots(&opening, &real_reply) == undine_slots(&opening, &generated_reply),
                "trial {trial}: the generated reply of claim {claim}"
            );
            // One untimed decoding of each, so that neither is timed cold.
            black_box(undine_decode(&opening, &real_reply));
            black_box(undine_decode(&opening, &generated_reply));
            for first_real in if trial % 2 == 0 { [true, false] } else { [false, true] } {
                let reply = if first_real { &real_reply } else { &generated_reply };
                let (_, ns) = timed(|| undine_decode(&opening, reply));
                if first_real {
                    real.push(ns);
                } else {
                    generated.push(ns);
                }
            }
        }
    }
    let undine_entry = PairedEntry {
        role: "paired",
        system: "undine",
        height: common.height,
        server: claim,
        real_decode_ns: stats(real),
        generated_decode_ns: stats(generated),
    };

    let color = 0usize;
    let mut real = Vec::with_capacity(common.trials);
    let mut generated = Vec::with_capacity(common.trials);
    {
        let server = cpus.on_setup_cpus(|| {
            let nodes = setup.coloring.class_nodes(color);
            TreePirServer::new(setup.tree_layout, color, &nodes, |node| setup.tree.label_of_heap(node)).0
        });
        for trial in 0..common.trials {
            let leaf = paired_leaf(trial);
            let location = *setup
                .coloring
                .locate(leaf)
                .iter()
                .find(|item| item.color == color)
                .expect("every path has one node of every class");
            let (query, request) = setup.tree_layout.request(location).expect("request");
            let real_reply = server.answer(&request).expect("answer");
            let generated_reply = query.encrypt_row(&setup.treepir_row(location)).expect("reply");
            assert_eq!(real_reply.len(), generated_reply.len(), "reply bytes");
            assert_eq!(
                query.decode_row(&real_reply).expect("decode"),
                query.decode_row(&generated_reply).expect("decode"),
                "trial {trial}: the generated row of color {color}"
            );
            black_box(query.decode(&real_reply).expect("decode"));
            black_box(query.decode(&generated_reply).expect("decode"));
            for first_real in if trial % 2 == 0 { [true, false] } else { [false, true] } {
                let reply = if first_real { &real_reply } else { &generated_reply };
                let (_, ns) = timed(|| query.decode(reply).expect("decode"));
                if first_real {
                    real.push(ns);
                } else {
                    generated.push(ns);
                }
            }
        }
    }
    let treepir_entry = PairedEntry {
        role: "paired",
        system: "treepir",
        height: common.height,
        server: color,
        real_decode_ns: stats(real),
        generated_decode_ns: stats(generated),
    };
    vec![
        serde_json::to_value(undine_entry).unwrap(),
        serde_json::to_value(treepir_entry).unwrap(),
    ]
}

// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct BenchMeta {
    seed: u64,
    trials: usize,
    role: String,
    height: u8,
    kappa: usize,
    leaves: u64,
    seeded_leaves: u64,
    replies: Option<String>,
    maintainer: Option<usize>,
    color: Option<usize>,
    pir_path: &'static str,
    setup_threads: usize,
    setup_cpus: Option<String>,
    timed_cpus: Option<String>,
    elapsed_seconds: f64,
    peak_rss_bytes: Option<u64>,
    timestamp_unix_ms: u64,
    git: Option<String>,
    notes: &'static str,
}

fn run_benchmark() {
    let role = env_string("ROLE").unwrap_or_else(|| "client".to_string());
    let common = Common::from_env();
    let start = Instant::now();
    let cpus = CpuPlan::new();
    let entries = match role.as_str() {
        "server" => run_server(&common, &cpus),
        "client" => run_client(&common, &cpus),
        other => panic!("{ENV_PREFIX}_ROLE must be server or client; got {other}"),
    };
    let meta = BenchMeta {
        seed: BENCH_SEED,
        trials: common.trials,
        height: common.height,
        kappa: common.kappa,
        leaves: common.leaves,
        seeded_leaves: common.seeded_leaves(),
        replies: (role == "client")
            .then(|| env_string("REPLIES").unwrap_or_else(|| "generated".to_string())),
        maintainer: (role == "server").then(|| env_usize("MAINTAINER", 0)),
        color: (role == "server").then(|| env_usize("COLOR", 0)),
        role,
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
        elapsed_seconds: start.elapsed().as_secs_f64(),
        peak_rss_bytes: peak_rss_bytes(),
        timestamp_unix_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64,
        git: git_sha(),
        notes: "UndineLog against TreePIR at h = kappa over one YPIR backend, on the same allocated log with a seeded suffix: each server's preparation, rebuilt per trial, and its answer; or the client's requests and its processing of the replies into a verified opening",
    };
    let document = serde_json::json!({ "meta": meta, "entries": entries });
    let out_dir = Path::new("results");
    let _ = fs::create_dir_all(out_dir);
    let path = out_dir.join("bench_treepir_comparison_local.json");
    fs::write(path, serde_json::to_vec_pretty(&document).expect("serialize"))
        .expect("write bench_treepir_comparison_local.json");
}

fn main() {
    run_benchmark();
}
