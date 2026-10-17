# Experiments

## Manifest Design

`scripts/reproduce.py` defines the authoritative relation among result cells,
measurements, and benchmark jobs. A job is scheduled only when at least one
result cell consumes its output. Jobs with identical workload semantics are
deduplicated.

The complete manifest contains:

- 12 result groups;
- 1,001 exact result cells;
- 465 timing series, of which 434 require precision and 31 are diagnostics;
- 77 deduplicated benchmark jobs: 22 per layout for the paired groups, 4
  for the UndineLog on-the-fly openings, 26 for the private openings, and 3
  for the comparison with TreePIR.

Manifest validation rejects uncovered cells, measurements without consumers,
duplicate job semantics, incompatible units, unexpected timing fields, and
deterministic outputs that vary across repetitions.

## Operating Points

Every job of the first nine result groups uses Merkle height 50; the
comparison with TreePIR uses a tree of height 32 with `kappa = h`. Height 16
is reserved for the standalone correctness check below. The thresholds are
8, 16, 32, and 64, each with segment size `kappa^2`: 64, 256, 1,024, and 4,096.
UndineLog benchmarks
take the threshold. BorgLog benchmarks take the segment size together with
the threshold or the primary width `Seg / kappa`. Apart from these
operating-point variables, the two layouts' jobs for one workload have the
same environment; the pipeline's unit tests enforce this.

## Workloads

### Appends (`tab:append`)

Benchmark: `bench_append_only_log_bulk_updates_append_local`

One job per layout and threshold appends 4 MiB batches with one local claim.
One independent repetition starts from fresh state and executes 64
consecutive batches. Each result is the mean per-batch maintainer processing
time, including active-prefix extension and suffix installation. The fragment
cell is the mean newly materialized fragment size per batch, computed from
the preorder cursor and the per-segment fragment size.

### Point Openings (`tab:point-open`)

Benchmarks: `bench_append_only_log_point_open_local` and
`bench_append_only_log_client_point_open_local`

One helper job and one client job per layout and threshold, after three 4 MiB
batches. Helper work includes response construction and serialization. Client
work starts after responses arrive. The clean case uses `kappa` correct
responses; the correcting case uses `3*kappa` responses of which `kappa` are
corrupted.

### Range Openings (`tab:range-open`)

Benchmark: `bench_append_only_log_range_open_local`

Threshold 64, segment size 4,096, sixteen committed 8 MiB batches. Queries
return 1, 4, 8, 32, or 128 MiB. Each layout has a helper job and a client job.

The helper job warms a server bound to one claim with the same appends as the
full server and times that server's handler for the range request plus
serialization of the response. Before timing, the helper's serialized answer
is checked to equal its claim's share of the full server's answer. The
request's full-segment and point-read counts and the response size come from
this job.

The client job times Combine+Verify from 192 clean responses and from 192
responses of which 64 are corrupted.

### Repair and Reconfiguration (`tab:maintenance`)

Benchmarks: `bench_append_only_log_repair_kernels_local` and
`bench_append_only_log_one_server_kappa_rescale_local`

One repair job and one reconfiguration job per layout and threshold, after a
12 MiB inserted-value prefix. Reported values are normalized per 4 MiB.

Repair begins without usable target state and measures receiver work, one
helper's serialized RPC work, and the modeled total helper payload, which is
the claim's whole fragment.

Reconfiguration measures both directions between `kappa` and `2*kappa`:

- UndineLog: the receiver's local four-segment stitch, with no helper and no
  download; one helper's partial-stitch response and the receiver's decode
  and undo for halving from helpers at `2*kappa`; and the undo from the source
  state of two diagonal segments, which is what helpers at `kappa` supply.
  The fragment size after doubling is reported by the benchmark.
- BorgLog: receiver and helper work and the download for threshold doubling
  and halving at a fixed segment size, and the local coupled
  segment-and-threshold double. The fragment size after doubling is computed
  from the segment size and the doubled threshold.

### On-the-Fly Openings (`tab:on-the-fly-open`)

Benchmark: `bench_append_only_log_on_the_fly_open_local` (UndineLog only)

One job per stored threshold. One maintainer with one claim stores a 12 MiB
log at the stored threshold and answers one point opening and range openings
of 1 and 4 MiB at 1, 2, 4, and 8 times that threshold. Above the stored
threshold each answer is timed twice: from the maintainer stored at the
stored threshold, and from a copy that ran the threshold doublings and stores
the log at the opened threshold. Before timing, the two serialized answers
are checked to be equal. The time of those doublings on the whole state is
measured as well. The trials of all series of a job are interleaved.

### Private Openings (`tab:private-maintainer`, `tab:private-single`, `tab:private-table`, `tab:private-client`)

Benchmark: `bench_private_opening_local` (UndineLog only)

A private opening of one position is a PIR query over the records a server
holds. One process measures one configuration.

**One claim of a maintainer** (`tab:private-maintainer`). The claim's primary
columns are built for a seeded log of 1, 4, 16, 64, and 256 GiB at
threshold 64, for a 4 GiB log at thresholds 8, 16, and 32, and for a 1 TiB
log at threshold 256, the largest log whose fragment fits the reference
host. The group is one segment,
and the pattern size is the public one of the log length and the
threshold, `max(2 log2 kappa - 1, floor((h + log2 kappa - 5) / 2))` leaves
as a power of two: `2^13` to `2^17` leaves at threshold 64 and `2^19` at
threshold 256 on the 1 TiB log. Three more jobs
at 64 GiB use patterns of `2^14`, `2^15`, and `2^17` leaves next to the
public `2^16`. The record of a pattern is the columns of the segments that
its subtree meets and one entry per level above. Each repetition
opens one position: the client's request
(key generation, query, serialization), the maintainer's first pass, the
complete answer from the serialized request to the serialized reply, and the
client's decoding of the reply. Preprocessing is sampled independently of
online openings and append updates. Each preprocessing repetition rebuilds
the entire hint, including padded columns, and all response-packing tables
on one core, using the same constructor and timing boundaries as the TreePIR
comparison. The artifact reports the two stages (`hint_ns`, `packing_ns`)
and their per-rebuild sum (`preprocess_ns`), without extrapolation. Tree
construction, fragment encoding, conversion to PIR cells, and construction
of the row table and upper-node cache are outside this hint-and-packing
total. Each build serves a checked opening. Between builds only the input
cell buffer is retained; the previous preprocessing is released.
The preprocessing size is recorded directly. Historical runs that report
`hint_sample_ns` and `packing_table_ns` remain estimates and must not be
substituted for these full-build measurements. Then each
repetition appends 4 MiB to the log and updates the preprocessing from the
changed records, timing the hint patch and the packing tables. Sampled hint
columns are used only for an untimed correctness check after these updates.

**A single server** (`tab:private-single`). The server stores every cell of
the tree in groups of 64 cells and runs the same request, answer, decoding,
and preprocessing measurements at 1 and 4 GiB with the public pattern size
of threshold 1, `2^10` and `2^11` leaves, and with patterns of `2^9` and
`2^11` leaves at 1 GiB. The job at 1 GiB also reports the database
dimensions and message sizes of the 16 GiB to 1 TiB logs at their own
pattern sizes.

**The table of complete openings** (`tab:private-table`). YPIR's server
answers over a table with one complete opening per pattern, for the claim of
the maintainer jobs at 1 and 4 GiB. It uses the tiled dense builder and compact
packing preprocessing shared with TreePIR, while retaining an independent
bitwise reference packer. Its reply is checked to equal the record-based
reply byte for byte.

**The client** (`tab:private-client`). On the 4 GiB log at every threshold
and at threshold 256, and on the 8, 16, and 32 GiB logs at threshold 64,
at the public
grouping, each of `3 kappa` claims answers one request honestly, and the
first `kappa` of them answer a second time with a lie about one node of the
path. Setup encodes only the group records and the records holding upper
nodes touched by this opening, for all `3 kappa` claims, using the same
segment encoder as a full fragment build. It respects the active prefix,
including final-segment defaults and zero padding. This avoids allocating
whole-tree fragments without changing any selected share. A claim's reply
is the row the query selects encrypted
under the client's key in the server's response format, in place of the
server's preprocessing and answer; the opened leaf is unwritten. The client's
request, its decoding plus Combine+Verify from `kappa` correct replies, from
the `3 kappa` honest replies, and from the same registry with the `kappa`
lying replies, and Combine+Verify from the public replies of the same three
registries are timed in one process; the lying registry's time is the least
release time of the protocol's schedule at that threshold.

The benchmark checks, inside every run, that each decoded opening equals the
stored entries, that the serialized sizes equal the reported ones, and that
after the appends the records, a recomputed sample of the
patched hint, and openings across the log equal those of the longer log, that
the single server's decoded cells hash to the root, and that a decoded
private reply equals the public reply of its claim.

The input cells are built on all cores before the timed repetitions, which
run on the pinned CPU. The result document records both CPU sets. Each
preprocessing batch discards one warmup rebuild; online and update batches
discard three warmup samples. A group stops once all its measurements meet
the existing precision target. If only online or update samples remain, one
fresh untimed build supplies their database; measured preprocessing samples
always rebuild it completely.

### UndineLog against TreePIR (`tab:treepir-server`, `tab:treepir-client`, `tab:treepir-check`)

Benchmark: `bench_treepir_comparison_local` (both systems in the UndineLog
crate, over the same YPIR backend)

The matched point is `h = kappa = 32`: a log of `2^32` leaves (128 GiB), the
perfect tree TreePIR needs without padding, 32 UndineLog maintainers and 32
TreePIR servers, and the same reply ciphertext count, which the runner checks.

**Servers** (`tab:treepir-server`). Two jobs each pair one UndineLog claim with
one TreePIR class: claim 0 with class 0, and claim 31 with class 31. Every
preprocessing repetition rebuilds both servers from the start, in an order
that alternates, and checks a fresh query on each. Online samples stop
independently: they are distributed over the requested builds. An online-only
batch makes one untimed build of each system and serves its requested queries
from it, retaining only one large server in memory at a time. They use the same valid Merkle tree as
the client: `2^32 - 2^20` allocated, unwritten leaves followed by `2^20`
seeded leaves. Every query opens a seeded leaf. UndineLog uses the existing
`BulkLog::build_with_default_prefix`, and each rebuilt fragment's root must
equal the root of the `SuffixTree` supplying TreePIR's labels. The result
records the seeded-leaf count and both roots; the runner checks them. Default
labels avoid hashing unwritten subtrees, while both systems materialize their
full PIR inputs with normal padding and the backend's zero-block shortcut.

The maintainer's fragment, including encoding the claim's primary columns and
hashing the seeded suffix, is built once per process on every online CPU and
kept in a temporary file; its wall time is recorded in the result, outside the
timed series. Every build reloads it untimed and constructs fresh PIR state. UndineLog's stages are the primary columns into PIR cells, the row
table and the upper nodes' cells, the hint (which gathers its columns from the cells), and
the packing tables; TreePIR's are the class's nodes in database order, the
database columns from their labels, the hint, and the packing tables. All
run on the pinned CPU. The initial `SuffixTree`
construction is fixture setup; tree hashing is excluded from the PIR
conversion, hint, packing and answer timers. These jobs discard one warmup
rebuild per process.

The storage columns are each server's dedicated PIR state: the PIR database
(UndineLog's is derived from the claim's primary columns and needs none of the
auxiliary rows; TreePIR's is the class's dense padded matrix), the hint, and
the packing tables. After the first rebuild's timers, each job counts the
hint's all-zero blocks of 2,048 rows of one column, which the hint's kernel
skips, and those that lie in the padding columns. The runner checks that
every zero block count lies between the padding blocks and all blocks.

**Clients** (`tab:treepir-client`). At `h = 32` the tree has `2^32 - 2^20`
allocated, unwritten leaves followed by `2^20` seeded ones.
Every opening is of a seeded leaf, so it returns and verifies
real content. Each repetition times
both clients, alternating: the requests in parts (indexing, key generation,
the query with its packing keys, serialization) on one opening and as a whole
on a second; then the processing of every reply into a verified opening, in
parts (decrypting the replies, UndineLog's Combine+Verify, the root from the
recovered path) and as a whole. UndineLog uses one secret key and one request
for its `kappa` maintainers; TreePIR uses one secret key per request, since
the backend reuses a public query matrix. Reconstruction is fault-free. Each
reply is the row its server returns, read from the tree without building the
server and encrypted under the client's key in the server's format.

**Generated and real replies** (`tab:treepir-check`). The `h = 32` client job
also builds UndineLog claim 0 and TreePIR class 0 on the opening tree. For
every repetition, the real reply and the generated reply for the same server
and query decode to the same row (checked exactly), and both are decoded in
turn. The runner rejects a batch whose mean decoding times differ by more
than 10%.

#### Height-16 correctness check

Height 16 tests the generated-reply method against every real server of both
systems. Every leaf is seeded, each real reply must decode to the same row
as its generated counterpart, and every opening must verify. This is a
correctness check, not a timing result: it has no rows in the publication
tables and no adaptive sampling or precision target.

The smoke gate separately runs the `treepir` Rust integration tests, including
an opening through every real color server on a small tree. Removing height 16
from the timing manifest therefore preserves automatic correctness coverage.

To run it once from the repository root:

```bash
BENCH_TREEPIR_COMPARISON_ROLE=client \
BENCH_TREEPIR_COMPARISON_HEIGHT=16 \
BENCH_TREEPIR_COMPARISON_SEEDED_LEAVES=65536 \
BENCH_TREEPIR_COMPARISON_REPLIES=real \
BENCH_TREEPIR_COMPARISON_TRIALS=1 \
BENCH_SAMPLE_GROUPS='{"client":1}' \
cargo bench --locked --manifest-path undine/Cargo.toml \
  --features bench --bench bench_treepir_comparison_local
```

A successful invocation exits with status 0 after checking eight opening
pairs per system. Its diagnostic JSON is not input to the publication tables.

## Execution and Sampling

Each benchmark binary writes a structured JSON document. The runner builds
each binary in its crate with the locked dependency graph, copies the exact
built binary into the isolated run directory, pins execution to one logical
CPU, captures stdout and stderr, hashes the result and raw-sample documents,
and checkpoints before and after every batch.

Full runs sample at 10, 20, 40, 80, 160, 320, 640, 1,280, 2,560, 5,120,
10,240 and 20,480 independent repetitions. The Student-t interval divides
the 5% error budget across all 12 declared looks. Sampling stops as soon as
all precision-required outputs of each sample group meet the familywise 95%,
5% relative-half-width target. Earlier samples remain in every subsequent
calculation. A job completes after every group has stopped; a group that
reaches 20,480 samples without meeting the target remains flagged and prevents
publication validation.
Jobs are resumed from each group's next required look; a
completed batch is never rerun. Smoke and canary modes use one measured
repetition for path coverage and host calibration, except that the paired
real/generated-reply check collects 10 retained pairs and three warmup pairs.
That check enforces a 10% tolerance between means and therefore needs repeated
measurements even in a quick gate run. Neither mode supports publication claims.

The list of deliberately omitted measurements is emitted in every run's
`audit-report.md`.

Independent groups keep expensive work from being repeated because a short
operation is noisy. Private maintainer/single-server groups are preprocessing,
online work, and (where applicable) updates. The TreePIR comparison separates
server preparation from online work. Client components and directly timed
totals share one group, so adaptive stopping never changes the allocation
history between the two openings; the paired reply check has its own group.
For the TreePIR client comparison, only request preparation, reply processing
and their per-opening total require precision. Indexing, key generation, query
generation, serialization, decryption, combination and standalone verification
remain diagnostic breakdowns, with all samples and their achieved intervals.
For the server comparison, database filling includes both cell conversion and
gathering. Hint plus packing is another per-rebuild sum. These combined costs,
color enumeration, full preparation and complete answers require precision;
their finer stages and the first pass are diagnostics. Sums pair stages from
the same rebuild before confidence intervals are computed, and the validator
checks each sum against the raw stages. A diagnostic cannot trigger another
batch, but missing or malformed diagnostic samples still fail validation.
This scope is declared in the manifest before the run, equally for both systems.
The benchmark echoes requested
counts; the runner validates exact counts and group-specific warmups, saves
all raw samples, and rechecks the schedule when resuming. Inactive groups
emit no samples. Groups share the same declared schedule, minimum sample count
and confidence target; converged preparation groups stay stopped while noisy
online groups collect more samples.

Local public-opening setup reclaims sole ownership of its server after dropping
the setup client and waiting for background fetches to release their references.
Client point-opening jobs destroy that server before timing captured replies;
point, range and on-the-fly helper jobs keep the reclaimed server for measurement.
This prevents setup-server destruction on a worker from competing with timed
operations on the pinned CPU. The same handoff is used for Borg and Undine.
The wait is outside all reported timers and fails after 30 seconds if an owner
does not release the server. Decoding, serialization and verification remain
inside their original timing boundaries.

Short public range helper timers and on-the-fly point-answer timers average
32 calls per sample. The JSON records `inner_repetitions = 32` or
`point_calls_per_sample = 32`; each average counts as one independent sample.
Response destruction stays inside the timer, as in the original single-call
helper measurement. In the TreePIR client comparison, one sample averages
eight complete opening pairs (`client_openings_per_sample = 8`). Each pair
times the components of one opening and the request/reply totals directly on
a separate opening; both use fresh keys and different leaves. Each operation
runs once in its normal client context. No small component is repeatedly timed
in isolation. The output retains every per-opening timing as
`raw_call_samples`, and the runner re-derives the batch means. Inner openings
do not increase the confidence interval's sample count. Reported percentiles
are of batch means, not single-request latency quantiles.

Untimed fixture caches under `results/runs/<run>/fixtures/` share exact input
state between adaptive batches and compatible jobs. Keys include the complete
workload and executable hash; loads verify the complete-file checksum and
reject corrupted data. Range snapshots include the root, primary columns and
auxiliary rows. Bulk snapshots include the append state and restore a fresh
mutable instance for every process. Hints, packing tables and measured
operations are never loaded from a cache. Partial writes are cache misses.

Every measured first pass (structured, materialized, single-server or TreePIR)
follows one untimed scan with the same query. This gives fresh and reused
databases the same warmup policy; otherwise independent groups could mix a
scan after a rebuild with scans after previous queries. JSON records
`first_pass_warmup_calls = 1` for these server roles. The complete-answer
timer still includes parsing and serialization. Historical first-pass results
without this field used no per-query scan warmup and should be distinguished
when comparing microbenchmarks; complete-answer timing boundaries are unchanged.

Each run builds in its own `build-targets/<layout>/` directory, bound to the
source snapshot. Crate-local Cargo caches and compiler wrappers are excluded.
The snapshot includes the compiler versions, build flags and Cargo configuration
hashes, and the runner checks it before and after building and before each
batch. The archived executable is hashed and bound to that snapshot; resumption
checks the executable again. Exact source and test files, including untracked
additions, are retained under `provenance/source-tree/`. Only manifest-declared `BENCH_*` settings reach a
benchmark, and profiling/tracing switches are removed. Build caches and untimed
fixture files are not included in a packaged reference run.
