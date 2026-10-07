# Experiments

## Manifest Design

`scripts/reproduce.py` defines the authoritative relation among result cells,
measurements, and benchmark jobs. A job is scheduled only when at least one
result cell consumes its output. Jobs with identical workload semantics are
deduplicated.

The complete manifest contains:

- 9 result groups;
- 713 exact result cells;
- 348 timing series;
- 70 deduplicated benchmark jobs: 22 per layout for the paired groups, 4
  for the UndineLog on-the-fly openings, and 22 for the private openings.

Manifest validation rejects uncovered cells, measurements without consumers,
duplicate job semantics, incompatible units, unexpected timing fields, and
deterministic outputs that vary across repetitions.

## Operating Points

Every job uses Merkle height 50. The thresholds are 8, 16, 32, and 64, each
with segment size `kappa^2`: 64, 256, 1,024, and 4,096. UndineLog benchmarks
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
client's decoding of the reply. Each repetition also computes the hint of 64
database columns spread over the row and the response-packing tables of one
reply ciphertext; the preprocessing cell scales the two to the whole database
by the column and ciphertext counts the benchmark reports. Then each
repetition appends 4 MiB to the log and updates the preprocessing from the
changed records, timing the hint patch and the packing tables.

**A single server** (`tab:private-single`). The server stores every cell of
the tree in groups of 64 cells and runs the same request, answer, decoding,
and preprocessing measurements at 1 and 4 GiB with the public pattern size
of threshold 1, `2^10` and `2^11` leaves, and with patterns of `2^9` and
`2^11` leaves at 1 GiB. The job at 1 GiB also reports the database
dimensions and message sizes of the 16 GiB to 1 TiB logs at their own
pattern sizes.

**The table of complete openings** (`tab:private-table`). YPIR's server
answers over a table with one complete opening per pattern, for the claim of
the maintainer jobs at 1 and 4 GiB. Its reply is checked to equal the
record-based reply byte for byte.

**The client** (`tab:private-client`). On a 12 MiB log at every threshold
and at threshold 256, and on the 1 and 4 GiB logs at threshold 64, at the
public grouping, each of `3 kappa` claims preprocesses its own primary
columns and answers one request honestly, and the first `kappa` of them
answer a second time with a lie about one node of the path. The client's
request, its decoding plus Combine+Verify from `kappa` correct replies, from
the `3 kappa` honest replies, and from the same registry with the `kappa`
lying replies, and Combine+Verify from the public replies of the same three
registries are timed in one process. Two further timings run the decoding and
Combine of the honest and of the lying registry under the protocol's
schedule: the opening is returned at a release time fixed in advance
(`CLIENT_RELEASE_MS`, past the slowest decoding of the threshold), so both
return at the release.

The benchmark checks, inside every run, that each decoded opening equals the
stored entries, that the serialized sizes equal the reported ones, that the
hint sample equals the stored hint, that after the appends the records, the
patched hint, and openings across the log equal those of the longer log, that
the single server's decoded cells hash to the root, and that a decoded
private reply equals the public reply of its claim.

The database is built and preprocessed on all cores before the timed
repetitions, which run on the pinned CPU. The result document records both
CPU sets.

## Execution and Sampling

Each benchmark binary writes a structured JSON document. The runner builds
each binary in its crate with the locked dependency graph, copies the exact
built binary into the isolated run directory, pins execution to one logical
CPU, captures stdout and stderr, hashes the result and raw-sample documents,
and checkpoints before and after every batch.

Full runs sample at 10, 20, 40, 80, and 160 independent repetitions and stop
as soon as all timing outputs of a job meet the familywise 95%, 5%
relative-half-width target. Jobs are resumed from the next required look; a
completed batch is never rerun. Smoke and canary modes use one measured
repetition for path coverage and host calibration, not for publication claims.

The list of deliberately omitted measurements is emitted in every run's
`audit-report.md`.
