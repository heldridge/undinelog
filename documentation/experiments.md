# Experiments

## Manifest Design

`scripts/reproduce.py` defines the authoritative relation among result cells,
measurements, and benchmark jobs. A job is scheduled only when at least one
result cell consumes its output. Jobs with identical workload semantics are
deduplicated.

The complete manifest contains:

- 5 result groups;
- 439 exact result cells;
- 210 timing series;
- 48 deduplicated benchmark jobs: 22 per layout for the paired groups and 4
  for the UndineLog on-the-fly openings.

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
