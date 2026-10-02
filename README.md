# UndineLog Evaluation Artifact

This repository reproduces the empirical comparison of the UndineLog and
BorgLog append-only logs. It contains both implementations, the benchmark
sources, and a paper-independent reproduction pipeline that measures the two
layouts in one run, on one host, under one statistical protocol.

- `undine/` is the UndineLog implementation: the balanced bivariate layout,
  in which the segment size is `kappa^2` for threshold `kappa`.
- `borg/` is the BorgLog implementation, configured with the same segment
  size `kappa^2` and threshold `kappa`.

The reference run `20261002T054218Z-full-bb80da561d` and all of its
measurements are included under `reference-results/`. No paper source or
typesetting toolchain is required. The reference run is immutable; new runs
are written under `results/runs/`.

## Quick Start

From the repository root, verify the packaged reference run:

```bash
./artifact verify
```

This performs no benchmarking. A successful report has:

```text
jobs: 48
result_cells: 439
timing_series: 210
publishable: true
```

Then build and test both implementations and the pipeline:

```bash
./artifact test
```

Run the short path-coverage experiment:

```bash
./artifact smoke
```

The smoke run covers every benchmark family of both layouts with reduced
work. It validates the pipeline and result shapes, but it is not a
replacement for the full statistical run.

## Prerequisites

- x86-64 Linux. The runner uses Linux file locks and CPU affinity.
- Python 3.10 or newer.
- Git.
- Rust and Cargo with edition 2024 support.
- `taskset`, normally provided by Linux `util-linux`.
- A C++ compiler, used by the build script of the PIR backend.
- About 20 GiB of free memory for the largest private-opening job.

The reference machine had an Intel Core Ultra 7 255U, 32 GiB RAM, and 14
logical CPUs, and ran Linux with Rust 1.98.0. The benchmark processes were
pinned to logical CPU 2, a performance core, with the `performance` energy
preference. Exact environment details are retained in
[`environment.json`](reference-results/20261002T054218Z-full-bb80da561d/provenance/environment.json).
Cargo may need network access once to fetch locked dependencies.

## Reviewer Commands

| Goal | Command |
| --- | --- |
| Build and test | `./artifact test` |
| Run a quick experiment | `./artifact smoke` |
| Test interruption/resume | `./artifact resume-canary` |
| Start full reproduction | `./artifact full` |
| Resume a run | `./artifact resume RUN_ID` |
| Validate a new completed run | `./artifact validate RUN_ID` |
| Package a full run as the reference | `./artifact package RUN_ID` |
| Verify the packaged reference | `./artifact verify` |

CPU selection is automatic. Pass `--cpu N` to `smoke`, `resume-canary`,
`full`, or `resume` to pin a specific allowed logical CPU. On a processor with
performance and efficiency cores the automatic choice is a maximum-frequency
CPU, which is a performance core.

## Full Reproduction

For unattended execution:

```bash
nohup ./artifact full > artifact-run.log 2>&1 &
```

Before starting the 66-job run, the command validates the manifest, runs the
pipeline unit tests and a smoke experiment, proves interruption/resume
behavior, and executes a publication-sized canary. Gates are cached by source
snapshot, so successful work is not repeated.

Every benchmark batch is checkpointed. If the process or host stops, identify
the directory under `results/runs/` and resume it:

```bash
./artifact resume RUN_ID
```

Do not edit source or orchestration files between starting and resuming a run;
the runner rejects source drift. Only one experiment process may use the
repository at a time.

The included full run took 1.51 hours of benchmark batches. Runtime depends on
host speed, load, and variance; the generated preflight reports a
host-calibrated range, which was 1.8 to 21.3 hours on the reference host
because adaptive series may stop at 10 repetitions or continue to 160. The
canary took 13.9 minutes.

## Statistical Protocol

Each timing is sampled with independent repetitions at declared looks of
10, 20, 40, 80, and 160. At each look, the runner computes a Student-t
interval with a Bonferroni allocation of the 5% error budget across all five
looks. Sampling stops when the familywise 95% interval has relative half-width
at most 5%; otherwise it continues to the next look.

Direct benchmarks discard three warmup repetitions per process. Append
benchmarks discard one complete warmup sequence, and one independent append
sample is a complete 64-batch sequence. Deterministic quantities must agree
across observations. A full run is publication-ready only when every consumed
measurement exists and all 210 timing series satisfy the stopping rule.

Jobs of the two layouts use the same benchmark, the same workload variables,
the same CPU, and the same protocol. They alternate in the schedule.

## Evaluation Map

The artifact generates nine result groups:

| Result group | What is measured |
| --- | --- |
| `tab:append` | Append time and fragment growth per 4 MiB batch, both layouts |
| `tab:point-open` | Point-opening helper work, response size, and client work, both layouts |
| `tab:range-open` | Range-opening helper work, response size, and client work at threshold 64, both layouts |
| `tab:maintenance` | Repair, threshold doubling and halving, and the resulting fragment size, both layouts |
| `tab:on-the-fly-open` | UndineLog openings answered above the stored threshold |
| `tab:private-maintainer` | Private openings of one position: one maintainer claim's answer, preprocessing, and preprocessing update on logs of 1 to 64 GiB |
| `tab:private-single` | The same measurements for a single server that stores the whole tree |
| `tab:private-table` | The maintainer's records against YPIR over the table of complete openings |
| `tab:private-client` | Client work for a private opening next to a public one |

The first five groups use thresholds 8, 16, 32, and 64 with segment size
`kappa^2`, except the range group, which uses threshold 64. The private-opening
groups use threshold 64 for the log-size series and all four thresholds at a
4 GiB log and for the client. The manifest schedules only
measurements consumed by these groups, and output parsing rejects unconsumed
timing fields. See [Experiments](documentation/experiments.md) for exact
workloads and benchmark targets.

## Reference Results

The packaged run is
[`reference-results/20261002T054218Z-full-bb80da561d/`](reference-results/20261002T054218Z-full-bb80da561d/).
Start with its [rendered result groups](reference-results/20261002T054218Z-full-bb80da561d/tables/README.md).
Machine-readable values, units, samples, confidence intervals, raw benchmark
documents, logs, environment data, and source hashes are retained alongside
them. The run has 6,660 independent timing samples in 69 batches, and every
one of its 210 timing series met the precision target.

`./artifact verify` checks the complete file-integrity manifest, both crates'
source hashes, job and measurement semantics, checkpoint completion, every raw
batch artifact, the raw-sample export, statistical precision, and regenerated
CSV/JSON/Markdown outputs. It does not compare timings from different
machines.

See [Results and Provenance](documentation/results.md) for the directory
schema and interpretation rules.

## Success Criteria

A reproduction succeeds when:

1. `./artifact test`, `./artifact smoke`, and `./artifact resume-canary` exit
   with status 0.
2. `./artifact full` exits with status 0.
3. `./artifact validate RUN_ID` reports 48 jobs, 439 result cells, 210 timing
   series, and `publishable: true`.
4. The completed run's `summary.json` lists no missing or unstable
   measurements.

Exit status 2 from a full run means measurement completed but at least one
series missed the precision requirement. Exit status 75 is used only by the
deliberate interruption canary. Other nonzero statuses indicate a failed
invariant, build, benchmark, or validation.

## Documentation

- [Implementation](documentation/implementation.md)
- [Experiments](documentation/experiments.md)
- [Results and Provenance](documentation/results.md)
- [Troubleshooting](documentation/troubleshooting.md)

The root README is the authoritative entry point. The documents above contain
only implementation and reproduction details needed to inspect this artifact.
