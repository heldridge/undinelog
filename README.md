# UndineLog Evaluation Artifact

This repository reproduces the empirical comparison of the UndineLog and
BorgLog append-only logs. It contains both implementations, the benchmark
sources, and a paper-independent reproduction pipeline that measures the two
layouts in one run, on one host, under one statistical protocol.

- `undine/` is the UndineLog implementation: the balanced bivariate layout,
  in which the segment size is `kappa^2` for threshold `kappa`.
- `borg/` is the BorgLog implementation, configured with the same segment
  size `kappa^2` and threshold `kappa`.

No paper source or typesetting toolchain is required. New runs are written
under `results/runs/`. A completed full run is packaged under
`reference-results/` and is immutable once packaged.

## Quick Start

Build and test both implementations and the pipeline:

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

When a reference run is packaged, verify it without benchmarking:

```bash
./artifact verify
```

## Prerequisites

- x86-64 Linux. The runner uses Linux file locks and CPU affinity.
- Python 3.10 or newer.
- Git.
- Rust and Cargo with edition 2024 support.
- `taskset`, normally provided by Linux `util-linux`.

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

Before starting the 48-job run, the command validates the manifest, runs the
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

Runtime depends on host speed, load, and variance. The generated preflight
reports a host-calibrated range.

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

The artifact generates five result groups:

| Result group | What is measured |
| --- | --- |
| `tab:append` | Append time and fragment growth per 4 MiB batch, both layouts |
| `tab:point-open` | Point-opening helper work, response size, and client work, both layouts |
| `tab:range-open` | Range-opening helper work, response size, and client work at threshold 64, both layouts |
| `tab:maintenance` | Repair, threshold doubling and halving, and the resulting fragment size, both layouts |
| `tab:on-the-fly-open` | UndineLog openings answered above the stored threshold |

Every group uses thresholds 8, 16, 32, and 64 with segment size `kappa^2`,
except the range group, which uses threshold 64. The manifest schedules only
measurements consumed by these groups, and output parsing rejects unconsumed
timing fields. See [Experiments](documentation/experiments.md) for exact
workloads and benchmark targets.

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
