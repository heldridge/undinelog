# Troubleshooting

## Dependency or Build Failure

Use the locked dependency graphs:

```bash
cargo fetch --locked --manifest-path undine/Cargo.toml
cargo fetch --locked --manifest-path borg/Cargo.toml
./artifact test
```

If an older compiler rejects edition 2024 or a dependency, use a newer stable
toolchain. Do not regenerate either `Cargo.lock`; their hashes are part of
artifact provenance.

## CPU Affinity Failure

The runner chooses an allowed maximum-frequency logical CPU and invokes
`taskset`. Install the Linux `util-linux` package if `taskset` is absent. In a
restricted container, inspect the allowed set with:

```bash
taskset --cpu-list --pid $$
```

Then pass one listed CPU explicitly:

```bash
./artifact smoke --cpu N
```

Use the same CPU when resuming a run. A requested CPU that differs from the
checkpoint is rejected.

## Interrupted Run

Each batch is checkpointed under `results/runs/RUN_ID/checkpoint.json`. Resume
without deleting partial output:

```bash
./artifact resume RUN_ID
```

The runner verifies the source snapshot, saved benchmark binaries, and all
completed batch artifacts before continuing. If source changed, restore the
exact committed artifact state rather than editing the checkpoint.

Use `./artifact resume-canary` to create a deliberate one-batch interruption
and prove that resume completes correctly.

## Full Run Stops With Status 2

Status 2 means all scheduled work completed but the result was not
publication-ready. Inspect:

```text
results/runs/RUN_ID/summary.json
results/runs/RUN_ID/audit-report.md
```

`missing_full_manifest_measurements` identifies absent outputs.
`unstable_measurements` identifies timing series that still exceeded the 5%
relative-half-width target at the run's declared maximum (20,480 repetitions
in the current protocol; 160 in earlier runs). Preserve the run as evidence;
do not remove samples or edit the summary.

## Timeout

The pipeline's per-build and per-benchmark timeout defaults to 24 hours, which
is separate from total experiment time. If a process is terminated
externally, resume the checkpointed run. A persistently stalled benchmark
should be treated as a failed reproduction rather than silently omitted.

## Verification Failure

`./artifact verify` prints the first violated invariant. Hash failures mean
the packaged reference or a measured source file changed. Manifest failures
mean workload or reporting semantics changed. Derived-output failures can be
diagnosed without benchmarking:

```bash
python3 scripts/export_results.py reference-results/RUN_ID --check
```

Run all documentation and package checks with:

```bash
python3 scripts/check_artifact.py
```
