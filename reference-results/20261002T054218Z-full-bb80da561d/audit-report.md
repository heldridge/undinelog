# Reproduction audit report

- Run: `20261002T054218Z-full-bb80da561d`
- Mode: `full`
- Source snapshot: `bb80da561dba3ca3113a206170cbaa8fcc6dbed513013c31fb3bd5990c011c9d`
- Benchmark CPU: 2
- Jobs selected: 48
- Manifest result cells: 439
- Manifest timing series: 210
- Uncovered cells: 0
- Measurements without consumers: 0
- Duplicate jobs: 0

## Statistical status

Publication-ready: **True**. Unstable timing series: 0. Missing full-manifest measurements: 0.

The stopping rule uses Student-t intervals and allocates the 5% error budget across looks at 10, 20, 40, 80, 160 independent repetitions. Each benchmark process discards explicit warmup work before collecting samples. A complete 64-batch append sequence is one independent repetition.

## Shared work

- `output:repair-borg-k16:modeled-download` -> `tab:maintenance|borg:k16|fragment`, `tab:maintenance|borg:k16|repair-download`
- `output:repair-borg-k32:modeled-download` -> `tab:maintenance|borg:k32|fragment`, `tab:maintenance|borg:k32|repair-download`
- `output:repair-borg-k64:modeled-download` -> `tab:maintenance|borg:k64|fragment`, `tab:maintenance|borg:k64|repair-download`
- `output:repair-borg-k8:modeled-download` -> `tab:maintenance|borg:k8|fragment`, `tab:maintenance|borg:k8|repair-download`
- `output:repair-undine-k16:modeled-download` -> `tab:maintenance|undine:k16|fragment`, `tab:maintenance|undine:k16|repair-download`
- `output:repair-undine-k32:modeled-download` -> `tab:maintenance|undine:k32|fragment`, `tab:maintenance|undine:k32|repair-download`
- `output:repair-undine-k64:modeled-download` -> `tab:maintenance|undine:k64|fragment`, `tab:maintenance|undine:k64|repair-download`
- `output:repair-undine-k8:modeled-download` -> `tab:maintenance|undine:k8|fragment`, `tab:maintenance|undine:k8|repair-download`

## Deliberately omitted work

- append batches other than 4 MiB
- append generic, codec, preparation-only, and prepared-without-slack paths
- point-helper core-only timing
- range sizes 2, 16, and 64 MiB
- range helper core-only, threshold-only, and split-range timing variants
- repair component, correction, replay, and rebuild timings
- reconfiguration component and all-helper serial timing variants
- segment sizes other than kappa^2

## Result validation

All generated result groups have the declared dimensions:

- `tab:append`: 4 rows, 6 columns
- `tab:point-open`: 4 rows, 10 columns
- `tab:range-open`: 5 rows, 11 columns
- `tab:maintenance`: 8 rows, 16 columns
- `tab:on-the-fly-open`: 16 rows, 12 columns

Machine-readable measurements and confidence intervals are in `summary.json` and `summary.tsv`. Reviewer-facing CSV, JSON, and Markdown result groups are under `tables/`.
