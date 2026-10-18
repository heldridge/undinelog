# Reproduction audit report

- Run: `20261018T005140Z-full-b8533f977d`
- Mode: `full`
- Source snapshot: `b8533f977daaf1dd8740497e6ce6ee70f040d5abadb5ecbbdedc08101231852a`
- Benchmark CPU: 2
- Jobs selected: 78
- Manifest result cells: 1037
- Manifest timing series: 484
- Precision-required timing series: 440
- Diagnostic timing series: 44
- Uncovered cells: 0
- Measurements without consumers: 0
- Duplicate jobs: 0

## Statistical status

Publication-ready: **True**. Unstable required timing series: 0. Diagnostic precision misses: 9. Missing full-manifest measurements: 0.

The stopping rule uses Student-t intervals and allocates the 5% error budget across looks at 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480 independent repetitions. Each benchmark process discards explicit warmup work before collecting samples. A complete 64-batch append sequence is one independent repetition.

## Shared work

- `output:private-maintainer-undine-log1gib-p13-k64:database_bytes` -> `tab:private-maintainer|log:k64-log1gib-p13|database`, `tab:private-table|log1gib|records-mib`
- `output:private-maintainer-undine-log1gib-p13-k64:server_answer_ns` -> `tab:private-maintainer|log:k64-log1gib-p13|answer`, `tab:private-table|log1gib|records-answer-ms`
- `output:private-maintainer-undine-log1gib-p13-k64:server_first_pass_ns` -> `tab:private-maintainer|log:k64-log1gib-p13|first-pass`, `tab:private-table|log1gib|records-first-pass-ms`
- `output:private-maintainer-undine-log4gib-p14-k64:database_bytes` -> `tab:private-maintainer|log:k64-log4gib-p14|database`, `tab:private-table|log4gib|records-mib`
- `output:private-maintainer-undine-log4gib-p14-k64:server_answer_ns` -> `tab:private-maintainer|log:k64-log4gib-p14|answer`, `tab:private-table|log4gib|records-answer-ms`
- `output:private-maintainer-undine-log4gib-p14-k64:server_first_pass_ns` -> `tab:private-maintainer|log:k64-log4gib-p14|first-pass`, `tab:private-table|log4gib|records-first-pass-ms`
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
- private openings of ranges
- private-opening baselines with one query per tree level, replicated full-copy servers, and BorgLog under the same PIR pipeline
- UndineLog against TreePIR at heights other than 16 and 32, at other thresholds than kappa = h, and with faulty replies
- private openings over an emulated wide-area network
- the single server's preprocessing update after an append
- the maintainer service's update that compares every record; the measured update reads the records an append changes

## Result validation

All generated result groups have the declared dimensions:

- `tab:append`: 4 rows, 6 columns
- `tab:point-open`: 4 rows, 10 columns
- `tab:range-open`: 5 rows, 11 columns
- `tab:maintenance`: 8 rows, 16 columns
- `tab:on-the-fly-open`: 16 rows, 12 columns
- `tab:private-maintainer`: 12 rows, 17 columns
- `tab:private-single`: 8 rows, 14 columns
- `tab:private-table`: 2 rows, 7 columns
- `tab:private-client`: 8 rows, 11 columns
- `tab:treepir-server`: 4 rows, 25 columns
- `tab:treepir-client`: 4 rows, 18 columns
- `tab:treepir-check`: 2 rows, 4 columns

Machine-readable measurements and confidence intervals are in `summary.json` and `summary.tsv`. Reviewer-facing CSV, JSON, and Markdown result groups are under `tables/`.
