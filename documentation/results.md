# Results and Provenance

## Included Run

The immutable reference is
[`20261005T202005Z-full-0c73f1f8f7`](../reference-results/20261005T202005Z-full-0c73f1f8f7/).
It completed all 68 jobs and 338 timing series and is marked
publication-ready. The run contains 15,530 independent timing samples and 123
raw batch records. Its
[audit report](../reference-results/20261005T202005Z-full-0c73f1f8f7/audit-report.md)
lists the shared measurements and the deliberately omitted work.

## Runs

New runs are written under `results/runs/RUN_ID/`. A completed,
publication-ready full run is packaged with `./artifact package RUN_ID`,
which copies it to `reference-results/RUN_ID/` without its compiled benchmark
binaries, writes an integrity manifest, and records the run in
`reference-results/reference.json`. The binaries' hashes remain in the
checkpoint, and retained batch outputs can be verified without executing them.

## Directory Schema

| Path | Contents |
| --- | --- |
| `manifest.json` | Jobs, selectors, measurements, formulas, cells, and consumers |
| `checkpoint.json` | Run state, seeds, batch metadata, commands, and hashes |
| `summary.json` | Values, independent samples, statistics, and publication status |
| `summary.tsv` | Flat measurement summary |
| `raw-samples.jsonl` | Independent samples and references to every raw batch |
| `jobs/JOB/batch-NNN/` | Raw benchmark JSON, raw samples, stdout, and stderr |
| `builds/` | Build commands and logs |
| `tables/tables.json` | All rendered result groups |
| `tables/*.csv` | One CSV per result group |
| `tables/README.md` | Human-readable tables with units and conventions |
| `audit-report.md` | Statistical status, shared measurements, and omitted work |
| `result-validation.json` | Expected and actual dimensions for every group |
| `provenance/environment.json` | Host, toolchain, CPU, memory, affinity, and the benchmark CPU's frequency-scaling settings and sibling threads |
| `provenance/source-snapshot.json` | Source file hashes and Git state |
| `integrity.json` | Size and SHA-256 for every other retained reference file |

## Interpreting Measurements

Timing records in `summary.json` include the original independent samples,
mean, standard deviation, standard error, ordinary 95% Student-t interval, and
the Bonferroni-adjusted sequential interval used for stopping. The `stable`
field records whether the adjusted relative half-width is at most 5%.

Value records are deterministic quantities such as response bytes, read
counts, and download sizes. Formula records derive table geometry and storage
quantities from declared parameters. Rendered CSV and Markdown are convenient
views; `summary.json` and `manifest.json` remain the authoritative
machine-readable values and semantics.

Timings cover the local operation identified by each manifest measurement.
They include serialization where the result group says so, but exclude network
transit. The two layouts are measured in the same run on the same CPU, so
their cells are comparable with each other. The artifact does not assert that
absolute timings should match on a different processor.

## Verification

```bash
./artifact verify
```

The verifier:

1. hashes every file declared by `integrity.json`;
2. checks both crates against the reference run's source snapshot and against
   `provenance/sources.json`;
3. rebuilds the current job manifest and compares all jobs, produced
   measurement semantics, and result cells with the reference manifest;
4. verifies all completed jobs, batches, return codes, result hashes, and raw
   sample hashes;
5. reconstructs independent samples from `raw-samples.jsonl`;
6. rechecks the stopping rule and publication status;
7. regenerates and byte-compares all JSON, CSV, and Markdown result views.

Validate a newly generated result with:

```bash
./artifact validate RUN_ID
```

The new-run command performs the same structural, sample, statistical, and
derived-output checks without requiring the reference run's identity.
