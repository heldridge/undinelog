# October 17 scope and convergence audit

This archive is diagnostic evidence, not a source of replacement paper values.
The full run `20261017T012510Z-full-a1dd1b9f80` was stopped at the author's
request. A future full run starts with empty samples.

## Scope correction

Height 16 was a correctness workload mistakenly scheduled as a publication
benchmark. The revised manifest contains only height-32 TreePIR comparisons:
77 jobs, 993 result cells and 457 timing series. `manifest-scope.json` checks
that the other jobs, measurements and result cells are unchanged. The 5%
relative-half-width target and all 12 sequential looks are unchanged.

Height 16 remains available as a separately documented, single-invocation
correctness check. The smoke gate now explicitly includes the existing
`treepir` integration tests, which reconstruct actual Merkle openings using
every real color server on a small tree. `pipeline-tests.log` records 60 passing
scheduler/validation tests. `treepir-tests.log` records the two passing Rust
integration tests; these were run from the implementation checkout with
`cargo test --locked --release --features bench --test treepir -- --test-threads=1`
pinned to CPU 0. The mirrored test and compiled source files are unchanged.

## What caused the extra sampling

Of the 76 completed jobs, 51 stopped at 10 samples, 12 at 20, five at 40, four
at 80 and four at 160. The saved checkpoint retains all observations, including
slow ones. `analyze.py` recomputes each stopping decision from the retained
prefixes, using each completed batch's declared sample counts:

```sh
python3 results/diagnostics/20261017-convergence/analyze.py
```

The result is `stopping-analysis.json`. The original checkpoint is preserved
before the user-requested stop, so its final interrupted batch still says
`running`; the analysis uses only completed batches. `source-run/superseded.json`
records the stop and excludes these results from publication use.

The height-16 client had completed 80 retained samples. Request preparation,
reply processing and total client time met the corrected 5% requirement after
10 samples, all with relative half-widths below 1%. Index selection did not.
In particular, UndineLog's indexing mean changed from about 3.1--3.3 us in the
first three processes to 7.0 us in the fourth, while its request and processing
totals remained close. This is evidence of component timing variability across
processes, not evidence of incorrect reconstruction. The logs do not identify
which allocator, cache or scheduling effect caused the shift.

The claim/class-31 server job took 235.5 minutes across its completed batches
and needed 40 samples. Its preparation totals were already within 0.4% at
10 samples. UndineLog's row-table/upper-node gathering time was normally about
72--75 ms, with calls of 95.6 and 99.9 ms. Its corrected interval narrowly
missed the target at 20 samples (5.07%), requiring another 20 complete builds.
TreePIR's first pass also needed 40 samples. No slow observations are removed,
and short components are not repeatedly timed in isolation to force convergence.

## Independent height-32 check

`h32-original-0/` holds one new invocation of the unchanged executable, pinned
to CPU 2, with 13 client samples and no repeated real-server timing check.
Three samples are discarded as declared warmups, leaving 10 independent
samples, each averaging eight complete opening pairs. All 104 pairs per system
verified. The output validator rechecks message dimensions, every per-opening
mean and each client total against request preparation plus reply processing.
This is a client-cost diagnostic; it does not repeat the large paired real-server
experiment or replace the full benchmark.

`h32-validation.json` compares this invocation with the archived October 16
height-32 invocation. `prior-source-comparison.json` confirms an identical
benchmark source and all 88 checked library/backend source files. Each
invocation is summarized separately under the current 12-look confidence rule;
the two runs are not pooled to claim a publication result.

In the new check, all request, reply-processing and total client timing series
have corrected relative half-widths below 0.6%. Several small components still
need more samples. One TreePIR indexing call took 183 us, compared with typical
calls around 12--20 us, widening its ten-sample interval to 54.7%. That call is
retained. Removing height 16 therefore fixes the scope error, but does not
prove that height 32 or every other published substep will converge quickly.
The new full run must still satisfy its declared precision rule.

The diagnostic was limited to one new invocation before its results were
available, because the existing source-identical invocation already provided
an independent comparison. `diagnostic-scope.json` records that decision.

## Integrity

`source-run/` contains the frozen manifest, checkpoint, environment and source
hashes, plus the raw result documents for the two large server comparison jobs
and the height-16 client. Their hashes match the checkpoint. The large executable
and build caches are deliberately not part of this archive. `integrity.json`
binds every archived evidence file by size and SHA-256.
