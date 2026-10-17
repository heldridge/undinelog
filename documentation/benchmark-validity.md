# Benchmark validation before the next full run

The October 16 audit checks whether the optimized implementation and the
revised benchmark harness still measure the intended operations. These
diagnostics are separate from the publication results. Paper values must come
from one completed, validated full run using the committed sources.

## Problems corrected

Cargo had reused a diagnostic executable even though its extra diagnostic
branches were absent from the recorded source. Each run now uses a fresh,
source-bound build directory, records the compiler configuration and executable
hash, and archives the exact source and test files. A regression test reproduces
the stale-file-timestamp case. The harness also removes inherited benchmark and
profiling switches before applying the manifest's settings.

Stopping client component and total timers independently changed the allocation
history seen by serialization. These timers now stay in one sampling group.
Each independent sample averages eight complete opening pairs with fresh keys
and different leaves, retaining every individual timing. A prototype that
repeated tiny components in isolation was rejected because it changed their
cost; its archived sources are labelled as a rejected prototype.

Local public-opening setup could return while a background fetch still held a
reference to its server. Dropping the local owners could then leave a worker to
destroy the large setup state during the client timer. The October 17 fix drops
the setup client, waits for sole ownership, and transfers or destroys that state
on the benchmark thread before measurement. Both Borg and Undine use this
handoff for the affected public-opening setup paths. This removes a possible
source of timing interference; it does not establish that every observed spike
had that cause. The two interrupted full runs are retained only as diagnostics.

## Evidence

- The [setup cleanup check](../results/diagnostics/20261017-setup-lifetime/validation-summary.json)
  runs nine targeted jobs after the ownership fix. The six point-opening jobs
  retain 160 samples each; all ten timing series meet the corrected 5% target,
  with relative half-widths from 0.43% to 1.65%. Output dimensions match the
  preceding run, and each client opening verifies. Range-helper and on-the-fly
  checks exercise the other changed setup paths. These are diagnostics, not
  replacement publication results.
- The [audit record](../results/diagnostics/20261016-benchmark-validity-audit/audit.json)
  identifies the checks, fixes and limitations. The archive retains build logs,
  source hashes, raw timings and both the rejected prototype and final method.
- The [final client diagnostics](../results/diagnostics/20261016-benchmark-validity-audit/opening-batches/preflight-summary.json)
  exercise 104 opening pairs per system at heights 16 and 32. Every opening
  verifies. At height 16, every server's real reply is checked against the
  generated reply for plaintext equality and serialized length. Reported means
  are recomputed from the individual timings, and complete client totals are
  checked against request preparation plus reply processing.
- The [implementation comparison](../results/diagnostics/20261016-maintainer-attribution/summary.json)
  uses the same diagnostic harness, seeded primary entries and message shapes
  with old and new implementations, compiled in separate fresh build directories.
  At the 4 GiB, kappa 64 shape, complete-answer time falls by 10.4--10.7% and
  first-pass time by 54--55%. Varying the number of preceding scans changes
  complete-answer means by less than 0.3%. These inputs are synthetic primary
  columns, not a Merkle tree; this experiment attributes the speedup to the
  implementation and does not supply publication results.

The exact fixture, selected-record, PIR, update and opening tests run in the
smoke gate. The canary exercises representative publication-sized workloads
and checks paired real/generated replies. Neither gate establishes statistical
precision. The audit used the earlier five-look schedule ending at 160 samples.
The current full-run schedule doubles from 10 through 20,480 samples, with the
confidence correction covering all 12 looks. The 95% confidence and 5%
relative-half-width targets are unchanged. Measurements are not overlaid from
earlier runs.

The reviewed audit archives are tracked explicitly. Other exploratory outputs,
compiled diagnostic binaries and fixture caches remain local and ignored.
Archived tool output retains its original paths and integrity hashes; runnable
commands and documentation outside the archives must use portable paths.

## October 17 scope and convergence audit

The full run `20261017T012510Z-full-a1dd1b9f80` was stopped at the author's
request after 76 jobs completed. Its partial measurements are retained for
diagnosis only; the next full run starts with empty samples.

The height-16 all-real-server client workload was initially removed from the
timing manifest because it was not intended for the paper, leaving a
separately documented, fixed-count correctness check. At that point the
manifest had 77 jobs, 993 result cells and 457 timing series. The remaining
job definitions and measured quantities were unchanged. The smoke gate also
included the existing `treepir` integration tests, retaining automatic opening
checks through every real color server on a small tree. Height 16 is now
restored for artifact-only performance results under the revised precision
policy described below.

The [convergence audit](../results/diagnostics/20261017-convergence/README.md)
recomputes stopping decisions from the saved samples. Of the 76 completed jobs,
51 stopped at 10 samples, 12 at 20, five at 40, four at 80 and four at 160.
The height-16 client request and reply-processing totals met the corrected
5% target at the first look, with relative half-widths below 1%. Its tiny
indexing timers continued to miss the target. Their means also differed across
processes; the evidence does not identify a single allocator, cache or
scheduling cause.

The expensive server comparison for claim/class 31 needed 40 samples. Its
preparation totals were already below 0.4% uncertainty at 10 samples, but
UndineLog's approximately 74 ms row-table/upper-node gathering stage had
occasional longer calls, and TreePIR's first pass also needed more samples.
Because gathering is part of the preparation group, those calls caused further
complete builds. This explains the runtime without establishing a defect in
the measured algorithms. The confidence target, stopping schedule and complete
set of samples are preserved; no outlier filtering or isolated repetition of
tiny client components is introduced.

A new height-32 diagnostic also verified all 104 opening pairs per system.
Its request, processing and total client intervals were below 0.6% after
10 retained samples. Several component timers still missed 5%; a single
183 us TreePIR indexing call widened that interval substantially. This confirms
that removing height 16 does not guarantee fast convergence for the remaining
suite. Both the result and the slow observation are retained in the audit.

## Precision follows the reported operation

The subsequent scope change retains every TreePIR comparison substep in its
original execution context, but stops sampling based on the reported totals.
Client request preparation, reply processing and total time retain the corrected
5% target. Index selection and other client substeps are diagnostics; the paper
no longer reports index selection as a separate timing row.

Server filling includes cell conversion and gathering. Hint plus packing is
also measured as a paired sum from each rebuild. These two costs, color
enumeration, full preparation and complete answers retain the precision target.
The finer preparation stages and first pass remain diagnostic measurements.
No timing boundaries, keys, input selection, opening checks, warmups, sample
schedule or retained observations change. In particular, a noisy 75 ms gather
cannot force a complete rebuild after its containing costs have converged.

That precision change produced 77 jobs, 1,001 result cells and 465 timing series:
434 precision-required results and 31 diagnostics. Diagnostics have explicit
flags and their achieved confidence intervals, even when they miss 5%.
The same declaration applies to both systems before the next fresh full run;
it does not selectively discard slow observations or reuse partial results.

The [targeted checks](../results/diagnostics/20261017-total-precision/checks.json)
validate the new aggregate fields in fresh server builds, an online-only batch
and complete client openings. The pipeline regressions check that a noisy
diagnostic cannot restart a converged group, a noisy required total still does,
and missing diagnostics, incorrect paired sums or changed precision declarations
are rejected. The [stopping replay](../results/diagnostics/20261017-total-precision/stopping-replay.json)
applies the new scope to the archived height-32 diagnostic and first server
batches: their totals already meet the target, while the noisy substep intervals
remain visible. This replay is a policy check, not a new performance result.

## Height-16 results in the artifact

The full manifest again includes the height-16 all-real-server client job,
publishing its totals and communication only in the artifact. It uses the
same precision policy as height 32: request preparation, reply processing and
total client CPU control repetitions; component timers remain diagnostics.
The job still checks every real reply against its generated counterpart and
verifies each opening. The smoke gate covers its real-server path, and the
canary exercises the height-16 workload.

The current manifest has 78 jobs, 1,037 result cells and 484 timing series:
440 precision-required results and 44 diagnostics. Outputs are managed by the
artifact runner under `results/runs/`, so the standalone Cargo command and its
crate-local output are no longer needed. The next full run starts fresh;
neither the earlier stopped run nor the standalone correctness check supplies
its published samples. The paper comparison remains at height 32.

## October 17 preparation audit

The completed full run `20261016T005529Z-full-2e1033e8d8` took 15.6 h: 10.3 h
of timed samples and 5.2 h of untimed setup, warmups and verification. The
[preparation audit](../results/diagnostics/20261017-preparation-audit/README.md)
breaks the untimed time down per job. Beyond the TreePIR over-sampling that
the precision policy above removes, two costs were avoidable without changing
any measured operation.

Fixtures were rebuilt from scratch in every run because the cache lived in the
run's own directory: 85 minutes for the 1 TiB maintainer's primary columns,
25 minutes for the 256 GiB ones, and about 2 h in all, repeated identically by
the stopped run of October 17. The cache now lives in `results/fixtures/`,
shared by every run. Its keys already include the benchmark executable's
digest and the complete workload, and loads verify the complete-file checksum,
so an entry is reused exactly when the measured binary and workload are
unchanged. Builds are reproducible: the nine benchmark binaries had identical
digests across the four full runs of October 16 and 17.

Warmups were checked by replaying the 1 GiB maintainer job with no discarded
repetition and three retained samples per group, and the 64 GiB job at
pattern 14 with four retained update samples. The first build of the
process was 3.1% slower for the hint, 4.6% for the packing tables and 4.2%
for their sum than the following builds, so preprocessing batches keep their
one discarded rebuild. The first append-and-update was within 1.3% of the
following ones in both replays (hint patch 0.993 and 0.987, packing rebuild
1.005 and 1.008), and the update warmups were dropped on that evidence. The
full run `20261017T171200Z-full-4bda1ee20f` then contradicted the replays:
the first hint patch of a process was 14-21% slower than the series median
in three of the four 64 GiB maintainer jobs and 4.5 times slower at 1 TiB,
where the second one was 11% slower as well. That single cold sample kept
the 1 TiB hint patch series from converging through 160 repetitions, so the
run was stopped, update batches again discard three repetitions, and the
next full run starts fresh. The stopped run's samples are the evidence and
supply nothing else. Timed series measured after a fixture cache hit agree
with the same series after an in-process fixture build to within 1.7% in the
full runs that discarded three update warmups, so the shared cache stays.
Online batches keep three, which cost seconds. Update timings now carry the
sample mode `update`; the manifest, checkpoint and summary record the
policy, and a resumed run must agree with it.
