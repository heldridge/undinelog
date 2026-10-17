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

The height-16 all-real-server client workload had incorrectly been included in
the timing manifest. It is now a separately documented, fixed-count correctness
check. The full manifest has 77 jobs, 993 result cells and 457 timing series.
The remaining job definitions and measured quantities are unchanged. The smoke
gate includes the existing `treepir` integration tests, so automatic checks
still reconstruct openings through every real color server on a small tree.

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
