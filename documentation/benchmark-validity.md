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
