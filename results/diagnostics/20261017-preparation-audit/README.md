# October 17 preparation audit

Diagnostic evidence for the preparation work around the timed sections of the
benchmarks. Nothing here is a source of paper values.

## Where the untimed time of the completed full run went

`analyze.py` splits every batch of a run into timed component samples and
untimed work (setup, discarded warmups, checks and verification):

```sh
python3 results/diagnostics/20261017-preparation-audit/analyze.py results/runs/RUN_ID
```

`breakdown-20261016.json` is its output for `20261016T005529Z-full-2e1033e8d8`,
the completed full run of October 16: 15.6 h in all, 10.3 h of timed samples
and 5.2 h untimed. The largest untimed items were:

| Item | Cost | Resolution |
| --- | --- | --- |
| TreePIR server job sampled to 80 rebuilds for a 75 ms stage | 6.3 h of its 7.5 h | precision policy (see `benchmark-validity.md`) |
| Fixtures rebuilt from scratch in every run | about 2 h (85 min for the 1 TiB primary columns, 25 min for 256 GiB) | shared fixture cache |
| Discarded warmup repetitions | 54 min (18 min TreePIR rebuilds, 12 min preprocessing builds, 8 min updates, the rest online) | all kept; see the warmup replay and its full-run check below |
| Repeat batches forced by sub-millisecond timers outside TreePIR | 25 min, mostly process setup repeated per batch | unchanged |

The 1 TiB maintainer job spent 89 min untimed in its first batch, writing its
fixture, and 3.5 min in its second, a cache hit. The stopped run of October 17
spent the same 85 min writing a byte-identical fixture, because the cache
directory was per run. The nine benchmark binaries had identical digests
across the four full runs of October 16 and 17, so a cache keyed by binary
digest and workload is hit across runs whenever the measured binary is
unchanged. `results/fixtures/` was seeded from the October 17 run's entries
(hard links), and a direct invocation of the 1 GiB maintainer job against it
reported `fixture cache hit` with 4.0 s of setup instead of 24.4 s.

## Warmup replay

`replay-1gib-no-warmups/` is one invocation of the unchanged private-opening
executable of the stopped run (`command.json` records the binary and
environment), replaying `private-maintainer-undine-log1gib-p13-k64` with no
discarded repetition and three retained samples per group, pinned to CPU 2.
The first sample of each series against the median of the other two:

| Series | Samples (s) | First / median of the rest |
| --- | --- | --- |
| hint | 0.950, 0.920, 0.924 | 1.031 |
| packing | 2.757, 2.626, 2.648 | 1.046 |
| hint + packing | 3.707, 3.546, 3.572 | 1.042 |
| update hint patch | 0.104, 0.104, 0.106 | 0.993 |
| update packing rebuild | 2.613, 2.602, 2.598 | 1.005 |

`replay-64gib-no-warmups/` repeats this for
`private-maintainer-undine-log64gib-p14-k64` with four retained update
samples, against the shared fixture cache:

| Series | Samples (s) | First / median of the rest |
| --- | --- | --- |
| update hint patch | 0.376, 0.380, 0.382, 0.381 | 0.987 |
| update packing rebuild | 5.209, 5.168, 5.137, 5.227 | 1.008 |

The first build of a process is 3-5% slower than the following ones, ten
times the 0.3% coefficient of variation of the retained preprocessing samples
in the full runs, so preprocessing batches keep their one discarded rebuild.
The first update is within 1.3% of the following ones in both replays, and
the update warmups were dropped on that evidence. The full run below showed
that the replays, each a single process, had missed a cold first update that
appears at 64 GiB and above, so update batches keep their three discarded
repetitions. Online warmups cost seconds and are unchanged.

## Full-run check of the dropped update warmups

The full run `20261017T171200Z-full-4bda1ee20f` ran with no discarded update
repetition and was stopped after 59 of 78 jobs (its checkpoint under
`results/runs/` is the evidence; it supplies no published value). For each
maintainer process, the first retained hint-patch samples divided by the
median of the whole series:

| Job | Processes | First sample / median | Second sample / median |
| --- | --- | --- | --- |
| 1 TiB, pattern 19, kappa 256 | 5 | 4.50, 1.09, 1.08, 1.11, 1.06 | 1.11, 0.99, 0.98, 1.05, 0.98 |
| 64 GiB, pattern 14 | 3 | 1.21, 1.20, 1.01 | 1.01, 1.00, 1.05 |
| 64 GiB, pattern 15 | 2 | 1.19, 1.04 | 1.00, 0.98 |
| 64 GiB, pattern 16 | 2 | 1.14, 1.02 | 1.11, 1.00 |
| 64 GiB, pattern 17 | 1 | 1.00 | 0.98 |
| 256 GiB, pattern 17 | 1 | 1.03 | 1.00 |
| 1 GiB to 16 GiB (six jobs) | 1 each | 0.97 to 1.01 | 0.97 to 1.05 |

The packing-table rebuilds of the same processes were within 1.6% of their
medians in every position. The 1 TiB hint-patch series had 160 samples with
median 324 ms; the one 1456 ms cold sample gave the series a sequential
relative half-width of 6.3% against the 5% target, so the pipeline kept
re-sampling it at about 37 s per repetition. Without that sample the series
met the target at 20 repetitions. Three discarded update repetitions cost
about 8 min per full run and remove the effect. The two replays above, one
process each, did not show it.

## Fixture reuse does not move the timed series

Cache hits already occurred inside every full run, whenever a later batch or
a compatible job reused a fixture written earlier in the same run. For the
same job and process type, the timed series after a hit against the same
series after the in-process build (means of the batch, hit / write):

| Run, job | Series | Hit / write |
| --- | --- | --- |
| Oct 16, 1 TiB maintainer | answer, first pass, decode, request | 1.008, 1.014, 1.017, 0.973 |
| Oct 17, 1 TiB maintainer | update packing, update hint (two batches) | 1.007, 1.003; 1.008, 0.980 |
| Oct 17, 64 GiB maintainer, pattern 16 | update packing, update hint | 1.001, 0.982 |

The second-scale series agree to within 1.7%, and the millisecond series to
within their own batch-to-batch variation. The shared cache changes only
which process writes an entry, not what any process measures: fixtures hold
untimed inputs, and hints, packing tables and every measured operation are
always built in the measuring process.

## Not changed here

Repeat batches driven by timers below a millisecond (the on-the-fly, point
client, reconfiguration and repair jobs) still repeat their process setup per
batch; the reconfiguration job at threshold 64 rebuilds a warmed cluster for
197 s to time 3.6 s of work. The maintainers' update group still pairs the
0.3 s hint patch with the 33 s packing rebuild under one precision target.
Both are measurement-definition changes and are left for a separate decision.
