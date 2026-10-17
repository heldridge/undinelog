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
| Discarded warmup repetitions | 54 min (18 min TreePIR rebuilds, 12 min preprocessing builds, 8 min updates, the rest online) | update warmups dropped; rebuild warmup kept |
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

The first build of a process is 3-5% slower than the following ones, ten
times the 0.3% coefficient of variation of the retained preprocessing samples
in the full runs, so preprocessing batches keep their one discarded rebuild.
The first update is within 1% of the following ones, so update batches
discard none of their former three warmups, which cost 35 s each at the large
log sizes. Online warmups cost seconds and are unchanged.

## Not changed here

Repeat batches driven by timers below a millisecond (the on-the-fly, point
client, reconfiguration and repair jobs) still repeat their process setup per
batch; the reconfiguration job at threshold 64 rebuilds a warmed cluster for
197 s to time 3.6 s of work. The maintainers' update group still pairs the
0.3 s hint patch with the 33 s packing rebuild under one precision target.
Both are measurement-definition changes and are left for a separate decision.
