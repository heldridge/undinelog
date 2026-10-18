# Implementation

## Evaluated Systems

The artifact holds two independent Rust crates with the same structure. Each
has three layers.

**Coded array.** `src/coded_array/` implements field arithmetic,
Reed-Solomon reconstruction, the segment layout, and storage. `src/unauth/`
adds client orchestration, claim-bound storage, point and aligned reads,
updates, repair, and threshold changes.

**Authenticated array.** `src/auth/` places a sparse Merkle tree over the
extendable coded array. It maps active tree nodes to coded-array positions,
constructs and verifies point or range openings, validates public update
batches, derives changed labels and field deltas, and applies each
maintainer's local fragment update.

**Append-only log.** `apps/appendOnlyLog/` is an append-only adapter over the
authenticated array. It maps fixed-size batches of canonical 32-byte field
encodings into contiguous authenticated positions, maintains the append
frontier, and implements point and range retrieval, repair, and
reconfiguration workloads.

The two crates differ in the coded array's layout.

- `undine/` (UndineLog) stores, per claim and segment, one primary column and
  one auxiliary row of a polynomial of degree below `kappa` in each of two
  variables. The segment size is `kappa^2`. Doubling the threshold stitches
  four segments into one locally; halving undoes the stitch. A maintainer also
  answers openings at `2^j kappa` from state stored at `kappa`.
- `borg/` (BorgLog) takes the segment size and the threshold as separate
  parameters. The artifact configures it with segment size `kappa^2`.
  Threshold doubling and halving keep the segment size and exchange data with
  helpers.

**Private openings.** `undine/src/private/` serves a private opening of one
position from a maintainer's primary columns. The columns are grouped into
records; a pattern is a group of leaves whose openings read the same records,
and the PIR database has one row per pattern. The answer and the
preprocessing work on the stored records, and the preprocessing follows an
append by patching the hint. The PIR backend is YPIR's SimplePIR variant,
kept as source under [`undine/third_party/`](../undine/third_party/README.md).

The networking implementation in `src/net/` uses length-prefixed binary RPC
messages and registry-derived maintainer assignments. Benchmark paths include
the construction and serialization work identified in the experiment
manifest; network transit latency is not timed.

## Parameters and Storage

| Evaluation term | Implementation field | Meaning |
| --- | --- | --- |
| segment size | `segment_len` or `n_grid` | Field-element capacity of one coded segment |
| threshold | `need` | Number of correct maintainer responses needed to decode |
| primary width | `s` | `segment_len / need` |

At segment size `kappa^2` a claim stores `2 * kappa` field elements per
segment in both layouts: a primary column of `kappa` values and `kappa`
auxiliary values. The authenticated layer uses preorder linearization to map
the active sparse Merkle tree into the growing coded array.

## Append-Only Log Encoding

The default build allocates 131,072 field entries, or 4 MiB, per append slot.
Benchmark features select 1, 2, or 8 MiB slots. A payload must be a multiple
of 32 bytes; each little-endian word must be a canonical BLS12-381
scalar-field encoding.

Benchmark workloads generate their accepted field entries from recorded
seeds.

## Source Map

Paths are relative to `undine/` or `borg/`.

- `src/field.rs`, `src/value.rs`: field and canonical 32-byte value encoding.
- `src/coded_array/`: coding, reconstruction, layout, and storage.
- `src/unauth/`: extendable coded-array operations and maintenance.
- `src/auth/`: sparse-Merkle authentication and client/server orchestration.
- `src/net/`: registry views, wire types, and binary RPC adapters.
- `apps/appendOnlyLog/`: log layout, codec, frontier, client, and server.
- `src/private/` (`undine/` only): private openings of one position.
- `third_party/` (`undine/` only): the PIR backend and its patches.
- `benches/`: the measured benchmark binaries.
- `benches/support/`: shared benchmark kernels.

At the repository root, `scripts/reproduce.py` holds the demand-driven
manifest, execution, statistics, and result rendering.

## Source Provenance

[`provenance/sources.json`](../provenance/sources.json) records where each
crate comes from and the size and SHA-256 of every measured source and
test file.

- `undine/` is the `undinelog` branch of the implementation repository at the
  recorded commit, with the crate manifest reduced to the append-only-log
  application and the measured benchmark targets. It includes the
  `third_party/` sources of that commit.
- `borg/` is the crate of the BorgLog evaluation artifact at the recorded
  commit, with one changed file,
  `benches/bench_append_only_log_range_open_local.rs`. The change makes the
  range benchmark time a one-claim helper's handler for the range request, as
  the UndineLog range benchmark does, and check that the helper's answer is
  its share of the full server's answer.

`python3 scripts/source_provenance.py --check` verifies the recorded hashes,
and `./artifact verify` also checks both crates against the source snapshot of
the packaged reference run.
