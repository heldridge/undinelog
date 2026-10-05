# Third-party sources

This directory holds the PIR backend used by private openings. Both crates are
MIT licensed and are kept here as source so that a build needs only this
repository and the crates.io registry.

| Directory | Upstream | Commit |
| --- | --- | --- |
| `ypir/` | https://github.com/menonsamir/ypir | `a73e550a469605c2fa23945a4d96c20fb2f4a839` |
| `spiral-rs/` | https://github.com/menonsamir/spiral-rs | `6929441` (the revision YPIR pins) |

`patches/ypir.diff` and `patches/spiral-rs.diff` are the complete differences
from those commits. They contain:

- **Stable toolchain.** The `stdarch_x86_avx512` feature gate and YPIR's
  pinned nightly toolchain file are removed; the crates build with the
  toolchain of this repository.
- **A scalar path and an AVX2 path next to the AVX-512 path.** Every
  routine that uses AVX-512 intrinsics keeps its upstream body under
  `cfg(target_feature = "avx512f")` and has a scalar body otherwise. The
  scalar dot product reduces every 2^15 terms, so its 64-bit sums stay in
  range for any database height. On x86-64 hosts without AVX-512, the
  multiply kernels of the response packing (`packing.rs`, module `avx2`)
  run AVX2 code selected at run time, and the two additions of each
  packing step are fused with their scalar multiplications.
- **Compact packing tables.** A response-packing table stores its
  gadget-decomposed polynomials in condensed form at their real size
  (`packing::Condensed`: the two CRT residues of a coefficient in one
  word, `poly_len` words per polynomial), where upstream keeps a matrix of
  twice that size with a zero half, and `fast_multiply_condensed` reads
  that layout. `factored::offline_from_hint` derives the table of each
  output from its prepacked hint columns and releases those columns, which
  the online packing never reads. A table is 96 MiB per response
  ciphertext instead of about 320 MiB; the responses are unchanged, byte
  for byte.
- **Local dependency.** YPIR depends on `../spiral-rs`.
- **Column-based building blocks.** `ypir/src/factored.rs` runs the SimplePIR
  variant's preprocessing, response packing, query generation, and decoding
  on a database described by its columns, following `YServer` step for step.
  Its test checks that the hint and the response equal `YServer`'s byte for
  byte, that a hint patched with `hint_of_row_range` and
  `add_scaled_to_hint_column` equals a recomputed one, and that a taller
  database keeps the public query of its first rows.
  `packing_table_of_output` derives the response-packing tables of one reply
  ciphertext; the test checks them against the tables of the whole hint.
  `YServer::perform_offline_precomputation_simplepir` returns its values with
  the lifetime of the parameters.
- **Quiet builds.** Both crates start with `#![allow(warnings)]`.
- YPIR's `rodeo/` directory (benchmark scripts and data) is left out.

The AVX-512 path is selected at compile time, so it is used only when the
build enables it, for example with `RUSTFLAGS="-C target-cpu=native"` on a
machine that has AVX-512.

## Tests

`third_party` is excluded from the workspace. YPIR's own tests run from its
directory, one at a time because two of them build 8 GB databases:

```sh
cd third_party/ypir && cargo test --release -- --test-threads=1
```
