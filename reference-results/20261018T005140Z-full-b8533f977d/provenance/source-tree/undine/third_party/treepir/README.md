# TreePIR reference

The balanced coloring and client index-selection code in
`src/private/treepir.rs` is a Rust port of the authors' Color Splitting
Algorithm and SubCSA implementation:

- Repository: https://github.com/PIR-PIXR/TreePIR
- Commit: `930063c5aefc441244abb4890fdf35383f3aa956`
- Source: `CSA/src/CSA.java` and `TreePIR-Indexing/src/SubCSA.java`
- License: MIT, reproduced in `LICENSE`.

The port preserves the reference's stable sorting of equal-size color
classes and its insertion order. Integer arithmetic replaces floating point,
and node IDs and class offsets use 64-bit integers. These changes avoid
the Java implementation's 32-bit node-ID overflow at large heights.

YPIR is the same vendored, compact-packing backend used by Undine in this
repository. Each color class is a conventional rectangular database. The
rectangle minimizes actual query-coefficient plus reply bytes over supported
power-of-two row counts; the packing-key cost is fixed and included in wire
measurements. A query retrieves a row and the client extracts its node locally.

Every class query uses a fresh secret. The backend has a fixed public query
matrix; reusing a secret across *different* queried rows would allow their
query difference to expose those rows. Undine sends the same query to every
maintainer, which does not have this problem.
