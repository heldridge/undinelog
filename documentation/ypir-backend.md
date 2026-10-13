# The YPIR Backend

The paper's Section 6 instantiates the private opening with the SimplePIR
variant of YPIR and states the parameters, the pattern-size rule, and the
update of the preprocessing after an append (Proposition 5). This document
holds the details the paper points to: the database layout, the request and
the reply, the preprocessing, the mechanics and the cost of the patch, and
the other changes of a maintainer's preprocessing. The code is
`undine/src/private/` (`pir.rs` for the backend, `opening.rs` for the
factored answer, `bulk.rs` for the bulk builder of the evaluation).

## Database

The 256-bit encodings of the entries of one slot are written as one string
of symbols of 14 bits: a group, one segment of `kappa` entries, is a vector
of `ceil(256 kappa / 14)` symbols and a cell slot one of 19. The matrix `D`
of the materialized records has one row per pattern, padded to a power of
two of at least `2^11` rows, and one column per symbol of a record, padded
to a multiple of `2^11` columns. The maintainer stores its columns, a
contiguous copy of its entries for the about `2R` nodes at height `gamma`
and above so that the first pass streams over them, and the map from
patterns to groups and cells; it never forms `D`. A column of `D` lists,
for one slot and one symbol position, that symbol of the group or cell each
pattern has in the slot.

The tables of the evaluation have at most `2^16` rows and `2^17` columns,
within the `2^18` by `2^18` for which YPIR's moduli bound the decoding
error of one reply ciphertext by `2^-40`.

## Request and reply

The query has one coefficient per pattern, 7 bytes, and the client adds
the key material with which the server packs its reply, about half a
mebibyte. The protocol of the paper's Section 5.3 generates this key afresh
for every opening; a client may also keep one key per maintainer and reuse
it across openings, which the backend supports and which removes the key
material from every request after the first. Query privacy is defined for
any number of queries under one key, so a kept key reveals no more about
the positions than fresh ones; it links the openings of the client at that
maintainer, which the channel the request arrives on may link as well.

The server computes the first pass by the factored answer (the paper's
equation (10)) and packs every `2^11` entries of the result into one
ciphertext of 12 KiB, so a reply has `ceil(((m+1)|c| + 19 U) / 2^11)`
ciphertexts for a record of `m+1` groups and `U` cells: three on the 1 GiB
log at `kappa = 64`, where `m+1 = 5`, `|c| = 1171`, and `U = 12`.

## Preprocessing

The backend preprocesses the database in two steps. The *hint* is
`Hint = A D`, where the public matrix `A` has one column `a_r` of `2^11`
values for every row `r` of the table and is derived from a public seed.
The *packing tables* are derived from the hint, one set per reply
ciphertext, from the hint of the `2^11` columns that the ciphertext packs;
a set holds 96 MiB, the key-switching work of the packing applied to the
hint in advance, and the hint of the ciphertext's columns is 32 MiB. The
maintainer computes the hint column by column from its stored entries.

## The pattern size

A coefficient of the query is `beta_q = 7` bytes. An entry of 32 bytes is
18.3 symbols, and a reply ciphertext carries `2^11` symbols in 12 KiB, so
an entry costs `beta_c = 110` bytes in the reply. The query is
`7 l_t / 2^gamma` bytes and the columns of the subtree are
`110 * 2^(gamma+1) / kappa` bytes, equal at
`2^gamma = sqrt(0.032 l_t kappa)`. In powers of two the backend uses

    gamma(l_t, kappa) = max(2 log2(kappa) - 1,
                            floor((ceil(log2(l_t)) + log2(kappa) - 5) / 2)),

the smallest pattern or the balance, whichever is larger. At `kappa = 64`
a pattern has `2^13` to `2^17` leaves on logs of `2^25` to `2^33` leaves,
and its record holds 5, 9, 17, 33, and 65 columns; at `kappa = 8` on the
4 GiB log it has `2^12` leaves and 129 columns of eight entries, since the
smaller segment of a smaller threshold makes a subtree span more columns
and the balance stops at a smaller pattern; and from `kappa = 128` on the
1 GiB log the pattern is the smallest one, with two columns. The request
and the reply both scale with `sqrt(l_t / kappa)`. A single server with
the whole tree is the case `kappa = 1` with a group of its choice. The
server's cost of a longer reply is its packing: time and one set of tables
per reply ciphertext.

## The patch after an append

Every update of UndineLog is an append, with three consequences.

1. Groups and cells keep their places: for a fixed pattern size a cell
   keeps its position, so an append adds groups, nodes, and patterns at
   the end, changes among the old entries only those that answer reads of
   the labels above the new leaves, and gives an old pattern at most a
   sibling that stops being a default, in an empty cell slot, or one new
   slot when `ceil(log2(l_t))` grows.
2. A change touches ranges of rows: the patterns that hold a given group
   are at most two neighbours, and the patterns that read a given node are
   the contiguous range below its sibling, so the rows in which a slot
   changes form few ranges.
3. The hint is linear: if the data in slot `u` changes from `c` to `c'` on
   the rows `[r_0, r_1)`, the hint of the column of symbol `k` of that slot
   changes by `(c'_k - c_k) * sum_{r_0 <= r < r_1} a_r`.

The maintainer compares the old and the new group of every slot and
collects the *runs*. A run of slot `u` is a maximal range of rows that had
the same data `c` in the slot before the append, an empty slot counting as
zero, and have the same data `c'` after it, where `c' != c` as vectors of
symbols: the slot holds another group or cell, or the same one with
changed entries. For each run it computes the sum once and adds its
multiples to the hint of the columns whose symbols differ. It then derives
the packing tables from the patched hint.

**Proposition 5 (the update equals a fresh preprocessing).** After the
update, the hint and the packing tables of a maintainer equal those
computed from scratch on the fragment of `M_{t+1}`.

*Proof.* Let `D` and `D'` be the tables before and after the append, with
`D` padded by zero rows and columns to the shape of `D'`. By (1) and (2) a
group, a node, and a slot keep their places, so column `(u, k)` of
`D' - D` is, on every run of slot `u`, the constant `c'_k - c_k` of that
run, and zero elsewhere. The matrix `A` of the larger table extends that
of the smaller one: the columns `a_r` of the old rows are derived from the
same seed and are unchanged. Hence `A D' = A D + A (D' - D)`, and the
second term is the sum of the patches over the runs. The packing tables
are a function of the hint.

## The update's cost and the other changes of the fragment

The patch reads the changed entries only: an append of `a` entries adds
about `2a / kappa^2` groups, `a / 2^gamma` patterns, and as many nodes
above the pattern roots, each entering a bounded number of runs, and the
changed labels on the frontier path enter one run each, so the patch is
proportional to the appended data, with a term per height of the tree and
one pass over the hint of a record's columns, where a fresh hint passes
over every column of the table. The packing tables are derived again after
every update, since a table is not a linear function of the hint's
columns, at one derivation per reply ciphertext whatever the log length; a
maintainer that receives several batches before it is asked derives them
once for all. The paper's Section 7.3 measures the three.

In steady state at 1 TiB the update of a full 4 MiB batch takes 40.2 s of a
75 s block interval, more than half a core, almost all of it the
derivation of the packing tables (one second per reply ciphertext, 40
ciphertexts); the hint patch itself is 107 to 282 ms across the logs.

When the number of patterns passes a power of two `D` gains rows, and when
`ceil(log2(l_t))` grows it gains a slot; both are covered by Proposition 5,
and the hint keeps its values. When the log length passes a threshold of
the pattern size the patterns double: the columns do not change, the
incidence map does, and the maintainer preprocesses its columns under the
new patterns, at most once per quadrupling of the log; since the step is
at a public log length, a maintainer builds the preprocessing of the next
pattern size ahead of it from the columns it holds and patches both until
the step, holding one more preprocessing for that stretch. A stitch or an undo
changes the threshold and with it every column, and the maintainer
preprocesses the fragment at the new threshold before it advertises it.

## Alternatives considered

Letting the maintainer evaluate the coded array under encryption needs
exact homomorphic arithmetic over the 255-bit coding field, and a backend
in the style of SealPIR needs a cache of derived plaintexts of the database
to be fast, which every append would invalidate. A client-held hint
(SimplePIR without YPIR's server-side hint handling) would have to be
refreshed for every maintainer after every append.
