# Supplementary Tables

The tables below are rendered from the reported run of the paper's
evaluation (the reference run `20261018T005140Z-full-b8533f977d`), by
`undinelog/scripts/render_eval_tables.py` and
`undinelog/scripts/render_supplementary_tables.py` of the implementation
repository. The paper's Section 7 states their numbers in the text and
points here.

## The client's work for one opening

At every threshold on the 4 GiB log and at kappa = 64 on the 8, 16, and
32 GiB logs. *Request* is key generation, the query, and its
serialization. *Private* decrypts the replies and runs Combine and Verify;
*public* runs Combine and Verify on the public replies of the same cIDs.
The kappa replies are correct and are combined directly, a lower bound.
The 3kappa replies are the registry of the protocol of the paper's
Section 5.3, measured twice from the same cIDs, all honest and with kappa
of them lying; Combine corrects in both cases, as the protocol does on
every opening.

| kappa | Log | Request (ms) | kappa honest, private (ms) | kappa honest, public (ms) | 3kappa honest, private (ms) | 3kappa honest, public (ms) | 3kappa with kappa lies, private (ms) | 3kappa with kappa lies, public (ms) |
|---|---|---|---|---|---|---|---|---|
| 8 | 4 GiB | 41.3 | 22.2 | 0.338 | 66.1 | 0.451 | 66.6 | 0.737 |
| 16 | 4 GiB | 37.4 | 44.3 | 0.456 | 132 | 1.32 | 135 | 3.02 |
| 32 | 4 GiB | 38.0 | 44.5 | 0.863 | 136 | 4.56 | 149 | 16.8 |
| 64 | 4 GiB | 35.8 | 107 | 2.24 | 333 | 17.5 | 380 | 64.2 |
| 256 | 4 GiB | 34.0 | 379 | 29.3 | 1,320 | 274 | 2,069 | 1,021 |
| 64 | 8 GiB | 38.0 | 106 | 2.31 | 330 | 18.2 | 379 | 66.6 |
| 64 | 16 GiB | 37.6 | 176 | 2.39 | 535 | 18.7 | 585 | 70.0 |
| 64 | 32 GiB | 41.4 | 176 | 2.45 | 537 | 19.3 | 588 | 72.8 |

## Range openings at kappa = 64

UndineLog (U) and BorgLog (B). *Response* is the size of one helper's
response; *corrupted* is the client correcting kappa corrupted responses
among 3kappa.

| Range (MiB) | Segments | Response (KiB) | Helper U (ms) | Helper B (ms) | Client U (s) | Client B (s) | Client corrupted U (s) | Client corrupted B (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 33.86 | 0.0303 | 0.673 | 0.129 | 0.154 | 0.680 | 0.779 |
| 4 | 64 | 130.0 | 0.0941 | 0.670 | 0.503 | 0.597 | 1.93 | 2.30 |
| 8 | 128 | 258.3 | 0.182 | 0.724 | 1.01 | 1.20 | 2.68 | 3.15 |
| 32 | 512 | 1,029 | 0.716 | 1.20 | 4.41 | 5.13 | 7.47 | 8.48 |
| 128 | 2,048 | 4,113 | 3.36 | 3.67 | 18.5 | 21.6 | 27.5 | 30.6 |

## Private opening of one position by one cID, by threshold and pattern size

Every threshold on the 4 GiB log at the public pattern size gamma, and
kappa = 64 on the 64 GiB log at pattern sizes from two below the public
one, gamma = 16, to one above. *Columns* is the number of columns in a
record.

| kappa | gamma | Columns | Database (MiB) | Request (KiB) | Reply (KiB) | Answer (ms) | Preprocessing size (MiB) | Preprocessing time (s) | Update (s) |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 12 | 129 | 1,020 | 686 | 120 | 252 | 1,280 | 20.8 | 9.90 |
| 16 | 13 | 65 | 512 | 574 | 120 | 190 | 1,280 | 15.6 | 9.07 |
| 32 | 13 | 17 | 256 | 574 | 60.0 | 95.5 | 641 | 7.93 | 4.51 |
| 64 | 14 | 9 | 128 | 518 | 72.0 | 100 | 769 | 7.64 | 5.32 |
| 64 | 14 | 9 | 2,050 | 1,360 | 72.0 | 241 | 769 | 30.1 | 5.58 |
| 64 | 15 | 17 | 2,050 | 910 | 120 | 299 | 1,280 | 32.3 | 8.92 |
| 64 | 16 | 33 | 2,050 | 686 | 228 | 440 | 2,440 | 40.7 | 16.6 |
| 64 | 17 | 65 | 2,050 | 574 | 456 | 699 | 4,870 | 60.1 | 33.0 |

## Stored columns against materialized records

One complete opening per pattern, answered by the same backend on the same
requests at kappa = 64 on the 1 and 4 GiB logs; the replies are identical.

| Log | Database, columns | Database, records | First pass, columns | First pass, records | Answer, columns | Answer, records |
|---|---|---|---|---|---|---|
| 1 GiB | 32.0 MiB | 41.5 MiB | 2.61 ms | 3.39 ms | 49.9 ms | 50.3 ms |
| 4 GiB | 128 MiB | 147 MiB | 9.38 ms | 14.7 ms | 100 ms | 106 ms |

Materialized sizes count logical 32-byte entries and are derived from the
raw shape fields as `patterns * (group_slots * record_bytes +
(slots - group_slots) * 32)`. The original `table_bytes` counter charged
every upper-sibling slot as a whole column instead of one entry. The paper
renderer corrects that accounting from the recorded shapes; the raw runs
and all timings are unchanged. These sizes exclude backend symbol encoding
and matrix padding.

## Answering above the stored threshold

A maintainer that stores kappa answers a point opening and a 4 MiB range
at the opened threshold on the fly (the paper's Proposition 2), against a
maintainer that stores it, with the size of the range response; and the
stitch of its whole fragment to the opened threshold.

| kappa stored | kappa opened | Point on the fly (us) | Point stored (us) | Range on the fly (us) | Range stored (us) | Range size (KiB) | Stitch (ms) |
|---|---|---|---|---|---|---|---|
| 8 | 16 | 11.5 | 8.08 | 1,320 | 484 | 520 | 4.52 |
| 8 | 32 | 16.1 | 7.97 | 1,310 | 193 | 259 | 6.62 |
| 8 | 64 | 24.0 | 7.66 | 1,270 | 89.7 | 130 | 8.24 |
| 16 | 32 | 11.3 | 7.87 | 527 | 195 | 259 | 2.20 |
| 16 | 64 | 15.4 | 7.60 | 554 | 90.0 | 130 | 3.87 |
| 16 | 128 | 22.6 | 7.29 | 573 | 43.1 | 65.6 | 7.15 |
| 32 | 64 | 11.3 | 7.69 | 234 | 90.1 | 130 | 1.85 |
| 32 | 128 | 14.9 | 7.29 | 256 | 43.1 | 65.6 | 5.20 |
| 32 | 256 | 20.9 | 7.01 | 273 | 23.4 | 33.5 | 16.4 |
| 64 | 128 | 10.9 | 7.30 | 112 | 43.8 | 65.6 | 3.73 |
| 64 | 256 | 14.2 | 7.15 | 127 | 24.5 | 33.5 | 15.2 |
| 64 | 512 | 19.9 | 6.77 | 137 | 14.9 | 17.5 | 63.2 |

## Private retrieval of a Merkle opening: TreePIR and UndineLog

TreePIR in its parallel setting of one server per sub-database, and the
paper's Section 5; the price of each row is counted in the paper's
Section 8.

| Property | TreePIR | UndineLog |
|---|---|---|
| No index computed over the tree before the first query (TreePIR: a coloring, Theta(N log log N)) | no | yes |
| The tree grows by appends with nothing of the layout recomputed (TreePIR: the coloring) | no | yes |
| Costs follow the filled part of the tree, not its capacity | no | yes |
| A filled prefix of a far taller tree (TreePIR: a perfect tree) | no | yes |
| The client prepares one query for the whole opening (TreePIR: h, one per class) | no | yes |
| Each server's share of the tree is its own choice | no | yes |
| Opening served when some servers do not answer | no | yes |
| Opening served, and privacy kept, when some servers lie (the paper's Theorem 4) | no | yes |
