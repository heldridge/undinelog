# Supplementary Tables

The tables below are rendered from the reported run of the paper's
evaluation (the reference run `20261009T161432Z-full-6ab859da3a` with the
client jobs of `20261011T230101Z-jobs-abf7ac3d05` overlaid), by
`undinelog/scripts/render_eval_tables.py` of the implementation repository.
The paper's Section 7 states their numbers in the text and points here.

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
| 8 | 4 GiB | 48.4 | 26.9 | 0.381 | 80.1 | 0.506 | 80.1 | 0.819 |
| 16 | 4 GiB | 43.9 | 54.6 | 0.520 | 162 | 1.50 | 165 | 3.41 |
| 32 | 4 GiB | 41.8 | 51.9 | 0.936 | 157 | 4.97 | 170 | 18.1 |
| 64 | 4 GiB | 38.7 | 123 | 2.44 | 380 | 19.3 | 429 | 68.6 |
| 256 | 4 GiB | 36.2 | 423 | 30.5 | 1,463 | 291 | 2,247 | 1,072 |
| 64 | 8 GiB | 42.1 | 125 | 2.50 | 387 | 20.2 | 438 | 72.5 |
| 64 | 16 GiB | 41.0 | 202 | 2.53 | 619 | 20.4 | 672 | 73.7 |
| 64 | 32 GiB | 46.2 | 201 | 2.62 | 614 | 21.0 | 669 | 76.9 |

## Range openings at kappa = 64

UndineLog (U) and BorgLog (B). *Response* is the size of one helper's
response; *corrupted* is the client correcting kappa corrupted responses
among 3kappa.

| Range (MiB) | Segments | Response (KiB) | Helper U (ms) | Helper B (ms) | Client U (s) | Client B (s) | Client corrupted U (s) | Client corrupted B (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 33.86 | 0.0323 | 0.679 | 0.130 | 0.157 | 0.687 | 0.791 |
| 4 | 64 | 130.0 | 0.0973 | 0.680 | 0.504 | 0.611 | 1.95 | 2.35 |
| 8 | 128 | 258.3 | 0.185 | 0.735 | 1.02 | 1.23 | 2.71 | 3.23 |
| 32 | 512 | 1,029 | 0.731 | 1.20 | 4.34 | 5.17 | 7.52 | 8.66 |
| 128 | 2,048 | 4,113 | 3.42 | 3.62 | 18.6 | 22.0 | 27.7 | 31.4 |

## Private opening of one position by one cID, by threshold and pattern size

Every threshold on the 4 GiB log at the public pattern size gamma, and
kappa = 64 on the 64 GiB log at pattern sizes from two below the public
one, gamma = 16, to one above. *Columns* is the number of columns in a
record.

| kappa | gamma | Columns | Database (MiB) | Request (KiB) | Reply (KiB) | Answer (ms) | Preprocessing size (MiB) | Preprocessing time (s) | Update (s) |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 12 | 129 | 1,020 | 686 | 120 | 351 | 1,280 | 38.9 | 11.3 |
| 16 | 13 | 65 | 512 | 574 | 120 | 238 | 1,280 | 24.8 | 10.5 |
| 32 | 13 | 17 | 256 | 574 | 60.0 | 125 | 641 | 12.5 | 6.12 |
| 64 | 14 | 9 | 128 | 518 | 72.0 | 119 | 769 | 10.5 | 6.56 |
| 64 | 14 | 9 | 2,050 | 1,360 | 72.0 | 425 | 769 | 64.6 | 6.42 |
| 64 | 15 | 17 | 2,050 | 910 | 120 | 468 | 1,280 | 67.7 | 10.6 |
| 64 | 16 | 33 | 2,050 | 686 | 240 | 642 | 2,560 | 77.6 | 20.2 |
| 64 | 17 | 65 | 2,050 | 574 | 456 | 928 | 4,870 | 97.1 | 38.5 |

## Stored columns against materialized records

One complete opening per pattern, answered by the same backend on the same
requests at kappa = 64 on the 1 and 4 GiB logs; the replies are identical.

| Log | Database, columns | Database, records | First pass, columns | First pass, records | Answer, columns | Answer, records |
|---|---|---|---|---|---|---|
| 1 GiB | 32.0 MiB | 41.5 MiB | 5.14 ms | 12.5 ms | 59.3 ms | 62.6 ms |
| 4 GiB | 128 MiB | 147 MiB | 20.7 ms | 42.9 ms | 119 ms | 134 ms |

Materialized sizes count logical 32-byte entries and are derived from the
raw shape fields as `patterns * (group_slots * record_bytes +
(slots - group_slots) * 32)`: 41.5 and 147.25 MiB before rounding.
The original `table_bytes` counter charged every upper-sibling slot as a
whole column instead of one entry. The paper renderer corrects that
accounting from the recorded shapes; the raw runs and all timings are
unchanged. These sizes exclude backend symbol encoding and matrix padding.

## Answering above the stored threshold

A maintainer that stores kappa answers a point opening and a 4 MiB range
at the opened threshold on the fly (the paper's Proposition 2), against a
maintainer that stores it, with the size of the range response; and the
stitch of its whole fragment to the opened threshold.

| kappa stored | kappa opened | Point on the fly (us) | Point stored (us) | Range on the fly (us) | Range stored (us) | Range size (KiB) | Stitch (ms) |
|---|---|---|---|---|---|---|---|
| 8 | 16 | 12.5 | 8.78 | 1,340 | 498 | 520 | 4.52 |
| 8 | 32 | 17.8 | 8.79 | 1,310 | 200 | 259 | 6.62 |
| 8 | 64 | 26.3 | 8.52 | 1,280 | 92.9 | 130 | 8.29 |
| 16 | 32 | 12.2 | 8.57 | 533 | 199 | 259 | 2.22 |
| 16 | 64 | 17.0 | 8.43 | 560 | 94.5 | 130 | 3.91 |
| 16 | 128 | 24.7 | 8.11 | 569 | 44.5 | 65.6 | 7.22 |
| 32 | 64 | 12.4 | 8.64 | 239 | 94.4 | 130 | 1.88 |
| 32 | 128 | 16.4 | 8.19 | 255 | 43.5 | 65.6 | 5.28 |
| 32 | 256 | 22.8 | 8.01 | 270 | 23.4 | 33.5 | 16.7 |
| 64 | 128 | 12.2 | 8.34 | 113 | 44.7 | 65.6 | 3.79 |
| 64 | 256 | 15.7 | 8.08 | 127 | 23.9 | 33.5 | 15.4 |
| 64 | 512 | 21.5 | 8.04 | 137 | 15.1 | 17.5 | 64.4 |

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
