# Evaluation result groups

Values are reviewer-facing renderings. Machine-readable source values, units, samples, and confidence intervals are in `../summary.json` and `../summary.tsv`.

## Append processing and fragment size

Result-group ID: `tab:append`. CSV: `tab-append.csv`.

One row per threshold, with Seg = kappa^2. Append cells are one maintainer's mean processing time for a 4 MiB batch in milliseconds. Fragment cells are the mean newly materialized fragment size per batch in MiB.

| panel | row | segment | kappa | undine-append | undine-fragment | borg-append | borg-fragment |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | k8 | 64 | 8 | 163.6 | 2.000 | 262.0 | 2.000 |
|  | k16 | 256 | 16 | 161.7 | 1.000 | 230.8 | 1.000 |
|  | k32 | 1024 | 32 | 161.1 | 0.500 | 224.2 | 0.500 |
|  | k64 | 4096 | 64 | 164.5 | 0.250 | 235.7 | 0.250 |

## Point openings

Result-group ID: `tab:point-open`. CSV: `tab-point-open.csv`.

Helper work is microseconds for one helper's response construction and serialization; response cells are its serialized bytes. Client cells are Combine+Verify in milliseconds from kappa correct responses, and from 3*kappa responses of which kappa are corrupted.

| panel | row | segment | kappa | undine-helper | undine-response | undine-client | undine-client-correcting | borg-helper | borg-response | borg-client | borg-client-correcting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | k8 | 64 | 8 | 8.8 | 1251 | 0.224 | 0.622 | 178.8 | 1251 | 0.276 | 0.700 |
|  | k16 | 256 | 16 | 8.2 | 1251 | 0.322 | 2.941 | 196.3 | 1251 | 0.371 | 3.030 |
|  | k32 | 1024 | 32 | 8.2 | 1251 | 0.610 | 14.329 | 232.6 | 1251 | 0.662 | 14.404 |
|  | k64 | 4096 | 64 | 7.9 | 1251 | 1.634 | 55.147 | 309.7 | 1251 | 1.687 | 55.300 |

## Range openings at kappa = 64

Result-group ID: `tab:range-open`. CSV: `tab-range-open.csv`.

Per-helper response size is KiB. Helper work is one helper's answer to the range request and its serialization, in milliseconds. Client work is seconds, from 192 responses; the corrupt case has 64 corrupted responses among them.

| panel | row | range-mib | full-segment-reads | point-reads | undine-per-helper-kib | undine-helper-work | undine-clean-client | undine-corrupt-client | borg-per-helper-kib | borg-helper-work | borg-clean-client | borg-corrupt-client |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | range-1mib | 1 | 16 | 41 | 33.9 | 0.030 | 0.129 | 0.680 | 33.9 | 0.673 | 0.154 | 0.779 |
|  | range-4mib | 4 | 64 | 37 | 130.0 | 0.094 | 0.503 | 1.931 | 130.0 | 0.670 | 0.597 | 2.305 |
|  | range-8mib | 8 | 128 | 35 | 258.3 | 0.182 | 1.006 | 2.683 | 258.3 | 0.724 | 1.202 | 3.149 |
|  | range-32mib | 32 | 512 | 31 | 1028.9 | 0.716 | 4.409 | 7.466 | 1028.9 | 1.200 | 5.128 | 8.482 |
|  | range-128mib | 128 | 2048 | 27 | 4112.7 | 3.361 | 18.488 | 27.467 | 4112.7 | 3.671 | 21.609 | 30.644 |

## Repair and reconfiguration

Result-group ID: `tab:maintenance`. CSV: `tab-maintenance.csv`.

The panel is the layout. All quantities are normalized per 4 MiB appended to the evaluated 12 MiB prefix. Times are milliseconds; fragment, download, and state values are MiB. Double and halve move the threshold between kappa and 2*kappa. The diagonal halve columns are the UndineLog undo from source state at kappa; the segment-double column is BorgLog's local segment-and-kappa double.

| panel | row | segment | kappa | fragment | repair-receiver | repair-helper | repair-download | double-receiver | double-helper | double-download | state-after-double | halve-receiver | halve-helper | halve-download | halve-diagonal-receiver | halve-diagonal-download | segment-double-receiver |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| undine | undine:k8 | 64 | 8 | 2.000 | 36.944 | 3.792 | 2.000 | 1.467 | -- | 0.000 | 1.000 | 16.388 | 0.712 | 1.000 | 2.211 | 1.000 | -- |
| undine | undine:k16 | 256 | 16 | 1.000 | 21.076 | 1.311 | 1.000 | 0.627 | -- | 0.000 | 0.501 | 12.707 | 0.311 | 0.501 | 0.891 | 0.501 | -- |
| undine | undine:k32 | 1024 | 32 | 0.501 | 15.432 | 0.543 | 0.501 | 0.309 | -- | 0.000 | 0.251 | 11.148 | 0.150 | 0.251 | 0.397 | 0.251 | -- |
| undine | undine:k64 | 4096 | 64 | 0.251 | 14.753 | 0.261 | 0.251 | 0.192 | -- | 0.000 | 0.128 | 11.035 | 0.081 | 0.128 | 0.238 | 0.128 | -- |
| borg | borg:k8 | 64 | 8 | 2.000 | 488.521 | 29.705 | 2.000 | 90.797 | 11.687 | 1.875 | 2.500 | 129.674 | 13.301 | 1.375 | -- | -- | 66.949 |
| borg | borg:k16 | 256 | 16 | 1.000 | 167.117 | 9.309 | 1.000 | 69.706 | 3.680 | 0.969 | 1.250 | 24.221 | 4.452 | 0.719 | -- | -- | 36.881 |
| borg | borg:k32 | 1024 | 32 | 0.501 | 110.598 | 3.361 | 0.501 | 64.937 | 1.931 | 0.493 | 0.626 | 18.267 | 1.788 | 0.368 | -- | -- | 29.339 |
| borg | borg:k64 | 4096 | 64 | 0.251 | 79.220 | 1.278 | 0.251 | 70.565 | 2.829 | 0.249 | 0.314 | 16.827 | 0.703 | 0.187 | -- | -- | 30.183 |

## UndineLog openings served above the stored threshold

Result-group ID: `tab:on-the-fly-open`. CSV: `tab-on-the-fly-open.csv`.

One maintainer stores a 12 MiB log at the stored threshold and answers at the opened threshold. On-the-fly cells are that answer; stored cells are the answer of a maintainer that stores the log at the opened threshold. Both include response serialization and are microseconds; response sizes are KiB. Stitch-state is the time to stitch the maintainer's whole state to the opened threshold, in milliseconds.

| panel | row | stored-kappa | open-kappa | point-on-the-fly | point-stored | point-bytes | range-1mib-on-the-fly | range-1mib-stored | range-1mib-kib | range-4mib-on-the-fly | range-4mib-stored | range-4mib-kib | stitch-state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | k8-open8 | 8 | 8 | -- | 8.6 | 1251 | -- | 355.1 | 263.9 | -- | 1488.8 | 1055.8 | -- |
|  | k8-open16 | 8 | 16 | 11.5 | 8.1 | 1251 | 335.2 | 122.7 | 130.2 | 1318.4 | 483.8 | 520.1 | 4.525 |
|  | k8-open32 | 8 | 32 | 16.1 | 8.0 | 1251 | 333.2 | 51.6 | 65.5 | 1307.5 | 193.4 | 258.6 | 6.624 |
|  | k8-open64 | 8 | 64 | 24.0 | 7.7 | 1251 | 340.9 | 28.3 | 33.7 | 1269.2 | 89.7 | 129.8 | 8.236 |
|  | k16-open16 | 16 | 16 | -- | 8.4 | 1251 | -- | 126.3 | 130.2 | -- | 491.6 | 520.1 | -- |
|  | k16-open32 | 16 | 32 | 11.3 | 7.9 | 1251 | 135.5 | 50.7 | 65.5 | 527.1 | 195.2 | 258.6 | 2.202 |
|  | k16-open64 | 16 | 64 | 15.4 | 7.6 | 1251 | 149.8 | 28.1 | 33.7 | 553.9 | 90.0 | 129.8 | 3.873 |
|  | k16-open128 | 16 | 128 | 22.6 | 7.3 | 1251 | 163.8 | 16.2 | 17.7 | 573.0 | 43.1 | 65.6 | 7.154 |
|  | k32-open32 | 32 | 32 | -- | 8.3 | 1251 | -- | 53.7 | 65.5 | -- | 202.5 | 258.6 | -- |
|  | k32-open64 | 32 | 64 | 11.3 | 7.7 | 1251 | 68.6 | 28.8 | 33.7 | 234.3 | 90.1 | 129.8 | 1.854 |
|  | k32-open128 | 32 | 128 | 14.9 | 7.3 | 1251 | 76.4 | 16.5 | 17.7 | 256.1 | 43.1 | 65.6 | 5.202 |
|  | k32-open256 | 32 | 256 | 20.9 | 7.0 | 1251 | 88.3 | 11.2 | 9.6 | 272.5 | 23.4 | 33.5 | 16.428 |
|  | k64-open64 | 64 | 64 | -- | 8.0 | 1251 | -- | 29.1 | 33.7 | -- | 90.7 | 129.8 | -- |
|  | k64-open128 | 64 | 128 | 10.9 | 7.3 | 1251 | 37.4 | 16.3 | 17.7 | 111.6 | 43.8 | 65.6 | 3.725 |
|  | k64-open256 | 64 | 256 | 14.2 | 7.2 | 1251 | 45.9 | 11.4 | 9.6 | 127.2 | 24.5 | 33.5 | 15.219 |
|  | k64-open512 | 64 | 512 | 19.9 | 6.8 | 1251 | 125.1 | 8.5 | 16.1 | 136.7 | 14.9 | 17.5 | 63.214 |

## Private openings of one position: one claim of a maintainer

Result-group ID: `tab:private-maintainer`. CSV: `tab-private-maintainer.csv`.

One claim answers from its primary columns, grouped into blocks of the public grouping of the log length (one segment per block at 1 GiB, doubling every two levels) unless the row gives another block size. A log of 1 GiB has 2^25 leaves. The record of a pattern is the two blocks of its subtree and one entry per level above. Database is the bytes of the primary columns; state adds the row table and the preprocessing; both are MiB. Request and reply are serialized KiB. The client's request (key, query, serialization), the first pass, the complete answer from serialized request to serialized reply, and the client's decoding of one reply are milliseconds on one core. Preprocessing is measured in seconds on one core: each repetition rebuilds the entire hint, including padding, and all response-packing tables. The two stages and their sum are reported separately; preprocessing size is MiB. Tree construction, fragment encoding and conversion to PIR cells are outside these timers. Each rebuild serves a checked opening. The update cells follow a 4 MiB append: the hint patch in milliseconds and the packing tables in seconds.

| panel | row | kappa | log-gib | record-kib | database | state | preprocess-size | request | reply-kib | client-request-ms | first-pass | answer | client-decode-ms | preprocess-core-s | preprocess-hint-s | preprocess-packing-s | update-hint-ms | update-packing-s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| log | log:k64-log1gib-p13 | 64 | 1 | 2.0 | 32.0 | 421.7 | 384.6 | 490.0 | 36.0 | 33.8 | 2.6 | 49.9 | 0.8 | 3.5 | 0.9 | 2.6 | 107.7 | 2.6 |
| log | log:k64-log4gib-p14 | 64 | 4 | 2.0 | 128.0 | 916.8 | 769.1 | 518.0 | 72.0 | 34.8 | 9.4 | 100.0 | 1.7 | 7.6 | 2.5 | 5.2 | 126.4 | 5.2 |
| log | log:k64-log16gib-p15 | 64 | 16 | 2.0 | 512.0 | 1870.5 | 1281.9 | 574.0 | 120.0 | 37.0 | 36.5 | 184.5 | 2.8 | 15.9 | 7.2 | 8.7 | 178.5 | 8.6 |
| log | log:k64-log64gib-p16 | 64 | 64 | 2.0 | 2048.0 | 4785.9 | 2435.6 | 686.0 | 228.1 | 40.6 | 157.9 | 440.3 | 5.1 | 40.7 | 24.3 | 16.4 | 282.7 | 16.3 |
| log | log:k64-log256gib-p17 | 64 | 256 | 2.0 | 8192.0 | 14264.1 | 4871.1 | 910.0 | 456.1 | 49.0 | 603.3 | 1150.9 | 10.3 | 121.3 | 88.9 | 32.4 | 550.7 | 32.5 |
| kappa | kappa:k8-log4gib-p12 | 8 | 4 | 0.2 | 1024.0 | 2478.3 | 1281.9 | 686.0 | 120.0 | 41.0 | 103.8 | 251.8 | 2.8 | 20.8 | 12.2 | 8.6 | 1200.3 | 8.7 |
| kappa | kappa:k16-log4gib-p13 | 16 | 4 | 0.5 | 512.0 | 1874.0 | 1281.9 | 574.0 | 120.0 | 37.1 | 42.8 | 189.9 | 2.7 | 15.6 | 7.0 | 8.5 | 487.4 | 8.6 |
| kappa | kappa:k32-log4gib-p13 | 32 | 4 | 1.0 | 256.0 | 937.1 | 640.9 | 574.0 | 60.0 | 36.7 | 19.7 | 95.5 | 1.4 | 7.9 | 3.6 | 4.3 | 226.4 | 4.3 |
| pattern | pattern:k64-log64gib-p14 | 64 | 64 | 2.0 | 2048.0 | 3133.6 | 769.1 | 1358.0 | 72.0 | 64.2 | 151.8 | 241.1 | 1.6 | 30.1 | 24.8 | 5.3 | 381.0 | 5.2 |
| pattern | pattern:k64-log64gib-p15 | 64 | 64 | 2.0 | 2048.0 | 3636.9 | 1281.9 | 910.0 | 120.0 | 48.7 | 150.1 | 298.5 | 2.8 | 32.3 | 23.6 | 8.7 | 289.4 | 8.6 |
| pattern | pattern:k64-log64gib-p17 | 64 | 64 | 2.0 | 2048.0 | 7219.3 | 4871.1 | 574.0 | 456.1 | 36.8 | 149.8 | 698.7 | 10.4 | 60.1 | 27.2 | 32.9 | 376.1 | 32.7 |
| large | large:k256-log1024gib-p19 | 256 | 1024 | 8.0 | 8192.0 | 14376.3 | 4999.3 | 910.0 | 468.1 | 48.5 | 570.1 | 1130.3 | 10.6 | 125.9 | 92.5 | 33.4 | 332.6 | 34.1 |

## Private openings of one position: a single server with the whole tree

Result-group ID: `tab:private-single`. CSV: `tab-private-single.csv`.

The server stores every cell of the tree in blocks of the given number of cells and answers with the code path of a maintainer: the record of a pattern is the two blocks of its subtree and one cell per level above. Database and state are GiB, the request is MiB, and the reply is KiB. Client request and decoding are milliseconds; first pass and answer are seconds on one core. Preprocessing is seconds on one core, measured as for a maintainer; its size is GiB. The rows at 16 and 64 GiB give the database dimensions and message sizes of those logs.

| panel | row | log-gib | record-kib | database | state | preprocess-size | request | reply-kib | client-request-ms | first-pass | answer | client-decode-ms | preprocess-core-s | preprocess-hint-s | preprocess-packing-s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| measured | measured:log1gib-p10 | 1 | 2.0 | 2.0 | 4.7 | 2.4 | 0.7 | 228.1 | 40.7 | 0.151 | 0.429 | 5.2 | 40.7 | 24.3 | 16.4 |
| measured | measured:log1gib-p9 | 1 | 2.0 | 2.0 | 3.6 | 1.3 | 0.9 | 120.0 | 48.8 | 0.150 | 0.298 | 2.8 | 32.2 | 23.5 | 8.7 |
| measured | measured:log1gib-p11 | 1 | 2.0 | 2.0 | 7.1 | 4.8 | 0.6 | 456.1 | 37.5 | 0.151 | 0.703 | 10.2 | 59.8 | 27.0 | 32.8 |
| measured | measured:log4gib-p11 | 4 | 2.0 | 8.0 | 13.9 | 4.8 | 0.9 | 456.1 | 48.9 | 0.603 | 1.155 | 10.4 | 122.7 | 90.2 | 32.5 |
| shape | shape:log16gib-p12 | 16 | 2.0 | 32.0 | -- | -- | 1.3 | 888.2 | -- | -- | -- | -- | -- | -- | -- |
| shape | shape:log64gib-p13 | 64 | 2.0 | 128.0 | -- | -- | 2.2 | 1776.4 | -- | -- | -- | -- | -- | -- | -- |
| shape | shape:log256gib-p14 | 256 | 2.0 | 512.0 | -- | -- | 4.0 | 3528.9 | -- | -- | -- | -- | -- | -- | -- |
| shape | shape:log1024gib-p15 | 1024 | 2.0 | 2048.0 | -- | -- | 7.5 | 7033.7 | -- | -- | -- | -- | -- | -- | -- |

## Records against the table of complete openings at kappa = 64

Result-group ID: `tab:private-table`. CSV: `tab-private-table.csv`.

The table holds one complete opening per pattern and is answered by YPIR's server; the records are the maintainer's database of the same claim. Both give the same reply bytes. Sizes are MiB; first pass and answer are milliseconds on one core.

| panel | row | log-gib | records-mib | table-mib | records-first-pass-ms | table-first-pass-ms | records-answer-ms | table-answer-ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | log1gib | 1 | 32.0 | 41.5 | 2.6 | 3.4 | 49.9 | 50.3 |
|  | log4gib | 4 | 128.0 | 147.2 | 9.4 | 14.7 | 100.0 | 106.3 |

## Client work for one private opening

Result-group ID: `tab:private-client`. CSV: `tab-private-client.csv`.

A 12 MiB log at every threshold and the 1 and 4 GiB logs at threshold 64, at the public grouping. Request and reply are serialized KiB; the reply is one claim's. Request is key generation, the query, and serialization in milliseconds. Private cells decode the replies and run Combine+Verify; public cells run Combine+Verify on the public replies of the same claims. The first pair uses kappa correct replies; the second uses 3*kappa replies of which kappa lie about one node. All are milliseconds.

| panel | row | kappa | log-mib | request-kib | reply-kib | client-request-ms | private-kappa-ms | public-kappa-ms | private-3kappa-honest-ms | public-3kappa-honest-ms | private-3kappa-ms | public-3kappa-ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | k8 | 8 | 4096 | 686.0 | 120.0 | 41.3 | 22.231 | 0.338 | 66.148 | 0.451 | 66.630 | 0.737 |
|  | k16 | 16 | 4096 | 574.0 | 120.0 | 37.4 | 44.328 | 0.456 | 132.499 | 1.316 | 134.548 | 3.024 |
|  | k32 | 32 | 4096 | 574.0 | 60.0 | 38.0 | 44.463 | 0.863 | 136.307 | 4.557 | 149.012 | 16.806 |
|  | k64 | 64 | 4096 | 518.0 | 72.0 | 35.8 | 106.828 | 2.238 | 333.241 | 17.492 | 379.932 | 64.155 |
|  | k256 | 256 | 4096 | 490.0 | 60.0 | 34.0 | 378.855 | 29.301 | 1319.846 | 273.523 | 2068.883 | 1020.815 |
| log | log:log8gib-k64 | 64 | 8192 | 574.0 | 72.0 | 38.0 | 106.082 | 2.308 | 329.969 | 18.195 | 379.109 | 66.630 |
| log | log:log16gib-k64 | 64 | 16384 | 574.0 | 120.0 | 37.6 | 176.046 | 2.392 | 535.295 | 18.680 | 585.436 | 70.047 |
| log | log:log32gib-k64 | 64 | 32768 | 686.0 | 120.0 | 41.4 | 175.562 | 2.452 | 536.653 | 19.329 | 588.025 | 72.798 |

## UndineLog against TreePIR: one server's preparation and answer

Result-group ID: `tab:treepir-server`. CSV: `tab-treepir-server.csv`.

h = kappa = 32: a log of 2^32 leaves (128 GiB), 32 UndineLog maintainers and 32 TreePIR servers over the same YPIR backend. Each pair of rows is one UndineLog claim and one TreePIR class, whose PIR servers are rebuilt from the start in every repetition and alternated, on the same valid Merkle tree as the client: 2^32 - 2^20 allocated, unwritten leaves followed by 2^20 seeded ones. Every query opens a seeded leaf. The fragment and the TreePIR label source must have the same root. Both systems materialize their full PIR inputs with normal padding; zero blocks retain the backend's shortcut. The maintainer's fragment is built once per process, outside these stages, and every repetition starts from it. The stages are seconds on one core: UndineLog gathers its primary columns into PIR cells, then reads the row table and the upper nodes, then builds the hint (which gathers its columns as it goes) and the packing tables; TreePIR enumerates its class, writes its database columns, then builds the hint and the packing tables. Prepare is the four stages of one rebuild. Precision is required for fill-s (including UndineLog's gathering), preprocess-s (hint plus packing), enumerate-s, prepare-s, and answer-ms. Other stage timings are diagnostics; their achieved intervals remain in the summary. PIR is the PIR database (UndineLog's is derived from the claim's primary columns and needs none of the auxiliary rows; TreePIR's is the class's dense padded matrix); hint and packing are the preprocessing; state is the three; all MiB. The hint's kernel skips an all-zero block of 2,048 rows of one column: zero-blocks counts those, padding-blocks those in the padding columns, out of hint-blocks. First pass and answer are milliseconds on one core; request, query, keys, and reply are serialized KiB.

| panel | row | system | server | outputs | pir-mib | hint-mib | packing-mib | state-mib | hint-blocks | zero-blocks | padding-blocks | fill-s | preprocess-s | gather-cells-s | gather-upper-s | enumerate-s | columns-s | hint-s | packing-s | prepare-s | first-pass-ms | answer-ms | request-kib | query-kib | key-kib | reply-kib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m0-c0 | m0-c0:undine | undine | 0 | 37 | 9417.0 | 1184.0 | 3558.9 | 14159.9 | 2424832 | 0 | 0 | 12.1 | 122.0 | 12.1 | 0.076 | -- | -- | 90.0 | 32.0 | 134.2 | 667.3 | 1188.2 | 910.0 | 448.0 | 462.0 | 444.1 |
| m0-c0 | m0-c0:treepir | treepir | 0 | 37 | 9472.0 | 1184.0 | 3558.9 | 14214.9 | 2424832 | 28095 | 28064 | 13.2 | 118.0 | -- | -- | 6.2 | 13.2 | 86.4 | 31.6 | 137.4 | 875.2 | 1309.8 | 910.0 | 448.0 | 462.0 | 444.1 |
| m31-c31 | m31-c31:undine | undine | 31 | 37 | 9417.0 | 1184.0 | 3558.9 | 14159.9 | 2424832 | 0 | 0 | 12.1 | 121.5 | 12.1 | 0.073 | -- | -- | 89.5 | 32.0 | 133.6 | 653.3 | 1195.6 | 910.0 | 448.0 | 462.0 | 444.1 |
| m31-c31 | m31-c31:treepir | treepir | 31 | 37 | 9472.0 | 1184.0 | 3558.9 | 14214.9 | 2424832 | 28095 | 28064 | 13.2 | 117.7 | -- | -- | 6.3 | 13.2 | 86.0 | 31.7 | 137.3 | 789.0 | 1335.1 | 910.0 | 448.0 | 462.0 | 444.1 |

## UndineLog against TreePIR: the client's work for one opening

Result-group ID: `tab:treepir-client`. CSV: `tab-treepir-client.csv`.

UndineLog prepares one request and sends it to kappa maintainers; TreePIR prepares h requests, each with its own secret key, since the backend reuses a public query matrix and one secret for different queries would let colluding servers subtract them. Upload counts every transmitted request and download every reply of one opening (MiB and KiB). Index, verify are microseconds; the other cells are milliseconds on one core. Index is the pattern and the opening plan, or the h colors and offsets of the path; keygen, query (with the packing keys), and serialize are the request's parts; decrypt parses and decrypts every reply and extracts the shares or the proof nodes; combine+verify is UndineLog's reconstruction and path check; verify recomputes the root from the recovered path. The totals are timed as a whole on a second opening. The h = 16 results are reported only in the artifact: every leaf is seeded, every reply comes from a real server, and each reply is checked against its generated counterpart for plaintext equality and serialized length. At h = 32 the log has 2^32 - 2^20 allocated, unwritten leaves followed by 2^20 seeded ones; every opening is of a seeded leaf. Each reply is the row its server returns, read from the tree and encrypted under the client's key, and the check table compares such replies with a real server's. Reconstruction is fault-free. Only request-total-ms, process-total-ms, and client-total-ms determine timing precision. All substeps, including index, are diagnostics with their achieved intervals in the summary.

| panel | row | system | height | replies | servers | queries | request-kib | upload-mib | download-kib | index-us | keygen-ms | query-ms | serialize-ms | request-total-ms | decrypt-ms | combine-verify-ms | verify-us | process-total-ms | client-total-ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h16-real | h16-real:undine | undine | 16 | real | 16 | 1 | 476.0 | 7.4 | 192.1 | 3.2 | 0.048 | 33.090 | 0.085 | 33.201 | 4.483 | 0.323 | 4.5 | 4.787 | 37.988 |
| h16-real | h16-real:treepir | treepir | 16 | real | 16 | 16 | 476.0 | 7.4 | 192.1 | 6.5 | 0.615 | 526.062 | 2.392 | 528.218 | 4.298 | -- | 9.6 | 4.317 | 532.535 |
| h32-generated | h32-generated:undine | undine | 32 | generated | 32 | 1 | 910.0 | 28.4 | 14211.6 | 9.5 | 0.051 | 48.047 | 0.183 | 48.326 | 320.126 | 1.045 | 8.4 | 321.204 | 369.531 |
| h32-generated | h32-generated:treepir | treepir | 32 | generated | 32 | 32 | 910.0 | 28.4 | 14211.6 | 12.1 | 1.308 | 1535.556 | 13.687 | 1542.174 | 320.318 | -- | 14.3 | 319.777 | 1861.950 |

## UndineLog against TreePIR: generated and real replies

Result-group ID: `tab:treepir-check`. CSV: `tab-treepir-check.csv`.

At h = kappa = 32, one real UndineLog claim and one real TreePIR server are built on the opening tree. For each query, the real reply and the generated reply for the same server decode to the same row (checked exactly), and their decoding is timed in turn, in alternating order. Milliseconds on one core.

| panel | row | system | server | real-decode-ms | generated-decode-ms |
| --- | --- | --- | --- | --- | --- |
|  | undine | undine | 0 | 10.703 | 10.769 |
|  | treepir | treepir | 0 | 10.213 | 10.145 |
