# Evaluation result groups

Values are reviewer-facing renderings. Machine-readable source values, units, samples, and confidence intervals are in `../summary.json` and `../summary.tsv`.

## Append processing and fragment size

Result-group ID: `tab:append`. CSV: `tab-append.csv`.

One row per threshold, with Seg = kappa^2. Append cells are one maintainer's mean processing time for a 4 MiB batch in milliseconds. Fragment cells are the mean newly materialized fragment size per batch in MiB.

| panel | row | segment | kappa | undine-append | undine-fragment | borg-append | borg-fragment |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | k8 | 64 | 8 | 162.0 | 2.000 | 261.8 | 2.000 |
|  | k16 | 256 | 16 | 160.9 | 1.000 | 233.5 | 1.000 |
|  | k32 | 1024 | 32 | 161.0 | 0.500 | 225.2 | 0.500 |
|  | k64 | 4096 | 64 | 165.9 | 0.250 | 237.3 | 0.250 |

## Point openings

Result-group ID: `tab:point-open`. CSV: `tab-point-open.csv`.

Helper work is microseconds for one helper's response construction and serialization; response cells are its serialized bytes. Client cells are Combine+Verify in milliseconds from kappa correct responses, and from 3*kappa responses of which kappa are corrupted.

| panel | row | segment | kappa | undine-helper | undine-response | undine-client | undine-client-correcting | borg-helper | borg-response | borg-client | borg-client-correcting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | k8 | 64 | 8 | 14.9 | 1251 | 0.209 | 1.130 | 173.9 | 1251 | 0.274 | 1.393 |
|  | k16 | 256 | 16 | 8.6 | 1251 | 0.307 | 9.778 | 192.7 | 1251 | 0.376 | 10.423 |
|  | k32 | 1024 | 32 | 8.2 | 1251 | 0.659 | 101.348 | 245.7 | 1251 | 0.655 | 102.948 |
|  | k64 | 4096 | 64 | 8.2 | 1251 | 1.596 | 764.441 | 307.8 | 1251 | 1.647 | 772.332 |

## Range openings at kappa = 64

Result-group ID: `tab:range-open`. CSV: `tab-range-open.csv`.

Per-helper response size is KiB. Helper work is one helper's answer to the range request and its serialization, in milliseconds. Client work is seconds, from 192 responses; the corrupt case has 64 corrupted responses among them.

| panel | row | range-mib | full-segment-reads | point-reads | undine-per-helper-kib | undine-helper-work | undine-clean-client | undine-corrupt-client | borg-per-helper-kib | borg-helper-work | borg-clean-client | borg-corrupt-client |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | range-1mib | 1 | 16 | 41 | 33.9 | 0.032 | 0.130 | 2.616 | 33.9 | 0.679 | 0.153 | 2.714 |
|  | range-4mib | 4 | 64 | 37 | 130.0 | 0.097 | 0.507 | 5.149 | 130.0 | 0.680 | 0.605 | 5.515 |
|  | range-8mib | 8 | 128 | 35 | 258.3 | 0.185 | 1.029 | 5.923 | 258.3 | 0.735 | 1.230 | 6.408 |
|  | range-32mib | 32 | 512 | 31 | 1028.9 | 0.731 | 4.374 | 10.643 | 1028.9 | 1.202 | 5.111 | 11.718 |
|  | range-128mib | 128 | 2048 | 27 | 4112.7 | 3.416 | 18.757 | 30.881 | 4112.7 | 3.624 | 21.916 | 34.145 |

## Repair and reconfiguration

Result-group ID: `tab:maintenance`. CSV: `tab-maintenance.csv`.

The panel is the layout. All quantities are normalized per 4 MiB appended to the evaluated 12 MiB prefix. Times are milliseconds; fragment, download, and state values are MiB. Double and halve move the threshold between kappa and 2*kappa. The diagonal halve columns are the UndineLog undo from source state at kappa; the segment-double column is BorgLog's local segment-and-kappa double.

| panel | row | segment | kappa | fragment | repair-receiver | repair-helper | repair-download | double-receiver | double-helper | double-download | state-after-double | halve-receiver | halve-helper | halve-download | halve-diagonal-receiver | halve-diagonal-download | segment-double-receiver |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| undine | undine:k8 | 64 | 8 | 2.000 | 36.917 | 3.850 | 2.000 | 1.414 | -- | 0.000 | 1.000 | 16.370 | 0.718 | 1.000 | 2.158 | 1.000 | -- |
| undine | undine:k16 | 256 | 16 | 1.000 | 21.829 | 1.335 | 1.000 | 0.625 | -- | 0.000 | 0.501 | 12.702 | 0.311 | 0.501 | 0.892 | 0.501 | -- |
| undine | undine:k32 | 1024 | 32 | 0.501 | 16.154 | 0.545 | 0.501 | 0.304 | -- | 0.000 | 0.251 | 11.243 | 0.152 | 0.251 | 0.396 | 0.251 | -- |
| undine | undine:k64 | 4096 | 64 | 0.251 | 15.101 | 0.262 | 0.251 | 0.209 | -- | 0.000 | 0.128 | 10.916 | 0.089 | 0.128 | 0.233 | 0.128 | -- |
| borg | borg:k8 | 64 | 8 | 2.000 | 433.046 | 29.675 | 2.000 | 91.111 | 11.650 | 1.875 | 2.500 | 205.302 | 13.405 | 1.375 | -- | -- | 68.340 |
| borg | borg:k16 | 256 | 16 | 1.000 | 158.421 | 9.215 | 1.000 | 71.740 | 3.733 | 0.969 | 1.250 | 24.463 | 4.494 | 0.719 | -- | -- | 37.113 |
| borg | borg:k32 | 1024 | 32 | 0.501 | 113.218 | 3.402 | 0.501 | 67.676 | 1.957 | 0.493 | 0.626 | 18.476 | 1.788 | 0.368 | -- | -- | 29.278 |
| borg | borg:k64 | 4096 | 64 | 0.251 | 80.740 | 1.279 | 0.251 | 70.969 | 2.792 | 0.249 | 0.314 | 17.376 | 0.713 | 0.187 | -- | -- | 29.952 |

## UndineLog openings served above the stored threshold

Result-group ID: `tab:on-the-fly-open`. CSV: `tab-on-the-fly-open.csv`.

One maintainer stores a 12 MiB log at the stored threshold and answers at the opened threshold. On-the-fly cells are that answer; stored cells are the answer of a maintainer that stores the log at the opened threshold. Both include response serialization and are microseconds; response sizes are KiB. Stitch-state is the time to stitch the maintainer's whole state to the opened threshold, in milliseconds.

| panel | row | stored-kappa | open-kappa | point-on-the-fly | point-stored | point-bytes | range-1mib-on-the-fly | range-1mib-stored | range-1mib-kib | range-4mib-on-the-fly | range-4mib-stored | range-4mib-kib | stitch-state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | k8-open8 | 8 | 8 | -- | 10.3 | 1251 | -- | 366.3 | 263.9 | -- | 1508.6 | 1055.8 | -- |
|  | k8-open16 | 8 | 16 | 12.5 | 8.8 | 1251 | 338.4 | 126.0 | 130.2 | 1337.3 | 498.3 | 520.1 | 4.520 |
|  | k8-open32 | 8 | 32 | 17.8 | 8.8 | 1251 | 334.7 | 51.4 | 65.5 | 1313.4 | 200.3 | 258.6 | 6.624 |
|  | k8-open64 | 8 | 64 | 26.3 | 8.5 | 1251 | 342.1 | 28.3 | 33.7 | 1276.2 | 92.9 | 129.8 | 8.291 |
|  | k16-open16 | 16 | 16 | -- | 9.7 | 1251 | -- | 131.1 | 130.2 | -- | 503.0 | 520.1 | -- |
|  | k16-open32 | 16 | 32 | 12.2 | 8.6 | 1251 | 136.3 | 52.7 | 65.5 | 532.6 | 199.5 | 258.6 | 2.217 |
|  | k16-open64 | 16 | 64 | 17.0 | 8.4 | 1251 | 149.6 | 29.3 | 33.7 | 559.9 | 94.5 | 129.8 | 3.912 |
|  | k16-open128 | 16 | 128 | 24.7 | 8.1 | 1251 | 161.1 | 16.5 | 17.7 | 568.7 | 44.5 | 65.6 | 7.215 |
|  | k32-open32 | 32 | 32 | -- | 9.8 | 1251 | -- | 54.3 | 65.5 | -- | 206.7 | 258.6 | -- |
|  | k32-open64 | 32 | 64 | 12.4 | 8.6 | 1251 | 69.5 | 30.0 | 33.7 | 239.0 | 94.4 | 129.8 | 1.884 |
|  | k32-open128 | 32 | 128 | 16.4 | 8.2 | 1251 | 76.3 | 16.7 | 17.7 | 255.0 | 43.5 | 65.6 | 5.280 |
|  | k32-open256 | 32 | 256 | 22.8 | 8.0 | 1251 | 89.0 | 11.5 | 9.6 | 269.9 | 23.4 | 33.5 | 16.673 |
|  | k64-open64 | 64 | 64 | -- | 10.4 | 1251 | -- | 31.9 | 33.7 | -- | 97.5 | 129.8 | -- |
|  | k64-open128 | 64 | 128 | 12.2 | 8.3 | 1251 | 38.1 | 17.1 | 17.7 | 113.1 | 44.7 | 65.6 | 3.792 |
|  | k64-open256 | 64 | 256 | 15.7 | 8.1 | 1251 | 45.1 | 11.4 | 9.6 | 126.8 | 23.9 | 33.5 | 15.418 |
|  | k64-open512 | 64 | 512 | 21.5 | 8.0 | 1251 | 125.8 | 8.7 | 16.1 | 136.7 | 15.1 | 17.5 | 64.413 |

## Private openings of one position: one claim of a maintainer

Result-group ID: `tab:private-maintainer`. CSV: `tab-private-maintainer.csv`.

One claim answers from its primary columns, grouped into blocks of the public grouping of the log length (one segment per block at 1 GiB, doubling every two levels) unless the row gives another block size. A log of 1 GiB has 2^25 leaves. The record of a pattern is the two blocks of its subtree and one entry per level above. Database is the bytes of the primary columns; state adds the row table and the preprocessing; both are MiB. Request and reply are serialized KiB. The client's request (key, query, serialization), the first pass, the complete answer from serialized request to serialized reply, and the client's decoding of one reply are milliseconds on one core. Preprocessing is core-seconds: the measured hint of 64 database columns scaled to all record columns, plus the measured packing tables of one reply ciphertext times the ciphertext count. The update cells follow a 4 MiB append: the hint patch in milliseconds and the packing tables in seconds.

| panel | row | kappa | log-gib | record-kib | database | state | request | reply-kib | client-request-ms | first-pass | answer | client-decode-ms | preprocess-core-s | update-hint-ms | update-packing-s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| log | log:k64-log1gib-p13 | 64 | 1 | 2.0 | 32.0 | 421.7 | 490.0 | 36.0 | 36.2 | 5.1 | 59.3 | 0.9 | 4.7 | 107.4 | 3.1 |
| log | log:k64-log4gib-p14 | 64 | 4 | 2.0 | 128.0 | 916.8 | 518.0 | 72.0 | 38.3 | 20.7 | 119.4 | 1.9 | 10.5 | 127.6 | 6.4 |
| log | log:k64-log16gib-p15 | 64 | 16 | 2.0 | 512.0 | 1870.5 | 574.0 | 120.0 | 41.3 | 81.4 | 233.2 | 3.3 | 25.8 | 178.0 | 10.2 |
| log | log:k64-log64gib-p16 | 64 | 64 | 2.0 | 2048.0 | 4914.1 | 686.0 | 240.1 | 45.2 | 329.2 | 642.4 | 6.2 | 77.6 | 282.1 | 19.9 |
| log | log:k64-log256gib-p17 | 64 | 256 | 2.0 | 8192.0 | 14264.1 | 910.0 | 456.1 | 53.9 | 1376.6 | 2021.5 | 11.6 | 307.5 | 538.1 | 37.7 |
| kappa | kappa:k8-log4gib-p12 | 8 | 4 | 0.2 | 1024.0 | 2478.3 | 686.0 | 120.0 | 44.4 | 203.5 | 351.2 | 3.2 | 38.9 | 1202.6 | 10.1 |
| kappa | kappa:k16-log4gib-p13 | 16 | 4 | 0.5 | 512.0 | 1874.0 | 574.0 | 120.0 | 40.2 | 86.0 | 237.5 | 3.3 | 24.8 | 492.0 | 10.0 |
| kappa | kappa:k32-log4gib-p13 | 32 | 4 | 1.0 | 256.0 | 937.1 | 574.0 | 60.0 | 40.9 | 40.4 | 124.5 | 1.6 | 12.5 | 227.1 | 5.9 |
| pattern | pattern:k64-log64gib-p14 | 64 | 64 | 2.0 | 2048.0 | 3133.6 | 1358.0 | 72.0 | 70.7 | 329.3 | 425.2 | 1.9 | 64.6 | 392.1 | 6.0 |
| pattern | pattern:k64-log64gib-p15 | 64 | 64 | 2.0 | 2048.0 | 3636.9 | 910.0 | 120.0 | 52.8 | 321.0 | 467.6 | 3.2 | 67.7 | 285.7 | 10.3 |
| pattern | pattern:k64-log64gib-p17 | 64 | 64 | 2.0 | 2048.0 | 7219.3 | 574.0 | 456.1 | 40.4 | 350.7 | 928.1 | 11.7 | 97.1 | 380.4 | 38.1 |
| large | large:k256-log1024gib-p19 | 256 | 1024 | 8.0 | 8192.0 | 14504.5 | 910.0 | 480.1 | 52.8 | 1458.6 | 2053.6 | 12.3 | 306.7 | 345.8 | 39.9 |

## Private openings of one position: a single server with the whole tree

Result-group ID: `tab:private-single`. CSV: `tab-private-single.csv`.

The server stores every cell of the tree in blocks of the given number of cells and answers with the code path of a maintainer: the record of a pattern is the two blocks of its subtree and one cell per level above. Database and state are GiB, the request is MiB, and the reply is KiB. Client request and decoding are milliseconds; first pass and answer are seconds on one core. Preprocessing is core-seconds, measured as for a maintainer. The rows at 16 and 64 GiB give the database dimensions and message sizes of those logs.

| panel | row | log-gib | record-kib | database | state | request | reply-kib | client-request-ms | first-pass | answer | client-decode-ms | preprocess-core-s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| measured | measured:log1gib-p10 | 1 | 2.0 | 2.0 | 4.8 | 0.7 | 240.1 | 44.9 | 0.335 | 0.640 | 6.2 | 78.1 |
| measured | measured:log1gib-p9 | 1 | 2.0 | 2.0 | 3.6 | 0.9 | 120.0 | 53.7 | 0.327 | 0.480 | 3.2 | 69.1 |
| measured | measured:log1gib-p11 | 1 | 2.0 | 2.0 | 7.1 | 0.6 | 456.1 | 41.4 | 0.356 | 0.940 | 11.9 | 98.0 |
| measured | measured:log4gib-p11 | 4 | 2.0 | 8.0 | 13.9 | 0.9 | 456.1 | 52.8 | 1.526 | 2.095 | 11.6 | 307.6 |
| shape | shape:log16gib-p12 | 16 | 2.0 | 32.0 | -- | 1.3 | 888.2 | -- | -- | -- | -- | -- |
| shape | shape:log64gib-p13 | 64 | 2.0 | 128.0 | -- | 2.2 | 1776.4 | -- | -- | -- | -- | -- |
| shape | shape:log256gib-p14 | 256 | 2.0 | 512.0 | -- | 4.0 | 3528.9 | -- | -- | -- | -- | -- |
| shape | shape:log1024gib-p15 | 1024 | 2.0 | 2048.0 | -- | 7.5 | 7045.7 | -- | -- | -- | -- | -- |

## Records against the table of complete openings at kappa = 64

Result-group ID: `tab:private-table`. CSV: `tab-private-table.csv`.

The table holds one complete opening per pattern and is answered by YPIR's server; the records are the maintainer's database of the same claim. Both give the same reply bytes. Sizes are MiB; first pass and answer are milliseconds on one core.

| panel | row | log-gib | records-mib | table-mib | records-first-pass-ms | table-first-pass-ms | records-answer-ms | table-answer-ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | log1gib | 1 | 32.0 | 136.0 | 5.1 | 12.5 | 59.3 | 62.6 |
|  | log4gib | 4 | 128.0 | 352.0 | 20.7 | 42.9 | 119.4 | 133.9 |

## Client work for one private opening

Result-group ID: `tab:private-client`. CSV: `tab-private-client.csv`.

A 12 MiB log at every threshold and the 1 and 4 GiB logs at threshold 64, at the public grouping. Request and reply are serialized KiB; the reply is one claim's. Request is key generation, the query, and serialization in milliseconds. Private cells decode the replies and run Combine+Verify; public cells run Combine+Verify on the public replies of the same claims. The first pair uses kappa correct replies; the second uses 3*kappa replies of which kappa lie about one node. All are milliseconds.

| panel | row | kappa | log-mib | request-kib | reply-kib | client-request-ms | private-kappa-ms | public-kappa-ms | private-3kappa-honest-ms | public-3kappa-honest-ms | private-3kappa-ms | public-3kappa-ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | k8 | 8 | 4096 | 686.0 | 120.0 | 45.4 | 25.200 | 0.369 | 74.651 | 0.472 | 75.954 | 1.356 |
|  | k16 | 16 | 4096 | 574.0 | 120.0 | 40.5 | 49.262 | 0.495 | 146.737 | 1.342 | 155.713 | 10.353 |
|  | k32 | 32 | 4096 | 574.0 | 60.0 | 40.4 | 49.270 | 0.920 | 149.281 | 4.717 | 244.065 | 99.577 |
|  | k64 | 64 | 4096 | 518.0 | 72.0 | 37.5 | 115.608 | 2.374 | 357.287 | 17.692 | 1064.287 | 721.439 |
|  | k256 | 256 | 4096 | 490.0 | 60.0 | 35.3 | 403.210 | 30.218 | 1399.747 | 274.996 | 48953.887 | 47782.408 |
| log | log:log8gib-k64 | 64 | 8192 | 574.0 | 72.0 | 39.7 | 116.561 | 2.483 | 359.866 | 18.353 | 1103.208 | 758.852 |
| log | log:log16gib-k64 | 64 | 16384 | 574.0 | 120.0 | 40.0 | 192.028 | 2.512 | 587.658 | 18.984 | 1372.553 | 800.702 |
| log | log:log32gib-k64 | 64 | 32768 | 686.0 | 120.0 | 44.6 | 189.645 | 2.613 | 580.050 | 19.703 | 1402.915 | 841.701 |
