# Evaluation result groups

Values are reviewer-facing renderings. Machine-readable source values, units, samples, and confidence intervals are in `../summary.json` and `../summary.tsv`.

## Append processing and fragment size

Result-group ID: `tab:append`. CSV: `tab-append.csv`.

One row per threshold, with Seg = kappa^2. Append cells are one maintainer's mean processing time for a 4 MiB batch in milliseconds. Fragment cells are the mean newly materialized fragment size per batch in MiB.

| panel | row | segment | kappa | undine-append | undine-fragment | borg-append | borg-fragment |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | k8 | 64 | 8 | 159.8 | 2.000 | 262.1 | 2.000 |
|  | k16 | 256 | 16 | 160.6 | 1.000 | 231.3 | 1.000 |
|  | k32 | 1024 | 32 | 160.6 | 0.500 | 225.4 | 0.500 |
|  | k64 | 4096 | 64 | 166.7 | 0.250 | 238.0 | 0.250 |

## Point openings

Result-group ID: `tab:point-open`. CSV: `tab-point-open.csv`.

Helper work is microseconds for one helper's response construction and serialization; response cells are its serialized bytes. Client cells are Combine+Verify in milliseconds from kappa correct responses, and from 3*kappa responses of which kappa are corrupted.

| panel | row | segment | kappa | undine-helper | undine-response | undine-client | undine-client-correcting | borg-helper | borg-response | borg-client | borg-client-correcting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | k8 | 64 | 8 | 9.5 | 1251 | 0.224 | 1.173 | 174.7 | 1251 | 0.272 | 1.399 |
|  | k16 | 256 | 16 | 8.4 | 1251 | 0.318 | 10.054 | 194.1 | 1251 | 0.369 | 10.378 |
|  | k32 | 1024 | 32 | 10.8 | 1251 | 0.646 | 102.861 | 239.2 | 1251 | 0.652 | 103.433 |
|  | k64 | 4096 | 64 | 7.7 | 1251 | 1.638 | 769.371 | 312.1 | 1251 | 1.673 | 770.366 |

## Range openings at kappa = 64

Result-group ID: `tab:range-open`. CSV: `tab-range-open.csv`.

Per-helper response size is KiB. Helper work is one helper's answer to the range request and its serialization, in milliseconds. Client work is seconds, from 192 responses; the corrupt case has 64 corrupted responses among them.

| panel | row | range-mib | full-segment-reads | point-reads | undine-per-helper-kib | undine-helper-work | undine-clean-client | undine-corrupt-client | borg-per-helper-kib | borg-helper-work | borg-clean-client | borg-corrupt-client |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | range-1mib | 1 | 16 | 41 | 33.9 | 0.033 | 0.131 | 2.612 | 33.9 | 0.681 | 0.153 | 2.711 |
|  | range-4mib | 4 | 64 | 37 | 130.0 | 0.095 | 0.512 | 5.143 | 130.0 | 0.687 | 0.605 | 5.524 |
|  | range-8mib | 8 | 128 | 35 | 258.3 | 0.181 | 1.016 | 5.920 | 258.3 | 0.738 | 1.231 | 6.393 |
|  | range-32mib | 32 | 512 | 31 | 1028.9 | 0.744 | 4.366 | 10.663 | 1028.9 | 1.197 | 5.102 | 11.708 |
|  | range-128mib | 128 | 2048 | 27 | 4112.7 | 3.268 | 18.810 | 30.801 | 4112.7 | 3.697 | 21.838 | 34.158 |

## Repair and reconfiguration

Result-group ID: `tab:maintenance`. CSV: `tab-maintenance.csv`.

The panel is the layout. All quantities are normalized per 4 MiB appended to the evaluated 12 MiB prefix. Times are milliseconds; fragment, download, and state values are MiB. Double and halve move the threshold between kappa and 2*kappa. The diagonal halve columns are the UndineLog undo from source state at kappa; the segment-double column is BorgLog's local segment-and-kappa double.

| panel | row | segment | kappa | fragment | repair-receiver | repair-helper | repair-download | double-receiver | double-helper | double-download | state-after-double | halve-receiver | halve-helper | halve-download | halve-diagonal-receiver | halve-diagonal-download | segment-double-receiver |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| undine | undine:k8 | 64 | 8 | 2.000 | 37.196 | 3.888 | 2.000 | 1.413 | -- | 0.000 | 1.000 | 16.316 | 0.719 | 1.000 | 2.112 | 1.000 | -- |
| undine | undine:k16 | 256 | 16 | 1.000 | 21.081 | 1.339 | 1.000 | 0.616 | -- | 0.000 | 0.501 | 12.694 | 0.313 | 0.501 | 0.871 | 0.501 | -- |
| undine | undine:k32 | 1024 | 32 | 0.501 | 16.395 | 0.546 | 0.501 | 0.302 | -- | 0.000 | 0.251 | 11.225 | 0.154 | 0.251 | 0.397 | 0.251 | -- |
| undine | undine:k64 | 4096 | 64 | 0.251 | 14.924 | 0.262 | 0.251 | 0.189 | -- | 0.000 | 0.128 | 11.029 | 0.081 | 0.128 | 0.234 | 0.128 | -- |
| borg | borg:k8 | 64 | 8 | 2.000 | 435.315 | 29.919 | 2.000 | 91.020 | 11.703 | 1.875 | 2.500 | 207.130 | 13.337 | 1.375 | -- | -- | 66.299 |
| borg | borg:k16 | 256 | 16 | 1.000 | 157.753 | 9.223 | 1.000 | 70.200 | 3.677 | 0.969 | 1.250 | 25.073 | 4.444 | 0.719 | -- | -- | 36.794 |
| borg | borg:k32 | 1024 | 32 | 0.501 | 113.138 | 3.360 | 0.501 | 68.113 | 1.952 | 0.493 | 0.626 | 18.681 | 1.787 | 0.368 | -- | -- | 29.425 |
| borg | borg:k64 | 4096 | 64 | 0.251 | 80.323 | 1.277 | 0.251 | 70.683 | 2.799 | 0.249 | 0.314 | 17.615 | 0.709 | 0.187 | -- | -- | 30.089 |

## UndineLog openings served above the stored threshold

Result-group ID: `tab:on-the-fly-open`. CSV: `tab-on-the-fly-open.csv`.

One maintainer stores a 12 MiB log at the stored threshold and answers at the opened threshold. On-the-fly cells are that answer; stored cells are the answer of a maintainer that stores the log at the opened threshold. Both include response serialization and are microseconds; response sizes are KiB. Stitch-state is the time to stitch the maintainer's whole state to the opened threshold, in milliseconds.

| panel | row | stored-kappa | open-kappa | point-on-the-fly | point-stored | point-bytes | range-1mib-on-the-fly | range-1mib-stored | range-1mib-kib | range-4mib-on-the-fly | range-4mib-stored | range-4mib-kib | stitch-state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | k8-open8 | 8 | 8 | -- | 10.3 | 1251 | -- | 362.2 | 263.9 | -- | 1486.2 | 1055.8 | -- |
|  | k8-open16 | 8 | 16 | 12.4 | 8.7 | 1251 | 336.1 | 123.7 | 130.2 | 1341.7 | 492.2 | 520.1 | 4.428 |
|  | k8-open32 | 8 | 32 | 18.0 | 8.7 | 1251 | 340.3 | 52.0 | 65.5 | 1336.4 | 198.5 | 258.6 | 6.522 |
|  | k8-open64 | 8 | 64 | 26.3 | 8.6 | 1251 | 350.3 | 28.4 | 33.7 | 1306.8 | 91.9 | 129.8 | 8.162 |
|  | k16-open16 | 16 | 16 | -- | 9.7 | 1251 | -- | 124.8 | 130.2 | -- | 481.6 | 520.1 | -- |
|  | k16-open32 | 16 | 32 | 12.1 | 8.6 | 1251 | 137.1 | 51.3 | 65.5 | 535.9 | 194.4 | 258.6 | 2.174 |
|  | k16-open64 | 16 | 64 | 16.8 | 8.4 | 1251 | 151.1 | 28.4 | 33.7 | 570.4 | 90.8 | 129.8 | 3.837 |
|  | k16-open128 | 16 | 128 | 24.6 | 8.2 | 1251 | 163.3 | 16.1 | 17.7 | 580.7 | 43.3 | 65.6 | 7.073 |
|  | k32-open32 | 32 | 32 | -- | 9.7 | 1251 | -- | 55.1 | 65.5 | -- | 209.9 | 258.6 | -- |
|  | k32-open64 | 32 | 64 | 12.5 | 8.6 | 1251 | 70.2 | 28.9 | 33.7 | 238.8 | 92.5 | 129.8 | 1.847 |
|  | k32-open128 | 32 | 128 | 16.1 | 8.1 | 1251 | 77.8 | 16.3 | 17.7 | 263.2 | 43.2 | 65.6 | 5.194 |
|  | k32-open256 | 32 | 256 | 22.6 | 7.8 | 1251 | 90.4 | 11.1 | 9.6 | 276.0 | 23.2 | 33.5 | 16.301 |
|  | k64-open64 | 64 | 64 | -- | 10.5 | 1251 | -- | 32.0 | 33.7 | -- | 98.1 | 129.8 | -- |
|  | k64-open128 | 64 | 128 | 12.3 | 8.2 | 1251 | 38.9 | 17.3 | 17.7 | 115.0 | 45.5 | 65.6 | 3.733 |
|  | k64-open256 | 64 | 256 | 15.9 | 8.1 | 1251 | 46.2 | 11.8 | 9.6 | 129.8 | 24.4 | 33.5 | 15.002 |
|  | k64-open512 | 64 | 512 | 21.4 | 7.9 | 1251 | 126.6 | 8.8 | 16.1 | 139.0 | 15.1 | 17.5 | 63.080 |

## Private openings of one position: one claim of a maintainer

Result-group ID: `tab:private-maintainer`. CSV: `tab-private-maintainer.csv`.

One claim answers from its primary columns, grouped into blocks of the public grouping of the log length (one segment per block at 1 GiB, doubling every two levels) unless the row gives another block size. A log of 1 GiB has 2^25 leaves. The record of a pattern is the two blocks of its subtree and one entry per level above. Database is the bytes of the primary columns; state adds the row table and the preprocessing; both are MiB. Request and reply are serialized KiB. The client's request (key, query, serialization), the first pass, the complete answer from serialized request to serialized reply, and the client's decoding of one reply are milliseconds on one core. Preprocessing is core-seconds: the measured hint of 64 database columns scaled to all record columns, plus the measured packing tables of one reply ciphertext times the ciphertext count. The update cells follow a 4 MiB append: the hint patch in milliseconds and the packing tables in seconds.

| panel | row | kappa | log-gib | record-kib | database | state | request | reply-kib | client-request-ms | first-pass | answer | client-decode-ms | preprocess-core-s | update-hint-ms | update-packing-s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| log | log:k64-log1gib-p13 | 64 | 1 | 2.0 | 32.0 | 421.7 | 490.0 | 36.0 | 35.4 | 7.6 | 54.1 | 0.9 | 4.4 | 311.2 | 2.9 |
| log | log:k64-log4gib-p14 | 64 | 4 | 2.0 | 128.0 | 916.8 | 518.0 | 72.0 | 36.4 | 29.6 | 119.8 | 1.8 | 10.2 | 498.7 | 5.8 |
| log | log:k64-log16gib-p15 | 64 | 16 | 2.0 | 512.0 | 1870.5 | 574.0 | 120.0 | 38.9 | 115.6 | 262.6 | 3.0 | 24.7 | 1033.6 | 9.6 |
| log | log:k64-log64gib-p16 | 64 | 64 | 2.0 | 2048.0 | 4914.1 | 686.0 | 240.1 | 43.0 | 488.0 | 787.2 | 5.9 | 76.8 | 2048.9 | 19.1 |
| kappa | kappa:k8-log4gib-p12 | 8 | 4 | 0.2 | 1024.0 | 2478.3 | 686.0 | 120.0 | 42.7 | 892.8 | 1053.1 | 3.0 | 38.0 | 2238.1 | 9.6 |
| kappa | kappa:k16-log4gib-p13 | 16 | 4 | 0.5 | 512.0 | 1874.0 | 574.0 | 120.0 | 38.7 | 264.9 | 413.7 | 3.0 | 24.5 | 1364.1 | 9.6 |
| kappa | kappa:k32-log4gib-p13 | 32 | 4 | 1.0 | 256.0 | 937.1 | 574.0 | 60.0 | 38.7 | 79.8 | 153.4 | 1.5 | 12.2 | 605.8 | 5.8 |
| pattern | pattern:k64-log64gib-p14 | 64 | 64 | 2.0 | 2048.0 | 3133.6 | 1358.0 | 72.0 | 69.4 | 486.0 | 570.5 | 1.8 | 64.2 | 807.6 | 5.8 |
| pattern | pattern:k64-log64gib-p15 | 64 | 64 | 2.0 | 2048.0 | 3636.9 | 910.0 | 120.0 | 51.5 | 470.4 | 612.8 | 3.0 | 66.3 | 1185.0 | 9.7 |
| pattern | pattern:k64-log64gib-p17 | 64 | 64 | 2.0 | 2048.0 | 7219.3 | 574.0 | 456.1 | 38.7 | 492.1 | 1041.5 | 11.3 | 94.1 | 3792.8 | 36.4 |

## Private openings of one position: a single server with the whole tree

Result-group ID: `tab:private-single`. CSV: `tab-private-single.csv`.

The server stores every cell of the tree in blocks of the given number of cells and answers with the code path of a maintainer: the record of a pattern is the two blocks of its subtree and one cell per level above. Database and state are GiB, the request is MiB, and the reply is KiB. Client request and decoding are milliseconds; first pass and answer are seconds on one core. Preprocessing is core-seconds, measured as for a maintainer. The rows at 16 and 64 GiB give the database dimensions and message sizes of those logs.

| panel | row | log-gib | record-kib | database | state | request | reply-kib | client-request-ms | first-pass | answer | client-decode-ms | preprocess-core-s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| measured | measured:log1gib-p10 | 1 | 2.0 | 2.0 | 4.8 | 0.7 | 240.1 | 43.0 | 0.489 | 0.781 | 6.0 | 75.9 |
| measured | measured:log1gib-p9 | 1 | 2.0 | 2.0 | 3.6 | 0.9 | 120.0 | 51.4 | 0.463 | 0.611 | 3.0 | 66.7 |
| measured | measured:log1gib-p11 | 1 | 2.0 | 2.0 | 7.1 | 0.6 | 456.1 | 38.5 | 0.510 | 1.063 | 11.3 | 95.0 |
| measured | measured:log4gib-p11 | 4 | 2.0 | 8.0 | 13.9 | 0.9 | 456.1 | 51.6 | 1.987 | 2.543 | 11.2 | 308.5 |
| shape | shape:log16gib-p12 | 16 | 2.0 | 32.0 | -- | 1.3 | 888.2 | -- | -- | -- | -- | -- |
| shape | shape:log64gib-p13 | 64 | 2.0 | 128.0 | -- | 2.2 | 1776.4 | -- | -- | -- | -- | -- |

## Records against the table of complete openings at kappa = 64

Result-group ID: `tab:private-table`. CSV: `tab-private-table.csv`.

The table holds one complete opening per pattern and is answered by YPIR's server; the records are the maintainer's database of the same claim. Both give the same reply bytes. Sizes are MiB; first pass and answer are milliseconds on one core.

| panel | row | log-gib | records-mib | table-mib | records-first-pass-ms | table-first-pass-ms | records-answer-ms | table-answer-ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | log1gib | 1 | 32.0 | 136.0 | 7.6 | 12.5 | 54.1 | 62.1 |
|  | log4gib | 4 | 128.0 | 352.0 | 29.6 | 42.1 | 119.8 | 133.9 |

## Client work for one private opening

Result-group ID: `tab:private-client`. CSV: `tab-private-client.csv`.

A 12 MiB log at the public grouping. Request and reply are serialized KiB; the reply is one claim's. Request is key generation, the query, and serialization in milliseconds. Private cells decode the replies and run Combine+Verify; public cells run Combine+Verify on the public replies of the same claims. The first pair uses kappa correct replies; the second uses 3*kappa replies of which kappa lie about one node. All are milliseconds.

| panel | row | kappa | request-kib | reply-kib | client-request-ms | private-kappa-ms | public-kappa-ms | private-3kappa-ms | public-3kappa-ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | k8 | 8 | 476.0 | 12.0 | 36.0 | 2.883 | 0.248 | 9.051 | 1.252 |
|  | k16 | 16 | 476.0 | 12.0 | 38.5 | 5.658 | 0.372 | 26.456 | 10.908 |
|  | k32 | 32 | 476.0 | 12.0 | 37.6 | 10.840 | 0.679 | 105.220 | 74.952 |
|  | k64 | 64 | 476.0 | 24.0 | 36.7 | 40.090 | 1.708 | 666.375 | 550.396 |
