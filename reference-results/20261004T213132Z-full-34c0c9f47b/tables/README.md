# Evaluation result groups

Values are reviewer-facing renderings. Machine-readable source values, units, samples, and confidence intervals are in `../summary.json` and `../summary.tsv`.

## Append processing and fragment size

Result-group ID: `tab:append`. CSV: `tab-append.csv`.

One row per threshold, with Seg = kappa^2. Append cells are one maintainer's mean processing time for a 4 MiB batch in milliseconds. Fragment cells are the mean newly materialized fragment size per batch in MiB.

| panel | row | segment | kappa | undine-append | undine-fragment | borg-append | borg-fragment |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | k8 | 64 | 8 | 160.6 | 2.000 | 261.2 | 2.000 |
|  | k16 | 256 | 16 | 161.8 | 1.000 | 232.2 | 1.000 |
|  | k32 | 1024 | 32 | 159.2 | 0.500 | 224.5 | 0.500 |
|  | k64 | 4096 | 64 | 166.6 | 0.250 | 236.7 | 0.250 |

## Point openings

Result-group ID: `tab:point-open`. CSV: `tab-point-open.csv`.

Helper work is microseconds for one helper's response construction and serialization; response cells are its serialized bytes. Client cells are Combine+Verify in milliseconds from kappa correct responses, and from 3*kappa responses of which kappa are corrupted.

| panel | row | segment | kappa | undine-helper | undine-response | undine-client | undine-client-correcting | borg-helper | borg-response | borg-client | borg-client-correcting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | k8 | 64 | 8 | 9.7 | 1251 | 0.224 | 1.181 | 181.7 | 1251 | 0.273 | 1.368 |
|  | k16 | 256 | 16 | 8.6 | 1251 | 0.322 | 9.889 | 206.7 | 1251 | 0.365 | 10.467 |
|  | k32 | 1024 | 32 | 11.0 | 1251 | 0.609 | 102.790 | 232.2 | 1251 | 0.693 | 103.299 |
|  | k64 | 4096 | 64 | 7.8 | 1251 | 1.678 | 774.565 | 305.8 | 1251 | 1.712 | 774.904 |

## Range openings at kappa = 64

Result-group ID: `tab:range-open`. CSV: `tab-range-open.csv`.

Per-helper response size is KiB. Helper work is one helper's answer to the range request and its serialization, in milliseconds. Client work is seconds, from 192 responses; the corrupt case has 64 corrupted responses among them.

| panel | row | range-mib | full-segment-reads | point-reads | undine-per-helper-kib | undine-helper-work | undine-clean-client | undine-corrupt-client | borg-per-helper-kib | borg-helper-work | borg-clean-client | borg-corrupt-client |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | range-1mib | 1 | 16 | 41 | 33.9 | 0.035 | 0.131 | 2.671 | 33.9 | 0.681 | 0.152 | 2.701 |
|  | range-4mib | 4 | 64 | 37 | 130.0 | 0.095 | 0.518 | 5.245 | 130.0 | 0.680 | 0.605 | 5.513 |
|  | range-8mib | 8 | 128 | 35 | 258.3 | 0.183 | 1.040 | 6.007 | 258.3 | 0.736 | 1.224 | 6.368 |
|  | range-32mib | 32 | 512 | 31 | 1028.9 | 0.768 | 4.456 | 10.833 | 1028.9 | 1.233 | 5.076 | 11.662 |
|  | range-128mib | 128 | 2048 | 27 | 4112.7 | 3.317 | 18.941 | 30.979 | 4112.7 | 3.784 | 21.790 | 33.986 |

## Repair and reconfiguration

Result-group ID: `tab:maintenance`. CSV: `tab-maintenance.csv`.

The panel is the layout. All quantities are normalized per 4 MiB appended to the evaluated 12 MiB prefix. Times are milliseconds; fragment, download, and state values are MiB. Double and halve move the threshold between kappa and 2*kappa. The diagonal halve columns are the UndineLog undo from source state at kappa; the segment-double column is BorgLog's local segment-and-kappa double.

| panel | row | segment | kappa | fragment | repair-receiver | repair-helper | repair-download | double-receiver | double-helper | double-download | state-after-double | halve-receiver | halve-helper | halve-download | halve-diagonal-receiver | halve-diagonal-download | segment-double-receiver |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| undine | undine:k8 | 64 | 8 | 2.000 | 35.869 | 3.821 | 2.000 | 1.412 | -- | 0.000 | 1.000 | 16.171 | 0.739 | 1.000 | 2.157 | 1.000 | -- |
| undine | undine:k16 | 256 | 16 | 1.000 | 21.180 | 1.316 | 1.000 | 0.616 | -- | 0.000 | 0.501 | 12.611 | 0.312 | 0.501 | 0.901 | 0.501 | -- |
| undine | undine:k32 | 1024 | 32 | 0.501 | 16.115 | 0.546 | 0.501 | 0.303 | -- | 0.000 | 0.251 | 11.180 | 0.150 | 0.251 | 0.395 | 0.251 | -- |
| undine | undine:k64 | 4096 | 64 | 0.251 | 14.673 | 0.259 | 0.251 | 0.190 | -- | 0.000 | 0.128 | 11.053 | 0.083 | 0.128 | 0.227 | 0.128 | -- |
| borg | borg:k8 | 64 | 8 | 2.000 | 433.453 | 29.705 | 2.000 | 91.511 | 11.743 | 1.875 | 2.500 | 207.601 | 13.386 | 1.375 | -- | -- | 66.274 |
| borg | borg:k16 | 256 | 16 | 1.000 | 158.081 | 9.295 | 1.000 | 70.492 | 3.698 | 0.969 | 1.250 | 24.372 | 4.496 | 0.719 | -- | -- | 36.536 |
| borg | borg:k32 | 1024 | 32 | 0.501 | 114.546 | 3.370 | 0.501 | 67.874 | 1.935 | 0.493 | 0.626 | 18.449 | 1.787 | 0.368 | -- | -- | 29.312 |
| borg | borg:k64 | 4096 | 64 | 0.251 | 80.242 | 1.297 | 0.251 | 70.133 | 2.797 | 0.249 | 0.314 | 17.494 | 0.718 | 0.187 | -- | -- | 29.852 |

## UndineLog openings served above the stored threshold

Result-group ID: `tab:on-the-fly-open`. CSV: `tab-on-the-fly-open.csv`.

One maintainer stores a 12 MiB log at the stored threshold and answers at the opened threshold. On-the-fly cells are that answer; stored cells are the answer of a maintainer that stores the log at the opened threshold. Both include response serialization and are microseconds; response sizes are KiB. Stitch-state is the time to stitch the maintainer's whole state to the opened threshold, in milliseconds.

| panel | row | stored-kappa | open-kappa | point-on-the-fly | point-stored | point-bytes | range-1mib-on-the-fly | range-1mib-stored | range-1mib-kib | range-4mib-on-the-fly | range-4mib-stored | range-4mib-kib | stitch-state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | k8-open8 | 8 | 8 | -- | 10.4 | 1251 | -- | 375.6 | 263.9 | -- | 1535.8 | 1055.8 | -- |
|  | k8-open16 | 8 | 16 | 12.7 | 8.9 | 1251 | 340.7 | 125.1 | 130.2 | 1335.1 | 499.0 | 520.1 | 4.469 |
|  | k8-open32 | 8 | 32 | 17.8 | 8.9 | 1251 | 337.3 | 52.1 | 65.5 | 1318.0 | 200.8 | 258.6 | 6.557 |
|  | k8-open64 | 8 | 64 | 26.0 | 8.6 | 1251 | 342.7 | 29.4 | 33.7 | 1283.5 | 94.1 | 129.8 | 8.243 |
|  | k16-open16 | 16 | 16 | -- | 9.5 | 1251 | -- | 127.3 | 130.2 | -- | 494.6 | 520.1 | -- |
|  | k16-open32 | 16 | 32 | 12.3 | 8.5 | 1251 | 135.8 | 51.5 | 65.5 | 529.9 | 194.8 | 258.6 | 2.181 |
|  | k16-open64 | 16 | 64 | 16.8 | 8.4 | 1251 | 148.1 | 28.7 | 33.7 | 557.4 | 92.5 | 129.8 | 3.868 |
|  | k16-open128 | 16 | 128 | 24.4 | 8.1 | 1251 | 161.6 | 16.3 | 17.7 | 568.3 | 43.5 | 65.6 | 7.177 |
|  | k32-open32 | 32 | 32 | -- | 9.7 | 1251 | -- | 56.8 | 65.5 | -- | 216.0 | 258.6 | -- |
|  | k32-open64 | 32 | 64 | 12.4 | 8.7 | 1251 | 69.4 | 30.7 | 33.7 | 241.4 | 96.8 | 129.8 | 1.867 |
|  | k32-open128 | 32 | 128 | 16.2 | 8.2 | 1251 | 76.3 | 17.0 | 17.7 | 256.4 | 44.7 | 65.6 | 5.279 |
|  | k32-open256 | 32 | 256 | 22.9 | 8.0 | 1251 | 89.1 | 11.5 | 9.6 | 269.9 | 24.3 | 33.5 | 16.681 |
|  | k64-open64 | 64 | 64 | -- | 10.0 | 1251 | -- | 30.1 | 33.7 | -- | 93.3 | 129.8 | -- |
|  | k64-open128 | 64 | 128 | 12.0 | 8.2 | 1251 | 37.7 | 16.8 | 17.7 | 112.5 | 45.0 | 65.6 | 3.793 |
|  | k64-open256 | 64 | 256 | 15.6 | 8.0 | 1251 | 45.0 | 11.5 | 9.6 | 126.5 | 24.2 | 33.5 | 15.473 |
|  | k64-open512 | 64 | 512 | 21.4 | 8.0 | 1251 | 126.0 | 8.8 | 16.1 | 138.2 | 14.9 | 17.5 | 64.464 |

## Private openings of one position: one claim of a maintainer

Result-group ID: `tab:private-maintainer`. CSV: `tab-private-maintainer.csv`.

One claim answers from its primary columns, grouped into blocks of the public grouping of the log length (one segment per block at 1 GiB, doubling every two levels) unless the row gives another block size. A log of 1 GiB has 2^25 leaves. The record of a pattern is the two blocks of its subtree and one entry per level above. Database is the bytes of the primary columns; state adds the row table and the preprocessing; both are MiB. Request and reply are serialized KiB. The client's request (key, query, serialization), the first pass, the complete answer from serialized request to serialized reply, and the client's decoding of one reply are milliseconds on one core. Preprocessing is core-seconds: the measured hint of 64 database columns scaled to all record columns, plus the measured packing tables of one reply ciphertext times the ciphertext count. The update cells follow a 4 MiB append: the hint patch in milliseconds and the packing tables in seconds.

| panel | row | kappa | log-gib | record-kib | database | state | request | reply-kib | client-request-ms | first-pass | answer | client-decode-ms | preprocess-core-s | update-hint-ms | update-packing-s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| log | log:k64-log1gib-r1 | 64 | 1 | 2.0 | 32.0 | 743.1 | 574.0 | 24.0 | 38.9 | 7.7 | 41.1 | 0.6 | 3.8 | 236.2 | 2.0 |
| log | log:k64-log4gib-r2 | 64 | 4 | 4.0 | 128.0 | 1207.2 | 686.0 | 36.0 | 43.8 | 26.4 | 75.2 | 0.9 | 9.2 | 346.8 | 3.0 |
| log | log:k64-log16gib-r4 | 64 | 16 | 8.0 | 512.0 | 2355.2 | 910.0 | 60.0 | 53.5 | 104.3 | 181.4 | 1.5 | 29.0 | 656.5 | 5.0 |
| log | log:k64-log64gib-r8 | 64 | 64 | 16.0 | 2048.0 | 5881.2 | 1358.0 | 120.0 | 69.8 | 373.9 | 506.9 | 3.0 | 105.9 | 1400.1 | 10.0 |
| kappa | kappa:k8-log4gib-r2 | 8 | 4 | 0.5 | 1024.0 | 1860.1 | 14798.0 | 12.0 | 602.4 | 320.0 | 342.7 | 0.3 | 73.4 | 4667.7 | 1.0 |
| kappa | kappa:k16-log4gib-r2 | 16 | 4 | 1.0 | 512.0 | 1018.1 | 4046.0 | 12.0 | 182.0 | 136.3 | 162.1 | 0.3 | 29.8 | 1435.0 | 1.0 |
| kappa | kappa:k32-log4gib-r2 | 32 | 4 | 2.0 | 256.0 | 1016.0 | 1358.0 | 24.0 | 71.8 | 62.3 | 96.5 | 0.6 | 15.0 | 554.5 | 2.0 |
| block | block:k64-log64gib-r1 | 64 | 64 | 2.0 | 2048.0 | 3210.3 | 7630.0 | 24.0 | 313.5 | 461.9 | 508.8 | 0.6 | 108.9 | 2041.0 | 2.0 |
| block | block:k64-log64gib-r2 | 64 | 64 | 4.0 | 2048.0 | 3477.4 | 4046.0 | 36.0 | 174.3 | 394.8 | 445.2 | 0.9 | 102.6 | 1371.7 | 3.0 |
| block | block:k64-log64gib-r4 | 64 | 64 | 8.0 | 2048.0 | 4140.7 | 2254.0 | 60.0 | 104.8 | 366.9 | 457.3 | 1.5 | 100.7 | 1160.1 | 5.0 |

## Private openings of one position: a single server with the whole tree

Result-group ID: `tab:private-single`. CSV: `tab-private-single.csv`.

The server stores every cell of the tree in blocks of the given number of cells and answers with the code path of a maintainer: the record of a pattern is the two blocks of its subtree and one cell per level above. Database and state are GiB, the request is MiB, and the reply is KiB. Client request and decoding are milliseconds; first pass and answer are seconds on one core. Preprocessing is core-seconds, measured as for a maintainer. The rows at 16 and 64 GiB give the database dimensions and message sizes of those logs.

| panel | row | log-gib | record-kib | database | state | request | reply-kib | client-request-ms | first-pass | answer | client-decode-ms | preprocess-core-s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| measured | measured:log1gib-c256 | 1 | 8.0 | 2.0 | 4.0 | 2.2 | 60.0 | 105.8 | 0.390 | 0.473 | 1.5 | 101.1 |
| measured | measured:log1gib-c512 | 1 | 16.0 | 2.0 | 5.7 | 1.3 | 120.0 | 70.6 | 0.370 | 0.504 | 3.1 | 106.0 |
| measured | measured:log1gib-c1024 | 1 | 32.0 | 2.0 | 8.8 | 0.9 | 228.1 | 52.0 | 0.354 | 0.613 | 5.8 | 117.5 |
| measured | measured:log4gib-c512 | 4 | 16.0 | 8.0 | 12.7 | 4.0 | 120.0 | 175.7 | 1.473 | 1.536 | 3.0 | 395.8 |
| shape | shape:log16gib-c512 | 16 | 16.0 | 32.0 | -- | 14.5 | 120.0 | -- | -- | -- | -- | -- |
| shape | shape:log64gib-c512 | 64 | 16.0 | 128.0 | -- | 56.5 | 120.0 | -- | -- | -- | -- | -- |

## Records against the table of complete openings at kappa = 64

Result-group ID: `tab:private-table`. CSV: `tab-private-table.csv`.

The table holds one complete opening per pattern and is answered by YPIR's server; the records are the maintainer's database of the same claim. Both give the same reply bytes. Sizes are MiB; first pass and answer are milliseconds on one core.

| panel | row | log-gib | records-mib | table-mib | records-first-pass-ms | table-first-pass-ms | records-answer-ms | table-answer-ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | log1gib | 1 | 32.0 | 512.0 | 7.7 | 27.9 | 41.1 | 62.5 |
|  | log4gib | 4 | 128.0 | 2176.0 | 26.4 | 83.6 | 75.2 | 134.0 |

## Client work for one private opening

Result-group ID: `tab:private-client`. CSV: `tab-private-client.csv`.

A 12 MiB log at the public grouping. Request and reply are serialized KiB; the reply is one claim's. Request is key generation, the query, and serialization in milliseconds. Private cells decode the replies and run Combine+Verify; public cells run Combine+Verify on the public replies of the same claims. The first pair uses kappa correct replies; the second uses 3*kappa replies of which kappa lie about one node. All are milliseconds.

| panel | row | kappa | request-kib | reply-kib | client-request-ms | private-kappa-ms | public-kappa-ms | private-3kappa-ms | public-3kappa-ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | k8 | 8 | 574.0 | 12.0 | 40.2 | 2.845 | 0.255 | 8.991 | 1.257 |
|  | k16 | 16 | 490.0 | 12.0 | 39.2 | 5.706 | 0.376 | 26.393 | 10.807 |
|  | k32 | 32 | 476.0 | 12.0 | 37.0 | 10.793 | 0.672 | 101.992 | 71.699 |
|  | k64 | 64 | 476.0 | 24.0 | 36.8 | 40.242 | 1.737 | 668.407 | 550.087 |
