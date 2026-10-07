# Evaluation result groups

Values are reviewer-facing renderings. Machine-readable source values, units, samples, and confidence intervals are in `../summary.json` and `../summary.tsv`.

## Append processing and fragment size

Result-group ID: `tab:append`. CSV: `tab-append.csv`.

One row per threshold, with Seg = kappa^2. Append cells are one maintainer's mean processing time for a 4 MiB batch in milliseconds. Fragment cells are the mean newly materialized fragment size per batch in MiB.

| panel | row | segment | kappa | undine-append | undine-fragment | borg-append | borg-fragment |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | k8 | 64 | 8 | 160.6 | 2.000 | 261.8 | 2.000 |
|  | k16 | 256 | 16 | 160.5 | 1.000 | 231.7 | 1.000 |
|  | k32 | 1024 | 32 | 159.8 | 0.500 | 224.7 | 0.500 |
|  | k64 | 4096 | 64 | 167.2 | 0.250 | 236.6 | 0.250 |

## Point openings

Result-group ID: `tab:point-open`. CSV: `tab-point-open.csv`.

Helper work is microseconds for one helper's response construction and serialization; response cells are its serialized bytes. Client cells are Combine+Verify in milliseconds from kappa correct responses, and from 3*kappa responses of which kappa are corrupted.

| panel | row | segment | kappa | undine-helper | undine-response | undine-client | undine-client-correcting | borg-helper | borg-response | borg-client | borg-client-correcting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | k8 | 64 | 8 | 10.6 | 1251 | 0.227 | 1.154 | 176.0 | 1251 | 0.272 | 1.394 |
|  | k16 | 256 | 16 | 9.9 | 1251 | 0.318 | 9.858 | 192.5 | 1251 | 0.367 | 10.430 |
|  | k32 | 1024 | 32 | 8.5 | 1251 | 0.603 | 102.301 | 230.0 | 1251 | 0.673 | 102.784 |
|  | k64 | 4096 | 64 | 8.5 | 1251 | 1.605 | 774.649 | 316.1 | 1251 | 1.641 | 774.305 |

## Range openings at kappa = 64

Result-group ID: `tab:range-open`. CSV: `tab-range-open.csv`.

Per-helper response size is KiB. Helper work is one helper's answer to the range request and its serialization, in milliseconds. Client work is seconds, from 192 responses; the corrupt case has 64 corrupted responses among them.

| panel | row | range-mib | full-segment-reads | point-reads | undine-per-helper-kib | undine-helper-work | undine-clean-client | undine-corrupt-client | borg-per-helper-kib | borg-helper-work | borg-clean-client | borg-corrupt-client |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | range-1mib | 1 | 16 | 41 | 33.9 | 0.030 | 0.131 | 2.621 | 33.9 | 0.681 | 0.154 | 2.705 |
|  | range-4mib | 4 | 64 | 37 | 130.0 | 0.094 | 0.518 | 5.168 | 130.0 | 0.681 | 0.599 | 5.503 |
|  | range-8mib | 8 | 128 | 35 | 258.3 | 0.182 | 1.039 | 5.924 | 258.3 | 0.738 | 1.215 | 6.369 |
|  | range-32mib | 32 | 512 | 31 | 1028.9 | 0.718 | 4.396 | 10.719 | 1028.9 | 1.203 | 5.075 | 11.676 |
|  | range-128mib | 128 | 2048 | 27 | 4112.7 | 3.331 | 18.974 | 31.077 | 4112.7 | 3.629 | 21.877 | 33.983 |

## Repair and reconfiguration

Result-group ID: `tab:maintenance`. CSV: `tab-maintenance.csv`.

The panel is the layout. All quantities are normalized per 4 MiB appended to the evaluated 12 MiB prefix. Times are milliseconds; fragment, download, and state values are MiB. Double and halve move the threshold between kappa and 2*kappa. The diagonal halve columns are the UndineLog undo from source state at kappa; the segment-double column is BorgLog's local segment-and-kappa double.

| panel | row | segment | kappa | fragment | repair-receiver | repair-helper | repair-download | double-receiver | double-helper | double-download | state-after-double | halve-receiver | halve-helper | halve-download | halve-diagonal-receiver | halve-diagonal-download | segment-double-receiver |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| undine | undine:k8 | 64 | 8 | 2.000 | 37.203 | 3.796 | 2.000 | 1.421 | -- | 0.000 | 1.000 | 16.356 | 0.713 | 1.000 | 2.217 | 1.000 | -- |
| undine | undine:k16 | 256 | 16 | 1.000 | 21.868 | 1.315 | 1.000 | 0.624 | -- | 0.000 | 0.501 | 12.623 | 0.311 | 0.501 | 0.889 | 0.501 | -- |
| undine | undine:k32 | 1024 | 32 | 0.501 | 15.492 | 0.539 | 0.501 | 0.308 | -- | 0.000 | 0.251 | 11.160 | 0.152 | 0.251 | 0.405 | 0.251 | -- |
| undine | undine:k64 | 4096 | 64 | 0.251 | 14.946 | 0.264 | 0.251 | 0.194 | -- | 0.000 | 0.128 | 10.999 | 0.080 | 0.128 | 0.227 | 0.128 | -- |
| borg | borg:k8 | 64 | 8 | 2.000 | 433.606 | 29.693 | 2.000 | 90.753 | 11.613 | 1.875 | 2.500 | 207.113 | 13.356 | 1.375 | -- | -- | 66.789 |
| borg | borg:k16 | 256 | 16 | 1.000 | 158.493 | 9.257 | 1.000 | 70.298 | 3.706 | 0.969 | 1.250 | 24.392 | 4.450 | 0.719 | -- | -- | 36.882 |
| borg | borg:k32 | 1024 | 32 | 0.501 | 113.304 | 3.361 | 0.501 | 67.596 | 1.935 | 0.493 | 0.626 | 18.623 | 1.815 | 0.368 | -- | -- | 29.255 |
| borg | borg:k64 | 4096 | 64 | 0.251 | 79.989 | 1.279 | 0.251 | 70.308 | 2.792 | 0.249 | 0.314 | 17.416 | 0.707 | 0.187 | -- | -- | 29.953 |

## UndineLog openings served above the stored threshold

Result-group ID: `tab:on-the-fly-open`. CSV: `tab-on-the-fly-open.csv`.

One maintainer stores a 12 MiB log at the stored threshold and answers at the opened threshold. On-the-fly cells are that answer; stored cells are the answer of a maintainer that stores the log at the opened threshold. Both include response serialization and are microseconds; response sizes are KiB. Stitch-state is the time to stitch the maintainer's whole state to the opened threshold, in milliseconds.

| panel | row | stored-kappa | open-kappa | point-on-the-fly | point-stored | point-bytes | range-1mib-on-the-fly | range-1mib-stored | range-1mib-kib | range-4mib-on-the-fly | range-4mib-stored | range-4mib-kib | stitch-state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | k8-open8 | 8 | 8 | -- | 10.5 | 1251 | -- | 360.4 | 263.9 | -- | 1496.7 | 1055.8 | -- |
|  | k8-open16 | 8 | 16 | 12.4 | 8.9 | 1251 | 339.7 | 124.5 | 130.2 | 1342.8 | 491.6 | 520.1 | 4.511 |
|  | k8-open32 | 8 | 32 | 17.8 | 8.8 | 1251 | 335.4 | 51.4 | 65.5 | 1307.3 | 193.9 | 258.6 | 6.564 |
|  | k8-open64 | 8 | 64 | 25.8 | 8.4 | 1251 | 343.7 | 28.2 | 33.7 | 1271.5 | 90.6 | 129.8 | 8.347 |
|  | k16-open16 | 16 | 16 | -- | 9.9 | 1251 | -- | 131.7 | 130.2 | -- | 502.2 | 520.1 | -- |
|  | k16-open32 | 16 | 32 | 12.4 | 8.7 | 1251 | 136.7 | 51.6 | 65.5 | 539.7 | 200.6 | 258.6 | 2.222 |
|  | k16-open64 | 16 | 64 | 17.1 | 8.6 | 1251 | 150.9 | 28.9 | 33.7 | 560.4 | 93.3 | 129.8 | 3.910 |
|  | k16-open128 | 16 | 128 | 24.7 | 8.3 | 1251 | 163.0 | 16.8 | 17.7 | 574.7 | 45.1 | 65.6 | 7.219 |
|  | k32-open32 | 32 | 32 | -- | 9.7 | 1251 | -- | 54.4 | 65.5 | -- | 203.8 | 258.6 | -- |
|  | k32-open64 | 32 | 64 | 12.2 | 8.5 | 1251 | 70.3 | 29.3 | 33.7 | 244.7 | 93.8 | 129.8 | 1.885 |
|  | k32-open128 | 32 | 128 | 16.4 | 8.4 | 1251 | 79.1 | 17.4 | 17.7 | 259.9 | 44.5 | 65.6 | 5.282 |
|  | k32-open256 | 32 | 256 | 22.5 | 8.0 | 1251 | 90.4 | 11.6 | 9.6 | 275.3 | 23.7 | 33.5 | 16.631 |
|  | k64-open64 | 64 | 64 | -- | 10.4 | 1251 | -- | 31.3 | 33.7 | -- | 97.7 | 129.8 | -- |
|  | k64-open128 | 64 | 128 | 12.3 | 8.3 | 1251 | 38.0 | 17.0 | 17.7 | 114.2 | 45.8 | 65.6 | 3.784 |
|  | k64-open256 | 64 | 256 | 15.6 | 8.1 | 1251 | 45.3 | 11.4 | 9.6 | 128.4 | 24.1 | 33.5 | 15.341 |
|  | k64-open512 | 64 | 512 | 21.2 | 7.9 | 1251 | 125.6 | 8.7 | 16.1 | 141.6 | 14.8 | 17.5 | 63.812 |

## Private openings of one position: one claim of a maintainer

Result-group ID: `tab:private-maintainer`. CSV: `tab-private-maintainer.csv`.

One claim answers from its primary columns, grouped into blocks of the public grouping of the log length (one segment per block at 1 GiB, doubling every two levels) unless the row gives another block size. A log of 1 GiB has 2^25 leaves. The record of a pattern is the two blocks of its subtree and one entry per level above. Database is the bytes of the primary columns; state adds the row table and the preprocessing; both are MiB. Request and reply are serialized KiB. The client's request (key, query, serialization), the first pass, the complete answer from serialized request to serialized reply, and the client's decoding of one reply are milliseconds on one core. Preprocessing is core-seconds: the measured hint of 64 database columns scaled to all record columns, plus the measured packing tables of one reply ciphertext times the ciphertext count. The update cells follow a 4 MiB append: the hint patch in milliseconds and the packing tables in seconds.

| panel | row | kappa | log-gib | record-kib | database | state | request | reply-kib | client-request-ms | first-pass | answer | client-decode-ms | preprocess-core-s | update-hint-ms | update-packing-s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| log | log:k64-log1gib-p13 | 64 | 1 | 2.0 | 32.0 | 421.7 | 490.0 | 36.0 | 35.5 | 5.3 | 51.6 | 0.9 | 4.5 | 105.0 | 2.9 |
| log | log:k64-log4gib-p14 | 64 | 4 | 2.0 | 128.0 | 916.8 | 518.0 | 72.0 | 36.3 | 20.5 | 112.4 | 1.8 | 10.2 | 128.1 | 5.9 |
| log | log:k64-log16gib-p15 | 64 | 16 | 2.0 | 512.0 | 1870.5 | 574.0 | 120.0 | 39.3 | 81.3 | 232.0 | 3.1 | 24.6 | 178.0 | 9.8 |
| log | log:k64-log64gib-p16 | 64 | 64 | 2.0 | 2048.0 | 4914.1 | 686.0 | 240.1 | 42.9 | 315.4 | 619.8 | 6.0 | 79.5 | 281.3 | 19.7 |
| kappa | kappa:k8-log4gib-p12 | 8 | 4 | 0.2 | 1024.0 | 2478.3 | 686.0 | 120.0 | 43.6 | 189.8 | 339.1 | 3.0 | 38.7 | 1201.9 | 10.0 |
| kappa | kappa:k16-log4gib-p13 | 16 | 4 | 0.5 | 512.0 | 1874.0 | 574.0 | 120.0 | 39.5 | 83.7 | 233.7 | 3.1 | 23.9 | 487.8 | 9.8 |
| kappa | kappa:k32-log4gib-p13 | 32 | 4 | 1.0 | 256.0 | 937.1 | 574.0 | 60.0 | 39.7 | 39.1 | 116.8 | 1.6 | 12.3 | 226.8 | 5.8 |
| pattern | pattern:k64-log64gib-p14 | 64 | 64 | 2.0 | 2048.0 | 3133.6 | 1358.0 | 72.0 | 69.8 | 326.8 | 416.3 | 1.8 | 64.5 | 385.9 | 6.0 |
| pattern | pattern:k64-log64gib-p15 | 64 | 64 | 2.0 | 2048.0 | 3636.9 | 910.0 | 120.0 | 52.3 | 319.8 | 465.2 | 3.0 | 67.9 | 293.2 | 10.1 |
| pattern | pattern:k64-log64gib-p17 | 64 | 64 | 2.0 | 2048.0 | 7219.3 | 574.0 | 456.1 | 38.5 | 324.7 | 876.4 | 11.5 | 95.3 | 380.0 | 37.5 |

## Private openings of one position: a single server with the whole tree

Result-group ID: `tab:private-single`. CSV: `tab-private-single.csv`.

The server stores every cell of the tree in blocks of the given number of cells and answers with the code path of a maintainer: the record of a pattern is the two blocks of its subtree and one cell per level above. Database and state are GiB, the request is MiB, and the reply is KiB. Client request and decoding are milliseconds; first pass and answer are seconds on one core. Preprocessing is core-seconds, measured as for a maintainer. The rows at 16 and 64 GiB give the database dimensions and message sizes of those logs.

| panel | row | log-gib | record-kib | database | state | request | reply-kib | client-request-ms | first-pass | answer | client-decode-ms | preprocess-core-s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| measured | measured:log1gib-p10 | 1 | 2.0 | 2.0 | 4.8 | 0.7 | 240.1 | 42.6 | 0.318 | 0.618 | 5.9 | 77.0 |
| measured | measured:log1gib-p9 | 1 | 2.0 | 2.0 | 3.6 | 0.9 | 120.0 | 51.9 | 0.322 | 0.470 | 3.0 | 67.3 |
| measured | measured:log1gib-p11 | 1 | 2.0 | 2.0 | 7.1 | 0.6 | 456.1 | 39.0 | 0.327 | 0.884 | 11.3 | 96.5 |
| measured | measured:log4gib-p11 | 4 | 2.0 | 8.0 | 13.9 | 0.9 | 456.1 | 51.8 | 1.378 | 1.928 | 11.5 | 311.5 |
| shape | shape:log16gib-p12 | 16 | 2.0 | 32.0 | -- | 1.3 | 888.2 | -- | -- | -- | -- | -- |
| shape | shape:log64gib-p13 | 64 | 2.0 | 128.0 | -- | 2.2 | 1776.4 | -- | -- | -- | -- | -- |

## Records against the table of complete openings at kappa = 64

Result-group ID: `tab:private-table`. CSV: `tab-private-table.csv`.

The table holds one complete opening per pattern and is answered by YPIR's server; the records are the maintainer's database of the same claim. Both give the same reply bytes. Sizes are MiB; first pass and answer are milliseconds on one core.

| panel | row | log-gib | records-mib | table-mib | records-first-pass-ms | table-first-pass-ms | records-answer-ms | table-answer-ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | log1gib | 1 | 32.0 | 136.0 | 5.3 | 13.1 | 51.6 | 62.1 |
|  | log4gib | 4 | 128.0 | 352.0 | 20.5 | 42.1 | 112.4 | 133.5 |

## Client work for one private opening

Result-group ID: `tab:private-client`. CSV: `tab-private-client.csv`.

A 12 MiB log at every threshold and the 1 and 4 GiB logs at threshold 64, at the public grouping. Request and reply are serialized KiB; the reply is one claim's. Request is key generation, the query, and serialization in milliseconds. Private cells decode the replies and run Combine+Verify; public cells run Combine+Verify on the public replies of the same claims. The first pair uses kappa correct replies; the second uses 3*kappa replies of which kappa lie about one node. All are milliseconds.

| panel | row | kappa | log-mib | request-kib | reply-kib | client-request-ms | private-kappa-ms | public-kappa-ms | private-3kappa-ms | public-3kappa-ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | k8 | 8 | 12 | 476.0 | 12.0 | 36.6 | 2.921 | 0.253 | 9.155 | 1.275 |
|  | k16 | 16 | 12 | 476.0 | 12.0 | 38.2 | 5.661 | 0.374 | 26.604 | 10.869 |
|  | k32 | 32 | 12 | 476.0 | 12.0 | 37.7 | 10.984 | 0.705 | 106.305 | 75.154 |
|  | k64 | 64 | 12 | 476.0 | 24.0 | 35.9 | 40.300 | 1.859 | 685.384 | 570.519 |
| log | log:log1gib-k64 | 64 | 1024 | 490.0 | 36.0 | 36.7 | 60.071 | 2.361 | 873.270 | 696.802 |
| log | log:log4gib-k64 | 64 | 4096 | 518.0 | 72.0 | 37.5 | 116.370 | 2.538 | 1083.376 | 738.765 |
