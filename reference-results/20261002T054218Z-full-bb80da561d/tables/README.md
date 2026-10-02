# Evaluation result groups

Values are reviewer-facing renderings. Machine-readable source values, units, samples, and confidence intervals are in `../summary.json` and `../summary.tsv`.

## Append processing and fragment size

Result-group ID: `tab:append`. CSV: `tab-append.csv`.

One row per threshold, with Seg = kappa^2. Append cells are one maintainer's mean processing time for a 4 MiB batch in milliseconds. Fragment cells are the mean newly materialized fragment size per batch in MiB.

| panel | row | segment | kappa | undine-append | undine-fragment | borg-append | borg-fragment |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | k8 | 64 | 8 | 161.2 | 2.000 | 262.7 | 2.000 |
|  | k16 | 256 | 16 | 159.9 | 1.000 | 231.5 | 1.000 |
|  | k32 | 1024 | 32 | 159.8 | 0.500 | 224.9 | 0.500 |
|  | k64 | 4096 | 64 | 163.9 | 0.250 | 236.0 | 0.250 |

## Point openings

Result-group ID: `tab:point-open`. CSV: `tab-point-open.csv`.

Helper work is microseconds for one helper's response construction and serialization; response cells are its serialized bytes. Client cells are Combine+Verify in milliseconds from kappa correct responses, and from 3*kappa responses of which kappa are corrupted.

| panel | row | segment | kappa | undine-helper | undine-response | undine-client | undine-client-correcting | borg-helper | borg-response | borg-client | borg-client-correcting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | k8 | 64 | 8 | 9.9 | 1251 | 0.212 | 1.143 | 175.4 | 1251 | 0.275 | 1.377 |
|  | k16 | 256 | 16 | 9.4 | 1251 | 0.319 | 9.819 | 196.8 | 1251 | 0.371 | 10.362 |
|  | k32 | 1024 | 32 | 8.1 | 1251 | 0.608 | 101.174 | 230.2 | 1251 | 0.690 | 101.952 |
|  | k64 | 4096 | 64 | 8.0 | 1251 | 1.670 | 766.741 | 323.6 | 1251 | 1.667 | 766.144 |

## Range openings at kappa = 64

Result-group ID: `tab:range-open`. CSV: `tab-range-open.csv`.

Per-helper response size is KiB. Helper work is one helper's answer to the range request and its serialization, in milliseconds. Client work is seconds, from 192 responses; the corrupt case has 64 corrupted responses among them.

| panel | row | range-mib | full-segment-reads | point-reads | undine-per-helper-kib | undine-helper-work | undine-clean-client | undine-corrupt-client | borg-per-helper-kib | borg-helper-work | borg-clean-client | borg-corrupt-client |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | range-1mib | 1 | 16 | 41 | 33.9 | 0.030 | 0.131 | 2.569 | 33.9 | 0.677 | 0.153 | 2.689 |
|  | range-4mib | 4 | 64 | 37 | 130.0 | 0.094 | 0.518 | 5.085 | 130.0 | 0.729 | 0.599 | 5.471 |
|  | range-8mib | 8 | 128 | 35 | 258.3 | 0.183 | 1.031 | 5.844 | 258.3 | 0.758 | 1.222 | 6.350 |
|  | range-32mib | 32 | 512 | 31 | 1028.9 | 0.788 | 4.389 | 10.643 | 1028.9 | 1.244 | 5.107 | 11.689 |
|  | range-128mib | 128 | 2048 | 27 | 4112.7 | 3.447 | 18.739 | 30.749 | 4112.7 | 4.101 | 21.728 | 34.028 |

## Repair and reconfiguration

Result-group ID: `tab:maintenance`. CSV: `tab-maintenance.csv`.

The panel is the layout. All quantities are normalized per 4 MiB appended to the evaluated 12 MiB prefix. Times are milliseconds; fragment, download, and state values are MiB. Double and halve move the threshold between kappa and 2*kappa. The diagonal halve columns are the UndineLog undo from source state at kappa; the segment-double column is BorgLog's local segment-and-kappa double.

| panel | row | segment | kappa | fragment | repair-receiver | repair-helper | repair-download | double-receiver | double-helper | double-download | state-after-double | halve-receiver | halve-helper | halve-download | halve-diagonal-receiver | halve-diagonal-download | segment-double-receiver |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| undine | undine:k8 | 64 | 8 | 2.000 | 37.462 | 3.815 | 2.000 | 1.444 | -- | 0.000 | 1.000 | 16.238 | 0.754 | 1.000 | 2.155 | 1.000 | -- |
| undine | undine:k16 | 256 | 16 | 1.000 | 22.186 | 1.320 | 1.000 | 0.623 | -- | 0.000 | 0.501 | 12.773 | 0.319 | 0.501 | 0.887 | 0.501 | -- |
| undine | undine:k32 | 1024 | 32 | 0.501 | 16.077 | 0.564 | 0.501 | 0.308 | -- | 0.000 | 0.251 | 11.120 | 0.151 | 0.251 | 0.409 | 0.251 | -- |
| undine | undine:k64 | 4096 | 64 | 0.251 | 15.189 | 0.262 | 0.251 | 0.193 | -- | 0.000 | 0.128 | 10.942 | 0.081 | 0.128 | 0.227 | 0.128 | -- |
| borg | borg:k8 | 64 | 8 | 2.000 | 434.423 | 29.783 | 2.000 | 91.324 | 11.701 | 1.875 | 2.500 | 205.049 | 13.475 | 1.375 | -- | -- | 66.260 |
| borg | borg:k16 | 256 | 16 | 1.000 | 159.304 | 9.317 | 1.000 | 69.649 | 3.707 | 0.969 | 1.250 | 24.363 | 4.524 | 0.719 | -- | -- | 36.741 |
| borg | borg:k32 | 1024 | 32 | 0.501 | 113.508 | 3.359 | 0.501 | 67.505 | 1.941 | 0.493 | 0.626 | 18.530 | 1.869 | 0.368 | -- | -- | 29.284 |
| borg | borg:k64 | 4096 | 64 | 0.251 | 80.222 | 1.332 | 0.251 | 70.023 | 2.796 | 0.249 | 0.314 | 17.360 | 0.716 | 0.187 | -- | -- | 29.872 |

## UndineLog openings served above the stored threshold

Result-group ID: `tab:on-the-fly-open`. CSV: `tab-on-the-fly-open.csv`.

One maintainer stores a 12 MiB log at the stored threshold and answers at the opened threshold. On-the-fly cells are that answer; stored cells are the answer of a maintainer that stores the log at the opened threshold. Both include response serialization and are microseconds; response sizes are KiB. Stitch-state is the time to stitch the maintainer's whole state to the opened threshold, in milliseconds.

| panel | row | stored-kappa | open-kappa | point-on-the-fly | point-stored | point-bytes | range-1mib-on-the-fly | range-1mib-stored | range-1mib-kib | range-4mib-on-the-fly | range-4mib-stored | range-4mib-kib | stitch-state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | k8-open8 | 8 | 8 | -- | 10.2 | 1251 | -- | 361.5 | 263.9 | -- | 1518.6 | 1055.8 | -- |
|  | k8-open16 | 8 | 16 | 12.6 | 8.9 | 1251 | 339.4 | 125.8 | 130.2 | 1355.8 | 509.1 | 520.1 | 4.450 |
|  | k8-open32 | 8 | 32 | 17.6 | 8.8 | 1251 | 339.2 | 51.3 | 65.5 | 1324.0 | 200.3 | 258.6 | 6.506 |
|  | k8-open64 | 8 | 64 | 25.9 | 8.5 | 1251 | 344.4 | 28.6 | 33.7 | 1288.3 | 91.0 | 129.8 | 8.206 |
|  | k16-open16 | 16 | 16 | -- | 9.5 | 1251 | -- | 125.0 | 130.2 | -- | 491.6 | 520.1 | -- |
|  | k16-open32 | 16 | 32 | 12.2 | 8.7 | 1251 | 139.8 | 51.5 | 65.5 | 536.6 | 196.6 | 258.6 | 2.174 |
|  | k16-open64 | 16 | 64 | 16.8 | 8.5 | 1251 | 150.7 | 28.2 | 33.7 | 562.2 | 91.3 | 129.8 | 3.858 |
|  | k16-open128 | 16 | 128 | 24.2 | 8.1 | 1251 | 162.3 | 16.1 | 17.7 | 573.9 | 43.2 | 65.6 | 7.142 |
|  | k32-open32 | 32 | 32 | -- | 9.7 | 1251 | -- | 55.1 | 65.5 | -- | 204.0 | 258.6 | -- |
|  | k32-open64 | 32 | 64 | 12.3 | 8.5 | 1251 | 69.5 | 28.7 | 33.7 | 239.5 | 93.1 | 129.8 | 1.852 |
|  | k32-open128 | 32 | 128 | 16.2 | 8.3 | 1251 | 78.0 | 16.8 | 17.7 | 260.0 | 44.2 | 65.6 | 5.223 |
|  | k32-open256 | 32 | 256 | 22.7 | 8.0 | 1251 | 91.2 | 11.6 | 9.6 | 275.2 | 24.3 | 33.5 | 16.482 |
|  | k64-open64 | 64 | 64 | -- | 10.0 | 1251 | -- | 31.1 | 33.7 | -- | 94.8 | 129.8 | -- |
|  | k64-open128 | 64 | 128 | 12.1 | 8.3 | 1251 | 38.0 | 16.9 | 17.7 | 114.2 | 45.3 | 65.6 | 3.769 |
|  | k64-open256 | 64 | 256 | 16.2 | 8.6 | 1251 | 46.9 | 12.5 | 9.6 | 130.8 | 25.5 | 33.5 | 15.353 |
|  | k64-open512 | 64 | 512 | 20.9 | 7.8 | 1251 | 125.1 | 8.6 | 16.1 | 140.5 | 15.5 | 17.5 | 63.956 |
