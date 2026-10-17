#!/usr/bin/env python3
"""Demand-driven reproduction of the artifact's eight evaluation result groups.

The manifest is authoritative: every scheduled benchmark metric has one or
more exact result-cell consumers, and every result cell has an explicit source.
Runs are isolated under ``results/runs`` and do not depend on paper sources.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import pathlib
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "results" / "runs"
GATES_ROOT = ROOT / "results" / "gates"
PIPELINE_VERSION = 2
FIELD_BYTES = 32
TREE_HEIGHT = 50
APPEND_SEQUENCE_BLOCKS = 64
APPEND_SEQUENCE_WARMUP_BLOCKS = 0
DEFAULT_MIN_REPETITIONS = 10
DIRECT_WARMUP_REPETITIONS = 3
APPEND_WARMUP_REPETITIONS = 1
# A repetition that rebuilds a server from the start has warm code and a warm
# allocator after one rebuild.
REBUILD_WARMUP_REPETITIONS = 1
DEFAULT_RELATIVE_HALF_WIDTH = 0.05
# Short operations can need thousands of observations when occasional delays
# dominate their variance. Declare the entire schedule before a run; the
# confidence calculation divides its error budget across all of these looks.
SEQUENTIAL_LOOKS = tuple(DEFAULT_MIN_REPETITIONS * 2**look for look in range(12))
DEFAULT_MAX_REPETITIONS = SEQUENTIAL_LOOKS[-1]
SEQUENTIAL_FAMILY_ALPHA = 0.05
MIB = 1024 * 1024
KIB = 1024

LAYOUTS = ("undine", "borg")
CRATE_ROOTS = {layout: ROOT / layout for layout in LAYOUTS}
KAPPAS = (8, 16, 32, 64)
RANGE_KAPPA = 64
RANGE_SIZES_MIB = (1, 4, 8, 32, 128)
ON_THE_FLY_LEVELS = 3
ON_THE_FLY_RANGES_MIB = (1, 4)
PREFIX_BATCHES = 3
BATCH_VALUES_4MIB = 4 * MIB // FIELD_BYTES
GIB = 1024 * MIB
PRIVATE_BENCHMARK = "bench_private_opening_local"
PRIVATE_PREFIX = "BENCH_PRIVATE_OPENING"
PRIVATE_KAPPA = 64
PRIVATE_LOGS_GIB = (1, 4, 16, 64, 256)
# The largest log measured, at the threshold whose fragment fits the
# memory of the reference host: (GiB, kappa).
PRIVATE_LARGE_POINTS = ((1024, 256),)
PRIVATE_SWEEP_LOG_GIB = 4
# The client is measured at every threshold on this log, and at the largest
# threshold on the larger logs as well. The client's work does not depend
# on the stored values or on how a reply was formed, so its 3*kappa claims
# hold the columns of a log of unwritten leaves, built without hashing in a
# few walks of the tree on all cores, and reply with the selected row
# encrypted under the client's key.
PRIVATE_CLIENT_LOG_GIB = 4
PRIVATE_CLIENT_LARGE_LOGS_GIB = (8, 16, 32)
# The client at a threshold above the sweep's: Combine over 3 kappa replies
# at the threshold of the largest maintainer job.
PRIVATE_CLIENT_LARGE_KAPPAS = (256,)
# Seconds of the client job's setup per GiB of log, on all cores.
# The public grouping: one segment per block up to 2^25 leaves, doubling
# with every two further levels of the active prefix.
PRIVATE_PATTERN_SHIFT = 5
PRIVATE_PATTERN_SWEEP_GIB = 64
PRIVATE_PATTERN_SWEEP_OFFSETS = (-2, -1, 1)
# Four of the hint builder's batches of 1,024 adjacent columns, so that the
# sample gathers its columns as the complete build does.
PRIVATE_HINT_CHECK_COLUMNS = 4 * 1024
PRIVATE_SINGLE_GROUP_CELLS = 64
PRIVATE_SINGLE_PATTERN_OFFSETS = (-1, 1)
PRIVATE_SINGLE_MEASURED_GIB = (1, 4)
PRIVATE_SINGLE_SHAPE_GIB = (16, 64, 256, 1024)
PRIVATE_TABLE_LOGS_GIB = (1, 4)
PRIVATE_SMOKE_LEAVES = 32768
PRIVATE_SMOKE_APPEND_LEAVES = 1024
TREEPIR_BENCHMARK = "bench_treepir_comparison_local"
TREEPIR_PREFIX = "BENCH_TREEPIR_COMPARISON"
# h = kappa = 32: a log of 2^32 leaves, 128 GiB, which is TreePIR's tree
# without padding and gives both systems 32 servers.
TREEPIR_HEIGHT = 32
# The opened leaves are seeded; the leaves before them are allocated and
# unwritten.
TREEPIR_SEEDED_LEAVES = 1 << 20
# (UndineLog claim, TreePIR class) of each pair of measured servers.
TREEPIR_SERVERS = ((0, 0), (TREEPIR_HEIGHT - 1, TREEPIR_HEIGHT - 1))
# Largest relative difference between the mean decoding time of a real
# server's replies and that of the generated replies for the same queries.
TREEPIR_PAIRED_TOLERANCE = 0.10
TREEPIR_SMOKE_HEIGHT = 8
TREEPIR_SMOKE_SEEDED_LEAVES = 100

TABLE_LABELS = (
    "tab:append",
    "tab:point-open",
    "tab:range-open",
    "tab:maintenance",
    "tab:on-the-fly-open",
    "tab:private-maintainer",
    "tab:private-single",
    "tab:private-table",
    "tab:private-client",
    "tab:treepir-server",
    "tab:treepir-client",
    "tab:treepir-check",
)

TABLE_METADATA: dict[str, dict[str, Any]] = {
    "tab:append": {
        "title": "Append processing and fragment size",
        "notes": (
            "One row per threshold, with Seg = kappa^2. Append cells are one "
            "maintainer's mean processing time for a 4 MiB batch in "
            "milliseconds. Fragment cells are the mean newly materialized "
            "fragment size per batch in MiB."
        ),
    },
    "tab:point-open": {
        "title": "Point openings",
        "notes": (
            "Helper work is microseconds for one helper's response "
            "construction and serialization; response cells are its serialized "
            "bytes. Client cells are Combine+Verify in milliseconds from kappa "
            "correct responses, and from 3*kappa responses of which kappa are "
            "corrupted."
        ),
    },
    "tab:range-open": {
        "title": "Range openings at kappa = 64",
        "notes": (
            "Per-helper response size is KiB. Helper work is one helper's "
            "answer to the range request and its serialization, in "
            "milliseconds. Client work is seconds, from 192 responses; the "
            "corrupt case has 64 corrupted responses among them."
        ),
    },
    "tab:maintenance": {
        "title": "Repair and reconfiguration",
        "notes": (
            "The panel is the layout. All quantities are normalized per 4 MiB "
            "appended to the evaluated 12 MiB prefix. Times are milliseconds; "
            "fragment, download, and state values are MiB. Double and halve "
            "move the threshold between kappa and 2*kappa. The diagonal halve "
            "columns are the UndineLog undo from source state at kappa; the "
            "segment-double column is BorgLog's local segment-and-kappa "
            "double."
        ),
    },
    "tab:on-the-fly-open": {
        "title": "UndineLog openings served above the stored threshold",
        "notes": (
            "One maintainer stores a 12 MiB log at the stored threshold and "
            "answers at the opened threshold. On-the-fly cells are that "
            "answer; stored cells are the answer of a maintainer that stores "
            "the log at the opened threshold. Both include response "
            "serialization and are microseconds; response sizes are KiB. "
            "Stitch-state is the time to stitch the maintainer's whole state "
            "to the opened threshold, in milliseconds."
        ),
    },
}

TABLE_METADATA.update(
    {
        "tab:private-maintainer": {
            "title": "Private openings of one position: one claim of a maintainer",
            "notes": (
                "One claim answers from its primary columns, grouped into "
                "blocks of the public grouping of the log length (one "
                "segment per block at 1 GiB, doubling every two levels) "
                "unless the row gives another block size. A log of 1 GiB "
                "has 2^25 leaves. The record of a pattern is the two blocks "
                "of its subtree and one entry per level above. Database is "
                "the bytes of the primary columns; state adds the row table "
                "and the preprocessing; both are MiB. Request and reply are "
                "serialized KiB. The client's request (key, query, "
                "serialization), the first pass, the complete answer from "
                "serialized request to serialized reply, and the client's "
                "decoding of one reply are milliseconds on one core. "
                "Preprocessing is measured in seconds on one core: each "
                "repetition rebuilds the entire hint, including padding, and "
                "all response-packing tables. The two stages and their sum "
                "are reported separately; preprocessing size is MiB. Tree "
                "construction, fragment encoding and conversion to PIR cells "
                "are outside these timers. Each rebuild serves a checked "
                "opening. The update cells follow a 4 MiB append: "
                "the hint patch in milliseconds and the packing tables in "
                "seconds."
            ),
        },
        "tab:private-single": {
            "title": "Private openings of one position: a single server with the whole tree",
            "notes": (
                "The server stores every cell of the tree in blocks of the "
                "given number of cells and answers with the code path of a "
                "maintainer: the record of a pattern is the two blocks of "
                "its subtree and one cell per level above. Database and "
                "state are GiB, the request is MiB, and the reply is KiB. "
                "Client request and decoding are milliseconds; first pass "
                "and answer are seconds on one core. Preprocessing is "
                "seconds on one core, measured as for a maintainer; its size "
                "is GiB. The rows at 16 "
                "and 64 GiB give the database dimensions and message sizes "
                "of those logs."
            ),
        },
        "tab:private-table": {
            "title": "Records against the table of complete openings at kappa = 64",
            "notes": (
                "The table holds one complete opening per pattern and is "
                "answered by YPIR's server; the records are the maintainer's "
                "database of the same claim. Both give the same reply bytes. "
                "Sizes are MiB; first pass and answer are milliseconds on "
                "one core."
            ),
        },
        "tab:private-client": {
            "title": "Client work for one private opening",
            "notes": (
                "A 12 MiB log at every threshold and the 1 and 4 GiB logs at "
                "threshold 64, at the public grouping. Request and "
                "reply are serialized KiB; the reply is one claim's. Request "
                "is key generation, the query, and serialization in "
                "milliseconds. Private cells decode the replies and run "
                "Combine+Verify; public cells run Combine+Verify on the "
                "public replies of the same claims. The first pair uses "
                "kappa correct replies; the second uses 3*kappa replies of "
                "which kappa lie about one node. All are milliseconds."
            ),
        },
        "tab:treepir-server": {
            "title": "UndineLog against TreePIR: one server's preparation and answer",
            "notes": (
                "h = kappa = 32: a log of 2^32 leaves (128 GiB), 32 UndineLog "
                "maintainers and 32 TreePIR servers over the same YPIR "
                "backend. Each pair of rows is one UndineLog claim and one "
                "TreePIR class, whose PIR servers are rebuilt from the start "
                "in every repetition and alternated, on the same valid Merkle "
                "tree as the client: "
                "2^32 - 2^20 allocated, unwritten leaves followed by 2^20 "
                "seeded ones. Every query opens a seeded leaf. The fragment "
                "and the TreePIR label source must have the same root. Both "
                "systems materialize their full PIR inputs with normal "
                "padding; zero blocks retain the backend's shortcut. The "
                "maintainer's fragment is built once per process, outside "
                "these stages, and every repetition starts from it. The "
                "stages are seconds on one core: UndineLog gathers its primary columns "
                "into PIR cells, then reads the row table and the upper "
                "nodes, then builds the hint (which gathers its columns as it "
                "goes) and the packing tables; TreePIR enumerates its class, "
                "writes its database columns, then builds the hint and the "
                "packing tables. Prepare is the four stages of one rebuild. "
                "PIR is the PIR database (UndineLog's is derived from the "
                "claim's primary columns and needs none of the auxiliary "
                "rows; TreePIR's is the class's dense padded matrix); hint "
                "and packing are the preprocessing; state is the three; all "
                "MiB. The hint's kernel skips an all-zero block of 2,048 rows "
                "of one column: zero-blocks counts those, padding-blocks "
                "those in the padding columns, out of hint-blocks. First "
                "pass and answer are milliseconds on one core; request, "
                "query, keys, and reply are serialized KiB."
            ),
        },
        "tab:treepir-client": {
            "title": "UndineLog against TreePIR: the client's work for one opening",
            "notes": (
                "UndineLog prepares one request and sends it to kappa "
                "maintainers; TreePIR prepares h requests, each with its own "
                "secret key, since the backend reuses a public query matrix "
                "and one secret for different queries would let colluding "
                "servers subtract them. Upload counts every transmitted "
                "request and download every reply of one opening (MiB and "
                "KiB). Index, verify are microseconds; the other cells are "
                "milliseconds on one core. Index is the pattern and the "
                "opening plan, or the h colors and offsets of the path; "
                "keygen, query (with the packing keys), and serialize are "
                "the request's parts; decrypt parses and decrypts every reply "
                "and extracts the shares or the proof nodes; combine+verify "
                "is UndineLog's reconstruction and path check; verify "
                "recomputes the root from the recovered path. The totals are "
                "timed as a whole on a second opening. At h = 32 the log has "
                "2^32 - 2^20 allocated, unwritten leaves followed by 2^20 "
                "seeded ones; every opening is of a seeded leaf. Each reply "
                "is the row its server returns, read from the tree and encrypted "
                "under the client's key, and the check table compares such "
                "replies with a real server's. Reconstruction is fault-free."
            ),
        },
        "tab:treepir-check": {
            "title": "UndineLog against TreePIR: generated and real replies",
            "notes": (
                "At h = kappa = 32, one real UndineLog claim and one real "
                "TreePIR server are built on the opening tree. For each "
                "query, the real reply and the generated reply for the same "
                "server decode to the same row (checked exactly), and their "
                "decoding is timed in turn, in alternating order. "
                "Milliseconds on one core."
            ),
        },
    }
)

OMITTED_EXPERIMENTS = (
    "append batches other than 4 MiB",
    "append generic, codec, preparation-only, and prepared-without-slack paths",
    "point-helper core-only timing",
    "range sizes 2, 16, and 64 MiB",
    "range helper core-only, threshold-only, and split-range timing variants",
    "repair component, correction, replay, and rebuild timings",
    "reconfiguration component and all-helper serial timing variants",
    "segment sizes other than kappa^2",
    "private openings of ranges",
    "private-opening baselines with one query per tree level, replicated "
    "full-copy servers, and BorgLog under the same PIR pipeline",
    "UndineLog against TreePIR at heights other than 32, at other "
    "thresholds than kappa = h, and with faulty replies",
    "private openings over an emulated wide-area network",
    "the single server's preprocessing update after an append",
    "the maintainer service's update that compares every record; the "
    "measured update reads the records an append changes",
)

RENDER_UNIT_CONTRACTS: dict[str, tuple[frozenset[str], ...]] = {
    "bytes-integer": (frozenset({"bytes"}),),
    "gib-1": (frozenset({"bytes"}),),
    "integer": (frozenset({"", "reads", "segments", "ciphertexts"}),),
    "kib-1": (frozenset({"bytes"}),),
    "mib-1": (frozenset({"bytes"}),),
    "mib-3": (frozenset({"bytes"}),),
    "mib-3-div3": (frozenset({"bytes"}),),
    "microseconds-1": (frozenset({"ns"}),),
    "milliseconds-1": (frozenset({"ns"}),),
    "milliseconds-3": (frozenset({"ns"}),),
    "milliseconds-3-div3": (frozenset({"ns"}),),
    # A hint sample scaled to all record columns, plus the packing tables of
    # one reply ciphertext times the ciphertext count.
    "preprocess-core-seconds-1": (
        frozenset({"ns"}),
        frozenset({"columns"}),
        frozenset({"columns"}),
        frozenset({"ns"}),
        frozenset({"ciphertexts"}),
    ),
    "seconds-1": (frozenset({"ns"}),),
    "seconds-3": (frozenset({"ns"}),),
    "text": (frozenset({""}),),
}


class PipelineError(RuntimeError):
    """A reproducibility invariant failed."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = pathlib.Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def atomic_write_text(path: pathlib.Path, payload: str) -> None:
    atomic_write_bytes(path, payload.encode("utf-8"))


def atomic_write_json(path: pathlib.Path, payload: Any) -> None:
    atomic_write_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode())


def run_capture(
    command: Sequence[str],
    *,
    cwd: pathlib.Path = ROOT,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    check: bool = True,
    stdin: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=None if env is None else dict(env),
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode != 0:
        stdout = completed.stdout.decode(errors="replace")[-4000:]
        stderr = completed.stderr.decode(errors="replace")[-4000:]
        raise PipelineError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return completed


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (numerator + denominator - 1) // denominator


def total_nodes_for_height(height: int) -> int:
    return (1 << (height + 1)) - 1


def cursor_preorder(height: int, prefix_leaves: int) -> int:
    if height < 0 or prefix_leaves < 0 or prefix_leaves > (1 << height):
        raise ValueError("invalid preorder cursor arguments")
    if prefix_leaves == 0:
        return 0
    cursor = 0
    remaining = prefix_leaves
    current_height = height
    while remaining:
        cursor += 1
        if current_height == 0:
            break
        left_leaves = 1 << (current_height - 1)
        if remaining <= left_leaves:
            current_height -= 1
        else:
            cursor += total_nodes_for_height(current_height - 1)
            remaining -= left_leaves
            current_height -= 1
    return cursor


def stored_elements_per_segment(layout: str, segment_len: int, kappa: int) -> int:
    """Field elements one claim stores per segment."""
    if layout == "undine":
        if segment_len != kappa * kappa:
            raise ValueError("the undine layout has Seg = kappa^2")
        return 2 * kappa
    if layout == "borg":
        primary = segment_len // kappa
        return primary + kappa * max(1, ceil_div(primary, kappa))
    raise ValueError(f"unknown layout: {layout}")


def sequence_state_growth_bytes(
    *,
    layout: str,
    tree_height: int,
    inserted_values_per_batch: int,
    segment_len: int,
    kappa: int,
    sequence_blocks: int = APPEND_SEQUENCE_BLOCKS,
    warmup_blocks: int = APPEND_SEQUENCE_WARMUP_BLOCKS,
) -> float:
    if sequence_blocks <= warmup_blocks:
        raise ValueError("sequence must contain at least one measured block")
    before = cursor_preorder(tree_height, warmup_blocks * inserted_values_per_batch)
    after = cursor_preorder(tree_height, sequence_blocks * inserted_values_per_batch)
    first_segment = ceil_div(before, segment_len)
    last_segment = ceil_div(after, segment_len)
    mean_new_segments = (last_segment - first_segment) / (
        sequence_blocks - warmup_blocks
    )
    stored_elements = stored_elements_per_segment(layout, segment_len, kappa)
    return mean_new_segments * stored_elements * FIELD_BYTES


@dataclass(frozen=True)
class JobSpec:
    id: str
    layout: str
    family: str
    benchmark: str
    features: tuple[str, ...]
    env: tuple[tuple[str, str], ...]
    result_relpath: str
    trial_env: str
    setup: str
    estimated_setup_seconds: float
    estimated_seconds_per_repetition: float

    def env_dict(self) -> dict[str, str]:
        return dict(self.env)

    def semantic_key(self) -> str:
        value = {
            "layout": self.layout,
            "benchmark": self.benchmark,
            "features": self.features,
            "env": self.env,
            "result_relpath": self.result_relpath,
            "trial_env": self.trial_env,
        }
        return sha256_bytes(canonical_json(value))


@dataclass(frozen=True)
class MeasurementSpec:
    id: str
    job_id: str | None
    selector: tuple[tuple[str, Any], ...] = ()
    path: tuple[str, ...] = ()
    kind: str = "timing"
    sample_mode: str = "direct"
    sample_group: str = "all"
    formula: str | None = None
    formula_args: tuple[Any, ...] = ()
    unit: str = ""
    description: str = ""

    def selector_dict(self) -> dict[str, Any]:
        return dict(self.selector)


@dataclass(frozen=True)
class CellSpec:
    table: str
    row: str
    column: str
    sources: tuple[str, ...]
    render: str
    description: str

    @property
    def id(self) -> str:
        return f"{self.table}|{self.row}|{self.column}"


@dataclass
class Manifest:
    jobs: dict[str, JobSpec] = field(default_factory=dict)
    measurements: dict[str, MeasurementSpec] = field(default_factory=dict)
    cells: dict[str, CellSpec] = field(default_factory=dict)
    table_rows: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    table_columns: dict[str, list[str]] = field(default_factory=dict)

    def add_job(self, job: JobSpec) -> JobSpec:
        if job.id in self.jobs:
            raise PipelineError(f"duplicate job id: {job.id}")
        self.jobs[job.id] = job
        return job

    def add_measurement(self, measurement: MeasurementSpec) -> str:
        previous = self.measurements.get(measurement.id)
        if previous is not None and previous != measurement:
            raise PipelineError(f"conflicting measurement id: {measurement.id}")
        self.measurements[measurement.id] = measurement
        return measurement.id

    def add_cell(self, cell: CellSpec) -> None:
        if cell.id in self.cells:
            raise PipelineError(f"duplicate table cell: {cell.id}")
        self.cells[cell.id] = cell

    def add_row(self, table: str, row: str, cells: Iterable[CellSpec]) -> None:
        if row in self.table_rows[table]:
            raise PipelineError(f"duplicate row {table}|{row}")
        self.table_rows[table].append(row)
        for cell in cells:
            if cell.table != table or cell.row != row:
                raise PipelineError(f"cell placed in wrong row: {cell.id}")
            self.add_cell(cell)

    def consumers(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)
        for cell in self.cells.values():
            for source in cell.sources:
                result[source].append(cell.id)
        return result

    def measurements_for_job(self, job_id: str) -> list[MeasurementSpec]:
        return [
            measurement
            for measurement in self.measurements.values()
            if measurement.job_id == job_id
        ]

    def as_dict(self) -> dict[str, Any]:
        consumers = self.consumers()
        return {
            "pipeline_version": PIPELINE_VERSION,
            "artifact_schema_version": 1,
            "result_groups": list(TABLE_LABELS),
            "jobs": [dataclasses.asdict(self.jobs[key]) for key in sorted(self.jobs)],
            "measurements": [
                {
                    **dataclasses.asdict(self.measurements[key]),
                    "consumers": sorted(consumers.get(key, [])),
                }
                for key in sorted(self.measurements)
            ],
            "cells": [
                dataclasses.asdict(self.cells[key]) | {"id": key}
                for key in sorted(self.cells)
            ],
            "table_rows": dict(self.table_rows),
            "table_columns": self.table_columns,
            "omitted_experiments": list(OMITTED_EXPERIMENTS),
        }


def validate_cell_unit_contract(cell_spec: CellSpec, manifest: Manifest) -> None:
    contract = RENDER_UNIT_CONTRACTS.get(cell_spec.render)
    if contract is None:
        raise PipelineError(
            f"{cell_spec.id}: no unit contract for renderer {cell_spec.render}"
        )
    if len(contract) != len(cell_spec.sources):
        raise PipelineError(
            f"{cell_spec.id}: renderer {cell_spec.render} expects "
            f"{len(contract)} sources, found {len(cell_spec.sources)}"
        )
    for index, (source, allowed) in enumerate(zip(cell_spec.sources, contract)):
        measurement = manifest.measurements.get(source)
        if measurement is None:
            raise PipelineError(f"{cell_spec.id}: unknown source {source}")
        if measurement.unit not in allowed:
            raise PipelineError(
                f"{cell_spec.id}: source {index + 1} has unit "
                f"{measurement.unit!r}, expected one of {sorted(allowed)!r}"
            )


def validate_measurement_value(
    measurement: MeasurementSpec,
    value: Any,
) -> None:
    numeric_units = {"ns", "bytes", "reads", "segments", "columns", "ciphertexts"}
    is_numeric = not isinstance(value, bool) and isinstance(value, (int, float))
    if is_numeric:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise PipelineError(
                f"{measurement.id}: invalid {measurement.unit or 'numeric'} "
                f"value {value!r}"
            )
    elif measurement.unit in numeric_units or measurement.kind == "timing":
        raise PipelineError(
            f"{measurement.id}: expected numeric {measurement.unit} value"
        )
    if measurement.kind == "timing" and is_numeric and float(value) <= 0:
        raise PipelineError(
            f"{measurement.id}: timing sample must be strictly positive"
        )


def selector(**values: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(values.items()))


def job_env(**values: Any) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((key, str(value)) for key, value in values.items()))


def formula_id(formula: str, *args: Any) -> str:
    suffix = sha256_bytes(canonical_json([formula, args]))[:12]
    readable = "-".join(str(value).replace(" ", "_") for value in args[:3])
    return f"formula:{formula}:{readable}:{suffix}"


def ensure_formula(
    manifest: Manifest,
    formula: str,
    *args: Any,
    unit: str = "",
    description: str = "",
) -> str:
    identifier = formula_id(formula, *args)
    return manifest.add_measurement(
        MeasurementSpec(
            id=identifier,
            job_id=None,
            kind="formula",
            formula=formula,
            formula_args=tuple(args),
            unit=unit,
            description=description,
        )
    )


def output_id(job_id: str, tag: str) -> str:
    return f"output:{job_id}:{tag}"


def ensure_output(
    manifest: Manifest,
    job: JobSpec,
    tag: str,
    *,
    entry_selector: tuple[tuple[str, Any], ...],
    path: str,
    kind: str,
    sample_mode: str = "direct",
    sample_group: str = "all",
    unit: str = "",
    description: str = "",
) -> str:
    identifier = output_id(job.id, tag)
    return manifest.add_measurement(
        MeasurementSpec(
            id=identifier,
            job_id=job.id,
            selector=entry_selector,
            path=tuple(path.split(".")),
            kind=kind,
            sample_mode=sample_mode,
            sample_group=sample_group,
            unit=unit,
            description=description,
        )
    )


def cell(
    table: str,
    row: str,
    column: str,
    sources: str | Sequence[str],
    render: str,
    description: str,
) -> CellSpec:
    if isinstance(sources, str):
        sources = (sources,)
    return CellSpec(table, row, column, tuple(sources), render, description)


def add_job(
    manifest: Manifest,
    *,
    identifier: str,
    layout: str,
    family: str,
    benchmark: str,
    features: tuple[str, ...],
    env: tuple[tuple[str, str], ...],
    result_relpath: str,
    trial_env: str,
    setup: str,
    setup_seconds: float,
    repetition_seconds: float,
) -> JobSpec:
    return manifest.add_job(
        JobSpec(
            id=identifier,
            layout=layout,
            family=family,
            benchmark=benchmark,
            features=features,
            env=env,
            result_relpath=result_relpath,
            trial_env=trial_env,
            setup=setup,
            estimated_setup_seconds=setup_seconds,
            estimated_seconds_per_repetition=repetition_seconds,
        )
    )


def build_manifest() -> Manifest:
    manifest = Manifest()
    _build_append_table(manifest)
    _build_point_table(manifest)
    _build_range_table(manifest)
    _build_maintenance_table(manifest)
    _build_on_the_fly_table(manifest)
    _build_private_maintainer_table(manifest)
    _build_private_single_table(manifest)
    _build_private_table_table(manifest)
    _build_private_client_table(manifest)
    _build_treepir_tables(manifest)
    validate_manifest(manifest)
    return manifest


def layout_env(layout: str, kappa: int, *, undine: str, borg: Mapping[str, Any]) -> dict[str, Any]:
    """Operating-point variables of one benchmark for one layout.

    The undine benchmarks take the threshold; the borg benchmarks take the
    segment size and the threshold or `s = Seg / kappa`. `borg` maps each
    variable to a function of `(segment_len, kappa)`.
    """
    segment_len = kappa * kappa
    if layout == "undine":
        return {undine: kappa}
    if layout == "borg":
        return {key: value(segment_len, kappa) for key, value in borg.items()}
    raise ValueError(f"unknown layout: {layout}")


def _dash(manifest: Manifest) -> str:
    return ensure_formula(manifest, "text", "--")


def _append_job(manifest: Manifest, layout: str, kappa: int) -> tuple[str, str]:
    segment_len = kappa * kappa
    prefix = "BENCH_APPEND_ONLY_LOG_BULK_UPDATES_APPEND"
    job = add_job(
        manifest,
        identifier=f"append-4mib-{layout}-k{kappa}",
        layout=layout,
        family=f"{layout}-append",
        benchmark="bench_append_only_log_bulk_updates_append_local",
        features=("bench",),
        env=job_env(
            **{
                f"{prefix}_TREE_HEIGHT": TREE_HEIGHT,
                f"{prefix}_LOCAL_CLAIMS": 1,
                f"{prefix}_SEQUENCE_BLOCKS": APPEND_SEQUENCE_BLOCKS,
                f"{prefix}_SEQUENCE_WARMUP_BLOCKS": APPEND_SEQUENCE_WARMUP_BLOCKS,
                f"{prefix}_SKIP_GENERIC": "true",
                f"{prefix}_METRICS": "prepared_with_slack",
            },
            **layout_env(
                layout,
                kappa,
                undine=f"{prefix}_KAPPA",
                borg={
                    f"{prefix}_SEGMENT_LEN": lambda seg, k: seg,
                    f"{prefix}_S": lambda seg, k: seg // k,
                },
            ),
        ),
        result_relpath="results/bench_append_only_log_bulk_updates_append_local.json",
        trial_env=f"{prefix}_TRIALS",
        setup="Each repetition starts a fresh maintainer and runs one complete 64-batch sequence.",
        setup_seconds=1.0,
        repetition_seconds=APPEND_SEQUENCE_BLOCKS * 0.3,
    )
    timing = ensure_output(
        manifest,
        job,
        "prepared-with-slack-sequence",
        entry_selector=selector(segment_len=segment_len, need=kappa),
        path="prepared_with_slack_next_bulk_updates_append_ns",
        kind="timing",
        sample_mode="append-sequence",
        unit="ns",
        description="Mean append time from one complete 64-batch sequence.",
    )
    fragment = ensure_formula(
        manifest,
        "append-fragment-growth-bytes",
        layout,
        TREE_HEIGHT,
        BATCH_VALUES_4MIB,
        segment_len,
        kappa,
        APPEND_SEQUENCE_BLOCKS,
        APPEND_SEQUENCE_WARMUP_BLOCKS,
        unit="bytes",
        description="Mean newly materialized fragment bytes per appended batch.",
    )
    return timing, fragment


def _build_append_table(manifest: Manifest) -> None:
    table = "tab:append"
    columns = ["segment", "kappa"]
    for layout in LAYOUTS:
        columns += [f"{layout}-append", f"{layout}-fragment"]
    manifest.table_columns[table] = columns
    for kappa in KAPPAS:
        row = f"k{kappa}"
        cells = [
            cell(table, row, "segment", ensure_formula(manifest, "integer", kappa * kappa),
                 "integer", "Segment size."),
            cell(table, row, "kappa", ensure_formula(manifest, "integer", kappa),
                 "integer", "Threshold."),
        ]
        for layout in LAYOUTS:
            timing, fragment = _append_job(manifest, layout, kappa)
            cells += [
                cell(table, row, f"{layout}-append", timing, "milliseconds-1",
                     f"{layout}: maintainer processing time of one 4 MiB batch."),
                cell(table, row, f"{layout}-fragment", fragment, "mib-3",
                     f"{layout}: fragment growth per 4 MiB batch."),
            ]
        manifest.add_row(table, row, cells)


def _point_helper_job(manifest: Manifest, layout: str, kappa: int) -> tuple[str, str]:
    segment_len = kappa * kappa
    prefix = "BENCH_APPEND_ONLY_LOG_POINT_OPEN"
    job = add_job(
        manifest,
        identifier=f"point-helper-{layout}-k{kappa}",
        layout=layout,
        family=f"{layout}-point-helper",
        benchmark="bench_append_only_log_point_open_local",
        features=("bench",),
        env=job_env(
            **{
                f"{prefix}_TREE_HEIGHT": TREE_HEIGHT,
                f"{prefix}_LOCAL_CLAIMS": 1,
                f"{prefix}_COMMITTED_BULK_UPDATE_BATCHES": PREFIX_BATCHES,
                f"{prefix}_METRICS": "serialized",
            },
            **layout_env(
                layout,
                kappa,
                undine=f"{prefix}_KAPPA",
                borg={
                    f"{prefix}_SEG": lambda seg, k: seg,
                    f"{prefix}_KAPPA": lambda seg, k: k,
                },
            ),
        ),
        result_relpath="results/bench_append_only_log_point_open_local.json",
        trial_env=f"{prefix}_TRIALS",
        setup="Warm one claim-bound helper through three 4 MiB appends.",
        setup_seconds=8.0,
        repetition_seconds=0.01,
    )
    match = selector(Seg=segment_len, kappa=kappa)
    helper = ensure_output(
        manifest, job, "serialized-helper-response", entry_selector=match,
        path="helper_open_share_serialized_ns", kind="timing", unit="ns",
        description="Construction and serialization of one helper response.",
    )
    response_bytes = ensure_output(
        manifest, job, "serialized-response-bytes", entry_selector=match,
        path="serialized_response_bytes", kind="value", unit="bytes",
        description="Serialized bytes in one helper response.",
    )
    return helper, response_bytes


def _point_client_job(manifest: Manifest, layout: str, kappa: int) -> tuple[str, str]:
    segment_len = kappa * kappa
    prefix = "BENCH_APPEND_ONLY_LOG_CLIENT_POINT_OPEN"
    job = add_job(
        manifest,
        identifier=f"point-client-{layout}-k{kappa}",
        layout=layout,
        family=f"{layout}-point-client",
        benchmark="bench_append_only_log_client_point_open_local",
        features=("bench",),
        env=job_env(
            **{
                f"{prefix}_TREE_HEIGHT": TREE_HEIGHT,
                f"{prefix}_COMMITTED_BULK_UPDATE_BATCHES": PREFIX_BATCHES,
            },
            **layout_env(
                layout,
                kappa,
                undine=f"{prefix}_KAPPA",
                borg={
                    f"{prefix}_SEG": lambda seg, k: seg,
                    f"{prefix}_KAPPA": lambda seg, k: k,
                },
            ),
        ),
        result_relpath="results/bench_append_only_log_client_point_open_local.json",
        trial_env=f"{prefix}_TRIALS",
        setup="Warm and split a three-batch point-opening fixture.",
        setup_seconds=12.0,
        repetition_seconds=max(0.01, (kappa / 8) ** 3 * 0.002),
    )
    match = selector(Seg=segment_len, kappa=kappa)
    clean = ensure_output(
        manifest, job, "clean-combine-verify", entry_selector=match,
        path="client_zero_bad_helpers_combine_verify_ns", kind="timing", unit="ns",
        description="Combine+Verify from kappa correct responses.",
    )
    corrupt = ensure_output(
        manifest, job, "corrupt-combine-verify", entry_selector=match,
        path="client_kappa_bad_helpers_combine_verify_ns", kind="timing", unit="ns",
        description="Combine+Verify from 3*kappa responses with kappa corrupted.",
    )
    return clean, corrupt


def _build_point_table(manifest: Manifest) -> None:
    table = "tab:point-open"
    columns = ["segment", "kappa"]
    for layout in LAYOUTS:
        columns += [
            f"{layout}-helper",
            f"{layout}-response",
            f"{layout}-client",
            f"{layout}-client-correcting",
        ]
    manifest.table_columns[table] = columns
    for kappa in KAPPAS:
        row = f"k{kappa}"
        cells = [
            cell(table, row, "segment", ensure_formula(manifest, "integer", kappa * kappa),
                 "integer", "Segment size."),
            cell(table, row, "kappa", ensure_formula(manifest, "integer", kappa),
                 "integer", "Threshold."),
        ]
        for layout in LAYOUTS:
            helper, response_bytes = _point_helper_job(manifest, layout, kappa)
            clean, corrupt = _point_client_job(manifest, layout, kappa)
            cells += [
                cell(table, row, f"{layout}-helper", helper, "microseconds-1",
                     f"{layout}: one helper's response construction and serialization."),
                cell(table, row, f"{layout}-response", response_bytes, "bytes-integer",
                     f"{layout}: serialized bytes of one helper response."),
                cell(table, row, f"{layout}-client", clean, "milliseconds-3",
                     f"{layout}: fault-free client Combine+Verify."),
                cell(table, row, f"{layout}-client-correcting", corrupt, "milliseconds-3",
                     f"{layout}: error-correcting client Combine+Verify."),
            ]
        manifest.add_row(table, row, cells)


def _range_job(manifest: Manifest, layout: str, role: str) -> JobSpec:
    kappa = RANGE_KAPPA
    prefix = "BENCH_APPEND_ONLY_LOG_RANGE_OPEN"
    metrics = {
        "helper": "serialized",
        "client": "correcting_clean,correcting_corrupt",
    }[role]
    return add_job(
        manifest,
        identifier=f"range-{role}-{layout}-k{kappa}",
        layout=layout,
        family=f"{layout}-range-{role}",
        benchmark="bench_append_only_log_range_open_local",
        features=("bench", "log-bulk-updates-8mb"),
        env=job_env(
            **{
                f"{prefix}_TREE_HEIGHT": TREE_HEIGHT,
                f"{prefix}_COMMITTED_BULK_UPDATE_BATCHES": 16,
                f"{prefix}_QUERY_BULK_UPDATES": 0,
                f"{prefix}_BYTE_OFFSET": 0,
                f"{prefix}_BYTE_LEN_LIST": ",".join(
                    str(value * MIB) for value in RANGE_SIZES_MIB
                ),
                f"{prefix}_SPLIT_COUNT": 1,
                f"{prefix}_METRICS": metrics,
            },
            **layout_env(
                layout,
                kappa,
                undine=f"{prefix}_KAPPA",
                borg={
                    f"{prefix}_SEG": lambda seg, k: seg,
                    f"{prefix}_KAPPA": lambda seg, k: k,
                },
            ),
        ),
        result_relpath="results/bench_append_only_log_range_open_local.json",
        trial_env=f"{prefix}_TRIALS",
        setup="Warm sixteen 8 MiB batches once and reuse the captured range state.",
        setup_seconds=240.0,
        repetition_seconds=0.02 if role == "helper" else 110.0,
    )


def _build_range_table(manifest: Manifest) -> None:
    table = "tab:range-open"
    kappa = RANGE_KAPPA
    columns = ["range-mib", "full-segment-reads", "point-reads"]
    for layout in LAYOUTS:
        columns += [
            f"{layout}-per-helper-kib",
            f"{layout}-helper-work",
            f"{layout}-clean-client",
            f"{layout}-corrupt-client",
        ]
    manifest.table_columns[table] = columns
    jobs = {
        (layout, role): _range_job(manifest, layout, role)
        for layout in LAYOUTS
        for role in ("helper", "client")
    }
    for range_mib in RANGE_SIZES_MIB:
        row = f"range-{range_mib}mib"
        match = selector(Seg=kappa * kappa, kappa=kappa, range_bytes=range_mib * MIB)
        counts_job = jobs[("undine", "helper")]
        cells = [
            cell(table, row, "range-mib", ensure_formula(manifest, "integer", range_mib),
                 "integer", "Returned range size in MiB."),
            cell(
                table, row, "full-segment-reads",
                ensure_output(
                    manifest, counts_job, f"{range_mib}mib-full-segments",
                    entry_selector=match, path="request_segments", kind="value",
                    unit="reads", description="Full coded-segment reads in the range request.",
                ),
                "integer", "Full segments.",
            ),
            cell(
                table, row, "point-reads",
                ensure_output(
                    manifest, counts_job, f"{range_mib}mib-point-reads",
                    entry_selector=match, path="request_points", kind="value",
                    unit="reads", description="Single-cell coded-array reads in the range request.",
                ),
                "integer", "Boundary/authentication points.",
            ),
        ]
        for layout in LAYOUTS:
            helper_job = jobs[(layout, "helper")]
            client_job = jobs[(layout, "client")]
            response_bytes = ensure_output(
                manifest, helper_job, f"{range_mib}mib-helper-bytes", entry_selector=match,
                path="helper_response_serialized_bytes", kind="value", unit="bytes",
                description="Serialized response bytes from one helper.",
            )
            helper = ensure_output(
                manifest, helper_job, f"{range_mib}mib-helper-time", entry_selector=match,
                path="helper_range_share_serialized_ns", kind="timing", unit="ns",
                description="One helper's answer to the range request and its serialization.",
            )
            clean = ensure_output(
                manifest, client_job, f"{range_mib}mib-clean-client", entry_selector=match,
                path="client_correction_budget_combine_verify_ns", kind="timing", unit="ns",
                description="Client Combine+Verify from clean oversampled responses.",
            )
            corrupt = ensure_output(
                manifest, client_job, f"{range_mib}mib-corrupt-client", entry_selector=match,
                path="client_kappa_failures_filtered_combine_verify_ns", kind="timing",
                unit="ns", description="Client Combine+Verify with 64 corrupted helpers.",
            )
            cells += [
                cell(table, row, f"{layout}-per-helper-kib", response_bytes, "kib-1",
                     f"{layout}: serialized bytes from one helper."),
                cell(table, row, f"{layout}-helper-work", helper, "milliseconds-3",
                     f"{layout}: one helper's work."),
                cell(table, row, f"{layout}-clean-client", clean, "seconds-3",
                     f"{layout}: client work with no corrupted helpers."),
                cell(table, row, f"{layout}-corrupt-client", corrupt, "seconds-3",
                     f"{layout}: client work with 64 corrupted helpers."),
            ]
        manifest.add_row(table, row, cells)


def _repair_job(manifest: Manifest, layout: str, kappa: int) -> dict[str, str]:
    segment_len = kappa * kappa
    prefix = "BENCH_APPEND_ONLY_LOG_REPAIR"
    job = add_job(
        manifest,
        identifier=f"repair-{layout}-k{kappa}",
        layout=layout,
        family=f"{layout}-repair",
        benchmark="bench_append_only_log_repair_kernels_local",
        features=("bench",),
        env=job_env(
            **{
                f"{prefix}_TREE_HEIGHT": TREE_HEIGHT,
                f"{prefix}_COMMITTED_BULK_UPDATE_BATCHES": PREFIX_BATCHES,
                f"{prefix}_HELPER_SURPLUS": 1,
                f"{prefix}_MAX_ERRORS": 0,
                f"{prefix}_METRICS": "helper_rpc,recover_install",
            },
            **layout_env(
                layout,
                kappa,
                undine=f"{prefix}_KAPPA",
                borg={
                    f"{prefix}_SEGMENT_LEN": lambda seg, k: seg,
                    f"{prefix}_S": lambda seg, k: seg // k,
                },
            ),
        ),
        result_relpath="results/bench_append_only_log_repair_kernels_local.json",
        trial_env=f"{prefix}_TRIALS",
        setup="Warm a 12 MiB prefix and prepare exact repair replies.",
        setup_seconds=20.0,
        repetition_seconds=3.0,
    )
    match = selector(segment_len=segment_len, need=kappa)
    return {
        "receiver": ensure_output(
            manifest, job, "receiver-recover-install", entry_selector=match,
            path="recoverer_recover_install_kernel_ns", kind="timing", unit="ns",
            description="Recovering maintainer reconstruction and installation.",
        ),
        "helper": ensure_output(
            manifest, job, "helper-rpc", entry_selector=match,
            path="helper_assist_net_adapter_ns", kind="timing", unit="ns",
            description="One helper's endpoint work and response serialization.",
        ),
        "download": ensure_output(
            manifest, job, "modeled-download", entry_selector=match,
            path="recover_helper_reply_bytes_modeled", kind="value", unit="bytes",
            description="Modeled raw helper data for repair.",
        ),
    }


RECONFIG_FIELDS: dict[str, dict[str, tuple[str, str, str]]] = {
    "undine": {
        "up_receiver": ("local_kappa_double_total_ns", "timing", "ns"),
        "up_download": ("kappa_double_helper_eval_raw_bytes_total", "value", "bytes"),
        "state_after": ("receiver_target_state_bytes", "value", "bytes"),
        "down_receiver": ("kappa_halve_prepared_total_ns", "timing", "ns"),
        "down_helper": ("kappa_halve_helper_eval_one_helper_serialized_ns", "timing", "ns"),
        "down_download": ("kappa_halve_helper_eval_raw_bytes_total", "value", "bytes"),
        "diagonal_receiver": ("kappa_halve_diagonal_unstitch_install_ns", "timing", "ns"),
        "diagonal_download": ("kappa_halve_diagonal_source_raw_bytes_total", "value", "bytes"),
    },
    "borg": {
        "up_receiver": ("local_kappa_double_total_ns", "timing", "ns"),
        "up_helper": ("helper_eval_one_helper_serialized_ns", "timing", "ns"),
        "up_download": ("helper_eval_raw_bytes_total", "value", "bytes"),
        "down_receiver": ("kappa_halve_prepared_total_ns", "timing", "ns"),
        "down_helper": ("kappa_halve_helper_eval_one_helper_serialized_ns", "timing", "ns"),
        "down_download": ("kappa_halve_helper_eval_raw_bytes_total", "value", "bytes"),
        "segment_receiver": ("local_segment_double_total_ns", "timing", "ns"),
    },
}

RECONFIG_METRICS = {
    "undine": "receiver_double,helper_halve,receiver_halve,receiver_halve_diagonal",
    "borg": "receiver_double,segment_double,receiver_halve,helper_double,helper_halve",
}


def _reconfig_job(manifest: Manifest, layout: str, kappa: int) -> dict[str, str]:
    segment_len = kappa * kappa
    prefix = "BENCH_APPEND_ONLY_LOG_KAPPA_RESCALE"
    job = add_job(
        manifest,
        identifier=f"reconfig-{layout}-k{kappa}",
        layout=layout,
        family=f"{layout}-reconfiguration",
        benchmark="bench_append_only_log_one_server_kappa_rescale_local",
        features=("bench",),
        env=job_env(
            **{
                "BENCH_APPEND_ONLY_LOG_REPAIR_TREE_HEIGHT": TREE_HEIGHT,
                f"{prefix}_COMMITTED_BULK_UPDATE_BATCHES": PREFIX_BATCHES,
                f"{prefix}_METRICS": RECONFIG_METRICS[layout],
            },
            **layout_env(
                layout,
                kappa,
                undine=f"{prefix}_KAPPA",
                borg={
                    f"{prefix}_SEGMENT_LEN": lambda seg, k: seg,
                    f"{prefix}_SOURCE_S": lambda seg, k: seg // k,
                },
            ),
        ),
        result_relpath="results/bench_append_only_log_one_server_kappa_rescale_local.json",
        trial_env=f"{prefix}_TRIALS",
        setup="Warm the 12 MiB source responsibility and prepare the reconfiguration inputs.",
        setup_seconds=25.0,
        repetition_seconds=5.0,
    )
    match = selector(segment_len=segment_len, source_need=kappa)
    return {
        name: ensure_output(
            manifest, job, name, entry_selector=match, path=path, kind=kind, unit=unit,
            description=f"Reconfiguration output {path}.",
        )
        for name, (path, kind, unit) in RECONFIG_FIELDS[layout].items()
    }


def _build_maintenance_table(manifest: Manifest) -> None:
    table = "tab:maintenance"
    columns = [
        "segment",
        "kappa",
        "fragment",
        "repair-receiver",
        "repair-helper",
        "repair-download",
        "double-receiver",
        "double-helper",
        "double-download",
        "state-after-double",
        "halve-receiver",
        "halve-helper",
        "halve-download",
        "halve-diagonal-receiver",
        "halve-diagonal-download",
        "segment-double-receiver",
    ]
    manifest.table_columns[table] = columns
    prefix_cursor = cursor_preorder(TREE_HEIGHT, PREFIX_BATCHES * BATCH_VALUES_4MIB)
    dash = _dash(manifest)
    for layout in LAYOUTS:
        for kappa in KAPPAS:
            segment_len = kappa * kappa
            row = f"{layout}:k{kappa}"
            repair = _repair_job(manifest, layout, kappa)
            reconfig = _reconfig_job(manifest, layout, kappa)

            def timed(column: str, source: str | None, description: str) -> CellSpec:
                if source is None:
                    return cell(table, row, column, dash, "text", description)
                return cell(table, row, column, source, "milliseconds-3-div3", description)

            def sized(column: str, source: str | None, description: str) -> CellSpec:
                if source is None:
                    return cell(table, row, column, dash, "text", description)
                return cell(table, row, column, source, "mib-3-div3", description)

            if layout == "undine":
                state_after = reconfig["state_after"]
            else:
                # BorgLog doubles kappa at a fixed segment size.
                state_after = ensure_formula(
                    manifest,
                    "maintainer-state-bytes",
                    "borg",
                    prefix_cursor,
                    segment_len,
                    2 * kappa,
                    unit="bytes",
                    description="Modeled fragment bytes after doubling kappa.",
                )
            manifest.add_row(
                table,
                row,
                [
                    cell(table, row, "segment", ensure_formula(manifest, "integer", segment_len),
                         "integer", "Segment size."),
                    cell(table, row, "kappa", ensure_formula(manifest, "integer", kappa),
                         "integer", "Threshold."),
                    sized("fragment", repair["download"],
                          "Maintainer fragment bytes per 4 MiB."),
                    timed("repair-receiver", repair["receiver"],
                          "Repair receiver work per 4 MiB."),
                    timed("repair-helper", repair["helper"],
                          "One repair helper's work per 4 MiB."),
                    sized("repair-download", repair["download"],
                          "Repair download per 4 MiB."),
                    timed("double-receiver", reconfig["up_receiver"],
                          "Kappa-double receiver work per 4 MiB."),
                    timed("double-helper", reconfig.get("up_helper"),
                          "Kappa-double helper work per 4 MiB."),
                    sized("double-download", reconfig["up_download"],
                          "Kappa-double download per 4 MiB."),
                    sized("state-after-double", state_after,
                          "Fragment bytes per 4 MiB after doubling kappa."),
                    timed("halve-receiver", reconfig["down_receiver"],
                          "Kappa-halve receiver work per 4 MiB, from helper replies."),
                    timed("halve-helper", reconfig["down_helper"],
                          "Kappa-halve helper work per 4 MiB."),
                    sized("halve-download", reconfig["down_download"],
                          "Kappa-halve download per 4 MiB."),
                    timed("halve-diagonal-receiver", reconfig.get("diagonal_receiver"),
                          "Kappa-halve receiver work per 4 MiB, from diagonal source state."),
                    sized("halve-diagonal-download", reconfig.get("diagonal_download"),
                          "Diagonal source state per 4 MiB."),
                    timed("segment-double-receiver", reconfig.get("segment_receiver"),
                          "Local segment-and-kappa double receiver work per 4 MiB."),
                ],
            )


def _on_the_fly_job(manifest: Manifest, kappa: int) -> JobSpec:
    prefix = "BENCH_APPEND_ONLY_LOG_ON_THE_FLY_OPEN"
    return add_job(
        manifest,
        identifier=f"on-the-fly-undine-k{kappa}",
        layout="undine",
        family="undine-on-the-fly-open",
        benchmark="bench_append_only_log_on_the_fly_open_local",
        features=("bench",),
        env=job_env(
            **{
                f"{prefix}_KAPPA": kappa,
                f"{prefix}_MAX_LEVELS": ON_THE_FLY_LEVELS,
                f"{prefix}_TREE_HEIGHT": TREE_HEIGHT,
                f"{prefix}_LOCAL_CLAIMS": 1,
                f"{prefix}_COMMITTED_BULK_UPDATE_BATCHES": PREFIX_BATCHES,
                f"{prefix}_RANGE_BYTE_LEN_LIST": ",".join(
                    str(value * MIB) for value in ON_THE_FLY_RANGES_MIB
                ),
            }
        ),
        result_relpath="results/bench_append_only_log_on_the_fly_open_local.json",
        trial_env=f"{prefix}_TRIALS",
        setup="Warm one claim-bound maintainer through three 4 MiB appends and stitch copies of it.",
        setup_seconds=10.0,
        repetition_seconds=0.5,
    )


def _build_on_the_fly_table(manifest: Manifest) -> None:
    table = "tab:on-the-fly-open"
    columns = ["stored-kappa", "open-kappa", "point-on-the-fly", "point-stored", "point-bytes"]
    for range_mib in ON_THE_FLY_RANGES_MIB:
        columns += [
            f"range-{range_mib}mib-on-the-fly",
            f"range-{range_mib}mib-stored",
            f"range-{range_mib}mib-kib",
        ]
    columns.append("stitch-state")
    manifest.table_columns[table] = columns
    dash = _dash(manifest)
    for kappa in KAPPAS:
        job = _on_the_fly_job(manifest, kappa)
        for levels in range(ON_THE_FLY_LEVELS + 1):
            open_kappa = kappa << levels
            row = f"k{kappa}-open{open_kappa}"

            def answer(query: str, byte_len: int, tag: str) -> tuple[str | None, str, str]:
                match = selector(query=query, open_kappa=open_kappa, query_byte_len=byte_len)
                on_the_fly = None
                if levels > 0:
                    on_the_fly = ensure_output(
                        manifest, job, f"open{open_kappa}-{tag}-on-the-fly",
                        entry_selector=match, path="helper_on_the_fly_serialized_ns",
                        kind="timing", unit="ns",
                        description="Answer of the maintainer stored at the stored threshold.",
                    )
                stored = ensure_output(
                    manifest, job, f"open{open_kappa}-{tag}-stored", entry_selector=match,
                    path="helper_stored_at_open_kappa_serialized_ns", kind="timing", unit="ns",
                    description="Answer of a maintainer stored at the opened threshold.",
                )
                size = ensure_output(
                    manifest, job, f"open{open_kappa}-{tag}-bytes", entry_selector=match,
                    path="serialized_response_bytes", kind="value", unit="bytes",
                    description="Serialized bytes of the answer.",
                )
                return on_the_fly, stored, size

            def timed(column: str, source: str | None, description: str) -> CellSpec:
                if source is None:
                    return cell(table, row, column, dash, "text", description)
                return cell(table, row, column, source, "microseconds-1", description)

            point_fly, point_stored, point_bytes = answer("point", 0, "point")
            cells = [
                cell(table, row, "stored-kappa", ensure_formula(manifest, "integer", kappa),
                     "integer", "Stored threshold."),
                cell(table, row, "open-kappa", ensure_formula(manifest, "integer", open_kappa),
                     "integer", "Opened threshold."),
                timed("point-on-the-fly", point_fly, "Point opening answered on the fly."),
                timed("point-stored", point_stored,
                      "Point opening answered from state stored at the opened threshold."),
                cell(table, row, "point-bytes", point_bytes, "bytes-integer",
                     "Serialized bytes of the point answer."),
            ]
            for range_mib in ON_THE_FLY_RANGES_MIB:
                fly, stored, size = answer("range", range_mib * MIB, f"range-{range_mib}mib")
                cells += [
                    timed(f"range-{range_mib}mib-on-the-fly", fly,
                          f"{range_mib} MiB range opening answered on the fly."),
                    timed(f"range-{range_mib}mib-stored", stored,
                          f"{range_mib} MiB range opening answered from state stored at "
                          "the opened threshold."),
                    cell(table, row, f"range-{range_mib}mib-kib", size, "kib-1",
                         f"Serialized KiB of the {range_mib} MiB range answer."),
                ]
            if levels > 0:
                stitch = ensure_output(
                    manifest, job, f"open{open_kappa}-stitch-state",
                    entry_selector=selector(query="stitch", open_kappa=open_kappa),
                    path="stitch_whole_state_ns", kind="timing", unit="ns",
                    description="Kappa-doubles of the maintainer's whole state.",
                )
                cells.append(cell(table, row, "stitch-state", stitch, "milliseconds-3",
                                  "Stitching the whole state to the opened threshold."))
            else:
                cells.append(cell(table, row, "stitch-state", dash, "text",
                                  "The maintainer already stores this threshold."))
            manifest.add_row(table, row, cells)


def leaves_of_gib(gib: int) -> int:
    return gib * GIB // FIELD_BYTES


def private_pattern_bits(leaves: int, kappa: int) -> int:
    """Leaves per pattern of the public grouping, as a power of two: the
    balance of the client's upload against its download,
    floor((h + log2 kappa - 5) / 2) for the height h of the active prefix,
    and at least the pattern whose subtree fills one segment,
    2 log2 kappa - 1. A server that stores the whole tree is kappa = 1."""
    height = 0 if leaves <= 1 else (leaves - 1).bit_length()
    log_kappa = kappa.bit_length() - 1
    balance = (height + log_kappa - PRIVATE_PATTERN_SHIFT) // 2
    return max(balance, 2 * log_kappa - 1, 0)


def private_reply_ciphertexts(pattern_bits: int, group_cells: int, entries_per_group: int) -> int:
    """Reply ciphertexts of a record: the groups a subtree of
    2^(pattern_bits + 1) - 1 cells meets, and the upper entries."""
    span = ceil_div((1 << (pattern_bits + 1)) - 1, group_cells)
    symbols = ceil_div((span + 1) * entries_per_group * 256, 14) + 19 * 24
    return ceil_div(symbols, 2048)


def _private_job(
    manifest: Manifest,
    *,
    identifier: str,
    role: str,
    env: Mapping[str, Any],
    setup: str,
    setup_seconds: float,
    repetition_seconds: float,
) -> JobSpec:
    existing = manifest.jobs.get(identifier)
    if existing is not None:
        return existing
    return add_job(
        manifest,
        identifier=identifier,
        layout="undine",
        family=f"undine-private-{role}",
        benchmark=PRIVATE_BENCHMARK,
        features=("bench",),
        env=job_env(
            **{
                f"{PRIVATE_PREFIX}_ROLE": role,
                f"{PRIVATE_PREFIX}_TREE_HEIGHT": TREE_HEIGHT,
                f"{PRIVATE_PREFIX}_HINT_CHECK_COLUMNS": PRIVATE_HINT_CHECK_COLUMNS,
                **{f"{PRIVATE_PREFIX}_{key}": value for key, value in env.items()},
            }
        ),
        result_relpath=f"results/{PRIVATE_BENCHMARK}.json",
        trial_env=f"{PRIVATE_PREFIX}_TRIALS",
        setup=setup,
        setup_seconds=setup_seconds,
        repetition_seconds=repetition_seconds,
    )


PRIVATE_SIZE_FIELDS = {
    "record_bytes": "Bytes of one record.",
    "database_bytes": "Bytes of the records the server answers from.",
    "request_bytes": "Serialized bytes of the client's request.",
    "reply_bytes": "Serialized bytes of the reply of one claim.",
}
PRIVATE_SERVER_TIMINGS = {
    "client_request_ns": "Key generation, the query, and the serialized request.",
    "server_first_pass_ns": "The query's inner product with every database column, after one untimed scan.",
    "server_answer_ns": "From the serialized request to the serialized reply.",
    "client_decode_ns": "From the serialized reply to the entries of the opening.",
    "hint_ns": "A complete fresh hint, including padded columns, on one core.",
    "packing_ns": "Fresh packing tables of every reply ciphertext, on one core.",
    "preprocess_ns": "The sum of the full hint and packing stages of each rebuild.",
}


def _private_outputs(
    manifest: Manifest,
    job: JobSpec,
    match: tuple[tuple[str, Any], ...],
    *,
    tag: str = "",
    sizes: Mapping[str, str] = PRIVATE_SIZE_FIELDS,
    timings: Mapping[str, str] = {},
    counts: Mapping[str, tuple[str, str]] = {},
    sample_mode: str = "direct",
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for name, description in sizes.items():
        outputs[name] = ensure_output(
            manifest, job, f"{tag}{name}", entry_selector=match, path=name,
            kind="value", unit="bytes", description=description,
        )
    for name, (unit, description) in counts.items():
        outputs[name] = ensure_output(
            manifest, job, f"{tag}{name}", entry_selector=match, path=name,
            kind="value", unit=unit, description=description,
        )
    for name, description in timings.items():
        role = dict(match).get("role")
        group = "all"
        if role in {"maintainer", "single"}:
            group = ("preprocess" if name in {"hint_ns", "packing_ns", "preprocess_ns"}
                     else "update" if name.startswith("update_") else "online")
        outputs[name] = ensure_output(
            manifest, job, f"{tag}{name}", entry_selector=match, path=name,
            kind="timing", unit="ns", sample_mode="rebuild" if group == "preprocess" else sample_mode,
            sample_group=group, description=description,
        )
    return outputs


def _private_server_outputs(
    manifest: Manifest, job: JobSpec, role: str, *, updates: bool
) -> dict[str, str]:
    timings = dict(PRIVATE_SERVER_TIMINGS)
    if updates:
        timings |= {
            "update_hint_ns": "The changed records and the patched hint after an append.",
            "update_packing_ns": "The packing tables of the patched hint.",
        }
    return _private_outputs(
        manifest,
        job,
        selector(role=role),
        sizes=PRIVATE_SIZE_FIELDS
        | {
            "serving_state_bytes": "Bytes of the records, the row table, and the preprocessing.",
            "preprocessed_bytes": "Bytes of the complete hint and packing tables.",
        },
        timings=timings,
        sample_mode="direct",
    )


def _private_preprocess_cell(
    table: str, row: str, outputs: Mapping[str, str]
) -> CellSpec:
    return cell(
        table,
        row,
        "preprocess-core-s",
        outputs["preprocess_ns"],
        "seconds-1",
        "Measured full hint and packing-table rebuild on one core.",
    )


def _private_maintainer(
    manifest: Manifest, kappa: int, gib: int, pattern_bits: int
) -> tuple[JobSpec, dict[str, str]]:
    leaves = leaves_of_gib(gib)
    database_gib = 2 * gib / kappa
    ciphertexts = private_reply_ciphertexts(pattern_bits, kappa * kappa, kappa)
    job = _private_job(
        manifest,
        identifier=f"private-maintainer-undine-log{gib}gib-p{pattern_bits}-k{kappa}",
        role="maintainer",
        env={
            "KAPPA": kappa,
            "LEAVES": leaves,
            "SEGMENTS_PER_RECORD": 1,
            "PATTERN_BITS": pattern_bits,
            "APPEND_LEAVES": BATCH_VALUES_4MIB,
        },
        setup=(
            "Build one claim's primary columns for the seeded log on all "
            "cores and convert them to PIR cells. Each repetition rebuilds "
            "all PIR preprocessing on one core; online and update samples stop independently."
        ),
        setup_seconds=30.0 + 8.0 * gib,
        repetition_seconds=4.0 + 3.5 * ciphertexts + 25.0 * database_gib,
    )
    return job, _private_server_outputs(manifest, job, "maintainer", updates=True)


def _private_server_cells(
    table: str,
    row: str,
    outputs: Mapping[str, str],
    *,
    size_render: str,
    request_render: str,
    answer_render: str,
) -> list[CellSpec]:
    return [
        cell(table, row, "record-kib", outputs["record_bytes"], "kib-1",
             "KiB of one record."),
        cell(table, row, "database", outputs["database_bytes"], size_render,
             "Records the server answers from."),
        cell(table, row, "state", outputs["serving_state_bytes"], size_render,
             "Records, row table, and preprocessing."),
        cell(table, row, "preprocess-size", outputs["preprocessed_bytes"], size_render,
             "Complete hint and packing tables."),
        cell(table, row, "request", outputs["request_bytes"], request_render,
             "Serialized request."),
        cell(table, row, "reply-kib", outputs["reply_bytes"], "kib-1",
             "Serialized reply of one claim."),
        cell(table, row, "client-request-ms", outputs["client_request_ns"],
             "milliseconds-1", "Key generation, the query, and serialization."),
        cell(table, row, "first-pass", outputs["server_first_pass_ns"], answer_render,
             "The query's inner product with every database column."),
        cell(table, row, "answer", outputs["server_answer_ns"], answer_render,
             "From the serialized request to the serialized reply."),
        cell(table, row, "client-decode-ms", outputs["client_decode_ns"],
             "milliseconds-1", "Decoding the reply of one claim."),
        _private_preprocess_cell(table, row, outputs),
        cell(table, row, "preprocess-hint-s", outputs["hint_ns"], "seconds-1",
             "Complete hint rebuild on one core."),
        cell(table, row, "preprocess-packing-s", outputs["packing_ns"], "seconds-1",
             "Rebuild all response-packing tables on one core."),
    ]


def _build_private_maintainer_table(manifest: Manifest) -> None:
    table = "tab:private-maintainer"
    manifest.table_columns[table] = [
        "kappa", "log-gib", "record-kib", "database", "state", "preprocess-size", "request",
        "reply-kib", "client-request-ms", "first-pass", "answer",
        "client-decode-ms", "preprocess-core-s", "preprocess-hint-s",
        "preprocess-packing-s", "update-hint-ms",
        "update-packing-s",
    ]
    points = [
        ("log", PRIVATE_KAPPA, gib, private_pattern_bits(leaves_of_gib(gib), PRIVATE_KAPPA))
        for gib in PRIVATE_LOGS_GIB
    ]
    points += [
        (
            "kappa",
            kappa,
            PRIVATE_SWEEP_LOG_GIB,
            private_pattern_bits(leaves_of_gib(PRIVATE_SWEEP_LOG_GIB), kappa),
        )
        for kappa in KAPPAS
        if kappa != PRIVATE_KAPPA
    ]
    sweep_bits = private_pattern_bits(leaves_of_gib(PRIVATE_PATTERN_SWEEP_GIB), PRIVATE_KAPPA)
    points += [
        ("pattern", PRIVATE_KAPPA, PRIVATE_PATTERN_SWEEP_GIB, sweep_bits + offset)
        for offset in PRIVATE_PATTERN_SWEEP_OFFSETS
    ]
    points += [
        ("large", kappa, gib, private_pattern_bits(leaves_of_gib(gib), kappa))
        for gib, kappa in PRIVATE_LARGE_POINTS
    ]
    for panel, kappa, gib, pattern_bits in points:
        _, outputs = _private_maintainer(manifest, kappa, gib, pattern_bits)
        row = f"{panel}:k{kappa}-log{gib}gib-p{pattern_bits}"
        cells = [
            cell(table, row, "kappa", ensure_formula(manifest, "integer", kappa),
                 "integer", "Threshold."),
            cell(table, row, "log-gib", ensure_formula(manifest, "integer", gib),
                 "integer", "GiB of the log."),
            *_private_server_cells(
                table, row, outputs,
                size_render="mib-1", request_render="kib-1",
                answer_render="milliseconds-1",
            ),
            cell(table, row, "update-hint-ms", outputs["update_hint_ns"],
                 "milliseconds-1", "The hint patch after a 4 MiB append."),
            cell(table, row, "update-packing-s", outputs["update_packing_ns"],
                 "seconds-1", "The packing tables after a 4 MiB append."),
        ]
        manifest.add_row(table, row, cells)


def _private_single(
    manifest: Manifest, gib: int, pattern_bits: int
) -> tuple[JobSpec, dict[str, str]]:
    env: dict[str, Any] = {
        "LEAVES": leaves_of_gib(gib),
        "CELLS_PER_RECORD": PRIVATE_SINGLE_GROUP_CELLS,
        "PATTERN_BITS": pattern_bits,
    }
    if gib == PRIVATE_SINGLE_MEASURED_GIB[0] and pattern_bits == private_pattern_bits(
        leaves_of_gib(gib), 1
    ):
        env["SHAPE_LEAVES_LIST"] = ",".join(
            str(leaves_of_gib(shape)) for shape in PRIVATE_SINGLE_SHAPE_GIB
        )
    job = _private_job(
        manifest,
        identifier=f"private-single-undine-log{gib}gib-p{pattern_bits}",
        role="single",
        env=env,
        setup=(
            "Build the cells of the whole tree on all cores. Each repetition "
            "rebuilds all PIR preprocessing on one core."
        ),
        setup_seconds=150.0 * gib,
        repetition_seconds=58.0 * gib + private_reply_ciphertexts(
            pattern_bits, PRIVATE_SINGLE_GROUP_CELLS, PRIVATE_SINGLE_GROUP_CELLS
        ),
    )
    return job, _private_server_outputs(manifest, job, "single", updates=False)


def _build_private_single_table(manifest: Manifest) -> None:
    table = "tab:private-single"
    manifest.table_columns[table] = [
        "log-gib", "record-kib", "database", "state", "preprocess-size", "request", "reply-kib",
        "client-request-ms", "first-pass", "answer", "client-decode-ms",
        "preprocess-core-s", "preprocess-hint-s", "preprocess-packing-s",
    ]
    dash = _dash(manifest)
    first_gib = PRIVATE_SINGLE_MEASURED_GIB[0]
    first_bits = private_pattern_bits(leaves_of_gib(first_gib), 1)
    points = [(first_gib, first_bits)]
    points += [(first_gib, first_bits + offset) for offset in PRIVATE_SINGLE_PATTERN_OFFSETS]
    points += [
        (gib, private_pattern_bits(leaves_of_gib(gib), 1))
        for gib in PRIVATE_SINGLE_MEASURED_GIB[1:]
    ]
    for gib, pattern_bits in points:
        _, outputs = _private_single(manifest, gib, pattern_bits)
        row = f"measured:log{gib}gib-p{pattern_bits}"
        manifest.add_row(
            table,
            row,
            [
                cell(table, row, "log-gib", ensure_formula(manifest, "integer", gib),
                     "integer", "GiB of the log."),
                *_private_server_cells(
                    table, row, outputs,
                    size_render="gib-1", request_render="mib-1",
                    answer_render="seconds-3",
                ),
            ],
        )
    shape_job, _ = _private_single(manifest, first_gib, first_bits)
    for gib in PRIVATE_SINGLE_SHAPE_GIB:
        outputs = _private_outputs(
            manifest,
            shape_job,
            selector(role="single-shape", leaves=leaves_of_gib(gib)),
            tag=f"shape-log{gib}gib-",
        )
        row = f"shape:log{gib}gib-p{private_pattern_bits(leaves_of_gib(gib), 1)}"
        cells = [
            cell(table, row, "log-gib", ensure_formula(manifest, "integer", gib),
                 "integer", "GiB of the log."),
            cell(table, row, "record-kib", outputs["record_bytes"], "kib-1",
                 "KiB of one record."),
            cell(table, row, "database", outputs["database_bytes"], "gib-1",
                 "Records the server answers from."),
            cell(table, row, "request", outputs["request_bytes"], "mib-1",
                 "Serialized request."),
            cell(table, row, "reply-kib", outputs["reply_bytes"], "kib-1",
                 "Serialized reply."),
        ]
        cells += [
            cell(table, row, column, dash, "text", "Measured at 1 and 4 GiB.")
            for column in (
                "state", "preprocess-size", "client-request-ms", "first-pass", "answer",
                "client-decode-ms", "preprocess-core-s", "preprocess-hint-s", "preprocess-packing-s",
            )
        ]
        manifest.add_row(table, row, cells)


def _build_private_table_table(manifest: Manifest) -> None:
    table = "tab:private-table"
    manifest.table_columns[table] = [
        "log-gib", "records-mib", "table-mib", "records-first-pass-ms",
        "table-first-pass-ms", "records-answer-ms", "table-answer-ms",
    ]
    for gib in PRIVATE_TABLE_LOGS_GIB:
        pattern_bits = private_pattern_bits(leaves_of_gib(gib), PRIVATE_KAPPA)
        _, records = _private_maintainer(manifest, PRIVATE_KAPPA, gib, pattern_bits)
        job = _private_job(
            manifest,
            identifier=f"private-table-undine-log{gib}gib-k{PRIVATE_KAPPA}",
            role="table",
            env={
                "KAPPA": PRIVATE_KAPPA,
                "LEAVES": leaves_of_gib(gib),
                "SEGMENTS_PER_RECORD": 1,
                "PATTERN_BITS": pattern_bits,
            },
            setup="Build the table of complete openings of one claim and preprocess it.",
            setup_seconds=60.0 * gib,
            repetition_seconds=1.0 * gib,
        )
        outputs = _private_outputs(
            manifest,
            job,
            selector(role="table"),
            sizes={"table_bytes": "Bytes of the table of complete openings."},
            timings={
                "server_first_pass_ns": "First pass over the table.",
                "server_answer_ns": "From the serialized request to the serialized reply.",
            },
        )
        row = f"log{gib}gib"
        manifest.add_row(
            table,
            row,
            [
                cell(table, row, "log-gib", ensure_formula(manifest, "integer", gib),
                     "integer", "GiB of the log."),
                cell(table, row, "records-mib", records["database_bytes"], "mib-1",
                     "The maintainer's records."),
                cell(table, row, "table-mib", outputs["table_bytes"], "mib-1",
                     "The table of complete openings."),
                cell(table, row, "records-first-pass-ms", records["server_first_pass_ns"],
                     "milliseconds-1", "First pass over the records."),
                cell(table, row, "table-first-pass-ms", outputs["server_first_pass_ns"],
                     "milliseconds-1", "First pass over the table."),
                cell(table, row, "records-answer-ms", records["server_answer_ns"],
                     "milliseconds-1", "Answer from the records."),
                cell(table, row, "table-answer-ms", outputs["server_answer_ns"],
                     "milliseconds-1", "Answer from the table."),
            ],
        )


def _build_private_client_table(manifest: Manifest) -> None:
    table = "tab:private-client"
    manifest.table_columns[table] = [
        "kappa", "log-mib", "request-kib", "reply-kib", "client-request-ms",
        "private-kappa-ms", "public-kappa-ms", "private-3kappa-honest-ms",
        "public-3kappa-honest-ms", "private-3kappa-ms", "public-3kappa-ms",
    ]
    settings = [
        (kappa, leaves_of_gib(PRIVATE_CLIENT_LOG_GIB), f"private-client-undine-k{kappa}", f"k{kappa}")
        for kappa in KAPPAS + PRIVATE_CLIENT_LARGE_KAPPAS
    ]
    settings += [
        (
            PRIVATE_KAPPA,
            leaves_of_gib(gib),
            f"private-client-undine-log{gib}gib-k{PRIVATE_KAPPA}",
            f"log:log{gib}gib-k{PRIVATE_KAPPA}",
        )
        for gib in PRIVATE_CLIENT_LARGE_LOGS_GIB
    ]
    for kappa, leaves, identifier, row in settings:
        # Only records touched by the opening are encoded. The canary
        # calibrates this conservative fixed estimate on the current host.
        setup_seconds = 10.0 + kappa / 8
        env = {
            "KAPPA": kappa,
            "LEAVES": leaves,
            "SEGMENTS_PER_RECORD": 1,
            "PATTERN_BITS": private_pattern_bits(leaves, kappa),
        }
        job = _private_job(
            manifest,
            identifier=identifier,
            role="client",
            env=env,
            setup=(
                "Only the selected group and upper-node records of the unwritten "
                "log for 3*kappa claims, and each claim's reply to one request."
            ),
            setup_seconds=setup_seconds,
            repetition_seconds=max(0.1, (kappa / 8) ** 3 * 0.004),
        )
        outputs = _private_outputs(
            manifest,
            job,
            selector(role="client"),
            sizes={
                "request_bytes": PRIVATE_SIZE_FIELDS["request_bytes"],
                "reply_bytes": PRIVATE_SIZE_FIELDS["reply_bytes"],
            },
            timings={
                "client_request_ns": PRIVATE_SERVER_TIMINGS["client_request_ns"],
                "client_private_zero_bad_ns": "Decoding kappa replies, then Combine+Verify.",
                "client_private_honest_ns": (
                    "Decoding 3*kappa honest replies, then Combine with error "
                    "correction and Verify."
                ),
                "client_private_kappa_bad_ns": (
                    "Decoding 3*kappa replies with kappa lying, then Combine+Verify."
                ),
                "client_public_zero_bad_ns": "Combine+Verify from kappa public replies.",
                "client_public_honest_ns": (
                    "Combine with error correction and Verify from 3*kappa honest "
                    "public replies."
                ),
                "client_public_kappa_bad_ns": (
                    "Combine+Verify from 3*kappa public replies with kappa lying."
                ),
            },
        )
        manifest.add_row(
            table,
            row,
            [
                cell(table, row, "kappa", ensure_formula(manifest, "integer", kappa),
                     "integer", "Threshold."),
                cell(table, row, "log-mib", ensure_formula(manifest, "integer", leaves * FIELD_BYTES // MIB),
                     "integer", "MiB of the log's leaves."),
                cell(table, row, "request-kib", outputs["request_bytes"], "kib-1",
                     "Serialized request."),
                cell(table, row, "reply-kib", outputs["reply_bytes"], "kib-1",
                     "Serialized reply of one claim."),
                cell(table, row, "client-request-ms", outputs["client_request_ns"],
                     "milliseconds-1", "Key generation, the query, and serialization."),
                cell(table, row, "private-kappa-ms", outputs["client_private_zero_bad_ns"],
                     "milliseconds-3", "Private opening from kappa replies."),
                cell(table, row, "public-kappa-ms", outputs["client_public_zero_bad_ns"],
                     "milliseconds-3", "Public opening from kappa replies."),
                cell(table, row, "private-3kappa-honest-ms", outputs["client_private_honest_ns"],
                     "milliseconds-3", "Private opening from 3*kappa honest replies, correcting."),
                cell(table, row, "public-3kappa-honest-ms", outputs["client_public_honest_ns"],
                     "milliseconds-3", "Public opening from 3*kappa honest replies, correcting."),
                cell(table, row, "private-3kappa-ms", outputs["client_private_kappa_bad_ns"],
                     "milliseconds-3", "Private opening from 3*kappa replies, kappa lying."),
                cell(table, row, "public-3kappa-ms", outputs["client_public_kappa_bad_ns"],
                     "milliseconds-3", "Public opening from 3*kappa replies, kappa lying."),
            ],
        )


def treepir_class_size(height: int, color: int) -> int:
    """Nodes of TreePIR's class `color` in a tree of height `height`: the
    2^(h+1) - 2 non-root nodes split into h balanced classes, the larger
    ones last."""
    nodes = (1 << (height + 1)) - 2
    small = height - nodes % height
    return nodes // height + (1 if color >= small else 0)


def _treepir_job(
    manifest: Manifest,
    *,
    identifier: str,
    role: str,
    env: Mapping[str, Any],
    setup: str,
    setup_seconds: float,
    repetition_seconds: float,
) -> JobSpec:
    return add_job(
        manifest,
        identifier=identifier,
        layout="undine",
        family=f"treepir-{role}",
        benchmark=TREEPIR_BENCHMARK,
        features=("bench",),
        env=job_env(
            **{f"{TREEPIR_PREFIX}_ROLE": role},
            **{f"{TREEPIR_PREFIX}_{key}": value for key, value in env.items()},
        ),
        result_relpath=f"results/{TREEPIR_BENCHMARK}.json",
        trial_env=f"{TREEPIR_PREFIX}_TRIALS",
        setup=setup,
        setup_seconds=setup_seconds,
        repetition_seconds=repetition_seconds,
    )


def _treepir_outputs(
    manifest: Manifest,
    job: JobSpec,
    match: tuple[tuple[str, Any], ...],
    tag: str,
    *,
    sizes: Mapping[str, str] = {},
    counts: Mapping[str, tuple[str, str]] = {},
    timings: Mapping[str, str] = {},
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for name, description in sizes.items():
        outputs[name] = ensure_output(
            manifest, job, f"{tag}{name}", entry_selector=match, path=name,
            kind="value", unit="bytes", description=description,
        )
    for name, (unit, description) in counts.items():
        outputs[name] = ensure_output(
            manifest, job, f"{tag}{name}", entry_selector=match, path=name,
            kind="value", unit=unit, description=description,
        )
    for name, description in timings.items():
        role = dict(match)["role"]
        if role == "server":
            group = "online" if name in {"first_pass_ns", "answer_ns"} else "preprocess"
        elif role == "paired":
            group = "paired"
        else:
            # Keep the allocator/workload history fixed: dropping the directly
            # timed opening once it converges changes serialization timings in
            # the component opening. These two openings are one sampling group.
            group = "client"
        outputs[name] = ensure_output(
            manifest, job, f"{tag}{name}", entry_selector=match, path=name,
            kind="timing", unit="ns", sample_mode="rebuild" if group == "preprocess" else "direct",
            sample_group=group,
            description=description,
        )
    return outputs


TREEPIR_MESSAGE_SIZES = {
    "request_bytes": "Serialized bytes of one request.",
    "query_bytes": "Bytes of the request's query.",
    "key_bytes": "Bytes of the request's packing keys.",
    "reply_bytes": "Serialized bytes of one reply.",
}
TREEPIR_SERVER_SIZES = {
    "pir_bytes": "The PIR database.",
    "hint_bytes": "The hint.",
    "packing_bytes": "The response-packing tables.",
    "serving_state_bytes": "The PIR database, the hint, and the packing tables.",
    **TREEPIR_MESSAGE_SIZES,
}
TREEPIR_SERVER_COUNTS = {
    "outputs": ("ciphertexts", "Ciphertexts of one reply."),
    "hint_blocks": ("", "Blocks of 2,048 rows of one database column."),
    "zero_hint_blocks": ("", "All-zero blocks, which the hint skips."),
    "padding_hint_blocks": ("", "Blocks of the padding columns."),
}
TREEPIR_SERVER_ONLINE = {
    "first_pass_ns": "The query's inner product with every database column.",
    "answer_ns": "From the serialized request to the serialized reply.",
}
TREEPIR_SERVER_STAGES = {
    "undine": {
        "gather_cells_ns": "The primary columns into PIR cells.",
        "gather_upper_ns": "The row table and the upper nodes' cells.",
        "hint_ns": "The hint, gathering its columns from the cells.",
        "packing_ns": "The response-packing tables.",
        "prepare_ns": "The four one-core stages of one rebuild.",
    },
    "treepir": {
        "enumerate_ns": "The class's nodes in database order.",
        "columns_ns": "The database columns from the class's labels.",
        "hint_ns": "The hint over the database columns.",
        "packing_ns": "The response-packing tables.",
        "prepare_ns": "The four stages of one rebuild.",
    },
}
TREEPIR_STAGE_COLUMNS = {
    "gather-cells-s": ("gather_cells_ns", "undine", "seconds-1"),
    "gather-upper-s": ("gather_upper_ns", "undine", "seconds-3"),
    "enumerate-s": ("enumerate_ns", "treepir", "seconds-1"),
    "columns-s": ("columns_ns", "treepir", "seconds-1"),
    "hint-s": ("hint_ns", None, "seconds-1"),
    "packing-s": ("packing_ns", None, "seconds-1"),
    "prepare-s": ("prepare_ns", None, "seconds-1"),
}
TREEPIR_CLIENT_TIMINGS = {
    "index_ns": "The pattern and the opening plan, or the h colors and offsets.",
    "keygen_ns": "Fresh secret keys: one, or one per TreePIR server.",
    "query_ns": "The encrypted queries and their packing keys.",
    "serialize_ns": "The serialized requests.",
    "decrypt_ns": "Parsing and decrypting every reply and extracting the opening.",
    "verify_ns": "The root from the recovered value and path.",
    "request_total_ns": "Preparing the requests, timed as a whole.",
    "process_total_ns": "The replies into a verified opening, timed as a whole.",
    "client_total_ns": "Preparing the requests and processing the replies.",
}
TREEPIR_CLIENT_COLUMNS = (
    ("index-us", "index_ns", "microseconds-1"),
    ("keygen-ms", "keygen_ns", "milliseconds-3"),
    ("query-ms", "query_ns", "milliseconds-3"),
    ("serialize-ms", "serialize_ns", "milliseconds-3"),
    ("request-total-ms", "request_total_ns", "milliseconds-3"),
    ("decrypt-ms", "decrypt_ns", "milliseconds-3"),
    ("combine-verify-ms", "combine_verify_ns", "milliseconds-3"),
    ("verify-us", "verify_ns", "microseconds-1"),
    ("process-total-ms", "process_total_ns", "milliseconds-3"),
    ("client-total-ms", "client_total_ns", "milliseconds-3"),
)


def _treepir_server_job(manifest: Manifest, claim: int, color: int) -> JobSpec:
    return _treepir_job(
        manifest,
        identifier=f"treepir-server-h{TREEPIR_HEIGHT}-m{claim}-c{color}",
        role="server",
        env={
            "HEIGHT": TREEPIR_HEIGHT,
            "MAINTAINER": claim,
            "COLOR": color,
            "SEEDED_LEAVES": TREEPIR_SEEDED_LEAVES,
        },
        setup=(
            "The valid Merkle tree of an unwritten prefix followed by seeded "
            "leaves, and the maintainer's fragment on it, built once and kept "
            "on disk; every repetition reloads the fragment untimed and "
            "rebuilds both PIR servers."
        ),
        setup_seconds=240.0,
        repetition_seconds=330.0,
    )


def _treepir_client_job(manifest: Manifest, height: int, replies: str) -> JobSpec:
    real = replies == "real"
    return _treepir_job(
        manifest,
        identifier=f"treepir-client-h{height}-{replies}",
        role="client",
        env={
            "HEIGHT": height,
            "REPLIES": replies,
            "SEEDED_LEAVES": min(TREEPIR_SEEDED_LEAVES, 1 << height),
        },
        setup=(
            "Every server of both systems, built on the tree."
            if real
            else "The records the openings read, encoded from the tree, then one "
            "real server of each system for the paired check."
        ),
        setup_seconds=60.0 if real else 480.0,
        repetition_seconds=3.0 if real else 10.0,
    )


def _build_treepir_tables(manifest: Manifest) -> None:
    dash = _dash(manifest)
    table = "tab:treepir-server"
    manifest.table_columns[table] = [
        "system", "server", "outputs", "pir-mib", "hint-mib", "packing-mib",
        "state-mib", "hint-blocks", "zero-blocks", "padding-blocks",
        *TREEPIR_STAGE_COLUMNS, "first-pass-ms", "answer-ms",
        "request-kib", "query-kib", "key-kib", "reply-kib",
    ]
    for claim, color in TREEPIR_SERVERS:
        job = _treepir_server_job(manifest, claim, color)
        for system, server in (("undine", claim), ("treepir", color)):
            outputs = _treepir_outputs(
                manifest,
                job,
                selector(role="server", system=system),
                f"{system}-",
                sizes=TREEPIR_SERVER_SIZES,
                counts=TREEPIR_SERVER_COUNTS,
                timings=TREEPIR_SERVER_STAGES[system] | TREEPIR_SERVER_ONLINE,
            )
            row = f"m{claim}-c{color}:{system}"
            cells = [
                cell(table, row, "system", ensure_formula(manifest, "text", system),
                     "text", "The system."),
                cell(table, row, "server", ensure_formula(manifest, "integer", server),
                     "integer", "The UndineLog claim or the TreePIR class."),
                cell(table, row, "outputs", outputs["outputs"], "integer",
                     "Ciphertexts of one reply."),
                cell(table, row, "pir-mib", outputs["pir_bytes"], "mib-1",
                     "The PIR database."),
                cell(table, row, "hint-mib", outputs["hint_bytes"], "mib-1",
                     "The hint."),
                cell(table, row, "packing-mib", outputs["packing_bytes"], "mib-1",
                     "The response-packing tables."),
                cell(table, row, "state-mib", outputs["serving_state_bytes"], "mib-1",
                     "The PIR database, the hint, and the packing tables."),
                cell(table, row, "hint-blocks", outputs["hint_blocks"], "integer",
                     "Blocks of 2,048 rows of one database column."),
                cell(table, row, "zero-blocks", outputs["zero_hint_blocks"], "integer",
                     "All-zero blocks, which the hint skips."),
                cell(table, row, "padding-blocks", outputs["padding_hint_blocks"], "integer",
                     "Blocks of the padding columns."),
            ]
            for column, (metric, owner, render) in TREEPIR_STAGE_COLUMNS.items():
                if owner in (None, system):
                    cells.append(cell(table, row, column, outputs[metric], render,
                                      TREEPIR_SERVER_STAGES[system][metric]))
                else:
                    cells.append(cell(table, row, column, dash, "text",
                                      "A stage of the other system."))
            cells += [
                cell(table, row, "first-pass-ms", outputs["first_pass_ns"],
                     "milliseconds-1", TREEPIR_SERVER_ONLINE["first_pass_ns"]),
                cell(table, row, "answer-ms", outputs["answer_ns"],
                     "milliseconds-1", TREEPIR_SERVER_ONLINE["answer_ns"]),
                cell(table, row, "request-kib", outputs["request_bytes"], "kib-1",
                     "Serialized request."),
                cell(table, row, "query-kib", outputs["query_bytes"], "kib-1",
                     "The request's query."),
                cell(table, row, "key-kib", outputs["key_bytes"], "kib-1",
                     "The request's packing keys."),
                cell(table, row, "reply-kib", outputs["reply_bytes"], "kib-1",
                     "Serialized reply."),
            ]
            manifest.add_row(table, row, cells)

    table = "tab:treepir-client"
    manifest.table_columns[table] = [
        "system", "height", "replies", "servers", "queries", "request-kib",
        "upload-mib", "download-kib",
        *(column for column, _, _ in TREEPIR_CLIENT_COLUMNS),
    ]
    # The all-real height-16 workload is a standalone correctness check,
    # not a publication measurement that needs adaptive timing precision.
    check_outputs: dict[str, dict[str, str]] = {}
    for height, replies in (
        (TREEPIR_HEIGHT, "generated"),
    ):
        job = _treepir_client_job(manifest, height, replies)
        for system in ("undine", "treepir"):
            timings = dict(TREEPIR_CLIENT_TIMINGS)
            if system == "undine":
                timings["combine_verify_ns"] = "Combining the shares and verifying the path."
            outputs = _treepir_outputs(
                manifest,
                job,
                selector(role="client", system=system),
                f"{system}-",
                sizes={
                    "request_bytes": TREEPIR_MESSAGE_SIZES["request_bytes"],
                    "upload_bytes": "Every transmitted request of one opening.",
                    "download_bytes": "Every reply of one opening.",
                },
                counts={
                    "servers": ("", "Servers contacted per opening."),
                    "queries": ("", "Distinct queries per opening."),
                },
                timings=timings,
            )
            row = f"h{height}-{replies}:{system}"
            cells = [
                cell(table, row, "system", ensure_formula(manifest, "text", system),
                     "text", "The system."),
                cell(table, row, "height", ensure_formula(manifest, "integer", height),
                     "integer", "Height of the tree, and the threshold."),
                cell(table, row, "replies", ensure_formula(manifest, "text", replies),
                     "text", "Where the replies come from."),
                cell(table, row, "servers", outputs["servers"], "integer",
                     "Servers contacted per opening."),
                cell(table, row, "queries", outputs["queries"], "integer",
                     "Distinct queries per opening."),
                cell(table, row, "request-kib", outputs["request_bytes"], "kib-1",
                     "Serialized bytes of one request."),
                cell(table, row, "upload-mib", outputs["upload_bytes"], "mib-1",
                     "Every transmitted request of one opening."),
                cell(table, row, "download-kib", outputs["download_bytes"], "kib-1",
                     "Every reply of one opening."),
            ]
            for column, metric, render in TREEPIR_CLIENT_COLUMNS:
                if metric in outputs:
                    cells.append(cell(table, row, column, outputs[metric], render,
                                      timings[metric]))
                else:
                    cells.append(cell(table, row, column, dash, "text",
                                      "UndineLog's reconstruction step."))
            manifest.add_row(table, row, cells)
            if replies == "generated":
                check_outputs[system] = _treepir_outputs(
                    manifest,
                    job,
                    selector(role="paired", system=system),
                    f"paired-{system}-",
                    counts={"server": ("", "The real server's claim or class.")},
                    timings={
                        "real_decode_ns": "Decoding the real server's reply.",
                        "generated_decode_ns": "Decoding the generated reply.",
                    },
                )

    table = "tab:treepir-check"
    manifest.table_columns[table] = [
        "system", "server", "real-decode-ms", "generated-decode-ms",
    ]
    for system, outputs in check_outputs.items():
        row = system
        manifest.add_row(
            table,
            row,
            [
                cell(table, row, "system", ensure_formula(manifest, "text", system),
                     "text", "The system."),
                cell(table, row, "server", outputs["server"], "integer",
                     "The real server's claim or class."),
                cell(table, row, "real-decode-ms", outputs["real_decode_ns"],
                     "milliseconds-3", "Decoding the real server's reply."),
                cell(table, row, "generated-decode-ms", outputs["generated_decode_ns"],
                     "milliseconds-3", "Decoding the generated reply."),
            ],
        )


def evaluate_formula(spec: MeasurementSpec) -> Any:
    name = spec.formula
    args = spec.formula_args
    if name in {"integer", "text"}:
        return args[0]
    if name == "maintainer-state-bytes":
        layout = str(args[0])
        cursor, segment_len, kappa = map(int, args[1:])
        return (
            ceil_div(cursor, segment_len)
            * stored_elements_per_segment(layout, segment_len, kappa)
            * FIELD_BYTES
        )
    if name == "append-fragment-growth-bytes":
        layout = str(args[0])
        (
            tree_height,
            values,
            segment_len,
            kappa,
            blocks,
            warmup,
        ) = map(int, args[1:])
        return sequence_state_growth_bytes(
            layout=layout,
            tree_height=tree_height,
            inserted_values_per_batch=values,
            segment_len=segment_len,
            kappa=kappa,
            sequence_blocks=blocks,
            warmup_blocks=warmup,
        )
    raise PipelineError(f"unknown formula {name!r} for {spec.id}")


def validate_manifest(manifest: Manifest) -> None:
    errors: list[str] = []
    if set(manifest.table_rows) != set(TABLE_LABELS):
        errors.append(
            "manifest table set differs from expected labels: "
            f"{sorted(set(TABLE_LABELS) ^ set(manifest.table_rows))}"
        )
    consumers = manifest.consumers()
    for cell_spec in manifest.cells.values():
        for source in cell_spec.sources:
            if source not in manifest.measurements:
                errors.append(f"{cell_spec.id}: unknown source {source}")
        try:
            validate_cell_unit_contract(cell_spec, manifest)
        except PipelineError as exc:
            errors.append(str(exc))
    for measurement in manifest.measurements.values():
        if not consumers.get(measurement.id):
            errors.append(f"measurement has no table consumer: {measurement.id}")
        if measurement.job_id is not None and measurement.job_id not in manifest.jobs:
            errors.append(
                f"{measurement.id}: unknown producing job {measurement.job_id}"
            )
        if measurement.kind == "formula":
            try:
                value = evaluate_formula(measurement)
                validate_measurement_value(measurement, value)
            except Exception as exc:  # noqa: BLE001 - aggregate validation failures
                errors.append(f"{measurement.id}: invalid formula: {exc}")
    semantic_jobs: dict[str, str] = {}
    for job in manifest.jobs.values():
        key = job.semantic_key()
        if key in semantic_jobs:
            errors.append(
                f"duplicate scheduled job semantics: {semantic_jobs[key]} and {job.id}"
            )
        semantic_jobs[key] = job.id
        timing = [
            item
            for item in manifest.measurements_for_job(job.id)
            if item.kind == "timing"
        ]
        if not timing:
            errors.append(f"job has no consumed timing metric: {job.id}")
    for table, rows in manifest.table_rows.items():
        columns = manifest.table_columns.get(table)
        if not columns:
            errors.append(f"{table}: missing column declaration")
            continue
        for row in rows:
            actual = {
                item.column
                for item in manifest.cells.values()
                if item.table == table and item.row == row
            }
            if actual != set(columns):
                errors.append(
                    f"{table}|{row}: columns differ: "
                    f"missing={sorted(set(columns)-actual)}, extra={sorted(actual-set(columns))}"
                )
    if errors:
        raise PipelineError("manifest validation failed:\n- " + "\n- ".join(errors))


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    maximum_iterations = 300
    epsilon = 3.0e-14
    floor = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        m2 = 2 * iteration
        numerator = iteration * (b - iteration) * x / (
            (qam + m2) * (a + m2)
        )
        d = 1.0 + numerator * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + numerator / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        result *= d * c
        numerator = -(a + iteration) * (qab + iteration) * x / (
            (a + m2) * (qap + m2)
        )
        d = 1.0 + numerator * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + numerator / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise PipelineError("incomplete-beta continued fraction did not converge")


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if a <= 0 or b <= 0 or not 0.0 <= x <= 1.0:
        raise ValueError("invalid incomplete-beta arguments")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    if value == 0.0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * regularized_incomplete_beta(
        degrees_of_freedom / 2.0, 0.5, x
    )
    return 1.0 - tail if value > 0 else tail


def student_t_ppf(probability: float, degrees_of_freedom: int) -> float:
    if not 0.5 < probability < 1.0:
        raise ValueError("only upper-half Student-t quantiles are supported")
    low = 0.0
    high = 1.0
    while student_t_cdf(high, degrees_of_freedom) < probability:
        high *= 2.0
        if high > 1.0e6:
            raise PipelineError("failed to bracket Student-t quantile")
    for _ in range(100):
        middle = (low + high) / 2.0
        if student_t_cdf(middle, degrees_of_freedom) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def summarize_samples(
    samples: Sequence[float],
    *,
    relative_target: float = DEFAULT_RELATIVE_HALF_WIDTH,
    looks: Sequence[int] = SEQUENTIAL_LOOKS,
) -> dict[str, Any]:
    numeric = [float(value) for value in samples]
    count = len(numeric)
    if count == 0:
        return {
            "n": 0,
            "mean": None,
            "standard_deviation": None,
            "ci95": None,
            "sequential_ci": None,
            "stable": False,
            "reason": "no samples",
        }
    mean = statistics.fmean(numeric)
    if count == 1:
        return {
            "n": 1,
            "mean": mean,
            "standard_deviation": None,
            "ci95": None,
            "sequential_ci": None,
            "stable": False,
            "reason": "fewer than two independent repetitions",
        }
    standard_deviation = statistics.stdev(numeric)
    standard_error = standard_deviation / math.sqrt(count)
    critical_95 = student_t_ppf(0.975, count - 1)
    half_width_95 = critical_95 * standard_error
    look_count = len(tuple(looks))
    per_look_alpha = SEQUENTIAL_FAMILY_ALPHA / look_count
    sequential_probability = 1.0 - per_look_alpha / 2.0
    critical_sequential = student_t_ppf(sequential_probability, count - 1)
    half_width_sequential = critical_sequential * standard_error
    relative_95 = math.inf if mean == 0 else abs(half_width_95 / mean)
    relative_sequential = (
        math.inf if mean == 0 else abs(half_width_sequential / mean)
    )
    at_declared_look = count in set(looks)
    stable = (
        count >= DEFAULT_MIN_REPETITIONS
        and at_declared_look
        and relative_sequential <= relative_target
    )
    return {
        "n": count,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "ci95": {
            "confidence": 0.95,
            "low": mean - half_width_95,
            "high": mean + half_width_95,
            "half_width": half_width_95,
            "relative_half_width": relative_95,
            "critical_value": critical_95,
        },
        "sequential_ci": {
            "familywise_confidence": 1.0 - SEQUENTIAL_FAMILY_ALPHA,
            "pointwise_confidence": 1.0 - per_look_alpha,
            "looks": list(looks),
            "low": mean - half_width_sequential,
            "high": mean + half_width_sequential,
            "half_width": half_width_sequential,
            "relative_half_width": relative_sequential,
            "critical_value": critical_sequential,
        },
        "stable": stable,
        "reason": (
            "sequential confidence target met"
            if stable
            else "sequential relative half-width exceeds target"
        ),
    }


def benchmark_warmup_repetitions(run_job: RunJob, mode: str) -> int:
    if mode not in ADAPTIVE_MODES:
        return 0
    modes = {
        measurement.sample_mode
        for measurement in run_job.measurements
        if measurement.kind == "timing"
    }
    if not modes:
        return 0
    if modes == {"append-sequence"}:
        return APPEND_WARMUP_REPETITIONS
    if modes == {"direct"}:
        return DIRECT_WARMUP_REPETITIONS
    if modes == {"rebuild"}:
        return REBUILD_WARMUP_REPETITIONS
    groups = {item.sample_group for item in run_job.measurements if item.kind == "timing"}
    if len(groups) > 1:
        return max(benchmark_warmup_repetitions(
            dataclasses.replace(run_job, measurements=tuple(item for item in run_job.measurements
                if item.kind == "timing" and item.sample_group == group)), mode
        ) for group in groups)
    raise PipelineError(
        f"{run_job.spec.id}: cannot select one warmup policy for sample modes "
        f"{sorted(modes)}"
    )


def source_files() -> list[pathlib.Path]:
    direct = [
        ROOT / "artifact",
        pathlib.Path(__file__).resolve(),
        ROOT / "scripts" / "check_artifact.py",
        ROOT / "scripts" / "check_docs.py",
        ROOT / "scripts" / "export_results.py",
        ROOT / "scripts" / "package_reference.py",
        ROOT / "scripts" / "source_provenance.py",
        ROOT / "scripts" / "test_reproduce.py",
        ROOT / "scripts" / "verify_results.py",
    ]
    for candidate in ROOT.glob("rust-toolchain*"):
        direct.append(candidate)
    for crate in CRATE_ROOTS.values():
        direct.extend([crate / "Cargo.toml", crate / "Cargo.lock"])
        direct.extend(crate.glob("rust-toolchain*"))
        for base in (
            crate / "src",
            crate / "benches",
            crate / "tests",
            crate / "apps" / "appendOnlyLog",
            crate / "third_party",
        ):
            if not base.exists():
                continue
            direct.extend(
                path
                for path in base.rglob("*")
                if path.is_file() and "target" not in path.relative_to(base).parts
            )
    return sorted(set(path.resolve() for path in direct if path.exists()))


def collect_source_snapshot() -> dict[str, Any]:
    files = {
        str(path.relative_to(ROOT)): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in source_files()
    }
    git_head = run_capture(["git", "rev-parse", "HEAD"]).stdout.decode().strip()
    git_status = run_capture(
        ["git", "status", "--short", "--untracked-files=all"]
    ).stdout.decode()
    git_diff = run_capture(["git", "diff", "--binary", "HEAD"]).stdout
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "git_head": git_head,
        "git_status_sha256": sha256_bytes(git_status.encode()),
        "git_diff_sha256": sha256_bytes(git_diff),
        "build_configuration": collect_build_configuration(),
        "files": files,
    }
    payload["snapshot_id"] = sha256_bytes(canonical_json(payload))
    return payload


def collect_build_configuration() -> dict[str, Any]:
    """Bind resumed builds to the compiler, flags and Cargo configuration."""
    configs: set[pathlib.Path] = set()
    for crate in CRATE_ROOTS.values():
        for directory in (crate, *crate.parents):
            configs.update(directory / ".cargo" / name for name in ("config", "config.toml"))
    cargo_home = pathlib.Path(os.environ.get("CARGO_HOME", pathlib.Path.home() / ".cargo"))
    configs.update(cargo_home / name for name in ("config", "config.toml"))
    flags = {
        key: value for key, value in os.environ.items()
        if key.startswith(("RUST", "CARGO_BUILD_", "CARGO_PROFILE_", "CARGO_TARGET_"))
        or key in {"CARGO_ENCODED_RUSTFLAGS", "CC", "CXX", "CFLAGS", "CXXFLAGS", "AR"}
    }
    # The pipeline overrides the target directory and disables compiler wrappers.
    for key in ("CARGO_TARGET_DIR", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER"):
        flags.pop(key, None)
    return {
        "rustc": run_capture(["rustc", "--version", "--verbose"]).stdout.decode().strip(),
        "cargo": run_capture(["cargo", "--version"]).stdout.decode().strip(),
        "flags": dict(sorted(flags.items())),
        "configs": {str(path): sha256_file(path) for path in sorted(configs) if path.is_file()},
        "compiler_wrappers": "disabled",
    }


def validate_current_source_snapshot(
    expected_snapshot_id: str,
    *,
    phase: str,
) -> None:
    actual_snapshot_id = str(collect_source_snapshot()["snapshot_id"])
    if actual_snapshot_id != expected_snapshot_id:
        raise PipelineError(
            f"source snapshot changed during {phase}: expected "
            f"{expected_snapshot_id}, found {actual_snapshot_id}; restore the "
            "recorded source state before resuming"
        )


def _read_cpu_text(cpu: int, relative: str) -> str | None:
    path = pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu}") / relative
    try:
        return path.read_text().strip()
    except OSError:
        return None


def benchmark_cpu_settings(cpu: int | None) -> dict[str, Any] | None:
    """Frequency-scaling settings and topology of the benchmark CPU."""
    if cpu is None:
        return None
    return {
        "cpu": cpu,
        "scaling_governor": _read_cpu_text(cpu, "cpufreq/scaling_governor"),
        "energy_performance_preference": _read_cpu_text(
            cpu, "cpufreq/energy_performance_preference"
        ),
        "scaling_driver": _read_cpu_text(cpu, "cpufreq/scaling_driver"),
        "cpuinfo_max_freq_khz": _read_cpu_integer(cpu, "cpufreq/cpuinfo_max_freq"),
        "core_id": _read_cpu_integer(cpu, "topology/core_id"),
        "thread_siblings_list": _read_cpu_text(cpu, "topology/thread_siblings_list"),
    }


def collect_environment(benchmark_cpu: int | None = None) -> dict[str, Any]:
    commands: dict[str, dict[str, Any]] = {}
    for name, command in (
        ("python", [sys.executable, "--version"]),
        ("rustc", ["rustc", "--version", "--verbose"]),
        ("cargo", ["cargo", "--version", "--verbose"]),
        ("git", ["git", "--version"]),
    ):
        try:
            completed = run_capture(command, check=False)
            commands[name] = {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout.decode(errors="replace").strip(),
                "stderr": completed.stderr.decode(errors="replace").strip(),
            }
        except OSError as exc:
            commands[name] = {"command": command, "error": str(exc)}
    proc_files: dict[str, str] = {}
    for name in ("cpuinfo", "meminfo"):
        path = pathlib.Path("/proc") / name
        if path.exists():
            proc_files[name] = path.read_text(errors="replace")
    return {
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "uname": list(platform.uname()),
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "process_cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "benchmark_cpu": benchmark_cpu_settings(benchmark_cpu),
        "commands": commands,
        "proc": proc_files,
    }


def _read_cpu_integer(cpu: int, relative: str) -> int | None:
    path = pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu}") / relative
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, PermissionError, ValueError):
        return None


def choose_benchmark_cpu(requested: int | None) -> int | None:
    if not hasattr(os, "sched_getaffinity"):
        if requested is not None:
            raise PipelineError("--cpu requires operating-system affinity support")
        return None
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        raise PipelineError("the process has an empty CPU affinity set")
    if requested is not None:
        if requested not in allowed:
            raise PipelineError(
                f"--cpu {requested} is outside the allowed CPU set {allowed}"
            )
        return requested

    frequencies = {
        cpu: _read_cpu_integer(cpu, "cpufreq/cpuinfo_max_freq") or 0
        for cpu in allowed
    }
    maximum = max(frequencies.values())
    candidates = [
        cpu for cpu in allowed if maximum == 0 or frequencies[cpu] == maximum
    ]
    first_core = _read_cpu_integer(allowed[0], "topology/core_id")
    separate_core = [
        cpu
        for cpu in candidates
        if cpu != 0
        and (
            first_core is None
            or _read_cpu_integer(cpu, "topology/core_id") != first_core
        )
    ]
    if separate_core:
        return min(separate_core)
    nonzero = [cpu for cpu in candidates if cpu != 0]
    return min(nonzero or candidates)


@dataclass(frozen=True)
class RunJob:
    spec: JobSpec
    measurements: tuple[MeasurementSpec, ...]
    env_overrides: tuple[tuple[str, str], ...] = ()
    features_override: tuple[str, ...] | None = None

    def effective_env(self) -> dict[str, str]:
        result = self.spec.env_dict()
        result.update(dict(self.env_overrides))
        return result

    def effective_features(self) -> tuple[str, ...]:
        return self.spec.features if self.features_override is None else self.features_override


def _measurements_with_tag(
    manifest: Manifest, job_id: str, fragments: Sequence[str]
) -> tuple[MeasurementSpec, ...]:
    selected = []
    for measurement in manifest.measurements_for_job(job_id):
        if any(fragment in measurement.id for fragment in fragments):
            selected.append(measurement)
    if not selected:
        raise PipelineError(f"no measurements selected for {job_id}: {fragments}")
    return tuple(selected)


def select_run_jobs(
    manifest: Manifest, mode: str, job_ids: Sequence[str] | None = None
) -> list[RunJob]:
    if mode == "full":
        return [
            RunJob(job, tuple(manifest.measurements_for_job(job.id)))
            for job in manifest.jobs.values()
        ]
    if mode == "jobs":
        if not job_ids:
            raise PipelineError("a jobs run names at least one job")
        unknown = [job_id for job_id in job_ids if job_id not in manifest.jobs]
        if unknown:
            raise PipelineError(f"unknown jobs: {', '.join(unknown)}")
        return [
            RunJob(manifest.jobs[job_id], tuple(manifest.measurements_for_job(job_id)))
            for job_id in dict.fromkeys(job_ids)
        ]
    if mode not in {"smoke", "canary"}:
        raise ValueError(f"unsupported execution mode: {mode}")

    selected: list[RunJob] = []

    def add(
        job_id: str,
        *,
        tags: Sequence[str] | None = None,
        overrides: Mapping[str, str | int] | None = None,
        features: tuple[str, ...] | None = None,
        measurements: Sequence[MeasurementSpec] | None = None,
    ) -> None:
        job = manifest.jobs[job_id]
        if measurements is None:
            picked = (
                tuple(manifest.measurements_for_job(job_id))
                if tags is None
                else _measurements_with_tag(manifest, job_id, tags)
            )
        else:
            picked = tuple(measurements)
        selected.append(
            RunJob(
                job,
                picked,
                tuple(sorted((key, str(value)) for key, value in (overrides or {}).items())),
                features,
            )
        )

    def without_fragment(job_id: str, fragment: str) -> tuple[MeasurementSpec, ...]:
        return tuple(
            measurement
            for measurement in manifest.measurements_for_job(job_id)
            if fragment not in measurement.id
        )

    one_mib = ("bench", "log-bulk-updates-1mb")
    for layout in LAYOUTS:
        if mode == "smoke":
            add(
                f"append-4mib-{layout}-k8",
                overrides={
                    "BENCH_APPEND_ONLY_LOG_BULK_UPDATES_APPEND_SEQUENCE_BLOCKS": 2,
                    "BENCH_APPEND_ONLY_LOG_BULK_UPDATES_APPEND_SEQUENCE_WARMUP_BLOCKS": 0,
                },
                features=one_mib,
            )
            add(
                f"point-helper-{layout}-k8",
                overrides={"BENCH_APPEND_ONLY_LOG_POINT_OPEN_COMMITTED_BULK_UPDATE_BATCHES": 1},
                features=one_mib,
            )
            add(
                f"point-client-{layout}-k8",
                overrides={
                    "BENCH_APPEND_ONLY_LOG_CLIENT_POINT_OPEN_COMMITTED_BULK_UPDATE_BATCHES": 1
                },
                features=one_mib,
            )
            for role in ("helper", "client"):
                job_id = f"range-{role}-{layout}-k{RANGE_KAPPA}"
                add(
                    job_id,
                    overrides={
                        "BENCH_APPEND_ONLY_LOG_RANGE_OPEN_COMMITTED_BULK_UPDATE_BATCHES": 1,
                        "BENCH_APPEND_ONLY_LOG_RANGE_OPEN_BYTE_LEN_LIST": MIB,
                    },
                    measurements=_measurements_with_tag(manifest, job_id, (":1mib-",)),
                )
            add(
                f"repair-{layout}-k8",
                overrides={"BENCH_APPEND_ONLY_LOG_REPAIR_COMMITTED_BULK_UPDATE_BATCHES": 1},
                features=one_mib,
            )
            add(
                f"reconfig-{layout}-k8",
                overrides={
                    "BENCH_APPEND_ONLY_LOG_KAPPA_RESCALE_COMMITTED_BULK_UPDATE_BATCHES": 1
                },
                features=one_mib,
            )
        else:
            kappa = RANGE_KAPPA
            add(f"append-4mib-{layout}-k{kappa}")
            add(f"point-helper-{layout}-k{kappa}")
            add(f"point-client-{layout}-k{kappa}")
            for role in ("helper", "client"):
                job_id = f"range-{role}-{layout}-k{kappa}"
                add(
                    job_id,
                    overrides={"BENCH_APPEND_ONLY_LOG_RANGE_OPEN_BYTE_LEN_LIST": MIB},
                    measurements=_measurements_with_tag(manifest, job_id, (":1mib-",)),
                )
            add(f"repair-{layout}-k{kappa}")
            add(f"reconfig-{layout}-k{kappa}")
    if mode == "smoke":
        # One 1 MiB batch is committed, so only the 1 MiB range fits.
        add(
            "on-the-fly-undine-k8",
            overrides={
                "BENCH_APPEND_ONLY_LOG_ON_THE_FLY_OPEN_COMMITTED_BULK_UPDATE_BATCHES": 1,
                "BENCH_APPEND_ONLY_LOG_ON_THE_FLY_OPEN_RANGE_BYTE_LEN_LIST": MIB,
            },
            features=one_mib,
            measurements=without_fragment("on-the-fly-undine-k8", "-range-4mib-"),
        )
    else:
        add(f"on-the-fly-undine-k{RANGE_KAPPA}")
    first_gib = PRIVATE_LOGS_GIB[0]
    if mode == "smoke":
        small = {f"{PRIVATE_PREFIX}_LEAVES": PRIVATE_SMOKE_LEAVES}
        smallest = KAPPAS[0]
        sweep_leaves = leaves_of_gib(PRIVATE_SWEEP_LOG_GIB)
        add(
            f"private-maintainer-undine-log{PRIVATE_SWEEP_LOG_GIB}gib"
            f"-p{private_pattern_bits(sweep_leaves, smallest)}-k{smallest}",
            overrides=small
            | {f"{PRIVATE_PREFIX}_APPEND_LEAVES": PRIVATE_SMOKE_APPEND_LEAVES},
        )
        add(
            f"private-single-undine-log{first_gib}gib"
            f"-p{private_pattern_bits(leaves_of_gib(first_gib), 1)}",
            overrides=small,
        )
        add(f"private-table-undine-log{first_gib}gib-k{PRIVATE_KAPPA}", overrides=small)
        add(f"private-client-undine-k{smallest}", overrides=small)
        claim, color = TREEPIR_SERVERS[0]
        add(
            f"treepir-server-h{TREEPIR_HEIGHT}-m{claim}-c{color}",
            overrides={
                f"{TREEPIR_PREFIX}_HEIGHT": TREEPIR_SMOKE_HEIGHT,
                f"{TREEPIR_PREFIX}_SEEDED_LEAVES": TREEPIR_SMOKE_SEEDED_LEAVES,
            },
        )
        for height, replies in (
            (TREEPIR_HEIGHT, "generated"),
        ):
            add(
                f"treepir-client-h{height}-{replies}",
                overrides={
                    f"{TREEPIR_PREFIX}_HEIGHT": TREEPIR_SMOKE_HEIGHT,
                    f"{TREEPIR_PREFIX}_SEEDED_LEAVES": TREEPIR_SMOKE_SEEDED_LEAVES,
                },
            )
    else:
        first_leaves = leaves_of_gib(first_gib)
        add(
            f"private-maintainer-undine-log{first_gib}gib"
            f"-p{private_pattern_bits(first_leaves, PRIVATE_KAPPA)}-k{PRIVATE_KAPPA}"
        )
        add(f"private-single-undine-log{first_gib}gib-p{private_pattern_bits(first_leaves, 1)}")
        add(f"private-table-undine-log{first_gib}gib-k{PRIVATE_KAPPA}")
        add(f"private-client-undine-k{PRIVATE_KAPPA}")
        claim, color = TREEPIR_SERVERS[0]
        add(f"treepir-server-h{TREEPIR_HEIGHT}-m{claim}-c{color}")
        add(f"treepir-client-h{TREEPIR_HEIGHT}-generated")
    return selected


# The modes that sample adaptively to the precision target: the full run,
# and a run of selected jobs, which measures them the same way but is never
# publishable on its own.
ADAPTIVE_MODES = frozenset({"full", "jobs"})


def run_id_for(mode: str, snapshot_id: str) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{mode}-{snapshot_id[:10]}"


def checkpoint_path(run_dir: pathlib.Path) -> pathlib.Path:
    return run_dir / "checkpoint.json"


def write_checkpoint(run_dir: pathlib.Path, state: dict[str, Any]) -> None:
    state["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    atomic_write_json(checkpoint_path(run_dir), state)


def mark_run_completed(run_dir: pathlib.Path, state: dict[str, Any]) -> None:
    state["status"] = "completed"
    state["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_checkpoint(run_dir, state)


def load_checkpoint(run_dir: pathlib.Path) -> dict[str, Any]:
    path = checkpoint_path(run_dir)
    if not path.exists():
        raise PipelineError(f"missing checkpoint: {path}")
    return json.loads(path.read_text())


def validate_resume_snapshot(
    state: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> None:
    if state.get("snapshot_id") != snapshot.get("snapshot_id"):
        raise PipelineError(
            "resume rejected: source/configuration snapshot changed "
            f"({state.get('snapshot_id')} != {snapshot.get('snapshot_id')})"
        )
    if state.get("pipeline_version") != PIPELINE_VERSION:
        raise PipelineError("resume rejected: pipeline version changed")


def initialize_run(
    manifest: Manifest,
    mode: str,
    selected_jobs: Sequence[RunJob],
    snapshot: Mapping[str, Any],
    benchmark_cpu: int | None,
) -> tuple[pathlib.Path, dict[str, Any]]:
    run_id = run_id_for(mode, str(snapshot["snapshot_id"]))
    run_dir = RUNS_ROOT / run_id
    suffix = 0
    while run_dir.exists():
        suffix += 1
        run_dir = RUNS_ROOT / f"{run_id}-{suffix}"
    run_dir.mkdir(parents=True)
    (run_dir / "provenance").mkdir()
    atomic_write_json(run_dir / "manifest.json", manifest.as_dict())
    atomic_write_json(run_dir / "provenance" / "source-snapshot.json", snapshot)
    # Hashes and a tracked diff alone cannot restore newly added source files.
    # Retain the exact measured sources and tests, including untracked additions.
    for relative, record in snapshot.get("files", {}).items():
        destination = run_dir / "provenance" / "source-tree" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
        if sha256_file(destination) != record["sha256"]:
            raise PipelineError(f"source changed while archiving {relative}")
    atomic_write_json(
        run_dir / "provenance" / "environment.json",
        collect_environment(benchmark_cpu),
    )
    git_status = run_capture(
        ["git", "status", "--short", "--untracked-files=all"]
    ).stdout.decode()
    git_diff = run_capture(["git", "diff", "--binary", "HEAD"]).stdout
    atomic_write_text(run_dir / "provenance" / "git-status.txt", git_status)
    atomic_write_bytes(run_dir / "provenance" / "git-diff.patch", git_diff)
    state: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "run_id": run_dir.name,
        "mode": mode,
        "status": "running",
        "snapshot_id": snapshot["snapshot_id"],
        "benchmark_cpu": benchmark_cpu,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selected_jobs": [run_job.spec.id for run_job in selected_jobs],
        "run_jobs": {
            run_job.spec.id: {
                "measurements": [item.id for item in run_job.measurements],
                "selectors": {
                    item.id: {
                        "selector": list(item.selector),
                        "path": list(item.path),
                        "kind": item.kind,
                        "unit": item.unit,
                        "sample_mode": item.sample_mode,
                        "sample_group": item.sample_group,
                    }
                    for item in run_job.measurements
                },
                "env_overrides": list(run_job.env_overrides),
                "features": list(run_job.effective_features()),
            }
            for run_job in selected_jobs
        },
        "binaries": {},
        "jobs": {
            run_job.spec.id: {
                "status": "pending",
                "batches": [],
                "samples": {
                    item.id: [] for item in run_job.measurements if item.kind == "timing"
                },
                "values": {
                    item.id: [] for item in run_job.measurements if item.kind == "value"
                },
                "seeds": [],
            }
            for run_job in selected_jobs
        },
        "interruption_requested": False,
        "resumed": False,
    }
    write_checkpoint(run_dir, state)
    return run_dir, state


def build_key(run_job: RunJob) -> str:
    return sha256_bytes(
        canonical_json(
            {
                "layout": run_job.spec.layout,
                "benchmark": run_job.spec.benchmark,
                "features": run_job.effective_features(),
            }
        )
    )[:16]


def isolated_build_target(run_dir: pathlib.Path, snapshot_id: str, layout: str) -> pathlib.Path:
    """A run may reuse only compiler outputs from its own frozen sources."""
    target_dir = run_dir.resolve() / "build-targets" / layout
    identity = {"snapshot_id": snapshot_id, "layout": layout}
    marker = target_dir / "artifact-build-identity.json"
    if target_dir.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != identity:
            raise PipelineError(f"unrecognized benchmark build cache: {target_dir}")
    else:
        target_dir.mkdir(parents=True)
        atomic_write_json(marker, identity)
    return target_dir


def build_benchmark_binary(
    run_dir: pathlib.Path,
    state: dict[str, Any],
    run_job: RunJob,
    *,
    timeout_seconds: float,
) -> pathlib.Path:
    key = build_key(run_job)
    existing = state["binaries"].get(key)
    if existing is not None:
        binary = run_dir / existing["path"]
        if not binary.exists():
            raise PipelineError(f"checkpointed binary is missing: {binary}")
        actual = sha256_file(binary)
        if actual != existing["sha256"]:
            raise PipelineError(f"checkpointed binary hash changed: {binary}")
        if existing.get("source_snapshot_id") != state["snapshot_id"]:
            raise PipelineError(f"checkpointed binary has no matching source build: {binary}")
        return binary
    validate_current_source_snapshot(
        str(state["snapshot_id"]), phase=f"before building {run_job.spec.benchmark}"
    )
    # Never consult a crate's shared target cache. Cargo freshness can otherwise
    # reuse a diagnostic executable after source restoration with preserved mtimes.
    # Only builds of this run's frozen sources may share dependencies; the marker
    # also protects resumed builds from an unrelated target directory.
    target_dir = isolated_build_target(run_dir, str(state["snapshot_id"]), run_job.spec.layout)
    environment = clean_benchmark_environment()
    environment.update({"RUSTC_WRAPPER": "", "RUSTC_WORKSPACE_WRAPPER": ""})
    features = ",".join(run_job.effective_features())
    command = [
        "cargo",
        "bench",
        "--locked",
        "--no-run",
        "--message-format=json-render-diagnostics",
        "--target-dir",
        str(target_dir),
        "--features",
        features,
        "--bench",
        run_job.spec.benchmark,
    ]
    build_dir = run_dir / "builds" / key
    build_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    completed = run_capture(
        command,
        cwd=CRATE_ROOTS[run_job.spec.layout],
        env=environment,
        timeout=timeout_seconds,
        check=False,
    )
    atomic_write_bytes(build_dir / "stdout.jsonl", completed.stdout)
    atomic_write_bytes(build_dir / "stderr.log", completed.stderr)
    if completed.returncode != 0:
        raise PipelineError(
            f"failed to build {run_job.spec.benchmark}; see {build_dir}"
        )
    executables: list[pathlib.Path] = []
    freshness: dict[pathlib.Path, bool] = {}
    for line in completed.stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("reason") != "compiler-artifact":
            continue
        target = message.get("target", {})
        if target.get("name") != run_job.spec.benchmark:
            continue
        if "bench" not in target.get("kind", []):
            continue
        executable = message.get("executable")
        if executable:
            path = pathlib.Path(executable)
            executables.append(path)
            freshness[path] = bool(message.get("fresh"))
    unique = sorted(set(executables))
    if len(unique) != 1 or not unique[0].exists():
        raise PipelineError(
            f"expected one executable for {run_job.spec.benchmark}, found {unique}"
        )
    if not unique[0].resolve().is_relative_to(target_dir):
        raise PipelineError(f"Cargo returned a binary outside the isolated build: {unique[0]}")
    destination = run_dir / "binaries" / key / run_job.spec.benchmark
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(unique[0], destination)
    destination.chmod(destination.stat().st_mode | 0o100)
    record = {
        "layout": run_job.spec.layout,
        "benchmark": run_job.spec.benchmark,
        "features": list(run_job.effective_features()),
        "path": str(destination.relative_to(run_dir)),
        "sha256": sha256_file(destination),
        "source_snapshot_id": state["snapshot_id"],
        "isolated_target_dir": str(target_dir.relative_to(run_dir.resolve())),
        "compiler_wrappers": "disabled",
        "cargo_fresh": freshness[unique[0]],
        "source_path": str(unique[0]),
        "command": command,
        "build_seconds": time.time() - started,
    }
    validate_current_source_snapshot(
        str(state["snapshot_id"]),
        phase=f"building benchmark {run_job.spec.benchmark}",
    )
    state["binaries"][key] = record
    write_checkpoint(run_dir, state)
    return destination


def clean_benchmark_environment() -> dict[str, str]:
    """Only manifest-declared BENCH_* settings may affect a measured process."""
    tracing = {"YPIR_PACK_PROFILE", "AUTH_WRITE_TRACE", "NET_DEMO_TRACE"}
    return {key: value for key, value in os.environ.items()
            if not key.startswith("BENCH_") and key not in tracing}


def validate_timing_policy(run_job: RunJob, document: Mapping[str, Any]) -> None:
    """Check measurement policy as well as workload dimensions in new runs."""
    meta = document.get("meta", {})
    benchmark = run_job.spec.benchmark
    expected: dict[str, int] = {}
    if benchmark == PRIVATE_BENCHMARK:
        expected["first_pass_warmup_calls"] = int(meta.get("role") != "client")
    elif benchmark == TREEPIR_BENCHMARK:
        expected["first_pass_warmup_calls"] = int(meta.get("role") == "server")
        expected["client_openings_per_sample"] = 8
    elif benchmark == "bench_append_only_log_on_the_fly_open_local":
        expected["point_calls_per_sample"] = 32
    for key, value in expected.items():
        if meta.get(key) != value:
            raise PipelineError(f"{run_job.spec.id}: timing policy mismatch for {key}")
    if benchmark == TREEPIR_BENCHMARK and meta.get("role") == "client":
        for entry in document.get("entries", []):
            if entry.get("role") != "client":
                continue
            for path, metric in metric_stats_paths(entry):
                calls = metric.get("raw_call_samples")
                if metric.get("inner_repetitions") != 8 or not isinstance(calls, list):
                    raise PipelineError(f"{run_job.spec.id}: missing per-opening timings for {path}")
                if len(calls) != 8 * len(metric["samples"]):
                    raise PipelineError(f"{run_job.spec.id}: incomplete opening batch for {path}")
                if any(not isinstance(value, int) or value <= 0 for value in calls):
                    raise PipelineError(f"{run_job.spec.id}: invalid per-opening timings for {path}")
                means = [sum(calls[offset:offset + 8]) // 8 for offset in range(0, len(calls), 8)]
                if means != metric["samples"]:
                    raise PipelineError(f"{run_job.spec.id}: opening batch means disagree for {path}")
            if "client_total_ns" in entry:
                request, process, total = (
                    entry[field]["raw_call_samples"]
                    for field in ("request_total_ns", "process_total_ns", "client_total_ns")
                )
                if not len(request) == len(process) == len(total) or total != [
                    left + right for left, right in zip(request, process)
                ]:
                    raise PipelineError(f"{run_job.spec.id}: per-opening client total differs from request plus processing")


def find_entry(document: Mapping[str, Any], measurement: MeasurementSpec) -> tuple[int, Mapping[str, Any]]:
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise PipelineError("benchmark output has no entries list")
    expected = measurement.selector_dict()
    matches = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        if all(entry.get(key) == value for key, value in expected.items()):
            matches.append((index, entry))
    if len(matches) != 1:
        raise PipelineError(
            f"{measurement.id}: expected one entry for {expected}, found {len(matches)}"
        )
    return matches[0]


def deep_get(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for component in path:
        if not isinstance(current, dict) or component not in current:
            raise PipelineError(f"missing JSON path {'.'.join(path)}")
        current = current[component]
    return current


def metric_stats_paths(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Mapping[str, Any]]]:
    if isinstance(value, dict):
        required = {"mean", "p50", "p95", "trials", "samples"}
        if required.issubset(value):
            yield path, value
            return
        for key, child in value.items():
            yield from metric_stats_paths(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from metric_stats_paths(child, path + (str(index),))


def requested_samples(document: Mapping[str, Any], measurement: MeasurementSpec) -> int:
    meta = document.get("meta", {})
    groups = meta.get("sample_groups")
    return int(meta.get("trials", -1) if groups is None else groups.get(measurement.sample_group, 0))


def validate_no_unconsumed_timings(
    document: Mapping[str, Any], measurements: Sequence[MeasurementSpec]
) -> None:
    expected: set[tuple[int, tuple[str, ...]]] = set()
    for measurement in measurements:
        if measurement.kind != "timing" or requested_samples(document, measurement) == 0:
            continue
        index, _ = find_entry(document, measurement)
        expected.add((index, measurement.path))
    entries = document.get("entries", [])
    actual: set[tuple[int, tuple[str, ...]]] = set()
    for index, entry in enumerate(entries):
        for path, stats in metric_stats_paths(entry):
            trials = int(stats.get("trials", 0))
            samples = stats.get("samples", [])
            if trials > 0 or samples:
                actual.add((index, path))
    extra = actual - expected
    missing = expected - actual
    if extra or missing:
        raise PipelineError(
            "timing-consumer validation failed: "
            f"unconsumed={sorted(extra)}, missing={sorted(missing)}"
        )


def extract_measurements(
    document: Mapping[str, Any],
    measurements: Sequence[MeasurementSpec],
    *,
    warmup_repetitions: int = 0,
    group_warmups: Mapping[str, int] | None = None,
) -> tuple[
    dict[str, list[float]],
    dict[str, Any],
    int | None,
    dict[str, list[float]],
]:
    validate_no_unconsumed_timings(document, measurements)
    timing: dict[str, list[float]] = {}
    raw_timing: dict[str, list[float]] = {}
    values: dict[str, Any] = {}
    meta = document.get("meta")
    if not isinstance(meta, dict):
        raise PipelineError("benchmark output has no meta object")
    declared_run_repetitions = int(meta.get("trials", -1))
    if warmup_repetitions < 0 or warmup_repetitions >= declared_run_repetitions:
        raise PipelineError(
            f"invalid warmup count {warmup_repetitions} for "
            f"{declared_run_repetitions} benchmark repetitions"
        )
    seed = meta.get("seed")
    for measurement in measurements:
        _, entry = find_entry(document, measurement)
        raw = deep_get(entry, measurement.path)
        if measurement.kind == "value":
            if isinstance(raw, (dict, list)):
                raise PipelineError(f"{measurement.id}: expected scalar value")
            validate_measurement_value(measurement, raw)
            values[measurement.id] = raw
            continue
        if measurement.kind != "timing":
            raise PipelineError(f"unexpected run measurement kind: {measurement.kind}")
        expected_trials = requested_samples(document, measurement)
        if expected_trials == 0:
            if not isinstance(raw, dict) or raw.get("trials") != 0 or raw.get("samples") != []:
                raise PipelineError(f"{measurement.id}: unrequested group produced samples")
            continue
        discard = (group_warmups or {}).get(measurement.sample_group, warmup_repetitions)
        if not 0 <= discard < expected_trials:
            raise PipelineError(f"{measurement.id}: invalid group warmup count")
        if not isinstance(raw, dict):
            raise PipelineError(f"{measurement.id}: expected MetricStats object")
        samples = raw.get("samples")
        if not isinstance(samples, list) or not samples:
            raise PipelineError(f"{measurement.id}: missing raw timing samples")
        inner = raw.get("inner_repetitions", 1)
        if type(inner) is not int or inner < 1:
            raise PipelineError(f"{measurement.id}: invalid inner repetition count")
        numeric = [float(value) for value in samples]
        for value in numeric:
            validate_measurement_value(measurement, value)
        raw_timing[measurement.id] = numeric
        declared_trials = int(raw.get("trials", -1))
        if declared_trials != len(numeric):
            raise PipelineError(
                f"{measurement.id}: trials={declared_trials}, samples={len(numeric)}"
            )
        if measurement.sample_mode in {"direct", "rebuild"}:
            if len(numeric) != expected_trials:
                raise PipelineError(
                    f"{measurement.id}: expected {expected_trials} direct samples, "
                    f"found {len(numeric)}"
                )
            timing[measurement.id] = numeric[discard:]
        elif measurement.sample_mode == "append-sequence":
            repetitions = declared_run_repetitions
            blocks = int(meta.get("sequence_blocks", -1))
            warmup = int(meta.get("sequence_warmup_blocks", -1))
            sampled_blocks = blocks - warmup
            if repetitions <= 0 or sampled_blocks <= 0:
                raise PipelineError(f"{measurement.id}: invalid append sequence meta")
            if len(numeric) != repetitions * sampled_blocks:
                raise PipelineError(
                    f"{measurement.id}: expected {repetitions * sampled_blocks} block "
                    f"samples, found {len(numeric)}"
                )
            sequence_means = [
                statistics.fmean(
                    numeric[offset : offset + sampled_blocks]
                )
                for offset in range(0, len(numeric), sampled_blocks)
            ]
            timing[measurement.id] = sequence_means[discard:]
        else:
            raise PipelineError(
                f"{measurement.id}: unknown sample mode {measurement.sample_mode}"
            )
    return timing, values, None if seed is None else int(seed), raw_timing


def validate_output_contract(
    run_job: RunJob, document: Mapping[str, Any], repetitions: int,
    expected_sample_groups: Mapping[str, int] | None = None,
    *, warmup_repetitions: int = 0, group_warmups: Mapping[str, int] | None = None,
) -> None:
    meta = document.get("meta")
    entries = document.get("entries")
    if not isinstance(meta, dict) or not isinstance(entries, list):
        raise PipelineError(f"{run_job.spec.id}: malformed benchmark result")
    if int(meta.get("trials", -1)) != repetitions:
        raise PipelineError(
            f"{run_job.spec.id}: output trials do not match invocation"
        )
    if meta.get("sample_groups") != expected_sample_groups:
        raise PipelineError(f"{run_job.spec.id}: sample groups differ from the invocation")
    if "seed" not in meta:
        raise PipelineError(f"{run_job.spec.id}: benchmark output omits its seed")
    env = run_job.effective_env()
    benchmark = run_job.spec.benchmark
    layout = run_job.spec.layout
    if benchmark == "bench_append_only_log_bulk_updates_append_local":
        expected_blocks = int(
            env["BENCH_APPEND_ONLY_LOG_BULK_UPDATES_APPEND_SEQUENCE_BLOCKS"]
        )
        expected_warmup = int(
            env[
                "BENCH_APPEND_ONLY_LOG_BULK_UPDATES_APPEND_SEQUENCE_WARMUP_BLOCKS"
            ]
        )
        if (
            meta.get("sequence_blocks") != expected_blocks
            or meta.get("sequence_warmup_blocks") != expected_warmup
        ):
            raise PipelineError(f"{run_job.spec.id}: append sequence meta mismatch")
        if not bool(meta.get("skip_generic")):
            raise PipelineError(f"{run_job.spec.id}: generic append path was enabled")
        if set(meta.get("metrics", [])) != {"prepared_with_slack"}:
            raise PipelineError(f"{run_job.spec.id}: append metric filter mismatch")
    elif benchmark == "bench_append_only_log_point_open_local":
        if set(meta.get("metrics", [])) != {"serialized"}:
            raise PipelineError(f"{run_job.spec.id}: point-helper metric filter mismatch")
    elif benchmark == "bench_append_only_log_range_open_local":
        expected = set(env["BENCH_APPEND_ONLY_LOG_RANGE_OPEN_METRICS"].split(","))
        if set(meta.get("metrics", [])) != expected:
            raise PipelineError(f"{run_job.spec.id}: range metric filter mismatch")
    elif benchmark == "bench_append_only_log_repair_kernels_local":
        expected = set(env["BENCH_APPEND_ONLY_LOG_REPAIR_METRICS"].split(","))
        if set(meta.get("metrics", [])) != expected:
            raise PipelineError(f"{run_job.spec.id}: repair metric filter mismatch")
    elif benchmark == "bench_append_only_log_one_server_kappa_rescale_local":
        expected = set(
            env["BENCH_APPEND_ONLY_LOG_KAPPA_RESCALE_METRICS"].split(",")
        )
        if set(meta.get("metrics", [])) != expected:
            raise PipelineError(
                f"{run_job.spec.id}: reconfiguration metric filter mismatch"
            )
    elif benchmark == "bench_append_only_log_on_the_fly_open_local":
        prefix = "BENCH_APPEND_ONLY_LOG_ON_THE_FLY_OPEN"
        if meta.get("stored_kappa") != int(env[f"{prefix}_KAPPA"]) or meta.get(
            "max_levels"
        ) != int(env[f"{prefix}_MAX_LEVELS"]):
            raise PipelineError(f"{run_job.spec.id}: on-the-fly parameter echo mismatch")
    elif benchmark == PRIVATE_BENCHMARK:
        role = env[f"{PRIVATE_PREFIX}_ROLE"]
        echoes: dict[str, Any] = {
            "role": role,
            "leaves": int(env[f"{PRIVATE_PREFIX}_LEAVES"]),
            "tree_height": int(env[f"{PRIVATE_PREFIX}_TREE_HEIGHT"]),
        }
        # Archived runs used sampled timing; new runs use samples only for
        # untimed update checks and must contain full rebuild timings below.
        check_key = (
            "HINT_CHECK_COLUMNS" if f"{PRIVATE_PREFIX}_HINT_CHECK_COLUMNS" in env
            else "HINT_SAMPLE_COLUMNS"
        )
        echoes[check_key.lower()] = int(env[f"{PRIVATE_PREFIX}_{check_key}"])
        if role == "single":
            echoes["cells_per_record"] = int(env[f"{PRIVATE_PREFIX}_CELLS_PER_RECORD"])
        else:
            echoes["kappa"] = int(env[f"{PRIVATE_PREFIX}_KAPPA"])
            echoes["segments_per_record"] = int(
                env[f"{PRIVATE_PREFIX}_SEGMENTS_PER_RECORD"]
            )
        if role == "maintainer":
            echoes["append_leaves"] = int(env[f"{PRIVATE_PREFIX}_APPEND_LEAVES"])
        for key, expected_value in echoes.items():
            if meta.get(key) != expected_value:
                raise PipelineError(
                    f"{run_job.spec.id}: private-opening parameter echo mismatch "
                    f"for {key}: {meta.get(key)!r} != {expected_value!r}"
                )
    elif benchmark == TREEPIR_BENCHMARK:
        role = env[f"{TREEPIR_PREFIX}_ROLE"]
        height = int(env[f"{TREEPIR_PREFIX}_HEIGHT"])
        echoes = {
            "role": role,
            "height": height,
            "kappa": height,
            "leaves": 1 << height,
            "seeded_leaves": min(
                int(env[f"{TREEPIR_PREFIX}_SEEDED_LEAVES"]), 1 << height
            ),
        }
        if role == "server":
            echoes["maintainer"] = int(env[f"{TREEPIR_PREFIX}_MAINTAINER"])
            echoes["color"] = int(env[f"{TREEPIR_PREFIX}_COLOR"])
        else:
            echoes["replies"] = env[f"{TREEPIR_PREFIX}_REPLIES"]
        for key, expected_value in echoes.items():
            if meta.get(key) != expected_value:
                raise PipelineError(
                    f"{run_job.spec.id}: TreePIR-comparison parameter echo "
                    f"mismatch for {key}: {meta.get(key)!r} != {expected_value!r}"
                )

    for measurement in run_job.measurements:
        find_entry(document, measurement)

    # Independently verify deterministic values emitted by the benchmark.
    if benchmark == "bench_append_only_log_repair_kernels_local":
        entry = entries[0]
        segment_len = int(entry["segment_len"])
        kappa = int(entry["need"])
        expected_segments = ceil_div(
            cursor_preorder(
                TREE_HEIGHT,
                int(entry["committed_bulk_update_batches"])
                * int(entry["bulk_updates_size"]),
            ),
            segment_len,
        )
        if int(entry["segment_count"]) != expected_segments:
            raise PipelineError(
                f"{run_job.spec.id}: active segment count fails preorder formula"
            )
        expected_download = (
            expected_segments
            * stored_elements_per_segment(layout, segment_len, kappa)
            * FIELD_BYTES
        )
        if int(entry["recover_helper_reply_bytes_modeled"]) != expected_download:
            raise PipelineError(
                f"{run_job.spec.id}: repair download fails storage formula"
            )
    elif benchmark == "bench_append_only_log_one_server_kappa_rescale_local":
        entry = entries[0]
        segments = int(entry["segment_count"])
        if layout == "borg":
            up_expected = (
                segments
                * int(entry["helper_target_count"])
                * int(entry["target_row_count"])
                * FIELD_BYTES
            )
            if int(entry["helper_eval_raw_bytes_total"]) != up_expected:
                raise PipelineError(
                    f"{run_job.spec.id}: kappa-double download fails formula"
                )
            down_expected = (
                segments
                * (
                    int(entry["kappa_halve_source_s"])
                    + (int(entry["kappa_halve_helper_count"]) - 1)
                    * int(entry["kappa_halve_target_row_count"])
                )
                * FIELD_BYTES
            )
            if int(entry["kappa_halve_helper_eval_raw_bytes_total"]) != down_expected:
                raise PipelineError(
                    f"{run_job.spec.id}: kappa-halve download fails formula"
                )
            if int(entry["segment_double_helper_eval_raw_bytes_total"]) != 0:
                raise PipelineError(
                    f"{run_job.spec.id}: local segment-double emitted helper payload"
                )
        else:
            kappa = int(entry["source_need"])
            stitched = ceil_div(segments, 4)
            if int(entry["target_segment_count"]) != stitched:
                raise PipelineError(
                    f"{run_job.spec.id}: stitched segment count is not ceil(segments / 4)"
                )
            if int(entry["kappa_double_helper_eval_raw_bytes_total"]) != 0:
                raise PipelineError(
                    f"{run_job.spec.id}: the local stitch emitted helper payload"
                )
            # Two field elements per helper claim and stitched segment.
            down_expected = (
                stitched * int(entry["kappa_halve_helper_count"]) * 2 * FIELD_BYTES
            )
            if int(entry["kappa_halve_helper_eval_raw_bytes_total"]) != down_expected:
                raise PipelineError(
                    f"{run_job.spec.id}: kappa-halve download fails formula"
                )
            # Source state of two diagonal segments: 4 kappa field elements.
            diagonal_expected = stitched * 4 * kappa * FIELD_BYTES
            if int(entry["kappa_halve_diagonal_source_raw_bytes_total"]) != diagonal_expected:
                raise PipelineError(
                    f"{run_job.spec.id}: diagonal source state fails formula"
                )
            if int(entry["receiver_source_state_bytes"]) != segments * 2 * kappa * FIELD_BYTES:
                raise PipelineError(
                    f"{run_job.spec.id}: source state fails storage formula"
                )
            if int(entry["receiver_target_state_bytes"]) != stitched * 4 * kappa * FIELD_BYTES:
                raise PipelineError(
                    f"{run_job.spec.id}: stitched state fails storage formula"
                )
    elif benchmark == "bench_append_only_log_range_open_local":
        for entry in entries:
            raw_bytes = int(entry["helper_response_field_elements"]) * FIELD_BYTES
            if int(entry["helper_response_serialized_bytes"]) < raw_bytes:
                raise PipelineError(
                    f"{run_job.spec.id}: serialized range response is smaller "
                    "than its field-element payload"
                )
    elif benchmark == PRIVATE_BENCHMARK:
        for entry in entries:
            if (
                entry["role"] in {"maintainer", "single"}
                and f"{PRIVATE_PREFIX}_HINT_CHECK_COLUMNS" in env
            ):
                validate_private_preprocessing(entry, (meta.get("sample_groups") or {}).get("preprocess", meta["trials"]), run_job.spec.id)
            leaves = int(entry["leaves"])
            if int(entry["log_bytes"]) != leaves * FIELD_BYTES:
                raise PipelineError(f"{run_job.spec.id}: log bytes fail 32 per leaf")
            if entry["role"] in {"single", "single-shape"}:
                cells_per_record = int(entry["cells_per_record"])
            else:
                kappa = int(entry["kappa"])
                cells_per_record = int(entry["segments_per_record"]) * kappa * kappa
                if int(entry["record_bytes"]) != cells_per_record // kappa * FIELD_BYTES:
                    raise PipelineError(
                        f"{run_job.spec.id}: record bytes fail kappa entries per segment"
                    )
            expected_records = ceil_div(
                cursor_preorder(TREE_HEIGHT, leaves), cells_per_record
            )
            if int(entry["records"]) != expected_records:
                raise PipelineError(
                    f"{run_job.spec.id}: record count fails the preorder formula"
                )
            if int(entry["database_bytes"]) != expected_records * int(entry["record_bytes"]):
                raise PipelineError(
                    f"{run_job.spec.id}: database bytes fail records times record bytes"
                )
            pattern_bits = int(entry["pattern_bits"])
            span = ceil_div((1 << (pattern_bits + 1)) - 1, cells_per_record)
            if (
                int(entry["patterns"]) != ceil_div(leaves, 1 << pattern_bits)
                or int(entry["group_slots"]) != span + 1
                or int(entry["slots"])
                != span + 1 + max(0, (leaves - 1).bit_length() - pattern_bits)
            ):
                raise PipelineError(
                    f"{run_job.spec.id}: patterns and slots fail the pattern size"
                )
            if entry["role"] != "single-shape" and pattern_bits != int(
                env[f"{PRIVATE_PREFIX}_PATTERN_BITS"]
            ):
                raise PipelineError(f"{run_job.spec.id}: pattern size echo mismatch")
            if not (
                int(entry["patterns"]) <= int(entry["db_rows"])
                and int(entry["record_columns"]) <= int(entry["db_cols"])
                and int(entry["outputs"]) * 2048 == int(entry["db_cols"])
                and int(entry["request_bytes"]) > 7 * int(entry["db_rows"])
            ):
                raise PipelineError(
                    f"{run_job.spec.id}: database dimensions are inconsistent"
                )
    elif benchmark == TREEPIR_BENCHMARK:
        validate_treepir_entries(run_job, env, entries,
                                 paired_warmups=(group_warmups or {}).get("paired", warmup_repetitions))
    elif benchmark == "bench_append_only_log_on_the_fly_open_local":
        for entry in entries:
            if entry.get("query") == "stitch":
                continue
            raw_bytes = int(entry["response_field_elements"]) * FIELD_BYTES
            if int(entry["serialized_response_bytes"]) < raw_bytes:
                raise PipelineError(
                    f"{run_job.spec.id}: serialized answer is smaller than its "
                    "field-element payload"
                )


def validate_private_preprocessing(
    entry: Mapping[str, Any], trials: int, job_id: str
) -> None:
    """Require complete-build samples and their paired sum, never scaled samples."""
    samples = {}
    for field in ("hint_ns", "packing_ns", "preprocess_ns"):
        metric = entry.get(field)
        if not isinstance(metric, dict):
            raise PipelineError(f"{job_id}: missing full preprocessing metric {field}")
        values = metric.get("samples")
        if (
            not isinstance(values, list)
            or len(values) != trials
            or metric.get("trials") != trials
            or any(type(value) is not int or value <= 0 for value in values)
        ):
            raise PipelineError(f"{job_id}: incomplete full preprocessing samples for {field}")
        samples[field] = values
    if samples["preprocess_ns"] != [
        hint + packing for hint, packing in zip(samples["hint_ns"], samples["packing_ns"])
    ]:
        raise PipelineError(f"{job_id}: preprocessing total is not hint plus packing per rebuild")
    expected_bytes = int(entry["outputs"]) * (2048 * 2048 * 8 + 100_859_904)
    if entry.get("preprocessed_bytes") != expected_bytes:
        raise PipelineError(f"{job_id}: preprocessing bytes fail the full hint and packing size")


def validate_treepir_entries(
    run_job: RunJob, env: Mapping[str, str], entries: Sequence[Mapping[str, Any]],
    *, paired_warmups: int = 0,
) -> None:
    """Independently derive the sizes and counts of the TreePIR comparison,
    and hold the generated replies to the real ones."""
    job_id = run_job.spec.id
    height = int(env[f"{TREEPIR_PREFIX}_HEIGHT"])
    leaves = 1 << height
    by_role: dict[tuple[str, str], Mapping[str, Any]] = {}
    for entry in entries:
        key = (str(entry["role"]), str(entry["system"]))
        if key in by_role:
            raise PipelineError(f"{job_id}: two entries for {key}")
        by_role[key] = entry
    role = env[f"{TREEPIR_PREFIX}_ROLE"]
    expected_keys = {(role, "undine"), (role, "treepir")}
    if role == "client" and env[f"{TREEPIR_PREFIX}_REPLIES"] == "generated":
        expected_keys |= {("paired", "undine"), ("paired", "treepir")}
    if set(by_role) != expected_keys:
        raise PipelineError(f"{job_id}: entries {sorted(by_role)} != {sorted(expected_keys)}")
    undine = by_role[(role, "undine")]
    treepir = by_role[(role, "treepir")]
    seeded = min(int(env[f"{TREEPIR_PREFIX}_SEEDED_LEAVES"]), leaves)
    for entry in (undine, treepir):
        if int(entry["leaves"]) != leaves or int(entry["log_bytes"]) != leaves * FIELD_BYTES:
            raise PipelineError(f"{job_id}: log size fails 2^h leaves of 32 bytes")
        if int(entry["seeded_leaves"]) != seeded or not 0 < seeded <= leaves:
            raise PipelineError(f"{job_id}: seeded suffix differs from the configured workload")
        if int(entry["request_bytes"]) <= int(entry["query_bytes"]) + int(entry["key_bytes"]):
            raise PipelineError(f"{job_id}: request smaller than its query and keys")
        if int(entry["query_bytes"]) != 7 * int(entry["db_rows"]):
            raise PipelineError(f"{job_id}: query bytes fail seven per database row")
        if int(entry["outputs"]) * 2048 != int(entry["db_cols"]):
            raise PipelineError(f"{job_id}: ciphertexts fail the padded columns")
    # The matched point: both systems reply with the same ciphertext count.
    if int(undine["outputs"]) != int(treepir["outputs"]):
        raise PipelineError(
            f"{job_id}: reply ciphertexts differ: {undine['outputs']} against "
            f"{treepir['outputs']}"
        )
    if role == "server":
        root = undine["tree_root"]
        if (
            not isinstance(root, list)
            or len(root) != FIELD_BYTES
            or any(type(byte) is not int or not 0 <= byte <= 255 for byte in root)
            or treepir["tree_root"] != root
        ):
            raise PipelineError(f"{job_id}: the servers do not use the same Merkle root")
        segments = ceil_div(cursor_preorder(height, leaves), height * height)
        if int(undine["database_bytes"]) != segments * height * FIELD_BYTES:
            raise PipelineError(f"{job_id}: primary columns fail kappa entries per segment")
        if int(undine["fragment_bytes"]) != 2 * int(undine["database_bytes"]):
            raise PipelineError(f"{job_id}: fragment is not twice the primary columns")
        color = int(env[f"{TREEPIR_PREFIX}_COLOR"])
        if int(treepir["class_nodes"]) != treepir_class_size(height, color):
            raise PipelineError(f"{job_id}: class size fails the balanced split")
        if int(treepir["database_bytes"]) != treepir_class_size(height, color) * FIELD_BYTES:
            raise PipelineError(f"{job_id}: class bytes fail 32 per node")
        for entry in (undine, treepir):
            if int(entry["preprocessed_bytes"]) != int(entry["hint_bytes"]) + int(
                entry["packing_bytes"]
            ):
                raise PipelineError(f"{job_id}: preprocessing fails hint plus packing")
            if int(entry["serving_state_bytes"]) != int(entry["pir_bytes"]) + int(
                entry["preprocessed_bytes"]
            ):
                raise PipelineError(f"{job_id}: state fails PIR plus preprocessing")
            # One hint column of 2,048 words per database column, and one
            # block per 2,048 rows of each column.
            if int(entry["hint_bytes"]) != int(entry["db_cols"]) * 2048 * 8:
                raise PipelineError(f"{job_id}: hint bytes fail one column per database column")
            if int(entry["hint_blocks"]) != int(entry["db_cols"]) * int(entry["db_rows"]) // 2048:
                raise PipelineError(f"{job_id}: hint blocks fail the database dimensions")
            padding = (int(entry["db_cols"]) - int(entry["record_columns"])) * (
                int(entry["db_rows"]) // 2048
            )
            if int(entry["padding_hint_blocks"]) != padding:
                raise PipelineError(f"{job_id}: padding blocks fail the padding columns")
            if not padding <= int(entry["zero_hint_blocks"]) <= int(entry["hint_blocks"]):
                raise PipelineError(f"{job_id}: zero blocks outside padding and all blocks")
        return
    for entry, servers, queries in ((undine, height, 1), (treepir, height, height)):
        if int(entry["servers"]) != servers or int(entry["queries"]) != queries:
            raise PipelineError(f"{job_id}: server or query count")
        if int(entry["upload_bytes"]) != servers * int(entry["request_bytes"]):
            raise PipelineError(f"{job_id}: upload fails one request per server")
        if int(entry["download_bytes"]) != servers * int(entry["reply_bytes"]):
            raise PipelineError(f"{job_id}: download fails one reply per server")
    for system in ("undine", "treepir"):
        paired = by_role.get(("paired", system))
        if paired is None or paired["real_decode_ns"]["trials"] == 0:
            continue
        real = statistics.fmean(paired["real_decode_ns"]["samples"][paired_warmups:])
        generated = statistics.fmean(paired["generated_decode_ns"]["samples"][paired_warmups:])
        if abs(generated - real) > TREEPIR_PAIRED_TOLERANCE * real:
            raise PipelineError(
                f"{job_id}: {system} generated replies decode in {generated:.0f} ns "
                f"on average against {real:.0f} ns for a real server's, beyond "
                f"the tolerance of {TREEPIR_PAIRED_TOLERANCE:.0%}"
            )


def expected_repetitions_for_mode(mode: str) -> int:
    return DEFAULT_MIN_REPETITIONS if mode in ADAPTIVE_MODES else 1


def job_complete(job_state: Mapping[str, Any], mode: str) -> bool:
    timing = job_state.get("samples", {})
    if not timing:
        return False
    if mode not in ADAPTIVE_MODES:
        return all(len(values) >= 1 for values in timing.values())
    if not all(len(values) >= DEFAULT_MIN_REPETITIONS for values in timing.values()):
        return False
    summaries = [summarize_samples(values) for values in timing.values()]
    return all(summary["stable"] or summary["n"] >= DEFAULT_MAX_REPETITIONS for summary in summaries)


def batch_repetitions(job_state: Mapping[str, Any], mode: str) -> int:
    if mode not in ADAPTIVE_MODES:
        return 1
    lengths = [len(values) for values in job_state["samples"].values()]
    current = min(lengths, default=0)
    if current >= DEFAULT_MAX_REPETITIONS:
        return 0
    next_look = next((look for look in SEQUENTIAL_LOOKS if look > current), None)
    if next_look is None:
        return 0
    return next_look - current


def group_batch_plan(run_job: RunJob, job_state: Mapping[str, Any], mode: str) -> tuple[dict[str, int], dict[str, int]]:
    """Retained samples and warmups, independently for each declared group."""
    groups: dict[str, list[MeasurementSpec]] = defaultdict(list)
    for measurement in run_job.measurements:
        if measurement.kind == "timing":
            groups[measurement.sample_group].append(measurement)
    counts, warmups = {}, {}
    for group, measurements in groups.items():
        substate = {"samples": {item.id: job_state["samples"][item.id] for item in measurements}}
        counts[group] = 0 if job_complete(substate, mode) else batch_repetitions(substate, mode)
        warmups[group] = (benchmark_warmup_repetitions(
            dataclasses.replace(run_job, measurements=tuple(measurements)), mode
        ) if counts[group] else 0)
        if (
            counts[group]
            and mode in {"smoke", "canary"}
            and run_job.spec.benchmark == TREEPIR_BENCHMARK
            and group == "paired"
        ):
            # These gates also enforce the 10% agreement between mean real
            # and generated reply timings. One pair cannot support that
            # check: use the full run's first-look sample count and warmups.
            counts[group] = DEFAULT_MIN_REPETITIONS
            warmups[group] = DIRECT_WARMUP_REPETITIONS
    return counts, warmups


def execute_batch(
    run_dir: pathlib.Path,
    state: dict[str, Any],
    run_job: RunJob,
    binary: pathlib.Path,
    *,
    repetitions: int,
    timeout_seconds: float,
) -> None:
    job_state = state["jobs"][run_job.spec.id]
    warmup_repetitions = benchmark_warmup_repetitions(run_job, state["mode"])
    invocation_repetitions = repetitions + warmup_repetitions
    grouped = any(item.sample_group != "all" for item in run_job.measurements if item.kind == "timing")
    group_counts, group_warmups = group_batch_plan(run_job, job_state, state["mode"]) if grouped else ({}, {})
    sample_groups = {group: count + group_warmups[group] for group, count in group_counts.items()} if grouped else None
    if grouped:
        repetitions = max(group_counts.values())
        invocation_repetitions = max(sample_groups.values())
        warmup_repetitions = 0  # Each group has its own discarded prefix.
    batch_index = len(job_state["batches"])
    batch_dir = run_dir / "jobs" / run_job.spec.id / f"batch-{batch_index:03d}"
    work_dir = batch_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    result_path = work_dir / run_job.spec.result_relpath
    environment = clean_benchmark_environment()
    environment["GIT_DIR"] = str(ROOT / ".git")
    environment["GIT_WORK_TREE"] = str(ROOT)
    effective_env = run_job.effective_env()
    # Only immutable, untimed inputs live here. Benchmarks bind each entry
    # to their executable and workload and verify its complete-file checksum.
    effective_env["BENCH_FIXTURE_DIR"] = str(run_dir / "fixtures")
    for key, value in list(effective_env.items()):
        effective_env[key] = value.replace("{RESULT_PATH}", str(result_path))
    effective_env[run_job.spec.trial_env] = str(invocation_repetitions)
    if sample_groups is not None:
        effective_env["BENCH_SAMPLE_GROUPS"] = json.dumps(sample_groups, sort_keys=True)
    environment.update(effective_env)
    command = [str(binary), "--bench"]
    benchmark_cpu = state.get("benchmark_cpu")
    if benchmark_cpu is not None:
        taskset = shutil.which("taskset")
        if taskset is None:
            raise PipelineError("CPU pinning requested but taskset is unavailable")
        command = [taskset, "--cpu-list", str(benchmark_cpu), *command]
    attempt = {
        "index": batch_index,
        "status": "running",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": command,
        "env": dict(sorted(effective_env.items())),
        "binary_sha256": sha256_file(binary),
        "repetitions": repetitions,
        "warmup_repetitions": warmup_repetitions,
        "invocation_repetitions": invocation_repetitions,
        "sample_groups": sample_groups,
        "group_warmups": group_warmups,
        "group_repetitions": group_counts,
    }
    job_state["status"] = "running"
    job_state["batches"].append(attempt)
    write_checkpoint(run_dir, state)
    started = time.time()
    try:
        completed = run_capture(
            command,
            cwd=work_dir,
            env=environment,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        attempt.update(
            {
                "status": "failed",
                "failure": f"timeout after {timeout_seconds} seconds",
                "elapsed_seconds": time.time() - started,
            }
        )
        job_state["status"] = "failed"
        write_checkpoint(run_dir, state)
        raise PipelineError(f"{run_job.spec.id}: benchmark timed out") from exc
    atomic_write_bytes(batch_dir / "stdout.log", completed.stdout)
    atomic_write_bytes(batch_dir / "stderr.log", completed.stderr)
    attempt["elapsed_seconds"] = time.time() - started
    attempt["returncode"] = completed.returncode
    if completed.returncode != 0:
        attempt["status"] = "failed"
        attempt["failure"] = "benchmark process returned nonzero"
        job_state["status"] = "failed"
        write_checkpoint(run_dir, state)
        raise PipelineError(
            f"{run_job.spec.id}: benchmark failed; see {batch_dir / 'stderr.log'}"
        )
    if not result_path.exists():
        attempt["status"] = "failed"
        attempt["failure"] = f"missing result {run_job.spec.result_relpath}"
        job_state["status"] = "failed"
        write_checkpoint(run_dir, state)
        raise PipelineError(f"{run_job.spec.id}: result file was not produced")
    result_bytes = result_path.read_bytes()
    try:
        document = json.loads(result_bytes)
        validate_timing_policy(run_job, document)
        validate_output_contract(run_job, document, invocation_repetitions, sample_groups,
                                 warmup_repetitions=warmup_repetitions, group_warmups=group_warmups)
        timing, values, seed, raw_timing = extract_measurements(
            document,
            run_job.measurements,
            warmup_repetitions=warmup_repetitions,
            group_warmups=group_warmups,
        )
        for measurement_id, value in values.items():
            previous = job_state["values"][measurement_id]
            if previous and any(item != value for item in previous):
                raise PipelineError(
                    f"{measurement_id}: deterministic value changed across batches"
                )
    except (json.JSONDecodeError, PipelineError, TypeError, ValueError) as exc:
        failed_result = batch_dir / "unvalidated-result.json"
        atomic_write_bytes(failed_result, result_bytes)
        attempt.update(
            {
                "status": "failed",
                "failure": f"result validation failed: {exc}",
                "unvalidated_result": str(failed_result.relative_to(run_dir)),
                "unvalidated_result_sha256": sha256_file(failed_result),
            }
        )
        job_state["status"] = "failed"
        write_checkpoint(run_dir, state)
        raise PipelineError(
            f"{run_job.spec.id}: benchmark result failed validation; "
            f"see {failed_result}"
        ) from exc
    for measurement_id, samples in timing.items():
        job_state["samples"][measurement_id].extend(samples)
    for measurement_id, value in values.items():
        job_state["values"][measurement_id].append(value)
    if seed is not None:
        job_state["seeds"].append(seed)
    saved_result = batch_dir / "result.json"
    atomic_write_bytes(saved_result, result_path.read_bytes())
    saved_raw = batch_dir / "raw-samples.json"
    atomic_write_json(
        saved_raw,
        {
            "raw_benchmark_samples": raw_timing,
            "independent_samples": timing,
            "seed": seed,
        },
    )
    attempt.update(
        {
            "status": "completed",
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "result": str(saved_result.relative_to(run_dir)),
            "result_sha256": sha256_file(saved_result),
            "raw_samples": str(saved_raw.relative_to(run_dir)),
            "raw_samples_sha256": sha256_file(saved_raw),
            "seed": seed,
            "sample_counts": {
                key: len(value) for key, value in timing.items()
            },
        }
    )
    job_state["status"] = "completed" if job_complete(job_state, state["mode"]) else "pending"
    write_checkpoint(run_dir, state)


def reconstruct_run_jobs(
    manifest: Manifest, state: Mapping[str, Any]
) -> list[RunJob]:
    result: list[RunJob] = []
    for job_id in state["selected_jobs"]:
        saved = state["run_jobs"][job_id]
        specs: list[MeasurementSpec] = []
        for identifier in saved["measurements"]:
            if identifier in manifest.measurements:
                specs.append(manifest.measurements[identifier])
                continue
            metadata = saved["selectors"][identifier]
            specs.append(
                MeasurementSpec(
                    id=identifier,
                    job_id=job_id,
                    selector=tuple(
                        (key, value) for key, value in metadata["selector"]
                    ),
                    path=tuple(metadata["path"]),
                    kind=metadata["kind"],
                    sample_mode=metadata["sample_mode"],
                    sample_group=metadata.get("sample_group", "all"),
                )
            )
        result.append(
            RunJob(
                manifest.jobs[job_id],
                tuple(specs),
                tuple((key, value) for key, value in saved["env_overrides"]),
                tuple(saved["features"]),
            )
        )
    return result


def _checked_run_artifact(run_dir: pathlib.Path, relative: str) -> pathlib.Path:
    path = (run_dir / relative).resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise PipelineError(
            f"resume rejected: artifact escapes run directory: {relative}"
        ) from exc
    return path


def validate_resume_artifacts(
    run_dir: pathlib.Path,
    state: Mapping[str, Any],
    run_jobs: Sequence[RunJob],
) -> None:
    binaries = state.get("binaries")
    jobs = state.get("jobs")
    if not isinstance(binaries, dict) or not isinstance(jobs, dict):
        raise PipelineError("resume rejected: malformed checkpoint state")

    for key, record in binaries.items():
        if not isinstance(record, dict):
            raise PipelineError(f"resume rejected: malformed binary record {key}")
        path = _checked_run_artifact(run_dir, str(record.get("path", "")))
        if not path.is_file():
            raise PipelineError(f"resume rejected: saved binary is missing: {path}")
        if sha256_file(path) != record.get("sha256"):
            raise PipelineError(f"resume rejected: saved binary hash changed: {path}")
        if state.get("pipeline_version", 1) >= 2 and record.get("source_snapshot_id") != state["snapshot_id"]:
            raise PipelineError(f"resume rejected: binary source snapshot differs: {path}")

    for run_job in run_jobs:
        job_state = jobs.get(run_job.spec.id)
        if not isinstance(job_state, dict):
            raise PipelineError(
                f"resume rejected: missing job state for {run_job.spec.id}"
            )
        expected_samples = {
            measurement.id: []
            for measurement in run_job.measurements
            if measurement.kind == "timing"
        }
        expected_values = {
            measurement.id: []
            for measurement in run_job.measurements
            if measurement.kind == "value"
        }
        expected_seeds: list[int] = []
        for attempt in job_state.get("batches", []):
            if attempt.get("status") != "completed":
                continue
            if any(item.sample_group != "all" for item in run_job.measurements if item.kind == "timing"):
                counts, warmups = group_batch_plan(
                    run_job, {"samples": expected_samples}, state["mode"]
                )
                requested = {group: count + warmups[group] for group, count in counts.items()}
                if (attempt.get("group_repetitions") != counts
                        or attempt.get("group_warmups") != warmups
                        or attempt.get("sample_groups") != requested
                        or attempt.get("invocation_repetitions") != max(requested.values())
                        or attempt.get("warmup_repetitions") != 0):
                    raise PipelineError(
                        f"resume rejected: sample-group schedule changed for {run_job.spec.id}"
                    )
            binary_record = binaries.get(build_key(run_job))
            if not isinstance(binary_record, dict):
                raise PipelineError(
                    f"resume rejected: missing binary for completed job "
                    f"{run_job.spec.id}"
                )
            if attempt.get("binary_sha256") != binary_record.get("sha256"):
                raise PipelineError(
                    f"resume rejected: batch binary does not match saved binary "
                    f"for {run_job.spec.id}"
                )
            result_path = _checked_run_artifact(
                run_dir, str(attempt.get("result", ""))
            )
            raw_path = _checked_run_artifact(
                run_dir, str(attempt.get("raw_samples", ""))
            )
            for path, hash_key in (
                (result_path, "result_sha256"),
                (raw_path, "raw_samples_sha256"),
            ):
                if not path.is_file():
                    raise PipelineError(
                        f"resume rejected: completed artifact is missing: {path}"
                    )
                if sha256_file(path) != attempt.get(hash_key):
                    raise PipelineError(
                        f"resume rejected: completed artifact hash changed: {path}"
                    )

            try:
                document = json.loads(result_path.read_bytes())
                repetitions = int(attempt["repetitions"])
                warmup_repetitions = int(attempt.get("warmup_repetitions", 0))
                invocation_repetitions = int(
                    attempt.get(
                        "invocation_repetitions",
                        repetitions + warmup_repetitions,
                    )
                )
                if state.get("pipeline_version", 1) >= 2:
                    validate_timing_policy(run_job, document)
                validate_output_contract(
                    run_job, document, invocation_repetitions, attempt.get("sample_groups"),
                    warmup_repetitions=warmup_repetitions, group_warmups=attempt.get("group_warmups"),
                )
                timing, values, seed, raw_timing = extract_measurements(
                    document,
                    run_job.measurements,
                    warmup_repetitions=warmup_repetitions,
                    group_warmups=attempt.get("group_warmups"),
                )
                saved_raw = json.loads(raw_path.read_bytes())
            except (json.JSONDecodeError, KeyError, PipelineError, TypeError, ValueError) as exc:
                raise PipelineError(
                    f"resume rejected: saved output no longer validates for "
                    f"{run_job.spec.id}: {exc}"
                ) from exc
            expected_raw = {
                "raw_benchmark_samples": raw_timing,
                "independent_samples": timing,
                "seed": seed,
            }
            if saved_raw != expected_raw:
                raise PipelineError(
                    f"resume rejected: raw samples disagree with result for "
                    f"{run_job.spec.id}"
                )
            for measurement_id, samples in timing.items():
                expected_samples[measurement_id].extend(samples)
            for measurement_id, value in values.items():
                expected_values[measurement_id].append(value)
            if seed is not None:
                expected_seeds.append(seed)

        if job_state.get("samples") != expected_samples:
            raise PipelineError(
                f"resume rejected: checkpoint samples disagree with saved "
                f"artifacts for {run_job.spec.id}"
            )
        if job_state.get("values") != expected_values:
            raise PipelineError(
                f"resume rejected: checkpoint values disagree with saved "
                f"artifacts for {run_job.spec.id}"
            )
        if job_state.get("seeds") != expected_seeds:
            raise PipelineError(
                f"resume rejected: checkpoint seeds disagree with saved "
                f"artifacts for {run_job.spec.id}"
            )


def execute_run(
    manifest: Manifest,
    run_dir: pathlib.Path,
    state: dict[str, Any],
    run_jobs: Sequence[RunJob],
    *,
    timeout_seconds: float,
    stop_after_batches: int | None,
) -> bool:
    completed_batches_this_process = 0
    for run_job in run_jobs:
        job_state = state["jobs"][run_job.spec.id]
        if job_complete(job_state, state["mode"]):
            job_state["status"] = "completed"
            continue
        validate_current_source_snapshot(
            str(state["snapshot_id"]),
            phase=f"starting job {run_job.spec.id}",
        )
        binary = build_benchmark_binary(
            run_dir, state, run_job, timeout_seconds=timeout_seconds
        )
        while not job_complete(job_state, state["mode"]):
            validate_current_source_snapshot(
                str(state["snapshot_id"]),
                phase=f"starting a batch for {run_job.spec.id}",
            )
            if any(item.sample_group != "all" for item in run_job.measurements if item.kind == "timing"):
                counts, _ = group_batch_plan(run_job, job_state, state["mode"])
                repetitions = max(counts.values())
            else:
                repetitions = batch_repetitions(job_state, state["mode"])
            if repetitions <= 0:
                break
            execute_batch(
                run_dir,
                state,
                run_job,
                binary,
                repetitions=repetitions,
                timeout_seconds=timeout_seconds,
            )
            completed_batches_this_process += 1
            if (
                stop_after_batches is not None
                and completed_batches_this_process >= stop_after_batches
            ):
                state["status"] = "interrupted"
                state["interruption_requested"] = True
                write_checkpoint(run_dir, state)
                return False
        validate_current_source_snapshot(
            str(state["snapshot_id"]),
            phase=f"finishing job {run_job.spec.id}",
        )
        job_state["status"] = "completed"
        write_checkpoint(run_dir, state)
    state["status"] = "benchmarks-complete"
    write_checkpoint(run_dir, state)
    return True


def resolve_measurements(
    manifest: Manifest, state: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    resolved: dict[str, dict[str, Any]] = {}
    for measurement in manifest.measurements.values():
        if measurement.kind == "formula":
            resolved[measurement.id] = {
                "kind": "formula",
                "unit": measurement.unit,
                "value": evaluate_formula(measurement),
                "formula": measurement.formula,
                "formula_args": list(measurement.formula_args),
            }
            continue
        if measurement.job_id not in state.get("jobs", {}):
            continue
        job_state = state["jobs"][measurement.job_id]
        if measurement.kind == "timing":
            samples = job_state.get("samples", {}).get(measurement.id)
            if samples:
                resolved[measurement.id] = {
                    "kind": "timing",
                    "unit": measurement.unit,
                    "samples": samples,
                    "summary": summarize_samples(samples),
                }
        elif measurement.kind == "value":
            values = job_state.get("values", {}).get(measurement.id)
            if values:
                if any(value != values[0] for value in values[1:]):
                    raise PipelineError(
                        f"{measurement.id}: deterministic output was not stable"
                    )
                resolved[measurement.id] = {
                    "kind": "value",
                    "unit": measurement.unit,
                    "value": values[0],
                    "observations": len(values),
                }
    for job_id in state.get("selected_jobs", []):
        job_state = state["jobs"][job_id]
        selectors = state["run_jobs"][job_id]["selectors"]
        for identifier, samples in job_state.get("samples", {}).items():
            if identifier in resolved or not samples:
                continue
            metadata = selectors[identifier]
            resolved[identifier] = {
                "kind": "timing",
                "unit": metadata.get("unit", "ns"),
                "samples": samples,
                "summary": summarize_samples(samples),
            }
        for identifier, values in job_state.get("values", {}).items():
            if identifier in resolved or not values:
                continue
            if any(value != values[0] for value in values[1:]):
                raise PipelineError(
                    f"{identifier}: deterministic output was not stable"
                )
            metadata = selectors[identifier]
            resolved[identifier] = {
                "kind": "value",
                "unit": metadata.get("unit", ""),
                "value": values[0],
                "observations": len(values),
            }
    return resolved


def measurement_value(record: Mapping[str, Any]) -> Any:
    if record["kind"] == "timing":
        return record["summary"]["mean"]
    return record["value"]


def render_cell_value(cell_spec: CellSpec, values: Sequence[Any]) -> str:
    render = cell_spec.render
    if render == "text":
        return str(values[0])
    if render == "integer":
        return str(int(round(values[0])))
    if render == "bytes-integer":
        return str(int(round(values[0])))
    if render == "kib-1":
        return f"{float(values[0]) / KIB:.1f}"
    if render == "mib-1":
        return f"{float(values[0]) / MIB:.1f}"
    if render == "gib-1":
        return f"{float(values[0]) / GIB:.1f}"
    if render == "mib-3":
        return f"{float(values[0]) / MIB:.3f}"
    if render == "mib-3-div3":
        return f"{float(values[0]) / 3.0 / MIB:.3f}"
    if render == "microseconds-1":
        return f"{float(values[0]) / 1.0e3:.1f}"
    if render == "milliseconds-1":
        return f"{float(values[0]) / 1.0e6:.1f}"
    if render == "milliseconds-3":
        return f"{float(values[0]) / 1.0e6:.3f}"
    if render == "milliseconds-3-div3":
        return f"{float(values[0]) / 3.0 / 1.0e6:.3f}"
    if render == "seconds-3":
        return f"{float(values[0]) / 1.0e9:.3f}"
    if render == "seconds-1":
        return f"{float(values[0]) / 1.0e9:.1f}"
    if render == "preprocess-core-seconds-1":
        hint_sample, record_columns, sample_columns, packing_table, outputs = (
            float(value) for value in values
        )
        total = hint_sample * record_columns / sample_columns + packing_table * outputs
        return f"{total / 1.0e9:.1f}"
    raise PipelineError(f"{cell_spec.id}: unknown renderer {render}")


def render_cell(
    cell_spec: CellSpec,
    resolved: Mapping[str, Mapping[str, Any]],
    *,
    allow_pending: bool,
) -> str:
    missing = [source for source in cell_spec.sources if source not in resolved]
    if missing:
        if allow_pending:
            return "pending"
        raise PipelineError(f"{cell_spec.id}: unresolved sources {missing}")
    values = [measurement_value(resolved[source]) for source in cell_spec.sources]
    return render_cell_value(cell_spec, values)


def render_result_tables(
    manifest: Manifest,
    resolved: Mapping[str, Mapping[str, Any]],
    *,
    allow_pending: bool,
) -> dict[str, list[dict[str, str]]]:
    tables: dict[str, list[dict[str, str]]] = {}
    for table in TABLE_LABELS:
        records: list[dict[str, str]] = []
        for row in manifest.table_rows[table]:
            panel = row.split(":", 1)[0] if ":" in row else ""
            record = {"panel": panel, "row": row}
            for column in manifest.table_columns[table]:
                record[column] = render_cell(
                    manifest.cells[f"{table}|{row}|{column}"],
                    resolved,
                    allow_pending=allow_pending,
                )
            records.append(record)
        tables[table] = records
    return tables


def markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def write_result_tables(
    run_dir: pathlib.Path,
    manifest: Manifest,
    tables: Mapping[str, Sequence[Mapping[str, str]]],
) -> None:
    table_dir = run_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(table_dir / "tables.json", tables)
    index = [
        "# Evaluation result groups",
        "",
        "Values are reviewer-facing renderings. Machine-readable source values, "
        "units, samples, and confidence intervals are in `../summary.json` and "
        "`../summary.tsv`.",
        "",
    ]
    for table in TABLE_LABELS:
        safe_name = table.replace(":", "-")
        columns = ["panel", "row", *manifest.table_columns[table]]
        csv_path = table_dir / f"{safe_name}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(tables[table])
        metadata = TABLE_METADATA[table]
        index.extend(
            [
                f"## {metadata['title']}",
                "",
                f"Result-group ID: `{table}`. CSV: `{csv_path.name}`.",
                "",
                str(metadata["notes"]),
                "",
            ]
        )
        index.append("| " + " | ".join(columns) + " |")
        index.append("| " + " | ".join("---" for _ in columns) + " |")
        for record in tables[table]:
            index.append(
                "| "
                + " | ".join(markdown_cell(str(record[column])) for column in columns)
                + " |"
            )
        index.append("")
    atomic_write_text(table_dir / "README.md", "\n".join(index))


def write_raw_samples(run_dir: pathlib.Path, state: Mapping[str, Any]) -> None:
    lines: list[str] = []
    for job_id in state["selected_jobs"]:
        job_state = state["jobs"][job_id]
        for measurement_id, samples in sorted(job_state["samples"].items()):
            for index, value in enumerate(samples):
                lines.append(
                    json.dumps(
                        {
                            "job": job_id,
                            "measurement": measurement_id,
                            "independent_repetition": index,
                            "value": value,
                            "unit": "ns",
                        },
                        sort_keys=True,
                    )
                )
        for batch in job_state["batches"]:
            if batch.get("status") != "completed":
                continue
            lines.append(
                json.dumps(
                    {
                        "job": job_id,
                        "batch": batch["index"],
                        "seed": batch.get("seed"),
                        "raw_benchmark_samples": batch.get("raw_samples"),
                        "raw_benchmark_samples_sha256": batch.get(
                            "raw_samples_sha256"
                        ),
                        "result": batch.get("result"),
                        "result_sha256": batch.get("result_sha256"),
                    },
                    sort_keys=True,
                )
            )
    atomic_write_text(
        run_dir / "raw-samples.jsonl", "\n".join(lines) + ("\n" if lines else "")
    )


def write_summary_outputs(
    manifest: Manifest,
    run_dir: pathlib.Path,
    state: Mapping[str, Any],
    resolved: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    consumers = manifest.consumers()
    selected_jobs_by_measurement = {
        identifier: job_id
        for job_id, run_job in state.get("run_jobs", {}).items()
        for identifier in run_job.get("measurements", [])
    }
    records = []
    for identifier in sorted(resolved):
        spec = manifest.measurements.get(identifier)
        record = {
            "id": identifier,
            "job": (
                selected_jobs_by_measurement.get(identifier)
                if spec is None
                else spec.job_id
            ),
            "description": "" if spec is None else spec.description,
            "consumers": consumers.get(identifier, []),
            **resolved[identifier],
        }
        records.append(record)
    unstable = [
        record["id"]
        for record in records
        if record["kind"] == "timing"
        and not record["summary"]["stable"]
        and state["mode"] in ADAPTIVE_MODES
    ]
    # A run of selected jobs lists only their measurements as missing; the
    # rest of the manifest is not its business.
    selected_measurements = {
        measurement.id
        for job_id in state["selected_jobs"]
        for measurement in manifest.measurements_for_job(job_id)
    }
    missing = [
        identifier
        for identifier, spec in manifest.measurements.items()
        if spec.kind != "formula"
        and identifier not in resolved
        and (state["mode"] != "jobs" or identifier in selected_measurements)
    ]
    summary = {
        "run_id": state["run_id"],
        "mode": state["mode"],
        "snapshot_id": state["snapshot_id"],
        "benchmark_cpu": state.get("benchmark_cpu"),
        "statistical_rule": {
            "independent_minimum": DEFAULT_MIN_REPETITIONS,
            "independent_maximum": DEFAULT_MAX_REPETITIONS,
            "looks": list(SEQUENTIAL_LOOKS),
            "target_relative_half_width": DEFAULT_RELATIVE_HALF_WIDTH,
            "familywise_confidence": 1.0 - SEQUENTIAL_FAMILY_ALPHA,
            "method": (
                "Student-t intervals with Bonferroni alpha allocation across "
                f"the {len(SEQUENTIAL_LOOKS)} declared sequential looks"
            ),
            "append_independent_unit": "one complete 64-batch sequence",
            "direct_warmup_repetitions_per_process": DIRECT_WARMUP_REPETITIONS,
            "rebuild_warmup_repetitions_per_process": REBUILD_WARMUP_REPETITIONS,
            "append_warmup_sequences_per_process": APPEND_WARMUP_REPETITIONS,
        },
        "records": records,
        "missing_full_manifest_measurements": missing,
        "unstable_measurements": unstable,
        "publishable": state["mode"] == "full" and not missing and not unstable,
        "partial": state["mode"] == "jobs",
        "selected_jobs": list(state["selected_jobs"]),
    }
    atomic_write_json(run_dir / "summary.json", summary)
    with (run_dir / "summary.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "measurement",
                "job",
                "kind",
                "unit",
                "n",
                "mean_or_value",
                "ci95_low",
                "ci95_high",
                "relative_half_width",
                "stable",
                "consumers",
            ]
        )
        for record in records:
            if record["kind"] == "timing":
                stats = record["summary"]
                ci = stats["ci95"]
                row = [
                    record["id"],
                    record["job"],
                    record["kind"],
                    record["unit"],
                    stats["n"],
                    stats["mean"],
                    "" if ci is None else ci["low"],
                    "" if ci is None else ci["high"],
                    "" if ci is None else ci["relative_half_width"],
                    stats["stable"],
                    ",".join(record["consumers"]),
                ]
            else:
                row = [
                    record["id"],
                    record["job"],
                    record["kind"],
                    record["unit"],
                    "",
                    record["value"],
                    "",
                    "",
                    "",
                    "",
                    ",".join(record["consumers"]),
                ]
            writer.writerow(row)
    return summary


def validate_result_dimensions(
    manifest: Manifest,
    tables: Mapping[str, Sequence[Mapping[str, str]]],
    resolved: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    resolved = {} if resolved is None else resolved
    evidence: dict[str, Any] = {}
    for label in TABLE_LABELS:
        expected_rows = len(manifest.table_rows[label])
        actual_rows = len(tables[label])
        expected_columns = len(manifest.table_columns[label])
        expected_keys = {"panel", "row", *manifest.table_columns[label]}
        for record in tables[label]:
            if set(record) != expected_keys:
                raise PipelineError(
                    f"{label}|{record.get('row', '?')}: result columns differ: "
                    f"missing={sorted(expected_keys - set(record))}, "
                    f"extra={sorted(set(record) - expected_keys)}"
                )
        if actual_rows != expected_rows:
            raise PipelineError(
                f"{label}: generated {actual_rows} rows, expected {expected_rows}"
            )
        table_cells = [
            cell_spec
            for cell_spec in manifest.cells.values()
            if cell_spec.table == label
        ]
        for cell_spec in table_cells:
            validate_cell_unit_contract(cell_spec, manifest)
        source_references = [
            source for cell_spec in table_cells for source in cell_spec.sources
        ]
        panel_counts: dict[str, int] = defaultdict(int)
        for record in tables[label]:
            panel_counts[record["panel"]] += 1
        evidence[label] = {
            "rows": actual_rows,
            "columns": expected_columns,
            "panel_rows": list(panel_counts.values()),
            "unit_checked_cells": len(table_cells),
            "resolved_value_source_references": sum(
                source in resolved for source in source_references
            ),
            "pending_value_source_references": sum(
                source not in resolved for source in source_references
            ),
        }
    return evidence


def write_audit_report(
    manifest: Manifest,
    run_dir: pathlib.Path,
    state: Mapping[str, Any],
    summary: Mapping[str, Any],
    result_validation: Mapping[str, Any],
) -> None:
    consumers = manifest.consumers()
    shared_outputs = [
        (identifier, cells)
        for identifier, cells in consumers.items()
        if len(cells) > 1
        and manifest.measurements[identifier].job_id is not None
    ]
    lines = [
        "# Reproduction audit report",
        "",
        f"- Run: `{state['run_id']}`",
        f"- Mode: `{state['mode']}`",
        f"- Source snapshot: `{state['snapshot_id']}`",
        f"- Benchmark CPU: {state.get('benchmark_cpu')}",
        f"- Jobs selected: {len(state['selected_jobs'])}",
        f"- Manifest result cells: {len(manifest.cells)}",
        f"- Manifest timing series: {sum(m.kind == 'timing' for m in manifest.measurements.values())}",
        f"- Uncovered cells: 0",
        f"- Measurements without consumers: 0",
        f"- Duplicate jobs: 0",
        "",
        "## Statistical status",
        "",
        (
            f"Publication-ready: **{summary['publishable']}**. "
            f"Unstable timing series: {len(summary['unstable_measurements'])}. "
            f"Missing full-manifest measurements: "
            f"{len(summary['missing_full_manifest_measurements'])}."
        ),
        "",
        "The stopping rule uses Student-t intervals and allocates the 5% error "
        f"budget across looks at {', '.join(map(str, SEQUENTIAL_LOOKS))} "
        "independent repetitions. Each benchmark process discards explicit "
        "warmup work before collecting samples. A complete 64-batch append "
        "sequence is one independent repetition.",
        "",
        "## Shared work",
        "",
    ]
    for identifier, cells in sorted(shared_outputs):
        lines.append(f"- `{identifier}` -> {', '.join(f'`{item}`' for item in cells)}")
    lines.extend(["", "## Deliberately omitted work", ""])
    lines.extend(f"- {item}" for item in OMITTED_EXPERIMENTS)
    lines.extend(
        [
            "",
            "## Result validation",
            "",
            "All generated result groups have the declared dimensions:",
            "",
        ]
    )
    for label, record in result_validation.items():
        lines.append(
            f"- `{label}`: {record['rows']} rows, {record['columns']} columns"
        )
    lines.extend(
        [
            "",
            "Machine-readable measurements and confidence intervals are in "
            "`summary.json` and `summary.tsv`. Reviewer-facing CSV, JSON, and "
            "Markdown result groups are under `tables/`.",
        ]
    )
    atomic_write_text(run_dir / "audit-report.md", "\n".join(lines) + "\n")


def finalize_run_outputs(
    manifest: Manifest,
    run_dir: pathlib.Path,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    expected_snapshot_id = str(state["snapshot_id"])
    validate_current_source_snapshot(
        expected_snapshot_id,
        phase="starting final output generation",
    )
    resolved = resolve_measurements(manifest, state)
    allow_pending = state["mode"] != "full"
    tables = render_result_tables(
        manifest, resolved, allow_pending=allow_pending
    )
    write_result_tables(run_dir, manifest, tables)
    validation = validate_result_dimensions(manifest, tables, resolved)
    atomic_write_json(run_dir / "result-validation.json", validation)
    write_raw_samples(run_dir, state)
    summary = write_summary_outputs(manifest, run_dir, state, resolved)
    write_audit_report(
        manifest,
        run_dir,
        state,
        summary,
        validation,
    )
    validate_current_source_snapshot(
        expected_snapshot_id,
        phase="finishing final output generation",
    )
    return {
        "summary": summary,
        "result_validation": validation,
    }


def runtime_estimate(manifest: Manifest) -> dict[str, Any]:
    setup = sum(job.estimated_setup_seconds for job in manifest.jobs.values())
    per_repetition = sum(
        job.estimated_seconds_per_repetition for job in manifest.jobs.values()
    )
    warmup_work = sum(
        benchmark_warmup_repetitions(
            RunJob(job, tuple(manifest.measurements_for_job(job.id))),
            "full",
        )
        * job.estimated_seconds_per_repetition
        for job in manifest.jobs.values()
    )
    minimum = (
        setup
        + DEFAULT_MIN_REPETITIONS * per_repetition
        + warmup_work
    )
    maximum = (
        len(SEQUENTIAL_LOOKS) * setup
        + DEFAULT_MAX_REPETITIONS * per_repetition
        + len(SEQUENTIAL_LOOKS) * warmup_work
    )
    unique_builds = {
        (job.benchmark, job.features) for job in manifest.jobs.values()
    }
    disk_bytes = len(unique_builds) * 35 * MIB + len(manifest.jobs) * 4 * MIB
    return {
        "setup_seconds": setup,
        "per_repetition_seconds": per_repetition,
        "warmup_seconds_per_process": warmup_work,
        "maximum_processes_per_job": len(SEQUENTIAL_LOOKS),
        "minimum_seconds": minimum,
        "maximum_seconds": maximum,
        "minimum_hours": minimum / 3600.0,
        "maximum_hours": maximum / 3600.0,
        "estimated_disk_bytes": disk_bytes,
        "estimated_disk_gib": disk_bytes / (1024**3),
        "unique_builds": len(unique_builds),
        "basis": (
            "Conservative static family estimates; the canary supplies measured "
            "preflight evidence before --full."
        ),
    }


def write_plan(
    manifest: Manifest,
    run_dir: pathlib.Path,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    consumers = manifest.consumers()
    estimate = runtime_estimate(manifest)
    lines = [
        "# Evaluation reproduction plan",
        "",
        f"- Source snapshot: `{snapshot['snapshot_id']}`",
        f"- Evaluation result groups: {len(TABLE_LABELS)}",
        f"- Exact result cells: {len(manifest.cells)}",
        f"- Deduplicated benchmark jobs: {len(manifest.jobs)}",
        f"- Consumed timing series: {sum(m.kind == 'timing' for m in manifest.measurements.values())}",
        f"- Uncovered cells: 0",
        f"- Duplicate jobs: 0",
        f"- Measurements without consumers: 0",
        f"- Estimated minimum full duration: {estimate['minimum_hours']:.1f} hours",
        f"- Estimated worst-case full duration: {estimate['maximum_hours']:.1f} hours",
        f"- Estimated run storage: {estimate['estimated_disk_gib']:.2f} GiB",
        "",
        "## Exact cell manifest",
        "",
    ]
    for table in TABLE_LABELS:
        lines.extend([f"### `{table}`", ""])
        for row in manifest.table_rows[table]:
            for column in manifest.table_columns[table]:
                item = manifest.cells[f"{table}|{row}|{column}"]
                source_text = []
                for source in item.sources:
                    measurement = manifest.measurements[source]
                    if measurement.job_id is None:
                        producer = f"formula `{measurement.formula}`"
                    else:
                        selector_text = json.dumps(
                            measurement.selector_dict(), sort_keys=True
                        )
                        producer = (
                            f"job `{measurement.job_id}`, entry `{selector_text}`, "
                            f"path `{'.'.join(measurement.path)}`"
                        )
                    source_text.append(f"`{source}` ({producer})")
                lines.append(
                    f"- `{row}` / `{column}` <- " + "; ".join(source_text)
                )
        lines.append("")
    lines.extend(["## Scheduled jobs", ""])
    for job_id in sorted(manifest.jobs):
        job = manifest.jobs[job_id]
        measurements = manifest.measurements_for_job(job_id)
        cell_consumers = sorted(
            {
                cell_id
                for measurement in measurements
                for cell_id in consumers[measurement.id]
            }
        )
        lines.extend(
            [
                f"### `{job_id}`",
                "",
                f"- Benchmark: `{job.benchmark}`",
                f"- Features: `{','.join(job.features)}`",
                f"- Result: `{job.result_relpath}`",
                f"- Setup: {job.setup}",
                f"- Environment: `{json.dumps(job.env_dict(), sort_keys=True)}`",
                f"- Consumed measurements: {len(measurements)}",
                f"- Exact consuming cells: {len(cell_consumers)}",
                "",
            ]
        )
        for measurement in sorted(measurements, key=lambda item: item.id):
            lines.append(
                f"- `{measurement.id}` -> "
                + ", ".join(f"`{item}`" for item in consumers[measurement.id])
            )
        lines.append("")
    lines.extend(["## Shared benchmark outputs", ""])
    shared = [
        (identifier, cells)
        for identifier, cells in consumers.items()
        if len(cells) > 1
        and manifest.measurements[identifier].job_id is not None
    ]
    for identifier, cells in sorted(shared):
        lines.append(f"- `{identifier}` -> " + ", ".join(f"`{item}`" for item in cells))
    lines.extend(["", "## Deliberately omitted work", ""])
    lines.extend(f"- {item}" for item in OMITTED_EXPERIMENTS)
    lines.extend(
        [
            "",
            "## Repetition rule",
            "",
            f"Full jobs begin with {DEFAULT_MIN_REPETITIONS} independent "
            "repetitions. At the declared looks of "
            f"{', '.join(map(str, SEQUENTIAL_LOOKS))} repetitions, the pipeline computes "
            "Student-t intervals with Bonferroni allocation of the overall 5% "
            "error probability. It stops a job only when every consumed timing "
            "series has a simultaneous relative half-width at most 5%, or after "
            f"{DEFAULT_MAX_REPETITIONS} repetitions. Results still outside the "
            f"target at {DEFAULT_MAX_REPETITIONS} are marked unstable and "
            "non-publishable.",
            "",
            f"Each direct-timing process executes and discards "
            f"{DIRECT_WARMUP_REPETITIONS} warmup repetitions. Each append "
            f"process discards {APPEND_WARMUP_REPETITIONS} complete warmup "
            "sequence before recording independent sequences. Benchmark "
            "processes are pinned to the CPU recorded in the run provenance.",
            "",
            "For append processing, one fresh 64-batch sequence is one independent "
            "repetition. The 64 within-sequence batch timings are retained but are "
            "not treated as independent samples.",
        ]
    )
    atomic_write_text(run_dir / "plan.md", "\n".join(lines) + "\n")
    atomic_write_json(run_dir / "runtime-estimate.json", estimate)
    return estimate


def gate_dir(snapshot_id: str) -> pathlib.Path:
    return GATES_ROOT / snapshot_id


def record_gate(
    snapshot_id: str,
    name: str,
    *,
    run_id: str,
    evidence: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "snapshot_id": snapshot_id,
        "gate": name,
        "run_id": run_id,
        "recorded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evidence": dict(evidence or {}),
    }
    atomic_write_json(gate_dir(snapshot_id) / f"{name}.json", payload)


def require_gates(snapshot_id: str, names: Sequence[str]) -> None:
    missing = []
    invalid = []
    for name in names:
        path = gate_dir(snapshot_id) / f"{name}.json"
        if not path.exists():
            missing.append(name)
            continue
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            invalid.append(name)
            continue
        if (
            record.get("snapshot_id") != snapshot_id
            or record.get("pipeline_version") != PIPELINE_VERSION
            or record.get("gate") != name
        ):
            invalid.append(name)
    if missing or invalid:
        raise PipelineError(
            "precondition gates are not satisfied for this source snapshot: "
            f"missing={missing}, invalid={invalid}"
        )


def run_unit_tests(run_dir: pathlib.Path) -> None:
    test_file = ROOT / "scripts" / "test_reproduce.py"
    if not test_file.exists():
        raise PipelineError(f"missing pipeline unit tests: {test_file}")
    command = [sys.executable, str(test_file)]
    completed = run_capture(command, check=False, timeout=300)
    atomic_write_bytes(run_dir / "unit-tests.stdout.log", completed.stdout)
    atomic_write_bytes(run_dir / "unit-tests.stderr.log", completed.stderr)
    if completed.returncode != 0:
        raise PipelineError(
            f"pipeline unit tests failed; see {run_dir / 'unit-tests.stderr.log'}"
        )
    # These tests exercise fixture restoration, selected records, independent
    # dense/structured PIR agreement and verified openings, including every
    # real TreePIR color server. Do not let a passing Python scheduler test
    # stand in for correctness of the measured Rust code.
    state = load_checkpoint(run_dir)
    snapshot_id = str(state["snapshot_id"])
    validate_current_source_snapshot(snapshot_id, phase="before Rust correctness tests")
    target = isolated_build_target(run_dir, snapshot_id, "undine")
    command = ["cargo", "test", "--locked", "--release", "--features", "bench",
               "--target-dir", str(target)]
    for test in ("benchmark_fixtures", "private_selected_records", "private_pir",
                 "private_update", "private_opening", "private_layout", "private_bulk",
                 "treepir"):
        command.extend(["--test", test])
    environment = clean_benchmark_environment()
    environment.update({"RUSTC_WRAPPER": "", "RUSTC_WORKSPACE_WRAPPER": ""})
    atomic_write_json(run_dir / "rust-correctness.command.json", command)
    completed = run_capture(command, cwd=CRATE_ROOTS["undine"], env=environment,
                            check=False, timeout=1800)
    atomic_write_bytes(run_dir / "rust-correctness.stdout.log", completed.stdout)
    atomic_write_bytes(run_dir / "rust-correctness.stderr.log", completed.stderr)
    if completed.returncode != 0:
        raise PipelineError(f"Rust correctness tests failed; see {run_dir / 'rust-correctness.stderr.log'}")
    validate_current_source_snapshot(snapshot_id, phase="after Rust correctness tests")


def full_preflight_risks() -> list[str]:
    return [
        "Every deduplicated job needs at least "
        f"{DEFAULT_MIN_REPETITIONS} independent repetitions.",
        "The 64-batch append sequences and the large corrupted range openings "
        "are the dominant CPU costs.",
        "The private server jobs build their input cells on all cores, then "
        "rebuild PIR preprocessing on the pinned core for each preprocessing sample; "
        "the single server at a 4 GiB log holds about 15 GiB of memory.",
        "Thermal throttling or competing host load can widen confidence intervals "
        f"and push individual jobs toward {DEFAULT_MAX_REPETITIONS} repetitions.",
        f"A timing series that misses the precision target at "
        f"{DEFAULT_MAX_REPETITIONS} repetitions "
        "will be retained and flagged; it will not be silently accepted.",
        "Machine load, thermal state, and CPU-frequency policy can make a new "
        "host's timings differ from the included reference run even when the "
        "implementation and workloads are identical.",
    ]


def calibrated_runtime_estimate(
    manifest: Manifest, state: Mapping[str, Any]
) -> dict[str, Any]:
    observed_by_family: dict[str, float] = defaultdict(float)
    canary_static_by_family: dict[str, float] = defaultdict(float)
    for job_id in state["selected_jobs"]:
        job = manifest.jobs[job_id]
        observed_by_family[job.family] += sum(
            float(batch.get("elapsed_seconds", 0))
            for batch in state["jobs"][job_id]["batches"]
            if batch.get("status") == "completed"
        )
        canary_static_by_family[job.family] += (
            job.estimated_setup_seconds + job.estimated_seconds_per_repetition
        )
    full_minimum_by_family: dict[str, float] = defaultdict(float)
    full_maximum_by_family: dict[str, float] = defaultdict(float)
    full_jobs_by_family: dict[str, int] = defaultdict(int)
    for job in manifest.jobs.values():
        warmup = benchmark_warmup_repetitions(
            RunJob(job, tuple(manifest.measurements_for_job(job.id))),
            "full",
        )
        full_minimum_by_family[job.family] += (
            job.estimated_setup_seconds
            + (DEFAULT_MIN_REPETITIONS + warmup)
            * job.estimated_seconds_per_repetition
        )
        full_maximum_by_family[job.family] += (
            len(SEQUENTIAL_LOOKS) * job.estimated_setup_seconds
            + (
                DEFAULT_MAX_REPETITIONS
                + len(SEQUENTIAL_LOOKS) * warmup
            )
            * job.estimated_seconds_per_repetition
        )
        full_jobs_by_family[job.family] += 1
    families: dict[str, Any] = {}
    minimum = 0.0
    maximum = 0.0
    for family in sorted(full_jobs_by_family):
        observed = observed_by_family.get(family, 0.0)
        canary_static = canary_static_by_family.get(family, 0.0)
        if observed <= 0 or canary_static <= 0:
            factor = 1.0
            basis = "static fallback; no canary observation"
        else:
            factor = observed / canary_static
            basis = "scaled from publication-sized canary elapsed time"
        family_minimum = factor * full_minimum_by_family[family]
        family_maximum = factor * full_maximum_by_family[family]
        minimum += family_minimum
        maximum += family_maximum
        families[family] = {
            "jobs": full_jobs_by_family[family],
            "canary_observed_seconds": observed,
            "canary_static_seconds": canary_static,
            "calibration_factor": factor,
            "minimum_seconds": family_minimum,
            "maximum_seconds": family_maximum,
            "basis": basis,
        }
    return {
        "method": (
            "Scale each benchmark family's static setup/work model by the "
            "publication-sized canary's observed-to-modeled elapsed-time ratio."
        ),
        "minimum_seconds": minimum,
        "maximum_seconds": maximum,
        "minimum_hours": minimum / 3600.0,
        "maximum_hours": maximum / 3600.0,
        "families": families,
    }


def write_preflight(
    manifest: Manifest,
    run_dir: pathlib.Path,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    static_estimate = runtime_estimate(manifest)
    calibrated_estimate = calibrated_runtime_estimate(manifest, state)
    benchmark_seconds = sum(
        float(batch.get("elapsed_seconds", 0))
        for job in state["jobs"].values()
        for batch in job["batches"]
        if batch.get("status") == "completed"
    )
    build_seconds = sum(
        float(record.get("build_seconds", 0))
        for record in state["binaries"].values()
    )
    report = {
        "snapshot_id": state["snapshot_id"],
        "canary_run": state["run_id"],
        "canary_benchmark_seconds": benchmark_seconds,
        "canary_build_seconds": build_seconds,
        "static_estimate": static_estimate,
        "canary_calibrated_estimate": calibrated_estimate,
        "risks": full_preflight_risks(),
        "full_command": "./artifact full",
        "full_starts_automatically": False,
    }
    lines = [
        "# Full-run preflight",
        "",
        f"- Source snapshot: `{state['snapshot_id']}`",
        f"- Canary run: `{state['run_id']}`",
        f"- Canary benchmark time: {benchmark_seconds / 60.0:.1f} minutes",
        f"- Canary build time: {build_seconds / 60.0:.1f} minutes",
        (
            "- Canary-calibrated full estimate: "
            f"{calibrated_estimate['minimum_hours']:.1f}--"
            f"{calibrated_estimate['maximum_hours']:.1f} hours"
        ),
        (
            "- Earlier static plan estimate: "
            f"{static_estimate['minimum_hours']:.1f}--"
            f"{static_estimate['maximum_hours']:.1f} hours"
        ),
        f"- Estimated isolated-run storage: {static_estimate['estimated_disk_gib']:.2f} GiB",
        "",
        "## Remaining risks",
        "",
        *[f"- {risk}" for risk in report["risks"]],
        "",
        "The full run is not started by canary or preflight. Its explicit command is:",
        "",
        "```bash",
        report["full_command"],
        "```",
    ]
    atomic_write_text(run_dir / "preflight.md", "\n".join(lines) + "\n")
    atomic_write_json(run_dir / "preflight.json", report)
    return report


def selected_run_elapsed(state: Mapping[str, Any]) -> float:
    return sum(
        float(batch.get("elapsed_seconds", 0))
        for job in state["jobs"].values()
        for batch in job["batches"]
        if batch.get("status") == "completed"
    )


def process_completed_run(
    manifest: Manifest,
    run_dir: pathlib.Path,
    state: dict[str, Any],
    *,
    was_resume: bool,
    was_interrupted: bool,
) -> dict[str, Any]:
    outputs = finalize_run_outputs(manifest, run_dir, state)
    snapshot_id = state["snapshot_id"]
    mode = state["mode"]
    common_evidence = {
        "result_validation": str(
            (run_dir / "result-validation.json").relative_to(ROOT)
        ),
    }
    if mode == "smoke":
        record_gate(
            snapshot_id,
            "smoke",
            run_id=state["run_id"],
            evidence=common_evidence
            | {"families": sorted({manifest.jobs[job].family for job in state["selected_jobs"]})},
        )
        record_gate(
            snapshot_id,
            "result-validation",
            run_id=state["run_id"],
            evidence=common_evidence,
        )
    if was_resume and was_interrupted:
        record_gate(
            snapshot_id,
            "resume",
            run_id=state["run_id"],
            evidence={"checkpoint": str(checkpoint_path(run_dir).relative_to(ROOT))},
        )
    if mode == "canary":
        preflight = write_preflight(manifest, run_dir, state)
        record_gate(
            snapshot_id,
            "canary",
            run_id=state["run_id"],
            evidence=common_evidence
            | {
                "families": sorted(
                    {manifest.jobs[job].family for job in state["selected_jobs"]}
                ),
                "benchmark_seconds": selected_run_elapsed(state),
            },
        )
        record_gate(
            snapshot_id,
            "preflight",
            run_id=state["run_id"],
            evidence={"preflight": str((run_dir / "preflight.json").relative_to(ROOT))},
        )
        outputs["preflight"] = preflight
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the artifact's empirical result groups"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan", action="store_true", help="validate and print the exact plan")
    modes.add_argument("--smoke", action="store_true", help="run reduced path-coverage workloads")
    modes.add_argument(
        "--canary",
        action="store_true",
        help="run one publication-sized point per benchmark family",
    )
    modes.add_argument("--full", action="store_true", help="run the complete adaptive manifest")
    modes.add_argument(
        "--jobs",
        metavar="JOB[,JOB...]",
        help=(
            "run the named jobs of the manifest with the adaptive sampling of "
            "a full run; the run is partial and never publishable"
        ),
    )
    modes.add_argument("--resume", metavar="RUN_ID", help="resume an interrupted run")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=24 * 60 * 60,
        help="per-build and per-benchmark timeout (default: 24 hours)",
    )
    parser.add_argument(
        "--cpu",
        type=int,
        help=(
            "logical CPU for benchmark processes; by default, select an "
            "available maximum-frequency CPU on a non-system core"
        ),
    )
    parser.add_argument(
        "--stop-after-batches",
        type=int,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


@contextlib.contextmanager
def global_run_lock() -> Iterator[None]:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = RUNS_ROOT / ".pipeline.lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.timeout_seconds <= 0:
        raise PipelineError("--timeout-seconds must be positive")
    if args.stop_after_batches is not None and args.stop_after_batches <= 0:
        raise PipelineError("--stop-after-batches must be positive")
    benchmark_cpu = choose_benchmark_cpu(args.cpu)
    manifest = build_manifest()
    snapshot = collect_source_snapshot()
    with global_run_lock():
        if args.plan:
            selected = [
                RunJob(job, tuple(manifest.measurements_for_job(job.id)))
                for job in manifest.jobs.values()
            ]
            run_dir, state = initialize_run(
                manifest, "plan", selected, snapshot, benchmark_cpu
            )
            estimate = write_plan(manifest, run_dir, snapshot)
            state["status"] = "completed"
            write_checkpoint(run_dir, state)
            record_gate(
                snapshot["snapshot_id"],
                "plan",
                run_id=state["run_id"],
                evidence={
                    "plan": str((run_dir / "plan.md").relative_to(ROOT)),
                    "jobs": len(manifest.jobs),
                    "cells": len(manifest.cells),
                    "uncovered_cells": 0,
                    "duplicate_jobs": 0,
                    "unconsumed_measurements": 0,
                    "runtime_estimate": estimate,
                },
            )
            print(f"Plan validated: {run_dir / 'plan.md'}")
            print(
                f"{len(manifest.jobs)} jobs, {len(manifest.cells)} cells, "
                "0 uncovered, 0 duplicate, 0 unconsumed"
            )
            return 0

        if args.resume:
            run_dir = RUNS_ROOT / args.resume
            state = load_checkpoint(run_dir)
            if state.get("status") == "completed":
                raise PipelineError(f"run is already complete: {args.resume}")
            validate_resume_snapshot(state, snapshot)
            saved_cpu = state.get("benchmark_cpu")
            if args.cpu is not None and benchmark_cpu != saved_cpu:
                raise PipelineError(
                    f"resume CPU mismatch: checkpoint uses {saved_cpu}, "
                    f"requested {benchmark_cpu}"
                )
            run_jobs = reconstruct_run_jobs(manifest, state)
            validate_resume_artifacts(run_dir, state, run_jobs)
            was_interrupted = bool(
                state.get("interruption_requested")
                or state.get("status") == "interrupted"
            )
            state["resumed"] = True
            state["status"] = "running"
            write_checkpoint(run_dir, state)
            completed = execute_run(
                manifest,
                run_dir,
                state,
                run_jobs,
                timeout_seconds=args.timeout_seconds,
                stop_after_batches=args.stop_after_batches,
            )
            if not completed:
                print(f"Run checkpointed for resume: {run_dir.name}")
                return 75
            process_completed_run(
                manifest,
                run_dir,
                state,
                was_resume=True,
                was_interrupted=was_interrupted,
            )
            mark_run_completed(run_dir, state)
            print(f"Resumed run completed: {run_dir}")
            return 0

        if args.smoke:
            mode = "smoke"
            require_gates(snapshot["snapshot_id"], ("plan",))
        elif args.canary:
            mode = "canary"
            require_gates(
                snapshot["snapshot_id"],
                (
                    "plan",
                    "unit-tests",
                    "smoke",
                    "resume",
                    "result-validation",
                ),
            )
        else:
            mode = "jobs" if args.jobs else "full"
            require_gates(
                snapshot["snapshot_id"],
                (
                    "plan",
                    "unit-tests",
                    "smoke",
                    "resume",
                    "result-validation",
                    "canary",
                    "preflight",
                ),
            )
        job_ids = [job_id.strip() for job_id in args.jobs.split(",")] if args.jobs else None
        run_jobs = select_run_jobs(manifest, mode, job_ids)
        run_dir, state = initialize_run(
            manifest, mode, run_jobs, snapshot, benchmark_cpu
        )
        if mode == "smoke":
            run_unit_tests(run_dir)
            record_gate(
                snapshot["snapshot_id"],
                "unit-tests",
                run_id=state["run_id"],
                evidence={
                    "stdout": str(
                        (run_dir / "unit-tests.stdout.log").relative_to(ROOT)
                    ),
                    "rust_correctness_stdout": str(
                        (run_dir / "rust-correctness.stdout.log").relative_to(ROOT)
                    ),
                    "rust_correctness_command": str(
                        (run_dir / "rust-correctness.command.json").relative_to(ROOT)
                    ),
                },
            )
        completed = execute_run(
            manifest,
            run_dir,
            state,
            run_jobs,
            timeout_seconds=args.timeout_seconds,
            stop_after_batches=args.stop_after_batches,
        )
        if not completed:
            print(f"Run checkpointed for resume: {run_dir.name}")
            return 75
        outputs = process_completed_run(
            manifest,
            run_dir,
            state,
            was_resume=False,
            was_interrupted=False,
        )
        mark_run_completed(run_dir, state)
        if mode == "full" and not outputs["summary"]["publishable"]:
            print(f"Full run completed with flagged results: {run_dir}")
            return 2
        if mode == "jobs" and outputs["summary"]["unstable_measurements"]:
            print(f"Jobs run completed with flagged results: {run_dir}")
            return 2
        print(f"{mode.capitalize()} run completed: {run_dir}")
        if mode == "canary":
            print(
                "Preflight complete. The full run was not started; use "
                "`./artifact full` explicitly."
            )
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
