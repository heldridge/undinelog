#!/usr/bin/env python3
"""Validate a completed artifact run without executing benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
from collections import defaultdict
from typing import Any, Mapping


ROOT = pathlib.Path(__file__).resolve().parents[1]
REFERENCE_ROOT = ROOT / "reference-results"
REFERENCE_POINTER = REFERENCE_ROOT / "reference.json"
SOURCE_PROVENANCE = ROOT / "provenance" / "sources.json"
CRATES = ("undine", "borg")


class VerificationError(RuntimeError):
    pass


def load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VerificationError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise VerificationError(f"invalid JSON in {path}: {exc}") from exc


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def relative_files(root: pathlib.Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "integrity.json"
    }


def verify_integrity(run_dir: pathlib.Path) -> int:
    integrity_path = run_dir / "integrity.json"
    if not integrity_path.exists():
        return 0
    integrity = load_json(integrity_path)
    declared = integrity.get("files")
    require(isinstance(declared, dict), "integrity.json has no file map")
    actual_paths = relative_files(run_dir)
    declared_paths = set(declared)
    require(
        actual_paths == declared_paths,
        "reference file set differs from integrity manifest: "
        f"missing={sorted(declared_paths - actual_paths)}, "
        f"extra={sorted(actual_paths - declared_paths)}",
    )
    for relative, record in declared.items():
        path = run_dir / relative
        require(path.is_file(), f"integrity file is missing: {relative}")
        require(
            path.stat().st_size == int(record["bytes"]),
            f"size mismatch for {relative}",
        )
        require(
            sha256_file(path) == record["sha256"],
            f"SHA-256 mismatch for {relative}",
        )
    return len(declared)


def reference_identity() -> dict[str, str]:
    """Run ID and source snapshot of the packaged reference run."""
    pointer = load_json(REFERENCE_POINTER)
    require(
        isinstance(pointer.get("run_id"), str)
        and isinstance(pointer.get("snapshot_id"), str),
        "reference pointer is malformed",
    )
    return {"run_id": pointer["run_id"], "snapshot_id": pointer["snapshot_id"]}


def reference_dir() -> pathlib.Path:
    return REFERENCE_ROOT / reference_identity()["run_id"]


def crate_files() -> set[str]:
    """Measured source files of both crates, relative to the repository."""
    result: set[str] = set()
    for crate in CRATES:
        root = ROOT / crate
        for name in ("Cargo.toml", "Cargo.lock"):
            result.add(f"{crate}/{name}")
        for base in ("src", "benches", "apps/appendOnlyLog", "third_party"):
            for path in (root / base).rglob("*"):
                if path.is_file():
                    result.add(path.relative_to(ROOT).as_posix())
    return result


def verify_file_records(records: Mapping[str, Any], label: str) -> int:
    for relative, record in records.items():
        path = ROOT / relative
        require(path.is_file(), f"{label} is missing: {relative}")
        require(
            path.stat().st_size == int(record["bytes"]),
            f"{label} size mismatch: {relative}",
        )
        require(
            sha256_file(path) == record["sha256"],
            f"{label} hash mismatch: {relative}",
        )
    return len(records)


def verify_reference_sources(run_dir: pathlib.Path) -> int:
    """Check that both crates are the sources the reference run measured.

    The run's source snapshot and the recorded origin of each crate must both
    agree with the packaged files, and neither may omit a measured file.
    """
    current = crate_files()
    snapshot = load_json(run_dir / "provenance" / "source-snapshot.json")
    measured = {
        relative: record
        for relative, record in snapshot.get("files", {}).items()
        if relative.split("/", 1)[0] in CRATES
    }
    require(
        set(measured) == current,
        "crate sources differ from the reference run's snapshot: "
        f"missing={sorted(set(measured) - current)}, extra={sorted(current - set(measured))}",
    )
    verify_file_records(measured, "reference-run source")

    provenance = load_json(SOURCE_PROVENANCE)
    recorded: dict[str, Any] = {}
    for crate in CRATES:
        origin = provenance.get("crates", {}).get(crate)
        require(isinstance(origin, dict), f"source provenance has no record for {crate}")
        recorded.update(origin.get("files", {}))
    require(
        set(recorded) == current,
        "crate sources differ from their recorded origin: "
        f"missing={sorted(set(recorded) - current)}, extra={sorted(current - set(recorded))}",
    )
    return verify_file_records(recorded, "recorded source")


def verify_derived_outputs(run_dir: pathlib.Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "export_results.py"),
        str(run_dir),
        "--check",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()
        raise VerificationError(f"derived result validation failed: {detail}")


def load_reproduction_module() -> Any:
    path = ROOT / "scripts" / "reproduce.py"
    spec = importlib.util.spec_from_file_location("artifact_reproduce", path)
    require(spec is not None and spec.loader is not None, "cannot load reproduce.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def verify_manifest_compatibility(reference_manifest: Mapping[str, Any]) -> None:
    module = load_reproduction_module()
    current = normalized(module.build_manifest().as_dict())
    reference_jobs = {record["id"]: record for record in reference_manifest["jobs"]}
    current_jobs = {record["id"]: record for record in current["jobs"]}
    require(
        reference_jobs == current_jobs,
        "current benchmark job manifest differs from the reference run",
    )
    def produced_semantics(record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in record.items()
            if key not in {"description", "consumers"}
        }

    reference_measurements = {
        record["id"]: produced_semantics(record)
        for record in reference_manifest["measurements"]
        if record.get("job_id") is not None
    }
    current_measurements = {
        record["id"]: produced_semantics(record)
        for record in current["measurements"]
        if record.get("job_id") is not None
    }
    require(
        reference_measurements == current_measurements,
        "current consumed benchmark measurements differ from the reference run",
    )
    reference_cells = {record["id"] for record in reference_manifest["cells"]}
    current_cells = {record["id"] for record in current["cells"]}
    require(
        reference_cells == current_cells,
        "current result cells differ from the reference run",
    )


def verify_batch_artifacts(run_dir: pathlib.Path, checkpoint: Mapping[str, Any]) -> int:
    checked = 0
    for job_id, job in checkpoint["jobs"].items():
        for batch in job.get("batches", []):
            require(batch.get("status") == "completed", f"{job_id}: incomplete batch")
            require(batch.get("returncode") == 0, f"{job_id}: nonzero batch return code")
            for path_key, hash_key in (
                ("result", "result_sha256"),
                ("raw_samples", "raw_samples_sha256"),
            ):
                relative = batch.get(path_key)
                expected = batch.get(hash_key)
                require(relative and expected, f"{job_id}: missing {path_key} provenance")
                path = run_dir / str(relative)
                require(path.is_file(), f"{job_id}: missing {relative}")
                require(
                    sha256_file(path) == expected,
                    f"{job_id}: hash mismatch for {relative}",
                )
                checked += 1
    return checked


def verify_raw_sample_export(
    run_dir: pathlib.Path, summary: Mapping[str, Any]
) -> tuple[int, int]:
    independent: dict[str, list[float]] = defaultdict(list)
    batch_records = 0
    path = run_dir / "raw-samples.jsonl"
    try:
        handle = path.open(encoding="utf-8")
    except FileNotFoundError as exc:
        raise VerificationError(f"missing required file: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VerificationError(
                    f"invalid raw-samples.jsonl line {line_number}: {exc}"
                ) from exc
            if "measurement" in record:
                independent[record["measurement"]].append(float(record["value"]))
            elif "batch" in record:
                batch_records += 1
            else:
                raise VerificationError(
                    f"unrecognized raw-samples.jsonl line {line_number}"
                )
    timing_records = {
        record["id"]: record
        for record in summary["records"]
        if record["kind"] == "timing"
    }
    require(
        set(independent) == set(timing_records),
        "raw sample export does not cover exactly the timing series",
    )
    for identifier, record in timing_records.items():
        expected = [float(value) for value in record["samples"]]
        require(
            independent[identifier] == expected,
            f"raw sample export differs for {identifier}",
        )
    return sum(map(len, independent.values())), batch_records


def verify_run(run_dir: pathlib.Path, *, reference: bool) -> dict[str, Any]:
    require(run_dir.is_dir(), f"run directory does not exist: {run_dir}")
    integrity_files = verify_integrity(run_dir)
    checkpoint = load_json(run_dir / "checkpoint.json")
    manifest = load_json(run_dir / "manifest.json")
    summary = load_json(run_dir / "summary.json")

    require(checkpoint.get("status") == "completed", "checkpoint is not complete")
    require(summary.get("run_id") == checkpoint.get("run_id"), "run IDs disagree")
    require(summary.get("mode") == checkpoint.get("mode"), "run modes disagree")
    require(
        summary.get("snapshot_id") == checkpoint.get("snapshot_id"),
        "source snapshots disagree",
    )
    selected = checkpoint.get("selected_jobs")
    jobs = checkpoint.get("jobs")
    require(isinstance(selected, list), "checkpoint selected_jobs is malformed")
    require(isinstance(jobs, dict), "checkpoint jobs is malformed")
    require(set(selected) == set(jobs), "selected job set differs from checkpoint jobs")
    require(
        all(record.get("status") == "completed" for record in jobs.values()),
        "one or more jobs are incomplete",
    )

    manifest_jobs = manifest.get("jobs")
    manifest_cells = manifest.get("cells")
    require(isinstance(manifest_jobs, list), "manifest jobs is malformed")
    require(isinstance(manifest_cells, list), "manifest cells is malformed")
    require(
        len({record["id"] for record in manifest_jobs}) == len(manifest_jobs),
        "manifest contains duplicate job IDs",
    )
    records = summary.get("records")
    require(isinstance(records, list), "summary records is malformed")
    record_ids = {record["id"] for record in records}
    require(len(record_ids) == len(records), "summary contains duplicate record IDs")
    run_jobs = checkpoint.get("run_jobs")
    require(isinstance(run_jobs, dict), "checkpoint run_jobs is malformed")
    expected_record_ids = {
        record["id"]
        for record in manifest.get("measurements", [])
        if record.get("kind") == "formula"
    }
    for job_id in selected:
        run_job = run_jobs.get(job_id)
        require(isinstance(run_job, dict), f"{job_id}: missing run-job selection")
        expected_record_ids.update(run_job.get("measurements", []))
    require(
        record_ids == expected_record_ids,
        "summary records differ from the selected measurement set",
    )
    timing = [record for record in records if record.get("kind") == "timing"]
    for record in timing:
        stats = record["summary"]
        require(stats.get("n") == len(record.get("samples", [])), f"n mismatch: {record['id']}")

    if checkpoint.get("mode") == "full":
        require(
            not summary.get("missing_full_manifest_measurements"),
            "summary has missing measurements",
        )
        require(
            not summary.get("unstable_measurements"),
            "summary has unstable timings",
        )
        for record in timing:
            stats = record["summary"]
            require(stats.get("stable") is True, f"unstable timing: {record['id']}")
            require(
                stats["sequential_ci"]["relative_half_width"] <= 0.05,
                f"precision target missed: {record['id']}",
            )
        require(summary.get("publishable") is True, "full run is not publishable")

    batch_artifacts = verify_batch_artifacts(run_dir, checkpoint)
    raw_samples, raw_batches = verify_raw_sample_export(run_dir, summary)
    verify_derived_outputs(run_dir)

    if reference:
        identity = reference_identity()
        require(checkpoint.get("run_id") == identity["run_id"], "wrong reference run ID")
        require(
            checkpoint.get("snapshot_id") == identity["snapshot_id"],
            "wrong reference source snapshot",
        )
        require(checkpoint.get("mode") == "full", "reference run is not a full run")
        require(integrity_files > 0, "reference run has no integrity manifest")
        require(
            len(selected) == len(manifest_jobs),
            "reference run did not select every manifest job",
        )
        verify_manifest_compatibility(manifest)
        exact_sources = verify_reference_sources(run_dir)
    else:
        exact_sources = 0

    return {
        "run_id": checkpoint["run_id"],
        "mode": checkpoint["mode"],
        "snapshot_id": checkpoint["snapshot_id"],
        "jobs": len(selected),
        "result_cells": len(manifest_cells),
        "timing_series": len(timing),
        "summary_records": len(records),
        "raw_independent_samples": raw_samples,
        "raw_batch_records": raw_batches,
        "checked_batch_artifacts": batch_artifacts,
        "integrity_files": integrity_files,
        "byte_identical_crate_sources": exact_sources,
        "publishable": bool(summary.get("publishable")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=pathlib.Path,
        default=None,
        help="completed run directory (default: included reference run)",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="require the reference run's identity and source compatibility",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.run_dir is None:
        require(
            REFERENCE_POINTER.is_file(),
            "no reference run is packaged yet; run `./artifact full` and "
            "`./artifact package RUN_ID` first",
        )
        run_dir = reference_dir().resolve()
        reference = True
    else:
        run_dir = args.run_dir.resolve()
        reference = args.reference
    report = verify_run(run_dir, reference=reference)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
