#!/usr/bin/env python3
"""Package a completed full run as the artifact's reference run.

The run directory is copied to `reference-results/RUN_ID/` without its
compiled benchmark binaries, an integrity manifest is written for every
retained file, and `reference-results/reference.json` is pointed at the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "results" / "runs"
REFERENCE_ROOT = ROOT / "reference-results"
EXCLUDED_TOP_LEVEL = {"binaries", "build-targets", "fixtures"}


class PackageError(RuntimeError):
    pass


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(run_id: str) -> pathlib.Path:
    source = RUNS_ROOT / run_id
    if not source.is_dir():
        raise PackageError(f"run directory does not exist: {source}")
    checkpoint = json.loads((source / "checkpoint.json").read_text(encoding="utf-8"))
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if checkpoint.get("mode") != "full" or checkpoint.get("status") != "completed":
        raise PackageError("only a completed full run can become the reference run")
    if summary.get("publishable") is not True:
        raise PackageError("the run is not publication-ready")
    if REFERENCE_ROOT.exists() and any(REFERENCE_ROOT.iterdir()):
        raise PackageError(
            f"{REFERENCE_ROOT.relative_to(ROOT)} already holds a reference run"
        )

    destination = REFERENCE_ROOT / run_id
    destination.mkdir(parents=True)
    for child in sorted(source.iterdir()):
        if child.name in EXCLUDED_TOP_LEVEL:
            continue
        if child.is_dir():
            shutil.copytree(child, destination / child.name)
        else:
            shutil.copy2(child, destination / child.name)

    files = {
        path.relative_to(destination).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    integrity = {
        "schema_version": 1,
        "algorithm": "sha256",
        "reference_run_id": run_id,
        "files": files,
    }
    (destination / "integrity.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True), encoding="utf-8"
    )
    pointer = {
        "schema_version": 1,
        "run_id": run_id,
        "snapshot_id": checkpoint["snapshot_id"],
    }
    (REFERENCE_ROOT / "reference.json").write_text(
        json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    destination = package(args.run_id)
    print(f"reference run packaged: {destination.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PackageError as exc:
        print(f"packaging failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
