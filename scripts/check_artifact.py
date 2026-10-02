#!/usr/bin/env python3
"""Validate the packaged repository's publication-artifact invariants."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

import check_docs
import source_provenance


ROOT = pathlib.Path(__file__).resolve().parents[1]
REFERENCE_ROOT = ROOT / "reference-results"
REFERENCE_POINTER = REFERENCE_ROOT / "reference.json"
FORBIDDEN_SUFFIXES = {".bib", ".pdf", ".tex"}
FORBIDDEN_PARTS = {"__pycache__", "target"}
TABLE_BENCHMARKS = {
    "bench_append_only_log_bulk_updates_append_local.rs",
    "bench_append_only_log_client_point_open_local.rs",
    "bench_append_only_log_one_server_kappa_rescale_local.rs",
    "bench_append_only_log_point_open_local.rs",
    "bench_append_only_log_range_open_local.rs",
    "bench_append_only_log_repair_kernels_local.rs",
}
MEASURED_BENCHMARKS = {
    "undine": TABLE_BENCHMARKS | {"bench_append_only_log_on_the_fly_open_local.rs"},
    "borg": TABLE_BENCHMARKS,
}
ABSOLUTE_WORKSPACE = re.compile(
    r"/(?:home|Users)/[^/\s]+/(?:[^/\s]+/)*"
    r"(?:paper-a-optimized|paper-a-publication-artifact|undinelog-publication-artifact)"
    r"(?:/|\b)"
)


class ArtifactCheckError(RuntimeError):
    pass


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and completed.returncode:
        detail = completed.stderr.decode(errors="replace").strip()
        raise ArtifactCheckError(f"git {' '.join(args)} failed: {detail}")
    return completed


def tracked_files() -> list[pathlib.Path]:
    output = git("ls-files", "-z").stdout
    return [
        ROOT / raw.decode()
        for raw in output.split(b"\0")
        if raw
    ]


def check_packaged_files(files: list[pathlib.Path]) -> None:
    errors: list[str] = []
    for path in files:
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden publication file: {relative}")
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            errors.append(f"forbidden build/cache path: {relative}")
        if path.is_file():
            prefix = path.read_bytes()[:4]
            if prefix == b"\x7fELF" or prefix[:2] == b"MZ":
                errors.append(f"compiled executable is tracked: {relative}")
    manifest_text = {
        crate: (ROOT / crate / "Cargo.toml").read_text(encoding="utf-8")
        for crate in MEASURED_BENCHMARKS
    }
    for crate, measured in MEASURED_BENCHMARKS.items():
        benches = {
            path.name
            for path in files
            if path.parent == ROOT / crate / "benches" and path.suffix == ".rs"
        }
        missing = measured - benches
        if missing:
            errors.append(f"{crate}: measured benchmark is missing: {sorted(missing)}")
        for name in sorted(measured):
            if f'name = "{pathlib.Path(name).stem}"' not in manifest_text[crate]:
                errors.append(f"{crate}: Cargo.toml does not declare {name}")
    reference_run = reference_run_dir()
    required = [
        ROOT / "README.md",
        ROOT / "artifact",
        ROOT / "scripts" / "reproduce.py",
        ROOT / "scripts" / "verify_results.py",
        ROOT / "provenance" / "sources.json",
    ]
    if reference_run is not None:
        required += [
            reference_run / "checkpoint.json",
            reference_run / "summary.json",
            reference_run / "integrity.json",
        ]
    for path in required:
        if not path.is_file():
            errors.append(f"required artifact file is missing: {path.relative_to(ROOT)}")
    for path in files:
        if REFERENCE_ROOT in path.parents or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if ABSOLUTE_WORKSPACE.search(text):
            errors.append(
                f"packaged command/content uses an absolute workspace path: "
                f"{path.relative_to(ROOT)}"
            )
    if errors:
        raise ArtifactCheckError("\n".join(errors))


def reference_run_dir() -> pathlib.Path | None:
    """Directory of the packaged reference run, if one is packaged."""
    if not REFERENCE_POINTER.is_file():
        return None
    pointer = json.loads(REFERENCE_POINTER.read_text(encoding="utf-8"))
    return REFERENCE_ROOT / str(pointer["run_id"])


def check_reference_identity(*, required: bool) -> tuple[int, int] | None:
    reference_run = reference_run_dir()
    if reference_run is None:
        if required:
            raise ArtifactCheckError("no reference run is packaged")
        return None
    checkpoint = json.loads((reference_run / "checkpoint.json").read_text())
    summary = json.loads((reference_run / "summary.json").read_text())
    manifest = json.loads((reference_run / "manifest.json").read_text())
    jobs = len(checkpoint.get("selected_jobs", []))
    if checkpoint.get("mode") != "full" or jobs != len(manifest.get("jobs", [])):
        raise ArtifactCheckError("reference run is not a complete full run")
    timing = [
        record
        for record in summary.get("records", [])
        if record.get("kind") == "timing"
    ]
    if summary.get("publishable") is not True:
        raise ArtifactCheckError("reference summary is not publication-ready")
    return jobs, len(timing)


def check_release_git() -> None:
    commits = git("rev-list", "--count", "HEAD").stdout.decode().strip()
    roots = git("rev-list", "--max-parents=0", "HEAD").stdout.decode().splitlines()
    parents = git("show", "-s", "--format=%P", "HEAD").stdout.decode().strip()
    status = git("status", "--short", "--untracked-files=all").stdout.decode()
    if commits != "1" or len(roots) != 1 or parents:
        raise ArtifactCheckError(
            "release repository must contain exactly one parentless root commit"
        )
    if status:
        raise ArtifactCheckError(f"release worktree is not clean:\n{status}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release",
        action="store_true",
        help="also require a packaged reference run, one root commit, and a clean worktree",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = tracked_files()
    check_packaged_files(files)
    reference = check_reference_identity(required=args.release)
    source_provenance.check()
    check_docs.check()
    if args.release:
        check_release_git()
    described = (
        "no reference run packaged yet"
        if reference is None
        else f"{reference[0]} jobs, {reference[1]} timing series"
    )
    print(f"artifact checks passed: {len(files)} tracked files, {described}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ArtifactCheckError,
        check_docs.DocumentationError,
        source_provenance.ProvenanceError,
    ) as exc:
        print(f"artifact check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
