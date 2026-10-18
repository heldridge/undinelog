#!/usr/bin/env python3
"""Render CSV, JSON, and Markdown result groups from a completed summary."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ExportError(RuntimeError):
    pass


def load_reproduction_module():
    path = ROOT / "scripts" / "reproduce.py"
    spec = importlib.util.spec_from_file_location("artifact_reproduce_export", path)
    if spec is None or spec.loader is None:
        raise ExportError("cannot load reproduce.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolved_from_summary(module, manifest, summary):
    records = {record["id"]: record for record in summary["records"]}
    resolved = {}
    for identifier, measurement in manifest.measurements.items():
        if measurement.kind == "formula":
            resolved[identifier] = {
                "kind": "formula",
                "unit": measurement.unit,
                "value": module.evaluate_formula(measurement),
                "formula": measurement.formula,
                "formula_args": list(measurement.formula_args),
            }
            continue
        record = records.get(identifier)
        if record is None:
            continue
        if measurement.kind == "timing":
            resolved[identifier] = {
                "kind": "timing",
                "unit": record["unit"],
                "samples": record["samples"],
                "summary": record["summary"],
            }
        else:
            resolved[identifier] = {
                "kind": "value",
                "unit": record["unit"],
                "value": record["value"],
                "observations": record.get("observations"),
            }
    return resolved


def export(run_dir: pathlib.Path, destination: pathlib.Path) -> None:
    module = load_reproduction_module()
    manifest = module.build_manifest()
    try:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExportError(f"missing summary: {run_dir / 'summary.json'}") from exc
    resolved = resolved_from_summary(module, manifest, summary)
    allow_pending = summary.get("mode") != "full"
    tables = module.render_result_tables(
        manifest, resolved, allow_pending=allow_pending
    )
    module.write_result_tables(destination, manifest, tables)
    validation = module.validate_result_dimensions(manifest, tables, resolved)
    module.atomic_write_json(destination / "result-validation.json", validation)


def compare_trees(expected: pathlib.Path, actual: pathlib.Path) -> None:
    expected_files = {
        path.relative_to(expected)
        for path in expected.rglob("*")
        if path.is_file()
        and (path.name == "result-validation.json" or "tables" in path.parts)
    }
    actual_files = {
        path.relative_to(actual)
        for path in actual.rglob("*")
        if path.is_file()
    }
    if expected_files != actual_files:
        raise ExportError(
            "derived result file set differs: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    for relative in sorted(expected_files):
        if (expected / relative).read_bytes() != (actual / relative).read_bytes():
            raise ExportError(f"derived result differs: {relative}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--write", action="store_true", help="write derived outputs")
    modes.add_argument("--check", action="store_true", help="check existing outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if args.write:
        export(run_dir, run_dir)
    else:
        with tempfile.TemporaryDirectory() as raw:
            generated = pathlib.Path(raw)
            export(run_dir, generated)
            compare_trees(run_dir, generated)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExportError as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
