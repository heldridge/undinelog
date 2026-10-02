#!/usr/bin/env python3
"""Record or check where the two measured crates come from.

`provenance/sources.json` lists every measured source file of `undine/` and
`borg/` with its size and SHA-256, next to the origin of each crate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "provenance" / "sources.json"
CRATES = ("undine", "borg")


class ProvenanceError(RuntimeError):
    pass


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crate_file_records(crate: str) -> dict[str, dict[str, object]]:
    root = ROOT / crate
    paths = [root / "Cargo.toml", root / "Cargo.lock"]
    for base in ("src", "benches", "apps/appendOnlyLog", "third_party"):
        paths.extend(
            path
            for path in (root / base).rglob("*")
            if path.is_file() and "target" not in path.relative_to(root).parts
        )
    return {
        path.relative_to(ROOT).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    }


def write(origins: dict[str, dict[str, object]]) -> None:
    payload = {
        "schema_version": 1,
        "crates": {
            crate: origins[crate] | {"files": crate_file_records(crate)}
            for crate in CRATES
        },
    }
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def check() -> int:
    try:
        payload = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProvenanceError(f"missing {PROVENANCE.relative_to(ROOT)}") from exc
    checked = 0
    for crate in CRATES:
        recorded = payload.get("crates", {}).get(crate, {}).get("files")
        if not isinstance(recorded, dict):
            raise ProvenanceError(f"no recorded files for {crate}")
        actual = crate_file_records(crate)
        if recorded != actual:
            changed = sorted(
                name
                for name in set(recorded) | set(actual)
                if recorded.get(name) != actual.get(name)
            )
            raise ProvenanceError(
                f"{crate} sources differ from their recorded origin: {changed}"
            )
        checked += len(actual)
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument(
        "--write",
        type=pathlib.Path,
        metavar="ORIGINS_JSON",
        help="JSON object mapping each crate to its origin description",
    )
    args = parser.parse_args()
    if args.check:
        print(f"source provenance checked: {check()} files")
    else:
        write(json.loads(args.write.read_text(encoding="utf-8")))
        print(f"source provenance written: {PROVENANCE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProvenanceError as exc:
        print(f"source provenance check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
