#!/usr/bin/env python3
"""Split a run's wall time per job into timed component samples and untimed work.

Usage: python3 analyze.py RUN_DIR [OUT_JSON]

Timed seconds sum the retained samples of component series; aggregate series
(prepare, preprocess, fill, any total) are left out so that stages are not
counted twice. Append samples are per-block means over 64 blocks and helper
samples average 32 calls, so their timed share is scaled by that factor.
Untimed seconds are the batch's elapsed time minus its timed seconds: setup,
discarded warmups, checks and verification.
"""
import json, pathlib, sys

AGGREGATE = ("prepare_ns", "preprocess_ns", "fill_ns")


def component(series: str) -> bool:
    name = series.rsplit(":", 1)[1]
    return not (name.endswith(AGGREGATE) or "total" in name or name.startswith("paired-"))


def main() -> int:
    run = pathlib.Path(sys.argv[1])
    checkpoint = json.load(open(run / "checkpoint.json"))
    manifest = json.load(open(run / "manifest.json"))
    benchmarks = {job["id"]: job["benchmark"] for job in manifest["jobs"]}
    rows = []
    for name, state in checkpoint["jobs"].items():
        samples = state["samples"]
        inner = 64 if "bulk_updates_append" in benchmarks[name] else 1
        if "helper" in name and "range" in name:
            inner = 32
        seen = {key: 0 for key in samples}
        batches = []
        for batch in state["batches"]:
            timed = 0.0
            for key, count in batch.get("sample_counts", {}).items():
                new = samples[key][seen[key]:seen[key] + count]
                seen[key] += count
                if component(key):
                    timed += inner * sum(new) / 1e9
            batches.append({
                "index": batch["index"],
                "elapsed_seconds": batch["elapsed_seconds"],
                "timed_seconds": timed,
                "untimed_seconds": batch["elapsed_seconds"] - timed,
                "group_repetitions": batch.get("group_repetitions"),
                "group_warmups": batch.get("group_warmups"),
            })
        rows.append({
            "job": name,
            "benchmark": benchmarks[name],
            "elapsed_seconds": sum(b["elapsed_seconds"] for b in batches),
            "timed_seconds": sum(b["timed_seconds"] for b in batches),
            "untimed_seconds": sum(b["untimed_seconds"] for b in batches),
            "batches": batches,
        })
    rows.sort(key=lambda row: -row["untimed_seconds"])
    summary = {
        "run": run.name,
        "elapsed_hours": sum(r["elapsed_seconds"] for r in rows) / 3600,
        "timed_hours": sum(r["timed_seconds"] for r in rows) / 3600,
        "untimed_hours": sum(r["untimed_seconds"] for r in rows) / 3600,
        "jobs": rows,
    }
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if out:
        out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{run.name}: elapsed {summary['elapsed_hours']:.1f} h, timed {summary['timed_hours']:.1f} h, untimed {summary['untimed_hours']:.1f} h")
    for row in rows[:12]:
        print(f"  {row['untimed_seconds']/60:7.1f} min untimed of {row['elapsed_seconds']/60:7.1f}  {row['job']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
