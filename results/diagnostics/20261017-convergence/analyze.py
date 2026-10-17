"""Recompute stopping decisions from the interrupted run's archived checkpoint."""

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
import reproduce


def main() -> None:
    state = json.loads((ARCHIVE / "source-run" / "checkpoint.json").read_text())
    completed = [job for job in state["jobs"].values() if job["status"] == "completed"]
    sample_counts = Counter(
        max(map(len, job["samples"].values()), default=0) for job in completed
    )
    report = {
        "source_run": state["run_id"],
        "publication_results": False,
        "completed_jobs": len(completed),
        "completed_jobs_by_maximum_samples": dict(sorted(sample_counts.items())),
        "jobs": [],
    }

    for key, job in state["jobs"].items():
        batches = [batch for batch in job["batches"] if batch["status"] == "completed"]
        cursors = {metric: 0 for metric in job["samples"]}
        row = {
            "job": key,
            "status": job["status"],
            "maximum_samples": max(map(len, job["samples"].values()), default=0),
            "completed_batch_seconds": sum(batch["elapsed_seconds"] for batch in batches),
            "looks": [],
        }
        for batch in batches:
            # Each completed batch records how many retained samples it appended.
            for metric, count in batch["sample_counts"].items():
                cursors[metric] += count
            summaries = {
                metric: reproduce.summarize_samples(job["samples"][metric][:count])
                for metric, count in cursors.items()
                if count
            }
            misses = [
                {
                    "metric": metric,
                    "n": summary["n"],
                    "mean_ns": summary["mean"],
                    "relative_half_width": summary["sequential_ci"]["relative_half_width"],
                }
                for metric, summary in summaries.items()
                if not summary["stable"]
            ]
            misses.sort(key=lambda item: item["relative_half_width"], reverse=True)
            row["looks"].append({
                "batch": batch["index"],
                "counts": batch.get("group_repetitions", {}),
                "seconds": batch["elapsed_seconds"],
                "misses": misses,
            })
        report["jobs"].append(row)

    (ARCHIVE / "stopping-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Recomputed stopping decisions for", len(report["jobs"]), "scheduled jobs.")
    print("Completed jobs by maximum sample count:", report["completed_jobs_by_maximum_samples"])


if __name__ == "__main__":
    main()
