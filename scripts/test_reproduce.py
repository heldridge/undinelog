#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = pathlib.Path(__file__).with_name("reproduce.py")
SPEC = importlib.util.spec_from_file_location("reproduce", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


UNDINE_ONLY_BENCHMARKS = {
    "bench_append_only_log_on_the_fly_open_local",
    MODULE.PRIVATE_BENCHMARK,
}


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = MODULE.build_manifest()

    def test_all_expected_tables_and_cells_are_covered(self) -> None:
        self.assertEqual(set(self.manifest.table_rows), set(MODULE.TABLE_LABELS))
        self.assertEqual(set(MODULE.TABLE_METADATA), set(MODULE.TABLE_LABELS))
        consumers = self.manifest.consumers()
        self.assertTrue(self.manifest.cells)
        for cell in self.manifest.cells.values():
            self.assertTrue(cell.sources, cell.id)
            for source in cell.sources:
                self.assertIn(source, self.manifest.measurements)
        for measurement in self.manifest.measurements.values():
            self.assertTrue(consumers[measurement.id], measurement.id)

    def test_jobs_are_semantically_unique(self) -> None:
        keys = [job.semantic_key() for job in self.manifest.jobs.values()]
        self.assertEqual(len(keys), len(set(keys)))

    def test_both_layouts_run_the_same_workloads(self) -> None:
        by_layout = {
            layout: {
                job.id.replace(f"-{layout}-", "-"): job
                for job in self.manifest.jobs.values()
                if job.layout == layout and job.benchmark not in UNDINE_ONLY_BENCHMARKS
            }
            for layout in MODULE.LAYOUTS
        }
        self.assertEqual(set(by_layout["undine"]), set(by_layout["borg"]))
        for key, undine in by_layout["undine"].items():
            borg = by_layout["borg"][key]
            self.assertEqual(undine.benchmark, borg.benchmark)
            self.assertEqual(undine.features, borg.features)
            self.assertEqual(undine.trial_env, borg.trial_env)

            def workload(job):
                # The operating point and the layout-specific reconfiguration
                # metric list are the only differences.
                return {
                    name: value
                    for name, value in job.env
                    if not name.endswith(
                        ("_KAPPA", "_SEG", "_SEGMENT_LEN", "_S", "_SOURCE_S", "_RESCALE_METRICS")
                    )
                }

            self.assertEqual(workload(undine), workload(borg), key)

    def test_operating_points_have_segment_equal_kappa_squared(self) -> None:
        for job in self.manifest.jobs.values():
            env = job.env_dict()
            if env.get(f"{MODULE.PRIVATE_PREFIX}_ROLE") == "single":
                # The single server stores the tree itself: it has a record
                # size and no threshold.
                self.assertFalse(any(name.endswith("_KAPPA") for name in env), job.id)
                continue
            kappa = int(job.id.rsplit("-k", 1)[1])
            if job.layout == "undine":
                thresholds = [value for name, value in env.items() if name.endswith("_KAPPA")]
                self.assertEqual(thresholds, [str(kappa)], job.id)
                self.assertFalse(
                    any(name.endswith(("_SEG", "_SEGMENT_LEN", "_S")) for name in env), job.id
                )
            else:
                segments = [
                    int(value)
                    for name, value in env.items()
                    if name.endswith(("_SEG", "_SEGMENT_LEN"))
                ]
                self.assertEqual(segments, [kappa * kappa], job.id)

    def test_on_the_fly_opening_is_measured_for_undine_only(self) -> None:
        jobs = [
            job
            for job in self.manifest.jobs.values()
            if job.benchmark == "bench_append_only_log_on_the_fly_open_local"
        ]
        self.assertEqual(len(jobs), len(MODULE.KAPPAS))
        self.assertTrue(all(job.layout == "undine" for job in jobs))

    def test_private_openings_are_measured_for_undine_only(self) -> None:
        jobs = [
            job
            for job in self.manifest.jobs.values()
            if job.benchmark == MODULE.PRIVATE_BENCHMARK
        ]
        self.assertTrue(all(job.layout == "undine" for job in jobs))
        roles = [job.env_dict()[f"{MODULE.PRIVATE_PREFIX}_ROLE"] for job in jobs]
        self.assertEqual(
            {role: roles.count(role) for role in set(roles)},
            {
                "maintainer": len(MODULE.PRIVATE_LOGS_GIB)
                + len(MODULE.KAPPAS)
                - 1
                + len(MODULE.PRIVATE_PATTERN_SWEEP_OFFSETS)
                + len(MODULE.PRIVATE_LARGE_POINTS),
                "single": 1
                + len(MODULE.PRIVATE_SINGLE_PATTERN_OFFSETS)
                + len(MODULE.PRIVATE_SINGLE_MEASURED_GIB)
                - 1,
                "table": len(MODULE.PRIVATE_TABLE_LOGS_GIB),
                "client": len(MODULE.KAPPAS) + len(MODULE.PRIVATE_CLIENT_LOGS_GIB),
            },
        )

    def test_private_patterns_follow_the_public_grouping(self) -> None:
        bits = MODULE.private_pattern_bits
        # kappa = 64: patterns of 2^13 to 2^16 leaves on logs of 1 to 64 GiB.
        self.assertEqual(
            [bits(MODULE.leaves_of_gib(gib), 64) for gib in (1, 4, 16, 64)],
            [13, 14, 15, 16],
        )
        # Smaller thresholds on the 4 GiB log, and the smallest pattern.
        self.assertEqual(
            [bits(MODULE.leaves_of_gib(4), kappa) for kappa in (8, 16, 32)],
            [12, 13, 13],
        )
        self.assertEqual(bits(3 * MODULE.BATCH_VALUES_4MIB, 64), 11)
        self.assertEqual(bits(MODULE.leaves_of_gib(1), 128), 13)
        # The single server is threshold 1.
        self.assertEqual(bits(MODULE.leaves_of_gib(1), 1), 10)
        self.assertEqual(bits(MODULE.leaves_of_gib(4), 1), 11)
        # A record of the 64 GiB log at kappa = 64 holds 33 columns; the
        # estimate allows 24 upper entries.
        self.assertEqual(MODULE.private_reply_ciphertexts(16, 64 * 64, 64), 20)
        sweep = {
            bits(MODULE.leaves_of_gib(MODULE.PRIVATE_PATTERN_SWEEP_GIB), MODULE.PRIVATE_KAPPA)
            + offset
            for offset in MODULE.PRIVATE_PATTERN_SWEEP_OFFSETS
        }
        for job in self.manifest.jobs.values():
            env = job.env_dict()
            role = env.get(f"{MODULE.PRIVATE_PREFIX}_ROLE")
            if role is None:
                continue
            leaves = int(env[f"{MODULE.PRIVATE_PREFIX}_LEAVES"])
            pattern = int(env[f"{MODULE.PRIVATE_PREFIX}_PATTERN_BITS"])
            if role == "single":
                self.assertEqual(
                    int(env[f"{MODULE.PRIVATE_PREFIX}_CELLS_PER_RECORD"]),
                    MODULE.PRIVATE_SINGLE_GROUP_CELLS,
                )
                self.assertIn(
                    pattern - bits(leaves, 1),
                    (0, *MODULE.PRIVATE_SINGLE_PATTERN_OFFSETS),
                    job.id,
                )
                continue
            kappa = int(env[f"{MODULE.PRIVATE_PREFIX}_KAPPA"])
            self.assertEqual(int(env[f"{MODULE.PRIVATE_PREFIX}_SEGMENTS_PER_RECORD"]), 1)
            if pattern != bits(leaves, kappa):
                self.assertEqual(role, "maintainer", job.id)
                self.assertIn(pattern, sweep, job.id)

    def test_fragment_sizes_agree_between_layouts_at_segment_kappa_squared(self) -> None:
        for kappa in MODULE.KAPPAS:
            self.assertEqual(
                MODULE.stored_elements_per_segment("undine", kappa * kappa, kappa),
                MODULE.stored_elements_per_segment("borg", kappa * kappa, kappa),
            )

    def test_jobs_mode_selects_named_jobs_with_full_sampling(self) -> None:
        names = ["private-client-undine-k8", "append-4mib-borg-k16", "private-client-undine-k8"]
        selected = MODULE.select_run_jobs(self.manifest, "jobs", names)
        self.assertEqual([run_job.spec.id for run_job in selected], names[:2])
        for run_job in selected:
            self.assertEqual(
                run_job.measurements,
                tuple(self.manifest.measurements_for_job(run_job.spec.id)),
            )
        self.assertEqual(
            MODULE.expected_repetitions_for_mode("jobs"),
            MODULE.expected_repetitions_for_mode("full"),
        )
        with self.assertRaises(MODULE.PipelineError):
            MODULE.select_run_jobs(self.manifest, "jobs", ["no-such-job"])
        with self.assertRaises(MODULE.PipelineError):
            MODULE.select_run_jobs(self.manifest, "jobs", [])

    def test_exact_full_job_count_is_stable(self) -> None:
        self.assertEqual(len(self.manifest.jobs), 72)
        self.assertEqual(len(self.manifest.cells), 763)

    def test_every_table_cell_satisfies_its_unit_contract(self) -> None:
        for cell in self.manifest.cells.values():
            MODULE.validate_cell_unit_contract(cell, self.manifest)

    def test_renderer_rejects_incompatible_source_unit(self) -> None:
        manifest = MODULE.Manifest()
        source = manifest.add_measurement(
            MODULE.MeasurementSpec(
                id="bytes",
                job_id=None,
                kind="formula",
                formula="integer",
                formula_args=(1,),
                unit="bytes",
            )
        )
        cell = MODULE.CellSpec(
            table="table",
            row="row",
            column="column",
            sources=(source,),
            render="milliseconds-3",
            description="invalid unit pairing",
        )
        with self.assertRaisesRegex(MODULE.PipelineError, "has unit 'bytes'"):
            MODULE.validate_cell_unit_contract(cell, manifest)


class StatisticsTests(unittest.TestCase):
    def test_student_t_reference_quantiles(self) -> None:
        self.assertAlmostEqual(MODULE.student_t_ppf(0.975, 4), 2.776445, places=5)
        self.assertAlmostEqual(MODULE.student_t_ppf(0.975, 19), 2.093024, places=5)

    def test_zero_variance_reaches_target_at_first_look(self) -> None:
        result = MODULE.summarize_samples(
            [100.0] * MODULE.DEFAULT_MIN_REPETITIONS
        )
        self.assertTrue(result["stable"])
        self.assertEqual(result["sequential_ci"]["half_width"], 0.0)

    def test_noisy_result_is_flagged(self) -> None:
        result = MODULE.summarize_samples([1.0, 100.0, 1.0, 100.0, 1.0])
        self.assertFalse(result["stable"])
        self.assertGreater(
            result["sequential_ci"]["relative_half_width"],
            MODULE.DEFAULT_RELATIVE_HALF_WIDTH,
        )

    def test_sequential_interval_is_more_conservative_than_plain_ci(self) -> None:
        result = MODULE.summarize_samples([10, 11, 9, 10, 12])
        self.assertGreater(
            result["sequential_ci"]["half_width"],
            result["ci95"]["half_width"],
        )


class ParsingTests(unittest.TestCase):
    @staticmethod
    def stats(samples: list[int]) -> dict[str, object]:
        ordered = sorted(samples)
        return {
            "mean": sum(samples) // len(samples),
            "p50": ordered[len(ordered) // 2],
            "p95": ordered[-1],
            "trials": len(samples),
            "samples": samples,
        }

    def test_append_blocks_become_sequence_repetitions(self) -> None:
        measurement = MODULE.MeasurementSpec(
            id="append",
            job_id="job",
            selector=MODULE.selector(segment_len=64, need=8),
            path=("timing",),
            kind="timing",
            sample_mode="append-sequence",
        )
        document = {
            "meta": {
                "seed": 42,
                "trials": 2,
                "sequence_blocks": 3,
                "sequence_warmup_blocks": 1,
            },
            "entries": [
                {
                    "segment_len": 64,
                    "need": 8,
                    "timing": self.stats([10, 20, 30, 50]),
                }
            ],
        }
        timing, values, seed, raw = MODULE.extract_measurements(
            document, [measurement]
        )
        self.assertEqual(timing["append"], [15.0, 40.0])
        self.assertEqual(raw["append"], [10.0, 20.0, 30.0, 50.0])
        self.assertEqual(values, {})
        self.assertEqual(seed, 42)

    def test_process_warmup_repetitions_are_not_independent_samples(self) -> None:
        measurement = MODULE.MeasurementSpec(
            id="used",
            job_id="job",
            selector=MODULE.selector(key=1),
            path=("used",),
            kind="timing",
        )
        document = {
            "meta": {"seed": 42, "trials": 5},
            "entries": [
                {
                    "key": 1,
                    "used": self.stats([1000, 900, 800, 10, 11]),
                }
            ],
        }
        timing, _, _, raw = MODULE.extract_measurements(
            document,
            [measurement],
            warmup_repetitions=3,
        )
        self.assertEqual(raw["used"], [1000.0, 900.0, 800.0, 10.0, 11.0])
        self.assertEqual(timing["used"], [10.0, 11.0])

    def test_append_warmup_discards_complete_sequences(self) -> None:
        measurement = MODULE.MeasurementSpec(
            id="append",
            job_id="job",
            selector=MODULE.selector(key=1),
            path=("timing",),
            kind="timing",
            sample_mode="append-sequence",
        )
        document = {
            "meta": {
                "seed": 42,
                "trials": 3,
                "sequence_blocks": 2,
                "sequence_warmup_blocks": 0,
            },
            "entries": [
                {
                    "key": 1,
                    "timing": self.stats([100, 200, 10, 20, 30, 40]),
                }
            ],
        }
        timing, _, _, _ = MODULE.extract_measurements(
            document,
            [measurement],
            warmup_repetitions=1,
        )
        self.assertEqual(timing["append"], [15.0, 35.0])

    def test_unconsumed_timing_is_rejected(self) -> None:
        measurement = MODULE.MeasurementSpec(
            id="used",
            job_id="job",
            selector=MODULE.selector(key=1),
            path=("used",),
            kind="timing",
        )
        document = {
            "meta": {"seed": 42, "trials": 1},
            "entries": [
                {
                    "key": 1,
                    "used": self.stats([10]),
                    "unused": self.stats([20]),
                }
            ],
        }
        with self.assertRaises(MODULE.PipelineError):
            MODULE.extract_measurements(document, [measurement])

    def test_direct_trial_count_mismatch_is_rejected(self) -> None:
        measurement = MODULE.MeasurementSpec(
            id="used",
            job_id="job",
            selector=MODULE.selector(key=1),
            path=("used",),
            kind="timing",
        )
        document = {
            "meta": {"seed": 42, "trials": 2},
            "entries": [{"key": 1, "used": self.stats([10])}],
        }
        with self.assertRaises(MODULE.PipelineError):
            MODULE.extract_measurements(document, [measurement])

    def test_nonfinite_timing_sample_is_rejected(self) -> None:
        measurement = MODULE.MeasurementSpec(
            id="used",
            job_id="job",
            selector=MODULE.selector(key=1),
            path=("used",),
            kind="timing",
            unit="ns",
        )
        document = {
            "meta": {"seed": 42, "trials": 1},
            "entries": [{"key": 1, "used": self.stats([float("nan")])}],
        }
        with self.assertRaisesRegex(MODULE.PipelineError, "invalid ns value"):
            MODULE.extract_measurements(document, [measurement])


class CheckpointAndResumeTests(unittest.TestCase):
    def test_checkpoint_round_trip_is_atomic_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run_dir = pathlib.Path(raw)
            state = {
                "pipeline_version": MODULE.PIPELINE_VERSION,
                "snapshot_id": "abc",
                "nested": {"samples": [1, 2, 3]},
            }
            MODULE.write_checkpoint(run_dir, state)
            loaded = MODULE.load_checkpoint(run_dir)
            self.assertEqual(loaded["snapshot_id"], "abc")
            self.assertEqual(loaded["nested"]["samples"], [1, 2, 3])
            leftovers = list(run_dir.glob(".checkpoint.json.*"))
            self.assertEqual(leftovers, [])

    def test_stale_source_snapshot_is_rejected(self) -> None:
        state = {
            "pipeline_version": MODULE.PIPELINE_VERSION,
            "snapshot_id": "old",
        }
        with self.assertRaises(MODULE.PipelineError):
            MODULE.validate_resume_snapshot(state, {"snapshot_id": "new"})

    def test_pipeline_version_change_is_rejected(self) -> None:
        state = {
            "pipeline_version": MODULE.PIPELINE_VERSION - 1,
            "snapshot_id": "same",
        }
        with self.assertRaises(MODULE.PipelineError):
            MODULE.validate_resume_snapshot(state, {"snapshot_id": "same"})

    def test_maximum_repetitions_complete_but_remain_unstable(self) -> None:
        samples = [1.0, 100.0] * (MODULE.DEFAULT_MAX_REPETITIONS // 2)
        state = {"samples": {"metric": samples}}
        self.assertTrue(MODULE.job_complete(state, "full"))
        self.assertFalse(MODULE.summarize_samples(samples)["stable"])

    def test_batches_advance_to_declared_looks(self) -> None:
        state = {"samples": {"metric": []}}
        expected = []
        current = 0
        for look in MODULE.SEQUENTIAL_LOOKS:
            repetitions = MODULE.batch_repetitions(state, "full")
            expected.append(repetitions)
            state["samples"]["metric"].extend([1.0] * repetitions)
            current += repetitions
            self.assertEqual(current, look)
        self.assertEqual(expected, [10, 10, 20, 40, 80])
        self.assertEqual(MODULE.batch_repetitions(state, "full"), 0)

    def test_resume_rejects_changed_saved_binary(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run_dir = pathlib.Path(raw)
            binary = run_dir / "binaries" / "key" / "bench"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"changed")
            state = {
                "binaries": {
                    "key": {
                        "path": str(binary.relative_to(run_dir)),
                        "sha256": MODULE.sha256_bytes(b"original"),
                    }
                },
                "jobs": {},
            }
            with self.assertRaisesRegex(
                MODULE.PipelineError, "saved binary hash changed"
            ):
                MODULE.validate_resume_artifacts(run_dir, state, [])

    def test_source_drift_is_rejected(self) -> None:
        with mock.patch.object(
            MODULE,
            "collect_source_snapshot",
            return_value={"snapshot_id": "changed"},
        ):
            with self.assertRaisesRegex(
                MODULE.PipelineError, "source snapshot changed"
            ):
                MODULE.validate_current_source_snapshot(
                    "expected",
                    phase="unit test",
                )

    def test_benchmarks_complete_remains_resumable_until_finalized(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run_dir = pathlib.Path(raw)
            state = {
                "mode": "full",
                "status": "running",
                "snapshot_id": "snapshot",
                "jobs": {},
            }
            completed = MODULE.execute_run(
                MODULE.Manifest(),
                run_dir,
                state,
                [],
                timeout_seconds=1,
                stop_after_batches=None,
            )
            self.assertTrue(completed)
            self.assertEqual(state["status"], "benchmarks-complete")
            self.assertNotEqual(state["status"], "completed")

            MODULE.mark_run_completed(run_dir, state)
            self.assertEqual(state["status"], "completed")
            self.assertIn("finished_utc", state)


class RuntimeEstimateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = MODULE.build_manifest()

    def test_canary_calibration_covers_every_full_family(self) -> None:
        selected = MODULE.select_run_jobs(self.manifest, "canary")
        state = {
            "selected_jobs": [run_job.spec.id for run_job in selected],
            "jobs": {
                run_job.spec.id: {
                    "batches": [{"status": "completed", "elapsed_seconds": 1.0}]
                }
                for run_job in selected
            },
        }
        estimate = MODULE.calibrated_runtime_estimate(self.manifest, state)
        expected_families = {
            job.family for job in self.manifest.jobs.values()
        }
        self.assertEqual(set(estimate["families"]), expected_families)
        self.assertGreater(estimate["minimum_seconds"], 0)
        self.assertGreater(
            estimate["maximum_seconds"], estimate["minimum_seconds"]
        )


class ResultOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = MODULE.build_manifest()

    def test_pending_result_groups_have_exact_dimensions(self) -> None:
        resolved = {
            identifier: {
                "kind": "formula",
                "value": MODULE.evaluate_formula(spec),
                "unit": spec.unit,
            }
            for identifier, spec in self.manifest.measurements.items()
            if spec.kind == "formula"
        }
        tables = MODULE.render_result_tables(
            self.manifest, resolved, allow_pending=True
        )
        evidence = MODULE.validate_result_dimensions(self.manifest, tables)
        self.assertEqual(set(evidence), set(MODULE.TABLE_LABELS))
        self.assertTrue(
            all(record["unit_checked_cells"] > 0 for record in evidence.values())
        )

    def test_result_groups_write_json_csv_and_markdown(self) -> None:
        tables = MODULE.render_result_tables(self.manifest, {}, allow_pending=True)
        with tempfile.TemporaryDirectory() as raw:
            run_dir = pathlib.Path(raw)
            MODULE.write_result_tables(run_dir, self.manifest, tables)
            output_dir = run_dir / "tables"
            self.assertTrue((output_dir / "tables.json").is_file())
            self.assertTrue((output_dir / "README.md").is_file())
            self.assertEqual(
                len(list(output_dir.glob("tab-*.csv"))),
                len(MODULE.TABLE_LABELS),
            )
            self.assertFalse(list(output_dir.glob("*.tex")))

    def test_smoke_only_measurement_is_retained_in_summary_inputs(self) -> None:
        state = {
            "selected_jobs": ["job"],
            "run_jobs": {
                "job": {
                    "selectors": {
                        "smoke:metric": {
                            "kind": "timing",
                            "unit": "ns",
                        }
                    }
                }
            },
            "jobs": {
                "job": {
                    "samples": {"smoke:metric": [123.0]},
                    "values": {},
                }
            },
        }
        resolved = MODULE.resolve_measurements(MODULE.Manifest(), state)
        self.assertEqual(resolved["smoke:metric"]["samples"], [123.0])
        self.assertEqual(resolved["smoke:metric"]["unit"], "ns")

    def test_timing_renderers_scale_nanoseconds(self) -> None:
        def render(name: str, value: float) -> str:
            cell = MODULE.CellSpec("table", "row", "column", (), name, "")
            return MODULE.render_cell_value(cell, [value])

        self.assertEqual(render("microseconds-1", 12_345), "12.3")
        self.assertEqual(render("milliseconds-3", 12_345_000), "12.345")
        self.assertEqual(render("milliseconds-3-div3", 36_000_000), "12.000")
        self.assertEqual(render("seconds-3", 5_610_000_000), "5.610")
        self.assertEqual(render("seconds-1", 5_660_000_000), "5.7")

    def test_preprocessing_renderer_scales_its_two_samples(self) -> None:
        cell = MODULE.CellSpec(
            "table", "row", "column", (), "preprocess-core-seconds-1", ""
        )
        # 64 sampled columns in 2 s, 6400 record columns; 1 s per table, 13
        # reply ciphertexts.
        self.assertEqual(
            MODULE.render_cell_value(
                cell, [2_000_000_000, 6400, 64, 1_000_000_000, 13]
            ),
            "213.0",
        )

    def test_size_renderers_scale_bytes(self) -> None:
        def render(name: str, value: float) -> str:
            cell = MODULE.CellSpec("table", "row", "column", (), name, "")
            return MODULE.render_cell_value(cell, [value])

        self.assertEqual(render("kib-1", 1536), "1.5")
        self.assertEqual(render("mib-3", 3 * 1024 * 1024), "3.000")
        self.assertEqual(render("mib-3-div3", 3 * 1024 * 1024), "1.000")
        self.assertEqual(render("mib-1", 3 * 1024 * 1024 + 512 * 1024), "3.5")
        self.assertEqual(render("gib-1", 5 * 1024**3 // 2), "2.5")


if __name__ == "__main__":
    unittest.main(verbosity=2)
