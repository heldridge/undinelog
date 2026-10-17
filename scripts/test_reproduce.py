#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import dataclasses
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import check_artifact
import verify_results


SCRIPT = pathlib.Path(__file__).with_name("reproduce.py")
SPEC = importlib.util.spec_from_file_location("reproduce", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


UNDINE_ONLY_BENCHMARKS = {
    "bench_append_only_log_on_the_fly_open_local",
    MODULE.PRIVATE_BENCHMARK,
    MODULE.TREEPIR_BENCHMARK,
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
            if job.benchmark == MODULE.TREEPIR_BENCHMARK:
                # The threshold is the height of the tree, kappa = h.
                height = int(env[f"{MODULE.TREEPIR_PREFIX}_HEIGHT"])
                self.assertIn(f"-h{height}-", job.id)
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

    def test_treepir_comparison_runs_at_the_matched_point(self) -> None:
        jobs = [
            job
            for job in self.manifest.jobs.values()
            if job.benchmark == MODULE.TREEPIR_BENCHMARK
        ]
        heights = {
            (job.env_dict()[f"{MODULE.TREEPIR_PREFIX}_ROLE"],
             int(job.env_dict()[f"{MODULE.TREEPIR_PREFIX}_HEIGHT"]))
            for job in jobs
        }
        self.assertEqual(
            heights,
            {
                ("server", MODULE.TREEPIR_HEIGHT),
                ("client", MODULE.TREEPIR_HEIGHT),
                ("client", MODULE.TREEPIR_ARTIFACT_HEIGHT),
            },
        )
        rebuilds = {
            measurement.sample_mode
            for job in jobs
            if job.env_dict()[f"{MODULE.TREEPIR_PREFIX}_ROLE"] == "server"
            for measurement in self.manifest.measurements_for_job(job.id)
            if measurement.kind == "timing"
        }
        self.assertEqual(rebuilds, {"rebuild", "direct"})

    def test_expensive_group_stops_independently_of_online_noise(self) -> None:
        job = next(job for job in self.manifest.jobs.values()
                   if job.benchmark == MODULE.PRIVATE_BENCHMARK
                   and job.env_dict()[f"{MODULE.PRIVATE_PREFIX}_ROLE"] == "maintainer")
        run_job = MODULE.RunJob(job, tuple(self.manifest.measurements_for_job(job.id)))
        samples = {item.id: ([1., 100.] * 10 if item.sample_group == "online" else [100.] * 10)
                   for item in run_job.measurements if item.kind == "timing"}
        counts, warmups = MODULE.group_batch_plan(run_job, {"samples": samples}, "full")
        self.assertEqual(counts, {"preprocess": 0, "online": 20, "update": 0})
        self.assertEqual(warmups, {"preprocess": 0, "online": 3, "update": 0})
        self.assertFalse(MODULE.job_complete({"samples": samples}, "full"))
        for item in run_job.measurements:
            if item.kind == "timing" and item.sample_group == "online":
                samples[item.id] = [1., 100.] * 80
        counts, warmups = MODULE.group_batch_plan(run_job, {"samples": samples}, "full")
        self.assertEqual(counts, {"preprocess": 0, "online": 160, "update": 0})
        self.assertEqual(warmups, {"preprocess": 0, "online": 3, "update": 0})
        for item in run_job.measurements:
            if item.kind == "timing" and item.sample_group == "online":
                samples[item.id] = [1., 100.] * (MODULE.DEFAULT_MAX_REPETITIONS // 2)
        self.assertTrue(MODULE.job_complete({"samples": samples}, "full"))

    def test_treepir_class_sizes_cover_the_tree(self) -> None:
        for height in (1, 2, 3, 8, 16, 32):
            sizes = [MODULE.treepir_class_size(height, color) for color in range(height)]
            self.assertEqual(sum(sizes), (1 << (height + 1)) - 2)
            self.assertLessEqual(max(sizes) - min(sizes), 1)
            self.assertEqual(sizes, sorted(sizes))

    def test_quick_gates_repeat_real_generated_reply_pairs(self) -> None:
        for mode in ("smoke", "canary"):
            with self.subTest(mode=mode):
                run_job = next(
                    job for job in MODULE.select_run_jobs(self.manifest, mode)
                    if job.spec.id == "treepir-client-h32-generated"
                )
                samples = {
                    item.id: [] for item in run_job.measurements if item.kind == "timing"
                }
                counts, warmups = MODULE.group_batch_plan(run_job, {"samples": samples}, mode)
                self.assertEqual(counts, {"client": 1, "paired": 10})
                self.assertEqual(warmups, {"client": 0, "paired": 3})
                for item in run_job.measurements:
                    if item.kind == "timing":
                        samples[item.id] = [100.] * counts[item.sample_group]
                counts, warmups = MODULE.group_batch_plan(run_job, {"samples": samples}, mode)
                self.assertEqual(counts, {"client": 0, "paired": 0})
                self.assertEqual(warmups, counts)

    def test_full_client_sampling_keeps_its_first_look(self) -> None:
        for job_id in ("treepir-client-h16-real", "treepir-client-h32-generated"):
            job = self.manifest.jobs[job_id]
            run_job = MODULE.RunJob(job, tuple(self.manifest.measurements_for_job(job.id)))
            state = {"samples": {
                item.id: [] for item in run_job.measurements if item.kind == "timing"
            }}
            groups = {item.sample_group for item in run_job.measurements if item.kind == "timing"}
            for mode in ("full", "jobs"):
                with self.subTest(job=job_id, mode=mode):
                    counts, warmups = MODULE.group_batch_plan(run_job, state, mode)
                    self.assertEqual(counts, {group: 10 for group in groups})
                    self.assertEqual(warmups, {group: 3 for group in groups})

    def test_client_substeps_cannot_restart_converged_totals(self) -> None:
        for job_id in ("treepir-client-h16-real", "treepir-client-h32-generated"):
            with self.subTest(job=job_id):
                job = self.manifest.jobs[job_id]
                run_job = MODULE.RunJob(job, tuple(self.manifest.measurements_for_job(job.id)))
                samples = {
                    item.id: ([1., 100.] * 5 if not item.precision_required else [100.] * 10)
                    for item in run_job.measurements if item.kind == "timing"
                }
                groups = {item.sample_group for item in run_job.measurements if item.kind == "timing"}
                stopped = {group: 0 for group in groups}
                counts, _ = MODULE.group_batch_plan(run_job, {"samples": samples}, "full")
                self.assertEqual(counts, stopped)
                self.assertTrue(MODULE.job_complete({"samples": samples}, "full", run_job.measurements))
                required = {item.path for item in run_job.measurements
                            if item.kind == "timing" and dict(item.selector)["role"] == "client"
                            and item.precision_required}
                self.assertEqual(required, {("request_total_ns",), ("process_total_ns",), ("client_total_ns",)})
                # Changing the stopping decision must not alter the two-opening workload.
                self.assertTrue(all(item.sample_group == "client" for item in run_job.measurements
                                    if item.kind == "timing" and dict(item.selector)["role"] == "client"))
                request = next(item for item in run_job.measurements if item.path == ("request_total_ns",))
                samples[request.id] = [1., 100.] * 5
                counts, _ = MODULE.group_batch_plan(run_job, {"samples": samples}, "full")
                self.assertEqual(counts, stopped | {"client": 10})
                self.assertFalse(MODULE.job_complete({"samples": samples}, "full", run_job.measurements))

    def test_server_gather_cannot_restart_converged_combined_costs(self) -> None:
        job = self.manifest.jobs["treepir-server-h32-m31-c31"]
        run_job = MODULE.RunJob(job, tuple(self.manifest.measurements_for_job(job.id)))
        samples = {item.id: ([1., 100.] * 5 if not item.precision_required else [100.] * 10)
                   for item in run_job.measurements if item.kind == "timing"}
        counts, _ = MODULE.group_batch_plan(run_job, {"samples": samples}, "full")
        self.assertEqual(counts, {"preprocess": 0, "online": 0})
        for item in run_job.measurements:
            if item.path == ("fill_ns",):
                samples[item.id] = [1., 100.] * 5
        counts, _ = MODULE.group_batch_plan(run_job, {"samples": samples}, "full")
        self.assertEqual(counts, {"preprocess": 10, "online": 0})

    def test_manifest_rejects_groups_with_no_required_precision(self) -> None:
        manifest = MODULE.build_manifest()
        for item in manifest.measurements_for_job("treepir-client-h32-generated"):
            if item.kind == "timing" and item.sample_group == "client":
                manifest.measurements[item.id] = dataclasses.replace(item, precision_required=False)
        with self.assertRaisesRegex(MODULE.PipelineError, "has no precision requirement"):
            MODULE.validate_manifest(manifest)

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
                "client": len(MODULE.KAPPAS)
                + len(MODULE.PRIVATE_CLIENT_LARGE_KAPPAS)
                + len(MODULE.PRIVATE_CLIENT_LARGE_LOGS_GIB),
            },
        )

    def test_private_preprocessing_uses_full_rebuild_measurements(self) -> None:
        for table in ("tab:private-maintainer", "tab:private-single"):
            for row in self.manifest.table_rows[table]:
                if row.startswith("shape:"):
                    continue
                time = self.manifest.cells[f"{table}|{row}|preprocess-core-s"]
                self.assertEqual(time.render, "seconds-1")
                self.assertEqual(len(time.sources), 1)
                measurement = self.manifest.measurements[time.sources[0]]
                self.assertEqual(measurement.path, ("preprocess_ns",))
                self.assertEqual(measurement.kind, "timing")
                self.assertEqual(measurement.sample_mode, "rebuild")
                job = self.manifest.jobs[measurement.job_id]
                run_job = MODULE.RunJob(job, tuple(self.manifest.measurements_for_job(job.id)))
                state = {"samples": {item.id: [] for item in run_job.measurements if item.kind == "timing"}}
                counts, warmups = MODULE.group_batch_plan(run_job, state, "full")
                self.assertEqual(counts["preprocess"], MODULE.DEFAULT_MIN_REPETITIONS)
                self.assertEqual(warmups["preprocess"], MODULE.REBUILD_WARMUP_REPETITIONS)
                self.assertEqual(warmups["online"], MODULE.DIRECT_WARMUP_REPETITIONS)
                self.assertIn(f"{MODULE.PRIVATE_PREFIX}_HINT_CHECK_COLUMNS", job.env_dict())
                self.assertNotIn(f"{MODULE.PRIVATE_PREFIX}_HINT_SAMPLE_COLUMNS", job.env_dict())
                size = self.manifest.cells[f"{table}|{row}|preprocess-size"]
                self.assertEqual(self.manifest.measurements[size.sources[0]].path, ("preprocessed_bytes",))

    def test_private_preprocessing_rejects_sampled_or_inconsistent_results(self) -> None:
        def metric(values):
            return {"samples": values, "trials": len(values)}

        entry = {
            "hint_ns": metric([100, 110]),
            "packing_ns": metric([200, 190]),
            "preprocess_ns": metric([300, 300]),
            "outputs": 2,
            "preprocessed_bytes": 2 * (2048 * 2048 * 8 + 100_859_904),
        }
        MODULE.validate_private_preprocessing(entry, 2, "test")
        for field, invalid in (
            ("hint_ns", None),
            ("packing_ns", metric([200])),
            ("preprocess_ns", metric([300, 301])),
            ("preprocessed_bytes", entry["preprocessed_bytes"] // 2),
        ):
            with self.subTest(field=field), self.assertRaises(MODULE.PipelineError):
                MODULE.validate_private_preprocessing(entry | {field: invalid}, 2, "test")

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
        self.assertEqual(len(self.manifest.jobs), 78)
        self.assertEqual(len(self.manifest.cells), 1037)
        self.assertEqual(
            sum(item.kind == "timing" for item in self.manifest.measurements.values()),
            484,
        )
        self.assertEqual(sum(item.kind == "timing" and item.precision_required
                             for item in self.manifest.measurements.values()), 440)

    def test_treepir_real_client_results_are_in_the_artifact(self) -> None:
        self.assertEqual(
            self.manifest.table_rows["tab:treepir-client"],
            ["h16-real:undine", "h16-real:treepir",
             "h32-generated:undine", "h32-generated:treepir"],
        )
        for mode in ("full", "smoke", "canary", "jobs"):
            with self.subTest(mode=mode):
                selected = MODULE.select_run_jobs(self.manifest, mode, ["treepir-client-h16-real"])
                run_job = next(job for job in selected if job.spec.id == "treepir-client-h16-real")
                env = run_job.effective_env()
                self.assertEqual(env[f"{MODULE.TREEPIR_PREFIX}_REPLIES"], "real")
                self.assertEqual(int(env[f"{MODULE.TREEPIR_PREFIX}_HEIGHT"]),
                                 MODULE.TREEPIR_SMOKE_HEIGHT if mode == "smoke" else 16)
                if mode != "smoke":
                    self.assertEqual(int(env[f"{MODULE.TREEPIR_PREFIX}_SEEDED_LEAVES"]), 1 << 16)
                timing = [item for item in run_job.measurements if item.kind == "timing"]
                self.assertEqual({item.sample_group for item in timing}, {"client"})
                self.assertEqual(sum(item.precision_required for item in timing), 6)
                self.assertEqual(len(timing), 19)

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


class PrecisionScopeTests(unittest.TestCase):
    def test_summary_reports_unstable_diagnostics_without_claiming_they_converged(self) -> None:
        manifest = MODULE.Manifest()
        manifest.add_measurement(MODULE.MeasurementSpec("total", "job", unit="ns"))
        manifest.add_measurement(MODULE.MeasurementSpec(
            "detail", "job", unit="ns", precision_required=False))
        state = {
            "run_id": "test", "mode": "full", "snapshot_id": "test", "selected_jobs": ["job"],
            "run_jobs": {"job": {"measurements": ["total", "detail"], "selectors": {}}},
            "jobs": {"job": {"samples": {"total": [100.] * 10, "detail": [1., 100.] * 5}}},
        }
        resolved = MODULE.resolve_measurements(manifest, state)
        with tempfile.TemporaryDirectory() as directory:
            summary = MODULE.write_summary_outputs(manifest, pathlib.Path(directory), state, resolved)
            columns = [line.split("\t") for line in (pathlib.Path(directory) / "summary.tsv").read_text().splitlines()]
        self.assertTrue(summary["publishable"])
        self.assertEqual(summary["unstable_measurements"], [])
        self.assertEqual(summary["diagnostic_precision_misses"], ["detail"])
        self.assertEqual(summary["precision_required_series"], 1)
        diagnostic = next(record for record in summary["records"] if record["id"] == "detail")
        self.assertFalse(diagnostic["summary"]["stable"])
        self.assertEqual(diagnostic["samples"], [1., 100.] * 5)
        self.assertTrue(all(len(row) == len(columns[0]) for row in columns))

    def test_resume_rejects_changed_precision_scope(self) -> None:
        manifest = MODULE.build_manifest()
        job = manifest.jobs["treepir-client-h32-generated"]
        run_job = MODULE.RunJob(job, tuple(manifest.measurements_for_job(job.id)))
        # An absent scope field means the historical all-required contract.
        state = {"binaries": {}, "jobs": {job.id: {}},
                 "run_jobs": {job.id: {"selectors": {}}}}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(MODULE.PipelineError, "precision requirement changed"):
                MODULE.validate_resume_artifacts(pathlib.Path(directory), state, [run_job])

    def test_preparation_totals_pair_the_same_rebuilds(self) -> None:
        def metric(values):
            return {"trials": len(values), "samples": values}

        entry = {"system": "undine", **{name: metric(values) for name, values in {
            "gather_cells_ns": [900, 800], "gather_upper_ns": [100, 200],
            "fill_ns": [1000, 1000], "hint_ns": [200, 300],
            "packing_ns": [300, 200], "preprocess_ns": [500, 500],
            "prepare_ns": [1500, 1500],
        }.items()}}
        MODULE.validate_treepir_preparation(entry, "test")
        # Matching stage means are insufficient: the per-rebuild sum must agree.
        with self.assertRaisesRegex(MODULE.PipelineError, "sum of stages"):
            MODULE.validate_treepir_preparation(entry | {"gather_upper_ns": metric([200, 100])}, "test")
        with self.assertRaisesRegex(MODULE.PipelineError, "unpaired preparation"):
            MODULE.validate_treepir_preparation(entry | {"gather_upper_ns": metric([100])}, "test")
        treepir = {"system": "treepir", **{name: metric(values) for name, values in {
            "enumerate_ns": [50, 60], "columns_ns": [900, 800],
            "fill_ns": [900, 800], "hint_ns": [200, 300],
            "packing_ns": [300, 200], "preprocess_ns": [500, 500],
            "prepare_ns": [1450, 1360],
        }.items()}}
        MODULE.validate_treepir_preparation(treepir, "test")
        # An online-only batch does not claim any fresh preprocessing samples.
        for item in (entry, treepir):
            empty = {name: metric([]) if isinstance(value, dict) else value
                     for name, value in item.items()}
            MODULE.validate_treepir_preparation(empty, "test")

    def test_verifier_accepts_diagnostic_uncertainty_but_not_relabeled_totals(self) -> None:
        noisy = MODULE.summarize_samples([1., 100.] * 5)
        record = {"id": "substep", "precision_required": False, "summary": noisy}
        declared = {"substep": {"precision_required": False}}
        verify_results.verify_precision_scope([record], declared, adaptive=True)
        with self.assertRaisesRegex(verify_results.VerificationError, "scope differs"):
            verify_results.verify_precision_scope([record], {"substep": {}}, adaptive=True)
        with self.assertRaisesRegex(verify_results.VerificationError, "unstable timing"):
            verify_results.verify_precision_scope(
                [record | {"precision_required": True}], {"substep": {}}, adaptive=True)

    def test_diagnostics_still_require_complete_samples(self) -> None:
        measurements = [MODULE.MeasurementSpec("total", "job"),
                        MODULE.MeasurementSpec("detail", "job", precision_required=False)]
        state = {"samples": {"total": [100.] * 10, "detail": []}}
        self.assertFalse(MODULE.job_complete(state, "full", measurements))
        state["samples"]["detail"] = [1., 100.] * 5
        self.assertTrue(MODULE.job_complete(state, "full", measurements))


class CorrectnessGateTests(unittest.TestCase):
    def test_smoke_keeps_real_treepir_servers_in_correctness_tests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = pathlib.Path(directory)
            with (
                mock.patch.object(MODULE, "run_capture", return_value=
                                  subprocess.CompletedProcess([], 0, b"", b"")) as run,
                mock.patch.object(MODULE, "load_checkpoint", return_value=
                                  {"snapshot_id": "test-snapshot"}),
                mock.patch.object(MODULE, "validate_current_source_snapshot"),
                mock.patch.object(MODULE, "isolated_build_target",
                                  return_value=run_dir / "target"),
            ):
                MODULE.run_unit_tests(run_dir)
            rust = next(call.args[0] for call in run.call_args_list
                        if call.args[0][:2] == ["cargo", "test"])
            targets = [rust[i + 1] for i, arg in enumerate(rust) if arg == "--test"]
            self.assertIn("treepir", targets)
            self.assertIn("private_opening", targets)
            self.assertIn("private_selected_records", targets)


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

    def test_confidence_budget_covers_all_twelve_looks(self) -> None:
        samples = [99., 101.] * 80
        current = MODULE.summarize_samples(samples)["sequential_ci"]
        previous = MODULE.summarize_samples(samples, looks=(10, 20, 40, 80, 160))["sequential_ci"]
        self.assertEqual(len(current["looks"]), 12)
        self.assertEqual(current["looks"][-1], 20480)
        self.assertAlmostEqual(current["pointwise_confidence"], 1.0 - 0.05 / 12)
        self.assertAlmostEqual((1.0 - current["pointwise_confidence"]) * 12, 0.05)
        self.assertGreater(current["half_width"], previous["half_width"])


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

    def test_group_counts_and_warmups_do_not_fabricate_samples(self) -> None:
        measurements = [MODULE.MeasurementSpec(
            id=group, job_id="job", selector=MODULE.selector(key=1), path=(group,),
            kind="timing", sample_group=group,
        ) for group in ("preprocess", "online", "update")]
        empty = {"mean": 0, "p50": 0, "p95": 0, "trials": 0, "samples": []}
        document = {
            "meta": {"seed": 42, "trials": 5,
                     "sample_groups": {"preprocess": 3, "online": 5, "update": 0}},
            "entries": [{"key": 1, "preprocess": self.stats([900, 10, 11]),
                         "online": self.stats([900, 800, 700, 20, 21]), "update": empty}],
        }
        timing, _, _, raw = MODULE.extract_measurements(
            document, measurements, group_warmups={"preprocess": 1, "online": 3, "update": 0})
        self.assertEqual(timing, {"preprocess": [10., 11.], "online": [20., 21.]})
        self.assertEqual(len(raw["online"]), 5)
        document["entries"][0]["update"] = self.stats([1])
        with self.assertRaises(MODULE.PipelineError):
            MODULE.extract_measurements(document, measurements)

    def test_inner_repetitions_are_one_independent_sample(self) -> None:
        measurement = MODULE.MeasurementSpec(id="used", job_id="job", selector=MODULE.selector(key=1),
                                             path=("used",), kind="timing")
        metric = self.stats([100, 101]) | {"inner_repetitions": 32}
        document = {"meta": {"trials": 2}, "entries": [{"key": 1, "used": metric}]}
        timing, _, _, _ = MODULE.extract_measurements(document, [measurement])
        self.assertEqual(timing["used"], [100., 101.])
        metric["inner_repetitions"] = 0
        with self.assertRaises(MODULE.PipelineError):
            MODULE.extract_measurements(document, [measurement])

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
        samples = [1.0] * (MODULE.DEFAULT_MAX_REPETITIONS - 1) + [1.0e9]
        state = {"samples": {"metric": samples}}
        self.assertTrue(MODULE.job_complete(state, "full"))
        self.assertFalse(MODULE.summarize_samples(samples)["stable"])

    def test_noisy_series_continue_beyond_the_old_limit(self) -> None:
        samples = [1.0, 100.0] * 80
        state = {"samples": {"metric": samples}}
        for mode in ("full", "jobs"):
            with self.subTest(mode=mode):
                self.assertFalse(MODULE.job_complete(state, mode))
                self.assertEqual(MODULE.batch_repetitions(state, mode), 160)

    def test_extended_sampling_can_reach_precision_without_discarding_samples(self) -> None:
        samples = [1.0, 100.0] * 80
        original = samples.copy()
        state = {"samples": {"metric": samples}}
        while not MODULE.job_complete(state, "full"):
            count = MODULE.batch_repetitions(state, "full")
            self.assertGreater(count, 0)
            samples.extend([1.0, 100.0] * (count // 2))
        self.assertGreater(len(samples), 160)
        self.assertLess(len(samples), MODULE.DEFAULT_MAX_REPETITIONS)
        self.assertTrue(MODULE.summarize_samples(samples)["stable"])
        self.assertEqual(samples[:160], original)

    def test_stable_series_still_stop_at_the_first_look(self) -> None:
        self.assertTrue(MODULE.job_complete({"samples": {"metric": [100.0] * 10}}, "full"))

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
        self.assertEqual(expected, [10, 10, 20, 40, 80, 160, 320, 640,
                                    1280, 2560, 5120, 10240])
        self.assertEqual(MODULE.DEFAULT_MAX_REPETITIONS, 20480)
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


class BuildIsolationTests(unittest.TestCase):
    def test_build_cache_is_owned_by_one_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            target = MODULE.isolated_build_target(root, "first", "undine")
            self.assertEqual(MODULE.isolated_build_target(root, "first", "undine"), target)
            with self.assertRaisesRegex(MODULE.PipelineError, "unrecognized benchmark build cache"):
                MODULE.isolated_build_target(root, "second", "undine")
            (root / "build-targets" / "borg").mkdir()
            with self.assertRaisesRegex(MODULE.PipelineError, "unrecognized benchmark build cache"):
                MODULE.isolated_build_target(root, "first", "borg")

    def test_timing_policy_rejects_an_old_client_executable(self) -> None:
        manifest = MODULE.build_manifest()
        job = manifest.jobs["treepir-client-h32-generated"]
        run_job = MODULE.RunJob(job, tuple(manifest.measurements_for_job(job.id)))
        document = {"meta": {"role": "client", "first_pass_warmup_calls": 0}}
        with self.assertRaisesRegex(MODULE.PipelineError, "client_openings_per_sample"):
            MODULE.validate_timing_policy(run_job, document)
        document["meta"]["client_openings_per_sample"] = 8
        MODULE.validate_timing_policy(run_job, document)

    def test_client_means_are_checked_against_all_per_opening_timings(self) -> None:
        manifest = MODULE.build_manifest()
        job = manifest.jobs["treepir-client-h32-generated"]
        run_job = MODULE.RunJob(job, tuple(manifest.measurements_for_job(job.id)))
        metric = {"mean": 15, "p50": 10, "p95": 20, "trials": 2,
                  "samples": [10, 20], "inner_repetitions": 8,
                  "raw_call_samples": [10] * 8 + [20] * 8}
        document = {"meta": {"role": "client", "first_pass_warmup_calls": 0,
                             "client_openings_per_sample": 8},
                    "entries": [{"role": "client", "index_ns": metric}]}
        MODULE.validate_timing_policy(run_job, document)
        metric["raw_call_samples"][0] = 90
        with self.assertRaisesRegex(MODULE.PipelineError, "batch means disagree"):
            MODULE.validate_timing_policy(run_job, document)

    def test_inherited_benchmark_settings_are_removed(self) -> None:
        with mock.patch.dict(os.environ, {
            "BENCH_SAMPLE_GROUPS": "bad",
            "BENCH_DIAGNOSTIC_TABLE_LEGACY": "1",
            "BENCH_PRIVATE_OPENING_DEFAULT_PREFIX": "1",
            "YPIR_PACK_PROFILE": "1",
            "AUTH_WRITE_TRACE": "1",
            "PATH": "preserved",
        }):
            environment = MODULE.clean_benchmark_environment()
        self.assertFalse(any(key.startswith("BENCH_") for key in environment))
        self.assertEqual(environment["PATH"], "preserved")
        self.assertNotIn("YPIR_PACK_PROFILE", environment)
        self.assertNotIn("AUTH_WRITE_TRACE", environment)

    @unittest.skipUnless(shutil.which("cargo"), "Cargo is required for the stale-cache regression")
    def test_rebuild_ignores_stale_executable_even_with_restored_source_mtime(self) -> None:
        # Reproduce the failure mode: a modified source with an older mtime
        # leaves a different executable in a shared Cargo target directory.
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            crate = root / "crate"
            (crate / "benches").mkdir(parents=True)
            (crate / "Cargo.toml").write_text(
                '[package]\nname="build-isolation-fixture"\nversion="0.1.0"\n'
                'edition="2021"\n[features]\nbench=[]\n'
                '[[bench]]\nname="probe"\nharness=false\n'
            )
            source = crate / "benches" / "probe.rs"
            source.write_text('fn main() { println!("diagnostic"); }\n')
            shared = root / "shared-target"
            subprocess.run(
                ["cargo", "bench", "--no-run", "--target-dir", str(shared),
                 "--features", "bench", "--bench", "probe"],
                cwd=crate, capture_output=True, check=True,
            )
            source.write_text('fn main() { println!("measured"); }\n')
            os.utime(source, (1, 1))
            manifest = MODULE.build_manifest()
            original = next(iter(manifest.jobs.values()))
            job = dataclasses.replace(original, layout="undine", benchmark="probe", features=("bench",))
            run_job = MODULE.RunJob(job, ())
            run_dir = root / "run"
            run_dir.mkdir()
            state = {"snapshot_id": "frozen", "binaries": {}}
            with (
                mock.patch.object(MODULE, "CRATE_ROOTS", {"undine": crate}),
                mock.patch.object(MODULE, "validate_current_source_snapshot"),
                mock.patch.dict(os.environ, {"CARGO_TARGET_DIR": str(shared)}),
            ):
                binary = MODULE.build_benchmark_binary(run_dir, state, run_job, timeout_seconds=60)
                self.assertEqual(subprocess.check_output([str(binary)]).strip(), b"measured")
                # Resume reuses only the hash-checked binary saved by this run.
                self.assertEqual(MODULE.build_benchmark_binary(run_dir, state, run_job,
                                                               timeout_seconds=60), binary)
                state["binaries"][MODULE.build_key(run_job)]["source_snapshot_id"] = "other"
                with self.assertRaisesRegex(MODULE.PipelineError, "matching source build"):
                    MODULE.build_benchmark_binary(run_dir, state, run_job, timeout_seconds=60)
            record = state["binaries"][MODULE.build_key(run_job)]
            self.assertTrue(pathlib.Path(record["source_path"]).is_relative_to(run_dir))
            self.assertIn("--target-dir", record["command"])


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

    def test_archived_preprocessing_renderer_scales_its_two_samples(self) -> None:
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


class ArtifactPackagingTests(unittest.TestCase):
    def check_file(self, relative: str, content: bytes) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for name in ("README.md", "artifact", "scripts/reproduce.py",
                         "scripts/verify_results.py", "provenance/sources.json", relative):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content if name == relative else b"")
            with mock.patch.multiple(
                check_artifact, ROOT=root, REFERENCE_ROOT=root / "reference-results",
                DIAGNOSTICS_ROOT=root / "results/diagnostics", MEASURED_BENCHMARKS={},
            ), mock.patch.object(check_artifact, "reference_run_dir", return_value=None):
                check_artifact.check_packaged_files(
                    [path for path in root.rglob("*") if path.is_file()]
                )

    def test_original_paths_allowed_only_in_evidence_archives(self) -> None:
        content = (b"/home/researcher/Documents/"
                   b"undinelog-publication-artifact/undine")
        for relative in ("reference-results/run/build.log",
                         "results/diagnostics/audit/build.log"):
            with self.subTest(relative=relative):
                self.check_file(relative, content)
        for relative in ("scripts/command.py", "documentation/commands.md"):
            with self.subTest(relative=relative):
                with self.assertRaisesRegex(check_artifact.ArtifactCheckError,
                                            "absolute workspace path"):
                    self.check_file(relative, content)

    def test_evidence_archives_still_reject_forbidden_files(self) -> None:
        for relative, content, message in (
            ("paper.tex", b"", "forbidden publication file"),
            ("target/cache", b"", "forbidden build/cache path"),
            ("binary", b"\x7fELF", "compiled executable"),
        ):
            with self.subTest(relative=relative):
                with self.assertRaisesRegex(check_artifact.ArtifactCheckError, message):
                    self.check_file(f"results/diagnostics/audit/{relative}", content)



class WarmupPolicyTests(unittest.TestCase):
    def test_maintainer_groups_have_their_declared_warmups(self) -> None:
        manifest = MODULE.build_manifest()
        job = next(job for job in manifest.jobs.values()
                   if job.benchmark == MODULE.PRIVATE_BENCHMARK
                   and job.env_dict()[f"{MODULE.PRIVATE_PREFIX}_ROLE"] == "maintainer")
        run_job = MODULE.RunJob(job, tuple(manifest.measurements_for_job(job.id)))
        modes = {item.sample_group: item.sample_mode
                 for item in run_job.measurements if item.kind == "timing"}
        self.assertEqual(modes, {"preprocess": "rebuild", "online": "direct", "update": "update"})
        state = {"samples": {item.id: [] for item in run_job.measurements if item.kind == "timing"}}
        _, warmups = MODULE.group_batch_plan(run_job, state, "full")
        self.assertEqual(warmups, {"preprocess": MODULE.REBUILD_WARMUP_REPETITIONS,
                                   "online": MODULE.DIRECT_WARMUP_REPETITIONS,
                                   "update": MODULE.UPDATE_WARMUP_REPETITIONS})
        self.assertEqual(MODULE.REBUILD_WARMUP_REPETITIONS, 1)
        self.assertEqual(MODULE.UPDATE_WARMUP_REPETITIONS, 0)

    def test_fixture_cache_is_shared_and_ignored(self) -> None:
        self.assertEqual(MODULE.FIXTURES_ROOT, MODULE.ROOT / "results" / "fixtures")
        ignore = (MODULE.ROOT / ".gitignore").read_text().splitlines()
        self.assertIn("/results/fixtures/", ignore)


if __name__ == "__main__":
    unittest.main(verbosity=2)
