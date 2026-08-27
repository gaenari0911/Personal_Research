"""Contracts for the same-allocation Stage A -> Stage B orchestrator."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_stage_b_after_stage_a as pipeline  # noqa: E402
import stage_a_control  # noqa: E402
import train_stage_b_probes as probe_cli  # noqa: E402
import cache_stage_a_features as cache_cli  # noqa: E402
import eval_stage_b as eval_cli  # noqa: E402


class StageBPipelineTests(unittest.TestCase):
    def test_parallel_final_metrics_uses_four_workers_then_one_merge(self) -> None:
        processes = [mock.Mock() for _ in range(4)]
        for process in processes:
            process.wait.return_value = 0
        with (
            mock.patch.object(eval_cli.subprocess, "Popen", side_effect=processes) as popen,
            mock.patch.object(eval_cli, "write_comparisons") as comparisons,
            mock.patch.object(eval_cli, "atomic_json") as atomic,
        ):
            eval_cli.run_parallel_final_metrics(Path("config.yaml"), Path("reports"), Path("representations"))
        self.assertEqual(popen.call_count, 4)
        for call in popen.call_args_list:
            self.assertIn("--parallel-child", call.args[0])
            self.assertEqual(call.kwargs["env"]["OMP_NUM_THREADS"], "2")
        comparisons.assert_called_once_with(Path("reports"), Path("representations"), "test")
        self.assertIn("FINAL_TEST_BATCH_COMPLETED.json", str(atomic.call_args.args[0]))

    def test_parallel_test_cache_uses_disjoint_two_gpu_workers_and_merged_audit(self) -> None:
        processes = [mock.Mock(), mock.Mock()]
        for process in processes:
            process.wait.return_value = 0
        args = SimpleNamespace(
            audit=Path("analysis/stage_b/test_clip_cache_audit.json"),
            model_cache=Path("model-cache"), batch_size=64,
        )
        merged = {
            "status": "PASS", "completed": 95, "expected_episodes": 95,
            "missing": [], "invalid": [],
        }
        with (
            mock.patch.object(cache_cli.subprocess, "Popen", side_effect=processes) as popen,
            mock.patch.object(cache_cli, "read_json", return_value={"status": "PASS", "records_this_run": []}),
            mock.patch.object(cache_cli, "audit_cache", return_value=merged),
            mock.patch.object(cache_cli, "atomic_json") as atomic,
        ):
            cache_cli.run_parallel_test_cache(args, [str(index) for index in range(95)], {}, ["GPU-A", "GPU-B"])
        self.assertEqual(popen.call_count, 2)
        for index, call in enumerate(popen.call_args_list):
            command = call.args[0]
            self.assertEqual(command[command.index("--test-num-shards") + 1], "2")
            self.assertEqual(command[command.index("--test-shard-index") + 1], str(index))
            self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], f"GPU-{'A' if index == 0 else 'B'}")
        atomic.assert_called_once_with(args.audit, merged)

    def test_parallel_probe_launcher_uses_four_two_thread_workers(self) -> None:
        processes = [mock.Mock() for _ in range(4)]
        for process in processes:
            process.wait.return_value = 0
        with mock.patch.object(probe_cli.subprocess, "Popen", side_effect=processes) as popen:
            probe_cli.run_parallel_variants(Path("config.yaml"))
        self.assertEqual(popen.call_count, 4)
        variants = []
        for call in popen.call_args_list:
            command = call.args[0]
            variants.append(command[command.index("--variant") + 1])
            self.assertIn("--parallel-child", command)
            self.assertEqual(call.kwargs["env"]["OMP_NUM_THREADS"], "2")
            self.assertEqual(call.kwargs["env"]["MKL_NUM_THREADS"], "2")
        self.assertEqual(variants, ["B0", "B1", "B2", "B3"])

    def test_orchestrator_success_order_and_final_status(self) -> None:
        reports = []
        extracts = []
        config = {"execution": {"minimum_remaining_walltime_seconds": 1}}
        with (
            mock.patch.object(pipeline, "_config", return_value=config),
            mock.patch.object(pipeline, "_validate_remaining_walltime", return_value={"status": "PASS"}),
            mock.patch.object(pipeline, "_validate_stage_a_outputs", return_value={"status": "PASS"}),
            mock.patch.object(
                pipeline, "_run_extract_pair",
                side_effect=lambda a, b, _d, _c, splits: extracts.append((a, b, splits)),
            ),
            mock.patch.object(pipeline, "_run_probe_training") as probe_training,
            mock.patch.object(pipeline, "_run_test_feature_cache") as test_cache,
            mock.patch.object(pipeline, "_run_final_test_eval") as final_eval,
            mock.patch.object(pipeline, "_validate_stage_b_outputs", return_value={"status": "PASS"}),
            mock.patch.object(pipeline, "_write_report", side_effect=lambda payload: reports.append(dict(payload))),
        ):
            result = pipeline.run(Path("synthetic.yaml"))
        self.assertEqual(
            extracts,
            [
                ("B0", "B1", ("train", "val")), ("B2", "B3", ("train", "val")),
                ("B0", "B1", ("test",)), ("B2", "B3", ("test",)),
            ],
        )
        probe_training.assert_called_once()
        test_cache.assert_called_once()
        final_eval.assert_called_once()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["current_stage"], "COMPLETE")
        self.assertEqual(reports[-1]["status"], "PASS")

    def test_orchestrator_failure_is_recorded(self) -> None:
        reports = []
        with (
            mock.patch.object(pipeline, "_config", return_value={}),
            mock.patch.object(pipeline, "_validate_remaining_walltime", return_value={"status": "PASS"}),
            mock.patch.object(pipeline, "_validate_stage_a_outputs", return_value={"status": "PASS"}),
            mock.patch.object(pipeline, "_run_extract_pair", side_effect=RuntimeError("synthetic extraction failure")),
            mock.patch.object(pipeline, "_write_report", side_effect=lambda payload: reports.append(dict(payload))),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic extraction failure"):
                pipeline.run(Path("synthetic.yaml"))
        self.assertEqual(reports[-1]["status"], "FAIL")
        self.assertIn("synthetic extraction failure", reports[-1]["error"])

    def test_stage_b_failure_preserves_stage_a_pass(self) -> None:
        status = {
            "stage_a": "PASS",
            "stage_b": "FAIL",
            "training": {variant: "PASS" for variant in ("B0", "B1", "B2", "B3")},
            "smoke": {variant: "PASS" for variant in ("B0", "B1", "B2", "B3")},
            "cache": "PASS",
        }
        saved = []
        written = []
        with (
            mock.patch.object(stage_a_control, "current_status", return_value=status),
            mock.patch.object(stage_a_control, "save_status", side_effect=lambda payload: saved.append(dict(payload))),
            mock.patch.object(stage_a_control, "read_json", return_value={"STAGE_A_GATE": "PASS"}),
            mock.patch.object(stage_a_control, "atomic_json", side_effect=lambda path, payload: written.append(dict(payload))),
        ):
            stage_a_control.record_failure("FINAL_GATE", 17)
        self.assertEqual(saved[-1]["stage_a"], "PASS")
        self.assertEqual(saved[-1]["stage_b"], "FAIL")
        self.assertEqual(written[-1]["STAGE_A_GATE"], "PASS")
        self.assertEqual(written[-1]["READY_FOR_PROBE_TEST_METRIC"], "YES")
        self.assertEqual(written[-1]["STAGE_B"], "FAIL")

    def test_auto_run_is_explicitly_configured_for_pbs(self) -> None:
        with mock.patch.dict("os.environ", {"PBS_JOBID": "synthetic.pleiades1"}, clear=True):
            self.assertTrue(stage_a_control.stage_b_auto_enabled())
        with mock.patch.dict(
            "os.environ", {"PBS_JOBID": "synthetic.pleiades1", "STAGE_B_AUTO_RUN": "0"}, clear=True
        ):
            self.assertFalse(stage_a_control.stage_b_auto_enabled())

    def test_extract_pair_pins_one_allocated_gpu_per_variant(self) -> None:
        processes = [mock.Mock() for _ in range(4)]
        for process in processes:
            process.wait.return_value = 0
        with mock.patch.object(pipeline.subprocess, "Popen", side_effect=processes) as popen:
            pipeline._run_extract_pair("B2", "B3", ["GPU-A", "GPU-B"], Path("config.yaml"))
        self.assertEqual(popen.call_count, 4)
        calls = popen.call_args_list
        self.assertEqual([call.kwargs["env"]["CUDA_VISIBLE_DEVICES"] for call in calls], ["GPU-A", "GPU-B", "GPU-A", "GPU-B"])
        self.assertEqual([call.args[0][call.args[0].index("--split") + 1] for call in calls], ["train", "train", "val", "val"])

    def test_test_extraction_uses_explicit_final_test_gate(self) -> None:
        processes = [mock.Mock(), mock.Mock()]
        for process in processes:
            process.wait.return_value = 0
        with mock.patch.object(pipeline.subprocess, "Popen", side_effect=processes) as popen:
            pipeline._run_extract_pair("B0", "B1", ["GPU-A", "GPU-B"], Path("config.yaml"), ("test",))
        for call in popen.call_args_list:
            self.assertIn("--final-test", call.args[0])
            self.assertEqual(call.args[0][call.args[0].index("--split") + 1], "test")

    def test_test_clip_cache_runs_after_selection_with_gate_and_gpu_pin(self) -> None:
        with mock.patch.object(pipeline, "_run") as run:
            pipeline._run_test_feature_cache("GPU-A")
        command = run.call_args.args[0]
        self.assertIn("tools/cache_stage_a_features.py", command)
        self.assertEqual(command[command.index("--split-scope") + 1], "test")
        self.assertIn("--final-test", command)
        self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "GPU-A")

    def test_final_output_gate_requires_metrics_provenance_and_figures(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            reports = root / "reports"
            representations = root / "representations"
            probes = root / "probes"
            config = {
                "dataset": {"expected_trajectories": {"train": 1, "val": 1, "test": 1}},
                "outputs": {
                    "report_root": str(reports),
                    "representation_root": str(representations),
                    "probe_root": str(probes),
                },
            }
            reports.mkdir(parents=True)
            (reports / "test_clip_cache_audit.json").write_text(
                json.dumps(
                    {
                        "status": "PASS", "split_scope": "test", "completed": 1,
                        "expected_episodes": 1, "test_split_used": True,
                        "final_test_gate": True, "missing": [], "invalid": [],
                    }
                ),
                encoding="utf-8",
            )
            for variant in ("B0", "B1", "B2", "B3"):
                directory = reports / "final_test" / variant
                directory.mkdir(parents=True)
                for filename in pipeline.REQUIRED_REPORT_FILES:
                    (directory / filename).write_text("{}\n", encoding="utf-8")
                (directory / "summary.json").write_text(
                    json.dumps(
                        {
                            "split": "test", "test_split_evaluated": True,
                            "provenance": {
                                "stage_a_checkpoint": {"state_dict_sha256": f"sha-{variant}"},
                                "probe_checkpoint_sha256": f"probe-{variant}",
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                figures = directory / "figures"
                figures.mkdir()
                for filename in pipeline.REQUIRED_VARIANT_FIGURES:
                    (figures / filename).touch()
                for split in ("train", "val", "test"):
                    manifest_dir = representations / variant / split
                    manifest_dir.mkdir(parents=True)
                    (manifest_dir / "manifest.json").write_text(
                        json.dumps(
                            {
                                "status": "COMPLETE", "variant": variant, "split": split,
                                "trajectory_count": 1, "test_split": split == "test",
                                "final_test_gate": split == "test",
                                "sampling_sha256": f"same-{split}",
                                "checkpoint": {"state_dict_sha256": f"sha-{variant}"},
                            }
                        ),
                        encoding="utf-8",
                    )
                probe_dir = probes / variant
                probe_dir.mkdir(parents=True)
                torch.save(
                    {
                        "schema_version": "stage-b-probes-v1", "variant": variant,
                        "selection_split": "val", "backbone_frozen": True,
                        "test_split_used": False,
                        "stage_a_checkpoint": {"state_dict_sha256": f"sha-{variant}"},
                    },
                    probe_dir / "selected_probes.pt",
                )
                (directory / "FINAL_TEST_COMPLETED.json").write_text(
                    json.dumps(
                        {
                            "status": "COMPLETE", "variant": variant, "split": "test",
                            "trajectory_count": 1, "exactly_once_gate": True,
                            "sampling_sha256": "same-test",
                            "probe_checkpoint_sha256": f"probe-{variant}",
                        }
                    ),
                    encoding="utf-8",
                )
            comparison = reports / "final_test" / "comparison"
            comparison.mkdir()
            for filename in (
                "B0_B1_B2_B3_summary.csv", "memory_depth_comparison.csv",
                "retention_distance_comparison.csv", "transition_robustness_comparison.csv",
            ):
                (comparison / filename).touch()
            comparison_figures = comparison / "figures"
            comparison_figures.mkdir()
            for filename in pipeline.REQUIRED_COMPARISON_FIGURES:
                (comparison_figures / filename).touch()
            self.assertEqual(pipeline._validate_stage_b_outputs(config)["status"], "PASS")
            (reports / "final_test/B3/figures/dashboard.html").unlink()
            with self.assertRaisesRegex(RuntimeError, "figure files missing for B3"):
                pipeline._validate_stage_b_outputs(config)

    def test_stage_a_gate_accepts_earlier_best_but_requires_complete_last(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "analysis").mkdir()
            (root / "analysis/stage_a_compute_plan.json").write_text(
                json.dumps({"selected_epochs_per_model": 2}), encoding="utf-8"
            )
            common_hash = "common-sha"
            for variant in ("B0", "B1", "B2", "B3"):
                (root / f"checkpoints/stage_a/{variant}").mkdir(parents=True)
                torch.save(
                    {"common_initialization_sha256": common_hash, "completed_epoch": 1, "global_update": 734},
                    root / f"checkpoints/stage_a/{variant}/best_val.pt",
                )
                torch.save(
                    {"common_initialization_sha256": common_hash, "completed_epoch": 2, "global_update": 1468},
                    root / f"checkpoints/stage_a/{variant}/last.pt",
                )
                (root / f"analysis/stage_a_training_{variant}.json").write_text(
                    json.dumps(
                        {
                            "status": "PASS", "epochs_completed": 2, "global_updates": 1468,
                            "test_split_used": False, "epochs": [{"collapse": False}, {"collapse": False}],
                            "common_initialization_sha256": common_hash,
                        }
                    ),
                    encoding="utf-8",
                )

            def fake_loader(path, variant, _device):
                payload = torch.load(path, map_location="cpu", weights_only=False)
                return None, {
                    "completed_epoch": payload["completed_epoch"],
                    "global_update": payload["global_update"],
                    "state_dict_sha256": f"state-{variant}-{Path(path).name}",
                }

            with (
                mock.patch.object(pipeline, "ROOT", root),
                mock.patch.object(pipeline, "load_frozen_stage_a_model", side_effect=fake_loader),
            ):
                result = pipeline._validate_stage_a_outputs({})
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["models"]["B2"]["completed_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
