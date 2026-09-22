import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from Utilities.lightning_batch_pipeline import (
    BatchConfig,
    BatchRunner,
    apply_thresholds,
    report_row,
    temporal_metrics,
)
from Utilities.lightning_fastlio_pipeline import PipelinePaths


HEADER = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]


def write_trajectory(path: Path, timestamps_s: list[float]) -> None:
    rows = [
        [time_s * 1e9, index, 0, 0, 0, 0, 0, 1]
        for index, time_s in enumerate(timestamps_s)
    ]
    pd.DataFrame(rows, columns=HEADER).to_csv(path, index=False)


class LightningBatchPipelineTests(unittest.TestCase):
    def write_config(self, root: Path, **overrides: object) -> Path:
        source = root / "session" / "sample_seq001_20260915_000000_000000"
        data = {
            "version": 1,
            "name": "test-batch",
            "local_root": str(root / "local"),
            "sensor_type": "stereo",
            "sequences": [str(source)],
            "methods": ["orbslam2"],
            "fast_lio": {"enabled": False, "workspace": str(root / "workspace")},
            "evaluation": {
                "max_time_difference_s": 0.02,
                "thresholds": {
                    "min_temporal_completion_ratio": None,
                    "max_vslam_vs_robot_odometry_translation_rmse_m": None,
                    "max_vslam_vs_fast_lio_translation_rmse_m": None,
                },
            },
            "execution": {"continue_on_error": True},
        }
        data.update(overrides)
        path = root / "batch.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path

    def test_config_loads_explicit_paths_and_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = BatchConfig.load(self.write_config(root))
            self.assertEqual(config.methods, ("orbslam2",))
            self.assertEqual(config.sensor_type, "stereo")
            self.assertFalse(config.fast_lio_enabled)
            self.assertEqual(config.batch_root, root / "local/batches/test-batch")
            self.assertTrue(config.sequences[0].is_absolute())

    def test_config_rejects_duplicate_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = str(root / "source")
            path = self.write_config(root, sequences=[source, source])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                BatchConfig.load(path)

    def test_fast_lio_threshold_requires_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_config(root)
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            data["evaluation"]["thresholds"][
                "max_vslam_vs_fast_lio_translation_rmse_m"
            ] = 1.0
            path.write_text(yaml.safe_dump(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires fast_lio"):
                BatchConfig.load(path)

    def test_temporal_completion_uses_reference_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, estimate = root / "reference.csv", root / "estimate.csv"
            write_trajectory(reference, [0, 2, 4, 6, 8, 10])
            write_trajectory(estimate, [2, 4, 6, 8])
            result = temporal_metrics(estimate, reference)
            self.assertAlmostEqual(result["temporal_completion_ratio"], 0.6)
            self.assertAlmostEqual(result["start_delay_s"], 2.0)
            self.assertAlmostEqual(result["early_stop_s"], 2.0)
            self.assertAlmostEqual(result["largest_estimate_gap_s"], 2.0)

    def test_temporal_completion_is_clamped_to_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, estimate = root / "reference.csv", root / "estimate.csv"
            write_trajectory(reference, [1, 2, 3, 4])
            write_trajectory(estimate, [0, 2, 4, 5])
            result = temporal_metrics(estimate, reference)
            self.assertEqual(result["temporal_completion_ratio"], 1.0)
            self.assertEqual(result["start_delay_s"], 0.0)
            self.assertEqual(result["early_stop_s"], 0.0)

    def test_report_flattens_primary_and_fast_lio_metrics(self):
        comparison = {
            "association": {
                "matched_poses": 4,
                "reference_coverage": 0.5,
                "estimate_coverage": 1.0,
            },
            "ape": {
                "translation": {"mean": 0.1, "std": 0.2, "rmse": 0.3, "max": 0.4},
                "rotation": {"mean": 1.0, "std": 2.0, "rmse": 3.0, "max": 4.0},
            },
            "rpe": {
                "translation_rmse_1m": {"rmse": 0.05},
                "rotation_rmse_1m": {"rmse": 0.5},
            },
            "path_lengths_m": {"reference": 10.0, "estimate": 8.0},
            "trajectory_length_ratio": 0.8,
        }
        result = {
            "sequence": "seq001",
            "source": "/nas/seq001",
            "method": "orbslam2",
            "status": "complete",
            "temporal": {"temporal_completion_ratio": 0.75},
            "metrics": {
                "length_ratio": 0.8,
                "weighted_rmse": 0.4,
                "comparisons": {
                    "vslam_vs_robot_odometry": comparison,
                    "fast_lio_vs_robot_odometry": comparison,
                    "vslam_vs_fast_lio": comparison,
                },
            },
        }
        row = report_row(result)
        self.assertEqual(row["translation_ape_rmse_m"], 0.3)
        self.assertEqual(row["vslam_vs_fast_lio_translation_rmse_m"], 0.3)
        self.assertEqual(row["temporal_completion_ratio"], 0.75)

    def test_thresholds_report_missing_and_out_of_range_metrics(self):
        failures = apply_thresholds(
            {"temporal_completion_ratio": 0.5, "translation_ape_rmse_m": None},
            {
                "min_temporal_completion_ratio": 0.8,
                "max_vslam_vs_robot_odometry_translation_rmse_m": 1.0,
                "max_vslam_vs_fast_lio_translation_rmse_m": None,
            },
        )
        self.assertEqual(len(failures), 2)
        self.assertIn("required >=", failures[0])
        self.assertIn("unavailable", failures[1])

    def test_reports_are_written_in_all_three_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = BatchConfig.load(self.write_config(root))
            runner = BatchRunner(config)
            runner.state["results"] = {
                "seq/orbslam2": {
                    "sequence": "seq",
                    "source": "/nas/seq",
                    "method": "orbslam2",
                    "status": "failed",
                    "error": "test failure",
                }
            }
            runner.write_reports()
            self.assertTrue(runner.report_json.is_file())
            self.assertTrue(runner.report_csv.is_file())
            self.assertTrue(runner.report_markdown.is_file())
            report = json.loads(runner.report_json.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["failed"], 1)
            self.assertIn(
                "test failure", runner.report_markdown.read_text(encoding="utf-8")
            )

    def test_preflight_passes_fast_lio_toggle_to_sequence_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = BatchConfig.load(self.write_config(root))
            runner = BatchRunner(config)
            checks = {
                "ok": True,
                "errors": [],
                "warnings": [],
                "estimated_required_bytes": 0,
                "free_bytes": 1,
            }
            with (
                patch.object(runner, "validate_methods", return_value=[]),
                patch(
                    "Utilities.lightning_batch_pipeline.Pipeline.source_checks",
                    return_value=checks,
                ) as source_checks,
            ):
                result = runner.preflight()
            self.assertTrue(result["ok"])
            source_checks.assert_called_once_with(
                require_orbslam2=False, require_fastlio=False
            )

    def test_verified_slam_and_metrics_are_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = BatchConfig.load(self.write_config(root))
            runner = BatchRunner(config)
            paths = PipelinePaths(config.sequences[0], config.local_root)
            paths.run_root.mkdir(parents=True)
            paths.sequence.mkdir(parents=True)
            (paths.sequence / "rgb.csv").write_text("rgb\n", encoding="utf-8")
            (paths.sequence / "calibration.yaml").write_text(
                "cameras: []\n", encoding="utf-8"
            )
            write_trajectory(paths.sequence / "groundtruth.csv", [0, 1, 2, 3])
            output = runner.output_path(paths, "orbslam2")
            output.mkdir(parents=True)
            trajectory = output / "00000_KeyFrameTrajectory.csv"
            write_trajectory(trajectory, [0, 1, 2, 3])
            metrics = {
                "schema_version": 2,
                "comparisons": {"vslam_vs_robot_odometry": {}},
                "length_ratio": 1.0,
                "weighted_rmse": 0.0,
            }
            (output / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            (output / "trajectory_report.pdf").write_bytes(b"pdf")
            key = runner.result_key(paths, "orbslam2")
            runner.state["results"][key] = {
                "slam_fingerprint": "slam-fp",
                "metrics_fingerprint": "metrics-fp",
            }
            with (
                patch.object(runner, "slam_fingerprint", return_value="slam-fp"),
                patch.object(runner, "metrics_fingerprint", return_value="metrics-fp"),
                patch("Utilities.lightning_batch_pipeline.run_single_baseline") as slam,
                patch(
                    "Utilities.lightning_batch_pipeline.evaluate_single_trajectory"
                ) as evaluate,
            ):
                result = runner.run_method(paths, "orbslam2", "extract-fp", None)
            self.assertEqual(result["status"], "complete")
            slam.assert_not_called()
            evaluate.assert_not_called()

    def test_batch_continues_to_next_method_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = BatchConfig.load(
                self.write_config(root, methods=["orbslam2", "orbslam3"])
            )
            runner = BatchRunner(config)
            source = str(config.sequences[0])
            preflight = {
                "ok": True,
                "global_errors": [],
                "sequence_checks": {
                    source: {
                        "ok": True,
                        "errors": [],
                        "warnings": [],
                        "topic_counts": {},
                    }
                },
                "estimated_required_bytes": 0,
                "free_bytes": 1,
            }
            seen: list[str] = []

            def run_method(
                paths: PipelinePaths, method: str, *_: object
            ) -> dict[str, object]:
                seen.append(method)
                result = {
                    "sequence": paths.sequence_name,
                    "source": str(paths.bag_root),
                    "method": method,
                    "status": "failed" if method == "orbslam2" else "complete",
                }
                runner.state["results"][runner.result_key(paths, method)] = result
                return result

            with (
                patch.object(runner, "preflight", return_value=preflight),
                patch.object(runner, "run_method", side_effect=run_method),
                patch(
                    "Utilities.lightning_batch_pipeline.Pipeline.source_identity",
                    return_value={},
                ),
                patch(
                    "Utilities.lightning_batch_pipeline.Pipeline.run_staging",
                    return_value="stage",
                ),
                patch(
                    "Utilities.lightning_batch_pipeline.Pipeline.run_extraction",
                    return_value="extract",
                ),
            ):
                exit_code = runner.run()
            self.assertEqual(seen, ["orbslam2", "orbslam3"])
            self.assertEqual(exit_code, 1)
            self.assertEqual(runner.state["status"], "failed")


if __name__ == "__main__":
    unittest.main()
