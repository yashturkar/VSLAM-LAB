import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Evaluate.pairwise_metrics import evaluate_pair, interpolate_trajectory, read_pose_trajectory, write_pairwise_metrics


HEADER = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]


def write_trajectory(path: Path, scale: float = 1.0, translation=(0.0, 0.0, 0.0)) -> None:
    rows = []
    for index in range(20):
        xyz = np.asarray([index * scale, np.sin(index / 3) * scale, 0.1 * index * scale]) + translation
        rows.append([1_000_000_000 + index * 100_000_000, *xyz, 0, 0, 0, 1])
    pd.DataFrame(rows, columns=HEADER).to_csv(path, index=False)


class PairwiseMetricsTests(unittest.TestCase):
    def test_nanosecond_timestamps_are_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.csv"
            write_trajectory(path)
            trajectory = read_pose_trajectory(path)
            self.assertAlmostEqual(trajectory.timestamps[0], 1.0)
            self.assertAlmostEqual(trajectory.timestamps[-1], 2.9)

    def test_sensor_aware_alignment_preserves_metric_scale_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path, estimate_path = root / "reference.csv", root / "estimate.csv"
            write_trajectory(reference_path)
            write_trajectory(estimate_path, scale=2.0, translation=(10, -4, 3))
            reference, estimate = read_pose_trajectory(reference_path), read_pose_trajectory(estimate_path)
            se3, _ = evaluate_pair(reference, estimate, "reference", "estimate", 0.02, correct_scale=False)
            sim3, _ = evaluate_pair(reference, estimate, "reference", "estimate", 0.02, correct_scale=True)
            self.assertGreater(se3["ape"]["translation"]["rmse"], 1.0)
            self.assertLess(sim3["ape"]["translation"]["rmse"], 1e-8)

    def test_three_pair_report_keeps_legacy_primary_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            odom, vslam, fast = root / "odom.csv", root / "vslam.csv", root / "fast.csv"
            write_trajectory(odom)
            write_trajectory(vslam, translation=(2, 1, 0))
            write_trajectory(fast, translation=(-2, 3, 0))
            output, _ = write_pairwise_metrics(root / "metrics.json", vslam, odom, fast, 0.02, "stereo", ["proxy"])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], 2)
            self.assertEqual(set(report["comparisons"]), {
                "vslam_vs_robot_odometry", "fast_lio_vs_robot_odometry", "vslam_vs_fast_lio"
            })
            self.assertEqual(report["warnings"], ["proxy"])
            self.assertAlmostEqual(report["rmse"]["translation"], 0.0, places=10)

    def test_cross_reference_comparison_interpolates_fast_lio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            odom, vslam, fast = root / "odom.csv", root / "vslam.csv", root / "fast.csv"
            write_trajectory(vslam)
            write_trajectory(fast)
            frame = pd.read_csv(fast)
            frame.iloc[:, 0] += 40_000_000
            frame.to_csv(fast, index=False)
            odometry_rows = []
            for index in range(100):
                t = index * 0.02
                odometry_rows.append([1_000_000_000 + index * 20_000_000, t * 10, np.sin(t * 10 / 3), t, 0, 0, 0, 1])
            pd.DataFrame(odometry_rows, columns=HEADER).to_csv(odom, index=False)
            output, _ = write_pairwise_metrics(root / "metrics.json", vslam, odom, fast, 0.02, "stereo")
            report = json.loads(output.read_text(encoding="utf-8"))
            direct = report["comparisons"]["vslam_vs_fast_lio"]
            self.assertEqual(direct["association"]["max_time_difference_s"], 1e-6)
            self.assertIn("Slerp", direct["reference_interpolation"])

    def test_interpolation_matches_linear_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.csv"
            write_trajectory(path)
            trajectory = read_pose_trajectory(path)
            targets = trajectory.timestamps[:-1] + 0.05
            interpolated = interpolate_trajectory(trajectory, targets)
            self.assertEqual(interpolated.num_poses, 19)
            np.testing.assert_allclose(interpolated.positions_xyz[:, 0], np.arange(19) + 0.5)


if __name__ == "__main__":
    unittest.main()
