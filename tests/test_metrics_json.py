import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from Evaluate.metrics_json import aligned_rotation_rmse, symmetric_coverage, trajectory_length, write_metrics_json


class MetricsTests(unittest.TestCase):
    def test_symmetric_coverage(self):
        self.assertEqual(symmetric_coverage(10.0, 10.0), 1.0)
        self.assertEqual(symmetric_coverage(5.0, 10.0), 0.5)
        self.assertEqual(symmetric_coverage(15.0, 10.0), 0.5)
        self.assertEqual(symmetric_coverage(25.0, 10.0), 0.0)
        self.assertEqual(symmetric_coverage(1.0, 0.0), 0.0)

    def test_trajectory_length(self):
        frame = pd.DataFrame([[0, 0, 0, 0], [1, 3, 4, 0], [2, 3, 4, 12]])
        self.assertEqual(trajectory_length(frame), 17.0)

    def test_rotation_rmse_matches_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluation = Path(directory)
            predicted = evaluation / "00000_KeyFrameTrajectory.tum"
            groundtruth = evaluation / "00000_gt.tum"
            predicted.write_text("10 0 0 0 0 0 0 1\n30 0 0 0 0 0 0.70710678 0.70710678\n", encoding="utf-8")
            groundtruth.write_text(
                "10 0 0 0 0 0 0 1\n20 0 0 0 0 0 0 1\n30 0 0 0 0 0 0.70710678 0.70710678\n",
                encoding="utf-8",
            )
            self.assertAlmostEqual(aligned_rotation_rmse(evaluation, "00000"), 0.0)

    def test_zero_coverage_writes_null_weighted_rmse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = root / "00000_KeyFrameTrajectory.csv"
            groundtruth = root / "groundtruth.csv"
            evaluation = root / "evaluation"
            evaluation.mkdir()
            pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 1], [1, 20, 0, 0, 0, 0, 0, 1]]).to_csv(trajectory, index=False)
            pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 1], [1, 10, 0, 0, 0, 0, 0, 1]]).to_csv(groundtruth, index=False)
            pd.DataFrame([{"traj_name": "00000_KeyFrameTrajectory.txt", "mean": 1, "std": 0, "rmse": 2, "max": 3}]).to_csv(evaluation / "ate.csv", index=False)
            output = write_metrics_json(root / "metrics.json", trajectory, groundtruth, evaluation, "00000")
            metrics = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(metrics["length_ratio"], 0.0)
            self.assertIsNone(metrics["weighted_rmse"])


if __name__ == "__main__":
    unittest.main()
