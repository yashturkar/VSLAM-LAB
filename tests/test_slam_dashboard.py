import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import yaml

from Web.slam_dashboard import create_run, discover_runs, discover_sequences, metrics_table, trajectory_frames


class SlamDashboardTests(unittest.TestCase):
    def make_capture(self, root: Path) -> Path:
        capture = root / "session_20260901_120000" / "walk_seq001_20260901_120100_123456"
        research = capture / "research-bag"
        research.mkdir(parents=True)
        (capture / "manifest.json").write_text(json.dumps({"name": capture.name, "status": "complete"}))
        (capture / "processing.json").write_text(json.dumps({"status": "complete"}))
        (research / "research-bag_0.mcap").write_bytes(b"mcap")
        counts = {
            "/cam_sync/cam0/image_preview/compressed": 20,
            "/cam_sync/cam1/image_preview/compressed": 19,
            "/odometry": 90,
            "/ouster/points": 18,
        }
        metadata = {"rosbag2_bagfile_information": {"topics_with_message_count": [
            {"topic_metadata": {"name": name}, "message_count": count} for name, count in counts.items()
        ]}}
        (research / "metadata.yaml").write_text(yaml.safe_dump(metadata))
        return capture

    def test_discovers_processed_sequences_and_topic_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = self.make_capture(root)
            records = discover_sequences(root)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].path, str(capture))
            self.assertEqual(records[0].stereo_frames, 19)
            self.assertEqual(records[0].odometry_poses, 90)
            self.assertEqual(records[0].status, "complete")

    def test_launch_creates_timestamped_state_and_detached_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = self.make_capture(root / "nas")
            process = MagicMock(pid=1234)
            with patch("Web.slam_dashboard.subprocess.Popen", return_value=process) as popen:
                run_dir = create_run(capture, "orbslam2", True, root / "local", root / "ws")
            state = json.loads((run_dir / "run.json").read_text())
            self.assertEqual(state["status"], "running")
            self.assertEqual(state["pid"], 1234)
            self.assertIn("orbslam2", run_dir.name)
            self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_worker_entrypoint_can_import_repository_modules(self):
        result = subprocess.run(
            [sys.executable, "Utilities/web_slam_worker.py", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--sequence", result.stdout)

    def test_discovers_completed_run_history(self):
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory)
            run = local / "web_results/sequence/20260901_120000_orbslam2"
            run.mkdir(parents=True)
            (run / "run.json").write_text(json.dumps({
                "status": "complete", "created_at": "2026-09-01T12:00:00Z", "run_dir": str(run)
            }))
            self.assertEqual(discover_runs(local)[0]["status"], "complete")

    def test_metrics_table_exposes_evo_rmse(self):
        metrics = {"comparisons": {"vslam_vs_robot_odometry": {
            "association": {"matched_poses": 10},
            "ape": {"translation": {"rmse": 0.2}, "rotation": {"rmse": 1.5}},
            "rpe": {"translation_rmse_1m": {"rmse": 0.1}, "rotation_rmse_1m": {"rmse": 0.8}},
        }}}
        table = metrics_table(metrics)
        self.assertEqual(table.iloc[0]["translation APE RMSE (m)"], 0.2)
        self.assertEqual(table.iloc[0]["matched poses"], 10)

    def test_trajectory_frames_applies_saved_evo_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            output = run / "output"
            sequence = root / "prepared"
            output.mkdir(parents=True)
            sequence.mkdir()
            columns = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]
            row = [[1, 1, 2, 3, 0, 0, 0, 1]]
            pd.DataFrame(row, columns=columns).to_csv(output / "00000_KeyFrameTrajectory.csv", index=False)
            pd.DataFrame(row, columns=columns).to_csv(sequence / "groundtruth.csv", index=False)
            fast_lio = sequence / "references/fast_lio"
            fast_lio.mkdir(parents=True)
            pd.DataFrame(row, columns=columns).to_csv(fast_lio / "trajectory.csv", index=False)
            (output / "metrics.json").write_text(json.dumps({"comparisons": {
                "vslam_vs_robot_odometry": {"alignment": {
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "translation_m": [10, 0, 0], "scale": 1,
                }},
                "fast_lio_vs_robot_odometry": {"alignment": {
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "translation_m": [0, 5, 0], "scale": 1,
                }},
            }}))
            frames = trajectory_frames({"run_dir": str(run), "prepared_sequence": str(sequence)})
            self.assertEqual(frames["vslam"].iloc[0]["x"], 11)
            self.assertEqual(frames["fast lio"].iloc[0]["y"], 7)


if __name__ == "__main__":
    unittest.main()
