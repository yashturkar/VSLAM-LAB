import csv
import fcntl
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from Utilities.lightning_fastlio_pipeline import (
    Pipeline,
    PipelinePaths,
    StateStore,
    object_fingerprint,
    parse_topic_counts,
    sampled_file_identity,
)


class LightningFastLIOPipelineTests(unittest.TestCase):
    def make_source(self, root: Path, name: str = "hw-redo-test_seq003_20260816_170343_578407") -> Path:
        source = root / "hw-redo-test_20260816_165717" / name
        research = source / "research-bag"
        research.mkdir(parents=True)
        (source / "manifest.json").write_text(json.dumps({"name": name, "status": "complete"}), encoding="utf-8")
        (source / "processing.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
        (research / "research-bag_0.mcap").write_bytes(b"mcap" * 1024)
        topics = [
            "/cam_sync/cam0/image_preview/compressed",
            "/cam_sync/cam1/image_preview/compressed",
            "/cam_sync/cam0/camera_info",
            "/cam_sync/cam1/camera_info",
            "/odometry",
            "/ouster/points",
            "/ouster/imu",
        ]
        metadata = {
            "rosbag2_bagfile_information": {
                "topics_with_message_count": [
                    {"topic_metadata": {"name": topic}, "message_count": 10} for topic in topics
                ]
            }
        }
        (research / "metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
        return source

    def test_derives_existing_clid_run_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            paths = PipelinePaths(source, root / "local")
            self.assertEqual(paths.sequence_name, "hw-redo-test_seq003")
            self.assertEqual(paths.run_name, "hw-redo-test_20260816_165717_seq003")
            self.assertEqual(paths.config.name, "orbslam2_fastlio_seq003.yaml")

    def test_sampled_identity_detects_changed_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.mcap"
            path.write_bytes(b"a" * (3 * 1024 * 1024))
            before = sampled_file_identity(path)
            with path.open("r+b") as stream:
                stream.seek(-1, os.SEEK_END)
                stream.write(b"b")
            after = sampled_file_identity(path)
            self.assertNotEqual(before["sampled_sha256"], after["sampled_sha256"])

    def test_topic_counts_are_read_from_rosbag_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_source(Path(directory))
            counts = parse_topic_counts(source / "research-bag/metadata.yaml")
            self.assertEqual(counts["/odometry"], 10)

    def test_staged_bag_requires_matching_identity_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            paths = PipelinePaths(source, root / "local")
            paths.staged_research.mkdir(parents=True)
            shutil.copy2(paths.source_mcap(), paths.staged_mcap())
            shutil.copy2(paths.source_research / "metadata.yaml", paths.staged_research / "metadata.yaml")
            pipeline = Pipeline(paths, root / "workspace")
            identity = pipeline.source_identity()
            self.assertTrue(pipeline.stage_valid(identity))
            with paths.staged_mcap().open("ab") as stream:
                stream.write(b"corrupt")
            self.assertFalse(pipeline.stage_valid(identity))

    def test_extraction_integrity_checks_every_referenced_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            paths = PipelinePaths(source, root / "local")
            sequence = paths.sequence
            (sequence / "rgb_0").mkdir(parents=True)
            (sequence / "rgb_1").mkdir()
            for folder in ("rgb_0", "rgb_1"):
                (sequence / folder / "1.png").write_bytes(b"png")
            with (sequence / "rgb.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["ts_rgb_0 (ns)", "path_rgb_0", "ts_rgb_1 (ns)", "path_rgb_1"])
                writer.writerow([1, "rgb_0/1.png", 1, "rgb_1/1.png"])
            (sequence / "groundtruth.csv").write_text("ts,x,y,z,qx,qy,qz,qw\n1,0,0,0,0,0,0,1\n", encoding="utf-8")
            (sequence / "calibration.yaml").write_text("cameras: []\n", encoding="utf-8")
            fingerprint = "source-fingerprint"
            (sequence / "extraction_metadata.json").write_text(
                json.dumps({
                    "status": "complete",
                    "source_bag": str(paths.staged_mcap()),
                    "source_fingerprint": fingerprint,
                    "stereo_pairs": 1,
                    "odometry_poses": 1,
                }),
                encoding="utf-8",
            )
            pipeline = Pipeline(paths, root / "workspace")
            self.assertTrue(pipeline.extraction_valid(fingerprint))
            (sequence / "rgb_1/1.png").unlink()
            self.assertFalse(pipeline.extraction_valid(fingerprint))

    def test_force_from_invalidates_selected_and_downstream_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = PipelinePaths(self.make_source(root), root / "local")
            pipeline = Pipeline(paths, root / "workspace", force_from="orbslam2")
            self.assertFalse(pipeline.forced("extract"))
            self.assertTrue(pipeline.forced("orbslam2"))
            self.assertTrue(pipeline.forced("metrics"))

    def test_state_updates_are_atomic_and_reloadable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = PipelinePaths(self.make_source(root), root / "local")
            store = StateStore(paths)
            store.update(status="running")
            store.stage("extract", status="complete", fingerprint=object_fingerprint("x"))
            loaded = json.loads(paths.state.read_text(encoding="utf-8"))
            self.assertEqual(loaded["status"], "running")
            self.assertEqual(loaded["stages"]["extract"]["status"], "complete")
            self.assertFalse(paths.state.with_suffix(".json.tmp").exists())

    def test_preflight_refuses_nas_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_source(Path(directory))
            paths = PipelinePaths(source, Path("/mnt/share/nas/unsafe-output"))
            pipeline = Pipeline(paths, Path(directory) / "workspace")
            with patch("shutil.which", return_value="/usr/bin/rsync"):
                checks = pipeline.source_checks()
            self.assertTrue(any("unsafe" in error.lower() for error in checks["errors"]))

    def test_preflight_does_not_create_local_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            local = root / "not-created/local"
            pipeline = Pipeline(PipelinePaths(source, local), root / "workspace")
            pipeline.source_checks()
            self.assertFalse(local.exists())

    def test_run_orchestrates_all_stages_and_records_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            paths = PipelinePaths(source, root / "local")
            pipeline = Pipeline(paths, root / "workspace")
            trajectory = root / "trajectory.csv"
            trajectory.write_text("ts,x,y,z,qx,qy,qz,qw\n1,0,0,0,0,0,0,1\n", encoding="utf-8")
            checks = {"ok": True, "errors": [], "warnings": [], "topic_counts": {}, "free_bytes": 1}
            with (
                patch.object(pipeline, "source_checks", return_value=checks),
                patch.object(pipeline, "source_identity", return_value={"source": "id"}),
                patch.object(pipeline, "run_staging", return_value="stage") as stage,
                patch.object(pipeline, "run_extraction", return_value="extract") as extract,
                patch.object(pipeline, "run_config", return_value="config") as config,
                patch.object(pipeline, "run_orbslam2", return_value=("orb", trajectory)) as orb,
                patch.object(pipeline, "run_fastlio", return_value=("fast", root / "fast.csv")) as fast,
                patch.object(pipeline, "run_metrics", return_value="metrics") as metrics,
            ):
                self.assertEqual(pipeline.run(), paths.output)
            for mocked in (stage, extract, config, orb, fast, metrics):
                mocked.assert_called_once()
            state = json.loads(paths.state.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "complete")

    def test_run_refuses_a_concurrent_lock_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            paths = PipelinePaths(source, root / "local")
            paths.run_root.mkdir(parents=True)
            pipeline = Pipeline(paths, root / "workspace")
            checks = {"ok": True, "errors": [], "warnings": [], "topic_counts": {}, "free_bytes": 1}
            with paths.lock.open("w", encoding="utf-8") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(pipeline, "source_checks", return_value=checks):
                    with self.assertRaisesRegex(RuntimeError, "Another pipeline"):
                        pipeline.run()


if __name__ == "__main__":
    unittest.main()
