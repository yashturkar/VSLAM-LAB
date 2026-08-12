import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from Datasets.dataset_files.dataset_lightning import LightningDataset
from Run.single_sequence import (
    SingleSequenceConfig,
    _trajectory,
    headless_environment,
    load_dataset,
)


class SingleSequenceTests(unittest.TestCase):
    def test_config_paths_and_registry_selector(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "single.yaml"
            config.write_text(
                "DATASET:\n"
                f"  base_path: {root}\n"
                "  name: sample\n"
                "  baseline: mast3rslam\n"
                "  dataset: lightning\n"
                "  output_dir: results\n",
                encoding="utf-8",
            )
            loaded = SingleSequenceConfig.load(config)
            self.assertEqual(loaded.output_dir, root / "results")
            self.assertIsInstance(load_dataset("lightning", config), LightningDataset)

    def test_config_accepts_baseline_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "single.yaml"
            config.write_text(
                "DATASET:\n"
                f"  base_path: {root}\n"
                "  name: sample\n"
                "  baseline: colmap\n"
                "  dataset: lightning\n"
                "  parameters:\n"
                "    max_rgb: 100\n",
                encoding="utf-8",
            )
            self.assertEqual(SingleSequenceConfig.load(config).parameters, {"max_rgb": 100})

    def test_empty_trajectory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run_folder = Path(directory)
            (run_folder / "00000_KeyFrameTrajectory.csv").write_text("timestamp,x,y,z\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "empty trajectory"):
                _trajectory(run_folder)

    def test_lightning_fixture_is_prepared_without_copying_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "sample"
            image_dir = root / "image_0"
            right_image_dir = root / "image_1"
            (root / "sequences" / name).mkdir(parents=True)
            (root / "poses").mkdir()
            image_dir.mkdir()
            right_image_dir.mkdir()
            cv2.imwrite(str(image_dir / "000000.png"), np.zeros((4, 6, 3), dtype=np.uint8))
            cv2.imwrite(str(image_dir / "000001.png"), np.zeros((4, 6, 3), dtype=np.uint8))
            cv2.imwrite(str(right_image_dir / "000000.png"), np.zeros((4, 6, 3), dtype=np.uint8))
            cv2.imwrite(str(right_image_dir / "000001.png"), np.zeros((4, 6, 3), dtype=np.uint8))
            (root / "sequences" / name / "times.txt").write_text("1.25\n1.35\n", encoding="utf-8")
            (root / "poses" / f"{name}.txt").write_text(
                "1 0 0 0 0 1 0 0 0 0 1 0\n1 0 0 0 0 1 0 0 0 0 1 0\n", encoding="utf-8"
            )
            (root / "config.yaml").write_text(
                "Camera:\n  fx: 10\n  fy: 11\n  cx: 3\n  cy: 2\n  fps: 5\n  bf: 2\n"
                "Stereo.R: [1, 0, 0, 0, 1, 0, 0, 0, 1]\n"
                "Stereo.T: [-0.2, 0, 0]\n",
                encoding="utf-8",
            )

            dataset = LightningDataset()
            sequence = dataset.prepare_local_sequence(root, name)
            self.assertTrue((sequence / "rgb_0").is_symlink())
            self.assertTrue((sequence / "rgb_1").is_symlink())
            rgb = pd.read_csv(sequence / "rgb.csv")
            groundtruth = pd.read_csv(sequence / "groundtruth.csv")
            self.assertEqual(int(rgb.iloc[0, 0]), 1_250_000_000)
            self.assertEqual(int(groundtruth.iloc[0, 0]), 1_250_000_000)
            self.assertAlmostEqual(dataset.rgb_hz, 10.0)
            self.assertEqual(list(rgb.columns), ["ts_rgb_0 (ns)", "path_rgb_0", "ts_rgb_1 (ns)", "path_rgb_1"])
            self.assertIn(
                "cam_model: radtan5",
                (sequence / "calibration.yaml").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "distortion_type: radtan5",
                (sequence / "calibration.yaml").read_text(encoding="utf-8"),
            )
            self.assertIn("cam_name: rgb_1", (sequence / "calibration.yaml").read_text(encoding="utf-8"))
            self.assertIn("0.2000000000000", (sequence / "calibration.yaml").read_text(encoding="utf-8"))
            calibration = sequence / "calibration.yaml"
            calibration.write_text(
                calibration.read_text(encoding="utf-8").replace("distortion_type: radtan5", "distortion_type: radtan"),
                encoding="utf-8",
            )
            LightningDataset().prepare_local_sequence(root, name)
            self.assertIn("distortion_type: radtan5", calibration.read_text(encoding="utf-8"))

    def test_nested_images_and_ancestor_calibration_are_discovered(self):
        with tempfile.TemporaryDirectory() as directory:
            processed = Path(directory) / "Processed"
            root = processed / "sample_extract"
            name = "sample_extract"
            image_dir = root / "sequences" / name / "image_0"
            image_dir.mkdir(parents=True)
            (root / "poses").mkdir()
            cv2.imwrite(str(image_dir / "000000.png"), np.zeros((4, 6, 3), dtype=np.uint8))
            (root / "sequences" / name / "times.txt").write_text("1.25\n", encoding="utf-8")
            (root / "poses" / f"{name}.txt").write_text("1 0 0 0 0 1 0 0 0 0 1 0\n", encoding="utf-8")
            (processed / "lightning.yaml").write_text(
                "Camera.fx: 10\nCamera.fy: 11\nCamera.cx: 3\nCamera.cy: 2\nCamera.fps: 5\n",
                encoding="utf-8",
            )
            sequence = LightningDataset().prepare_local_sequence(root, name)
            self.assertEqual((sequence / "rgb_0").resolve(), image_dir.resolve())

    def test_legacy_absolute_module_path_remains_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module = root / "dataset_custom.py"
            module.write_text(
                "from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB\n"
                "class CUSTOM_dataset(DatasetVSLAMLAB):\n"
                "    def __init__(self, benchmark_path, dataset_name='custom'):\n"
                "        self.benchmark_path = benchmark_path\n"
                "        self.dataset_name = dataset_name\n"
                "    def download_sequence_data(self, name): pass\n"
                "    def create_rgb_folder(self, name): pass\n"
                "    def create_rgb_csv(self, name): pass\n"
                "    def create_calibration_yaml(self, name): pass\n",
                encoding="utf-8",
            )
            dataset = load_dataset(str(module), benchmark_path=root)
            self.assertEqual(dataset.benchmark_path, root)

    def test_headless_environment_restores_values(self):
        import os

        original = os.environ.get("DISPLAY")
        with headless_environment(True):
            self.assertEqual(os.environ["DISPLAY"], "")
        self.assertEqual(os.environ.get("DISPLAY"), original)


if __name__ == "__main__":
    unittest.main()
