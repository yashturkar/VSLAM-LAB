import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from Datasets.dataset_files.dataset_lightning import LightningDataset
from Run.single_sequence import SingleSequenceConfig, headless_environment, load_dataset


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

    def test_lightning_fixture_is_prepared_without_copying_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "sample"
            image_dir = root / "image_0"
            (root / "sequences" / name).mkdir(parents=True)
            (root / "poses").mkdir()
            image_dir.mkdir()
            cv2.imwrite(str(image_dir / "000000.png"), np.zeros((4, 6, 3), dtype=np.uint8))
            (root / "sequences" / name / "times.txt").write_text("1.25\n", encoding="utf-8")
            (root / "poses" / f"{name}.txt").write_text("1 0 0 0 0 1 0 0 0 0 1 0\n", encoding="utf-8")
            (root / "config.yaml").write_text("Camera:\n  fx: 10\n  fy: 11\n  cx: 3\n  cy: 2\n  fps: 5\n", encoding="utf-8")

            dataset = LightningDataset()
            sequence = dataset.prepare_local_sequence(root, name)
            self.assertTrue((sequence / "rgb_0").is_symlink())
            rgb = pd.read_csv(sequence / "rgb.csv")
            groundtruth = pd.read_csv(sequence / "groundtruth.csv")
            self.assertEqual(int(rgb.iloc[0, 0]), 1_250_000_000)
            self.assertEqual(int(groundtruth.iloc[0, 0]), 1_250_000_000)
            self.assertIn("cam_model: radtan5", (sequence / "calibration.yaml").read_text(encoding="utf-8"))

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
