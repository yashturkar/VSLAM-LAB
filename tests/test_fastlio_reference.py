import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Run.fastlio_reference import (
    apply_clock_mapping,
    fast_lio_reference_enabled,
    fit_clock_mapping,
    load_fast_lio_settings,
)


class FastLIOReferenceTests(unittest.TestCase):
    def test_disabled_config_does_not_require_source_bag_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "single.yaml"
            config.write_text("DATASET:\n  name: sample\n", encoding="utf-8")
            self.assertFalse(fast_lio_reference_enabled(config))

    def test_clock_fit_rejects_large_outlier(self):
        sensor = 1000.0 + np.arange(20, dtype=float) * 0.1
        record = 2000.0 + np.arange(20, dtype=float) * 0.10001
        record[8] += 2.0
        mapping = fit_clock_mapping(sensor, record)
        self.assertEqual(mapping["outliers"], 1)
        self.assertAlmostEqual(mapping["slope"], 1.0001, places=5)

    def test_clock_mapping_writes_canonical_nanosecond_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, corrected, csv_path = root / "raw.tum", root / "corrected.tum", root / "trajectory.csv"
            raw.write_text("10.0 1 2 3 0 0 0 1\n11.0 2 2 3 0 0 0 1\n", encoding="utf-8")
            mapping = {"sensor_origin_s": 10.0, "record_origin_s": 100.0, "slope": 1.0, "intercept_s": 0.0}
            self.assertEqual(apply_clock_mapping(raw, corrected, csv_path, mapping), 2)
            frame = pd.read_csv(csv_path)
            self.assertEqual(int(frame.iloc[0, 0]), 100_000_000_000)
            self.assertEqual(list(frame.columns), ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])

    def test_config_discovers_source_bag_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root / "sample"
            sequence.mkdir()
            bag = root / "bag.mcap"
            bag.touch()
            (sequence / "extraction_metadata.json").write_text(f'{{"source_bag": "{bag}"}}\n', encoding="utf-8")
            config = root / "single.yaml"
            config.write_text(
                f"DATASET:\n  base_path: {root}\n  name: sample\n  baseline: orbslam2\n  dataset: lightning\n"
                "EVALUATION:\n  fast_lio:\n    enabled: true\n",
                encoding="utf-8",
            )
            settings = load_fast_lio_settings(config)
            self.assertTrue(settings["enabled"])
            self.assertEqual(settings["source_bag"], bag)


if __name__ == "__main__":
    unittest.main()
