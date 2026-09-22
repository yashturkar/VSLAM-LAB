import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from Utilities.clid_relight import (
    DEFAULT_RELIGHTING_CONFIG,
    effective_power,
    load_components,
    load_schedule,
    prepare,
    relight_packed,
    render_profile,
    unpack_rggb,
    write_mono_calibration,
)


class ClidRelightTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(DEFAULT_RELIGHTING_CONFIG.read_text())

    def test_schedule_validation_and_sparse_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            schedule = Path(directory) / "schedule.yaml"
            schedule.write_text("0: {light: 0.25, exposure_ev: -1}\n2: {light: 1, exposure_ev: 2}\n")
            self.assertEqual(load_schedule(schedule, 4), [
                {"light": .25, "exposure_ev": -1.},
                {"light": .25, "exposure_ev": -1.},
                {"light": 1., "exposure_ev": 2.},
                {"light": 1., "exposure_ev": 2.},
            ])
            invalid = (
                "1: {light: 0, exposure_ev: 0}",
                '"0": {light: 0, exposure_ev: 0}',
                "0: {light: true, exposure_ev: 0}",
                "0: {light: .nan, exposure_ev: 0}",
                "0: {light: -0.1, exposure_ev: 0}",
                "0: {light: 0, exposure_ev: 2.1}",
                "0: {light: 0}",
                "0: {light: 0, exposure_ev: 0, extra: 1}",
                "0: {light: 0, exposure_ev: 0}\n0: {light: 1, exposure_ev: 0}",
                "4: {light: 0, exposure_ev: 0}",
                "[]",
            )
            for text in invalid:
                schedule.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    load_schedule(schedule, 4)

    def test_power_and_relighting_invariants(self):
        self.assertEqual(float(effective_power(self.config, 0)), 0.)
        self.assertEqual(float(effective_power(self.config, 1)), 1.)
        self.assertAlmostEqual(float(effective_power(self.config, .5)), .6447163, places=6)
        shape = (4, 2, 3)
        components = {
            "ambient": np.full(shape, .1, np.float32),
            "flash": np.full(shape, .2, np.float32),
            "correction1": np.full(shape, .3, np.float32),
            "correction2": np.full(shape, -.1, np.float32),
        }
        np.testing.assert_allclose(relight_packed(components, 0, 0, clip=False), components["ambient"])
        np.testing.assert_allclose(
            relight_packed(components, 1, 0, clip=False), components["ambient"] + components["flash"]
        )
        np.testing.assert_allclose(
            relight_packed(components, .5, 1, clip=False),
            2 * relight_packed(components, .5, 0, clip=False),
        )

    def test_component_validation_unpack_and_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "frame.npz"
            arrays = {key: np.full((4, 4, 5), .2, np.float16) for key in
                      ("ambient", "flash", "correction1", "correction2")}
            np.savez(archive, **arrays)
            loaded = load_components(archive, (4, 4, 5))
            self.assertTrue(all(value.dtype == np.float32 for value in loaded.values()))
            mosaic = unpack_rggb(loaded["ambient"])
            self.assertEqual(mosaic.shape, (8, 10))
            viewer = render_profile(loaded["ambient"], "viewer-rgb", self.config)
            gray = render_profile(loaded["ambient"], "linear-gray", self.config)
            self.assertEqual(viewer.shape, (8, 10, 3))
            self.assertEqual(gray.shape, (8, 10))
            self.assertEqual(viewer.dtype, np.uint8)
            self.assertEqual(gray.dtype, np.uint8)
            bad = Path(directory) / "bad.npz"
            arrays["ambient"] = arrays["ambient"].astype(np.float32)
            np.savez(bad, **arrays)
            with self.assertRaises(ValueError):
                load_components(bad, (4, 4, 5))

    def test_mono_calibration_uses_scaled_rectified_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.yaml"
            projection = [20., 0., 8., 0., 0., 22., 6., 0., 0., 0., 1., 0.]
            write_mono_calibration(path, {"projection": projection, "output_size": [16, 12], "fps": 10.})
            camera = json.loads(path.read_text())["cameras"]
            self.assertEqual(len(camera), 1)
            self.assertEqual(camera[0]["focal_length"], [20., 22.])
            self.assertEqual(camera[0]["principal_point"], [8., 6.])
            self.assertEqual(camera[0]["image_dimension"], [16, 12])
            self.assertEqual(camera[0]["distortion_coefficients"], [0.] * 5)

    def _source(self, root: Path) -> Path:
        source = root / "source"
        (source / "raw_0").mkdir(parents=True)
        (source / "clid_components_0").mkdir()
        (source / "rgb_0").mkdir()
        frames = []
        rgb_rows = ["ts_rgb_0 (ns),path_rgb_0"]
        for index, stamp in enumerate((100, 200, 300)):
            stem = str(stamp)
            np.save(source / "raw_0" / f"{stem}.npy", np.zeros((6, 8), np.uint16))
            values = {key: np.full((4, 2, 2), .1 + index * .1, np.float16)
                      for key in ("ambient", "flash", "correction1", "correction2")}
            np.savez(source / "clid_components_0" / f"{stem}.npz", **values)
            cv2.imwrite(str(source / "rgb_0" / f"{stem}.png"), np.zeros((6, 8, 3), np.uint8))
            frames.append({
                "index": index, "stem": stem, "timestamp_ns": stamp,
                "raw_file": f"raw_0/{stem}.npy", "components_file": f"clid_components_0/{stem}.npz",
            })
            rgb_rows.append(f"{stamp},rgb_0/{stem}.png")
        manifest = {
            "schema_version": 1, "status": "complete",
            "identity": {"config": {"sha256": self.config["accepted_producer_config_sha256"][0]}},
            "outputs": {"component_keys": ["ambient", "flash", "correction1", "correction2"],
                        "component_dtype": "float16", "component_shape": [4, 2, 2]},
            "frames": frames, "extracted_frame_count": 3,
        }
        (source / "clid_components_manifest.json").write_text(json.dumps(manifest))
        (source / "rgb.csv").write_text("\n".join(rgb_rows) + "\n")
        (source / "groundtruth.csv").write_text(
            "ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw\n100,0,0,0,0,0,0,1\n"
        )
        (source / "extraction_metadata.json").write_text(json.dumps({"source_bag": str(root / "bag.mcap")}))
        (root / "bag.mcap").touch()
        return source

    def test_prepare_renders_metadata_and_reuses_verified_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            schedule = root / "schedule.yaml"
            schedule.write_text("0: {light: 0.5, exposure_ev: 0}\n2: {light: 1, exposure_ev: 1}\n")
            identity_x = np.tile(np.arange(4, dtype=np.float32), (4, 1))
            identity_y = np.tile(np.arange(4, dtype=np.float32)[:, None], (1, 4))
            calibration = {
                "source_bag": str(root / "bag.mcap"), "source_size": [6, 6], "output_size": [4, 4],
                "scale": 2/3, "camera_matrix": np.eye(3).reshape(-1).tolist(), "distortion": [0.] * 5,
                "rectification": np.eye(3).reshape(-1).tolist(),
                "projection": [2., 0., 2., 0., 0., 2., 2., 0., 0., 0., 1., 0.],
                "fps": 10., "baseline_m": .1,
            }
            with patch("Utilities.clid_relight.scaled_camera", return_value=(calibration, (identity_x, identity_y))):
                output = prepare(source, schedule, output_root=root / "out")
            self.assertEqual(cv2.imread(str(output / "rgb_0/100.png"), -1).shape, (4, 4, 3))
            self.assertEqual(len(json.loads((output / "calibration.yaml").read_text())["cameras"]), 1)
            self.assertEqual(len((output / "relight.csv").read_text().splitlines()), 4)
            self.assertEqual(json.loads((output / "relight.json").read_text())["status"], "complete")
            with patch("Utilities.clid_relight.scaled_camera", return_value=(calibration, (identity_x, identity_y))), \
                 patch("Utilities.clid_relight.load_components", side_effect=AssertionError("cache not reused")):
                self.assertEqual(prepare(source, schedule, output_root=root / "out"), output)


if __name__ == "__main__":
    unittest.main()
