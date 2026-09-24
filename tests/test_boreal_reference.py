import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation

from Utilities.boreal_reference import materialize_camera_reference


class BorealReferenceTests(unittest.TestCase):
    def test_interpolates_selected_times_and_transforms_rectified_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root / "sequence"
            sequence.mkdir()
            image_ns = [500_000_000, 1_500_000_000, 2_500_000_000]
            pd.DataFrame({"ts_rgb_0 (ns)": image_ns}).to_csv(sequence / "rgb.csv", index=False)

            trajectory = root / "trajectory.csv"
            quaternions = Rotation.from_euler("z", [[0], [90], [180], [270]], degrees=True).as_quat()
            rows = [[index * 1_000_000_000, index, 0, 0, *quaternion]
                    for index, quaternion in enumerate(quaternions)]
            pd.DataFrame(rows, columns=["ts (ns)", "tx (m)", "ty (m)", "tz (m)",
                                             "qx", "qy", "qz", "qw"]).to_csv(trajectory, index=False)
            geometry = root / "geometry.yaml"
            geometry.write_text(yaml.safe_dump({
                "schema_version": 1, "method": "test_lio", "pose_frame": "imu_link",
                "camera_frame": "rectified_camera1_link",
                "imu_to_camera1": {"xyz_m": [1, 0, 0], "rpy_rad": [0, 0, 0]},
            }))
            calibration = root / "left.yaml"
            r1 = Rotation.from_euler("z", 90, degrees=True).as_matrix()
            calibration.write_text(yaml.safe_dump({"rectification_matrix": {"data": r1.ravel().tolist()}}))

            provenance = materialize_camera_reference(trajectory, sequence, calibration, geometry)
            result = pd.read_csv(sequence / "groundtruth.csv")
            self.assertEqual(result.iloc[:, 0].tolist(), image_ns)
            expected_first = [0.5 + np.sqrt(0.5), np.sqrt(0.5), 0]
            np.testing.assert_allclose(result.iloc[0, 1:4].to_numpy(dtype=float), expected_first, atol=1e-9)
            first_rotation = Rotation.from_quat(result.iloc[0, 4:8].to_numpy(dtype=float))
            np.testing.assert_allclose(first_rotation.as_euler("xyz", degrees=True), [0, 0, -45], atol=1e-8)
            self.assertEqual(provenance["reference_frames"], 3)
            self.assertEqual(json.loads((sequence / "groundtruth_reference.json").read_text())["method"], "test_lio")
            self.assertEqual(materialize_camera_reference(trajectory, sequence, calibration, geometry), provenance)


if __name__ == "__main__":
    unittest.main()
