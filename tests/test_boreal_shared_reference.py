import shutil
import tempfile
import unittest
from pathlib import Path

from Utilities.boreal_reference import ensure_full_reference
from Utilities.boreal_shared_reference import (publish_reference, shared_trajectory,
                                                source_fingerprint, validate_trajectory)


class SharedBorealReferenceTests(unittest.TestCase):
    def test_publish_reuse_by_content_and_detect_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original/backpack_test"
            (source / "lidar").mkdir(parents=True)
            (source / "imu").mkdir()
            (source / "camera_left/4.0").mkdir(parents=True)
            (source / "camera_right/4.0").mkdir(parents=True)
            for stamp in (0, 1_000_000_000, 2_000_000_000):
                (source / "lidar" / f"{stamp}.csv").write_text(f"scan {stamp}\n")
                for side in ("left", "right"):
                    (source / f"camera_{side}/4.0/{stamp}.png").write_bytes(b"image")
            (source / "imu/imu.csv").write_text("timestamp\n0\n")
            local = root / "local/trajectory.csv"
            local.parent.mkdir()
            local.write_text("ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw\n"
                             "0,0,0,0,0,0,0,1\n"
                             "1000000000,0.5,0,0,0,0,0,1\n"
                             "2000000000,1,0,0,0,0,0,1\n")
            quality = validate_trajectory(source, local)
            self.assertEqual(quality["pose_count"], 3)
            self.assertEqual(quality["camera_reference_pairs"], 3)
            shared_root = root / "shared"
            published = publish_reference(source, local, shared_root)
            self.assertEqual(shared_trajectory(source, shared_root), published)
            self.assertEqual(publish_reference(source, local, shared_root), published)
            relocated = root / "relocated/backpack_test"
            shutil.copytree(source, relocated)
            self.assertEqual(source_fingerprint(source), source_fingerprint(relocated))
            self.assertEqual(shared_trajectory(relocated, shared_root), published)
            self.assertEqual(ensure_full_reference(relocated, root / "empty_local", shared_root), published)
            (relocated / "imu/imu.csv").write_text("timestamp\n1\n")
            self.assertIsNone(shared_trajectory(relocated, shared_root))
            with self.assertRaises(ValueError):
                ensure_full_reference(relocated, root / "empty_local", shared_root)


if __name__ == "__main__":
    unittest.main()
