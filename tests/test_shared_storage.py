import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import shared_storage


class SharedStorageTests(unittest.TestCase):
    def test_relocates_and_links_git_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "storage-config"
            config.write_text(str(root / "shared") + "\n", encoding="utf-8")
            baseline = root / "workspace" / "Baselines" / "Example"
            (baseline / ".git").mkdir(parents=True)
            (baseline / "checkpoint.pth").write_bytes(b"weights")
            with patch.object(shared_storage, "STORAGE_CONFIG", config):
                shared_storage.relocate_baseline(baseline)
            self.assertTrue(baseline.is_symlink())
            self.assertEqual((baseline / "checkpoint.pth").read_bytes(), b"weights")

    def test_occupied_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "storage-config"
            shared = root / "shared"
            config.write_text(str(shared) + "\n", encoding="utf-8")
            baseline = root / "workspace" / "Baselines" / "Example"
            (baseline / ".git").mkdir(parents=True)
            (shared / "baselines" / "Example").mkdir(parents=True)
            with patch.object(shared_storage, "STORAGE_CONFIG", config):
                with self.assertRaises(FileExistsError):
                    shared_storage.relocate_baseline(baseline)


if __name__ == "__main__":
    unittest.main()
