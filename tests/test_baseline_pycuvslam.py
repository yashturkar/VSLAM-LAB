import tempfile
import unittest
from pathlib import Path

from Baselines.baseline_files.baseline_pycuvslam import PYCUVSLAM_baseline


class PyCuVSLAMBaselineTests(unittest.TestCase):
    def test_installation_requires_real_elf_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "bin" / "x86_64" / "cuvslam"
            runtime.mkdir(parents=True)
            (root / "install_pycuvslam.txt").touch()
            (runtime / "libcuvslam.so").write_bytes(b"\x7fELF-runtime")
            binding = runtime / "pycuvslam.so"
            binding.write_bytes(b"\x7fELF-binding")

            baseline = PYCUVSLAM_baseline()
            baseline.baseline_path = root
            self.assertTrue(baseline.is_installed()[0])

            binding.write_text("version https://git-lfs.github.com/spec/v1\n", encoding="utf-8")
            self.assertFalse(baseline.is_installed()[0])


if __name__ == "__main__":
    unittest.main()
