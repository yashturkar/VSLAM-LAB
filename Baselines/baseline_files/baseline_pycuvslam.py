from pathlib import Path
from Baselines.BaselineVSLAMLAB import BaselineVSLAMLAB

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


class PYCUVSLAM_baseline(BaselineVSLAMLAB):
    """PyCuVSLAM helper for VSLAM-LAB Baselines."""

    def __init__(self, baseline_name: str = 'pycuvslam', baseline_folder: str = 'PyCuVSLAM') -> None:

        default_parameters = {'verbose': 1, 'mode': 'mono'}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = (0.850, 0.700, 0.300)
        self.modes = ['mono', 'rgbd', 'stereo', 'stereo-vi']
        self.cam_models = ['pinhole', 'radtan4', 'radtan5', 'equid4']

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_python(exp_it, exp, dataset, sequence_name)

    @staticmethod
    def _is_elf(path: Path) -> bool:
        try:
            with path.open("rb") as file:
                return file.read(4) == b"\x7fELF"
        except OSError:
            return False

    def is_installed(self) -> tuple[bool, str]:
        install_log = self.baseline_path / 'install_pycuvslam.txt'
        runtime = self.baseline_path / 'bin' / 'x86_64' / 'cuvslam'
        extensions = (runtime / 'libcuvslam.so', runtime / 'pycuvslam.so')
        is_installed = install_log.is_file() and all(self._is_elf(path) for path in extensions)
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)')
