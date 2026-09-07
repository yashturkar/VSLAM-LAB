"""Native prepared BorealHDR sequences, with optional reference poses."""

from Datasets.dataset_files.dataset_lightning import LightningDataset


class BorealHDRDataset(LightningDataset):
    def __init__(self, dataset_name="borealhdr"):
        super().__init__(dataset_name)

    def prepare_local_sequence(self, base_path, sequence_name):
        self.dataset_path = base_path
        self.sequence_names = [sequence_name]
        sequence = base_path / sequence_name
        for name in ("rgb_0", "rgb_1", "rgb.csv", "calibration.yaml"):
            if not (sequence / name).exists():
                raise FileNotFoundError(f"Prepare BorealHDR first: missing {sequence / name}")
        self._update_rgb_hz(sequence / "rgb.csv")
        return sequence
