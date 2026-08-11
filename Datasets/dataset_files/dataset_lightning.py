"""Adapter for locally produced LIGHTNING monocular sequences."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from utilities import write_csv_rows


class LightningDataset(DatasetVSLAMLAB):
    """Prepare a local KITTI-pose sequence for the VSLAM-LAB pipeline."""

    def __init__(self, dataset_name: str = "lightning") -> None:
        super().__init__(dataset_name)

    def download_sequence_data(self, sequence_name: str) -> None:
        if not self.sequence_path(sequence_name).exists():
            raise FileNotFoundError(
                f"LIGHTNING is local-only; provide the sequence at {self.sequence_path(sequence_name)}"
            )

    def create_rgb_folder(self, sequence_name: str) -> None:
        # Local single-sequence preparation creates this link before the standard pipeline runs.
        if not self.rgb_path(sequence_name).exists():
            raise FileNotFoundError(f"Missing image directory: {self.rgb_path(sequence_name)}")

    def create_rgb_csv(self, sequence_name: str) -> None:
        if not self.rgb_csv_path(sequence_name).exists():
            raise FileNotFoundError(f"Missing timestamp file: {self.rgb_csv_path(sequence_name)}")

    def create_calibration_yaml(self, sequence_name: str) -> None:
        if not self.calibration_yaml_path(sequence_name).exists():
            raise FileNotFoundError(f"Missing calibration: {self.calibration_yaml_path(sequence_name)}")

    def prepare_local_sequence(self, base_path: Path, sequence_name: str) -> Path:
        """Normalize the Streamlit-style LIGHTNING layout without duplicating images."""
        self.dataset_path = base_path
        self.sequence_names = [sequence_name]
        sequence_path = self.sequence_path(sequence_name)
        sequence_path.mkdir(parents=True, exist_ok=True)

        # Already prepared layouts need no conversion.
        required = (self.rgb_path(sequence_name), self.rgb_csv_path(sequence_name),
                    self.calibration_yaml_path(sequence_name), self.groundtruth_csv_path(sequence_name))
        if all(path.exists() for path in required):
            return sequence_path

        image_source = base_path / "image_0"
        times_path = base_path / "sequences" / sequence_name / "times.txt"
        poses_path = base_path / "poses" / f"{sequence_name}.txt"
        calibration_source = base_path / "config.yaml"
        missing = [path for path in (image_source, times_path, poses_path, calibration_source) if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing LIGHTNING input(s): " + ", ".join(map(str, missing)))

        rgb_path = self.rgb_path(sequence_name)
        if not rgb_path.exists():
            rgb_path.symlink_to(os.path.relpath(image_source, rgb_path.parent), target_is_directory=True)

        times = [float(line.strip()) for line in times_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        images = sorted(
            path for path in image_source.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if len(times) != len(images):
            raise ValueError(f"LIGHTNING timestamp/image count mismatch: {len(times)} != {len(images)}")
        write_csv_rows(
            self.rgb_csv_path(sequence_name),
            ["ts_rgb_0 (ns)", "path_rgb_0"],
            [[int(timestamp * 1e9), f"rgb_0/{image.name}"] for timestamp, image in zip(times, images)],
        )
        self._write_groundtruth(sequence_name, poses_path, times)
        self._write_calibration(sequence_name, calibration_source, images[0] if images else None)
        return sequence_path

    def _write_groundtruth(self, sequence_name: str, poses_path: Path, times: list[float]) -> None:
        rows: list[list[Any]] = []
        pose_lines = [line for line in poses_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(pose_lines) != len(times):
            raise ValueError(f"LIGHTNING timestamp/pose count mismatch: {len(times)} != {len(pose_lines)}")
        for timestamp, line in zip(times, pose_lines):
            values = [float(value) for value in line.split()]
            if len(values) != 12:
                raise ValueError("Each LIGHTNING pose must contain a row-major 3x4 matrix")
            matrix = np.array(values, dtype=float).reshape(3, 4)
            qx, qy, qz, qw = Rotation.from_matrix(matrix[:, :3]).as_quat()
            tx, ty, tz = matrix[:, 3]
            rows.append([int(timestamp * 1e9), tx, ty, tz, qx, qy, qz, qw])
        write_csv_rows(
            self.groundtruth_csv_path(sequence_name),
            ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"],
            rows,
        )

    def _write_calibration(self, sequence_name: str, source: Path, first_image: Path | None) -> None:
        with open(source, encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        camera = config.get("Camera", {}) if isinstance(config.get("Camera"), dict) else {}

        def value(name: str, default: float) -> float:
            return float(camera.get(name, config.get(f"Camera.{name}", default)))

        width, height = 1920, 1200
        if first_image is not None:
            image = cv2.imread(str(first_image))
            if image is not None:
                height, width = image.shape[:2]
        rgb0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "mono",
            "cam_model": "radtan5",
            "focal_length": [value("fx", 1446.9127793242951), value("fy", 1451.5846408378259)],
            "principal_point": [value("cx", 964.9426652255537), value("cy", 607.0681454495964)],
            "distortion_type": "radtan",
            "distortion_coefficients": [
                value("k1", -0.13902893236244782), value("k2", 0.23675668936161912),
                value("p1", -0.0006401710568311474), value("p2", 0.000710816965242213),
                value("k3", -0.2731326697949815),
            ],
            "image_dimension": [width, height],
            "fps": value("fps", self.rgb_hz),
            "T_BS": np.eye(4),
        }
        self.rgb_hz = float(rgb0["fps"])
        self.write_calibration_yaml(sequence_name, rgb=[rgb0])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        if not self.groundtruth_csv_path(sequence_name).exists():
            raise FileNotFoundError(f"Missing ground truth: {self.groundtruth_csv_path(sequence_name)}")

    def remove_unused_files(self, sequence_name: str) -> None:
        return
