"""Adapter for locally produced LIGHTNING mono or stereo sequences."""

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
        required = (
            self.rgb_path(sequence_name),
            self.rgb_csv_path(sequence_name),
            self.calibration_yaml_path(sequence_name),
            self.groundtruth_csv_path(sequence_name),
        )
        if all(path.exists() for path in required):
            raw_times = base_path / "sequences" / sequence_name / "times.txt"
            calibration_source = self._find_calibration(base_path)
            if raw_times.is_file() and calibration_source.is_file():
                times = [float(line) for line in raw_times.read_text(encoding="utf-8").splitlines() if line.strip()]
                left_images = self._images(self.rgb_path(sequence_name))
                right_source = base_path / "sequences" / sequence_name / "image_1"
                right_path = sequence_path / "rgb_1"
                if right_source.is_dir() and not right_path.exists():
                    right_path.symlink_to(os.path.relpath(right_source, right_path.parent), target_is_directory=True)
                right_images = self._images(right_path) if right_path.is_dir() else []
                self._write_rgb_csv(sequence_name, times, left_images, right_images)
                self._write_calibration(
                    sequence_name,
                    calibration_source,
                    left_images[0] if left_images else None,
                    times,
                    stereo=bool(right_images),
                )
            return sequence_path

        nested_sequence = base_path / "sequences" / sequence_name
        image_source = base_path / "image_0"
        if not image_source.is_dir():
            image_source = nested_sequence / "image_0"
        right_source = base_path / "image_1"
        if not right_source.is_dir():
            right_source = nested_sequence / "image_1"
        times_path = base_path / "sequences" / sequence_name / "times.txt"
        poses_path = base_path / "poses" / f"{sequence_name}.txt"
        calibration_source = self._find_calibration(base_path)
        missing = [path for path in (image_source, times_path, poses_path, calibration_source) if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing LIGHTNING input(s): " + ", ".join(map(str, missing)))

        rgb_path = self.rgb_path(sequence_name)
        if not rgb_path.exists():
            rgb_path.symlink_to(os.path.relpath(image_source, rgb_path.parent), target_is_directory=True)
        right_path = sequence_path / "rgb_1"
        if right_source.is_dir() and not right_path.exists():
            right_path.symlink_to(os.path.relpath(right_source, right_path.parent), target_is_directory=True)

        times = [float(line.strip()) for line in times_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        left_images = self._images(image_source)
        right_images = self._images(right_source) if right_source.is_dir() else []
        self._write_rgb_csv(sequence_name, times, left_images, right_images)
        self._write_groundtruth(sequence_name, poses_path, times)
        self._write_calibration(
            sequence_name,
            calibration_source,
            left_images[0] if left_images else None,
            times,
            stereo=bool(right_images),
        )
        return sequence_path

    @staticmethod
    def _images(folder: Path) -> list[Path]:
        return sorted(
            path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

    def _write_rgb_csv(
        self, sequence_name: str, times: list[float], left_images: list[Path], right_images: list[Path]
    ) -> None:
        if len(times) != len(left_images):
            raise ValueError(f"LIGHTNING timestamp/left-image count mismatch: {len(times)} != {len(left_images)}")
        timestamps = [int(timestamp * 1e9) for timestamp in times]
        if right_images:
            if len(right_images) != len(left_images):
                raise ValueError(
                    f"LIGHTNING left/right image count mismatch: {len(left_images)} != {len(right_images)}"
                )
            write_csv_rows(
                self.rgb_csv_path(sequence_name),
                ["ts_rgb_0 (ns)", "path_rgb_0", "ts_rgb_1 (ns)", "path_rgb_1"],
                [
                    [timestamp, f"rgb_0/{left.name}", timestamp, f"rgb_1/{right.name}"]
                    for timestamp, left, right in zip(timestamps, left_images, right_images)
                ],
            )
            return
        write_csv_rows(
            self.rgb_csv_path(sequence_name),
            ["ts_rgb_0 (ns)", "path_rgb_0"],
            [[timestamp, f"rgb_0/{image.name}"] for timestamp, image in zip(timestamps, left_images)],
        )

    @staticmethod
    def _find_calibration(base_path: Path) -> Path:
        """Find per-sequence config.yaml or a shared ancestor lightning.yaml."""
        direct = base_path / "config.yaml"
        if direct.is_file():
            return direct
        for parent in (base_path, *base_path.parents):
            candidate = parent / "lightning.yaml"
            if candidate.is_file():
                return candidate
        return direct

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

    def _write_calibration(
        self,
        sequence_name: str,
        source: Path,
        first_image: Path | None,
        times: list[float] | None = None,
        stereo: bool = False,
    ) -> None:
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
        fps = value("fps", self.rgb_hz)
        if times and len(times) > 1:
            intervals = np.diff(np.asarray(times, dtype=float))
            positive_intervals = intervals[intervals > 0]
            if positive_intervals.size:
                fps = float(1.0 / np.median(positive_intervals))
        rgb0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "mono",
            "cam_model": "radtan5",
            "focal_length": [
                value("fx", 1446.9127793242951),
                value("fy", 1451.5846408378259),
            ],
            "principal_point": [
                value("cx", 964.9426652255537),
                value("cy", 607.0681454495964),
            ],
            "distortion_type": "radtan5",
            "distortion_coefficients": [
                value("k1", -0.13902893236244782),
                value("k2", 0.23675668936161912),
                value("p1", -0.0006401710568311474),
                value("p2", 0.000710816965242213),
                value("k3", -0.2731326697949815),
            ],
            "image_dimension": [width, height],
            "fps": fps,
            "T_BS": np.eye(4),
        }
        self.rgb_hz = float(rgb0["fps"])
        cameras = [rgb0]
        if stereo:
            right_extrinsics = np.eye(4)
            rotation = config.get("Stereo.R")
            translation = config.get("Stereo.T")
            if rotation is not None and translation is not None:
                left_to_right_rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
                left_to_right_translation = np.asarray(translation, dtype=float).reshape(3)
                right_extrinsics[:3, :3] = left_to_right_rotation.T
                right_extrinsics[:3, 3] = -left_to_right_rotation.T @ left_to_right_translation
            else:
                right_extrinsics[0, 3] = value("bf", 240.0) / rgb0["focal_length"][0]
            rgb1 = dict(rgb0)
            rgb1.update({"cam_name": "rgb_1", "T_BS": right_extrinsics})
            cameras.append(rgb1)
        self.write_calibration_yaml(sequence_name, rgb=cameras)

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        if not self.groundtruth_csv_path(sequence_name).exists():
            raise FileNotFoundError(f"Missing ground truth: {self.groundtruth_csv_path(sequence_name)}")

    def remove_unused_files(self, sequence_name: str) -> None:
        return
