"""Cache Boreal FAST-LIO and interpolate rectified left-camera reference poses."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation, Slerp

from Utilities.boreal_fastlio import CONFIG as FAST_LIO_CONFIG
from Utilities.boreal_fastlio import OUTPUT_ROOT as FAST_LIO_OUTPUT_ROOT
from Utilities.boreal_fastlio import run as run_fast_lio
from Utilities.boreal_shared_reference import SHARED_ROOT, shared_trajectory
from Run.fastlio_reference import file_sha256

REFERENCE_CONFIG = Path(__file__).resolve().parents[1] / "configs/borealhdr_reference.yaml"
POSE_HEADER = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]


def _raw_scan_span(source: Path) -> tuple[int, int]:
    scans = sorted((source / "lidar").glob("*.csv"), key=lambda path: int(path.stem))
    if len(scans) < 3:
        raise ValueError(f"Fewer than three Boreal lidar scans: {source}")
    return int(scans[0].stem), int(scans[-1].stem)


def _validated_full_reference(folder: Path, source: Path, first_ns: int, last_ns: int) -> Path | None:
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        trajectory = folder / "trajectory.csv"
        if (manifest.get("status") != "complete" or manifest.get("duration_s") != 0
                or Path(manifest["source"]).resolve() != source.resolve()
                or manifest.get("config_sha256") != file_sha256(FAST_LIO_CONFIG)
                or not trajectory.is_file()):
            return None
        frame = pd.read_csv(trajectory)
        if len(frame) < 3 or list(frame.columns[:8]) != POSE_HEADER:
            return None
        values = frame.iloc[:, :8].to_numpy(dtype=float)
        stamps = frame.iloc[:, 0].to_numpy(dtype=np.int64)
        if (not np.isfinite(values).all() or (np.diff(stamps) <= 0).any()
                or (stamps[-1] - stamps[0]) < 0.9 * (last_ns - first_ns)):
            return None
        return trajectory
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def ensure_full_reference(source: Path, output_root: Path = FAST_LIO_OUTPUT_ROOT,
                          shared_root: Path = SHARED_ROOT,
                          shared_only: bool = False) -> Path:
    """Reuse a verified full run or generate one from the source CSVs."""
    source = source.resolve()
    shared = shared_trajectory(source, shared_root)
    if shared is not None:
        print(f"Boreal FAST-LIO: reusing shared {shared}", flush=True)
        return shared
    if (shared_root / source.name).exists():
        raise ValueError(f"Shared Boreal reference is present but invalid: {shared_root / source.name}")
    if shared_only:
        raise FileNotFoundError(f"No validated shared Boreal reference for {source.name} under {shared_root}; "
                                "refusing local FAST-LIO generation")
    first_ns, last_ns = _raw_scan_span(source)
    parent = output_root.resolve() / source.name
    for folder in sorted(parent.glob("*_full"), reverse=True):
        trajectory = _validated_full_reference(folder, source, first_ns, last_ns)
        if trajectory is not None:
            print(f"Boreal FAST-LIO: reusing {trajectory}", flush=True)
            return trajectory
    print(f"Boreal FAST-LIO: generating full reference for {source.name}", flush=True)
    output = run_fast_lio(source, duration=0, rate=1.0, output_root=output_root)
    trajectory = _validated_full_reference(output, source, first_ns, last_ns)
    if trajectory is None:
        raise RuntimeError(f"Generated FAST-LIO reference failed validation: {output}")
    return trajectory


def _camera_geometry(reference_config: Path, left_calibration: Path) -> tuple[np.ndarray, Rotation, dict[str, Any]]:
    geometry = yaml.safe_load(reference_config.read_text(encoding="utf-8"))
    if geometry.get("schema_version") != 1 or geometry.get("pose_frame") != "imu_link":
        raise ValueError(f"Unsupported Boreal reference geometry: {reference_config}")
    mount = geometry["imu_to_camera1"]
    offset = np.asarray(mount["xyz_m"], dtype=float)
    rpy = np.asarray(mount["rpy_rad"], dtype=float)
    if offset.shape != (3,) or rpy.shape != (3,) or not np.isfinite(offset).all() or not np.isfinite(rpy).all():
        raise ValueError("Invalid Boreal camera mounting transform")
    calibration = yaml.safe_load(left_calibration.read_text(encoding="utf-8"))
    rectification = np.asarray(calibration["rectification_matrix"]["data"], dtype=float).reshape(3, 3)
    if not np.allclose(rectification.T @ rectification, np.eye(3), atol=1e-3) or np.linalg.det(rectification) < 0.99:
        raise ValueError("Invalid left camera rectification rotation")
    # R1 maps raw optical vectors into rectified optical coordinates.
    orientation = Rotation.from_euler("xyz", rpy) * Rotation.from_matrix(rectification).inv()
    return offset, orientation, geometry


def _update_prepared_metadata(sequence: Path, provenance: dict[str, Any]) -> None:
    marker = sequence / "borealhdr.json"
    if not marker.is_file():
        return
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    metadata.update({
        "groundtruth": "boreal_fast_lio",
        "groundtruth_reference": str(sequence / "groundtruth_reference.json"),
        "groundtruth_coverage": provenance["reference_coverage"],
    })
    temporary = marker.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(marker)


def materialize_camera_reference(
    trajectory: Path,
    sequence: Path,
    left_calibration: Path,
    reference_config: Path = REFERENCE_CONFIG,
) -> dict[str, Any]:
    """Write in-range FAST-LIO camera poses at this variant's image timestamps."""
    sequence = sequence.resolve()
    rgb = sequence / "rgb.csv"
    destination = sequence / "groundtruth.csv"
    provenance_path = sequence / "groundtruth_reference.json"
    if destination.exists() and not provenance_path.exists():
        raise FileExistsError(f"Existing unowned Boreal reference will not be overwritten: {destination}")
    offset, camera_rotation, geometry = _camera_geometry(reference_config, left_calibration)
    fingerprint = hashlib.sha256(json.dumps({
        "schema_version": 1,
        "trajectory_sha256": file_sha256(trajectory),
        "rgb_sha256": file_sha256(rgb),
        "left_calibration_sha256": file_sha256(left_calibration),
        "geometry_sha256": file_sha256(reference_config),
    }, sort_keys=True).encode()).hexdigest()
    if provenance_path.exists() and destination.exists():
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint and existing.get("groundtruth_sha256") == file_sha256(destination):
            _update_prepared_metadata(sequence, existing)
            print(f"Camera reference: SKIP (verified) {destination}", flush=True)
            return existing

    reference = pd.read_csv(trajectory)
    if list(reference.columns[:8]) != POSE_HEADER:
        raise ValueError(f"Unexpected FAST-LIO trajectory format: {trajectory}")
    reference_ns = reference.iloc[:, 0].to_numpy(dtype=np.int64)
    positions = reference.iloc[:, 1:4].to_numpy(dtype=float)
    quaternions = reference.iloc[:, 4:8].to_numpy(dtype=float)
    image_ns = pd.read_csv(rgb).iloc[:, 0].to_numpy(dtype=np.int64)
    if (len(reference_ns) < 3 or len(image_ns) < 3 or (np.diff(reference_ns) <= 0).any()
            or (np.diff(image_ns) <= 0).any() or not np.isfinite(positions).all()
            or not np.isfinite(quaternions).all()):
        raise ValueError("Invalid FAST-LIO or image timestamps/poses")
    inside = (image_ns >= reference_ns[0]) & (image_ns <= reference_ns[-1])
    selected_ns = image_ns[inside]
    if len(selected_ns) < 3 or len(selected_ns) / len(image_ns) < 0.9:
        raise ValueError(f"FAST-LIO covers only {len(selected_ns)}/{len(image_ns)} Boreal images")
    origin_ns = reference_ns[0]
    ref_s = (reference_ns - origin_ns) / 1e9
    selected_s = (selected_ns - origin_ns) / 1e9
    camera_centers = np.column_stack([
        np.interp(selected_s, ref_s, positions[:, axis]) for axis in range(3)
    ])
    imu_rotation = Slerp(ref_s, Rotation.from_quat(quaternions))(selected_s)
    camera_centers += imu_rotation.apply(offset)
    camera_quaternions = (imu_rotation * camera_rotation).as_quat()

    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(POSE_HEADER)
        for stamp, xyz, quat in zip(selected_ns, camera_centers, camera_quaternions):
            writer.writerow([int(stamp), *[f"{value:.12g}" for value in xyz],
                             *[f"{value:.12g}" for value in quat]])
    temporary.replace(destination)
    provenance = {
        "schema_version": 1,
        "method": geometry["method"],
        "reference_quality": "experimental FAST-LIO reference; not surveyed ground truth",
        "source_trajectory": str(trajectory.resolve()),
        "source_manifest": str((trajectory.parent / "manifest.json").resolve()),
        "pose_frame": geometry["camera_frame"],
        "source_pose_frame": geometry["pose_frame"],
        "geometry_config": str(reference_config.resolve()),
        "left_calibration": str(left_calibration.resolve()),
        "interpolation": "linear translation and quaternion Slerp at selected image timestamps",
        "input_frames": len(image_ns),
        "reference_frames": len(selected_ns),
        "reference_coverage": len(selected_ns) / len(image_ns),
        "first_reference_timestamp_ns": int(selected_ns[0]),
        "last_reference_timestamp_ns": int(selected_ns[-1]),
        "fingerprint": fingerprint,
        "groundtruth_sha256": file_sha256(destination),
    }
    temporary = provenance_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    temporary.replace(provenance_path)
    _update_prepared_metadata(sequence, provenance)
    print(f"Camera reference: {len(selected_ns)}/{len(image_ns)} frames -> {destination}", flush=True)
    return provenance
