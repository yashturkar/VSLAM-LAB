"""Publish and verify portable Boreal FAST-LIO reference sidecars."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from Run.fastlio_reference import file_sha256

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/fastlio/boreal_vlp16.yaml"
SHARED_ROOT = Path("/mnt/share/nas/eph/BorealHDR-references/fastlio-v1")
POSE_HEADER = ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"]


def source_fingerprint(source: Path) -> dict:
    """Path-independent sensor identity; all scan sizes and three scan hashes."""
    scans = sorted((source / "lidar").glob("*.csv"), key=lambda path: int(path.stem))
    imu = source / "imu/imu.csv"
    if len(scans) < 3 or not imu.is_file():
        raise ValueError(f"Expected lidar scans and imu/imu.csv under {source}")
    digest = hashlib.sha256()
    digest.update(b"boreal-source-fingerprint-v1\n")
    digest.update(file_sha256(imu).encode())
    for scan in scans:
        digest.update(f"{scan.name}:{scan.stat().st_size}\n".encode())
    for scan in (scans[0], scans[len(scans) // 2], scans[-1]):
        digest.update(file_sha256(scan).encode())
    return {"algorithm": "imu-sha256+all-scan-names-sizes+first-middle-last-scan-sha256-v1",
            "sha256": digest.hexdigest(), "scan_count": len(scans),
            "first_scan_ns": int(scans[0].stem), "last_scan_ns": int(scans[-1].stem)}


def validate_trajectory(source: Path, trajectory: Path) -> dict:
    fingerprint = source_fingerprint(source)
    frame = pd.read_csv(trajectory)
    if list(frame.columns[:8]) != POSE_HEADER or len(frame) < 3:
        raise ValueError(f"Invalid trajectory columns or pose count: {trajectory}")
    values = frame.iloc[:, :8].to_numpy(dtype=float)
    stamps = frame.iloc[:, 0].to_numpy(dtype=np.int64)
    if not np.isfinite(values).all() or (np.diff(stamps) <= 0).any():
        raise ValueError("Nonfinite or nonmonotonic FAST-LIO poses")
    quaternions = values[:, 4:8]
    if not np.allclose(np.linalg.norm(quaternions, axis=1), 1, atol=1e-2):
        raise ValueError("FAST-LIO quaternions are not unit length")
    sensor_span = (fingerprint["last_scan_ns"] - fingerprint["first_scan_ns"]) / 1e9
    pose_span = float((stamps[-1] - stamps[0]) / 1e9)
    coverage = float(pose_span / sensor_span)
    if coverage < 0.9:
        raise ValueError(f"FAST-LIO temporal coverage {coverage:.1%} is below 90%")
    xyz = values[:, 1:4]
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    speed = steps / (np.diff(stamps) / 1e9)
    if not np.isfinite(speed).all() or speed.max() > 10:
        raise ValueError(f"Implausible FAST-LIO step speed: {speed.max():.2f} m/s")
    quality = {"status": "passed", "pose_count": len(frame), "pose_span_s": pose_span,
               "sensor_span_s": sensor_span, "temporal_coverage_ratio": coverage,
               "path_length_m": float(steps.sum()), "maximum_step_speed_m_s": float(speed.max())}
    left = source / "camera_left/4.0"
    right = source / "camera_right/4.0"
    if left.is_dir() and right.is_dir():
        left_stamps = {int(path.stem) for path in left.glob("*.png")}
        right_stamps = {int(path.stem) for path in right.glob("*.png")}
        pairs = sorted(left_stamps & right_stamps)
        if len(pairs) < 3:
            raise ValueError("Fewer than three synchronized 4 ms stereo pairs")
        covered = int(sum(stamps[0] <= stamp <= stamps[-1] for stamp in pairs))
        camera_coverage = float(covered / len(pairs))
        quality.update(camera_pairs=len(pairs), camera_reference_pairs=covered,
                       camera_timestamp_coverage_ratio=camera_coverage)
        if camera_coverage < 0.9:
            raise ValueError(f"Only {covered}/{len(pairs)} camera pairs covered by FAST-LIO")
    gnss_file = source / "gnss/gnss.csv"
    if gnss_file.is_file():
        fixes = pd.read_csv(gnss_file)
        fix_ns = fixes["timestamp"].to_numpy(dtype=np.int64)
        lat = np.deg2rad(fixes["latitude"].to_numpy(dtype=float))
        lon = np.deg2rad(fixes["longitude"].to_numpy(dtype=float))
        if len(fix_ns) < 3 or (np.diff(fix_ns) <= 0).any() or not np.isfinite(lat).all() or not np.isfinite(lon).all():
            raise ValueError("Invalid GNSS sanity-check trace")
        east_north = np.column_stack(((lon - lon[0]) * 6_371_000 * np.cos(lat[0]),
                                      (lat - lat[0]) * 6_371_000))
        overlap = (stamps >= fix_ns[0]) & (stamps <= fix_ns[-1])
        if overlap.sum() < 3:
            raise ValueError("Insufficient FAST-LIO/GNSS timestamp overlap")
        matched = np.column_stack([np.interp(stamps[overlap], fix_ns, east_north[:, axis])
                                   for axis in range(2)])
        centered_pose = xyz[overlap, :2] - xyz[overlap, :2].mean(axis=0)
        centered_gnss = matched - matched.mean(axis=0)
        u, _, vt = np.linalg.svd(centered_pose.T @ centered_gnss)
        rotation = u @ np.diag([1.0, np.linalg.det(u @ vt)]) @ vt
        residuals = centered_pose @ rotation - centered_gnss
        rmse = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
        gnss_length = float(np.linalg.norm(np.diff(matched, axis=0), axis=1).sum())
        lio_length = float(np.linalg.norm(np.diff(xyz[overlap, :2], axis=0), axis=1).sum())
        ratio = lio_length / gnss_length if gnss_length > 0 else float("inf")
        quality.update(gnss_aligned_2d_rmse_m=rmse, gnss_overlap_poses=int(overlap.sum()),
                       horizontal_path_ratio_to_gnss=ratio)
        if rmse > 5 or not 0.5 <= ratio <= 2.0:
            raise ValueError(f"GNSS sanity check failed: RMSE {rmse:.2f} m, path ratio {ratio:.2f}")
    return quality


def shared_trajectory(source: Path, root: Path = SHARED_ROOT) -> Path | None:
    folder = root / source.name
    manifest_path = folder / "manifest.json"
    trajectory = folder / "trajectory.csv"
    if not manifest_path.is_file() or not trajectory.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("schema_version") != 1 or manifest.get("status") != "validated"
                or manifest.get("sequence") != source.name
                or manifest.get("source_fingerprint") != source_fingerprint(source)
                or manifest.get("fastlio_config_sha256") != file_sha256(CONFIG)
                or manifest.get("trajectory_sha256") != file_sha256(trajectory)
                or manifest.get("quality", {}).get("status") != "passed"):
            raise ValueError("Manifest, source, FAST-LIO config, or trajectory mismatch")
        validate_trajectory(source, trajectory)
        return trajectory
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"Shared Boreal reference rejected: {folder}: {error}", flush=True)
        return None


def publish_reference(source: Path, local_trajectory: Path, root: Path = SHARED_ROOT) -> Path:
    source = source.resolve()
    local_trajectory = local_trajectory.resolve()
    quality = validate_trajectory(source, local_trajectory)
    root.mkdir(parents=True, exist_ok=True)
    existing = shared_trajectory(source, root)
    if existing is not None:
        print(f"Shared Boreal reference: SKIP (verified) {existing}", flush=True)
        return existing
    target = root / source.name
    if target.exists():
        raise FileExistsError(f"Unverified shared reference exists; refusing overwrite: {target}")
    staging = root / f".staging-{source.name}-{uuid.uuid4().hex}"
    staging.mkdir()
    for name in ("trajectory.csv", "trajectory_raw.tum", "fast_lio.log", "player.log", "recorder.log"):
        path = local_trajectory.parent / name
        if path.is_file():
            shutil.copy2(path, staging / name)
    shutil.copy2(CONFIG, staging / "fastlio_config.yaml")
    original_manifest = local_trajectory.parent / "manifest.json"
    if original_manifest.is_file():
        shutil.copy2(original_manifest, staging / "generation_manifest.json")
    if (source / "gnss/gnss.csv").is_file():
        from Utilities.plot_boreal_fastlio import plot
        plot(staging / "trajectory.csv", source / "gnss/gnss.csv", staging)
    manifest = {"schema_version": 1, "status": "validated", "reference_quality": "experimental FAST-LIO; not surveyed ground truth",
                "sequence": source.name, "source_fingerprint": source_fingerprint(source),
                "fastlio_config_sha256": file_sha256(CONFIG),
                "trajectory_sha256": file_sha256(staging / "trajectory.csv"),
                "quality": quality, "published_at": datetime.now(timezone.utc).isoformat()}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    staging.rename(target)
    print(f"Shared Boreal reference: published {target}", flush=True)
    return target / "trajectory.csv"
