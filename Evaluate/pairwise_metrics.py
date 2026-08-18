"""Sensor-aware EVO comparisons for VSLAM, robot odometry, and FAST-LIO."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evo.core import lie_algebra, metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from scipy.spatial.transform import Rotation, Slerp


def read_pose_trajectory(path: Path) -> PoseTrajectory3D:
    frame = pd.read_csv(path)
    if frame.empty or len(frame.columns) < 8:
        raise ValueError(f"Trajectory must contain timestamp, xyz, and quaternion columns: {path}")
    values = frame.iloc[:, :8].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 3:
        raise ValueError(f"Trajectory has fewer than three usable poses: {path}")
    timestamps = values[:, 0]
    timestamp_label = str(frame.columns[0]).lower()
    if "ns" in timestamp_label or np.max(np.abs(timestamps)) > 1e12:
        timestamps = timestamps / 1e9
    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    values = values[order]
    unique = np.concatenate(([True], np.diff(timestamps) > 0))
    timestamps, values = timestamps[unique], values[unique]
    xyzw = values[:, 4:8]
    wxyz = xyzw[:, [3, 0, 1, 2]]
    return PoseTrajectory3D(positions_xyz=values[:, 1:4], orientations_quat_wxyz=wxyz, timestamps=timestamps)


def _statistics(metric: metrics.PE) -> dict[str, float]:
    return {
        statistic.value: float(metric.get_statistic(statistic))
        for statistic in metrics.StatisticsType
        if statistic is not metrics.StatisticsType.sse
    }


def interpolate_trajectory(reference: PoseTrajectory3D, target_timestamps: np.ndarray) -> PoseTrajectory3D:
    """Interpolate a lower-rate reference at in-range target timestamps."""
    target_timestamps = np.asarray(target_timestamps, dtype=float)
    target_timestamps = target_timestamps[
        (target_timestamps >= reference.timestamps[0]) & (target_timestamps <= reference.timestamps[-1])
    ]
    if target_timestamps.size < 3:
        raise ValueError("Fewer than three target timestamps overlap the reference trajectory")
    positions = np.column_stack(
        [np.interp(target_timestamps, reference.timestamps, reference.positions_xyz[:, axis]) for axis in range(3)]
    )
    rotations = Rotation.from_quat(reference.orientations_quat_wxyz, scalar_first=True)
    orientations = Slerp(reference.timestamps, rotations)(target_timestamps).as_quat(scalar_first=True)
    return PoseTrajectory3D(
        positions_xyz=positions,
        orientations_quat_wxyz=orientations,
        timestamps=target_timestamps,
    )


def evaluate_pair(
    reference_full: PoseTrajectory3D,
    estimate_full: PoseTrajectory3D,
    reference_name: str,
    estimate_name: str,
    max_time_difference_s: float,
    correct_scale: bool = False,
) -> tuple[dict[str, Any], PoseTrajectory3D]:
    reference, estimate = sync.associate_trajectories(
        deepcopy(reference_full),
        deepcopy(estimate_full),
        max_diff=max_time_difference_s,
        first_name=reference_name,
        snd_name=estimate_name,
    )
    if reference.num_poses < 3:
        raise ValueError(f"{estimate_name} vs {reference_name} has only {reference.num_poses} associated poses")
    rotation, translation, scale = estimate.align(reference, correct_scale=correct_scale)

    aligned_full = deepcopy(estimate_full)
    if correct_scale:
        aligned_full.scale(scale)
    aligned_full.transform(lie_algebra.se3(rotation, translation))

    translation_ape = metrics.APE(metrics.PoseRelation.translation_part)
    translation_ape.process_data((reference, estimate))
    rotation_ape = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
    rotation_ape.process_data((reference, estimate))

    rpe_values: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for key, relation, unit in (
        ("translation_rmse_1m", metrics.PoseRelation.translation_part, "m"),
        ("rotation_rmse_1m", metrics.PoseRelation.rotation_angle_deg, "deg"),
    ):
        try:
            evaluator = metrics.RPE(
                relation,
                delta=1.0,
                delta_unit=metrics.Unit.meters,
                rel_delta_tol=0.1,
                all_pairs=False,
                pairs_from_reference=True,
            )
            evaluator.process_data((reference, estimate))
            rpe_values[key] = {
                "rmse": float(evaluator.get_statistic(metrics.StatisticsType.rmse)),
                "pairs": int(len(evaluator.error)),
                "unit": unit,
            }
        except Exception as error:  # EVO raises several data-dependent exception types.
            rpe_values[key] = {"rmse": None, "pairs": 0, "unit": unit}
            errors[key] = str(error)

    reference_length = float(reference.path_length)
    estimate_length = float(estimate.path_length)
    ratio = estimate_length / reference_length if reference_length > 0 else None
    result = {
        "reference": reference_name,
        "estimate": estimate_name,
        "alignment": {
            "type": "Sim(3) Umeyama" if correct_scale else "SE(3) Umeyama (no scale correction)",
            "rotation": np.asarray(rotation).tolist(),
            "translation_m": np.asarray(translation).tolist(),
            "scale": float(scale),
        },
        "association": {
            "max_time_difference_s": max_time_difference_s,
            "matched_poses": int(reference.num_poses),
            "reference_poses": int(reference_full.num_poses),
            "estimate_poses": int(estimate_full.num_poses),
            "reference_coverage": float(reference.num_poses / reference_full.num_poses),
            "estimate_coverage": float(estimate.num_poses / estimate_full.num_poses),
            "first_timestamp_s": float(reference.timestamps[0]),
            "last_timestamp_s": float(reference.timestamps[-1]),
        },
        "ape": {
            "translation": {"unit": "m", **_statistics(translation_ape)},
            "rotation": {"unit": "deg", **_statistics(rotation_ape)},
        },
        "rpe": rpe_values,
        "path_lengths_m": {"reference": reference_length, "estimate": estimate_length},
        "trajectory_length_ratio": ratio,
        "metric_errors": errors,
    }
    return result, aligned_full


def _legacy_summary(primary: dict[str, Any]) -> dict[str, Any]:
    translation = primary["ape"]["translation"]
    ratio = primary["trajectory_length_ratio"]
    coverage = None if ratio is None else max(0.0, 1.0 - abs(1.0 - ratio))
    weighted = translation["rmse"] / coverage**2 if coverage and coverage > 0 else None
    return {
        "status": "SUCCESS",
        "rmse": {
            "translation": translation["rmse"],
            "rotation": primary["ape"]["rotation"]["rmse"],
            "total": translation["rmse"],
        },
        "ate": {key: translation[key] for key in ("mean", "std", "rmse", "max")},
        "trajectory_length": primary["path_lengths_m"]["estimate"],
        "length_ratio": coverage,
        "gt_trajectory_length": primary["path_lengths_m"]["reference"],
        "weighted_rmse": weighted,
    }


def write_pairwise_metrics(
    output_path: Path,
    vslam_path: Path,
    odometry_path: Path,
    fast_lio_path: Path | None,
    max_time_difference_s: float,
    sensor_type: str,
    warnings: list[str] | None = None,
) -> tuple[Path, dict[str, PoseTrajectory3D]]:
    odometry = read_pose_trajectory(odometry_path)
    vslam = read_pose_trajectory(vslam_path)
    primary, vslam_aligned = evaluate_pair(
        odometry,
        vslam,
        "robot_odometry",
        "vslam",
        max_time_difference_s,
        correct_scale=sensor_type.lower() == "mono",
    )
    comparisons: dict[str, Any] = {"vslam_vs_robot_odometry": primary}
    aligned = {"robot_odometry": odometry, "vslam": vslam_aligned}
    if fast_lio_path is not None:
        fast_lio = read_pose_trajectory(fast_lio_path)
        fast_vs_odom, fast_aligned = evaluate_pair(
            odometry, fast_lio, "robot_odometry", "fast_lio", max_time_difference_s
        )
        interpolated_fast_lio = interpolate_trajectory(fast_lio, vslam.timestamps)
        vslam_vs_fast, _ = evaluate_pair(
            interpolated_fast_lio, vslam, "fast_lio_interpolated", "vslam", 1e-6,
            correct_scale=sensor_type.lower() == "mono",
        )
        vslam_vs_fast["reference_interpolation"] = "linear position and quaternion Slerp at VSLAM timestamps"
        comparisons.update({
            "fast_lio_vs_robot_odometry": fast_vs_odom,
            "vslam_vs_fast_lio": vslam_vs_fast,
        })
        aligned["fast_lio"] = fast_aligned

    report = {
        "schema_version": 2,
        "primary_reference": "robot_odometry",
        **_legacy_summary(primary),
        "comparisons": comparisons,
        "warnings": warnings or [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return output_path, aligned


def write_combined_report(trajectories: dict[str, PoseTrajectory3D], output: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    colors = {"robot_odometry": "green", "vslam": "magenta", "fast_lio": "blue"}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, trajectory in trajectories.items():
        xyz = trajectory.positions_xyz
        axes[0].plot(xyz[:, 0], xyz[:, 1], label=name.replace("_", " "), color=colors.get(name))
        axes[1].plot(xyz[:, 0], xyz[:, 2], label=name.replace("_", " "), color=colors.get(name))
    for axis, x_label, y_label in ((axes[0], "x (m)", "y (m)"), (axes[1], "x (m)", "z (m)")):
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.axis("equal")
        axis.grid(True, alpha=0.3)
    axes[0].legend()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
