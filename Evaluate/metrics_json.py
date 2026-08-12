"""Generate compact machine-readable metrics for VSLAM-LAB runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from path_constants import TRAJECTORY_FILE_NAME
from utilities import read_trajectory_csv


def _read_trajectory(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if path.suffix == ".csv":
        return read_trajectory_csv(path)
    try:
        frame = pd.read_csv(path, header=None, sep=r"\s+")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    return frame if not frame.empty else None


def trajectory_length(trajectory: pd.DataFrame | Path | str) -> float | None:
    """Return the 3-D path length, or ``None`` for an unreadable trajectory."""
    if not isinstance(trajectory, pd.DataFrame):
        path = Path(trajectory)
        if not path.exists():
            return None
        trajectory = _read_trajectory(path)
    if trajectory is None or len(trajectory.index) < 2 or len(trajectory.columns) < 4:
        return None
    xyz = trajectory.iloc[:, 1:4].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)
    if len(xyz) < 2:
        return None
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def symmetric_coverage(predicted_length: float | None, groundtruth_length: float | None) -> float | None:
    """Score length agreement in [0, 1], penalising truncation and overshoot."""
    if predicted_length is None or groundtruth_length is None:
        return None
    if groundtruth_length <= 0:
        return 0.0
    return float(max(0.0, 1.0 - abs(predicted_length - groundtruth_length) / groundtruth_length))


def extract_ate_metrics(ate_csv: Path, exp_it: str) -> dict[str, float] | None:
    if not ate_csv.exists():
        return None
    try:
        frame = pd.read_csv(ate_csv)
    except (OSError, pd.errors.ParserError):
        return None
    if frame.empty:
        return None
    expected = f"{exp_it}_{TRAJECTORY_FILE_NAME}.txt"
    matches = frame[frame.get("traj_name", pd.Series(dtype=str)).astype(str) == expected]
    if matches.empty:
        matches = frame[frame.get("traj_name", pd.Series(dtype=str)).astype(str).str.contains(exp_it, na=False)]
    if matches.empty and len(frame.index) == 1:
        matches = frame
    if matches.empty:
        return None
    row = matches.iloc[0]
    try:
        return {name: float(row[name]) for name in ("mean", "std", "rmse", "max")}
    except (KeyError, TypeError, ValueError):
        return None


def aligned_rotation_rmse(evaluation_folder: Path, exp_it: str) -> float | None:
    """Compute absolute rotation RMSE from evo's timestamp-aligned trajectories."""
    predicted_path = evaluation_folder / f"{exp_it}_{TRAJECTORY_FILE_NAME}.tum"
    groundtruth_path = evaluation_folder / f"{exp_it}_gt.tum"
    predicted = _read_trajectory(predicted_path)
    groundtruth = _read_trajectory(groundtruth_path)
    if predicted is None or groundtruth is None:
        return None
    if len(predicted.columns) < 8 or len(groundtruth.columns) < 8:
        return None
    try:
        predicted = predicted.iloc[:, :8].apply(pd.to_numeric, errors="coerce").dropna()
        groundtruth = groundtruth.iloc[:, :8].apply(pd.to_numeric, errors="coerce").dropna()
        matched = predicted.merge(groundtruth, on=predicted.columns[0], suffixes=("_pred", "_gt"))
        if matched.empty:
            return None
        pred_rotation = Rotation.from_quat(matched.iloc[:, 4:8].to_numpy(dtype=float))
        gt_rotation = Rotation.from_quat(matched.iloc[:, 11:15].to_numpy(dtype=float))
        errors = (pred_rotation.inv() * gt_rotation).magnitude()
    except (TypeError, ValueError):
        return None
    return float(np.sqrt(np.mean(np.square(errors))))


def build_metrics(
    trajectory_path: Path,
    groundtruth_path: Path,
    evaluation_folder: Path,
    exp_it: str,
    status: str = "SUCCESS",
) -> dict[str, Any]:
    predicted_length = trajectory_length(evaluation_folder / f"{exp_it}_{TRAJECTORY_FILE_NAME}.tum")
    if predicted_length is None:
        predicted_length = trajectory_length(trajectory_path)
    groundtruth_length = trajectory_length(groundtruth_path)
    coverage = symmetric_coverage(predicted_length, groundtruth_length)
    ate = extract_ate_metrics(evaluation_folder / "ate.csv", exp_it)
    translation_rmse = ate["rmse"] if ate else None
    weighted_rmse = None
    if translation_rmse is not None and coverage is not None and coverage > 0:
        weighted_rmse = float(translation_rmse / coverage**2)

    trajectory_exists = trajectory_path.exists()
    return {
        "status": "SUCCESS" if trajectory_exists and ate else ("FAILURE" if not trajectory_exists else status),
        "rmse": {
            "translation": translation_rmse,
            "rotation": aligned_rotation_rmse(evaluation_folder, exp_it),
            "total": translation_rmse,
        },
        "ate": ate,
        "trajectory_length": predicted_length,
        "length_ratio": coverage,
        "gt_trajectory_length": groundtruth_length,
        "weighted_rmse": weighted_rmse,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def write_metrics_json(
    output_path: Path,
    trajectory_path: Path,
    groundtruth_path: Path,
    evaluation_folder: Path,
    exp_it: str,
    status: str = "SUCCESS",
) -> Path:
    """Write metrics atomically and return the resulting path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = build_metrics(trajectory_path, groundtruth_path, evaluation_folder, exp_it, status)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, allow_nan=False)
        file.write("\n")
    temporary.replace(output_path)
    return output_path


def write_experiment_metrics(exp: Any, dataset: Any, sequence_name: str, exp_it: str) -> Path:
    """Write the canonical metrics file for one experiment sequence/run."""
    run_folder = Path(exp.folder) / dataset.dataset_folder / sequence_name
    evaluation_folder = run_folder / "vslamlab_evaluation"
    trajectory_path = run_folder / f"{exp_it}_{TRAJECTORY_FILE_NAME}.csv"
    return write_metrics_json(
        evaluation_folder / "metrics.json",
        trajectory_path,
        run_folder / "groundtruth.csv",
        evaluation_folder,
        exp_it,
    )
