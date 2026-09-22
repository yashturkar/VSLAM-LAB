#!/usr/bin/env python3
"""Run resumable VSLAM-LAB evaluations across explicitly configured CLID sequences."""

from __future__ import annotations

import argparse
import csv
import fcntl
import inspect
import json
import math
import os
import re
import shutil
import sys
import threading
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Baselines.get_baseline import get_baseline, list_available_baselines  # noqa: E402
from Evaluate.pairwise_metrics import read_pose_trajectory  # noqa: E402
from Run.fastlio_reference import generate_fast_lio_reference  # noqa: E402
from Run.single_sequence import evaluate_single_trajectory, run_single_baseline  # noqa: E402
from Utilities.lightning_fastlio_pipeline import (  # noqa: E402
    NAS_ROOT,
    Pipeline,
    PipelinePaths,
    atomic_json,
    is_relative_to,
    nearest_existing,
    object_fingerprint,
    sha256_file,
)
from path_constants import VSLAM_LAB_DIR  # noqa: E402


FORCE_STAGES = ("stage", "extract", "slam", "fastlio", "metrics", "report")
REPORT_COLUMNS = (
    "sequence",
    "source",
    "method",
    "status",
    "elapsed_s",
    "temporal_completion_ratio",
    "start_delay_s",
    "early_stop_s",
    "estimate_duration_s",
    "reference_duration_s",
    "largest_estimate_gap_s",
    "translation_ape_mean_m",
    "translation_ape_std_m",
    "translation_ape_rmse_m",
    "translation_ape_max_m",
    "rotation_ape_mean_deg",
    "rotation_ape_std_deg",
    "rotation_ape_rmse_deg",
    "rotation_ape_max_deg",
    "translation_rpe_rmse_1m_m",
    "rotation_rpe_rmse_1m_deg",
    "matched_poses",
    "reference_coverage",
    "estimate_coverage",
    "reference_path_length_m",
    "estimate_path_length_m",
    "trajectory_length_ratio",
    "symmetric_length_coverage",
    "weighted_rmse",
    "fast_lio_vs_odometry_translation_rmse_m",
    "vslam_vs_fast_lio_translation_rmse_m",
    "trajectory",
    "metrics_path",
    "report_path",
    "config_path",
    "log_path",
    "error",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _Tee:
    def __init__(self, terminal: Any, log: Any) -> None:
        self.terminal = terminal
        self.log = log

    def write(self, value: str) -> int:
        self.terminal.write(value)
        self.log.write(value)
        return len(value)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


@contextmanager
def tee_output(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        with (
            redirect_stdout(_Tee(sys.stdout, stream)),
            redirect_stderr(_Tee(sys.stderr, stream)),
        ):
            yield


def _resolved_path(value: Any, config_path: Path, field: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _optional_number(
    mapping: dict[str, Any],
    key: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"evaluation.thresholds.{key} must be a number or null")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"evaluation.thresholds.{key} must be a number or null"
        ) from error
    if (
        not math.isfinite(number)
        or number < minimum
        or (maximum is not None and number > maximum)
    ):
        limit = (
            f" between {minimum} and {maximum}"
            if maximum is not None
            else f" at least {minimum}"
        )
        raise ValueError(f"evaluation.thresholds.{key} must be{limit}")
    return number


@dataclass(frozen=True)
class BatchConfig:
    path: Path
    name: str
    local_root: Path
    sensor_type: str
    sequences: tuple[Path, ...]
    methods: tuple[str, ...]
    fast_lio_enabled: bool
    fast_lio_workspace: Path
    max_time_difference_s: float
    thresholds: dict[str, float | None]
    continue_on_error: bool

    @property
    def batch_root(self) -> Path:
        return self.local_root / "batches" / self.name

    @classmethod
    def load(cls, config_path: str | Path) -> "BatchConfig":
        path = Path(config_path).expanduser().resolve()
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"Batch config does not exist: {path}") from None
        if not isinstance(data, dict):
            raise ValueError("Batch config must be a YAML mapping")
        if data.get("version") != 1:
            raise ValueError("Batch config version must be 1")
        name = data.get("name")
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", name
        ):
            raise ValueError("name must contain only letters, numbers, '.', '_' or '-'")
        local_root = _resolved_path(data.get("local_root"), path, "local_root")
        sensor_type = str(data.get("sensor_type", "stereo")).lower()
        if sensor_type != "stereo":
            raise ValueError(
                "The CLID batch runner currently requires sensor_type: stereo"
            )

        raw_sequences = data.get("sequences")
        if not isinstance(raw_sequences, list) or not raw_sequences:
            raise ValueError(
                "sequences must be a non-empty list of explicit sequence roots"
            )
        sequences = tuple(
            _resolved_path(value, path, f"sequences[{index}]")
            for index, value in enumerate(raw_sequences)
        )
        if len(set(sequences)) != len(sequences):
            raise ValueError("sequences contains duplicate paths")

        raw_methods = data.get("methods")
        if not isinstance(raw_methods, list) or not raw_methods:
            raise ValueError("methods must be a non-empty list")
        methods = tuple(str(value).strip().lower() for value in raw_methods)
        if any(not value for value in methods) or len(set(methods)) != len(methods):
            raise ValueError("methods must contain unique non-empty baseline names")

        fast_lio = data.get("fast_lio", {})
        if not isinstance(fast_lio, dict) or not isinstance(
            fast_lio.get("enabled", False), bool
        ):
            raise ValueError("fast_lio must be a mapping with a boolean enabled field")
        fast_lio_enabled = fast_lio.get("enabled", False)
        workspace = _resolved_path(
            fast_lio.get("workspace", "~/humble_ws"), path, "fast_lio.workspace"
        )

        evaluation = data.get("evaluation", {})
        if not isinstance(evaluation, dict):
            raise ValueError("evaluation must be a mapping")
        max_difference = float(evaluation.get("max_time_difference_s", 0.02))
        if not math.isfinite(max_difference) or max_difference <= 0:
            raise ValueError("evaluation.max_time_difference_s must be positive")
        threshold_data = evaluation.get("thresholds", {})
        if not isinstance(threshold_data, dict):
            raise ValueError("evaluation.thresholds must be a mapping")
        supported_thresholds = {
            "min_temporal_completion_ratio",
            "max_vslam_vs_robot_odometry_translation_rmse_m",
            "max_vslam_vs_fast_lio_translation_rmse_m",
        }
        unknown_thresholds = set(threshold_data) - supported_thresholds
        if unknown_thresholds:
            raise ValueError(
                "Unknown evaluation threshold(s): "
                + ", ".join(sorted(unknown_thresholds))
            )
        thresholds = {
            "min_temporal_completion_ratio": _optional_number(
                threshold_data, "min_temporal_completion_ratio", maximum=1.0
            ),
            "max_vslam_vs_robot_odometry_translation_rmse_m": _optional_number(
                threshold_data, "max_vslam_vs_robot_odometry_translation_rmse_m"
            ),
            "max_vslam_vs_fast_lio_translation_rmse_m": _optional_number(
                threshold_data, "max_vslam_vs_fast_lio_translation_rmse_m"
            ),
        }
        if (
            not fast_lio_enabled
            and thresholds["max_vslam_vs_fast_lio_translation_rmse_m"] is not None
        ):
            raise ValueError(
                "The VSLAM-vs-FAST-LIO threshold requires fast_lio.enabled: true"
            )

        execution = data.get("execution", {})
        if not isinstance(execution, dict) or not isinstance(
            execution.get("continue_on_error", True), bool
        ):
            raise ValueError(
                "execution must be a mapping with a boolean continue_on_error field"
            )
        return cls(
            path=path,
            name=name,
            local_root=local_root,
            sensor_type=sensor_type,
            sequences=sequences,
            methods=methods,
            fast_lio_enabled=fast_lio_enabled,
            fast_lio_workspace=workspace,
            max_time_difference_s=max_difference,
            thresholds=thresholds,
            continue_on_error=execution.get("continue_on_error", True),
        )


def temporal_metrics(
    trajectory_path: Path, reference_path: Path
) -> dict[str, float | None]:
    estimate = read_pose_trajectory(trajectory_path)
    reference = read_pose_trajectory(reference_path)
    estimate_times = estimate.timestamps
    reference_times = reference.timestamps
    reference_duration = float(reference_times[-1] - reference_times[0])
    estimate_duration = float(estimate_times[-1] - estimate_times[0])
    overlap = max(
        0.0,
        min(float(estimate_times[-1]), float(reference_times[-1]))
        - max(float(estimate_times[0]), float(reference_times[0])),
    )
    completion = (
        min(1.0, overlap / reference_duration) if reference_duration > 0 else None
    )
    gaps = [float(b - a) for a, b in zip(estimate_times[:-1], estimate_times[1:])]
    return {
        "temporal_completion_ratio": completion,
        "start_delay_s": max(0.0, float(estimate_times[0] - reference_times[0])),
        "early_stop_s": max(0.0, float(reference_times[-1] - estimate_times[-1])),
        "estimate_duration_s": estimate_duration,
        "reference_duration_s": reference_duration,
        "largest_estimate_gap_s": max(gaps) if gaps else None,
    }


def trajectory_valid(path: Path) -> bool:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        if len(rows) < 4 or len(rows[0]) < 8:
            return False
        for row in rows[1:4]:
            if len(row) < 8:
                return False
            [float(value) for value in row[:8]]
        return True
    except (OSError, ValueError):
        return False


def metrics_valid(path: Path, fast_lio_enabled: bool) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {"vslam_vs_robot_odometry"}
    if fast_lio_enabled:
        expected.update({"fast_lio_vs_robot_odometry", "vslam_vs_fast_lio"})
    return (
        data.get("schema_version") == 2 and set(data.get("comparisons", {})) == expected
    )


def _stat(comparison: dict[str, Any], kind: str, axis: str, statistic: str) -> Any:
    return comparison.get(kind, {}).get(axis, {}).get(statistic)


def report_row(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics") or {}
    comparisons = metrics.get("comparisons", {})
    primary = comparisons.get("vslam_vs_robot_odometry", {})
    fast_odom = comparisons.get("fast_lio_vs_robot_odometry", {})
    vslam_fast = comparisons.get("vslam_vs_fast_lio", {})
    association = primary.get("association", {})
    lengths = primary.get("path_lengths_m", {})
    temporal = result.get("temporal", {})
    row = {
        "sequence": result.get("sequence"),
        "source": result.get("source"),
        "method": result.get("method"),
        "status": result.get("status"),
        "elapsed_s": result.get("elapsed_s"),
        **{
            key: temporal.get(key)
            for key in (
                "temporal_completion_ratio",
                "start_delay_s",
                "early_stop_s",
                "estimate_duration_s",
                "reference_duration_s",
                "largest_estimate_gap_s",
            )
        },
        "translation_ape_mean_m": _stat(primary, "ape", "translation", "mean"),
        "translation_ape_std_m": _stat(primary, "ape", "translation", "std"),
        "translation_ape_rmse_m": _stat(primary, "ape", "translation", "rmse"),
        "translation_ape_max_m": _stat(primary, "ape", "translation", "max"),
        "rotation_ape_mean_deg": _stat(primary, "ape", "rotation", "mean"),
        "rotation_ape_std_deg": _stat(primary, "ape", "rotation", "std"),
        "rotation_ape_rmse_deg": _stat(primary, "ape", "rotation", "rmse"),
        "rotation_ape_max_deg": _stat(primary, "ape", "rotation", "max"),
        "translation_rpe_rmse_1m_m": primary.get("rpe", {})
        .get("translation_rmse_1m", {})
        .get("rmse"),
        "rotation_rpe_rmse_1m_deg": primary.get("rpe", {})
        .get("rotation_rmse_1m", {})
        .get("rmse"),
        "matched_poses": association.get("matched_poses"),
        "reference_coverage": association.get("reference_coverage"),
        "estimate_coverage": association.get("estimate_coverage"),
        "reference_path_length_m": lengths.get("reference"),
        "estimate_path_length_m": lengths.get("estimate"),
        "trajectory_length_ratio": primary.get("trajectory_length_ratio"),
        "symmetric_length_coverage": metrics.get("length_ratio"),
        "weighted_rmse": metrics.get("weighted_rmse"),
        "fast_lio_vs_odometry_translation_rmse_m": _stat(
            fast_odom, "ape", "translation", "rmse"
        ),
        "vslam_vs_fast_lio_translation_rmse_m": _stat(
            vslam_fast, "ape", "translation", "rmse"
        ),
        "trajectory": result.get("trajectory"),
        "metrics_path": result.get("metrics_path"),
        "report_path": result.get("report_path"),
        "config_path": result.get("config_path"),
        "log_path": result.get("log_path"),
        "error": result.get("error"),
    }
    return {column: row.get(column) for column in REPORT_COLUMNS}


def apply_thresholds(
    row: dict[str, Any], thresholds: dict[str, float | None]
) -> list[str]:
    failures: list[str] = []
    checks = (
        (
            "min_temporal_completion_ratio",
            "temporal_completion_ratio",
            lambda value, limit: value < limit,
            ">=",
        ),
        (
            "max_vslam_vs_robot_odometry_translation_rmse_m",
            "translation_ape_rmse_m",
            lambda value, limit: value > limit,
            "<=",
        ),
        (
            "max_vslam_vs_fast_lio_translation_rmse_m",
            "vslam_vs_fast_lio_translation_rmse_m",
            lambda value, limit: value > limit,
            "<=",
        ),
    )
    for threshold_name, field, failed, relation in checks:
        limit = thresholds.get(threshold_name)
        if limit is None:
            continue
        value = row.get(field)
        if value is None:
            failures.append(f"{field} is unavailable (required {relation} {limit})")
        elif failed(float(value), limit):
            failures.append(
                f"{field}={float(value):.6g}, required {relation} {limit:.6g}"
            )
    return failures


class BatchRunner:
    def __init__(self, config: BatchConfig, force_from: str | None = None) -> None:
        self.config = config
        self.force_from = force_from
        self.state_path = config.batch_root / "batch.json"
        self.lock_path = config.batch_root / ".batch.lock"
        self.log_path = config.batch_root / "batch.log"
        self.report_json = config.batch_root / "report.json"
        self.report_csv = config.batch_root / "report.csv"
        self.report_markdown = config.batch_root / "report.md"
        try:
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.state = {
                "schema_version": 1,
                "name": config.name,
                "config": str(config.path),
                "status": "pending",
                "results": {},
            }

    def forced(self, stage: str) -> bool:
        return self.force_from is not None and FORCE_STAGES.index(
            stage
        ) >= FORCE_STAGES.index(self.force_from)

    def log(self, message: str) -> None:
        line = (
            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
        )
        print(line, flush=True)
        self.config.batch_root.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def save_state(self) -> None:
        self.state["updated_at"] = utc_now()
        atomic_json(self.state_path, self.state)

    def validate_methods(self) -> list[str]:
        errors: list[str] = []
        available = set(list_available_baselines())
        for method in self.config.methods:
            if method not in available:
                errors.append(f"Unknown baseline: {method}")
                continue
            baseline = get_baseline(method)
            if "stereo" not in getattr(baseline, "modes", []):
                errors.append(f"Baseline does not support stereo: {method}")
                continue
            installed, reason = baseline.is_installed()
            if not installed:
                errors.append(f"Baseline is unavailable: {method} ({reason})")
        return errors

    def preflight(self) -> dict[str, Any]:
        global_errors = self.validate_methods()
        if is_relative_to(self.config.local_root, NAS_ROOT):
            global_errors.append(
                f"Local output root must not be on NAS: {self.config.local_root}"
            )
        for source in self.config.sequences:
            if is_relative_to(self.config.local_root, source):
                global_errors.append(
                    f"Local output root is inside source tree: {source}"
                )
        ancestor = nearest_existing(self.config.local_root)
        if not os.access(ancestor, os.W_OK):
            global_errors.append(f"Local output ancestor is not writable: {ancestor}")

        sequence_checks: dict[str, Any] = {}
        total_required = 0
        for source in self.config.sequences:
            paths = PipelinePaths(source, self.config.local_root)
            pipeline = Pipeline(paths, self.config.fast_lio_workspace)
            checks = pipeline.source_checks(
                require_orbslam2=False,
                require_fastlio=self.config.fast_lio_enabled,
            )
            sequence_checks[str(source)] = checks
            total_required += int(checks.get("estimated_required_bytes", 0))
        free = shutil.disk_usage(ancestor).free
        if free < total_required:
            global_errors.append(
                f"Insufficient aggregate local space: need about {total_required / 1024**3:.1f} GiB, "
                f"have {free / 1024**3:.1f} GiB"
            )
        return {
            "ok": not global_errors
            and all(item["ok"] for item in sequence_checks.values()),
            "global_errors": global_errors,
            "sequence_checks": sequence_checks,
            "estimated_required_bytes": total_required,
            "free_bytes": free,
        }

    def config_path(self, paths: PipelinePaths, method: str) -> Path:
        return paths.run_root / f"{method}_{paths.sequence_name}.yaml"

    def output_path(self, paths: PipelinePaths, method: str) -> Path:
        return paths.dataset_root / "output" / f"{method}_{self.config.sensor_type}"

    def write_method_config(self, paths: PipelinePaths, method: str) -> Path:
        output = self.output_path(paths, method)
        content = {
            "DATASET": {
                "base_path": str(paths.dataset_root),
                "name": paths.sequence_name,
                "baseline": method,
                "dataset": "lightning",
                "sensor_type": self.config.sensor_type,
                "groundtruth_format": "kitti",
                "output_dir": str(output),
            },
            "EVALUATION": {
                "max_time_difference_s": self.config.max_time_difference_s,
                "fast_lio": {
                    "enabled": self.config.fast_lio_enabled,
                    "workspace": str(self.config.fast_lio_workspace),
                },
            },
        }
        destination = self.config_path(paths, method)
        text = yaml.safe_dump(content, sort_keys=False)
        if not destination.is_file() or destination.read_text(encoding="utf-8") != text:
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(destination)
        return destination

    def slam_fingerprint(
        self, paths: PipelinePaths, method: str, extraction_fingerprint: str
    ) -> str:
        baseline_module = inspect.getmodule(get_baseline(method).__class__)
        baseline_source = (
            Path(baseline_module.__file__).resolve()
            if baseline_module and baseline_module.__file__
            else None
        )
        return object_fingerprint(
            {
                "extraction": extraction_fingerprint,
                "method": method,
                "sensor_type": self.config.sensor_type,
                "rgb": sha256_file(paths.sequence / "rgb.csv"),
                "calibration": sha256_file(paths.sequence / "calibration.yaml"),
                "pixi_lock": sha256_file(VSLAM_LAB_DIR / "pixi.lock"),
                "runner": sha256_file(VSLAM_LAB_DIR / "Run/single_sequence.py"),
                "baseline_registry": sha256_file(
                    VSLAM_LAB_DIR / "Baselines/get_baseline.py"
                ),
                "baseline_source": sha256_file(baseline_source)
                if baseline_source
                else None,
            }
        )

    def metrics_fingerprint(self, paths: PipelinePaths, slam_fingerprint: str) -> str:
        fast_lio_manifest = paths.sequence / "references/fast_lio/manifest.json"
        return object_fingerprint(
            {
                "slam": slam_fingerprint,
                "fast_lio": self.config.fast_lio_enabled,
                "fast_lio_manifest": (
                    sha256_file(fast_lio_manifest)
                    if self.config.fast_lio_enabled and fast_lio_manifest.is_file()
                    else None
                ),
                "max_time_difference_s": self.config.max_time_difference_s,
                "evaluator": sha256_file(
                    VSLAM_LAB_DIR / "Evaluate/pairwise_metrics.py"
                ),
            }
        )

    def result_key(self, paths: PipelinePaths, method: str) -> str:
        return f"{paths.run_name}/{method}"

    def configured_result_keys(self) -> set[str]:
        return {
            self.result_key(PipelinePaths(source, self.config.local_root), method)
            for source in self.config.sequences
            for method in self.config.methods
        }

    def run_method(
        self,
        paths: PipelinePaths,
        method: str,
        extraction_fingerprint: str,
        fast_error: str | None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        output = self.output_path(paths, method)
        config_path = self.write_method_config(paths, method)
        trajectory = output / "00000_KeyFrameTrajectory.csv"
        metrics_path = output / "metrics.json"
        pdf_path = output / "trajectory_report.pdf"
        method_log = output / "run.log"
        key = self.result_key(paths, method)
        previous = self.state.get("results", {}).get(key, {})
        slam_fp = self.slam_fingerprint(paths, method, extraction_fingerprint)
        result: dict[str, Any] = {
            "sequence": paths.sequence_name,
            "source": str(paths.bag_root),
            "method": method,
            "status": "running",
            "config_path": str(config_path),
            "trajectory": str(trajectory),
            "metrics_path": str(metrics_path),
            "report_path": str(pdf_path),
            "log_path": str(method_log),
            "started_at": utc_now(),
            "slam_fingerprint": slam_fp,
        }
        self.state.setdefault("results", {})[key] = result
        self.save_state()
        try:
            if (
                not self.forced("slam")
                and previous.get("slam_fingerprint") == slam_fp
                and trajectory_valid(trajectory)
            ):
                self.log(f"{paths.sequence_name}/{method}: SLAM SKIP (verified)")
            else:
                self.log(f"{paths.sequence_name}/{method}: SLAM starting")
                stop_heartbeat = threading.Event()

                def heartbeat() -> None:
                    while not stop_heartbeat.wait(30):
                        self.log(
                            f"{paths.sequence_name}/{method}: SLAM active for "
                            f"{time.monotonic() - started:.0f}s; runner exposes no frame percentage"
                        )

                monitor = threading.Thread(target=heartbeat, daemon=True)
                monitor.start()
                try:
                    with tee_output(method_log):
                        trajectory = run_single_baseline(config_path, headless=True)
                finally:
                    stop_heartbeat.set()
                    monitor.join(timeout=2)
                if not trajectory_valid(trajectory):
                    raise RuntimeError(f"{method} produced an invalid trajectory")
                result["trajectory"] = str(trajectory)
                self.log(f"{paths.sequence_name}/{method}: SLAM complete")

            if fast_error is not None:
                raise RuntimeError(f"FAST-LIO reference failed: {fast_error}")
            metrics_fp = self.metrics_fingerprint(paths, slam_fp)
            result["metrics_fingerprint"] = metrics_fp
            if (
                not self.forced("metrics")
                and previous.get("metrics_fingerprint") == metrics_fp
                and metrics_valid(metrics_path, self.config.fast_lio_enabled)
                and pdf_path.is_file()
                and pdf_path.stat().st_size > 0
            ):
                self.log(f"{paths.sequence_name}/{method}: metrics SKIP (verified)")
            else:
                self.log(f"{paths.sequence_name}/{method}: evaluation starting")
                with tee_output(method_log):
                    evaluate_single_trajectory(config_path, trajectory)
                if not metrics_valid(metrics_path, self.config.fast_lio_enabled):
                    raise RuntimeError("Evaluation produced invalid metrics")
                self.log(f"{paths.sequence_name}/{method}: evaluation complete")

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            temporal = temporal_metrics(trajectory, paths.sequence / "groundtruth.csv")
            result.update(metrics=metrics, temporal=temporal)
            threshold_failures = apply_thresholds(
                report_row(result), self.config.thresholds
            )
            result["threshold_failures"] = threshold_failures
            result["status"] = "unhealthy" if threshold_failures else "complete"
        except Exception as error:
            result.update(status="failed", error=str(error))
            self.log(f"{paths.sequence_name}/{method}: FAILED - {error}")
        result.update(elapsed_s=time.monotonic() - started, finished_at=utc_now())
        self.state["results"][key] = result
        self.save_state()
        self.write_reports()
        return result

    def write_reports(self) -> None:
        self.config.batch_root.mkdir(parents=True, exist_ok=True)
        results = list(self.state.get("results", {}).values())
        rows = [report_row(result) for result in results]
        report = {
            "schema_version": 1,
            "name": self.config.name,
            "config": str(self.config.path),
            "status": self.state.get("status", "unknown"),
            "generated_at": utc_now(),
            "fast_lio_enabled": self.config.fast_lio_enabled,
            "thresholds": self.config.thresholds,
            "summary": {
                status: sum(result.get("status") == status for result in results)
                for status in ("complete", "unhealthy", "failed", "running")
            },
            "results": results,
        }
        atomic_json(self.report_json, report)

        temporary_csv = self.report_csv.with_suffix(".csv.tmp")
        with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=REPORT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        temporary_csv.replace(self.report_csv)

        lines = [
            f"# {self.config.name} SLAM report",
            "",
            f"Generated: {report['generated_at']}",
            f"FAST-LIO enabled: {self.config.fast_lio_enabled}",
            "",
            "| Sequence | Method | Status | Translation APE RMSE (m) | Temporal completion | Path coverage |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in rows:

            def display(value: Any, digits: int = 3) -> str:
                return (
                    "—"
                    if value is None or value == ""
                    else f"{float(value):.{digits}f}"
                )

            lines.append(
                f"| {row['sequence']} | {row['method']} | {row['status']} | "
                f"{display(row['translation_ape_rmse_m'])} | {display(row['temporal_completion_ratio'])} | "
                f"{display(row['symmetric_length_coverage'])} |"
            )
        failures = [
            result
            for result in results
            if result.get("status") in {"failed", "unhealthy"}
        ]
        if failures:
            lines.extend(["", "## Failures and threshold violations", ""])
            for result in failures:
                details = result.get("error") or "; ".join(
                    result.get("threshold_failures", [])
                )
                lines.append(
                    f"- `{result.get('sequence')}/{result.get('method')}`: {details}"
                )
        temporary_md = self.report_markdown.with_suffix(".md.tmp")
        temporary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary_md.replace(self.report_markdown)

    def record_preflight_failure(
        self, source: Path, method: str, errors: list[str]
    ) -> None:
        paths = PipelinePaths(source, self.config.local_root)
        key = self.result_key(paths, method)
        self.state.setdefault("results", {})[key] = {
            "sequence": paths.sequence_name,
            "source": str(source),
            "method": method,
            "status": "failed",
            "error": "Preflight failed: " + "; ".join(errors),
            "finished_at": utc_now(),
            "log_path": str(paths.log),
        }

    def run(
        self,
        selected_sequences: set[str] | None = None,
        selected_methods: set[str] | None = None,
    ) -> int:
        self.config.batch_root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("w", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    f"Another batch runner owns {self.lock_path}"
                ) from error
            return self._run(selected_sequences, selected_methods)

    def _run(
        self,
        selected_sequences: set[str] | None = None,
        selected_methods: set[str] | None = None,
    ) -> int:
        preflight = self.preflight()
        if preflight["global_errors"]:
            raise RuntimeError(
                "Batch preflight failed:\n- " + "\n- ".join(preflight["global_errors"])
            )
        if selected_sequences:
            selectable = {
                name
                for source in self.config.sequences
                for name in (
                    PipelinePaths(source, self.config.local_root).sequence_name,
                    PipelinePaths(source, self.config.local_root).run_name,
                )
            }
            unknown = selected_sequences - selectable
            if unknown:
                raise ValueError(
                    "Selected sequence(s) are not in the config: "
                    + ", ".join(sorted(unknown))
                )
        self.config.batch_root.mkdir(parents=True, exist_ok=True)
        self.state.update(
            status="running",
            started_at=utc_now(),
            pid=os.getpid(),
            pid_started_at=utc_now(),
            config_fingerprint=object_fingerprint(
                yaml.safe_load(self.config.path.read_text(encoding="utf-8"))
            ),
            preflight=preflight,
        )
        configured_keys = self.configured_result_keys()
        self.state["results"] = {
            key: value
            for key, value in self.state.get("results", {}).items()
            if key in configured_keys
        }
        self.save_state()
        any_failure = False
        stop = False
        for source in self.config.sequences:
            paths = PipelinePaths(source, self.config.local_root)
            if (
                selected_sequences
                and paths.sequence_name not in selected_sequences
                and paths.run_name not in selected_sequences
            ):
                continue
            methods = [
                method
                for method in self.config.methods
                if not selected_methods or method in selected_methods
            ]
            checks = preflight["sequence_checks"][str(source)]
            if not checks["ok"]:
                for method in methods:
                    self.record_preflight_failure(source, method, checks["errors"])
                any_failure = True
                self.save_state()
                self.write_reports()
                if not self.config.continue_on_error:
                    break
                continue
            paths.run_root.mkdir(parents=True, exist_ok=True)
            pipeline_force = (
                self.force_from if self.force_from in {"stage", "extract"} else None
            )
            pipeline = Pipeline(
                paths, self.config.fast_lio_workspace, force_from=pipeline_force
            )
            with paths.lock.open("w", encoding="utf-8") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError(f"Another pipeline owns {paths.lock}") from error
                try:
                    self.log(f"{paths.sequence_name}: staging/extraction starting")
                    source_identity = pipeline.source_identity()
                    stage_fp = pipeline.run_staging(source_identity, 1, 2)
                    extract_fp = pipeline.run_extraction(
                        stage_fp, checks["topic_counts"], 2, 2
                    )
                    pipeline.state.update(
                        status="prepared",
                        current_stage=None,
                        finished_at=utc_now(),
                        pid=None,
                    )
                    self.log(f"{paths.sequence_name}: staging/extraction ready")
                    configs = {
                        method: self.write_method_config(paths, method)
                        for method in methods
                    }
                    fast_error: str | None = None
                    if self.config.fast_lio_enabled and methods:
                        try:
                            self.log(
                                f"{paths.sequence_name}: FAST-LIO reference starting"
                            )
                            generate_fast_lio_reference(
                                configs[methods[0]],
                                force=self.forced("fastlio"),
                                progress=lambda event: self.log(
                                    f"{paths.sequence_name}: FAST-LIO {event.get('pose_count', 0)} poses"
                                ),
                            )
                            self.log(f"{paths.sequence_name}: FAST-LIO reference ready")
                        except Exception as error:
                            fast_error = str(error)
                            any_failure = True
                            self.log(
                                f"{paths.sequence_name}: FAST-LIO FAILED - {error}"
                            )
                    for method in methods:
                        result = self.run_method(paths, method, extract_fp, fast_error)
                        if result["status"] != "complete":
                            any_failure = True
                            if not self.config.continue_on_error:
                                stop = True
                                break
                except Exception as error:
                    any_failure = True
                    pipeline.state.update(
                        status="failed",
                        finished_at=utc_now(),
                        error=str(error),
                        pid=None,
                    )
                    for method in methods:
                        self.record_preflight_failure(source, method, [str(error)])
                    self.save_state()
                    self.write_reports()
                    self.log(f"{paths.sequence_name}: FAILED - {error}")
                    if not self.config.continue_on_error:
                        stop = True
            if stop:
                break
        result_statuses = {
            key: result.get("status")
            for key, result in self.state.get("results", {}).items()
            if key in configured_keys
        }
        any_failure = any(
            status in {"failed", "unhealthy"} for status in result_statuses.values()
        )
        pending = configured_keys - set(result_statuses)
        final_status = "failed" if any_failure else "partial" if pending else "complete"
        self.state.update(
            status=final_status,
            finished_at=utc_now(),
            pid=None,
        )
        self.save_state()
        self.write_reports()
        self.log(f"Batch {self.state['status']}: {self.report_markdown}")
        return 1 if any_failure else 0


def print_preflight(config: BatchConfig, preflight: dict[str, Any]) -> int:
    print("PASS" if preflight["ok"] else "FAIL")
    for error in preflight["global_errors"]:
        print(f"ERROR: {error}")
    for source, checks in preflight["sequence_checks"].items():
        print(f"{'PASS' if checks['ok'] else 'FAIL'} {source}")
        for warning in checks["warnings"]:
            print(f"  WARNING: {warning}")
        for error in checks["errors"]:
            print(f"  ERROR: {error}")
    print(
        f"Estimated local space: {preflight['estimated_required_bytes'] / 1024**3:.1f} GiB"
    )
    print(f"Available local space: {preflight['free_bytes'] / 1024**3:.1f} GiB")
    return 0 if preflight["ok"] else 2


def print_status(config: BatchConfig) -> int:
    state_path = config.batch_root / "batch.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print(f"No batch state found at {state_path}")
        return 1
    status = state.get("status", "unknown")
    if status == "running":
        try:
            os.kill(int(state.get("pid")), 0)
        except (TypeError, ValueError, ProcessLookupError, PermissionError, OSError):
            status = "stale"
    print(f"Status: {status}")
    print(f"Batch root: {config.batch_root}")
    for result in state.get("results", {}).values():
        suffix = f" - {result['error']}" if result.get("error") else ""
        print(
            f"  {result.get('sequence')}/{result.get('method')}: {result.get('status')}{suffix}"
        )
    print(f"Report: {config.batch_root / 'report.md'}")
    return 1 if status in {"failed", "stale"} else 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="action", required=True)
    for action in ("run", "check", "status"):
        command = subparsers.add_parser(action)
        command.add_argument("config", type=Path)
        if action == "run":
            command.add_argument("--force-from", choices=FORCE_STAGES)
            command.add_argument("--sequence", action="append", default=[])
            command.add_argument("--method", action="append", default=[])
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = BatchConfig.load(args.config)
        if args.action == "status":
            return print_status(config)
        runner = BatchRunner(config, getattr(args, "force_from", None))
        if args.action == "check":
            return print_preflight(config, runner.preflight())
        selected_methods = set(args.method) if args.method else None
        unknown_methods = (
            selected_methods - set(config.methods) if selected_methods else set()
        )
        if unknown_methods:
            raise ValueError(
                "Selected method(s) are not in the config: "
                + ", ".join(sorted(unknown_methods))
            )
        return runner.run(
            set(args.sequence) if args.sequence else None, selected_methods
        )
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
