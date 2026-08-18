"""Generate and cache a FAST-LIO trajectory from a sequence's source MCAP."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shlex
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from path_constants import VSLAM_LAB_DIR

FAST_LIO_REVISION = "17b36d293a14df37d57e1751a337a32e2f164692"
DEFAULT_WORKSPACE = Path("~/humble_ws")
DEFAULT_CONFIG = VSLAM_LAB_DIR / "configs/fastlio/ouster_os0_128.yaml"
RVIZ_CONFIG = VSLAM_LAB_DIR / "configs/fastlio/fastlio.rviz"
RECORDER = VSLAM_LAB_DIR / "Utilities/record_fastlio_path.py"
PLAYER = VSLAM_LAB_DIR / "Utilities/play_fastlio_bag.py"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_fast_lio_settings(config_yaml: str | Path) -> dict[str, Any]:
    config_path = Path(config_yaml).expanduser().resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    dataset = data.get("DATASET")
    if not isinstance(dataset, dict):
        raise ValueError("Single-sequence config must contain a DATASET mapping")
    base_path = Path(dataset["base_path"]).expanduser()
    if not base_path.is_absolute():
        base_path = (config_path.parent / base_path).resolve()
    sequence = base_path / str(dataset["name"])
    evaluation = data.get("EVALUATION", {})
    fast_lio = evaluation.get("fast_lio", {}) if isinstance(evaluation, dict) else {}
    if fast_lio is True:
        fast_lio = {"enabled": True}
    if not isinstance(fast_lio, dict):
        raise ValueError("EVALUATION.fast_lio must be a mapping or true")

    workspace = Path(fast_lio.get("workspace", DEFAULT_WORKSPACE)).expanduser().resolve()
    fast_lio_config = Path(fast_lio.get("config", DEFAULT_CONFIG)).expanduser()
    if not fast_lio_config.is_absolute():
        fast_lio_config = (config_path.parent / fast_lio_config).resolve()
    metadata_path = sequence / "extraction_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"FAST-LIO requires source-bag provenance: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_value = fast_lio.get("source_bag") or metadata.get("source_bag")
    if not source_value:
        raise ValueError(f"Missing source_bag in {metadata_path}")
    source_bag = Path(source_value).expanduser().resolve()
    return {
        "config_path": config_path,
        "sequence": sequence,
        "metadata_path": metadata_path,
        "source_bag": source_bag,
        "workspace": workspace,
        "fast_lio_config": fast_lio_config,
        "enabled": bool(fast_lio.get("enabled", False)),
        "max_time_difference_s": float(evaluation.get("max_time_difference_s", 0.02)),
    }


def extract_ouster_clock_samples(source_bag: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sensor_times: list[float] = []
    record_times: list[float] = []
    description: dict[str, Any] | None = None
    with source_bag.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, channel, record, message in reader.iter_decoded_messages(topics=["/ouster/points"]):
            field = next((item for item in message.fields if item.name == "t"), None)
            if field is None or int(field.datatype) != 6:
                raise ValueError("Ouster PointCloud2 must contain a uint32 't' field")
            if description is None:
                description = {
                    "topic": channel.topic,
                    "frame": message.header.frame_id,
                    "width": int(message.width),
                    "height": int(message.height),
                    "point_step": int(message.point_step),
                    "fields": [item.name for item in message.fields],
                }
            point_count = int(message.width) * int(message.height)
            byte_order = ">u4" if message.is_bigendian else "<u4"
            offsets = np.ndarray(
                shape=(point_count,),
                dtype=byte_order,
                buffer=message.data,
                offset=int(field.offset),
                strides=(int(message.point_step),),
            )
            duration_ns = int(offsets.max())
            if not 1_000_000 <= duration_ns <= 120_000_000:
                continue
            header_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
            sensor_times.append((header_ns + duration_ns) / 1e9)
            record_times.append(record.log_time / 1e9)
    if len(sensor_times) < 3 or description is None:
        raise ValueError("At least three valid Ouster scans are required")
    return np.asarray(sensor_times), np.asarray(record_times), description


def fit_clock_mapping(sensor_times: np.ndarray, record_times: np.ndarray) -> dict[str, Any]:
    if sensor_times.shape != record_times.shape or sensor_times.size < 3:
        raise ValueError("Clock fit requires matching arrays with at least three samples")
    sensor_origin, record_origin = float(sensor_times[0]), float(record_times[0])
    x, y = sensor_times - sensor_origin, record_times - record_origin
    keep = np.ones(sensor_times.size, dtype=bool)
    for _ in range(5):
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        residual = y - (slope * x + intercept)
        median = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - median)))
        updated = np.abs(residual - median) <= max(0.005, 6.0 * 1.4826 * mad)
        if updated.sum() < 3 or np.array_equal(updated, keep):
            break
        keep = updated
    slope, intercept = np.polyfit(x[keep], y[keep], 1)
    residual_ms = np.abs(y - (slope * x + intercept)) * 1000.0
    return {
        "sensor_origin_s": sensor_origin,
        "record_origin_s": record_origin,
        "slope": float(slope),
        "intercept_s": float(intercept),
        "samples": int(sensor_times.size),
        "inliers": int(keep.sum()),
        "outliers": int((~keep).sum()),
        "residual_ms": {
            "median_inlier": float(np.median(residual_ms[keep])),
            "p95_inlier": float(np.percentile(residual_ms[keep], 95)),
            "maximum_inlier": float(np.max(residual_ms[keep])),
        },
    }


def apply_clock_mapping(raw_path: Path, corrected_path: Path, csv_path: Path, mapping: dict[str, Any]) -> int:
    rows: list[list[Any]] = []
    temporary = corrected_path.with_suffix(corrected_path.suffix + ".tmp")
    with raw_path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as destination:
        for line in source:
            values = line.split()
            if len(values) != 8:
                continue
            timestamp = float(values[0])
            corrected = mapping["record_origin_s"] + (
                mapping["slope"] * (timestamp - mapping["sensor_origin_s"]) + mapping["intercept_s"]
            )
            destination.write(f"{corrected:.9f} {' '.join(values[1:])}\n")
            rows.append([round(corrected * 1e9), *map(float, values[1:])])
    temporary.replace(corrected_path)
    csv_tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with csv_tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])
        writer.writerows(rows)
    csv_tmp.replace(csv_path)
    return len(rows)


def _git_revision(path: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def build_fingerprint(settings: dict[str, Any]) -> dict[str, Any]:
    source_bag = settings["source_bag"]
    workspace = settings["workspace"]
    library = workspace / "install/spark_fast_lio/lib/libspark_lio_component.so"
    required = [
        Path("/opt/ros/humble/setup.zsh"),
        workspace / "install/setup.zsh",
        library,
        settings["fast_lio_config"],
        settings["metadata_path"],
        source_bag,
        RECORDER,
        PLAYER,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("FAST-LIO setup is incomplete: " + ", ".join(map(str, missing)))
    return {
        "source_bag": {"path": str(source_bag), "size": source_bag.stat().st_size, "mtime_ns": source_bag.stat().st_mtime_ns},
        "extraction_metadata_sha256": file_sha256(settings["metadata_path"]),
        "config_sha256": file_sha256(settings["fast_lio_config"]),
        "recorder_sha256": file_sha256(RECORDER),
        "player_sha256": file_sha256(PLAYER),
        "fast_lio_revision": _git_revision(workspace / "src/spark-fast-lio"),
        "expected_fast_lio_revision": FAST_LIO_REVISION,
        "component": {"size": library.stat().st_size, "mtime_ns": library.stat().st_mtime_ns},
    }


def _shell_command(workspace: Path, command: list[str]) -> list[str]:
    setup = " && ".join(
        f"source {shlex.quote(str(path))}"
        for path in (Path("/opt/ros/humble/setup.zsh"), workspace / "install/setup.zsh")
    )
    return ["/usr/bin/zsh", "-c", f"{setup} && exec {shlex.join(command)}"]


def _stop(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def _cached(manifest_path: Path, trajectory: Path, fingerprint: dict[str, Any]) -> bool:
    if not trajectory.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("status") == "complete" and manifest.get("fingerprint") == fingerprint and sum(1 for _ in trajectory.open()) >= 4


def generate_fast_lio_reference(
    config_yaml: str | Path,
    force: bool = False,
    rviz: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    settings = load_fast_lio_settings(config_yaml)
    output = settings["sequence"] / "references/fast_lio"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, trajectory = output / "manifest.json", output / "trajectory.csv"
    fingerprint = build_fingerprint(settings)
    if not force and not rviz and _cached(manifest_path, trajectory, fingerprint):
        print(f"Reusing FAST-LIO reference: {trajectory}")
        if progress is not None:
            progress({"cached": True, "percent": 100.0, "pose_count": sum(1 for _ in trajectory.open()) - 1})
        return trajectory

    sensor_times, record_times, pointcloud = extract_ouster_clock_samples(settings["source_bag"])
    clock_mapping = fit_clock_mapping(sensor_times, record_times)
    raw_path, corrected_path = output / "trajectory_raw.tum", output / "trajectory_corrected.tum"
    ready_file = output / ".recorder_ready"
    ready_file.unlink(missing_ok=True)
    domain_id = 100 + os.getpid() % 100
    environment = os.environ.copy()
    environment.update({"ROS_DOMAIN_ID": str(domain_id), "ROS_LOG_DIR": str(output / "ros_logs")})
    environment.pop("PYTHONPATH", None)
    Path(environment["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "source_bag": str(settings["source_bag"]),
        "pointcloud": pointcloud,
        "clock_mapping": clock_mapping,
        "ros_domain_id": domain_id,
    }
    write_json(manifest_path, report)
    recorder = mapper = player = viewer = None
    logs = {name: output / f"{name}.log" for name in ("recorder", "fast_lio", "rosbag", "rviz")}
    try:
        with logs["recorder"].open("w", encoding="utf-8") as log:
            recorder = subprocess.Popen(
                _shell_command(settings["workspace"], ["/usr/bin/python3", str(RECORDER), str(raw_path), "--ready-file", str(ready_file)]),
                stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True,
            )
        deadline = time.monotonic() + 15
        while not ready_file.exists() and time.monotonic() < deadline:
            if recorder.poll() is not None:
                raise RuntimeError(f"FAST-LIO recorder exited; inspect {logs['recorder']}")
            time.sleep(0.1)
        if not ready_file.exists():
            raise RuntimeError("Timed out waiting for the FAST-LIO recorder")

        mapper_command = [
            "ros2", "run", "spark_fast_lio", "spark_lio_mapping", "--ros-args",
            "--params-file", str(settings["fast_lio_config"]),
            "-r", "lidar:=/ouster/points", "-r", "imu:=/ouster/imu",
            "-r", "path:=/fast_lio/path", "-r", "odometry:=/fast_lio/odometry",
            "-r", "cloud_registered:=/fast_lio/cloud_registered",
        ]
        if rviz:
            mapper_command.extend(["-p", "publish.scan_publish_en:=true"])
        with logs["fast_lio"].open("w", encoding="utf-8") as log:
            mapper = subprocess.Popen(_shell_command(settings["workspace"], mapper_command), stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
        time.sleep(3)
        if mapper.poll() is not None:
            raise RuntimeError(f"FAST-LIO exited during startup; inspect {logs['fast_lio']}")
        if rviz:
            with logs["rviz"].open("w", encoding="utf-8") as log:
                viewer = subprocess.Popen(
                    _shell_command(settings["workspace"], ["rviz2", "-d", str(RVIZ_CONFIG)]),
                    stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True,
                )

        bag_dir = settings["source_bag"].parent
        player_command = ["/usr/bin/python3", str(PLAYER), str(bag_dir), "--rate", "1.0"]
        with logs["rosbag"].open("w", encoding="utf-8") as log:
            player = subprocess.Popen(_shell_command(settings["workspace"], player_command), stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
            playback_duration = max(float(record_times[-1] - record_times[0]), 0.001)
            timeout_s = max(180, int(playback_duration * 2 + 60))
            playback_started = time.monotonic()
            while player.poll() is None:
                elapsed = time.monotonic() - playback_started
                if elapsed > timeout_s:
                    raise subprocess.TimeoutExpired(player.args, timeout_s)
                pose_count = max(0, sum(1 for _ in raw_path.open()) if raw_path.exists() else 0)
                event = {
                    "cached": False,
                    "elapsed_s": elapsed,
                    "duration_s": playback_duration,
                    "percent": min(99.0, elapsed / playback_duration * 100.0),
                    "eta_s": max(0.0, playback_duration - elapsed),
                    "pose_count": pose_count,
                }
                if progress is not None:
                    progress(event)
                elif int(elapsed) % 15 == 0:
                    print(
                        f"FAST-LIO playback {event['percent']:.1f}% "
                        f"({pose_count} poses, ETA {event['eta_s']:.0f}s)",
                        flush=True,
                    )
                time.sleep(1.0)
            exit_code = player.returncode
        if exit_code != 0:
            raise RuntimeError(f"rosbag playback exited with {exit_code}; inspect {logs['rosbag']}")
        time.sleep(5)
        if mapper.poll() is not None:
            raise RuntimeError(f"FAST-LIO exited unexpectedly; inspect {logs['fast_lio']}")
    except Exception as error:
        report.update({"status": "failed", "finished_at": datetime.now(timezone.utc).isoformat(), "error": str(error)})
        write_json(manifest_path, report)
        raise
    finally:
        for process in (player, mapper, recorder, viewer):
            _stop(process)
        ready_file.unlink(missing_ok=True)

    pose_count = apply_clock_mapping(raw_path, corrected_path, trajectory, clock_mapping)
    if pose_count < 3:
        raise RuntimeError(f"FAST-LIO produced only {pose_count} path poses; inspect {logs['fast_lio']}")
    report.update({
        "status": "complete", "finished_at": datetime.now(timezone.utc).isoformat(), "pose_count": pose_count,
        "trajectory": str(trajectory), "raw_trajectory": str(raw_path), "corrected_trajectory": str(corrected_path),
        "logs": {name: str(path) for name, path in logs.items()},
    })
    write_json(manifest_path, report)
    if progress is not None:
        progress({"cached": False, "percent": 100.0, "pose_count": pose_count, "complete": True})
    print(f"Generated FAST-LIO reference with {pose_count} poses: {trajectory}")
    return trajectory


def fast_lio_reference_enabled(config_yaml: str | Path) -> bool:
    config_path = Path(config_yaml).expanduser().resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    evaluation = data.get("EVALUATION", {})
    if not isinstance(evaluation, dict):
        raise ValueError("EVALUATION must be a mapping")
    fast_lio = evaluation.get("fast_lio", {})
    if fast_lio is True:
        return True
    if fast_lio in (False, None):
        return False
    if not isinstance(fast_lio, dict):
        raise ValueError("EVALUATION.fast_lio must be a mapping or boolean")
    return bool(fast_lio.get("enabled", False))
