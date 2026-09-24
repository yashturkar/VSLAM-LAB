#!/usr/bin/env python3
"""Run the installed SPARK FAST-LIO on BorealHDR lidar and IMU CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from Run.fastlio_reference import _shell_command, _stop, file_sha256  # noqa: E402

PLAYER = REPO / "Utilities/boreal_fastlio_player.py"
RECORDER = REPO / "Utilities/record_fastlio_path.py"
CONFIG = REPO / "configs/fastlio/boreal_vlp16.yaml"
WORKSPACE = Path("/home/yashturkar/humble_ws")
OUTPUT_ROOT = Path("/mnt/share/local/eph/VSLAM/borealhdr/fastlio_reference")


def output_csv(raw: Path, destination: Path) -> int:
    rows = []
    for line in raw.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 8:
            continue
        timestamp = fields[0].split(".")
        nanoseconds = int(timestamp[0]) * 1_000_000_000 + int(timestamp[1].ljust(9, "0")[:9])
        rows.append([nanoseconds, *fields[1:]])
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])
        writer.writerows(rows)
    return len(rows)


def run(source: Path, duration: float, rate: float, output_root: Path) -> Path:
    source = source.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not (source / "lidar").is_dir() or not (source / "imu/imu.csv").is_file():
        raise ValueError(f"Expected Boreal lidar/ and imu/imu.csv under {source}")
    if output_root == source or output_root.is_relative_to(source) or output_root.is_relative_to(Path("/mnt/share/nas")):
        raise ValueError("Output must be outside the source tree and NAS")
    required = [Path("/opt/ros/humble/setup.zsh"), WORKSPACE / "install/setup.zsh",
                WORKSPACE / "install/spark_fast_lio/lib/libspark_lio_component.so", CONFIG]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing FAST-LIO prerequisites: " + ", ".join(missing))
    name = "full" if duration == 0 else f"pilot_{duration:g}s"
    tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = output_root / source.name / f"{tag}_{name}"
    output.mkdir(parents=True, exist_ok=False)
    ready = output / ".recorder_ready"
    raw = output / "trajectory_raw.tum"
    environment = os.environ.copy()
    environment.update({"ROS_DOMAIN_ID": str(100 + os.getpid() % 100), "ROS_LOG_DIR": str(output / "ros_logs")})
    environment.pop("PYTHONPATH", None)
    Path(environment["ROS_LOG_DIR"]).mkdir()
    manifest = {
        "status": "running", "source": str(source), "output": str(output),
        "duration_s": duration, "rate": rate, "config": str(CONFIG),
        "config_sha256": file_sha256(CONFIG), "ros_domain_id": environment["ROS_DOMAIN_ID"],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    processes = []
    logs = []
    try:
        recorder_cmd = ["/usr/bin/python3", str(RECORDER), str(raw), "--ready-file", str(ready)]
        recorder_log = (output / "recorder.log").open("w", encoding="utf-8")
        logs.append(recorder_log)
        recorder = subprocess.Popen(_shell_command(WORKSPACE, recorder_cmd), stdout=recorder_log,
                                    stderr=subprocess.STDOUT, env=environment, start_new_session=True)
        processes.append(recorder)
        deadline = time.monotonic() + 20
        while not ready.exists() and time.monotonic() < deadline:
            if recorder.poll() is not None:
                raise RuntimeError("Path recorder exited during startup")
            time.sleep(0.1)
        if not ready.exists():
            raise RuntimeError("Path recorder did not become ready")

        mapper_cmd = ["ros2", "run", "spark_fast_lio", "spark_lio_mapping", "--ros-args",
                      "--params-file", str(CONFIG), "-r", "lidar:=/boreal/lidar", "-r", "imu:=/boreal/imu",
                      "-r", "path:=/fast_lio/path", "-r", "odometry:=/fast_lio/odometry"]
        mapper_log = (output / "fast_lio.log").open("w", encoding="utf-8")
        logs.append(mapper_log)
        mapper = subprocess.Popen(_shell_command(WORKSPACE, mapper_cmd), stdout=mapper_log,
                                  stderr=subprocess.STDOUT, env=environment, start_new_session=True)
        processes.append(mapper)
        time.sleep(3)
        if mapper.poll() is not None:
            raise RuntimeError("FAST-LIO exited during startup")

        player_cmd = ["/usr/bin/python3", str(PLAYER), str(source), "--duration", str(duration), "--rate", str(rate)]
        player_log = (output / "player.log").open("w", encoding="utf-8")
        logs.append(player_log)
        player = subprocess.Popen(_shell_command(WORKSPACE, player_cmd), stdout=player_log,
                                  stderr=subprocess.STDOUT, env=environment, start_new_session=True)
        processes.append(player)
        started = time.monotonic()
        timeout = max(120, (duration or 3600) / rate * 4 + 60)
        while player.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed > timeout:
                raise TimeoutError(f"Boreal playback exceeded {timeout:.0f}s")
            if mapper.poll() is not None or recorder.poll() is not None:
                raise RuntimeError("FAST-LIO or path recorder exited during playback")
            pose_count = len(raw.read_text(encoding="utf-8").splitlines()) if raw.exists() else 0
            print(f"Playback {elapsed:.0f}s; {pose_count} reference poses; logs: {output}", flush=True)
            time.sleep(10)
        if player.returncode:
            raise RuntimeError(f"Boreal player exited with code {player.returncode}")
        time.sleep(5)
    except Exception as error:
        manifest.update({"status": "failed", "error": str(error)})
        raise
    finally:
        for process in reversed(processes):
            _stop(process)
        for log in logs:
            log.close()
        ready.unlink(missing_ok=True)
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    trajectory = output / "trajectory.csv"
    count = output_csv(raw, trajectory)
    if count < 3:
        manifest.update({"status": "failed", "error": f"Only {count} poses recorded"})
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"FAST-LIO produced only {count} poses; inspect {output}")
    manifest.update({"status": "complete", "pose_count": count, "trajectory": str(trajectory)})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"FAST-LIO reference: {trajectory} ({count} poses)", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--duration", type=float, default=30, help="Sensor seconds; 0 runs the full sequence")
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.duration < 0 or args.rate <= 0:
        parser.error("duration must be nonnegative and rate must be positive")
    run(args.source, args.duration, args.rate, args.output_root)


if __name__ == "__main__":
    main()
