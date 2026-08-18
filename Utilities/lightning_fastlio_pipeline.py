#!/usr/bin/env python3
"""Resumable local ORB-SLAM2 + FAST-LIO pipeline for processed CLID sequences."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from mcap.reader import make_reader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Run.fastlio_reference import build_fingerprint, generate_fast_lio_reference, load_fast_lio_settings  # noqa: E402
from Run.single_sequence import evaluate_single_trajectory, run_single_baseline  # noqa: E402
from Utilities.extract_lightning_mcap import extract_sequence  # noqa: E402
from path_constants import VSLAM_LAB_DIR  # noqa: E402


DEFAULT_LOCAL_ROOT = Path("/mnt/share/local/eph/VSLAM")
NAS_ROOT = Path("/mnt/share/nas")
STAGES = ("stage", "extract", "config", "orbslam2", "fastlio", "metrics")
REQUIRED_TOPICS = (
    "/cam_sync/cam0/image_preview/compressed",
    "/cam_sync/cam1/image_preview/compressed",
    "/cam_sync/cam0/camera_info",
    "/cam_sync/cam1/camera_info",
    "/odometry",
    "/ouster/points",
    "/ouster/imu",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sampled_file_identity(path: Path, block_size: int = 1024 * 1024) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        digest.update(stream.read(block_size))
        if stat.st_size > block_size:
            stream.seek(max(0, stat.st_size - block_size))
            digest.update(stream.read(block_size))
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sampled_sha256": digest.hexdigest()}


def object_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def nearest_existing(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def parse_topic_counts(metadata_path: Path) -> dict[str, int]:
    data = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    information = data.get("rosbag2_bagfile_information", {})
    result: dict[str, int] = {}
    for item in information.get("topics_with_message_count", []):
        topic = item.get("topic_metadata", {}).get("name")
        if topic:
            result[str(topic)] = int(item.get("message_count", 0))
    return result


def _strip_capture_timestamp(name: str) -> str:
    return re.sub(r"_20\d{6}_\d{6}_\d+$", "", name)


class PipelinePaths:
    def __init__(
        self,
        bag_root: Path,
        local_root: Path,
        run_name: str | None = None,
        sequence_name: str | None = None,
    ) -> None:
        self.bag_root = bag_root.expanduser().resolve()
        self.local_root = local_root.expanduser().resolve()
        source_name = self.bag_root.name
        manifest_path = self.bag_root / "manifest.json"
        if manifest_path.is_file():
            try:
                source_name = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("name") or source_name)
            except (OSError, json.JSONDecodeError):
                pass
        self.sequence_name = sequence_name or _strip_capture_timestamp(source_name)
        sequence_token = re.search(r"(?:^|_)(seq\d+)(?:_|$)", source_name)
        default_run = (
            f"{self.bag_root.parent.name}_{sequence_token.group(1)}"
            if sequence_token
            else source_name
        )
        self.run_name = run_name or default_run
        self.run_root = self.local_root / "runs" / self.run_name
        self.source_research = self.bag_root / "research-bag"
        self.staged_research = self.run_root / "research-bag"
        self.dataset_root = self.run_root / "vslamlab"
        self.sequence = self.dataset_root / self.sequence_name
        self.output = self.dataset_root / "output/orbslam2_stereo_fastlio"
        self.config = self.run_root / f"orbslam2_fastlio_{sequence_token.group(1) if sequence_token else 'sequence'}.yaml"
        self.state = self.run_root / "pipeline.json"
        self.log = self.run_root / "pipeline.log"
        self.lock = self.run_root / ".pipeline.lock"

    def source_mcap(self) -> Path:
        files = sorted(self.source_research.glob("*.mcap"))
        if len(files) != 1:
            raise ValueError(f"Expected exactly one research-bag MCAP; found {len(files)} in {self.source_research}")
        return files[0]

    def staged_mcap(self) -> Path:
        return self.staged_research / self.source_mcap().name


class StateStore:
    def __init__(self, paths: PipelinePaths) -> None:
        self.paths = paths
        self._mutex = threading.RLock()
        try:
            self.data = json.loads(paths.state.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.data = {
                "schema_version": 1,
                "status": "pending",
                "bag_root": str(paths.bag_root),
                "run_root": str(paths.run_root),
                "sequence_name": paths.sequence_name,
                "stages": {},
            }

    def save(self) -> None:
        with self._mutex:
            self.data["updated_at"] = utc_now()
            atomic_json(self.paths.state, self.data)

    def update(self, **values: Any) -> None:
        with self._mutex:
            self.data.update(values)
            self.save()

    def stage(self, name: str, **values: Any) -> None:
        with self._mutex:
            current = self.data.setdefault("stages", {}).setdefault(name, {})
            current.update(values)
            self.save()


class Pipeline:
    def __init__(self, paths: PipelinePaths, workspace: Path, force_from: str | None = None) -> None:
        self.paths = paths
        self.workspace = workspace.expanduser().resolve()
        self.force_from = force_from
        self.state = StateStore(paths)
        self._last_console_progress: dict[str, float] = {}

    def log(self, message: str) -> None:
        line = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        self.paths.log.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.log.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def forced(self, stage: str) -> bool:
        return self.force_from is not None and STAGES.index(stage) >= STAGES.index(self.force_from)

    def begin_stage(self, stage: str, number: int, total: int) -> float:
        started = time.monotonic()
        self.state.update(status="running", current_stage=stage, pid=os.getpid(), pid_started_at=utc_now())
        self.state.stage(stage, status="running", started_at=utc_now(), progress={})
        self.log(f"[{number}/{total}] {stage}: starting")
        return started

    def finish_stage(self, stage: str, started: float, fingerprint: str, artifacts: dict[str, Any] | None = None) -> None:
        self.state.stage(
            stage,
            status="complete",
            finished_at=utc_now(),
            duration_s=time.monotonic() - started,
            fingerprint=fingerprint,
            artifacts=artifacts or {},
            progress={"percent": 100.0},
        )
        self.log(f"{stage}: complete")

    def skip_stage(self, stage: str, fingerprint: str) -> None:
        self.state.stage(stage, status="cached", checked_at=utc_now(), fingerprint=fingerprint, progress={"percent": 100.0})
        self.log(f"{stage}: SKIP (verified)")

    def progress(self, stage: str, event: dict[str, Any], interval_s: float = 5.0) -> None:
        self.state.stage(stage, progress=event, heartbeat_at=utc_now())
        now = time.monotonic()
        if now - self._last_console_progress.get(stage, 0.0) < interval_s and event.get("percent") != 100.0:
            return
        self._last_console_progress[stage] = now
        percent = event.get("percent")
        suffix = f" {percent:.1f}%" if isinstance(percent, (int, float)) else ""
        details = event.get("message", "")
        self.log(f"{stage}:{suffix}{' - ' + details if details else ''}")

    def source_checks(self) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        if not self.paths.bag_root.is_dir():
            errors.append(f"Bag root does not exist: {self.paths.bag_root}")
        for name in ("manifest.json", "processing.json", "research-bag/metadata.yaml"):
            if not (self.paths.bag_root / name).is_file():
                errors.append(f"Missing source file: {name}")
        source_mcap: Path | None = None
        if not errors:
            try:
                source_mcap = self.paths.source_mcap()
            except ValueError as error:
                errors.append(str(error))
        manifest: dict[str, Any] = {}
        processing: dict[str, Any] = {}
        for path, destination in (
            (self.paths.bag_root / "manifest.json", manifest),
            (self.paths.bag_root / "processing.json", processing),
        ):
            if path.is_file():
                try:
                    destination.update(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError) as error:
                    errors.append(f"Invalid JSON {path}: {error}")
        if manifest and manifest.get("status") != "complete":
            errors.append(f"Source manifest status is {manifest.get('status')!r}, not 'complete'")
        if processing and processing.get("status") != "complete":
            errors.append(f"Processing status is {processing.get('status')!r}, not 'complete'")
        topic_counts: dict[str, int] = {}
        metadata = self.paths.source_research / "metadata.yaml"
        if metadata.is_file():
            try:
                topic_counts = parse_topic_counts(metadata)
                missing_topics = [topic for topic in REQUIRED_TOPICS if topic_counts.get(topic, 0) <= 0]
                if missing_topics:
                    errors.append("Missing or empty required topic(s): " + ", ".join(missing_topics))
            except (OSError, yaml.YAMLError, TypeError, ValueError) as error:
                errors.append(f"Could not parse research-bag metadata: {error}")
        if source_mcap is not None:
            try:
                with source_mcap.open("rb") as stream:
                    summary = make_reader(stream).get_summary()
                if summary is None or not summary.channels:
                    errors.append(f"MCAP has no readable summary/channels: {source_mcap}")
                else:
                    summary_topics = {channel.topic for channel in summary.channels.values()}
                    missing_summary = [topic for topic in REQUIRED_TOPICS if topic not in summary_topics]
                    if missing_summary:
                        errors.append("MCAP summary lacks required topic(s): " + ", ".join(missing_summary))
            except Exception as error:
                errors.append(f"MCAP is not readable: {error}")

        local = self.paths.local_root
        if is_relative_to(local, NAS_ROOT) or is_relative_to(local, self.paths.bag_root):
            errors.append(f"Local output root is unsafe: {local}")
        ancestor = nearest_existing(local)
        if not os.access(ancestor, os.W_OK):
            errors.append(f"Local output ancestor is not writable: {ancestor}")

        tools = {
            "rsync": shutil.which("rsync"),
            "orbslam2": VSLAM_LAB_DIR / ".pixi/envs/orbslam2/bin/vslamlab_orbslam2_stereo",
            "ros_humble": Path("/opt/ros/humble/setup.zsh"),
            "fast_lio_setup": self.workspace / "install/setup.zsh",
            "fast_lio_library": self.workspace / "install/spark_fast_lio/lib/libspark_lio_component.so",
        }
        for name, path in tools.items():
            if not path or not Path(path).is_file():
                errors.append(f"Missing runtime prerequisite {name}: {path}")

        source_size = source_mcap.stat().st_size if source_mcap else 0
        staged_size = self.paths.staged_mcap().stat().st_size if source_mcap and self.paths.staged_mcap().exists() else 0
        extracted_exists = (self.paths.sequence / "extraction_metadata.json").is_file()
        estimated_needed = max(0, source_size - staged_size)
        if not extracted_exists:
            estimated_needed += int(source_size * 0.35) + 5 * 1024**3
        free = shutil.disk_usage(ancestor).free
        if free < estimated_needed:
            errors.append(f"Insufficient local space: need about {estimated_needed / 1024**3:.1f} GiB, have {free / 1024**3:.1f} GiB")
        elif free < estimated_needed * 1.2:
            warnings.append("Local free space is within 20% of the conservative estimate")

        return {
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
            "source_mcap": str(source_mcap) if source_mcap else None,
            "source_size": source_size,
            "topic_counts": topic_counts,
            "estimated_required_bytes": estimated_needed,
            "free_bytes": free,
        }

    def source_identity(self) -> dict[str, Any]:
        source = self.paths.source_mcap()
        return {
            "mcap": sampled_file_identity(source),
            "metadata_sha256": sha256_file(self.paths.source_research / "metadata.yaml"),
            "manifest_sha256": sha256_file(self.paths.bag_root / "manifest.json"),
            "processing_sha256": sha256_file(self.paths.bag_root / "processing.json"),
        }

    def stage_valid(self, source_identity: dict[str, Any]) -> bool:
        staged = self.paths.staged_mcap()
        if not staged.is_file() or not (self.paths.staged_research / "metadata.yaml").is_file():
            return False
        return (
            sampled_file_identity(staged) == source_identity["mcap"]
            and sha256_file(self.paths.staged_research / "metadata.yaml") == source_identity["metadata_sha256"]
        )

    def run_staging(self, source_identity: dict[str, Any], number: int, total: int) -> str:
        fingerprint = object_fingerprint(source_identity)
        if not self.forced("stage") and self.stage_valid(source_identity):
            self.skip_stage("stage", fingerprint)
            return fingerprint
        started = self.begin_stage("stage", number, total)
        self.paths.staged_research.mkdir(parents=True, exist_ok=True)
        command = [
            "rsync", "-a", "--partial", "--append-verify",
            f"{self.paths.source_research}/", f"{self.paths.staged_research}/",
        ]
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        total_bytes = source_identity["mcap"]["size"]
        initial_bytes = self.paths.staged_mcap().stat().st_size if self.paths.staged_mcap().exists() else 0
        copy_started = time.monotonic()
        try:
            while process.poll() is None:
                current = self.paths.staged_mcap().stat().st_size if self.paths.staged_mcap().exists() else 0
                percent = min(99.9, current / max(total_bytes, 1) * 100.0)
                elapsed = max(time.monotonic() - copy_started, 1e-9)
                rate = max(0.0, (current - initial_bytes) / elapsed)
                eta = (total_bytes - current) / rate if rate > 0 else None
                message = f"{current / 1024**3:.1f}/{total_bytes / 1024**3:.1f} GiB"
                if rate > 0:
                    message += f", {rate / 1024**2:.1f} MiB/s, ETA {eta:.0f}s"
                self.progress("stage", {"bytes": current, "total_bytes": total_bytes, "percent": percent, "rate_bps": rate, "eta_s": eta, "message": message})
                time.sleep(2)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        stderr = process.stderr.read() if process.stderr else ""
        if process.returncode != 0:
            raise RuntimeError(f"rsync failed with {process.returncode}: {stderr.strip()}")
        if not self.stage_valid(source_identity):
            raise RuntimeError("Staged research bag failed identity verification")
        self.finish_stage("stage", started, fingerprint, {"mcap": str(self.paths.staged_mcap())})
        return fingerprint

    def extraction_valid(self, stage_fingerprint: str) -> bool:
        metadata_path = self.paths.sequence / "extraction_metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        expected = metadata.get("pipeline_source_fingerprint") or metadata.get("source_fingerprint")
        if expected != stage_fingerprint or metadata.get("status") != "complete":
            return False
        try:
            if Path(metadata["source_bag"]).expanduser().resolve() != self.paths.staged_mcap():
                return False
        except (KeyError, TypeError, ValueError):
            return False
        for name in ("rgb.csv", "groundtruth.csv", "calibration.yaml"):
            path = self.paths.sequence / name
            if not path.is_file() or path.stat().st_size == 0:
                return False
        try:
            with (self.paths.sequence / "rgb.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != int(metadata["stereo_pairs"]):
                return False
            for row in rows:
                for key in ("path_rgb_0", "path_rgb_1"):
                    image = self.paths.sequence / row[key]
                    if not image.is_file() or image.stat().st_size == 0:
                        return False
            with (self.paths.sequence / "groundtruth.csv").open(encoding="utf-8") as stream:
                if sum(1 for _ in stream) - 1 != int(metadata["odometry_poses"]):
                    return False
        except (OSError, KeyError, TypeError, ValueError):
            return False
        return True

    def run_extraction(self, stage_fingerprint: str, topic_counts: dict[str, int], number: int, total: int) -> str:
        extractor_sha = sha256_file(Path(__file__).with_name("extract_lightning_mcap.py"))
        inputs = {"stage": stage_fingerprint, "extractor": extractor_sha, "topics": REQUIRED_TOPICS[:5]}
        fingerprint = object_fingerprint(inputs)
        if not self.forced("extract") and self.extraction_valid(stage_fingerprint):
            self.skip_stage("extract", fingerprint)
            return fingerprint
        started = self.begin_stage("extract", number, total)
        expected = min(topic_counts.get(REQUIRED_TOPICS[0], 0), topic_counts.get(REQUIRED_TOPICS[1], 0)) or None

        def report(event: dict[str, Any]) -> None:
            frames, frame_total = event["frames"], event.get("total")
            percent = frames / frame_total * 100.0 if frame_total else None
            eta = (frame_total - frames) / max(event["rate_fps"], 1e-9) if frame_total else None
            self.progress("extract", {**event, "percent": percent, "eta_s": eta, "message": f"{frames}/{frame_total or '?'} stereo frames, {event['rate_fps']:.1f} fps"})

        metadata = extract_sequence(
            self.paths.staged_mcap(),
            self.paths.sequence,
            expected_pairs=expected,
            overwrite_images=self.forced("extract"),
            progress=report,
            source_fingerprint=stage_fingerprint,
        )
        metadata["pipeline_source_fingerprint"] = stage_fingerprint
        atomic_json(self.paths.sequence / "extraction_metadata.json", metadata)
        if not self.extraction_valid(stage_fingerprint):
            raise RuntimeError("Extracted sequence failed integrity validation")
        self.finish_stage("extract", started, fingerprint, {"sequence": str(self.paths.sequence), "stereo_pairs": metadata["stereo_pairs"], "odometry_poses": metadata["odometry_poses"]})
        return fingerprint

    def config_content(self) -> str:
        data = {
            "DATASET": {
                "base_path": str(self.paths.dataset_root),
                "name": self.paths.sequence_name,
                "baseline": "orbslam2",
                "dataset": "lightning",
                "sensor_type": "stereo",
                "groundtruth_format": "kitti",
                "output_dir": "output/orbslam2_stereo_fastlio",
            },
            "EVALUATION": {
                "max_time_difference_s": 0.02,
                "fast_lio": {"enabled": True, "workspace": str(self.workspace)},
            },
        }
        return yaml.safe_dump(data, sort_keys=False)

    def run_config(self, extraction_fingerprint: str, number: int, total: int) -> str:
        content = self.config_content()
        fingerprint = object_fingerprint({"extraction": extraction_fingerprint, "content": content})
        if not self.forced("config") and self.paths.config.is_file() and self.paths.config.read_text(encoding="utf-8") == content:
            self.skip_stage("config", fingerprint)
            return fingerprint
        started = self.begin_stage("config", number, total)
        temporary = self.paths.config.with_suffix(".yaml.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(self.paths.config)
        self.finish_stage("config", started, fingerprint, {"config": str(self.paths.config)})
        return fingerprint

    def trajectory_valid(self) -> bool:
        trajectory = self.paths.output / "00000_KeyFrameTrajectory.csv"
        try:
            with trajectory.open(newline="", encoding="utf-8") as stream:
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

    def orb_fingerprint(self, config_fingerprint: str) -> str:
        inputs = {
            "config": config_fingerprint,
            "rgb": sampled_file_identity(self.paths.sequence / "rgb.csv"),
            "calibration": sha256_file(self.paths.sequence / "calibration.yaml"),
            "pixi_lock": sha256_file(VSLAM_LAB_DIR / "pixi.lock"),
            "single_sequence": sha256_file(VSLAM_LAB_DIR / "Run/single_sequence.py"),
        }
        return object_fingerprint(inputs)

    def run_orbslam2(self, config_fingerprint: str, number: int, total: int) -> tuple[str, Path]:
        fingerprint = self.orb_fingerprint(config_fingerprint)
        saved = self.state.data.get("stages", {}).get("orbslam2", {})
        trajectory = self.paths.output / "00000_KeyFrameTrajectory.csv"
        if not self.forced("orbslam2") and saved.get("fingerprint") == fingerprint and self.trajectory_valid():
            self.skip_stage("orbslam2", fingerprint)
            return fingerprint, trajectory
        started = self.begin_stage("orbslam2", number, total)
        try:
            extracted = json.loads((self.paths.sequence / "extraction_metadata.json").read_text(encoding="utf-8"))
            frame_count = int(extracted.get("stereo_pairs", 0))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            frame_count = 0
        stop = threading.Event()

        def heartbeat() -> None:
            while not stop.wait(10):
                elapsed = time.monotonic() - started
                self.progress("orbslam2", {"elapsed_s": elapsed, "frames": frame_count, "message": f"active for {elapsed:.0f}s on {frame_count or '?'} frames; external runner exposes no frame percentage"}, interval_s=30)

        monitor = threading.Thread(target=heartbeat, daemon=True)
        monitor.start()
        try:
            trajectory = run_single_baseline(self.paths.config, headless=True)
        finally:
            stop.set()
            monitor.join(timeout=2)
        if not self.trajectory_valid():
            raise RuntimeError("ORB-SLAM2 produced an invalid trajectory")
        self.finish_stage("orbslam2", started, fingerprint, {"trajectory": str(trajectory)})
        return fingerprint, trajectory

    def run_fastlio(self, number: int, total: int) -> tuple[str, Path]:
        settings = load_fast_lio_settings(self.paths.config)
        fast_inputs = build_fingerprint(settings)
        desired = object_fingerprint(fast_inputs)
        manifest_path = self.paths.sequence / "references/fast_lio/manifest.json"
        trajectory = self.paths.sequence / "references/fast_lio/trajectory.csv"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        valid = (
            manifest.get("status") == "complete"
            and manifest.get("fingerprint") == fast_inputs
            and trajectory.is_file()
            and sum(1 for _ in trajectory.open()) >= 4
        )
        saved = self.state.data.get("stages", {}).get("fastlio", {})
        if not self.forced("fastlio") and valid and saved.get("fingerprint") == desired:
            self.skip_stage("fastlio", desired)
            return desired, trajectory
        started = self.begin_stage("fastlio", number, total)

        def report(event: dict[str, Any]) -> None:
            message = f"{event.get('pose_count', 0)} poses"
            if isinstance(event.get("eta_s"), (int, float)):
                message += f", ETA {event['eta_s']:.0f}s"
            self.progress("fastlio", {**event, "message": message})

        trajectory = generate_fast_lio_reference(self.paths.config, force=self.forced("fastlio"), progress=report)
        self.finish_stage("fastlio", started, desired, {"trajectory": str(trajectory)})
        return desired, trajectory

    def metrics_valid(self) -> bool:
        metrics = self.paths.output / "metrics.json"
        report = self.paths.output / "trajectory_report.pdf"
        try:
            data = json.loads(metrics.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        required = {"vslam_vs_robot_odometry", "fast_lio_vs_robot_odometry", "vslam_vs_fast_lio"}
        return data.get("schema_version") == 2 and set(data.get("comparisons", {})) == required and report.is_file() and report.stat().st_size > 0

    def run_metrics(self, orb_fingerprint: str, fast_fingerprint: str, trajectory: Path, number: int, total: int) -> str:
        desired = object_fingerprint({
            "orb": orb_fingerprint,
            "fastlio": fast_fingerprint,
            "evaluator": sha256_file(VSLAM_LAB_DIR / "Evaluate/pairwise_metrics.py"),
        })
        saved = self.state.data.get("stages", {}).get("metrics", {})
        if not self.forced("metrics") and saved.get("fingerprint") == desired and self.metrics_valid():
            self.skip_stage("metrics", desired)
            return desired
        started = self.begin_stage("metrics", number, total)
        metrics = evaluate_single_trajectory(self.paths.config, trajectory)
        if not self.metrics_valid():
            raise RuntimeError("Final metrics failed integrity validation")
        self.finish_stage("metrics", started, desired, {"metrics": str(metrics), "report": str(self.paths.output / "trajectory_report.pdf")})
        return desired

    def run(self) -> Path:
        checks = self.source_checks()
        if not checks["ok"]:
            raise RuntimeError("Preflight failed:\n- " + "\n- ".join(checks["errors"]))
        self.paths.run_root.mkdir(parents=True, exist_ok=True)
        with self.paths.lock.open("w", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Another pipeline owns {self.paths.lock}") from error
            self.state.update(status="running", started_at=utc_now(), current_stage="preflight", pid=os.getpid(), checks=checks)
            self.log("[preflight] checks passed; NAS source will be opened read-only")
            try:
                source_identity = self.source_identity()
                stage_fp = self.run_staging(source_identity, 1, 6)
                extract_fp = self.run_extraction(stage_fp, checks["topic_counts"], 2, 6)
                config_fp = self.run_config(extract_fp, 3, 6)
                orb_fp, trajectory = self.run_orbslam2(config_fp, 4, 6)
                fast_fp, _ = self.run_fastlio(5, 6)
                self.run_metrics(orb_fp, fast_fp, trajectory, 6, 6)
                self.state.update(status="complete", current_stage=None, finished_at=utc_now(), pid=None)
                self.log(f"Pipeline complete: {self.paths.output}")
                return self.paths.output
            except KeyboardInterrupt:
                self.state.update(status="interrupted", current_stage=self.state.data.get("current_stage"), finished_at=utc_now(), pid=None)
                self.log("Pipeline interrupted; rerun the same command to resume")
                raise
            except Exception as error:
                stage = self.state.data.get("current_stage")
                if stage in STAGES:
                    self.state.stage(stage, status="failed", finished_at=utc_now(), error=str(error))
                self.state.update(status="failed", finished_at=utc_now(), error=str(error), pid=None)
                self.log(f"Pipeline failed in {stage}: {error}")
                raise


def print_checks(checks: dict[str, Any]) -> None:
    print("PASS" if checks["ok"] else "FAIL")
    for warning in checks["warnings"]:
        print(f"WARNING: {warning}")
    for error in checks["errors"]:
        print(f"ERROR: {error}")
    print(f"Source MCAP: {checks['source_mcap']}")
    print(f"Required local space: {checks['estimated_required_bytes'] / 1024**3:.1f} GiB")
    print(f"Available local space: {checks['free_bytes'] / 1024**3:.1f} GiB")
    for topic in REQUIRED_TOPICS:
        print(f"{topic}: {checks['topic_counts'].get(topic, 0)} messages")


def process_alive(pid: Any, bag_root: Path | None = None) -> bool:
    try:
        numeric_pid = int(pid)
        os.kill(numeric_pid, 0)
        if bag_root is not None:
            command = Path(f"/proc/{numeric_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            return "lightning_fastlio_pipeline.py" in command and str(bag_root) in command
        return True
    except (TypeError, ValueError, ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


def print_status(paths: PipelinePaths) -> int:
    try:
        state = json.loads(paths.state.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        fast_manifest = paths.sequence / "references/fast_lio/manifest.json"
        try:
            fast_complete = json.loads(fast_manifest.read_text(encoding="utf-8")).get("status") == "complete"
        except (OSError, json.JSONDecodeError):
            fast_complete = False
        existing = {
            "stage": paths.staged_research.is_dir() and any(paths.staged_research.glob("*.mcap")),
            "extract": (paths.sequence / "extraction_metadata.json").is_file(),
            "config": paths.config.is_file(),
            "orbslam2": (paths.output / "00000_KeyFrameTrajectory.csv").is_file(),
            "fastlio": fast_complete,
            "metrics": (paths.output / "metrics.json").is_file() and (paths.output / "trajectory_report.pdf").is_file(),
        }
        if not any(existing.values()):
            print(f"No pipeline state or existing artifacts found at {paths.run_root}")
            return 1
        status = "untracked-complete" if all(existing.values()) else "untracked-partial"
        state = {
            "status": status,
            "stages": {
                name: {"status": "existing-unverified" if present else "pending"}
                for name, present in existing.items()
            },
        }
        print("Note: these artifacts predate pipeline fingerprints; strict run mode will verify or regenerate them.")
    status = state.get("status", "unknown")
    if status == "running" and not process_alive(state.get("pid"), paths.bag_root):
        status = "stale"
    print(f"Status: {status}")
    print(f"Run root: {paths.run_root}")
    if state.get("current_stage"):
        print(f"Current stage: {state['current_stage']}")
    for name in STAGES:
        stage = state.get("stages", {}).get(name, {})
        label = stage.get("status", "pending")
        progress = stage.get("progress", {})
        percent = progress.get("percent")
        suffix = f" ({percent:.1f}%)" if isinstance(percent, (int, float)) else ""
        message = f" - {progress['message']}" if progress.get("message") else ""
        print(f"  {name:10s} {label}{suffix}{message}")
    if state.get("error"):
        print(f"Error: {state['error']}")
    metrics_path = paths.output / "metrics.json"
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            for name, result in metrics.get("comparisons", {}).items():
                rmse = result["ape"]["translation"]["rmse"]
                print(f"  {name}: translation APE RMSE {rmse:.3f} m")
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            print("WARNING: metrics.json exists but could not be summarized")
    print(f"Log: {paths.log}")
    return 2 if status in {"failed", "stale"} else 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="action", required=True)
    for action in ("run", "check", "status"):
        command = subparsers.add_parser(action)
        command.add_argument("bag_root", type=Path)
        command.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
        command.add_argument("--run-name")
        command.add_argument("--sequence-name")
        command.add_argument("--workspace", type=Path, default=Path("~/humble_ws"))
        if action == "run":
            command.add_argument("--force-from", choices=STAGES)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    paths = PipelinePaths(args.bag_root, args.local_root, args.run_name, args.sequence_name)
    pipeline = Pipeline(paths, args.workspace, getattr(args, "force_from", None))
    if args.action == "check":
        checks = pipeline.source_checks()
        print_checks(checks)
        return 0 if checks["ok"] else 2
    if args.action == "status":
        return print_status(paths)
    try:
        output = pipeline.run()
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
