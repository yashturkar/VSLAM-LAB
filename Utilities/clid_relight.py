"""Prepare CLID component relighting variants and run monocular VSLAM-LAB."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
import threading
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from Utilities.extract_lightning_mcap import camera_values, read_camera_info, write_csv  # noqa: E402

DEFAULT_OUTPUT = Path("/mnt/share/local/eph/VSLAM/relighting")
DEFAULT_RELIGHTING_CONFIG = REPO / "configs/clid_v63_relighting_v1.yaml"
COMPONENT_KEYS = ("ambient", "flash", "correction1", "correction2")
CAMERA_INFO_TOPIC = "/cam_sync/cam0/camera_info"
RENDERER_VERSION = 1


class UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader which refuses duplicate keys at every mapping level."""


def _unique_mapping(loader: UniqueKeyLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


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
def tee_output(path: Path) -> Iterator[None]:
    with path.open("a", encoding="utf-8") as stream:
        with redirect_stdout(_Tee(sys.stdout, stream)), redirect_stderr(_Tee(sys.stderr, stream)):
            yield


def load_schedule(path: Path, count: int) -> list[dict[str, float]]:
    try:
        schedule = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid schedule YAML: {error}") from error
    if not isinstance(schedule, dict) or 0 not in schedule:
        raise ValueError("Schedule must be a frame mapping starting at integer frame 0")
    normalized: dict[int, dict[str, float]] = {}
    for frame, values in schedule.items():
        if type(frame) is not int or not 0 <= frame < count:
            raise ValueError(f"Schedule frame {frame!r} must be an integer in 0..{count - 1}")
        if not isinstance(values, dict) or set(values) != {"light", "exposure_ev"}:
            raise ValueError(f"Frame {frame} must contain exactly light and exposure_ev")
        for key in ("light", "exposure_ev"):
            value = values[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Frame {frame} {key} must be a finite number")
        light, ev = float(values["light"]), float(values["exposure_ev"])
        if not 0.0 <= light <= 1.0:
            raise ValueError(f"Frame {frame} light must be in [0, 1]")
        if not -2.0 <= ev <= 2.0:
            raise ValueError(f"Frame {frame} exposure_ev must be in [-2, 2]")
        normalized[frame] = {"light": light, "exposure_ev": ev}
    dense: list[dict[str, float]] = []
    current = normalized[0]
    for frame in range(count):
        current = normalized.get(frame, current)
        dense.append(dict(current))
    return dense


def effective_power(config: dict[str, Any], commanded: float) -> np.float32:
    response = config["power_response"]
    commanded_table = np.asarray(response["commanded"], dtype=np.float32)
    effective_table = np.asarray(response["effective"], dtype=np.float32)
    if commanded_table.shape != effective_table.shape or commanded_table.ndim != 1:
        raise ValueError("Invalid power response table")
    if len(commanded_table) < 2 or np.any(np.diff(commanded_table) <= 0):
        raise ValueError("Commanded power response values must be strictly increasing")
    if commanded < 0 or commanded > 1:
        raise ValueError("Commanded light must be in [0, 1]")
    return np.float32(np.interp(np.float32(commanded), commanded_table, effective_table))


def load_components(path: Path, expected_shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(COMPONENT_KEYS):
            raise ValueError(f"Invalid component keys in {path}: {archive.files}")
        arrays: dict[str, np.ndarray] = {}
        for key in COMPONENT_KEYS:
            original = archive[key]
            if original.dtype != np.float16 or original.shape != expected_shape:
                raise ValueError(
                    f"{path}:{key} must be float16 {expected_shape}; got {original.dtype} {original.shape}"
                )
            value = np.asarray(original, dtype=np.float32)
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite values in {path}:{key}")
            arrays[key] = value
    return arrays


def relight_packed(components: dict[str, np.ndarray], q: float, ev: float, *, clip: bool = True) -> np.ndarray:
    q32 = np.float32(q)
    packed = (
        components["ambient"]
        + q32 * components["flash"]
        + q32 * (np.float32(1.0) - q32) * components["correction1"]
        + q32 * (np.float32(1.0) - q32) * (np.float32(2.0) * q32 - np.float32(1.0))
        * components["correction2"]
    )
    packed *= np.float32(2.0**ev)
    return np.clip(packed, 0.0, 1.0) if clip else packed


def unpack_rggb(packed: np.ndarray) -> np.ndarray:
    if packed.ndim != 3 or packed.shape[0] != 4:
        raise ValueError(f"Expected packed [4,H,W], got {packed.shape}")
    mosaic = np.empty((packed.shape[1] * 2, packed.shape[2] * 2), dtype=np.float32)
    mosaic[0::2, 0::2] = packed[0]
    mosaic[0::2, 1::2] = packed[1]
    mosaic[1::2, 0::2] = packed[2]
    mosaic[1::2, 1::2] = packed[3]
    return mosaic


def demosaic_linear(packed_ldr: np.ndarray) -> np.ndarray:
    mosaic = unpack_rggb(packed_ldr)
    peak = max(1.0, float(np.max(mosaic)))
    encoded = np.round(np.clip(mosaic / peak, 0.0, 1.0) * 65535.0).astype(np.uint16)
    rgb = cv2.cvtColor(encoded, cv2.COLOR_BAYER_RG2BGR)
    return rgb.astype(np.float32) / np.float32(65535.0) * np.float32(peak)


def linear_to_srgb(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return np.where(
        value <= 0.0031308,
        np.float32(12.92) * value,
        np.float32(1.055) * np.power(value, np.float32(1.0 / 2.4)) - np.float32(0.055),
    )


def render_profile(packed_ldr: np.ndarray, profile: str, config: dict[str, Any]) -> np.ndarray:
    rgb = demosaic_linear(packed_ldr)
    if profile == "linear-gray":
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return np.round(np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8)
    if profile != "viewer-rgb":
        raise ValueError(f"Unknown profile: {profile}")
    display = config["viewer_profile"]
    gains = np.asarray(display["white_balance_rgb"], dtype=np.float32)
    rgb = np.maximum(rgb, 0.0) * gains[None, None, :] * np.float32(display["display_exposure"])
    rgb = rgb / (np.float32(1.0) + rgb)
    return np.round(linear_to_srgb(rgb) * 255.0).astype(np.uint8)


def read_rgb_rows(path: Path) -> list[tuple[int, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        try:
            next(reader)
        except StopIteration as error:
            raise ValueError(f"Empty RGB CSV: {path}") from error
        rows = [(int(row[0]), row[1]) for row in reader if row]
    if len(rows) < 3 or any(a[0] >= b[0] for a, b in zip(rows, rows[1:])):
        raise ValueError("rgb.csv must contain at least three strictly increasing frames")
    return rows


def load_contract(sequence: Path, relighting_config: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[tuple[int, str]]]:
    manifest_path = sequence / "clid_components_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("CLID component manifest must be schema 1 with status complete")
    config = yaml.safe_load(relighting_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Relighting config must be a schema-version 1 mapping")
    producer_hash = manifest.get("identity", {}).get("config", {}).get("sha256")
    if producer_hash not in config.get("accepted_producer_config_sha256", []):
        raise ValueError(
            f"Producer calibration hash {producer_hash!r} is not accepted by {relighting_config}"
        )
    outputs = manifest.get("outputs", {})
    if tuple(outputs.get("component_keys", ())) != COMPONENT_KEYS:
        raise ValueError("Manifest component key contract does not match CLID v6.3")
    if outputs.get("component_dtype") != "float16":
        raise ValueError("Manifest component dtype must be float16")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or len(frames) != manifest.get("extracted_frame_count"):
        raise ValueError("Manifest frame inventory/count is invalid")
    rows = read_rgb_rows(sequence / "rgb.csv")
    if len(rows) != len(frames):
        raise ValueError(f"rgb.csv and component manifest counts differ: {len(rows)} != {len(frames)}")
    for index, (frame, row) in enumerate(zip(frames, rows)):
        stem = Path(row[1]).stem
        if frame.get("index") != index or frame.get("stem") != stem or str(row[0]) != stem:
            raise ValueError(f"Frame identity mismatch at index {index}")
        component_path = sequence / str(frame.get("components_file", ""))
        raw_path = sequence / str(frame.get("raw_file", ""))
        if not component_path.is_file() or not raw_path.is_file():
            raise FileNotFoundError(f"Missing component provenance at frame {index}")
    return manifest, config, frames, rows


def scaled_camera(sequence: Path, output_size: tuple[int, int]) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray]]:
    metadata = json.loads((sequence / "extraction_metadata.json").read_text(encoding="utf-8"))
    bag = Path(metadata["source_bag"]).expanduser().resolve()
    if not bag.is_file():
        raise FileNotFoundError(f"Staged source MCAP is required for CameraInfo: {bag}")
    message = read_camera_info(bag, (CAMERA_INFO_TOPIC,))[CAMERA_INFO_TOPIC]
    camera, distortion, rectification, projection, source_size = camera_values(message)
    sx, sy = output_size[0] / source_size[0], output_size[1] / source_size[1]
    if not math.isclose(sx, sy, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"Component and source aspect ratios differ: scale {sx} x {sy}")
    scaled_camera_matrix = camera.copy()
    scaled_camera_matrix[0, :] *= sx
    scaled_camera_matrix[1, :] *= sy
    scaled_projection = projection.copy()
    scaled_projection[0, :] *= sx
    scaled_projection[1, :] *= sy
    maps = cv2.initUndistortRectifyMap(
        scaled_camera_matrix,
        distortion,
        rectification,
        scaled_projection[:, :3],
        output_size,
        cv2.CV_16SC2,
    )
    return {
        "source_bag": str(bag),
        "source_size": list(source_size),
        "output_size": list(output_size),
        "scale": sx,
        "camera_matrix": scaled_camera_matrix.reshape(-1).tolist(),
        "distortion": distortion.tolist(),
        "rectification": rectification.reshape(-1).tolist(),
        "projection": scaled_projection.reshape(-1).tolist(),
        "fps": float(metadata["fps"]),
        "baseline_m": float(metadata.get("baseline_m", 0.0)),
    }, maps


def write_mono_calibration(path: Path, calibration: dict[str, Any]) -> None:
    projection = np.asarray(calibration["projection"]).reshape(3, 4)
    width, height = calibration["output_size"]
    content = {
        "cameras": [{
            "cam_name": "rgb_0",
            "cam_type": "mono",
            "cam_model": "radtan5",
            "distortion_type": "radtan5",
            "focal_length": [float(projection[0, 0]), float(projection[1, 1])],
            "principal_point": [float(projection[0, 2]), float(projection[1, 2])],
            "distortion_coefficients": [0.0] * 5,
            "image_dimension": [width, height],
            "fps": calibration["fps"],
            "T_BS": np.eye(4).reshape(-1).tolist(),
        }]
    }
    atomic_json(path, content)


def _variant_valid(sequence: Path, fingerprint: str, stems: list[str]) -> bool:
    marker = sequence / "relight.json"
    if not marker.is_file():
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    support = ("rgb.csv", "calibration.yaml", "groundtruth.csv", "relight.csv", "schedule.yaml")
    inventory = metadata.get("output_inventory")
    if not isinstance(inventory, dict) or set(inventory) != set(stems):
        return False
    images = {stem: sequence / "rgb_0" / f"{stem}.png" for stem in stems}
    return metadata.get("status") == "complete" and metadata.get("fingerprint") == fingerprint and all(
        (sequence / name).is_file() for name in support
    ) and all(path.is_file() and path.stat().st_size == inventory[stem] for stem, path in images.items())


@contextmanager
def variant_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def print_schedule(dense: list[dict[str, float]], config: dict[str, Any]) -> None:
    print("Resolved relighting schedule:", flush=True)
    start = 0
    for end in range(1, len(dense) + 1):
        if end < len(dense) and dense[end] == dense[start]:
            continue
        item = dense[start]
        q = effective_power(config, item["light"])
        print(
            f"  Frames {start}-{end - 1}: light={item['light']:g}, effective={float(q):.7f}, "
            f"exposure={item['exposure_ev']:+g} EV",
            flush=True,
        )
        start = end


def prepare(
    source: Path,
    schedule_yaml: Path,
    profile: str = "viewer-rgb",
    output_root: Path = DEFAULT_OUTPUT,
    relighting_config: Path = DEFAULT_RELIGHTING_CONFIG,
) -> Path:
    source = source.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    schedule_yaml = schedule_yaml.expanduser().resolve()
    relighting_config = relighting_config.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output_root.is_relative_to(source) or output_root.is_relative_to(Path("/mnt/share/nas")):
        raise ValueError("Output must be local and outside the source/NAS tree")
    manifest, config, frames, rows = load_contract(source, relighting_config)
    dense = load_schedule(schedule_yaml, len(frames))
    component_shape = tuple(manifest["outputs"]["component_shape"])
    if len(component_shape) != 3 or component_shape[0] != 4:
        raise ValueError(f"Invalid packed component shape: {component_shape}")
    output_size = (component_shape[2] * 2, component_shape[1] * 2)
    calibration, maps = scaled_camera(source, output_size)
    inventory = []
    for frame in frames:
        component = source / frame["components_file"]
        stat = component.stat()
        inventory.append((frame["stem"], stat.st_size, stat.st_mtime_ns))
    fingerprint = object_fingerprint({
        "renderer_version": RENDERER_VERSION,
        "source": str(source),
        "manifest_sha256": sha256_file(source / "clid_components_manifest.json"),
        "manifest_identity": manifest["identity"],
        "component_inventory": inventory,
        "schedule": dense,
        "profile": profile,
        "profile_version": config["profiles"][profile]["version"],
        "relighting_config_sha256": sha256_file(relighting_config),
        "calibration": calibration,
    })
    variant_root = output_root / source.name / "variants" / fingerprint
    destination = variant_root / "dataset" / source.name
    stems = [frame["stem"] for frame in frames]
    with variant_lock(variant_root / ".prepare.lock"):
        if _variant_valid(destination, fingerprint, stems):
            print(f"Preparation: SKIP (verified) {destination}", flush=True)
            print_schedule(dense, config)
            return destination
        image_folder = destination / "rgb_0"
        image_folder.mkdir(parents=True, exist_ok=True)
        relight_rows: list[list[Any]] = []
        previous: dict[str, float] | None = None
        for index, (frame, rgb_row, setting) in enumerate(zip(frames, rows, dense)):
            q = effective_power(config, setting["light"])
            target = image_folder / f"{frame['stem']}.png"
            reused = False
            if target.is_file():
                cached = cv2.imread(str(target), cv2.IMREAD_UNCHANGED)
                channels_ok = cached is not None and (
                    (profile == "linear-gray" and cached.ndim == 2)
                    or (profile == "viewer-rgb" and cached.ndim == 3 and cached.shape[2] == 3)
                )
                reused = channels_ok and cached.shape[:2] == (output_size[1], output_size[0]) and cached.dtype == np.uint8
            if not reused:
                components = load_components(source / frame["components_file"], component_shape)
                packed = relight_packed(components, q, setting["exposure_ev"])
                rendered = render_profile(packed, profile, config)
                rectified = cv2.remap(rendered, *maps, interpolation=cv2.INTER_LINEAR)
                encoded = cv2.cvtColor(rectified, cv2.COLOR_RGB2BGR) if rectified.ndim == 3 else rectified
                temporary = target.with_name(target.stem + ".tmp.png")
                if not cv2.imwrite(str(temporary), encoded):
                    raise OSError(f"Cannot write rendered frame: {temporary}")
                temporary.replace(target)
            relight_rows.append([
                index, rgb_row[0], setting["light"], float(q), setting["exposure_ev"],
                frame["components_file"], f"rgb_0/{frame['stem']}.png", profile,
            ])
            changed = previous != setting
            if changed or (index + 1) % 50 == 0 or index == len(frames) - 1:
                print(
                    f"Prepare: {index + 1}/{len(frames)} ({100 * (index + 1) / len(frames):.1f}%) | "
                    f"frame {index} | light={setting['light']:g} effective={float(q):.7f} | "
                    f"exposure={setting['exposure_ev']:+g} EV" + (" | reused" if reused else ""),
                    flush=True,
                )
            previous = setting
        write_csv(destination / "rgb.csv", ["ts_rgb_0 (ns)", "path_rgb_0"], [
            [stamp, f"rgb_0/{Path(path).stem}.png"] for stamp, path in rows
        ])
        shutil.copy2(source / "groundtruth.csv", destination / "groundtruth.csv")
        write_mono_calibration(destination / "calibration.yaml", calibration)
        write_csv(destination / "relight.csv", [
            "frame_num", "timestamp_ns", "light_commanded", "light_effective", "exposure_ev",
            "component_source", "output_path", "profile",
        ], relight_rows)
        resolved = {index: item for index, item in enumerate(dense)}
        temporary_schedule = destination / "schedule.yaml.tmp"
        temporary_schedule.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        temporary_schedule.replace(destination / "schedule.yaml")
        metadata = {
            "schema_version": 1,
            "status": "complete",
            "fingerprint": fingerprint,
            "renderer_version": RENDERER_VERSION,
            "source": str(source),
            "source_manifest_sha256": sha256_file(source / "clid_components_manifest.json"),
            "relighting_config": str(relighting_config),
            "relighting_config_sha256": sha256_file(relighting_config),
            "profile": profile,
            "profile_version": config["profiles"][profile]["version"],
            "frames": len(frames),
            "output_inventory": {
                stem: (destination / "rgb_0" / f"{stem}.png").stat().st_size for stem in stems
            },
            "calibration": calibration,
            "stereo_status": "blocked: clid_components_1 is not available",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(destination / "relight.json", metadata)
    print(f"Prepared relighted sequence: {destination}", flush=True)
    return destination


def temporal_completion(trajectory: Path, groundtruth: Path) -> dict[str, float]:
    def bounds(path: Path) -> tuple[float, float]:
        with path.open(newline="", encoding="utf-8") as stream:
            values = [float(row[0]) for row in list(csv.reader(stream))[1:] if row]
        if max(map(abs, values)) > 1e12:
            values = [value / 1e9 for value in values]
        return min(values), max(values)

    estimate_start, estimate_end = bounds(trajectory)
    reference_start, reference_end = bounds(groundtruth)
    duration = reference_end - reference_start
    overlap = max(0.0, min(estimate_end, reference_end) - max(estimate_start, reference_start))
    return {
        "reference_duration_s": duration,
        "trajectory_duration_s": estimate_end - estimate_start,
        "overlap_duration_s": overlap,
        "completion_ratio": overlap / duration if duration > 0 else 0.0,
    }


def run_slam(args: argparse.Namespace, prepared: Path) -> Path:
    from Baselines.get_baseline import get_baseline, list_available_baselines
    from Run.single_sequence import evaluate_single_trajectory, run_single_baseline

    if args.slam not in list_available_baselines():
        raise ValueError(f"Unknown SLAM baseline: {args.slam}")
    baseline = get_baseline(args.slam)
    if "mono" not in baseline.modes:
        raise ValueError(f"{args.slam} does not support mono mode")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    result = args.output_root / args.sequence.resolve().name / "results" / f"{timestamp}_{args.slam}_{args.profile}"
    result.mkdir(parents=True)
    config_path = result / "config.yaml"
    config = {
        "DATASET": {
            "base_path": str(prepared.parent),
            "name": prepared.name,
            "baseline": args.slam,
            "dataset": "lightning",
            "sensor_type": "mono",
            "groundtruth_format": "kitti",
            "output_dir": str(result),
        },
        "EVALUATION": {"max_time_difference_s": args.max_time_difference_s, "fast_lio": {"enabled": False}},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    for name in ("relight.json", "relight.csv", "schedule.yaml"):
        shutil.copy2(prepared / name, result / name)
    log_path = result / "run.log"
    state = {
        "status": "running", "source": str(args.sequence.resolve()), "prepared_sequence": str(prepared),
        "baseline": args.slam, "mode": "mono", "profile": args.profile, "config": str(config_path),
        "log": str(log_path), "started_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(result / "run.json", state)
    print_schedule(load_schedule(args.schedule_yaml, len(read_rgb_rows(prepared / "rgb.csv"))),
                   yaml.safe_load(args.relighting_config.read_text(encoding="utf-8")))
    headless = args.action == "run"
    stop = threading.Event()
    started = time.monotonic()

    def heartbeat() -> None:
        while not stop.wait(30):
            print(f"SLAM active for {time.monotonic() - started:.0f}s; runner exposes no frame percentage", flush=True)

    monitor = threading.Thread(target=heartbeat, daemon=True)
    monitor.start()
    try:
        with tee_output(log_path):
            print(f"SLAM starting: {args.slam} mono ({args.profile})", flush=True)
            trajectory = run_single_baseline(config_path, headless=headless)
            stop.set()
            monitor.join(timeout=2)
            print("SLAM complete; evaluation starting", flush=True)
            evaluate_single_trajectory(config_path, trajectory)
        metrics = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
        temporal = temporal_completion(trajectory, prepared / "groundtruth.csv")
        state.update(status="complete", trajectory=str(trajectory), metrics=metrics, temporal=temporal)
        print(
            f"Evaluation complete: APE RMSE={metrics['rmse']['translation']:.4f} m, "
            f"temporal completion={temporal['completion_ratio']:.1%}", flush=True,
        )
    except Exception as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        stop.set()
        monitor.join(timeout=2)
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(result / "run.json", state)
    print(f"Results: {result}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "demo"))
    parser.add_argument("sequence", type=Path, help="Prepared LIGHTNING sequence containing CLID components")
    parser.add_argument("--schedule-yaml", required=True, type=Path)
    parser.add_argument("--profile", choices=("viewer-rgb", "linear-gray"), default="viewer-rgb")
    parser.add_argument("--mode", choices=("mono", "stereo"), default="mono")
    parser.add_argument("--slam", default="orbslam2")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--relighting-config", type=Path, default=DEFAULT_RELIGHTING_CONFIG)
    parser.add_argument("--max-time-difference-s", type=float, default=0.02)
    parser.add_argument("--headless", action="store_true", help="Accepted for clarity; run is always headless")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.sequence = args.sequence.expanduser().resolve()
    args.schedule_yaml = args.schedule_yaml.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.relighting_config = args.relighting_config.expanduser().resolve()
    if args.mode == "stereo":
        parser.error("Stereo relighting requires synchronized clid_components_1, which is not available yet")
    if args.max_time_difference_s <= 0:
        parser.error("--max-time-difference-s must be positive")
    if args.action == "demo" and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        parser.error("Demo requires an X display; use run for headless SLAM")
    prepared = prepare(
        args.sequence, args.schedule_yaml, args.profile, args.output_root, args.relighting_config
    )
    if args.action != "prepare":
        run_slam(args, prepared)


if __name__ == "__main__":
    main()
