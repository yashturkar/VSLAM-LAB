#!/usr/bin/env python3
"""Prepare a processed CLID bag for stereo relit SLAM using IllumiNet release 2.

NAS input and release artifacts are read-only. All generated files live on a
caller-selected local disk. Inference is deliberately a separate, immutable
stage: the IllumiNet release refuses to resume or overwrite partial exports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import select
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from Utilities.clid_relight import (  # noqa: E402
    atomic_json, load_schedule, object_fingerprint, sha256_file, temporal_completion,
)
from Utilities.borealhdr import (  # noqa: E402
    AE50_GAIN, AE50_SOURCE, AE50_TARGET, AE50_THRESHOLD, ae50_next_exposure,
)
from Utilities.clid_auto_light import AutoLightController, AutoLightPolicy  # noqa: E402
from Utilities.extract_lightning_mcap import (  # noqa: E402
    camera_values, read_camera_info, timestamp_ns, write_calibration, write_csv,
)

DEFAULT_ARTIFACTS = Path('/mnt/share/nas/eph/clid-hdr-releases/clid-hdr-v634-release-2')
RELEASE_MANIFEST_SHA256 = 'e4629a19c5ff9370b56b6fed7f2fb23746860c1471f1597dadfff2b1dd927739'
SIDES = ('left', 'right')
KEYS = ('A', 'F', 'C1', 'C2')
INFO_TOPICS = ('/cam_sync/cam0/camera_info', '/cam_sync/cam1/camera_info')
RAW_TOPICS = ('/cam_sync/cam0/image_raw', '/cam_sync/cam1/image_raw')
ODOMETRY_TOPIC = '/odometry'
RENDER_VERSION = 1
AE50_FEEDBACK_VERSION = 1
AE50_INITIAL_MS = 4.0
AE50_MIN_MS = 1.0
AE50_MAX_MS = 16.0
AUTO_LIGHT_VERSION = 2


def local_root_default() -> Path | None:
    if root := os.environ.get('VSLAM_LOCAL_ROOT'):
        return Path(root)
    config = REPO / '.vslamlab-storage'
    if config.is_file() and (root := config.read_text().strip()):
        return Path(root) / 'clid_stereo'
    return None


def inference_command(artifacts: Path, source: Path, coefficients: Path, review: Path,
                      illuminet: Path | None, python: Path | None) -> tuple[list[str], Path, dict[str, str] | None]:
    arguments = ['--manifest', str(artifacts / 'manifest.json'), '--artifacts', str(artifacts),
                 '--camera', 'both', '--sequence', str(source),
                 '--coefficients-output', str(coefficients), '--output', str(review)]
    if illuminet is not None and python is not None:
        illuminet = illuminet.expanduser().resolve()
        python = python.expanduser().resolve()
        if not python.is_file() or not (illuminet / 'scripts/infer_clid_hdr_release.py').is_file():
            raise FileNotFoundError('Explicit IllumiNet checkout or Python runtime is missing')
        environment = os.environ.copy()
        environment['PATH'] = str(python.parent) + os.pathsep + environment.get('PATH', '')
        if shutil.which('ffmpeg', path=environment['PATH']) is None:
            raise FileNotFoundError(f'ffmpeg is unavailable in the inference runtime PATH: {python.parent}')
        return [str(python), str(illuminet / 'scripts/infer_clid_hdr_release.py'), *arguments], illuminet, environment
    pixi = shutil.which('pixi')
    if pixi is None:
        raise FileNotFoundError('pixi is required for packaged CLID inference')
    prefix = [pixi, 'run', '--frozen', '--platform', 'linux-64-cuda128',
              '--manifest-path', str(REPO / 'pixi.toml'), '-e', 'clid-infer']
    subprocess.run([*prefix, 'python', '-c',
                    'import shutil, torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"; '
                    'cap = torch.cuda.get_device_capability(); arch = f"sm_{cap[0]}{cap[1]}"; '
                    'assert arch in torch.cuda.get_arch_list(), f"PyTorch lacks {arch} kernels"; '
                    'assert shutil.which("ffmpeg"), "ffmpeg unavailable"; '
                    'torch.ones(1, device="cuda").square()'], cwd=REPO, check=True)
    return [*prefix, 'illuminet-clid-hdr-release', *arguments], REPO, None


def local_destination(path: Path, source: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_relative_to(Path('/mnt/share/nas')) or resolved.is_relative_to(source):
        raise ValueError(f'Output must not be in NAS or the source tree: {resolved}')
    if resolved.is_relative_to(Path('/mnt/share')) and not resolved.is_relative_to(Path('/mnt/share/local')):
        raise ValueError(f'Output must be under /mnt/share/local, not shared NAS: {resolved}')
    mounts = []
    for line in Path('/proc/mounts').read_text().splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        mount = Path(fields[1].replace('\\040', ' '))
        if resolved == mount or resolved.is_relative_to(mount):
            mounts.append((len(mount.parts), fields[2]))
    if resolved.is_relative_to(Path('/mnt/share/local')) and not any(
        line.split()[1].replace('\\040', ' ') == '/mnt/share/local'
        for line in Path('/proc/mounts').read_text().splitlines() if len(line.split()) >= 3
    ):
        raise ValueError('/mnt/share/local is not mounted; refusing to write onto the underlying disk')
    if mounts:
        filesystem = max(mounts)[1]
        if filesystem.startswith(('nfs', 'cifs', 'smb', 'fuse.sshfs', 'fuse.rclone')):
            raise ValueError(f'Output resolves to a network filesystem ({filesystem}): {resolved}')
    return resolved


def check_inference_space(bag: Path, work: Path) -> None:
    """Reserve approximate full-sequence coefficient and review space."""
    with bag.open('rb') as stream:
        summary = make_reader(stream).get_summary()
    if summary is None or summary.statistics is None:
        raise ValueError(f'MCAP summary unavailable for disk preflight: {bag}')
    counts = {channel.topic: summary.statistics.channel_message_counts.get(channel.id, 0)
              for channel in summary.channels.values()}
    if not all(counts.get(topic, 0) for topic in RAW_TOPICS):
        raise ValueError(f'Both RAW camera topics are required for disk preflight: {bag}')
    frames = max(counts[topic] for topic in RAW_TOPICS)
    required = frames * 50 * 1024**2 + 2 * 1024**3
    free = shutil.disk_usage(work).free
    print(f'Inference disk preflight: ~{required / 1024**3:.1f} GiB needed for {frames} frames/camera; '
          f'{free / 1024**3:.1f} GiB free (estimate)', flush=True)
    if free < required:
        raise OSError(f'Insufficient local space for estimated stereo inference: '
                      f'{free / 1024**3:.1f} GiB free < {required / 1024**3:.1f} GiB estimated')


def source_bag(source: Path) -> Path:
    if not (source / 'manifest.json').is_file():
        raise FileNotFoundError(source / 'manifest.json')
    bags = sorted((source / 'research-bag').glob('*.mcap'))
    if len(bags) != 1:
        raise ValueError(f'Expected exactly one research-bag MCAP in {source}')
    return bags[0]


def source_identity(source: Path, bag: Path) -> dict[str, Any]:
    stat = bag.stat()
    return {
        'source': str(source), 'bag': str(bag), 'bag_size': stat.st_size,
        'bag_mtime_ns': stat.st_mtime_ns,
        'capture_manifest_sha256': sha256_file(source / 'manifest.json'),
    }


def rgb_pairs(path: Path) -> list[int]:
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        required = {'ts_rgb_0 (ns)', 'path_rgb_0', 'ts_rgb_1 (ns)', 'path_rgb_1'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Stereo rgb.csv columns missing: {path}')
        rows = list(reader)
    stamps = []
    for row in rows:
        left, right = int(row['ts_rgb_0 (ns)']), int(row['ts_rgb_1 (ns)'])
        if left != right or Path(row['path_rgb_0']).stem != str(left) or Path(row['path_rgb_1']).stem != str(right):
            raise ValueError(f'Non-synchronous or mismatched preview pair: {row}')
        for side in (0, 1):
            if not (path.parent / row[f'path_rgb_{side}']).is_file():
                raise FileNotFoundError(path.parent / row[f'path_rgb_{side}'])
        stamps.append(left)
    if len(stamps) < 3 or len(set(stamps)) != len(stamps) or stamps != sorted(stamps):
        raise ValueError('Need at least three unique, increasing exact stereo timestamps')
    return stamps


def paired_timestamps(path: Path) -> list[int]:
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['timestamp_ns']:
            raise ValueError(f'Invalid RAW pairing index: {path}')
        stamps = [int(row['timestamp_ns']) for row in reader]
    if len(stamps) < 3 or stamps != sorted(set(stamps)):
        raise ValueError('Need at least three unique increasing RAW stereo timestamps')
    return stamps


def index_source(bag: Path, sequence: Path, identity: dict[str, Any],
                 frames: dict[str, dict[int, dict[str, Any]]], coefficients_hash: str) -> dict[str, Any]:
    """Pair verified inference RAW headers and read odometry; never decode previews."""
    paired = sorted(set(frames['left']) & set(frames['right']))
    if len(paired) < 3:
        raise ValueError(f'Only {len(paired)} exact RAW stereo timestamp matches')
    odometry = {}
    with bag.open('rb') as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, _, _, message in reader.iter_decoded_messages(topics=[ODOMETRY_TOPIC]):
            stamp = timestamp_ns(message)
            pose = message.pose.pose
            odometry.setdefault(stamp, [stamp, pose.position.x, pose.position.y, pose.position.z,
                                         pose.orientation.x, pose.orientation.y, pose.orientation.z,
                                         pose.orientation.w])
            if len(odometry) % 500 == 0:
                print(f'Odometry index: {len(odometry)} poses', flush=True)
    if not odometry:
        raise ValueError('No odometry messages in source bag')
    intervals = np.diff(np.asarray(paired, np.int64)) / 1e9
    fps = float(1 / np.median(intervals[intervals > 0]))
    sequence.mkdir(parents=True, exist_ok=True)
    write_csv(sequence / 'pairing.csv', ['timestamp_ns'], [[stamp] for stamp in paired])
    write_csv(sequence / 'groundtruth.csv',
              ['ts (ns)', 'tx (m)', 'ty (m)', 'tz (m)', 'qx', 'qy', 'qz', 'qw'],
              [odometry[stamp] for stamp in sorted(odometry)])
    metadata = {'schema_version': 1, 'status': 'complete', 'source_bag': str(bag),
                'source_fingerprint': identity, 'coefficients_manifest_sha256': coefficients_hash,
                'stereo_pairs': len(paired),
                'left_raw_frames': len(frames['left']),
                'right_raw_frames': len(frames['right']),
                'odometry_poses': len(odometry), 'fps': fps,
                'topics': {'left': RAW_TOPICS[0], 'right': RAW_TOPICS[1], 'odometry': ODOMETRY_TOPIC}}
    atomic_json(sequence / 'extraction_metadata.json', metadata)
    print(f'RAW index: {len(paired)} exact stereo pairs, {len(odometry)} odometry poses, {fps:.3f} Hz', flush=True)
    return metadata


def validate_base(sequence: Path, identity: dict[str, Any], coefficients_hash: str | None = None) -> list[int] | None:
    marker = sequence / 'extraction_metadata.json'
    if not marker.is_file():
        return None
    try:
        metadata = json.loads(marker.read_text())
        if metadata.get('source_fingerprint') != identity or metadata.get('status') != 'complete':
            return None
        if coefficients_hash is not None and metadata.get('coefficients_manifest_sha256') != coefficients_hash:
            return None
        stamps = paired_timestamps(sequence / 'pairing.csv')
        if len(stamps) != metadata.get('stereo_pairs'):
            return None
        if not (sequence / 'groundtruth.csv').is_file():
            return None
        return stamps
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def validate_coefficients(root: Path, release: dict[str, Any], identity: dict[str, Any],
                          release_manifest_hash: str | None = None) -> dict[str, dict[int, dict[str, Any]]] | None:
    group_path = root / 'stereo-manifest.json'
    if not group_path.is_file():
        return None
    try:
        group = json.loads(group_path.read_text())
        if group.get('complete') is not True or group.get('is_full_sequence') is not True:
            return None
        if release_manifest_hash is not None and group.get('release_manifest_sha256') != release_manifest_hash:
            return None
        if group.get('checkpoint_sha256') != release['sha256'][release['checkpoint']]:
            return None
        if group.get('exact_common_timestamp_count', 0) < 3:
            return None
        result = {}
        for side in SIDES:
            folder = root / side
            manifest_path = folder / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            contract = release['cameras'][side]
            if not (manifest.get('complete') and manifest.get('is_full_sequence')):
                return None
            if manifest.get('camera') != side or manifest.get('camera_serial') != contract['serial']:
                return None
            if manifest.get('topic') != contract['input_topic'] or manifest.get('metadata_topic') != contract['metadata_topic']:
                return None
            if manifest.get('checkpoint_sha256') != group['checkpoint_sha256']:
                return None
            if manifest.get('config_sha256') != release['sha256'][contract['calibration']]:
                return None
            if manifest.get('bag') != identity['bag'] or manifest.get('bag_bytes') != identity['bag_size'] or manifest.get('bag_mtime_ns') != identity['bag_mtime_ns']:
                return None
            if manifest.get('capture_manifest_sha256') != identity['capture_manifest_sha256']:
                return None
            if manifest.get('frame_count') != len(manifest.get('frames', [])):
                return None
            if sha256_file(manifest_path) != group['cameras'][side]['manifest_sha256']:
                return None
            frames = {}
            for frame in manifest['frames']:
                stamp = int(frame['header_stamp_ns'])
                if stamp in frames:
                    return None
                info = frame['coefficients']
                file = folder / info['path']
                if file.stat().st_size != info['bytes']:
                    return None
                frames[stamp] = {'file': file, 'sha256': info['sha256'], 'record': frame}
            if len(frames) != manifest['expected_frames']:
                return None
            result[side] = frames
        if len(set(result['left']) & set(result['right'])) != group['exact_common_timestamp_count']:
            return None
        return result
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def read_component(path: Path, expected_hash: str) -> dict[str, np.ndarray]:
    if sha256_file(path) != expected_hash:
        raise ValueError(f'Component checksum mismatch: {path}')
    with np.load(path, allow_pickle=False) as archive:
        if not set(KEYS).issubset(archive.files):
            raise ValueError(f'Missing coefficient arrays: {path}')
        values = {key: archive[key] for key in KEYS}
    if any(value.dtype != np.float32 or value.shape != (4, 400, 640) or not np.isfinite(value).all() for value in values.values()):
        raise ValueError(f'Invalid float32 coefficients: {path}')
    quantized = {key: value.astype(np.float16) for key, value in values.items()}
    if any(not np.isfinite(value).all() for value in quantized.values()):
        raise ValueError(f'Coefficient cannot be represented as finite float16: {path}')
    return quantized


def synthetic_bayer(values: dict[str, np.ndarray], calibration: dict[str, Any], light: float,
                    ev: float, maximum: int = 65535) -> np.ndarray:
    """Calibrated, unrectified Bayer response; AE meters this before display gamma."""
    response = calibration['power_response']
    q = np.float32(np.interp(light, response['commanded'], response['effective']))
    a, f, c1, c2 = (values[key].astype(np.float32) for key in KEYS)
    linear = (a + q * f + q * (1 - q) * c1 + q * (1 - q) * (2 * q - 1) * c2) * np.float32(2.0 ** ev)
    site = np.clip(linear, 0, 1)
    alpha = np.asarray([calibration['response']['bayer_alpha'][key] for key in ('R', 'G1', 'G2', 'B')], np.float32)
    site = np.power(site, alpha[:, None, None])
    mosaic = np.empty((site.shape[1] * 2, site.shape[2] * 2), np.float32)
    mosaic[0::2, 0::2], mosaic[0::2, 1::2] = site[0], site[1]
    mosaic[1::2, 0::2], mosaic[1::2, 1::2] = site[2], site[3]
    return np.round(mosaic * maximum).astype(np.uint16)


def render(values: dict[str, np.ndarray], calibration: dict[str, Any], light: float, ev: float) -> np.ndarray:
    bayer = synthetic_bayer(values, calibration, light, ev)
    # In OpenCV's Bayer aliases, RG2RGB yields BGR bytes for our R-at-(0,0)
    # mosaic. Verify with the red-site regression test before changing this.
    bgr = cv2.cvtColor(bayer, cv2.COLOR_BAYER_RG2RGB).astype(np.float32) / 65535.0
    return np.round(np.power(np.clip(bgr, 0, 1), 1 / 2.2) * 255).astype(np.uint8)


def stereo_maps(bag: Path, size: tuple[int, int]) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray, float]:
    info = read_camera_info(bag, INFO_TOPICS)
    maps = {}
    projections = []
    for side, topic in zip(SIDES, INFO_TOPICS):
        camera, distortion, rectification, projection, source_size = camera_values(info[topic])
        sx, sy = size[0] / source_size[0], size[1] / source_size[1]
        if not math.isclose(sx, sy, abs_tol=1e-9):
            raise ValueError(f'CLID output aspect ratio differs from {side} CameraInfo: {source_size} -> {size}')
        camera[:2, :] *= sx
        projection[:2, :] *= sx
        maps[side] = cv2.initUndistortRectifyMap(camera, distortion, rectification, projection[:, :3], size, cv2.CV_16SC2)
        projections.append(projection)
    if not np.allclose(projections[0][:, :3], projections[1][:, :3], rtol=0, atol=1e-6):
        raise ValueError('Stereo rectified projection matrices have different intrinsics')
    baseline = -float(projections[1][0, 3] - projections[0][0, 3]) / float(projections[0][0, 0])
    if not 0 < baseline < 5:
        raise ValueError(f'Implausible stereo baseline: {baseline} m')
    return maps, projections[0], baseline


def valid_ae_trace(destination: Path, stamps: list[int], light: float) -> bool:
    """Do not reuse an AE variant with incomplete or inconsistent feedback."""
    try:
        with (destination / 'ae_trace.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        dense = load_schedule(destination / 'schedule.yaml', len(stamps))
        if len(rows) != len(stamps):
            return False
        for index, (stamp, row, setting) in enumerate(zip(stamps, rows, dense)):
            if int(row['frame_num']) != index or int(row['timestamp_ns']) != stamp:
                return False
            exposure = float(row['applied_exposure_ms'])
            if not AE50_MIN_MS <= exposure <= AE50_MAX_MS:
                return False
            if not math.isclose(exposure, AE50_INITIAL_MS * 2 ** setting['exposure_ev'], abs_tol=1e-9):
                return False
            if not math.isclose(float(row['light_commanded']), light, abs_tol=1e-12):
                return False
            if index + 1 < len(rows) and not math.isclose(
                float(row['applied_next_ms']), float(rows[index + 1]['applied_exposure_ms']), abs_tol=1e-9
            ):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
        return False


def valid_auto_light_trace(destination: Path, stamps: list[int], policy: AutoLightPolicy) -> bool:
    try:
        with (destination / 'auto_light_trace.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        dense = load_schedule(destination / 'schedule.yaml', len(stamps))
        if len(rows) != len(stamps):
            return False
        last_change = None
        for index, (stamp, row, setting) in enumerate(zip(stamps, rows, dense)):
            light = float(row['light_commanded'])
            ms = float(row['applied_exposure_ms'])
            right_mean = float(row['right_mean_dn_12bit'])
            right_saturated = float(row['right_saturated_fraction'])
            if int(row['frame_num']) != index or int(row['timestamp_ns']) != stamp:
                return False
            if not policy.light_min <= light <= 1 or not 1 <= ms <= 16:
                return False
            if not math.isfinite(right_mean) or not 0 <= right_mean <= 4095:
                return False
            if not math.isfinite(right_saturated) or not 0 <= right_saturated <= 1:
                return False
            if not math.isclose(light, setting['light'], abs_tol=1e-9) or not math.isclose(
                ms, 4 * 2 ** setting['exposure_ev'], abs_tol=1e-9):
                return False
            if index and not math.isclose(light, float(rows[index - 1]['applied_next_light']), abs_tol=1e-9):
                return False
            if index and not math.isclose(ms, float(rows[index - 1]['applied_next_ms']), abs_tol=1e-9):
                return False
            if index and abs(light - float(rows[index - 1]['light_commanded'])) > 1e-12:
                if abs(light - float(rows[index - 1]['light_commanded'])) > policy.max_step + 1e-9:
                    return False
                if last_change is not None and stamp - last_change < math.ceil(1e9 / policy.rate_hz):
                    return False
                last_change = stamp
        return True
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
        return False


def prepare_variant(base: Path, bag: Path, coefficients: Path, frames: dict[str, dict[int, dict[str, Any]]],
                    artifacts: Path, release: dict[str, Any], schedule: Path | None, output: Path,
                    ae_light: float | None = None, auto_policy: AutoLightPolicy | None = None) -> Path:
    stamps = [stamp for stamp in paired_timestamps(base / 'pairing.csv') if stamp in frames['left'] and stamp in frames['right']]
    if len(stamps) < 3:
        raise ValueError('Fewer than three exact preview/left-RAW/right-RAW timestamp matches')
    if sum(value is not None for value in (schedule, ae_light, auto_policy)) != 1:
        raise ValueError('Choose exactly one of schedule YAML, AE50 fixed light, or AE50 auto-light')
    if ae_light is not None and (not math.isfinite(ae_light) or not 0 <= ae_light <= 1):
        raise ValueError('AE50 light must be a finite fraction in [0, 1]')
    dense = load_schedule(schedule, len(stamps)) if schedule is not None else None
    calibrations = {side: json.loads((artifacts / release['cameras'][side]['calibration']).read_text()) for side in SIDES}
    fingerprint = object_fingerprint({
        'renderer_version': RENDER_VERSION, 'bag': str(bag),
        'coefficients_manifest': sha256_file(coefficients / 'stereo-manifest.json'),
        'base_metadata': sha256_file(base / 'extraction_metadata.json'),
        'calibrations': {side: sha256_file(artifacts / release['cameras'][side]['calibration']) for side in SIDES},
        'policy': {'schedule': dense} if dense is not None else {
            'ae50_source': AE50_SOURCE, 'feedback_version': AE50_FEEDBACK_VERSION,
            'light': ae_light, 'initial_ms': AE50_INITIAL_MS,
            'min_ms': AE50_MIN_MS, 'max_ms': AE50_MAX_MS,
        } if auto_policy is None else {'auto_light_version': AUTO_LIGHT_VERSION,
            'ae50_source': AE50_SOURCE, 'feedback_version': AE50_FEEDBACK_VERSION,
            'light_min': auto_policy.light_min, 'preferred_ms': auto_policy.preferred_ms,
            'rate_hz': auto_policy.rate_hz, 'max_step': auto_policy.max_step,
            'initial_ms': AE50_INITIAL_MS, 'max_ms': AE50_MAX_MS},
    })
    destination = output / 'variants' / fingerprint / 'dataset' / base.name
    marker = destination / 'relight.json'
    if marker.is_file():
        previous = json.loads(marker.read_text())
        if previous.get('status') == 'complete' and previous.get('fingerprint') == fingerprint:
            inventory = previous.get('image_bytes', {})
            support = ('rgb.csv', 'calibration.yaml', 'groundtruth.csv', 'relight.csv')
            if ae_light is not None:
                support += ('schedule.yaml', 'ae_trace.csv')
            if auto_policy is not None:
                support += ('schedule.yaml', 'auto_light_trace.csv')
            if all((destination / name).is_file() for name in support) and len(inventory) == 2 * len(stamps) and all(
                (destination / path).is_file() and (destination / path).stat().st_size == size
                for path, size in inventory.items()
            ) and rgb_pairs(destination / 'rgb.csv') == stamps and (
                ae_light is None or valid_ae_trace(destination, stamps, ae_light)
            ) and (
                auto_policy is None or valid_auto_light_trace(destination, stamps, auto_policy)
            ):
                print(f'Relight: SKIP (verified) {destination}', flush=True)
                return destination
    maps, projection, baseline = stereo_maps(bag, (1280, 800))
    metadata = json.loads((base / 'extraction_metadata.json').read_text())
    for side in SIDES:
        (destination / ('rgb_0' if side == 'left' else 'rgb_1')).mkdir(parents=True, exist_ok=True)
    rows = []
    trace = []
    ae_rows = []
    auto_rows = []
    image_bytes = {}
    exposure_ms = AE50_INITIAL_MS
    controller = AutoLightController(auto_policy) if auto_policy is not None else None
    interval_ns = int(np.median(np.diff(np.asarray(stamps, dtype=np.int64))))
    if dense is None:
        dense = []
    for index, stamp in enumerate(stamps):
        setting = dense[index] if ae_light is None and controller is None else {
            'light': controller.light if controller is not None else ae_light,
            'exposure_ev': math.log2((controller.exposure_ms if controller is not None else exposure_ms)
                                     / AE50_INITIAL_MS),
        }
        row = [stamp, f'rgb_0/{stamp}.png', stamp, f'rgb_1/{stamp}.png']
        for side, relative in zip(SIDES, (row[1], row[3])):
            info = frames[side][stamp]
            packed = read_component(info['file'], info['sha256'])
            if side == 'left' and ae_light is not None:
                bayer12 = synthetic_bayer(packed, calibrations['left'], ae_light,
                                          setting['exposure_ev'], maximum=4095)
                mean_dn, requested_ms = ae50_next_exposure(bayer12, exposure_ms)
                next_ms = min(AE50_MAX_MS, max(AE50_MIN_MS, requested_ms))
                clamp = 'min' if requested_ms < AE50_MIN_MS else 'max' if requested_ms > AE50_MAX_MS else 'none'
                ae_rows.append([index, stamp, ae_light, setting['exposure_ev'], exposure_ms,
                                mean_dn, mean_dn - AE50_TARGET, requested_ms, next_ms, clamp])
            if side == 'left' and controller is not None:
                bayer12 = synthetic_bayer(packed, calibrations['left'], setting['light'],
                                          setting['exposure_ev'], maximum=4095)
                mean_dn, requested_ms = ae50_next_exposure(bayer12, controller.exposure_ms)
                sampled = {key: value[:, ::8, ::8] for key, value in packed.items()}

                @lru_cache(maxsize=256)
                def meter(light: float, ms: float) -> float:
                    candidate = synthetic_bayer(sampled, calibrations['left'], light,
                                                math.log2(ms / AE50_INITIAL_MS), maximum=4095)
                    return float(np.mean(cv2.cvtColor(candidate, cv2.COLOR_BAYER_RG2GRAY)))

                next_stamp = stamps[index + 1] if index + 1 < len(stamps) else stamp + interval_ns
                decision = controller.step(requested_ms, next_stamp, meter)

                def full_meter(light: float, ms: float) -> float:
                    candidate = synthetic_bayer(packed, calibrations['left'], light,
                                                math.log2(ms / AE50_INITIAL_MS), maximum=4095)
                    return float(np.mean(cv2.cvtColor(candidate, cv2.COLOR_BAYER_RG2GRAY)))

                requested_full = full_meter(setting['light'], min(16.0, max(1.0, requested_ms)))
                predicted_full = full_meter(float(decision['next_light']), float(decision['next_ms']))
                auto_rows.append([index, stamp, setting['light'],
                                  AE50_INITIAL_MS * 2 ** setting['exposure_ev'], mean_dn,
                                  mean_dn - AE50_TARGET, requested_ms, requested_full,
                                  decision['ideal_light'], decision['next_light'], decision['next_ms'],
                                  decision['light_limit'], decision['exposure_limit'],
                                  predicted_full, predicted_full - requested_full])
            if side == 'right' and controller is not None:
                right_bayer = synthetic_bayer(packed, calibrations['right'], setting['light'],
                                              setting['exposure_ev'], maximum=4095)
                right_mean = float(np.mean(cv2.cvtColor(right_bayer, cv2.COLOR_BAYER_RG2GRAY)))
                right_saturated = float(np.count_nonzero(right_bayer >= 4094) / right_bayer.size)
                auto_rows[-1].extend([right_mean, right_saturated])
            image = cv2.remap(render(packed, calibrations[side], setting['light'], setting['exposure_ev']),
                              *maps[side], interpolation=cv2.INTER_LINEAR)
            target = destination / relative
            temporary = target.with_name(f'{target.stem}.tmp.png')
            if not cv2.imwrite(str(temporary), image):
                raise OSError(f'Could not write {temporary}')
            temporary.replace(target)
            image_bytes[relative] = target.stat().st_size
        if ae_light is not None or controller is not None:
            dense.append(setting)
            if ae_light is not None:
                exposure_ms = next_ms
        rows.append(row)
        trace.append([index, stamp, setting['light'], setting['exposure_ev'],
                      float(np.interp(setting['light'], calibrations['left']['power_response']['commanded'], calibrations['left']['power_response']['effective'])),
                      float(np.interp(setting['light'], calibrations['right']['power_response']['commanded'], calibrations['right']['power_response']['effective']))])
        if index == 0 or (index + 1) % 25 == 0 or index == len(stamps) - 1:
            ae_status = (f', mean={mean_dn:.1f}/4095 DN, next={next_ms:.3f} ms'
                         f' ({clamp} clamp)' if ae_light is not None else
                         f', mean={mean_dn:.1f}/4095 DN, next light={decision["next_light"]:.3f},'
                         f' next={decision["next_ms"]:.3f} ms'
                         if controller is not None else '')
            print(f'Relight: {index + 1}/{len(stamps)} pairs ({100 * (index + 1) / len(stamps):.1f}%) | light={setting["light"]:g}, EV={setting["exposure_ev"]:+g}{ae_status}', flush=True)
    write_csv(destination / 'rgb.csv', ['ts_rgb_0 (ns)', 'path_rgb_0', 'ts_rgb_1 (ns)', 'path_rgb_1'], rows)
    write_csv(destination / 'relight.csv', ['frame_num', 'timestamp_ns', 'light_commanded', 'exposure_ev', 'left_effective_power', 'right_effective_power'], trace)
    if ae_light is not None:
        (destination / 'schedule.yaml').write_text(yaml.safe_dump({index: setting for index, setting in enumerate(dense)}, sort_keys=False))
        write_csv(destination / 'ae_trace.csv', ['frame_num', 'timestamp_ns', 'light_commanded',
                  'exposure_ev', 'applied_exposure_ms', 'left_mean_dn_12bit', 'error_dn',
                  'requested_next_ms', 'applied_next_ms', 'clamp'], ae_rows)
    if controller is not None:
        (destination / 'schedule.yaml').write_text(yaml.safe_dump({index: setting for index, setting in enumerate(dense)}, sort_keys=False))
        write_csv(destination / 'auto_light_trace.csv', ['frame_num', 'timestamp_ns', 'light_commanded',
                  'applied_exposure_ms', 'left_mean_dn_12bit', 'error_dn', 'ae50_requested_ms',
                  'requested_mean_dn_full', 'ideal_next_light', 'applied_next_light',
                  'applied_next_ms', 'light_limit', 'exposure_limit', 'predicted_next_mean_dn_full',
                  'prediction_error_dn', 'right_mean_dn_12bit', 'right_saturated_fraction'], auto_rows)
    shutil.copy2(base / 'groundtruth.csv', destination / 'groundtruth.csv')
    write_calibration(destination / 'calibration.yaml', projection, baseline, float(metadata['fps']), (1280, 800))
    atomic_json(marker, {
        'schema_version': 1, 'status': 'complete', 'fingerprint': fingerprint,
        'source_bag': str(bag), 'base_sequence': str(base), 'coefficients': str(coefficients),
        'schedule_yaml': str(schedule) if schedule is not None else str(destination / 'schedule.yaml'),
        'ae50': None if ae_light is None else {
            'source': AE50_SOURCE, 'light_commanded': ae_light, 'initial_exposure_ms': AE50_INITIAL_MS,
            'min_exposure_ms': AE50_MIN_MS, 'max_exposure_ms': AE50_MAX_MS,
            'target_dn_12bit': AE50_TARGET, 'deadband_dn': AE50_THRESHOLD, 'gain_ms_per_dn': AE50_GAIN,
            'feedback': 'left synthetic unrectified BayerRG12; frame n controls frame n+1',
            'trace': 'ae_trace.csv', 'feedback_version': AE50_FEEDBACK_VERSION,
        }, 'frame_count': len(stamps), 'baseline_m': baseline,
        'auto_light': None if auto_policy is None else {
            'version': AUTO_LIGHT_VERSION, 'ae50_source': AE50_SOURCE,
            'light_min': auto_policy.light_min, 'light_max': 1.0,
            'light_rate_hz': auto_policy.rate_hz, 'light_max_step': auto_policy.max_step,
            'preferred_exposure_ms': auto_policy.preferred_ms,
            'trace': 'auto_light_trace.csv', 'feedback': 'left synthetic BayerRG12; frame n controls n+1',
            'search': '8x subsampled Bayer candidate grid; full-resolution prediction logged',
        },
        'resolution': [1280, 800], 'image_bytes': image_bytes,
        'display': 'camera-specific Bayer response, demosaic, gamma 2.2; no independent display exposure',
        'component_storage': 'IllumiNet float32 loaded and quantized to float16 before synthesis',
        'groundtruth_caveat': 'Extracted odometry is body-frame, not independently surveyed camera GT',
        'completed_at': datetime.now(timezone.utc).isoformat(),
    })
    print(f'Stereo SLAM sequence: {destination}', flush=True)
    return destination


def launch_control_plot(prepared: Path) -> subprocess.Popen[str]:
    """Start the file-open plot before SLAM can read the first frame."""
    if not os.environ.get('DISPLAY'):
        raise RuntimeError('Live control plot requires an X display; use demo with DISPLAY set')
    script = Path(__file__).with_name('clid_control_plot.py')
    process = subprocess.Popen([sys.executable, str(script), str(prepared)],
                               stdout=subprocess.PIPE, text=True)
    try:
        ready, _, _ = select.select([process.stdout], [], [], 10)
        if not ready or process.stdout.readline().strip() != 'READY':
            raise RuntimeError('CLID control plot did not start; check the graphical display')
    except Exception:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3)
        raise
    process.stdout.close()
    print('CLID control plot: watching prepared left-frame reads', flush=True)
    return process


def run_slam(prepared: Path, slam: str, headless: bool, output: Path,
             control_plot: bool = False) -> Path:
    from Baselines.get_baseline import get_baseline, list_available_baselines
    from Run.single_sequence import evaluate_single_trajectory, run_single_baseline
    if slam not in list_available_baselines() or 'stereo' not in get_baseline(slam).modes:
        raise ValueError(f'{slam} is not an available stereo SLAM baseline')
    result = output / 'results' / f'{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}_{slam}_stereo'
    result.mkdir(parents=True)
    config = result / 'config.yaml'
    config.write_text(yaml.safe_dump({
        'DATASET': {'base_path': str(prepared.parent), 'name': prepared.name, 'baseline': slam,
                    'dataset': 'lightning', 'sensor_type': 'stereo', 'groundtruth_format': 'kitti',
                    'output_dir': str(result)},
        'EVALUATION': {'max_time_difference_s': 0.02, 'fast_lio': {'enabled': False}},
    }, sort_keys=False))
    plotter = None
    try:
        if control_plot:
            plotter = launch_control_plot(prepared)
        trajectory = run_single_baseline(config, headless=headless)
        evaluate_single_trajectory(config, trajectory)
    finally:
        if plotter is not None and plotter.poll() is None:
            plotter.terminate()
            try:
                plotter.wait(timeout=3)
            except subprocess.TimeoutExpired:
                plotter.kill()
                plotter.wait()
    temporal = temporal_completion(trajectory, prepared / 'groundtruth.csv')
    atomic_json(result / 'clid_run.json', {'prepared': str(prepared), 'trajectory': str(trajectory),
                                           'temporal': temporal, 'slam': slam, 'mode': 'stereo'})
    print(f'SLAM results: {result}', flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'status', 'prepare-coefficients', 'prepare', 'run', 'demo'))
    parser.add_argument('source', type=Path, help='Processed CLID sequence root with manifest.json and research-bag/*.mcap')
    parser.add_argument('--schedule-yaml', type=Path)
    parser.add_argument('--ae', action='store_true', help='Use BorealHDR classical AE50 on CLID left Bayer feedback')
    parser.add_argument('--light', type=float, help='Fixed commanded light fraction [0,1] for --ae')
    parser.add_argument('--auto-light', action='store_true', help='Allocate AE50 brightness to exposure and light')
    parser.add_argument('--light-min', type=float, help='Auto-light floor; default 0.1 commanded')
    parser.add_argument('--preferred-exposure-ms', type=float, help='Soft exposure preference; default 8 ms')
    parser.add_argument('--light-rate-hz', type=float, help='Maximum light-update frequency; default 3 Hz')
    parser.add_argument('--light-max-step', type=float, help='Maximum commanded light change per update; default 0.1')
    parser.add_argument('--local-root', type=Path, default=local_root_default(),
                        help='Local output root; or set VSLAM_LOCAL_ROOT / .vslamlab-storage')
    parser.add_argument('--illuminet', type=Path, help='Optional development checkout override')
    parser.add_argument('--illuminet-python', type=Path, help='Python for --illuminet override')
    parser.add_argument('--artifacts', type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument('--slam', default='orbslam3')
    parser.add_argument('--control-plot', action='store_true',
                        help='Show live commanded light and exposure ms during demo')
    parser.add_argument('--ae-plot', action='store_true', help='Alias for --control-plot in demo')
    args = parser.parse_args()
    if (args.control_plot or args.ae_plot) and args.action != 'demo':
        parser.error('--control-plot/--ae-plot requires demo')
    if args.action == 'prepare-coefficients' and args.schedule_yaml is not None:
        parser.error('--schedule-yaml is not used by prepare-coefficients')
    options = (args.light_min, args.preferred_exposure_ms, args.light_rate_hz, args.light_max_step)
    if args.action in ('check', 'status', 'prepare-coefficients') and (
        args.ae or args.light is not None or args.auto_light or any(v is not None for v in options)
    ):
        parser.error('AE/light controls are only used by prepare/run/demo')
    if args.ae and args.schedule_yaml is not None:
        parser.error('--ae and --schedule-yaml are mutually exclusive')
    if args.auto_light and (not args.ae or args.light is not None):
        parser.error('--auto-light requires --ae and cannot use fixed --light')
    if not args.auto_light and any(v is not None for v in options):
        parser.error('auto-light settings require --auto-light')
    if not args.auto_light and args.ae != (args.light is not None):
        parser.error('--ae requires --light, and --light requires --ae')
    if args.light is not None and (not math.isfinite(args.light) or not 0 <= args.light <= 1):
        parser.error('--light must be a finite fraction in [0, 1]')
    try:
        auto_policy = AutoLightPolicy(
            light_min=0.1 if args.light_min is None else args.light_min,
            preferred_ms=8.0 if args.preferred_exposure_ms is None else args.preferred_exposure_ms,
            rate_hz=3.0 if args.light_rate_hz is None else args.light_rate_hz,
            max_step=0.1 if args.light_max_step is None else args.light_max_step,
        ) if args.auto_light else None
    except ValueError as error:
        parser.error(str(error))
    if args.local_root is None:
        parser.error('choose --local-root on a local disk, or set VSLAM_LOCAL_ROOT')
    if (args.illuminet is None) != (args.illuminet_python is None):
        parser.error('--illuminet and --illuminet-python must be used together')
    source = args.source.expanduser().resolve()
    bag = source_bag(source)
    output = local_destination(args.local_root, source)
    identity = source_identity(source, bag)
    key = object_fingerprint(identity)[:12]
    work = local_destination(output / f'{source.name}_{key}', source)
    base = local_destination(work / 'base' / source.name, source)
    coefficients = local_destination(work / 'coefficients', source)
    review = local_destination(work / 'review', source)
    artifacts = args.artifacts.expanduser().resolve()
    manifest_path = artifacts / 'manifest.json'
    if sha256_file(manifest_path) != RELEASE_MANIFEST_SHA256:
        raise ValueError(f'Release manifest checksum mismatch: {manifest_path}')
    release = json.loads(manifest_path.read_text())
    for name, digest in release['sha256'].items():
        if sha256_file(artifacts / name) != digest:
            raise ValueError(f'Release artifact checksum mismatch: {name}')
    frames = validate_coefficients(coefficients, release, identity, sha256_file(manifest_path))
    coefficients_hash = sha256_file(coefficients / 'stereo-manifest.json') if frames else None
    base_stamps = validate_base(base, identity, coefficients_hash) if frames else None
    state = {'source': str(source), 'bag': str(bag), 'work': str(work),
             'coefficients_path': str(coefficients),
             'raw_index': 'ready' if base_stamps else ('partial/stale' if base.exists() else 'missing'),
             'coefficients': 'ready' if frames else ('partial/stale' if coefficients.exists() else 'missing'),
             'exact_pairs': len(set(base_stamps or []) & set(frames['left']) & set(frames['right'])) if frames else 0}
    print(json.dumps(state, indent=2), flush=True)
    if args.action in ('check', 'status'):
        return
    schedule = None
    if args.action != 'prepare-coefficients':
        if not args.ae:
            if args.schedule_yaml is None:
                parser.error('--schedule-yaml or --ae --light is required for prepare/run/demo')
            schedule = args.schedule_yaml.expanduser().resolve()
            if not schedule.is_file():
                raise FileNotFoundError(schedule)
    work.mkdir(parents=True, exist_ok=True)
    if frames is None:
        if coefficients.exists() or review.exists():
            raise RuntimeError(f'Inference is partial/stale; preserved for inspection: {coefficients}, {review}')
        check_inference_space(bag, work)
        print('Stereo inference: starting with camera-specific RAW and calibrations', flush=True)
        command, cwd, inference_env = inference_command(
            artifacts, source, coefficients, review, args.illuminet, args.illuminet_python)
        subprocess.run(command, cwd=cwd, env=inference_env, check=True)
        frames = validate_coefficients(coefficients, release, identity, sha256_file(manifest_path))
        if frames is None:
            raise RuntimeError(f'Inference output failed validation: {coefficients}')
        coefficients_hash = sha256_file(coefficients / 'stereo-manifest.json')
    else:
        print(f'Stereo inference: SKIP (verified) {len(frames["left"])} left, {len(frames["right"])} right', flush=True)
    if args.action == 'prepare-coefficients':
        print(f'Coefficients ready: {len(frames["left"])} left, {len(frames["right"])} right -> {coefficients}', flush=True)
        return
    if base_stamps is None:
        if base.exists():
            raise RuntimeError(f'RAW/odometry index is partial/stale; inspect before replacing: {base}')
        print('RAW/odometry indexing: starting from verified inference timestamps (no preview images)', flush=True)
        index_source(bag, base, identity, frames, coefficients_hash)
        base_stamps = validate_base(base, identity, coefficients_hash)
        if base_stamps is None:
            raise RuntimeError('RAW/odometry index did not validate')
    else:
        print(f'RAW/odometry index: SKIP (verified) {len(base_stamps)} pairs', flush=True)
    prepared = prepare_variant(base, bag, coefficients, frames, artifacts, release, schedule, work,
                               ae_light=args.light, auto_policy=auto_policy)
    print(f'Prepared {len(rgb_pairs(prepared / "rgb.csv"))} exact stereo pairs', flush=True)
    if args.action in ('run', 'demo'):
        run_slam(prepared, args.slam, headless=args.action == 'run', output=work,
                 control_plot=args.control_plot or args.ae_plot)


if __name__ == '__main__':
    main()
