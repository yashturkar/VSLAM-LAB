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
from Utilities.extract_lightning_mcap import (  # noqa: E402
    camera_values, read_camera_info, timestamp_ns, write_calibration, write_csv,
)

DEFAULT_ROOT = Path('/mnt/share/local/eph/VSLAM/clid_stereo')
DEFAULT_ILLUMINET = Path('/home/yashturkar/Workspace/IllumiNet')
DEFAULT_PYTHON = Path('/mnt/share/local/eph/VSLAM/envs/illuminet-v634/bin/python')
DEFAULT_ARTIFACTS = Path('/mnt/share/nas/eph/clid-hdr-releases/clid-hdr-v634-release-2')
SIDES = ('left', 'right')
KEYS = ('A', 'F', 'C1', 'C2')
INFO_TOPICS = ('/cam_sync/cam0/camera_info', '/cam_sync/cam1/camera_info')
RAW_TOPICS = ('/cam_sync/cam0/image_raw', '/cam_sync/cam1/image_raw')
ODOMETRY_TOPIC = '/odometry'
RENDER_VERSION = 1


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
    if mounts:
        filesystem = max(mounts)[1]
        if filesystem.startswith(('nfs', 'cifs', 'smb', 'fuse.sshfs', 'fuse.rclone')):
            raise ValueError(f'Output resolves to a network filesystem ({filesystem}): {resolved}')
    return resolved


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


def render(values: dict[str, np.ndarray], calibration: dict[str, Any], light: float, ev: float) -> np.ndarray:
    response = calibration['power_response']
    q = np.float32(np.interp(light, response['commanded'], response['effective']))
    a, f, c1, c2 = (values[key].astype(np.float32) for key in KEYS)
    linear = (a + q * f + q * (1 - q) * c1 + q * (1 - q) * (2 * q - 1) * c2) * np.float32(2.0 ** ev)
    site = np.clip(linear, 0, 1)
    alpha = np.asarray([calibration['response']['bayer_alpha'][key] for key in ('R', 'G1', 'G2', 'B')], np.float32)
    site = np.power(site, alpha[:, None, None])
    mosaic = np.empty((800, 1280), np.float32)
    mosaic[0::2, 0::2], mosaic[0::2, 1::2] = site[0], site[1]
    mosaic[1::2, 0::2], mosaic[1::2, 1::2] = site[2], site[3]
    bayer = np.round(mosaic * 65535).astype(np.uint16)
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


def prepare_variant(base: Path, bag: Path, coefficients: Path, frames: dict[str, dict[int, dict[str, Any]]],
                    artifacts: Path, release: dict[str, Any], schedule: Path, output: Path) -> Path:
    stamps = [stamp for stamp in paired_timestamps(base / 'pairing.csv') if stamp in frames['left'] and stamp in frames['right']]
    if len(stamps) < 3:
        raise ValueError('Fewer than three exact preview/left-RAW/right-RAW timestamp matches')
    dense = load_schedule(schedule, len(stamps))
    calibrations = {side: json.loads((artifacts / release['cameras'][side]['calibration']).read_text()) for side in SIDES}
    fingerprint = object_fingerprint({
        'renderer_version': RENDER_VERSION, 'bag': str(bag),
        'coefficients_manifest': sha256_file(coefficients / 'stereo-manifest.json'),
        'base_metadata': sha256_file(base / 'extraction_metadata.json'),
        'calibrations': {side: sha256_file(artifacts / release['cameras'][side]['calibration']) for side in SIDES},
        'schedule': dense,
    })
    destination = output / 'variants' / fingerprint / 'dataset' / base.name
    marker = destination / 'relight.json'
    if marker.is_file():
        previous = json.loads(marker.read_text())
        if previous.get('status') == 'complete' and previous.get('fingerprint') == fingerprint:
            inventory = previous.get('image_bytes', {})
            support = ('rgb.csv', 'calibration.yaml', 'groundtruth.csv', 'relight.csv')
            if all((destination / name).is_file() for name in support) and len(inventory) == 2 * len(stamps) and all(
                (destination / path).is_file() and (destination / path).stat().st_size == size
                for path, size in inventory.items()
            ) and rgb_pairs(destination / 'rgb.csv') == stamps:
                print(f'Relight: SKIP (verified) {destination}', flush=True)
                return destination
    maps, projection, baseline = stereo_maps(bag, (1280, 800))
    metadata = json.loads((base / 'extraction_metadata.json').read_text())
    for side in SIDES:
        (destination / ('rgb_0' if side == 'left' else 'rgb_1')).mkdir(parents=True, exist_ok=True)
    rows = []
    trace = []
    image_bytes = {}
    for index, (stamp, setting) in enumerate(zip(stamps, dense)):
        row = [stamp, f'rgb_0/{stamp}.png', stamp, f'rgb_1/{stamp}.png']
        for side, relative in zip(SIDES, (row[1], row[3])):
            info = frames[side][stamp]
            packed = read_component(info['file'], info['sha256'])
            image = cv2.remap(render(packed, calibrations[side], setting['light'], setting['exposure_ev']),
                              *maps[side], interpolation=cv2.INTER_LINEAR)
            target = destination / relative
            temporary = target.with_name(f'{target.stem}.tmp.png')
            if not cv2.imwrite(str(temporary), image):
                raise OSError(f'Could not write {temporary}')
            temporary.replace(target)
            image_bytes[relative] = target.stat().st_size
        rows.append(row)
        trace.append([index, stamp, setting['light'], setting['exposure_ev'],
                      float(np.interp(setting['light'], calibrations['left']['power_response']['commanded'], calibrations['left']['power_response']['effective'])),
                      float(np.interp(setting['light'], calibrations['right']['power_response']['commanded'], calibrations['right']['power_response']['effective']))])
        if index == 0 or (index + 1) % 25 == 0 or index == len(stamps) - 1:
            print(f'Relight: {index + 1}/{len(stamps)} pairs ({100 * (index + 1) / len(stamps):.1f}%) | light={setting["light"]:g}, EV={setting["exposure_ev"]:+g}', flush=True)
    write_csv(destination / 'rgb.csv', ['ts_rgb_0 (ns)', 'path_rgb_0', 'ts_rgb_1 (ns)', 'path_rgb_1'], rows)
    write_csv(destination / 'relight.csv', ['frame_num', 'timestamp_ns', 'light_commanded', 'exposure_ev', 'left_effective_power', 'right_effective_power'], trace)
    shutil.copy2(base / 'groundtruth.csv', destination / 'groundtruth.csv')
    write_calibration(destination / 'calibration.yaml', projection, baseline, float(metadata['fps']), (1280, 800))
    atomic_json(marker, {
        'schema_version': 1, 'status': 'complete', 'fingerprint': fingerprint,
        'source_bag': str(bag), 'base_sequence': str(base), 'coefficients': str(coefficients),
        'schedule_yaml': str(schedule), 'frame_count': len(stamps), 'baseline_m': baseline,
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
    parser.add_argument('action', choices=('check', 'status', 'prepare', 'run', 'demo'))
    parser.add_argument('source', type=Path, help='Processed CLID sequence root with manifest.json and research-bag/*.mcap')
    parser.add_argument('--schedule-yaml', type=Path)
    parser.add_argument('--local-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--illuminet', type=Path, default=DEFAULT_ILLUMINET)
    parser.add_argument('--illuminet-python', type=Path, default=DEFAULT_PYTHON)
    parser.add_argument('--artifacts', type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument('--slam', default='orbslam3')
    parser.add_argument('--control-plot', action='store_true',
                        help='Show live commanded light and exposure EV during demo')
    args = parser.parse_args()
    if args.control_plot and args.action != 'demo':
        parser.error('--control-plot requires demo')
    source = args.source.expanduser().resolve()
    bag = source_bag(source)
    output = local_destination(args.local_root, source)
    identity = source_identity(source, bag)
    key = object_fingerprint(identity)[:12]
    work = local_destination(output / f'{source.name}_{key}', source)
    base = local_destination(work / 'base' / source.name, source)
    coefficients = local_destination(work / 'coefficients', source)
    review = local_destination(work / 'review', source)
    illuminet = args.illuminet.expanduser().resolve()
    python = args.illuminet_python.expanduser().resolve()
    artifacts = args.artifacts.expanduser().resolve()
    manifest_path = illuminet / 'releases/clid-hdr-v634-release-2/manifest.json'
    release = json.loads(manifest_path.read_text())
    if not python.is_file() or not (illuminet / 'scripts/infer_clid_hdr_release.py').is_file():
        raise FileNotFoundError('IllumiNet release-2 code or Python runtime is missing')
    inference_env = os.environ.copy()
    inference_env['PATH'] = str(python.parent) + os.pathsep + inference_env.get('PATH', '')
    if shutil.which('ffmpeg', path=inference_env['PATH']) is None:
        raise FileNotFoundError(f'ffmpeg is unavailable in the inference runtime PATH: {python.parent}')
    if not artifacts.is_dir():
        raise FileNotFoundError(artifacts)
    for name, digest in release['sha256'].items():
        if sha256_file(artifacts / name) != digest:
            raise ValueError(f'Release artifact checksum mismatch: {name}')
    frames = validate_coefficients(coefficients, release, identity, sha256_file(manifest_path))
    coefficients_hash = sha256_file(coefficients / 'stereo-manifest.json') if frames else None
    base_stamps = validate_base(base, identity, coefficients_hash) if frames else None
    state = {'source': str(source), 'bag': str(bag), 'work': str(work),
             'raw_index': 'ready' if base_stamps else ('partial/stale' if base.exists() else 'missing'),
             'coefficients': 'ready' if frames else ('partial/stale' if coefficients.exists() else 'missing'),
             'exact_pairs': len(set(base_stamps or []) & set(frames['left']) & set(frames['right'])) if frames else 0}
    print(json.dumps(state, indent=2), flush=True)
    if args.action in ('check', 'status'):
        return
    if args.schedule_yaml is None:
        parser.error('--schedule-yaml is required for prepare/run/demo')
    schedule = args.schedule_yaml.expanduser().resolve()
    if not schedule.is_file():
        raise FileNotFoundError(schedule)
    work.mkdir(parents=True, exist_ok=True)
    if frames is None:
        if coefficients.exists() or review.exists():
            raise RuntimeError(f'Inference is partial/stale; preserved for inspection: {coefficients}, {review}')
        print('Stereo inference: starting with camera-specific RAW and calibrations', flush=True)
        command = [str(python), str(illuminet / 'scripts/infer_clid_hdr_release.py'),
                   '--manifest', str(manifest_path), '--artifacts', str(artifacts),
                   '--camera', 'both', '--sequence', str(source),
                   '--coefficients-output', str(coefficients), '--output', str(review)]
        subprocess.run(command, cwd=illuminet, env=inference_env, check=True)
        frames = validate_coefficients(coefficients, release, identity, sha256_file(manifest_path))
        if frames is None:
            raise RuntimeError(f'Inference output failed validation: {coefficients}')
        coefficients_hash = sha256_file(coefficients / 'stereo-manifest.json')
    else:
        print(f'Stereo inference: SKIP (verified) {len(frames["left"])} left, {len(frames["right"])} right', flush=True)
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
    prepared = prepare_variant(base, bag, coefficients, frames, artifacts, release, schedule, work)
    print(f'Prepared {len(rgb_pairs(prepared / "rgb.csv"))} exact stereo pairs', flush=True)
    if args.action in ('run', 'demo'):
        run_slam(prepared, args.slam, headless=args.action == 'run', output=work,
                 control_plot=args.control_plot)


if __name__ == '__main__':
    main()
