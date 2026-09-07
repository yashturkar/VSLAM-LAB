"""Prepare and run recorded BorealHDR stereo at a fixed 4 ms exposure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from Utilities.extract_lightning_mcap import write_calibration, write_csv  # noqa: E402

DEFAULT_DATA = Path('/mnt/share/local/eph/BorealHDR')
DEFAULT_CODE = Path('/home/yashturkar/Workspace/TFR24_BorealHDR/BorealHDR')
DEFAULT_OUTPUT = Path('/mnt/share/local/eph/VSLAM/borealhdr')


def discover(root):
    return sorted(p.name for p in root.glob('backpack_*')
                  if (p / 'camera_left/4.0').is_dir() and (p / 'camera_right/4.0').is_dir())


def prepare(root, name, code, output):
    if name not in discover(root):
        raise ValueError(f'Unknown or unavailable sequence: {name}')
    source = (root / name).resolve()
    output = output.resolve()
    if output.is_relative_to(source) or output.is_relative_to(Path('/mnt/share/nas')):
        raise ValueError('Output must be local and outside the source sequence')
    season = 'april_2023' if name.startswith('backpack_2023-04-') else 'september_2023'
    calibration = code / 'calibration_files/calibration_vo' / season
    camera_configs = [yaml.safe_load((calibration / f'{side}.yaml').read_text()) for side in ('left', 'right')]
    left = {int(p.stem): p for p in (source / 'camera_left/4.0').glob('*.png')}
    right = {int(p.stem): p for p in (source / 'camera_right/4.0').glob('*.png')}
    stamps = sorted(left.keys() & right.keys())
    if len(stamps) < 3:
        raise ValueError('Fewer than three exactly synchronized stereo pairs')
    fingerprint = hashlib.sha256(json.dumps({
        'version': 1, 'source': str(source), 'calibration': camera_configs,
        'files': [(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                  for stamp in stamps for p in (left[stamp], right[stamp])],
    }, sort_keys=True).encode()).hexdigest()
    sequence = output / 'sequences' / name
    marker = sequence / 'borealhdr.json'
    if marker.is_file():
        metadata = json.loads(marker.read_text())
        if metadata.get('fingerprint') == fingerprint and all(
            (sequence / folder / f'{stamp}.png').is_file()
            for folder in ('rgb_0', 'rgb_1') for stamp in stamps
        ) and all((sequence / file).is_file() for file in ('rgb.csv', 'calibration.yaml')):
            print(f'Preparation: SKIP (verified) {sequence}', flush=True)
            return sequence
    first = cv2.imread(str(left[stamps[0]]), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise ValueError('Cannot decode first image')
    height, width = first.shape[:2]
    size = (width, height)
    projection = np.array(camera_configs[0]['projection_matrix']['data']).reshape(3, 4)
    right_projection = np.array(camera_configs[1]['projection_matrix']['data']).reshape(3, 4)
    baseline = -right_projection[0, 3] / right_projection[0, 0]
    if baseline <= 0:
        raise ValueError('Invalid stereo baseline')
    projection[:2] *= np.array([width / camera_configs[0]['image_width'], height / camera_configs[0]['image_height']])[:, None]
    maps = []
    for config in camera_configs:
        intrinsic = np.array(config['camera_matrix']['data']).reshape(3, 3)
        intrinsic[:2] *= np.array([width / config['image_width'], height / config['image_height']])[:, None]
        maps.append(cv2.initUndistortRectifyMap(
            intrinsic, np.array(config['distortion_coefficients']['data']),
            np.array(config['rectification_matrix']['data']).reshape(3, 3),
            projection[:, :3], size, cv2.CV_16SC2))
    for folder in ('rgb_0', 'rgb_1'):
        (sequence / folder).mkdir(parents=True, exist_ok=True)
    for i, stamp in enumerate(stamps):
        for index, path in enumerate((left[stamp], right[stamp])):
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None or raw.shape != (height, width) or raw.dtype != np.uint16:
                raise ValueError(f'Expected 12-bit Bayer in uint16 PNG: {path}')
            gray = cv2.cvtColor(raw, cv2.COLOR_BAYER_RG2GRAY)
            gray = np.clip(gray / 16.0, 0, 255).astype(np.uint8)
            rectified = cv2.remap(gray, *maps[index], interpolation=cv2.INTER_LINEAR)
            if not cv2.imwrite(str(sequence / f'rgb_{index}' / f'{stamp}.png'), rectified):
                raise OSError(f'Cannot save rectified frame {stamp}')
        if i % 50 == 0:
            print(f'Prepare: {i + 1}/{len(stamps)} pairs', flush=True)
    fps = float(1e9 / np.median(np.diff(np.array(stamps, dtype=np.int64))))
    write_csv(sequence / 'rgb.csv', ['ts_rgb_0 (ns)', 'path_rgb_0', 'ts_rgb_1 (ns)', 'path_rgb_1'],
              [[stamp, f'rgb_0/{stamp}.png', stamp, f'rgb_1/{stamp}.png'] for stamp in stamps])
    write_calibration(sequence / 'calibration.yaml', projection, baseline, fps, size)
    marker.write_text(json.dumps({'fingerprint': fingerprint, 'source': str(source),
        'calibration': str(calibration), 'exposure_ms': 4, 'pairs': len(stamps),
        'unmatched_left': len(left)-len(stamps), 'unmatched_right': len(right)-len(stamps),
        'fps': fps, 'groundtruth': 'not supplied'}, indent=2))
    return sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['list', 'prepare', 'run', 'demo'])
    parser.add_argument('sequence', nargs='?')
    parser.add_argument('--slam', default='orbslam2')
    parser.add_argument('--mode', choices=['mono', 'stereo'], default='stereo')
    parser.add_argument('--root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--code', type=Path, default=DEFAULT_CODE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    args.code = args.code.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.action == 'list':
        print('\n'.join(discover(args.root)))
        return
    if not args.sequence:
        parser.error('A sequence name is required')
    if args.action == 'demo' and sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        parser.error('Demo requires an X display; run from a desktop terminal or use run for headless SLAM')
    from Baselines.get_baseline import get_baseline, list_available_baselines
    from Run.single_sequence import run_single_baseline
    if args.slam not in list_available_baselines() or args.mode not in get_baseline(args.slam).modes:
        parser.error('Choose a registered SLAM with support for the requested camera mode')
    sequence = prepare(args.root, args.sequence, args.code, args.output)
    if args.action == 'prepare':
        return
    run = args.output / 'results' / args.sequence / (datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '_' + args.slam)
    run.mkdir(parents=True)
    config = run / 'config.yaml'
    config.write_text(yaml.safe_dump({'DATASET': {'base_path': str(sequence.parent), 'name': args.sequence,
        'dataset': 'borealhdr', 'baseline': args.slam, 'sensor_type': args.mode, 'output_dir': str(run.resolve())}}))
    print(f'Configuration: {config}', flush=True)
    state = {'sequence': args.sequence, 'baseline': args.slam, 'mode': args.mode,
             'exposure_ms': 4, 'source': str(args.root / args.sequence),
             'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
             'groundtruth': 'not supplied', 'config': str(config)}
    state_path = run / 'run.json'
    state_path.write_text(json.dumps(state, indent=2))
    try:
        trajectory = run_single_baseline(config, headless=args.action == 'run')
        from Evaluate.pairwise_metrics import read_pose_trajectory, write_combined_report
        poses = read_pose_trajectory(trajectory)
        import pandas as pd
        input_stamps = pd.read_csv(sequence / 'rgb.csv').iloc[:, 0].to_numpy(dtype=np.int64)
        input_duration = float((input_stamps[-1] - input_stamps[0]) / 1e9)
        trajectory_duration = float(poses.timestamps[-1] - poses.timestamps[0])
        state.update(input_pairs=len(input_stamps), input_duration_s=input_duration,
                     trajectory_duration_s=trajectory_duration)
        print(f'Saved {poses.num_poses} poses spanning {trajectory_duration:.1f}s; '
              f'input spans {input_duration:.1f}s. Keyframes are sparse; check temporal coverage.', flush=True)
        write_combined_report({'vslam': poses}, run / 'trajectory_report.pdf',
                              f'{args.slam}: {args.sequence} (no reference)')
        state.update(status='complete', trajectory=str(trajectory), poses=poses.num_poses)
    except Exception as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        state['finished_at'] = datetime.now(timezone.utc).isoformat()
        state_path.write_text(json.dumps(state, indent=2))
    print(f'Trajectory: {trajectory}', flush=True)
    print('Reference poses are not supplied; no ground-truth metrics computed.', flush=True)


if __name__ == '__main__':
    main()
