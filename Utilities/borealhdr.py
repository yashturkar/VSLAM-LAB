"""Prepare and run BorealHDR stereo with fixed or scheduled exposure."""

from __future__ import annotations

import argparse
import csv
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
BRACKETS = (1., 2., 4., 8., 16., 32.)


class ExposureLoader(yaml.SafeLoader):
    """Reject duplicate frame keys instead of silently overwriting them."""


def _unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if type(key) is not int or key in result:
            raise ValueError('Exposure keys must be unique integer frame numbers')
        result[key] = loader.construct_object(value_node)
    return result


ExposureLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def load_exposures(path, count):
    if path is None:
        return [4.] * count
    schedule = yaml.load(Path(path).read_text(), Loader=ExposureLoader)
    if not isinstance(schedule, dict) or 0 not in schedule:
        raise ValueError('Exposure YAML must be a frame_num: exposure_ms mapping starting at frame 0')
    for frame, value in schedule.items():
        if not 0 <= frame < count:
            raise ValueError(f'Exposure frame {frame} is outside 0..{count - 1}')
        if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'Exposure at frame {frame} must be finite and positive (milliseconds)')
    result = []
    current = schedule[0]
    for frame in range(count):
        current = schedule.get(frame, current)
        result.append(float(current))
    return result


def stereo_table(source, bracket):
    tables = [{int(p.stem): p for p in (source / f'camera_{side}' / str(float(bracket))).glob('*.png')}
              for side in ('left', 'right')]
    common = tables[0].keys() & tables[1].keys()
    dropped = len(tables[0]) + len(tables[1]) - 2 * len(common)
    if dropped:
        print(f'Bracket {bracket}: excluding {dropped} unpaired camera images', flush=True)
    return [(stamp, tables[0][stamp], tables[1][stamp]) for stamp in sorted(common)]


def exposure_rows(source, exposures):
    # BorealHDR's emulator groups the nth image from each exposure bracket into
    # one cycle. Keep real acquisition timestamps when selecting that bracket.
    selected = [min(BRACKETS, key=lambda b: abs(np.log(b / exposure))) for exposure in exposures]
    tables = {bracket: stereo_table(source, bracket) for bracket in set(selected)}
    if any(len(table) != len(exposures) for table in tables.values()):
        raise ValueError('Exposure brackets have unequal cycle counts; refusing ambiguous frame indexing')
    rows = [(i, exposures[i], bracket, *tables[bracket][i]) for i, bracket in enumerate(selected)]
    if any(a[3] >= b[3] for a, b in zip(rows, rows[1:])):
        raise ValueError('Selected exposure timestamps are not strictly increasing')
    return rows


def response_curve(code):
    values = np.loadtxt(code / 'calibration_files/pcalib_forest2024.txt') * 16.
    values[0], values[-1] = 0., 4095.
    if values.shape != (256,) or not np.isfinite(values).all() or (np.diff(values) <= 0).any():
        raise ValueError('Invalid BorealHDR camera response calibration')
    return values


def convert_exposure(gray, target, source, curve):
    if target == source:
        return gray
    digital = np.linspace(0, 4095, len(curve))
    radiance = np.interp(gray, digital, curve)
    return np.interp(np.clip(radiance * target / source, 0, 4095), curve, digital).astype(np.uint16)


def print_exposure_schedule(sequence):
    """Report cached input exposures, without implying live SLAM progress."""
    with (sequence / 'exposure.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    print('Prepared input exposure schedule (not live SLAM progress):', flush=True)
    start = 0
    for end in range(1, len(rows) + 1):
        first = rows[start]
        if end < len(rows) and all(rows[end][key] == first[key]
                                  for key in ('exposure_ms', 'source_bracket_ms')):
            continue
        print(f"  Frames {first['frame_num']}–{rows[end - 1]['frame_num']}: "
              f"left = right = {float(first['exposure_ms']):g} ms "
              f"(source bracket {float(first['source_bracket_ms']):g} ms)", flush=True)
        start = end


def discover(root):
    return sorted(p.name for p in root.glob('backpack_*')
                  if (p / 'camera_left/4.0').is_dir() and (p / 'camera_right/4.0').is_dir())


def prepare(root, name, code, output, exposure_yaml=None):
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
    exposures = load_exposures(exposure_yaml, len(stamps))
    rows = exposure_rows(source, exposures) if exposure_yaml else [
        (i, 4., 4., stamp, left[stamp], right[stamp]) for i, stamp in enumerate(stamps)]
    stamps = [row[3] for row in rows]
    left = {row[3]: row[4] for row in rows}
    right = {row[3]: row[5] for row in rows}
    needs_curve = any(row[1] != row[2] for row in rows)
    curve = response_curve(code) if needs_curve else None
    fingerprint = hashlib.sha256(json.dumps({
        'version': 2, 'source': str(source), 'calibration': camera_configs,
        'exposures': exposures, 'curve': curve.tolist() if needs_curve else None,
        'files': [(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                  for stamp in stamps for p in (left[stamp], right[stamp])],
    }, sort_keys=True).encode()).hexdigest()
    variant = output / 'exposure_variants' / fingerprint if exposure_yaml else output
    sequence = variant / 'sequences' / name
    marker = sequence / 'borealhdr.json'
    if marker.is_file():
        metadata = json.loads(marker.read_text())
        if metadata.get('fingerprint') == fingerprint and all(
            (sequence / folder / f'{stamp}.png').is_file()
            for folder in ('rgb_0', 'rgb_1') for stamp in stamps
        ) and all((sequence / file).is_file() for file in ('rgb.csv', 'calibration.yaml', 'exposure.csv')):
            print(f'Preparation: SKIP (verified) {sequence}', flush=True)
            print_exposure_schedule(sequence)
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
            gray = convert_exposure(gray, rows[i][1], rows[i][2], curve)
            gray = np.clip(gray / 16.0, 0, 255).astype(np.uint8)
            rectified = cv2.remap(gray, *maps[index], interpolation=cv2.INTER_LINEAR)
            if not cv2.imwrite(str(sequence / f'rgb_{index}' / f'{stamp}.png'), rectified):
                raise OSError(f'Cannot save rectified frame {stamp}')
        if i % 50 == 0 or i == len(stamps) - 1 or rows[i][1:3] != rows[i - 1][1:3]:
            print(f'Prepare: {i + 1}/{len(stamps)} pairs | frame {i} | '
                  f'left = right = {rows[i][1]:g} ms '
                  f'(source bracket {rows[i][2]:g} ms)', flush=True)
    fps = float(1e9 / np.median(np.diff(np.array(stamps, dtype=np.int64))))
    write_csv(sequence / 'rgb.csv', ['ts_rgb_0 (ns)', 'path_rgb_0', 'ts_rgb_1 (ns)', 'path_rgb_1'],
              [[stamp, f'rgb_0/{stamp}.png', stamp, f'rgb_1/{stamp}.png'] for stamp in stamps])
    write_calibration(sequence / 'calibration.yaml', projection, baseline, fps, size)
    write_csv(sequence / 'exposure.csv', ['frame_num', 'exposure_ms', 'source_bracket_ms', 'timestamp_ns', 'left_source', 'right_source'], rows)
    if exposure_yaml:
        (sequence / 'exposure.yaml').write_text(yaml.safe_dump({i: e for i, e in enumerate(exposures)}))
    marker.write_text(json.dumps({'fingerprint': fingerprint, 'source': str(source),
        'calibration': str(calibration), 'exposure_ms': None if exposure_yaml else 4,
        'exposure_mode': 'scheduled' if exposure_yaml else 'fixed', 'pairs': len(stamps),
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
    parser.add_argument('--exposure-yaml', type=Path, help='Zero-based frame: exposure_ms mapping; hold until next entry')
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
    sequence = prepare(args.root, args.sequence, args.code, args.output, args.exposure_yaml)
    if args.action == 'prepare':
        return
    run = args.output / 'results' / args.sequence / (datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '_' + args.slam)
    run.mkdir(parents=True)
    if args.exposure_yaml:
        import shutil
        for filename in ('exposure.yaml', 'exposure.csv', 'borealhdr.json'):
            shutil.copy2(sequence / filename, run / filename)
    config = run / 'config.yaml'
    config.write_text(yaml.safe_dump({'DATASET': {'base_path': str(sequence.parent), 'name': args.sequence,
        'dataset': 'borealhdr', 'baseline': args.slam, 'sensor_type': args.mode, 'output_dir': str(run.resolve())}}))
    print(f'Configuration: {config}', flush=True)
    state = {'sequence': args.sequence, 'baseline': args.slam, 'mode': args.mode,
             'exposure_ms': None if args.exposure_yaml else 4,
             'exposure_mode': 'scheduled' if args.exposure_yaml else 'fixed',
             'prepared_sequence': str(sequence), 'source': str(args.root / args.sequence),
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
