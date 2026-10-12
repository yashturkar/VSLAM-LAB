"""Prepare and run BorealHDR stereo with fixed, scheduled, or AE50 exposure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import select
import subprocess
import sys
from bisect import bisect_left
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
AE50_SOURCE = 'norlab-ulaval/BorealHDR@042befac24efb256ab2a3c787c758a88a490c290:Metric_Classical'
AE50_TARGET = 2048
AE50_GAIN = 0.002
AE50_THRESHOLD = 15


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
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError(f'Exposure at frame {frame} must be finite and nonnegative (milliseconds)')
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
    selected = [source_bracket(exposure) for exposure in exposures]
    tables = {bracket: stereo_table(source, bracket) for bracket in set(selected)}
    if any(len(table) != len(exposures) for table in tables.values()):
        raise ValueError('Exposure brackets have unequal cycle counts; refusing ambiguous frame indexing')
    rows = [(i, exposures[i], bracket, *tables[bracket][i]) for i, bracket in enumerate(selected)]
    if any(a[3] >= b[3] for a, b in zip(rows, rows[1:])):
        raise ValueError('Selected exposure timestamps are not strictly increasing')
    return rows


def source_bracket(exposure):
    """Select the nearest log bracket; zero is rendered from the shortest one."""
    if exposure == 0:
        return BRACKETS[0]
    return min(BRACKETS, key=lambda b: abs(math.log(b / exposure)))


def ae50_source_bracket(exposure, image_for_bracket):
    """Pinned BorealHDR closer_least_sat selection on one camera's Bayer images."""
    higher = next((bracket for bracket in BRACKETS if bracket >= exposure), None)
    if higher is None:
        return BRACKETS[-1]
    index = BRACKETS.index(higher)
    if index:
        previous = BRACKETS[index - 1]

        def saturation(bracket):
            image = image_for_bracket(bracket)
            return (np.count_nonzero(image == 0) + np.count_nonzero(image == 4094)) / image.size

        if saturation(higher) > saturation(previous):
            return previous
    return higher


def ae50_next_exposure(bayer, exposure):
    """Exact Metric_Classical(50) update from the pinned BorealHDR revision."""
    gray = cv2.cvtColor(bayer, cv2.COLOR_BAYER_RG2GRAY)
    mean = float(np.mean(gray))
    error = mean - AE50_TARGET
    if abs(error) < AE50_THRESHOLD:
        return mean, exposure
    if error < 0:
        return mean, float(np.abs(exposure + AE50_GAIN * np.abs(error)))
    return mean, float(np.abs(exposure - AE50_GAIN * np.abs(error)))


def ae50_tables(source, reference):
    """Align all brackets to 4 ms capture cycles; omit incomplete cycles."""
    tables = {bracket: stereo_table(source, bracket) for bracket in BRACKETS}
    if any(not table for table in tables.values()):
        raise ValueError('AE50 requires paired images in every exposure bracket')
    period = float(np.median(np.diff([row[0] for row in reference])))
    tolerance = period / 4
    offsets = {bracket: tables[bracket][0][0] - reference[0][0] for bracket in BRACKETS}
    times = {bracket: [row[0] for row in table] for bracket, table in tables.items()}
    aligned = {bracket: [] for bracket in BRACKETS}
    for stamp, _, _ in reference:
        matches = {}
        for bracket in BRACKETS:
            expected = stamp + offsets[bracket]
            index = bisect_left(times[bracket], expected)
            candidates = [i for i in (index - 1, index) if 0 <= i < len(times[bracket])]
            if not candidates:
                break
            closest = min(candidates, key=lambda i: abs(times[bracket][i] - expected))
            if abs(times[bracket][closest] - expected) > tolerance:
                break
            matches[bracket] = tables[bracket][closest]
        if len(matches) == len(BRACKETS):
            for bracket, row in matches.items():
                aligned[bracket].append(row)
    dropped = len(reference) - len(aligned[4.])
    if dropped:
        print(f'AE50: excluding {dropped} incomplete exposure cycles', flush=True)
    if len(aligned[4.]) < 3:
        raise ValueError('AE50 found fewer than three complete exposure cycles')
    return aligned


def response_curve(code, *, ae=False):
    # RESEARCH PROVENANCE — BorealHDR AE50 comparison:
    # BorealHDR@042befac's Image_Emulator hard-codes pcalib_inside1.txt and uses
    # closer_least_sat bracket selection independently for each camera. Our first
    # AE implementation reused this harness's forest2024 CRF and nearest-log
    # bracket selection. It reproduced Metric_Classical's controller equation,
    # but NOT its upstream image inputs; old AE runs are not exact emulator
    # comparisons. Use the pinned CRF for --ae, retaining forest2024 only for
    # fixed/YAML schedules so their existing behavior does not silently change.
    filename = 'pcalib_inside1.txt' if ae else 'pcalib_forest2024.txt'
    values = np.loadtxt(code / 'calibration_files' / filename) * 16.
    values[0], values[-1] = 0., 4095.
    if values.shape != (256,) or not np.isfinite(values).all() or (np.diff(values) <= 0).any():
        raise ValueError('Invalid BorealHDR camera response calibration')
    return values


def convert_exposure(gray, target, source, curve, *, force_curve=False):
    if target == source and not force_curve:
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
    keys = ('exposure_ms', 'source_bracket_ms')
    if rows and 'right_source_bracket_ms' in rows[0]:
        keys += ('right_source_bracket_ms',)
    for end in range(1, len(rows) + 1):
        first = rows[start]
        if end < len(rows) and all(rows[end][key] == first[key] for key in keys):
            continue
        bracket_note = (f"left source bracket {float(first['source_bracket_ms']):g} ms; "
                        f"right source bracket {float(first['right_source_bracket_ms']):g} ms"
                        if 'right_source_bracket_ms' in first else
                        f"source bracket {float(first['source_bracket_ms']):g} ms")
        print(f"  Frames {first['frame_num']}–{rows[end - 1]['frame_num']}: "
              f"left = right = {float(first['exposure_ms']):g} ms "
              f"({bracket_note})", flush=True)
        start = end


def discover(root):
    return sorted(p.name for p in root.glob('backpack_*')
                  if (p / 'camera_left/4.0').is_dir() and (p / 'camera_right/4.0').is_dir())


def prepare(root, name, code, output, exposure_yaml=None, ae=False):
    if ae and exposure_yaml is not None:
        raise ValueError('--ae and --exposure-yaml are mutually exclusive')
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
    unmatched_left = len(left) - len(stamps)
    unmatched_right = len(right) - len(stamps)
    if len(stamps) < 3:
        raise ValueError('Fewer than three exactly synchronized stereo pairs')
    original_cycles = len(stamps)
    exposures = None if ae else load_exposures(exposure_yaml, len(stamps))
    tables = ae50_tables(source, [(stamp, left[stamp], right[stamp]) for stamp in stamps]) if ae else None
    if ae:
        stamps = [row[0] for row in tables[4.]]
    rows = None if ae else (exposure_rows(source, exposures) if exposure_yaml else [
        (i, 4., 4., stamp, left[stamp], right[stamp]) for i, stamp in enumerate(stamps)])
    if rows is not None:
        stamps = [row[3] for row in rows]
        left = {row[3]: row[4] for row in rows}
        right = {row[3]: row[5] for row in rows}
    needs_curve = ae or any(row[1] != row[2] for row in rows)
    curve = response_curve(code, ae=ae) if needs_curve else None
    source_files = ([p for bracket in BRACKETS for side in ('left', 'right')
                     for p in sorted((source / f'camera_{side}' / str(bracket)).glob('*.png'))] if ae else
                    [p for stamp in stamps for p in (left[stamp], right[stamp])])
    fingerprint_input = {
        'version': 4 if ae else 2, 'source': str(source), 'calibration': camera_configs,
        'exposures': exposures, 'curve': curve.tolist() if needs_curve else None,
        'files': [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in source_files],
    }
    if ae:
        fingerprint_input['ae'] = {'source': AE50_SOURCE, 'target': AE50_TARGET,
                                   'gain': AE50_GAIN, 'threshold': AE50_THRESHOLD,
                                   'initial_ms': 4., 'selection': 'closer_least_sat_per_camera',
                                   'response_curve': 'pcalib_inside1.txt'}
    fingerprint = hashlib.sha256(json.dumps(fingerprint_input, sort_keys=True).encode()).hexdigest()
    variant = output / 'exposure_variants' / fingerprint if exposure_yaml or ae else output
    sequence = variant / 'sequences' / name
    marker = sequence / 'borealhdr.json'
    if marker.is_file():
        metadata = json.loads(marker.read_text())
        cached_files = ('rgb.csv', 'calibration.yaml', 'exposure.csv', 'exposure.yaml', 'ae_trace.csv') if ae else (
            'rgb.csv', 'calibration.yaml', 'exposure.csv')
        if ae and (sequence / 'exposure.csv').is_file():
            with (sequence / 'exposure.csv').open(newline='') as stream:
                cached_stamps = [(int(row['timestamp_ns']), int(row['right_timestamp_ns']))
                                 for row in csv.DictReader(stream)]
        else:
            cached_stamps = [(stamp, stamp) for stamp in stamps]
        if metadata.get('fingerprint') == fingerprint and len(cached_stamps) == len(stamps) and all(
            (sequence / folder / f'{stamp}.png').is_file()
            for left_stamp, right_stamp in cached_stamps
            for folder, stamp in (('rgb_0', left_stamp), ('rgb_1', right_stamp))
        ) and all((sequence / file).is_file() for file in cached_files):
            print(f'Preparation: SKIP (verified) {sequence}', flush=True)
            print_exposure_schedule(sequence)
            return sequence
    first_path = tables[4.][0][1] if ae else left[stamps[0]]
    first = cv2.imread(str(first_path), cv2.IMREAD_UNCHANGED)
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
    ae_trace = []
    if ae:
        rows = []
        current_exposure = 4.
    for i in range(len(stamps)):
        if ae:
            selected = []
            raw_images = []
            for index in (0, 1):
                decoded = {}

                def image_for_bracket(bracket):
                    if bracket not in decoded:
                        path = tables[bracket][i][index + 1]
                        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                        if raw is None or raw.shape != (height, width) or raw.dtype != np.uint16:
                            raise ValueError(f'Expected 12-bit Bayer in uint16 PNG: {path}')
                        decoded[bracket] = raw
                    return decoded[bracket]

                bracket = ae50_source_bracket(current_exposure, image_for_bracket)
                selected.append((bracket, tables[bracket][i][0], tables[bracket][i][index + 1]))
                raw_images.append(image_for_bracket(bracket))
            left_bracket, stamp, left_path = selected[0]
            right_bracket, right_stamp, right_path = selected[1]
            if rows and (rows[-1][3] >= stamp or rows[-1][7] >= right_stamp):
                raise ValueError('Selected exposure timestamps are not strictly increasing')
            rows.append((i, current_exposure, left_bracket, stamp, left_path, right_path,
                         right_bracket, right_stamp))
        else:
            stamp = stamps[i]
        for index, path in enumerate((rows[i][4], rows[i][5])):
            if ae:
                bracket = rows[i][2] if index == 0 else rows[i][6]
                raw = convert_exposure(raw_images[index], rows[i][1], bracket, curve,
                                       force_curve=True)
                if index == 0:
                    mean, next_exposure = ae50_next_exposure(raw, current_exposure)
                gray = cv2.cvtColor(raw, cv2.COLOR_BAYER_RG2GRAY)
            else:
                raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if raw is None or raw.shape != (height, width) or raw.dtype != np.uint16:
                    raise ValueError(f'Expected 12-bit Bayer in uint16 PNG: {path}')
                gray = cv2.cvtColor(raw, cv2.COLOR_BAYER_RG2GRAY)
                gray = convert_exposure(gray, rows[i][1], rows[i][2], curve)
            gray = np.clip(gray / 16.0, 0, 255).astype(np.uint8)
            rectified = cv2.remap(gray, *maps[index], interpolation=cv2.INTER_LINEAR)
            output_stamp = right_stamp if ae and index == 1 else stamp
            if not cv2.imwrite(str(sequence / f'rgb_{index}' / f'{output_stamp}.png'), rectified):
                raise OSError(f'Cannot save rectified frame {output_stamp}')
        if ae:
            ae_trace.append((i, stamp, current_exposure,
                             math.log2(current_exposure / 4.) if current_exposure > 0 else '',
                             mean, mean - AE50_TARGET, next_exposure, left_bracket, right_bracket))
            current_exposure = next_exposure
        if i % 50 == 0 or i == len(stamps) - 1 or rows[i][1:3] != rows[i - 1][1:3]:
            bracket_note = (f'left source bracket {rows[i][2]:g} ms; '
                            f'right source bracket {rows[i][6]:g} ms' if ae else
                            f'source bracket {rows[i][2]:g} ms')
            print(f'Prepare: {i + 1}/{len(stamps)} pairs | frame {i} | '
                  f'left = right = {rows[i][1]:g} ms '
                  f'({bracket_note})', flush=True)
    if ae:
        stamps = [row[3] for row in rows]
    fps = float(1e9 / np.median(np.diff(np.array(stamps, dtype=np.int64))))
    write_csv(sequence / 'rgb.csv', ['ts_rgb_0 (ns)', 'path_rgb_0', 'ts_rgb_1 (ns)', 'path_rgb_1'],
              [[stamp, f'rgb_0/{stamp}.png', rows[i][7] if ae else stamp,
                f'rgb_1/{rows[i][7] if ae else stamp}.png'] for i, stamp in enumerate(stamps)])
    write_calibration(sequence / 'calibration.yaml', projection, baseline, fps, size)
    exposure_columns = ['frame_num', 'exposure_ms', 'source_bracket_ms', 'timestamp_ns',
                        'left_source', 'right_source']
    if ae:
        exposure_columns += ['right_source_bracket_ms', 'right_timestamp_ns']
    write_csv(sequence / 'exposure.csv', exposure_columns, rows)
    if exposure_yaml or ae:
        (sequence / 'exposure.yaml').write_text(yaml.safe_dump({i: row[1] for i, row in enumerate(rows)}))
    if ae:
        write_csv(sequence / 'ae_trace.csv', ['frame_num', 'timestamp_ns', 'exposure_ms', 'stops_vs_4ms',
                  'mean_dn', 'error_dn', 'next_exposure_ms', 'source_bracket_ms',
                  'right_source_bracket_ms'], ae_trace)
    marker.write_text(json.dumps({'fingerprint': fingerprint, 'source': str(source),
        'calibration': str(calibration), 'exposure_ms': None if exposure_yaml or ae else 4,
        'exposure_mode': 'ae50' if ae else ('scheduled' if exposure_yaml else 'fixed'),
        'ae_controller': AE50_SOURCE if ae else None, 'pairs': len(stamps),
        'ae_response_curve': 'pcalib_inside1.txt' if ae else None,
        'ae_bracket_selection': 'closer_least_sat_per_camera' if ae else None,
        'unmatched_left': unmatched_left, 'unmatched_right': unmatched_right,
        'incomplete_cycles': original_cycles - len(stamps) if ae else 0,
        'fps': fps, 'groundtruth': 'not supplied'}, indent=2))
    return sequence


def launch_ae_plot(sequence):
    """Wait until the frame-open watcher is ready before SLAM can read images."""
    script = Path(__file__).with_name('boreal_ae_plot.py')
    process = subprocess.Popen([sys.executable, str(script), str(sequence)],
                               stdout=subprocess.PIPE, text=True)
    ready, _, _ = select.select([process.stdout], [], [], 10)
    if not ready or process.stdout.readline().strip() != 'READY':
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3)
        raise RuntimeError('AE plot did not start; check the graphical display')
    process.stdout.close()
    print('AE exposure plot: watching prepared left-frame reads', flush=True)
    return process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['list', 'prepare', 'run', 'demo'])
    parser.add_argument('sequence', nargs='?')
    parser.add_argument('--slam', default='orbslam2')
    parser.add_argument('--mode', choices=['mono', 'stereo'], default='stereo')
    parser.add_argument('--root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--code', type=Path, default=DEFAULT_CODE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--reference-root', type=Path,
                        default=Path('/mnt/share/nas/eph/BorealHDR-references/fastlio-v1'),
                        help='Read-only shared FAST-LIO reference bundles')
    parser.add_argument('--shared-only', action='store_true',
                        help='Require a validated shared reference; never generate FAST-LIO locally')
    parser.add_argument('--exposure-yaml', type=Path, help='Zero-based frame: exposure_ms mapping; hold until next entry')
    parser.add_argument('--ae', action='store_true', help='Use BorealHDR classical-50 auto-exposure')
    parser.add_argument('--ae-plot', action='store_true',
                        help='Show frame-synchronized exposure plot alongside demo --ae')
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    args.code = args.code.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.reference_root = args.reference_root.expanduser().resolve()
    if args.ae and args.exposure_yaml:
        parser.error('--ae and --exposure-yaml are mutually exclusive')
    if args.ae_plot and (args.action != 'demo' or not args.ae):
        parser.error('--ae-plot requires demo --ae')
    if args.action == 'list':
        print('\n'.join(discover(args.root)))
        return
    if not args.sequence:
        parser.error('A sequence name is required')
    if args.action == 'demo' and sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        parser.error('Demo requires an X display; run from a desktop terminal or use run for headless SLAM')
    from Baselines.get_baseline import get_baseline, list_available_baselines
    from Run.single_sequence import evaluate_single_trajectory, run_single_baseline
    if args.slam not in list_available_baselines() or args.mode not in get_baseline(args.slam).modes:
        parser.error('Choose a registered SLAM with support for the requested camera mode')
    from Utilities.boreal_reference import ensure_full_reference, materialize_camera_reference
    reference_trajectory = None
    if args.shared_only and args.action in ('run', 'demo'):
        reference_trajectory = ensure_full_reference(
            args.root / args.sequence, args.output / 'fastlio_reference',
            args.reference_root, shared_only=True)
    sequence = prepare(args.root, args.sequence, args.code, args.output, args.exposure_yaml,
                       ae=True) if args.ae else prepare(args.root, args.sequence, args.code, args.output, args.exposure_yaml)
    if args.action == 'prepare':
        return
    season = 'april_2023' if args.sequence.startswith('backpack_2023-04-') else 'september_2023'
    left_calibration = args.code / 'calibration_files/calibration_vo' / season / 'left.yaml'
    if reference_trajectory is None:
        reference_trajectory = ensure_full_reference(
            args.root / args.sequence, args.output / 'fastlio_reference', args.reference_root)
    reference = materialize_camera_reference(reference_trajectory, sequence, left_calibration)
    run = args.output / 'results' / args.sequence / (datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '_' + args.slam)
    run.mkdir(parents=True)
    if args.exposure_yaml or args.ae:
        import shutil
        filenames = ('exposure.yaml', 'exposure.csv', 'borealhdr.json', 'ae_trace.csv') if args.ae else (
            'exposure.yaml', 'exposure.csv', 'borealhdr.json')
        for filename in filenames:
            shutil.copy2(sequence / filename, run / filename)
    config = run / 'config.yaml'
    config.write_text(yaml.safe_dump({'DATASET': {'base_path': str(sequence.parent), 'name': args.sequence,
        'dataset': 'borealhdr', 'baseline': args.slam, 'sensor_type': args.mode, 'output_dir': str(run.resolve())},
        'EVALUATION': {'max_time_difference_s': 0.02, 'reference_name': 'boreal_fast_lio',
        'reference_metadata': reference,
        'warnings': ['Boreal FAST-LIO is an experimental lidar/IMU reference, not surveyed ground truth.']}}))
    print(f'Configuration: {config}', flush=True)
    state = {'sequence': args.sequence, 'baseline': args.slam, 'mode': args.mode,
             'exposure_ms': None if args.exposure_yaml or args.ae else 4,
             'exposure_mode': 'ae50' if args.ae else ('scheduled' if args.exposure_yaml else 'fixed'),
             'ae_controller': AE50_SOURCE if args.ae else None,
             'ae_response_curve': 'pcalib_inside1.txt' if args.ae else None,
             'ae_bracket_selection': 'closer_least_sat_per_camera' if args.ae else None,
             'prepared_sequence': str(sequence), 'source': str(args.root / args.sequence),
             'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
             'groundtruth': 'boreal_fast_lio', 'reference_trajectory': str(reference_trajectory),
             'reference_frames': reference['reference_frames'],
             'reference_coverage': reference['reference_coverage'], 'config': str(config)}
    state_path = run / 'run.json'
    state_path.write_text(json.dumps(state, indent=2))
    plotter = None
    try:
        if args.ae_plot:
            plotter = launch_ae_plot(sequence)
        trajectory = run_single_baseline(config, headless=args.action == 'run')
        from Evaluate.pairwise_metrics import read_pose_trajectory
        try:
            poses = read_pose_trajectory(trajectory)
        except ValueError as error:
            log = run / '.vslamlab_run' / args.sequence / 'system_output_00000.txt'
            raise RuntimeError(f'{args.slam} produced no evaluable trajectory: {error}; '
                               f'check SLAM log: {log}') from error
        import pandas as pd
        input_stamps = pd.read_csv(sequence / 'rgb.csv').iloc[:, 0].to_numpy(dtype=np.int64)
        input_duration = float((input_stamps[-1] - input_stamps[0]) / 1e9)
        trajectory_duration = float(poses.timestamps[-1] - poses.timestamps[0])
        state.update(input_pairs=len(input_stamps), input_duration_s=input_duration,
                     trajectory_duration_s=trajectory_duration)
        print(f'Saved {poses.num_poses} poses spanning {trajectory_duration:.1f}s; '
              f'input spans {input_duration:.1f}s. Keyframes are sparse; check temporal coverage.', flush=True)
        metrics_path = evaluate_single_trajectory(config, trajectory)
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        state.update(status='complete', trajectory=str(trajectory), poses=poses.num_poses,
                     metrics=str(metrics_path), trajectory_report=str(run / 'trajectory_report.pdf'),
                     translation_ape_rmse_m=metrics['rmse']['translation'],
                     temporal_completion_ratio=metrics['temporal_completion_ratio'])
    except Exception as error:
        state.update(status='failed', error=str(error))
        for field in ('exit_code', 'signal_name', 'log_file', 'partial_trajectory', 'pose_rows'):
            if hasattr(error, field):
                value = getattr(error, field)
                state[field] = str(value) if isinstance(value, Path) else value
        raise
    finally:
        if plotter is not None and plotter.poll() is None:
            plotter.terminate()
            try:
                plotter.wait(timeout=3)
            except subprocess.TimeoutExpired:
                plotter.kill()
                plotter.wait()
        state['finished_at'] = datetime.now(timezone.utc).isoformat()
        state_path.write_text(json.dumps(state, indent=2))
    print(f'Trajectory: {trajectory}', flush=True)
    print(f"Boreal FAST-LIO reference: {reference['reference_frames']}/{reference['input_frames']} "
          f"selected frames; translation APE RMSE: {state['translation_ape_rmse_m']:.3f} m; "
          f"temporal completion: {state['temporal_completion_ratio']:.1%}", flush=True)
    print(f"Metrics: {metrics_path}", flush=True)
    print(f"Trajectory report: {run / 'trajectory_report.pdf'}", flush=True)


if __name__ == '__main__':
    main()
