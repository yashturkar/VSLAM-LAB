"""Read-only BorealHDR capture-cycle, prepared-stereo, and AE trace audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from bisect import bisect_left
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from Utilities.boreal_ae_controllers import KimController, ZhangController
from Utilities.borealhdr import (BRACKETS, DEFAULT_CODE, DEFAULT_DATA, DEFAULT_OUTPUT,
                                 ae50_next_exposure, convert_exposure, discover, response_curve)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def raw_cycles(source: Path):
    """Independently assign raw stereo pairs to 4 ms cycles by capture phase."""
    errors = []
    tables = {}
    unpaired = {}
    for bracket in BRACKETS:
        sides = [{int(p.stem) for p in (source / f'camera_{side}' / str(bracket)).glob('*.png')}
                 for side in ('left', 'right')]
        tables[bracket] = sorted(sides[0] & sides[1])
        unpaired[str(bracket)] = len(sides[0] ^ sides[1])
        if not tables[bracket]:
            errors.append(f'{bracket:g} ms has no paired stereo images')
    reference = tables[4.]
    if len(reference) < 3:
        errors.append('4 ms has fewer than three stereo pairs')
        return {'sequence': source.name, 'status': 'fail', 'errors': errors}, {}
    period = int(round(float(np.median(np.diff(reference)))))
    if period <= 0:
        errors.append('nonpositive 4 ms capture period')
        return {'sequence': source.name, 'status': 'fail', 'errors': errors}, {}
    tolerance = period / 4
    mappings = {bracket: {} for bracket in BRACKETS}
    phases = {}
    residuals = []
    complete = 0
    for bracket, stamps in tables.items():
        if stamps:
            # A median over the first few samples is less sensitive to timestamp
            # jitter than anchoring phase to one image or assuming frame zero exists.
            phases[bracket] = (0 if bracket == 4. else int(round(float(np.median(
                [(stamp - reference[0]) % period for stamp in stamps[:min(10, len(stamps))]])))))
    for cycle, reference_stamp in enumerate(reference):
        found = {}
        for bracket, stamps in tables.items():
            if not stamps:
                continue
            expected = reference_stamp + phases[bracket]
            index = bisect_left(stamps, expected)
            candidates = [j for j in (index - 1, index) if 0 <= j < len(stamps)]
            if not candidates:
                continue
            best = min(candidates, key=lambda j: abs(stamps[j] - expected))
            residual = abs(stamps[best] - expected)
            if residual <= tolerance:
                found[bracket] = stamps[best]
                residuals.append(residual)
        if len(found) == len(BRACKETS):
            complete += 1
        for bracket, stamp in found.items():
            if stamp in mappings[bracket]:
                errors.append(f'{bracket:g} ms image {stamp} assigned to two cycles')
            mappings[bracket][stamp] = cycle
    phase_order = [phases.get(bracket) for bracket in (4., 8., 16., 32., 1., 2.)]
    if None not in phase_order and phase_order != sorted(phase_order):
        errors.append(f'unexpected bracket capture order: {phase_order}')
    warnings_list = []
    if sum(unpaired.values()):
        warnings_list.append(f'{sum(unpaired.values())} images have no exact-timestamp stereo mate')
    if complete != len(reference):
        warnings_list.append(f'{len(reference) - complete} reference cycles lack a complete bracket set')
    result = {
        'sequence': source.name, 'status': 'fail' if errors else 'pass',
        'reference_cycles': len(reference), 'complete_cycles': complete,
        'incomplete_cycles': len(reference) - complete,
        'period_ms': round(period / 1e6, 4),
        'phase_ms': {str(b): round(p / 1e6, 4) for b, p in phases.items()},
        'paired_images': {str(b): len(stamps) for b, stamps in tables.items()},
        'unpaired_images': unpaired,
        'max_phase_residual_ms': round(max(residuals, default=0) / 1e6, 4),
        'errors': errors, 'warnings': warnings_list,
    }
    return result, mappings


def stereo_image_matches(sequence: Path, rows: list[dict], sample_count: int):
    """Measure rectified epipolar evidence, without claiming hardware sync."""
    orb = cv2.ORB_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    samples = []
    for i in np.unique(np.linspace(0, len(rows) - 1, min(sample_count, len(rows)), dtype=int)):
        row = rows[int(i)]
        left = cv2.imread(str(sequence / row['path_rgb_0']), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(sequence / row['path_rgb_1']), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None or left.shape != right.shape:
            samples.append({'frame': int(i), 'error': 'missing image or shape mismatch'})
            continue
        left_points, left_descriptors = orb.detectAndCompute(left, None)
        right_points, right_descriptors = orb.detectAndCompute(right, None)
        if left_descriptors is None or right_descriptors is None:
            samples.append({'frame': int(i), 'good_matches': 0, 'epipolar_matches': 0})
            continue
        pairs = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
        good = [m for pair in pairs if len(pair) == 2 for m, n in [pair]
                if m.distance < 0.75 * n.distance]
        epipolar = []
        vertical = []
        positive_disparity = 0
        for match in good:
            x_left, y_left = left_points[match.queryIdx].pt
            x_right, y_right = right_points[match.trainIdx].pt
            if x_left > x_right:
                positive_disparity += 1
                vertical.append(abs(y_left - y_right))
                if abs(y_left - y_right) < 2:
                    epipolar.append(abs(y_left - y_right))
        samples.append({'frame': int(i), 'good_matches': len(good),
                        'epipolar_matches': len(epipolar),
                        'positive_disparity_fraction': positive_disparity / len(good) if good else None,
                        'median_vertical_px': float(np.median(vertical)) if vertical else None})
    counts = [row.get('epipolar_matches', 0) for row in samples]
    reliable_vertical = [row['median_vertical_px'] for row in samples
                         if row.get('good_matches', 0) >= 30 and row.get('median_vertical_px') is not None]
    return {'samples': samples, 'median_epipolar_matches': float(np.median(counts)) if counts else 0,
            'weak_samples': sum(count < 15 for count in counts),
            'median_vertical_px': float(np.median(reliable_vertical)) if reliable_vertical else None,
            'reliable_geometry_samples': len(reliable_vertical)}


def replay_controller(sequence: Path, exposure: list[dict], trace: list[dict], method: str,
                      code: Path, curve_name: str, frame_limit: int | None):
    curve = response_curve(code, ae=curve_name == 'pcalib_inside1.txt')
    controller = KimController() if method == 'kim' else ZhangController(curve) if method == 'zhang' else None
    failures = []
    max_error = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        for i, (row, recorded) in enumerate(zip(exposure, trace)):
            if frame_limit is not None and i >= frame_limit:
                break
            raw = cv2.imread(row['left_source'], cv2.IMREAD_UNCHANGED)
            if raw is None:
                failures.append(f'frame {i}: left raw image missing')
                continue
            current = float(row['exposure_ms'])
            rendered = convert_exposure(raw, current, float(row['source_bracket_ms']), curve,
                                        force_curve=True)
            if controller is None:
                _, expected = ae50_next_exposure(rendered, current)
            else:
                expected, _, _ = controller.update(rendered, current)
            error = abs(expected - float(recorded['next_exposure_ms']))
            max_error = max(max_error, error)
            # Older traces round the meter to a few decimal places; 0.1 us
            # tolerates that serialization without masking a meaningful step.
            if error > 1e-4:
                failures.append(f'frame {i}: next exposure differs by {error:.6g} ms')
    return {'frames_checked': min(len(trace), frame_limit) if frame_limit is not None else len(trace),
            'frames_available': len(trace), 'max_next_exposure_error_ms': max_error,
            'failures': failures[:20], 'failure_count': len(failures)}


def prepared_variant(sequence: Path, raw_mappings: dict, *, samples: int, replay: bool,
                     replay_frames: int | None = None, code: Path, stereo_runs: set[str]):
    marker = json.loads((sequence / 'borealhdr.json').read_text())
    superseded_reasons = []
    if (sequence.name.startswith('backpack_2023-09-27-') and
            marker.get('rectification_version') != 'september_2023_stereo_fit_v1'):
        superseded_reasons.append('predates fitted September stereo rectification')
    if (marker.get('ae_method') or marker.get('exposure_mode') == 'ae50') and (
            'stereo' in stereo_runs and
            marker.get('ae_bracket_selection') != 'closer_least_sat_shared_lower'):
        superseded_reasons.append('predates shared-bracket stereo AE')
    exposure = csv_rows(sequence / 'exposure.csv')
    rgb = csv_rows(sequence / 'rgb.csv')
    errors = []
    warnings_list = []
    if marker.get('rectification_version') and not (sequence / 'left_rectification.yaml').is_file():
        errors.append('fitted stereo variant lacks left_rectification.yaml for camera reference')
    if not exposure or len(exposure) != len(rgb) or len(exposure) != marker.get('pairs'):
        errors.append('exposure.csv, rgb.csv, and marker pair counts differ')
    mismatch = 0
    cycle_mismatch = 0
    missing_images = 0
    missing_sources = 0
    bad_rows = []
    for i, (row, image_row) in enumerate(zip(exposure, rgb)):
        left_stamp = int(row['timestamp_ns'])
        right_stamp = int(row.get('right_timestamp_ns') or left_stamp)
        left_bracket = float(row['source_bracket_ms'])
        right_bracket = float(row.get('right_source_bracket_ms') or left_bracket)
        if int(row['frame_num']) != i:
            bad_rows.append(i)
        if (left_stamp != int(image_row['ts_rgb_0 (ns)']) or
                right_stamp != int(image_row['ts_rgb_1 (ns)'])):
            bad_rows.append(i)
        if (Path(row['left_source']).stem != str(left_stamp) or
                Path(row['right_source']).stem != str(right_stamp)):
            bad_rows.append(i)
        for side, source, bracket in (
                ('left', Path(row['left_source']), left_bracket),
                ('right', Path(row['right_source']), right_bracket)):
            if source.parent.name != str(bracket) or source.parent.parent.name != f'camera_{side}':
                bad_rows.append(i)
            if not source.is_file():
                missing_sources += 1
        if left_stamp != right_stamp or left_bracket != right_bracket:
            mismatch += 1
        left_cycle = raw_mappings.get(left_bracket, {}).get(left_stamp)
        right_cycle = raw_mappings.get(right_bracket, {}).get(right_stamp)
        if left_cycle is None or left_cycle != right_cycle:
            cycle_mismatch += 1
        for column in ('path_rgb_0', 'path_rgb_1'):
            if not (sequence / image_row[column]).is_file():
                missing_images += 1
    if bad_rows:
        errors.append(f'{len(set(bad_rows))} rows disagree with source/image timestamps or frame numbering')
    if missing_images:
        errors.append(f'{missing_images} prepared image files missing')
    if missing_sources:
        errors.append(f'{missing_sources} raw source image files missing')
    if cycle_mismatch:
        errors.append(f'{cycle_mismatch} rows select images from different or unknown capture cycles')
    shared_policy = marker.get('ae_bracket_selection') == 'closer_least_sat_shared_lower'
    if mismatch and (shared_policy or 'stereo' in stereo_runs):
        errors.append(f'{mismatch} left/right stereo pairs have different timestamps or brackets')
    elif mismatch:
        warnings_list.append(f'{mismatch} per-camera pairs differ; only valid for mono use')
    times = [int(row['ts_rgb_0 (ns)']) for row in rgb]
    if any(a >= b for a, b in zip(times, times[1:])):
        errors.append('left timestamps are not strictly increasing')
    method = marker.get('ae_method') or ('classical' if marker.get('exposure_mode') == 'ae50' else None)
    trace = csv_rows(sequence / 'ae_trace.csv') if method and (sequence / 'ae_trace.csv').is_file() else []
    if method and len(trace) != len(exposure):
        errors.append('AE trace count differs from exposure count')
    for i, (row, recorded) in enumerate(zip(exposure, trace)):
        if int(recorded['frame_num']) != i or abs(float(row['exposure_ms']) - float(recorded['exposure_ms'])) > 1e-6:
            errors.append(f'AE trace frame {i} differs from exposure schedule')
        if i + 1 < len(exposure) and abs(float(recorded['next_exposure_ms']) - float(exposure[i + 1]['exposure_ms'])) > 1e-5:
            errors.append(f'AE trace next exposure differs at frame {i}')
    image_check = stereo_image_matches(sequence, rgb, samples) if not missing_images and rgb and samples else None
    if image_check and image_check['weak_samples']:
        warnings_list.append(f"{image_check['weak_samples']}/{len(image_check['samples'])} image samples have fewer than 15 epipolar matches")
    if image_check and image_check['reliable_geometry_samples'] >= 3 and image_check['median_vertical_px'] > 3:
        errors.append(f"rectified stereo matches have median vertical residual {image_check['median_vertical_px']:.2f} px (>3 px)")
    controller_check = None
    if replay and method and len(trace) == len(exposure):
        curve_name = marker.get('ae_response_curve') or 'pcalib_forest2024.txt'
        try:
            controller_check = replay_controller(sequence, exposure, trace, method, code,
                                                 curve_name, replay_frames)
            if controller_check['failure_count']:
                errors.append(f"{controller_check['failure_count']} controller replay mismatches")
        except (OSError, ValueError, KeyError) as error:
            errors.append(f'controller replay failed: {error}')
    return {'sequence': sequence.name, 'variant': str(sequence),
            'method': method or marker.get('exposure_mode'), 'policy': marker.get('ae_bracket_selection'),
            'rectification_version': marker.get('rectification_version'),
            'used_by_stereo': 'stereo' in stereo_runs, 'pairs': len(rgb),
            'superseded': bool(superseded_reasons), 'superseded_reasons': superseded_reasons,
            'timestamp_or_bracket_mismatches': mismatch, 'capture_cycle_mismatches': cycle_mismatch,
            'missing_images': missing_images, 'missing_sources': missing_sources,
            'image_geometry': image_check,
            'controller_replay': controller_check,
            'status': 'fail' if errors else 'pass', 'errors': errors[:20], 'warnings': warnings_list}


def run_audit(root: Path, variants_root: Path, results_root: Path, *, samples: int,
              replay: bool, replay_frames: int | None, code: Path):
    raw = []
    mappings = {}
    names = discover(root)
    for i, name in enumerate(names, 1):
        result, mappings[name] = raw_cycles(root / name)
        raw.append(result)
        print(f'Raw {i}/{len(names)}: {name}: {result["status"]}', flush=True)
    stereo_usage = {}
    for path in results_root.glob('backpack_*/*/run.json'):
        try:
            run = json.loads(path.read_text())
            stereo_usage.setdefault(str(Path(run['prepared_sequence']).resolve()), set()).add(run['mode'])
        except (KeyError, OSError, ValueError):
            continue
    prepared = []
    markers = sorted(variants_root.glob('*/sequences/backpack_*/borealhdr.json'))
    for i, marker in enumerate(markers, 1):
        sequence = marker.parent
        if sequence.name not in mappings:
            continue
        try:
            result = prepared_variant(sequence, mappings[sequence.name], samples=samples,
                                      replay=replay, code=code,
                                      replay_frames=replay_frames,
                                      stereo_runs=stereo_usage.get(str(sequence.resolve()), set()))
        except (OSError, ValueError, KeyError) as error:
            result = {'sequence': sequence.name, 'variant': str(sequence),
                      'status': 'fail', 'errors': [str(error)]}
        prepared.append(result)
        print(f'Prepared {i}/{len(markers)}: {sequence.name}: {result["status"]}', flush=True)
    return {'schema_version': 1, 'generated_at': datetime.now(timezone.utc).isoformat(),
            'raw_sequences': raw, 'prepared_variants': prepared,
            'summary': {'raw_checked': len(raw), 'raw_failed': sum(r['status'] == 'fail' for r in raw),
                        'raw_unpaired_images': sum(sum(r.get('unpaired_images', {}).values()) for r in raw),
                        'raw_incomplete_cycles': sum(r.get('incomplete_cycles', 0) for r in raw),
                        'prepared_checked': len(prepared),
                        'prepared_failed': sum(r['status'] == 'fail' for r in prepared),
                        'current_failed': sum(r['status'] == 'fail' and not r.get('superseded')
                                              for r in prepared),
                        'superseded_failed': sum(r['status'] == 'fail' and r.get('superseded')
                                                 for r in prepared),
                        'methods_found': sorted({r.get('method') for r in prepared if r.get('method')})}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--code', type=Path, default=DEFAULT_CODE)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--samples', type=int, default=12,
                        help='Evenly spaced stereo pairs to check geometrically per variant (0 disables)')
    parser.add_argument('--replay', action='store_true', help='Recompute AE controller decisions from raw frames')
    parser.add_argument('--replay-frames', type=int,
                        help='Replay only the first N AE frames per variant; trace continuity is still checked for all frames')
    parser.add_argument('--strict-history', action='store_true',
                        help='Also fail the command when superseded prepared variants fail checks')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    report = (args.report.expanduser().resolve() if args.report else
              output / 'audits' / f'borealhdr_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json')
    if (output.is_relative_to(Path('/mnt/share/nas')) or
            report.is_relative_to(Path('/mnt/share/nas')) or
            report.is_relative_to(root)):
        parser.error('Audit report must be outside the raw dataset and NAS')
    if args.samples < 0:
        parser.error('--samples must be nonnegative')
    if args.replay_frames is not None and (not args.replay or args.replay_frames < 1):
        parser.error('--replay-frames requires --replay and a positive number')
    result = run_audit(root, output / 'exposure_variants',
                       output / 'results', samples=args.samples, replay=args.replay,
                       replay_frames=args.replay_frames,
                       code=args.code.expanduser().resolve())
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2))
    print(f'Summary: {result["summary"]}', flush=True)
    print(f'Report: {report}', flush=True)
    if (result['summary']['raw_failed'] or result['summary']['current_failed'] or
            (args.strict_history and result['summary']['superseded_failed'])):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
