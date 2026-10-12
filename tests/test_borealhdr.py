import tempfile
import unittest
import json
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import yaml

from Utilities.borealhdr import (discover, prepare, main, load_exposures, exposure_rows,
                                convert_exposure, ae50_next_exposure, ae50_source_bracket,
                                ae50_tables, response_curve, source_bracket)
from Datasets.get_dataset import get_dataset
from Run.single_sequence import BaselineRunError


class BorealHDRTests(unittest.TestCase):
    def test_shared_only_checks_reference_before_preparing_images(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, 'argv', ['borealhdr', 'run', 'example', '--shared-only',
                                            '--output', directory]), \
                 patch('Utilities.boreal_reference.ensure_full_reference',
                       side_effect=FileNotFoundError('shared reference missing')) as reference, \
                 patch('Utilities.borealhdr.prepare') as preparation:
                with self.assertRaisesRegex(FileNotFoundError, 'shared reference missing'):
                    main()
            self.assertTrue(reference.call_args.kwargs['shared_only'])
            preparation.assert_not_called()

    def test_schedule_validation_and_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'exposure.yaml'
            path.write_text('0: 4\n2: 8\n')
            self.assertEqual(load_exposures(path, 4), [4., 4., 8., 8.])
            for invalid in ('1: 4', '0: -1', '0: .nan', '0: true', '0: 4\n4: 8',
                            '0: 4\n0: 8', '0: 4\n-1: 8', '"0": 4', '[]'):
                path.write_text(invalid)
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    load_exposures(path, 4)
            path.write_text('0: 4\n1: 0\n')
            self.assertEqual(load_exposures(path, 3), [4., 0., 0.])

    def test_stereo_uses_same_bracket_and_retains_real_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for bracket, offset in ((4., 10), (8., 20)):
                for side in ('left', 'right'):
                    folder = source / f'camera_{side}' / str(bracket)
                    folder.mkdir(parents=True)
                    for i in range(3):
                        (folder / f'{1000 + i * 100 + offset}.png').touch()
            rows = exposure_rows(source, [4., 8., 4.])
            self.assertEqual([row[3] for row in rows], [1010, 1120, 1210])
            for row in rows:
                self.assertEqual(row[4].name, row[5].name)
                self.assertEqual(row[4].parent.name, row[5].parent.name)
            (source / 'camera_right/8.0/1120.png').unlink()
            with self.assertRaises(ValueError):
                exposure_rows(source, [4., 8., 4.])

    def test_intermediate_exposure_uses_response_curve_and_clips(self):
        gray = np.array([[1000, 3000]], dtype=np.uint16)
        curve = np.linspace(0, 4095, 256)
        np.testing.assert_array_equal(convert_exposure(gray, 6., 4., curve), [[1500, 4095]])
        np.testing.assert_array_equal(convert_exposure(gray, 4., 4., None), gray)

    def test_ae50_matches_pinned_classical_controller(self):
        # Golden outputs follow Metric_Classical(50) at BorealHDR 042befac.
        for dn, exposure, expected in ((1024, 4., 6.048),
                                       (3000, 4., 2.096),
                                       (2040, 4., 4.),
                                       (2033, 4., 4.03),
                                       (4095, 1., 3.094),
                                       (4095, 4.094, 0.)):
            bayer = np.full((12, 16), dn, dtype=np.uint16)
            mean, result = ae50_next_exposure(bayer, exposure)
            self.assertEqual(mean, dn)
            self.assertAlmostEqual(result, expected)
        self.assertEqual(source_bracket(0.), 1.)
        self.assertEqual(source_bracket(100.), 32.)
        np.testing.assert_array_equal(convert_exposure(np.full((2, 2), 1000, dtype=np.uint16),
                                                       0., 1., np.linspace(0, 4095, 256)),
                                      np.zeros((2, 2), dtype=np.uint16))

    def test_ae50_matches_pinned_bracket_rule_and_crf(self):
        normal = np.full((4, 4), 1024, dtype=np.uint16)
        saturated = np.full((4, 4), 4094, dtype=np.uint16)
        images = {bracket: normal for bracket in (1., 2., 4., 8., 16., 32.)}
        # 5.537 ms is nearer to 4 ms in log space, but BorealHDR takes 8 ms.
        self.assertEqual(ae50_source_bracket(5.537, images.__getitem__), 8.)
        images[8.] = saturated
        self.assertEqual(ae50_source_bracket(5.537, images.__getitem__), 4.)
        self.assertEqual(ae50_source_bracket(0., images.__getitem__), 1.)
        self.assertEqual(ae50_source_bracket(40., images.__getitem__), 32.)
        with tempfile.TemporaryDirectory() as directory:
            calibration = Path(directory) / 'calibration_files'
            calibration.mkdir()
            forest = np.linspace(0., 4095. / 16., 256)
            inside = forest.copy()
            inside[1:-1] *= 0.9
            np.savetxt(calibration / 'pcalib_forest2024.txt', forest)
            np.savetxt(calibration / 'pcalib_inside1.txt', inside)
            self.assertNotEqual(response_curve(Path(directory), ae=True)[128],
                                response_curve(Path(directory))[128])

    def test_ae_aligns_brackets_and_skips_incomplete_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            reference = []
            for bracket in (1., 2., 4., 8., 16., 32.):
                for side in ('left', 'right'):
                    folder = source / f'camera_{side}' / str(bracket)
                    folder.mkdir(parents=True)
                    for i in range(4):
                        if bracket == 16. and i == 1:
                            continue
                        stamp = 1000 + i * 300 + int(bracket)
                        path = folder / f'{stamp}.png'
                        path.touch()
                        if bracket == 4. and side == 'left':
                            reference.append((stamp, path, folder.parent.parent /
                                              'camera_right/4.0' / path.name))
            aligned = ae50_tables(source, reference)
            self.assertEqual(len(aligned[4.]), 3)
            self.assertEqual([row[0] for row in aligned[16.]], [1016, 1616, 1916])

    def test_ae_cli_routing_and_conflict(self):
        for action in ('prepare', 'run', 'demo'):
            with self.subTest(action=action), \
                 patch.object(sys, 'argv', ['borealhdr', action, 'example', '--ae']), \
                 patch.dict('os.environ', {'DISPLAY': ':99'}), \
                 patch('Utilities.borealhdr.prepare', side_effect=RuntimeError('reached prepare')) as preparation:
                with self.assertRaisesRegex(RuntimeError, 'reached prepare'):
                    main()
                self.assertTrue(preparation.call_args.kwargs['ae'])
        for method in ('zhang', 'kim'):
            with self.subTest(method=method), \
                 patch.object(sys, 'argv', ['borealhdr', 'prepare', 'example', '--ae', method]), \
                 patch('Utilities.borealhdr.prepare', side_effect=RuntimeError('routed')) as preparation:
                with self.assertRaisesRegex(RuntimeError, 'routed'):
                    main()
                self.assertEqual(preparation.call_args.kwargs['ae'], method)
        with patch.object(sys, 'argv', ['borealhdr', 'prepare', 'example', '--ae',
                                       '--exposure-yaml', 'schedule.yaml']):
            with self.assertRaises(SystemExit):
                main()
        with patch.object(sys, 'argv', ['borealhdr', 'run', 'example', '--ae', '--ae-plot']):
            with self.assertRaises(SystemExit):
                main()

    def test_ae_prepares_stereo_trace_replay_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = 'backpack_2023-04-20-09-29-14'
            calibration = root / 'code/calibration_files/calibration_vo/april_2023'
            calibration.mkdir(parents=True)
            np.savetxt(root / 'code/calibration_files/pcalib_forest2024.txt',
                       np.linspace(0, 4095 / 16, 256))
            np.savetxt(root / 'code/calibration_files/pcalib_inside1.txt',
                       np.linspace(0, 4095 / 16, 256))
            for side in ('left', 'right'):
                config = {'image_width': 16, 'image_height': 12,
                          'camera_matrix': {'data': [10., 0, 8, 0, 10, 6, 0, 0, 1]},
                          'distortion_coefficients': {'data': [0.] * 5},
                          'rectification_matrix': {'data': np.eye(3).ravel().tolist()},
                          'projection_matrix': {'data': [10., 0, 8, -2 if side == 'right' else 0,
                                                         0, 10, 6, 0, 0, 0, 1, 0]}}
                (calibration / f'{side}.yaml').write_text(yaml.safe_dump(config))
                for bracket in (1., 2., 4., 8., 16., 32.):
                    folder = root / 'data' / name / f'camera_{side}' / str(bracket)
                    folder.mkdir(parents=True)
                    for i in range(4):
                        stamp = 1_000_000_000 + i * 300_000_000 + int(bracket * 1_000_000)
                        value = 1024 if side == 'left' else 3072
                        cv2.imwrite(str(folder / f'{stamp}.png'),
                                    np.full((12, 16), value, dtype=np.uint16))
            # Upstream emulates each camera independently: left 8 ms falls back
            # to 4 ms for saturation while right remains on 8 ms.
            left_8_frame_1 = (root / 'data' / name / 'camera_left/8.0' /
                              f'{1_000_000_000 + 300_000_000 + 8_000_000}.png')
            cv2.imwrite(str(left_8_frame_1), np.full((12, 16), 4094, dtype=np.uint16))
            sequence = prepare(root / 'data', name, root / 'code', root / 'out', ae=True)
            self.assertIn('exposure_variants', sequence.parts)
            trace = pd.read_csv(sequence / 'ae_trace.csv')
            exposure = pd.read_csv(sequence / 'exposure.csv')
            self.assertEqual(trace['exposure_ms'].iloc[0], 4.)
            self.assertEqual(trace['mean_dn'].iloc[0], 1024.)
            self.assertAlmostEqual(trace['next_exposure_ms'].iloc[0], 6.048)
            self.assertAlmostEqual(trace['exposure_ms'].iloc[1], 6.048)
            self.assertEqual(exposure['source_bracket_ms'].iloc[1], 4.)
            self.assertEqual(exposure['right_source_bracket_ms'].iloc[1], 8.)
            self.assertNotEqual(exposure['timestamp_ns'].iloc[1],
                                exposure['right_timestamp_ns'].iloc[1])
            rgb = pd.read_csv(sequence / 'rgb.csv')
            self.assertEqual(rgb['ts_rgb_0 (ns)'].iloc[1], exposure['timestamp_ns'].iloc[1])
            self.assertEqual(rgb['ts_rgb_1 (ns)'].iloc[1], exposure['right_timestamp_ns'].iloc[1])
            self.assertTrue((sequence / rgb['path_rgb_1'].iloc[1]).is_file())
            self.assertEqual(yaml.safe_load((sequence / 'exposure.yaml').read_text())[1], 6.048)
            metadata = json.loads((sequence / 'borealhdr.json').read_text())
            self.assertEqual(metadata['exposure_mode'], 'ae50')
            self.assertEqual(metadata['ae_response_curve'], 'pcalib_inside1.txt')
            self.assertEqual(metadata['ae_bracket_selection'], 'closer_least_sat_per_camera')
            with patch('Utilities.borealhdr.cv2.imread', side_effect=AssertionError('cached')):
                self.assertEqual(prepare(root / 'data', name, root / 'code', root / 'out', ae=True), sequence)

    def test_zhang_and_kim_preparations_have_distinct_traces_and_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = 'backpack_2023-04-20-09-29-14'
            calibration = root / 'code/calibration_files/calibration_vo/april_2023'
            calibration.mkdir(parents=True)
            np.savetxt(root / 'code/calibration_files/pcalib_inside1.txt',
                       np.linspace(0, 4095 / 16, 256))
            for side in ('left', 'right'):
                config = {'image_width': 32, 'image_height': 24,
                          'camera_matrix': {'data': [20., 0, 16, 0, 20, 12, 0, 0, 1]},
                          'distortion_coefficients': {'data': [0.] * 5},
                          'rectification_matrix': {'data': np.eye(3).ravel().tolist()},
                          'projection_matrix': {'data': [20., 0, 16, -2 if side == 'right' else 0,
                                                         0, 20, 12, 0, 0, 0, 1, 0]}}
                (calibration / f'{side}.yaml').write_text(yaml.safe_dump(config))
                for bracket in (1., 2., 4., 8., 16., 32.):
                    folder = root / 'data' / name / f'camera_{side}' / str(bracket)
                    folder.mkdir(parents=True)
                    for i in range(3):
                        stamp = 1_000_000_000 + i * 300_000_000 + int(bracket * 1_000_000)
                        raw = np.random.default_rng(i + int(bracket) + (side == 'right')).integers(
                            400, 3600, size=(24, 32), dtype=np.uint16)
                        cv2.imwrite(str(folder / f'{stamp}.png'), raw)
            prepared = {}
            for method in ('zhang', 'kim'):
                prepared[method] = prepare(root / 'data', name, root / 'code', root / 'out', ae=method)
                metadata = json.loads((prepared[method] / 'borealhdr.json').read_text())
                trace = pd.read_csv(prepared[method] / 'ae_trace.csv')
                self.assertEqual(metadata['ae_method'], method)
                self.assertEqual(metadata['exposure_mode'], f'ae_{method}')
                self.assertEqual(trace['ae_method'].tolist(), [method] * 3)
                self.assertTrue(trace['next_exposure_ms'].notna().all())
                self.assertTrue((trace['next_exposure_ms'] >= 0).all())
                with patch('Utilities.borealhdr.cv2.imread', side_effect=AssertionError('cached')):
                    self.assertEqual(prepare(root / 'data', name, root / 'code', root / 'out', ae=method),
                                     prepared[method])
            self.assertNotEqual(prepared['zhang'], prepared['kim'])

    def test_cli_routes_headless_and_demo_and_records_results(self):
        for action, headless, ae, ae_plot, crash, few in [
            ('run', True, False, False, False, False),
            ('demo', False, False, False, False, False),
            ('run', True, True, False, False, False),
            ('demo', False, True, True, False, False),
            ('run', True, True, False, True, False),
            ('run', True, True, False, False, True),
        ]:
            with self.subTest(action=action), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sequence = root / 'sequences/example'
                sequence.mkdir(parents=True)
                (sequence / 'rgb.csv').write_text('ts_rgb_0 (ns),path_rgb_0\n'
                                                 '1000000000,rgb_0/1.png\n2000000000,rgb_0/2.png\n'
                                                 '3000000000,rgb_0/3.png\n')
                if ae:
                    for filename in ('exposure.yaml', 'exposure.csv', 'borealhdr.json', 'ae_trace.csv'):
                        (sequence / filename).write_text('{}' if filename == 'borealhdr.json' else 'ae-test')
                calibration = root / 'code/calibration_files/calibration_vo/september_2023'
                calibration.mkdir(parents=True)
                (calibration / 'left.yaml').write_text(yaml.safe_dump({
                    'rectification_matrix': {'data': np.eye(3).ravel().tolist()}}))
                reference = root / 'fast_lio.csv'
                reference.write_text('ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw\n'
                                     '0,0,0,0,0,0,0,1\n'
                                     '1000000000,1,0,0,0,0,0,1\n'
                                     '2000000000,2,1,0,0,0,0,1\n'
                                     '3000000000,3,0,0,0,0,0,1\n'
                                     '4000000000,4,0,0,0,0,0,1\n')
                trajectory = root / 'trajectory.csv'
                trajectory.write_text('ts (ns),x,y,z,qx,qy,qz,qw\n'
                                      '1000000000,0,0,0,0,0,0,1\n'
                                      '2000000000,1,0,0,0,0,0,1\n'
                                      '3000000000,1,1,0,0,0,0,1\n')
                if few:
                    trajectory.write_text('ts (ns),x,y,z,qx,qy,qz,qw\n'
                                          '1000000000,0,0,0,0,0,0,1\n')
                argv = ['borealhdr', action, 'example', '--output', str(root),
                        '--code', str(root / 'code')]
                argv += ['--ae'] if ae else []
                argv += ['--ae-plot'] if ae_plot else []
                with patch.object(sys, 'argv', argv), \
                     patch.dict('os.environ', {'DISPLAY': ':99'}), \
                     patch('Utilities.borealhdr.prepare', return_value=sequence), \
                     patch('Utilities.borealhdr.launch_ae_plot') as plot_launcher, \
                     patch('Utilities.boreal_reference.ensure_full_reference', return_value=reference), \
                     patch('Run.single_sequence.run_single_baseline', return_value=trajectory) as runner:
                    plot_launcher.return_value.poll.return_value = None
                    if crash:
                        runner.side_effect = BaselineRunError('orbslam3', 139, root / 'native.log', trajectory, 1)
                        with self.assertRaises(BaselineRunError):
                            main()
                    elif few:
                        with self.assertRaisesRegex(RuntimeError, 'no evaluable trajectory.*check SLAM log'):
                            main()
                    else:
                        main()
                if ae_plot:
                    plot_launcher.assert_called_once_with(sequence)
                    plot_launcher.return_value.terminate.assert_called_once()
                else:
                    plot_launcher.assert_not_called()
                self.assertEqual(runner.call_args.kwargs['headless'], headless)
                state = next((root / 'results').glob('*/*/run.json'))
                saved = json.loads(state.read_text())
                if crash:
                    self.assertEqual(saved['status'], 'failed')
                    self.assertEqual(saved['exit_code'], 139)
                    self.assertEqual(saved['signal_name'], 'SIGSEGV')
                    self.assertEqual(saved['pose_rows'], 1)
                    self.assertEqual(saved['log_file'], str(root / 'native.log'))
                    continue
                if few:
                    self.assertEqual(saved['status'], 'failed')
                    self.assertIn('no evaluable trajectory', saved['error'])
                    continue
                self.assertEqual(saved['status'], 'complete')
                if ae:
                    self.assertEqual(json.loads(state.read_text())['exposure_mode'], 'ae50')
                    self.assertTrue((state.parent / 'ae_trace.csv').is_file())
                self.assertTrue((state.parent / 'trajectory_report.pdf').is_file())
                self.assertTrue((state.parent / 'metrics.json').is_file())
                metrics = json.loads((state.parent / 'metrics.json').read_text())
                self.assertEqual(metrics['primary_reference'], 'boreal_fast_lio')
                self.assertIn('vslam_vs_boreal_fast_lio', metrics['comparisons'])
                self.assertEqual(len(pd.read_csv(sequence / 'groundtruth.csv')), 3)

    def test_preparation_scales_calibration_pairs_timestamps_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = 'backpack_2023-04-20-09-29-14'
            calibration = root / 'code/calibration_files/calibration_vo/april_2023'
            calibration.mkdir(parents=True)
            stamps = [1681997354862757120 + i * 300000000 for i in range(4)]
            for side in ('left', 'right'):
                folder = root / 'data' / name / f'camera_{side}/4.0'
                folder.mkdir(parents=True)
                for stamp in stamps:
                    cv2.imwrite(str(folder / f'{stamp}.png'), np.full((12, 16), 1024, dtype=np.uint16))
                config = {'image_width': 32, 'image_height': 24,
                          'camera_matrix': {'data': [20., 0, 16, 0, 20, 12, 0, 0, 1]},
                          'distortion_coefficients': {'data': [0.] * 5},
                          'rectification_matrix': {'data': np.eye(3).ravel().tolist()},
                          'projection_matrix': {'data': [20., 0, 16, -4 if side == 'right' else 0,
                                                         0, 20, 12, 0, 0, 0, 1, 0]}}
                (calibration / f'{side}.yaml').write_text(yaml.safe_dump(config))
            self.assertEqual(discover(root / 'data'), [name])
            sequence = prepare(root / 'data', name, root / 'code', root / 'out')
            frame = pd.read_csv(sequence / 'rgb.csv')
            self.assertEqual(frame.iloc[0, 0], stamps[0])
            image = cv2.imread(str(sequence / 'rgb_0' / f'{stamps[0]}.png'), -1)
            self.assertEqual(image.dtype, np.uint8)
            self.assertEqual(int(image[6, 8]), 64)
            camera = yaml.safe_load((sequence / 'calibration.yaml').read_text())['cameras']
            self.assertEqual(camera[0]['focal_length'], [10., 10.])
            self.assertAlmostEqual(camera[1]['T_BS'][3], .2)
            self.assertFalse((sequence / 'groundtruth.csv').exists())
            dataset = get_dataset('borealhdr')
            self.assertEqual(dataset.prepare_local_sequence(sequence.parent, name), sequence)
            with patch('Utilities.borealhdr.cv2.imread', side_effect=AssertionError('cached')):
                self.assertEqual(prepare(root / 'data', name, root / 'code', root / 'out'), sequence)
            for side in ('left', 'right'):
                folder = root / 'data' / name / f'camera_{side}/8.0'
                folder.mkdir()
                for stamp in stamps:
                    cv2.imwrite(str(folder / f'{stamp + 50000000}.png'), np.full((12, 16), 2048, dtype=np.uint16))
            schedule = root / 'schedule.yaml'
            schedule.write_text('0: 4\n2: 8\n')
            variant = prepare(root / 'data', name, root / 'code', root / 'out', schedule)
            self.assertNotEqual(variant, sequence)
            log = pd.read_csv(variant / 'exposure.csv')
            self.assertEqual(log['exposure_ms'].tolist(), [4, 4, 8, 8])
            self.assertEqual(log['timestamp_ns'].tolist(), stamps[:2] + [s + 50000000 for s in stamps[2:]])
            for side in (0, 1):
                rendered = cv2.imread(str(variant / f'rgb_{side}' / f'{stamps[2] + 50000000}.png'), -1)
                self.assertEqual(int(rendered[6, 8]), 128)
            with patch('Utilities.borealhdr.cv2.imread', side_effect=AssertionError('cached')):
                self.assertEqual(prepare(root / 'data', name, root / 'code', root / 'out', schedule), variant)
            schedule.write_text('0: 8\n')
            changed = prepare(root / 'data', name, root / 'code', root / 'out', schedule)
            self.assertNotEqual(changed, variant)
            self.assertTrue((variant / 'exposure.csv').is_file())


if __name__ == '__main__':
    unittest.main()
