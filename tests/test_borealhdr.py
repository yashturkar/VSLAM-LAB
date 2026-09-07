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

from Utilities.borealhdr import discover, prepare, main, load_exposures, exposure_rows, convert_exposure
from Datasets.get_dataset import get_dataset


class BorealHDRTests(unittest.TestCase):
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

    def test_cli_routes_headless_and_demo_and_records_results(self):
        for action, headless in [('run', True), ('demo', False)]:
            with self.subTest(action=action), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sequence = root / 'sequences/example'
                sequence.mkdir(parents=True)
                (sequence / 'rgb.csv').write_text('ts_rgb_0 (ns),path_rgb_0\n'
                                                 '1000000000,rgb_0/1.png\n3000000000,rgb_0/3.png\n')
                trajectory = root / 'trajectory.csv'
                trajectory.write_text('ts (ns),x,y,z,qx,qy,qz,qw\n'
                                      '1000000000,0,0,0,0,0,0,1\n'
                                      '2000000000,1,0,0,0,0,0,1\n'
                                      '3000000000,1,1,0,0,0,0,1\n')
                with patch.object(sys, 'argv', ['borealhdr', action, 'example', '--output', str(root)]), \
                     patch.dict('os.environ', {'DISPLAY': ':99'}), \
                     patch('Utilities.borealhdr.prepare', return_value=sequence), \
                     patch('Run.single_sequence.run_single_baseline', return_value=trajectory) as runner:
                    main()
                self.assertEqual(runner.call_args.kwargs['headless'], headless)
                state = next((root / 'results').glob('*/*/run.json'))
                self.assertEqual(json.loads(state.read_text())['status'], 'complete')
                self.assertTrue((state.parent / 'trajectory_report.pdf').is_file())

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
