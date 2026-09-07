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

from Utilities.borealhdr import discover, prepare, main
from Datasets.get_dataset import get_dataset


class BorealHDRTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
