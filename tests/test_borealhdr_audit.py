import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from Utilities.borealhdr_audit import prepared_variant, raw_cycles, stereo_image_matches


class BorealHDRAuditTests(unittest.TestCase):
    def test_raw_audit_keeps_cycle_indices_when_first_bracket_pair_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'backpack_test'
            phases = {4.: 0, 8.: 40, 16.: 80, 32.: 120, 1.: 160, 2.: 200}
            for bracket, phase in phases.items():
                for side in ('left', 'right'):
                    folder = source / f'camera_{side}' / str(bracket)
                    folder.mkdir(parents=True)
                    for cycle in range(4):
                        if bracket == 1. and cycle == 0:
                            continue
                        (folder / f'{1000 + 300 * cycle + phase}.png').touch()
            report, mapping = raw_cycles(source)
            self.assertEqual(report['status'], 'pass')
            self.assertEqual(report['complete_cycles'], 3)
            self.assertEqual(mapping[1.][1460], 1)
            self.assertEqual(mapping[1.][2060], 3)

    def test_prepared_audit_rejects_asynchronous_stereo(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'backpack_test'
            sequence = Path(directory) / 'variant/sequences/backpack_test'
            sequence.mkdir(parents=True)
            for side in ('left', 'right'):
                (sequence / f'rgb_{0 if side == "left" else 1}').mkdir()
            left = source / 'camera_left/4.0/1300.png'
            right = source / 'camera_right/8.0/1340.png'
            left.parent.mkdir(parents=True)
            right.parent.mkdir(parents=True)
            left.touch()
            right.touch()
            (sequence / 'rgb_0/1300.png').touch()
            (sequence / 'rgb_1/1340.png').touch()
            (sequence / 'borealhdr.json').write_text(json.dumps({
                'pairs': 1, 'exposure_mode': 'ae50',
                'ae_bracket_selection': 'closer_least_sat_shared_lower'}))
            with (sequence / 'exposure.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['frame_num', 'exposure_ms', 'source_bracket_ms', 'timestamp_ns',
                                 'left_source', 'right_source', 'right_source_bracket_ms',
                                 'right_timestamp_ns'])
                writer.writerow([0, 4, 4, 1300, left, right, 8, 1340])
            with (sequence / 'rgb.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['ts_rgb_0 (ns)', 'path_rgb_0', 'ts_rgb_1 (ns)', 'path_rgb_1'])
                writer.writerow([1300, 'rgb_0/1300.png', 1340, 'rgb_1/1340.png'])
            result = prepared_variant(sequence, {4.: {1300: 1}, 8.: {1340: 1}},
                                      samples=0, replay=False, code=Path(directory),
                                      stereo_runs={'stereo'})
            self.assertEqual(result['timestamp_or_bracket_mismatches'], 1)
            self.assertEqual(result['capture_cycle_mismatches'], 0)
            self.assertEqual(result['status'], 'fail')
            self.assertFalse(result['superseded'])
            marker = sequence / 'borealhdr.json'
            metadata = json.loads(marker.read_text())
            metadata['ae_bracket_selection'] = 'closer_least_sat_per_camera'
            marker.write_text(json.dumps(metadata))
            historical = prepared_variant(sequence, {4.: {1300: 1}, 8.: {1340: 1}},
                                          samples=0, replay=False, code=Path(directory),
                                          stereo_runs={'stereo'})
            self.assertTrue(historical['superseded'])
            self.assertEqual(historical['status'], 'fail')
            right.unlink()
            result = prepared_variant(sequence, {4.: {1300: 1}, 8.: {1340: 1}},
                                      samples=0, replay=False, code=Path(directory),
                                      stereo_runs={'stereo'})
            self.assertEqual(result['missing_sources'], 1)

    def test_image_check_reports_systematic_vertical_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = Path(directory)
            (sequence / 'rgb_0').mkdir()
            (sequence / 'rgb_1').mkdir()
            image = np.random.default_rng(4).integers(0, 256, size=(480, 640), dtype=np.uint8)
            shifted = np.zeros_like(image)
            shifted[8:, :-20] = image[:-8, 20:]
            cv2.imwrite(str(sequence / 'rgb_0/0.png'), image)
            cv2.imwrite(str(sequence / 'rgb_1/0.png'), shifted)
            result = stereo_image_matches(sequence, [{
                'path_rgb_0': 'rgb_0/0.png', 'path_rgb_1': 'rgb_1/0.png'}], 1)
            self.assertGreater(result['samples'][0]['good_matches'], 30)
            self.assertGreater(result['median_vertical_px'], 3)


if __name__ == '__main__':
    unittest.main()
