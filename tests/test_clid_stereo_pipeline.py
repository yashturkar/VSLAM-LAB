import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from Utilities.clid_stereo_pipeline import (
    local_destination, prepare_variant, read_component, render, rgb_pairs, stereo_maps,
    validate_coefficients,
)
from Utilities.clid_relight import sha256_file


class StereoPipelineTests(unittest.TestCase):
    def test_rgb_pairs_use_exact_equal_stamps_and_existing_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for side in (0, 1):
                (root / f'rgb_{side}').mkdir()
                for stamp in (100, 200, 300):
                    (root / f'rgb_{side}' / f'{stamp}.png').write_bytes(b'png')
            file = root / 'rgb.csv'
            with file.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['ts_rgb_0 (ns)', 'path_rgb_0', 'ts_rgb_1 (ns)', 'path_rgb_1'])
                for stamp in (100, 200, 300):
                    writer.writerow([stamp, f'rgb_0/{stamp}.png', stamp, f'rgb_1/{stamp}.png'])
            self.assertEqual(rgb_pairs(file), [100, 200, 300])
            text = file.read_text().replace('300,rgb_1/300.png', '301,rgb_1/300.png')
            file.write_text(text)
            with self.assertRaisesRegex(ValueError, 'Non-synchronous'):
                rgb_pairs(file)

    def test_local_destination_rejects_nas_and_source(self):
        source = Path('/mnt/share/nas/eph/sequence')
        self.assertEqual(local_destination(Path('/mnt/share/local/eph/VSLAM/run'), source),
                         Path('/mnt/share/local/eph/VSLAM/run'))
        for output in (Path('/mnt/share/nas/eph/output'), source / 'output', Path('/mnt/share/other/output')):
            with self.subTest(output=output), self.assertRaises(ValueError):
                local_destination(output, source)

    def test_float32_coefficients_are_validated_and_quantized(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'frame.npz'
            arrays = {key: np.full((4, 400, 640), 0.125, np.float32) for key in ('A', 'F', 'C1', 'C2')}
            np.savez_compressed(file, **arrays)
            values = read_component(file, sha256_file(file))
            self.assertTrue(all(value.dtype == np.float16 for value in values.values()))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                read_component(file, 'wrong')

    def test_render_uses_per_camera_power_and_response(self):
        arrays = {key: np.zeros((4, 400, 640), np.float16) for key in ('A', 'F', 'C1', 'C2')}
        arrays['F'].fill(0.25)
        calibration = {'power_response': {'commanded': [0, 1], 'effective': [0, 1]},
                       'response': {'bayer_alpha': {key: 1 for key in ('R', 'G1', 'G2', 'B')}}}
        dark = render(arrays, calibration, 0, 0)
        bright = render(arrays, calibration, 1, 1)
        self.assertEqual(dark.shape, (800, 1280, 3))
        self.assertEqual(dark.dtype, np.uint8)
        self.assertEqual(int(dark.max()), 0)
        self.assertGreater(int(bright.min()), 0)
        right = json.loads(json.dumps(calibration))
        right['power_response']['effective'] = [0, 0.5]
        self.assertLess(float(render(arrays, right, 1, 1).mean()), float(bright.mean()))

    def test_render_keeps_red_bayer_site_red_in_bgr_png_bytes(self):
        arrays = {key: np.zeros((4, 400, 640), np.float16) for key in ('A', 'F', 'C1', 'C2')}
        arrays['A'][0].fill(1)
        calibration = {'power_response': {'commanded': [0, 1], 'effective': [0, 1]},
                       'response': {'bayer_alpha': {key: 1 for key in ('R', 'G1', 'G2', 'B')}}}
        bgr = render(arrays, calibration, 0, 0)
        self.assertGreater(int(bgr[400, 640, 2]), 250)
        self.assertEqual(int(bgr[400, 640, 0]), 0)

    def test_stereo_projection_scales_baseline_without_changing_meters(self):
        class Info:
            width, height = 1920, 1200
            k = [900, 0, 960, 0, 900, 600, 0, 0, 1]
            d = [0, 0, 0, 0, 0]
            r = [1, 0, 0, 0, 1, 0, 0, 0, 1]
            def __init__(self, tx):
                self.p = [900, 0, 960, tx, 0, 900, 600, 0, 0, 0, 1, 0]
        info = {'/cam_sync/cam0/camera_info': Info(0), '/cam_sync/cam1/camera_info': Info(-90)}
        with patch('Utilities.clid_stereo_pipeline.read_camera_info', return_value=info):
            maps, projection, baseline = stereo_maps(Path('/ignored.mcap'), (1280, 800))
        self.assertEqual(set(maps), {'left', 'right'})
        self.assertAlmostEqual(float(projection[0, 0]), 600)
        self.assertAlmostEqual(baseline, 0.1)

    def test_coefficient_manifest_requires_full_stereo_and_correct_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {'bag': '/source/bag.mcap', 'bag_size': 10, 'bag_mtime_ns': 123,
                        'capture_manifest_sha256': 'capture'}
            release = {'checkpoint': 'best.pt', 'sha256': {'best.pt': 'checkpoint', 'left.json': 'left-hash',
                       'right.json': 'right-hash'}, 'cameras': {}}
            group = {'complete': True, 'is_full_sequence': True, 'checkpoint_sha256': 'checkpoint',
                     'exact_common_timestamp_count': 3, 'cameras': {}}
            for side in ('left', 'right'):
                (root / side).mkdir()
                release['cameras'][side] = {'serial': side, 'input_topic': f'/{side}/raw',
                                            'metadata_topic': f'/{side}/meta', 'calibration': f'{side}.json'}
                records = []
                for index, stamp in enumerate((100, 200, 300)):
                    file = root / side / f'{index}.npz'
                    file.write_bytes(b'coefficient')
                    records.append({'header_stamp_ns': stamp, 'coefficients': {'path': file.name,
                                    'bytes': file.stat().st_size, 'sha256': sha256_file(file)}})
                manifest = {'complete': True, 'is_full_sequence': True, 'camera': side,
                            'camera_serial': side, 'topic': f'/{side}/raw', 'metadata_topic': f'/{side}/meta',
                            'checkpoint_sha256': 'checkpoint', 'config_sha256': f'{side}-hash',
                            'bag': identity['bag'], 'bag_bytes': 10, 'bag_mtime_ns': 123,
                            'capture_manifest_sha256': 'capture', 'expected_frames': 3,
                            'frame_count': 3, 'frames': records}
                path = root / side / 'manifest.json'
                path.write_text(json.dumps(manifest))
                group['cameras'][side] = {'manifest_sha256': sha256_file(path)}
            (root / 'stereo-manifest.json').write_text(json.dumps(group))
            self.assertEqual(len(validate_coefficients(root, release, identity)['left']), 3)
            group['is_full_sequence'] = False
            (root / 'stereo-manifest.json').write_text(json.dumps(group))
            self.assertIsNone(validate_coefficients(root, release, identity))

    def test_prepare_variant_writes_exact_stereo_pairs_and_scaled_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, coefficients, artifacts = (root / name for name in ('base', 'coefficients', 'artifacts'))
            for folder in (base, coefficients, artifacts):
                folder.mkdir()
            for side in (0, 1):
                (base / f'rgb_{side}').mkdir()
                for stamp in (100, 200, 300):
                    (base / f'rgb_{side}' / f'{stamp}.png').write_bytes(b'preview')
            with (base / 'rgb.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['ts_rgb_0 (ns)', 'path_rgb_0', 'ts_rgb_1 (ns)', 'path_rgb_1'])
                for stamp in (100, 200, 300):
                    writer.writerow([stamp, f'rgb_0/{stamp}.png', stamp, f'rgb_1/{stamp}.png'])
            (base / 'groundtruth.csv').write_text('ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw\n')
            (base / 'extraction_metadata.json').write_text(json.dumps({'fps': 10}))
            (coefficients / 'stereo-manifest.json').write_text('{}')
            release = {'cameras': {}}
            frames = {}
            for side in ('left', 'right'):
                name = f'{side}.json'
                release['cameras'][side] = {'calibration': name}
                (artifacts / name).write_text(json.dumps({
                    'side': side, 'power_response': {'commanded': [0, 1], 'effective': [0, 1]},
                }))
                frames[side] = {stamp: {'file': root / f'{side}-{stamp}.npz', 'sha256': 'hash'}
                                for stamp in (100, 200, 300)}
            schedule = root / 'schedule.yaml'
            schedule.write_text('0: {light: 0.2, exposure_ev: 0}\n2: {light: 0.8, exposure_ev: 1}\n')
            projection = np.array([[600., 0, 640, 0], [0, 600, 400, 0], [0, 0, 1, 0]])
            dummy_maps = {'left': (None, None), 'right': (None, None)}
            calls = []
            def synthetic_render(_values, calibration, light, ev):
                calls.append((calibration['side'], light, ev))
                return np.zeros((800, 1280, 3), np.uint8)
            with patch('Utilities.clid_stereo_pipeline.stereo_maps', return_value=(dummy_maps, projection, 0.1)), \
                 patch('Utilities.clid_stereo_pipeline.read_component', return_value={}), \
                 patch('Utilities.clid_stereo_pipeline.render', side_effect=synthetic_render), \
                 patch('Utilities.clid_stereo_pipeline.cv2.remap', side_effect=lambda image, *_a, **_kw: image):
                destination = prepare_variant(base, root / 'bag.mcap', coefficients, frames,
                                              artifacts, release, schedule, root / 'out')
                again = prepare_variant(base, root / 'bag.mcap', coefficients, frames,
                                        artifacts, release, schedule, root / 'out')
            self.assertEqual(destination, again)
            self.assertEqual(rgb_pairs(destination / 'rgb.csv'), [100, 200, 300])
            self.assertEqual(len(calls), 6)
            self.assertEqual(calls[0], ('left', 0.2, 0.0))
            self.assertEqual(calls[-1], ('right', 0.8, 1.0))
            calibration = json.loads((destination / 'calibration.yaml').read_text())
            self.assertEqual(calibration['cameras'][0]['image_dimension'], [1280, 800])
            self.assertAlmostEqual(calibration['cameras'][1]['T_BS'][3], 0.1)


if __name__ == '__main__':
    unittest.main()
