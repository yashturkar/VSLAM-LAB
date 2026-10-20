import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Utilities.boreal_ae_plot import FrameOpenMonitor
from Utilities.clid_control_plot import load_control_trace
from Utilities.clid_stereo_pipeline import launch_control_plot, run_slam


@unittest.skipUnless(sys.platform.startswith('linux'), 'Requires Linux inotify')
class ClidControlPlotTests(unittest.TestCase):
    def _sequence(self, root: Path) -> Path:
        images = root / 'rgb_0'
        images.mkdir()
        with (root / 'relight.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['frame_num', 'timestamp_ns', 'light_commanded', 'exposure_ev',
                             'left_effective_power', 'right_effective_power'])
            writer.writerow([0, 100, .3, -1, .4, .41])
            writer.writerow([1, 200, .4, -.5, .5, .51])
            writer.writerow([2, 300, .5, 0, .6, .61])
        for stamp in (100, 200, 300):
            (images / f'{stamp}.png').write_bytes(b'frame')
        return root

    def test_trace_and_actual_left_frame_open(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = self._sequence(Path(directory))
            values, lookup = load_control_trace(sequence)
            self.assertEqual(len(values), 3)
            self.assertEqual(values[1]['exposure_ev'], -.5)
            self.assertEqual(lookup['200.png'], 1)
            monitor = FrameOpenMonitor(sequence / 'rgb_0')
            try:
                with (sequence / 'rgb_0' / '200.png').open('rb') as stream:
                    stream.read()
                names, overflow = monitor.read()
                self.assertEqual(names, ['200.png'])
                self.assertFalse(overflow)
            finally:
                monitor.close()

    def test_invalid_frame_order_and_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = self._sequence(Path(directory))
            file = sequence / 'relight.csv'
            original = file.read_text()
            for replacement, message in (
                (original.replace('1,200', '3,200'), 'consecutive'),
                (original.replace('1,200', '1,100'), 'unique'),
                (original.replace('0.4,0.41', 'nan,0.41'), 'Non-finite'),
                (original.replace('0.5,0.51', '1.5,0.51'), 'effective power'),
            ):
                with self.subTest(message=message):
                    file.write_text(replacement)
                    with self.assertRaisesRegex(ValueError, message):
                        load_control_trace(sequence)

    def test_plot_requires_display_and_is_cleaned_up_if_slam_fails(self):
        with patch.dict(os.environ, {'DISPLAY': ''}):
            with self.assertRaisesRegex(RuntimeError, 'X display'):
                launch_control_plot(Path('/not-used'))
        with tempfile.TemporaryDirectory() as directory:
            events = []
            class Plotter:
                def poll(self):
                    return None
                def terminate(self):
                    events.append('plot_stopped')
                def wait(self, timeout=None):
                    return 0
            baseline = type('Baseline', (), {'modes': ('stereo',)})()
            def fail_slam(_config, headless):
                events.append(('slam_started', headless))
                raise RuntimeError('native failure')
            with patch('Baselines.get_baseline.list_available_baselines', return_value=['orbslam3']), \
                 patch('Baselines.get_baseline.get_baseline', return_value=baseline), \
                 patch('Utilities.clid_stereo_pipeline.launch_control_plot',
                       side_effect=lambda _sequence: events.append('plot_started') or Plotter()), \
                 patch('Run.single_sequence.run_single_baseline', side_effect=fail_slam):
                with self.assertRaisesRegex(RuntimeError, 'native failure'):
                    run_slam(Path(directory) / 'prepared', 'orbslam3', False, Path(directory), True)
            self.assertEqual(events, ['plot_started', ('slam_started', False), 'plot_stopped'])
