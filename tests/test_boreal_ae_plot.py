import csv
import sys
import tempfile
import unittest
from pathlib import Path

from Utilities.boreal_ae_plot import FrameOpenMonitor, load_exposure_trace


@unittest.skipUnless(sys.platform.startswith('linux'), 'Requires Linux inotify')
class BorealAEPlotTests(unittest.TestCase):
    def test_exposure_lookup_and_frame_open_events(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = Path(directory)
            images = sequence / 'rgb_0'
            images.mkdir()
            with (sequence / 'exposure.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['frame_num', 'exposure_ms', 'timestamp_ns'])
                writer.writerow([0, 4., 100])
                writer.writerow([1, 6.5, 200])
            exposures, lookup = load_exposure_trace(sequence)
            self.assertEqual(exposures, [4., 6.5])
            self.assertEqual(lookup['200.png'], 1)
            (images / '100.png').write_bytes(b'frame')
            (images / '200.png').write_bytes(b'frame')
            monitor = FrameOpenMonitor(images)
            try:
                with (images / '200.png').open('rb') as stream:
                    stream.read()
                names, overflow = monitor.read()
                self.assertIn('200.png', names)
                self.assertNotIn('100.png', names)
                self.assertFalse(overflow)
                self.assertEqual(monitor.read(), ([], False))
            finally:
                monitor.close()

    def test_rejects_nonconsecutive_frame_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = Path(directory)
            (sequence / 'exposure.csv').write_text('frame_num,exposure_ms,timestamp_ns\n0,4,100\n2,8,200\n')
            with self.assertRaisesRegex(ValueError, 'consecutive'):
                load_exposure_trace(sequence)
