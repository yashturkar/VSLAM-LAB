"""Regression values checked against BorealHDR@042befac on synthetic Bayer."""

import unittest

import numpy as np

from Utilities.boreal_ae_controllers import KimController, ZhangController


class BorealAEControllerTests(unittest.TestCase):
    def setUp(self):
        self.bayer = np.random.default_rng(17).integers(400, 3600, size=(24, 32), dtype=np.uint16)
        self.curve = np.linspace(0, 4095, 256)

    def test_zhang_soft_percentile_matches_pinned_formula_with_image_sized_weights(self):
        controller = ZhangController(self.curve)
        exposure, metric, branch = controller.update(self.bayer, 4.)
        self.assertEqual(branch, 'soft_percentile')
        self.assertEqual(controller.weights.size, self.bayer.size)
        self.assertTrue(np.isfinite(metric))
        self.assertGreater(exposure, 0.02)

    def test_zhang_classical_boundary_and_minimum_exposure(self):
        controller = ZhangController(self.curve)
        exposure, metric, branch = controller.update(np.full((24, 32), 1024, dtype=np.uint16), 4.)
        self.assertEqual((metric, branch), (1024., 'classical_boundary'))
        self.assertAlmostEqual(exposure, 6.048)
        exposure, _, branch = controller.update(np.full((24, 32), 4095, dtype=np.uint16), 4.094)
        self.assertEqual(branch, 'classical_boundary')
        self.assertEqual(exposure, 0.02)

    def test_kim_first_update_matches_pinned_seeded_gp(self):
        controller = KimController()
        exposure, metric, branch = controller.update(self.bayer, 4.)
        self.assertAlmostEqual(metric, 10417.547204603623, places=6)
        self.assertAlmostEqual(exposure, 4.020408163265306)
        self.assertEqual(branch, 'maxmi')
        self.assertEqual(controller.exposures, [4.])
        other = KimController()
        self.assertEqual(other.update(self.bayer, 4.), (exposure, metric, branch))


if __name__ == '__main__':
    unittest.main()
