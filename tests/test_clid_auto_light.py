import unittest

from Utilities.clid_auto_light import AutoLightController, AutoLightPolicy


def meter(light, exposure_ms):
    return 1000 * (light + 0.5) * exposure_ms / 4


class AutoLightTests(unittest.TestCase):
    def test_policy_rejects_invalid_controls(self):
        for values in ({'light_min': 1}, {'light_min': float('nan')},
                       {'preferred_ms': 17}, {'rate_hz': 0}, {'max_step': 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                AutoLightPolicy(**values)

    def test_light_replaces_exposure_above_preference(self):
        controller = AutoLightController(AutoLightPolicy())
        decision = controller.step(9.0, 100_000_000, meter)
        self.assertAlmostEqual(decision['ideal_light'], 0.18)
        self.assertAlmostEqual(decision['next_light'], 0.18)
        self.assertLessEqual(decision['next_ms'], 8.0)
        self.assertAlmostEqual(decision['predicted_dn'], decision['target_dn'], delta=0.02)

    def test_rate_and_step_limit_hold_exposure_at_preference(self):
        controller = AutoLightController(AutoLightPolicy(max_step=0.2))
        first = controller.step(16.0, 100_000_000, meter)
        self.assertAlmostEqual(first['next_light'], 0.3)
        self.assertEqual(first['light_limit'], 'step')
        self.assertEqual(first['next_ms'], 8.0)
        second = controller.step(16.0, 200_000_000, meter)
        self.assertEqual(second['light_limit'], 'rate')
        self.assertAlmostEqual(second['next_light'], 0.3)
        self.assertEqual(second['next_ms'], 8.0)
        third = controller.step(16.0, 500_000_000, meter)
        self.assertAlmostEqual(third['next_light'], 0.5)
        self.assertEqual(third['light_limit'], 'step')

    def test_full_light_uses_exposure_above_preference(self):
        controller = AutoLightController(AutoLightPolicy())
        controller.light = 1.0
        decision = controller.step(12.0, 100_000_000, meter)
        self.assertAlmostEqual(decision['next_light'], 1.0)
        self.assertAlmostEqual(decision['next_ms'], 12.0, places=3)

    def test_reduces_exposure_when_light_cannot_decrease_yet(self):
        controller = AutoLightController(AutoLightPolicy())
        controller.light = 0.8
        controller.last_light_change_ns = 100_000_000
        decision = controller.step(4.0, 200_000_000, meter)
        self.assertAlmostEqual(decision['next_light'], 0.8)
        self.assertAlmostEqual(decision['next_ms'], 4.0, places=3)
        self.assertEqual(decision['light_limit'], 'rate')

    def test_light_search_does_not_assume_monotone_clid_response(self):
        def nonmonotone(light, exposure_ms):
            return (3000 if 0.3 <= light < 0.4 else 1000 + 1000 * light) * exposure_ms / 8
        controller = AutoLightController(AutoLightPolicy(light_min=0, max_step=1))
        decision = controller.step(16, 100_000_000, nonmonotone)
        self.assertAlmostEqual(decision['ideal_light'], 0.3)


if __name__ == '__main__':
    unittest.main()
