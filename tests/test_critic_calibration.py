import unittest
from types import SimpleNamespace

from scripts.evaluate_critic_calibration import (
    calibration_rollout_config,
    metrics,
    recommended_leaf_weight,
)


class CriticCalibrationTest(unittest.TestCase):
    def test_cli_horizon_controls_actual_monte_carlo_rollout(self):
        config = calibration_rollout_config(
            SimpleNamespace(horizon_battles=10, max_continuation_steps=500)
        )
        self.assertEqual(config.horizon_battles, 10)
        self.assertEqual(config.max_continuation_steps, 500)
        self.assertEqual(config.leaf_value_weight, 0.0)

    def test_reports_calibration_and_recommends_weight(self):
        good_rows = [
            {"predicted": value + 0.05, "monteCarlo": value}
            for value in (-1.0, -0.5, 0.0, 0.5, 1.0)
        ]
        good = metrics(good_rows)
        self.assertGreater(good["correlation"], 0.99)
        self.assertLess(good["mae"], 0.1)
        self.assertEqual(recommended_leaf_weight(good), 1.0)
        optimistic_slice = dict(good, states=30, bias=0.4)
        self.assertEqual(
            recommended_leaf_weight(good, [optimistic_slice]), 0.5
        )

        bad_rows = [
            {"predicted": predicted, "monteCarlo": actual}
            for predicted, actual in zip(
                (1.0, 0.5, 0.0, -0.5, -1.0),
                (-1.0, -0.5, 0.0, 0.5, 1.0),
            )
        ]
        bad = metrics(bad_rows)
        self.assertLess(bad["correlation"], 0.0)
        self.assertEqual(recommended_leaf_weight(bad), 0.25)


if __name__ == "__main__":
    unittest.main()
