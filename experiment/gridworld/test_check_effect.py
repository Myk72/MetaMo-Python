"""Small invariants for the targeted appraisal-effect replay."""

import unittest

import numpy as np

from experiment.gridworld.check_effect import (
    ContextTargetPredictor,
    _environment_state,
    _max_shifts,
)


class EffectCheckTests(unittest.TestCase):
    def test_exact_target_only_matches_recorded_context(self):
        predictor = ContextTargetPredictor(np.array([1.0, 2.0]), np.array([0.2]))
        np.testing.assert_array_equal(predictor.predict(np.array([1.0, 2.0])), [0.2])
        np.testing.assert_array_equal(predictor.predict(np.array([1.0, 2.1])), [0.0])
        self.assertEqual(predictor.matches, 1)

    def test_reconstructed_lava_cells_support_membership(self):
        state = _environment_state({
            "pos": [1, 2], "mineral_pos": [3, 4],
            "lava_cells": [[2, 2], [4, 4]],
        })
        self.assertIn((2, 2), state["lava_cells"])
        self.assertEqual(state["pos"], (1, 2))

    def test_max_shift_reports_largest_probe(self):
        keys = (
            "exact_task_score_max_shift", "exact_safety_score_max_shift",
            "exact_consensus_score_max_shift", "exact_normalized_task_max_shift",
            "exact_normalized_safety_max_shift", "exact_score_max_shift",
            "nn_score_max_shift",
        )
        result = _max_shifts([{key: value for key in keys} for value in (0.1, 0.2)])
        self.assertTrue(all(value == 0.2 for value in result.values()))


if __name__ == "__main__":
    unittest.main()
