import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from applications.gridworld.evaluation.neural_self_model import (
    DIM, FEATURES, Dataset, NeuralSettings, run_neural_experiment,
)
from applications.gridworld.evaluation.neural_robustness import (
    FrozenForecasters, RobustnessSettings, _seed_plan, group_report,
    paired_comparison, run_robustness, situation_masks,
)


def fixture_data(count=4):
    return Dataset(
        np.zeros((count, len(FEATURES))), np.full((count, DIM), 0.4),
        np.full((count, DIM), 0.5), np.full((count, DIM), 0.6),
        np.arange(count), np.zeros(count, dtype=int), np.zeros(count, dtype=int),
    )


class NeuralRobustnessTests(unittest.TestCase):
    def test_groups_have_explicit_boundaries_and_allow_overlap(self):
        data = fixture_data()
        data.features[1:3, FEATURES.index("move.mineral_collected")] = 1
        data.features[2, FEATURES.index("stimulus.hazard_pressure")] = 0.4
        data.features[3, FEATURES.index("move.hazard_risk")] = 0.75
        data.target[3] = 0.9
        masks = situation_masks(data, threshold=0.3)
        np.testing.assert_array_equal(masks["ordinary"], [True, False, False, False])
        np.testing.assert_array_equal(masks["collection"], [False, True, True, False])
        np.testing.assert_array_equal(masks["near_lava"], [False, False, True, True])
        np.testing.assert_array_equal(masks["large_change"], [False, False, False, True])

    def test_paired_comparison_resamples_episode_means_not_windows(self):
        data = fixture_data()
        data.episode[:] = [1, 1, 1, 2]
        neural = data.target.copy()
        reference = data.target.copy()
        reference[:3] += 0.1
        reference[3] += 0.3
        report = paired_comparison(neural, reference, data, RobustnessSettings(), seed=3)
        self.assertAlmostEqual(report["mean_episode_mae_reduction"], 0.2)
        self.assertFalse(report["coverage_sufficient"])
        self.assertEqual(report["episode_win_fraction"], 1)
        self.assertGreater(report["bootstrap_95_percent_interval"][0], 0)

    def test_absent_and_single_episode_groups_are_not_claimed_supported(self):
        data = fixture_data(1)
        report = group_report(data, {"nominal_hold": data.nominal, "ridge": data.nominal,
                                    "neural_32": data.target}, 1.0, "nominal_hold",
                              RobustnessSettings(), seed=1)
        self.assertEqual(report["collection"]["status"], "no observations")
        comparison = report["all"]["comparisons"]["ridge"]
        self.assertIsNone(comparison["bootstrap_95_percent_interval"])
        self.assertFalse(comparison["coverage_sufficient"])

    def test_old_or_repeated_episode_seeds_are_rejected(self):
        with self.assertRaises(ValueError):
            _seed_plan(NeuralSettings(), RobustnessSettings(seed_start=7000000))
        with self.assertRaises(ValueError):
            RobustnessSettings(episodes_per_split=60, seed_stride=20).validate()
        ranges, old = _seed_plan(NeuralSettings(), RobustnessSettings())
        self.assertFalse(old.intersection(seed for block in ranges for seed in block))

    def test_saved_inference_is_bounded_and_does_not_read_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.npz"
            weights = {
                "input_mean": np.zeros(len(FEATURES)), "input_scale": np.ones(len(FEATURES)),
                "residual_mean": np.zeros(DIM), "residual_scale": np.ones(DIM),
                "ridge": np.zeros((len(FEATURES) + 1, DIM)),
                "neural.0.weight": np.ones((2, len(FEATURES)), dtype=np.float32),
                "neural.0.bias": np.zeros(2, dtype=np.float32),
                "neural.2.weight": np.ones((DIM, 2), dtype=np.float32),
                "neural.2.bias": np.zeros(DIM, dtype=np.float32),
            }
            np.savez(path, **weights)
            model = FrozenForecasters(path, horizon=5)
            data = fixture_data()
            data.features[0] = 1
            prediction = model.predict(data)
            data.target[:] = np.nan
            again = model.predict(data)
            for name in prediction:
                np.testing.assert_array_equal(prediction[name], again[name])
                self.assertTrue(np.all((again[name] >= 0) & (again[name] <= 1)))
            self.assertFalse(model.arrays["ridge"].flags.writeable)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch dependency is absent")
    def test_end_to_end_uses_saved_models_and_disjoint_new_episodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "robustness"
            original = NeuralSettings(
                splits=1, pretrain_episodes=2, train_episodes=3, validation_episodes=2,
                test_episodes=2, max_steps=8, horizon=3, max_epochs=2, patience=2,
                initializations=1,
            )
            run_neural_experiment(original, output_dir=source)
            before = (source / "split_1" / "model.npz").read_bytes()
            report = run_robustness(source, RobustnessSettings(episodes_per_split=2,
                                    bootstrap_samples=20), output_dir=output)
            self.assertEqual(report["aggregate_groups"]["all"]["windows"], 12)
            self.assertEqual(report["fresh_lava_layouts_verified"], 2)
            self.assertTrue(report["splits"][0]["live_behavior_unchanged"])
            self.assertTrue(report["splits"][0]["original_training_episode_replayed_exactly"])
            self.assertTrue(report["saved_artifacts_unchanged"])
            self.assertEqual(before, (source / "split_1" / "model.npz").read_bytes())
            self.assertTrue((output / "protocol.json").is_file())
            self.assertTrue((output / "split_1" / "groups.npz").is_file())


if __name__ == "__main__":
    unittest.main()
