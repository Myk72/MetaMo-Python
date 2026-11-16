import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from applications.gridworld.evaluation.neural_self_model import (
    DIM, FEATURES, Dataset, NeuralSettings, Scaling, run_neural_experiment,
)
from applications.gridworld.evaluation.neural_robustness import FrozenForecasters, load_dataset
from applications.gridworld.evaluation.neural_learning_curve import (
    CurveSettings, check_freshness, improvement_gate, nested_training,
    run_learning_curve, seed_plan,
)


def fixture(episodes):
    ids = np.repeat(episodes, 3)
    count = len(ids)
    return Dataset(np.zeros((count, len(FEATURES))), np.full((count, DIM), 0.4),
                   np.full((count, DIM), 0.5), np.full((count, DIM), 0.6),
                   ids, np.tile(np.arange(3), len(episodes)), np.zeros(count, dtype=int))


class LearningCurveDesignTests(unittest.TestCase):
    def test_subsets_are_nested_whole_episodes_and_preserve_base(self):
        base, extra = fixture([10, 11]), fixture([20, 21, 22, 23])
        small = nested_training(base, extra, 2)
        middle = nested_training(base, extra, 4)
        large = nested_training(base, extra, 6)
        self.assertIs(small, base)
        np.testing.assert_array_equal(middle.episode, [10]*3 + [11]*3 + [20]*3 + [21]*3)
        for field in Dataset.__dataclass_fields__:
            np.testing.assert_array_equal(getattr(middle, field), getattr(large, field)[:len(middle)])
        with self.assertRaises(ValueError):
            nested_training(base, fixture([11, 20]), 3)

    def test_seed_plan_excludes_old_data_and_keeps_roles_apart(self):
        original = NeuralSettings()
        settings = CurveSettings()
        plans = seed_plan(original, settings)
        seeds = [seed for plan in plans for block in plan.values() for seed in block]
        self.assertEqual(len(seeds), len(set(seeds)))
        with self.assertRaises(ValueError):
            check_freshness(original, seed_plan(original, CurveSettings(seed_start=7000000)), [])
        previous = {"splits": [{"fresh_seed_start": settings.seed_start,
                                 "fresh_seed_stop_exclusive": settings.seed_start + 60}]}
        with self.assertRaises(ValueError):
            check_freshness(original, plans, [previous])
        with self.assertRaises(ValueError):
            CurveSettings(train_sizes=(80, 80, 320)).validate(original)

    def test_success_requires_large_change_gain_and_ordinary_guard(self):
        def group(small, big, ridge):
            return {"windows": 10, "metrics": {
                "neural_80": {"balanced_mae": small}, "neural_320": {"balanced_mae": big},
                "ridge_320": {"balanced_mae": ridge}}}
        groups = {"all": group(1, 0.9, 1), "ordinary": group(1, 1.01, 1),
                  "large_change": group(2, 1.5, 1.6)}
        self.assertTrue(improvement_gate(groups, 80, 320, 0.02)["more_data_helps"])
        groups["ordinary"] = group(1, 1.03, 1)
        self.assertFalse(improvement_gate(groups, 80, 320, 0.02)["more_data_helps"])
        groups["large_change"]["windows"] = 0
        self.assertIsNone(improvement_gate(groups, 80, 320, 0.02)["more_data_helps"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch dependency is absent")
    def test_end_to_end_sizes_share_test_and_use_only_their_training_for_scaling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / "source", root / "curve"
            original = NeuralSettings(splits=1, pretrain_episodes=2, train_episodes=3,
                                      validation_episodes=2, test_episodes=2, max_steps=10, horizon=3,
                                      max_epochs=2, patience=2, initializations=1)
            run_neural_experiment(original, output_dir=source)
            report = run_learning_curve(source, (), destination, settings=CurveSettings(
                train_sizes=(3, 5, 7), validation_episodes=3, test_episodes=3, bootstrap_samples=20))
            folder = destination / "split_1"
            base = load_dataset(source / "split_1" / "train.npz")
            extra, validation, test = [load_dataset(folder / name) for name in
                                       ("extra_training.npz", "validation.npz", "test.npz")]
            roles = [set(data.episode) for data in (base, extra, validation, test)]
            self.assertEqual(sum(map(len, roles)), len(set.union(*roles)))
            self.assertEqual(report["aggregate_groups"]["all"]["windows"], 24)
            self.assertTrue(report["source_artifacts_unchanged"])
            self.assertTrue(report["splits"][0]["q_and_existing_self_model_frozen"])
            self.assertTrue(report["splits"][0]["live_behavior_unchanged"])
            self.assertLessEqual((destination / "selections.json").stat().st_mtime_ns,
                                 (folder / "test.npz").stat().st_mtime_ns)
            with np.load(folder / "predictions.npz") as predictions:
                for size in (3, 5, 7):
                    frozen = FrozenForecasters(folder / f"model_{size}.npz", original.horizon)
                    expected = Scaling.fit(nested_training(base, extra, size))
                    np.testing.assert_array_equal(frozen.scaling.input_mean, expected.input_mean)
                    inference = frozen.predict(test)
                    np.testing.assert_allclose(inference["neural_32"], predictions[f"neural_{size}"], atol=1e-6)
                    np.testing.assert_allclose(inference["ridge"], predictions[f"ridge_{size}"], atol=1e-12)
            saved = json.loads((folder / "selection.json").read_text())
            self.assertEqual([row["train_windows"] for row in saved["sizes"]], [24, 40, 56])


if __name__ == "__main__":
    unittest.main()
