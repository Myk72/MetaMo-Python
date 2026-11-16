import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from applications.gridworld.evaluation.neural_self_model import (
    DIM, FEATURES, Dataset, NeuralSettings, Scaling, fit_neural, neural_predict, run_neural_experiment,
)
from applications.gridworld.evaluation.neural_robustness import RobustnessSettings, run_robustness
from applications.gridworld.evaluation.neural_weighted import (
    WeightingSettings, _fresh_again, choose_strength, point_estimate_success, run_weighting, training_weights,
)


def data_fixture():
    rng = np.random.default_rng(21)
    features = rng.normal(size=(40, len(FEATURES)))
    current = np.full((40, DIM), 0.4)
    nominal = np.full((40, DIM), 0.5)
    target = np.full((40, DIM), 0.55)
    target[-4:] = 0.9
    return Dataset(features, current, nominal, target, np.repeat([1, 2], 20),
                   np.tile(np.arange(20), 2), np.zeros(40, dtype=int))


class WeightingDesignTests(unittest.TestCase):
    def test_only_large_change_training_rows_get_extra_weight(self):
        data = data_fixture()
        weights = training_weights(data, threshold=0.3, strength=4)
        np.testing.assert_array_equal(weights[:36], 1)
        np.testing.assert_array_equal(weights[36:], 4)
        np.testing.assert_array_equal(training_weights(data, 0.3, 1), 1)
        with self.assertRaises(ValueError):
            training_weights(data, 0.3, float("nan"))

    def test_selection_obeys_both_guards_and_can_reject_weighting(self):
        base = {"strength": 1.0, "validation": {"all": 1, "ordinary": 1, "large_change": 2}}
        # Strong large-change improvement, but each candidate violates a guard.
        worse_ordinary = {"strength": 4.0, "validation": {"all": 0.8, "ordinary": 1.03, "large_change": 0.5}}
        worse_overall = {"strength": 8.0, "validation": {"all": 1.03, "ordinary": 0.9, "large_change": 0.4}}
        self.assertEqual(choose_strength([base, worse_ordinary, worse_overall], WeightingSettings()), 1)
        eligible = {"strength": 2.0, "validation": {"all": 1.01, "ordinary": 1.01, "large_change": 1.5}}
        self.assertEqual(choose_strength([base, worse_ordinary, eligible], WeightingSettings()), 2)
        self.assertTrue(point_estimate_success(eligible["validation"], base["validation"], WeightingSettings()))
        self.assertFalse(point_estimate_success(base["validation"], base["validation"], WeightingSettings()))

    def test_fresh_test_must_exclude_prior_robustness_episodes(self):
        previous = {"splits": [{"fresh_seed_start": 9000000, "fresh_seed_stop_exclusive": 9000060,
                                 "lava_layout_sha256": []}]}
        with self.assertRaises(ValueError):
            _fresh_again(NeuralSettings(), RobustnessSettings(seed_start=9000000), previous)
        with self.assertRaises(ValueError):
            WeightingSettings(strengths=(2.0, 4.0)).validate()


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch dependency is absent")
class WeightedTrainingTests(unittest.TestCase):
    def test_all_one_control_replicates_and_nonuniform_weights_change_fit(self):
        import torch
        before = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            train = data_fixture()
            scaling = Scaling.fit(train)
            settings = NeuralSettings(max_epochs=3, patience=3)
            control, _ = fit_neural(train, train, scaling, settings, seed=101)
            identical, _ = fit_neural(train, train, scaling, settings, seed=101,
                                      sample_weights=np.ones(len(train)))
            weighted, _ = fit_neural(train, train, scaling, settings, seed=101,
                                     sample_weights=training_weights(train, 0.3, 4))
            expected = neural_predict(control, train, scaling)
            np.testing.assert_array_equal(expected, neural_predict(identical, train, scaling))
            self.assertGreater(np.max(np.abs(expected - neural_predict(weighted, train, scaling))), 1e-6)
            with self.assertRaises(ValueError):
                fit_neural(train, train, scaling, settings, seed=1, sample_weights=np.zeros(len(train)))
        finally:
            torch.set_num_threads(before)

    def test_small_end_to_end_selection_is_saved_before_fresh_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, previous, destination = root / "original", root / "previous", root / "weighted"
            settings = NeuralSettings(
                splits=1, pretrain_episodes=2, train_episodes=4, validation_episodes=4,
                test_episodes=2, max_steps=10, horizon=3, max_epochs=3, patience=3, initializations=1,
            )
            run_neural_experiment(settings, output_dir=source)
            run_robustness(source, RobustnessSettings(episodes_per_split=2, bootstrap_samples=20),
                           output_dir=previous)
            report = run_weighting(source, previous / "protocol.json", destination,
                                   settings=WeightingSettings(strengths=(1.0, 2.0)),
                                   evaluation=RobustnessSettings(episodes_per_split=3,
                                                                 seed_start=9000000, bootstrap_samples=20))
            self.assertTrue(report["original_artifacts_unchanged"])
            self.assertTrue(report["live_behavior_unchanged"])
            self.assertTrue(report["q_and_existing_self_model_frozen"])
            self.assertEqual(report["fresh_lava_layouts_verified"], 3)
            self.assertEqual(report["aggregate_groups"]["all"]["windows"], 24)
            selection = destination / "selections.json"
            fresh = destination / "fresh_reference" / "split_1" / "fresh.npz"
            self.assertLessEqual(selection.stat().st_mtime_ns, fresh.stat().st_mtime_ns)
            chosen = report["selections"][0]["selected_strength"]
            with np.load(destination / "split_1" / "predictions.npz") as saved:
                np.testing.assert_array_equal(saved["weighted_selected"], saved[f"weight_{chosen:g}"])
                np.testing.assert_array_equal(saved["weight_1"], saved["neural_32"])


if __name__ == "__main__":
    unittest.main()
