import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from applications.gridworld.agents.metamo import MetaMoAgent
from applications.gridworld.environment import GridWorld
from applications.gridworld.evaluation.neural_self_model import (
    COORDINATES, DIM, FEATURES, Dataset, ForecastInput, NeuralSettings,
    Scaling, balanced_mae, episode_dataset, fit_neural, freeze_input,
    neural_predict, rule_predictions, run_neural_experiment,
)


def synthetic_data(count, seed):
    rng = np.random.default_rng(seed)
    features = rng.uniform(-1.0, 1.0, (count, len(FEATURES)))
    nominal = np.full((count, DIM), 0.5)
    # A deterministic relation the optimizer should learn; no GridWorld assumptions.
    residual = 0.15 * np.tanh(2.0 * features[:, :1])
    target = nominal + residual * np.linspace(0.5, 1.0, DIM)
    return Dataset(features, nominal.copy(), nominal, target,
                   np.full(count, seed), np.arange(count), np.zeros(count, dtype=int))


class NeuralSelfModelDataTests(unittest.TestCase):
    def test_horizon_alignment_stays_inside_episode(self):
        inputs = [ForecastInput((i / 10,) * DIM, ((i + 1) / 10,) * DIM,
                                (0.0,) * len(FEATURES), 0) for i in range(6)]
        observed = [((i + 1) / 10,) * DIM for i in range(6)]
        data = episode_dataset(inputs, observed, episode=123, horizon=3)
        self.assertEqual(len(data), 4)
        np.testing.assert_allclose(data.target[0], 0.3)
        np.testing.assert_allclose(data.target[-1], 0.6)
        np.testing.assert_array_equal(data.episode, 123)
        np.testing.assert_allclose(data.nominal[0], 0.1)
        with self.assertRaises(ValueError):
            episode_dataset(inputs, observed, episode=123, horizon=7)

    def test_inputs_are_frozen_before_the_environment_moves(self):
        env = GridWorld(seed=701, max_steps=8)
        state = env.reset()
        agent = MetaMoAgent(seed=702, record_transitions=True)
        agent.reset_episode()
        action, alpha = agent.select_action(state)
        frozen = freeze_input(agent, action, alpha)
        self.assertEqual(env.step_count, 0)
        features = frozen.features
        following, _, done, _ = env.step(action)
        agent.commit_motivational_transition(next_env_state=following, done=done,
                                            executed_action=action)
        agent.mot.G[:] = 0
        self.assertEqual(frozen.features, features)
        self.assertEqual(len(features), len(FEATURES))
        self.assertNotEqual(frozen.current[:len(agent.mot.G)], tuple(agent.mot.G))

    def test_scaling_is_fitted_only_on_training_examples(self):
        train = synthetic_data(50, 1)
        validation = synthetic_data(25, 2)
        scaling = Scaling.fit(train)
        old_mean = scaling.input_mean.copy()
        validation.features[:] = 1000
        transformed = scaling.inputs(validation)
        self.assertGreater(float(transformed.min()), 100)
        np.testing.assert_array_equal(scaling.input_mean, old_mean)
        np.testing.assert_allclose(scaling.residual_mean, (train.target - train.nominal).mean(axis=0))

    def test_nominal_trend_extrapolates_known_transition(self):
        data = Dataset(np.zeros((1, len(FEATURES))), np.full((1, DIM), 0.2),
                       np.full((1, DIM), 0.3), np.full((1, DIM), 0.7),
                       np.array([1]), np.array([0]), np.array([0]))
        predictions = rule_predictions(data, horizon=5)
        np.testing.assert_allclose(predictions["nominal_trend"], data.target)
        self.assertLess(balanced_mae(predictions["nominal_trend"], data.target), 1e-12)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch dependency is absent")
class NeuralSelfModelTrainingTests(unittest.TestCase):
    def test_small_network_learns_a_known_nonlinear_change(self):
        import torch
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            train, validation = synthetic_data(256, 4), synthetic_data(96, 5)
            scaling = Scaling.fit(train)
            network, selection = fit_neural(
                train, validation, scaling,
                NeuralSettings(max_epochs=120, patience=25, batch_size=64), seed=6,
            )
            baseline = scaling.states(np.zeros_like(validation.target), validation.nominal)
            prediction = neural_predict(network, validation, scaling)
            self.assertLess(balanced_mae(prediction, validation.target),
                            0.7 * balanced_mae(baseline, validation.target))
            self.assertGreater(selection["best_epoch"], 0)
            self.assertTrue(np.all((prediction >= 0) & (prediction <= 1)))
        finally:
            torch.set_num_threads(previous_threads)

    def test_small_end_to_end_run_splits_episodes_and_saves_predictions(self):
        settings = NeuralSettings(
            splits=1, pretrain_episodes=2, train_episodes=3, validation_episodes=2,
            test_episodes=2, max_steps=8, horizon=3, max_epochs=4, patience=3,
            initializations=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            report = run_neural_experiment(settings, output_dir=folder)
            split = report["splits"][0]
            self.assertEqual(split["samples"], {"train": 18, "validation": 12, "test": 12})
            self.assertTrue(split["live_behavior_unchanged"])
            self.assertTrue(split["q_and_existing_self_model_frozen"])
            episodes = []
            for name in ("train", "validation", "test"):
                with np.load(folder / "split_1" / f"{name}.npz") as saved:
                    episodes.append(set(saved["episode"]))
            self.assertFalse(episodes[0] & episodes[1] or episodes[0] & episodes[2] or episodes[1] & episodes[2])
            self.assertTrue((folder / "split_1" / "model.npz").is_file())
            self.assertEqual(len(split["test_metrics"]["neural_32"]["coordinate_mae"]), len(COORDINATES))


if __name__ == "__main__":
    unittest.main()
