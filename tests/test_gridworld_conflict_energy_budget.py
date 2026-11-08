import unittest

from applications.gridworld.environment import GridWorld
from applications.gridworld.evaluation.conflict_energy_budget import (
    EnergyBudgetGridWorld, EnergySettings, _auc, run_energy_experiment,
)


class ConflictEnergyBudgetTests(unittest.TestCase):
    def test_energy_cost_and_depletion_penalty_apply_once(self):
        env = EnergyBudgetGridWorld(seed=11, initial_energy=10, max_steps=20)
        env.lava_cells = {(9, 9)}
        env.mineral_pos = [8, 8]
        state, reward, done, _ = env.step(3)
        self.assertEqual(state["energy"], 5)
        self.assertEqual(reward, -0.5)
        self.assertFalse(done)
        state, reward, done, info = env.step(3)
        self.assertEqual(state["energy"], 0)
        self.assertEqual(reward, -100.5)
        self.assertTrue(done)
        self.assertEqual(info["event"], "dead")
        self.assertEqual(env.step(3)[1], 0)
        self.assertEqual(env.reset()["energy"], 10)

    def test_collection_recharges_before_final_move_cost(self):
        env = EnergyBudgetGridWorld(seed=12, initial_energy=5, max_steps=20)
        env.lava_cells = {(9, 9)}
        env.mineral_pos = [0, 1]
        state, reward, done, info = env.step(3)
        self.assertEqual(state["energy"], 95)
        self.assertEqual(reward, 99.5)
        self.assertFalse(done)
        self.assertEqual(info["event"], "mineral")

    def test_time_limit_is_not_depletion_and_standard_world_is_unchanged(self):
        env = EnergyBudgetGridWorld(seed=13, max_steps=1)
        env.mineral_pos = [8, 8]
        state, reward, done, info = env.step(3)
        self.assertEqual(state["energy"], 55)
        self.assertEqual(reward, -0.5)
        self.assertTrue(done)
        self.assertNotEqual(info["event"], "dead")
        original = GridWorld(seed=13, max_steps=1)
        original.mineral_pos = [8, 8]
        self.assertEqual(original.step(3)[0]["energy"], 100)

    def test_warning_auc_direction_ties_and_missing_class(self):
        self.assertEqual(_auc([0.1, 0.9], [False, True]), 1)
        self.assertEqual(_auc([0.9, 0.1], [False, True]), 0)
        self.assertEqual(_auc([0.5, 0.5], [False, True]), 0.5)
        self.assertIsNone(_auc([0.2, 0.3], [False, False]))

    def test_small_experiment_keeps_deaths_in_safety_data_and_freezes_models(self):
        result = run_energy_experiment(EnergySettings(
            splits=1, pretrain_episodes=2, calibration_episodes=2, test_episodes=2,
            max_steps=12, forecast_horizon=2, continuation_moves=3,
            audit_every=2, initial_energy=20,
        ))
        split = result["splits"][0]
        self.assertTrue(split["live_trajectory_parity"])
        self.assertTrue(split["q_and_model_frozen"])
        self.assertGreater(split["candidate_branches"], 0)
        self.assertEqual(
            split["candidate_branches"],
            split["depleted_branches"] + split["surviving_branches"],
        )
        self.assertGreater(split["depleted_branches"], 0)
        self.assertEqual(len(result["held_out_rows"]), split["candidate_branches"])
        self.assertGreater(split["conflict_labels_missing_due_to_early_depletion"], 0)


if __name__ == "__main__":
    unittest.main()
