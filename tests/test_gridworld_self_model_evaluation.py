import math
import unittest

from applications.gridworld.evaluation.self_model import run_self_model_evaluation


class GridWorldSelfModelEvaluationTests(unittest.TestCase):
    def test_held_out_run_freezes_learning_and_preserves_behavior(self):
        report = run_self_model_evaluation(
            train_seeds=(101, 102, 103),
            held_out_seeds=(201, 202),
            max_steps=6,
            agent_seed=19,
        )
        self.assertEqual(report.training_episodes, 3)
        self.assertEqual(report.held_out_episodes, 2)
        self.assertGreater(report.training_steps, 0)
        self.assertGreater(report.scored_steps, 0)
        self.assertLess(report.scored_steps, report.held_out_steps)
        self.assertGreater(sum(report.training_samples_by_action.values()), 0)
        self.assertEqual(report.model_updates_during_held_out, 0)
        self.assertEqual(report.collection_updates_during_held_out, 0)
        self.assertEqual(report.action_only_updates_during_held_out, 0)
        self.assertEqual(report.contextual_updates_during_held_out, 0)
        self.assertLessEqual(report.contextual_forecast_steps, report.scored_steps)
        self.assertTrue(report.behavior_unchanged)
        self.assertTrue(report.safety_passed)
        self.assertEqual(report.committed_goal_mae, 0.0)
        self.assertEqual(report.committed_modulator_mae, 0.0)
        for error in (
            report.learned_stimulus_mae,
            report.nominal_stimulus_mae,
            report.contextual_stimulus_mae,
            report.action_only_stimulus_mae,
            report.persistence_stimulus_mae,
            report.learned_appraisal_mae,
            report.nominal_appraisal_mae,
            report.contextual_appraisal_mae,
            report.action_only_appraisal_mae,
            report.persistence_appraisal_mae,
        ):
            self.assertTrue(math.isfinite(error))
            self.assertGreaterEqual(error, 0.0)
        self.assertEqual(len(report.stimulus_feature_mae), 6)
        self.assertTrue(all(len(errors) == 5 for errors in report.stimulus_feature_mae.values()))

    def test_rejects_overlapping_or_duplicate_seeds(self):
        for training, held_out in (
            ((1,), (1,)),
            ((1, 1), (2,)),
            ((1,), (2, 2)),
        ):
            with self.subTest(training=training, held_out=held_out):
                with self.assertRaises(ValueError):
                    run_self_model_evaluation(training, held_out, max_steps=3)

    def test_sparse_model_matches_persistence_reference(self):
        report = run_self_model_evaluation(
            train_seeds=(301,),
            held_out_seeds=(401,),
            max_steps=4,
            agent_seed=7,
        )
        self.assertTrue(
            all(count < 10 for count in report.training_samples_by_action.values())
        )
        self.assertEqual(report.contextual_stimulus_mae, report.persistence_stimulus_mae)
        self.assertEqual(report.contextual_appraisal_mae, report.persistence_appraisal_mae)
        self.assertEqual(report.action_only_stimulus_mae, report.persistence_stimulus_mae)
        self.assertEqual(report.action_only_appraisal_mae, report.persistence_appraisal_mae)
        self.assertLess(report.learned_stimulus_mae, report.persistence_stimulus_mae)


if __name__ == "__main__":
    unittest.main()
