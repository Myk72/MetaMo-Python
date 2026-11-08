import unittest

from applications.gridworld.agents.metamo import MetaMoAgent
from applications.gridworld.environment import GridWorld
from applications.gridworld.evaluation.future_conflict import (
    BranchResult,
    CandidateProbe,
    DisagreementCase,
    FutureConflictModel,
    _branch,
    _choice_audit,
    run_future_conflict_evaluation,
)


class FutureConflictTests(unittest.TestCase):
    def test_predictor_only_changes_after_completed_target(self):
        model = FutureConflictModel(min_context_samples=2, prior_count=0)
        probe = CandidateProbe(0, "UP", 0.8, 0.5, 0.4, 0.1, 3.0, False, False)
        self.assertEqual(model.predict(probe, contextual=True), 0.8)
        model.observe(probe, 0.4)
        model.observe(probe, 0.6)
        self.assertAlmostEqual(model.predict(probe, contextual=True), 0.5)
        self.assertEqual(model.pooled.count, 2)

    def test_choice_audit_links_delayed_collection_to_paired_loss(self):
        learned = CandidateProbe(0, "UP", 0.4, 0.2, 0.1, 0.3, 2.0, False, False)
        reference = CandidateProbe(1, "DOWN", 0.8, 0.4, 0.7, 0.1, 3.0, True, False)
        case = DisagreementCase(
            learned_probe=learned,
            reference_probe=reference,
            learned_forecast=0.4,
            reference_forecast=0.8,
            learned_outcome=BranchResult(0.4, -0.5, -5.0, 0, 0, 0),
            reference_outcome=BranchResult(0.8, 99.5, 95.0, 1, 0, 0),
        )
        audit = _choice_audit([case])
        self.assertEqual(audit.groups["collection_delayed"].count, 1)
        self.assertEqual(audit.reward_losses_without_safety_improvement, 1)
        self.assertEqual(audit.mean_task_score_difference, -0.6)
        self.assertEqual(audit.mean_actual_conflict_difference, -0.4)

    def test_branch_does_not_advance_live_agent_or_environment(self):
        env = GridWorld(seed=19, max_steps=8)
        state = env.reset()
        agent = MetaMoAgent(seed=19)
        agent.reset_episode()
        action, _ = agent.select_action(state)
        pending = agent._pending_state
        before_step = env.step_count
        result = _branch(
            agent, env, state, action, forecast_horizon=2, rollout_horizon=3
        )
        self.assertIs(agent._pending_state, pending)
        self.assertEqual(env.step_count, before_step)
        self.assertIsNotNone(result.future_conflict)

    def test_small_evaluation_is_frozen_and_has_labels(self):
        report = run_future_conflict_evaluation(
            split_count=1,
            pretrain_episodes=2,
            calibration_episodes=2,
            held_out_episodes=2,
            max_steps=8,
            forecast_horizon=2,
            rollout_horizon=3,
        )
        self.assertGreater(report.total_held_out_labels, 0)
        self.assertEqual(report.splits[0].model_updates_during_test, 0)
        self.assertEqual(report.splits[0].live_action_mismatches, 0)


if __name__ == "__main__":
    unittest.main()
