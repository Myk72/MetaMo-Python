import unittest
from dataclasses import replace

from applications.gridworld.agents.metamo import MetaMoAgent
from applications.gridworld.environment import GridWorld
from applications.gridworld.predictive_self_model import GridWorldPredictiveSelfModel
from applications.gridworld.transition_record import MotivationalSnapshot


class GridWorldPredictiveSelfModelTests(unittest.TestCase):
    def make_cycle(self, *, max_steps=5):
        env = GridWorld(seed=23, max_steps=max_steps)
        agent = MetaMoAgent(seed=23, record_transitions=True)
        agent.reset_episode()
        state = env.reset()
        action, _ = agent.select_action(state)
        return env, agent, action

    def test_forecast_is_frozen_before_environment_step(self):
        env, agent, action = self.make_cycle()
        pending = agent._pending_transition
        forecast = pending.forecast

        self.assertEqual(forecast.transition_id, pending.transition_id)
        self.assertEqual(forecast.predicted_committed_state, pending.staged_state)
        self.assertEqual(
            forecast.predicted_next_stimulus.hazard_pressure,
            pending.move_inputs.hazard_risk,
        )
        self.assertEqual(
            forecast.predicted_next_stimulus.safe_mobility,
            pending.move_inputs.safe_mobility_after,
        )
        expected_appraisal = agent.runtime.appraisal_profile.appraise(
            agent._pending_state, forecast.predicted_next_stimulus
        )
        self.assertEqual(
            forecast.predicted_next_appraisal,
            MotivationalSnapshot.from_state(expected_appraisal),
        )

        next_state, _, done, _ = env.step(action)
        agent.commit_motivational_transition(
            next_env_state=next_state, done=done, executed_action=action
        )
        record = agent.transition_records[0]
        self.assertIs(record.forecast, forecast)
        self.assertEqual(record.committed_state, forecast.predicted_committed_state)
        self.assertEqual(record.move_inputs, pending.move_inputs)
        self.assertEqual(
            forecast.predicted_next_stimulus.hazard_pressure,
            record.next_stimulus.hazard_pressure,
        )
        self.assertEqual(
            forecast.predicted_next_stimulus.safe_mobility,
            record.next_stimulus.safe_mobility,
        )
        expected_observed_appraisal = agent.runtime.appraisal_profile.appraise(
            agent.mot, record.next_stimulus
        )
        self.assertEqual(
            record.observed_next_appraisal,
            MotivationalSnapshot.from_state(expected_observed_appraisal),
        )
        errors = agent.predictive_self_model.evaluate(record)
        self.assertEqual(errors.committed_goal_mae, 0.0)
        self.assertEqual(errors.committed_modulator_mae, 0.0)
        self.assertIsNotNone(errors.next_appraisal_modulator_mae)
        self.assertGreaterEqual(errors.next_stimulus_mae, 0.0)

    def test_terminal_observation_has_no_next_appraisal_target(self):
        env, agent, action = self.make_cycle(max_steps=1)
        next_state, _, done, _ = env.step(action)
        self.assertTrue(done)
        agent.commit_motivational_transition(
            next_env_state=next_state, done=done, executed_action=action
        )
        record = agent.transition_records[0]
        self.assertIsNone(record.observed_next_appraisal)
        self.assertIsNone(
            GridWorldPredictiveSelfModel.evaluate(record).next_appraisal_modulator_mae
        )
        self.assertEqual(agent.predictive_self_model.sample_count(record.action_id), 0)

    def test_agent_learns_once_after_nonterminal_commit(self):
        env, agent, action = self.make_cycle()
        pending = agent._pending_transition
        self.assertEqual(agent.predictive_self_model.sample_count(pending.action_id), 0)
        next_state, _, done, _ = env.step(action)
        self.assertFalse(done)
        agent.commit_motivational_transition(
            next_env_state=next_state, done=done, executed_action=action
        )
        record = agent.transition_records[0]
        self.assertEqual(agent.predictive_self_model.sample_count(record.action_id), 1)
        self.assertAlmostEqual(
            agent.predictive_self_model.mean_change(record.action_id)[1],
            record.next_stimulus.hazard_pressure
            - record.current_stimulus.hazard_pressure,
        )
        agent.commit_motivational_transition(
            next_env_state=next_state, done=done, executed_action=action
        )
        self.assertEqual(agent.predictive_self_model.sample_count(record.action_id), 1)

    def test_forecast_evaluation_rejects_wrong_transition_id(self):
        env, agent, action = self.make_cycle()
        next_state, _, done, _ = env.step(action)
        agent.commit_motivational_transition(
            next_env_state=next_state, done=done, executed_action=action
        )
        record = agent.transition_records[0]
        wrong_forecast = replace(record.forecast, transition_id=record.transition_id + 1)
        with self.assertRaises(ValueError):
            agent.predictive_self_model.evaluate(
                replace(record, forecast=wrong_forecast)
            )


if __name__ == "__main__":
    unittest.main()
