import unittest

import numpy as np

from applications.gridworld.agents.metamo import MetaMoAgent
from applications.gridworld.environment import GridWorld
from applications.gridworld.evaluation.runner import VariantSpec, _run_episode
from applications.gridworld.runtime import build_stimulus
from applications.gridworld.transition_record import MotivationalSnapshot


class GridWorldTransitionRecordingTests(unittest.TestCase):
    def make_pair(self, *, record_transitions=True, seed=17):
        env = GridWorld(seed=seed, max_steps=8)
        agent = MetaMoAgent(seed=seed, record_transitions=record_transitions)
        agent.reset_episode()
        return env, agent, env.reset()

    def test_training_update_pairs_one_selected_transition_with_observation(self):
        env, agent, state = self.make_pair()
        before = MotivationalSnapshot.from_state(agent.mot)
        current_stimulus = build_stimulus(state, agent.mot)
        action, alpha = agent.select_action(state)
        pending = agent._pending_transition
        staged = MotivationalSnapshot.from_state(agent._pending_state)
        next_state, reward, done, info = env.step(action)

        agent.update(state, action, reward, next_state, done, info.get("event"), alpha)

        self.assertEqual(len(agent.transition_records), 1)
        record = agent.transition_records[0]
        self.assertEqual(record.transition_id, pending.transition_id)
        self.assertEqual(record.action_index, action)
        self.assertEqual(record.action_id, pending.action_id)
        self.assertEqual(record.current_state, before)
        self.assertEqual(record.current_stimulus, current_stimulus)
        self.assertEqual(record.staged_state, staged)
        self.assertEqual(record.committed_state, MotivationalSnapshot.from_state(agent.mot))
        self.assertEqual(record.next_stimulus, build_stimulus(next_state, agent.mot))
        self.assertEqual(record.done, done)
        self.assertIsNone(agent._pending_transition)
        agent.mot.G[0] = 0.0
        self.assertEqual(record.committed_state, staged)

    def test_evaluation_commit_records_once_and_reset_starts_new_episode(self):
        env, agent, state = self.make_pair()
        action, _ = agent.select_action(state)
        next_state, _, done, _ = env.step(action)
        agent.commit_motivational_transition(
            next_env_state=next_state, done=done, executed_action=action
        )
        agent.commit_motivational_transition(
            next_env_state=next_state, done=done, executed_action=action
        )
        self.assertEqual(len(agent.transition_records), 1)
        first = agent.transition_records[0]

        agent.reset_episode()
        state = env.reset()
        action, _ = agent.select_action(state)
        next_state, _, done, _ = env.step(action)
        agent.commit_motivational_transition(
            next_env_state=next_state, done=done, executed_action=action
        )
        second = agent.transition_records[1]
        self.assertEqual(len(agent.transition_records), 2)
        self.assertNotEqual(first.transition_id, second.transition_id)
        self.assertNotEqual(first.episode_id, second.episode_id)
        self.assertEqual(second.step_index, 0)

    def test_evaluation_runner_passes_outcome_to_recorder(self):
        agent = MetaMoAgent(seed=11, record_transitions=True)
        variant = VariantSpec(
            name="recording_test",
            kind="metamo",
            factory=lambda seed: agent,
        )
        log = _run_episode(
            agent,
            variant,
            env_seed=11,
            max_steps=3,
            danger_distance=2,
            danger_mineral_probability=0.5,
            train=False,
        )
        self.assertEqual(len(agent.transition_records), log.total_steps)
        self.assertEqual(
            [record.step_index for record in agent.transition_records],
            list(range(log.total_steps)),
        )

    def test_missing_or_mismatched_outcome_does_not_commit(self):
        env, agent, state = self.make_pair()
        action, _ = agent.select_action(state)
        original = MotivationalSnapshot.from_state(agent.mot)
        next_state, _, done, _ = env.step(action)

        with self.assertRaises(ValueError):
            agent.commit_motivational_transition()
        with self.assertRaises(ValueError):
            agent.commit_motivational_transition(
                next_env_state=next_state, done=done, executed_action=(action + 1) % 4
            )
        with self.assertRaises(RuntimeError):
            agent.select_action(next_state)
        self.assertEqual(MotivationalSnapshot.from_state(agent.mot), original)
        self.assertEqual(agent.transition_records, [])

        agent.commit_motivational_transition(
            next_env_state=next_state, done=done, executed_action=action
        )
        self.assertEqual(len(agent.transition_records), 1)

    def test_recording_does_not_change_training_trajectory(self):
        env_plain, plain, state_plain = self.make_pair(record_transitions=False)
        env_recorded, recorded, state_recorded = self.make_pair(record_transitions=True)
        for _ in range(4):
            self.assertEqual(state_plain, state_recorded)
            action_plain, alpha_plain = plain.select_action(state_plain)
            action_recorded, alpha_recorded = recorded.select_action(state_recorded)
            self.assertEqual(action_plain, action_recorded)
            next_plain, reward_plain, done_plain, info_plain = env_plain.step(action_plain)
            next_recorded, reward_recorded, done_recorded, info_recorded = (
                env_recorded.step(action_recorded)
            )
            self.assertEqual((next_plain, reward_plain, done_plain),
                             (next_recorded, reward_recorded, done_recorded))
            plain.update(
                state_plain, action_plain, reward_plain, next_plain,
                done_plain, info_plain.get("event"), alpha_plain,
            )
            recorded.update(
                state_recorded, action_recorded, reward_recorded, next_recorded,
                done_recorded, info_recorded.get("event"), alpha_recorded,
            )
            np.testing.assert_array_equal(plain.mot.G, recorded.mot.G)
            np.testing.assert_array_equal(plain.mot.M, recorded.mot.M)
            self.assertEqual(plain.q_table.keys(), recorded.q_table.keys())
            for key in plain.q_table:
                np.testing.assert_array_equal(plain.q_table[key], recorded.q_table[key])
            state_plain, state_recorded = next_plain, next_recorded
            if done_plain:
                break
        self.assertEqual(len(recorded.transition_records), env_recorded.step_count)


if __name__ == "__main__":
    unittest.main()
