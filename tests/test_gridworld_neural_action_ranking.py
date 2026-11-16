import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from applications.gridworld.agents.metamo import MetaMoAgent
from applications.gridworld.environment import GridWorld
from applications.gridworld.evaluation.future_conflict import _probes
from applications.gridworld.evaluation.neural_self_model import DIM, FEATURES, freeze_input
from applications.gridworld.evaluation.neural_robustness import FrozenForecasters
from applications.gridworld.evaluation.neural_action_ranking import (
    RankingSettings, UtilityReadout, branch, candidate_data, choose, collect,
    evaluate, fit_readouts, ranking_metrics, vector,
)
from applications.gridworld.evaluation.neural_self_model import NeuralSettings


class ActionRankingTests(unittest.TestCase):
    def agent(self):
        agent = MetaMoAgent(seed=54, record_transitions=True)
        agent.epsilon = 0
        agent.predictive_self_model.learning_enabled = False
        agent.reset_episode()
        return agent

    def test_candidate_inputs_match_selected_forecast_and_change_action(self):
        env = GridWorld(seed=51, max_steps=12)
        state = env.reset()
        agent = self.agent()
        action, alpha = agent.select_action(state)
        probes = _probes(copy.deepcopy(agent), state)
        data = candidate_data(agent, state, probes)
        row = list(data.action).index(action)
        np.testing.assert_allclose(data.features[row], freeze_input(agent, action, alpha).features, rtol=0, atol=1e-12)
        for index, candidate in enumerate(data.action):
            self.assertEqual(data.features[index, FEATURES.index(('action.UP','action.DOWN','action.LEFT','action.RIGHT')[candidate])], 1)
        self.assertEqual(env.step_count, 0)

    def test_forced_selected_branch_matches_real_five_move_continuation_and_leaves_source_unchanged(self):
        env = GridWorld(seed=51, max_steps=12)
        state = env.reset()
        agent = self.agent()
        action, _ = agent.select_action(state)
        before_agent, before_env = agent.rng.getstate(), env.rng.getstate()
        result = branch(agent, env, state, action, 5)
        self.assertEqual(agent.rng.getstate(), before_agent)
        self.assertEqual(env.rng.getstate(), before_env)
        self.assertEqual(env.step_count, 0)
        loss, collections = 0, env.minerals_collected
        for step in range(5):
            if step:
                action, _ = agent.select_action(state)
            energy = env.energy
            state, _, done, _ = env.step(action)
            loss += max(0, energy-env.energy)
            agent.commit_motivational_transition(next_env_state=state, done=done, executed_action=action)
        np.testing.assert_array_equal(result['state'], vector(agent.mot))
        self.assertEqual(result['moves'], 5)
        self.assertEqual(result['utility'], 100*(env.minerals_collected-collections)-loss)

    def test_early_terminal_has_mask_and_keeps_utility(self):
        env = GridWorld(seed=51, max_steps=2)
        state = env.reset()
        agent = self.agent()
        action, _ = agent.select_action(state)
        result = branch(agent, env, state, action, 5)
        self.assertFalse(result['complete'])
        self.assertEqual(result['moves'], 2)
        self.assertTrue(np.isfinite(result['utility']))

    def test_ties_do_not_create_fake_ranking_accuracy(self):
        self.assertEqual(choose([1,1,1], 2), 2)
        scores = ranking_metrics(np.array([0.,1.,2.]), np.array([0.,0.,0.]), 0)
        self.assertFalse(scores['informative'])
        self.assertEqual(scores['pairwise_pairs'], 0)
        scores = ranking_metrics(np.array([1.,1.]), np.array([0.,100.]), 0)
        self.assertEqual(scores['pairwise_correct_sum'], 0.5)
        self.assertEqual(scores['regret'], 100)
        scores = ranking_metrics(np.array([0.999999, 1.0]), np.array([0.,100.]), 0, fixed_choice=0)
        self.assertEqual(scores['choice'], 0)
        self.assertEqual(scores['pairwise_correct_sum'], 1)

    def test_readout_recovers_known_relation_without_changing_scaling_at_inference(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(100,3)); target = 2*x[:,0]-x[:,1]+3
        fitted = UtilityReadout.fit(x,target,0.0001)
        self.assertLess(float(np.abs(fitted.predict(x)-target).mean()), 1e-4)
        before = fitted.mean.copy()
        fitted.predict(x+100)
        np.testing.assert_array_equal(fitted.mean,before)

    def test_small_collection_readout_evaluation_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'model.npz'
            np.savez(path,input_mean=np.zeros(len(FEATURES)),input_scale=np.ones(len(FEATURES)),
                     residual_mean=np.zeros(DIM),residual_scale=np.ones(DIM),ridge=np.zeros((len(FEATURES)+1,DIM)),
                     **{'neural.0.weight':np.zeros((32,len(FEATURES))), 'neural.0.bias':np.zeros(32),
                        'neural.2.weight':np.zeros((DIM,32)), 'neural.2.bias':np.zeros(DIM)})
            model=FrozenForecasters(path,5)
            settings=RankingSettings(calibration_episodes=2,test_episodes=2,replicas=2,audit_stride=3,bootstrap_samples=20)
            original=NeuralSettings(max_steps=10)
            agent=self.agent()
            calibration=collect(agent,range(51,53),original,settings,model,None,'calibration')
            readout,direct=fit_readouts(calibration,1)
            test=collect(agent,range(55,57),original,settings,model,None,'test')
            report,decisions=evaluate(test,readout,direct,20,1)
            self.assertGreater(report['audited_states'],0)
            self.assertEqual(report['incomplete_branches'],0)
            self.assertEqual(len(decisions),report['audited_states'])
            self.assertIn('actual_state_readout',report['methods'])
            self.assertEqual(report['state_mae']['neural_32']['selected']['count'],len(test)*2)


if __name__ == '__main__':
    unittest.main()
