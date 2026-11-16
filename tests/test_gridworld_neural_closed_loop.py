import copy
import unittest
import numpy as np

from applications.gridworld.agents.metamo import MetaMoAgent
from applications.gridworld.environment import GridWorld
from applications.gridworld.runtime import transition_for_action
from applications.gridworld.evaluation.neural_self_model import NeuralSettings, FEATURES
from applications.gridworld.evaluation.neural_action_ranking import UtilityReadout, vector
from applications.gridworld.evaluation.neural_closed_loop import run_episode, paired_summary, ClosedLoopSettings


class NominalModel:
    def predict(self,data):
        return {'neural_32':data.nominal.copy()}


class ClosedLoopTests(unittest.TestCase):
    def template(self):
        agent=MetaMoAgent(seed=42)
        agent.epsilon=0
        agent.predictive_self_model.learning_enabled=False
        return agent

    def direct(self):
        coefficients=np.zeros(len(FEATURES)+1)
        coefficients[FEATURES.index('action.RIGHT')]=1000
        return UtilityReadout(np.zeros(len(FEATURES)),np.ones(len(FEATURES)),coefficients)

    def test_passive_baseline_exactly_matches_original_episode(self):
        settings=NeuralSettings(max_steps=12)
        template=self.template()
        result,trace=run_episode(template,123,'current_selector',settings,NominalModel(),None,None)
        agent=copy.deepcopy(template); agent.reset_episode(); agent.rng.seed(123)
        env=GridWorld(seed=123,max_steps=12); state=env.reset()
        actions=[]; rewards=[]
        for _ in range(12):
            action,_=agent.select_action(state)
            state,reward,done,_=env.step(action); agent.commit_motivational_transition()
            actions.append(action); rewards.append(reward)
            if done: break
        self.assertEqual(actions,[row['action'] for row in trace])
        self.assertEqual(result['reward'],sum(rewards))
        self.assertEqual(result['minerals'],env.minerals_collected)
        self.assertEqual(result['forecast_targets'],8)
        self.assertEqual(template.q_table,{})

    def test_override_commits_matching_state_preserves_shortlist_and_tail(self):
        original=NeuralSettings(max_steps=20)
        result,trace=run_episode(self.template(),123,'direct_features',original,NominalModel(),None,self.direct())
        self.assertGreater(result['changed_choices'],0)
        self.assertEqual(result['tail_fallbacks'],4)
        agent=self.template(); agent.reset_episode()
        env=GridWorld(seed=123,max_steps=20); state=env.reset()
        committed=[]
        for row in trace:
            self.assertIn(row['action'],row['eligible'])
            if row['step']>=16: self.assertEqual(row['action'],row['proposed'])
            _,staged,_,_=transition_for_action(state,agent.mot,row['action'],runtime=agent.runtime)
            agent.mot=staged; committed.append(vector(staged))
            state,_,_,_=env.step(row['action'])
        for row in trace[:16]:
            np.testing.assert_allclose(row['target'],committed[row['step']+4],atol=1e-12)

    def test_nn_readout_is_used_at_each_supported_decision(self):
        class Counter:
            count=0
            def predict(self,x):
                self.count+=1
                return np.arange(len(x),dtype=float)
        readout=Counter()
        result,_=run_episode(self.template(),124,'neural_state',NeuralSettings(max_steps=10),NominalModel(),readout,None)
        self.assertEqual(readout.count,6)
        self.assertEqual(result['tail_fallbacks'],4)

    def test_paired_intervals_and_missing_pair_rejection(self):
        rows=[]
        for seed in range(4):
            for method,value in [('current_selector',1),('neural_state',2)]:
                rows.append({'episode':seed,'method':method,'minerals':value,'energy_loss':0,'depleted':False,
                             'zero_collection':False,'utility':value*100})
        result=paired_summary(rows,'neural_state','current_selector',100,1)
        self.assertEqual(result['minerals']['bootstrap_95_percent_interval'],[1.,1.])
        with self.assertRaises(ValueError): paired_summary(rows[:-1],'neural_state','current_selector',100,1)
        with self.assertRaises(ValueError): ClosedLoopSettings(episodes_per_split=0).validate()


if __name__=='__main__':
    unittest.main()
