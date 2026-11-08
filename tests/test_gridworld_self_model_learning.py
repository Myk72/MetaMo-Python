import unittest
from dataclasses import replace

from applications.gridworld.predictive_self_model import GridWorldPredictiveSelfModel
from applications.gridworld.profile import GRIDWORLD_APPRAISAL_PROFILE
from applications.gridworld.state import create_initial_motivational_state
from applications.gridworld.stimulus import GridWorldStimulus
from applications.gridworld.transition_record import (
    GridWorldTransitionRecord,
    MotivationalSnapshot,
    MoveForecastInputs,
)


def stimulus(hazard: float) -> GridWorldStimulus:
    return GridWorldStimulus(
        goal_proximity=0.5,
        hazard_pressure=hazard,
        energy_deficit=0.2,
        time_pressure=0.3,
        safe_progress=0.4,
        safe_mobility=0.6,
    )


class GridWorldSelfModelLearningTests(unittest.TestCase):
    def setUp(self):
        self.state = create_initial_motivational_state()
        self.snapshot = MotivationalSnapshot.from_state(self.state)
        self.model = GridWorldPredictiveSelfModel(
            GRIDWORLD_APPRAISAL_PROFILE,
            min_samples=2,
            prior_count=0.0,
            forecast_mode="action_only",
        )

    def record(self, transition_id, action_id, before, after, *, done=False):
        forecast = self.model.forecast(
            transition_id=transition_id,
            action_id=action_id,
            staged_state=self.state,
            current_stimulus=before,
        )
        return GridWorldTransitionRecord(
            transition_id=transition_id,
            episode_id=0,
            step_index=transition_id,
            action_index=0,
            action_id=action_id,
            current_state=self.snapshot,
            current_stimulus=before,
            staged_state=self.snapshot,
            committed_state=self.snapshot,
            next_stimulus=after,
            done=done,
            forecast=forecast,
        )

    def test_learns_action_mean_only_after_feedback_and_sparse_fallback(self):
        first = self.record(1, "east", stimulus(0.2), stimulus(0.4))
        self.assertEqual(first.forecast.predicted_next_stimulus, stimulus(0.2))
        self.model.observe(first)
        self.assertEqual(self.model.sample_count("east"), 1)

        second = self.record(2, "east", stimulus(0.2), stimulus(0.6))
        self.assertEqual(second.forecast.predicted_next_stimulus, stimulus(0.2))
        self.model.observe(second)
        self.assertEqual(self.model.sample_count("east"), 2)
        self.assertAlmostEqual(self.model.mean_change("east")[1], 0.3)

        learned = self.model.forecast(
            transition_id=3,
            action_id="east",
            staged_state=self.state,
            current_stimulus=stimulus(0.2),
        )
        self.assertAlmostEqual(learned.predicted_next_stimulus.hazard_pressure, 0.5)
        self.assertNotEqual(
            learned.predicted_next_appraisal, second.forecast.predicted_next_appraisal
        )
        unseen_action = self.model.forecast(
            transition_id=4,
            action_id="west",
            staged_state=self.state,
            current_stimulus=stimulus(0.2),
        )
        self.assertEqual(unseen_action.predicted_next_stimulus, stimulus(0.2))

    def test_clips_prediction_and_rejects_duplicate_update(self):
        first = self.record(1, "east", stimulus(0.2), stimulus(0.9))
        second = self.record(2, "east", stimulus(0.2), stimulus(0.9))
        self.model.observe(first)
        self.model.observe(second)
        clipped = self.model.forecast(
            transition_id=3,
            action_id="east",
            staged_state=self.state,
            current_stimulus=stimulus(0.9),
        )
        self.assertEqual(clipped.predicted_next_stimulus.hazard_pressure, 1.0)
        with self.assertRaises(ValueError):
            self.model.observe(second)
        self.assertEqual(self.model.sample_count("east"), 2)

    def test_terminal_and_disabled_learning_do_not_update_statistics(self):
        terminal = self.record(
            1, "east", stimulus(0.2), stimulus(0.8), done=True
        )
        self.model.observe(terminal)
        self.assertEqual(self.model.sample_count("east"), 0)

        self.model.learning_enabled = False
        paused = self.record(2, "east", stimulus(0.2), stimulus(0.8))
        self.model.observe(paused)
        self.assertEqual(self.model.sample_count("east"), 0)
        self.model.learning_enabled = True
        with self.assertRaises(ValueError):
            self.model.observe(paused)

    def test_contextual_model_keeps_hazard_and_mobility_conditions_separate(self):
        model = GridWorldPredictiveSelfModel(
            GRIDWORLD_APPRAISAL_PROFILE,
            min_samples=2,
            prior_count=0.0,
            forecast_mode="contextual",
        )

        def observe(transition_id, before, after):
            forecast = model.forecast(
                transition_id=transition_id,
                action_id="east",
                staged_state=self.state,
                current_stimulus=before,
            )
            model.observe(GridWorldTransitionRecord(
                transition_id=transition_id,
                episode_id=0,
                step_index=transition_id,
                action_index=0,
                action_id="east",
                current_state=self.snapshot,
                current_stimulus=before,
                staged_state=self.snapshot,
                committed_state=self.snapshot,
                next_stimulus=after,
                done=False,
                forecast=forecast,
            ))

        for transition_id in (1, 2):
            observe(transition_id, stimulus(0.2), stimulus(0.4))
        high_hazard = stimulus(0.6)
        self.assertEqual(model.context_sample_count("east", high_hazard), 0)
        self.assertEqual(
            model.forecast(
                transition_id=3, action_id="east", staged_state=self.state,
                current_stimulus=high_hazard,
            ).predicted_next_stimulus,
            high_hazard,
        )
        self.assertAlmostEqual(
            model.forecast(
                transition_id=4, action_id="east", staged_state=self.state,
                current_stimulus=stimulus(0.2),
            ).predicted_next_stimulus.hazard_pressure,
            0.4,
        )
        for transition_id in (5, 6):
            observe(transition_id, high_hazard, stimulus(0.5))
        self.assertAlmostEqual(
            model.forecast(
                transition_id=7, action_id="east", staged_state=self.state,
                current_stimulus=high_hazard,
            ).predicted_next_stimulus.hazard_pressure,
            0.5,
        )
        low_mobility = GridWorldStimulus(0.5, 0.2, 0.2, 0.3, 0.4, 0.25)
        self.assertEqual(model.context_sample_count("east", low_mobility), 0)

    def test_move_forecast_uses_candidate_risk_and_keeps_respawn_uncertain(self):
        model = GridWorldPredictiveSelfModel(
            GRIDWORLD_APPRAISAL_PROFILE, forecast_mode="move_informed"
        )
        move = MoveForecastInputs(
            hazard_risk=0.75,
            mineral_progress=1.0,
            mineral_collected=1.0,
            safe_mobility_after=0.25,
        )
        before = stimulus(0.2)
        forecast = model.forecast(
            transition_id=10,
            action_id="east",
            staged_state=self.state,
            current_stimulus=before,
            move_inputs=move,
        )
        self.assertEqual(forecast.predicted_next_stimulus.hazard_pressure, 0.75)
        self.assertEqual(forecast.predicted_next_stimulus.safe_mobility, 0.25)
        self.assertEqual(forecast.predicted_next_stimulus.energy_deficit, 0.0)
        # A collected mineral respawns randomly; its new distance is not inferred
        # from the pre-collection progress estimate.
        self.assertEqual(
            forecast.predicted_next_stimulus.goal_proximity,
            before.goal_proximity,
        )

    def test_respawn_learning_uses_only_prior_collection_outcomes(self):
        model = GridWorldPredictiveSelfModel(
            GRIDWORLD_APPRAISAL_PROFILE,
            min_collection_samples=3,
            collection_history_limit=5,
            forecast_mode="move_informed",
        )
        collection = MoveForecastInputs(0.0, 1.0, 1.0, 0.5)
        before = stimulus(0.2)

        def record(transition_id, next_proximity, move_inputs):
            forecast = model.forecast(
                transition_id=transition_id,
                action_id="east",
                staged_state=self.state,
                current_stimulus=before,
                move_inputs=move_inputs,
            )
            return GridWorldTransitionRecord(
                transition_id=transition_id,
                episode_id=0,
                step_index=transition_id,
                action_index=0,
                action_id="east",
                current_state=self.snapshot,
                current_stimulus=before,
                staged_state=self.snapshot,
                committed_state=self.snapshot,
                next_stimulus=replace(before, goal_proximity=next_proximity),
                done=False,
                move_inputs=move_inputs,
                forecast=forecast,
            )

        model.observe(record(1, 0.9, MoveForecastInputs(0.0, 1.0, 0.0, 0.5)))
        self.assertEqual(model.collection_observation_count, 0)
        for transition_id, next_proximity in ((2, 0.2), (3, 0.4), (4, 0.8)):
            pending = record(transition_id, next_proximity, collection)
            self.assertEqual(pending.forecast.predicted_next_stimulus.goal_proximity, 0.5)
            model.observe(pending)

        self.assertEqual(model.collection_observation_count, 3)
        self.assertEqual(model.collection_proximity_interval(), (0.2, 0.8))
        learned = model.forecast(
            transition_id=5, action_id="east", staged_state=self.state,
            current_stimulus=before, move_inputs=collection,
        )
        self.assertEqual(learned.predicted_next_stimulus.goal_proximity, 0.4)
        model.learning_enabled = False
        model.observe(record(6, 1.0, collection))
        self.assertEqual(model.collection_observation_count, 3)


if __name__ == "__main__":
    unittest.main()
