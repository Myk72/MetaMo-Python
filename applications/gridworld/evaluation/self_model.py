"""Held-out prediction and behavior-parity evaluation for the GridWorld self-model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, astuple, dataclass, replace
from math import fsum
from typing import Sequence

import numpy as np

from applications.gridworld.agents.metamo import MetaMoAgent
from applications.gridworld.config import ACTION_IDS
from applications.gridworld.environment import GridWorld
from applications.gridworld.predictive_self_model import GridWorldPredictiveSelfModel
from applications.gridworld.stimulus import GridWorldStimulus
from applications.gridworld.transition_record import MotivationalSnapshot
from dynamics.stability import is_in_safe_region


def _mae(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("comparison vectors must have the same nonzero length")
    return fsum(abs(a - b) for a, b in zip(left, right)) / len(left)


def _q_tables_match(first: MetaMoAgent, second: MetaMoAgent) -> bool:
    if first.q_table.keys() != second.q_table.keys():
        return False
    return all(
        np.array_equal(first.q_table[key], second.q_table[key])
        and np.array_equal(first.visit_counts[key], second.visit_counts[key])
        for key in first.q_table
    )


@dataclass(frozen=True)
class SelfModelEvaluationReport:
    training_episodes: int
    held_out_episodes: int
    training_steps: int
    held_out_steps: int
    scored_steps: int
    training_samples_by_action: dict[str, int]
    training_collection_samples: int
    held_out_collection_steps: int
    learned_collection_forecast_steps: int
    collection_proximity_interval: tuple[float, float] | None
    collection_interval_coverage: float | None
    learned_collection_proximity_mae: float | None
    nominal_collection_proximity_mae: float | None
    learned_collection_appraisal_mae: float | None
    nominal_collection_appraisal_mae: float | None
    learned_stimulus_mae: float
    nominal_stimulus_mae: float
    contextual_stimulus_mae: float
    action_only_stimulus_mae: float
    persistence_stimulus_mae: float
    learned_appraisal_mae: float
    nominal_appraisal_mae: float
    contextual_appraisal_mae: float
    action_only_appraisal_mae: float
    persistence_appraisal_mae: float
    contextual_forecast_steps: int
    stimulus_feature_mae: dict[str, dict[str, float]]
    committed_goal_mae: float
    committed_modulator_mae: float
    action_mismatches: int
    reward_mismatches: int
    state_mismatches: int
    safety_status_mismatches: int
    unsafe_steps: int
    q_tables_match: bool
    model_updates_during_held_out: int
    collection_updates_during_held_out: int
    contextual_updates_during_held_out: int
    action_only_updates_during_held_out: int

    @property
    def prediction_improved(self) -> bool:
        return (
            self.learned_appraisal_mae < self.persistence_appraisal_mae
            and self.learned_appraisal_mae < self.contextual_appraisal_mae
            and self.learned_appraisal_mae < self.action_only_appraisal_mae
        )

    @property
    def learning_adds_value(self) -> bool:
        return (
            self.held_out_collection_steps > 0
            and self.learned_collection_forecast_steps > 0
            and self.learned_collection_appraisal_mae
            < self.nominal_collection_appraisal_mae
        )

    @property
    def behavior_unchanged(self) -> bool:
        return (
            self.action_mismatches == 0
            and self.reward_mismatches == 0
            and self.state_mismatches == 0
            and self.safety_status_mismatches == 0
            and self.q_tables_match
            and self.model_updates_during_held_out == 0
            and self.collection_updates_during_held_out == 0
            and self.contextual_updates_during_held_out == 0
            and self.action_only_updates_during_held_out == 0
        )

    @property
    def safety_passed(self) -> bool:
        return self.safety_status_mismatches == 0 and self.unsafe_steps == 0


def run_self_model_evaluation(
    train_seeds: Sequence[int],
    held_out_seeds: Sequence[int],
    *,
    max_steps: int = 40,
    agent_seed: int = 42,
) -> SelfModelEvaluationReport:
    """Train on early episodes, then compare frozen forecasts on disjoint seeds."""
    train_seeds = tuple(train_seeds)
    held_out_seeds = tuple(held_out_seeds)
    if not train_seeds or not held_out_seeds or max_steps < 2:
        raise ValueError("provide both seed sets and at least two steps per episode")
    if (
        len(set(train_seeds)) != len(train_seeds)
        or len(set(held_out_seeds)) != len(held_out_seeds)
        or set(train_seeds) & set(held_out_seeds)
    ):
        raise ValueError("training and held-out episode seeds must be unique and disjoint")

    plain = MetaMoAgent(seed=agent_seed, record_transitions=False)
    observed = MetaMoAgent(seed=agent_seed, record_transitions=True)
    nominal = GridWorldPredictiveSelfModel(
        observed.runtime.appraisal_profile, forecast_mode="move_nominal"
    )
    contextual = GridWorldPredictiveSelfModel(
        observed.runtime.appraisal_profile, forecast_mode="contextual"
    )
    action_only = GridWorldPredictiveSelfModel(
        observed.runtime.appraisal_profile, forecast_mode="action_only"
    )
    training_steps = held_out_steps = scored_steps = 0
    held_out_collection_steps = learned_collection_forecast_steps = 0
    collection_interval_hits = 0
    contextual_forecast_steps = 0
    action_mismatches = reward_mismatches = state_mismatches = 0
    safety_status_mismatches = unsafe_steps = 0
    learned_stimulus_errors: list[float] = []
    nominal_stimulus_errors: list[float] = []
    contextual_stimulus_errors: list[float] = []
    action_only_stimulus_errors: list[float] = []
    persistence_stimulus_errors: list[float] = []
    learned_appraisal_errors: list[float] = []
    nominal_appraisal_errors: list[float] = []
    contextual_appraisal_errors: list[float] = []
    action_only_appraisal_errors: list[float] = []
    persistence_appraisal_errors: list[float] = []
    committed_goal_errors: list[float] = []
    committed_modulator_errors: list[float] = []
    learned_collection_proximity_errors: list[float] = []
    nominal_collection_proximity_errors: list[float] = []
    learned_collection_appraisal_errors: list[float] = []
    nominal_collection_appraisal_errors: list[float] = []
    feature_error_totals = {
        name: {"move_informed": 0.0, "move_nominal": 0.0, "contextual": 0.0,
               "action_only": 0.0, "persistence": 0.0}
        for name in GridWorldStimulus.__dataclass_fields__
    }

    def run_episode(seed: int, *, train: bool) -> None:
        nonlocal training_steps, held_out_steps, scored_steps, contextual_forecast_steps
        nonlocal held_out_collection_steps, learned_collection_forecast_steps
        nonlocal collection_interval_hits
        nonlocal action_mismatches, reward_mismatches, state_mismatches
        nonlocal safety_status_mismatches, unsafe_steps

        env_plain = GridWorld(seed=seed, max_steps=max_steps)
        env_observed = GridWorld(seed=seed, max_steps=max_steps)
        state_plain = env_plain.reset()
        state_observed = env_observed.reset()
        plain.reset_episode()
        observed.reset_episode()
        if state_plain != state_observed:
            state_mismatches += 1

        for _ in range(max_steps):
            action_plain, alpha_plain = plain.select_action(state_plain)
            action_observed, alpha_observed = observed.select_action(state_observed)
            nominal_forecast = None
            contextual_forecast = None
            action_only_forecast = None
            if not train:
                pending = observed._pending_transition
                nominal_forecast = nominal.forecast(
                    transition_id=pending.transition_id,
                    action_id=pending.action_id,
                    staged_state=observed._pending_state,
                    current_stimulus=pending.current_stimulus,
                    move_inputs=pending.move_inputs,
                )
                contextual_forecast = contextual.forecast(
                    transition_id=pending.transition_id,
                    action_id=pending.action_id,
                    staged_state=observed._pending_state,
                    current_stimulus=pending.current_stimulus,
                )
                action_only_forecast = action_only.forecast(
                    transition_id=pending.transition_id,
                    action_id=pending.action_id,
                    staged_state=observed._pending_state,
                    current_stimulus=pending.current_stimulus,
                )
            if action_plain != action_observed:
                action_mismatches += 1

            next_plain, reward_plain, done_plain, info_plain = env_plain.step(action_plain)
            next_observed, reward_observed, done_observed, info_observed = (
                env_observed.step(action_observed)
            )
            if reward_plain != reward_observed:
                reward_mismatches += 1

            if train:
                plain.update(
                    state_plain, action_plain, reward_plain, next_plain,
                    done_plain, info_plain.get("event"), alpha_plain,
                )
                observed.update(
                    state_observed, action_observed, reward_observed, next_observed,
                    done_observed, info_observed.get("event"), alpha_observed,
                )
                contextual.observe(observed.transition_records[-1])
                action_only.observe(observed.transition_records[-1])
                training_steps += 1
            else:
                plain.commit_motivational_transition(
                    next_env_state=next_plain,
                    done=done_plain,
                    executed_action=action_plain,
                )
                observed.commit_motivational_transition(
                    next_env_state=next_observed,
                    done=done_observed,
                    executed_action=action_observed,
                )
                held_out_steps += 1

            if (
                next_plain != next_observed
                or done_plain != done_observed
                or not np.array_equal(plain.mot.G, observed.mot.G)
                or not np.array_equal(plain.mot.M, observed.mot.M)
            ):
                state_mismatches += 1
            safe_plain = is_in_safe_region(plain.mot)
            safe_observed = is_in_safe_region(observed.mot)
            if safe_plain != safe_observed:
                safety_status_mismatches += 1
            if not safe_plain or not safe_observed:
                unsafe_steps += 1

            if not train and not done_observed:
                record = observed.transition_records[-1]
                error = observed.predictive_self_model.evaluate(record)
                nominal_error = nominal.evaluate(
                    replace(record, forecast=nominal_forecast)
                )
                contextual_error = contextual.evaluate(
                    replace(record, forecast=contextual_forecast)
                )
                action_only_error = action_only.evaluate(
                    replace(record, forecast=action_only_forecast)
                )
                if error.next_appraisal_modulator_mae is None:
                    raise RuntimeError("nonterminal transition has no appraisal target")
                baseline_appraisal = observed.runtime.appraisal_profile.appraise(
                    observed.mot, record.current_stimulus
                )
                baseline_snapshot = MotivationalSnapshot.from_state(baseline_appraisal)
                learned_stimulus_errors.append(error.next_stimulus_mae)
                nominal_stimulus_errors.append(nominal_error.next_stimulus_mae)
                contextual_stimulus_errors.append(contextual_error.next_stimulus_mae)
                action_only_stimulus_errors.append(action_only_error.next_stimulus_mae)
                persistence_stimulus_errors.append(
                    _mae(astuple(record.current_stimulus), astuple(record.next_stimulus))
                )
                learned_appraisal_errors.append(error.next_appraisal_modulator_mae)
                nominal_appraisal_errors.append(nominal_error.next_appraisal_modulator_mae)
                contextual_appraisal_errors.append(
                    contextual_error.next_appraisal_modulator_mae
                )
                action_only_appraisal_errors.append(
                    action_only_error.next_appraisal_modulator_mae
                )
                persistence_appraisal_errors.append(
                    _mae(
                        baseline_snapshot.modulators,
                        record.observed_next_appraisal.modulators,
                    )
                )
                committed_goal_errors.append(error.committed_goal_mae)
                committed_modulator_errors.append(error.committed_modulator_mae)
                if record.move_inputs is not None and record.move_inputs.mineral_collected >= 0.5:
                    held_out_collection_steps += 1
                    if training_collection_samples >= observed.predictive_self_model.min_collection_samples:
                        learned_collection_forecast_steps += 1
                    actual_proximity = record.next_stimulus.goal_proximity
                    learned_collection_proximity_errors.append(abs(
                        record.forecast.predicted_next_stimulus.goal_proximity - actual_proximity
                    ))
                    nominal_collection_proximity_errors.append(abs(
                        nominal_forecast.predicted_next_stimulus.goal_proximity - actual_proximity
                    ))
                    learned_collection_appraisal_errors.append(
                        error.next_appraisal_modulator_mae
                    )
                    nominal_collection_appraisal_errors.append(
                        nominal_error.next_appraisal_modulator_mae
                    )
                    if collection_interval is not None and (
                        collection_interval[0] <= actual_proximity <= collection_interval[1]
                    ):
                        collection_interval_hits += 1
                if observed.predictive_self_model.context_sample_count(
                    record.action_id, record.current_stimulus
                ) >= observed.predictive_self_model.min_samples:
                    contextual_forecast_steps += 1
                for name, totals in feature_error_totals.items():
                    actual = getattr(record.next_stimulus, name)
                    totals["move_informed"] += abs(
                        getattr(record.forecast.predicted_next_stimulus, name) - actual
                    )
                    totals["move_nominal"] += abs(
                        getattr(nominal_forecast.predicted_next_stimulus, name) - actual
                    )
                    totals["contextual"] += abs(
                        getattr(contextual_forecast.predicted_next_stimulus, name) - actual
                    )
                    totals["action_only"] += abs(
                        getattr(action_only_forecast.predicted_next_stimulus, name) - actual
                    )
                    totals["persistence"] += abs(
                        getattr(record.current_stimulus, name) - actual
                    )
                scored_steps += 1

            state_plain, state_observed = next_plain, next_observed
            if done_plain or done_observed:
                break

        if train:
            plain.decay_epsilon()
            observed.decay_epsilon()

    for seed in train_seeds:
        run_episode(seed, train=True)

    training_samples = {
        action_id: observed.predictive_self_model.sample_count(action_id)
        for action_id in ACTION_IDS
    }
    training_collection_samples = observed.predictive_self_model.collection_observation_count
    collection_interval = observed.predictive_self_model.collection_proximity_interval()
    observed.predictive_self_model.learning_enabled = False
    contextual.learning_enabled = False
    action_only.learning_enabled = False
    plain.epsilon = observed.epsilon = 0.0
    for seed in held_out_seeds:
        run_episode(seed, train=False)
    if scored_steps == 0:
        raise ValueError("held-out episodes produced no nonterminal transitions")
    later_samples = sum(
        observed.predictive_self_model.sample_count(action_id)
        for action_id in ACTION_IDS
    )
    later_action_only_samples = sum(
        action_only.sample_count(action_id) for action_id in ACTION_IDS
    )
    later_contextual_samples = sum(
        contextual.sample_count(action_id) for action_id in ACTION_IDS
    )

    return SelfModelEvaluationReport(
        training_episodes=len(train_seeds),
        held_out_episodes=len(held_out_seeds),
        training_steps=training_steps,
        held_out_steps=held_out_steps,
        scored_steps=scored_steps,
        training_samples_by_action=training_samples,
        training_collection_samples=training_collection_samples,
        held_out_collection_steps=held_out_collection_steps,
        learned_collection_forecast_steps=learned_collection_forecast_steps,
        collection_proximity_interval=collection_interval,
        collection_interval_coverage=(
            collection_interval_hits / held_out_collection_steps
            if collection_interval is not None and held_out_collection_steps else None
        ),
        learned_collection_proximity_mae=(
            fsum(learned_collection_proximity_errors) / held_out_collection_steps
            if held_out_collection_steps else None
        ),
        nominal_collection_proximity_mae=(
            fsum(nominal_collection_proximity_errors) / held_out_collection_steps
            if held_out_collection_steps else None
        ),
        learned_collection_appraisal_mae=(
            fsum(learned_collection_appraisal_errors) / held_out_collection_steps
            if held_out_collection_steps else None
        ),
        nominal_collection_appraisal_mae=(
            fsum(nominal_collection_appraisal_errors) / held_out_collection_steps
            if held_out_collection_steps else None
        ),
        learned_stimulus_mae=fsum(learned_stimulus_errors) / scored_steps,
        nominal_stimulus_mae=fsum(nominal_stimulus_errors) / scored_steps,
        contextual_stimulus_mae=fsum(contextual_stimulus_errors) / scored_steps,
        action_only_stimulus_mae=fsum(action_only_stimulus_errors) / scored_steps,
        persistence_stimulus_mae=fsum(persistence_stimulus_errors) / scored_steps,
        learned_appraisal_mae=fsum(learned_appraisal_errors) / scored_steps,
        nominal_appraisal_mae=fsum(nominal_appraisal_errors) / scored_steps,
        contextual_appraisal_mae=fsum(contextual_appraisal_errors) / scored_steps,
        action_only_appraisal_mae=fsum(action_only_appraisal_errors) / scored_steps,
        persistence_appraisal_mae=fsum(persistence_appraisal_errors) / scored_steps,
        contextual_forecast_steps=contextual_forecast_steps,
        stimulus_feature_mae={
            name: {mode: total / scored_steps for mode, total in totals.items()}
            for name, totals in feature_error_totals.items()
        },
        committed_goal_mae=fsum(committed_goal_errors) / scored_steps,
        committed_modulator_mae=fsum(committed_modulator_errors) / scored_steps,
        action_mismatches=action_mismatches,
        reward_mismatches=reward_mismatches,
        state_mismatches=state_mismatches,
        safety_status_mismatches=safety_status_mismatches,
        unsafe_steps=unsafe_steps,
        q_tables_match=_q_tables_match(plain, observed),
        model_updates_during_held_out=later_samples - sum(training_samples.values()),
        collection_updates_during_held_out=(
            observed.predictive_self_model.collection_observation_count
            - training_collection_samples
        ),
        contextual_updates_during_held_out=(
            later_contextual_samples - sum(training_samples.values())
        ),
        action_only_updates_during_held_out=(
            later_action_only_samples - sum(training_samples.values())
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-episodes", type=int, default=20)
    parser.add_argument("--held-out-episodes", type=int, default=8)
    parser.add_argument("--held-out-seed-start", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--agent-seed", type=int, default=42)
    args = parser.parse_args()
    report = run_self_model_evaluation(
        train_seeds=range(1000, 1000 + args.train_episodes),
        held_out_seeds=range(
            args.held_out_seed_start,
            args.held_out_seed_start + args.held_out_episodes,
        ),
        max_steps=args.max_steps,
        agent_seed=args.agent_seed,
    )
    print(json.dumps(
        {**asdict(report),
         "prediction_improved": report.prediction_improved,
         "learning_adds_value": report.learning_adds_value,
         "behavior_unchanged": report.behavior_unchanged,
         "safety_passed": report.safety_passed},
        indent=2,
    ))


if __name__ == "__main__":
    main()
