"""Evaluate whether a self-model can anticipate later task/safety conflict.

All candidate outcomes come from cloned GridWorld branches. The live policy and
Q values are frozen during calibration and held-out evaluation. This is a
shadow experiment, not an action-selection change.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from math import fsum, sqrt

import numpy as np

from applications.gridworld.agents.metamo import MetaMoAgent
from applications.gridworld.environment import GridWorld
from applications.gridworld.runtime import (
    build_candidates,
    build_stimulus,
    perspective_candidate_scores,
    transition_for_action,
)
from dynamics.stability import is_in_safe_region


def _mean(values: list[float]) -> float | None:
    return fsum(values) / len(values) if values else None


@dataclass(frozen=True)
class CandidateProbe:
    action: int
    action_id: str
    current_conflict: float
    base_score: float
    task_score: float
    safety_score: float
    q_value: float
    collection: bool
    hazard_high: bool

    @property
    def context(self) -> tuple[str, bool, bool, bool]:
        return (
            self.action_id,
            self.collection,
            self.hazard_high,
            self.current_conflict >= 0.5,
        )


@dataclass(frozen=True)
class BranchResult:
    future_conflict: float | None
    first_reward: float
    reward: float
    collections: int
    lava_steps: int
    unsafe_steps: int
    energy_depleted: bool = False


@dataclass(frozen=True)
class DisagreementCase:
    learned_probe: CandidateProbe
    reference_probe: CandidateProbe
    learned_forecast: float
    reference_forecast: float
    learned_outcome: BranchResult
    reference_outcome: BranchResult

    @property
    def collection_pattern(self) -> str:
        if self.reference_probe.collection and not self.learned_probe.collection:
            return "collection_delayed"
        if self.learned_probe.collection and not self.reference_probe.collection:
            return "collection_advanced"
        if self.learned_probe.collection:
            return "both_collect"
        return "neither_collects"


@dataclass(frozen=True)
class _Stats:
    count: int = 0
    total: float = 0.0

    def add(self, value: float) -> "_Stats":
        return _Stats(self.count + 1, self.total + value)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


class FutureConflictModel:
    """Shrunken residual means for the five-decision conflict target."""

    def __init__(self, *, min_context_samples: int = 10, prior_count: float = 20.0):
        if min_context_samples < 1 or not np.isfinite(prior_count) or prior_count < 0:
            raise ValueError("invalid conflict model settings")
        self.min_context_samples = min_context_samples
        self.prior_count = prior_count
        self.pooled = _Stats()
        self.contexts: dict[tuple[str, bool, bool, bool], _Stats] = {}

    def predict(self, probe: CandidateProbe, *, contextual: bool) -> float:
        stats = self.contexts.get(probe.context, _Stats()) if contextual else self.pooled
        if contextual and stats.count < self.min_context_samples:
            stats = self.pooled
        weight = stats.count / (stats.count + self.prior_count) if stats.count else 0.0
        return float(np.clip(probe.current_conflict + weight * stats.mean, 0.0, 2.0))

    def observe(self, probe: CandidateProbe, target: float) -> None:
        if not np.isfinite(target) or not 0.0 <= target <= 2.0:
            raise ValueError("future conflict target must be in [0, 2]")
        residual = target - probe.current_conflict
        self.pooled = self.pooled.add(residual)
        self.contexts[probe.context] = self.contexts.get(probe.context, _Stats()).add(residual)


def _probes(agent: MetaMoAgent, state: dict) -> tuple[CandidateProbe, ...]:
    stimulus = build_stimulus(state, agent.mot, runtime=agent.runtime)
    candidates = build_candidates(state, agent.mot, runtime=agent.runtime)
    perspectives = perspective_candidate_scores(
        agent.mot, stimulus, candidates, state, runtime=agent.runtime
    )
    risks = np.asarray([float(c.metadata["hazard_risk"]) for c in candidates])
    exploit = agent._valid_actions(state, avoid_lava=agent.mask_lava_on_exploit)
    eligible = agent._hard_safety_actions(exploit, risks)
    q_key = agent._encode(state)
    q_values = agent._q_values(q_key)
    if agent.selector_mode == "legacy_additive":
        shortlist = eligible
        base = agent._normalize_preferences(perspectives.consensus, exploit)
    else:
        shortlist, _ = agent._q_shortlist(
            q_values, agent.visit_counts[q_key], eligible
        )
        _, _, _, base = agent._composed_preferences(perspectives, exploit)
    task = agent._normalize_preferences(perspectives.task, exploit)
    safety = agent._normalize_preferences(perspectives.safety, exploit)
    return tuple(
        CandidateProbe(
            action=action,
            action_id=candidates[action].id,
            current_conflict=float(abs(task[action] - safety[action])),
            base_score=float(base[action]),
            task_score=float(task[action]),
            safety_score=float(safety[action]),
            q_value=float(q_values[action]),
            collection=float(candidates[action].metadata["mineral_collected"]) >= 0.5,
            hazard_high=float(candidates[action].metadata["hazard_risk"]) >= 0.5,
        )
        for action in shortlist
    )


def _branch(
    agent: MetaMoAgent,
    environment: GridWorld,
    state: dict,
    first_action: int,
    *,
    forecast_horizon: int,
    rollout_horizon: int,
) -> BranchResult:
    """Force one action, then follow the frozen live selector in a clone."""
    branch_agent = copy.deepcopy(agent)
    branch_env = copy.deepcopy(environment)
    branch_agent.record_transitions = False
    branch_agent._pending_state = None
    branch_agent._pending_action = None
    branch_agent._pending_transition = None
    branch_agent.epsilon = 0.0
    stimulus = build_stimulus(state, branch_agent.mot, runtime=branch_agent.runtime)
    candidates = build_candidates(state, branch_agent.mot, runtime=branch_agent.runtime)
    _, staged, _, _ = transition_for_action(
        state, branch_agent.mot, first_action,
        stimulus=stimulus, candidates=candidates, runtime=branch_agent.runtime,
    )
    branch_agent.mot = staged
    next_state, reward, done, info = branch_env.step(first_action)
    rewards = [float(reward)]
    collections = int(info.get("event") == "mineral")
    lava_steps = int(next_state["in_lava"])
    unsafe_steps = int(not is_in_safe_region(branch_agent.mot))
    future_conflicts: list[float] = []
    for _ in range(rollout_horizon):
        if done:
            break
        action, alpha = branch_agent.select_action(next_state)
        if len(future_conflicts) < forecast_horizon:
            future_conflicts.append(float(alpha["task_safety_disagreement"]))
        following, reward, done, info = branch_env.step(action)
        branch_agent.commit_motivational_transition()
        rewards.append(float(reward))
        collections += int(info.get("event") == "mineral")
        lava_steps += int(following["in_lava"])
        unsafe_steps += int(not is_in_safe_region(branch_agent.mot))
        next_state = following
    return BranchResult(
        future_conflict=_mean(future_conflicts) if len(future_conflicts) == forecast_horizon else None,
        first_reward=float(rewards[0]),
        reward=fsum(rewards),
        collections=collections,
        lava_steps=lava_steps,
        unsafe_steps=unsafe_steps,
        energy_depleted=bool(branch_env.energy <= 0.0),
    )


@dataclass(frozen=True)
class ConflictSplit:
    agent_seed: int
    calibration_labels: int
    held_out_labels: int
    target_mean: float
    target_std: float
    target_above_half_fraction: float
    reference_mae: float
    pooled_mae: float
    contextual_mae: float
    ready_contexts: int
    ranking_disagreements: int
    mean_reward_learned: float | None
    mean_reward_reference: float | None
    mean_collections_learned: float | None
    mean_collections_reference: float | None
    mean_lava_learned: float | None
    mean_lava_reference: float | None
    mean_unsafe_learned: float | None
    mean_unsafe_reference: float | None
    mean_conflict_learned: float | None
    mean_conflict_reference: float | None
    live_action_mismatches: int
    model_updates_during_test: int


@dataclass(frozen=True)
class FutureConflictReport:
    forecast_horizon: int
    rollout_horizon: int
    splits: tuple[ConflictSplit, ...]
    total_held_out_labels: int
    total_ranking_disagreements: int
    target_std: float
    reference_mae: float
    pooled_mae: float
    contextual_mae: float
    target_variable: bool
    prediction_supported: bool
    decision_supported: bool
    choice_audit: "ChoiceAudit"


@dataclass(frozen=True)
class ChoiceGroup:
    count: int
    mean_reward_difference: float | None
    mean_first_reward_difference: float | None
    mean_collection_difference: float | None
    mean_actual_conflict_difference: float | None
    reward_losses: int
    safety_improvements: int


@dataclass(frozen=True)
class ChoiceAudit:
    groups: dict[str, ChoiceGroup]
    total_cases: int
    complete_conflict_pairs: int
    mean_task_score_difference: float | None
    mean_safety_score_difference: float | None
    mean_q_difference: float | None
    mean_predicted_conflict_difference: float | None
    mean_actual_conflict_difference: float | None
    reward_losses: int
    reward_gains: int
    reward_losses_without_safety_improvement: int
    safety_improvements: int


def _actual_conflict_difference(case: DisagreementCase) -> float | None:
    left = case.learned_outcome.future_conflict
    right = case.reference_outcome.future_conflict
    return left - right if left is not None and right is not None else None


def _safety_improved(case: DisagreementCase) -> bool:
    return (
        case.learned_outcome.lava_steps < case.reference_outcome.lava_steps
        or case.learned_outcome.unsafe_steps < case.reference_outcome.unsafe_steps
    )


def _choice_audit(cases: list[DisagreementCase]) -> ChoiceAudit:
    patterns = (
        "collection_delayed", "collection_advanced", "both_collect", "neither_collects"
    )
    groups: dict[str, ChoiceGroup] = {}
    for pattern in patterns:
        group = [case for case in cases if case.collection_pattern == pattern]
        conflict_differences = [
            value for case in group
            if (value := _actual_conflict_difference(case)) is not None
        ]
        groups[pattern] = ChoiceGroup(
            count=len(group),
            mean_reward_difference=_mean([
                case.learned_outcome.reward - case.reference_outcome.reward
                for case in group
            ]),
            mean_first_reward_difference=_mean([
                case.learned_outcome.first_reward - case.reference_outcome.first_reward
                for case in group
            ]),
            mean_collection_difference=_mean([
                case.learned_outcome.collections - case.reference_outcome.collections
                for case in group
            ]),
            mean_actual_conflict_difference=_mean(conflict_differences),
            reward_losses=sum(
                case.learned_outcome.reward < case.reference_outcome.reward
                for case in group
            ),
            safety_improvements=sum(_safety_improved(case) for case in group),
        )
    complete_differences = [
        value for case in cases
        if (value := _actual_conflict_difference(case)) is not None
    ]
    return ChoiceAudit(
        groups=groups,
        total_cases=len(cases),
        complete_conflict_pairs=len(complete_differences),
        mean_task_score_difference=_mean([
            case.learned_probe.task_score - case.reference_probe.task_score
            for case in cases
        ]),
        mean_safety_score_difference=_mean([
            case.learned_probe.safety_score - case.reference_probe.safety_score
            for case in cases
        ]),
        mean_q_difference=_mean([
            case.learned_probe.q_value - case.reference_probe.q_value
            for case in cases
        ]),
        mean_predicted_conflict_difference=_mean([
            case.learned_forecast - case.reference_forecast for case in cases
        ]),
        mean_actual_conflict_difference=_mean(complete_differences),
        reward_losses=sum(
            case.learned_outcome.reward < case.reference_outcome.reward for case in cases
        ),
        reward_gains=sum(
            case.learned_outcome.reward > case.reference_outcome.reward for case in cases
        ),
        reward_losses_without_safety_improvement=sum(
            case.learned_outcome.reward < case.reference_outcome.reward
            and not _safety_improved(case) for case in cases
        ),
        safety_improvements=sum(_safety_improved(case) for case in cases),
    )


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    center = _mean(values)
    return fsum((value - center) ** 2 for value in values) / len(values)


def _train_q(agent: MetaMoAgent, seed: int, max_steps: int) -> None:
    env = GridWorld(seed=seed, max_steps=max_steps)
    state = env.reset()
    agent.reset_episode()
    for _ in range(max_steps):
        action, alpha = agent.select_action(state)
        following, reward, done, info = env.step(action)
        agent.update(state, action, reward, following, done, info.get("event"), alpha)
        state = following
        if done:
            break
    agent.decay_epsilon()


def _audit_episode(
    agent: MetaMoAgent,
    seed: int,
    *,
    max_steps: int,
    forecast_horizon: int,
    rollout_horizon: int,
    model: FutureConflictModel | None,
) -> tuple[list[tuple[CandidateProbe, float, float, float]], list[DisagreementCase], int]:
    env = GridWorld(seed=seed, max_steps=max_steps)
    state = env.reset()
    agent.reset_episode()
    accuracy: list[tuple[CandidateProbe, float, float, float]] = []
    disagreements: list[DisagreementCase] = []
    live_mismatches = 0
    for _ in range(max_steps):
        selected_action, _ = agent.select_action(state)
        probes = _probes(agent, state)
        if selected_action not in {probe.action for probe in probes}:
            live_mismatches += 1
        predictions = []
        for probe in probes:
            pooled = model.predict(probe, contextual=False) if model is not None else probe.current_conflict
            contextual = model.predict(probe, contextual=True) if model is not None else probe.current_conflict
            predictions.append((probe, pooled, contextual))
        if model is not None:
            def rank(use_context: bool) -> int:
                return max(
                    range(len(predictions)),
                    key=lambda index: (
                        predictions[index][0].base_score
                        + float(predictions[index][0].collection)
                        - 0.25 * (
                            predictions[index][2] if use_context
                            else predictions[index][0].current_conflict
                        ),
                        -predictions[index][0].action,
                    ),
                )
            learned_index = rank(True)
            reference_index = rank(False)
        else:
            learned_index = reference_index = -1
        # Clone only actions needed for accuracy or a paired disagreement.
        # Calibration uses every shortlist action; held-out uses the same set.
        outcomes: list[BranchResult] = []
        for probe, pooled, contextual in predictions:
            outcome = _branch(
                agent, env, state, probe.action,
                forecast_horizon=forecast_horizon,
                rollout_horizon=forecast_horizon,
            )
            outcomes.append(outcome)
            if outcome.future_conflict is not None:
                accuracy.append((probe, outcome.future_conflict, pooled, contextual))
        if model is not None and learned_index != reference_index:
            if rollout_horizon == forecast_horizon:
                pair = (outcomes[learned_index], outcomes[reference_index])
            else:
                pair = tuple(
                    _branch(
                        agent, env, state, probes[index].action,
                        forecast_horizon=forecast_horizon,
                        rollout_horizon=rollout_horizon,
                    )
                    for index in (learned_index, reference_index)
                )
            disagreements.append(DisagreementCase(
                learned_probe=probes[learned_index],
                reference_probe=probes[reference_index],
                learned_forecast=predictions[learned_index][2],
                reference_forecast=predictions[reference_index][2],
                learned_outcome=pair[0],
                reference_outcome=pair[1],
            ))
        following, _, done, _ = env.step(selected_action)
        agent.commit_motivational_transition()
        state = following
        if done:
            break
    return accuracy, disagreements, live_mismatches


def run_future_conflict_evaluation(
    *,
    split_count: int = 3,
    pretrain_episodes: int = 20,
    calibration_episodes: int = 10,
    held_out_episodes: int = 6,
    max_steps: int = 40,
    forecast_horizon: int = 5,
    rollout_horizon: int = 20,
    seed_stride: int = 100000,
    agent_seed: int = 80,
) -> FutureConflictReport:
    if (
        split_count < 1 or pretrain_episodes < 1 or calibration_episodes < 1
        or held_out_episodes < 1 or forecast_horizon < 1
        or rollout_horizon < forecast_horizon or max_steps <= forecast_horizon
        or seed_stride < pretrain_episodes + calibration_episodes + held_out_episodes
    ):
        raise ValueError("invalid conflict evaluation settings")
    splits: list[ConflictSplit] = []
    all_labels: list[float] = []
    all_reference_errors: list[float] = []
    all_pooled_errors: list[float] = []
    all_context_errors: list[float] = []
    all_cases: list[DisagreementCase] = []
    for split_index in range(split_count):
        base = 5000000 + split_index * seed_stride
        agent = MetaMoAgent(seed=agent_seed + split_index)
        for seed in range(base, base + pretrain_episodes):
            _train_q(agent, seed, max_steps)
        agent.epsilon = 0.0
        model = FutureConflictModel()
        calibration: list[tuple[CandidateProbe, float, float, float]] = []
        for seed in range(base + pretrain_episodes,
                          base + pretrain_episodes + calibration_episodes):
            examples, _, _ = _audit_episode(
                agent, seed, max_steps=max_steps,
                forecast_horizon=forecast_horizon,
                rollout_horizon=forecast_horizon, model=None,
            )
            calibration.extend(examples)
        for probe, target, _, _ in calibration:
            model.observe(probe, target)
        frozen_count = model.pooled.count
        test: list[tuple[CandidateProbe, float, float, float]] = []
        pairs: list[DisagreementCase] = []
        mismatches = 0
        for seed in range(base + pretrain_episodes + calibration_episodes,
                          base + pretrain_episodes + calibration_episodes + held_out_episodes):
            examples, episode_pairs, episode_mismatches = _audit_episode(
                agent, seed, max_steps=max_steps,
                forecast_horizon=forecast_horizon,
                rollout_horizon=rollout_horizon, model=model,
            )
            test.extend(examples)
            pairs.extend(episode_pairs)
            mismatches += episode_mismatches
        if not test:
            raise RuntimeError("no complete held-out conflict targets")
        labels = [target for _, target, _, _ in test]
        ref_errors = [abs(probe.current_conflict - target) for probe, target, _, _ in test]
        pooled_errors = [abs(pooled - target) for _, target, pooled, _ in test]
        context_errors = [abs(context - target) for _, target, _, context in test]
        all_labels.extend(labels)
        all_reference_errors.extend(ref_errors)
        all_pooled_errors.extend(pooled_errors)
        all_context_errors.extend(context_errors)
        all_cases.extend(pairs)
        splits.append(ConflictSplit(
            agent_seed=agent_seed + split_index,
            calibration_labels=len(calibration),
            held_out_labels=len(test),
            target_mean=_mean(labels),
            target_std=sqrt(_variance(labels)),
            target_above_half_fraction=_mean([float(label >= 0.5) for label in labels]),
            reference_mae=_mean(ref_errors),
            pooled_mae=_mean(pooled_errors),
            contextual_mae=_mean(context_errors),
            ready_contexts=sum(
                stats.count >= model.min_context_samples for stats in model.contexts.values()
            ),
            ranking_disagreements=len(pairs),
            mean_reward_learned=_mean([case.learned_outcome.reward for case in pairs]),
            mean_reward_reference=_mean([case.reference_outcome.reward for case in pairs]),
            mean_collections_learned=_mean([case.learned_outcome.collections for case in pairs]),
            mean_collections_reference=_mean([case.reference_outcome.collections for case in pairs]),
            mean_lava_learned=_mean([case.learned_outcome.lava_steps for case in pairs]),
            mean_lava_reference=_mean([case.reference_outcome.lava_steps for case in pairs]),
            mean_unsafe_learned=_mean([case.learned_outcome.unsafe_steps for case in pairs]),
            mean_unsafe_reference=_mean([case.reference_outcome.unsafe_steps for case in pairs]),
            mean_conflict_learned=_mean([
                case.learned_outcome.future_conflict for case in pairs
                if case.learned_outcome.future_conflict is not None
            ]),
            mean_conflict_reference=_mean([
                case.reference_outcome.future_conflict for case in pairs
                if case.reference_outcome.future_conflict is not None
            ]),
            live_action_mismatches=mismatches,
            model_updates_during_test=model.pooled.count - frozen_count,
        ))
    frozen_splits = tuple(splits)
    disagreement_count = sum(split.ranking_disagreements for split in frozen_splits)
    variable = all(split.target_std >= 0.05 for split in frozen_splits)
    prediction = variable and all(
        split.contextual_mae < min(split.reference_mae, split.pooled_mae)
        for split in frozen_splits
    )
    unchanged = all(
        split.live_action_mismatches == 0 and split.model_updates_during_test == 0
        for split in frozen_splits
    )
    paired = [split for split in frozen_splits if split.ranking_disagreements]
    decision = (
        prediction and unchanged and disagreement_count >= 10 and len(paired) == split_count
        and all(
            split.mean_reward_learned >= split.mean_reward_reference
            and split.mean_collections_learned >= split.mean_collections_reference
            and split.mean_lava_learned <= split.mean_lava_reference
            and split.mean_unsafe_learned <= split.mean_unsafe_reference
            and split.mean_conflict_learned is not None
            and split.mean_conflict_reference is not None
            and split.mean_conflict_learned < split.mean_conflict_reference
            for split in paired
        )
    )
    return FutureConflictReport(
        forecast_horizon=forecast_horizon,
        rollout_horizon=rollout_horizon,
        splits=frozen_splits,
        total_held_out_labels=len(all_labels),
        total_ranking_disagreements=disagreement_count,
        target_std=sqrt(_variance(all_labels)),
        reference_mae=_mean(all_reference_errors),
        pooled_mae=_mean(all_pooled_errors),
        contextual_mae=_mean(all_context_errors),
        target_variable=variable,
        prediction_supported=prediction,
        decision_supported=decision,
        choice_audit=_choice_audit(all_cases),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--pretrain-episodes", type=int, default=20)
    parser.add_argument("--calibration-episodes", type=int, default=10)
    parser.add_argument("--held-out-episodes", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--forecast-horizon", type=int, default=5)
    parser.add_argument("--rollout-horizon", type=int, default=20)
    args = parser.parse_args()
    report = run_future_conflict_evaluation(
        split_count=args.splits,
        pretrain_episodes=args.pretrain_episodes,
        calibration_episodes=args.calibration_episodes,
        held_out_episodes=args.held_out_episodes,
        max_steps=args.max_steps,
        forecast_horizon=args.forecast_horizon,
        rollout_horizon=args.rollout_horizon,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
