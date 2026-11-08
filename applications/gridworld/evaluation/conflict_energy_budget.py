"""Test conflict as a safety warning in an experimental energy-budget world.

Scenario: movement costs 5 energy, initial energy is 60, collecting a mineral
recharges before the movement cost, and depletion terminates with -100 reward.
The standard GridWorld, its profiles, and the live selector are not changed.

Protocol: train Q, freeze it, calibrate the existing conflict estimator on
separate episodes, then freeze that estimator too. Every third live decision
audits the safe Q shortlist in cloned worlds. A branch contains the forced
action plus ten continuation moves under the frozen policy. Time-limit tails
are skipped, so a non-depleting branch always completes that safety horizon.
Conflict MAE uses only complete five-decision targets; safety metrics retain
early deaths. Candidate rows and full settings can be saved for later analysis.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from applications.gridworld.agents.metamo import MetaMoAgent
from applications.gridworld.config import ACTION_DELTAS, GRID_SIZE, REWARD_LAVA
from applications.gridworld.environment import GridWorld
from applications.gridworld.evaluation.future_conflict import (
    BranchResult, CandidateProbe, FutureConflictModel, _branch, _mean, _probes,
)


class EnergyBudgetGridWorld(GridWorld):
    """Experimental extension: charge each move after any mineral recharge."""

    def __init__(self, *, energy_cost: float = 5.0, initial_energy: float = 60.0,
                 depletion_penalty: float = 100.0, **kwargs):
        if (
            not np.isfinite(energy_cost) or energy_cost <= 0
            or not np.isfinite(initial_energy) or not 0 < initial_energy <= 100
            or not np.isfinite(depletion_penalty) or depletion_penalty < 0
        ):
            raise ValueError("invalid experimental energy settings")
        self.energy_cost = energy_cost
        self.initial_energy = initial_energy
        self.depletion_penalty = depletion_penalty
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self.energy = self.initial_energy
        return self._get_state()

    def step(self, action: int):
        if self.done:
            return self._get_state(), 0.0, True, {"event": None}
        _, reward, _, info = super().step(action)
        self.energy = max(0.0, self.energy - self.energy_cost)
        if self.energy <= 0.0:
            self.done = True
            info["event"] = "dead"
            reward -= self.depletion_penalty
        return self._get_state(), reward, self.done, info


@dataclass(frozen=True)
class EnergySettings:
    splits: int = 3
    pretrain_episodes: int = 20
    calibration_episodes: int = 12
    test_episodes: int = 8
    max_steps: int = 40
    forecast_horizon: int = 5
    continuation_moves: int = 10
    audit_every: int = 3
    seed_start: int = 6000000
    seed_stride: int = 100000
    agent_seed: int = 90
    energy_cost: float = 5.0
    initial_energy: float = 60.0
    depletion_penalty: float = 100.0

    def validate(self):
        if (
            min(self.splits, self.pretrain_episodes, self.calibration_episodes,
                self.test_episodes, self.audit_every, self.forecast_horizon) < 1
            or self.continuation_moves < self.forecast_horizon
            or self.max_steps <= self.continuation_moves
            or self.seed_stride < self.pretrain_episodes + self.calibration_episodes + self.test_episodes
        ):
            raise ValueError("invalid energy experiment settings")

    def environment(self, seed: int) -> EnergyBudgetGridWorld:
        return EnergyBudgetGridWorld(
            seed=seed, max_steps=self.max_steps, energy_cost=self.energy_cost,
            initial_energy=self.initial_energy, depletion_penalty=self.depletion_penalty,
        )


def _energy_shortfalls(agent: MetaMoAgent, state: dict,
                       probes: tuple[CandidateProbe, ...], cost: float) -> list[float]:
    """Pre-action reference: required energy minus remaining energy.

    Use shortest paths through visible non-lava cells to the current mineral.
    For a collecting action, assume no knowledge of the random respawn: credit
    the known recharge and set remaining distance to the just-reached goal to 0.
    Higher values indicate a tighter energy budget, not a death probability.
    """
    target = state["mineral_pos"]
    lava = set(state["lava_cells"])
    distances = {target: 0}
    pending = deque([target])
    while pending:
        row, col = pending.popleft()
        for dr, dc in ACTION_DELTAS:
            cell = (row + dr, col + dc)
            if (0 <= cell[0] < GRID_SIZE and 0 <= cell[1] < GRID_SIZE
                    and cell not in lava and cell not in distances):
                distances[cell] = distances[(row, col)] + 1
                pending.append(cell)
    result = []
    for probe in probes:
        pos = agent.runtime.profile.project_move(state, probe.action)
        energy = state["energy"] + (REWARD_LAVA if pos in lava else 0.0)
        if probe.collection:
            energy = min(100.0, energy + 100.0)
        after = max(0.0, energy - cost)
        distance = distances.get(pos, GRID_SIZE ** 2 + 1)
        result.append(cost * distance - after)
    return result


@dataclass(frozen=True)
class EnergyRow:
    episode_seed: int
    step: int
    live_action: int
    energy_before: float
    probe: CandidateProbe
    predicted_conflict: float | None
    energy_shortfall: float
    outcome: BranchResult


def _train(agent: MetaMoAgent, env: EnergyBudgetGridWorld):
    state = env.reset()
    agent.reset_episode()
    for _ in range(env.max_steps):
        action, alpha = agent.select_action(state)
        following, reward, done, info = env.step(action)
        agent.update(state, action, reward, following, done, info.get("event"), alpha)
        state = following
        if done:
            break
    agent.decay_epsilon()


def _collect(agent: MetaMoAgent, seed: int, settings: EnergySettings,
             model: FutureConflictModel | None) -> list[EnergyRow]:
    env = settings.environment(seed)
    state = env.reset()
    agent.reset_episode()
    # Mirror the live trajectory without running the candidate probes. This
    # checks action, RNG, state, and reward parity throughout the episode.
    mirror = copy.deepcopy(agent)
    mirror_env = copy.deepcopy(env)
    mirror_state = copy.deepcopy(state)
    rows = []
    for step in range(settings.max_steps):
        action, _ = agent.select_action(state)
        mirror_action, _ = mirror.select_action(mirror_state)
        if action != mirror_action:
            raise RuntimeError("candidate auditing changed the live action")
        enough_time = env.max_steps - env.step_count >= settings.continuation_moves + 1
        if step % settings.audit_every == 0 and enough_time:
            probes = _probes(agent, state)
            forecasts = [model.predict(p, contextual=True) if model else None for p in probes]
            margins = _energy_shortfalls(agent, state, probes, settings.energy_cost)
            for probe, prediction, shortfall in zip(probes, forecasts, margins):
                outcome = _branch(
                    agent, env, state, probe.action,
                    forecast_horizon=settings.forecast_horizon,
                    rollout_horizon=settings.continuation_moves,
                )
                rows.append(EnergyRow(
                    seed, step, action, float(state["energy"]), probe,
                    prediction, shortfall, outcome,
                ))
        following, reward, done, _ = env.step(action)
        mirror_following, mirror_reward, mirror_done, _ = mirror_env.step(mirror_action)
        agent.commit_motivational_transition()
        mirror.commit_motivational_transition()
        if (
            following != mirror_following or reward != mirror_reward or done != mirror_done
            or not np.array_equal(agent.mot.G, mirror.mot.G)
            or not np.array_equal(agent.mot.M, mirror.mot.M)
            or agent.rng.getstate() != mirror.rng.getstate()
        ):
            raise RuntimeError("candidate auditing changed the live trajectory")
        state, mirror_state = following, mirror_following
        if done:
            break
    return rows


def _auc(scores: list[float], labels: list[bool]) -> float | None:
    """Probability that a randomly chosen death has a higher warning score.

    Give ties half credit. None means one outcome class is absent. Candidate
    branches are correlated: these values are descriptive, not significance tests.
    """
    positive = np.asarray([s for s, label in zip(scores, labels) if label])
    negative = np.asarray([s for s, label in zip(scores, labels) if not label])
    if not len(positive) or not len(negative):
        return None
    differences = positive[:, None] - negative[None, :]
    return float(np.mean((differences > 0) + 0.5 * (differences == 0)))


def _policy_score(row: EnergyRow, learned: bool) -> float:
    conflict = row.predicted_conflict if learned else row.probe.current_conflict
    return row.probe.base_score + float(row.probe.collection) - 0.25 * conflict


def _summarize(rows: list[EnergyRow]) -> dict:
    if not rows:
        raise ValueError("no held-out energy audit rows")
    groups: dict[tuple[int, int], list[EnergyRow]] = {}
    for row in rows:
        groups.setdefault((row.episode_seed, row.step), []).append(row)
    decisions = {name: [] for name in ("live", "reference_conflict", "learned_conflict", "energy_reference")}
    for group in groups.values():
        decisions["live"].append(next(row for row in group if row.probe.action == row.live_action))
        for name, learned in (("reference_conflict", False), ("learned_conflict", True)):
            decisions[name].append(max(group, key=lambda row: (_policy_score(row, learned), -row.probe.action)))
        decisions["energy_reference"].append(max(
            group, key=lambda row: (-row.energy_shortfall, _policy_score(row, False), -row.probe.action)
        ))
    summaries = {}
    for name, chosen in decisions.items():
        summaries[name] = {
            "depleted_branches": sum(row.outcome.energy_depleted for row in chosen),
            "mean_reward": _mean([row.outcome.reward for row in chosen]),
            "mean_collections": _mean([row.outcome.collections for row in chosen]),
            "lava_steps": sum(row.outcome.lava_steps for row in chosen),
            "motivational_unsafe_steps": sum(row.outcome.unsafe_steps for row in chosen),
        }
    pairs = list(zip(decisions["learned_conflict"], decisions["reference_conflict"]))
    changed = [(a, b) for a, b in pairs if a.probe.action != b.probe.action]
    complete = [row for row in rows if row.outcome.future_conflict is not None]
    labels = [row.outcome.energy_depleted for row in rows]
    return {
        "audited_states": len(groups),
        "candidate_branches": len(rows),
        "depleted_branches": sum(labels),
        "surviving_branches": len(labels) - sum(labels),
        "states_with_both_depleting_and_surviving_actions": sum(
            any(row.outcome.energy_depleted for row in group)
            and any(not row.outcome.energy_depleted for row in group)
            for group in groups.values()
        ),
        "complete_conflict_labels": len(complete),
        "conflict_labels_missing_due_to_early_depletion": len(rows) - len(complete),
        "current_conflict_mae_on_complete_labels": _mean([
            abs(row.probe.current_conflict - row.outcome.future_conflict) for row in complete
        ]),
        "learned_conflict_mae_on_complete_labels": _mean([
            abs(row.predicted_conflict - row.outcome.future_conflict) for row in complete
        ]),
        "depletion_auc": {
            "current_conflict": _auc([row.probe.current_conflict for row in rows], labels),
            "learned_conflict": _auc([row.predicted_conflict for row in rows], labels),
            "energy_reference": _auc([row.energy_shortfall for row in rows], labels),
        },
        "choice_outcomes": summaries,
        "changed_choices": len(changed),
        "depletions_prevented_vs_reference": sum(
            not a.outcome.energy_depleted and b.outcome.energy_depleted for a, b in changed
        ),
        "depletions_introduced_vs_reference": sum(
            a.outcome.energy_depleted and not b.outcome.energy_depleted for a, b in changed
        ),
        "mean_reward_difference_on_changed_choices": _mean([
            a.outcome.reward - b.outcome.reward for a, b in changed
        ]),
    }


def _check_frozen_q(agent: MetaMoAgent, values: dict, visits: dict):
    for key, value in agent.q_table.items():
        expected = values.get(key, np.zeros(agent.ACTIONS))
        expected_visits = visits.get(key, np.zeros(agent.ACTIONS))
        if not np.array_equal(value, expected) or not np.array_equal(agent.visit_counts[key], expected_visits):
            raise RuntimeError("Q values or visit counts changed after pretraining")


def run_energy_experiment(settings: EnergySettings = EnergySettings(),
                          progress: Callable[[str], None] | None = None) -> dict:
    settings.validate()
    splits = []
    traces = []
    for index in range(settings.splits):
        prefix = f"Split {index + 1}/{settings.splits}"
        base = settings.seed_start + index * settings.seed_stride
        agent = MetaMoAgent(seed=settings.agent_seed + index)
        if progress:
            progress(f"{prefix}: pretraining Q")
        for seed in range(base, base + settings.pretrain_episodes):
            _train(agent, settings.environment(seed))
        agent.epsilon = 0.0
        q_before = copy.deepcopy(agent.q_table)
        visits_before = copy.deepcopy(agent.visit_counts)
        calibration_start = base + settings.pretrain_episodes
        test_start = calibration_start + settings.calibration_episodes
        model = FutureConflictModel()
        calibration_rows = []
        if progress:
            progress(f"{prefix}: calibrating conflict model")
        for seed in range(calibration_start, test_start):
            calibration_rows.extend(_collect(agent, seed, settings, model=None))
        for row in calibration_rows:
            if row.outcome.future_conflict is not None:
                model.observe(row.probe, row.outcome.future_conflict)
        if model.pooled.count == 0:
            raise RuntimeError("no complete conflict targets for calibration")
        model_before = (model.pooled, dict(model.contexts))
        held_out = []
        if progress:
            progress(f"{prefix}: held-out paired branches")
        for seed in range(test_start, test_start + settings.test_episodes):
            held_out.extend(_collect(agent, seed, settings, model=model))
        if model_before != (model.pooled, model.contexts):
            raise RuntimeError("conflict model changed during held-out evaluation")
        _check_frozen_q(agent, q_before, visits_before)
        summary = _summarize(held_out)
        split = {
            "agent_seed": settings.agent_seed + index,
            "pretrain_seed_start": base,
            "calibration_seed_start": calibration_start,
            "test_seed_start": test_start,
            "calibration_candidate_branches": len(calibration_rows),
            "calibration_complete_conflict_labels": model.pooled.count,
            "live_trajectory_parity": True,
            "q_and_model_frozen": True,
            **summary,
        }
        splits.append(split)
        traces.extend(asdict(row) for row in held_out)
        if progress:
            auc = summary["depletion_auc"]
            progress(f"{prefix}: {summary['depleted_branches']}/{summary['candidate_branches']} "
                     f"branches depleted; warning AUC {auc}")
    eligible = all(s["depleted_branches"] >= 10 and s["surviving_branches"] >= 10 for s in splits)
    actionable = sum(s["states_with_both_depleting_and_surviving_actions"] for s in splits) >= 10
    warning = eligible and all(
        s["depletion_auc"]["learned_conflict"] > max(
            0.5, s["depletion_auc"]["current_conflict"], s["depletion_auc"]["energy_reference"]
        ) for s in splits
    )
    decision = eligible and actionable and all(
        s["choice_outcomes"]["learned_conflict"]["depleted_branches"]
        < s["choice_outcomes"]["reference_conflict"]["depleted_branches"]
        and s["choice_outcomes"]["learned_conflict"]["mean_reward"]
        >= s["choice_outcomes"]["reference_conflict"]["mean_reward"]
        and s["choice_outcomes"]["learned_conflict"]["mean_collections"]
        >= s["choice_outcomes"]["reference_conflict"]["mean_collections"]
        and s["choice_outcomes"]["learned_conflict"]["lava_steps"]
        <= s["choice_outcomes"]["reference_conflict"]["lava_steps"]
        and s["choice_outcomes"]["learned_conflict"]["motivational_unsafe_steps"]
        <= s["choice_outcomes"]["reference_conflict"]["motivational_unsafe_steps"]
        for s in splits
    )
    return {
        "settings": asdict(settings),
        "method": {
            "target": "mean conflict across next five decisions, if all occur",
            "safety_outcome": "energy <= 0 after forced action or within continuation moves",
            "energy_reference": "known shortest non-lava distance energy minus post-move energy",
            "conflict_weight": 0.25,
            "candidate_rows_are_correlated": True,
            "results_are_exploratory": True,
            "conflict_mae_excludes_early_deaths": True,
        },
        "splits": splits,
        "enough_depletion_and_survival_examples": eligible,
        "enough_states_with_action_dependent_safety": actionable,
        "conflict_warning_adds_value_over_energy_reference": warning,
        "conflict_ranking_improves_safety_without_task_loss": decision,
        "held_out_rows": traces,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_energy_experiment(progress=lambda message: print(message, file=sys.stderr, flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "held_out_rows"}, indent=2))
    print(f"Full result and candidate traces saved to {args.output}")


if __name__ == "__main__":
    main()
