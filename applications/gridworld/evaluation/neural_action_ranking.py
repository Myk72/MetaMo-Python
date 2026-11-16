"""Shadow action-ranking audit of frozen five-step neural state forecasts.

Force each safety/Q-eligible first action in clones, then resume the existing
policy for four moves. Forecasts use pre-action information only. A shared
ridge readout, calibrated on separate branches with actual future states,
maps state change to 100 * collections - cumulative energy loss. An actual-
future-state readout and a direct feature-to-utility baseline diagnose whether
forecast errors or the representation/readout limit action ranking.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, astuple, dataclass
from pathlib import Path

import numpy as np

from applications.gridworld.agents.metamo import MetaMoAgent
from applications.gridworld.config import ACTION_IDS
from applications.gridworld.environment import GridWorld
from applications.gridworld.runtime import build_candidates, build_stimulus, transition_for_action
from applications.gridworld.transition_record import MotivationalSnapshot, MoveForecastInputs
from applications.gridworld.evaluation.future_conflict import _probes
from applications.gridworld.evaluation.neural_self_model import (
    DIM, FEATURES, Dataset, NeuralSettings, _pretrain, _check_frozen_q, freeze_input, balanced_mae,
)
from applications.gridworld.evaluation.neural_robustness import FrozenForecasters, _fingerprint, _layout


@dataclass(frozen=True)
class RankingSettings:
    calibration_episodes: int = 20
    test_episodes: int = 24
    audit_stride: int = 5
    replicas: int = 3
    seed_start: int = 16000000
    seed_stride: int = 100000
    ridge_penalty: float = 1.0
    bootstrap_samples: int = 2000

    def validate(self):
        if (min(self.calibration_episodes, self.test_episodes, self.audit_stride,
                self.replicas, self.bootstrap_samples) < 1
                or self.seed_stride < self.calibration_episodes + self.test_episodes
                or not np.isfinite(self.ridge_penalty) or self.ridge_penalty <= 0):
            raise ValueError("invalid action-ranking settings")


def vector(state):
    snapshot = MotivationalSnapshot.from_state(state)
    return np.asarray(snapshot.goals + snapshot.modulators)


def candidate_data(agent, state, probes) -> Dataset:
    stimulus = build_stimulus(state, agent.mot, runtime=agent.runtime)
    candidates = build_candidates(state, agent.mot, runtime=agent.runtime)
    current, features, nominal = vector(agent.mot), [], []
    for probe in probes:
        action, staged, _, _ = transition_for_action(state, agent.mot, probe.action,
            stimulus=stimulus, candidates=candidates, runtime=agent.runtime)
        move = MoveForecastInputs(**{name: float(action.metadata[name]) for name in MoveForecastInputs.__dataclass_fields__})
        next_vector = vector(staged)
        features.append(tuple(current) + tuple(next_vector) + astuple(stimulus)
                        + tuple(float(index == probe.action) for index in range(agent.ACTIONS))
                        + astuple(move) + (probe.task_score, probe.safety_score, probe.base_score,
                                          float(np.tanh(probe.q_value / 100.0))))
        nominal.append(next_vector)
    count = len(probes)
    return Dataset(np.asarray(features), np.tile(current, (count, 1)), np.asarray(nominal),
                   np.zeros((count, DIM)), np.zeros(count, dtype=int), np.zeros(count, dtype=int),
                   np.asarray([probe.action for probe in probes]))


def branch(agent, environment, state, action, horizon, random_seed=None):
    """Exactly horizon total moves, including the forced first action."""
    agent, env = copy.deepcopy(agent), copy.deepcopy(environment)
    agent.record_transitions = False
    agent._pending_state = agent._pending_action = agent._pending_transition = None
    if random_seed is not None:
        agent.rng.seed(random_seed)
        env.rng.seed(random_seed + 1)
    _, staged, _, _ = transition_for_action(state, agent.mot, action, runtime=agent.runtime)
    agent.mot = staged
    collections, loss, moves, reward_sum = 0, 0.0, 0, 0.0
    for index in range(horizon):
        if index:
            action, _ = agent.select_action(state)
            agent.commit_motivational_transition()
        before = env.energy
        state, reward, done, _ = env.step(action)
        loss += max(0.0, before - env.energy)
        reward_sum += reward
        moves += 1
        if done:
            break
    collections = env.minerals_collected - environment.minerals_collected
    return {"state": vector(agent.mot).tolist(), "collections": collections, "energy_loss": loss,
            "utility": 100.0 * collections - loss, "moves": moves, "reward": reward_sum,
            "depleted": bool(env.energy <= 0), "complete": moves == horizon}


def readout_inputs(current, future, features):
    # Current state and known first-action cues are shared by all readouts.
    indices = [FEATURES.index("move.mineral_collected"), FEATURES.index("move.hazard_risk")]
    return np.column_stack((current, future - current, features[:, indices]))


@dataclass(frozen=True)
class UtilityReadout:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray

    @classmethod
    def fit(cls, x, target, penalty):
        mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)
        design = np.column_stack(((x - mean) / scale, np.ones(len(x))))
        regularizer = np.eye(design.shape[1]) * penalty
        regularizer[-1, -1] = 0
        coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ target)
        return cls(mean, scale, coefficients)

    def predict(self, x):
        return np.column_stack(((x - self.mean) / self.scale, np.ones(len(x)))) @ self.coefficients


def choose(scores, live_index):
    best = np.flatnonzero(np.isclose(scores, max(scores), rtol=0, atol=1e-9))
    return live_index if live_index in best else int(best[0])


def ranking_metrics(scores, outcomes, live_index, *, fixed_choice=None):
    selected = choose(scores, live_index) if fixed_choice is None else fixed_choice
    best = float(max(outcomes))
    pairs = []
    for left in range(len(outcomes)):
        for right in range(left):
            truth = outcomes[left] - outcomes[right]
            if abs(truth) > 1e-9:
                delta = scores[left] - scores[right]
                pairs.append(0.5 if abs(delta) <= 1e-9 else float(delta * truth > 0))
    return {"choice": selected, "utility": float(outcomes[selected]),
            "regret": best - float(outcomes[selected]),
            "optimal": bool(abs(outcomes[selected] - best) <= 1e-9),
            "informative": bool(np.ptp(outcomes) > 1e-9),
            "pairwise_correct_sum": float(sum(pairs)), "pairwise_pairs": len(pairs),
            "changed_choice": selected != live_index}


def collect(agent, seeds, original, settings, forecaster, progress, label):
    audits = []
    for count, seed in enumerate(seeds):
        env = GridWorld(seed=seed, max_steps=original.max_steps)
        state = env.reset()
        agent.reset_episode()
        agent.transition_records.clear()
        mirror, mirror_env = copy.deepcopy(agent), copy.deepcopy(env)
        mirror.record_transitions = False
        mirror_state = copy.deepcopy(state)
        for step in range(original.max_steps):
            action, alpha = agent.select_action(state)
            other_action, _ = mirror.select_action(mirror_state)
            if action != other_action:
                raise RuntimeError("audit altered live action")
            if step % settings.audit_stride == 0 and step + original.horizon <= original.max_steps:
                probes = _probes(copy.deepcopy(agent), state)
                if len(probes) >= 2:
                    data = candidate_data(agent, state, probes)
                    live_index = next(index for index, probe in enumerate(probes) if probe.action == action)
                    frozen = freeze_input(agent, action, alpha)
                    np.testing.assert_allclose(data.features[live_index], frozen.features, rtol=0, atol=1e-12)
                    # All candidate predictions are frozen before any branch outcome is read.
                    predictions = forecaster.predict(data)
                    outcomes = [[branch(agent, env, state, probe.action, original.horizon,
                                        None if replicate == 0 else seed * 10000 + step * 100 + replicate * 2)
                                 for replicate in range(settings.replicas)] for probe in probes]
                    audits.append({"episode": seed, "step": step, "live_index": live_index,
                                   "actions": data.action.tolist(), "current": data.current.tolist(),
                                   "features": data.features.tolist(),
                                   "predictions": {name: predictions[name].tolist() for name in
                                                   ("neural_32", "ridge", "nominal_trend")},
                                   "outcomes": outcomes})
            following, reward, done, _ = env.step(action)
            other, other_reward, other_done, _ = mirror_env.step(other_action)
            agent.commit_motivational_transition(next_env_state=following, done=done, executed_action=action)
            mirror.commit_motivational_transition()
            if (following != other or reward != other_reward or done != other_done
                    or not np.array_equal(agent.mot.G, mirror.mot.G)
                    or not np.array_equal(agent.mot.M, mirror.mot.M)
                    or agent.rng.getstate() != mirror.rng.getstate()):
                raise RuntimeError("audit altered live trajectory")
            state, mirror_state = following, other
            agent.transition_records.clear()
            if done:
                break
        if progress and ((count + 1) % 5 == 0 or count + 1 == len(seeds)):
            progress(f"{label}: {count + 1}/{len(seeds)} episodes, {len(audits)} audited states")
    return audits


def fit_readouts(audits, penalty):
    representations, raw, targets = [], [], []
    for audit in audits:
        for index, outcomes in enumerate(audit["outcomes"]):
            for outcome in outcomes:
                if outcome["complete"]:
                    current = np.asarray(audit["current"])[index:index + 1]
                    features = np.asarray(audit["features"])[index:index + 1]
                    representations.append(readout_inputs(current, np.asarray([outcome["state"]]), features)[0])
                    raw.append(features[0])
                    targets.append(outcome["utility"])
    if not targets:
        raise ValueError("no full-horizon calibration outcomes")
    return (UtilityReadout.fit(np.asarray(representations), np.asarray(targets), penalty),
            UtilityReadout.fit(np.asarray(raw), np.asarray(targets), penalty))


def evaluate(audits, readout, direct, bootstrap_samples, seed):
    decisions = []
    errors = {method: {kind: [] for kind in ("selected", "alternative")} for method in ("neural_32", "ridge", "nominal_trend")}
    incomplete = energy_events = branches = 0
    for audit in audits:
        current, features = np.asarray(audit["current"]), np.asarray(audit["features"])
        live_index = audit["live_index"]
        utility = np.asarray([np.mean([row["utility"] for row in outcomes]) for outcomes in audit["outcomes"]])
        scores = {method: readout.predict(readout_inputs(current, np.asarray(prediction), features))
                  for method, prediction in audit["predictions"].items()}
        scores["direct_features"] = direct.predict(features)
        scores["current_selector"] = features[:, FEATURES.index("policy.composed_score")]
        complete = all(row["complete"] for outcomes in audit["outcomes"] for row in outcomes)
        if complete:
            actual = np.asarray([np.mean([row["state"] for row in outcomes], axis=0) for outcomes in audit["outcomes"]])
            scores["actual_state_readout"] = readout.predict(readout_inputs(current, actual, features))
        for index, outcomes in enumerate(audit["outcomes"]):
            for row in outcomes:
                branches += 1
                energy_events += int(row["energy_loss"] > 0)
                incomplete += int(not row["complete"])
                if row["complete"]:
                    for method, prediction in audit["predictions"].items():
                        errors[method]["selected" if index == live_index else "alternative"].append(
                            balanced_mae(np.asarray([prediction[index]]), np.asarray([row["state"]])))
        methods = {}
        for name, value in scores.items():
            # Preserve the recorded random tie choice; use the real composed
            # scores (not a one-hot choice) for the selector's pairwise ranking.
            result = ranking_metrics(value, utility, live_index,
                                     fixed_choice=live_index if name == "current_selector" else None)
            selected = audit["outcomes"][result["choice"]]
            result.update(collections=float(np.mean([row["collections"] for row in selected])),
                          energy_loss=float(np.mean([row["energy_loss"] for row in selected])))
            methods[name] = result
        decisions.append({"episode": audit["episode"], "step": audit["step"], "methods": methods})
    summary = {}
    names = sorted({method for row in decisions for method in row["methods"]})
    for name in names:
        rows = [row["methods"][name] for row in decisions if name in row["methods"]]
        informative = [row for row in rows if row["informative"]]
        pairs = sum(row["pairwise_pairs"] for row in rows)
        summary[name] = {"states": len(rows), "informative_states": len(informative),
                         "mean_utility": float(np.mean([row["utility"] for row in rows])),
                         "mean_regret": float(np.mean([row["regret"] for row in rows])),
                         "mean_collections": float(np.mean([row["collections"] for row in rows])),
                         "mean_energy_loss": float(np.mean([row["energy_loss"] for row in rows])),
                         "changed_choices": sum(row["changed_choice"] for row in rows),
                         "optimal_fraction_informative": float(np.mean([row["optimal"] for row in informative])) if informative else None,
                         "pairwise_accuracy": sum(row["pairwise_correct_sum"] for row in rows) / pairs if pairs else None}
    comparisons = {}
    rng = np.random.default_rng(seed)
    for reference in ("current_selector", "ridge", "direct_features", "actual_state_readout"):
        paired = [row for row in decisions if reference in row["methods"]]
        episode_gains = [np.mean([row["methods"]["neural_32"]["utility"] - row["methods"][reference]["utility"]
                                 for row in paired if row["episode"] == episode])
                         for episode in sorted({row["episode"] for row in paired})]
        interval = None
        if len(episode_gains) >= 2:
            values = np.asarray(episode_gains)
            samples = values[rng.integers(len(values), size=(bootstrap_samples, len(values)))].mean(axis=1)
            interval = np.quantile(samples, [0.025, 0.975]).tolist()
        comparisons[reference] = {"episodes": len(episode_gains), "mean_episode_utility_gain": float(np.mean(episode_gains)) if episode_gains else None,
                                  "bootstrap_95_percent_interval": interval}
    return {"audited_states": len(audits), "branches": branches, "energy_loss_branches": energy_events,
            "incomplete_branches": incomplete, "methods": summary, "neural_comparisons": comparisons,
            "state_mae": {method: {kind: {"count": len(values), "mae": float(np.mean(values)) if values else None}
                                   for kind, values in groups.items()} for method, groups in errors.items()}}, decisions


def _write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_ranking(source, curve, output_dir, settings=RankingSettings(), progress=None):
    settings.validate()
    original = NeuralSettings(**json.loads((source / "settings.json").read_text()))
    original.validate()
    if original.horizon != 5:
        raise ValueError("this action audit requires the trained five-step model")
    for protected in (source, curve):
        if output_dir.resolve() == protected.resolve() or protected.resolve() in output_dir.resolve().parents:
            raise ValueError("outputs must be outside source directories")
    old_protocol = json.loads((curve / "protocol.json").read_text())
    old_layouts = set(old_protocol["new_lava_layout_sha256"].values())
    old_seeds = set(map(int, old_protocol["new_lava_layout_sha256"]))
    count = original.pretrain_episodes + original.train_episodes + original.validation_episodes + original.test_episodes
    old_seeds.update(original.seed_start + index * original.seed_stride + offset
                     for index in range(original.splits) for offset in range(count))
    for path in old_protocol["previous_protocol_sha256"]:
        prior = json.loads(Path(path).read_text())
        for split in prior["splits"]:
            old_seeds.update(range(split["fresh_seed_start"], split["fresh_seed_stop_exclusive"]))
    old_layouts.update(_layout(seed, original.max_steps) for seed in old_seeds)
    seeds = [settings.seed_start + index * settings.seed_stride + offset for index in range(original.splits)
             for offset in range(settings.calibration_episodes + settings.test_episodes)]
    layouts = [_layout(seed, original.max_steps) for seed in seeds]
    if old_seeds.intersection(seeds) or old_layouts.intersection(layouts) or len(set(layouts)) != len(layouts):
        raise ValueError("choose a new seed block: previous seed/layout overlap")
    model_paths = [curve / f"split_{index + 1}" / "model_320.npz" for index in range(original.splits)]
    hashes = {str(path): _fingerprint(path) for path in model_paths}
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {"settings": asdict(settings), "original_settings": asdict(original), "model_sha256": hashes,
                "objective": "100 * mineral collections - cumulative energy loss over exactly five total moves",
                "candidate_set": "existing hard-safety mask and Q shortlist; states with >=2 candidates, every fifth step by default",
                "readout": "fixed-penalty ridge on actual future-state deltas, current state and known immediate collection/hazard cues; calibration only",
                "direct_baseline": "ridge from same 44 pre-action inputs directly to utility, on same calibration branches",
                "replicas": "replica 0 preserves snapshot RNG; other replicas share fresh continuation RNG seeds across actions",
                "tie_break": "retain live action on equal predicted scores; otherwise smallest action index",
                "decision_criterion": "NN utility gain over current selector has positive episode-bootstrap lower bound in every split; no higher mean energy loss",
                "limits": ["Five-step branch ranking is not an episode-level deployment trial.",
                           "Action alternatives are outside the original selected-action training distribution.",
                           "Actual-state readout is an unavailable-at-decision diagnostic, not a performance bound.",
                           "Oracle regret uses finite replica means and can be optimistic; bootstrap resamples whole source episodes.",
                           "Full-horizon-only state accuracy/readout fitting; early-terminal utility retained.",
                           "No energy variation means no evidence of improved safety."],
                "new_lava_layout_sha256": dict(zip(seeds, layouts))}
    _write(output_dir / "protocol.json", protocol)
    prepared = []
    for index, path in enumerate(model_paths):
        folder = output_dir / f"split_{index + 1}"
        folder.mkdir(exist_ok=True)
        agent = MetaMoAgent(seed=original.agent_seed + index)
        base = original.seed_start + index * original.seed_stride
        _pretrain(agent, range(base, base + original.pretrain_episodes), original)
        agent.epsilon = 0
        agent.record_transitions = True
        agent.predictive_self_model.learning_enabled = False
        q, visits = copy.deepcopy(agent.q_table), copy.deepcopy(agent.visit_counts)
        model = FrozenForecasters(path, original.horizon)
        base = settings.seed_start + index * settings.seed_stride
        agent.rng.seed(base)
        calibration = collect(agent, range(base, base + settings.calibration_episodes), original,
                              settings, model, progress, f"Split {index + 1} calibration")
        _write(folder / "calibration.json", calibration)
        readout, direct = fit_readouts(calibration, settings.ridge_penalty)
        np.savez_compressed(folder / "readouts.npz", **{f"state_{key}": value for key, value in readout.__dict__.items()},
                            **{f"direct_{key}": value for key, value in direct.__dict__.items()})
        prepared.append((agent, model, readout, direct, q, visits))
    reports = []
    for index, (agent, model, readout, direct, q, visits) in enumerate(prepared):
        folder = output_dir / f"split_{index + 1}"
        start = settings.seed_start + index * settings.seed_stride + settings.calibration_episodes
        agent.rng.seed(start)
        audits = collect(agent, range(start, start + settings.test_episodes), original, settings, model,
                         progress, f"Split {index + 1} held-out")
        _check_frozen_q(agent, q, visits)
        if agent.predictive_self_model.collection_observation_count or any(
                agent.predictive_self_model.sample_count(action) for action in ACTION_IDS):
            raise RuntimeError("diagnostic self-model learned during audit")
        report, decisions = evaluate(audits, readout, direct, settings.bootstrap_samples, start)
        report.update(split=index + 1, live_behavior_unchanged=True, q_frozen=True)
        _write(folder / "test.json", audits)
        _write(folder / "decisions.json", decisions)
        _write(folder / "report.json", report)
        reports.append(report)
        if progress:
            progress(f"Split {index + 1}: NN gain over current selector {report['neural_comparisons']['current_selector']}")
    for path, fingerprint in hashes.items():
        if _fingerprint(Path(path)) != fingerprint:
            raise RuntimeError("forecast model changed")
    aggregate = {}
    for method in reports[0]["methods"]:
        rows = [report["methods"][method] for report in reports]
        aggregate[method] = {key: float(np.average([row[key] for row in rows], weights=[row["states"] for row in rows]))
                             for key in ("mean_utility", "mean_regret", "mean_collections", "mean_energy_loss")}
    return {"protocol": protocol, "splits": reports, "aggregate": aggregate,
            "audited_states": sum(report["audited_states"] for report in reports),
            "branches": sum(report["branches"] for report in reports), "forecast_models_unchanged": True,
            "nn_improves_current_selector_in_every_split": all(
                report["neural_comparisons"]["current_selector"]["bootstrap_95_percent_interval"] is not None
                and report["neural_comparisons"]["current_selector"]["bootstrap_95_percent_interval"][0] > 0
                and report["methods"]["neural_32"]["mean_energy_loss"] <= report["methods"]["current_selector"]["mean_energy_loss"]
                for report in reports)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/self_model/neural_five_step"))
    parser.add_argument("--curve", type=Path, default=Path("artifacts/self_model/neural_learning_curve"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/self_model/neural_action_ranking"))
    parser.add_argument("--summary", type=Path, default=Path("artifacts/self_model/neural_action_ranking.summary.json"))
    args = parser.parse_args()
    report = run_ranking(args.source, args.curve, args.output_dir, progress=lambda value: print(value, flush=True))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    _write(args.summary, report)
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
