"""Replay switch-feasible targets through MAGUS and the actual GridWorld selector.

This is a focused diagnosis, not a new training run. It uses the saved target
dataset and trained NN. A frozen Q agent is trained once per requested seed;
each stored decision is then replayed with rule, exact-target, and NN appraisal
under identical Q values and RNG state.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from applications.gridworld.evaluation.runner import _train_agent, _variant_specs
from applications.gridworld.learned_appraisal import with_learned_appraisal
from applications.gridworld.runtime import (
    build_candidates, build_stimulus, perspective_candidate_scores,
)
from experiment.gridworld.targets import (
    _reconstruct_candidates, _reconstruct_state, _scores_for_residual,
)
from experiment.gridworld.train_test import _load_inputs
from experiment.network import ResidualMLP
from core.features import GoalChangeFeedback
from dynamics.stability import DefaultStabilityPolicy
from magus.decision import MagusDecision
from applications.gridworld.profile import GRIDWORLD_DECISION_PROFILE
from openpsi.hybrid import HybridAppraisalConfig, HybridAppraisalMode, bound_appraisal_residual
from openpsi.learned import LearnedAppraisalManifest


class ContextTargetPredictor:
    """Apply an exact target only at its recorded manifest input vector."""

    def __init__(self, input_vector: np.ndarray, target: np.ndarray):
        self.input_vector = np.asarray(input_vector, dtype=float)
        self.target = np.asarray(target, dtype=float)
        self.matches = 0

    def predict(self, input_vector: np.ndarray) -> np.ndarray:
        if np.allclose(input_vector, self.input_vector, rtol=0.0, atol=1e-10):
            self.matches += 1
            return self.target.copy()
        return np.zeros_like(self.target)


def _environment_state(values: dict) -> dict:
    """Undo JSON's list conversion so lava membership has normal semantics."""
    state = dict(values)
    state["pos"] = tuple(values["pos"])
    state["mineral_pos"] = tuple(values["mineral_pos"])
    state["lava_cells"] = tuple(tuple(cell) for cell in values["lava_cells"])
    return state


def _margin(scores: np.ndarray, oracle: int) -> float:
    return float(scores[oracle] - max(scores[index] for index in range(len(scores)) if index != oracle))


def _replay_selector(agent, environment_state: dict, *, predictor=None,
                     manifest: LearnedAppraisalManifest | None = None) -> dict[str, Any]:
    replay = copy.deepcopy(agent)
    if predictor is not None:
        if manifest is None:
            raise ValueError("learned appraisal requires a manifest")
        replay.runtime = with_learned_appraisal(
            replay.runtime,
            mode=HybridAppraisalMode.ENABLED,
            predictor=predictor,
            manifest=manifest,
        )
    captured: dict[str, Any] = {}
    stimulus = build_stimulus(environment_state, replay.mot, runtime=replay.runtime)
    candidates = build_candidates(environment_state, replay.mot, runtime=replay.runtime)
    perspectives = perspective_candidate_scores(
        replay.mot, stimulus, candidates, environment_state, runtime=replay.runtime,
    )
    captured["task_scores"] = perspectives.task.copy()
    captured["safety_scores"] = perspectives.safety.copy()
    captured["consensus_scores"] = perspectives.consensus.copy()
    exploit_actions = replay._valid_actions(
        environment_state, avoid_lava=replay.mask_lava_on_exploit,
    )
    normalized_task, normalized_safety, _, _ = replay._composed_preferences(
        perspectives, exploit_actions,
    )
    captured["normalized_task_scores"] = normalized_task
    captured["normalized_safety_scores"] = normalized_safety
    captured["scoring_context_matches"] = int(getattr(predictor, "matches", 0))
    original = replay._argmax_with_random_tie

    def capture(scores, actions):
        scores = np.asarray(scores, dtype=float)
        captured["shortlist"] = tuple(int(action) for action in actions)
        captured["tied_actions"] = tuple(int(action) for action in replay._tied_best_actions(scores, actions))
        captured["scores"] = scores.copy()
        return original(scores, actions)

    replay._argmax_with_random_tie = capture
    executed, info = replay.select_action(environment_state)
    if "shortlist" not in captured:
        raise AssertionError("selector did not reach the exploit/tie-breaking path")
    return {
        "executed": int(executed),
        "greedy": int(info["greedy_action"]),
        "shortlist": captured["shortlist"],
        "ties": captured["tied_actions"],
        "scores": captured["scores"],
        "task_scores": captured["task_scores"],
        "safety_scores": captured["safety_scores"],
        "consensus_scores": captured["consensus_scores"],
        "normalized_task_scores": captured["normalized_task_scores"],
        "normalized_safety_scores": captured["normalized_safety_scores"],
        "scoring_context_matches": captured["scoring_context_matches"],
        "predictor_matches": int(getattr(predictor, "matches", 0)),
    }


def _train_frozen_q(seed: int, episodes: int, max_steps: int, danger_probability: float):
    variant = _variant_specs()["QTrainMetaMoEval"]
    agent = variant.factory(seed)
    _train_agent(
        agent, variant, episodes,
        seed_start=seed * 100000,
        max_steps=max_steps,
        danger_distance=2,
        danger_mineral_probability=danger_probability,
    )
    agent.epsilon = 0.0
    return agent


def _one_context(
    row: dict, record: dict, manifest: LearnedAppraisalManifest,
    model: ResidualMLP, trained_agents: dict[int, Any],
) -> tuple[dict, list[dict]]:
    context = next(item for item in record["contexts"] if item["perspective"] == row["perspective"])
    oracle = int(row["oracle_action_index"])
    target_prediction = np.asarray(row["target_normalized"], dtype=float)
    nn_prediction = model.predict(np.asarray(row["input_vector"], dtype=float))
    pressure = float(row["boundary_pressure"])
    target_residual, _ = bound_appraisal_residual(
        manifest, HybridAppraisalConfig(), target_prediction, pressure,
    )
    nn_residual, _ = bound_appraisal_residual(
        manifest, HybridAppraisalConfig(), nn_prediction, pressure,
    )
    rule_state = _reconstruct_state(context["rule_state"])
    candidates = _reconstruct_candidates(record["candidates"])
    feedback = GoalChangeFeedback(context["goal_change_feedback"])
    decision = MagusDecision(profile=GRIDWORLD_DECISION_PROFILE)
    stability = DefaultStabilityPolicy()
    nn_scores = _scores_for_residual(
        rule_state=rule_state,
        residual=nn_residual,
        output_modulator_names=manifest.output_modulator_names,
        candidates=candidates,
        feedback=feedback,
        decision=decision,
        stability=stability,
    )
    target_scores = np.asarray(row["target_scores"], dtype=float)
    baseline_scores = np.asarray(row["baseline_scores"], dtype=float)
    if not np.allclose(target_residual, row["target_residual"], rtol=0.0, atol=1e-10):
        raise AssertionError("exact target changed during replay")
    if int(np.argmax(target_scores)) != oracle:
        raise AssertionError("switch-feasible target does not select oracle")
    target_norm = float(np.linalg.norm(target_residual))
    nn_norm = float(np.linalg.norm(nn_residual))
    cosine = (
        float(np.dot(target_residual, nn_residual) / (target_norm * nn_norm))
        if target_norm > 0 and nn_norm > 0 else None
    )
    case = {
        "target_id": row["target_id"],
        "split": row["split"],
        "group_id": row["group_id"],
        "regime": row["regime"],
        "perspective": row["perspective"],
        "oracle_action": oracle,
        "baseline_magus_action": int(np.argmax(baseline_scores)),
        "target_magus_action": int(np.argmax(target_scores)),
        "nn_magus_action": int(np.argmax(nn_scores)),
        "baseline_oracle_margin": _margin(baseline_scores, oracle),
        "target_oracle_margin": _margin(target_scores, oracle),
        "nn_oracle_margin": _margin(nn_scores, oracle),
        "target_residual_norm": target_norm,
        "nn_residual_norm": nn_norm,
        "nn_target_norm_ratio": nn_norm / target_norm if target_norm else None,
        "nn_target_direction_cosine": cosine,
    }
    environment_state = _environment_state(record["environment_state"])
    motivational_state = _reconstruct_state(record["motivational_state"])
    probe_rows = []
    for agent_seed, trained in trained_agents.items():
        base = copy.deepcopy(trained)
        base.mot = motivational_state.copy()
        base._pending_state = None
        base._pending_action = None
        base.epsilon = 0.0
        rule = _replay_selector(base, environment_state)
        exact_predictor = ContextTargetPredictor(
            np.asarray(row["input_vector"], dtype=float), target_prediction,
        )
        exact = _replay_selector(
            base, environment_state, predictor=exact_predictor, manifest=manifest,
        )
        nn = _replay_selector(base, environment_state, predictor=model, manifest=manifest)
        probe_rows.append({
            "target_id": row["target_id"],
            "split": row["split"],
            "perspective": row["perspective"],
            "agent_seed": agent_seed,
            "exact_context_matched": exact["predictor_matches"] > 0,
            "exact_scoring_context_matched": exact["scoring_context_matches"] > 0,
            "rule_action": rule["executed"],
            "exact_action": exact["executed"],
            "nn_action": nn["executed"],
            "rule_greedy": rule["greedy"],
            "exact_greedy": exact["greedy"],
            "nn_greedy": nn["greedy"],
            "rule_shortlist": "|".join(map(str, rule["shortlist"])),
            "exact_shortlist": "|".join(map(str, exact["shortlist"])),
            "nn_shortlist": "|".join(map(str, nn["shortlist"])),
            "rule_ties": "|".join(map(str, rule["ties"])),
            "exact_ties": "|".join(map(str, exact["ties"])),
            "nn_ties": "|".join(map(str, nn["ties"])),
            "exact_shortlist_changed": exact["shortlist"] != rule["shortlist"],
            "nn_shortlist_changed": nn["shortlist"] != rule["shortlist"],
            "exact_ties_changed": exact["ties"] != rule["ties"],
            "nn_ties_changed": nn["ties"] != rule["ties"],
            "exact_greedy_changed": exact["greedy"] != rule["greedy"],
            "nn_greedy_changed": nn["greedy"] != rule["greedy"],
            "exact_action_changed": exact["executed"] != rule["executed"],
            "nn_action_changed": nn["executed"] != rule["executed"],
            "exact_score_max_shift": float(np.max(np.abs(exact["scores"] - rule["scores"]))),
            "nn_score_max_shift": float(np.max(np.abs(nn["scores"] - rule["scores"]))),
            "exact_task_score_max_shift": float(np.max(np.abs(exact["task_scores"] - rule["task_scores"]))),
            "nn_task_score_max_shift": float(np.max(np.abs(nn["task_scores"] - rule["task_scores"]))),
            "exact_safety_score_max_shift": float(np.max(np.abs(exact["safety_scores"] - rule["safety_scores"]))),
            "nn_safety_score_max_shift": float(np.max(np.abs(nn["safety_scores"] - rule["safety_scores"]))),
            "exact_consensus_score_max_shift": float(np.max(np.abs(exact["consensus_scores"] - rule["consensus_scores"]))),
            "nn_consensus_score_max_shift": float(np.max(np.abs(nn["consensus_scores"] - rule["consensus_scores"]))),
            "exact_normalized_task_max_shift": float(np.max(np.abs(exact["normalized_task_scores"] - rule["normalized_task_scores"]))),
            "nn_normalized_task_max_shift": float(np.max(np.abs(nn["normalized_task_scores"] - rule["normalized_task_scores"]))),
            "exact_normalized_safety_max_shift": float(np.max(np.abs(exact["normalized_safety_scores"] - rule["normalized_safety_scores"]))),
            "nn_normalized_safety_max_shift": float(np.max(np.abs(nn["normalized_safety_scores"] - rule["normalized_safety_scores"]))),
        })
    return case, probe_rows


def _rates(rows: list[dict], names: tuple[str, ...]) -> dict[str, float]:
    return {name + "_rate": float(np.mean([bool(row[name]) for row in rows])) for name in names}


def _max_shifts(rows: list[dict]) -> dict[str, float]:
    names = (
        "exact_task_score_max_shift",
        "exact_safety_score_max_shift",
        "exact_consensus_score_max_shift",
        "exact_normalized_task_max_shift",
        "exact_normalized_safety_max_shift",
        "exact_score_max_shift",
        "nn_score_max_shift",
    )
    return {name: max(float(row[name]) for row in rows) for name in names}


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    if args.q_train_episodes < 1 or args.max_steps < 1:
        raise ValueError("training episodes and max steps must be positive")
    if not 0 <= args.danger_probability <= 1:
        raise ValueError("danger probability must be in [0, 1]")
    seeds = tuple(int(item.strip()) for item in args.agent_seeds.split(",") if item.strip())
    splits = tuple(item.strip() for item in args.splits.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("agent seeds must be nonempty and unique")
    if not splits or not set(splits) <= {"test", "ood"}:
        raise ValueError("splits must contain test and/or ood")
    base_manifest, targets, records = _load_inputs(Path(args.targets_dir))
    manifest = LearnedAppraisalManifest.read_json(Path(args.model_dir) / "manifest.json")
    if (
        manifest.goal_names != base_manifest.goal_names
        or manifest.modulator_names != base_manifest.modulator_names
        or manifest.feature_names != base_manifest.feature_names
        or manifest.output_modulator_names != base_manifest.output_modulator_names
        or manifest.residual_bounds != base_manifest.residual_bounds
        or manifest.appraisal_profile_name != base_manifest.appraisal_profile_name
    ):
        raise ValueError("model and target coordinate contracts differ")
    model = ResidualMLP.load(manifest.artifact_path)
    selected = [row for row in targets if row["split"] in splits and row["status"] == "action_switch_feasible"]
    if args.max_cases is not None:
        selected = selected[:args.max_cases]
    if not selected:
        raise ValueError("no switch-feasible targets in requested split(s)")
    trained_agents = {
        seed: _train_frozen_q(seed, args.q_train_episodes, args.max_steps, args.danger_probability)
        for seed in seeds
    }
    cases, probes = [], []
    for row in selected:
        case, case_probes = _one_context(row, records[row["record_id"]], manifest, model, trained_agents)
        cases.append(case)
        probes.extend(case_probes)
    by_perspective = {name: sum(row["perspective"] == name for row in cases)
                      for name in ("base", "safety", "growth")}
    summary = {
        "design": "matched exact-target versus NN replay on switch-feasible stored contexts",
        "splits": splits,
        "cases": len(cases),
        "agent_seeds": seeds,
        "selector_probes": len(probes),
        "cases_by_perspective": by_perspective,
        "direct_magus": {
            "exact_target_oracle_rate": float(np.mean([row["target_magus_action"] == row["oracle_action"] for row in cases])),
            "nn_oracle_rate": float(np.mean([row["nn_magus_action"] == row["oracle_action"] for row in cases])),
            "median_target_norm": float(np.median([row["target_residual_norm"] for row in cases])),
            "median_nn_norm": float(np.median([row["nn_residual_norm"] for row in cases])),
            "median_nn_target_norm_ratio": float(np.median([row["nn_target_norm_ratio"] for row in cases])),
        },
        "selector": {
            **_rates(probes, (
                "exact_context_matched", "exact_scoring_context_matched",
                "exact_shortlist_changed", "nn_shortlist_changed",
                "exact_ties_changed", "nn_ties_changed", "exact_greedy_changed",
                "nn_greedy_changed", "exact_action_changed", "nn_action_changed",
            )),
            **_max_shifts(probes),
        },
        "selector_by_perspective": {
            name: {
                **_rates([row for row in probes if row["perspective"] == name], (
                    "exact_context_matched", "exact_scoring_context_matched",
                    "exact_ties_changed", "nn_ties_changed",
                    "exact_greedy_changed", "nn_greedy_changed",
                    "exact_action_changed", "nn_action_changed",
                )),
                **_max_shifts([row for row in probes if row["perspective"] == name]),
            }
            for name in ("base", "safety", "growth")
            if any(row["perspective"] == name for row in probes)
        },
        "limitations": [
            "cases are selected because direct MAGUS target search found a switch; rates do not describe all GridWorld states",
            "stored collector motivational states are replayed against independently trained frozen Q tables",
            "an exact correction is applied only to the recorded appraisal context",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("cases.csv", cases), ("selector_probes.csv", probes)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets_dir")
    parser.add_argument("model_dir")
    parser.add_argument("--splits", default="test")
    parser.add_argument("--agent-seeds", default="0,1,2,3,4")
    parser.add_argument("--q-train-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--danger-probability", type=float, default=0.55)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--output-dir", default="experiment/results/gridworld-effect-check")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
