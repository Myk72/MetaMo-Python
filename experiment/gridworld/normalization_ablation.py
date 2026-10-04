"""Compare current and fixed-reference selector normalization offline.

The fixed-reference arm centers each corrected branch as usual, but divides
by the rule branch's scale rather than recomputing its scale after appraisal.
Its zero-residual decision is therefore identical to the application. It is
an experimental counterfactual: this file does not modify GridWorld behavior.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np

from applications.gridworld.learned_appraisal import with_learned_appraisal
from applications.gridworld.runtime import build_candidates, build_stimulus
from experiment.gridworld.check_effect import (
    ContextTargetPredictor,
    _environment_state,
    _replay_selector,
    _train_frozen_q,
)
from experiment.gridworld.selector_reachability import (
    MATERIAL_SCORE_TOLERANCE,
    _choice,
    _sample_records,
    _score,
)
from experiment.gridworld.targets import (
    _record_seed,
    _reconstruct_state,
    _search_candidates,
    select_oracle_action,
)
from experiment.gridworld.train_test import _load_inputs
from openpsi.hybrid import HybridAppraisalConfig, HybridAppraisalMode


def _reference_scale(values: np.ndarray, actions: list[int]) -> float:
    eligible = np.asarray([values[action] for action in actions], dtype=float)
    centered = eligible - float(np.mean(eligible))
    scale = float(np.max(np.abs(centered)))
    return scale if scale > 1e-12 else 1.0


def _fixed_branch(values: np.ndarray, actions: list[int], rule_scale: float) -> np.ndarray:
    result = np.zeros_like(values, dtype=float)
    eligible = np.asarray([values[action] for action in actions], dtype=float)
    centered = eligible - float(np.mean(eligible))
    for action, value in zip(actions, centered / rule_scale):
        result[action] = float(value)
    return result


def _fixed_composed(agent, scored: dict, actions: list[int], scales: tuple[float, float]) -> np.ndarray:
    task = _fixed_branch(scored["raw_task"], actions, scales[0])
    safety = _fixed_branch(scored["raw_safety"], actions, scales[1])
    return (
        agent.task_weight * task
        + agent.safety_weight * safety
        - agent.disagreement_penalty * np.abs(task - safety)
    )


def _witness(current: dict | None, normalized: np.ndarray, residual: np.ndarray,
             perspective: str, ties: tuple[int, ...]) -> dict:
    candidate = {
        "norm": float(np.linalg.norm(residual)),
        "normalized": normalized.tolist(),
        "residual": residual.tolist(),
        "perspective": perspective,
        "ties": ties,
    }
    return candidate if current is None or candidate["norm"] < current["norm"] else current


def _audit_record(record: dict, trained, manifest, random_samples: int, q_seed: int) -> dict:
    state = _environment_state(record["environment_state"])
    agent = copy.deepcopy(trained)
    agent.mot = _reconstruct_state(record["motivational_state"])
    agent.epsilon = 0.0
    if agent.selector_mode != "composed":
        raise ValueError("normalization ablation requires the composed selector")
    stimulus = build_stimulus(state, agent.mot, runtime=agent.runtime)
    candidates = build_candidates(state, agent.mot, runtime=agent.runtime)
    actions = agent._valid_actions(state, avoid_lava=agent.mask_lava_on_exploit)
    risk = np.asarray([
        float(candidate.metadata.get("hazard_risk", candidate.metadata.get("risk", 0.0)))
        for candidate in candidates
    ])
    eligible = agent._hard_safety_actions(actions, risk)
    key = agent._encode(state)
    shortlist, _ = agent._q_shortlist(
        agent._q_values(key), agent.visit_counts[key], eligible,
    )
    baseline = _score(agent, state, stimulus, candidates, agent.runtime, actions)
    scales = (
        _reference_scale(baseline["raw_task"], actions),
        _reference_scale(baseline["raw_safety"], actions),
    )
    fixed_zero = _fixed_composed(agent, baseline, actions, scales)
    if not np.allclose(fixed_zero, baseline["composed"], rtol=0.0, atol=1e-10):
        raise AssertionError("fixed-reference arm does not match zero-residual selector")
    actual = _replay_selector(agent, state)
    if actual["shortlist"] != tuple(shortlist) or not np.allclose(
        actual["scores"], baseline["composed"], rtol=0.0, atol=1e-10,
    ):
        raise AssertionError("offline baseline differs from the application selector")
    rule_greedy, rule_ties = _choice(agent, baseline["composed"], shortlist)
    if actual["ties"] != rule_ties:
        raise AssertionError("offline tie set differs from the application selector")
    paired_actions: dict[tuple[int, ...], int] = {}

    def paired_action(ties: tuple[int, ...]) -> int:
        if ties not in paired_actions:
            rng = copy.deepcopy(agent.rng)
            rng.random()  # Match the selector's epsilon draw, even when epsilon is zero.
            paired_actions[ties] = int(rng.choice(ties))
        return paired_actions[ties]

    if paired_action(rule_ties) != actual["executed"]:
        raise AssertionError("paired RNG replay differs from the application selector")
    oracle, _ = select_oracle_action(record)
    predictor = ContextTargetPredictor(
        np.zeros(manifest.input_size), np.zeros(manifest.output_size),
    )
    enabled = with_learned_appraisal(
        agent.runtime, mode=HybridAppraisalMode.ENABLED,
        predictor=predictor, manifest=manifest,
    )
    result = {
        "record_id": record["record_id"],
        "split": record["split"],
        "regime": record["regime"]["name"],
        "agent_seed": q_seed,
        "oracle_action": oracle,
        "oracle_shortlisted": oracle in shortlist,
        "rule_greedy": rule_greedy,
        "rule_ties": "|".join(map(str, rule_ties)),
        "rule_executed": actual["executed"],
        "oracle_in_rule_ties": oracle in rule_ties,
        "task_rule_scale": scales[0],
        "safety_rule_scale": scales[1],
        "evaluations": 0,
        "current_max_score_shift": 0.0,
        "fixed_max_score_shift": 0.0,
    }
    witness = {
        mode: {"tie": None, "action": None, "robust": None,
               "oracle_recruited": None, "oracle_unique": None}
        for mode in ("current", "fixed")
    }
    for context in record["contexts"]:
        if context["perspective"] == "base":
            continue  # Base appraisal is not used to choose the current action.
        predictor.input_vector = np.asarray(context["input_vector"], dtype=float)
        search = _search_candidates(
            manifest=manifest,
            config=HybridAppraisalConfig(),
            boundary_pressure=float(context["boundary_pressure"]),
            seed=_record_seed(record["record_id"], context["perspective"]),
            random_samples=random_samples,
        )
        for normalized, residual in search:
            if not np.any(residual):
                continue
            predictor.target = normalized
            before = predictor.matches
            scored = _score(agent, state, stimulus, candidates, enabled, actions)
            if predictor.matches == before:
                raise AssertionError("searched context did not reach appraisal scoring")
            result["evaluations"] += 1
            for mode, scores in (
                ("current", scored["composed"]),
                ("fixed", _fixed_composed(agent, scored, actions, scales)),
            ):
                shift = float(np.max(np.abs(scores - baseline["composed"])))
                result[mode + "_max_score_shift"] = max(
                    result[mode + "_max_score_shift"], shift,
                )
                _, ties = _choice(agent, scores, shortlist)
                if ties != rule_ties:
                    witness[mode]["tie"] = _witness(
                        witness[mode]["tie"], normalized, residual,
                        context["perspective"], ties,
                    )
                if paired_action(ties) != actual["executed"]:
                    witness[mode]["action"] = _witness(
                        witness[mode]["action"], normalized, residual,
                        context["perspective"], ties,
                    )
                if not set(ties).intersection(rule_ties):
                    witness[mode]["robust"] = _witness(
                        witness[mode]["robust"], normalized, residual,
                        context["perspective"], ties,
                    )
                if oracle in ties and oracle not in rule_ties:
                    witness[mode]["oracle_recruited"] = _witness(
                        witness[mode]["oracle_recruited"], normalized, residual,
                        context["perspective"], ties,
                    )
                if ties == (oracle,) and rule_ties != (oracle,):
                    witness[mode]["oracle_unique"] = _witness(
                        witness[mode]["oracle_unique"], normalized, residual,
                        context["perspective"], ties,
                    )
    for mode in ("current", "fixed"):
        for kind in ("tie", "action", "robust", "oracle_recruited", "oracle_unique"):
            found = witness[mode][kind]
            result[f"{mode}_{kind}_change_feasible"] = found is not None
            result[f"{mode}_{kind}_minimum_norm"] = None if found is None else found["norm"]
            result[f"{mode}_{kind}_witness_perspective"] = None if found is None else found["perspective"]
            result[f"{mode}_{kind}_witness_normalized"] = None if found is None else json.dumps(found["normalized"])
            result[f"{mode}_{kind}_witness_residual"] = None if found is None else json.dumps(found["residual"])
            result[f"{mode}_{kind}_witness_ties"] = None if found is None else "|".join(map(str, found["ties"]))
            result[f"{mode}_{kind}_witness_executed"] = (
                None if found is None else paired_action(found["ties"])
            )
    result["fixed_only_tie_change"] = (
        result["fixed_tie_change_feasible"] and not result["current_tie_change_feasible"]
    )
    result["fixed_only_robust_change"] = (
        result["fixed_robust_change_feasible"] and not result["current_robust_change_feasible"]
    )
    result["fixed_only_action_change"] = (
        result["fixed_action_change_feasible"] and not result["current_action_change_feasible"]
    )
    result["fixed_material_score_shift"] = (
        result["fixed_max_score_shift"] > MATERIAL_SCORE_TOLERANCE
    )
    action_witness = result["fixed_action_witness_executed"]
    result["fixed_action_witness_oracle_gain"] = (
        action_witness is not None
        and actual["executed"] != oracle
        and action_witness == oracle
    )
    result["fixed_action_witness_oracle_loss"] = (
        action_witness is not None
        and actual["executed"] == oracle
        and action_witness != oracle
    )
    return result


def _summary(rows: list[dict], args: argparse.Namespace) -> dict:
    return {
        "design": "paired normalization ablation; same bounded corrections, Q, safety, and ties",
        "normalizations": {
            "current": "center each branch and divide by its corrected max-absolute deviation",
            "fixed": "center each branch and divide by its rule-baseline max-absolute deviation",
        },
        "splits": args.splits,
        "agent_seeds": args.agent_seeds,
        "records": len({row["record_id"] for row in rows}),
        "probes": len(rows),
        "candidate_evaluations": sum(row["evaluations"] for row in rows),
        "oracle_shortlisted": sum(row["oracle_shortlisted"] for row in rows),
        "current_tie_change_feasible": sum(row["current_tie_change_feasible"] for row in rows),
        "fixed_tie_change_feasible": sum(row["fixed_tie_change_feasible"] for row in rows),
        "fixed_only_tie_change": sum(row["fixed_only_tie_change"] for row in rows),
        "current_action_change_feasible": sum(row["current_action_change_feasible"] for row in rows),
        "fixed_action_change_feasible": sum(row["fixed_action_change_feasible"] for row in rows),
        "fixed_action_changed_records": len({
            row["record_id"] for row in rows if row["fixed_action_change_feasible"]
        }),
        "fixed_only_action_change": sum(row["fixed_only_action_change"] for row in rows),
        "fixed_action_witness_oracle_gain": sum(row["fixed_action_witness_oracle_gain"] for row in rows),
        "fixed_action_witness_oracle_loss": sum(row["fixed_action_witness_oracle_loss"] for row in rows),
        "current_robust_change_feasible": sum(row["current_robust_change_feasible"] for row in rows),
        "fixed_robust_change_feasible": sum(row["fixed_robust_change_feasible"] for row in rows),
        "fixed_only_robust_change": sum(row["fixed_only_robust_change"] for row in rows),
        "current_oracle_unique_feasible": sum(row["current_oracle_unique_change_feasible"] for row in rows),
        "fixed_oracle_unique_feasible": sum(row["fixed_oracle_unique_change_feasible"] for row in rows),
        "fixed_oracle_recruited_feasible": sum(row["fixed_oracle_recruited_change_feasible"] for row in rows),
        "fixed_material_score_shift": sum(row["fixed_material_score_shift"] for row in rows),
        "max_current_score_shift": max(row["current_max_score_shift"] for row in rows),
        "max_fixed_score_shift": max(row["fixed_max_score_shift"] for row in rows),
        "caveats": [
            "fixed-reference scoring is an experiment-only counterfactual, not GridWorld behavior",
            "finite sampled search cannot establish global infeasibility",
            "reachability is not evidence of improved episode outcomes",
        ],
    }


def run(args: argparse.Namespace) -> dict:
    if args.max_records_per_split < 1 or args.q_train_episodes < 1 or args.max_steps < 1:
        raise ValueError("record count, Q episodes, and max steps must be positive")
    if args.random_samples < 0:
        raise ValueError("random samples must be non-negative")
    if not 0.0 <= args.danger_probability <= 1.0:
        raise ValueError("danger probability must be in [0, 1]")
    splits = tuple(item.strip() for item in args.splits.split(",") if item.strip())
    seeds = tuple(int(item.strip()) for item in args.agent_seeds.split(",") if item.strip())
    if not splits or not set(splits) <= {"test", "ood"} or len(set(splits)) != len(splits):
        raise ValueError("splits must be unique test and/or ood")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("agent seeds must be nonempty and unique")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    manifest, _, records = _load_inputs(Path(args.targets_dir))
    selected = _sample_records(records, splits, args.max_records_per_split, args.seed)
    if not selected:
        raise ValueError("no records in requested split(s)")
    trained = {
        seed: _train_frozen_q(
            seed, args.q_train_episodes, args.max_steps, args.danger_probability,
        )
        for seed in seeds
    }
    rows = []
    for index, record in enumerate(selected, start=1):
        for seed, agent in trained.items():
            rows.append(_audit_record(record, agent, manifest, args.random_samples, seed))
        if not args.quiet:
            print(f"{index}/{len(selected)} {record['record_id']}", flush=True)
    summary = _summary(rows, args)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "normalization_ablation_cases.csv").open(
        "w", newline="", encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "normalization_ablation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets_dir")
    parser.add_argument("--splits", default="test,ood")
    parser.add_argument("--agent-seeds", default="0")
    parser.add_argument("--max-records-per-split", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--q-train-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--danger-probability", type=float, default=0.55)
    parser.add_argument("--random-samples", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", default="experiment/results/gridworld-normalization-ablation")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
