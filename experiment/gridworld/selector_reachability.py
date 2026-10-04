"""Audit whether bounded appraisal residuals can affect the real GridWorld selector.

This is an offline, selector-aware search, not NN training. It reuses held-out
collected states, freezes a trained Q table, and tests exactly the same bounded
residual vectors and composed selector used by the enabled appraisal wrapper.
The safety mask, Q shortlist, and random-tie semantics remain unchanged.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np

from applications.gridworld.learned_appraisal import with_learned_appraisal
from applications.gridworld.runtime import (
    build_candidates,
    build_stimulus,
    perspective_candidate_scores,
)
from experiment.gridworld.check_effect import (
    ContextTargetPredictor,
    _environment_state,
    _replay_selector,
    _train_frozen_q,
)
from experiment.gridworld.targets import (
    _record_seed,
    _reconstruct_state,
    _search_candidates,
    select_oracle_action,
)
from experiment.gridworld.train_test import _load_inputs
from openpsi.hybrid import HybridAppraisalConfig, HybridAppraisalMode


MATERIAL_SCORE_TOLERANCE = 1e-9


def _sample_records(records: dict, splits: tuple[str, ...], limit: int, seed: int) -> list[dict]:
    """Choose a reproducible, regime-balanced subset within each split."""
    rng = np.random.default_rng(seed)
    selected = []
    for split in splits:
        by_regime: dict[str, list[dict]] = {}
        for record in records.values():
            if record["split"] == split:
                by_regime.setdefault(record["regime"]["name"], []).append(record)
        buckets = []
        for name in sorted(by_regime):
            group = sorted(by_regime[name], key=lambda row: row["record_id"])
            rng.shuffle(group)
            buckets.append(group)
        while len([row for row in selected if row["split"] == split]) < limit:
            advanced = False
            for bucket in buckets:
                if bucket:
                    selected.append(bucket.pop())
                    advanced = True
                    if len([row for row in selected if row["split"] == split]) == limit:
                        break
            if not advanced:
                break
    return selected


def _score(agent, state: dict, stimulus, candidates, runtime, exploit_actions):
    perspectives = perspective_candidate_scores(
        agent.mot, stimulus, candidates, state, runtime=runtime,
    )
    task, safety, _, composed = agent._composed_preferences(
        perspectives, exploit_actions,
    )
    return {
        "raw_task": perspectives.task,
        "raw_safety": perspectives.safety,
        "task": task,
        "safety": safety,
        "composed": composed,
    }


def _choice(agent, scores: np.ndarray, shortlist: list[int]) -> tuple[int, tuple[int, ...]]:
    greedy = agent._diagnostic_argmax(scores, shortlist)
    tied = tuple(agent._tied_best_actions(scores, shortlist))
    return int(greedy), tied


def _margin(scores: np.ndarray, oracle: int, shortlist: list[int]) -> float | None:
    if oracle not in shortlist:
        return None
    alternatives = [float(scores[action]) for action in shortlist if action != oracle]
    return float(scores[oracle] - max(alternatives)) if alternatives else None


def _prefer(current: dict | None, candidate: dict) -> dict:
    """Keep the smallest legal residual, then the larger score shift."""
    if current is None:
        return candidate
    return min(
        (current, candidate),
        key=lambda item: (item["residual_norm"], -item["composed_max_shift"]),
    )


def _audit_record(record: dict, agent, manifest, random_samples: int) -> dict:
    state = _environment_state(record["environment_state"])
    replay = copy.deepcopy(agent)
    replay.mot = _reconstruct_state(record["motivational_state"])
    replay.epsilon = 0.0
    if replay.selector_mode != "composed":
        raise ValueError("reachability audit requires the composed selector")
    stimulus = build_stimulus(state, replay.mot, runtime=replay.runtime)
    candidates = build_candidates(state, replay.mot, runtime=replay.runtime)
    exploit = replay._valid_actions(state, avoid_lava=replay.mask_lava_on_exploit)
    risk = np.asarray([
        float(candidate.metadata.get("hazard_risk", candidate.metadata.get("risk", 0.0)))
        for candidate in candidates
    ])
    eligible = replay._hard_safety_actions(exploit, risk)
    key = replay._encode(state)
    q_values = replay._q_values(key)
    shortlist, _ = replay._q_shortlist(q_values, replay.visit_counts[key], eligible)
    baseline = _score(replay, state, stimulus, candidates, replay.runtime, exploit)
    baseline_greedy, baseline_ties = _choice(replay, baseline["composed"], shortlist)
    actual_selector = _replay_selector(replay, state)
    if (
        actual_selector["shortlist"] != tuple(shortlist)
        or actual_selector["ties"] != baseline_ties
        or not np.allclose(
            actual_selector["scores"], baseline["composed"], rtol=0.0, atol=1e-10,
        )
    ):
        raise AssertionError("offline selector scoring does not match the application")
    non_tied = [action for action in shortlist if action not in baseline_ties]
    baseline_action_gap = (
        float(max(baseline["composed"][action] for action in baseline_ties)
              - max(baseline["composed"][action] for action in non_tied))
        if non_tied else None
    )
    oracle, _ = select_oracle_action(record)
    baseline_margin = _margin(baseline["composed"], oracle, shortlist)

    predictor = ContextTargetPredictor(
        np.zeros(manifest.input_size), np.zeros(manifest.output_size),
    )
    enabled_runtime = with_learned_appraisal(
        replay.runtime,
        mode=HybridAppraisalMode.ENABLED,
        predictor=predictor,
        manifest=manifest,
    )
    predictor.target = np.zeros(manifest.output_size)
    zero_scores = _score(replay, state, stimulus, candidates, enabled_runtime, exploit)
    if not np.allclose(
        zero_scores["composed"], baseline["composed"], rtol=0.0, atol=1e-10,
    ):
        raise AssertionError("enabled zero residual does not reproduce rule selector")

    result = {
        "record_id": record["record_id"],
        "split": record["split"],
        "regime": record["regime"]["name"],
        "oracle_action": oracle,
        "oracle_eligible": oracle in eligible,
        "oracle_shortlisted": oracle in shortlist,
        "baseline_greedy": baseline_greedy,
        "baseline_ties": "|".join(map(str, baseline_ties)),
        "baseline_action_gap": baseline_action_gap,
        "baseline_oracle_margin": baseline_margin,
        "search_candidates": 0,
        "contexts_scored": 0,
        "max_raw_task_shift": 0.0,
        "max_raw_safety_shift": 0.0,
        "max_normalized_task_shift": 0.0,
        "max_normalized_safety_shift": 0.0,
        "max_composed_shift": 0.0,
        "greedy_index_change_observed": False,
        "material_greedy_change_feasible": False,
        "tie_change_feasible": False,
        "robust_action_change_feasible": False,
        "oracle_tie_feasible": False,
        "oracle_unique_feasible": False,
        "best_oracle_margin_improvement": 0.0,
        "minimum_material_greedy_change_norm": None,
        "minimum_tie_change_norm": None,
        "minimum_robust_action_change_norm": None,
        "minimum_oracle_unique_norm": None,
    }
    best = {name: None for name in ("greedy_index", "material_greedy", "tie", "robust", "oracle_unique")}
    for context in record["contexts"]:
        if context["perspective"] == "base":
            continue  # Base appraisal occurs after the selector has chosen an action.
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
            before_matches = predictor.matches
            scored = _score(replay, state, stimulus, candidates, enabled_runtime, exploit)
            if predictor.matches == before_matches:
                raise AssertionError("recorded perspective was not reached during scoring")
            result["search_candidates"] += 1
            result["contexts_scored"] += 1
            raw_task_shift = float(np.max(np.abs(scored["raw_task"] - baseline["raw_task"])))
            raw_safety_shift = float(np.max(np.abs(scored["raw_safety"] - baseline["raw_safety"])))
            task_shift = float(np.max(np.abs(scored["task"] - baseline["task"])))
            safety_shift = float(np.max(np.abs(scored["safety"] - baseline["safety"])))
            composed_shift = float(np.max(np.abs(scored["composed"] - baseline["composed"])))
            result["max_raw_task_shift"] = max(result["max_raw_task_shift"], raw_task_shift)
            result["max_raw_safety_shift"] = max(result["max_raw_safety_shift"], raw_safety_shift)
            result["max_normalized_task_shift"] = max(result["max_normalized_task_shift"], task_shift)
            result["max_normalized_safety_shift"] = max(result["max_normalized_safety_shift"], safety_shift)
            result["max_composed_shift"] = max(result["max_composed_shift"], composed_shift)
            greedy, ties = _choice(replay, scored["composed"], shortlist)
            if baseline_margin is not None:
                margin = _margin(scored["composed"], oracle, shortlist)
                result["best_oracle_margin_improvement"] = max(
                    result["best_oracle_margin_improvement"], margin - baseline_margin,
                )
            candidate = {
                "perspective": context["perspective"],
                "residual_norm": float(np.linalg.norm(residual)),
                "composed_max_shift": composed_shift,
                "greedy": greedy,
                "ties": ties,
            }
            if greedy != baseline_greedy:
                best["greedy_index"] = _prefer(best["greedy_index"], candidate)
                if composed_shift > MATERIAL_SCORE_TOLERANCE:
                    best["material_greedy"] = _prefer(best["material_greedy"], candidate)
            if ties != baseline_ties:
                best["tie"] = _prefer(best["tie"], candidate)
            if not set(ties).intersection(baseline_ties):
                best["robust"] = _prefer(best["robust"], candidate)
            if oracle in ties and oracle not in baseline_ties:
                result["oracle_tie_feasible"] = True
            if ties == (oracle,) and baseline_ties != (oracle,):
                best["oracle_unique"] = _prefer(best["oracle_unique"], candidate)

    for name, field in (
        ("material_greedy", "material_greedy_change"),
        ("tie", "tie_change"),
        ("robust", "robust_action_change"),
        ("oracle_unique", "oracle_unique"),
    ):
        result[field + "_feasible"] = best[name] is not None
        if best[name] is not None:
            result["minimum_" + field + "_norm"] = best[name]["residual_norm"]
    result["greedy_index_change_observed"] = best["greedy_index"] is not None
    result["best_robust_perspective"] = (
        None if best["robust"] is None else best["robust"]["perspective"]
    )
    result["twice_max_shift_over_action_gap"] = (
        None if baseline_action_gap is None or baseline_action_gap <= 0
        else 2.0 * result["max_composed_shift"] / baseline_action_gap
    )
    return result


def _summarize(rows: list[dict], *, splits, seeds, random_samples) -> dict:
    return {
        "design": "sampled bounded-residual reachability at the final composed selector",
        "splits": splits,
        "agent_seeds": seeds,
        "random_samples_per_context": random_samples,
        "records": len({row["record_id"] for row in rows}),
        "probes": len(rows),
        "candidate_context_evaluations": sum(row["search_candidates"] for row in rows),
        "oracle_shortlisted": sum(row["oracle_shortlisted"] for row in rows),
        "greedy_index_change_observed": sum(row["greedy_index_change_observed"] for row in rows),
        "material_greedy_change_feasible": sum(row["material_greedy_change_feasible"] for row in rows),
        "tie_change_feasible": sum(row["tie_change_feasible"] for row in rows),
        "robust_action_change_feasible": sum(row["robust_action_change_feasible"] for row in rows),
        "oracle_unique_feasible": sum(row["oracle_unique_feasible"] for row in rows),
        "max_raw_task_shift": max(row["max_raw_task_shift"] for row in rows),
        "max_raw_safety_shift": max(row["max_raw_safety_shift"] for row in rows),
        "max_normalized_task_shift": max(row["max_normalized_task_shift"] for row in rows),
        "max_normalized_safety_shift": max(row["max_normalized_safety_shift"] for row in rows),
        "max_composed_shift": max(row["max_composed_shift"] for row in rows),
        "median_action_gap": float(np.median([
            row["baseline_action_gap"] for row in rows
            if row["baseline_action_gap"] is not None
        ])) if any(row["baseline_action_gap"] is not None for row in rows) else None,
        "max_twice_shift_over_action_gap": max(
            (row["twice_max_shift_over_action_gap"] for row in rows
             if row["twice_max_shift_over_action_gap"] is not None),
            default=None,
        ),
        "caveats": [
            "sampled residual search cannot prove global infeasibility",
            "the one-step safety-first oracle is diagnostic, not the GridWorld reward objective",
            "frozen Q tables and stored collector states are matched replay, not episode outcomes",
        ],
    }


def run(args: argparse.Namespace) -> dict:
    if args.max_records_per_split < 1 or args.q_train_episodes < 1 or args.max_steps < 1:
        raise ValueError("record count, Q training episodes, and max steps must be positive")
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
        raise ValueError("no records for requested split(s)")
    trained_agents = {
        seed: _train_frozen_q(
            seed, args.q_train_episodes, args.max_steps, args.danger_probability,
        )
        for seed in seeds
    }
    rows = []
    for index, record in enumerate(selected, start=1):
        for seed, agent in trained_agents.items():
            row = _audit_record(record, agent, manifest, args.random_samples)
            row["agent_seed"] = seed
            rows.append(row)
        if not args.quiet:
            print(f"{index}/{len(selected)} {record['record_id']}", flush=True)
    summary = _summarize(
        rows, splits=splits, seeds=seeds, random_samples=args.random_samples,
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "selector_reachability_cases.csv").open(
        "w", newline="", encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "selector_reachability_summary.json").write_text(
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
    parser.add_argument("--max-records-per-split", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--q-train-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--danger-probability", type=float, default=0.55)
    parser.add_argument("--random-samples", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", default="experiment/results/gridworld-selector-reachability")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
