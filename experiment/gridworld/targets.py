"""Generate Phase-3 minimum-intervention appraisal residual targets.

Targets are produced by a deterministic bounded search.  Every candidate
residual passes through the exact runtime guard and the unchanged MAGUS scorer.
The strict target remains zero unless it gives the safety-first oracle action
the requested margin.  A separate shadow-training target may use the smallest
bounded residual that either selects the oracle action or materially improves
its margin.  No learned correction is enabled by this module.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from applications.gridworld.learned_appraisal import (
    gridworld_learned_appraisal_manifest,
)
from applications.gridworld.profile import GRIDWORLD_DECISION_PROFILE, GRIDWORLD_SCHEMA
from core.features import GoalChangeFeedback
from core.state import Action, MotivationalState
from dynamics.stability import DefaultStabilityPolicy
from experiment.gridworld.baseline import (
    PROJECT_ROOT,
    _jsonable,
    _source_fingerprint,
    _write_json,
)
from magus.decision import MagusDecision
from openpsi.hybrid import (
    HybridAppraisalConfig,
    appraisal_residual_constraint_activity,
    bound_appraisal_residual,
)
from openpsi.learned import LearnedAppraisalManifest, file_sha256


TARGET_VERSION = 2
SCORE_TOLERANCE = 1e-10
DEFAULT_IMPROVEMENT_THRESHOLD = 0.001
AUDIT_MARGIN_THRESHOLDS = (0.0, 0.005, 0.01, 0.02)


def safety_first_outcome_key(outcome: dict[str, Any]) -> tuple:
    """Lexicographic oracle: immediate safety dominates task reward."""

    return (
        int(not outcome["dead"]),
        int(not outcome["in_lava"]),
        int(not outcome["unsafe_zone"]),
        int(outcome["lava_distance_after"]),
        float(outcome["energy_after"]),
        int(outcome["mineral_collected"]),
        float(outcome["progress"]),
        float(outcome["reward"]),
        int(outcome["movement"]),
    )


def select_oracle_action(record: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Select one oracle action, using consensus score only to break true ties."""

    outcomes = record["counterfactuals"]
    keys = [safety_first_outcome_key(outcome) for outcome in outcomes]
    best_key = max(keys)
    tied = [idx for idx, key in enumerate(keys) if key == best_key]
    consensus_scores = record["consensus_rule_scores"]
    selected = max(tied, key=lambda idx: (float(consensus_scores[idx]), -idx))
    return selected, {
        "ordering": [
            "alive",
            "not_in_lava",
            "outside_unsafe_zone",
            "lava_distance_after",
            "energy_after",
            "mineral_collected",
            "progress",
            "reward",
            "movement",
        ],
        "winning_key": list(best_key),
        "tied_action_indices": tied,
        "tie_breaker": "highest rule consensus score, then lowest action index",
    }


def _reconstruct_state(values: dict[str, Any]) -> MotivationalState:
    return MotivationalState(
        G=np.asarray(values["G"], dtype=float),
        M=np.asarray(values["M"], dtype=float),
        schema=GRIDWORLD_SCHEMA,
    )


def _reconstruct_candidates(values: list[dict[str, Any]]) -> list[Action]:
    return [
        Action(
            id=item["id"],
            goal_correlations=np.asarray(item["goal_correlations"], dtype=float),
            delta_g=np.asarray(item["delta_g"], dtype=float),
            schema=GRIDWORLD_SCHEMA,
            metadata=dict(item["metadata"]),
        )
        for item in values
    ]


def _scores_for_residual(
    *,
    rule_state: MotivationalState,
    residual: np.ndarray,
    output_modulator_names: tuple[str, ...],
    candidates: list[Action],
    feedback: GoalChangeFeedback,
    decision: MagusDecision,
    stability: DefaultStabilityPolicy,
) -> np.ndarray:
    corrected = rule_state.copy()
    for idx, name in enumerate(output_modulator_names):
        modulator_idx = corrected.modulator_index(name)
        corrected.M[modulator_idx] += residual[idx]
    corrected.M = np.clip(corrected.M, 0.0, 1.0)
    decision_state = stability.raise_boundary_caution(corrected)
    return np.asarray(
        [
            decision.evaluate_candidate(decision_state, candidate, feedback)[1].total_score
            for candidate in candidates
        ],
        dtype=float,
    )


def _oracle_margin(scores: np.ndarray, oracle_index: int) -> float:
    alternatives = np.delete(scores, oracle_index)
    return float(scores[oracle_index] - np.max(alternatives))


def _normalized_search_vectors(
    output_size: int,
    rng: np.random.Generator,
    random_samples: int,
) -> Iterable[np.ndarray]:
    yield np.zeros(output_size, dtype=float)

    axis_levels = (-1.0, -0.5, 0.5, 1.0)
    for idx in range(output_size):
        for level in axis_levels:
            vector = np.zeros(output_size, dtype=float)
            vector[idx] = level
            yield vector

    for left, right in itertools.combinations(range(output_size), 2):
        for left_level, right_level in itertools.product((-1.0, 1.0), repeat=2):
            vector = np.zeros(output_size, dtype=float)
            vector[left] = left_level
            vector[right] = right_level
            yield vector

    for corner in itertools.product((-1.0, 1.0), repeat=output_size):
        yield np.asarray(corner, dtype=float)

    for _ in range(random_samples):
        yield rng.uniform(-1.0, 1.0, size=output_size)


def _search_candidates(
    *,
    manifest: LearnedAppraisalManifest,
    config: HybridAppraisalConfig,
    boundary_pressure: float,
    seed: int,
    random_samples: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    unique: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] = {}
    for normalized in _normalized_search_vectors(
        manifest.output_size,
        rng,
        random_samples,
    ):
        residual, _ = bound_appraisal_residual(
            manifest,
            config,
            normalized,
            boundary_pressure,
        )
        key = tuple(float(value) for value in np.round(residual, decimals=12))
        existing = unique.get(key)
        if existing is None or tuple(normalized) < tuple(existing[0]):
            unique[key] = (normalized.copy(), residual.copy())
    return sorted(
        unique.values(),
        key=lambda item: (
            float(np.linalg.norm(item[1])),
            tuple(float(value) for value in item[1]),
            tuple(float(value) for value in item[0]),
        ),
    )


def _record_seed(record_id: str, perspective: str) -> int:
    digest = sha256(f"{record_id}:{perspective}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _minimum_norm_evaluation(
    items: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not items:
        return None
    return min(
        items,
        key=lambda item: (
            item["norm"],
            -item["margin"],
            tuple(float(value) for value in item["residual"]),
        ),
    )


def select_target_evaluations(
    evaluations: list[dict[str, Any]],
    *,
    oracle_index: int,
    required_margin: float,
    improvement_threshold: float,
) -> dict[str, Any]:
    """Select strict and shadow targets from deterministic search results."""

    if not evaluations:
        raise ValueError("evaluations must contain the zero-residual baseline")
    baseline = evaluations[0]
    nonzero = [item for item in evaluations if item["norm"] > 0.0]
    strict_choice = _minimum_norm_evaluation(
        [
            item
            for item in nonzero
            if item["action_index"] == oracle_index
            and item["margin"] >= required_margin
        ]
    )
    action_switch_choice = _minimum_norm_evaluation(
        [item for item in nonzero if item["action_index"] == oracle_index]
    )
    improvement_choice = _minimum_norm_evaluation(
        [
            item
            for item in nonzero
            if item["margin_improvement"] >= improvement_threshold
        ]
    )

    strict_baseline_sufficient = (
        baseline["action_index"] == oracle_index
        and baseline["margin"] >= required_margin
    )
    strict_target = baseline
    if strict_baseline_sufficient:
        strict_status = "baseline_sufficient"
    elif strict_choice is not None:
        strict_target = strict_choice
        strict_status = "feasible_residual"
    else:
        strict_status = "no_feasible_residual"

    target = baseline
    if baseline["action_index"] == oracle_index:
        status = "baseline_sufficient"
    elif action_switch_choice is not None:
        target = action_switch_choice
        status = "action_switch_feasible"
    elif improvement_choice is not None:
        target = improvement_choice
        status = "margin_improvement_only"
    else:
        status = "no_safe_improvement"

    return {
        "status": status,
        "target": target,
        "strict_status": strict_status,
        "strict_target": strict_target,
        "strict_baseline_sufficient": strict_baseline_sufficient,
        "strict_choice": strict_choice,
        "action_switch_choice": action_switch_choice,
    }


def generate_context_target(
    *,
    record: dict[str, Any],
    context: dict[str, Any],
    oracle_index: int,
    oracle_evidence: dict[str, Any],
    manifest: LearnedAppraisalManifest,
    config: HybridAppraisalConfig,
    decision: MagusDecision,
    stability: DefaultStabilityPolicy,
    required_margin: float,
    random_samples: int,
    improvement_threshold: float = DEFAULT_IMPROVEMENT_THRESHOLD,
) -> dict[str, Any]:
    candidates = _reconstruct_candidates(record["candidates"])
    rule_state = _reconstruct_state(context["rule_state"])
    feedback = GoalChangeFeedback(context["goal_change_feedback"])
    zero = np.zeros(manifest.output_size, dtype=float)
    baseline_scores = _scores_for_residual(
        rule_state=rule_state,
        residual=zero,
        output_modulator_names=manifest.output_modulator_names,
        candidates=candidates,
        feedback=feedback,
        decision=decision,
        stability=stability,
    )
    stored_scores = np.asarray(
        [item["total_score"] for item in context["rule_scores"]],
        dtype=float,
    )
    if not np.allclose(baseline_scores, stored_scores, rtol=0.0, atol=SCORE_TOLERANCE):
        raise ValueError(
            f"stored rule scores do not replay for {record['record_id']}:{context['perspective']}"
        )

    baseline_margin = _oracle_margin(baseline_scores, oracle_index)
    baseline_action_index = int(np.argmax(baseline_scores))
    if improvement_threshold <= 0.0 or not np.isfinite(improvement_threshold):
        raise ValueError("improvement_threshold must be finite and positive")

    def evaluation(
        normalized: np.ndarray,
        residual: np.ndarray,
        scores: np.ndarray,
    ) -> dict[str, Any]:
        margin = _oracle_margin(scores, oracle_index)
        return {
            "normalized": normalized.copy(),
            "residual": residual.copy(),
            "scores": scores.copy(),
            "norm": float(np.linalg.norm(residual)),
            "margin": margin,
            "margin_improvement": margin - baseline_margin,
            "action_index": int(np.argmax(scores)),
        }

    baseline = evaluation(zero, zero, baseline_scores)
    evaluations = [baseline]
    strict_baseline_sufficient = (
        baseline_action_index == oracle_index
        and baseline_margin >= required_margin
    )

    if not strict_baseline_sufficient:
        candidates_to_search = _search_candidates(
            manifest=manifest,
            config=config,
            boundary_pressure=float(context["boundary_pressure"]),
            seed=_record_seed(record["record_id"], context["perspective"]),
            random_samples=random_samples,
        )
        for normalized, residual in candidates_to_search:
            if not np.any(residual):
                continue
            scores = _scores_for_residual(
                rule_state=rule_state,
                residual=residual,
                output_modulator_names=manifest.output_modulator_names,
                candidates=candidates,
                feedback=feedback,
                decision=decision,
                stability=stability,
            )
            evaluations.append(evaluation(normalized, residual, scores))

    selection = select_target_evaluations(
        evaluations,
        oracle_index=oracle_index,
        required_margin=required_margin,
        improvement_threshold=improvement_threshold,
    )
    strict_choice = selection["strict_choice"]
    action_switch_choice = selection["action_switch_choice"]
    best = min(
        evaluations,
        key=lambda item: (
            -item["margin"],
            item["norm"],
            tuple(float(value) for value in item["residual"]),
        ),
    )

    strict_target = selection["strict_target"]
    strict_status = selection["strict_status"]
    target = selection["target"]
    status = selection["status"]

    target_normalized = target["normalized"]
    target_residual = target["residual"]
    target_scores = target["scores"]

    replay_residual, _ = bound_appraisal_residual(
        manifest,
        config,
        target_normalized,
        float(context["boundary_pressure"]),
    )
    replay_scores = _scores_for_residual(
        rule_state=rule_state,
        residual=replay_residual,
        output_modulator_names=manifest.output_modulator_names,
        candidates=candidates,
        feedback=feedback,
        decision=decision,
        stability=stability,
    )
    replay_verified = bool(
        np.allclose(replay_residual, target_residual, rtol=0.0, atol=1e-12)
        and np.allclose(replay_scores, target_scores, rtol=0.0, atol=1e-12)
    )
    if not replay_verified:
        raise AssertionError("stored appraisal target failed replay verification")

    target_margin = _oracle_margin(target_scores, oracle_index)
    target_action_index = int(np.argmax(target_scores))
    oracle_selected = target_action_index == oracle_index
    margin_satisfied = oracle_selected and target_margin >= required_margin
    margin_improvement = target_margin - baseline_margin
    if status == "baseline_sufficient" and (
        not oracle_selected or np.any(target_residual)
    ):
        raise AssertionError("baseline-sufficient target must be zero and select the oracle")
    if status == "action_switch_feasible" and not oracle_selected:
        raise AssertionError("action-switch target does not select the oracle")
    if status == "margin_improvement_only" and (
        oracle_selected
        or margin_improvement < improvement_threshold
        or not np.any(target_residual)
    ):
        raise AssertionError("margin-improvement target does not meet its contract")
    if status == "no_safe_improvement" and np.any(target_residual):
        raise AssertionError("no-improvement targets must fall back to zero residual")

    strict_replay_residual, _ = bound_appraisal_residual(
        manifest,
        config,
        strict_target["normalized"],
        float(context["boundary_pressure"]),
    )
    strict_replay_scores = _scores_for_residual(
        rule_state=rule_state,
        residual=strict_replay_residual,
        output_modulator_names=manifest.output_modulator_names,
        candidates=candidates,
        feedback=feedback,
        decision=decision,
        stability=stability,
    )
    strict_replay_verified = bool(
        np.allclose(
            strict_replay_residual,
            strict_target["residual"],
            rtol=0.0,
            atol=1e-12,
        )
        and np.allclose(
            strict_replay_scores,
            strict_target["scores"],
            rtol=0.0,
            atol=1e-12,
        )
    )
    if not strict_replay_verified:
        raise AssertionError("stored strict target failed replay verification")
    if strict_status in {"baseline_sufficient", "feasible_residual"}:
        strict_action = int(np.argmax(strict_target["scores"]))
        if (
            strict_action != oracle_index
            or strict_target["margin"] < required_margin
        ):
            raise AssertionError("strict target does not satisfy its margin")
    elif np.any(strict_target["residual"]):
        raise AssertionError("infeasible strict target must be zero")

    thresholds = sorted(set((*AUDIT_MARGIN_THRESHOLDS, required_margin)))
    threshold_feasibility: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        eligible = [
            item
            for item in evaluations
            if item["action_index"] == oracle_index
            and item["margin"] >= threshold
        ]
        choice = _minimum_norm_evaluation(eligible)
        threshold_feasibility[f"{threshold:.6g}"] = {
            "feasible": choice is not None,
            "minimum_residual_norm": None if choice is None else choice["norm"],
        }

    pressure = float(context["boundary_pressure"])
    best_constraints = appraisal_residual_constraint_activity(
        manifest,
        config,
        best["normalized"],
        pressure,
    )
    target_constraints = appraisal_residual_constraint_activity(
        manifest,
        config,
        target_normalized,
        pressure,
    )
    strict_constraints = appraisal_residual_constraint_activity(
        manifest,
        config,
        strict_target["normalized"],
        pressure,
    )

    return {
        "target_version": TARGET_VERSION,
        "target_id": f"{record['record_id']}:{context['perspective']}",
        "record_id": record["record_id"],
        "split": record["split"],
        "group_id": record["group_id"],
        "regime": record["regime"]["name"],
        "perspective": context["perspective"],
        "input_vector": context["input_vector"],
        "features": context["features"],
        "boundary_pressure": context["boundary_pressure"],
        "oracle_action_index": oracle_index,
        "oracle_action_id": candidates[oracle_index].id,
        "oracle_evidence": oracle_evidence,
        "baseline_action_index": baseline_action_index,
        "baseline_action_id": candidates[baseline_action_index].id,
        "baseline_scores": baseline_scores.tolist(),
        "baseline_oracle_margin": baseline_margin,
        "status": status,
        "target_normalized": target_normalized.tolist(),
        "target_residual": target_residual.tolist(),
        "target_residual_norm": float(np.linalg.norm(target_residual)),
        "target_scores": target_scores.tolist(),
        "target_action_index": target_action_index,
        "target_action_id": candidates[target_action_index].id,
        "target_oracle_margin": target_margin,
        "target_margin_improvement": margin_improvement,
        "margin_required": required_margin,
        "improvement_threshold": improvement_threshold,
        "margin_satisfied": margin_satisfied,
        "oracle_selected": oracle_selected,
        "replay_verified": replay_verified,
        "target_constraint_activity": target_constraints,
        "strict_status": strict_status,
        "strict_target_normalized": strict_target["normalized"].tolist(),
        "strict_target_residual": strict_target["residual"].tolist(),
        "strict_target_residual_norm": strict_target["norm"],
        "strict_target_scores": strict_target["scores"].tolist(),
        "strict_target_oracle_margin": strict_target["margin"],
        "strict_margin_satisfied": (
            strict_target["action_index"] == oracle_index
            and strict_target["margin"] >= required_margin
        ),
        "strict_replay_verified": strict_replay_verified,
        "strict_constraint_activity": strict_constraints,
        "search_candidates_evaluated": len(evaluations),
        "best_searched_margin": best["margin"],
        "best_margin_improvement": best["margin_improvement"],
        "best_searched_normalized": best["normalized"].tolist(),
        "best_searched_residual": best["residual"].tolist(),
        "best_searched_residual_norm": best["norm"],
        "best_searched_scores": best["scores"].tolist(),
        "best_constraint_activity": best_constraints,
        "baseline_oracle_selected": baseline_action_index == oracle_index,
        "oracle_selection_feasible": (
            baseline_action_index == oracle_index
            or action_switch_choice is not None
        ),
        "action_switch_feasible": (
            baseline_action_index != oracle_index
            and action_switch_choice is not None
        ),
        "minimum_action_switch_residual_norm": (
            action_switch_choice["norm"]
            if baseline_action_index != oracle_index
            and action_switch_choice is not None
            else None
        ),
        "minimum_oracle_reinforcement_residual_norm": (
            action_switch_choice["norm"]
            if baseline_action_index == oracle_index
            and action_switch_choice is not None
            else None
        ),
        "required_margin_feasible": (
            strict_baseline_sufficient or strict_choice is not None
        ),
        "required_margin_correction_feasible": (
            not strict_baseline_sufficient and strict_choice is not None
        ),
        "minimum_required_margin_residual_norm": (
            None if strict_choice is None else strict_choice["norm"]
        ),
        "margin_threshold_feasibility": threshold_feasibility,
    }


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is invalid JSON") from error


def _numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "p95": None}
    return {
        "count": int(data.size),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p95": float(np.quantile(data, 0.95)),
    }


def _count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _constraint_counts(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    flag_names = ("prediction_clipped", "norm_capped")
    list_names = (
        "coordinate_saturated",
        "direction_clipped",
        "boundary_attenuated",
    )
    result: dict[str, Any] = {
        name: sum(bool(row[field][name]) for row in rows)
        for name in flag_names
    }
    for name in list_names:
        item_counts: dict[str, int] = {}
        for row in rows:
            for item in row[field][name]:
                item_counts[item] = item_counts.get(item, 0) + 1
        result[name] = dict(sorted(item_counts.items()))
    return result


def build_feasibility_audit(
    rows: list[dict[str, Any]],
    *,
    required_margin: float,
    improvement_threshold: float,
) -> dict[str, Any]:
    """Aggregate sampled control authority without changing target selection."""

    threshold_keys = sorted(
        {key for row in rows for key in row["margin_threshold_feasibility"]},
        key=float,
    )
    threshold_summary: dict[str, Any] = {}
    for key in threshold_keys:
        feasible = [
            row["margin_threshold_feasibility"][key]
            for row in rows
            if key in row["margin_threshold_feasibility"]
        ]
        norms = [
            item["minimum_residual_norm"]
            for item in feasible
            if item["minimum_residual_norm"] is not None
        ]
        threshold_summary[key] = {
            "feasible_count": sum(bool(item["feasible"]) for item in feasible),
            "infeasible_count": sum(not bool(item["feasible"]) for item in feasible),
            "minimum_norm_summary": _numeric_summary(norms),
        }

    grouped_counts: dict[str, Any] = {}
    for dimension in ("split", "regime", "perspective"):
        groups = sorted({str(row[dimension]) for row in rows})
        grouped_counts[dimension] = {
            group: _count_values(
                [row for row in rows if str(row[dimension]) == group],
                "status",
            )
            for group in groups
        }

    return {
        "audit_version": 1,
        "method": "deterministic sampled control-authority audit",
        "scope": {
            "target_count": len(rows),
            "required_margin": required_margin,
            "improvement_threshold": improvement_threshold,
        },
        "status_counts": _count_values(rows, "status"),
        "strict_status_counts": _count_values(rows, "strict_status"),
        "grouped_status_counts": grouped_counts,
        "control_authority": {
            "baseline_oracle_selected_count": sum(
                bool(row["baseline_oracle_selected"]) for row in rows
            ),
            "oracle_selection_feasible_count": sum(
                bool(row["oracle_selection_feasible"]) for row in rows
            ),
            "action_switch_feasible_count": sum(
                bool(row["action_switch_feasible"]) for row in rows
            ),
            "required_margin_feasible_count": sum(
                bool(row["required_margin_feasible"]) for row in rows
            ),
            "required_margin_correction_feasible_count": sum(
                bool(row["required_margin_correction_feasible"]) for row in rows
            ),
            "baseline_margin": _numeric_summary(
                row["baseline_oracle_margin"] for row in rows
            ),
            "best_searched_margin": _numeric_summary(
                row["best_searched_margin"] for row in rows
            ),
            "best_margin_improvement": _numeric_summary(
                row["best_margin_improvement"] for row in rows
            ),
            "minimum_action_switch_norm": _numeric_summary(
                row["minimum_action_switch_residual_norm"]
                for row in rows
                if row["minimum_action_switch_residual_norm"] is not None
            ),
            "minimum_oracle_reinforcement_norm": _numeric_summary(
                row["minimum_oracle_reinforcement_residual_norm"]
                for row in rows
                if row["minimum_oracle_reinforcement_residual_norm"] is not None
            ),
            "minimum_required_margin_norm": _numeric_summary(
                row["minimum_required_margin_residual_norm"]
                for row in rows
                if row["minimum_required_margin_residual_norm"] is not None
            ),
        },
        "margin_thresholds": threshold_summary,
        "constraint_activity": {
            "best_margin_candidate": _constraint_counts(rows, "best_constraint_activity"),
            "selected_shadow_target": _constraint_counts(rows, "target_constraint_activity"),
            "selected_strict_target": _constraint_counts(rows, "strict_constraint_activity"),
        },
        "constraint_definitions": {
            "coordinate_saturated": "normalized coordinate reached its manifest bound",
            "direction_clipped": "negative correction was removed for a nonnegative safety modulator",
            "boundary_attenuated": "non-safety correction was reduced by boundary pressure",
            "norm_capped": "projected residual exceeded the runtime L2 norm cap",
        },
    }


def generate_targets(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    required_margin: float = 0.02,
    random_samples: int = 256,
    improvement_threshold: float = DEFAULT_IMPROVEMENT_THRESHOLD,
) -> Path:
    if required_margin < 0.0 or not np.isfinite(required_margin):
        raise ValueError("required_margin must be finite and non-negative")
    if random_samples < 0:
        raise ValueError("random_samples must be non-negative")
    if improvement_threshold <= 0.0 or not np.isfinite(improvement_threshold):
        raise ValueError("improvement_threshold must be finite and positive")

    dataset = Path(dataset_dir)
    if not dataset.is_absolute():
        dataset = PROJECT_ROOT / dataset
    output = Path(output_dir)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if output.exists():
        raise FileExistsError(f"target output already exists: {output}")

    dataset_manifest_path = dataset / "manifest.json"
    transition_path = dataset / "transitions.jsonl"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("status") != "complete":
        raise ValueError("source dataset is not complete")
    if file_sha256(transition_path) != dataset_manifest.get("records_sha256"):
        raise ValueError("source transition checksum does not match its manifest")

    manifest = LearnedAppraisalManifest.from_dict(
        dataset_manifest["learned_appraisal_manifest"]
    )
    expected_manifest = gridworld_learned_appraisal_manifest()
    if manifest != expected_manifest:
        raise ValueError("dataset learned-appraisal manifest differs from runtime")

    output.mkdir(parents=True)
    source_digest, _ = _source_fingerprint()
    target_manifest: dict[str, Any] = {
        "target_version": TARGET_VERSION,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_at_utc": None,
        "source_sha256": source_digest,
        "dataset_directory": str(dataset),
        "dataset_manifest_sha256": file_sha256(dataset_manifest_path),
        "dataset_records_sha256": file_sha256(transition_path),
        "learned_appraisal_manifest": manifest.to_dict(),
        "search": {
            "method": "deterministic sampled minimum-norm constrained search",
            "required_margin": required_margin,
            "improvement_threshold": improvement_threshold,
            "random_samples": random_samples,
            "runtime_guard": asdict(HybridAppraisalConfig()),
        },
        "target_policy": {
            "strict": "minimum norm oracle selection at required margin, otherwise zero",
            "shadow_training": (
                "zero when the oracle is already selected; otherwise minimum norm "
                "oracle-selection candidate, then minimum norm material margin "
                "improvement, otherwise zero"
            ),
            "runtime_effect": "none; generated targets are offline artifacts",
        },
        "oracle": {
            "type": "lexicographic safety-first one-step outcome oracle",
            "ordering": [
                "alive",
                "not_in_lava",
                "outside_unsafe_zone",
                "lava_distance_after",
                "energy_after",
                "mineral_collected",
                "progress",
                "reward",
                "movement",
            ],
        },
    }
    _write_json(output / "manifest.json", target_manifest)

    decision = MagusDecision(profile=GRIDWORLD_DECISION_PROFILE)
    stability = DefaultStabilityPolicy()
    guard_config = HybridAppraisalConfig()
    target_path = output / "targets.jsonl"
    status_counts: dict[str, int] = {}
    strict_status_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    nonzero_counts: dict[str, int] = {}
    strict_nonzero_counts: dict[str, int] = {}
    verified_count = 0
    strict_verified_count = 0
    total = 0
    groups_by_split: dict[str, set[str]] = {}
    audit_rows: list[dict[str, Any]] = []

    try:
        with target_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in _iter_jsonl(transition_path):
                oracle_index, oracle_evidence = select_oracle_action(record)
                for context in record["contexts"]:
                    target = generate_context_target(
                        record=record,
                        context=context,
                        oracle_index=oracle_index,
                        oracle_evidence=oracle_evidence,
                        manifest=manifest,
                        config=guard_config,
                        decision=decision,
                        stability=stability,
                        required_margin=required_margin,
                        random_samples=random_samples,
                        improvement_threshold=improvement_threshold,
                    )
                    handle.write(
                        json.dumps(
                            _jsonable(target),
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                    handle.write("\n")
                    total += 1
                    status_counts[target["status"]] = status_counts.get(target["status"], 0) + 1
                    strict_status_counts[target["strict_status"]] = (
                        strict_status_counts.get(target["strict_status"], 0) + 1
                    )
                    split_counts[target["split"]] = split_counts.get(target["split"], 0) + 1
                    if target["target_residual_norm"] > 0.0:
                        nonzero_counts[target["split"]] = nonzero_counts.get(target["split"], 0) + 1
                    if target["strict_target_residual_norm"] > 0.0:
                        strict_nonzero_counts[target["split"]] = (
                            strict_nonzero_counts.get(target["split"], 0) + 1
                        )
                    verified_count += int(target["replay_verified"])
                    strict_verified_count += int(target["strict_replay_verified"])
                    groups_by_split.setdefault(target["split"], set()).add(target["group_id"])
                    audit_rows.append(target)

        split_items = list(groups_by_split.items())
        for idx, (left_name, left) in enumerate(split_items):
            for right_name, right in split_items[idx + 1 :]:
                overlap = left.intersection(right)
                if overlap:
                    raise AssertionError(
                        f"target group leakage between {left_name} and {right_name}: {sorted(overlap)}"
                    )

        audit = build_feasibility_audit(
            audit_rows,
            required_margin=required_margin,
            improvement_threshold=improvement_threshold,
        )
        audit_path = output / "feasibility_audit.json"
        _write_json(audit_path, audit)

        target_manifest.update(
            {
                "status": "complete",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "target_count": total,
                "status_counts": status_counts,
                "strict_status_counts": strict_status_counts,
                "split_counts": split_counts,
                "nonzero_target_counts": nonzero_counts,
                "strict_nonzero_target_counts": strict_nonzero_counts,
                "replay_verified_count": verified_count,
                "strict_replay_verified_count": strict_verified_count,
                "group_overlap_count": 0,
                "targets_sha256": file_sha256(target_path),
                "feasibility_audit_sha256": file_sha256(audit_path),
            }
        )
        _write_json(output / "manifest.json", target_manifest)
    except Exception as error:
        target_manifest.update(
            {
                "status": "failed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "target_count": total,
            }
        )
        _write_json(output / "manifest.json", target_manifest)
        raise
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--required-margin", type=float, default=0.02)
    parser.add_argument("--random-samples", type=int, default=256)
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=DEFAULT_IMPROVEMENT_THRESHOLD,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = Path(args.dataset_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or (
        PROJECT_ROOT / "experiment" / "results" / f"gridworld-targets-{timestamp}"
    )
    result = generate_targets(
        dataset,
        output,
        required_margin=args.required_margin,
        random_samples=args.random_samples,
        improvement_threshold=args.improvement_threshold,
    )
    print(f"Learned-appraisal targets complete: {result}")


if __name__ == "__main__":
    main()
