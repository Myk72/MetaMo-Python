"""Train and test a bounded neural correction to MetaMo rule appraisal.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from experiment.metamo_only.environment import (
    APPRAISAL_PROFILE, DECISION_PROFILE, FEATURES, SCHEMA, Case, sample_case,
)
from experiment.network import ResidualMLP
from category.bimonad import MetaMoPseudoBimonad
from dynamics.stability import DefaultStabilityPolicy, boundary_pressure
from magus.decision import MagusDecision
from openpsi.appraisal import OpenPsiAppraisal
from openpsi.hybrid import (
    HybridAppraisalConfig, HybridAppraisalHistory, HybridAppraisalMode,
    HybridOpenPsiAppraisal, bound_appraisal_residual,
)
from openpsi.learned import LearnedAppraisalManifest, ManifestFeatureEncoder, file_sha256


BOUNDS = (0.05, 0.05, 0.05, 0.05, 0.02, 0.02)


def make_manifest() -> LearnedAppraisalManifest:
    return LearnedAppraisalManifest(
        model_id="metamo-resource-residual-mlp-v1",
        application_name="metamo_appraisal_lab",
        appraisal_profile_name=APPRAISAL_PROFILE.name,
        goal_names=tuple(SCHEMA.goal_names),
        modulator_names=tuple(SCHEMA.modulator_names),
        feature_names=FEATURES,
        output_modulator_names=tuple(SCHEMA.modulators.openpsi_modulators),
        residual_bounds=BOUNDS,
    )


def _search_vectors() -> tuple[np.ndarray, ...]:
    """A small fixed search, including zero, coordinate, and paired corrections."""
    vectors = [np.zeros(6)]
    for index in range(6):
        for magnitude in ((-1.0, -0.5, 0.5, 1.0) if index < 4 else (0.5, 1.0)):
            vector = np.zeros(6)
            vector[index] = magnitude
            vectors.append(vector)
    for approach, caution in ((-1.0, 1.0), (1.0, 0.0), (-0.5, 0.5)):
        vector = np.zeros(6)
        vector[2], vector[5] = approach, caution
        vectors.append(vector)
    return tuple(vectors)


def _score(
    case: Case,
    rule_state,
    feedback,
    prediction: np.ndarray,
    manifest: LearnedAppraisalManifest,
    decision: MagusDecision,
    stability: DefaultStabilityPolicy,
) -> tuple[int, np.ndarray, float]:
    """Replay the hybrid wrapper's projection and MAGUS action selection."""
    residual, _ = bound_appraisal_residual(
        manifest, HybridAppraisalConfig(), prediction,
        boundary_pressure(case.state),
    )
    corrected = rule_state.copy()
    for output_idx, name in enumerate(manifest.output_modulator_names):
        idx = corrected.modulator_index(name)
        corrected.M[idx] = np.clip(corrected.M[idx] + residual[output_idx], 0.0, 1.0)
    decision_state = stability.raise_boundary_caution(corrected)
    result = decision.decide(decision_state, list(case.candidates), feedback=feedback)
    scores = np.asarray([item.total_score for item in result.candidate_scores], dtype=float)
    chosen = next(i for i, action in enumerate(case.candidates) if action.id == result.action.id)
    return chosen, scores, float(np.linalg.norm(corrected.M - rule_state.M))


def make_target(
    case: Case,
    manifest: LearnedAppraisalManifest,
    encoder: ManifestFeatureEncoder,
    decision: MagusDecision,
    stability: DefaultStabilityPolicy,
) -> tuple[np.ndarray, np.ndarray, str, int]:
    """Choose the smallest searched safe correction that selects the outcome oracle.

    If none switches the action, retain only a material improvement in the
    oracle score margin. A zero target explicitly means no justified change.
    """
    base = OpenPsiAppraisal(profile=APPRAISAL_PROFILE)
    rule_state, feedback = base.appraise_with_feedback(case.state, case.stimulus)
    features = APPRAISAL_PROFILE.stimulus_features(case.state, case.stimulus)
    encoded = encoder.encode(case.state, APPRAISAL_PROFILE, features, rule_state)
    oracle = case.oracle_index
    baseline_action, baseline_scores, _ = _score(
        case, rule_state, feedback, np.zeros(6), manifest, decision, stability,
    )
    if baseline_action == oracle:
        return encoded, np.zeros(6), "baseline_sufficient", baseline_action
    baseline_margin = baseline_scores[oracle] - max(
        score for i, score in enumerate(baseline_scores) if i != oracle
    )
    switched = []
    improved = []
    for vector in _search_vectors()[1:]:
        chosen, scores, norm = _score(
            case, rule_state, feedback, vector, manifest, decision, stability,
        )
        margin = scores[oracle] - max(score for i, score in enumerate(scores) if i != oracle)
        if norm <= 1e-12:
            continue
        item = (norm, -margin, tuple(vector), vector)
        if chosen == oracle:
            switched.append(item)
        elif margin >= baseline_margin + 0.001:
            improved.append(item)
    if switched:
        chosen = min(switched, key=lambda item: item[:3])
        return encoded, chosen[3].copy(), "action_switch_feasible", baseline_action
    if improved:
        chosen = min(improved, key=lambda item: item[:3])
        return encoded, chosen[3].copy(), "margin_improvement_only", baseline_action
    return encoded, np.zeros(6), "no_safe_improvement", baseline_action


def _cases(seed: int, count: int) -> list[Case]:
    rng = np.random.default_rng(seed)
    return [sample_case(rng) for _ in range(count)]


def _challenge_cases(
    *, seed: int, requested: int, pool_limit: int,
    manifest: LearnedAppraisalManifest,
    encoder: ManifestFeatureEncoder,
    decision: MagusDecision,
    stability: DefaultStabilityPolicy,
) -> tuple[list[Case], list[np.ndarray], int]:
    """Select a labeled sensitivity stratum from an independent fixed seed.

    Selection uses target feasibility, never the trained model. This stratum
    is reported separately and must not be interpreted as population benefit.
    """
    rng = np.random.default_rng(seed)
    selected, targets = [], []
    scanned = 0
    for _ in range(pool_limit):
        case = sample_case(rng)
        scanned += 1
        _, target, status, _ = make_target(
            case, manifest, encoder, decision, stability,
        )
        if status == "action_switch_feasible":
            selected.append(case)
            targets.append(target)
            if len(selected) >= requested:
                break
    return selected, targets, scanned


def _build_dataset(cases, split, manifest, encoder, decision, stability):
    vectors, targets, rows = [], [], []
    for index, case in enumerate(cases):
        encoded, target, status, rule_index = make_target(
            case, manifest, encoder, decision, stability,
        )
        vectors.append(encoded)
        targets.append(target)
        rows.append({
            "split": split,
            "case": index,
            "status": status,
            "rule_action": case.candidates[rule_index].id,
            "oracle_action": case.candidates[case.oracle_index].id,
            "target_norm": float(np.linalg.norm(target * np.asarray(BOUNDS))),
            **{f"feature_{name}": case.stimulus.numeric(name) for name in FEATURES},
            **{f"target_{name}": float(value) for name, value in zip(manifest.output_modulator_names, target)},
        })
    return np.asarray(vectors), np.asarray(targets), rows


def evaluate(cases: list[Case], manifest: LearnedAppraisalManifest, predictor: ResidualMLP) -> tuple[dict, list[dict]]:
    """Run rule, shadow, and enabled arms through the real MetaMo transition."""
    rule = MetaMoPseudoBimonad(
        OpenPsiAppraisal(profile=APPRAISAL_PROFILE),
        MagusDecision(profile=DECISION_PROFILE),
    )
    histories = {}
    hybrids = {}
    for mode in (HybridAppraisalMode.SHADOW, HybridAppraisalMode.ENABLED):
        histories[mode] = HybridAppraisalHistory()
        hybrids[mode] = MetaMoPseudoBimonad(
            HybridOpenPsiAppraisal(
                base=OpenPsiAppraisal(profile=APPRAISAL_PROFILE),
                manifest=manifest,
                encoder=ManifestFeatureEncoder(manifest, boundary_pressure),
                predictor=predictor,
                config=HybridAppraisalConfig(mode=mode),
                history=histories[mode],
            ),
            MagusDecision(profile=DECISION_PROFILE),
        )
    rows = []
    for index, case in enumerate(cases):
        actions = {}
        next_states = {}
        main_records = {}
        for mode, engine in (("rule", rule), ("shadow", hybrids[HybridAppraisalMode.SHADOW]),
                             ("enabled", hybrids[HybridAppraisalMode.ENABLED])):
            history = histories.get(HybridAppraisalMode(mode)) if mode != "rule" else None
            prior_records = len(history) if history is not None else 0
            action, next_state, _ = engine.step_with_diagnostics(
                case.state, case.stimulus, list(case.candidates),
                record_diagnostics=False,
            )
            actions[mode] = action.id
            next_states[mode] = next_state
            if history is not None:
                main_records[mode] = history.records[prior_records]
        if actions["rule"] != actions["shadow"]:
            raise AssertionError("shadow mode changed the rule action")
        if not np.array_equal(next_states["rule"].G, next_states["shadow"].G) or not np.array_equal(
            next_states["rule"].M, next_states["shadow"].M
        ):
            raise AssertionError("shadow mode changed the MetaMo state")
        if main_records["enabled"].status == "fallback":
            raise AssertionError("learned appraisal fell back; inspect model/manifest")
        utilities = {action.id: case.utilities[i] for i, action in enumerate(case.candidates)}
        risks = {action.id: action.metadata["risk"] for action in case.candidates}
        rows.append({
            "case": index,
            **{f"feature_{name}": case.stimulus.numeric(name) for name in FEATURES},
            "oracle_action": case.candidates[case.oracle_index].id,
            "rule_action": actions["rule"],
            "shadow_action": actions["shadow"],
            "enabled_action": actions["enabled"],
            "rule_utility": utilities[actions["rule"]],
            "enabled_utility": utilities[actions["enabled"]],
            "rule_risk": risks[actions["rule"]],
            "enabled_risk": risks[actions["enabled"]],
            "applied_residual_norm": float(np.linalg.norm(
                main_records["enabled"].applied_residual
            )),
            **{
                f"predicted_{name}": float(value)
                for name, value in zip(manifest.output_modulator_names, main_records["enabled"].predicted_normalized)
            },
            **{
                f"applied_{name}": float(value)
                for name, value in zip(manifest.modulator_names, main_records["enabled"].applied_residual)
            },
            "hard_safety_rejections": case.rejected_by_hard_safety,
        })
    summary = {
        "cases": len(rows),
        "rule_mean_utility": float(np.mean([row["rule_utility"] for row in rows])),
        "shadow_mean_utility": float(np.mean([row["rule_utility"] for row in rows])),
        "enabled_mean_utility": float(np.mean([row["enabled_utility"] for row in rows])),
        "rule_oracle_match": float(np.mean([row["rule_action"] == row["oracle_action"] for row in rows])),
        "enabled_oracle_match": float(np.mean([row["enabled_action"] == row["oracle_action"] for row in rows])),
        "enabled_action_change_rate": float(np.mean([row["rule_action"] != row["enabled_action"] for row in rows])),
        "rule_mean_risk": float(np.mean([row["rule_risk"] for row in rows])),
        "enabled_mean_risk": float(np.mean([row["enabled_risk"] for row in rows])),
        "mean_applied_residual_norm": float(np.mean([row["applied_residual_norm"] for row in rows])),
        "nonzero_applied_rate": float(np.mean([row["applied_residual_norm"] > 1e-12 for row in rows])),
        "hard_safety_rejections": int(sum(row["hard_safety_rejections"] for row in rows)),
        "shadow_records": len(histories[HybridAppraisalMode.SHADOW]),
        "enabled_records": len(histories[HybridAppraisalMode.ENABLED]),
    }
    return summary, rows


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    if min(args.train_cases, args.validation_cases, args.test_cases, args.epochs) < 1:
        raise ValueError("case counts and epochs must be positive")
    if args.challenge_cases < 0 or args.challenge_pool < 0:
        raise ValueError("challenge counts must be non-negative")
    if args.challenge_cases > 0 and args.challenge_pool == 0:
        raise ValueError("a challenge pool is required when challenge cases are requested")
    manifest = make_manifest()
    encoder = ManifestFeatureEncoder(manifest, boundary_pressure)
    decision = MagusDecision(profile=DECISION_PROFILE)
    stability = DefaultStabilityPolicy()
    train_cases = _cases(args.seed + 1000, args.train_cases)
    validation_cases = _cases(args.seed + 2000, args.validation_cases)
    test_cases = _cases(args.seed + 3000, args.test_cases)
    train_x, train_y, train_rows = _build_dataset(
        train_cases, "train", manifest, encoder, decision, stability,
    )
    validation_x, validation_y, validation_rows = _build_dataset(
        validation_cases, "validation", manifest, encoder, decision, stability,
    )
    _, _, test_rows = _build_dataset(
        test_cases, "test", manifest, encoder, decision, stability,
    )
    model = ResidualMLP(manifest.input_size, manifest.output_size, seed=args.seed)
    training = model.fit(
        train_x, train_y, validation_x, validation_y,
        epochs=args.epochs,
    )
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "model.npz"
    model.save(artifact)
    final_manifest = replace(
        manifest,
        artifact_path=artifact.name,
        artifact_sha256=file_sha256(artifact),
    )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(final_manifest.to_dict(), indent=2), encoding="utf-8")
    loaded_manifest = LearnedAppraisalManifest.read_json(manifest_path)
    loaded_model = ResidualMLP.load(loaded_manifest.artifact_path)
    evaluation, case_rows = evaluate(test_cases, loaded_manifest, loaded_model)
    challenge = {"status": "not_requested"}
    challenge_rows = []
    if args.challenge_cases:
        challenge_cases, challenge_targets, scanned = _challenge_cases(
            seed=args.seed + 4000,
            requested=args.challenge_cases,
            pool_limit=args.challenge_pool,
            manifest=manifest,
            encoder=encoder,
            decision=decision,
            stability=stability,
        )
        challenge = {
            "status": "complete" if len(challenge_cases) == args.challenge_cases else "pool_exhausted",
            "requested": args.challenge_cases,
            "found": len(challenge_cases),
            "pool_scanned": scanned,
            "selection": "independent cases with exact bounded target action switch",
            "population_inference": "none; this is a conditional sensitivity stratum",
        }
        if challenge_cases:
            exact_verified = []
            for case, target in zip(challenge_cases, challenge_targets):
                rule_state, feedback = OpenPsiAppraisal(profile=APPRAISAL_PROFILE).appraise_with_feedback(
                    case.state, case.stimulus,
                )
                chosen, _, _ = _score(
                    case, rule_state, feedback, target, manifest, decision, stability,
                )
                exact_verified.append(chosen == case.oracle_index)
            challenge_evaluation, challenge_rows = evaluate(
                challenge_cases, loaded_manifest, loaded_model,
            )
            challenge["exact_target_switch_rate"] = float(np.mean(exact_verified))
            challenge["nn_evaluation"] = challenge_evaluation
    target_rows = train_rows + validation_rows + test_rows
    with (output / "targets.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(target_rows[0]))
        writer.writeheader()
        writer.writerows(target_rows)
    with (output / "test_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    if challenge_rows:
        with (output / "challenge_cases.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(challenge_rows[0]))
            writer.writeheader()
            writer.writerows(challenge_rows)
    summary = {
        "design": "independent resource-choice cases; no RL, no LLM",
        "seed": args.seed,
        "split_sizes": {"train": len(train_cases), "validation": len(validation_cases), "test": len(test_cases)},
        "target_status_counts": {
            split: {status: sum(row["status"] == status for row in rows)
                    for status in sorted({row["status"] for row in rows})}
            for split, rows in (("train", train_rows), ("validation", validation_rows), ("test", test_rows))
        },
        "training": training,
        "evaluation": evaluation,
        "challenge": challenge,
        "caveat": "one-step outcome oracle; results do not establish long-horizon benefit",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="experiment/results/metamo_only-run")
    parser.add_argument("--train-cases", type=int, default=120)
    parser.add_argument("--validation-cases", type=int, default=40)
    parser.add_argument("--test-cases", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--challenge-cases", type=int, default=0)
    parser.add_argument("--challenge-pool", type=int, default=5000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
