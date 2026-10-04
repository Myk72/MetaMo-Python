"""Train bounded GridWorld appraisal corrections and evaluate held-out effects.

Use after ``experiment.gridworld.collect`` and ``experiment.gridworld.targets``.
The decision audit uses stored one-step counterfactuals; the episode audit
keeps Q learning fixed while switching only the appraisal mode at evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from applications.gridworld.evaluation.runner import (
    _evaluate_agent, _train_agent, _variant_specs,
)
from applications.gridworld.environment import GridWorld
from applications.gridworld.learned_appraisal import (
    gridworld_learned_appraisal_manifest, with_learned_appraisal,
)
from applications.gridworld.profile import GRIDWORLD_DECISION_PROFILE
from dynamics.stability import DefaultStabilityPolicy
from experiment.gridworld.targets import (
    _reconstruct_candidates, _reconstruct_state, _scores_for_residual,
    safety_first_outcome_key,
)
from experiment.network import ResidualMLP
from core.features import GoalChangeFeedback
from magus.decision import MagusDecision
from openpsi.hybrid import (
    HybridAppraisalConfig, HybridAppraisalHistory, HybridAppraisalMode,
    bound_appraisal_residual,
)
from openpsi.learned import LearnedAppraisalManifest, file_sha256


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_inputs(targets_dir: Path) -> tuple[LearnedAppraisalManifest, list[dict], dict[str, dict]]:
    meta_path = targets_dir / "manifest.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "complete":
        raise ValueError("target generation did not complete")
    target_path = targets_dir / "targets.jsonl"
    if file_sha256(target_path) != meta["targets_sha256"]:
        raise ValueError("target checksum mismatch")
    dataset_dir = Path(meta["dataset_directory"])
    source_path = dataset_dir / "transitions.jsonl"
    if file_sha256(source_path) != meta["dataset_records_sha256"]:
        raise ValueError("source transition checksum mismatch")
    manifest = LearnedAppraisalManifest.from_dict(meta["learned_appraisal_manifest"])
    if manifest != gridworld_learned_appraisal_manifest():
        raise ValueError("target coordinate contract differs from current GridWorld")
    targets = _jsonl(target_path)
    records = {record["record_id"]: record for record in _jsonl(source_path)}
    if len(records) == 0 or len(targets) == 0:
        raise ValueError("target/source dataset is empty")
    return manifest, targets, records


def _arrays(rows: list[dict], manifest: LearnedAppraisalManifest) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("a required split is empty")
    x = np.asarray([row["input_vector"] for row in rows], dtype=float)
    y = np.asarray([row["target_normalized"] for row in rows], dtype=float)
    if x.shape != (len(rows), manifest.input_size) or y.shape != (len(rows), manifest.output_size):
        raise ValueError("target tensors disagree with the appraisal manifest")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("target tensors contain non-finite values")
    return x, y


def _paired_seed_ci(values: list[float]) -> list[float] | None:
    """A descriptive 95% bootstrap interval over independent seed groups."""
    if len(values) < 2:
        return None
    rng = np.random.default_rng(20261004)
    samples = rng.choice(np.asarray(values, dtype=float), size=(2000, len(values)), replace=True)
    means = samples.mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _audit_decisions(
    rows: list[dict], records: dict[str, dict],
    manifest: LearnedAppraisalManifest, model: ResidualMLP,
) -> tuple[dict, list[dict]]:
    decision = MagusDecision(profile=GRIDWORLD_DECISION_PROFILE)
    stability = DefaultStabilityPolicy()
    audited = []
    for row in rows:
        record = records[row["record_id"]]
        context = next(
            item for item in record["contexts"]
            if item["perspective"] == row["perspective"]
        )
        prediction = model.predict(np.asarray(row["input_vector"], dtype=float))
        residual, _ = bound_appraisal_residual(
            manifest, HybridAppraisalConfig(), prediction,
            float(row["boundary_pressure"]),
        )
        candidates = _reconstruct_candidates(record["candidates"])
        scores = _scores_for_residual(
            rule_state=_reconstruct_state(context["rule_state"]),
            residual=residual,
            output_modulator_names=manifest.output_modulator_names,
            candidates=candidates,
            feedback=GoalChangeFeedback(context["goal_change_feedback"]),
            decision=decision,
            stability=stability,
        )
        chosen = int(np.argmax(scores))
        baseline = int(row["baseline_action_index"])
        oracle = int(row["oracle_action_index"])
        target = int(row["target_action_index"])
        outcomes = record["counterfactuals"]
        audited.append({
            "target_id": row["target_id"],
            "group_id": row["group_id"],
            "split": row["split"],
            "regime": row["regime"],
            "perspective": row["perspective"],
            "status": row["status"],
            "baseline_action": baseline,
            "target_action": target,
            "enabled_action": chosen,
            "oracle_action": oracle,
            "target_replay_verified": bool(row["replay_verified"]),
            "enabled_changed_action": chosen != baseline,
            "enabled_improved_oracle_match": chosen == oracle and baseline != oracle,
            "baseline_oracle_match": baseline == oracle,
            "target_oracle_match": target == oracle,
            "enabled_oracle_match": chosen == oracle,
            "enabled_outcome_better": (
                safety_first_outcome_key(outcomes[chosen])
                > safety_first_outcome_key(outcomes[baseline])
            ),
            "enabled_outcome_worse": (
                safety_first_outcome_key(outcomes[chosen])
                < safety_first_outcome_key(outcomes[baseline])
            ),
            "baseline_reward": float(outcomes[baseline]["reward"]),
            "enabled_reward": float(outcomes[chosen]["reward"]),
            "baseline_lava": bool(outcomes[baseline]["in_lava"]),
            "enabled_lava": bool(outcomes[chosen]["in_lava"]),
            "applied_residual_norm": float(np.linalg.norm(residual)),
        })
    n = len(audited)
    counts = Counter(item["status"] for item in audited)
    grouped = {}
    for item in audited:
        grouped.setdefault(item["group_id"], []).append(item)
    reward_group_deltas = [
        float(np.mean([item["enabled_reward"] - item["baseline_reward"] for item in items]))
        for items in grouped.values()
    ]
    oracle_group_deltas = [
        float(np.mean([
            int(item["enabled_oracle_match"]) - int(item["baseline_oracle_match"])
            for item in items
        ]))
        for items in grouped.values()
    ]
    metrics = {
        "cases": n,
        "independent_seed_groups": len(grouped),
        "target_status_counts": dict(sorted(counts.items())),
        "target_replay_verified_rate": float(np.mean([item["target_replay_verified"] for item in audited])),
        "target_oracle_match_rate": float(np.mean([item["target_oracle_match"] for item in audited])),
        "enabled_action_change_rate": float(np.mean([item["enabled_changed_action"] for item in audited])),
        "baseline_oracle_match_rate": float(np.mean([item["baseline_oracle_match"] for item in audited])),
        "enabled_oracle_match_rate": float(np.mean([item["enabled_oracle_match"] for item in audited])),
        "enabled_outcome_better_rate": float(np.mean([item["enabled_outcome_better"] for item in audited])),
        "enabled_outcome_worse_rate": float(np.mean([item["enabled_outcome_worse"] for item in audited])),
        "mean_reward_delta": float(np.mean([item["enabled_reward"] - item["baseline_reward"] for item in audited])),
        "seed_group_reward_delta_ci95": _paired_seed_ci(reward_group_deltas),
        "seed_group_oracle_match_delta_ci95": _paired_seed_ci(oracle_group_deltas),
        "mean_applied_residual_norm": float(np.mean([item["applied_residual_norm"] for item in audited])),
    }
    return metrics, audited


def _episode_metrics(logs) -> dict:
    return {
        "episodes": len(logs),
        "mean_reward": float(np.mean([log.total_reward for log in logs])),
        "mean_completion": float(np.mean([log.completion_rate() for log in logs])),
        "mean_lava_steps": float(np.mean([log.lava_steps for log in logs])),
        "mean_unsafe_rate": float(np.mean([log.unsafe_rate() for log in logs])),
        "survival_rate": float(np.mean([log.survived for log in logs])),
    }


def _selector_probes(
    trained, manifest: LearnedAppraisalManifest, model: ResidualMLP,
    *, seed: int, episodes: int, max_steps: int, danger_probability: float,
) -> dict:
    """Compare both selectors on identical frozen-Q, rule-trajectory states."""
    behavior = copy.deepcopy(trained)
    comparisons = []
    for episode in range(episodes):
        env = GridWorld(
            seed=2_000_000 + seed * 100000 + episode,
            max_steps=max_steps,
            danger_mineral_probability=danger_probability,
        )
        state = env.reset()
        behavior.reset_episode()
        for _ in range(max_steps):
            enabled = copy.deepcopy(behavior)
            enabled.runtime = with_learned_appraisal(
                enabled.runtime,
                mode=HybridAppraisalMode.ENABLED,
                predictor=model,
                manifest=manifest,
            )
            enabled_action, enabled_info = enabled.select_action(state)
            rule_action, rule_info = behavior.select_action(state)
            rule_copy = copy.deepcopy(env)
            enabled_copy = copy.deepcopy(env)
            rule_next, rule_reward, _, _ = rule_copy.step(rule_action)
            enabled_next, enabled_reward, _, _ = enabled_copy.step(enabled_action)
            comparisons.append({
                "action_changed": enabled_action != rule_action,
                "greedy_action_changed": (
                    enabled_info["greedy_action"] != rule_info["greedy_action"]
                ),
                "reward_delta": enabled_reward - rule_reward,
                "lava_delta": int(enabled_next["in_lava"]) - int(rule_next["in_lava"]),
            })
            state, _, done, _ = env.step(rule_action)
            behavior.commit_motivational_transition()
            if done:
                break
    return {
        "states": len(comparisons),
        "executed_action_change_rate": float(np.mean([row["action_changed"] for row in comparisons])),
        "greedy_action_change_rate": float(np.mean([row["greedy_action_changed"] for row in comparisons])),
        "mean_one_step_reward_delta": float(np.mean([row["reward_delta"] for row in comparisons])),
        "mean_one_step_lava_delta": float(np.mean([row["lava_delta"] for row in comparisons])),
        "scope": "counterfactual actions on states visited by the rule policy; not full enabled trajectories",
    }


def _audit_episodes(
    manifest: LearnedAppraisalManifest, model: ResidualMLP,
    *, train_episodes: int, eval_episodes: int, agent_seeds: tuple[int, ...],
    max_steps: int, danger_probability: float,
) -> dict:
    if train_episodes == 0 or eval_episodes == 0:
        return {"status": "skipped"}
    variant = _variant_specs()["QTrainMetaMoEval"]
    seed_rows = []
    for seed in agent_seeds:
        trained = variant.factory(seed)
        _train_agent(
            trained, variant, train_episodes, seed_start=seed * 100000,
            max_steps=max_steps, danger_distance=2,
            danger_mineral_probability=danger_probability,
        )
        arm_logs = {}
        for mode in ("rule", "shadow", "enabled"):
            agent = copy.deepcopy(trained)
            if mode != "rule":
                agent.runtime = with_learned_appraisal(
                    agent.runtime,
                    mode=HybridAppraisalMode(mode),
                    predictor=model,
                    manifest=manifest,
                    history=HybridAppraisalHistory(max_records=None),
                )
            arm_logs[mode] = _evaluate_agent(
                agent, variant, eval_episodes,
                seed_start=1_000_000 + seed * 100000,
                max_steps=max_steps, danger_distance=2,
                danger_mineral_probability=danger_probability,
                audit_compositionality=True,
            )
        for rule, shadow in zip(arm_logs["rule"], arm_logs["shadow"]):
            if vars(rule) != vars(shadow):
                raise AssertionError("shadow changed the GridWorld episode")
        seed_rows.append({
            "seed": seed,
            "rule": _episode_metrics(arm_logs["rule"]),
            "shadow": _episode_metrics(arm_logs["shadow"]),
            "enabled": _episode_metrics(arm_logs["enabled"]),
            "selector_probes": _selector_probes(
                trained, manifest, model,
                seed=seed, episodes=eval_episodes,
                max_steps=max_steps, danger_probability=danger_probability,
            ),
        })
    return {
        "status": "complete",
        "design": "same trained Q table and paired environment seeds; appraisal switched only at evaluation",
        "by_seed": seed_rows,
        "mean_reward_delta": float(np.mean([
            row["enabled"]["mean_reward"] - row["rule"]["mean_reward"]
            for row in seed_rows
        ])),
        "mean_lava_steps_delta": float(np.mean([
            row["enabled"]["mean_lava_steps"] - row["rule"]["mean_lava_steps"]
            for row in seed_rows
        ])),
        "reward_delta_ci95_by_agent_seed": _paired_seed_ci([
            row["enabled"]["mean_reward"] - row["rule"]["mean_reward"]
            for row in seed_rows
        ]),
        "lava_steps_delta_ci95_by_agent_seed": _paired_seed_ci([
            row["enabled"]["mean_lava_steps"] - row["rule"]["mean_lava_steps"]
            for row in seed_rows
        ]),
    }


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    if args.epochs < 1 or args.q_train_episodes < 0 or args.eval_episodes < 0 or args.max_steps < 1:
        raise ValueError("epoch and episode counts are invalid")
    if not 0.0 <= args.danger_probability <= 1.0:
        raise ValueError("danger probability must be in [0, 1]")
    base_manifest, targets, records = _load_inputs(Path(args.targets_dir))
    splits = {name: [row for row in targets if row["split"] == name]
              for name in ("train", "validation", "test", "ood")}
    groups = {name: {row["group_id"] for row in rows} for name, rows in splits.items()}
    for left in groups:
        for right in groups:
            if left < right and groups[left].intersection(groups[right]):
                raise ValueError(f"data leakage between {left} and {right} seed groups")
    train_x, train_y = _arrays(splits["train"], base_manifest)
    val_x, val_y = _arrays(splits["validation"], base_manifest)
    model = ResidualMLP(base_manifest.input_size, base_manifest.output_size, seed=args.seed)
    training = model.fit(train_x, train_y, val_x, val_y, epochs=args.epochs)
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "model.npz"
    model.save(artifact)
    manifest = replace(
        base_manifest,
        model_id="gridworld-bounded-residual-mlp-v1",
        artifact_path=artifact.name,
        artifact_sha256=file_sha256(artifact),
    )
    (output / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    manifest = LearnedAppraisalManifest.read_json(output / "manifest.json")
    model = ResidualMLP.load(manifest.artifact_path)
    audits = {}
    all_rows = []
    for split in ("test", "ood"):
        if splits[split]:
            audits[split], rows = _audit_decisions(splits[split], records, manifest, model)
            all_rows.extend(rows)
    if all_rows:
        with (output / "decision_cases.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    seeds = tuple(int(item) for item in args.agent_seeds.split(",") if item.strip())
    if not seeds:
        raise ValueError("at least one agent seed is required")
    episodes = _audit_episodes(
        manifest, model,
        train_episodes=args.q_train_episodes,
        eval_episodes=args.eval_episodes,
        agent_seeds=seeds,
        max_steps=args.max_steps,
        danger_probability=args.danger_probability,
    )
    summary = {
        "design": "bounded residual MLP, disjoint environment-seed splits, fixed safety guard",
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "training": training,
        "decision_audits": audits,
        "frozen_q_episode_audit": episodes,
        "caveat": "the decision oracle is safety-first one-step; episode results are required for outcome claims",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets_dir", help="complete output from experiment.gridworld.targets")
    parser.add_argument("--output-dir", default="experiment/results/gridworld-run")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--q-train-episodes", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--agent-seeds", default="0,1,2")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--danger-probability", type=float, default=0.55)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
