"""Collect Phase-2 GridWorld learned-appraisal transitions.

The collector is offline and rule-only.  Each environment transition stores
three appraisal contexts (base, safety, and growth) plus one-step
counterfactual outcomes for every legal action.  Complete seed groups are
assigned to exactly one of train, validation, test, or OOD.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from applications.gridworld.learned_appraisal import (
    gridworld_learned_appraisal_manifest,
)
from applications.gridworld.profile import (
    GRIDWORLD_APPRAISAL_PROFILE,
    GRIDWORLD_DECISION_PROFILE,
    GRIDWORLD_PROFILE,
)
from category.bimonad import MetaMoPseudoBimonad
from dynamics.coherence import DefaultCoherencePolicy
from dynamics.stability import DefaultStabilityPolicy
from experiment.gridworld.baseline import (
    PROJECT_ROOT,
    _jsonable,
    _source_fingerprint,
    _write_json,
)
from magus.decision import MagusDecision
from openpsi.appraisal import OpenPsiAppraisal
from openpsi.learned import ManifestFeatureEncoder, file_sha256
from applications.gridworld.environment import GridWorld


DATASET_VERSION = 1


@dataclass(frozen=True)
class CollectionRegime:
    name: str
    danger_mineral_probability: float
    lava_cell_count: int = 4
    mineral_spawn_band: int = 2
    is_ood: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("regime name must be non-empty")
        if not 0.0 <= self.danger_mineral_probability <= 1.0:
            raise ValueError("danger_mineral_probability must be in [0, 1]")
        if self.lava_cell_count <= 0:
            raise ValueError("lava_cell_count must be positive")
        if self.mineral_spawn_band < 0:
            raise ValueError("mineral_spawn_band must be non-negative")


DEFAULT_REGIMES = (
    CollectionRegime("low", 0.20),
    CollectionRegime("medium", 0.55),
    CollectionRegime("high", 0.80),
)
DEFAULT_OOD_REGIMES = (
    CollectionRegime(
        "extreme_layout",
        0.95,
        lava_cell_count=8,
        mineral_spawn_band=3,
        is_ood=True,
    ),
)


@dataclass(frozen=True)
class CollectionConfig:
    train_seeds: tuple[int, ...] = tuple(range(10))
    validation_seeds: tuple[int, ...] = (10, 11)
    test_seeds: tuple[int, ...] = (12, 13)
    ood_seeds: tuple[int, ...] = (100, 101, 102, 103)
    episodes_per_seed: int = 6
    max_steps: int = 75
    exploration_probability: float = 0.35
    danger_distance: int = 2
    regimes: tuple[CollectionRegime, ...] = DEFAULT_REGIMES
    ood_regimes: tuple[CollectionRegime, ...] = DEFAULT_OOD_REGIMES

    def __post_init__(self) -> None:
        if self.episodes_per_seed <= 0:
            raise ValueError("episodes_per_seed must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not 0.0 <= self.exploration_probability <= 1.0:
            raise ValueError("exploration_probability must be in [0, 1]")
        if self.danger_distance < 0:
            raise ValueError("danger_distance must be non-negative")
        if not self.regimes or not self.ood_regimes:
            raise ValueError("normal and OOD regimes must both be configured")

        split_sets = {
            "train": set(self.train_seeds),
            "validation": set(self.validation_seeds),
            "test": set(self.test_seeds),
            "ood": set(self.ood_seeds),
        }
        for name, values in split_sets.items():
            if not values:
                raise ValueError(f"{name} seed set must not be empty")
        names = tuple(split_sets)
        for idx, left in enumerate(names):
            for right in names[idx + 1 :]:
                overlap = split_sets[left].intersection(split_sets[right])
                if overlap:
                    raise ValueError(
                        f"seed groups overlap between {left} and {right}: {sorted(overlap)}"
                    )


def parse_seed_spec(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(part))
    if not values:
        raise ValueError("seed specification must not be empty")
    if len(values) != len(set(values)):
        raise ValueError("seed specification contains duplicates")
    return tuple(values)


def _environment_state(state: dict) -> dict[str, Any]:
    return {
        "pos": state["pos"],
        "mineral_pos": state["mineral_pos"],
        "energy": float(state["energy"]),
        "in_lava": bool(state["in_lava"]),
        "lava_distance": int(state["lava_distance"]),
        "lava_cells": state["lava_cells"],
        "dx_mineral": int(state["dx_mineral"]),
        "dy_mineral": int(state["dy_mineral"]),
        "step": int(state["step"]),
    }


def _motivational_state(state) -> dict[str, Any]:
    return {
        "G": state.G.tolist(),
        "M": state.M.tolist(),
    }


def _candidate(candidate) -> dict[str, Any]:
    return {
        "id": candidate.id,
        "goal_correlations": candidate.goal_correlations.tolist(),
        "delta_g": candidate.delta_g.tolist(),
        "metadata": dict(candidate.metadata),
    }


def _counterfactual_outcomes(
    env: GridWorld,
    before: dict,
    danger_distance: int,
) -> list[dict[str, Any]]:
    before_distance = abs(before["dx_mineral"]) + abs(before["dy_mineral"])
    outcomes: list[dict[str, Any]] = []
    for action_idx, action_id in enumerate(GRIDWORLD_PROFILE.action_ids):
        branch = copy.deepcopy(env)
        minerals_before = branch.minerals_collected
        after, reward, done, info = branch.step(action_idx)
        mineral_collected = branch.minerals_collected > minerals_before
        if mineral_collected:
            progress = float(before_distance)
        else:
            after_distance = abs(after["dx_mineral"]) + abs(after["dy_mineral"])
            progress = float(before_distance - after_distance)
        outcomes.append(
            {
                "action_index": action_idx,
                "action_id": action_id,
                "next_state": _environment_state(after),
                "reward": float(reward),
                "event": info.get("event"),
                "done": bool(done),
                "dead": bool(after["energy"] <= 0.0),
                "in_lava": bool(after["in_lava"]),
                "unsafe_zone": bool(
                    after["in_lava"]
                    or after["lava_distance"] <= danger_distance
                ),
                "mineral_collected": mineral_collected,
                "movement": bool(after["pos"] != before["pos"]),
                "progress": progress,
                "energy_after": float(after["energy"]),
                "lava_distance_after": int(after["lava_distance"]),
            }
        )
    return outcomes


def _context_record(
    name: str,
    state,
    stimulus,
    candidates,
    bimonad: MetaMoPseudoBimonad,
    encoder: ManifestFeatureEncoder,
) -> dict[str, Any]:
    features = GRIDWORLD_APPRAISAL_PROFILE.stimulus_features(state, stimulus)
    rule_state, feedback = bimonad.appraisal.appraise_with_feedback(state, stimulus)
    decision_state = bimonad.stability_policy.raise_boundary_caution(rule_state)
    result = bimonad.decision.decide(
        decision_state,
        candidates,
        feedback=feedback,
    )
    input_vector = encoder.encode(
        state,
        GRIDWORLD_APPRAISAL_PROFILE,
        features,
        rule_state,
    )
    return {
        "perspective": name,
        "source_state": _motivational_state(state),
        "features": features.as_dict(),
        "rule_state": _motivational_state(rule_state),
        "decision_state": _motivational_state(decision_state),
        "goal_change_feedback": feedback.as_dict(),
        "boundary_pressure": encoder.pressure(state),
        "input_vector": input_vector.tolist(),
        "rule_chosen_action": result.action.id,
        "rule_scores": [asdict(score) for score in result.candidate_scores],
    }


def _consensus_scores(
    bimonad: MetaMoPseudoBimonad,
    safety_state,
    growth_state,
    stimulus,
    candidates,
) -> np.ndarray:
    context_a = bimonad._decision_context(safety_state, stimulus)
    context_b = bimonad._decision_context(growth_state, stimulus)
    scores = []
    for candidate in candidates:
        score_a = bimonad.decision.score_candidate(context_a, candidate)
        score_b = bimonad.decision.score_candidate(context_b, candidate)
        scores.append(((score_a + score_b) / 2.0) - 0.25 * abs(score_a - score_b))
    return np.asarray(scores, dtype=float)


def _direct_consensus_transition(
    bimonad: MetaMoPseudoBimonad,
    current_state,
    safety_state,
    growth_state,
    stimulus,
    candidate,
) -> tuple[Any, dict[str, Any]]:
    action, merged_target = bimonad.consensus_transition(
        safety_state,
        growth_state,
        stimulus,
        [candidate],
    )
    projected_target = bimonad.stability_policy.project_to_safe_region(merged_target)
    projection_delta = merged_target.distance_to(projected_target)
    blend = bimonad.coherence_policy.measure_blend(current_state, projected_target)
    final_state = blend.state
    final_projection_delta = 0.0
    if not bimonad.stability_policy.is_in_safe_region(final_state):
        projected_final = bimonad.stability_policy.project_to_safe_region(final_state)
        final_projection_delta = final_state.distance_to(projected_final)
        final_state = projected_final

    _, _, single_path_diagnostics = bimonad.step_with_diagnostics(
        current_state,
        stimulus,
        [candidate],
        record_diagnostics=False,
    )
    telemetry = {
        "action_id": action.id,
        "merged_target": _motivational_state(merged_target),
        "projected_target": _motivational_state(projected_target),
        "projection_delta": projection_delta,
        "blend_alpha": blend.alpha,
        "base_blend_alpha": blend.base_alpha,
        "self_model_drift": asdict(blend.drift),
        "final_state": _motivational_state(final_state),
        "final_projection_delta": final_projection_delta,
        "final_in_safe_region": bimonad.stability_policy.is_in_safe_region(final_state),
        "single_path_law_diagnostics": asdict(single_path_diagnostics),
        "single_path_diagnostic_scope": (
            "rule-only transition from the unsplit current state using the selected candidate"
        ),
    }
    return final_state, telemetry


def _iter_split_jobs(config: CollectionConfig) -> Iterable[tuple[str, int, CollectionRegime]]:
    normal_splits = (
        ("train", config.train_seeds),
        ("validation", config.validation_seeds),
        ("test", config.test_seeds),
    )
    for split, seeds in normal_splits:
        for seed in seeds:
            for regime in config.regimes:
                yield split, seed, regime
    for seed in config.ood_seeds:
        for regime in config.ood_regimes:
            yield "ood", seed, regime


def _validate_record(record: dict[str, Any], manifest) -> None:
    if record["dataset_version"] != DATASET_VERSION:
        raise ValueError("dataset version mismatch")
    if record["split"] not in {"train", "validation", "test", "ood"}:
        raise ValueError("invalid dataset split")
    if len(record["candidates"]) != len(GRIDWORLD_PROFILE.action_ids):
        raise ValueError("candidate set is incomplete")
    counterfactual_ids = [item["action_id"] for item in record["counterfactuals"]]
    if counterfactual_ids != list(GRIDWORLD_PROFILE.action_ids):
        raise ValueError("counterfactual action order does not match the profile")
    if {context["perspective"] for context in record["contexts"]} != {
        "base",
        "safety",
        "growth",
    }:
        raise ValueError("base, safety, and growth contexts are required")
    for context in record["contexts"]:
        vector = np.asarray(context["input_vector"], dtype=float)
        if vector.shape != (manifest.input_size,) or not np.all(np.isfinite(vector)):
            raise ValueError("context input vector is invalid")


def collect_dataset(output_dir: str | Path, config: CollectionConfig) -> Path:
    output = Path(output_dir)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if output.exists():
        raise FileExistsError(f"dataset output already exists: {output}")
    output.mkdir(parents=True)

    learned_manifest = gridworld_learned_appraisal_manifest()
    encoder = ManifestFeatureEncoder(
        manifest=learned_manifest,
        boundary_pressure=DefaultStabilityPolicy().boundary_pressure,
    )
    bimonad = MetaMoPseudoBimonad(
        OpenPsiAppraisal(profile=GRIDWORLD_APPRAISAL_PROFILE),
        MagusDecision(profile=GRIDWORLD_DECISION_PROFILE),
        stability_policy=DefaultStabilityPolicy(),
        coherence_policy=DefaultCoherencePolicy(),
    )
    started = datetime.now(timezone.utc)
    source_digest, _ = _source_fingerprint()
    dataset_manifest: dict[str, Any] = {
        "dataset_version": DATASET_VERSION,
        "status": "running",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": None,
        "source_sha256": source_digest,
        "learned_appraisal_manifest": learned_manifest.to_dict(),
        "collection_config": asdict(config),
        "record_count": 0,
        "context_count": 0,
        "split_counts": {},
        "split_groups": {},
    }
    _write_json(output / "manifest.json", dataset_manifest)

    record_path = output / "transitions.jsonl"
    split_counts = {name: 0 for name in ("train", "validation", "test", "ood")}
    split_groups: dict[str, set[str]] = {
        name: set() for name in ("train", "validation", "test", "ood")
    }
    record_ids: set[str] = set()

    try:
        with record_path.open("w", encoding="utf-8", newline="\n") as handle:
            for job_index, (split, seed, regime) in enumerate(_iter_split_jobs(config)):
                group_id = f"seed-{seed}"
                split_groups[split].add(group_id)
                for episode in range(config.episodes_per_seed):
                    env_seed = (seed * 100000) + (job_index * 1000) + episode
                    env = GridWorld(
                        seed=env_seed,
                        max_steps=config.max_steps,
                        lava_cell_count=regime.lava_cell_count,
                        mineral_spawn_band=regime.mineral_spawn_band,
                        danger_mineral_probability=regime.danger_mineral_probability,
                    )
                    env_state = env.reset()
                    mot_state = GRIDWORLD_PROFILE.create_initial_state()
                    behavior_rng = np.random.default_rng(env_seed + 7919)

                    while not env.done:
                        stimulus = GRIDWORLD_PROFILE.build_stimulus(env_state, mot_state)
                        candidates = GRIDWORLD_PROFILE.build_candidates(env_state, mot_state)
                        safety_state, growth_state = GRIDWORLD_PROFILE.build_consensus_states(
                            env_state,
                            mot_state,
                            stimulus,
                        )
                        consensus_scores = _consensus_scores(
                            bimonad,
                            safety_state,
                            growth_state,
                            stimulus,
                            candidates,
                        )
                        if behavior_rng.random() < config.exploration_probability:
                            action_index = int(behavior_rng.integers(0, len(candidates)))
                            action_source = "uniform_exploration"
                        else:
                            action_index = int(np.argmax(consensus_scores))
                            action_source = "rule_consensus"

                        counterfactuals = _counterfactual_outcomes(
                            env,
                            env_state,
                            config.danger_distance,
                        )
                        contexts = [
                            _context_record(
                                "base",
                                mot_state,
                                stimulus,
                                candidates,
                                bimonad,
                                encoder,
                            ),
                            _context_record(
                                "safety",
                                safety_state,
                                stimulus,
                                candidates,
                                bimonad,
                                encoder,
                            ),
                            _context_record(
                                "growth",
                                growth_state,
                                stimulus,
                                candidates,
                                bimonad,
                                encoder,
                            ),
                        ]
                        next_mot_state, transition_telemetry = _direct_consensus_transition(
                            bimonad,
                            mot_state,
                            safety_state,
                            growth_state,
                            stimulus,
                            candidates[action_index],
                        )
                        next_env_state, reward, done, info = env.step(action_index)
                        selected_counterfactual = counterfactuals[action_index]
                        if (
                            selected_counterfactual["next_state"]
                            != _environment_state(next_env_state)
                            or selected_counterfactual["reward"] != float(reward)
                            or selected_counterfactual["event"] != info.get("event")
                        ):
                            raise AssertionError(
                                "selected counterfactual does not match the actual transition"
                            )

                        record_id = (
                            f"{split}:{group_id}:{regime.name}:"
                            f"episode-{episode}:step-{env_state['step']}"
                        )
                        if record_id in record_ids:
                            raise AssertionError(f"duplicate record id: {record_id}")
                        record_ids.add(record_id)
                        record = {
                            "dataset_version": DATASET_VERSION,
                            "record_id": record_id,
                            "split": split,
                            "group_id": group_id,
                            "seed": seed,
                            "environment_seed": env_seed,
                            "regime": asdict(regime),
                            "episode": episode,
                            "step": int(env_state["step"]),
                            "environment_state": _environment_state(env_state),
                            "motivational_state": _motivational_state(mot_state),
                            "stimulus": asdict(stimulus),
                            "candidates": [_candidate(candidate) for candidate in candidates],
                            "contexts": contexts,
                            "consensus_rule_scores": consensus_scores.tolist(),
                            "behavior_action_index": action_index,
                            "behavior_action_id": candidates[action_index].id,
                            "behavior_action_source": action_source,
                            "counterfactuals": counterfactuals,
                            "actual_outcome": selected_counterfactual,
                            "actual_reward": float(reward),
                            "actual_event": info.get("event"),
                            "actual_done": bool(done),
                            "transition_telemetry": transition_telemetry,
                        }
                        _validate_record(record, learned_manifest)
                        handle.write(
                            json.dumps(
                                _jsonable(record),
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                        )
                        handle.write("\n")
                        split_counts[split] += 1

                        env_state = next_env_state
                        mot_state = next_mot_state
                        if done:
                            break

        group_sets = list(split_groups.items())
        for idx, (left_name, left) in enumerate(group_sets):
            for right_name, right in group_sets[idx + 1 :]:
                overlap = left.intersection(right)
                if overlap:
                    raise AssertionError(
                        f"group leakage between {left_name} and {right_name}: {sorted(overlap)}"
                    )

        dataset_manifest.update(
            {
                "status": "complete",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "record_count": sum(split_counts.values()),
                "context_count": 3 * sum(split_counts.values()),
                "split_counts": split_counts,
                "split_groups": {
                    name: sorted(groups) for name, groups in split_groups.items()
                },
                "records_sha256": file_sha256(record_path),
                "validation": {
                    "unique_record_ids": len(record_ids),
                    "group_overlap_count": 0,
                    "contexts_per_record": 3,
                    "counterfactuals_per_record": len(GRIDWORLD_PROFILE.action_ids),
                },
            }
        )
        _write_json(output / "manifest.json", dataset_manifest)
    except Exception as error:
        dataset_manifest.update(
            {
                "status": "failed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "split_counts": split_counts,
            }
        )
        _write_json(output / "manifest.json", dataset_manifest)
        raise
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")
    parser.add_argument("--train-seeds", default="0-9")
    parser.add_argument("--validation-seeds", default="10-11")
    parser.add_argument("--test-seeds", default="12-13")
    parser.add_argument("--ood-seeds", default="100-103")
    parser.add_argument("--episodes-per-seed", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=75)
    parser.add_argument("--exploration-probability", type=float, default=0.35)
    parser.add_argument("--danger-distance", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or (
        PROJECT_ROOT / "experiment" / "results" / f"gridworld-dataset-{timestamp}"
    )
    config = CollectionConfig(
        train_seeds=parse_seed_spec(args.train_seeds),
        validation_seeds=parse_seed_spec(args.validation_seeds),
        test_seeds=parse_seed_spec(args.test_seeds),
        ood_seeds=parse_seed_spec(args.ood_seeds),
        episodes_per_seed=args.episodes_per_seed,
        max_steps=args.max_steps,
        exploration_probability=args.exploration_probability,
        danger_distance=args.danger_distance,
    )
    result = collect_dataset(output, config)
    print(f"Learned-appraisal dataset complete: {result}")


if __name__ == "__main__":
    main()
