"""Replicate the collection-only self-model test across independent seed splits."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import fsum

from applications.gridworld.evaluation.self_model import (
    SelfModelEvaluationReport,
    run_self_model_evaluation,
)


@dataclass(frozen=True)
class CollectionSplitResult:
    train_seed_start: int
    test_seed_start: int
    agent_seed: int
    report: SelfModelEvaluationReport


@dataclass(frozen=True)
class CollectionRobustnessReport:
    splits: tuple[CollectionSplitResult, ...]
    minimum_train_collections: int
    minimum_test_collections: int
    total_training_collections: int
    total_test_collections: int
    learned_proximity_mae: float | None
    nominal_proximity_mae: float | None
    learned_appraisal_mae: float | None
    nominal_appraisal_mae: float | None
    interval_scored_collections: int
    interval_coverage: float | None
    eligible_splits: int
    improved_splits: int
    behavior_unchanged: bool
    safety_passed: bool
    robustness_passed: bool


def _weighted_collection_mae(
    splits: tuple[CollectionSplitResult, ...], field: str
) -> float | None:
    total = sum(split.report.held_out_collection_steps for split in splits)
    if total == 0:
        return None
    return fsum(
        getattr(split.report, field) * split.report.held_out_collection_steps
        for split in splits
        if split.report.held_out_collection_steps
    ) / total


def run_collection_robustness_evaluation(
    *,
    split_count: int = 3,
    train_episodes: int = 60,
    test_episodes: int = 16,
    max_steps: int = 40,
    train_seed_start: int = 100000,
    test_seed_start: int = 800000,
    seed_stride: int = 100000,
    agent_seed: int = 42,
    minimum_train_collections: int = 20,
    minimum_test_collections: int = 10,
) -> CollectionRobustnessReport:
    """Use new environment and policy seeds for each train/test pair."""
    if (
        split_count < 1
        or train_episodes < 1
        or test_episodes < 1
        or max_steps < 2
        or seed_stride < 1
        or minimum_train_collections < 1
        or minimum_test_collections < 1
    ):
        raise ValueError("split counts, episode counts, steps, and thresholds must be positive")
    if train_episodes > seed_stride or test_episodes > seed_stride:
        raise ValueError("episode count exceeds seed stride")

    seed_pairs: list[tuple[range, range]] = []
    all_seeds: set[int] = set()
    for index in range(split_count):
        training = range(
            train_seed_start + index * seed_stride,
            train_seed_start + index * seed_stride + train_episodes,
        )
        testing = range(
            test_seed_start + index * seed_stride,
            test_seed_start + index * seed_stride + test_episodes,
        )
        if all_seeds.intersection(training) or all_seeds.intersection(testing) or set(training).intersection(testing):
            raise ValueError("train and test seed ranges overlap across splits")
        all_seeds.update(training)
        all_seeds.update(testing)
        seed_pairs.append((training, testing))

    splits = tuple(
        CollectionSplitResult(
            train_seed_start=training.start,
            test_seed_start=testing.start,
            agent_seed=agent_seed + index,
            report=run_self_model_evaluation(
                training,
                testing,
                max_steps=max_steps,
                agent_seed=agent_seed + index,
            ),
        )
        for index, (training, testing) in enumerate(seed_pairs)
    )
    total_test = sum(split.report.held_out_collection_steps for split in splits)
    interval_scored = sum(
        split.report.held_out_collection_steps
        for split in splits
        if split.report.collection_interval_coverage is not None
    )
    interval_hits = fsum(
        split.report.collection_interval_coverage * split.report.held_out_collection_steps
        for split in splits
        if split.report.collection_interval_coverage is not None
    )
    eligible = tuple(
        split for split in splits
        if split.report.training_collection_samples >= minimum_train_collections
        and split.report.held_out_collection_steps >= minimum_test_collections
        and split.report.learned_collection_forecast_steps == split.report.held_out_collection_steps
    )
    improved = tuple(
        split for split in eligible
        if split.report.learning_adds_value
        and split.report.learned_collection_proximity_mae
        < split.report.nominal_collection_proximity_mae
    )
    behavior_unchanged = all(split.report.behavior_unchanged for split in splits)
    safety_passed = all(split.report.safety_passed for split in splits)
    return CollectionRobustnessReport(
        splits=splits,
        minimum_train_collections=minimum_train_collections,
        minimum_test_collections=minimum_test_collections,
        total_training_collections=sum(split.report.training_collection_samples for split in splits),
        total_test_collections=total_test,
        learned_proximity_mae=_weighted_collection_mae(splits, "learned_collection_proximity_mae"),
        nominal_proximity_mae=_weighted_collection_mae(splits, "nominal_collection_proximity_mae"),
        learned_appraisal_mae=_weighted_collection_mae(splits, "learned_collection_appraisal_mae"),
        nominal_appraisal_mae=_weighted_collection_mae(splits, "nominal_collection_appraisal_mae"),
        interval_scored_collections=interval_scored,
        interval_coverage=(interval_hits / interval_scored if interval_scored else None),
        eligible_splits=len(eligible),
        improved_splits=len(improved),
        behavior_unchanged=behavior_unchanged,
        safety_passed=safety_passed,
        robustness_passed=(
            len(eligible) == split_count
            and len(improved) == split_count
            and behavior_unchanged
            and safety_passed
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--train-episodes", type=int, default=60)
    parser.add_argument("--test-episodes", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=40)
    args = parser.parse_args()
    report = run_collection_robustness_evaluation(
        split_count=args.splits,
        train_episodes=args.train_episodes,
        test_episodes=args.test_episodes,
        max_steps=args.max_steps,
    )
    print(json.dumps({
        "settings": {
            "split_count": args.splits,
            "train_episodes_per_split": args.train_episodes,
            "test_episodes_per_split": args.test_episodes,
            "max_steps": args.max_steps,
            "minimum_train_collections": report.minimum_train_collections,
            "minimum_test_collections": report.minimum_test_collections,
            "interval_percentiles": [10, 90],
        },
        "splits": [
            {
                "train_seed_start": split.train_seed_start,
                "test_seed_start": split.test_seed_start,
                "agent_seed": split.agent_seed,
                "training_collections": split.report.training_collection_samples,
                "test_collections": split.report.held_out_collection_steps,
                "learned_proximity_mae": split.report.learned_collection_proximity_mae,
                "nominal_proximity_mae": split.report.nominal_collection_proximity_mae,
                "learned_appraisal_mae": split.report.learned_collection_appraisal_mae,
                "nominal_appraisal_mae": split.report.nominal_collection_appraisal_mae,
                "interval_coverage": split.report.collection_interval_coverage,
                "learning_adds_value": split.report.learning_adds_value,
                "behavior_unchanged": split.report.behavior_unchanged,
                "safety_passed": split.report.safety_passed,
            }
            for split in report.splits
        ],
        "total_training_collections": report.total_training_collections,
        "total_test_collections": report.total_test_collections,
        "learned_proximity_mae": report.learned_proximity_mae,
        "nominal_proximity_mae": report.nominal_proximity_mae,
        "learned_appraisal_mae": report.learned_appraisal_mae,
        "nominal_appraisal_mae": report.nominal_appraisal_mae,
        "interval_scored_collections": report.interval_scored_collections,
        "interval_coverage": report.interval_coverage,
        "eligible_splits": report.eligible_splits,
        "improved_splits": report.improved_splits,
        "behavior_unchanged": report.behavior_unchanged,
        "safety_passed": report.safety_passed,
        "robustness_passed": report.robustness_passed,
    }, indent=2))


if __name__ == "__main__":
    main()
