"""Run a small learned-appraisal control-authority check without training.

This command intentionally avoids the multi-seed Q-learning evaluation and
neural-network training.  It collects a small current-schema dataset, runs the
deterministic bounded residual search, and reports whether the global
modulator channel can correct any initially non-oracle action rankings.

Example:

    python -m experiment.gridworld.quick_reachability
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from experiment.gridworld.baseline import PROJECT_ROOT, _write_json
from experiment.gridworld.collect import CollectionConfig, collect_dataset
from experiment.gridworld.targets import generate_targets


def summarize_audit(audit: dict[str, Any]) -> dict[str, Any]:
    """Convert the detailed feasibility audit into one actionable result."""

    scope = audit["scope"]
    authority = audit["control_authority"]
    target_count = int(scope["target_count"])
    baseline_correct = int(authority["baseline_oracle_selected_count"])
    initially_incorrect = target_count - baseline_correct
    action_switches = int(authority["action_switch_feasible_count"])
    correctable_rate = (
        action_switches / initially_incorrect if initially_incorrect else None
    )

    if initially_incorrect == 0:
        result = "inconclusive"
        recommendation = (
            "The pilot found no initially incorrect rankings. Collect more diverse "
            "states before deciding whether to train."
        )
    elif action_switches == 0:
        result = "global_residual_lacks_control_authority"
        recommendation = (
            "Do not train the neural residual yet. Test an application-local, "
            "bounded action-conditioned appraisal interface."
        )
    else:
        result = "global_residual_has_control_authority"
        recommendation = (
            "Inspect the correctable cases and target consistency. Train only in "
            "shadow mode if recurring patterns remain across grouped splits."
        )

    return {
        "result": result,
        "target_contexts": target_count,
        "baseline_oracle_selected": baseline_correct,
        "initially_incorrect": initially_incorrect,
        "action_switch_feasible": action_switches,
        "correctable_rate_among_initially_incorrect": correctable_rate,
        "status_counts": audit["status_counts"],
        "recommendation": recommendation,
    }


def run_quick_check(
    output_dir: str | Path,
    *,
    episodes_per_seed: int = 1,
    max_steps: int = 10,
    random_samples: int = 0,
    required_margin: float = 0.02,
    improvement_threshold: float = 0.001,
) -> Path:
    """Collect and audit a compact dataset using four disjoint seed groups."""

    output = Path(output_dir)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if output.exists():
        raise FileExistsError(f"quick-check output already exists: {output}")

    dataset_dir = output / "dataset"
    target_dir = output / "targets"
    started = perf_counter()

    print("[1/2] Collecting a small rule-only dataset (no training) ...")
    collect_dataset(
        dataset_dir,
        CollectionConfig(
            train_seeds=(0,),
            validation_seeds=(1,),
            test_seeds=(2,),
            ood_seeds=(100,),
            episodes_per_seed=episodes_per_seed,
            max_steps=max_steps,
        ),
    )

    print("[2/2] Testing bounded corrections with the current runtime guard ...")
    generate_targets(
        dataset_dir,
        target_dir,
        required_margin=required_margin,
        random_samples=random_samples,
        improvement_threshold=improvement_threshold,
    )

    with (target_dir / "feasibility_audit.json").open(encoding="utf-8") as handle:
        audit = json.load(handle)
    with (dataset_dir / "manifest.json").open(encoding="utf-8") as handle:
        dataset_manifest = json.load(handle)

    summary = summarize_audit(audit)
    summary.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": perf_counter() - started,
            "training_performed": False,
            "dataset_records": dataset_manifest["record_count"],
            "random_samples_per_context": random_samples,
            "dataset_directory": str(dataset_dir),
            "target_directory": str(target_dir),
        }
    )
    summary_path = output / "quick_reachability_summary.json"
    _write_json(summary_path, summary)

    rate = summary["correctable_rate_among_initially_incorrect"]
    rate_text = "n/a" if rate is None else f"{100.0 * rate:.1f}%"
    print("Quick reachability check complete")
    print(f"  records: {summary['dataset_records']}")
    print(f"  initially incorrect contexts: {summary['initially_incorrect']}")
    print(f"  feasible action switches: {summary['action_switch_feasible']}")
    print(f"  correctable rate: {rate_text}")
    print(f"  result: {summary['result']}")
    print(f"  next: {summary['recommendation']}")
    print(f"  summary: {summary_path}")
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        help="new output directory; defaults to a UTC timestamp",
    )
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--random-samples",
        type=int,
        default=0,
        help="extra random candidates per context; zero is sufficient for the pilot",
    )
    parser.add_argument("--required-margin", type=float, default=0.02)
    parser.add_argument("--improvement-threshold", type=float, default=0.001)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or (
        PROJECT_ROOT
        / "experiment"
        / "results"
        / f"quick-reachability-{timestamp}"
    )
    run_quick_check(
        output,
        episodes_per_seed=args.episodes_per_seed,
        max_steps=args.max_steps,
        random_samples=args.random_samples,
        required_margin=args.required_margin,
        improvement_threshold=args.improvement_threshold,
    )


if __name__ == "__main__":
    main()
