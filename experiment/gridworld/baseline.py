"""Create an immutable Phase-0 learned-appraisal baseline snapshot.

The command refuses to reuse an output directory.  It records the exact source
fingerprint, environment, arguments, deterministic appraisal contracts, unit
test log, and a paired GridWorld evaluation.

Example:

    python -m experiment.gridworld.baseline
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np

from applications.gridworld.profile import (
    GRIDWORLD_APPRAISAL_PROFILE,
    GRIDWORLD_DECISION_PROFILE,
    GRIDWORLD_PROFILE,
)
from category.bimonad import MetaMoPseudoBimonad
from core import config as core_config
from magus.decision import MagusDecision
from openpsi.appraisal import OpenPsiAppraisal
from applications.gridworld.environment import GridWorld


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = (
    "applications/gridworld",
    "category",
    "core",
    "dynamics",
    "magus",
    "openpsi",
    "experiment/gridworld",
)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_jsonable(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _source_files() -> list[Path]:
    files: list[Path] = []
    for root_name in SOURCE_ROOTS:
        root = PROJECT_ROOT / root_name
        if root.exists():
            files.extend(root.rglob("*.py"))
    for name in ("requirements.txt", "README.md", "experiment/network.py"):
        path = PROJECT_ROOT / name
        if path.is_file():
            files.append(path)
    return sorted(set(files), key=lambda path: path.as_posix())


def _source_fingerprint() -> tuple[str, list[str]]:
    digest = sha256()
    relative_paths: list[str] = []
    for path in _source_files():
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        relative_paths.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), relative_paths


def _dependency_versions() -> dict[str, str]:
    packages = (
        "numpy",
        "pygame",
        "matplotlib",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _numeric_core_config() -> dict[str, int | float]:
    names = (
        "LAMBDA_IND",
        "LAMBDA_TRANS",
        "THETA_SAFE",
        "G_MAX",
        "C_CONTRACT",
        "EPSILON",
        "ETA_BOUNDARY",
        "ALPHA_0",
        "BETA_0",
        "LAX_DISTRIBUTIVE_DELTA",
        "PARALLEL_COMPOSITION_DELTA",
    )
    return {name: getattr(core_config, name) for name in names}


def _state_values(state) -> dict[str, Any]:
    return {
        "goal_names": list(state.schema.goal_names),
        "modulator_names": list(state.schema.modulator_names),
        "G": state.G.tolist(),
        "M": state.M.tolist(),
    }


def _transition_contract(
    *,
    name: str,
    state,
    stimulus,
    candidates,
    appraisal_profile,
    decision_profile,
) -> dict[str, Any]:
    appraisal = OpenPsiAppraisal(profile=appraisal_profile)
    decision = MagusDecision(profile=decision_profile)
    bimonad = MetaMoPseudoBimonad(appraisal, decision)
    features = appraisal_profile.stimulus_features(state, stimulus)
    rule_state, feedback = appraisal.appraise_with_feedback(state, stimulus)
    trace = bimonad.compute_transition_trace(state, stimulus, candidates)
    action, final_state, diagnostics = bimonad.step_with_diagnostics(
        state,
        stimulus,
        candidates,
        record_diagnostics=False,
    )
    return {
        "name": name,
        "source_state": _state_values(state),
        "features": features.as_dict(),
        "rule_state": _state_values(rule_state),
        "goal_change_feedback": feedback.as_dict(),
        "candidate_ids": [candidate.id for candidate in candidates],
        "candidate_scores": [
            asdict(score) for score in trace.decision_result.candidate_scores
        ],
        "chosen_action": action.id,
        "proposed_delta_g": trace.proposed_delta_g.tolist(),
        "damped_delta_g": trace.damped_delta_g.tolist(),
        "projected_state": _state_values(trace.projected_state),
        "final_state": _state_values(final_state),
        "diagnostics": asdict(diagnostics),
    }


def deterministic_contracts() -> dict[str, Any]:
    env = GridWorld(seed=1729, max_steps=25, danger_mineral_probability=0.55)
    env_state = env.reset()
    grid_state = GRIDWORLD_PROFILE.create_initial_state()
    grid_stimulus = GRIDWORLD_PROFILE.build_stimulus(env_state, grid_state)
    grid_candidates = GRIDWORLD_PROFILE.build_candidates(env_state, grid_state)

    return {
        "gridworld": _transition_contract(
            name="gridworld_rule_baseline_v1",
            state=grid_state,
            stimulus=grid_stimulus,
            candidates=grid_candidates,
            appraisal_profile=GRIDWORLD_APPRAISAL_PROFILE,
            decision_profile=GRIDWORLD_DECISION_PROFILE,
        ),
    }


def _run_logged(command: list[str], log_path: Path) -> None:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("COMMAND\n")
        handle.write(" ".join(command))
        handle.write("\n\nSTDOUT\n")
        handle.write(result.stdout)
        handle.write("\nSTDERR\n")
        handle.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}; see {log_path}"
        )


def _snapshot_checksums(output_dir: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name == "checksums.json":
            continue
        relative = path.relative_to(output_dir).as_posix()
        checksums[relative] = sha256(path.read_bytes()).hexdigest()
    return checksums


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        help=(
            "new snapshot directory; defaults to a UTC timestamp under "
            "experiment/results"
        ),
    )
    parser.add_argument("--train-episodes", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--agent-seeds", default="0-4")
    parser.add_argument("--regimes", default="low,medium,high")
    parser.add_argument("--variants", default="BaselineCompactQ,MetaMo")
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--test-seed-start", type=int, default=2000)
    parser.add_argument("--danger-distance", type=int, default=2)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-gridworld", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    started = datetime.now(timezone.utc)
    default_name = f"baseline-{started.strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "experiment" / "results" / default_name
    )
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if output_dir.exists():
        raise FileExistsError(
            f"baseline output already exists and will not be overwritten: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    fingerprint, source_files = _source_fingerprint()
    manifest = {
        "format_version": 1,
        "status": "running",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": None,
        "git_commit": _run_git("rev-parse", "HEAD"),
        "git_status": _run_git("status", "--short"),
        "source_sha256": fingerprint,
        "source_files": source_files,
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": _dependency_versions(),
        "core_config": _numeric_core_config(),
        "arguments": vars(args),
        "output_directory": str(output_dir),
        "commands": {},
    }
    _write_json(output_dir / "manifest.json", manifest)

    try:
        _write_json(output_dir / "deterministic_contracts.json", deterministic_contracts())

        if not args.skip_tests:
            test_command = [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-v",
            ]
            manifest["commands"]["unit_tests"] = test_command
            _run_logged(test_command, output_dir / "unit_tests.log")

        if not args.skip_gridworld:
            gridworld_dir = output_dir / "gridworld"
            gridworld_command = [
                sys.executable,
                "-m",
                "applications.gridworld.evaluation.runner",
                "--train-episodes",
                str(args.train_episodes),
                "--eval-episodes",
                str(args.eval_episodes),
                "--max-steps",
                str(args.max_steps),
                "--agent-seeds",
                args.agent_seeds,
                "--regimes",
                args.regimes,
                "--variants",
                args.variants,
                "--train-seed-start",
                str(args.train_seed_start),
                "--test-seed-start",
                str(args.test_seed_start),
                "--danger-distance",
                str(args.danger_distance),
                "--output-dir",
                str(gridworld_dir),
                "--quiet-seeds",
            ]
            manifest["commands"]["gridworld"] = gridworld_command
            _run_logged(gridworld_command, output_dir / "gridworld.log")

        manifest["status"] = "complete"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(output_dir / "manifest.json", manifest)
        _write_json(output_dir / "checksums.json", _snapshot_checksums(output_dir))
    except Exception as error:
        manifest["status"] = "failed"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = f"{type(error).__name__}: {error}"
        _write_json(output_dir / "manifest.json", manifest)
        _write_json(output_dir / "checksums.json", _snapshot_checksums(output_dir))
        raise

    return output_dir


def main(argv: list[str] | None = None) -> None:
    output_dir = run(parse_args(argv))
    print(f"Baseline snapshot complete: {output_dir}")


if __name__ == "__main__":
    main()
