"""Evaluate saved neural forecasts on fresh standard-GridWorld episodes.

No forecast model is fitted or selected here. Reconstruct the original frozen
Q policy by replaying its seeded pretraining, verify one recorded training
episode, then collect new episodes. All forecasters see the same frozen inputs.
This tests new layouts from the original distribution, not arbitrary actions
or a changed environment. Inference uses saved arrays and needs only NumPy.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from applications.gridworld.agents.metamo import MetaMoAgent
from applications.gridworld.config import ACTION_IDS, hazard_pressure
from applications.gridworld.environment import GridWorld
from applications.gridworld.evaluation.neural_self_model import (
    COORDINATES, DIM, FEATURES, GOALS, RULES, Dataset, NeuralSettings, Scaling,
    _check_frozen_q, _collect, _pretrain, metrics, rule_predictions,
)


@dataclass(frozen=True)
class RobustnessSettings:
    episodes_per_split: int = 60
    seed_start: int = 8000000
    seed_stride: int = 100000
    large_change_quantile: float = 0.90
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 8300
    minimum_group_episodes: int = 10

    def validate(self):
        if (self.episodes_per_split < 1 or self.seed_stride < self.episodes_per_split
                or not 0 < self.large_change_quantile < 1
                or self.bootstrap_samples < 1 or self.minimum_group_episodes < 2):
            raise ValueError("invalid robustness settings")


def load_dataset(path: Path) -> Dataset:
    with np.load(path, allow_pickle=False) as saved:
        return Dataset(**{name: saved[name].copy() for name in Dataset.__dataclass_fields__})


def subset(data: Dataset, mask: np.ndarray) -> Dataset:
    return Dataset(**{name: getattr(data, name)[mask] for name in Dataset.__dataclass_fields__})


def row_balanced_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    error = np.abs(prediction - target)
    return 0.5 * (error[:, :len(GOALS)].mean(axis=1) + error[:, len(GOALS):].mean(axis=1))


def situation_masks(data: Dataset, threshold: float) -> dict[str, np.ndarray]:
    """Large-change membership is retrospective, never a prediction input."""
    collection = data.features[:, FEATURES.index("move.mineral_collected")] > 0.5
    near_lava = (
        np.maximum(data.features[:, FEATURES.index("stimulus.hazard_pressure")],
                   data.features[:, FEATURES.index("move.hazard_risk")]) >= hazard_pressure(2)
    )
    return {
        "all": np.ones(len(data), dtype=bool),
        "ordinary": ~collection & ~near_lava,
        "collection": collection,
        "near_lava": near_lava,
        "large_change": row_balanced_error(data.target, data.current) >= threshold,
    }


class FrozenForecasters:
    """Read-only NumPy inference for the original 1-hidden-layer tanh network."""

    def __init__(self, path: Path, horizon: int):
        with np.load(path, allow_pickle=False) as saved:
            self.arrays = {name: saved[name].copy() for name in saved.files}
        for value in self.arrays.values():
            if not np.all(np.isfinite(value)):
                raise ValueError("nonfinite saved model")
            value.flags.writeable = False
        self.scaling = Scaling(**{name: self.arrays[name] for name in Scaling.__dataclass_fields__})
        self.horizon = horizon
        if (self.arrays["ridge"].shape != (len(FEATURES) + 1, DIM)
                or self.arrays["neural.0.weight"].shape[1] != len(FEATURES)
                or self.arrays["neural.2.weight"].shape[0] != DIM
                or np.any(self.scaling.input_scale <= 0)
                or np.any(self.scaling.residual_scale <= 0)):
            raise ValueError("incompatible saved model dimensions/scales")

    def predict(self, data: Dataset) -> dict[str, np.ndarray]:
        # Prediction never reads data.target, even though Dataset also carries labels.
        inputs = self.scaling.inputs(data)
        a = self.arrays
        hidden = np.tanh(inputs.astype(np.float32) @ a["neural.0.weight"].T + a["neural.0.bias"])
        residual = hidden @ a["neural.2.weight"].T + a["neural.2.bias"]
        predictions = rule_predictions(data, self.horizon)
        predictions["mean_residual"] = self.scaling.states(np.zeros_like(data.nominal), data.nominal)
        predictions["ridge"] = self.scaling.states(
            np.column_stack((inputs, np.ones(len(data)))) @ a["ridge"], data.nominal)
        predictions["neural_32"] = self.scaling.states(residual, data.nominal)
        if any(not np.all(np.isfinite(value)) for value in predictions.values()):
            raise ValueError("nonfinite prediction")
        return predictions


def paired_comparison(neural: np.ndarray, reference: np.ndarray, data: Dataset,
                      settings: RobustnessSettings, seed: int) -> dict:
    """Intervals concern mean episode error reduction, conditional on fixed models."""
    difference = row_balanced_error(reference, data.target) - row_balanced_error(neural, data.target)
    episode_means = np.asarray([difference[data.episode == episode].mean()
                                for episode in np.unique(data.episode)])
    interval = None
    if len(episode_means) >= 2:
        rng = np.random.default_rng(seed)
        sampled = episode_means[rng.integers(
            len(episode_means), size=(settings.bootstrap_samples, len(episode_means)))].mean(axis=1)
        interval = np.quantile(sampled, [0.025, 0.975]).tolist()
    coordinate_reduction = np.abs(reference - data.target).mean(axis=0) - np.abs(neural - data.target).mean(axis=0)
    return {
        "mean_episode_mae_reduction": float(episode_means.mean()),
        "bootstrap_95_percent_interval": interval,
        "episode_win_fraction": float(np.mean(episode_means > 0)),
        "window_worse_fraction": float(np.mean(difference < 0)),
        "coordinate_mae_reduction": dict(zip(COORDINATES, coordinate_reduction.tolist())),
        "coverage_sufficient": len(episode_means) >= settings.minimum_group_episodes,
    }


def group_report(data: Dataset, predictions: dict[str, np.ndarray], threshold: float,
                 selected_rule: str, settings: RobustnessSettings, seed: int, *,
                 focal_model: str = "neural_32", references: tuple[str, ...] | None = None) -> dict:
    groups = {}
    for index, (name, mask) in enumerate(situation_masks(data, threshold).items()):
        group = {"windows": int(mask.sum()), "episodes": int(len(np.unique(data.episode[mask])))}
        if not np.any(mask):
            groups[name] = {**group, "status": "no observations", "metrics": {}, "comparisons": {}}
            continue
        selected = subset(data, mask)
        scores = {}
        for model, prediction in predictions.items():
            errors = row_balanced_error(prediction[mask], selected.target)
            scores[model] = {
                **metrics(prediction[mask], selected),
                "p95_window_balanced_error": float(np.quantile(errors, 0.95)),
                "maximum_window_balanced_error": float(errors.max()),
            }
        groups[name] = {
            **group,
            "status": ("adequate episode coverage" if group["episodes"] >= settings.minimum_group_episodes
                       else "insufficient episode coverage"),
            "metrics": scores,
            "comparisons": {reference: paired_comparison(
                predictions[focal_model][mask], predictions[reference][mask], selected,
                settings, seed + index,
            ) for reference in (references if references is not None else (selected_rule, "ridge"))},
        }
    return groups


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _layout(seed: int, max_steps: int) -> str:
    env = GridWorld(seed=seed, max_steps=max_steps)
    env.reset()  # Match the constructor + explicit reset in the collector.
    return hashlib.sha256(repr(sorted(env.lava_cells)).encode()).hexdigest()


def _seed_plan(original: NeuralSettings, settings: RobustnessSettings) -> tuple[list[range], set[int]]:
    old_count = (original.pretrain_episodes + original.train_episodes
                 + original.validation_episodes + original.test_episodes)
    old = {original.seed_start + split * original.seed_stride + offset
           for split in range(original.splits) for offset in range(old_count)}
    ranges = [range(settings.seed_start + split * settings.seed_stride,
                    settings.seed_start + split * settings.seed_stride + settings.episodes_per_split)
              for split in range(original.splits)]
    fresh = [seed for seeds in ranges for seed in seeds]
    if len(set(fresh)) != len(fresh) or old.intersection(fresh):
        raise ValueError("fresh evaluation seeds overlap previous data or another split")
    return ranges, old


def run_robustness(source: Path, settings: RobustnessSettings = RobustnessSettings(), *,
                   output_dir: Path | None = None,
                   progress: Callable[[str], None] | None = None) -> dict:
    settings.validate()
    original = NeuralSettings(**json.loads((source / "settings.json").read_text(encoding="utf-8")))
    original.validate()
    ranges, old_seeds = _seed_plan(original, settings)
    if output_dir is not None:
        if output_dir.resolve() == source.resolve() or source.resolve() in output_dir.resolve().parents:
            raise ValueError("robustness outputs must be outside the original experiment directory")
        output_dir.mkdir(parents=True, exist_ok=True)
    old_layouts = {_layout(seed, original.max_steps) for seed in old_seeds}
    fresh_layouts = [[_layout(seed, original.max_steps) for seed in seeds] for seeds in ranges]
    flattened = [layout for layouts in fresh_layouts for layout in layouts]
    if len(set(flattened)) != len(flattened) or old_layouts.intersection(flattened):
        raise ValueError("evaluation reuses a lava layout; choose a new seed block before running")

    # Freeze settings, thresholds, selections and source fingerprints before collecting any new outcomes.
    prepared = []
    manifest = {
        "settings": asdict(settings), "original_settings": asdict(original),
        "source_settings_sha256": _fingerprint(source / "settings.json"),
        "groups": {
            "ordinary": "selected move does not collect; current and destination lava distances exceed 2",
            "collection": "selected first action collects a mineral (known before execution)",
            "near_lava": "current or selected destination cell is within Manhattan distance 2 of lava",
            "large_change": "actual balanced absolute state change >= training quantile threshold; retrospective only",
        },
        "limits": [
            "Groups overlap; subgroup results are descriptive, with no multiple-comparison correction.",
            "Fresh randomized layouts follow the original standard-GridWorld distribution.",
            "Only full forecast horizons are scored; final horizon-1 origins per episode are omitted.",
            "Intervals resample paired whole episodes with subgroup observations, conditional on frozen models.",
            "Aggregate MAE weights windows; interval point estimates weight episodes equally.",
            "No action intervention, counterfactual validation, or model selection uses these results.",
        ],
        "splits": [],
    }
    for index, seeds in enumerate(ranges):
        folder = source / f"split_{index + 1}"
        selection = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        selected_rule = selection["selected_rule"]
        if selected_rule not in RULES:
            raise ValueError("unrecognized previously selected rule")
        train = load_dataset(folder / "train.npz")
        threshold = float(np.quantile(row_balanced_error(train.target, train.current),
                                     settings.large_change_quantile))
        model = FrozenForecasters(folder / "model.npz", original.horizon)
        old_test = load_dataset(folder / "test.npz")
        with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
            for name, predicted in model.predict(old_test).items():
                np.testing.assert_allclose(predicted, saved[name], atol=1e-6, rtol=0,
                                           err_msg="saved inference no longer matches original test predictions")
        prepared.append((train, threshold, model, selected_rule))
        manifest["splits"].append({
            "fresh_seed_start": seeds.start, "fresh_seed_stop_exclusive": seeds.stop,
            "lava_layout_sha256": fresh_layouts[index], "selected_rule": selected_rule,
            "large_change_threshold": threshold,
            "source_sha256": {name: _fingerprint(folder / name) for name in
                              ("model.npz", "train.npz", "test.npz", "predictions.npz", "report.json")},
        })
    if output_dir is not None:
        (output_dir / "protocol.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    reports = []
    for index, (train, threshold, model, selected_rule) in enumerate(prepared):
        prefix = f"Split {index + 1}/{original.splits}"
        if progress:
            progress(f"{prefix}: reconstructing frozen Q policy; saved forecast models remain unchanged")
        base = original.seed_start + index * original.seed_stride
        agent = MetaMoAgent(seed=original.agent_seed + index)
        _pretrain(agent, range(base, base + original.pretrain_episodes), original)
        agent.epsilon = 0.0
        agent.record_transitions = True
        agent.predictive_self_model.learning_enabled = False
        q_before, visits_before = copy.deepcopy(agent.q_table), copy.deepcopy(agent.visit_counts)
        first_seed = base + original.pretrain_episodes
        check = _collect(agent, range(first_seed, first_seed + 1), original, None, "replay check")
        expected = subset(train, train.episode == first_seed)
        for name in Dataset.__dataclass_fields__:
            np.testing.assert_array_equal(getattr(check, name), getattr(expected, name),
                                          err_msg="policy/feature replay differs from original experiment")
        # Start a fresh reproducible tie-breaking stream; Q and policy rules stay fixed.
        agent.rng.seed(settings.seed_start + index)
        data = _collect(agent, ranges[index], original, progress, f"{prefix} fresh test")
        _check_frozen_q(agent, q_before, visits_before)
        if agent.predictive_self_model.collection_observation_count or any(
                agent.predictive_self_model.sample_count(action) for action in ACTION_IDS):
            raise RuntimeError("existing self-model learned during evaluation")
        prediction = model.predict(data)
        groups = group_report(data, prediction, threshold, selected_rule, settings,
                              settings.bootstrap_seed + index * 100)
        report = {
            "split": index + 1, "selected_rule": selected_rule, "groups": groups,
            "original_training_episode_replayed_exactly": True,
            "saved_inference_matches_original_test": True,
            "live_behavior_unchanged": True, "q_and_existing_self_model_frozen": True,
        }
        reports.append(report)
        if output_dir is not None:
            folder = output_dir / f"split_{index + 1}"
            folder.mkdir(exist_ok=True)
            data.save(folder / "fresh.npz")
            np.savez_compressed(folder / "predictions.npz", **prediction)
            np.savez_compressed(folder / "groups.npz", **situation_masks(data, threshold))
            (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if progress:
            scores = groups["all"]["metrics"]
            progress(f"{prefix}: rule={scores[selected_rule]['balanced_mae']:.6f}, "
                     f"ridge={scores['ridge']['balanced_mae']:.6f}, NN={scores['neural_32']['balanced_mae']:.6f}")

    # Descriptive pooled metrics, not a claim of independent prediction windows.
    aggregate = {}
    for group in reports[0]["groups"]:
        rows = [report["groups"][group] for report in reports]
        populated = [row for row in rows if row["windows"]]
        models = {}
        for name in (*RULES, "mean_residual", "ridge", "neural_32"):
            if not populated:
                continue
            models[name] = {
                metric: float(np.average([row["metrics"][name][metric] for row in populated],
                                         weights=[row["windows"] for row in populated]))
                for metric in ("balanced_mae", "goal_mae", "modulator_mae")
            }
            models[name]["coordinate_mae"] = {
                coordinate: float(np.average(
                    [row["metrics"][name]["coordinate_mae"][coordinate] for row in populated],
                    weights=[row["windows"] for row in populated])) for coordinate in COORDINATES
            }
        aggregate[group] = {"windows": sum(row["windows"] for row in rows),
                            "episodes": sum(row["episodes"] for row in rows), "metrics": models}
    for index, record in enumerate(manifest["splits"]):
        for name, expected_hash in record["source_sha256"].items():
            if _fingerprint(source / f"split_{index + 1}" / name) != expected_hash:
                raise RuntimeError("original experiment artifact changed during robustness evaluation")
    return {
        "protocol": manifest, "coordinate_names": COORDINATES, "feature_names": FEATURES,
        "splits": reports, "aggregate_groups": aggregate,
        "saved_artifacts_unchanged": True, "fresh_lava_layouts_verified": len(flattened),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/self_model/neural_five_step"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/self_model/neural_robustness"))
    parser.add_argument("--summary", type=Path, default=Path("artifacts/self_model/neural_robustness.summary.json"))
    parser.add_argument("--episodes-per-split", type=int, default=60)
    args = parser.parse_args()
    report = run_robustness(args.source, RobustnessSettings(episodes_per_split=args.episodes_per_split),
                            output_dir=args.output_dir, progress=lambda text: print(text, flush=True))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, group in report["aggregate_groups"].items():
        print(f"{name}: {group['windows']} windows, {group['episodes']} episodes")
        for model in ("nominal_trend", "ridge", "neural_32"):
            if group["metrics"]:
                print(f"  {model}: {group['metrics'][model]['balanced_mae']:.6f}")
    print(f"Summary: {args.summary}; raw artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
