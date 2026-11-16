"""Controlled large-change weighting experiment; forecasting only, no intervention.

Reuse original train/validation episodes, scaling, architecture, optimizer,
initializations and epoch-selection criterion. Change only training row weights.
Validation chooses a weight under explicit ordinary/overall degradation limits.
All selections are saved before generating new test episodes. Prior robustness
outcomes motivated this hypothesis but are never used for fitting or selection.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from applications.gridworld.evaluation.neural_self_model import (
    COORDINATES, NeuralSettings, _torch, fit_neural, neural_predict,
)
from applications.gridworld.evaluation.neural_robustness import (
    FrozenForecasters, RobustnessSettings, _fingerprint, _layout, _seed_plan,
    group_report, load_dataset, row_balanced_error, run_robustness, situation_masks,
)


@dataclass(frozen=True)
class WeightingSettings:
    strengths: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    ordinary_tolerance: float = 0.02
    overall_tolerance: float = 0.02

    def validate(self):
        if (1.0 not in self.strengths or len(set(self.strengths)) != len(self.strengths)
                or any(not np.isfinite(value) or value < 1 for value in self.strengths)
                or any(not np.isfinite(value) or value < 0 for value in
                       (self.ordinary_tolerance, self.overall_tolerance))):
            raise ValueError("weights must include 1, be unique and >=1; tolerances must be nonnegative")


def training_weights(train, threshold: float, strength: float) -> np.ndarray:
    if not np.isfinite(strength) or strength < 1 or not np.isfinite(threshold):
        raise ValueError("invalid weighting strength or threshold")
    large = row_balanced_error(train.target, train.current) >= threshold
    return np.where(large, strength, 1.0)


def validation_scores(prediction, data, threshold: float) -> dict:
    errors = row_balanced_error(prediction, data.target)
    masks = situation_masks(data, threshold)
    if any(not np.any(masks[name]) for name in ("all", "ordinary", "large_change")):
        raise ValueError("validation needs ordinary and large-change examples")
    return {name: float(errors[masks[name]].mean()) for name in ("all", "ordinary", "large_change")}


def choose_strength(candidates: list[dict], settings: WeightingSettings) -> float:
    """Choose using validation only; the unweighted control is always eligible."""
    baseline = next(row["validation"] for row in candidates if row["strength"] == 1.0)
    eligible = [row for row in candidates
                if row["validation"]["ordinary"] <= baseline["ordinary"] * (1 + settings.ordinary_tolerance)
                and row["validation"]["all"] <= baseline["all"] * (1 + settings.overall_tolerance)]
    return min(eligible, key=lambda row: (row["validation"]["large_change"], row["strength"]))["strength"]


def point_estimate_success(selected: dict, reference: dict, settings: WeightingSettings) -> bool:
    return bool(selected["large_change"] < reference["large_change"]
                and selected["ordinary"] <= reference["ordinary"] * (1 + settings.ordinary_tolerance)
                and selected["all"] <= reference["all"] * (1 + settings.overall_tolerance))


def _write(path: Path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _fresh_again(original, evaluation, previous):
    ranges, _ = _seed_plan(original, evaluation)
    prior_seeds = {seed for split in previous["splits"]
                   for seed in range(split["fresh_seed_start"], split["fresh_seed_stop_exclusive"])}
    prior_layouts = {layout for split in previous["splits"] for layout in split["lava_layout_sha256"]}
    for seeds in ranges:
        if prior_seeds.intersection(seeds):
            raise ValueError("test seeds overlap the preceding robustness experiment")
        if any(_layout(seed, original.max_steps) in prior_layouts for seed in seeds):
            raise ValueError("test lava layout overlaps the preceding robustness experiment")


def run_weighting(source: Path, previous_protocol: Path, output_dir: Path, *,
                  settings: WeightingSettings = WeightingSettings(),
                  evaluation: RobustnessSettings = RobustnessSettings(seed_start=9000000, bootstrap_seed=9300),
                  progress: Callable[[str], None] | None = None) -> dict:
    settings.validate()
    evaluation.validate()
    original = NeuralSettings(**json.loads((source / "settings.json").read_text(encoding="utf-8")))
    original.validate()
    previous = json.loads(previous_protocol.read_text(encoding="utf-8"))
    # Refuse writing inside either earlier experiment's artifact directory.
    for protected in (source, previous_protocol.parent):
        if output_dir.resolve() == protected.resolve() or protected.resolve() in output_dir.resolve().parents:
            raise ValueError("weighted outputs must be outside earlier experiment directories")
    _fresh_again(original, evaluation, previous)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {"settings.json": _fingerprint(source / "settings.json")}
    for index in range(original.splits):
        for name in ("train.npz", "validation.npz", "model.npz", "report.json"):
            relative = f"split_{index + 1}/{name}"
            source_hashes[relative] = _fingerprint(source / relative)
    protocol = {
        "weighting": asdict(settings), "evaluation": asdict(evaluation),
        "original_settings": asdict(original), "source_sha256": source_hashes,
        "previous_protocol_sha256": _fingerprint(previous_protocol),
        "training": "weight per-row standardized residual MSE; normalize weights by their training mean",
        "large_change": "balanced absolute actual state change >= original training 90th percentile (default)",
        "epoch_and_initialization_selection": "unchanged: minimize overall validation balanced MAE",
        "strength_selection": "minimize validation large-change MAE subject to ordinary and overall tolerances vs weight 1",
        "test_point_estimate_gate": "large-change MAE improves vs original NN; ordinary and overall within tolerances in every split",
        "limits": [
            "Fresh test confirms a hypothesis motivated by earlier test results; earlier tests are not reused.",
            "Large-change membership uses outcomes only for training weights and retrospective evaluation, never inference inputs.",
            "Subgroups overlap. Intervals resample paired episodes; exploratory comparisons are not multiplicity-adjusted.",
            "Test selection is forbidden: nonselected weight results remain diagnostic.",
            "Forecasts are on-policy, standard GridWorld, full horizons only; no decision-benefit claim.",
        ],
    }
    _write(output_dir / "protocol.json", protocol)
    torch = _torch()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _run(source, output_dir, original, settings, evaluation, protocol, progress, torch)
    finally:
        torch.set_num_threads(old_threads)


def _run(source, output_dir, original, settings, evaluation, protocol, progress, torch):
    prepared, selections = [], []
    for index in range(original.splits):
        folder = source / f"split_{index + 1}"
        destination = output_dir / f"split_{index + 1}"
        destination.mkdir(exist_ok=True)
        train = load_dataset(folder / "train.npz")
        validation = load_dataset(folder / "validation.npz")
        old_report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        frozen = FrozenForecasters(folder / "model.npz", original.horizon)
        threshold = float(np.quantile(row_balanced_error(train.target, train.current),
                                     evaluation.large_change_quantile))
        candidates, models = [], {}
        for strength in sorted(settings.strengths):
            fits = []
            weights = training_weights(train, threshold, strength)
            for initialization in range(original.initializations):
                seed = original.network_seed + index * 100 + initialization
                network, selection = fit_neural(train, validation, frozen.scaling, original, seed,
                                                sample_weights=weights)
                fits.append((network, selection))
            network, selected = min(fits, key=lambda pair: pair[1]["validation_balanced_mae"])
            prediction = neural_predict(network, validation, frozen.scaling)
            scores = validation_scores(prediction, validation, threshold)
            if strength == 1.0:
                np.testing.assert_allclose(prediction, frozen.predict(validation)["neural_32"], atol=1e-6, rtol=0,
                                           err_msg="unweighted control no longer reproduces saved NN")
                if (selected["seed"] != old_report["neural_selection"]["seed"]
                        or selected["best_epoch"] != old_report["neural_selection"]["best_epoch"]):
                    raise RuntimeError("unweighted checkpoint selection changed")
            candidates.append({"strength": strength, "validation": scores, "selection": selected,
                               "initializations": [fit[1] for fit in fits]})
            models[strength] = network
            # Every candidate is saved before fresh test outcomes exist.
            np.savez_compressed(destination / f"weight_{strength:g}.npz", **frozen.scaling.__dict__,
                                ridge=frozen.arrays["ridge"],
                                **{f"neural.{name}": value.detach().numpy()
                                   for name, value in network.state_dict().items()})
            if progress:
                progress(f"Split {index + 1}/{original.splits}, weight {strength:g}: "
                         f"validation large-change={scores['large_change']:.6f}, "
                         f"ordinary={scores['ordinary']:.6f}, overall={scores['all']:.6f}")
        strength = choose_strength(candidates, settings)
        selection = {"split": index + 1, "selected_strength": strength,
                     "large_change_threshold": threshold,
                     "training_large_change_windows": int(np.sum(training_weights(train, threshold, 2) == 2)),
                     "validation_group_counts": {name: int(mask.sum()) for name, mask in
                                                  situation_masks(validation, threshold).items()},
                     "unweighted_control_reproduced": True, "candidates": candidates}
        _write(destination / "selection.json", selection)
        selections.append(selection)
        prepared.append((models, frozen.scaling, threshold, old_report["selected_rule"], strength))
        if progress:
            progress(f"Split {index + 1}: selected weight {strength:g} using validation only")
    _write(output_dir / "selections.json", selections)

    # Reuse the verified mirror collector and seed/layout checks. All models
    # and choices above are frozen before even generating these observations.
    reference = run_robustness(source, evaluation, output_dir=output_dir / "fresh_reference", progress=progress)
    reports = []
    for index, (models, scaling, threshold, selected_rule, strength) in enumerate(prepared):
        destination = output_dir / f"split_{index + 1}"
        data = load_dataset(output_dir / "fresh_reference" / f"split_{index + 1}" / "fresh.npz")
        with np.load(output_dir / "fresh_reference" / f"split_{index + 1}" / "predictions.npz",
                     allow_pickle=False) as saved:
            predictions = {name: saved[name].copy() for name in saved.files}
        for weight, network in models.items():
            predictions[f"weight_{weight:g}"] = neural_predict(network, data, scaling)
        np.testing.assert_allclose(predictions["weight_1"], predictions["neural_32"], atol=1e-6, rtol=0)
        # Avoid reporting roundoff between Torch and NumPy as a weight-1 gain.
        predictions["weight_1"] = predictions["neural_32"]
        predictions["weighted_selected"] = predictions[f"weight_{strength:g}"]
        groups = group_report(data, predictions, threshold, selected_rule, evaluation,
                              evaluation.bootstrap_seed + index * 100, focal_model="weighted_selected",
                              references=("neural_32", "ridge", selected_rule))
        def scores(model):
            return {group: groups[group]["metrics"][model]["balanced_mae"]
                    for group in ("all", "ordinary", "large_change")}
        if any(not groups[group]["windows"] for group in ("ordinary", "large_change")):
            success = None
            beats_ridge = None
        else:
            success = point_estimate_success(scores("weighted_selected"), scores("neural_32"), settings)
            beats_ridge = scores("weighted_selected")["large_change"] < scores("ridge")["large_change"]
        report = {"split": index + 1, "selected_strength": strength, "groups": groups,
                  "point_estimate_success_vs_original_nn": success,
                  "large_change_point_estimate_beats_ridge": beats_ridge}
        np.savez_compressed(destination / "predictions.npz", **predictions)
        _write(destination / "report.json", report)
        # Raw reports retain per-episode numbers; keep the reviewed summary compact.
        for group in groups.values():
            for metrics in group["metrics"].values():
                metrics.pop("episode_balanced_mae", None)
        reports.append(report)
        if progress and success is not None:
            progress(f"Split {index + 1}: test large-change error original="
                     f"{scores('neural_32')['large_change']:.6f}, selected="
                     f"{scores('weighted_selected')['large_change']:.6f}, "
                     f"ridge={scores('ridge')['large_change']:.6f}; point-estimate gate={success}")

    aggregate = {}
    for name in reports[0]["groups"]:
        rows = [report["groups"][name] for report in reports]
        populated = [row for row in rows if row["windows"]]
        aggregate[name] = {"windows": sum(row["windows"] for row in rows),
                           "episodes": sum(row["episodes"] for row in rows), "metrics": {}}
        if populated:
            for model in populated[0]["metrics"]:
                aggregate[name]["metrics"][model] = {
                    metric: float(np.average([row["metrics"][model][metric] for row in populated],
                                             weights=[row["windows"] for row in populated]))
                    for metric in ("balanced_mae", "goal_mae", "modulator_mae")}
    for relative, expected in protocol["source_sha256"].items():
        if _fingerprint(source / relative) != expected:
            raise RuntimeError("original artifact changed during weighted experiment")
    return {
        "protocol": protocol, "coordinate_names": COORDINATES,
        "selections": [{**selection, "candidates": [
            {key: value for key, value in candidate.items() if key != "initializations"}
            for candidate in selection["candidates"]]} for selection in selections],
        "splits": reports, "aggregate_groups": aggregate,
        "point_estimate_gate_passed_in_every_split": all(
            report["point_estimate_success_vs_original_nn"] is True for report in reports),
        "original_artifacts_unchanged": True,
        "fresh_lava_layouts_verified": reference["fresh_lava_layouts_verified"],
        "live_behavior_unchanged": all(report["live_behavior_unchanged"] for report in reference["splits"]),
        "q_and_existing_self_model_frozen": all(
            report["q_and_existing_self_model_frozen"] for report in reference["splits"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/self_model/neural_five_step"))
    parser.add_argument("--previous-protocol", type=Path,
                        default=Path("artifacts/self_model/neural_robustness/protocol.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/self_model/neural_weighted"))
    parser.add_argument("--summary", type=Path, default=Path("artifacts/self_model/neural_weighted.summary.json"))
    parser.add_argument("--episodes-per-split", type=int, default=60)
    args = parser.parse_args()
    report = run_weighting(args.source, args.previous_protocol, args.output_dir,
                           evaluation=RobustnessSettings(episodes_per_split=args.episodes_per_split,
                                                         seed_start=9000000, bootstrap_seed=9300),
                           progress=lambda message: print(message, flush=True))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    _write(args.summary, report)
    for name, group in report["aggregate_groups"].items():
        print(name, {model: group["metrics"].get(model, {}).get("balanced_mae")
                     for model in ("neural_32", "weighted_selected", "ridge")})
    print(f"Point-estimate gate passed in every split: {report['point_estimate_gate_passed_in_every_split']}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
