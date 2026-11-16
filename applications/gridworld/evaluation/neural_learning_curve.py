"""Nested-data learning curves for the existing unweighted neural self-model.

Reuse the original 80 training episodes and extend to 160/320. Reconstruct the
same frozen Q policy; generate fresh shared validation/test episodes. Train-only
scaling is refitted for each size. Validation chooses NN checkpoints/seeds and
ridge penalties; all choices precede test collection. No live intervention.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from applications.gridworld.agents.metamo import MetaMoAgent
from applications.gridworld.config import ACTION_IDS
from applications.gridworld.evaluation.neural_self_model import (
    COORDINATES, FEATURES, RULES, Dataset, NeuralSettings, Scaling,
    _check_frozen_q, _collect, _combine, _pretrain, _torch, balanced_mae,
    fit_neural, fit_ridge, neural_predict, rule_predictions,
)
from applications.gridworld.evaluation.neural_robustness import (
    RobustnessSettings, _fingerprint, _layout, group_report, load_dataset,
    paired_comparison, row_balanced_error, situation_masks, subset,
)


@dataclass(frozen=True)
class CurveSettings:
    train_sizes: tuple[int, ...] = (80, 160, 320)
    validation_episodes: int = 40
    test_episodes: int = 60
    seed_start: int = 14000000
    seed_stride: int = 100000
    role_stride: int = 10000
    large_change_quantile: float = 0.90
    ordinary_tolerance: float = 0.02
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 14300

    def validate(self, original: NeuralSettings):
        if (len(self.train_sizes) < 2 or any(type(value) is not int for value in self.train_sizes)
                or tuple(sorted(set(self.train_sizes))) != tuple(self.train_sizes)
                or self.train_sizes[0] != original.train_episodes
                or min(self.validation_episodes, self.test_episodes, self.bootstrap_samples) < 1
                or self.role_stride < max(self.train_sizes[-1] - self.train_sizes[0],
                                          self.validation_episodes, self.test_episodes)
                or self.seed_stride < 3 * self.role_stride
                or not 0 < self.large_change_quantile < 1
                or not np.isfinite(self.ordinary_tolerance) or self.ordinary_tolerance < 0):
            raise ValueError("invalid curve sizes, disjoint seed ranges, or evaluation settings")


def nested_training(base: Dataset, extra: Dataset, size: int) -> Dataset:
    """Extend by complete episodes in ascending seed order; never split rows randomly."""
    base_episodes, extra_episodes = np.unique(base.episode), np.unique(extra.episode)
    if np.intersect1d(base_episodes, extra_episodes).size:
        raise ValueError("base and additional training episodes overlap")
    if size < len(base_episodes) or size > len(base_episodes) + len(extra_episodes):
        raise ValueError("requested training size unavailable")
    if size == len(base_episodes):
        return base
    return _combine([base, subset(extra, np.isin(extra.episode, extra_episodes[:size - len(base_episodes)]))])


def seed_plan(original: NeuralSettings, settings: CurveSettings) -> list[dict[str, range]]:
    settings.validate(original)
    return [{
        "extra": range(base, base + settings.train_sizes[-1] - settings.train_sizes[0]),
        "validation": range(base + settings.role_stride,
                            base + settings.role_stride + settings.validation_episodes),
        "test": range(base + 2 * settings.role_stride,
                      base + 2 * settings.role_stride + settings.test_episodes),
    } for base in (settings.seed_start + index * settings.seed_stride for index in range(original.splits))]


def check_freshness(original, plans, previous_protocols):
    old_count = original.pretrain_episodes + original.train_episodes + original.validation_episodes + original.test_episodes
    old_seeds = {original.seed_start + split * original.seed_stride + offset
                 for split in range(original.splits) for offset in range(old_count)}
    for protocol in previous_protocols:
        for split in protocol["splits"]:
            old_seeds.update(range(split["fresh_seed_start"], split["fresh_seed_stop_exclusive"]))
    fresh = [seed for plan in plans for seeds in plan.values() for seed in seeds]
    if len(set(fresh)) != len(fresh) or old_seeds.intersection(fresh):
        raise ValueError("new data overlap an earlier experiment or another role/split")
    old_layouts = {_layout(seed, original.max_steps) for seed in old_seeds}
    layouts = {_seed: _layout(_seed, original.max_steps) for _seed in fresh}
    if len(set(layouts.values())) != len(layouts) or old_layouts.intersection(layouts.values()):
        raise ValueError("new data reuse a lava layout")
    return layouts


def improvement_gate(groups: dict, smallest: int, largest: int, tolerance: float) -> dict:
    if any(not groups[name]["windows"] for name in ("all", "ordinary", "large_change")):
        return {"more_data_helps": None, "largest_nn_beats_matched_ridge_on_large_changes": None}
    def score(group, model):
        return groups[group]["metrics"][model]["balanced_mae"]
    small, large = f"neural_{smallest}", f"neural_{largest}"
    return {
        "more_data_helps": bool(score("large_change", large) < score("large_change", small)
                                and all(score(group, large) <= (1 + tolerance) * score(group, small)
                                        for group in ("ordinary", "all"))),
        "largest_nn_beats_matched_ridge_on_large_changes": bool(
            score("large_change", large) < score("large_change", f"ridge_{largest}")),
    }


def _write(path: Path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_learning_curve(source: Path, previous_protocols: tuple[Path, ...], output_dir: Path, *,
                       settings: CurveSettings = CurveSettings(),
                       progress: Callable[[str], None] | None = None) -> dict:
    original = NeuralSettings(**json.loads((source / "settings.json").read_text(encoding="utf-8")))
    original.validate()
    settings.validate(original)
    for protected in (source, *(path.parent for path in previous_protocols)):
        if output_dir.resolve() == protected.resolve() or protected.resolve() in output_dir.resolve().parents:
            raise ValueError("curve outputs must be outside previous experiment directories")
    plans = seed_plan(original, settings)
    previous = [json.loads(path.read_text(encoding="utf-8")) for path in previous_protocols]
    layouts = check_freshness(original, plans, previous)
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {"settings.json": _fingerprint(source / "settings.json"),
              **{f"split_{index + 1}/train.npz": _fingerprint(source / f"split_{index + 1}" / "train.npz")
                 for index in range(original.splits)}}
    protocol = {
        "settings": asdict(settings), "original_settings": asdict(original), "source_sha256": hashes,
        "previous_protocol_sha256": {str(path): _fingerprint(path) for path in previous_protocols},
        "seed_plan": [{name: {"start": seeds.start, "stop_exclusive": seeds.stop}
                       for name, seeds in plan.items()} for plan in plans],
        "new_lava_layout_sha256": layouts,
        "training": "unweighted normalized residual MSE; identical architecture, optimizer, initialization seeds and epoch budget",
        "validation": "new shared episodes; select rule once, ridge penalty and NN epoch/initialization separately at each size",
        "large_change": "threshold from original smallest training set, frozen for all sizes within each split",
        "point_estimate_gate": "largest NN improves large-change MAE over smallest NN; ordinary/overall degrade <=2% by default",
        "limits": [
            "More episodes also permit more optimizer updates per epoch; this is not a fixed-compute study.",
            "One nested training sequence per policy split; no claim that data alone limit accuracy.",
            "All sizes use the same fresh test episodes; none are selected using test error.",
            "Paired bootstrap resamples episodes; subgroup comparisons are exploratory and unadjusted.",
            "Standard GridWorld, on-policy, complete horizons only; no action intervention.",
        ],
    }
    _write(output_dir / "protocol.json", protocol)
    torch = _torch()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _run(source, output_dir, original, settings, plans, protocol, progress)
    finally:
        torch.set_num_threads(threads)


def _run(source, output_dir, original, settings, plans, protocol, progress):
    prepared, selections = [], []
    for index, plan in enumerate(plans):
        prefix = f"Split {index + 1}/{original.splits}"
        folder = output_dir / f"split_{index + 1}"
        folder.mkdir(exist_ok=True)
        base_data = load_dataset(source / f"split_{index + 1}" / "train.npz")
        if len(np.unique(base_data.episode)) != settings.train_sizes[0]:
            raise ValueError("base dataset does not match smallest training size")
        threshold = float(np.quantile(row_balanced_error(base_data.target, base_data.current),
                                     settings.large_change_quantile))
        if progress:
            progress(f"{prefix}: reconstructing frozen policy and extending training data")
        agent = MetaMoAgent(seed=original.agent_seed + index)
        base = original.seed_start + index * original.seed_stride
        _pretrain(agent, range(base, base + original.pretrain_episodes), original)
        agent.epsilon = 0.0
        agent.record_transitions = True
        agent.predictive_self_model.learning_enabled = False
        q_before, visits_before = copy.deepcopy(agent.q_table), copy.deepcopy(agent.visit_counts)
        first = base + original.pretrain_episodes
        replay = _collect(agent, range(first, first + 1), original, None, "policy replay")
        expected = subset(base_data, base_data.episode == first)
        for name in Dataset.__dataclass_fields__:
            np.testing.assert_array_equal(getattr(replay, name), getattr(expected, name))
        agent.rng.seed(plan["extra"].start)
        extra = _collect(agent, plan["extra"], original, progress, f"{prefix} extra training")
        extra.save(folder / "extra_training.npz")
        agent.rng.seed(plan["validation"].start)
        validation = _collect(agent, plan["validation"], original, progress, f"{prefix} shared validation")
        validation.save(folder / "validation.npz")
        selected_rule = min(RULES, key=lambda name: balanced_mae(
            rule_predictions(validation, original.horizon)[name], validation.target))
        models, rows = {}, []
        for size in settings.train_sizes:
            train = nested_training(base_data, extra, size)
            if set(train.episode).intersection(validation.episode):
                raise RuntimeError("training/validation leakage")
            scaling = Scaling.fit(train)
            ridge, ridge_selection = fit_ridge(train, validation, scaling, original.ridge_penalties)
            fits = []
            for initialization in range(original.initializations):
                seed = original.network_seed + index * 100 + initialization
                network, selection = fit_neural(train, validation, scaling, original, seed)
                fits.append((network, selection))
                if progress:
                    progress(f"{prefix}, {size} episodes, NN seed {seed}: validation MAE "
                             f"{selection['validation_balanced_mae']:.6f}, epoch {selection['best_epoch']}")
            network, selected = min(fits, key=lambda pair: pair[1]["validation_balanced_mae"])
            models[size] = (network, scaling, ridge)
            np.savez_compressed(folder / f"model_{size}.npz", **scaling.__dict__, ridge=ridge,
                                **{f"neural.{name}": value.detach().numpy()
                                   for name, value in network.state_dict().items()})
            rows.append({"train_episodes": size, "train_windows": len(train),
                         "train_episode_ids": np.unique(train.episode).tolist(),
                         "large_change_training_windows": int(situation_masks(train, threshold)["large_change"].sum()),
                         "ridge_selection": ridge_selection, "neural_selection": selected,
                         "neural_initializations": [fit[1] for fit in fits]})
        selection = {"split": index + 1, "large_change_threshold": threshold,
                     "selected_rule": selected_rule, "sizes": rows}
        _write(folder / "selection.json", selection)
        selections.append(selection)
        prepared.append((agent, q_before, visits_before, models, threshold, selected_rule))
    # All splits and sizes are fixed before collecting the first test trajectory.
    _write(output_dir / "selections.json", selections)
    evaluation = RobustnessSettings(bootstrap_samples=settings.bootstrap_samples)
    reports = []
    for index, (agent, q_before, visits_before, models, threshold, selected_rule) in enumerate(prepared):
        folder = output_dir / f"split_{index + 1}"
        agent.rng.seed(plans[index]["test"].start)
        test = _collect(agent, plans[index]["test"], original, progress, f"Split {index + 1} shared fresh test")
        _check_frozen_q(agent, q_before, visits_before)
        if agent.predictive_self_model.collection_observation_count or any(
                agent.predictive_self_model.sample_count(action) for action in ACTION_IDS):
            raise RuntimeError("existing self-model learned during collection")
        test.save(folder / "test.npz")
        prediction = rule_predictions(test, original.horizon)
        for size, (network, scaling, ridge) in models.items():
            prediction[f"ridge_{size}"] = scaling.states(
                np.column_stack((scaling.inputs(test), np.ones(len(test)))) @ ridge, test.nominal)
            prediction[f"neural_{size}"] = neural_predict(network, test, scaling)
        largest, smallest = settings.train_sizes[-1], settings.train_sizes[0]
        groups = group_report(test, prediction, threshold, selected_rule, evaluation,
                              settings.bootstrap_seed + index * 100, focal_model=f"neural_{largest}",
                              references=tuple(f"neural_{size}" for size in settings.train_sizes[:-1])
                              + (f"ridge_{largest}", selected_rule))
        for group, mask in situation_masks(test, threshold).items():
            groups[group]["nn_vs_matched_ridge"] = {
                str(size): paired_comparison(prediction[f"neural_{size}"][mask],
                                             prediction[f"ridge_{size}"][mask], subset(test, mask),
                                             evaluation, settings.bootstrap_seed + index * 100)
                for size in settings.train_sizes if np.any(mask)}
        report = {"split": index + 1, "groups": groups,
                  **improvement_gate(groups, smallest, largest, settings.ordinary_tolerance),
                  "live_behavior_unchanged": True, "q_and_existing_self_model_frozen": True,
                  "original_policy_replay_matched": True}
        np.savez_compressed(folder / "predictions.npz", **prediction)
        np.savez_compressed(folder / "groups.npz", **situation_masks(test, threshold))
        _write(folder / "report.json", report)
        for group in groups.values():
            for score in group["metrics"].values():
                score.pop("episode_balanced_mae", None)
        reports.append(report)
        if progress:
            progress(f"Split {index + 1}: more-data point-estimate gate={report['more_data_helps']}, "
                     f"largest NN beats ridge on large changes={report['largest_nn_beats_matched_ridge_on_large_changes']}")
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
    for name, fingerprint in protocol["source_sha256"].items():
        if _fingerprint(source / name) != fingerprint:
            raise RuntimeError("source artifact changed during experiment")
    return {"protocol": {key: value for key, value in protocol.items() if key != "new_lava_layout_sha256"},
            "feature_names": FEATURES, "coordinate_names": COORDINATES,
            "splits": reports, "aggregate_groups": aggregate,
            "new_layouts_verified": len(protocol["new_lava_layout_sha256"]),
            "source_artifacts_unchanged": True,
            "more_data_gate_passed_in_every_split": all(report["more_data_helps"] is True for report in reports),
            "largest_nn_beats_matched_ridge_on_large_changes_in_every_split": all(
                report["largest_nn_beats_matched_ridge_on_large_changes"] is True for report in reports),
            "selections": [{**selection, "sizes": [
                {key: value for key, value in row.items() if key not in ("train_episode_ids", "neural_initializations")}
                for row in selection["sizes"]]} for selection in selections]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/self_model/neural_five_step"))
    parser.add_argument("--previous-protocols", type=Path, nargs="+", default=[
        Path("artifacts/self_model/neural_robustness/protocol.json"),
        Path("artifacts/self_model/neural_weighted/fresh_reference/protocol.json")])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/self_model/neural_learning_curve"))
    parser.add_argument("--summary", type=Path, default=Path("artifacts/self_model/neural_learning_curve.summary.json"))
    args = parser.parse_args()
    report = run_learning_curve(args.source, tuple(args.previous_protocols), args.output_dir,
                                progress=lambda message: print(message, flush=True))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    _write(args.summary, report)
    for group in ("all", "ordinary", "large_change"):
        print(group, {name: score["balanced_mae"] for name, score in report["aggregate_groups"][group]["metrics"].items()})
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
