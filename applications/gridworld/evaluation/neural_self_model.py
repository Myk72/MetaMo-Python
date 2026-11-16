"""Compare a small neural self-model with rules and linear regression.

Target: the actual committed motivational vector x[t+5], conditional on the
selected action and the continuation of a frozen MetaMo policy. Inputs are
copied before env.step. The known staged state x[t+1] is a shared reference;
learned models predict the residual x[t+5] - x[t+1]. All output coordinates
are clipped to [0, 1] for evaluation, never inserted into the live agent.

Train Q first, freeze Q/epsilon, then collect disjoint whole-episode training,
validation, and test sets. Fit scaling on training data only. Validation selects
the rule, ridge penalty, neural epoch, and neural initialization. Test outcomes
are read only after these selections. No random row splitting of overlapping
five-step windows is allowed. CPU PyTorch is an optional experiment dependency.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
from dataclasses import asdict, astuple, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from applications.gridworld.agents.metamo import MetaMoAgent
from applications.gridworld.config import ACTION_IDS
from applications.gridworld.environment import GridWorld
from applications.gridworld.schema import GRIDWORLD_SCHEMA
from applications.gridworld.stimulus import GridWorldStimulus


GOALS = GRIDWORLD_SCHEMA.goal_names
MODULATORS = GRIDWORLD_SCHEMA.modulator_names
COORDINATES = tuple(f"goal.{name}" for name in GOALS) + tuple(f"modulator.{name}" for name in MODULATORS)
DIM = len(COORDINATES)
FEATURES = (
    tuple(f"current.{name}" for name in COORDINATES)
    + tuple(f"nominal.{name}" for name in COORDINATES)
    + tuple(f"stimulus.{name}" for name in GridWorldStimulus.__dataclass_fields__)
    + tuple(f"action.{name}" for name in ACTION_IDS)
    + ("move.hazard_risk", "move.mineral_progress", "move.mineral_collected", "move.safe_mobility_after")
    + ("policy.task_score", "policy.safety_score", "policy.composed_score", "policy.tanh_q_over_100")
)
RULES = ("persistence", "nominal_hold", "nominal_trend")


@dataclass(frozen=True)
class NeuralSettings:
    splits: int = 3
    pretrain_episodes: int = 30
    train_episodes: int = 80
    validation_episodes: int = 20
    test_episodes: int = 24
    max_steps: int = 40
    horizon: int = 5
    hidden_units: int = 32
    max_epochs: int = 300
    patience: int = 30
    initializations: int = 3
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    ridge_penalties: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    seed_start: int = 7000000
    seed_stride: int = 100000
    agent_seed: int = 100
    network_seed: int = 1000

    def validate(self):
        if (
            min(self.splits, self.pretrain_episodes, self.train_episodes,
                self.validation_episodes, self.test_episodes, self.hidden_units,
                self.max_epochs, self.patience, self.initializations, self.batch_size) < 1
            or self.horizon < 2 or self.max_steps < self.horizon
            or self.seed_stride < self.pretrain_episodes + self.train_episodes + self.validation_episodes + self.test_episodes
            or not np.isfinite(self.learning_rate) or self.learning_rate <= 0
            or not np.isfinite(self.weight_decay) or self.weight_decay < 0
            or not self.ridge_penalties
            or any(not np.isfinite(value) or value <= 0 for value in self.ridge_penalties)
        ):
            raise ValueError("invalid neural experiment settings")


@dataclass(frozen=True)
class ForecastInput:
    """Only values available after selection and before executing that action."""

    current: tuple[float, ...]
    nominal: tuple[float, ...]
    features: tuple[float, ...]
    action: int


@dataclass(frozen=True)
class Dataset:
    features: np.ndarray
    current: np.ndarray
    nominal: np.ndarray
    target: np.ndarray
    episode: np.ndarray
    step: np.ndarray
    action: np.ndarray

    def __len__(self):
        return len(self.target)

    def save(self, path: Path):
        np.savez_compressed(path, **self.__dict__)


def freeze_input(agent: MetaMoAgent, action: int, alpha: dict) -> ForecastInput:
    pending = agent._pending_transition
    if pending is None or pending.move_inputs is None or action != pending.action_index:
        raise ValueError("freeze_input requires the selected pending transition")
    current = pending.current_state.goals + pending.current_state.modulators
    nominal = pending.staged_state.goals + pending.staged_state.modulators
    features = (
        current + nominal + astuple(pending.current_stimulus)
        + tuple(float(index == action) for index in range(len(ACTION_IDS)))
        + astuple(pending.move_inputs)
        + (float(alpha["task_score"]), float(alpha["safety_score"]),
           float(alpha["composed_motivation_score"]), float(np.tanh(alpha["q_value"] / 100.0)))
    )
    if len(features) != len(FEATURES) or not np.all(np.isfinite(features)):
        raise ValueError("invalid pre-action feature vector")
    return ForecastInput(current, nominal, features, action)


def episode_dataset(inputs: list[ForecastInput], observed: list[tuple[float, ...]],
                    episode: int, horizon: int) -> Dataset:
    if horizon < 1 or len(inputs) != len(observed):
        raise ValueError("invalid episode alignment")
    count = len(inputs) - horizon + 1
    if count <= 0:
        raise ValueError("episode has no complete forecast horizon")
    rows = inputs[:count]
    data = Dataset(
        features=np.asarray([row.features for row in rows], dtype=float),
        current=np.asarray([row.current for row in rows], dtype=float),
        nominal=np.asarray([row.nominal for row in rows], dtype=float),
        target=np.asarray(observed[horizon - 1:], dtype=float),
        episode=np.full(count, episode, dtype=np.int64),
        step=np.arange(count, dtype=np.int64),
        action=np.asarray([row.action for row in rows], dtype=np.int64),
    )
    for values in (data.current, data.nominal, data.target):
        if values.shape != (count, DIM) or not np.all(np.isfinite(values)):
            raise ValueError("invalid motivational coordinates")
        if np.any(values < 0) or np.any(values > 1):
            raise ValueError("experiment assumes GridWorld coordinates in [0, 1]")
    return data


def _combine(parts: list[Dataset]) -> Dataset:
    if not parts:
        raise ValueError("no full-horizon episodes collected")
    return Dataset(**{
        name: np.concatenate([getattr(part, name) for part in parts])
        for name in Dataset.__dataclass_fields__
    })


def _collect(agent: MetaMoAgent, seeds: range, settings: NeuralSettings,
             progress: Callable[[str], None] | None, label: str) -> Dataset:
    parts = []
    for index, seed in enumerate(seeds):
        env = GridWorld(seed=seed, max_steps=settings.max_steps)
        state = env.reset()
        agent.reset_episode()
        agent.transition_records.clear()
        mirror = copy.deepcopy(agent)
        mirror.record_transitions = False
        mirror_env = copy.deepcopy(env)
        mirror_state = copy.deepcopy(state)
        inputs, observed = [], []
        for _ in range(settings.max_steps):
            action, alpha = agent.select_action(state)
            other_action, _ = mirror.select_action(mirror_state)
            if action != other_action:
                raise RuntimeError("recording changed action selection")
            inputs.append(freeze_input(agent, action, alpha))
            following, reward, done, _ = env.step(action)
            other_state, other_reward, other_done, _ = mirror_env.step(other_action)
            agent.commit_motivational_transition(
                next_env_state=following, done=done, executed_action=action
            )
            mirror.commit_motivational_transition()
            if (
                following != other_state or reward != other_reward or done != other_done
                or not np.array_equal(agent.mot.G, mirror.mot.G)
                or not np.array_equal(agent.mot.M, mirror.mot.M)
                or agent.rng.getstate() != mirror.rng.getstate()
            ):
                raise RuntimeError("recording changed the live trajectory")
            snapshot = agent.transition_records[-1].committed_state
            observed.append(snapshot.goals + snapshot.modulators)
            state, mirror_state = following, other_state
            if done:
                break
        if len(inputs) < settings.horizon:
            raise RuntimeError("episode ended before a full horizon; inspect terminal censoring")
        parts.append(episode_dataset(inputs, observed, seed, settings.horizon))
        agent.transition_records.clear()
        if progress and ((index + 1) % 20 == 0 or index + 1 == len(seeds)):
            progress(f"{label}: collected {index + 1}/{len(seeds)} episodes")
    return _combine(parts)


def _pretrain(agent: MetaMoAgent, seeds: range, settings: NeuralSettings):
    for seed in seeds:
        env = GridWorld(seed=seed, max_steps=settings.max_steps)
        state = env.reset()
        agent.reset_episode()
        for _ in range(settings.max_steps):
            action, alpha = agent.select_action(state)
            following, reward, done, info = env.step(action)
            agent.update(state, action, reward, following, done, info.get("event"), alpha)
            state = following
            if done:
                break
        agent.decay_epsilon()


@dataclass(frozen=True)
class Scaling:
    input_mean: np.ndarray
    input_scale: np.ndarray
    residual_mean: np.ndarray
    residual_scale: np.ndarray

    @classmethod
    def fit(cls, train: Dataset) -> "Scaling":
        residual = train.target - train.nominal
        return cls(
            train.features.mean(axis=0), np.maximum(train.features.std(axis=0), 1e-6),
            residual.mean(axis=0), np.maximum(residual.std(axis=0), 1e-3),
        )

    def inputs(self, data: Dataset) -> np.ndarray:
        return (data.features - self.input_mean) / self.input_scale

    def residuals(self, data: Dataset) -> np.ndarray:
        return (data.target - data.nominal - self.residual_mean) / self.residual_scale

    def states(self, normalized_residual: np.ndarray, nominal: np.ndarray) -> np.ndarray:
        return np.clip(nominal + self.residual_mean + normalized_residual * self.residual_scale, 0.0, 1.0)


def rule_predictions(data: Dataset, horizon: int) -> dict[str, np.ndarray]:
    return {
        "persistence": data.current,
        "nominal_hold": data.nominal,
        "nominal_trend": np.clip(data.current + horizon * (data.nominal - data.current), 0, 1),
    }


def balanced_mae(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.shape != target.shape or not len(target) or not np.all(np.isfinite(prediction)):
        raise ValueError("invalid predictions")
    error = np.abs(prediction - target)
    return float(0.5 * (error[:, :len(GOALS)].mean() + error[:, len(GOALS):].mean()))


def metrics(prediction: np.ndarray, data: Dataset) -> dict:
    error = np.abs(prediction - data.target)
    return {
        "balanced_mae": balanced_mae(prediction, data.target),
        "goal_mae": float(error[:, :len(GOALS)].mean()),
        "modulator_mae": float(error[:, len(GOALS):].mean()),
        "coordinate_mae": dict(zip(COORDINATES, error.mean(axis=0).tolist())),
        "episode_balanced_mae": {
            str(seed): balanced_mae(prediction[data.episode == seed], data.target[data.episode == seed])
            for seed in np.unique(data.episode)
        },
    }


def _design(data: Dataset, scaling: Scaling) -> np.ndarray:
    return np.column_stack((scaling.inputs(data), np.ones(len(data))))


def fit_ridge(train: Dataset, validation: Dataset, scaling: Scaling,
              penalties: tuple[float, ...]) -> tuple[np.ndarray, dict]:
    x = _design(train, scaling)
    y = scaling.residuals(train)
    penalty = np.eye(x.shape[1])
    penalty[-1, -1] = 0.0  # The intercept is not regularized.
    candidates = []
    for strength in penalties:
        coefficients = np.linalg.solve(x.T @ x + strength * penalty, x.T @ y)
        prediction = scaling.states(_design(validation, scaling) @ coefficients, validation.nominal)
        candidates.append((balanced_mae(prediction, validation.target), strength, coefficients))
    score, strength, coefficients = min(candidates, key=lambda candidate: candidate[0])
    return coefficients, {"penalty": strength, "validation_balanced_mae": score}


def _torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("This optional experiment needs PyTorch: python -m pip install torch") from error
    return torch


def fit_neural(train: Dataset, validation: Dataset, scaling: Scaling,
               settings: NeuralSettings, seed: int, *,
               sample_weights: np.ndarray | None = None) -> tuple[object, dict]:
    torch = _torch()
    # Local CPU generator control; do not consume the GridWorld agent RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        network = torch.nn.Sequential(
            torch.nn.Linear(len(FEATURES), settings.hidden_units),
            torch.nn.Tanh(),
            torch.nn.Linear(settings.hidden_units, DIM),
        )
        torch.nn.init.zeros_(network[-1].weight)
        torch.nn.init.zeros_(network[-1].bias)
    optimizer = torch.optim.AdamW(network.parameters(), lr=settings.learning_rate,
                                  weight_decay=settings.weight_decay)
    x = torch.as_tensor(scaling.inputs(train), dtype=torch.float32)
    y = torch.as_tensor(scaling.residuals(train), dtype=torch.float32)
    weights = None
    if sample_weights is not None:
        values = np.asarray(sample_weights, dtype=float)
        if values.shape != (len(train),) or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("sample weights must be positive finite values, one per training row")
        # Global mean normalization preserves the expected loss/gradient scale.
        # All-one weights take the original path for exact control replication.
        if not np.all(values == values[0]):
            weights = torch.as_tensor(values / values.mean(), dtype=torch.float32)
    vx = torch.as_tensor(scaling.inputs(validation), dtype=torch.float32)
    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_epoch, stale = 0, 0
    best_weights = None
    history = []
    for epoch in range(settings.max_epochs + 1):
        if epoch:
            network.train()
            permutation = rng.permutation(len(train))
            for start in range(0, len(train), settings.batch_size):
                indices = permutation[start:start + settings.batch_size]
                optimizer.zero_grad(set_to_none=True)
                output = network(x[indices])
                if weights is None:
                    loss = torch.nn.functional.mse_loss(output, y[indices])
                else:
                    row_loss = (output - y[indices]).square().mean(dim=1)
                    loss = (weights[indices] * row_loss).mean()
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("nonfinite neural training loss")
                loss.backward()
                optimizer.step()
        network.eval()
        with torch.no_grad():
            prediction = scaling.states(network(vx).numpy(), validation.nominal)
        score = balanced_mae(prediction, validation.target)
        if score < best_score - 1e-8:
            best_score, best_epoch, stale = score, epoch, 0
            best_weights = {name: value.detach().clone() for name, value in network.state_dict().items()}
        else:
            stale += 1
        if epoch % 10 == 0:
            history.append({"epoch": epoch, "validation_balanced_mae": score})
        if stale >= settings.patience:
            break
    network.load_state_dict(best_weights)
    network.eval()
    return network, {
        "seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
        "validation_balanced_mae": best_score,
        "parameter_count": sum(parameter.numel() for parameter in network.parameters()),
        "validation_history": history,
    }


def neural_predict(network: object, data: Dataset, scaling: Scaling) -> np.ndarray:
    torch = _torch()
    network.eval()
    with torch.no_grad():
        residual = network(torch.as_tensor(scaling.inputs(data), dtype=torch.float32)).numpy()
    return scaling.states(residual, data.nominal)


def _episode_improvement(neural: dict, baseline: dict, seed: int) -> dict:
    """Positive differences mean lower neural error. Resample whole episodes."""
    keys = sorted(neural["episode_balanced_mae"])
    difference = np.asarray([
        baseline["episode_balanced_mae"][key] - neural["episode_balanced_mae"][key] for key in keys
    ])
    rng = np.random.default_rng(seed)
    sampled = difference[rng.integers(0, len(difference), size=(2000, len(difference)))].mean(axis=1)
    return {
        "mean_episode_mae_reduction": float(difference.mean()),
        "bootstrap_95_percent_interval": np.quantile(sampled, [0.025, 0.975]).tolist(),
        "episodes": len(difference),
    }


def _check_frozen_q(agent: MetaMoAgent, values: dict, visits: dict):
    for key, value in agent.q_table.items():
        if (not np.array_equal(value, values.get(key, np.zeros(agent.ACTIONS)))
                or not np.array_equal(agent.visit_counts[key], visits.get(key, np.zeros(agent.ACTIONS)))):
            raise RuntimeError("Q learning occurred during forecast data collection")


def run_neural_experiment(settings: NeuralSettings = NeuralSettings(), *,
                          output_dir: Path | None = None,
                          progress: Callable[[str], None] | None = None) -> dict:
    settings.validate()
    torch = _torch()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _run(settings, output_dir, progress, torch)
    finally:
        torch.set_num_threads(previous_threads)


def _run(settings: NeuralSettings, output_dir: Path | None,
         progress: Callable[[str], None] | None, torch) -> dict:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "settings.json").write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
    reports = []
    for index in range(settings.splits):
        prefix = f"Split {index + 1}/{settings.splits}"
        base = settings.seed_start + index * settings.seed_stride
        agent = MetaMoAgent(seed=settings.agent_seed + index)
        if progress:
            progress(f"{prefix}: training the continuation policy")
        _pretrain(agent, range(base, base + settings.pretrain_episodes), settings)
        agent.epsilon = 0.0
        agent.record_transitions = True
        agent.predictive_self_model.learning_enabled = False
        q_before = copy.deepcopy(agent.q_table)
        visits_before = copy.deepcopy(agent.visit_counts)
        train_start = base + settings.pretrain_episodes
        validation_start = train_start + settings.train_episodes
        test_start = validation_start + settings.validation_episodes
        train = _collect(agent, range(train_start, validation_start), settings, progress, f"{prefix} training")
        validation = _collect(agent, range(validation_start, test_start), settings, progress, f"{prefix} validation")
        scaling = Scaling.fit(train)
        validation_rules = rule_predictions(validation, settings.horizon)
        selected_rule = min(RULES, key=lambda name: balanced_mae(validation_rules[name], validation.target))
        ridge, ridge_selection = fit_ridge(train, validation, scaling, settings.ridge_penalties)
        candidates = []
        for initialization in range(settings.initializations):
            seed = settings.network_seed + index * 100 + initialization
            network, selection = fit_neural(train, validation, scaling, settings, seed)
            candidates.append((network, selection))
            if progress:
                progress(f"{prefix}: NN seed {seed}, validation MAE {selection['validation_balanced_mae']:.6f}, "
                         f"selected epoch {selection['best_epoch']}")
        network, selection = min(candidates, key=lambda candidate: candidate[1]["validation_balanced_mae"])
        # The test trajectories have not even been generated during selection.
        test = _collect(agent, range(test_start, test_start + settings.test_episodes),
                        settings, progress, f"{prefix} test")
        _check_frozen_q(agent, q_before, visits_before)
        if agent.predictive_self_model.collection_observation_count or any(
            agent.predictive_self_model.sample_count(action) for action in ACTION_IDS
        ):
            raise RuntimeError("diagnostic self-model learned during data collection")
        predictions = rule_predictions(test, settings.horizon)
        predictions["mean_residual"] = scaling.states(np.zeros_like(test.target), test.nominal)
        predictions["ridge"] = scaling.states(_design(test, scaling) @ ridge, test.nominal)
        predictions["neural_32"] = neural_predict(network, test, scaling)
        scores = {name: metrics(prediction, test) for name, prediction in predictions.items()}
        nn = scores["neural_32"]
        baselines = (*RULES, "mean_residual", "ridge")
        report = {
            "agent_seed": settings.agent_seed + index,
            "pretrain_seed_start": base, "train_seed_start": train_start,
            "validation_seed_start": validation_start, "test_seed_start": test_start,
            "samples": {"train": len(train), "validation": len(validation), "test": len(test)},
            "selected_rule": selected_rule, "ridge_selection": ridge_selection,
            "neural_selection": selection,
            "neural_initializations": [
                {**candidate_selection,
                 "test_balanced_mae": balanced_mae(neural_predict(candidate, test, scaling), test.target)}
                for candidate, candidate_selection in candidates
            ],
            "test_metrics": scores,
            "mean_absolute_actual_five_step_change": dict(zip(
                COORDINATES, np.abs(test.target - test.current).mean(axis=0).tolist()
            )),
            "neural_vs_selected_rule": _episode_improvement(nn, scores[selected_rule], settings.network_seed + index),
            "neural_vs_ridge": _episode_improvement(nn, scores["ridge"], settings.network_seed + 100 + index),
            "neural_beats_all_baselines_for_goals_and_modulators": all(
                nn["goal_mae"] < scores[name]["goal_mae"] and nn["modulator_mae"] < scores[name]["modulator_mae"]
                for name in baselines
            ),
            "live_behavior_unchanged": True,
            "q_and_existing_self_model_frozen": True,
        }
        reports.append(report)
        if output_dir is not None:
            folder = output_dir / f"split_{index + 1}"
            folder.mkdir(exist_ok=True)
            for name, data in (("train", train), ("validation", validation), ("test", test)):
                data.save(folder / f"{name}.npz")
            np.savez_compressed(folder / "predictions.npz", **predictions)
            np.savez_compressed(folder / "model.npz", **scaling.__dict__, ridge=ridge,
                                **{f"neural.{name}": value.detach().numpy() for name, value in network.state_dict().items()})
            (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if progress:
            progress(f"{prefix}: test MAE rule={scores[selected_rule]['balanced_mae']:.6f}, "
                     f"ridge={scores['ridge']['balanced_mae']:.6f}, NN={nn['balanced_mae']:.6f}")
    names = (*RULES, "mean_residual", "ridge", "neural_32")
    aggregate = {
        name: {metric: float(np.mean([report["test_metrics"][name][metric] for report in reports]))
               for metric in ("balanced_mae", "goal_mae", "modulator_mae")}
        for name in names
    }
    return {
        "settings": asdict(settings), "feature_names": FEATURES, "coordinate_names": COORDINATES,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "protocol": {
            "environment": "standard GridWorld", "target": f"committed x[t+{settings.horizon}]",
            "learned_residual": f"x[t+{settings.horizon}] - nominal x[t+1]",
            "delta_prediction": "predicted future state minus current state; identical absolute error",
            "input_capture": "after selecting the proposed action, before env.step",
            "neural_architecture": f"{len(FEATURES)} inputs -> {settings.hidden_units} tanh -> {DIM} linear outputs",
            "fit_loss": "MSE of training-standardized residuals",
            "selection_metric": "equal-weight mean of goal MAE and modulator MAE",
            "point_estimate_gate": "neural goal and modulator MAE each beat every baseline in every split",
            "normalization": "training episodes only", "output_bounds": [0, 1],
            "data_split": "disjoint entire episodes; no cross-episode labels",
            "continuation": "frozen Q, epsilon zero, existing randomized tie resolution",
            "limits": "on-policy forecasts only; overlapping windows; bootstrap resamples whole test episodes",
            "deployment": "diagnostic experiment; no neural action selection",
        },
        "splits": reports, "mean_across_splits": aggregate,
        "neural_point_estimates_improved_in_every_split": all(
            report["neural_beats_all_baselines_for_goals_and_modulators"] for report in reports
        ),
        "neural_ridge_advantage_has_positive_interval_in_every_split": all(
            report["neural_vs_ridge"]["bootstrap_95_percent_interval"][0] > 0 for report in reports
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/self_model/neural_five_step"))
    parser.add_argument("--summary", type=Path, default=Path("artifacts/self_model/neural_five_step.summary.json"))
    args = parser.parse_args()
    report = run_neural_experiment(output_dir=args.output_dir,
                                   progress=lambda message: print(message, flush=True))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["mean_across_splits"], indent=2))
    print(f"Neural point estimates improved in every split: {report['neural_point_estimates_improved_in_every_split']}")
    print(f"Summary: {args.summary}; datasets, weights and predictions: {args.output_dir}")


if __name__ == "__main__":
    main()
