"""Matched whole-episode comparison of frozen direct and neural controllers.

Evaluation-only override of actions inside the existing safety/Q shortlist.
No model or policy learning. Last horizon-1 moves use the original selector.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from applications.gridworld.agents.metamo import MetaMoAgent
from applications.gridworld.environment import GridWorld
from applications.gridworld.runtime import transition_for_action
from applications.gridworld.evaluation.future_conflict import _probes
from applications.gridworld.evaluation.neural_self_model import NeuralSettings, _pretrain, _check_frozen_q, balanced_mae
from applications.gridworld.evaluation.neural_robustness import FrozenForecasters, _fingerprint, _layout
from applications.gridworld.evaluation.neural_action_ranking import UtilityReadout, candidate_data, choose, readout_inputs, vector

METHODS = ("current_selector", "direct_features", "neural_state")


@dataclass(frozen=True)
class ClosedLoopSettings:
    episodes_per_split: int = 60
    seed_start: int = 18000000
    seed_stride: int = 100000
    bootstrap_samples: int = 2000

    def validate(self):
        if min(self.episodes_per_split, self.bootstrap_samples) < 1 or self.seed_stride < self.episodes_per_split:
            raise ValueError("invalid closed-loop settings")


def run_episode(template, seed, method, original, model, readout, direct):
    if method not in METHODS:
        raise ValueError("unknown controller")
    agent = copy.deepcopy(template)
    agent.record_transitions = False
    agent.reset_episode()
    agent.rng.seed(seed)
    q, visits = copy.deepcopy(agent.q_table), copy.deepcopy(agent.visit_counts)
    env = GridWorld(seed=seed, max_steps=original.max_steps)
    state = env.reset()
    loss = reward_total = 0.0
    lava = changed = opportunities = fallbacks = 0
    trace, committed = [], []
    for step in range(original.max_steps):
        proposed, _ = agent.select_action(state)
        probes = _probes(copy.deepcopy(agent), state)
        data = candidate_data(agent, state, probes)
        live_index = list(data.action).index(proposed)
        forecasts = model.predict(data)["neural_32"]
        eligible_horizon = step + original.horizon <= original.max_steps
        selected = live_index
        scores = None
        if method != "current_selector" and eligible_horizon:
            scores = (direct.predict(data.features) if method == "direct_features" else
                      readout.predict(readout_inputs(data.current, forecasts, data.features)))
            selected = choose(scores, live_index)
        elif method != "current_selector":
            fallbacks += 1
        action = int(data.action[selected])
        if action not in [probe.action for probe in probes]:
            raise RuntimeError("controller escaped the eligible shortlist")
        changed += int(action != proposed)
        opportunities += int(eligible_horizon and len(probes) > 1)
        if action != proposed:
            chosen, staged, _, _ = transition_for_action(state, agent.mot, action, runtime=agent.runtime)
            agent._pending_action, agent._pending_state = chosen, staged
        before = env.energy
        following, reward, done, _ = env.step(action)
        agent.commit_motivational_transition()
        committed.append(vector(agent.mot))
        loss += max(0.0, before-env.energy)
        lava += int(following["in_lava"])
        reward_total += reward
        trace.append({"step": step, "proposed": proposed, "action": action,
                      "eligible": data.action.tolist(), "position": list(state["pos"]),
                      "forecast": forecasts[selected].tolist(), "full_horizon_requested": eligible_horizon,
                      "scores": None if scores is None else scores.tolist(), "reward": reward})
        state = following
        if done:
            break
    _check_frozen_q(agent, q, visits)
    errors = []
    for index, row in enumerate(trace):
        target_index = index + original.horizon - 1
        if target_index < len(committed):
            target = committed[target_index]
            error = balanced_mae(np.asarray([row["forecast"]]), np.asarray([target]))
            errors.append(error)
            row.update(target=target.tolist(), forecast_mae=error)
    result = {"episode": seed, "method": method, "minerals": env.minerals_collected,
              "energy_loss": loss, "lava_steps": lava, "depleted": bool(env.energy <= 0),
              "zero_collection": env.minerals_collected == 0, "reward": reward_total,
              "utility": 100 * env.minerals_collected - loss, "steps": len(trace),
              "changed_choices": changed, "multi_action_opportunities": opportunities,
              "tail_fallbacks": fallbacks, "forecast_targets": len(errors),
              "nn_state_mae": float(np.mean(errors)) if errors else None,
              "q_frozen": True, "shortlist_preserved": True}
    return result, trace


def paired_summary(rows, candidate, reference, samples, seed):
    left = {row["episode"]: row for row in rows if row["method"] == candidate}
    right = {row["episode"]: row for row in rows if row["method"] == reference}
    if not left or left.keys() != right.keys():
        raise ValueError("paired controllers must have identical episode seeds")
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(left), size=(samples, len(left)))
    report = {}
    for metric in ("minerals", "energy_loss", "depleted", "zero_collection", "utility"):
        deltas = np.asarray([float(left[key][metric])-float(right[key][metric]) for key in sorted(left)])
        interval = np.quantile(deltas[draws].mean(axis=1), [0.025,0.975]).tolist() if len(left) >= 2 else None
        report[metric] = {"mean_difference": float(deltas.mean()), "bootstrap_95_percent_interval": interval}
    return report


def summarize(rows, settings, seed):
    means = {method: {metric: float(np.mean([row[metric] for row in rows if row["method"] == method]))
                      for metric in ("minerals", "energy_loss", "lava_steps", "depleted", "zero_collection",
                                     "reward", "utility", "steps", "changed_choices", "nn_state_mae")}
             for method in METHODS}
    comparisons = {f"{candidate}_vs_{reference}": paired_summary(rows, candidate, reference,
                                                                settings.bootstrap_samples, seed+index)
                   for index,(candidate,reference) in enumerate((("direct_features","current_selector"),
                       ("neural_state","current_selector"),("neural_state","direct_features")))}
    support = {}
    for candidate in METHODS[1:]:
        comparison = comparisons[f"{candidate}_vs_current_selector"]
        interval = comparison["minerals"]["bootstrap_95_percent_interval"]
        support[candidate] = bool(interval is not None and interval[0] > 0
                                 and comparison["energy_loss"]["mean_difference"] <= 0
                                 and comparison["depleted"]["mean_difference"] <= 0)
    return {"means": means, "comparisons": comparisons, "support_vs_current_selector": support}


def _write(path, value):
    path.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")


def run_closed_loop(source, curve, ranking, output_dir, settings=ClosedLoopSettings(), progress=None):
    settings.validate()
    original = NeuralSettings(**json.loads((source/"settings.json").read_text()))
    original.validate()
    for protected in (source,curve,ranking):
        if output_dir.resolve() == protected.resolve() or protected.resolve() in output_dir.resolve().parents:
            raise ValueError("outputs must be outside source directories")
    curve_protocol = json.loads((curve/"protocol.json").read_text())
    rank_protocol = json.loads((ranking/"protocol.json").read_text())
    old_seeds = set(map(int,curve_protocol["new_lava_layout_sha256"])) | set(map(int,rank_protocol["new_lava_layout_sha256"]))
    old_count = original.pretrain_episodes + original.train_episodes + original.validation_episodes + original.test_episodes
    old_seeds.update(original.seed_start+i*original.seed_stride+j for i in range(original.splits) for j in range(old_count))
    for path in curve_protocol["previous_protocol_sha256"]:
        previous = json.loads(Path(path).read_text())
        for row in previous["splits"]:
            old_seeds.update(range(row["fresh_seed_start"],row["fresh_seed_stop_exclusive"]))
    seeds = [settings.seed_start+i*settings.seed_stride+j for i in range(original.splits) for j in range(settings.episodes_per_split)]
    layouts = [_layout(seed,original.max_steps) for seed in seeds]
    if (old_seeds.intersection(seeds) or len(set(layouts)) != len(layouts)
            or {_layout(seed,original.max_steps) for seed in old_seeds}.intersection(layouts)):
        raise ValueError("fresh seed/layout collision; choose a new block before observing outcomes")
    inputs = [source/"settings.json",curve/"protocol.json",ranking/"protocol.json"]
    for index in range(original.splits):
        inputs.extend([curve/f"split_{index+1}"/"model_320.npz",ranking/f"split_{index+1}"/"readouts.npz"])
    fingerprints = {str(path):_fingerprint(path) for path in inputs}
    output_dir.mkdir(parents=True,exist_ok=True)
    protocol = {"settings":asdict(settings),"original_settings":asdict(original),"source_sha256":fingerprints,
                "methods":METHODS,"new_lava_layout_sha256":dict(zip(seeds,layouts)),
                "primary_metric":"total episode mineral collections; paired episode bootstrap per policy split",
                "support_criterion":"positive mineral-gain interval in each split without increased mean energy loss or depletion",
                "tail_policy":"last horizon-1 steps use current selector for every experimental controller",
                "frozen":"Q, forecast models, readouts, online self-model; no training or test-driven selection",
                "matching":"identical initial environment and policy RNG seeds; subsequent trajectories and random draws can diverge",
                "limits":["Models/readouts were trained for original-policy continuation; repeated intervention changes that distribution.",
                          "No energy-loss variation means no evidence of safety improvement.",
                          "State MAE is diagnostic and compares different visited states across controllers.",
                          "Bootstrap intervals resample whole paired episodes; exploratory comparisons are unadjusted."]}
    _write(output_dir/"protocol.json",protocol)
    reports=[]; all_rows=[]
    for index in range(original.splits):
        folder=output_dir/f"split_{index+1}"; folder.mkdir(exist_ok=True)
        model=FrozenForecasters(curve/f"split_{index+1}"/"model_320.npz",original.horizon)
        with np.load(ranking/f"split_{index+1}"/"readouts.npz",allow_pickle=False) as saved:
            readout=UtilityReadout(**{key:saved['state_'+key].copy() for key in ('mean','scale','coefficients')})
            direct=UtilityReadout(**{key:saved['direct_'+key].copy() for key in ('mean','scale','coefficients')})
        template=MetaMoAgent(seed=original.agent_seed+index)
        base=original.seed_start+index*original.seed_stride
        _pretrain(template,range(base,base+original.pretrain_episodes),original)
        template.epsilon=0
        template.predictive_self_model.learning_enabled=False
        rows=[]
        with (folder/"trajectories.jsonl").open('w',encoding='utf-8') as stream:
            for episode in range(settings.episodes_per_split):
                seed=settings.seed_start+index*settings.seed_stride+episode
                for method in METHODS:
                    result,trace=run_episode(template,seed,method,original,model,readout,direct)
                    rows.append(result)
                    stream.write(json.dumps({"episode":seed,"method":method,"trace":trace})+'\n')
                if progress and ((episode+1)%10==0 or episode+1==settings.episodes_per_split):
                    progress(f"Split {index+1}: {episode+1}/{settings.episodes_per_split} matched environments completed")
        report={"split":index+1,**summarize(rows,settings,settings.seed_start+index)}
        _write(folder/"episodes.json",rows); _write(folder/"report.json",report)
        reports.append(report); all_rows.extend(rows)
        if progress:
            progress(f"Split {index+1} mean minerals: "+str({key:value['minerals'] for key,value in report['means'].items()}))
    for path,expected in fingerprints.items():
        if _fingerprint(Path(path)) != expected:
            raise RuntimeError("source artifact changed")
    return {"protocol":protocol,"splits":reports,"aggregate_means":{
                method:{metric:float(np.mean([row[metric] for row in all_rows if row['method']==method]))
                        for metric in reports[0]['means'][method]} for method in METHODS},
            "matched_environments":len(seeds),"controller_episodes":len(all_rows),"source_artifacts_unchanged":True,
            "support_in_every_split":{method:all(row['support_vs_current_selector'][method] for row in reports) for method in METHODS[1:]}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('artifacts/self_model/neural_five_step'))
    parser.add_argument('--curve',type=Path,default=Path('artifacts/self_model/neural_learning_curve'))
    parser.add_argument('--ranking',type=Path,default=Path('artifacts/self_model/neural_action_ranking'))
    parser.add_argument('--output-dir',type=Path,default=Path('artifacts/self_model/neural_closed_loop'))
    parser.add_argument('--summary',type=Path,default=Path('artifacts/self_model/neural_closed_loop.summary.json'))
    args=parser.parse_args()
    result=run_closed_loop(args.source,args.curve,args.ranking,args.output_dir,progress=lambda message:print(message,flush=True))
    args.summary.parent.mkdir(parents=True,exist_ok=True); _write(args.summary,result)
    print(json.dumps(result['aggregate_means'],indent=2)); print(f'Summary: {args.summary}')


if __name__=='__main__':
    main()
