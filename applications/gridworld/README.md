# GridWorld

GridWorld is a rule-based MetaMo application with a tabular Q-learning
baseline. It contains the environment, task-specific motivational model,
agents, controlled evaluation tools, and an optional visual simulation.

```text
applications/gridworld/
|-- agents/        baseline and MetaMo-guided Q-learning agents
|-- assets/        simulation sprites and audio
|-- evaluation/    experiment runners, metrics, plots, and smoke checks
|-- simulation/    optional pygame visualization
|-- config.py      environment and calibration constants
|-- environment.py GridWorld environment
|-- schema.py      goal and modulator coordinates
|-- stimulus.py    normalized appraisal input
|-- profile.py     task-specific appraisal and action semantics
|-- runtime.py     binding to the existing MetaMo pipeline
`-- state.py       initial motivational state
```

## Variables

The two MetaMo overgoals remain `individuation` and `transcendence`.

The application uses three primary goals:

| Goal | Initial value | Operational meaning |
|---|---:|---|
| `mineral_acquisition` | 0.75 | make progress toward and collect minerals |
| `energy_preservation` | 0.65 | avoid energy loss and recover energy |
| `navigation_efficiency` | 0.45 | move productively without boundary waste |

Two anti-goals make hazards explicit:

| Anti-goal | Initial value | Activated by |
|---|---:|---|
| `lava_exposure` | 0.85 | local lava risk |
| `boundary_collision` | 0.35 | an action that tries to leave the grid |

The appraisal stimulus contains six normalized signals:

| Signal | Meaning |
|---|---|
| `goal_proximity` | closeness to the current mineral |
| `hazard_pressure` | calibrated pressure from nearby lava |
| `energy_deficit` | missing energy as a fraction of full energy |
| `time_pressure` | fraction of the episode budget already used |
| `safe_progress` | mineral proximity discounted by nearby hazard |
| `safe_mobility` | fraction of directions that stay in bounds and avoid lava |

## Runtime process

For every GridWorld step:

1. The application builds six appraisal signals and four action candidates.
2. Rule appraisal updates motivation through MetaMo's existing goal-update,
   projection, coherence, law-audit, and safety pipeline.
3. Task and safety perspectives score the actions.
4. Q-learning supplies a near-optimal action shortlist.
5. The composed selector chooses within the shortlist and rejects immediate
   lava entry when a safer legal action exists.
6. Q updates occur during training only. Evaluation freezes Q while preserving
   MetaMo's motivational transition.

## Evaluation variants

The default evaluation compares:

| Variant | Purpose |
|---|---|
| `BaselineCompactQ` | Q-learning control |
| `MetaMoTaskSelector` | task perspective only |
| `MetaMoSafetySelector` | safety perspective only |
| `MetaMoComposedSelector` | composed task and safety perspectives |
| `MetaMo` | composed selector with hard immediate-lava safety |

### Selector-withdrawal experiment

The selector-withdrawal experiment separates improvement learned into the
Q-table from improvement supplied by MetaMo while actions are being chosen:

| Variant | Training policy | Evaluation policy | Question |
|---|---|---|---|
| `QTrainQEval` | Q only | Q only | baseline |
| `MetaMoTrainQEval` | MetaMo-guided | Q only | did guided experience improve Q itself? |
| `QTrainMetaMoEval` | Q only | MetaMo | what does MetaMo add only at decision time? |
| `MetaMoTrainMetaMoEval` | MetaMo-guided | MetaMo | total combined effect |

## Run

Run the lightweight validation:

```powershell
python -m unittest discover -s tests
```

Run and plot the default controlled evaluation:

```powershell
python -m applications.gridworld.evaluation.runner `
  --output-dir eval_results/gridworld

python -m applications.gridworld.evaluation.analyze_ablation `
  --input-dir eval_results/gridworld

python -m applications.gridworld.evaluation.plot_results `
  --input-dir eval_results/gridworld
```

Run the standalone-Q learning-transfer diagnostic:

```powershell
python -m applications.gridworld.evaluation.learning_transfer --quiet-seeds
```

Run the visual simulation after installing `pygame`:

```powershell
python -m applications.gridworld.simulation.main
```

The non-visual application requires NumPy. Generated results are written under
`eval_results/` and are not application source.

## Predictive self-model experiments

`MetaMoAgent(record_transitions=True)` freezes a forecast before each action,
records the observed outcome, and updates the diagnostic predictor afterward.
Ordinary moves use the known GridWorld rules. After mineral collection, the
predictor learns the median proximity of the respawned mineral from a bounded
history. OpenPsi appraises the predicted stimulus to forecast the next
motivational response. These forecasts do not select live actions.

The retained experiments cover prediction accuracy, replication, and whether
forecasts help decisions:

| Experiment | Purpose |
|---|---|
| `evaluation.self_model` | Compare rules, learned collection statistics, contextual/action averages, and persistence on held-out episodes; verify unchanged live behavior. |
| `evaluation.collection_robustness` | Replicate collection forecasting across independent seed splits with minimum sample counts. |
| `evaluation.future_conflict` | Predict future task/safety disagreement and audit changed choices through paired cloned rollouts. |
| `evaluation.conflict_energy_budget` | Test whether conflict predicts energy depletion better than a direct energy-and-distance reference. |

Run from the repository root:

```powershell
python -m applications.gridworld.evaluation.self_model
python -m applications.gridworld.evaluation.collection_robustness
python -m applications.gridworld.evaluation.future_conflict --rollout-horizon 10
python -m applications.gridworld.evaluation.conflict_energy_budget --output artifacts/self_model/conflict_energy_budget.json
```

Collection learning improved prediction in the completed replication. Future
conflict was also predictable, but the tested conflict penalty did not reliably
improve reward or safety. The experimental energy world adds an energy cost per
move and a depletion penalty; it is not the standard GridWorld environment.
Across 137 audited states, its current-conflict rule, learned-conflict rule, and
direct energy reference produced 61, 62, and 56 depleted continuations,
respectively. These are correlated branch outcomes, not independent episodes.

The energy experiment's settings and results are retained in
[`conflict_energy_budget.summary.json`](../../artifacts/self_model/conflict_energy_budget.summary.json).
Full candidate traces are regenerable and ignored by Git. Conflict accuracy is
scored only where all future decisions occur; safety measurements include early
depletion. No neural network or learned intervention is enabled in the agent.

The earlier valence-only ranking, short/long-horizon wrapper, and
boundary-pressure probe have been retired. Their findings remain constraints on
future work: improved mood prediction did not imply better task choices, and
the simple pressure reference outperformed the tested statistical corrections.
