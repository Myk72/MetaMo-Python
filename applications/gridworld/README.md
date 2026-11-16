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
| `evaluation.neural_self_model` | Compare a 32-unit neural network with rules and linear regression for five-step changes in all goals and modulators. |
| `evaluation.neural_robustness` | Evaluate the saved forecasters on fresh layouts, separating ordinary moves, collection, nearby lava, and large state changes. |
| `evaluation.neural_weighted` | Test whether emphasizing large-change examples during training improves the same NN without materially degrading ordinary forecasts. |
| `evaluation.neural_learning_curve` | Compare the same unweighted NN and ridge on nested 80/160/320-episode training sets, with shared fresh validation/test episodes. |
| `evaluation.neural_action_ranking` | Audit frozen 320-episode forecasts on alternative actions in cloned five-move branches, with a calibrated task readout. |
| `evaluation.neural_closed_loop` | Compare the original selector, direct predictor and NN/readout across complete matched episodes with frozen models. |

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

### Neural prediction of motivational changes

This optional experiment predicts the actual committed motivational state five
moves ahead under the existing frozen policy. It uses the current state,
selected action, stimulus, candidate estimates, selector scores, and the known
one-step transition. The neural network learns a correction to that transition;
it has one hidden layer of 32 tanh units and 13 outputs (seven goals and six
modulators). Training uses PyTorch's
[AdamW optimizer](https://docs.pytorch.org/docs/2.13/generated/torch.optim.AdamW.html).

```powershell
# Optional dependency, only needed for this experiment:
python -m pip install torch
python -m applications.gridworld.evaluation.neural_self_model
```

Each of three independent splits pretrains Q for 30 episodes, freezes the
policy, and collects 80 training, 20 validation, and 24 test episodes. Five-step
windows never cross episode boundaries. Both learned models receive the same
features. Scaling uses training data only; validation selects the ridge penalty,
neural epoch/initialization, and rule reference before test episodes are collected.
Rule references include persistence, holding the known one-step forecast, and
extrapolating its change. The primary score gives equal weight to goal and
modulator mean absolute error; individual coordinates are also reported.

The run writes a summary to
`artifacts/self_model/neural_five_step.summary.json` and regenerable datasets,
weights, validation histories, and test predictions under
`artifacts/self_model/neural_five_step/`. Error intervals resample whole test
episodes because adjacent prediction windows overlap. These forecasts are
evaluated on actions the existing policy selected; they do not establish better
action selection or accuracy for arbitrary counterfactual actions.

The initial run's mean balanced errors were 0.005206 for the selected rule
(linear extrapolation), 0.003037 for ridge regression, and 0.002920 for the neural
network. Neural goal and modulator errors were lower in every split, but two of
the three episode-bootstrap intervals for its improvement over ridge included
zero. The gain over the rule reference is clear in this experiment; the smaller
gain over linear regression remains preliminary. Full coordinate errors and
selection details are in
[`neural_five_step.summary.json`](../../artifacts/self_model/neural_five_step.summary.json).

### Robustness of the saved neural model

After the neural experiment has saved its datasets and weights, run:

```powershell
python -m applications.gridworld.evaluation.neural_robustness
```

This evaluation needs only NumPy. It loads the existing neural and ridge
weights, scaling, and validation-selected rule without fitting or selecting a
new forecaster. It reconstructs each Q policy from its original seeded
pretraining, checks an original training episode exactly, and checks that saved
weights reproduce the original test predictions. Evaluation then uses a fresh
tie-breaking random stream with Q and the existing diagnostic learner frozen.

The default run evaluates 60 new episodes for each of the three saved models.
Both the episode seeds and initial lava layouts must differ from the original
training, validation, and test runs, including Q pretraining. All forecasters
receive the same inputs, captured before execution. A mirror agent verifies
that recording changes neither actions nor the resulting trajectory.

The report separates these situations:

| Situation | Definition |
|---|---|
| Ordinary movement | The selected move does not collect a mineral, and both current and destination cells are more than two steps from lava. |
| Collection | The selected first move collects a mineral; this is known before execution. |
| Near lava | The current or selected destination cell is within Manhattan distance two of lava. |
| Large change | The observed five-step change exceeds or equals the original training set's 90th percentile of balanced absolute change. |

Groups overlap. Large-change membership is a retrospective diagnostic, never
an input to the predictor. Its threshold, all group definitions, source hashes,
and seeds are written to `protocol.json` before fresh outcomes are collected.
For each group and split, the report includes goal/modulator MAE, every
coordinate's MAE, the 95th percentile and maximum of window error, and paired
neural comparisons against ridge and the previously selected rule. Positive
error reductions favor the neural model. Comparisons include episode win
fractions and the fraction of windows where the neural model is worse.

Bootstrap intervals resample whole episodes containing group observations,
since five-step windows overlap. Their point estimates weight episodes equally;
pooled MAE weights windows. Groups observed in fewer than ten episodes per
split are flagged as insufficient coverage. This coverage rule is descriptive,
not a statistical power guarantee. Subgroup intervals are exploratory and are
not adjusted for multiple comparisons. Only complete forecast horizons are
scored, and these layouts follow the original environment distribution: this
does not establish robustness to changed physics or untested actions.

The summary is `artifacts/self_model/neural_robustness.summary.json`. Raw fresh
datasets, predictions, group masks, and detailed per-split reports are saved
under `artifacts/self_model/neural_robustness/`. No neural predictions enter
MetaMo's action selector. Override `--source`, `--output-dir`, `--summary`, or
`--episodes-per-split` when reproducing the experiment with separate artifacts.

The completed run covered 180 fresh episodes and 6,480 complete forecast
windows. Pooled balanced MAE was 0.005597 for the previously selected rule,
0.003226 for ridge, and 0.003123 for the NN. The NN's pooled error was 3.2%
lower than ridge overall and 5.6% lower near lava, but 7.0% higher for large
motivational changes. The large-change disadvantage appeared in every split;
each of its episode-bootstrap intervals still included zero. Two of the three
overall NN-versus-ridge intervals also included zero. These results support
continued comparison with ridge, not replacing it based solely on average
error. Per-coordinate results and uncertainty are retained in
[`neural_robustness.summary.json`](../../artifacts/self_model/neural_robustness.summary.json).

### Weighting large changes during neural training

With the original neural artifacts and robustness protocol available, run:

```powershell
python -m applications.gridworld.evaluation.neural_weighted
```

This controlled follow-up reuses the original training/validation episodes,
training-only scaling, 32-unit network, AdamW settings, and three initialization
seeds per split. An example is a large change if its actual five-step balanced
absolute state change is at least the training set's 90th percentile. Test
outcomes never determine this threshold. The experiment tries weights of
1, 2, 4, and 8 for those training examples, with weight 1 for all other examples.

For a batch of B examples, training minimizes
`mean_i((w_i / mean_training_weight) * mean_coordinate(squared_normalized_residual_error))`.
Normalizing by the full training set's mean weight keeps the expected loss
scale comparable. The 1x candidate follows the original loss computation
exactly and must reproduce the saved model's validation predictions, selected
epoch, and initialization. Weighting changes training emphasis, not the network's
inputs, target, or architecture. Actual change is never required at inference.

For each strength, epoch and initialization selection still minimize overall
validation balanced MAE, as in the original experiment. Strength selection
then minimizes large-change validation MAE, subject to both ordinary and
overall validation MAE staying within 2% of the 1x control. Weight 1 remains an
eligible fallback. The 2% tolerance is an experimental design choice, not an
established acceptable degradation for deployed agents.

All three splits' selected weights and candidate models are saved before any
fresh test trajectories are generated. The test uses another 60 episodes per
split, with seeds and lava layouts excluded from both preceding experiments.
The existing mirror checks verify unchanged live actions and frozen Q/online
statistics. Comparisons include the original NN, ridge, and rule reference;
unselected strengths are reported only as diagnostics, with no test-driven
reselection. Success at the point-estimate level requires lower large-change
error than the original NN and both ordinary/overall errors within the 2%
tolerances in every split. Episode-bootstrap intervals are reported separately;
passing this point-estimate rule alone does not establish statistical certainty.

Outputs go to `artifacts/self_model/neural_weighted/`: the protocol, selections,
candidate weights and training histories, new reference trajectories, predictions,
and detailed reports. The reviewed summary is
`artifacts/self_model/neural_weighted.summary.json`. The experiment requires the
same optional PyTorch dependency as the original neural training run. It tests
forecast accuracy; it does not change MetaMo's action selection.

The completed run selected strengths 4x, 2x, and 1x across the three splits;
every weighted candidate in the third split violated a validation guard.
On 180 new episodes (6,480 windows), the selected models reduced pooled
large-change MAE from 0.003243 to 0.003110 (4.1%), while ordinary-movement
error increased by 0.74% and overall error was essentially unchanged.
Ridge still had lower large-change MAE (0.003014). The first two splits passed
the point-estimate criterion, but only the second had a positive entire
episode-bootstrap interval for large-change improvement over the original NN.
The third retained its original model, so the predeclared requirement of
improvement in every split was not met. Weighting gave a modest partial gain,
not a consistent solution to large-change prediction. Full results are in
[`neural_weighted.summary.json`](../../artifacts/self_model/neural_weighted.summary.json).

### Learning curve with more training episodes

After the preceding experiments, run:

```powershell
python -m applications.gridworld.evaluation.neural_learning_curve
```

Each of the three splits starts with its original 80 training episodes and
collects 240 more under the same frozen Q policy. Training sets contain 80,
160, and 320 complete episodes: every smaller set is contained in the next.
Each split also receives 40 fresh validation episodes and 60 fresh test episodes,
shared across all training sizes and forecasters. New seeds and lava layouts
are checked against all preceding neural experiments and against each other.
The default seed block was chosen using layout uniqueness alone, before looking
at any trajectory outcomes.

Both the NN and ridge are retrained at each size. Each receives identical
examples and its scaling is fitted only on that training subset. The network
keeps its 32-unit architecture, unweighted loss, AdamW settings, three
initialization seeds, and maximum of 300 epochs. The new shared validation set
chooses each NN's epoch/initialization and each ridge penalty. Consequently the
80-episode model is a new controlled baseline, not the earlier checkpoint
selected with an older validation set. All sizes and all splits finish model
selection before any test episode is generated. No test error selects a size.

The large-change threshold is computed once from the original 80 training
episodes and held fixed for every size. Reports compare every coordinate and
situation, each NN against ridge trained on the same data, and the 320-episode
NN against the smaller NNs. The point-estimate success criterion requires a
large-change improvement from 80 to 320 episodes, with ordinary and overall
errors no more than 2% worse in every split. Beating ridge on large changes is
reported separately. Paired episode-bootstrap intervals quantify uncertainty;
overlapping prediction windows are not treated as independent samples.

This tests the practical benefit of more training experience. A larger dataset
also permits more optimizer updates per epoch, so it is not a fixed-compute
sample-efficiency experiment. It uses one nested data sequence per policy split;
it does not isolate training-data ordering or establish benefits for decisions.

Outputs go to `artifacts/self_model/neural_learning_curve/`: the protocol and
layout hashes, additional training data, shared validation/test data, selected
models and histories, group masks, and predictions. The compact summary is
`artifacts/self_model/neural_learning_curve.summary.json`. Source datasets and
the original models are preserved. The same optional PyTorch dependency is
required for training; saved models also support NumPy inference.

The completed run evaluated every model on the same 180 new test episodes
(6,480 forecast windows). Pooled balanced MAE was:

| Training episodes per split | NN | Ridge |
|---|---:|---:|
| 80 | 0.002883 | 0.003003 |
| 160 | 0.002741 | 0.002995 |
| 320 | 0.002611 | 0.002961 |

From 80 to 320 episodes, NN error decreased by 9.4% overall, 11.3% on ordinary
movement, and 12.8% on large changes. The predeclared more-data criterion passed
in all three splits. Episode-bootstrap intervals for the 320-versus-80 NN
improvement were positive in every split for all three of those groups.
The 320-episode NN also beat matched ridge overall in every split, with positive
intervals, and its pooled overall error was 11.8% lower. Large changes remain a
limitation: pooled NN error was 0.002399 versus ridge's 0.002311 (NN 3.8% higher),
and the NN beat ridge on that subgroup's mean error in only one split. These
results support more training experience for this forecaster, while retaining
ridge as a comparator for large changes. Full results and selections are in
[`neural_learning_curve.summary.json`](../../artifacts/self_model/neural_learning_curve.summary.json).

### Can state forecasts rank actions usefully?

After the learning-curve experiment has saved its 320-episode models, run:

```powershell
python -m applications.gridworld.evaluation.neural_action_ranking
```

The state NN does not predict rewards or collections directly. This audit
therefore separates forecasting from the conversion of a future state into a
task score. The fixed objective is `100 * mineral collections - cumulative
energy loss` over **five total moves**, including the candidate first action.
Energy loss sums decreases before recharge can hide earlier damage. Minerals
spawn outside lava, so collection and lava loss cannot coincide in one step.

At every fifth live decision with five moves remaining and at least two
eligible actions, reconstruct the original 44 input features for every action
in MetaMo's existing safety-filtered Q shortlist. The reconstruction must match
the original frozen input for the action actually selected. Freeze all model
predictions, then force each candidate in a cloned environment and agent and
follow the existing frozen policy for four more moves. Three continuations per
candidate share random conditions across actions. The first preserves the
snapshot RNG; two others reseed future randomness while keeping the current
world fixed. The untouched live policy continues independently; mirror checks
verify unchanged actions, state, reward, and RNG.

For each of three policy splits, 20 separate calibration episodes fit two
fixed-penalty ridge readouts (penalty 1, training-only standardization):

- A shared state readout maps actual future motivational change, current state,
  and known first-move collection/hazard cues to utility. At test time, apply
  the same readout to NN, regression, and nominal-trend forecasts.
- A direct baseline maps the original 44 pre-action features to utility using
  the same calibration branches. This checks whether the intermediate state
  forecast contributes useful information.

Freeze every readout before collecting 24 held-out episodes per split. Also
score the readout with actual future states as a diagnostic: poor ranking here
would implicate the state representation/readout, not merely forecast error.
This diagnostic is unavailable before acting and is not an upper bound on
performance. Compare all methods with MetaMo's actual selected action. Original
forecast weights and Q values remain frozen throughout; no PyTorch is needed.

Report selected-versus-alternative state-prediction error, action regret,
collections, energy loss, changed choices, and pairwise ranking accuracy on
non-tied outcomes. Report best-action agreement only on states with different
candidate utilities; equal predicted scores retain the live choice when
possible. Utility is averaged over the three continuations. Regret uses the
best sampled candidate mean, so finite-rollout noise can make it optimistic.
The current selector's pairwise ranking uses its composed candidate scores;
its evaluated choice remains the recorded action, including randomized ties.
Paired bootstrap intervals resample complete source episodes. The criterion
for NN decision support requires a positive lower confidence bound on utility
gain over the current selector in every split, without higher mean energy loss.

This is a five-step shadow audit, not repeated neural control of an episode.
Early-terminal utility remains in the audit; undefined full-horizon state
targets are masked and excluded from readout fitting. States with only one
eligible action are outside the ranking comparison. If safety filtering leaves
no energy-loss outcomes, the run cannot demonstrate a safety improvement.
Inputs for alternative actions may be outside the NN's original on-policy
training distribution; their prediction errors are reported separately.

Raw calibration/test branches, readout coefficients, predictions and decisions
are saved under `artifacts/self_model/neural_action_ranking/`; the reviewed
summary is `artifacts/self_model/neural_action_ranking.summary.json`. Seed and
lava-layout checks exclude all preceding neural experiments. The source
checkpoints and live selector are preserved.

The completed audit covered 374 multi-action situations from 72 held-out
episodes, with 3,528 five-move branches. Mean mineral collections per selected
branch were:

| Method | Collections |
|---|---:|
| Current selector | 0.201 |
| NN forecast + state readout | 0.374 |
| Regression forecast + same readout | 0.373 |
| Nominal-trend forecast + same readout | 0.386 |
| Direct input-to-utility regression | 0.397 |
| Actual-future-state readout (diagnostic only) | 0.443 |

NN utility gains over the current selector had positive episode-bootstrap
intervals in all three splits. Its comparisons with regression all included
zero; the direct predictor had higher mean utility in every split, with a
clear interval in the second. Thus the short-horizon improvement does not
establish a decision advantage specific to the NN. The actual-state diagnostic
was close to the best sampled candidate, so this readout can use future-state
information effectively when supplied accurate states.

The NN's state MAE was 0.002648 on selected actions and 0.003431 on alternatives;
regression's corresponding errors were 0.002983 and 0.004168. Better alternative-
state accuracy therefore did not translate into a clear ranking advantage.
All branches completed the horizon and none lost energy, preventing a claim of
safety improvement. These are isolated first-action interventions with the old
policy continuing afterward, not whole episodes controlled by the readout.
Results are retained in
[`neural_action_ranking.summary.json`](../../artifacts/self_model/neural_action_ranking.summary.json).

### Repeated action selection over complete episodes

After the action-ranking audit, run:

```powershell
python -m applications.gridworld.evaluation.neural_closed_loop
```

This evaluation reuses the saved 320-episode NN and the audit's calibration-only
state/direct readouts. There is no further fitting. Three policy splits each
receive 60 fresh environments; all three controllers start from identical
environment and agent RNG seeds in each environment (540 controller episodes).
Subsequent trajectories and random draws may diverge as their actions differ.
Seeds and initial lava layouts exclude all preceding neural experiments.

At every move, ask the original selector for its proposed action and construct
the existing safety-filtered Q shortlist. The baseline executes its proposal.
The direct controller maximizes its predicted utility from the 44 current
features. The NN controller predicts each candidate's five-step motivational
state and maximizes the shared state readout's utility. Ties retain the original
proposal where possible. When the action changes, the experiment stages and
commits the motivational transition for that executed action; it never inserts
the predicted future state into MetaMo. The final four moves revert to the
original selector because a full five-move forecast horizon is unavailable.
This override is confined to the evaluator; the application's selector is unchanged.

Measure total minerals, cumulative energy loss, lava contacts, energy depletion,
zero-collection episodes, reward, and changed choices across complete episodes.
Paired episode-bootstrap intervals compare both controllers with the baseline
and the NN with the direct controller. Support requires a positive mineral-gain
interval in every split without increased mean energy loss or depletion.
All Q values, model weights, readouts and online statistics remain frozen.

The original NN assumes continuation of MetaMo's old policy, so repeated new
choices change its forecasting conditions. Record its five-step state error
under each controller as a diagnostic, including the different states visited;
do not treat these differing trajectories as identical prediction tasks.
No observed energy loss means the experiment cannot show a safety improvement.

The run saves the protocol, per-episode metrics and complete action/forecast
traces under `artifacts/self_model/neural_closed_loop/`, and a reviewed summary
at `artifacts/self_model/neural_closed_loop.summary.json`. Tests verify baseline
replay, correct action-specific motivational commits, shortlist preservation,
tail fallback and paired analysis. The experiment needs only NumPy.

The completed comparison used 180 matched environments and 540 complete
40-move controller episodes. Pooled results were:

| Controller | Mean minerals per episode | Zero-collection episodes |
|---|---:|---:|
| Original MetaMo selector | 2.128 | 19.4% |
| Direct utility predictor | 2.883 | 10.0% |
| NN forecast + state readout | 4.311 | 2.8% |

The NN controller's mineral-gain intervals over MetaMo were positive in every
split. It also beat the direct controller with positive intervals in splits
two and three; the first split's interval included zero. The direct controller
failed the all-split criterion: in split two it collected 1.733 minerals versus
MetaMo's 2.450, with a negative paired interval. This reverses the impression
from the shorter branch audit, where its pooled performance was best.

No controller lost energy, entered lava, or depleted its energy in these runs,
so the results do not establish safety improvements. NN state MAE was 0.002934
under the original selector, 0.004926 under the direct controller, and 0.004447
under NN control. These diagnostics concern different visited states and
continuation policies; the behavioral improvement does not imply better state
prediction under intervention. All 540 traces passed checks for shortlist
membership, tail fallback, executed-action counts, rewards, and horizon labels.
The controllers remain evaluation-only. Results are retained in
[`neural_closed_loop.summary.json`](../../artifacts/self_model/neural_closed_loop.summary.json).

The earlier valence-only ranking, short/long-horizon wrapper, and
boundary-pressure probe have been retired. Their findings remain constraints on
future work: improved mood prediction did not imply better task choices, and
the simple pressure reference outperformed the tested statistical corrections.
