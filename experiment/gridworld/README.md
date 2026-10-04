# GridWorld learned-appraisal experiment

This experiment uses the existing GridWorld application without changing its
default Q learner, decision selector, goal update, law audit, or safety guard.
The only application adapter is `applications/gridworld/learned_appraisal.py`.

Run from the repository root. Use fresh output names on repeated runs:

```powershell
python -m experiment.gridworld.collect `
  --train-seeds 0 --validation-seeds 1 --test-seeds 2 --ood-seeds 100 `
  --episodes-per-seed 1 --max-steps 10 `
  --output-dir experiment/results/gridworld-data-1

python -m experiment.gridworld.targets experiment/results/gridworld-data-1 `
  --random-samples 0 `
  --output-dir experiment/results/gridworld-targets-1

python -m experiment.gridworld.train_test experiment/results/gridworld-targets-1 `
  --epochs 50 --q-train-episodes 2 --eval-episodes 2 `
  --agent-seeds 0,1 --max-steps 20 `
  --output-dir experiment/results/gridworld-model-1
```

These values are a functional pilot, **not** enough for statistical inference.
Expand the seed groups and episode counts only after checking target
feasibility; the bounded target search is the costliest stage. The script
reports seed-group bootstrap intervals when there are at least two groups.

Collection uses disjoint environment-seed groups and stores the rule-appraised
base, safety, and growth perspectives plus one-step outcomes for all actions.
The target generator tries small normalized residuals through the *same*
direction, coordinate, boundary, and norm caps as the enabled wrapper. It
keeps zero when the oracle is already selected, selects the smallest feasible
action-switch correction, or records a margin-only improvement when switching
is not found. The safety-first one-step oracle is a diagnostic reference, not
the GridWorld reward objective.

Training uses a 24-unit residual MLP with weighted MSE and validation-based
epoch selection. Test and OOD cases are never used for fitting. Decision replay
reports whether the trained correction changes MAGUS's choice and whether the
exact target could have done so. A separate episode audit trains Q once per
agent seed, freezes it, and evaluates rule, shadow, and enabled appraisal on
paired environment seeds. Matched selector probes also compare rule and
enabled actions on identical frozen-Q states visited by the rule policy. The
shadow arm must match rule behavior.

Decision-replay gains and episode gains are distinct claims: the Q shortlist
and task/safety composition can prevent a MAGUS score change from affecting
the actual GridWorld action. Do not treat a low validation MSE or a nonzero
correction alone as evidence of improved decisions. Use multiple agent seeds
and report paired reward, completion, lava, unsafe-zone, and survival outcomes.

For a focused follow-up after a larger run, replay only the saved
`action_switch_feasible` test and OOD targets against the actual selector:

```powershell
python -m experiment.gridworld.check_effect `
  experiment/results/YOUR_TARGETS experiment/results/YOUR_MODEL `
  --splits test,ood --agent-seeds 0,1,2,3,4 `
  --q-train-episodes 20 --max-steps 50 `
  --output-dir experiment/results/YOUR_EFFECT_CHECK
```

This check compares rule appraisal, the exact bounded target, and the trained
NN on the same stored context and frozen Q table. It records direct MAGUS
choices, raw and normalized task/safety scores, Q shortlists, tie sets, and
executed actions. The exact target is applied only when its input matches the
stored context; it is a reachability diagnostic, not a deployable policy.

To test whether *any searched legal correction* can change the final composed
selector, independent of the NN or direct-MAGUS target, run:

```powershell
python -m experiment.gridworld.selector_reachability `
  experiment/results/YOUR_TARGETS `
  --splits test,ood --max-records-per-split 30 `
  --agent-seeds 0,1,2 --q-train-episodes 20 --max-steps 50 `
  --random-samples 0 `
  --output-dir experiment/results/YOUR_SELECTOR_CHECK
```

The deterministic search tries axis, coordinate-pair, and corner corrections
through the runtime's actual bounds. It holds the application Q table, safety
filter, and tie rule fixed, and verifies its baseline scores against the real
selector. The output distinguishes a raw branch-score shift, a surviving
normalized-score shift, a changed tie set, and a robust action switch. An
unchanged result under this finite search is evidence of low sampled control
authority, not a proof that no legal correction exists anywhere.

To isolate how much of that bottleneck comes from recomputing the branch
normalization scale after each correction, run the paired experiment-only
ablation:

```powershell
python -m experiment.gridworld.normalization_ablation `
  experiment/results/YOUR_TARGETS `
  --splits test,ood --max-records-per-split 30 `
  --agent-seeds 0,1,2 --q-train-episodes 20 --max-steps 50 `
  --random-samples 0 `
  --output-dir experiment/results/YOUR_NORMALIZATION_ABLATION
```

Both arms use the same states, Q tables, legal residuals, safety filter, and
tie draws. The fixed-reference arm divides each centered corrected branch by
the *rule baseline's* scale, while the current arm recomputes that scale after
correction. Their zero-residual decisions are identical. The CSV saves
minimum-norm witnesses for changed tie sets and matched-RNG actions, plus
whether those actions gain or lose one-step oracle agreement. Fixed-reference
scoring is only a counterfactual in this experiment; it is not enabled in the
GridWorld application.

All generated data, models, and summaries go under `experiment/results/`,
which is ignored by Git. The source code and instructions stay in this folder.
