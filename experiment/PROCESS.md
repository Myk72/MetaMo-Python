# How the two learned-appraisal experiments work

## The shared hybrid mechanism

MetaMo's existing rule appraiser first computes its normal modulator state
`M_rule`. The neural network sees the current goals, modulators, stimulus,
rule-appraised modulators, and boundary-pressure signal. It predicts six
normalized correction coordinates. The hybrid wrapper scales those predictions
by small per-modulator bounds, enforces one-sided safety directions for
`threshold` and `securing`, attenuates other corrections near the safety
boundary, and applies an overall norm cap. The resulting state is:

```text
M_hybrid = clip(M_rule + bounded_correction, 0, 1)
```

For example, a normalized `approach` prediction of `+0.6` with a `0.05`
coordinate bound proposes `+0.03`. If boundary pressure is `0.5`, the guard
reduces that non-safety correction to `+0.015` before applying the norm cap.
The network cannot set goals, replace MAGUS, remove hard safety, or change the
goal-update and law-audit rules. Invalid predictions fall back to rule
appraisal. `shadow` mode computes predictions but applies no correction; it
must behave exactly like the rule system.

Both tasks use the same small 24-hidden-unit residual MLP. Its supervised loss
is weighted mean-squared error against normalized correction targets, plus
small weight decay. Validation error selects the saved epoch. A low loss is
only a model-fitting check; it is **not** evidence that decisions improved.

## Experiment 1: MetaMo-only resource choice

This task removes RL. Each independently seeded case presents gather, rest,
and inspect choices, with opportunity, hazard, fatigue, urgency, and
uncertainty signals. A fixed catastrophic-risk filter removes ineligible
choices in *all* modes. MetaMo alone chooses among the remaining actions.
The environment computes a separate one-step utility for each action; the
highest-utility eligible action serves as the diagnostic oracle.

The process is:

1. Run rule appraisal and MAGUS. Record the rule action and oracle action.
2. Try zero and a fixed set of small correction vectors through the *runtime*
   bound function, then replay MAGUS. If the rule already chooses the oracle,
   use a zero target. Otherwise choose the smallest searched correction that
   switches the action. If none switches it, retain only a material improvement
   in the oracle's score margin; if none helps, use zero.
3. Train on one seed group, choose the epoch on a separate validation group,
   and evaluate on an untouched test group. Reload the saved, checksummed
   model and compare rule, shadow, and enabled MetaMo transitions.
4. Optionally sample an independent *challenge stratum* containing cases for
   which an exact bounded target can switch the action. Replay each exact
   target and the NN separately. This checks whether a decision change is
   possible and whether the network learned it. Because cases are selected by
   feasibility, challenge-set rates are **not** population performance rates.

If an exact target switches an action but the NN does not, the mechanism has
some decision authority but the current model, targets, or data are inadequate.
If even exact targets rarely switch actions on ordinary cases, training a
larger network alone will not solve the reachability problem.

## Experiment 2: GridWorld

GridWorld tests the same idea in an application with a Q learner and a
task/safety-composed MetaMo selector. Experimental code stays here; the thin
opt-in adapter in `applications/gridworld/learned_appraisal.py` attaches a
predictor to an isolated runtime. Ordinary GridWorld runs remain rule-based.

The process has three commands, documented in `gridworld/README.md`:

1. **Collect.** Run rule-only episodes with disjoint train, validation, test,
   and out-of-distribution environment-seed groups. Store each transition's
   base, safety, and growth appraisal contexts and one-step counterfactual
   outcomes for all four actions. Replay the behavior action to check that
   stored counterfactuals match the real environment transition.
2. **Generate targets.** A safety-first one-step oracle ranks the stored
   outcomes. For each appraisal context, replay rule scores, try small
   normalized corrections through the same runtime guard, and retain the
   minimum-intervention action switch or a margin-only improvement. Store
   zero when no correction helps. This oracle is a diagnostic reference, not
   a replacement for GridWorld reward or long-horizon evaluation.
3. **Train and test.** Fit the shared MLP on training targets, select the epoch
   on validation targets, and audit held-out test and OOD contexts. This first
   measures whether corrections change **MAGUS** choices. Then train Q once
   per agent seed, freeze the Q table, and compare rule, shadow, and enabled
   appraisal on paired episode seeds. Matched counterfactual probes also
   compare the actual rule and enabled **selector** on identical frozen-Q
   states. The episode comparison measures reward, completion, lava exposure,
   unsafe-zone rate, and survival. The application-level compositionality
   audit remains active.

These levels must stay separate: a correction can change a MAGUS score yet
still fail to pass GridWorld's Q shortlist or composed selector. A selector
change can, in turn, fail to improve a complete episode. Only the last step
supports a GridWorld outcome claim.

## How to interpret a result

Read the results in this order: exact-target switch feasibility, NN action
changes on held-out cases, paired outcome changes, then safety. Report
ordinary, challenge, and OOD cases separately. Use multiple independent
seeds for uncertainty intervals; a one-seed pilot only verifies the pipeline.
Nonzero corrections, low MSE, or a selected challenge-set success rate alone
do not show a significant application benefit.

Run commands are in `metamo_only/README.md` and `gridworld/README.md`. Current
pilot observations and their limits are in `RESULTS.md`.
