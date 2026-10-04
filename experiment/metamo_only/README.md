# MetaMo-only learned appraisal experiment

This deterministic resource-choice task isolates MetaMo appraisal from
Q-learning and LLMs. Each case presents three possible actions: gather, rest,
or inspect. The environment supplies a separate one-step outcome utility, but
only MetaMo chooses the action. A fixed catastrophic-risk filter is applied to
all comparison modes before MetaMo scores candidates.

Run from the repository root:

```powershell
python -m experiment.metamo_only.experiment `
  --train-cases 120 --validation-cases 40 --test-cases 60 `
  --seed 100 `
  --challenge-cases 10 --challenge-pool 5000 `
  --output-dir experiment/results/metamo_only-run-1
```

The command writes a trained `model.npz`, its checksummed `manifest.json`,
`targets.csv`, `test_cases.csv`, and `summary.json`. When a challenge set is
requested it also writes `challenge_cases.csv`. Use a fresh output directory
for each run; existing results are never overwritten.

The full experiment follows one path:

1. Run existing rule appraisal and MAGUS action scoring. The scenario's
   one-step utility identifies an independent outcome-preferred action.
2. Search small normalized corrections to the six rule-appraised modulators.
   Every candidate passes through the *same* direction, coordinate, boundary,
   and L2 caps as the runtime hybrid appraiser. Keep zero when the rule already
   chooses the preferred action or no searched correction helps.
3. Train a 24-hidden-unit NumPy MLP to predict those normalized targets from
   the manifest-ordered goal, modulator, stimulus, rule-output, and boundary
   features. The loss is weighted residual MSE with small weight decay;
   validation MSE selects the saved epoch. Train, validation, and test cases
   use independent seeds.
4. Reload the saved model into the existing `HybridOpenPsiAppraisal`. Compare
   rule, shadow, and enabled modes through MetaMo's normal transition and
   report action changes, outcome utility, oracle agreement, risk, and applied
   correction size. Shadow must exactly match rule action and state.

The experiment is deliberately honest about no-effect results. A nonzero
correction does not necessarily cross a MAGUS action-score boundary, and this
one-step synthetic task cannot prove long-horizon benefit. No GridWorld agent,
Q update, law audit, goal-update rule, or core safety code is modified by this
experiment.

The challenge cases are sampled with a separate seed and selected only when
the exact bounded target switches the rule action. They test mechanism
sensitivity, not population-level performance, and are never used for fitting.

For a larger held-out replication, use 1,000 training, 200 validation, and
500 test cases and repeat with independent `--seed` values (for example,
100, 101, and 102). Keep a fresh output directory for each seed and compare
ordinary test cases separately from selected challenge cases. The smoke tests
run from the repository root with
`python -m unittest experiment.metamo_only.test_experiment`.
