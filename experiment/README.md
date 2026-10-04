# Hybrid learned appraisal experiments

This folder contains the complete experimental workflow for bounded neural
corrections to MetaMo's existing rule appraisal. It is **not** an application:
no model is loaded by default, and no core decision, goal-update, law-audit,
or safety behavior is replaced. Both tasks use the same residual MLP and the
same `openpsi.hybrid` projection and fail-closed runtime contract.

- `metamo_only/`: a deterministic resource-choice task in which MetaMo alone
  selects gather, rest, or inspect. Run its experiment as described in its
  README. An optional independently seeded challenge stratum contains cases
  where an exact bounded target can switch the action; it is reported apart
  from ordinary held-out cases.
- `gridworld/`: grouped GridWorld collection, bounded target generation,
  training, decision replay, and frozen-Q episode evaluation. See its README.
- `network.py`: one small NumPy residual MLP shared by both tasks.

The minimum causal comparison is rule versus shadow versus enabled appraisal.
Exact searched targets provide a reachability upper bound. Report action
switches and outcomes, not only residual size or prediction loss. Keep ordinary
test cases, challenge cases, and GridWorld OOD cases separate; no challenge
case is used in model fitting.

`RESULTS.md` records the current pilots and their limits.
`PROCESS.md` explains the two workflows and how to interpret their outputs.

The experiments depend on the repository's `openpsi/hybrid.py` and
`openpsi/learned.py`, and GridWorld evaluation also depends on
`applications/gridworld/`. Those reusable/runtime files are outside this
folder; all experiment-specific runners, datasets, model artifacts, reports,
and tests belong here. Generated output is written to `experiment/results/`.
