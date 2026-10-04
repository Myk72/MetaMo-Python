"""Checks for the experiment-only fixed-reference selector comparison."""

import tempfile
from pathlib import Path
import unittest

import numpy as np

from applications.gridworld.learned_appraisal import gridworld_learned_appraisal_manifest
from experiment.gridworld.check_effect import _train_frozen_q
from experiment.gridworld.collect import CollectionConfig, collect_dataset
from experiment.gridworld.normalization_ablation import (
    _audit_record,
    _fixed_branch,
    _reference_scale,
)
from experiment.gridworld.targets import _iter_jsonl


class NormalizationAblationTests(unittest.TestCase):
    def test_fixed_reference_keeps_scale_change(self):
        actions = [0, 1, 2]
        rule = np.asarray([-1.0, 0.0, 1.0])
        corrected = 2.0 * rule
        scale = _reference_scale(rule, actions)
        np.testing.assert_allclose(_fixed_branch(rule, actions, scale), rule)
        np.testing.assert_allclose(_fixed_branch(corrected, actions, scale), corrected)

    def test_smoke_replays_real_selector_and_matched_rng(self):
        with tempfile.TemporaryDirectory() as root:
            dataset = Path(root) / "data"
            collect_dataset(dataset, CollectionConfig(
                train_seeds=(0,), validation_seeds=(1,),
                test_seeds=(2,), ood_seeds=(3,),
                episodes_per_seed=1, max_steps=1,
            ))
            record = next(
                row for row in _iter_jsonl(dataset / "transitions.jsonl")
                if row["split"] == "test"
            )
            agent = _train_frozen_q(0, 1, 5, 0.55)
            result = _audit_record(
                record, agent, gridworld_learned_appraisal_manifest(), 0, 0,
            )
            self.assertGreater(result["evaluations"], 0)
            self.assertGreaterEqual(result["fixed_max_score_shift"], 0.0)


if __name__ == "__main__":
    unittest.main()
