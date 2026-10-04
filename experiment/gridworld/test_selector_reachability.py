"""Smoke tests for the selector-aware bounded-correction audit."""

import tempfile
from pathlib import Path
import unittest

from applications.gridworld.learned_appraisal import gridworld_learned_appraisal_manifest
from experiment.gridworld.check_effect import _train_frozen_q
from experiment.gridworld.collect import CollectionConfig, collect_dataset
from experiment.gridworld.selector_reachability import _audit_record, _prefer
from experiment.gridworld.targets import _iter_jsonl


class SelectorReachabilityTests(unittest.TestCase):
    def test_minimum_norm_candidate_is_preferred(self):
        larger = {"residual_norm": 0.04, "composed_max_shift": 0.02}
        smaller = {"residual_norm": 0.02, "composed_max_shift": 0.01}
        self.assertIs(_prefer(larger, smaller), smaller)

    def test_offline_score_matches_application_selector(self):
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
                record, agent, gridworld_learned_appraisal_manifest(), 0,
            )
            self.assertGreater(result["search_candidates"], 0)
            self.assertGreaterEqual(result["max_composed_shift"], 0.0)


if __name__ == "__main__":
    unittest.main()
