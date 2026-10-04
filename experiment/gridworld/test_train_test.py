"""Shareable smoke tests for the GridWorld residual experiment."""

import tempfile
from pathlib import Path
import unittest

from applications.gridworld.learned_appraisal import (
    gridworld_learned_appraisal_manifest, with_learned_appraisal,
)
from applications.gridworld.runtime import make_gridworld_runtime
from experiment.gridworld.collect import CollectionConfig, collect_dataset
from experiment.gridworld.targets import generate_targets
from experiment.gridworld.train_test import parse_args, run
from openpsi.hybrid import HybridAppraisalMode
from openpsi.learned import ZeroResidualPredictor


class GridWorldResidualExperimentTests(unittest.TestCase):
    def test_opt_in_adapter_keeps_default_rule_runtime(self):
        rule = make_gridworld_runtime()
        manifest = gridworld_learned_appraisal_manifest()
        shadow = with_learned_appraisal(
            rule,
            mode=HybridAppraisalMode.SHADOW,
            predictor=ZeroResidualPredictor(manifest.output_size),
            manifest=manifest,
        )
        self.assertEqual(rule.name, "gridworld:rule")
        self.assertIsNot(rule.bimonad, shadow.bimonad)
        self.assertEqual(shadow.bimonad.appraisal.config.mode, HybridAppraisalMode.SHADOW)

    def test_collect_target_train_and_replay(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            data = base / "data"
            targets = base / "targets"
            model = base / "model"
            collect_dataset(data, CollectionConfig(
                train_seeds=(0,), validation_seeds=(1,),
                test_seeds=(2,), ood_seeds=(3,),
                episodes_per_seed=1, max_steps=1,
            ))
            generate_targets(data, targets, random_samples=0)
            result = run(parse_args([
                str(targets), "--output-dir", str(model),
                "--epochs", "2", "--q-train-episodes", "0",
                "--eval-episodes", "0", "--agent-seeds", "0",
            ]))
            self.assertGreater(result["decision_audits"]["test"]["cases"], 0)
            self.assertEqual(
                result["decision_audits"]["test"]["target_replay_verified_rate"], 1.0,
            )
            self.assertTrue((model / "manifest.json").is_file())
            self.assertTrue((model / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
