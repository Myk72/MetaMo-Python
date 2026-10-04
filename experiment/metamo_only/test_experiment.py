"""Controls for the standalone learned-appraisal experiment."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiment.metamo_only.environment import DECISION_PROFILE, sample_case
from experiment.metamo_only.experiment import make_manifest, make_target, parse_args, run
from dynamics.stability import DefaultStabilityPolicy, boundary_pressure
from magus.decision import MagusDecision
from openpsi.learned import LearnedAppraisalManifest, ManifestFeatureEncoder


class MetaMoAppraisalLabTests(unittest.TestCase):
    def test_targets_respect_manifest_and_hard_safety(self):
        manifest = make_manifest()
        encoder = ManifestFeatureEncoder(manifest, boundary_pressure)
        decision = MagusDecision(profile=DECISION_PROFILE)
        stability = DefaultStabilityPolicy()
        rng = np.random.default_rng(12)
        for _ in range(8):
            case = sample_case(rng)
            self.assertGreaterEqual(len(case.candidates), 2)
            self.assertTrue(all(action.metadata["risk"] < 0.65 for action in case.candidates))
            features, target, status, baseline = make_target(
                case, manifest, encoder, decision, stability,
            )
            self.assertEqual(features.shape, (manifest.input_size,))
            self.assertEqual(target.shape, (manifest.output_size,))
            self.assertTrue(np.all(np.isfinite(features)))
            self.assertTrue(np.all(np.abs(target) <= 1.0))
            self.assertGreaterEqual(target[4], 0.0)
            self.assertGreaterEqual(target[5], 0.0)
            self.assertLessEqual(
                np.linalg.norm(target * np.asarray(manifest.residual_bounds)), 0.08 + 1e-12,
            )
            self.assertIn(status, {
                "baseline_sufficient", "action_switch_feasible",
                "margin_improvement_only", "no_safe_improvement",
            })
            self.assertIn(baseline, range(len(case.candidates)))

    def test_training_artifact_and_shadow_control(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "lab"
            args = parse_args([
                "--train-cases", "6", "--validation-cases", "3",
                "--test-cases", "4", "--epochs", "5",
                "--output-dir", str(output),
            ])
            summary = run(args)
            self.assertEqual(summary["evaluation"]["cases"], 4)
            self.assertEqual(
                summary["evaluation"]["shadow_mean_utility"],
                summary["evaluation"]["rule_mean_utility"],
            )
            self.assertGreaterEqual(summary["evaluation"]["shadow_records"], 4)
            manifest = LearnedAppraisalManifest.read_json(output / "manifest.json")
            self.assertTrue(Path(manifest.artifact_path).is_file())
            self.assertEqual(
                json.loads((output / "summary.json").read_text())["split_sizes"]["test"], 4,
            )
            with (output / "test_cases.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertIn("predicted_approach", rows[0])
            self.assertIn("applied_approach", rows[0])
            with self.assertRaises(FileExistsError):
                run(args)


if __name__ == "__main__":
    unittest.main()
