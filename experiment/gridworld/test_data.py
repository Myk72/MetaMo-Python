import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiment.gridworld.collect import (
    CollectionConfig,
    CollectionRegime,
    collect_dataset,
)
from experiment.gridworld.targets import (
    generate_targets,
    safety_first_outcome_key,
    select_target_evaluations,
    select_oracle_action,
)
from experiment.gridworld.quick_reachability import summarize_audit
from openpsi.learned import file_sha256


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class LearnedAppraisalDataTests(unittest.TestCase):
    def small_config(self):
        return CollectionConfig(
            train_seeds=(0,),
            validation_seeds=(1,),
            test_seeds=(2,),
            ood_seeds=(100,),
            episodes_per_seed=1,
            max_steps=2,
            exploration_probability=0.5,
            regimes=(CollectionRegime("low", 0.2),),
            ood_regimes=(
                CollectionRegime(
                    "extreme_layout",
                    0.95,
                    lava_cell_count=8,
                    mineral_spawn_band=3,
                    is_ood=True,
                ),
            ),
        )

    def test_collection_and_target_generation_are_complete_and_grouped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "dataset"
            target_dir = root / "targets"

            collect_dataset(dataset_dir, self.small_config())
            dataset_manifest = json.loads(
                (dataset_dir / "manifest.json").read_text(encoding="utf-8")
            )
            transitions = read_jsonl(dataset_dir / "transitions.jsonl")

            self.assertEqual(dataset_manifest["status"], "complete")
            self.assertEqual(dataset_manifest["record_count"], 8)
            self.assertEqual(dataset_manifest["context_count"], 24)
            self.assertEqual(len(transitions), 8)
            self.assertEqual(set(dataset_manifest["split_counts"]), {
                "train", "validation", "test", "ood"
            })
            for record in transitions:
                self.assertEqual(len(record["counterfactuals"]), 4)
                self.assertEqual(len(record["contexts"]), 3)
                self.assertEqual(
                    record["actual_outcome"],
                    record["counterfactuals"][record["behavior_action_index"]],
                )

            generate_targets(
                dataset_dir,
                target_dir,
                required_margin=0.0,
                random_samples=4,
            )
            target_manifest = json.loads(
                (target_dir / "manifest.json").read_text(encoding="utf-8")
            )
            audit = json.loads(
                (target_dir / "feasibility_audit.json").read_text(encoding="utf-8")
            )
            targets = read_jsonl(target_dir / "targets.jsonl")

            self.assertEqual(target_manifest["status"], "complete")
            self.assertEqual(target_manifest["target_count"], 24)
            self.assertEqual(target_manifest["replay_verified_count"], 24)
            self.assertEqual(target_manifest["strict_replay_verified_count"], 24)
            self.assertEqual(
                target_manifest["feasibility_audit_sha256"],
                file_sha256(target_dir / "feasibility_audit.json"),
            )
            self.assertEqual(audit["scope"]["target_count"], 24)
            self.assertEqual(
                sum(audit["status_counts"].values()),
                24,
            )
            self.assertEqual(len(targets), 24)
            for target in targets:
                residual = np.asarray(target["target_residual"], dtype=float)
                strict_residual = np.asarray(
                    target["strict_target_residual"], dtype=float
                )
                self.assertLessEqual(np.linalg.norm(residual), 0.08 + 1e-12)
                self.assertLessEqual(
                    np.linalg.norm(strict_residual),
                    0.08 + 1e-12,
                )
                self.assertGreaterEqual(residual[4], 0.0)
                self.assertGreaterEqual(residual[5], 0.0)
                self.assertTrue(target["replay_verified"])
                self.assertTrue(target["strict_replay_verified"])
                self.assertEqual(
                    target["baseline_oracle_selected"],
                    target["baseline_action_index"]
                    == target["oracle_action_index"],
                )
                if target["action_switch_feasible"]:
                    self.assertFalse(target["baseline_oracle_selected"])
                    self.assertIsNotNone(
                        target["minimum_action_switch_residual_norm"]
                    )
                else:
                    self.assertIsNone(
                        target["minimum_action_switch_residual_norm"]
                    )
                if target["status"] == "baseline_sufficient":
                    self.assertTrue(target["oracle_selected"])
                    np.testing.assert_array_equal(residual, np.zeros(6))
                elif target["status"] == "action_switch_feasible":
                    self.assertTrue(target["oracle_selected"])
                elif target["status"] == "margin_improvement_only":
                    self.assertGreaterEqual(
                        target["target_margin_improvement"],
                        target["improvement_threshold"],
                    )
                elif target["status"] == "no_safe_improvement":
                    np.testing.assert_array_equal(residual, np.zeros(6))
                else:
                    self.fail(f"unknown target status: {target['status']}")

                if target["strict_status"] in {
                    "baseline_sufficient",
                    "feasible_residual",
                }:
                    self.assertTrue(target["strict_margin_satisfied"])
                elif target["strict_status"] == "no_feasible_residual":
                    np.testing.assert_array_equal(strict_residual, np.zeros(6))
                else:
                    self.fail(f"unknown strict status: {target['strict_status']}")

            with self.assertRaises(FileExistsError):
                collect_dataset(dataset_dir, self.small_config())
            with self.assertRaises(FileExistsError):
                generate_targets(dataset_dir, target_dir)

    def test_shadow_and_strict_target_policies_are_independent(self):
        def item(norm, margin, action_index):
            return {
                "norm": norm,
                "margin": margin,
                "margin_improvement": margin + 0.2,
                "action_index": action_index,
                "residual": np.asarray([norm]),
            }

        baseline = item(0.0, -0.2, 1)
        improvement = item(0.01, -0.1, 1)
        switch = item(0.02, 0.0, 0)
        strict = item(0.03, 0.03, 0)
        selection = select_target_evaluations(
            [baseline, improvement, switch, strict],
            oracle_index=0,
            required_margin=0.02,
            improvement_threshold=0.001,
        )

        self.assertEqual(selection["status"], "action_switch_feasible")
        self.assertIs(selection["target"], switch)
        self.assertEqual(selection["strict_status"], "feasible_residual")
        self.assertIs(selection["strict_target"], strict)

        improvement_only = select_target_evaluations(
            [baseline, improvement],
            oracle_index=0,
            required_margin=0.02,
            improvement_threshold=0.001,
        )
        self.assertEqual(improvement_only["status"], "margin_improvement_only")
        self.assertIs(improvement_only["target"], improvement)
        self.assertEqual(
            improvement_only["strict_status"],
            "no_feasible_residual",
        )

        no_improvement = item(0.01, -0.2, 1)
        no_safe = select_target_evaluations(
            [baseline, no_improvement],
            oracle_index=0,
            required_margin=0.02,
            improvement_threshold=0.001,
        )
        self.assertEqual(no_safe["status"], "no_safe_improvement")
        self.assertIs(no_safe["target"], baseline)

    def test_safety_first_oracle_dominates_reward(self):
        safe = {
            "dead": False,
            "in_lava": False,
            "unsafe_zone": False,
            "lava_distance_after": 3,
            "energy_after": 80.0,
            "mineral_collected": False,
            "progress": -1.0,
            "reward": -0.5,
            "movement": True,
        }
        risky_reward = {
            **safe,
            "in_lava": True,
            "unsafe_zone": True,
            "lava_distance_after": 0,
            "energy_after": 100.0,
            "mineral_collected": True,
            "progress": 10.0,
            "reward": 100.0,
        }
        self.assertGreater(
            safety_first_outcome_key(safe),
            safety_first_outcome_key(risky_reward),
        )

        record = {
            "counterfactuals": [risky_reward, safe],
            "consensus_rule_scores": [100.0, 0.0],
        }
        oracle_index, evidence = select_oracle_action(record)
        self.assertEqual(oracle_index, 1)
        self.assertEqual(evidence["tied_action_indices"], [1])

    def test_quick_summary_stops_when_global_residual_cannot_switch_actions(self):
        audit = {
            "scope": {"target_count": 12},
            "control_authority": {
                "baseline_oracle_selected_count": 5,
                "action_switch_feasible_count": 0,
            },
            "status_counts": {
                "baseline_sufficient": 5,
                "margin_improvement_only": 7,
            },
        }
        summary = summarize_audit(audit)

        self.assertEqual(summary["initially_incorrect"], 7)
        self.assertEqual(summary["action_switch_feasible"], 0)
        self.assertEqual(
            summary["result"],
            "global_residual_lacks_control_authority",
        )
        self.assertEqual(summary["correctable_rate_among_initially_incorrect"], 0.0)


if __name__ == "__main__":
    unittest.main()
