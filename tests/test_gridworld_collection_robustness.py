import unittest

from applications.gridworld.evaluation.collection_robustness import (
    run_collection_robustness_evaluation,
)


class GridWorldCollectionRobustnessTests(unittest.TestCase):
    def test_independent_splits_report_sparse_evidence_without_claiming_success(self):
        report = run_collection_robustness_evaluation(
            split_count=2,
            train_episodes=2,
            test_episodes=2,
            max_steps=4,
            train_seed_start=100,
            test_seed_start=1000,
            seed_stride=100,
            minimum_train_collections=20,
            minimum_test_collections=10,
        )
        self.assertEqual(len(report.splits), 2)
        self.assertEqual([split.train_seed_start for split in report.splits], [100, 200])
        self.assertEqual([split.test_seed_start for split in report.splits], [1000, 1100])
        self.assertEqual(report.eligible_splits, 0)
        self.assertFalse(report.robustness_passed)
        self.assertTrue(report.behavior_unchanged)
        self.assertTrue(report.safety_passed)

    def test_rejects_overlapping_splits_before_running(self):
        with self.assertRaises(ValueError):
            run_collection_robustness_evaluation(
                split_count=2,
                train_episodes=2,
                test_episodes=2,
                train_seed_start=100,
                test_seed_start=100,
                seed_stride=10,
            )


if __name__ == "__main__":
    unittest.main()
