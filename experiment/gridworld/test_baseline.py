import json
from pathlib import Path
import tempfile
import unittest

from experiment.gridworld import baseline


class LearnedAppraisalBaselineTests(unittest.TestCase):
    def test_snapshot_records_contracts_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "snapshot"
            args = baseline.parse_args(
                [
                    "--output-dir",
                    str(output),
                    "--skip-tests",
                    "--skip-gridworld",
                ]
            )

            result = baseline.run(args)
            self.assertEqual(result, output)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            contracts = json.loads(
                (output / "deterministic_contracts.json").read_text(encoding="utf-8")
            )
            checksums = json.loads(
                (output / "checksums.json").read_text(encoding="utf-8")
            )

            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(manifest["source_sha256"]), 64)
            self.assertIn("gridworld", contracts)
            self.assertEqual(set(contracts), {"gridworld"})
            self.assertIn("deterministic_contracts.json", checksums)
            self.assertIn("manifest.json", checksums)

            with self.assertRaises(FileExistsError):
                baseline.run(args)


if __name__ == "__main__":
    unittest.main()
