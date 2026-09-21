from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ARTIFACT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARTIFACT_DIR))

import run_benchmark  # noqa: E402


class OfflineBenchmarkTests(unittest.TestCase):
    def test_both_recovery_controls_pass_without_claiming_api_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "trial"
            summary = run_benchmark.run_all(destination)

            self.assertEqual(summary["status"], "offline_control_passed")
            self.assertFalse(summary["comparison_eligible"])
            self.assertTrue(summary["semantic_summaries_match"])
            self.assertIsNone(summary["unavailable"]["provider_tokens"])
            self.assertIsNone(summary["unavailable"]["provider_cost_usd"])
            for arm in ("custom_harness", "agents_api"):
                run_dir = destination / arm
                manifest = json.loads((run_dir / "run-manifest.json").read_text())
                self.assertTrue(manifest["passed"])
                self.assertEqual(manifest["initial_exit_code"], 75)
                self.assertEqual(manifest["resume_exit_code"], 0)
                self.assertTrue((run_dir / "trace.jsonl").is_file())
                self.assertTrue((run_dir / "report.json").is_file())
                self.assertTrue((run_dir / "initial-stdout.log").is_file())
                self.assertTrue((run_dir / "resume-stderr.log").is_file())

            agents_metrics = summary["arms"]["agents_api"]["metrics"]
            self.assertTrue(agents_metrics["simulated"])
            self.assertFalse(agents_metrics["api_derived"])
            self.assertIsNone(agents_metrics["cost_usd"])
            usage = agents_metrics["usage"]
            self.assertTrue(
                usage is None or all(value is None for value in usage.values())
            )

    def test_existing_destination_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "existing"
            destination.mkdir()
            sentinel = destination / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_benchmark.run_all(destination)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
