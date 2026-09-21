from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ARTIFACT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARTIFACT_DIR))

import custom_harness as harness  # noqa: E402


class ToolTests(unittest.TestCase):
    def test_three_tools_have_exact_deterministic_result(self) -> None:
        normalized = harness.normalize_records(harness.RAW_RECORDS)
        self.assertEqual(
            [record["id"] for record in normalized["records"]],
            ["job-001", "job-002", "job-003", "job-004"],
        )
        self.assertEqual(normalized["source_count"], 6)
        self.assertEqual(normalized["valid_count"], 4)
        self.assertEqual(
            [item["reason"] for item in normalized["dropped"]],
            ["duplicate_id", "invalid_latency"],
        )

        summary = harness.summarize_records(normalized)
        self.assertEqual(
            summary,
            {
                "count": 4,
                "error_count": 1,
                "error_rate": 0.25,
                "mean_latency_ms": 140.0,
                "p95_latency_ms": 200,
            },
        )
        self.assertEqual(
            harness.write_report(normalized, summary),
            harness.write_report(normalized, summary),
        )


class DurableRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name) / "run"
        self.run_id = "custom-test-001"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def read_json(self, name: str) -> dict:
        return json.loads((self.run_dir / name).read_text(encoding="utf-8"))

    def read_trace(self) -> list[dict]:
        return [
            json.loads(line)
            for line in (self.run_dir / "trace.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]

    def test_checkpoint_failure_resume_and_idempotent_second_resume(self) -> None:
        with self.assertRaises(harness.PlannedInterruption) as raised:
            harness.run(self.run_dir, self.run_id)
        self.assertEqual(raised.exception.exit_code, 75)

        failed_state = self.read_json("state.json")
        self.assertEqual(failed_state["status"], "retry_pending")
        self.assertEqual(failed_state["next_step"], 1)
        self.assertEqual(
            failed_state["completed_steps"],
            [{"index": 0, "tool": "normalize_records"}],
        )
        self.assertEqual(failed_state["last_error"]["code"], "TEMP_UNAVAILABLE")
        self.assertEqual(failed_state["metrics"]["tool_calls"], 2)
        self.assertEqual(failed_state["metrics"]["retries"], 1)
        self.assertTrue((self.run_dir / "metrics.json").is_file())
        self.assertFalse((self.run_dir / "report.json").exists())

        report = harness.run(self.run_dir, self.run_id, resume=True)
        completed_state = self.read_json("state.json")
        metrics = self.read_json("metrics.json")
        self.assertEqual(completed_state["status"], "completed")
        self.assertEqual(completed_state["next_step"], 3)
        self.assertEqual(metrics["tool_calls"], 4)
        self.assertEqual(metrics["successful_tool_calls"], 3)
        self.assertEqual(metrics["retries"], 1)
        self.assertGreaterEqual(metrics["wall_time_ms"], 0)
        self.assertEqual(report["summary"]["count"], 4)
        self.assertEqual(report, self.read_json("report.json"))

        trace_before = self.read_trace()
        starts_before = [
            event for event in trace_before if event["event"] == "tool_call_started"
        ]
        report_bytes = (self.run_dir / "report.json").read_bytes()

        second_report = harness.run(self.run_dir, self.run_id, resume=True)
        trace_after = self.read_trace()
        starts_after = [
            event for event in trace_after if event["event"] == "tool_call_started"
        ]
        self.assertEqual(second_report, report)
        self.assertEqual((self.run_dir / "report.json").read_bytes(), report_bytes)
        self.assertEqual(starts_after, starts_before)
        self.assertEqual(self.read_json("metrics.json")["tool_calls"], 4)

        normalize_starts = [
            event
            for event in trace_after
            if event["event"] == "tool_call_started"
            and event["tool"] == "normalize_records"
        ]
        summarize_starts = [
            event
            for event in trace_after
            if event["event"] == "tool_call_started"
            and event["tool"] == "summarize_records"
        ]
        self.assertEqual(len(normalize_starts), 1)
        self.assertEqual([item["attempt"] for item in summarize_starts], [1, 2])
        self.assertTrue(
            any(
                event["event"] == "tool_call_failed"
                and event["error_code"] == "TEMP_UNAVAILABLE"
                for event in trace_after
            )
        )
        self.assertGreaterEqual(
            sum(event["event"] == "checkpoint_saved" for event in trace_after), 5
        )

    def test_cli_exits_75_once_then_resume_completes(self) -> None:
        command = [
            sys.executable,
            str(ARTIFACT_DIR / "custom_harness.py"),
            "--simulate",
            "--run-dir",
            str(self.run_dir),
        ]
        first = subprocess.run(
            command + ["--run-id", self.run_id],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(first.returncode, 75, first.stderr)
        self.assertEqual(json.loads(first.stdout)["error_code"], "TEMP_UNAVAILABLE")

        second = subprocess.run(
            command + ["--resume", self.run_id],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(second.returncode, 0, second.stderr)
        payload = json.loads(second.stdout)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["metrics"]["tool_calls"], 4)
        self.assertEqual(payload["metrics"]["retries"], 1)


if __name__ == "__main__":
    unittest.main()
