from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import responses_live  # noqa: E402


class FakeResponses:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.payloads:
            raise AssertionError("unexpected provider call")
        return self.payloads.pop(0)


class FakeClient:
    def __init__(self, payloads):
        self.responses = FakeResponses(payloads)


def fixture():
    return responses_live.load_experiment()


def call_payload(identifier, tool, arguments, usage=10):
    return {
        "id": identifier,
        "model": fixture()["model"]["requested_id"],
        "output": [
            {
                "type": "function_call",
                "call_id": f"call-{identifier}",
                "name": tool,
                "arguments": json.dumps(arguments, sort_keys=True),
            }
        ],
        "usage": {
            "input_tokens": usage,
            "input_tokens_details": {"cached_tokens": 1},
            "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 1},
            "total_tokens": usage + 2,
        },
    }


def terminal_payload(identifier="r-final"):
    return {
        "id": identifier,
        "model": fixture()["model"]["requested_id"],
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "receipt committed"}],
            }
        ],
        "usage": {
            "input_tokens": 5,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 7,
        },
    }


def test_live_custom_arm_persists_failure_and_resumes_full_transcript(tmp_path):
    exp = fixture()
    normalized = responses_live.custom_harness.normalize_records(exp["task"]["raw_records"])
    summary = responses_live.custom_harness.summarize_records(normalized)
    payloads = [
        call_payload(
            "r-normalize",
            "normalize_records",
            {"dataset_id": exp["task"]["dataset_id"], "records": exp["task"]["raw_records"]},
        ),
        call_payload("r-summary-fail", "summarize_records", {"normalized": normalized}),
        call_payload("r-summary-ok", "summarize_records", {"normalized": normalized}),
        call_payload("r-write", "write_report", {"normalized": normalized, "summary": summary}),
        terminal_payload(),
    ]
    client = FakeClient(payloads)
    run_dir = tmp_path / "run"

    with pytest.raises(responses_live.PlannedInterruption):
        responses_live.run(run_dir, "paired-001", client=client)
    failed = json.loads((run_dir / "state.json").read_text())
    assert failed["checkpoint"] == "C2_FAILURE_PERSISTED"
    assert failed["status"] == "retry_pending"
    assert failed["completed_tools"] == ["normalize_records"]
    assert failed["metrics"]["model_calls"] == 2
    assert failed["metrics"]["tool_attempts"] == 2
    assert failed["metrics"]["retry_count"] == 1
    assert not (run_dir / "report.json").exists()

    report = responses_live.run(run_dir, "paired-001", resume=True, client=client)
    assert report == exp["task"]["expected_report"]
    state = json.loads((run_dir / "state.json").read_text())
    assert state["checkpoints"] == exp["checkpoints"]
    assert state["completed_tools"] == [
        "normalize_records",
        "summarize_records",
        "write_report",
    ]
    assert state["attempts"] == {
        "normalize_records": 1,
        "summarize_records": 2,
        "write_report": 1,
    }
    assert state["metrics"]["model_calls"] == 5
    assert state["metrics"]["provider_requests"] == 5
    assert state["metrics"]["tool_attempts"] == 4
    assert state["metrics"]["tool_successes"] == 3
    assert state["metrics"]["input_tokens"] == 45
    assert state["metrics"]["cached_input_tokens"] == 4
    assert state["metrics"]["output_tokens"] == 10
    assert state["metrics"]["reasoning_tokens"] == 4
    assert state["metrics"]["total_tokens"] == 55
    assert json.loads((run_dir / "report.json").read_text()) == report
    assert state["receipts"]["write_report"]["sha256"] == hashlib.sha256(
        (run_dir / "report.json").read_bytes()
    ).hexdigest()
    assert state["metrics"]["restart_gap_ms"] >= 0
    assert state["metrics"]["recovery_wall_ms"] >= 0
    assert state["metrics"]["end_to_end_wall_ms"] >= state["metrics"]["active_wall_ms"]

    # A completed resume is read-only and makes no provider request.
    again = responses_live.run(
        run_dir, "paired-001", resume=True, client=FakeClient([])
    )
    assert again == report

    for kwargs in client.responses.calls:
        assert kwargs["store"] is False
        assert kwargs["parallel_tool_calls"] is False
        assert "previous_response_id" not in kwargs
        assert "conversation" not in kwargs
        assert kwargs["include"] == ["reasoning.encrypted_content"]
    resume_input = client.responses.calls[2]["input"]
    assert any(item.get("id") == "r-normalize" for item in resume_input if isinstance(item, dict)) is False
    # Response IDs live in state; replay uses the provider output items themselves.
    assert sum(item.get("name") == "normalize_records" for item in resume_input if isinstance(item, dict)) == 1
    assert sum(item.get("type") == "function_call_output" for item in resume_input if isinstance(item, dict)) == 2


def test_wrong_tool_and_wrong_returned_model_fail_closed(tmp_path):
    exp = fixture()
    wrong_tool = FakeClient([
        call_payload("r-wrong", "summarize_records", {"normalized": {}})
    ])
    with pytest.raises(responses_live.ResponsesProtocolError, match="expected tool"):
        responses_live.run(tmp_path / "wrong-tool", "wrong-tool", client=wrong_tool)

    payload = call_payload(
        "r-model",
        "normalize_records",
        {"dataset_id": exp["task"]["dataset_id"], "records": exp["task"]["raw_records"]},
    )
    payload["model"] = "substituted-model"
    with pytest.raises(responses_live.ResponsesProtocolError, match="differs from frozen"):
        responses_live.run(
            tmp_path / "wrong-model", "wrong-model", client=FakeClient([payload])
        )


def test_cli_without_key_fails_before_network_and_does_not_leak(tmp_path):
    environment = dict(os.environ)
    environment.pop("OPENAI_API_KEY", None)
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "responses_live.py"),
            "--live",
            "--run-dir",
            str(tmp_path / "cli"),
            "--run-id",
            "cli-no-key",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert process.returncode == 2
    assert "OPENAI_API_KEY" in process.stderr
    assert "Traceback" not in process.stderr
    assert not (tmp_path / "cli").exists()


def test_provider_kwargs_are_exact_and_tools_strip_benchmark_metadata():
    exp = fixture()
    captured = {}

    class StopResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop after capture")

    class StopClient:
        responses = StopResponses()

    with tempfile.TemporaryDirectory() as temporary:
        with pytest.raises(responses_live.ResponsesLiveError, match="stop after capture"):
            responses_live.run(Path(temporary) / "run", "capture", client=StopClient())
    assert captured["model"] == exp["model"]["requested_id"]
    assert captured["instructions"] == exp["task"]["developer_instruction"]
    assert captured["reasoning"] == {"effort": "low"}
    assert captured["text"] == {"verbosity": "low"}
    assert captured["service_tier"] == "default"
    assert all(tool["strict"] is True for tool in captured["tools"])
    assert all("side_effect" not in tool for tool in captured["tools"])
