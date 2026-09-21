from __future__ import annotations

import json
import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import managed_live  # noqa: E402


EXPERIMENT = json.loads((ROOT / "experiment.json").read_text(encoding="utf-8"))
TASK = EXPERIMENT["task"]
NORMALIZED = {
    "source_count": 6,
    "valid_count": 4,
    "records": TASK["expected_normalized_records"],
    "dropped": TASK["expected_report"]["input"]["dropped"],
}
SUMMARY = TASK["expected_report"]["summary"]


def action(name: str, call_id: str, arguments: dict, turn_id: str = "turn-1"):
    return {
        "type": "function_call",
        "turn_id": turn_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def session(
    status: str,
    actions=(),
    *,
    usage=None,
    model: str = "gpt-5.6-terra",
    session_id: str = "sess-one",
    error=None,
):
    return SimpleNamespace(
        id=session_id,
        status=status,
        required_actions=list(actions),
        usage=usage,
        error=error,
        agent=SimpleNamespace(model=model),
    )


NORMALIZE_ACTION = action(
    "normalize_records",
    "call-normalize",
    {"dataset_id": TASK["dataset_id"], "records": TASK["raw_records"]},
)
SUMMARY_FAIL_ACTION = action(
    "summarize_records", "call-summary-fail", {"normalized": NORMALIZED}
)
SUMMARY_OK_ACTION = action(
    "summarize_records", "call-summary-ok", {"normalized": NORMALIZED}, "turn-2"
)
WRITE_ACTION = action(
    "write_report",
    "call-write",
    {"normalized": NORMALIZED, "summary": SUMMARY},
    "turn-2",
)


class FakeSessions:
    def __init__(self, owner, created):
        self.owner = owner
        self.created = created

    def create(self, **kwargs):
        self.owner.create_calls.append(kwargs)
        self.owner.provider_calls.append(("create", kwargs))
        return self.created


class FakeAdapter:
    def __init__(self, created, retrieved=(), turns=()):
        self.create_calls = []
        self.retrieve_calls = []
        self.submit_calls = []
        self.list_turn_calls = []
        self.provider_calls = []
        self._retrieved = list(retrieved)
        self._turns = list(turns)
        self.sessions = FakeSessions(self, created)

    def retrieve_session(self, session_id):
        self.retrieve_calls.append(session_id)
        self.provider_calls.append(("retrieve", session_id))
        if not self._retrieved:
            raise AssertionError("unexpected retrieve")
        value = self._retrieved.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    def submit_tool_result(
        self,
        session_id,
        turn_id,
        call_id,
        *,
        output=None,
        error=None,
        idempotency_key=None,
    ):
        call = {
            "session_id": session_id,
            "turn_id": turn_id,
            "call_id": call_id,
            "output": output,
            "error": error,
            "idempotency_key": idempotency_key,
        }
        self.submit_calls.append(call)
        self.provider_calls.append(("submit", call))

    def list_turns(self, session_id):
        self.list_turn_calls.append(session_id)
        self.provider_calls.append(("list_turns", session_id))
        return self._turns


class IncrementingClock:
    def __init__(self, value=1_000_000_000, step=1_000_000):
        self.value = value
        self.step = step

    def __call__(self):
        value = self.value
        self.value += self.step
        return value


class IncrementingUTC:
    def __init__(self):
        self.second = 0

    def __call__(self):
        value = f"2026-09-22T00:00:{self.second:02d}.000Z"
        self.second += 1
        return value


def driver(adapter, *, pid=200):
    return managed_live.ManagedLiveDriver(
        adapter=adapter,
        poll_interval=0,
        timeout=60,
        sleep=lambda _: None,
        monotonic_ns=IncrementingClock(),
        utc_now=IncrementingUTC(),
        process_id=lambda: pid,
    )


def run_to_fault(tmp_path: Path, adapter: FakeAdapter, run_id="run-1"):
    with pytest.raises(managed_live.PlannedInterruption) as caught:
        driver(adapter, pid=100).run(tmp_path, run_id)
    assert caught.value.exit_code == 75
    return json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))


def complete_adapter(*, repeat_pending_failure=True, usage=None):
    retrieved = []
    if repeat_pending_failure:
        retrieved.append(session("requires_action", [SUMMARY_FAIL_ACTION]))
    retrieved.extend(
        [
            session("requires_action", [SUMMARY_OK_ACTION]),
            session("requires_action", [WRITE_ACTION]),
            session("idle", usage=usage),
        ]
    )
    return FakeAdapter(
        session("requires_action", [NORMALIZE_ACTION]),
        retrieved=retrieved,
        turns=[SimpleNamespace(id="turn-2", status="completed", usage=usage)],
    )


def test_start_uses_exact_frozen_inline_agent_shape_and_exits_at_c2(tmp_path):
    adapter = FakeAdapter(
        session("requires_action", [NORMALIZE_ACTION]),
        retrieved=[session("requires_action", [SUMMARY_FAIL_ACTION])],
    )

    state = run_to_fault(tmp_path, adapter)

    assert len(adapter.create_calls) == 1
    call = adapter.create_calls[0]
    assert call["environment"] == {"type": "none"}
    assert call["input"] == managed_live._task_input(EXPERIMENT)
    assert call["stream"] is False
    assert call["agent"] == {
        "model": "gpt-5.6-terra",
        "instructions": TASK["developer_instruction"],
        "reasoning": {"effort": "low"},
        "text": {"format": {"type": "text"}, "verbosity": "low"},
        "service_tier": "default",
        "tools": [
            {key: tool[key] for key in ("type", "name", "description", "parameters")}
            for tool in EXPERIMENT["tools"]
        ],
    }
    assert call["metadata"] == {
        "benchmark_id": EXPERIMENT["benchmark_id"],
        "logical_run_id": "run-1",
    }
    assert state["session_id"] == "sess-one"
    assert state["checkpoint"] == "C2_FAILURE_PERSISTED"
    assert state["checkpoints"] == list(managed_live.CHECKPOINTS[:3])
    assert state["fault"]["consumed"] is True
    assert [(x["tool"], x["status"]) for x in state["tool_attempts"]] == [
        ("normalize_records", "success"),
        ("summarize_records", "error"),
    ]
    assert [call["call_id"] for call in adapter.submit_calls] == [
        "call-normalize",
        "call-summary-fail",
    ]
    assert json.loads(adapter.submit_calls[-1]["error"]) == EXPERIMENT["fault"]["result"]["error"]
    assert not (tmp_path / "report.json").exists()


def test_resume_retrieves_same_session_reuses_pending_failure_and_never_normalizes(tmp_path):
    first = FakeAdapter(
        session("requires_action", [NORMALIZE_ACTION]),
        retrieved=[session("requires_action", [SUMMARY_FAIL_ACTION])],
    )
    before = run_to_fault(tmp_path, first)
    normalize_result = before["receipts"]["call-normalize"]["result"]
    resumed = complete_adapter(repeat_pending_failure=True)

    report = driver(resumed).run(tmp_path, "run-1", resume=True)

    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert report == TASK["expected_report"]
    assert resumed.retrieve_calls[0] == "sess-one"
    assert resumed.create_calls == []
    assert [call["call_id"] for call in resumed.submit_calls] == [
        "call-summary-fail",
        "call-summary-ok",
        "call-write",
    ]
    assert state["receipts"]["call-normalize"]["result"] == normalize_result
    assert sum(x["tool"] == "normalize_records" for x in state["tool_attempts"]) == 1
    assert [(x["tool"], x["status"]) for x in state["tool_attempts"]] == [
        ("normalize_records", "success"),
        ("summarize_records", "error"),
        ("summarize_records", "success"),
        ("write_report", "success"),
    ]
    assert state["checkpoints"] == list(managed_live.CHECKPOINTS)
    assert state["status"] == "completed"
    assert state["resume_count"] == 1
    assert state["report_receipt"]["sha256"] == hashlib.sha256(
        (tmp_path / "report.json").read_bytes()
    ).hexdigest()
    assert json.loads((tmp_path / "report.json").read_text()) == report


def test_current_required_actions_are_authority_and_wrong_order_is_an_outcome(tmp_path):
    wrong = action(
        "summarize_records", "call-wrong", {"normalized": NORMALIZED}
    )
    adapter = FakeAdapter(session("requires_action", [wrong]))

    with pytest.raises(managed_live.ManagedLiveProtocolError, match="expected 'normalize_records'"):
        driver(adapter).run(tmp_path, "wrong-order")

    state = json.loads((tmp_path / "state.json").read_text())
    assert state["tool_attempts"] == []
    assert adapter.submit_calls == []


def test_strict_arguments_reject_modified_fixture_without_executing_tool(tmp_path):
    changed = json.loads(json.dumps(NORMALIZE_ACTION))
    changed["arguments"]["records"][0]["latency_ms"] = "81"
    adapter = FakeAdapter(session("requires_action", [changed]))

    with pytest.raises(managed_live.ManagedLiveProtocolError, match="frozen fixture"):
        driver(adapter).run(tmp_path, "bad-args")

    state = json.loads((tmp_path / "state.json").read_text())
    assert state["checkpoint"] == "C0_INITIALIZED"
    assert state["receipts"] == {}


def test_usage_remains_null_unless_provider_returns_it(tmp_path):
    first = FakeAdapter(
        session("requires_action", [NORMALIZE_ACTION]),
        retrieved=[session("requires_action", [SUMMARY_FAIL_ACTION])],
    )
    run_to_fault(tmp_path, first, "null-usage")
    resumed = complete_adapter(repeat_pending_failure=False, usage=None)
    driver(resumed).run(tmp_path, "null-usage", resume=True)

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["usage"] is None
    assert all(value is None for value in metrics["cost_usd"].values())


def test_provider_usage_is_copied_without_inventing_missing_fields(tmp_path):
    first = FakeAdapter(
        session("requires_action", [NORMALIZE_ACTION]),
        retrieved=[session("requires_action", [SUMMARY_FAIL_ACTION])],
    )
    run_to_fault(tmp_path, first, "with-usage")
    usage = {
        "input_tokens": 23,
        "input_tokens_details": {"cached_tokens": 7},
        "output_tokens": 11,
        "total_tokens": 34,
    }
    resumed = complete_adapter(repeat_pending_failure=False, usage=usage)
    driver(resumed).run(tmp_path, "with-usage", resume=True)

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["usage"] == {
        "input_tokens": 23,
        "cached_input_tokens": 7,
        "output_tokens": 11,
        "reasoning_tokens": None,
        "total_tokens": 34,
    }


def test_trace_has_required_shape_monotonic_sequences_and_exact_checkpoints(tmp_path):
    first = FakeAdapter(
        session("requires_action", [NORMALIZE_ACTION]),
        retrieved=[session("requires_action", [SUMMARY_FAIL_ACTION])],
    )
    run_to_fault(tmp_path, first, "trace-run")
    resumed = complete_adapter(repeat_pending_failure=False)
    driver(resumed).run(tmp_path, "trace-run", resume=True)

    events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    required = {
        "schema_version",
        "seq",
        "run_id",
        "arm",
        "process_instance_id",
        "wall_time_utc",
        "monotonic_ns",
        "event",
        "checkpoint",
        "resumed",
    }
    assert all(required <= event.keys() for event in events)
    checkpoints = [
        event["checkpoint"]
        for event in events
        if event["event"] == "checkpoint_saved"
    ]
    assert checkpoints == list(managed_live.CHECKPOINTS)
    assert any(event["resumed"] is True for event in events)


def test_resume_rejects_a_different_logical_run_id(tmp_path):
    first = FakeAdapter(
        session("requires_action", [NORMALIZE_ACTION]),
        retrieved=[session("requires_action", [SUMMARY_FAIL_ACTION])],
    )
    run_to_fault(tmp_path, first, "original")

    with pytest.raises(managed_live.ManagedLiveError, match="run_id"):
        driver(complete_adapter()).run(tmp_path, "different", resume=True)


def test_model_substitution_is_rejected_before_any_tool_executes(tmp_path):
    adapter = FakeAdapter(
        session("requires_action", [NORMALIZE_ACTION], model="different-model")
    )

    with pytest.raises(managed_live.ManagedLiveProtocolError, match="expected"):
        driver(adapter).run(tmp_path, "wrong-model")

    assert not (tmp_path / "state.json").exists()
    assert adapter.submit_calls == []


def test_live_cli_fails_explicitly_without_key_and_makes_no_run_directory(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    run_dir = tmp_path / "never-created"

    code = managed_live.main(
        ["--live", "--run-dir", str(run_dir), "--run-id", "no-key"]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "OPENAI_API_KEY is not set" in captured.err
    assert not run_dir.exists()


def test_build_live_adapter_disables_sdk_retries(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.beta = SimpleNamespace(
                agents=SimpleNamespace(sessions=SimpleNamespace())
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    managed_live._build_live_adapter("secret-value")

    assert captured == {"api_key": "secret-value", "max_retries": 0}


def test_active_key_is_redacted_from_failure_artifacts(tmp_path):
    secret = "sk-test-must-not-leak"
    adapter = FakeAdapter(
        session("in_progress"),
        retrieved=[RuntimeError(f"Authorization: Bearer {secret}")],
    )
    benchmark_driver = managed_live.ManagedLiveDriver(
        adapter=adapter,
        api_key=secret,
        poll_interval=0,
        monotonic_ns=IncrementingClock(),
        utc_now=IncrementingUTC(),
    )

    with pytest.raises(managed_live.ManagedLiveError):
        benchmark_driver.run(tmp_path, "redaction")

    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in tmp_path.iterdir()
        if path.is_file()
    )
    assert secret not in serialized
    assert "[REDACTED]" in serialized


def test_completed_resume_is_idempotent_and_does_not_contact_provider(tmp_path):
    first = FakeAdapter(
        session("requires_action", [NORMALIZE_ACTION]),
        retrieved=[session("requires_action", [SUMMARY_FAIL_ACTION])],
    )
    run_to_fault(tmp_path, first, "done")
    resumed = complete_adapter(repeat_pending_failure=False)
    expected = driver(resumed).run(tmp_path, "done", resume=True)
    report_stat = (tmp_path / "report.json").stat()
    offline = FakeAdapter(session("failed"))

    actual = driver(offline).run(tmp_path, "done", resume=True)

    assert actual == expected
    assert offline.provider_calls == []
    assert (tmp_path / "report.json").stat().st_mtime_ns == report_stat.st_mtime_ns
