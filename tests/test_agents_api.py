from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import agents_api  # noqa: E402


class FakeEvents:
    def __init__(self, stream_values=()):
        self.created = []
        self.stream_values = list(stream_values)

    def create(self, session_id, **kwargs):
        self.created.append((session_id, kwargs))
        return None

    def stream(self, session_id):
        return FakeStream(self.stream_values)


class FakeStream:
    def __init__(self, values):
        self.values = values
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return iter(self.values)

    def __exit__(self, exc_type, exc, traceback):
        self.exited = True


class FakeTurns:
    def __init__(self, pages):
        self.pages = list(pages)
        self.list_calls = []
        self.retrieve_calls = []

    def list(self, session_id, **kwargs):
        self.list_calls.append((session_id, kwargs))
        page = self.pages.pop(0) if len(self.pages) > 1 else self.pages[0]
        return SimpleNamespace(data=page)

    def retrieve(self, turn_id, **kwargs):
        self.retrieve_calls.append((turn_id, kwargs))
        return SimpleNamespace(id=turn_id, **kwargs)


class FakeSessions:
    def __init__(self, *, created, retrieved=(), turn_pages=((),), stream_values=()):
        self.created_value = created
        self.retrieved = list(retrieved)
        self.create_calls = []
        self.retrieve_calls = []
        self.turns = FakeTurns(turn_pages)
        self.events = FakeEvents(stream_values)

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return self.created_value

    def retrieve(self, session_id):
        self.retrieve_calls.append(session_id)
        return self.retrieved.pop(0) if len(self.retrieved) > 1 else self.retrieved[0]


def fake_client(sessions):
    return SimpleNamespace(beta=SimpleNamespace(agents=SimpleNamespace(sessions=sessions)))


class IncrementingClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        current = self.value
        self.value += 0.1
        return current


def test_create_session_uses_official_beta_sessions_shape():
    created = SimpleNamespace(id="sess_1", status="in_progress")
    sessions = FakeSessions(created=created)
    adapter = agents_api.AgentsAPIAdapter(client=fake_client(sessions))
    tools = [{"type": "function", "name": "normalize_records", "parameters": {"type": "object"}}]

    result = adapter.create_session(
        "process records",
        model="gpt-test",
        instructions="Use all three tools.",
        tools=tools,
        metadata={"run_id": "run-1"},
        stream=True,
    )

    assert result is created
    assert sessions.create_calls == [
        {
            "environment": {"type": "none"},
            "input": "process records",
            "stream": True,
            "agent": {
                "model": "gpt-test",
                "instructions": "Use all three tools.",
                "tools": tools,
            },
            "metadata": {"run_id": "run-1"},
        }
    ]


def test_polling_run_preserves_session_turn_usage_and_server_timestamps():
    created = SimpleNamespace(
        id="sess_durable",
        status="in_progress",
        created_at=1_700_000_000,
        last_active_at=1_700_000_001,
        required_actions=[],
        usage=None,
        error=None,
    )
    retrieved = SimpleNamespace(
        id="sess_durable",
        status="idle",
        created_at=1_700_000_000,
        last_active_at=1_700_000_004,
        required_actions=[],
        usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        error=None,
    )
    queued = SimpleNamespace(id="turn_durable", status="queued", error=None, usage=None)
    completed = SimpleNamespace(
        id="turn_durable",
        status="completed",
        created_at=1_700_000_001,
        started_at=1_700_000_002,
        completed_at=1_700_000_004,
        error=None,
        usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    sessions = FakeSessions(
        created=created,
        retrieved=[retrieved],
        turn_pages=[[queued], [completed]],
    )
    seen = []
    adapter = agents_api.AgentsAPIAdapter(
        client=fake_client(sessions),
        poll_interval=0,
        sleep=lambda _: None,
        monotonic=IncrementingClock(),
    )

    result = adapter.run("do work", model="gpt-test", on_event=seen.append)

    assert result.status == "completed"
    assert result.session_id == "sess_durable"
    assert result.turn_ids == ("turn_durable",)
    assert result.usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    assert result.server_timestamps == {
        "session_created_at": 1_700_000_000,
        "session_last_active_at": 1_700_000_004,
        "turn_created_at": 1_700_000_001,
        "turn_started_at": 1_700_000_002,
        "turn_completed_at": 1_700_000_004,
    }
    assert [event.type for event in seen] == ["adapter.session.poll", "adapter.session.poll"]
    assert sessions.retrieve_calls == ["sess_durable"]
    assert sessions.turns.list_calls[0] == ("sess_durable", {"order": "asc", "limit": 100})


def test_streaming_run_normalizes_events_and_output():
    stream = FakeStream(
        [
            {
                "type": "agent.session.created",
                "event_id": "evt_1",
                "session": {"id": "sess_stream", "status": "in_progress", "created_at": 11},
            },
            {
                "type": "agent.session.turn.created",
                "event_id": "evt_2",
                "session_id": "sess_stream",
                "turn": {"id": "turn_stream", "status": "in_progress", "created_at": 12},
            },
            {
                "type": "agent.session.turn.output_text.delta",
                "event_id": "evt_3",
                "session_id": "sess_stream",
                "turn_id": "turn_stream",
                "delta": "hello ",
            },
            {
                "type": "agent.session.turn.output_text.delta",
                "event_id": "evt_4",
                "session_id": "sess_stream",
                "turn_id": "turn_stream",
                "delta": "world",
            },
            {
                "type": "agent.session.turn.completed",
                "event_id": "evt_5",
                "session_id": "sess_stream",
                "turn": {
                    "id": "turn_stream",
                    "status": "completed",
                    "created_at": 12,
                    "started_at": 13,
                    "completed_at": 14,
                    "usage": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10},
                },
            },
        ]
    )
    sessions = FakeSessions(created=stream)
    adapter = agents_api.AgentsAPIAdapter(client=fake_client(sessions), monotonic=IncrementingClock())

    result = adapter.run("say hello", model="gpt-test", stream=True)

    assert stream.entered and stream.exited
    assert result.status == "completed"
    assert result.session_id == "sess_stream"
    assert result.turn_ids == ("turn_stream",)
    assert result.output_text == "hello world"
    assert result.usage["total_tokens"] == 10
    assert [event.event_id for event in result.events] == ["evt_1", "evt_2", "evt_3", "evt_4", "evt_5"]


def test_failed_polling_run_records_safe_error():
    secret = "test-secret-never-serialize-this"
    failed_session = SimpleNamespace(
        id="sess_failed",
        status="failed",
        created_at=20,
        last_active_at=22,
        required_actions=[],
        usage=None,
        error=f"provider rejected {secret}",
    )
    failed_turn = SimpleNamespace(
        id="turn_failed",
        status="failed",
        created_at=20,
        started_at=21,
        completed_at=22,
        usage=None,
        error={"code": "tool_error", "message": f"failed near {secret}"},
    )
    sessions = FakeSessions(created=failed_session, turn_pages=[[failed_turn]])
    adapter = agents_api.AgentsAPIAdapter(client=fake_client(sessions), api_key=secret)

    result = adapter.run("fail safely", model="gpt-test")

    assert result.status == "failed"
    assert result.error == {"code": "tool_error", "message": "failed near [REDACTED]"}
    assert secret not in json.dumps(result.to_dict())


def test_submit_tool_result_reuses_durable_ids_and_serializes_json():
    sessions = FakeSessions(created=SimpleNamespace(id="unused"))
    adapter = agents_api.AgentsAPIAdapter(client=fake_client(sessions))

    adapter.submit_tool_result(
        "sess_123",
        "turn_123",
        "call_123",
        output={"ok": True},
        idempotency_key="result:call_123",
    )

    assert sessions.events.created == [
        (
            "sess_123",
            {
                "events": [
                    {
                        "type": "agent.session.input.tool_result",
                        "turn_id": "turn_123",
                        "call_id": "call_123",
                        "success": True,
                        "output": '{"ok": true}',
                    }
                ],
                "idempotency_key": "result:call_123",
            },
        )
    ]


def test_failed_tool_result_and_followup_message_match_current_event_shapes():
    sessions = FakeSessions(created=SimpleNamespace(id="unused"))
    adapter = agents_api.AgentsAPIAdapter(client=fake_client(sessions))

    adapter.submit_tool_result("sess", "turn", "call", error="TEMP_UNAVAILABLE")
    adapter.send_message("sess", "continue", idempotency_key="message-1")

    assert sessions.events.created[0][1]["events"][0] == {
        "type": "agent.session.input.tool_result",
        "turn_id": "turn",
        "call_id": "call",
        "success": False,
        "error": "TEMP_UNAVAILABLE",
    }
    assert sessions.events.created[1] == (
        "sess",
        {
            "events": [
                {
                    "type": "agent.session.input.message",
                    "input": [
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": "continue"}],
                        }
                    ],
                }
            ],
            "idempotency_key": "message-1",
        },
    )


def test_normalized_events_redact_credentials_but_keep_token_usage():
    secret = "test-secret-never-print"
    event = agents_api.normalize_event(
        {
            "type": "agent.session.error",
            "session_id": "sess",
            "authorization": f"Bearer {secret}",
            "api_key": secret,
            "message": f"request for {secret} failed",
            "usage": {"input_tokens": 3},
        },
        known_secret=secret,
    )
    serialized = json.dumps(event.to_dict())

    assert secret not in serialized
    assert event.data["authorization"] == "[REDACTED]"
    assert event.data["usage"]["input_tokens"] == 3


def test_live_cli_without_key_fails_clearly_and_does_not_print_secret(tmp_path):
    environment = os.environ.copy()
    environment.pop("OPENAI_API_KEY", None)
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "agents_api.py"),
            "--live",
            "--prompt",
            "hello",
            "--output",
            str(tmp_path / "never-written.json"),
        ],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )

    assert process.returncode == 2
    assert "OPENAI_API_KEY" in process.stderr
    assert "--simulate" in process.stderr
    assert not (tmp_path / "never-written.json").exists()


def test_simulation_persists_failure_then_resumes_with_same_ids(tmp_path, capsys):
    first = agents_api.main(
        ["--simulate", "--run-dir", str(tmp_path), "--run-id", "trial-1"]
    )
    capsys.readouterr()
    failed_state = json.loads((tmp_path / "state.json").read_text())
    failed_metrics = json.loads((tmp_path / "metrics.json").read_text())

    assert first == 75
    assert failed_state["status"] == "retry_pending"
    assert failed_state["next_step"] == 1
    assert failed_state["simulated"] is True
    assert failed_metrics["api_derived"] is False
    assert failed_metrics["usage"] == {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }
    assert failed_metrics["execution_mode"] == "simulation"
    assert failed_metrics["comparison_eligible"] is False
    durable_ids = (
        failed_state["session_id"],
        tuple(failed_state["turn_ids"]),
        dict(failed_state["call_ids"]),
    )

    second = agents_api.main(
        [
            "--simulate",
            "--run-dir",
            str(tmp_path),
            "--resume",
            "trial-1",
        ]
    )
    second_output = json.loads(capsys.readouterr().out)
    complete_state = json.loads((tmp_path / "state.json").read_text())
    report = json.loads((tmp_path / "report.json").read_text())
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]

    assert second == 0
    assert complete_state["status"] == "completed"
    assert (
        complete_state["session_id"],
        tuple(complete_state["turn_ids"]),
        complete_state["call_ids"],
    ) == durable_ids
    assert report["summary"] == {
        "count": 4,
        "error_count": 1,
        "error_rate": 0.25,
        "mean_latency_ms": 140.0,
        "p95_latency_ms": 200,
    }
    assert set(report) == {"task", "input", "summary", "normalized_records_sha256"}
    assert report["input"] == {
        "source_count": 6,
        "valid_count": 4,
        "dropped": [
            {"id": "job-001", "index": 3, "reason": "duplicate_id"},
            {"id": "job-005", "index": 5, "reason": "invalid_latency"},
        ],
    }
    assert report["normalized_records_sha256"] == (
        "d4ce35528f777b9c847c0ce2b52e2db6511db8a030a1741e087cd560a403c98c"
    )
    assert metrics["tool_calls"] == 4
    assert metrics["successful_tool_calls"] == 3
    assert metrics["retries"] == 1
    assert metrics["usage"]["total_tokens"] is None
    assert metrics["provider_trace_status"] == "not_applicable"
    assert [item["event"] for item in trace].count("tool_call_failed") == 1
    assert all(item["simulated"] is True for item in trace)
    assert all(item["comparison_eligible"] is False for item in trace)
    assert second_output["execution_mode"] == "simulation"
    assert second_output["comparison_eligible"] is False
    assert second_output["provider_trace_status"] == "not_applicable"


def test_tool_result_requires_exactly_one_outcome():
    sessions = FakeSessions(created=SimpleNamespace(id="unused"))
    adapter = agents_api.AgentsAPIAdapter(client=fake_client(sessions))

    with pytest.raises(ValueError, match="exactly one"):
        adapter.submit_tool_result("sess", "turn", "call")
    with pytest.raises(ValueError, match="exactly one"):
        adapter.submit_tool_result("sess", "turn", "call", output="ok", error="bad")
