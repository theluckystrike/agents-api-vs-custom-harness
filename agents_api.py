#!/usr/bin/env python3
"""Small, mockable adapter for the beta OpenAI Agents sessions API.

The library surface intentionally depends only on the Python standard library.
Live mode imports ``openai`` lazily; tests and ``--simulate`` therefore run on a
machine with neither the SDK nor ``OPENAI_API_KEY`` installed.

Official API references used by this adapter:
https://developers.openai.com/api/reference/python/resources/beta/subresources/agents/subresources/sessions/methods/create
https://developers.openai.com/api/docs/guides/agents-api/sessions
https://developers.openai.com/api/docs/guides/agents-api/tools/functions
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


TERMINAL_TURN_STATUSES = frozenset({"completed", "failed", "cancelled"})
TERMINAL_SESSION_STATUSES = frozenset({"failed"})
SENSITIVE_KEYS = frozenset(
    {
        "api-key",
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "openai_api_key",
        "proxy-authorization",
        "secret",
        "set-cookie",
    }
)


class AgentsAPIError(RuntimeError):
    """Base error for the adapter."""


class AgentsAPIUnavailableError(AgentsAPIError):
    """Live Agents API cannot be used in the current environment."""


class AgentsAPIProtocolError(AgentsAPIError):
    """The service or injected mock returned an unusable response."""


class AgentsAPITimeoutError(AgentsAPIError):
    """A session did not reach a reportable state before the local timeout."""


@dataclasses.dataclass(frozen=True)
class AgentEvent:
    """A stable event shape shared by streamed and synthesized poll events."""

    type: str
    event_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    observed_at: str = ""
    data: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class AgentRunResult:
    """Serializable evidence from one initial session turn."""

    mode: str
    model: str
    status: str
    session_id: str | None
    turn_ids: tuple[str, ...]
    started_at: str
    completed_at: str
    elapsed_ms: int
    usage: dict[str, Any] | None
    server_timestamps: dict[str, int | None]
    events: tuple[AgentEvent, ...]
    output_text: str = ""
    required_actions: tuple[dict[str, Any], ...] = ()
    error: dict[str, str] | None = None
    simulated: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["turn_ids"] = list(self.turn_ids)
        value["events"] = [event.to_dict() for event in self.events]
        value["required_actions"] = list(self.required_actions)
        return value


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _object_to_dict(value: Any) -> Any:
    """Turn SDK models, test doubles, and plain JSON values into JSON values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _object_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_object_to_dict(item) for item in value]
    if dataclasses.is_dataclass(value):
        return _object_to_dict(dataclasses.asdict(value))
    for method_name in ("model_dump", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _object_to_dict(method())
            except TypeError:
                # A third-party object's method may require unrelated args.
                pass
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            str(key): _object_to_dict(item)
            for key, item in attributes.items()
            if not str(key).startswith("_")
        }
    return str(value)


def _redact(value: Any, *, known_secret: str | None = None) -> Any:
    """Remove credentials before an API value or exception becomes output."""

    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in SENSITIVE_KEYS:
                clean[key_text] = "[REDACTED]"
            else:
                clean[key_text] = _redact(item, known_secret=known_secret)
        return clean
    if isinstance(value, list):
        return [_redact(item, known_secret=known_secret) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact(item, known_secret=known_secret) for item in value)
    if isinstance(value, str) and known_secret:
        return value.replace(known_secret, "[REDACTED]")
    return value


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _error_dict(error: Any, *, known_secret: str | None = None) -> dict[str, str] | None:
    if error in (None, ""):
        return None
    converted = _redact(_object_to_dict(error), known_secret=known_secret)
    if isinstance(converted, Mapping):
        code = str(converted.get("code") or type(error).__name__)
        message = str(converted.get("message") or converted.get("error") or code)
    else:
        code = type(error).__name__ if not isinstance(error, str) else "api_error"
        message = str(converted)
    return {"code": code, "message": message}


def _turn_from_event(value: Any) -> Any:
    return _field(value, "turn")


def normalize_event(value: Any, *, known_secret: str | None = None) -> AgentEvent:
    """Normalize any SDK event without coupling to generated SDK classes."""

    payload = _redact(_object_to_dict(value), known_secret=known_secret)
    if not isinstance(payload, dict):
        payload = {"value": payload}
    nested_session = payload.get("session") or {}
    nested_turn = payload.get("turn") or {}
    session_id = payload.get("session_id") or _field(nested_session, "id")
    turn_id = payload.get("turn_id") or _field(nested_turn, "id")
    return AgentEvent(
        type=str(payload.get("type") or "agent.session.event.unknown"),
        event_id=_optional_str(payload.get("event_id")),
        session_id=_optional_str(session_id),
        turn_id=_optional_str(turn_id),
        observed_at=_utc_now(),
        data=payload,
    )


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _data_from_page(page: Any) -> list[Any]:
    data = _field(page, "data", None)
    if data is not None:
        return list(data)
    if isinstance(page, Iterable) and not isinstance(page, (str, bytes, Mapping)):
        return list(page)
    return []


def _latest_usage(session: Any, turns: Sequence[Any], events: Sequence[AgentEvent]) -> dict[str, Any] | None:
    candidates: list[Any] = [_field(session, "usage")]
    candidates.extend(_field(turn, "usage") for turn in turns)
    for event in events:
        candidates.append(_field(event.data.get("turn", {}), "usage"))
        candidates.append(_field(event.data.get("session", {}), "usage"))
    for usage in reversed(candidates):
        if usage is not None:
            converted = _object_to_dict(usage)
            if isinstance(converted, dict):
                return converted
    return None


def _session_id_from_event(event: AgentEvent) -> str | None:
    return event.session_id or _optional_str(_field(event.data.get("session", {}), "id"))


def _output_delta(event: AgentEvent) -> str:
    if event.type.endswith("output_text.delta"):
        return str(event.data.get("delta") or "")
    if event.type.endswith("output_text.done"):
        return str(event.data.get("text") or "")
    return ""


class AgentsAPIAdapter:
    """Thin beta-sessions wrapper whose client is straightforward to fake.

    Pass an object with ``client.beta.agents.sessions`` to avoid importing the
    OpenAI SDK. When no client is injected, the adapter validates credentials
    and lazily creates ``openai.OpenAI``.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        api_key: str | None = None,
        poll_interval: float = 0.5,
        timeout: float = 300.0,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if poll_interval < 0:
            raise ValueError("poll_interval must be non-negative")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._known_secret = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = client if client is not None else self._build_live_client(api_key)
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._sleep = sleep
        self._monotonic = monotonic

    @staticmethod
    def _build_live_client(api_key: str | None) -> Any:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise AgentsAPIUnavailableError(
                "Live mode requires OPENAI_API_KEY; use --simulate for an offline run."
            )
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise AgentsAPIUnavailableError(
                "Live mode requires the OpenAI Python SDK with beta Agents sessions support."
            ) from exc
        return OpenAI(api_key=key)

    @property
    def sessions(self) -> Any:
        try:
            return self._client.beta.agents.sessions
        except AttributeError as exc:
            raise AgentsAPIUnavailableError(
                "The installed OpenAI SDK does not expose beta.agents.sessions; upgrade the SDK."
            ) from exc

    def create_session(
        self,
        input_text: str,
        *,
        model: str,
        instructions: str = "",
        tools: Sequence[Mapping[str, Any]] = (),
        agent_id: str | None = None,
        metadata: Mapping[str, str] | None = None,
        environment: Mapping[str, Any] | None = None,
        stream: bool = False,
    ) -> Any:
        """Call the official ``client.beta.agents.sessions.create`` method."""

        if not input_text:
            raise ValueError("input_text must not be empty")
        if not model and not agent_id:
            raise ValueError("model is required when agent_id is not supplied")
        kwargs: dict[str, Any] = {
            "environment": dict(environment or {"type": "none"}),
            "input": input_text,
            "stream": stream,
        }
        agent: dict[str, Any] = {}
        if model:
            agent["model"] = model
        if instructions:
            agent["instructions"] = instructions
        if tools:
            agent["tools"] = [dict(tool) for tool in tools]
        if agent_id:
            kwargs["agent_id"] = agent_id
            if agent:
                kwargs["agent"] = agent
        else:
            kwargs["agent"] = agent
        if metadata is not None:
            kwargs["metadata"] = dict(metadata)
        return self.sessions.create(**kwargs)

    def retrieve_session(self, session_id: str) -> Any:
        return self.sessions.retrieve(session_id)

    def list_turns(self, session_id: str) -> list[Any]:
        page = self.sessions.turns.list(session_id, order="asc", limit=100)
        return _data_from_page(page)

    def list_items(self, session_id: str) -> list[Any]:
        """Return saved messages/tool calls for output inspection or recovery."""

        page = self.sessions.items.list(session_id, order="asc", limit=100)
        return _data_from_page(page)

    def retrieve_turn(self, session_id: str, turn_id: str) -> Any:
        return self.sessions.turns.retrieve(turn_id, session_id=session_id)

    def stream_events(self, session_id: str) -> Iterator[AgentEvent]:
        """Yield normalized live events and close SDK streams when supported."""

        source = self.sessions.events.stream(session_id)
        yield from self._normalize_stream(source)

    def _normalize_stream(self, source: Any) -> Iterator[AgentEvent]:
        manager = source if hasattr(source, "__enter__") else contextlib.nullcontext(source)
        with manager as events:
            for raw_event in events:
                yield normalize_event(raw_event, known_secret=self._known_secret)

    def send_message(
        self,
        session_id: str,
        text: str,
        *,
        idempotency_key: str | None = None,
    ) -> Any:
        event = {
            "type": "agent.session.input.message",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                }
            ],
        }
        kwargs: dict[str, Any] = {"events": [event]}
        if idempotency_key:
            kwargs["idempotency_key"] = idempotency_key
        return self.sessions.events.create(session_id, **kwargs)

    def submit_tool_result(
        self,
        session_id: str,
        turn_id: str,
        call_id: str,
        *,
        output: Any = None,
        error: str | None = None,
        idempotency_key: str | None = None,
    ) -> Any:
        """Return one pending function result using its durable IDs."""

        if (error is None) == (output is None):
            raise ValueError("provide exactly one of output or error")
        event: dict[str, Any] = {
            "type": "agent.session.input.tool_result",
            "turn_id": turn_id,
            "call_id": call_id,
            "success": error is None,
        }
        if error is None:
            event["output"] = output if isinstance(output, str) else json.dumps(output, sort_keys=True)
        else:
            event["error"] = error
        kwargs: dict[str, Any] = {"events": [event]}
        if idempotency_key:
            kwargs["idempotency_key"] = idempotency_key
        return self.sessions.events.create(session_id, **kwargs)

    def run(
        self,
        input_text: str,
        *,
        model: str,
        instructions: str = "",
        tools: Sequence[Mapping[str, Any]] = (),
        agent_id: str | None = None,
        metadata: Mapping[str, str] | None = None,
        environment: Mapping[str, Any] | None = None,
        stream: bool = False,
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> AgentRunResult:
        """Start a session and report its initial turn via streaming or polling."""

        started_iso = _utc_now()
        started_clock = self._monotonic()
        try:
            response = self.create_session(
                input_text,
                model=model,
                instructions=instructions,
                tools=tools,
                agent_id=agent_id,
                metadata=metadata,
                environment=environment,
                stream=stream,
            )
            if stream:
                return self._result_from_stream(
                    response,
                    model=model,
                    started_iso=started_iso,
                    started_clock=started_clock,
                    on_event=on_event,
                )
            return self._result_from_polling(
                response,
                model=model,
                started_iso=started_iso,
                started_clock=started_clock,
                on_event=on_event,
            )
        except AgentsAPIError:
            raise
        except Exception as exc:
            clean = _error_dict(exc, known_secret=self._known_secret)
            message = clean["message"] if clean else type(exc).__name__
            raise AgentsAPIError(f"Agents API request failed: {message}") from exc

    def poll_session(
        self,
        session_id: str,
        *,
        model: str = "unknown",
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> AgentRunResult:
        """Recover or continue an existing session through the polling path."""

        started_iso = _utc_now()
        started_clock = self._monotonic()
        try:
            session = self.retrieve_session(session_id)
            actual_id = _optional_str(_field(session, "id"))
            if actual_id != session_id:
                raise AgentsAPIProtocolError(
                    f"sessions.retrieve returned {actual_id!r} for requested session {session_id!r}."
                )
            return self._result_from_polling(
                session,
                model=model,
                started_iso=started_iso,
                started_clock=started_clock,
                on_event=on_event,
            )
        except AgentsAPIError:
            raise
        except Exception as exc:
            clean = _error_dict(exc, known_secret=self._known_secret)
            message = clean["message"] if clean else type(exc).__name__
            raise AgentsAPIError(f"Agents API polling failed: {message}") from exc

    def _result_from_stream(
        self,
        source: Any,
        *,
        model: str,
        started_iso: str,
        started_clock: float,
        on_event: Callable[[AgentEvent], None] | None,
    ) -> AgentRunResult:
        events: list[AgentEvent] = []
        session_id: str | None = None
        turns: dict[str, Any] = {}
        session: Any = None
        status = "in_progress"
        output_parts: list[str] = []
        for event in self._normalize_stream(source):
            events.append(event)
            if on_event:
                on_event(event)
            session_id = session_id or _session_id_from_event(event)
            if isinstance(event.data.get("session"), Mapping):
                session = event.data["session"]
                status = str(_field(session, "status", status))
            turn = event.data.get("turn")
            turn_id = event.turn_id or _optional_str(_field(turn, "id"))
            if turn_id:
                turns[turn_id] = turn or {"id": turn_id}
                turn_status = _field(turn, "status")
                if turn_status:
                    status = str(turn_status)
            delta = _output_delta(event)
            if delta:
                # Prefer deltas; a final done event may contain their concatenation.
                if event.type.endswith(".delta"):
                    output_parts.append(delta)
                elif not output_parts:
                    output_parts.append(delta)
            if event.type.endswith("turn.completed"):
                status = "completed"
            elif event.type.endswith("turn.failed"):
                status = "failed"
            elif event.type.endswith("turn.cancelled"):
                status = "cancelled"
            elif event.type.endswith("requires_action"):
                status = "requires_action"

        if not session_id:
            raise AgentsAPIProtocolError("The session event stream ended without a session_id.")
        turn_values = list(turns.values())
        error = _error_dict(
            _field(turn_values[-1], "error") if turn_values else None,
            known_secret=self._known_secret,
        )
        error = error or _error_dict(_field(session, "error"), known_secret=self._known_secret)
        return self._make_result(
            model=model,
            status=status,
            session_id=session_id,
            turns=turn_values,
            session=session,
            events=events,
            started_iso=started_iso,
            started_clock=started_clock,
            output_text="".join(output_parts),
            error=error,
        )

    def _result_from_polling(
        self,
        created: Any,
        *,
        model: str,
        started_iso: str,
        started_clock: float,
        on_event: Callable[[AgentEvent], None] | None,
    ) -> AgentRunResult:
        session_id = _optional_str(_field(created, "id"))
        if not session_id:
            raise AgentsAPIProtocolError("sessions.create returned no session id.")
        session = created
        turns: list[Any] = []
        events: list[AgentEvent] = []
        previous_signature: tuple[Any, ...] | None = None

        while True:
            turns = self.list_turns(session_id)
            signature = (
                _field(session, "status"),
                tuple((_field(turn, "id"), _field(turn, "status")) for turn in turns),
            )
            if signature != previous_signature:
                poll_event = self._poll_event(session, session_id, turns)
                events.append(poll_event)
                if on_event:
                    on_event(poll_event)
                previous_signature = signature

            latest_turn = turns[-1] if turns else None
            turn_status = str(_field(latest_turn, "status", ""))
            session_status = str(_field(session, "status", ""))
            if turn_status in TERMINAL_TURN_STATUSES:
                status = turn_status
                break
            if session_status in {"requires_action", *TERMINAL_SESSION_STATUSES}:
                status = session_status
                break
            if self._monotonic() - started_clock >= self.timeout:
                raise AgentsAPITimeoutError(
                    f"Session {session_id} did not reach a terminal or actionable state within "
                    f"{self.timeout:g}s."
                )
            self._sleep(self.poll_interval)
            session = self.retrieve_session(session_id)

        error = _error_dict(_field(latest_turn, "error"), known_secret=self._known_secret)
        error = error or _error_dict(_field(session, "error"), known_secret=self._known_secret)
        return self._make_result(
            model=model,
            status=status,
            session_id=session_id,
            turns=turns,
            session=session,
            events=events,
            started_iso=started_iso,
            started_clock=started_clock,
            output_text="",
            error=error,
        )

    def _poll_event(self, session: Any, session_id: str, turns: Sequence[Any]) -> AgentEvent:
        session_data = _redact(_object_to_dict(session), known_secret=self._known_secret)
        turn_data = [_redact(_object_to_dict(turn), known_secret=self._known_secret) for turn in turns]
        latest = turns[-1] if turns else None
        return AgentEvent(
            type="adapter.session.poll",
            session_id=session_id,
            turn_id=_optional_str(_field(latest, "id")),
            observed_at=_utc_now(),
            data={"session": session_data, "turns": turn_data},
        )

    def _make_result(
        self,
        *,
        model: str,
        status: str,
        session_id: str,
        turns: Sequence[Any],
        session: Any,
        events: Sequence[AgentEvent],
        started_iso: str,
        started_clock: float,
        output_text: str,
        error: dict[str, str] | None,
    ) -> AgentRunResult:
        completed_iso = _utc_now()
        elapsed_ms = max(0, round((self._monotonic() - started_clock) * 1000))
        turn_ids = tuple(
            str(turn_id)
            for turn in turns
            if (turn_id := _field(turn, "id")) is not None
        )
        latest_turn = turns[-1] if turns else None
        required_actions_raw = _field(session, "required_actions", []) or []
        required_actions = tuple(
            _redact(_object_to_dict(action), known_secret=self._known_secret)
            for action in required_actions_raw
        )
        timestamps = {
            "session_created_at": _field(session, "created_at"),
            "session_last_active_at": _field(session, "last_active_at"),
            "turn_created_at": _field(latest_turn, "created_at"),
            "turn_started_at": _field(latest_turn, "started_at"),
            "turn_completed_at": _field(latest_turn, "completed_at"),
        }
        return AgentRunResult(
            mode="live",
            model=model,
            status=status,
            session_id=session_id,
            turn_ids=turn_ids,
            started_at=started_iso,
            completed_at=completed_iso,
            elapsed_ms=elapsed_ms,
            usage=_latest_usage(session, turns, events),
            server_timestamps=timestamps,
            events=tuple(events),
            output_text=output_text,
            required_actions=required_actions,
            error=error,
            simulated=False,
        )


def _sim_id(kind: str, run_id: str, ordinal: int = 0) -> str:
    digest = hashlib.sha256(f"agents-api-sim:{run_id}:{kind}:{ordinal}".encode()).hexdigest()[:16]
    return f"sim_{kind}_{digest}"


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _sim_event(
    event: str,
    *,
    run_id: str,
    session_id: str,
    turn_id: str,
    call_id: str | None = None,
    step: int | None = None,
    tool: str | None = None,
    attempt: int | None = None,
    error: Mapping[str, Any] | None = None,
    **details: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "event": event,
        "run_id": run_id,
        "session_id": session_id,
        "turn_id": turn_id,
        "simulated": True,
        "execution_mode": "simulation",
        "comparison_eligible": False,
        "provider_trace_status": "not_applicable",
        "source": "deterministic_simulation_not_openai_api",
        "timestamp": _utc_now(),
    }
    if call_id is not None:
        result["call_id"] = call_id
    if step is not None:
        result["step"] = step
    if tool is not None:
        result["tool"] = tool
    if attempt is not None:
        result["attempt"] = attempt
    if error is not None:
        result["error"] = dict(error)
    result.update(details)
    return result


def run_simulation(run_dir: Path, run_id: str, *, resume: str | None = None) -> int:
    """Run the deterministic offline comparison arm.

    The first invocation persists a deliberate retryable failure and returns
    EX_TEMPFAIL (75). ``--resume RUN_ID`` reuses all durable IDs and completes.
    Nothing produced here is represented as API-derived measurement.
    """

    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    trace_path = run_dir / "trace.jsonl"
    metrics_path = run_dir / "metrics.json"
    report_path = run_dir / "report.json"
    session_id = _sim_id("session", run_id)
    turn_id = _sim_id("turn", run_id)
    calls = {
        "normalize_records": _sim_id("call", run_id, 1),
        "summarize_records": _sim_id("call", run_id, 2),
        "write_report": _sim_id("call", run_id, 3),
    }
    now = _utc_now()

    if resume is None:
        if state_path.exists():
            raise AgentsAPIError(
                f"Simulation state already exists at {state_path}; resume it with --resume {run_id}."
            )
        events = [
            _sim_event("run_started", run_id=run_id, session_id=session_id, turn_id=turn_id),
            _sim_event(
                "tool_call_started",
                run_id=run_id,
                session_id=session_id,
                turn_id=turn_id,
                call_id=calls["normalize_records"],
                step=0,
                tool="normalize_records",
                attempt=1,
            ),
            _sim_event(
                "tool_call_succeeded",
                run_id=run_id,
                session_id=session_id,
                turn_id=turn_id,
                call_id=calls["normalize_records"],
                step=0,
                tool="normalize_records",
                attempt=1,
            ),
            _sim_event(
                "checkpoint_saved",
                run_id=run_id,
                session_id=session_id,
                turn_id=turn_id,
                step=0,
                tool="normalize_records",
            ),
            _sim_event(
                "tool_call_started",
                run_id=run_id,
                session_id=session_id,
                turn_id=turn_id,
                call_id=calls["summarize_records"],
                step=1,
                tool="summarize_records",
                attempt=1,
            ),
            _sim_event(
                "tool_call_failed",
                run_id=run_id,
                session_id=session_id,
                turn_id=turn_id,
                call_id=calls["summarize_records"],
                step=1,
                tool="summarize_records",
                attempt=1,
                error_code="TEMP_UNAVAILABLE",
                retryable=True,
            ),
            _sim_event(
                "retry_scheduled",
                run_id=run_id,
                session_id=session_id,
                turn_id=turn_id,
                call_id=calls["summarize_records"],
                step=1,
                tool="summarize_records",
                attempt=2,
            ),
            _sim_event(
                "run_paused",
                run_id=run_id,
                session_id=session_id,
                turn_id=turn_id,
                step=1,
                tool="summarize_records",
                error_code="TEMP_UNAVAILABLE",
                retryable=True,
            ),
        ]
        for event in events:
            _append_jsonl(trace_path, event)
        state = {
            "schema_version": 1,
            "arm": "agents_api",
            "run_id": run_id,
            "workflow_version": "durable-three-tool-pipeline-v1",
            "status": "retry_pending",
            "next_step": 1,
            "completed_steps": [{"index": 0, "tool": "normalize_records"}],
            "session_id": session_id,
            "turn_ids": [turn_id],
            "call_ids": calls,
            "attempts": {"normalize_records": 1, "summarize_records": 1, "write_report": 0},
            "results": {
                "normalize_records": {
                    "source_count": 6,
                    "valid_count": 4,
                    "records": [
                        {"id": "job-001", "latency_ms": 120, "status": "ok"},
                        {"id": "job-002", "latency_ms": 200, "status": "error"},
                        {"id": "job-003", "latency_ms": 80, "status": "ok"},
                        {"id": "job-004", "latency_ms": 160, "status": "ok"},
                    ],
                    "dropped": [
                        {"id": "job-001", "index": 3, "reason": "duplicate_id"},
                        {"id": "job-005", "index": 5, "reason": "invalid_latency"},
                    ],
                },
            },
            "transient_failure_injected": True,
            "last_error": {"code": "TEMP_UNAVAILABLE", "retryable": True},
            "metrics": {"wall_time_ms": 0.0, "tool_calls": 2, "successful_tool_calls": 1, "retries": 1},
            "simulated": True,
            "execution_mode": "simulation",
            "comparison_eligible": False,
            "provider_trace_status": "not_applicable",
            "source": "deterministic_simulation_not_openai_api",
            "created_at": now,
            "updated_at": _utc_now(),
        }
        metrics = {
            "schema_version": 1,
            "arm": "agents_api",
            "run_id": run_id,
            "status": "retry_pending",
            "model": None,
            "session_id": session_id,
            "turn_ids": [turn_id],
            "tool_calls": 2,
            "successful_tool_calls": 1,
            "retry_count": 1,
            "retries": 1,
            "wall_time_ms": 0,
            "usage": {"input_tokens": None, "output_tokens": None, "total_tokens": None},
            "cost_usd": None,
            "simulated": True,
            "execution_mode": "simulation",
            "comparison_eligible": False,
            "provider_trace_status": "not_applicable",
            "api_derived": False,
            "source": "deterministic_simulation_not_openai_api",
            "started_at": now,
            "completed_at": None,
        }
        _atomic_write_json(state_path, state)
        _atomic_write_json(metrics_path, metrics)
        print(json.dumps({"status": state["status"], "run_id": run_id, "simulated": True}))
        return 75

    if resume != run_id:
        raise AgentsAPIError("--resume must equal --run-id so a different logical run cannot reuse state.")
    if not state_path.exists():
        raise AgentsAPIError(f"No persisted simulation state exists at {state_path}.")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("run_id") != run_id or state.get("session_id") != session_id:
        raise AgentsAPIError("Persisted state does not belong to this logical simulation run.")
    if state.get("status") == "completed":
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "status": "completed",
                    "run_id": run_id,
                    "report": report,
                    "execution_mode": "simulation",
                    "comparison_eligible": False,
                    "provider_trace_status": "not_applicable",
                },
                sort_keys=True,
            )
        )
        return 0
    if state.get("status") != "retry_pending":
        raise AgentsAPIError(f"Simulation state is not resumable: {state.get('status')!r}.")

    resume_events = [
        _sim_event("run_resumed", run_id=run_id, session_id=session_id, turn_id=turn_id),
        _sim_event(
            "step_skipped",
            run_id=run_id,
            session_id=session_id,
            turn_id=turn_id,
            step=0,
            tool="normalize_records",
            reason="checkpoint_present",
        ),
        _sim_event(
            "tool_call_started",
            run_id=run_id,
            session_id=session_id,
            turn_id=turn_id,
            call_id=calls["summarize_records"],
            step=1,
            tool="summarize_records",
            attempt=2,
        ),
        _sim_event(
            "tool_call_succeeded",
            run_id=run_id,
            session_id=session_id,
            turn_id=turn_id,
            call_id=calls["summarize_records"],
            step=1,
            tool="summarize_records",
            attempt=2,
        ),
        _sim_event(
            "checkpoint_saved",
            run_id=run_id,
            session_id=session_id,
            turn_id=turn_id,
            step=1,
            tool="summarize_records",
        ),
        _sim_event(
            "tool_call_started",
            run_id=run_id,
            session_id=session_id,
            turn_id=turn_id,
            call_id=calls["write_report"],
            step=2,
            tool="write_report",
            attempt=1,
        ),
        _sim_event(
            "tool_call_succeeded",
            run_id=run_id,
            session_id=session_id,
            turn_id=turn_id,
            call_id=calls["write_report"],
            step=2,
            tool="write_report",
            attempt=1,
        ),
        _sim_event("run_completed", run_id=run_id, session_id=session_id, turn_id=turn_id),
    ]
    for event in resume_events:
        _append_jsonl(trace_path, event)
    completed_at = _utc_now()
    report = {
        "task": "durable-three-tool-pipeline-v1",
        "input": {
            "source_count": 6,
            "valid_count": 4,
            "dropped": [
                {"id": "job-001", "index": 3, "reason": "duplicate_id"},
                {"id": "job-005", "index": 5, "reason": "invalid_latency"},
            ],
        },
        "summary": {
            "count": 4,
            "error_count": 1,
            "error_rate": 0.25,
            "mean_latency_ms": 140.0,
            "p95_latency_ms": 200,
        },
        "normalized_records_sha256": "d4ce35528f777b9c847c0ce2b52e2db6511db8a030a1741e087cd560a403c98c",
    }
    state.update(
        status="completed",
        next_step=None,
        completed_steps=[
            {"index": 0, "tool": "normalize_records"},
            {"index": 1, "tool": "summarize_records"},
            {"index": 2, "tool": "write_report"},
        ],
        attempts={"normalize_records": 1, "summarize_records": 2, "write_report": 1},
        results={
            **state.get("results", {}),
            "summarize_records": report["summary"],
            "write_report": report,
        },
        last_error=None,
        metrics={"wall_time_ms": 0.0, "tool_calls": 4, "successful_tool_calls": 3, "retries": 1},
        updated_at=completed_at,
    )
    metrics = {
        "schema_version": 1,
        "arm": "agents_api",
        "run_id": run_id,
        "status": "completed",
        "model": None,
        "session_id": session_id,
        "turn_ids": [turn_id],
        "tool_calls": 4,
        "successful_tool_calls": 3,
        "retry_count": 1,
        "retries": 1,
        "wall_time_ms": 0,
        "usage": {"input_tokens": None, "output_tokens": None, "total_tokens": None},
        "cost_usd": None,
        "simulated": True,
        "execution_mode": "simulation",
        "comparison_eligible": False,
        "provider_trace_status": "not_applicable",
        "api_derived": False,
        "source": "deterministic_simulation_not_openai_api",
        "started_at": state.get("created_at"),
        "completed_at": completed_at,
    }
    _atomic_write_json(report_path, report)
    _atomic_write_json(state_path, state)
    _atomic_write_json(metrics_path, metrics)
    print(
        json.dumps(
            {
                "status": "completed",
                "run_id": run_id,
                "report": report,
                "execution_mode": "simulation",
                "comparison_eligible": False,
                "provider_trace_status": "not_applicable",
            },
            sort_keys=True,
        )
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--simulate", action="store_true", help="run deterministic offline control flow")
    mode.add_argument("--live", action="store_true", help="call the beta Agents sessions API")
    parser.add_argument("--run-dir", type=Path, help="directory for simulation state and evidence")
    parser.add_argument("--run-id", help="durable logical run ID for a new simulation")
    parser.add_argument("--resume", metavar="RUN_ID", help="resume a persisted simulation run")
    parser.add_argument(
        "--model",
        default="gpt-5.6-terra",
        help="live-mode model (frozen benchmark default: gpt-5.6-terra)",
    )
    parser.add_argument("--prompt", help="initial live-mode input")
    parser.add_argument("--instructions", default="", help="live-mode agent instructions")
    parser.add_argument("--stream", action="store_true", help="stream live session events")
    parser.add_argument("--output", type=Path, help="also write the live result as JSON")
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.simulate:
            if args.run_dir is None:
                parser.error("--simulate requires --run-dir")
            if args.run_id and args.resume:
                parser.error("--run-id and --resume are mutually exclusive in simulation mode")
            run_id = args.resume or args.run_id
            if not run_id:
                parser.error("--simulate requires either --run-id or --resume RUN_ID")
            return run_simulation(args.run_dir, run_id, resume=args.resume)
        if args.resume:
            parser.error("--resume is valid only with --simulate")
        if not args.prompt:
            parser.error("--live requires --prompt")
        adapter = AgentsAPIAdapter(poll_interval=args.poll_interval, timeout=args.timeout)
        result = adapter.run(
            args.prompt,
            model=args.model,
            instructions=args.instructions,
            stream=args.stream,
        ).to_dict()
        if args.output:
            _atomic_write_json(args.output, result)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "completed" else 1
    except AgentsAPIError as exc:
        # Deliberately print only the sanitized adapter message, never config or keys.
        print(f"agents_api: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
