#!/usr/bin/env python3
"""Live managed-Agents arm for the preregistered recovery benchmark.

This module deliberately has no simulation path.  Provider calls are made only
when ``--live`` is explicit, credentials exist, and the installed OpenAI SDK
exposes beta Agents sessions.  Tests inject an adapter and never contact the
provider.

The recovery protocol follows the current Agents API contract: only current
``session.required_actions`` authorizes local function execution, and saved
tool results are returned with their original session, turn, and call IDs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from agents_api import AgentsAPIAdapter
from custom_harness import normalize_records, summarize_records, write_report


SCHEMA_VERSION = "1.0.0"
ARM = "agents_api"
EXIT_TEMPORARY_FAILURE = 75
CHECKPOINTS = (
    "C0_INITIALIZED",
    "C1_NORMALIZED",
    "C2_FAILURE_PERSISTED",
    "C3_SUMMARIZED",
    "C4_REPORT_COMMITTED",
    "C5_VERIFIED",
)
MAX_TOOL_ATTEMPTS = 8
MAX_MODEL_CALLS = 8

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


class ManagedLiveError(RuntimeError):
    """Base class for benchmark driver failures."""


class ManagedLiveUnavailable(ManagedLiveError):
    """Live mode cannot be started in this environment."""


class ManagedLiveProtocolError(ManagedLiveError):
    """The managed session violated the frozen benchmark protocol."""


class ManagedLiveTimeout(ManagedLiveError):
    """The managed session did not reach an actionable state in time."""


class PlannedInterruption(ManagedLiveError):
    """Required crash boundary after the durable injected failure."""

    exit_code = EXIT_TEMPORARY_FAILURE

    def __init__(self, run_id: str) -> None:
        super().__init__(
            f"run {run_id!r} paused after C2_FAILURE_PERSISTED: TEMP_UNAVAILABLE"
        )
        self.run_id = run_id


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    for method_name in ("model_dump", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _plain(method())
            except TypeError:
                pass
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            str(key): _plain(item)
            for key, item in attributes.items()
            if not str(key).startswith("_")
        }
    return str(value)


def _redact(value: Any, known_secret: str | None = None) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]"
                if str(key).lower() in SENSITIVE_KEYS
                else _redact(item, known_secret)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item, known_secret) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact(item, known_secret) for item in value)
    if isinstance(value, str):
        clean = value.replace(known_secret, "[REDACTED]") if known_secret else value
        return re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", clean)
    return value


def _safe_error(error: BaseException, known_secret: str | None = None) -> dict[str, str]:
    return {
        "code": type(error).__name__,
        "message": str(_redact(str(error), known_secret)),
    }


def _atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("wb") as handle:
            handle.write(_canonical_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManagedLiveError(f"required file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManagedLiveError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManagedLiveError(f"expected a JSON object in {path}")
    return value


def _load_experiment(path: Path) -> tuple[dict[str, Any], str]:
    value = _load_json(path)
    if value.get("protocol_status") != "preregistered":
        raise ManagedLiveError("experiment is not the frozen preregistered protocol")
    return value, hashlib.sha256(path.read_bytes()).hexdigest()


def _api_tools(experiment: Mapping[str, Any]) -> list[dict[str, Any]]:
    allowed = ("type", "name", "description", "parameters")
    return [
        {key: _plain(tool[key]) for key in allowed if key in tool}
        for tool in experiment["tools"]
    ]


def _task_input(experiment: Mapping[str, Any]) -> str:
    """Render the exact same frozen user input used by the Responses arm."""

    task = experiment["task"]
    return (
        f'{task["user_instruction"]}\n\n'
        f'Dataset ID: {task["dataset_id"]}\n'
        "Records (preserve exactly):\n"
        + json.dumps(task["raw_records"], ensure_ascii=False, separators=(",", ":"))
    )


def _usage_from_provider(value: Any) -> dict[str, int | None] | None:
    if value is None:
        return None
    usage = _plain(value)
    if not isinstance(usage, Mapping):
        return None
    input_details = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    return {
        "input_tokens": usage.get("input_tokens"),
        "cached_input_tokens": (
            input_details.get("cached_tokens")
            if isinstance(input_details, Mapping)
            else None
        ),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": (
            output_details.get("reasoning_tokens")
            if isinstance(output_details, Mapping)
            else None
        ),
        "total_tokens": usage.get("total_tokens"),
    }


def _iso_to_epoch_ms(value: str) -> float:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000


class ManagedLiveDriver:
    """Durable controller for exactly one managed-session benchmark run."""

    def __init__(
        self,
        *,
        adapter: Any,
        experiment_path: str | Path | None = None,
        api_key: str | None = None,
        poll_interval: float = 0.25,
        timeout: float = 300.0,
        sleep: Callable[[float], None] = time.sleep,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        utc_now: Callable[[], str] = _utc_now,
        process_id: Callable[[], int] = os.getpid,
    ) -> None:
        if poll_interval < 0:
            raise ValueError("poll_interval must be non-negative")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.adapter = adapter
        self.experiment_path = Path(experiment_path or Path(__file__).with_name("experiment.json"))
        self.experiment, self.config_sha256 = _load_experiment(self.experiment_path)
        self.api_key = api_key
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.sleep = sleep
        self.monotonic_ns = monotonic_ns
        self.utc_now = utc_now
        self.process_id = process_id
        self.run_dir = Path(".")
        self.state: dict[str, Any] = {}
        self.process_instance_id = ""
        self.resumed = False

    def run(self, run_dir: str | Path, run_id: str, *, resume: bool = False) -> dict[str, Any]:
        if not run_id or not run_id.strip():
            raise ManagedLiveError("logical run ID must not be empty")
        self.run_dir = Path(run_dir)
        current_pid = self.process_id()
        self.process_instance_id = f"proc-{current_pid}-{uuid.uuid4().hex}"
        self.resumed = resume
        invocation_started_ns = self.monotonic_ns()
        invocation_started_utc = self.utc_now()

        if resume:
            self.state = self._load_state(run_id)
            if self.state["checkpoint"] != "C2_FAILURE_PERSISTED":
                if self.state["checkpoint"] == "C5_VERIFIED":
                    return _load_json(self.run_dir / "report.json")
                raise ManagedLiveProtocolError(
                    "resume is permitted only from C2_FAILURE_PERSISTED"
                )
            self.state["resume_count"] += 1
            self.state["processes"].append(
                {
                    "process_instance_id": self.process_instance_id,
                    "pid": current_pid,
                    "resumed": True,
                    "started_at": invocation_started_utc,
                }
            )
            self.state["timing"]["recovery_started_at"] = invocation_started_utc
            self._save_state()
            self._write_manifest()
            self._trace("process_resumed", checkpoint=self.state["checkpoint"])
            session = self._retrieve_session(self.state["session_id"])
        else:
            if (self.run_dir / "state.json").exists():
                raise ManagedLiveError(
                    f"state already exists in {self.run_dir}; use --resume {run_id}"
                )
            self.run_dir.mkdir(parents=True, exist_ok=True)
            session = self._create_session(run_id)
            session_id = str(_field(session, "id") or "")
            if not session_id:
                raise ManagedLiveProtocolError("created session did not return an ID")
            returned_model = str(_field(_field(session, "agent", {}), "model", ""))
            requested_model = self.experiment["model"]["requested_id"]
            if returned_model and returned_model != requested_model:
                raise ManagedLiveProtocolError(
                    f"provider returned model {returned_model!r}, expected {requested_model!r}"
                )
            self.state = self._initial_state(
                run_id,
                session_id,
                returned_model or requested_model,
                invocation_started_utc,
                invocation_started_ns,
                current_pid,
            )
            self._write_manifest()
            self._checkpoint("C0_INITIALIZED")

        try:
            report = self._drive(session)
        except PlannedInterruption:
            self.state["timing"]["initial_finished_at"] = self.utc_now()
            self.state["timing"]["initial_wall_ms"] = round(
                (self.monotonic_ns() - invocation_started_ns) / 1_000_000, 3
            )
            self._save_state()
            self._write_metrics()
            raise
        except Exception as exc:
            self.state["status"] = "failed"
            self.state["error"] = _safe_error(exc, self.api_key)
            self._save_state()
            self._write_metrics()
            raise

        recovery_wall_ms = round(
            (self.monotonic_ns() - invocation_started_ns) / 1_000_000, 3
        )
        self.state["timing"]["recovery_wall_ms"] = recovery_wall_ms
        initial = self.state["timing"].get("initial_wall_ms")
        if initial is not None:
            self.state["timing"]["active_wall_ms"] = round(initial + recovery_wall_ms, 3)
        self.state["timing"]["finished_at"] = self.utc_now()
        self.state["timing"]["end_to_end_wall_ms"] = round(
            _iso_to_epoch_ms(self.state["timing"]["finished_at"])
            - _iso_to_epoch_ms(self.state["timing"]["started_at"]),
            3,
        )
        if self.state["timing"].get("initial_finished_at"):
            self.state["timing"]["restart_gap_ms"] = round(
                _iso_to_epoch_ms(self.state["timing"]["recovery_started_at"])
                - _iso_to_epoch_ms(self.state["timing"]["initial_finished_at"]),
                3,
            )
        self._save_state()
        self._write_metrics()
        return report

    def _initial_state(
        self,
        run_id: str,
        session_id: str,
        returned_model: str,
        started_at: str,
        started_ns: int,
        current_pid: int,
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "benchmark_id": self.experiment["benchmark_id"],
            "config_sha256": self.config_sha256,
            "execution_mode": "live",
            "comparison_eligible": True,
            "arm": ARM,
            "run_id": run_id,
            "session_id": session_id,
            "returned_model": returned_model,
            "checkpoint": None,
            "checkpoints": [],
            "status": "running",
            "processes": [
                {
                    "process_instance_id": self.process_instance_id,
                    "pid": current_pid,
                    "resumed": False,
                    "started_at": started_at,
                }
            ],
            "turn_ids": [],
            "call_ids": [],
            "receipts": {},
            "tool_attempts": [],
            "fault": {"consumed": False, "key": None, "result": None},
            "resume_count": 0,
            "duplicate_tool_executions": 0,
            "duplicate_side_effects": 0,
            "provider_requests": 1,
            "provider_usage": None,
            "provider_trace_status": "unavailable",
            "provider_trace_error": "provider trace export is not implemented by this driver",
            "trace_seq": 0,
            "checkpoint_bytes": {},
            "timing": {
                "started_at": started_at,
                "started_monotonic_ns": started_ns,
                "initial_finished_at": None,
                "recovery_started_at": None,
                "finished_at": None,
                "initial_wall_ms": None,
                "restart_gap_ms": None,
                "recovery_wall_ms": None,
                "active_wall_ms": None,
                "end_to_end_wall_ms": None,
            },
            "report_receipt": None,
            "validation": None,
            "error": None,
        }

    def _create_session(self, run_id: str) -> Any:
        model = self.experiment["model"]
        task = self.experiment["task"]
        payload = {
            "environment": {"type": "none"},
            "input": _task_input(self.experiment),
            "stream": False,
            "agent": {
                "model": model["requested_id"],
                "instructions": task["developer_instruction"],
                "reasoning": {"effort": model["reasoning_effort"]},
                "text": {
                    "format": {"type": "text"},
                    "verbosity": model["text_verbosity"],
                },
                "service_tier": model["service_tier"],
                "tools": _api_tools(self.experiment),
            },
            "metadata": {
                "benchmark_id": self.experiment["benchmark_id"],
                "logical_run_id": run_id,
            },
        }
        # Direct access is intentional: the generic adapter omits benchmark-only
        # inline agent settings such as reasoning, text, and service tier.
        try:
            return self.adapter.sessions.create(**payload)
        except Exception as exc:
            clean = _safe_error(exc, self.api_key)
            raise ManagedLiveError(
                f"managed session creation failed: {clean['message']}"
            ) from None

    def _retrieve_session(self, session_id: str) -> Any:
        self.state["provider_requests"] += 1
        try:
            session = self.adapter.retrieve_session(session_id)
        except Exception as exc:
            clean = _safe_error(exc, self.api_key)
            raise ManagedLiveError(
                f"managed session retrieval failed: {clean['message']}"
            ) from None
        self._capture_session(session)
        self._save_state()
        return session

    def _list_turns(self, session_id: str) -> list[Any]:
        self.state["provider_requests"] += 1
        try:
            turns = list(self.adapter.list_turns(session_id))
        except Exception as exc:
            clean = _safe_error(exc, self.api_key)
            raise ManagedLiveError(
                f"managed turn listing failed: {clean['message']}"
            ) from None
        for turn in turns:
            turn_id = str(_field(turn, "id") or "")
            if turn_id and turn_id not in self.state["turn_ids"]:
                self.state["turn_ids"].append(turn_id)
        self._save_state()
        return turns

    def _submit_receipt(self, receipt: dict[str, Any]) -> None:
        self.state["provider_requests"] += 1
        kwargs: dict[str, Any] = {"idempotency_key": receipt["idempotency_key"]}
        if receipt["success"]:
            kwargs["output"] = receipt["result"]
        else:
            kwargs["error"] = _canonical_bytes(receipt["result"]["error"]).decode(
                "utf-8"
            )
        try:
            self.adapter.submit_tool_result(
                self.state["session_id"],
                receipt["turn_id"],
                receipt["call_id"],
                **kwargs,
            )
        except Exception as exc:
            clean = _safe_error(exc, self.api_key)
            raise ManagedLiveError(
                f"managed tool-result submission failed: {clean['message']}"
            ) from None
        receipt["submitted"] = True
        receipt["submitted_at"] = self.utc_now()
        self._save_state()
        self._trace(
            "tool_result_submitted",
            tool=receipt["tool"],
            attempt=receipt["attempt"],
            turn_id=receipt["turn_id"],
            call_id=receipt["call_id"],
            arguments_sha256=receipt["arguments_sha256"],
            result_sha256=receipt["result_sha256"],
            status="success" if receipt["success"] else "error",
            error_code=None if receipt["success"] else "TEMP_UNAVAILABLE",
        )

    def _capture_session(self, session: Any) -> None:
        session_id = str(_field(session, "id") or "")
        if session_id != self.state["session_id"]:
            raise ManagedLiveProtocolError(
                f"retrieved session {session_id!r} does not match saved session"
            )
        returned_model = str(_field(_field(session, "agent", {}), "model", ""))
        if returned_model and returned_model != self.experiment["model"]["requested_id"]:
            raise ManagedLiveProtocolError(
                f"managed session model changed to {returned_model!r}"
            )
        usage = _field(session, "usage")
        if usage is not None:
            self.state["provider_usage"] = _usage_from_provider(usage)

    def _drive(self, session: Any) -> dict[str, Any]:
        deadline_ns = self.monotonic_ns() + int(self.timeout * 1_000_000_000)
        while True:
            self._capture_session(session)
            actions = list(_field(session, "required_actions", []) or [])
            status = str(_field(session, "status", ""))

            if actions:
                if status != "requires_action":
                    raise ManagedLiveProtocolError(
                        "required_actions were returned without requires_action status"
                    )
                if len(actions) != 1:
                    raise ManagedLiveProtocolError(
                        "sequential benchmark requires exactly one pending action"
                    )
                self._handle_action(actions[0])
                session = self._await_session(deadline_ns)
                continue

            if status == "failed":
                error = _redact(_plain(_field(session, "error")), self.api_key)
                raise ManagedLiveProtocolError(f"managed session failed: {error}")

            if status == "idle":
                if self.state["checkpoint"] != "C4_REPORT_COMMITTED":
                    raise ManagedLiveProtocolError(
                        f"session became idle at {self.state['checkpoint']} before report commit"
                    )
                turns = self._list_turns(self.state["session_id"])
                if not turns:
                    raise ManagedLiveProtocolError("idle session has no recorded turn")
                terminal = str(_field(turns[-1], "status", ""))
                if terminal != "completed":
                    raise ManagedLiveProtocolError(
                        f"latest managed turn ended with {terminal!r}, not 'completed'"
                    )
                usage = _field(turns[-1], "usage")
                if usage is not None:
                    self.state["provider_usage"] = _usage_from_provider(usage)
                if len(set(self.state["turn_ids"])) > MAX_MODEL_CALLS:
                    raise ManagedLiveProtocolError("maximum model-call limit exceeded")
                return self._verify()

            if status not in {"in_progress", "requires_action"}:
                raise ManagedLiveProtocolError(f"unknown managed session status {status!r}")
            session = self._await_session(deadline_ns)

    def _await_session(self, deadline_ns: int) -> Any:
        if self.monotonic_ns() >= deadline_ns:
            raise ManagedLiveTimeout("managed session timed out")
        if self.poll_interval:
            self.sleep(self.poll_interval)
        return self._retrieve_session(self.state["session_id"])

    def _handle_action(self, raw_action: Any) -> None:
        action = _plain(raw_action)
        if not isinstance(action, Mapping) or action.get("type") != "function_call":
            raise ManagedLiveProtocolError("pending action is not a function_call")
        name = str(action.get("name") or "")
        turn_id = str(action.get("turn_id") or "")
        call_id = str(action.get("call_id") or "")
        arguments = action.get("arguments")
        if not name or not turn_id or not call_id or not isinstance(arguments, Mapping):
            raise ManagedLiveProtocolError("pending function action is missing typed fields")
        arguments = _plain(arguments)
        arguments_hash = _sha256(arguments)
        if turn_id not in self.state["turn_ids"]:
            self.state["turn_ids"].append(turn_id)
        if call_id not in self.state["call_ids"]:
            self.state["call_ids"].append(call_id)

        saved = self.state["receipts"].get(call_id)
        if saved is not None:
            if (
                saved["tool"] != name
                or saved["turn_id"] != turn_id
                or saved["arguments_sha256"] != arguments_hash
            ):
                raise ManagedLiveProtocolError("pending call conflicts with its durable receipt")
            self._trace(
                "saved_tool_result_reused",
                tool=name,
                attempt=saved["attempt"],
                turn_id=turn_id,
                call_id=call_id,
                arguments_sha256=arguments_hash,
                result_sha256=saved["result_sha256"],
                status="success" if saved["success"] else "error",
                error_code=None if saved["success"] else "TEMP_UNAVAILABLE",
            )
            self._submit_receipt(saved)
            return

        expected = {
            "C0_INITIALIZED": "normalize_records",
            "C1_NORMALIZED": "summarize_records",
            "C2_FAILURE_PERSISTED": "summarize_records",
            "C3_SUMMARIZED": "write_report",
        }.get(self.state["checkpoint"])
        if name != expected:
            raise ManagedLiveProtocolError(
                f"expected {expected!r} at {self.state['checkpoint']}, got {name!r}"
            )
        if len(self.state["tool_attempts"]) >= MAX_TOOL_ATTEMPTS:
            raise ManagedLiveProtocolError("maximum tool-attempt limit exceeded")
        self._validate_arguments(name, arguments)
        attempt = 1 + sum(
            item["tool"] == name for item in self.state["tool_attempts"]
        )
        started_ns = self.monotonic_ns()
        self._trace(
            "tool_attempt_started",
            tool=name,
            attempt=attempt,
            turn_id=turn_id,
            call_id=call_id,
            arguments_sha256=arguments_hash,
            status="started",
        )

        if name == "summarize_records" and not self.state["fault"]["consumed"]:
            result = _plain(self.experiment["fault"]["result"])
            fault_key = _sha256(
                {
                    "logical_run_id": self.state["run_id"],
                    "tool_name": name,
                    "canonical_arguments_sha256": arguments_hash,
                }
            )
            self.state["fault"] = {
                "consumed": True,
                "key": fault_key,
                "result": result,
            }
            receipt = self._record_receipt(
                name,
                attempt,
                turn_id,
                call_id,
                arguments_hash,
                result,
                success=False,
                started_ns=started_ns,
            )
            self._checkpoint("C2_FAILURE_PERSISTED")
            self._submit_receipt(receipt)
            self._trace(
                "planned_process_exit",
                tool=name,
                attempt=attempt,
                turn_id=turn_id,
                call_id=call_id,
                arguments_sha256=arguments_hash,
                result_sha256=receipt["result_sha256"],
                status="error",
                error_code="TEMP_UNAVAILABLE",
                exit_code=EXIT_TEMPORARY_FAILURE,
            )
            raise PlannedInterruption(self.state["run_id"])

        if name == "normalize_records":
            result = normalize_records(arguments["records"])
            checkpoint = "C1_NORMALIZED"
        elif name == "summarize_records":
            if not self.state["fault"]["consumed"]:
                raise ManagedLiveProtocolError("summary success occurred before injected failure")
            first_summary = next(
                item
                for item in self.state["tool_attempts"]
                if item["tool"] == "summarize_records"
            )
            if first_summary["arguments_sha256"] != arguments_hash:
                raise ManagedLiveProtocolError("summary retry arguments changed after failure")
            result = summarize_records(arguments["normalized"])
            checkpoint = "C3_SUMMARIZED"
        elif name == "write_report":
            result = self._commit_report(arguments)
            checkpoint = "C4_REPORT_COMMITTED"
        else:  # pragma: no cover - guarded by the frozen stage map
            raise ManagedLiveProtocolError(f"unknown tool {name!r}")

        receipt = self._record_receipt(
            name,
            attempt,
            turn_id,
            call_id,
            arguments_hash,
            result,
            success=True,
            started_ns=started_ns,
        )
        self._checkpoint(checkpoint)
        self._submit_receipt(receipt)

    def _validate_arguments(self, name: str, arguments: Mapping[str, Any]) -> None:
        task = self.experiment["task"]
        if name == "normalize_records":
            expected = {
                "dataset_id": task["dataset_id"],
                "records": task["raw_records"],
            }
        elif name == "summarize_records":
            normalize_receipt = self._successful_receipt("normalize_records")
            expected = {"normalized": normalize_receipt["result"]}
        elif name == "write_report":
            expected = {
                "normalized": self._successful_receipt("normalize_records")["result"],
                "summary": self._successful_receipt("summarize_records")["result"],
            }
        else:
            raise ManagedLiveProtocolError(f"function {name!r} is not registered")
        if _canonical_bytes(arguments) != _canonical_bytes(expected):
            raise ManagedLiveProtocolError(f"{name} arguments differ from the frozen fixture")

    def _successful_receipt(self, tool: str) -> dict[str, Any]:
        values = [
            receipt
            for receipt in self.state["receipts"].values()
            if receipt["tool"] == tool and receipt["success"]
        ]
        if len(values) != 1:
            raise ManagedLiveProtocolError(
                f"expected one successful durable receipt for {tool}, found {len(values)}"
            )
        return values[0]

    def _record_receipt(
        self,
        tool: str,
        attempt: int,
        turn_id: str,
        call_id: str,
        arguments_sha256: str,
        result: Any,
        *,
        success: bool,
        started_ns: int,
    ) -> dict[str, Any]:
        semantic_attempt = attempt
        receipt = {
            "session_id": self.state["session_id"],
            "turn_id": turn_id,
            "call_id": call_id,
            "tool": tool,
            "attempt": attempt,
            "semantic_attempt": semantic_attempt,
            "arguments_sha256": arguments_sha256,
            "result": _redact(_plain(result), self.api_key),
            "result_sha256": _sha256(result),
            "success": success,
            "idempotency_key": _sha256(
                {
                    "logical_run_id": self.state["run_id"],
                    "tool_name": tool,
                    "canonical_arguments_sha256": arguments_sha256,
                    "semantic_attempt": semantic_attempt,
                }
            ),
            "submitted": False,
            "submitted_at": None,
        }
        self.state["receipts"][call_id] = receipt
        self.state["tool_attempts"].append(
            {
                "tool": tool,
                "attempt": attempt,
                "turn_id": turn_id,
                "call_id": call_id,
                "arguments_sha256": arguments_sha256,
                "result_sha256": receipt["result_sha256"],
                "status": "success" if success else "error",
                "error_code": None if success else "TEMP_UNAVAILABLE",
                "duration_ms": round(
                    (self.monotonic_ns() - started_ns) / 1_000_000, 3
                ),
            }
        )
        self._save_state()
        self._trace(
            "tool_attempt_finished",
            tool=tool,
            attempt=attempt,
            turn_id=turn_id,
            call_id=call_id,
            arguments_sha256=arguments_sha256,
            result_sha256=receipt["result_sha256"],
            status="success" if success else "error",
            error_code=None if success else "TEMP_UNAVAILABLE",
        )
        return receipt

    def _commit_report(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        report = write_report(arguments["normalized"], arguments["summary"])
        expected = self.experiment["task"]["expected_report"]
        if _canonical_bytes(report) != _canonical_bytes(expected):
            raise ManagedLiveProtocolError("write_report produced a noncanonical report")
        report_path = self.run_dir / "report.json"
        if report_path.exists():
            existing = _load_json(report_path)
            if _canonical_bytes(existing) != _canonical_bytes(report):
                raise ManagedLiveProtocolError("existing report conflicts with retry")
        else:
            _atomic_json_write(report_path, report)
        report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
        self.state["report_receipt"] = {
            "path": report_path.name,
            "sha256": report_sha,
            "commits": 1,
        }
        self._save_state()
        return {"ok": True, "report_sha256": report_sha, "path": report_path.name}

    def _verify(self) -> dict[str, Any]:
        expected_sequence = [
            ("normalize_records", "success"),
            ("summarize_records", "error"),
            ("summarize_records", "success"),
            ("write_report", "success"),
        ]
        actual_sequence = [
            (item["tool"], item["status"]) for item in self.state["tool_attempts"]
        ]
        report = _load_json(self.run_dir / "report.json")
        report_sha = hashlib.sha256((self.run_dir / "report.json").read_bytes()).hexdigest()
        validation = {
            "checkpoint_order": self.state["checkpoints"]
            == list(CHECKPOINTS[:-1]),
            "tool_attempt_order": actual_sequence == expected_sequence,
            "semantic_report_exact": _canonical_bytes(report)
            == _canonical_bytes(self.experiment["task"]["expected_report"]),
            "one_session": bool(self.state["session_id"]),
            "distinct_processes": len(self.state["processes"]) == 2
            and len({item["pid"] for item in self.state["processes"]}) == 2,
            "one_report_commit": self.state["report_receipt"]
            == {"path": "report.json", "sha256": report_sha, "commits": 1},
            "normalization_attempts": sum(
                item["tool"] == "normalize_records"
                for item in self.state["tool_attempts"]
            ),
            "duplicate_side_effects": self.state["duplicate_side_effects"],
            "no_secret_in_artifacts": self._secret_scan_passes(),
        }
        if validation["normalization_attempts"] != 1 or not all(
            value is True
            for key, value in validation.items()
            if key not in {"normalization_attempts", "duplicate_side_effects"}
        ):
            raise ManagedLiveProtocolError(f"final validation failed: {validation}")
        if validation["duplicate_side_effects"] != 0:
            raise ManagedLiveProtocolError(f"final validation failed: {validation}")
        self.state["validation"] = validation
        self.state["status"] = "completed"
        self._checkpoint("C5_VERIFIED")
        self._trace("run_verified", status="success")
        return report

    def _secret_scan_passes(self) -> bool:
        if not self.api_key:
            return True
        secret = self.api_key.encode("utf-8")
        for name in (
            "run-manifest.json",
            "state.json",
            "trace.jsonl",
            "report.json",
            "metrics.json",
        ):
            path = self.run_dir / name
            if path.exists() and secret in path.read_bytes():
                return False
        return True

    def _load_state(self, run_id: str) -> dict[str, Any]:
        state = _load_json(self.run_dir / "state.json")
        checks = {
            "schema_version": SCHEMA_VERSION,
            "benchmark_id": self.experiment["benchmark_id"],
            "config_sha256": self.config_sha256,
            "arm": ARM,
            "run_id": run_id,
        }
        for key, expected in checks.items():
            if state.get(key) != expected:
                raise ManagedLiveError(
                    f"saved {key} {state.get(key)!r} does not match {expected!r}"
                )
        serialized = _canonical_bytes(state)
        if self.api_key and self.api_key.encode("utf-8") in serialized:
            raise ManagedLiveError("saved state contains the active API key")
        return state

    def _checkpoint(self, checkpoint: str) -> None:
        expected_index = len(self.state["checkpoints"])
        if expected_index >= len(CHECKPOINTS) or CHECKPOINTS[expected_index] != checkpoint:
            raise ManagedLiveProtocolError(
                f"checkpoint {checkpoint} is out of order after {self.state['checkpoints']}"
            )
        self.state["checkpoint"] = checkpoint
        self.state["checkpoints"].append(checkpoint)
        self.state["checkpoint_bytes"][checkpoint] = 0
        while True:
            size = len(_canonical_bytes(_redact(self.state, self.api_key))) + 1
            if self.state["checkpoint_bytes"][checkpoint] == size:
                break
            self.state["checkpoint_bytes"][checkpoint] = size
        self._save_state()
        self._write_metrics()
        self._trace("checkpoint_saved", checkpoint=checkpoint, checkpoint_bytes=size)

    def _save_state(self) -> None:
        _atomic_json_write(self.run_dir / "state.json", _redact(self.state, self.api_key))

    def _write_metrics(self) -> None:
        attempts = self.state.get("tool_attempts", [])
        checkpoints = self.state.get("checkpoint_bytes", {})
        usage = self.state["provider_usage"]
        usage_complete = isinstance(usage, Mapping) and all(
            isinstance(usage.get(key), int)
            for key in (
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "total_tokens",
            )
        )
        metrics = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.state["run_id"],
            "arm": ARM,
            "execution_mode": "live",
            "comparison_eligible": self.state["comparison_eligible"],
            "status": self.state["status"],
            "checkpoint": self.state["checkpoint"],
            "timing": self.state["timing"],
            # The API exposes turns and aggregate usage, not the managed
            # harness's internal count of model invocations. A turn is not a
            # model call, so leave this unknown rather than writing one.
            "model_calls": None,
            "tool_attempts": len(attempts),
            "tool_successes": sum(item["status"] == "success" for item in attempts),
            "injected_failures": sum(
                item["error_code"] == "TEMP_UNAVAILABLE" for item in attempts
            ),
            "retry_count": max(
                0,
                sum(item["tool"] == "summarize_records" for item in attempts) - 1,
            ),
            "resume_count": self.state["resume_count"],
            "process_count": len(self.state["processes"]),
            "provider_requests": self.state["provider_requests"],
            "duplicate_tool_executions": self.state["duplicate_tool_executions"],
            "duplicate_side_effects": self.state["duplicate_side_effects"],
            "usage": usage,
            "provider_usage_complete": usage_complete,
            "input_tokens": usage.get("input_tokens") if usage_complete else None,
            "cached_input_tokens": usage.get("cached_input_tokens") if usage_complete else None,
            "output_tokens": usage.get("output_tokens") if usage_complete else None,
            "reasoning_tokens": usage.get("reasoning_tokens") if usage_complete else None,
            "total_tokens": usage.get("total_tokens") if usage_complete else None,
            **self.state["timing"],
            "cost_usd": {
                "uncached_input": None,
                "cached_input": None,
                "output": None,
                "agents_session": None,
                "total": None,
            },
            "checkpoint_bytes": checkpoints,
            "checkpoint_c2_bytes": checkpoints.get("C2_FAILURE_PERSISTED"),
            "checkpoint_peak_bytes": max(checkpoints.values(), default=None),
            "provider_trace_status": self.state["provider_trace_status"],
            "provider_trace_error": self.state["provider_trace_error"],
            "validation": self.state["validation"],
        }
        _atomic_json_write(self.run_dir / "metrics.json", metrics)

    def _write_manifest(self) -> None:
        try:
            sdk_version: str | None = importlib.metadata.version("openai")
        except importlib.metadata.PackageNotFoundError:
            sdk_version = None
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "benchmark_id": self.experiment["benchmark_id"],
            "config_sha256": self.config_sha256,
            "run_id": self.state["run_id"],
            "arm": ARM,
            "execution_mode": "live",
            "session_id": self.state["session_id"],
            "requested_model": self.experiment["model"]["requested_id"],
            "returned_model": self.state["returned_model"],
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "openai_sdk_version": sdk_version,
            "api_surface": "client.beta.agents.sessions",
            "agent_configuration": _redact(
                {
                    "environment": {"type": "none"},
                    "model": self.experiment["model"],
                    "instructions_sha256": _sha256(
                        self.experiment["task"]["developer_instruction"]
                    ),
                    "tools_sha256": _sha256(_api_tools(self.experiment)),
                },
                self.api_key,
            ),
            "pricing_snapshot": self.experiment.get("pricing_snapshot"),
            "processes": self.state["processes"],
            "created_at": self.state["timing"]["started_at"],
            "provider_retries": 0,
        }
        _atomic_json_write(self.run_dir / "run-manifest.json", manifest)

    def _trace(self, event: str, *, checkpoint: str | None = None, **fields: Any) -> None:
        self.state["trace_seq"] += 1
        payload = {
            "schema_version": SCHEMA_VERSION,
            "seq": self.state["trace_seq"],
            "run_id": self.state["run_id"],
            "arm": ARM,
            "process_instance_id": self.process_instance_id,
            "wall_time_utc": self.utc_now(),
            "monotonic_ns": self.monotonic_ns(),
            "event": event,
            "checkpoint": checkpoint or self.state.get("checkpoint"),
            "resumed": self.resumed,
            **fields,
        }
        payload = _redact(payload, self.api_key)
        # Save the allocated sequence before appending. A crash may leave a gap,
        # but can never cause a duplicate or decreasing sequence on resume.
        self._save_state()
        path = self.run_dir / "trace.jsonl"
        with path.open("ab") as handle:
            handle.write(_canonical_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())


def _build_live_adapter(api_key: str) -> AgentsAPIAdapter:
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ManagedLiveUnavailable(
            "live managed arm requires the OpenAI Python SDK with beta Agents sessions support"
        ) from exc
    client = OpenAI(api_key=api_key, max_retries=0)
    adapter = AgentsAPIAdapter(client=client, api_key=api_key)
    # Resolve the surface before creating any remote session.
    try:
        adapter.sessions
    except Exception as exc:
        raise ManagedLiveUnavailable(
            "installed OpenAI SDK does not expose beta.agents.sessions"
        ) from exc
    return adapter


def run_live(
    run_dir: str | Path,
    run_id: str,
    *,
    resume: bool = False,
    adapter: Any | None = None,
    client: Any | None = None,
    api_key: str | None = None,
    **driver_options: Any,
) -> dict[str, Any]:
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if adapter is not None and client is not None:
        raise ValueError("provide adapter or client, not both")
    if adapter is None and client is not None:
        adapter = AgentsAPIAdapter(client=client, api_key=key)
    if adapter is None:
        if not key:
            raise ManagedLiveUnavailable(
                "live managed arm unavailable: OPENAI_API_KEY is not set"
            )
        adapter = _build_live_adapter(key)
    driver = ManagedLiveDriver(adapter=adapter, api_key=key, **driver_options)
    return driver.run(run_dir, run_id, resume=resume)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="explicitly enable provider-backed execution",
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--run-id", help="start a new logical run")
    identity.add_argument("--resume", metavar="RUN_ID", help="resume the saved managed session")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.live:
        print(
            json.dumps(
                {"status": "unavailable", "error": "managed_live requires explicit --live"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    run_id = args.resume or args.run_id
    try:
        report = run_live(
            args.run_dir,
            run_id,
            resume=args.resume is not None,
        )
    except PlannedInterruption as exc:
        print(
            json.dumps(
                {
                    "status": "retry_pending",
                    "run_id": exc.run_id,
                    "checkpoint": "C2_FAILURE_PERSISTED",
                    "error_code": "TEMP_UNAVAILABLE",
                    "exit_code": exc.exit_code,
                },
                sort_keys=True,
            )
        )
        return exc.exit_code
    except ManagedLiveUnavailable as exc:
        print(
            json.dumps({"status": "unavailable", "error": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    except ManagedLiveError as exc:
        print(
            json.dumps({"status": "error", "error": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 1

    print(json.dumps({"status": "completed", "run_id": run_id, "report": report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
