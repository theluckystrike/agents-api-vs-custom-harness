#!/usr/bin/env python3
"""Live custom Responses API arm for the preregistered recovery benchmark.

The implementation is intentionally a small application-owned loop.  It uses
``store=False`` and replays the full response transcript; it never uses an
Agents runner, Conversations, or ``previous_response_id``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import custom_harness


EXIT_TEMPORARY_FAILURE = 75
SCHEMA_VERSION = 1
ARM = "custom_harness"
CHECKPOINTS = (
    "C0_INITIALIZED",
    "C1_NORMALIZED",
    "C2_FAILURE_PERSISTED",
    "C3_SUMMARIZED",
    "C4_REPORT_COMMITTED",
    "C5_VERIFIED",
)


class ResponsesLiveError(RuntimeError):
    """Base error for an invalid or unavailable live benchmark run."""


class ResponsesLiveUnavailable(ResponsesLiveError):
    """The live OpenAI client cannot be constructed in this environment."""


class ResponsesProtocolError(ResponsesLiveError):
    """The provider or model violated the frozen benchmark contract."""


class PlannedInterruption(ResponsesLiveError):
    """The preregistered process boundary after the injected failure."""

    exit_code = EXIT_TEMPORARY_FAILURE


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def as_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): as_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(item) for item in value]
    if dataclasses.is_dataclass(value):
        return as_json(dataclasses.asdict(value))
    for method_name in ("model_dump", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            return as_json(method())
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            str(key): as_json(item)
            for key, item in attributes.items()
            if not str(key).startswith("_")
        }
    return str(value)


def redact_text(text: str, secret: str | None) -> str:
    return text.replace(secret, "[REDACTED]") if secret else text


def load_experiment(path: Path | None = None) -> dict[str, Any]:
    source = path or Path(__file__).with_name("experiment.json")
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("protocol_status") != "preregistered":
        raise ResponsesProtocolError("experiment is not preregistered")
    return value


def provider_tools(experiment: Mapping[str, Any]) -> list[dict[str, Any]]:
    allowed = ("type", "name", "description", "parameters")
    return [
        {**{key: tool[key] for key in allowed}, "strict": True}
        for tool in experiment["tools"]
    ]


def initial_input(experiment: Mapping[str, Any]) -> list[dict[str, Any]]:
    task = experiment["task"]
    content = (
        f'{task["user_instruction"]}\n\n'
        f'Dataset ID: {task["dataset_id"]}\n'
        "Records (preserve exactly):\n"
        + json.dumps(task["raw_records"], ensure_ascii=False, separators=(",", ":"))
    )
    return [{"role": "user", "content": content}]


def _initial_metrics() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "arm": ARM,
        "execution_mode": "live",
        "comparison_eligible": True,
        "model_calls": 0,
        "provider_requests": 0,
        "tool_attempts": 0,
        "tool_successes": 0,
        "injected_failures": 0,
        "retry_count": 0,
        "resume_count": 0,
        "process_count": 1,
        "input_tokens": None,
        "cached_input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
        "provider_usage_complete": True,
        "model_call_latency_ms": [],
        "tool_attempt_latency_ms": [],
        "initial_wall_ms": None,
        "restart_gap_ms": None,
        "recovery_wall_ms": None,
        "active_wall_ms": None,
        "end_to_end_wall_ms": None,
        "total_cost_usd": None,
    }


def _initial_state(experiment: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "arm": ARM,
        "execution_mode": "live",
        "comparison_eligible": True,
        "run_id": run_id,
        "status": "running",
        "checkpoint": "C0_INITIALIZED",
        "checkpoints": ["C0_INITIALIZED"],
        "expected_next_tool": "normalize_records",
        "completed_tools": [],
        "attempts": {},
        "results": {},
        "fault_consumed": False,
        "receipts": {},
        "transcript": initial_input(experiment),
        "response_ids": [],
        "returned_models": [],
        "created_at": now,
        "initial_process_completed_at": None,
        "resume_process_started_at": None,
        "completed_at": None,
        "metrics": _initial_metrics(),
        "last_error": None,
    }


def load_state(path: Path, run_id: str) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResponsesLiveError(f"no checkpoint exists for run {run_id!r}") from exc
    if state.get("run_id") != run_id or state.get("arm") != ARM:
        raise ResponsesLiveError("checkpoint does not belong to this run and arm")
    return state


class TraceWriter:
    def __init__(self, path: Path, state: Mapping[str, Any], resumed: bool) -> None:
        self.path = path
        self.run_id = str(state["run_id"])
        self.resumed = resumed
        self.process_instance_id = f"pid-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        if path.exists():
            self.seq = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)
        else:
            self.seq = 0

    def emit(self, event: str, checkpoint: str, **fields: Any) -> None:
        self.seq += 1
        payload = {
            "schema_version": SCHEMA_VERSION,
            "seq": self.seq,
            "run_id": self.run_id,
            "arm": ARM,
            "process_instance_id": self.process_instance_id,
            "wall_time_utc": utc_now(),
            "monotonic_ns": time.monotonic_ns(),
            "event": event,
            "checkpoint": checkpoint,
            "resumed": self.resumed,
            **fields,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_bytes(payload).decode("utf-8") + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _usage(value: Any) -> dict[str, Any] | None:
    data = as_json(value)
    return data if isinstance(data, dict) else None


def _add_usage(metrics: dict[str, Any], usage: Mapping[str, Any] | None) -> None:
    if not usage:
        metrics["provider_usage_complete"] = False
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
        ):
            metrics[key] = None
        return
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    values = {
        "input_tokens": usage.get("input_tokens"),
        "cached_input_tokens": input_details.get("cached_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": output_details.get("reasoning_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }
    for key, value in values.items():
        if not isinstance(value, int):
            metrics["provider_usage_complete"] = False
            for token_key in values:
                metrics[token_key] = None
            return
    if not metrics["provider_usage_complete"]:
        return
    for key, value in values.items():
        metrics[key] = int(metrics[key] or 0) + value


def _tool_calls(output: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in output if item.get("type") == "function_call"]


def _parse_arguments(call: Mapping[str, Any]) -> dict[str, Any]:
    value = call.get("arguments")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ResponsesProtocolError("function arguments are not valid JSON") from exc
    if not isinstance(value, dict):
        raise ResponsesProtocolError("function arguments must be a JSON object")
    return value


def _validate_arguments(
    name: str,
    arguments: Mapping[str, Any],
    state: Mapping[str, Any],
    experiment: Mapping[str, Any],
) -> None:
    task = experiment["task"]
    if name == "normalize_records":
        expected = {"dataset_id": task["dataset_id"], "records": task["raw_records"]}
    elif name == "summarize_records":
        expected = {"normalized": state["results"].get("normalize_records")}
    elif name == "write_report":
        expected = {
            "normalized": state["results"].get("normalize_records"),
            "summary": state["results"].get("summarize_records"),
        }
    else:
        raise ResponsesProtocolError(f"unknown tool requested: {name}")
    if arguments != expected:
        raise ResponsesProtocolError(
            f"arguments for {name} differ from the frozen expected value"
        )


def _report_receipt(run_dir: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    report_path = run_dir / "report.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing != report:
            raise ResponsesProtocolError("existing report receipt does not match arguments")
        digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
        return {"ok": True, "sha256": digest, "duplicate": True}
    atomic_json(report_path, report)
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    return {"ok": True, "sha256": digest, "duplicate": False}


def _execute_tool(
    name: str,
    arguments: Mapping[str, Any],
    state: dict[str, Any],
    experiment: Mapping[str, Any],
    run_dir: Path,
) -> tuple[dict[str, Any], str]:
    _validate_arguments(name, arguments, state, experiment)
    if name == "normalize_records":
        result = custom_harness.normalize_records(arguments["records"])
        checkpoint = "C1_NORMALIZED"
    elif name == "summarize_records":
        result = custom_harness.summarize_records(arguments["normalized"])
        checkpoint = "C3_SUMMARIZED"
    elif name == "write_report":
        report = custom_harness.write_report(arguments["normalized"], arguments["summary"])
        if report != experiment["task"]["expected_report"]:
            raise ResponsesProtocolError("tool report differs from frozen expected report")
        result = _report_receipt(run_dir, report)
        state["receipts"]["write_report"] = result
        checkpoint = "C4_REPORT_COMMITTED"
    else:  # pragma: no cover - guarded above
        raise ResponsesProtocolError(f"unknown tool: {name}")
    state["results"][name] = (
        experiment["task"]["expected_report"] if name == "write_report" else result
    )
    return result, checkpoint


def build_live_client(api_key: str | None = None) -> Any:
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ResponsesLiveUnavailable(
            "live Responses benchmark requires OPENAI_API_KEY"
        )
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ResponsesLiveUnavailable(
            "live Responses benchmark requires a compatible OpenAI Python SDK"
        ) from exc
    return OpenAI(api_key=key, max_retries=0)


def _response_create(client: Any, experiment: Mapping[str, Any], transcript: list[Any]) -> Any:
    model = experiment["model"]
    return client.responses.create(
        model=model["requested_id"],
        instructions=experiment["task"]["developer_instruction"],
        # Give the SDK an immutable snapshot. The controller appends to its
        # transcript after the request returns and must not mutate a captured
        # request object in tests, logs, or asynchronous transports.
        input=json.loads(json.dumps(transcript, ensure_ascii=False)),
        tools=provider_tools(experiment),
        store=False,
        parallel_tool_calls=False,
        reasoning={"effort": model["reasoning_effort"]},
        text={"verbosity": model["text_verbosity"]},
        service_tier=model["service_tier"],
        include=["reasoning.encrypted_content"],
    )


def _persist(run_dir: Path, state: Mapping[str, Any]) -> None:
    atomic_json(run_dir / "state.json", state)
    atomic_json(run_dir / "metrics.json", state["metrics"])


def _manifest(run_dir: Path, state: Mapping[str, Any], experiment: Mapping[str, Any]) -> None:
    value = {
        "schema_version": SCHEMA_VERSION,
        "arm": ARM,
        "logical_run_id": state["run_id"],
        "execution_mode": "live",
        "comparison_eligible": True,
        "benchmark_id": experiment["benchmark_id"],
        "configuration_sha256": sha256_json(experiment),
        "requested_model": experiment["model"]["requested_id"],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "created_at": state["created_at"],
    }
    atomic_json(run_dir / "run-manifest.json", value)


def run(
    run_dir: str | Path,
    run_id: str,
    *,
    resume: bool = False,
    client: Any | None = None,
    experiment_path: Path | None = None,
) -> dict[str, Any]:
    """Run one process of the live custom arm.

    The initial process deliberately raises :class:`PlannedInterruption` after
    durably recording C2.  The caller must start a replacement process with
    ``resume=True`` and the same run ID.
    """

    destination = Path(run_dir)
    state_path = destination / "state.json"
    experiment = load_experiment(experiment_path)
    # Credential/SDK preflight happens before a new run directory is created.
    live_client = client or build_live_client()
    process_started_ns = time.monotonic_ns()
    if resume:
        state = load_state(state_path, run_id)
        if state["status"] == "completed":
            return dict(state["results"]["write_report"])
        if state["status"] != "retry_pending" or state["checkpoint"] != "C2_FAILURE_PERSISTED":
            raise ResponsesLiveError("checkpoint is not at the registered resume boundary")
        state["metrics"]["resume_count"] += 1
        state["metrics"]["process_count"] += 1
        state["resume_process_started_at"] = utc_now()
        if state.get("initial_process_completed_at"):
            end = datetime.fromisoformat(
                state["initial_process_completed_at"].replace("Z", "+00:00")
            )
            start = datetime.fromisoformat(
                state["resume_process_started_at"].replace("Z", "+00:00")
            )
            state["metrics"]["restart_gap_ms"] = round(
                (start - end).total_seconds() * 1000, 3
            )
    else:
        if state_path.exists():
            raise ResponsesLiveError("checkpoint already exists; use --resume")
        destination.mkdir(parents=True, exist_ok=True)
        state = _initial_state(experiment, run_id)

    trace = TraceWriter(destination / "trace.jsonl", state, resume)
    trace.emit("run_resumed" if resume else "run_started", state["checkpoint"])
    _persist(destination, state)
    _manifest(destination, state, experiment)
    requested_model = experiment["model"]["requested_id"]
    max_calls = int(experiment["execution"]["max_model_calls_per_run"])
    max_tools = int(experiment["execution"]["max_tool_attempts_per_run"])

    while True:
        if state["metrics"]["model_calls"] >= max_calls:
            raise ResponsesProtocolError("model call limit exceeded")
        call_started = time.monotonic_ns()
        trace.emit("model_call_started", state["checkpoint"])
        try:
            response = _response_create(live_client, experiment, state["transcript"])
        except Exception as exc:
            message = redact_text(str(exc), os.environ.get("OPENAI_API_KEY"))
            state["last_error"] = {
                "code": type(exc).__name__,
                "message": message,
                "retryable": False,
            }
            _persist(destination, state)
            trace.emit(
                "provider_request_failed",
                state["checkpoint"],
                status="error",
                error_code=type(exc).__name__,
            )
            raise ResponsesLiveError(f"provider request failed: {message}") from exc
        latency_ms = round((time.monotonic_ns() - call_started) / 1_000_000, 3)
        payload = as_json(response)
        if not isinstance(payload, dict):
            raise ResponsesProtocolError("Responses API returned a non-object")
        returned_model = payload.get("model")
        if returned_model != requested_model:
            raise ResponsesProtocolError(
                f"returned model {returned_model!r} differs from frozen {requested_model!r}"
            )
        output = payload.get("output")
        if not isinstance(output, list):
            raise ResponsesProtocolError("response.output is not an array")
        state["metrics"]["model_calls"] += 1
        state["metrics"]["provider_requests"] += 1
        state["metrics"]["model_call_latency_ms"].append(latency_ms)
        _add_usage(state["metrics"], _usage(payload.get("usage")))
        state["response_ids"].append(payload.get("id"))
        state["returned_models"].append(returned_model)
        state["transcript"].extend(output)
        trace.emit(
            "model_call_completed",
            state["checkpoint"],
            response_id=payload.get("id"),
            duration_ms=latency_ms,
        )

        calls = _tool_calls(output)
        if not calls:
            if state["checkpoint"] != "C4_REPORT_COMMITTED":
                raise ResponsesProtocolError("model ended before committing the report")
            report = json.loads((destination / "report.json").read_text(encoding="utf-8"))
            if report != experiment["task"]["expected_report"]:
                raise ResponsesProtocolError("final report failed semantic verification")
            state["checkpoint"] = "C5_VERIFIED"
            state["checkpoints"].append("C5_VERIFIED")
            state["status"] = "completed"
            state["completed_at"] = utc_now()
            elapsed_ms = round((time.monotonic_ns() - process_started_ns) / 1_000_000, 3)
            state["metrics"]["recovery_wall_ms" if resume else "initial_wall_ms"] = elapsed_ms
            initial_ms = state["metrics"].get("initial_wall_ms") or 0
            recovery_ms = state["metrics"].get("recovery_wall_ms") or 0
            state["metrics"]["active_wall_ms"] = round(initial_ms + recovery_ms, 3)
            created = datetime.fromisoformat(state["created_at"].replace("Z", "+00:00"))
            completed = datetime.fromisoformat(
                state["completed_at"].replace("Z", "+00:00")
            )
            state["metrics"]["end_to_end_wall_ms"] = round(
                (completed - created).total_seconds() * 1000, 3
            )
            _persist(destination, state)
            trace.emit("run_completed", state["checkpoint"])
            return report
        if len(calls) != 1:
            raise ResponsesProtocolError("parallel or multiple function calls are not allowed")

        call = calls[0]
        name = str(call.get("name") or "")
        expected = state["expected_next_tool"]
        if name != expected:
            raise ResponsesProtocolError(
                f"expected tool {expected!r}, provider requested {name!r}"
            )
        if state["metrics"]["tool_attempts"] >= max_tools:
            raise ResponsesProtocolError("tool attempt limit exceeded")
        arguments = _parse_arguments(call)
        arguments_hash = sha256_json(arguments)
        call_id = str(call.get("call_id") or "")
        if not call_id:
            raise ResponsesProtocolError("function call has no call_id")
        attempt = int(state["attempts"].get(name, 0)) + 1
        state["attempts"][name] = attempt
        state["metrics"]["tool_attempts"] += 1
        tool_started = time.monotonic_ns()
        trace.emit(
            "tool_call_started",
            state["checkpoint"],
            tool=name,
            attempt=attempt,
            call_id=call_id,
            arguments_sha256=arguments_hash,
        )

        if name == "summarize_records" and not state["fault_consumed"]:
            _validate_arguments(name, arguments, state, experiment)
            error = experiment["fault"]["result"]
            state["fault_consumed"] = True
            state["metrics"]["injected_failures"] += 1
            state["metrics"]["retry_count"] += 1
            state["transcript"].append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(error, sort_keys=True),
                }
            )
            state["checkpoint"] = "C2_FAILURE_PERSISTED"
            state["checkpoints"].append("C2_FAILURE_PERSISTED")
            state["status"] = "retry_pending"
            state["last_error"] = error["error"]
            state["initial_process_completed_at"] = utc_now()
            elapsed_ms = round((time.monotonic_ns() - process_started_ns) / 1_000_000, 3)
            state["metrics"]["initial_wall_ms"] = elapsed_ms
            state["metrics"]["tool_attempt_latency_ms"].append(
                round((time.monotonic_ns() - tool_started) / 1_000_000, 3)
            )
            _persist(destination, state)
            trace.emit(
                "tool_call_failed",
                state["checkpoint"],
                tool=name,
                attempt=attempt,
                call_id=call_id,
                arguments_sha256=arguments_hash,
                status="retryable_error",
                error_code="TEMP_UNAVAILABLE",
            )
            trace.emit("run_paused", state["checkpoint"], exit_code=75)
            raise PlannedInterruption("pre-registered transient failure persisted")

        result, checkpoint = _execute_tool(
            name, arguments, state, experiment, destination
        )
        duration_ms = round((time.monotonic_ns() - tool_started) / 1_000_000, 3)
        state["metrics"]["tool_attempt_latency_ms"].append(duration_ms)
        state["metrics"]["tool_successes"] += 1
        state["completed_tools"].append(name)
        if name == "normalize_records":
            state["expected_next_tool"] = "summarize_records"
        elif name == "summarize_records":
            state["expected_next_tool"] = "write_report"
            state["last_error"] = None
        else:
            state["expected_next_tool"] = None
        state["checkpoint"] = checkpoint
        state["checkpoints"].append(checkpoint)
        output_value = result
        state["transcript"].append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(output_value, sort_keys=True),
            }
        )
        _persist(destination, state)
        trace.emit(
            "tool_call_succeeded",
            checkpoint,
            tool=name,
            attempt=attempt,
            call_id=call_id,
            arguments_sha256=arguments_hash,
            result_sha256=sha256_json(output_value),
            duration_ms=duration_ms,
            status="success",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--run-id")
    identity.add_argument("--resume", metavar="RUN_ID")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = args.resume or args.run_id
    secret = os.environ.get("OPENAI_API_KEY")
    try:
        report = run(args.run_dir, run_id, resume=args.resume is not None)
    except PlannedInterruption as exc:
        print(json.dumps({"status": "retry_pending", "run_id": run_id, "exit_code": 75}))
        return exc.exit_code
    except ResponsesLiveError as exc:
        print(f"responses_live: {redact_text(str(exc), secret)}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "completed", "run_id": run_id, "report": report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
