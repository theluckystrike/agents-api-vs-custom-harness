#!/usr/bin/env python3
"""Minimal durable three-tool harness used by the comparison benchmark.

The first simulated invocation deliberately pauses after a retryable failure in
``summarize_records``.  A second invocation with ``--resume`` continues from
the JSON checkpoint without repeating completed tool calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = 1
WORKFLOW_VERSION = "durable-three-tool-pipeline-v1"
EXIT_TEMPORARY_FAILURE = 75

RAW_RECORDS: tuple[dict[str, str], ...] = (
    {"id": " job-003 ", "latency_ms": "80", "status": "ok"},
    {"id": "job-001", "latency_ms": "120", "status": "ok"},
    {"id": "job-002", "latency_ms": "200", "status": "error"},
    {"id": "job-001", "latency_ms": "120", "status": "ok"},
    {"id": "job-004", "latency_ms": "160", "status": "ok"},
    {"id": "job-005", "latency_ms": "not-a-number", "status": "ok"},
)


class HarnessError(RuntimeError):
    """Base error for an invalid or failed harness run."""


class TransientToolError(HarnessError):
    """A retryable tool error carrying a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PlannedInterruption(HarnessError):
    """Controller failpoint that makes recovery observable across processes."""

    exit_code = EXIT_TEMPORARY_FAILURE

    def __init__(self, run_id: str, step: str, code: str) -> None:
        super().__init__(f"run {run_id!r} paused after {step}: {code}")
        self.run_id = run_id
        self.step = step
        self.code = code


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def normalize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Normalize, validate, de-duplicate, and sort the fixed input records."""

    normalized: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    seen: set[str] = set()

    for index, raw in enumerate(records):
        record_id = str(raw.get("id", "")).strip()
        status = str(raw.get("status", "")).strip().lower()
        if not record_id:
            dropped.append({"index": index, "reason": "missing_id"})
            continue
        if record_id in seen:
            dropped.append({"id": record_id, "index": index, "reason": "duplicate_id"})
            continue
        try:
            latency_ms = int(str(raw.get("latency_ms", "")).strip())
        except (TypeError, ValueError):
            dropped.append({"id": record_id, "index": index, "reason": "invalid_latency"})
            continue
        if latency_ms < 0:
            dropped.append({"id": record_id, "index": index, "reason": "invalid_latency"})
            continue
        if status not in {"ok", "error"}:
            dropped.append({"id": record_id, "index": index, "reason": "invalid_status"})
            continue
        seen.add(record_id)
        normalized.append(
            {"id": record_id, "latency_ms": latency_ms, "status": status}
        )

    normalized.sort(key=lambda row: row["id"])
    return {
        "source_count": len(records),
        "valid_count": len(normalized),
        "records": normalized,
        "dropped": dropped,
    }


def summarize_records(normalized: Mapping[str, Any]) -> dict[str, Any]:
    """Return deterministic aggregate statistics using nearest-rank p95."""

    records = list(normalized.get("records", []))
    if not records:
        raise ValueError("at least one normalized record is required")
    latencies = sorted(int(row["latency_ms"]) for row in records)
    error_count = sum(row["status"] == "error" for row in records)
    count = len(records)
    p95_index = max(0, math.ceil(0.95 * count) - 1)
    return {
        "count": count,
        "error_count": error_count,
        "error_rate": error_count / count,
        "mean_latency_ms": sum(latencies) / count,
        "p95_latency_ms": latencies[p95_index],
    }


def write_report(
    normalized: Mapping[str, Any], summary: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the canonical report value (disk persistence is the controller's job)."""

    records = list(normalized["records"])
    digest = hashlib.sha256(_canonical_bytes(records)).hexdigest()
    return {
        "task": WORKFLOW_VERSION,
        "input": {
            "source_count": normalized["source_count"],
            "valid_count": normalized["valid_count"],
            "dropped": list(normalized["dropped"]),
        },
        "summary": dict(summary),
        "normalized_records_sha256": digest,
    }


TOOLS: tuple[
    tuple[str, Callable[[dict[str, Any]], dict[str, Any]]], ...
] = (
    ("normalize_records", lambda _results: normalize_records(RAW_RECORDS)),
    (
        "summarize_records",
        lambda results: summarize_records(results["normalize_records"]),
    ),
    (
        "write_report",
        lambda results: write_report(
            results["normalize_records"], results["summarize_records"]
        ),
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_event(path: Path, event: str, run_id: str, **fields: Any) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": _utc_now(),
        "event": event,
        "run_id": run_id,
        **fields,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_bytes(payload).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _initial_state(run_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow_version": WORKFLOW_VERSION,
        "run_id": run_id,
        "status": "running",
        "next_step": 0,
        "completed_steps": [],
        "results": {},
        "attempts": {},
        "transient_failure_injected": False,
        "last_error": None,
        "metrics": {
            "wall_time_ms": 0.0,
            "tool_calls": 0,
            "successful_tool_calls": 0,
            "retries": 0,
        },
    }


def _load_state(path: Path, run_id: str) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HarnessError(f"no checkpoint exists for run {run_id!r}") from exc
    except json.JSONDecodeError as exc:
        raise HarnessError(f"checkpoint is not valid JSON: {path}") from exc
    if state.get("schema_version") != SCHEMA_VERSION:
        raise HarnessError("unsupported checkpoint schema")
    if state.get("workflow_version") != WORKFLOW_VERSION:
        raise HarnessError("checkpoint belongs to another workflow version")
    if state.get("run_id") != run_id:
        raise HarnessError(
            f"checkpoint run_id {state.get('run_id')!r} does not match {run_id!r}"
        )
    return state


def run(
    run_dir: str | Path,
    run_id: str,
    *,
    resume: bool = False,
    pause_on_transient: bool = True,
) -> dict[str, Any]:
    """Execute or resume the durable workflow.

    The normal benchmark path uses ``pause_on_transient=True``.  It creates a
    durable failed-attempt checkpoint and raises :class:`PlannedInterruption`,
    allowing the next process to prove that completed work is not repeated.
    """

    destination = Path(run_dir)
    state_path = destination / "state.json"
    trace_path = destination / "trace.jsonl"
    metrics_path = destination / "metrics.json"
    report_path = destination / "report.json"
    invocation_started = time.perf_counter_ns()

    if resume:
        state = _load_state(state_path, run_id)
        _append_event(
            trace_path,
            "run_resumed",
            run_id,
            next_step=state["next_step"],
            status=state["status"],
        )
        for completed in state["completed_steps"]:
            _append_event(
                trace_path,
                "step_skipped",
                run_id,
                step=completed["index"],
                tool=completed["tool"],
                reason="checkpoint_completed",
            )
    else:
        if state_path.exists():
            raise HarnessError(
                f"checkpoint already exists at {state_path}; use --resume {run_id}"
            )
        destination.mkdir(parents=True, exist_ok=True)
        state = _initial_state(run_id)
        _append_event(trace_path, "run_started", run_id, next_step=0)

    base_wall_time_ms = float(state["metrics"]["wall_time_ms"])

    def checkpoint(reason: str, step: int | None = None) -> None:
        elapsed_ms = (time.perf_counter_ns() - invocation_started) / 1_000_000
        state["metrics"]["wall_time_ms"] = round(base_wall_time_ms + elapsed_ms, 3)
        _atomic_json_write(state_path, state)
        _atomic_json_write(metrics_path, state["metrics"])
        _append_event(
            trace_path,
            "checkpoint_saved",
            run_id,
            reason=reason,
            step=step,
            next_step=state["next_step"],
            status=state["status"],
        )

    if state["status"] == "completed":
        _append_event(trace_path, "run_already_completed", run_id)
        return dict(state["results"]["write_report"])

    state["status"] = "running"
    while state["next_step"] < len(TOOLS):
        step_index = int(state["next_step"])
        tool_name, tool = TOOLS[step_index]
        attempt = int(state["attempts"].get(tool_name, 0)) + 1
        state["attempts"][tool_name] = attempt
        state["metrics"]["tool_calls"] += 1
        tool_started = time.perf_counter_ns()
        _append_event(
            trace_path,
            "tool_call_started",
            run_id,
            step=step_index,
            tool=tool_name,
            attempt=attempt,
        )

        try:
            if tool_name == "summarize_records" and not state[
                "transient_failure_injected"
            ]:
                state["transient_failure_injected"] = True
                raise TransientToolError(
                    "TEMP_UNAVAILABLE", "deliberate one-time transient failure"
                )
            result = tool(state["results"])
        except TransientToolError as exc:
            duration_ms = (time.perf_counter_ns() - tool_started) / 1_000_000
            state["metrics"]["retries"] += 1
            state["status"] = "retry_pending"
            state["last_error"] = {
                "code": exc.code,
                "message": str(exc),
                "retryable": True,
                "step": step_index,
                "tool": tool_name,
            }
            _append_event(
                trace_path,
                "tool_call_failed",
                run_id,
                step=step_index,
                tool=tool_name,
                attempt=attempt,
                duration_ms=round(duration_ms, 3),
                error_code=exc.code,
                retryable=True,
            )
            checkpoint("transient_failure", step_index)
            _append_event(
                trace_path,
                "retry_scheduled",
                run_id,
                step=step_index,
                tool=tool_name,
                next_attempt=attempt + 1,
            )
            if pause_on_transient:
                _append_event(
                    trace_path,
                    "run_paused",
                    run_id,
                    step=step_index,
                    tool=tool_name,
                    exit_code=EXIT_TEMPORARY_FAILURE,
                )
                raise PlannedInterruption(run_id, tool_name, exc.code) from exc
            continue

        duration_ms = (time.perf_counter_ns() - tool_started) / 1_000_000
        state["results"][tool_name] = result
        state["metrics"]["successful_tool_calls"] += 1
        state["completed_steps"].append(
            {"index": step_index, "tool": tool_name}
        )
        state["next_step"] = step_index + 1
        state["last_error"] = None
        _append_event(
            trace_path,
            "tool_call_succeeded",
            run_id,
            step=step_index,
            tool=tool_name,
            attempt=attempt,
            duration_ms=round(duration_ms, 3),
        )
        checkpoint("step_completed", step_index)

    state["status"] = "completed"
    report = dict(state["results"]["write_report"])
    _atomic_json_write(report_path, report)
    checkpoint("run_completed", len(TOOLS) - 1)
    _append_event(
        trace_path,
        "run_completed",
        run_id,
        report_path=report_path.name,
        metrics=dict(state["metrics"]),
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="run the deterministic no-key benchmark scenario",
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--run-id", help="start a new logical run")
    identity.add_argument("--resume", metavar="RUN_ID", help="resume a checkpointed run")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.simulate:
        print(
            json.dumps({"error": "only --simulate mode is supported"}, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    run_id = args.resume or args.run_id
    try:
        report = run(
            args.run_dir,
            run_id,
            resume=args.resume is not None,
            pause_on_transient=True,
        )
    except PlannedInterruption as exc:
        print(
            json.dumps(
                {
                    "status": "retry_pending",
                    "run_id": exc.run_id,
                    "tool": exc.step,
                    "error_code": exc.code,
                    "exit_code": exc.exit_code,
                },
                sort_keys=True,
            )
        )
        return exc.exit_code
    except HarnessError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, sort_keys=True))
        return 1

    state = _load_state(args.run_dir / "state.json", run_id)
    print(
        json.dumps(
            {
                "status": state["status"],
                "run_id": run_id,
                "report": report,
                "metrics": state["metrics"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
