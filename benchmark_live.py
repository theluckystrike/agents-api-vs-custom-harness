#!/usr/bin/env python3
"""Plan, preflight, execute, validate, and analyze the paired live benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
PROGRAMS = {
    "agents_api": ROOT / "managed_live.py",
    "custom_harness": ROOT / "responses_live.py",
}
EXPECTED_INITIAL_EXIT = 75
EXPECTED_RESUME_EXIT = 0
SECRET_PATTERNS = (
    re.compile(r"Authorization\s*:\s*Bearer", re.I),
    re.compile(r"api[_-]?key\s*[=:]\s*(?!\[REDACTED\]|true|false|null)", re.I),
)


class BenchmarkError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


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


def experiment() -> dict[str, Any]:
    return json.loads((ROOT / "experiment.json").read_text(encoding="utf-8"))


def ordered_arms(pair_ordinal: int, *, warmup: bool = False) -> list[str]:
    if warmup:
        return ["agents_api", "custom_harness"]
    key = "odd" if pair_ordinal % 2 else "even"
    return list(experiment()["execution"]["pair_order"][key])


def build_plan() -> dict[str, Any]:
    exp = experiment()
    measured = int(exp["execution"]["measured_pairs"])
    return {
        "benchmark_id": exp["benchmark_id"],
        "configuration_sha256": sha256_json(exp),
        "requested_model": exp["model"]["requested_id"],
        "warmup": {"pair_ordinal": 0, "excluded": True, "order": ordered_arms(0, warmup=True)},
        "measured_pairs": [
            {"pair_ordinal": number, "order": ordered_arms(number)}
            for number in range(1, measured + 1)
        ],
        "replacement_policy": exp["exclusions"]["replacement_policy"],
        "run_arms_concurrently": False,
        "comparison_eligible": False,
        "status": "plan_only",
    }


def preflight() -> dict[str, Any]:
    exp = experiment()
    key_present = bool(os.environ.get("OPENAI_API_KEY"))
    sdk_agents_sessions = False
    sdk_responses = False
    sdk_surface_error = None
    try:
        sdk_version = importlib.metadata.version("openai")
    except importlib.metadata.PackageNotFoundError:
        sdk_version = None
    if sdk_version is not None:
        try:
            from openai import OpenAI  # type: ignore[import-not-found]

            probe = OpenAI(api_key="preflight-placeholder-not-a-credential", max_retries=0)
            sdk_agents_sessions = hasattr(
                getattr(getattr(probe, "beta", None), "agents", None), "sessions"
            )
            sdk_responses = hasattr(getattr(probe, "responses", None), "create")
            close = getattr(probe, "close", None)
            if callable(close):
                close()
        except Exception as exc:  # pragma: no cover - SDK-specific construction failure
            sdk_surface_error = f"{type(exc).__name__}: {exc}"
    prices = exp.get("pricing_snapshot") or {}
    rates = prices.get("per_million_tokens") or {}
    pricing_ready = (
        prices.get("status") == "frozen_before_first_live_pair"
        and bool(prices.get("official_source_url"))
        and all(isinstance(rates.get(key), (int, float)) for key in (
            "uncached_input", "cached_input", "output"
        ))
        and isinstance(
            (prices.get("agents_api_non_token_charges") or {}).get("value_usd_per_run"),
            (int, float),
        )
    )
    programs = {name: path.is_file() for name, path in PROGRAMS.items()}
    checks = {
        "openai_api_key_present": key_present,
        "openai_sdk_version": sdk_version,
        "openai_sdk_agents_sessions_surface": sdk_agents_sessions,
        "openai_sdk_responses_surface": sdk_responses,
        "openai_sdk_surface_error": sdk_surface_error,
        "python_version_supported": sys.version_info >= (3, 10),
        "pricing_snapshot_ready": pricing_ready,
        "programs_present": programs,
        "model_frozen": exp["model"].get("substitution_allowed") is False,
        "protocol_preregistered": exp.get("protocol_status") == "preregistered",
    }
    ready = (
        key_present
        and sdk_version is not None
        and sdk_agents_sessions
        and sdk_responses
        and checks["python_version_supported"]
        and pricing_ready
        and all(programs.values())
        and checks["model_frozen"]
        and checks["protocol_preregistered"]
    )
    return {"ready": ready, "checked_at": utc_now(), "checks": checks}


def _invoke(command: list[str], run_dir: Path, phase: str) -> dict[str, Any]:
    started_at = utc_now()
    started_monotonic = time.monotonic_ns()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate()
    elapsed_ms = round((time.monotonic_ns() - started_monotonic) / 1_000_000, 3)
    (run_dir / f"stdout-{phase}.log").write_text(stdout, encoding="utf-8")
    (run_dir / f"stderr-{phase}.log").write_text(stderr, encoding="utf-8")
    return {
        "phase": phase,
        "pid": process.pid,
        "started_at": started_at,
        "completed_at": utc_now(),
        "elapsed_ms": elapsed_ms,
        "exit_code": process.returncode,
    }


def run_arm_processes(arm: str, run_dir: Path, run_id: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    program = PROGRAMS[arm]
    initial = _invoke(
        [sys.executable, str(program), "--live", "--run-dir", str(run_dir), "--run-id", run_id],
        run_dir,
        "initial",
    )
    resume = _invoke(
        [sys.executable, str(program), "--live", "--run-dir", str(run_dir), "--resume", run_id],
        run_dir,
        "resume",
    )
    process_evidence = {"initial": initial, "resume": resume}
    atomic_json(run_dir / "processes.json", process_evidence)
    return process_evidence


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def add_costs(metrics: Mapping[str, Any], exp: Mapping[str, Any], arm: str) -> dict[str, Any]:
    result = dict(metrics)
    fields = ("input_tokens", "cached_input_tokens", "output_tokens")
    if not all(isinstance(metrics.get(field), int) for field in fields):
        result.update(
            uncached_input_cost_usd=None,
            cached_input_cost_usd=None,
            output_cost_usd=None,
            agents_api_non_token_cost_usd=None,
            total_cost_usd=None,
        )
        return result
    input_tokens = int(metrics["input_tokens"])
    cached_tokens = int(metrics["cached_input_tokens"])
    output_tokens = int(metrics["output_tokens"])
    if input_tokens > 272_000:
        # The frozen snapshot intentionally covers only the standard band.
        result.update(
            uncached_input_cost_usd=None,
            cached_input_cost_usd=None,
            output_cost_usd=None,
            agents_api_non_token_cost_usd=None,
            total_cost_usd=None,
            cost_error="input exceeded the frozen standard-context price band",
        )
        return result
    rates = exp["pricing_snapshot"]["per_million_tokens"]
    uncached_cost = max(0, input_tokens - cached_tokens) * rates["uncached_input"] / 1_000_000
    cached_cost = cached_tokens * rates["cached_input"] / 1_000_000
    output_cost = output_tokens * rates["output"] / 1_000_000
    non_token = (
        exp["pricing_snapshot"]["agents_api_non_token_charges"]["value_usd_per_run"]
        if arm == "agents_api"
        else 0.0
    )
    result.update(
        uncached_input_cost_usd=uncached_cost,
        cached_input_cost_usd=cached_cost,
        output_cost_usd=output_cost,
        agents_api_non_token_cost_usd=non_token,
        total_cost_usd=uncached_cost + cached_cost + output_cost + non_token,
    )
    return result


def _secret_scan(run_dir: Path) -> tuple[bool, list[str]]:
    findings = []
    known_secret = os.environ.get("OPENAI_API_KEY")
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if known_secret and known_secret in text:
            findings.append(f"{path.name}: exact API key")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(f"{path.name}: {pattern.pattern}")
    return not findings, sorted(set(findings))


def _exclusion_reason(run_dir: Path, state: Mapping[str, Any] | None) -> str | None:
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in run_dir.glob("stderr-*.log")
    ).lower()
    state_text = json.dumps(state or {}).lower()
    combined = text + "\n" + state_text
    patterns = (
        ("provider_rate_limit", ("rate_limit", "rate limit", "429")),
        ("provider_5xx", ("server_error", "status 500", "status 502", "status 503", "status 504")),
        ("network_loss", ("connectionerror", "connection error", "network", "timed out")),
        ("machine_sleep", ("machine_sleep", "machine sleep")),
        ("model_mismatch", ("differs from frozen", "model mismatch")),
    )
    for reason, needles in patterns:
        if any(needle in combined for needle in needles):
            return reason
    return None


def validate_trial(
    arm: str,
    run_dir: Path,
    run_id: str,
    process_evidence: Mapping[str, Any],
    exp: Mapping[str, Any],
) -> dict[str, Any]:
    state = _read_json(run_dir / "state.json")
    metrics = _read_json(run_dir / "metrics.json")
    report = _read_json(run_dir / "report.json")
    manifest = _read_json(run_dir / "run-manifest.json")
    trace_events = []
    trace_path = run_dir / "trace.jsonl"
    if trace_path.exists():
        try:
            trace_events = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except json.JSONDecodeError:
            trace_events = []
    secret_ok, secret_findings = _secret_scan(run_dir)
    starts = [
        row
        for row in trace_events
        if row.get("event") in {"tool_call_started", "tool_attempt_started"}
    ]
    tool_sequence = [row.get("tool") for row in starts]
    trace_seq = [row.get("seq") for row in trace_events]
    receipt = None
    if isinstance(state, dict):
        receipt = state.get("report_receipt")
        if receipt is None:
            receipt = (state.get("receipts") or {}).get("write_report")
    report_sha = (
        hashlib.sha256((run_dir / "report.json").read_bytes()).hexdigest()
        if (run_dir / "report.json").exists()
        else None
    )
    attempts = (state or {}).get("attempts") if isinstance(state, dict) else None
    if attempts is None and isinstance(state, dict):
        attempts = {}
        for item in state.get("tool_attempts") or []:
            tool = item.get("tool")
            if tool:
                attempts[tool] = attempts.get(tool, 0) + 1
    canonical_config_sha = sha256_json(exp)
    file_config_sha = hashlib.sha256((ROOT / "experiment.json").read_bytes()).hexdigest()
    manifest_config_ok = isinstance(manifest, dict) and (
        manifest.get("configuration_sha256") == canonical_config_sha
        or manifest.get("config_sha256") == file_config_sha
    )
    checks = {
        "initial_exit_75": process_evidence["initial"]["exit_code"] == EXPECTED_INITIAL_EXIT,
        "resume_exit_0": process_evidence["resume"]["exit_code"] == EXPECTED_RESUME_EXIT,
        "different_processes": process_evidence["initial"]["pid"] != process_evidence["resume"]["pid"],
        "state_present": isinstance(state, dict),
        "metrics_present": isinstance(metrics, dict),
        "logical_run_id_matches": isinstance(state, dict) and state.get("run_id") == run_id,
        "terminal_state_completed": isinstance(state, dict) and state.get("status") == "completed",
        "checkpoints_exact": isinstance(state, dict) and state.get("checkpoints") == exp["checkpoints"],
        "tool_sequence_exact": tool_sequence == [
            "normalize_records", "summarize_records", "summarize_records", "write_report"
        ],
        "attempt_counts_exact": attempts == {
            "normalize_records": 1, "summarize_records": 2, "write_report": 1
        },
        "trace_sequence_monotonic": bool(trace_seq)
        and trace_seq == list(range(1, len(trace_seq) + 1)),
        "report_exact": report == exp["task"]["expected_report"],
        "report_receipt_matches_bytes": isinstance(receipt, dict)
        and receipt.get("sha256") == report_sha,
        "secret_scan_passed": secret_ok,
        "configuration_hash_matches": manifest_config_ok,
        "one_retry": isinstance(metrics, dict) and metrics.get("retry_count") == 1,
        "one_injected_failure": isinstance(metrics, dict)
        and metrics.get("injected_failures") == 1,
        "four_tool_attempts": isinstance(metrics, dict) and metrics.get("tool_attempts") == 4,
        "three_tool_successes": isinstance(metrics, dict) and metrics.get("tool_successes") == 3,
        "two_processes": isinstance(metrics, dict) and metrics.get("process_count") == 2,
    }
    if arm == "agents_api":
        session_ids = []
        if isinstance(state, dict):
            if state.get("session_id"):
                session_ids.append(state["session_id"])
            session_ids.extend(state.get("session_ids") or [])
        checks["one_managed_session"] = len(set(session_ids)) == 1
    invalid_reason = _exclusion_reason(run_dir, state if isinstance(state, dict) else None)
    costed = add_costs(metrics or {}, exp, arm)
    checks["provider_usage_and_cost_available"] = (
        costed.get("provider_usage_complete") is True
        and isinstance(costed.get("total_tokens"), int)
        and isinstance(costed.get("total_cost_usd"), (int, float))
    )
    atomic_json(run_dir / "metrics.json", costed)
    result = {
        "arm": arm,
        "run_id": run_id,
        "valid": invalid_reason is None,
        "invalid_reason": invalid_reason,
        "task_passed": all(checks.values()),
        "checks": checks,
        "secret_findings": secret_findings,
        "metrics": costed,
        "returned_models": (
            (state.get("returned_models") or [state.get("returned_model")])
            if isinstance(state, dict)
            else []
        ),
    }
    atomic_json(run_dir / "validation.json", result)
    return result


def percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def summarize_values(values: Sequence[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "median": statistics.median(values) if values else None,
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "raw": list(values),
    }


def bootstrap_median_difference(
    differences: Sequence[float], *, resamples: int, seed: int
) -> dict[str, Any]:
    if not differences:
        return {"median_difference": None, "ci95": [None, None], "resamples": resamples, "seed": seed}
    rng = random.Random(seed)
    boot = [
        statistics.median(rng.choice(differences) for _ in differences)
        for _ in range(resamples)
    ]
    return {
        "median_difference": statistics.median(differences),
        "ci95": [percentile(boot, 0.025), percentile(boot, 0.975)],
        "resamples": resamples,
        "seed": seed,
    }


def analyze_pairs(pairs: Sequence[Mapping[str, Any]], exp: Mapping[str, Any]) -> dict[str, Any]:
    valid = [pair for pair in pairs if pair.get("valid") and not pair.get("warmup")]
    passing = [pair for pair in valid if all(row.get("task_passed") for row in pair["arms"].values())]
    analysis = {
        "valid_pairs": len(valid),
        "successful_pairs": len(passing),
        "arm_successes": {
            arm: sum(bool(pair["arms"][arm].get("task_passed")) for pair in valid)
            for arm in PROGRAMS
        },
        "recovery_gate_passed": False,
        "paired_metrics": {},
    }
    floor = 29
    analysis["recovery_gate_passed"] = all(
        count >= floor for count in analysis["arm_successes"].values()
    ) and len(valid) >= int(exp["execution"]["measured_pairs"])
    for metric in exp["analysis"]["paired_metrics"]:
        agents_values = []
        custom_values = []
        differences = []
        for pair in passing:
            a = pair["arms"]["agents_api"]["metrics"].get(metric)
            c = pair["arms"]["custom_harness"]["metrics"].get(metric)
            if isinstance(a, (int, float)) and isinstance(c, (int, float)):
                agents_values.append(float(a))
                custom_values.append(float(c))
                differences.append(float(a) - float(c))
        analysis["paired_metrics"][metric] = {
            "agents_api": summarize_values(agents_values),
            "custom_harness": summarize_values(custom_values),
            "agents_minus_custom": bootstrap_median_difference(
                differences,
                resamples=int(exp["execution"]["bootstrap_resamples"]),
                seed=int(exp["execution"]["bootstrap_seed"]),
            ),
        }
    analysis["winner_language_allowed"] = analysis["recovery_gate_passed"]
    return analysis


ArmRunner = Callable[[str, Path, str], dict[str, Any]]


def _run_pair(
    series_dir: Path,
    pair_ordinal: int,
    *,
    warmup: bool,
    arm_runner: ArmRunner,
    exp: Mapping[str, Any],
) -> dict[str, Any]:
    label = "warmup-000" if warmup else f"pair-{pair_ordinal:03d}"
    pair_dir = series_dir / label
    pair_dir.mkdir(parents=True, exist_ok=False)
    arms = {}
    arm_start_times = []
    for arm in ordered_arms(pair_ordinal, warmup=warmup):
        run_id = f"{series_dir.name}-{label}-{arm}"
        run_dir = pair_dir / arm
        process_evidence = arm_runner(arm, run_dir, run_id)
        arm_start_times.append(
            datetime.fromisoformat(
                process_evidence["initial"]["started_at"].replace("Z", "+00:00")
            )
        )
        arms[arm] = validate_trial(arm, run_dir, run_id, process_evidence, exp)
    separation = abs((arm_start_times[1] - arm_start_times[0]).total_seconds())
    pair_invalid_reason = None
    for row in arms.values():
        if not row["valid"]:
            pair_invalid_reason = row["invalid_reason"]
            break
    if separation > int(exp["execution"]["max_pair_start_separation_seconds"]):
        pair_invalid_reason = "pair_start_separation_exceeded"
    agent_models = set(arms["agents_api"].get("returned_models") or [])
    custom_models = set(arms["custom_harness"].get("returned_models") or [])
    if agent_models and custom_models and agent_models != custom_models:
        pair_invalid_reason = "returned_model_differs_between_arms"
    returned = {
        "pair_ordinal": pair_ordinal,
        "warmup": warmup,
        "order": ordered_arms(pair_ordinal, warmup=warmup),
        "arm_start_separation_seconds": separation,
        "valid": pair_invalid_reason is None,
        "invalid_reason": pair_invalid_reason,
        "arms": arms,
    }
    atomic_json(pair_dir / "pair.json", returned)
    return returned


def run_series(
    output_dir: Path,
    *,
    arm_runner: ArmRunner = run_arm_processes,
    max_replacement_pairs: int = 30,
) -> dict[str, Any]:
    if output_dir.exists():
        raise BenchmarkError(f"refusing to overwrite existing series: {output_dir}")
    readiness = preflight()
    if arm_runner is run_arm_processes and not readiness["ready"]:
        raise BenchmarkError("live preflight failed; inspect --preflight output")
    output_dir.mkdir(parents=True)
    exp = experiment()
    manifest = {
        "schema_version": 1,
        "benchmark_id": exp["benchmark_id"],
        "configuration_sha256": sha256_json(exp),
        "created_at": utc_now(),
        "preflight": readiness,
        "plan": build_plan(),
    }
    atomic_json(output_dir / "series-manifest.json", manifest)
    pairs = [_run_pair(output_dir, 0, warmup=True, arm_runner=arm_runner, exp=exp)]
    measured_target = int(exp["execution"]["measured_pairs"])
    valid_count = 0
    pair_ordinal = 1
    maximum_attempts = measured_target + max_replacement_pairs
    while valid_count < measured_target:
        if pair_ordinal > maximum_attempts:
            raise BenchmarkError("replacement-pair ceiling reached before 30 valid pairs")
        pair = _run_pair(
            output_dir,
            pair_ordinal,
            warmup=False,
            arm_runner=arm_runner,
            exp=exp,
        )
        pairs.append(pair)
        if pair["valid"]:
            valid_count += 1
        interim = {
            "status": "running",
            "pairs": pairs,
            "analysis": analyze_pairs(pairs, exp),
        }
        atomic_json(output_dir / "series.json", interim)
        pair_ordinal += 1
    result = {
        "status": "completed",
        "completed_at": utc_now(),
        "pairs": pairs,
        "analysis": analyze_pairs(pairs, exp),
    }
    atomic_json(output_dir / "series.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-replacement-pairs", type=int, default=30)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.plan:
        print(json.dumps(build_plan(), indent=2, sort_keys=True))
        return 0
    if args.preflight:
        result = preflight()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ready"] else 2
    if args.output_dir is None:
        print("benchmark_live: --execute requires --output-dir", file=sys.stderr)
        return 2
    try:
        result = run_series(
            args.output_dir.resolve(),
            max_replacement_pairs=args.max_replacement_pairs,
        )
    except BenchmarkError as exc:
        print(f"benchmark_live: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), **result["analysis"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
