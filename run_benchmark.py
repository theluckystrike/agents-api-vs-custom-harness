#!/usr/bin/env python3
"""Run the two-process, no-key recovery controls for both benchmark arms.

This runner deliberately does not calculate a winner.  Both arms are scripted
offline controls, so their wall time, tokens, and cost are not API evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent
ARMS = {
    "custom_harness": ROOT / "custom_harness.py",
    "agents_api": ROOT / "agents_api.py",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def invoke(command: list[str], run_dir: Path, phase: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    (run_dir / f"{phase}-stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_dir / f"{phase}-stderr.log").write_text(completed.stderr, encoding="utf-8")
    return completed


def run_arm(arm: str, program: Path, destination: Path, logical_run_id: str) -> dict[str, Any]:
    destination.mkdir(parents=True)
    started_at = utc_now()
    initial_command = [
        sys.executable,
        str(program),
        "--simulate",
        "--run-dir",
        str(destination),
        "--run-id",
        logical_run_id,
    ]
    resume_command = [
        sys.executable,
        str(program),
        "--simulate",
        "--run-dir",
        str(destination),
        "--resume",
        logical_run_id,
    ]
    initial = invoke(initial_command, destination, "initial")
    resumed = invoke(resume_command, destination, "resume")

    metrics_path = destination / "metrics.json"
    report_path = destination / "report.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    invariants = {
        "initial_exit_is_75": initial.returncode == 75,
        "resume_exit_is_0": resumed.returncode == 0,
        "three_successful_tool_calls": bool(metrics and metrics.get("successful_tool_calls") == 3),
        "four_total_tool_attempts": bool(metrics and metrics.get("tool_calls") == 4),
        "one_retry": bool(metrics and metrics.get("retries") == 1),
        "report_exists": report is not None,
    }
    manifest = {
        "schema_version": 1,
        "arm": arm,
        "logical_run_id": logical_run_id,
        "execution_mode": "deterministic_simulation",
        "comparison_eligible": False,
        "started_at": started_at,
        "completed_at": utc_now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "openai_api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
        "initial_exit_code": initial.returncode,
        "resume_exit_code": resumed.returncode,
        "invariants": invariants,
        "passed": all(invariants.values()),
    }
    atomic_json(destination / "run-manifest.json", manifest)
    return {"manifest": manifest, "metrics": metrics, "report": report}


def run_all(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {output_dir}")
    output_dir.mkdir(parents=True)
    run_token = output_dir.name.replace(" ", "-")
    results = {
        arm: run_arm(arm, program, output_dir / arm, f"{run_token}-{arm}")
        for arm, program in ARMS.items()
    }
    custom_summary = (results["custom_harness"].get("report") or {}).get("summary")
    agents_summary = (results["agents_api"].get("report") or {}).get("summary")
    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "status": (
            "offline_control_passed"
            if all(row["manifest"]["passed"] for row in results.values())
            and custom_summary == agents_summary
            else "offline_control_failed"
        ),
        "execution_mode": "deterministic_simulation",
        "comparison_eligible": False,
        "semantic_summaries_match": custom_summary == agents_summary,
        "arms": {
            arm: {
                "passed": row["manifest"]["passed"],
                "metrics": row["metrics"],
            }
            for arm, row in results.items()
        },
        "unavailable": {
            "live_agents_api_trial": "OPENAI_API_KEY and a compatible OpenAI SDK were not available",
            "live_custom_responses_trial": "not implemented or run in this no-key control",
            "provider_tokens": None,
            "provider_cost_usd": None,
            "provider_latency_ms": None,
        },
        "limitations": [
            "The Agents API arm is a scripted state-machine simulation, not an API call.",
            "The custom arm is a deterministic local harness, not a Responses API model loop.",
            "No latency, token, cost, quality, or recovery winner can be inferred from this run.",
        ],
    }
    atomic_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / datetime.now(timezone.utc).strftime("simulation-%Y%m%dT%H%M%SZ"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_all(args.output_dir.resolve())
    except (FileExistsError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"benchmark: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), **summary}, sort_keys=True))
    return 0 if summary["status"] == "offline_control_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
