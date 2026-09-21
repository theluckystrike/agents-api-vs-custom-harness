from __future__ import annotations

import json
import copy
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark_live  # noqa: E402
import managed_live  # noqa: E402
import responses_live  # noqa: E402


def test_plan_freezes_one_warmup_and_30_alternating_pairs():
    plan = benchmark_live.build_plan()
    assert plan["status"] == "plan_only"
    assert plan["comparison_eligible"] is False
    assert plan["warmup"]["excluded"] is True
    assert len(plan["measured_pairs"]) == 30
    assert plan["measured_pairs"][0]["order"] == ["agents_api", "custom_harness"]
    assert plan["measured_pairs"][1]["order"] == ["custom_harness", "agents_api"]
    assert plan["run_arms_concurrently"] is False


def test_both_live_arms_receive_identical_frozen_user_input():
    exp = benchmark_live.experiment()
    assert managed_live._task_input(exp) == responses_live.initial_input(exp)[0]["content"]


def test_preflight_fails_honestly_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = benchmark_live.preflight()
    assert result["ready"] is False
    assert result["checks"]["openai_api_key_present"] is False
    assert result["checks"]["pricing_snapshot_ready"] is True
    assert result["checks"]["programs_present"]["custom_harness"] is True
    assert isinstance(result["checks"]["openai_sdk_agents_sessions_surface"], bool)
    assert isinstance(result["checks"]["openai_sdk_responses_surface"], bool)


def test_cost_calculation_uses_frozen_rates_and_null_policy():
    exp = benchmark_live.experiment()
    costed = benchmark_live.add_costs(
        {"input_tokens": 1000, "cached_input_tokens": 200, "output_tokens": 100},
        exp,
        "agents_api",
    )
    assert costed["uncached_input_cost_usd"] == pytest.approx(0.0016)
    assert costed["cached_input_cost_usd"] == pytest.approx(0.00004)
    assert costed["output_cost_usd"] == pytest.approx(0.0012)
    assert costed["total_cost_usd"] == pytest.approx(0.00284)
    missing = benchmark_live.add_costs(
        {"input_tokens": None, "cached_input_tokens": None, "output_tokens": None},
        exp,
        "custom_harness",
    )
    assert missing["total_cost_usd"] is None


def test_paired_bootstrap_is_reproducible_and_direction_is_agents_minus_custom():
    first = benchmark_live.bootstrap_median_difference(
        [2.0, 4.0, 6.0], resamples=1000, seed=20260921
    )
    second = benchmark_live.bootstrap_median_difference(
        [2.0, 4.0, 6.0], resamples=1000, seed=20260921
    )
    assert first == second
    assert first["median_difference"] == 4.0
    assert first["ci95"][0] <= 4.0 <= first["ci95"][1]


def test_analyzer_withholds_winner_below_recovery_gate():
    exp = benchmark_live.experiment()
    pairs = []
    for index in range(1, 9):
        pairs.append(
            {
                "warmup": False,
                "valid": True,
                "arms": {
                    "agents_api": {
                        "task_passed": True,
                        "metrics": {
                            "recovery_wall_ms": 10 + index,
                            "active_wall_ms": 20 + index,
                            "total_tokens": 100 + index,
                            "total_cost_usd": 0.01,
                        },
                    },
                    "custom_harness": {
                        "task_passed": True,
                        "metrics": {
                            "recovery_wall_ms": 8 + index,
                            "active_wall_ms": 18 + index,
                            "total_tokens": 90 + index,
                            "total_cost_usd": 0.009,
                        },
                    },
                },
            }
        )
    result = benchmark_live.analyze_pairs(pairs, exp)
    assert result["valid_pairs"] == 8
    assert result["successful_pairs"] == 8
    assert result["winner_language_allowed"] is False
    assert result["paired_metrics"]["recovery_wall_ms"]["agents_minus_custom"]["median_difference"] == 2


def test_execute_refuses_existing_directory_and_failed_real_preflight(tmp_path, monkeypatch):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(benchmark_live.BenchmarkError, match="overwrite"):
        benchmark_live.run_series(existing)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    target = tmp_path / "not-created"
    with pytest.raises(benchmark_live.BenchmarkError, match="preflight"):
        benchmark_live.run_series(target)
    assert not target.exists()


def test_small_fake_series_runs_sequentially_and_writes_analysis(tmp_path, monkeypatch):
    exp = copy.deepcopy(benchmark_live.experiment())
    exp["execution"]["measured_pairs"] = 2
    exp["execution"]["bootstrap_resamples"] = 100
    monkeypatch.setattr(benchmark_live, "experiment", lambda: exp)
    calls = []

    def fake_runner(arm, run_dir, run_id):
        calls.append((arm, run_id))
        run_dir.mkdir(parents=True, exist_ok=False)
        state = {
            "run_id": run_id,
            "checkpoints": exp["checkpoints"],
            "session_id": f"session-{run_id}" if arm == "agents_api" else None,
        }
        metrics = {
            "retry_count": 1,
            "tool_attempts": 4,
            "tool_successes": 3,
            "process_count": 2,
            "recovery_wall_ms": 5,
            "active_wall_ms": 9,
            "input_tokens": 100,
            "cached_input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 120,
        }
        (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
        (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (run_dir / "report.json").write_text(
            json.dumps(exp["task"]["expected_report"]), encoding="utf-8"
        )
        (run_dir / "trace.jsonl").write_text("{}\n", encoding="utf-8")
        (run_dir / "stderr-initial.log").write_text("", encoding="utf-8")
        (run_dir / "stderr-resume.log").write_text("", encoding="utf-8")
        return {
            "initial": {
                "pid": len(calls) * 2,
                "started_at": "2026-09-21T17:00:00Z",
                "exit_code": 75,
            },
            "resume": {
                "pid": len(calls) * 2 + 1,
                "started_at": "2026-09-21T17:00:01Z",
                "exit_code": 0,
            },
        }

    result = benchmark_live.run_series(tmp_path / "series", arm_runner=fake_runner)
    assert result["status"] == "completed"
    assert len(result["pairs"]) == 3  # one excluded warmup plus two measured
    assert result["analysis"]["valid_pairs"] == 2
    assert [arm for arm, _ in calls] == [
        "agents_api",
        "custom_harness",
        "agents_api",
        "custom_harness",
        "custom_harness",
        "agents_api",
    ]
    saved = json.loads((tmp_path / "series" / "series.json").read_text())
    assert saved["status"] == "completed"
