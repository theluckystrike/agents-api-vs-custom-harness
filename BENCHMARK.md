# Managed Agents API vs a minimal custom harness

Status: pre-registered protocol. Do not edit the hypotheses, fixture, outcome definitions, exclusion
rules, or sample size after the first live trial. Amendments must be appended, dated, and analyzed
separately.

## Implementation status

As of 2026-09-21, this directory contains offline, deterministic `--simulate` control paths. The
custom harness proves its local checkpoint/failure/resume path, and the managed-arm adapter simulates
durable managed-session identifiers. These are engineering smoke tests only. A live custom Responses
three-tool driver, a live managed Agents API three-tool driver, the common evaluator/supervisor, and
30 paired live runs have **not** yet been implemented and evidenced end to end. Consequently, there
is no live comparison result, latency/cost finding, or winning arm yet. Nothing in this pre-registration
should be read as evidence that either API completed the task.

## Question

For one small but genuinely resumable tool workflow, what does OpenAI's managed Agents API buy us
relative to a minimal loop built directly on the Responses API?

This benchmark compares orchestration, persistence, recovery, observability, latency, and cost. It
does **not** compare model quality: both arms use the same model, instructions, fixture, function
schemas, local function implementations, failure schedule, and output evaluator.

The managed arm uses an `environment: {"type":"none"}` Agents API session and handles pending
function calls through `required_actions`. The custom arm calls the Responses API directly with
server-side response storage disabled and owns the tool loop and transcript checkpoint. This is a
comparison of a managed session runtime against application-owned orchestration, not "OpenAI versus
no OpenAI."

Official API facts used by the design:

- [Create an agent session](https://developers.openai.com/api/reference/typescript/resources/beta/subresources/agents/subresources/sessions/methods/create)
  documents managed sessions, inline agent configuration, function tools, usage, and session state.
- [Agents API function tools](https://developers.openai.com/api/docs/guides/agents-api/tools/functions)
  documents `required_actions`, returning tool results with the original turn and call IDs, and
  recovery by retrieving the session after a disconnect.
- [Agents API tracing](https://developers.openai.com/api/docs/guides/agents-api/tracing)
  documents post-turn traces. Trace availability can lag the agent response, so trace-export delay
  is measured separately and is not included in task latency.
- [Create a response](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
  documents custom function calls, token usage, `store`, and conversation-state controls used by the
  custom arm.

The Agents API is beta. Every run therefore records the OpenAI SDK version, API surface, returned
model identifier, and UTC timestamp. If the API contract changes, create a new benchmark revision;
do not silently adapt an in-progress series.

## Pre-registered hypotheses

1. Both arms should recover successfully in at least 29 of 30 valid paired trials and produce the
   exact same semantic report.
2. The managed arm should require fewer application-owned durable-state bytes and fewer executable
   orchestration lines than the custom arm.
3. No directional latency, token, or dollar-cost winner is assumed. Those results are empirical.

Correct recovery is a gate, not a metric that can be traded for speed. If an arm falls below 29 of
30 successful resumed trials, report that failure prominently and do not declare it the winner on
latency or cost.

## Fixed task

The agent receives six deliberately untidy pipeline-run records. It must use these three distinct
local function tools in order:

1. `normalize_records`: trim and lowercase strings, parse `latency_ms` as a base-10 integer, keep the
   first occurrence of an ID, reject later duplicates and invalid latency values, and sort accepted
   records by ID.
2. `summarize_records`: calculate record count, error count, error rate, arithmetic mean latency, and
   nearest-rank p95 latency from normalized records.
3. `write_report`: validate and atomically write the canonical report. This is the only tool with a
   material side effect.

The raw fixture is embedded in `experiment.json`; no network or wall-clock data enters the task. Its
canonical interpretation is:

```json
[
  {"id":" job-003 ","latency_ms":"80","status":"ok"},
  {"id":"job-001","latency_ms":"120","status":"ok"},
  {"id":"job-002","latency_ms":"200","status":"error"},
  {"id":"job-001","latency_ms":"120","status":"ok"},
  {"id":"job-004","latency_ms":"160","status":"ok"},
  {"id":"job-005","latency_ms":"not-a-number","status":"ok"}
]
```

The semantic expected result is fixed before any run:

```json
{
  "task": "durable-three-tool-pipeline-v1",
  "input": {
    "source_count": 6,
    "valid_count": 4,
    "dropped": [
      {"id": "job-001", "index": 3, "reason": "duplicate_id"},
      {"id": "job-005", "index": 5, "reason": "invalid_latency"}
    ]
  },
  "summary": {
    "count": 4,
    "error_count": 1,
    "error_rate": 0.25,
    "mean_latency_ms": 140.0,
    "p95_latency_ms": 200
  },
  "normalized_records_sha256": "d4ce35528f777b9c847c0ce2b52e2db6511db8a030a1741e087cd560a403c98c"
}
```

JSON object-key order, whitespace, run IDs, timestamps, hashes, and additive provenance fields are
ignored by the semantic evaluator. The values above and the normalized four-record set are not.
`p95_latency_ms` uses nearest rank: after sorting, select element `ceil(0.95 * n)` using one-based
indexing.

## Controlled failure and resume

There are three distinct tools and four expected tool attempts:

| Attempt | Tool | Expected result | Durable checkpoint |
|---:|---|---|---|
| 1 | `normalize_records` | success | `C1_NORMALIZED` |
| 2 | `summarize_records` | injected `TEMP_UNAVAILABLE` | `C2_FAILURE_PERSISTED` |
| 3 | `summarize_records` | success after resume | `C3_SUMMARIZED` |
| 4 | `write_report` | success | `C4_REPORT_COMMITTED` |

The fault injector is keyed by `(logical_run_id, tool_name, canonical_arguments_sha256)`. The first
`summarize_records` attempt returns this structured result and marks the injection as consumed using
an atomic, fsynced write:

```json
{
  "ok": false,
  "error": {
    "code": "TEMP_UNAVAILABLE",
    "message": "pre-registered transient failure",
    "retryable": true,
    "retry_after_ms": 0
  }
}
```

The arm must durably record the failed tool result before the fault controller exits the process
with code `75`. The initial invocation is considered wrong if it returns zero, exits before C2 is
durable, or proceeds to a successful summary in the same process. The supervisor then starts a new
process with `--resume LOGICAL_RUN_ID`.

On resume, the managed arm retrieves the existing session and trusts current `required_actions`,
not a historical function-call item, to decide whether a result is pending. It retains the original
session, turn, and call IDs and keeps a local receipt for any side effect. The custom arm loads its
atomically written state and transcript. Neither arm may re-run the successful normalization.

Every tool uses a stable idempotency key derived from the logical run ID, tool name, canonical
arguments hash, and semantic attempt number. `write_report` writes a temporary file, fsyncs it,
renames it, and records its SHA-256 receipt before reporting success. A resumed duplicate request
returns that receipt without writing again. Exactly one committed report and zero duplicate side
effects are required.

## Checkpoint contract

Each arm has the same logical checkpoints:

- `C0_INITIALIZED`: frozen config hash, logical run ID, arm, and start time are durable.
- `C1_NORMALIZED`: normalized output and rejection list are durable.
- `C2_FAILURE_PERSISTED`: `TEMP_UNAVAILABLE`, its tool/call identity, and consumed-fault marker are
  durable; this is the mandatory crash boundary.
- `C3_SUMMARIZED`: the successful summary is durable after the replacement process resumes.
- `C4_REPORT_COMMITTED`: report hash and one side-effect receipt are durable.
- `C5_VERIFIED`: semantic evaluator and trace-invariant results are durable.

`state.json` is the application-owned recovery record. The managed arm should contain only the
session pointer plus local tool receipts and fault state needed for safe client-side execution; the
custom arm additionally contains its replayable transcript/FSM state. Checkpoint writes must be
atomic. Secrets, chain-of-thought, and full hidden reasoning must never be written.

`trace.jsonl` is append-only and must contain monotonically increasing `seq` values. Every event has
`schema_version`, `run_id`, `arm`, `process_instance_id`, `wall_time_utc`, `monotonic_ns`, `event`,
`checkpoint`, `resumed`, and, when applicable, `tool`, `attempt`, `call_id`, `arguments_sha256`,
`result_sha256`, `status`, and `error_code`. Provider traces are exported separately; they do not
replace this common trace.

## Arms

### A — managed Agents API

- Create one inline managed agent session for the logical run with `environment.type = none`.
- Supply the frozen instruction text and the three function schemas from `experiment.json`.
- Execute local functions only when they appear in current `required_actions`.
- Submit success or error results using the returned session, turn, and call IDs.
- After process replacement, retrieve the same session and continue it; creating a replacement
  session makes the trial a recovery failure.
- Persist only the remote pointer and the local receipts/fault state required for safe recovery.
- Export the managed trace after the turn terminates. Record export lag independently.

### B — minimal custom Responses harness

- Call the Responses API directly with `store: false`; do not use the Agents API, Agents SDK runner,
  Conversations API, `previous_response_id`, or another orchestration framework.
- Implement only a sequential model/tool loop, strict argument validation, bounded retry policy,
  append-only trace, atomic checkpoint, idempotency receipt, and terminal-output validation.
- Persist and replay the response/tool transcript required to continue after process replacement.
- No handoffs, planning framework, memory package, hosted tools, or background worker is allowed.

Both implementations may share fixture loading, canonical JSON hashing, the three tool bodies, the
fault injector, evaluator, and metric writer. Arm-specific orchestration code must remain separable
so executable lines and durable bytes can be attributed honestly.

## Fairness controls

- Model: `gpt-5.6-terra`, reasoning effort `low`, text verbosity `low`, service tier `default`. If
  that exact model is unavailable, stop before the first measured live pair and issue a new protocol
  revision. Never substitute a model mid-series.
- Use the exact returned model identifier in every result. A mismatched pair is invalid.
- Identical developer instruction, user task, tool names/descriptions/JSON Schemas, strict argument
  validation, fixture bytes, local tool code, output schema, retry limit, and evaluator.
- Sequential tool execution only. No parallel calls, streaming, speculative execution, or hidden
  retries in either arm.
- One warm-up run per arm is excluded and visibly retained. Then run 30 measured pairs. Odd pairs
  run managed first; even pairs run custom first. Never run both simultaneously.
- Use a fresh logical run ID and fresh output directory per arm/trial. Use the same machine and
  network, and start the second arm within five minutes of the first.
- Retry only the pre-registered tool failure. Provider/API retries are disabled. Rate limits,
  unrelated provider 5xx responses, machine sleep, network loss, or model mismatch invalidate the
  entire pair; retain it, label the reason, and append a replacement pair.
- Model mistakes, invalid tool arguments, failure to retry, extra tool calls, wrong reports, and
  duplicate side effects are outcomes, never exclusions.
- The maximum is eight model calls and eight tool attempts per logical run. Crossing either limit is
  a task failure and halts that run.
- Freeze dependency lockfiles and record OS, architecture, Python version, SDK version, git commit,
  dirty-worktree hash, configuration SHA-256, and pricing snapshot before starting.

## Measures

All durations use a monotonic clock. Record both phases and the end-to-end total:

- `initial_wall_ms`: C0 through durable C2 and process exit.
- `restart_gap_ms`: first-process exit through replacement-process start.
- `recovery_wall_ms`: replacement-process start through C5.
- `active_wall_ms`: `initial_wall_ms + recovery_wall_ms`, excluding supervisor gap.
- `end_to_end_wall_ms`: C0 through C5, including restart gap.
- per-model-call and per-tool-attempt latency.
- model calls, tool attempts, tool successes, injected failures, retry count, resume count, provider
  requests, and duplicate tool/side-effect counts.
- input, cached-input, output, reasoning, and total tokens exactly as returned. Unavailable fields are
  `null`, never guessed as zero.
- model cost in USD from the frozen official pricing snapshot. Calculate uncached input, cached input,
  and output components separately; record any Agents API/session charge separately rather than
  hiding it in token cost. Unknown price components are `null` and make total cost `null`.
- application-owned checkpoint bytes at every checkpoint; report C2 and peak values.
- executable, nonblank, noncomment lines in arm-specific orchestration modules, reported as a
  descriptive maintainability proxy, not as proof of quality.
- trace completeness and provider-trace export lag.

Primary analysis is paired. First report the 30 raw pairs. For successful pairs report each arm's
median, p25, p75, p95, and the paired median difference for `recovery_wall_ms`, `active_wall_ms`,
total tokens, and total cost. Bootstrap the paired median difference with 10,000 resamples using seed
`20260921` and report the 95% percentile interval. Also report success counts, exact-report counts,
duplicate side effects, checkpoint bytes, and lines of code without collapsing them into one score.

## Required outputs and invariants

Each arm/trial directory contains:

- `run-manifest.json`: frozen environment/config/provenance and process identities.
- `state.json`: latest atomic application checkpoint.
- `trace.jsonl`: common append-only event trace.
- `report.json`: final task report, absent before C4.
- `metrics.json`: raw counts, usage, durations, costs, and validator results.
- `stdout.log` and `stderr.log` for both process invocations.
- `provider-trace.json` for the managed arm when export is available; otherwise an explicit
  `provider_trace_status` and error, never a fabricated empty trace.

A live trial passes only if all of these are true:

1. First invocation exits 75 after C2; second invocation is a distinct process and reaches C5.
2. One logical run ID is used throughout. The managed arm uses one session ID.
3. Checkpoints occur exactly in order C0, C1, C2, C3, C4, C5.
4. Tool attempts are normalize success, summarize transient failure, summarize success, write success.
5. Normalization is not repeated after resume; exactly one retry occurs.
6. Exactly one report side effect is committed and its stored SHA-256 matches `report.json`.
7. The report is semantically equal to the fixed expected result.
8. Usage fields come from provider responses and all cost math is reproducible from the saved price
   snapshot.
9. No secret or authorization header appears in any artifact.

## Five-minute no-key verification

Simulation mode is mandatory so anyone can verify checkpoints and recovery without an OpenAI key.
It uses a deterministic scripted planner that requests the same canonical tool sequence; it does not
pretend to run a model or provider.

The current minimal-harness contract is:

```bash
cd agents-api-vs-custom-harness
unset OPENAI_API_KEY

# Expected: exits 75 after C2_FAILURE_PERSISTED.
python custom_harness.py --simulate --run-dir runs/smoke/custom --run-id smoke-custom

# Expected: exits 0, skips normalize_records, and reaches C5_VERIFIED.
python custom_harness.py --simulate --run-dir runs/smoke/custom --resume smoke-custom
```

The managed-arm simulator must expose the same `--simulate`, `--run-dir`, `--run-id`, and `--resume`
semantics. A top-level runner may wrap these commands, but must preserve exit 75 at the fault boundary.

Simulation artifacts set `execution_mode: "simulation"`, `comparison_eligible: false`, provider
request counts to zero, token/cost fields to `null`, and provider trace status to `not_applicable`.
They still must pass all local task, checkpoint, restart, idempotency, trace, report, and secret-scan
invariants. Simulation proves the recovery machinery is reproducible; only the 30 live paired trials
can support claims about API latency, tokens, cost, or managed-runtime reliability.

## Interpretation limits

This is one small, sequential, three-tool workflow with a single known failure point. It does not
generalize by itself to multi-agent handoffs, long-context memory, hosted sandboxes, concurrent
tools, human approval, or production outage recovery. The fixture is deterministic and favors
measurement over ecological breadth. Managed remote state and local custom state also have different
privacy, retention, operational, and failure-domain properties that byte count alone cannot price.

Publish raw runs, failed trials, exclusions, lockfiles, evaluator code, pricing snapshot, and the
frozen `experiment.json`. Any public conclusion must say "for this task and configuration" and must
separate simulation results from live results.
