# Agents API vs a custom harness: a preregistered recovery benchmark

This repository is an executable benchmark scaffold for one narrow question:
on the same three-tool workflow, what does a managed Agents API session buy us
over application-owned orchestration?

The honest result today is **not a winner**. The no-key recovery controls pass,
but no live paired API trial has run. The checked-in simulation proves that both
local implementations can persist a deliberate transient failure, exit, resume
in a new process, skip completed work, and produce the same report. It does not
measure OpenAI model quality, provider latency, token use, cost, or managed
recovery reliability.

That distinction is machine-readable in every simulated run:

```json
{
  "execution_mode": "deterministic_simulation",
  "comparison_eligible": false,
  "status": "offline_control_passed"
}
```

## Five-minute quickstart

Simulation uses only the Python standard library and does not need an API key.

```bash
git clone https://github.com/theluckystrike/agents-api-vs-custom-harness.git
cd agents-api-vs-custom-harness
python3 run_benchmark.py
python3 -m unittest discover -s tests -v
```

The runner creates a timestamped directory under `runs/`. For each arm it:

1. invokes the workflow and expects exit code `75` after the persisted
   `TEMP_UNAVAILABLE` failure;
2. starts a second process with `--resume`;
3. verifies four tool attempts, three successes, one retry, and one report; and
4. writes `summary.json` without turning unavailable provider metrics into zero.

The frozen example is in
`runs/simulation-20260921T170000Z/summary.json`. Both arms went `75 -> 0`, made
four tool attempts, recorded one retry, and produced byte-identical reports.
Provider tokens, cost, and latency remain `null`.

## The task

Six deliberately untidy pipeline records move through three tools:

- `normalize_records` trims and validates fields, removes a duplicate, rejects
  an invalid latency, and sorts four accepted records;
- `summarize_records` calculates count, error rate, mean latency, and nearest-rank
  p95, but its first attempt returns the planned transient failure; and
- `write_report` produces the canonical report.

The first process must stop after the failed summary attempt is durable. The
replacement process must not repeat normalization. It retries the summary,
writes the report, and reaches a completed checkpoint. A third resume is
idempotent.

The expected summary is fixed before any live run:

```json
{
  "count": 4,
  "error_count": 1,
  "error_rate": 0.25,
  "mean_latency_ms": 140.0,
  "p95_latency_ms": 200
}
```

## What is implemented

- `custom_harness.py` is the small application-owned checkpoint/retry loop.
- `agents_api.py` is a standard-library, mockable adapter around the current
  `client.beta.agents.sessions` interface, plus an explicitly labeled offline
  session simulator.
- `managed_live.py` is the provider-backed managed-session arm. It persists the
  session pointer and tool receipts, stops at the required failure boundary,
  and resumes the same session in a replacement process.
- `responses_live.py` is the provider-backed application-owned Responses loop.
  It uses `store=False`, disables SDK retries, persists the replay transcript,
  and never uses managed conversation state.
- `benchmark_live.py` performs the dependency/credential preflight, supervises
  the warm-up and 30 order-balanced pairs sequentially, validates every
  artifact, retains invalid pairs, computes frozen-price cost, and withholds
  winner language until the recovery gate passes.
- `run_benchmark.py` supervises both offline two-process controls and preserves
  stdout, stderr, state, metrics, traces, manifests, and reports.
- `experiment.json` freezes the fixture, tools, failure, model configuration,
  checkpoints, expected output, exclusions, and analysis plan.
- `BENCHMARK.md` preregisters 30 order-balanced live pairs and the rules for
  interpreting them.

The adapter follows the official Agents API shape: OpenAI manages sessions,
orchestration, compaction, and recovery while the application supplies tools
and chooses an environment. The official quickstart also requires an API key
with Agents and Responses permissions, and uses the beta Agents session
namespace. See the [Agents API overview](https://developers.openai.com/api/docs/guides/agents-api/overview),
[quickstart](https://developers.openai.com/api/docs/guides/agents-api/quickstart),
and [session creation reference](https://developers.openai.com/api/reference/python/resources/beta/subresources/agents/subresources/sessions/methods/create).

## What is not implemented or claimed

There is still no provider-backed result or set of 30 paired runs in this
revision. Both live arms and the common supervisor are implemented and pass
mocked contract tests; that does not prove the beta API accepts the run or that
either arm recovers in production. Do not compare simulator wall clock against
provider execution.

The remaining gate is a scoped `OPENAI_API_KEY` with Agents and Responses
permissions. The live series then runs one excluded warm-up per arm followed by
30 sequential paired trials. The exact returned model must match the frozen
model; substitutions require a new protocol revision. Invalid and failed
trials stay visible.

## Why publish before the live result?

Publishing the protocol first removes several easy degrees of freedom. The
model, sample size, arm order, failure boundary, output evaluator, exclusions,
primary measures, and stop conditions cannot quietly change after seeing the
numbers. It also gives reviewers something more useful than a benchmark chart:
they can inspect whether the recovery claim is falsifiable before provider data
exists.

For this revision, the only supported conclusion is: **the offline recovery
control is reproducible, the live implementation is contract-tested, and the
provider comparison is pending credentials and execution.**

## Live preflight and execution

The beta Agents surface is currently pinned to an exact commit of the official
OpenAI Python SDK because the latest PyPI release inspected before trial zero
did not expose `beta.agents.sessions`. Use Python 3.10 or newer; the frozen
development environment uses Python 3.13.11.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock

# Prints every readiness check. With no key, exits 2 before creating a run.
.venv/bin/python benchmark_live.py --preflight

# Shows the frozen warm-up and 30-pair order without calling the provider.
.venv/bin/python benchmark_live.py --plan

# Provider-backed execution; refuses to overwrite OUTPUT_DIR.
OPENAI_API_KEY=... .venv/bin/python benchmark_live.py \
  --execute --output-dir runs/live-YYYYMMDDTHHMMSSZ
```

The checked-in `evidence/live-preflight-20260921T171810Z.json` shows that the
pinned SDK exposes both required API surfaces and that execution was stopped
solely because no key was present. It is readiness evidence, not a live result.

## Reproduce individual arms

```bash
# Both first commands intentionally exit 75.
python3 custom_harness.py --simulate --run-dir /tmp/custom-run --run-id custom-1
python3 custom_harness.py --simulate --run-dir /tmp/custom-run --resume custom-1

python3 agents_api.py --simulate --run-dir /tmp/agents-run --run-id agents-1
python3 agents_api.py --simulate --run-dir /tmp/agents-run --resume agents-1
```

Run the focused suite with either test runner:

```bash
python3 -m unittest discover -s tests -v
python3 -m pytest -q tests  # if pytest is installed
```

## License

MIT. See `LICENSE`.
