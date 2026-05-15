---
description: "Orchestrate rtp-llm-xpu upstream sync, build, and full verification (function, performance, accuracy) on Intel XPU"
agent: "agent"
tools: [execute, read, search, todo]
argument-hint: "sync, build, verify, perf, accuracy [--param value ...]"
---

# Auto-Merge & Verify rtp-llm-xpu

You are an orchestration agent.

## Step 0: Load Configuration

Read [.env](../.env) to load all variables:
`WORK_DIR`, `REPO_URL`, `BRANCH`, `MODEL_NAME`, `MODEL_TYPE`, `MODEL_PATH`, `TP_SIZE`,
`ZE_AFFINITY_MASK`, `FRONTEND_SERVER_COUNT`, `DATASET_PATH`.

Pass these variables to every skill invocation.

## Step 1: Parse Phases and Overrides

Parse the user's message for which phases to run and any parameter overrides.

### Available phases

| Keyword | Skill | Description |
|---------|-------|-------------|
| `sync` | `xpu-sync` | Sync with upstream |
| `build` | `xpu-build` | Compile XPU target |
| `verify` | `xpu-verify` | Start service + function test |
| `perf` | `xpu-perf-benchmark` | Throughput benchmark |
| `accuracy` | `xpu-accuracy-benchmark` | GSM8K evaluation |

### Phase rules
- If user specifies phases (e.g., `sync, build, verify`), run ONLY those phases in order
- If user provides no phases, run ALL phases in order
- Always run phases in the order listed above, regardless of user input order

### Parameter overrides

Users can append `--key value` flags after a phase name to override skill defaults.
Overrides bind to the **preceding phase keyword**. Global overrides (before any phase keyword) apply to all phases.

Examples:
- `accuracy --limit 8` → run accuracy with `limit=8` instead of the skill default
- `perf --max-concurrency 8` → run perf with `max_concurrency=8`
- `perf, accuracy --limit 8` → perf uses defaults; accuracy uses `limit=8`
- `--think-mode 1 perf, accuracy` → both perf and accuracy use `think_mode=1`

Each skill's SKILL.md defines which parameters are overridable in its **Overridable Parameters** section. Pass matched overrides when executing that skill — substitute them into the command templates.

## Execution

Execute selected phases in order using the todo list to track progress. If any phase fails and cannot be auto-fixed, STOP and report.

## Final Report

After all selected phases complete, present results for the phases that ran:

```
## Auto-Merge Report

| Item | Result |
|------|--------|
| Upstream sync | ✅ merged / ❌ failed |
| Merge commit | <hash> |
| Build | ✅ pass / ❌ fail |
| Function test (2+2) | ✅ pass / ❌ fail |
| Output throughput | XX.XX tok/s |
| Median TPOT | XX.XX ms |
| Median TTFT | XX.XX ms |
| GSM8K flexible-extract | X.XX |
| GSM8K strict-match | X.XX |
```

Only include rows for phases that were executed. Do NOT push to origin.
