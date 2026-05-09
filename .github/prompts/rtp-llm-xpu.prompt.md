---
description: "Orchestrate rtp-llm-xpu upstream sync, build, and full verification (function, performance, accuracy) on Intel XPU"
agent: "agent"
tools: [execute, read, search, todo]
argument-hint: "merge, build, verify, perf, accuracy"
---

# Auto-Merge & Verify rtp-llm-xpu

You are an orchestration agent.

## Step 0: Load Configuration

Read [.env](../.env) to load all variables:
`WORK_DIR`, `REPO_URL`, `BRANCH`, `MODEL_NAME`, `MODEL_TYPE`, `MODEL_PATH`, `TP_SIZE`,
`ZE_AFFINITY_MASK`, `FRONTEND_SERVER_COUNT`, `DATASET_PATH`.

Pass these variables to every skill invocation.

## Step 1: Parse Phases

Parse the user's message for which phases to run. The available phases are:

| Keyword | Skill | Description |
|---------|-------|-------------|
| `merge` | `xpu-merge` | Sync with upstream |
| `build` | `xpu-build` | Compile XPU target |
| `verify` | `xpu-verify` | Start service + function test |
| `perf` | `xpu-perf-benchmark` | Throughput benchmark |
| `accuracy` | `xpu-accuracy-benchmark` | GSM8K evaluation |

Rules:
- If user specifies phases (e.g., `merge, build, verify`), run ONLY those phases in order
- If user provides no phases, run ALL phases in order
- Always run phases in the order listed above, regardless of user input order

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
