# rtp-llm-xpu Copilot Skills — User Guide

## Overview

This project includes a set of **VS Code Copilot skills** that automate the full lifecycle of maintaining an Intel XPU fork of [alibaba/rtp-llm](https://github.com/alibaba/rtp-llm) — from syncing upstream, building, to running functional, performance, and accuracy benchmarks.

All skills are orchestrated through a single prompt file and configured via a shared `.env` file.

## File Layout

```
.github/
├── .env                                  # Shared configuration (edit this)
├── prompts/
│   └── rtp-llm-xpu.prompt.md            # Orchestration prompt
└── skills/
    ├── xpu-sync/SKILL.md                # Phase: sync
    ├── xpu-build/SKILL.md               # Phase: build
    ├── xpu-verify/SKILL.md              # Phase: verify
    ├── xpu-perf-benchmark/SKILL.md      # Phase: perf
    └── xpu-accuracy-benchmark/SKILL.md  # Phase: accuracy
```

## Configuration

Edit `.github/.env` before first use:

```bash
# Git
WORK_DIR=/workspace/rtp-llm-xpu
REPO_URL=https://github.com/aslanxie/rtp-llm-xpu.git

# Model
MODEL_NAME=Qwen3-8B
MODEL_TYPE=qwen_3
MODEL_PATH=/workspace/Qwen3-8B
TP_SIZE=1

# XPU device
ZE_AFFINITY_MASK=4
FRONTEND_SERVER_COUNT=1

# Benchmark
DATASET_PATH=/workspace/ShareGPT_V3_unfiltered_cleaned_split.json
```

| Variable | Description |
|----------|-------------|
| `WORK_DIR` | Absolute path to local rtp-llm-xpu workspace |
| `REPO_URL` | Git URL of your fork |
| `MODEL_NAME` | Model name used in API calls (e.g., `Qwen3-8B`) |
| `MODEL_TYPE` | Model type for start_server.py (e.g., `qwen_3`) |
| `MODEL_PATH` | Local path to model checkpoint / tokenizer |
| `TP_SIZE` | Tensor parallelism size |
| `ZE_AFFINITY_MASK` | Intel XPU device affinity mask |
| `FRONTEND_SERVER_COUNT` | Number of frontend server instances |
| `DATASET_PATH` | Path to ShareGPT dataset JSON for perf benchmark |

## Usage

In VS Code Copilot Chat (Agent mode), type:

```
/rtp-llm-xpu <phases>
```

### Available Phases

| Phase | Skill | What It Does | Duration |
|-------|-------|--------------|----------|
| `sync` | `xpu-sync` | Fetch upstream `alibaba/rtp-llm:main` and sync into local `main`. Auto-resolves conflicts. | ~1 min |
| `build` | `xpu-build` | Bazel build with `--config=xpu`. Auto-fixes common build errors and retries. | ~10-30 min |
| `verify` | `xpu-verify` | Start the inference service, health-check, run a "2+2" function test. | ~2-3 min |
| `perf` | `xpu-perf-benchmark` | Run `vllm bench serve` with 100 ShareGPT prompts. Reports throughput, TPOT, TTFT. | ~15-20 min |
| `accuracy` | `xpu-accuracy-benchmark` | Run `lm-eval` GSM8K (10 items, 8-shot). Reports flexible-extract and strict-match scores. | ~5-10 min |

### Examples

```
# Run all 5 phases in order
/rtp-llm-xpu

# Merge upstream and build only
/rtp-llm-xpu sync, build

# Full pipeline minus accuracy
/rtp-llm-xpu sync, build, verify, perf

# Just run performance benchmark (service must already be running)
/rtp-llm-xpu perf

# Just run accuracy benchmark (service must already be running)
/rtp-llm-xpu accuracy
```

### Phase Order

Phases always execute in this fixed order regardless of input order:

**sync → build → verify → perf → accuracy**

### Dependencies

```
sync ──► build ──► verify ──► perf
                        │
                        └────► accuracy
```

- `perf` and `accuracy` require the service to be running (started by `verify`)
- If you skip `verify`, ensure the service is already running on `http://localhost:8088`

## Output Report

After execution, a summary table is generated with only the rows for phases that ran:

```
## Auto-Merge Report

| Item                    | Result          |
|-------------------------|-----------------|
| Upstream sync           | ✅ synced       |
| Merge commit            | abc1234         |
| Build                   | ✅ pass         |
| Function test (2+2)     | ✅ pass         |
| Output throughput       | 17.30 tok/s     |
| Median TPOT             | 57.36 ms        |
| Median TTFT             | 93.38 ms        |
| GSM8K flexible-extract  | 0.80            |
| GSM8K strict-match      | 0.80            |
```

> **Note:** The orchestrator does NOT push to origin. Review the results and push manually.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Service not healthy for `perf`/`accuracy` | Run `verify` phase first, or start the service manually |
| Build fails after sync | The `build` skill auto-retries with fixes. If it still fails, check the error log and fix manually |
| Merge conflicts | The `sync` skill auto-resolves most conflicts. XPU-specific files keep local changes; upstream files accept upstream changes |
| Benchmark metrics truncated | The `perf` skill uses `tail -60` to capture full output. Do not reduce this value |
