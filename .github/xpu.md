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
    ├── xpu-service/SKILL.md             # Phase: service (launch helper)
    ├── xpu-verify/SKILL.md              # Phase: verify
    ├── xpu-perf-benchmark/SKILL.md      # Phase: perf
    ├── xpu-accuracy-benchmark/SKILL.md  # Phase: accuracy
    └── xpu-kill-service/SKILL.md        # Phase: kill
```

## Configuration

Edit `.github/.env` before first use:

```bash
# Git
WORK_DIR=/workspace/rtp-llm-xpu
REPO_URL=https://github.com/aslanxie/rtp-llm-xpu.git
BRANCH=main

# Model
MODEL_NAME=Qwen3-8B
MODEL_TYPE=qwen_3
MODEL_PATH=/workspace/Qwen3-8B
TP_SIZE=1

# XPU device
ZE_AFFINITY_MASK=0,1
FRONTEND_SERVER_COUNT=1

# Benchmark
DATASET_PATH=/workspace/ShareGPT_V3_unfiltered_cleaned_split.json
```

| Variable | Description |
|----------|-------------|
| `WORK_DIR` | Absolute path to local rtp-llm-xpu workspace |
| `REPO_URL` | Git URL of your fork |
| `BRANCH` | Branch name (default: `main`) |
| `MODEL_NAME` | Model name used in API calls (e.g., `Qwen3-8B`) |
| `MODEL_TYPE` | Model type for start_server.py (e.g., `qwen_3`) |
| `MODEL_PATH` | Local path to model checkpoint / tokenizer |
| `TP_SIZE` | Tensor parallelism size |
| `ZE_AFFINITY_MASK` | Intel XPU device affinity. Single value (e.g., `0`) = single GPU mode. Two comma-separated values (e.g., `0,1`) = PD disaggregation on 2 GPUs |
| `FRONTEND_SERVER_COUNT` | Number of frontend server instances |
| `DATASET_PATH` | Path to ShareGPT dataset JSON for perf benchmark |

## Deployment Modes

The `ZE_AFFINITY_MASK` value determines the deployment mode:

| ZE_AFFINITY_MASK | Mode | Description |
|------------------|------|-------------|
| `0` | Standard | Single GPU, service on port 8088 |
| `0,1` | PD Disaggregation | 2 GPUs: first device = PREFILL (port 8088), second device = DECODE (port 9088) |

In PD mode:
- DECODE server starts **first** on the second device
- PREFILL server starts **second** on the first device
- Client always connects to port **8088** (PREFILL endpoint)
- KV cache is transferred via TCP (`--cache_store_rdma_mode 0`)

## Thinking Mode (Qwen3)

Qwen3 supports chain-of-thought reasoning via `<think>...</think>`. Controlled by `--think_mode INT`:

| Value | Behavior | Recommended For |
|-------|----------|-----------------|
| `0` | Thinking disabled | Perf benchmark, few-shot accuracy eval (`--num_fewshot 5`) |
| `1` | Thinking enabled (default) | Zero-shot accuracy eval (`--num_fewshot 0`) |

The `xpu-service` skill documents how to set thinking mode at server level or per-request.

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
| `verify` | `xpu-verify` | Launch service (via xpu-service), health-check, run a "2+2" function test. | ~2-3 min |
| `perf` | `xpu-perf-benchmark` | Run `vllm bench serve` with 100 ShareGPT prompts. Reports throughput, TPOT, TTFT. | ~15-20 min |
| `accuracy` | `xpu-accuracy-benchmark` | Run `lm-eval` GSM8K (10 items, 5-shot). Reports flexible-extract and strict-match scores. | ~5-10 min |
| `kill` | `xpu-kill-service` | Kill all running rtp_llm service processes and free ports 8088/9088. | ~5 sec |

### Skill Dependencies

The `xpu-verify`, `xpu-perf-benchmark`, and `xpu-accuracy-benchmark` skills all delegate service launch to the **xpu-service** skill, which handles mode detection and server startup.

The `xpu-kill-service` skill is standalone and can be used at any time to stop running services.

### Examples

```
# Run all 5 phases in order (kill is not included by default)
/rtp-llm-xpu

# Merge upstream and build only
/rtp-llm-xpu sync, build

# Full pipeline minus accuracy
/rtp-llm-xpu sync, build, verify, perf

# Just run performance benchmark (service must already be running)
/rtp-llm-xpu perf

# Just run accuracy benchmark (service must already be running)
/rtp-llm-xpu accuracy

# Kill running services
/rtp-llm-xpu kill

# Run perf benchmark then kill the service
/rtp-llm-xpu perf, kill
```

### Parameter Overrides

Append `--key value` after a phase name to override that skill's defaults.
Each skill lists its overridable parameters in the **Overridable Parameters** section of its SKILL.md.

```
# Run accuracy with only 8 GSM8K items (default: 64)
/rtp-llm-xpu accuracy --limit 8

# Run perf with higher concurrency
/rtp-llm-xpu perf --max-concurrency 16

# Run both perf and accuracy — accuracy uses limit=8, perf uses defaults
/rtp-llm-xpu perf, accuracy --limit 8

# Global override: enable thinking mode for both perf and accuracy
/rtp-llm-xpu --think-mode 1 perf, accuracy

# Accuracy with zero-shot (for thinking mode evaluation)
/rtp-llm-xpu accuracy --num-fewshot 0
```

Override binding rules:
- Flags **after** a phase keyword bind to that phase only
- Flags **before** any phase keyword apply to all phases
- See the orchestration prompt for full details

### Phase Order

Phases always execute in this fixed order regardless of input order:

**sync → build → verify → perf → accuracy → kill**

### Dependencies

```
sync ──► build ──► verify ──► perf
                      │
                      └────► accuracy ──► kill (optional)

verify uses: xpu-service (for launch)
perf uses:   xpu-service (for launch)
accuracy uses: xpu-service (for launch)
kill:         standalone (no dependencies)
```

- `perf` and `accuracy` require the service to be running (started by `verify`)
- If you skip `verify`, ensure the service is already running on `http://localhost:8088`
- `kill` can run independently at any time

## Output Report

After execution, a summary table is generated with only the rows for phases that ran:

```
## Auto-Merge Report

| Item                    | Result          |
|-------------------------|-----------------|
| Upstream sync           | ✅ synced       |
| Merge commit            | abc1234         |
| Build                   | ✅ pass         |
| Deploy mode             | PD (0,1)        |
| Function test (2+2)     | ✅ pass         |
| Output throughput       | 17.30 tok/s     |
| Median TPOT             | 57.36 ms        |
| Median TTFT             | 93.38 ms        |
| GSM8K flexible-extract  | 0.80            |
| GSM8K strict-match      | 0.80            |
| Service killed          | ✅ done         |
```

> **Note:** The orchestrator does NOT push to origin. Review the results and push manually.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Service not healthy for `perf`/`accuracy` | Run `verify` phase first, or start the service manually |
| Build fails after sync | The `build` skill auto-retries with fixes. If it still fails, check the error log and fix manually |
| Merge conflicts | The `sync` skill auto-resolves most conflicts. XPU-specific files keep local changes; upstream files accept upstream changes |
| Benchmark metrics truncated | The `perf` skill uses `tail -60` to capture full output. Do not reduce this value |
| PD mode DECODE OOM | Reduce `--max_context_batch_size` (default 4) or `num_concurrent` in lm-eval |
| Timeout / retry in lm-eval | Add `timeout=300` to model_args, or reduce `num_concurrent` |
| Long tail latency (last requests slow) | Normal for thinking mode (long generations). Use `--think_mode 0` for perf bench |
| Stale service after code changes | Run `kill` then `verify` to restart with fresh code |
