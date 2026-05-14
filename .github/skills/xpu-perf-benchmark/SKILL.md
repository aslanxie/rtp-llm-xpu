---
name: xpu-perf-benchmark
description: 'Run vllm serving benchmark on rtp-llm-xpu with ShareGPT dataset. Use when: measuring throughput, latency, TPOT, TTFT on Intel XPU. Requires running service on port 8088.'
---

# Performance Benchmark (vllm bench serve)

## Inputs (from .env)

- **WORK_DIR** — workspace path
- **MODEL_NAME** — model name
- **MODEL_TYPE** — model type (e.g., `qwen_3`)
- **MODEL_PATH** — tokenizer / checkpoint path
- **TP_SIZE** — tensor parallelism size
- **ZE_AFFINITY_MASK** — XPU device mask. Single value (e.g., `0`) = single GPU. Two values (e.g., `0,1`) = PD disaggregation.
- **FRONTEND_SERVER_COUNT** — number of frontend servers
- **DATASET_PATH** — ShareGPT dataset path

## When to Use
- Measuring decode throughput and latency after build or optimization
- Comparing performance before/after code changes

## Procedure

Follow these steps EXACTLY in order. Do NOT run any commands outside this procedure.

### 1. Check Service Health

```
export no_proxy="localhost,127.0.0.1"
curl -sS --max-time 5 http://localhost:8088/health
```

- If response is `"ok"` or `{"status":"ok"}`: service is running. Skip to **Step 3**.
- If connection refused or timeout: no service running. Continue to **Step 2**.

### 2. Launch Service

Use the **xpu-service** skill to launch the service with `--think_mode 0` (recommended for perf benchmarks to avoid long chain-of-thought generations that skew latency metrics). It will automatically select the correct mode based on `ZE_AFFINITY_MASK`:
- Single value (e.g., `0`) → **Standard Mode** (single GPU on port 8088)
- Two values (e.g., `0,1`) → **PD Mode** (DECODE on second device:9088, PREFILL on first device:8088)

Follow the xpu-service skill procedure, then return here.

### 3. Run Benchmark

It is required to capture no less than 56 lines in the output of benchmark command to extract key metrics.

```
cd $WORK_DIR
export no_proxy="localhost,127.0.0.1"

vllm bench serve \
  --backend openai-chat \
  --model "$MODEL_NAME" \
  --tokenizer "$MODEL_PATH/" \
  --base-url http://localhost:8088 \
  --dataset-name sharegpt \
  --dataset-path $DATASET_PATH \
  --num-prompts 100 \
  --request-rate inf \
  --max-concurrency 1 \
  --seed 42 \
  --metric-percentiles 10,20,30,40,50,60,70,80,90,95,99 \
  --endpoint "/v1/chat/completions" 2>&1 | tail -60
```

For PD mode, consider `--max-concurrency 4` to test batched decode throughput.

IMPORTANT: You MUST use `tail -60` (not less) to capture the full metrics output including all percentile rows.

This takes ~15-20 minutes with 100 prompts at concurrency 1.

### 4. Extract Metrics

From the benchmark output, extract:
- **Output token throughput (tok/s)** — primary throughput metric
- **Median TPOT (ms)** — per-token decode latency
- **Median TTFT (ms)** — time to first token
- **P99 TPOT (ms)** — tail latency
- **Successful / Failed requests** — must be 100/0

## Output
- Report: mode (standard/PD), all key metrics in a table
