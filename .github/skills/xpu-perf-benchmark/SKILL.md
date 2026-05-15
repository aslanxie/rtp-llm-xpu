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

### 1. Ensure Service Is Ready

Use the **xpu-verify** skill to ensure the service is running with up-to-date code. It will automatically:
- Detect code changes or upstream merges and restart the service if needed
- Launch the service if not running
- Skip restart if the service is healthy and no changes detected

Pass `--think_mode 0` when launching (recommended for perf benchmarks to avoid long chain-of-thought generations that skew latency metrics).

Follow the xpu-verify skill procedure, then return here.

### 2. Run Benchmark

It is required to capture no less than 56 lines in the output of benchmark command to extract key metrics.

```
cd $WORK_DIR
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
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
  --metric-percentiles 25,50,75,90,95,99 \
  --endpoint "/v1/chat/completions" 2>&1 | tee ./logs/perf_benchmark.log | tail -60
```

For PD mode, consider `--max-concurrency 4` to test batched decode throughput.

IMPORTANT: You MUST use `tee ./logs/perf_benchmark.log | tail -60` to both persist the full log and capture the metrics output.

This takes ~15-20 minutes with 100 prompts at concurrency 1.

### 3. Extract Metrics

From the benchmark output, extract:
- **Output token throughput (tok/s)** — primary throughput metric
- **Median TPOT (ms)** — per-token decode latency
- **Median TTFT (ms)** — time to first token
- **P99 TPOT (ms)** — tail latency
- **Successful / Failed requests** — must be 100/0

## Output
- Report: mode (standard/PD), all key metrics in a table
