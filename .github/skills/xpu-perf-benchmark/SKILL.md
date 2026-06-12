---
name: xpu-perf-benchmark
description: 'Run vllm serving benchmark on rtp-llm-xpu with ShareGPT dataset. Use when: measuring throughput, latency, TPOT, TTFT on Intel XPU. Requires running service on port 8088.'
---

# Performance Benchmark (vllm bench serve)

## Inputs (from .env)

- **WORK_DIR** — workspace path
- **MODEL_NAME** — model name
- **MODEL_TYPE** — model type (e.g., `qwen_3`)
- **MODEL_PATH** — tokenizer / checkpoint path (for Qwen3-8B benchmarking, use `/workspace/Qwen3-8B`, not `/workspace/Qwen3-8B-Base`)
- **TP_SIZE** — tensor parallelism size
- **ZE_AFFINITY_MASK** — XPU device mask. Single value (e.g., `0`) = single GPU. Two values (e.g., `0,1`) = PD disaggregation.
- **FRONTEND_SERVER_COUNT** — number of frontend servers
- **DATASET_PATH** — ShareGPT dataset path

## Overridable Parameters

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| `max_concurrency` | `--max-concurrency` | 1 (standard) / 4 (PD) | Max concurrent requests |
| `num_prompts` | `--num-prompts` | 100 | Number of prompts to benchmark |
| `request_rate` | `--request-rate` | inf | Request rate (requests/sec) |
| `output_len` | `--sharegpt-output-len` | 256 | Fixed output length per request |

When the orchestration prompt passes overrides, substitute them into the command below.

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

OUTPUT_LEN=${OUTPUT_LEN:-256}

vllm bench serve \
  --backend openai-chat \
  --model "$MODEL_NAME" \
  --tokenizer "$MODEL_PATH/" \
  --base-url http://localhost:8088 \
  --dataset-name sharegpt \
  --dataset-path $DATASET_PATH \
  --num-prompts ${NUM_PROMPTS:-100} \
  --request-rate ${REQUEST_RATE:-inf} \
  --max-concurrency ${MAX_CONCURRENCY:-1} \
  --seed 42 \
  --sharegpt-output-len $OUTPUT_LEN \
  --extra-body '{"temperature": 0, "max_tokens": '$OUTPUT_LEN', "extra_configs": {"ignore_eos": true, "min_new_tokens": '$OUTPUT_LEN'}}' \
  --metric-percentiles 25,50,75,90,95,99 \
  --endpoint "/v1/chat/completions" 2>&1 | tee ./logs/perf_benchmark.log | tail -60
```

For PD mode, default `max_concurrency` is 4 (unless overridden).

IMPORTANT: You MUST use `tee ./logs/perf_benchmark.log | tail -60` to both persist the full log and capture the metrics output.

#### Deterministic Token Counts

The `--extra-body` flags ensure every request generates exactly `OUTPUT_LEN` tokens so that throughput/latency comparisons across commits are apples-to-apples:

| extra-body key | Why it's needed |
|---|---|
| `"max_tokens": OUTPUT_LEN` | rtp-llm's `ChatCompletionRequest` has `max_tokens` but **not** `max_completion_tokens`. vllm bench sends `max_completion_tokens` natively which rtp-llm silently ignores. Passing `max_tokens` via `extra_body` ensures the server respects the limit. |
| `"temperature": 0` | Greedy decoding for reproducibility. |
| `"extra_configs": {"ignore_eos": true}` | Tells the C++ engine to ignore EOS tokens so output isn't truncated early. |
| `"extra_configs": {"min_new_tokens": OUTPUT_LEN}` | Ensures the Python renderer's `_check_finish_reason()` suppresses EOS/stop-word checks until `min_new_tokens` tokens have been generated, preventing early termination (e.g., at `<\|im_end\|>` after thinking). |

### 3. Extract Metrics

From the benchmark output, extract:
- **Total generated tokens** — must equal `num_prompts × OUTPUT_LEN` (e.g., 25600 for 100 × 256). If not, token counting is non-deterministic — investigate.
- **Output token throughput (tok/s)** — primary throughput metric
- **Median TPOT (ms)** — per-token decode latency
- **Median TTFT (ms)** — time to first token
- **P99 TPOT (ms)** — tail latency
- **Successful / Failed requests** — must be 100/0

## Output
- Report: mode (standard/PD), all key metrics in a table
