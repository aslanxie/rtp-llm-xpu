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
- **ZE_AFFINITY_MASK** — XPU device mask
- **FRONTEND_SERVER_COUNT** — number of frontend servers
- **DATASET_PATH** — ShareGPT dataset path

## When to Use
- Measuring decode throughput and latency after build or optimization
- Comparing performance before/after code changes

## Procedure

Follow these steps EXACTLY in order. Do NOT run any commands outside this procedure.

### 1. Check Service Health

```
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
curl -sS --max-time 5 http://localhost:8088/health
```

- If response is `"ok"` or `{"status":"ok"}`: service is running. Skip to **Step 3**.
- If connection refused or timeout: no service running. Continue to **Step 2**.

### 2. Launch Service

Kill any stale processes first:
```
pkill -9 -f 'rtp_llm/start_server.py' 2>/dev/null
pkill -9 -f 'rtp_llm_backend_server' 2>/dev/null
sleep 2
ps -ef | grep rtp_llm | grep -v grep || echo "Clean"
```

Start the service in an **async terminal**:
```
cd $WORK_DIR
export PYTHONPATH=$(pwd):$PYTHONPATH
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK FRONTEND_SERVER_COUNT=$FRONTEND_SERVER_COUNT \
python3 rtp_llm/start_server.py \
  --checkpoint_path $MODEL_PATH \
  --model_type $MODEL_TYPE \
  --tp_size $TP_SIZE
```

Poll until healthy (max 180 seconds, check every 5 seconds). Use `--max-time 5` to prevent curl from blocking:
```
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
curl -sS --max-time 5 http://localhost:8088/health
```

Expected response: `"ok"` or `{"status":"ok"}`

### 3. Run Benchmark

It is required to capture no less than 56 lines in the output of benchmark command to extract key metrics.

```
cd $WORK_DIR
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

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
- Report all key metrics in a table
