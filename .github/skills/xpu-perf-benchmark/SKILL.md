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
- **ZE_AFFINITY_MASK** — XPU device mask. `0` = single GPU, `0,1` = PD disaggregation on 2 GPUs
- **FRONTEND_SERVER_COUNT** — number of frontend servers
- **DATASET_PATH** — ShareGPT dataset path

## Determine Mode

Check `ZE_AFFINITY_MASK`:
- If it contains a comma (e.g., `0,1`): use **PD Mode** (Step 2B). The first device is PREFILL, the second is DECODE.
- If single value (e.g., `0`): use **Standard Mode** (Step 2A).

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

#### 2A. Standard Mode (single XPU)

Start in an **async terminal**:
```
cd $WORK_DIR
export PYTHONPATH=$(pwd):$PYTHONPATH
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK FRONTEND_SERVER_COUNT=$FRONTEND_SERVER_COUNT \
python3 rtp_llm/start_server.py \
  --checkpoint_path $MODEL_PATH \
  --model_type $MODEL_TYPE \
  --tp_size $TP_SIZE \
  --think_mode 0 \
  --concurrency_limit 16 \
  --warm_up 0 \
  2>&1 | tee /tmp/standard.log
```

#### 2B. PD Mode (2 XPUs — e.g., ZE_AFFINITY_MASK=0,1)

Parse device indices from ZE_AFFINITY_MASK: first value = PREFILL device, second = DECODE device.

**Shared env** (paste into BOTH terminals):
```
cd $WORK_DIR
export PYTHONPATH=$(pwd):$PYTHONPATH
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_ENDPOINT="https://hf-mirror.com"
export no_proxy="localhost,127.0.0.1"
export MODEL_SERVICE_CONFIG='{"service_id":"local","use_local":true,"role_endpoints":[{"group":"default","prefill_endpoint":{"type":"Vipserver","address":"127.0.0.1:8088","protocol":"http","path":"/"},"decode_endpoint":{"type":"Vipserver","address":"127.0.0.1:9088","protocol":"http","path":"/"}}]}'
```

**DECODE (start FIRST)** in async terminal — uses second device (e.g., `1`):
```
export REMOTE_RPC_SERVER_IP=localhost
export REMOTE_SERVER_PORT=8088
export START_PORT=9088
ZE_AFFINITY_MASK=1 FRONTEND_SERVER_COUNT=1 \
python3 rtp_llm/start_server.py \
  --checkpoint_path $MODEL_PATH \
  --model_type $MODEL_TYPE \
  --tp_size $TP_SIZE \
  --role_type DECODE \
  --cache_store_rdma_mode 0 \
  --seq_size_per_block 64 \
  --think_mode 0 \
  --concurrency_limit 16 \
  --max_context_batch_size 4 \
  --concurrency_with_block true \
  --warm_up 0 \
  2>&1 | tee /tmp/decode.log
```

Wait until `curl -sS --max-time 5 http://localhost:9088/health` returns ok.

**PREFILL (start SECOND)** in async terminal — uses first device (e.g., `0`):
```
export REMOTE_RPC_SERVER_IP=localhost
export REMOTE_SERVER_PORT=9088
export START_PORT=8088
ZE_AFFINITY_MASK=0 FRONTEND_SERVER_COUNT=1 \
python3 rtp_llm/start_server.py \
  --checkpoint_path $MODEL_PATH \
  --model_type $MODEL_TYPE \
  --tp_size $TP_SIZE \
  --role_type PREFILL \
  --cache_store_rdma_mode 0 \
  --seq_size_per_block 64 \
  --think_mode 0 \
  --concurrency_limit 64 \
  --concurrency_with_block true \
  --warm_up 0 \
  2>&1 | tee /tmp/prefill.log
```

Poll until healthy (max 180 seconds):
```
curl -sS --max-time 5 http://localhost:8088/health
curl -sS --max-time 5 http://localhost:9088/health
```

Expected response: `"ok"` or `{"status":"ok"}`

**Note:** `--think_mode 0` is recommended for perf benchmarks to avoid long chain-of-thought generations that skew latency metrics.

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
