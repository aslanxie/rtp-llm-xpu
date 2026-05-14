---
name: xpu-verify
description: 'Start rtp-llm-xpu service on Intel XPU and verify basic function with a chat completion request. Use when: testing service health, validating model output, smoke testing after build.'
---

# Verify rtp-llm-xpu Service

## Inputs (from .env)

- **WORK_DIR** — workspace path
- **MODEL_NAME** — model name (e.g., `Qwen3-8B`)
- **MODEL_TYPE** — model type (e.g., `qwen_3`)
- **MODEL_PATH** — checkpoint path (e.g., `/workspace/Qwen3-8B`)
- **TP_SIZE** — tensor parallelism size
- **ZE_AFFINITY_MASK** — XPU device mask. `0` = single GPU, `0,1` = PD disaggregation on 2 GPUs
- **FRONTEND_SERVER_COUNT** — number of frontend servers

## Determine Mode

Check `ZE_AFFINITY_MASK`:
- If it contains a comma (e.g., `0,1`): use **PD Mode** (Step 2B). The first device is PREFILL, the second is DECODE.
- If single value (e.g., `0`): use **Standard Mode** (Step 2A).

## Procedure

Follow these steps EXACTLY in order. Do NOT run any commands outside this procedure.

### 1. Check Existing Service

```
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
curl -sS --max-time 5 http://localhost:8088/health
```

- If response is `"ok"` or `{"status":"ok"}`: service is already running. Skip to **Step 3**.
- If connection refused or timeout: no service running. Continue to **Step 2**.

### 2. Kill Legacy Processes & Start Service

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
  --concurrency_limit 64 \
  --concurrency_with_block true \
  --warm_up 0 \
  2>&1 | tee /tmp/prefill.log
```

### 3. Wait for Health Check

Poll until healthy (max 180 seconds, check every 5 seconds). Use `--max-time 5` to prevent curl from blocking:
```
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
curl -sS --max-time 5 http://localhost:8088/health
```

For PD mode, also check:
```
curl -sS --max-time 5 http://localhost:9088/health
```

Expected response: `"ok"` or `{"status":"ok"}`

### 4. Function Test

```
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
curl -sS http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'$MODEL_NAME'",
    "messages": [{"role": "user", "content": "Hello, what is 2+2?"}],
    "max_tokens": 256
  }'
```

### 5. Validate Response

- Response must contain a valid JSON with `choices[0].message.content`
- The content must include a correct answer (4)
- Mark PASS if reasonable, FAIL if empty/error/wrong

## Output
- Report: mode (standard/PD), health check status, function test PASS/FAIL, response snippet
- Keep the service running for subsequent benchmark skills
