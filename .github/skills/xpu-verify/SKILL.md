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
- **ZE_AFFINITY_MASK** — XPU device mask
- **FRONTEND_SERVER_COUNT** — number of frontend servers

## When to Use
- After a successful build to validate the service works
- Smoke testing model inference on XPU

## Procedure

Follow these steps EXACTLY in order. Do NOT run any commands outside this procedure.

### 1. Check Existing Service

```
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
curl -sS --max-time 5 http://localhost:8088/health
```

- If response is `"ok"` or `{"status":"ok"}`: service is already running. Skip to **Step 4**.
- If connection refused or timeout: no service running. Continue to **Step 2**.

### 2. Kill Legacy Processes & Start Service

Kill any stale processes first:
```
pkill -9 -f 'rtp_llm/start_server.py' 2>/dev/null
pkill -9 -f 'rtp_llm_backend_server' 2>/dev/null
sleep 2
ps -ef | grep rtp_llm | grep -v grep || echo "Clean"
```

Then start the service in an **async terminal**:
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

### 3. Wait for Health Check

Poll until healthy (max 180 seconds, check every 5 seconds). Use `--max-time 5` to prevent curl from blocking:
```
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
curl -sS --max-time 5 http://localhost:8088/health
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
- Report: health check status, function test PASS/FAIL, response snippet
- Keep the service running for subsequent benchmark skills
