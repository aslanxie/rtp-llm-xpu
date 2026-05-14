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
- **ZE_AFFINITY_MASK** — XPU device mask. Single value (e.g., `0`) = single GPU mode. Two values (e.g., `0,1`) = PD disaggregation mode.
- **FRONTEND_SERVER_COUNT** — number of frontend servers

## Procedure

Follow these steps EXACTLY in order. Do NOT run any commands outside this procedure.

### 1. Check Existing Service

```
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
curl -sS --max-time 5 http://localhost:8088/health
```

- If response is `"ok"` or `{"status":"ok"}`: service is already running. Skip to **Step 3**.
- If connection refused or timeout: no service running. Continue to **Step 2**.

### 2. Launch Service

Use the **xpu-service** skill to launch the service. It will automatically select the correct mode based on `ZE_AFFINITY_MASK`:
- Single value (e.g., `0`) → **Standard Mode** (single GPU on port 8088)
- Two values (e.g., `0,1`) → **PD Mode** (DECODE on second device:9088, PREFILL on first device:8088)

Follow the xpu-service skill procedure, then return here for verification.

### 3. Wait for Health Check

Poll until healthy (max 180 seconds, check every 5 seconds). Use `--max-time 5` to prevent curl from blocking:
```
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
curl -sS --max-time 5 http://localhost:8088/health
```

For PD mode (ZE_AFFINITY_MASK has two values), also check:
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
