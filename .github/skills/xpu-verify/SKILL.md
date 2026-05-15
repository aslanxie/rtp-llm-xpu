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

### 1. Check Whether Service Restart Is Needed

Determine if the service must be (re)started by checking for code changes or upstream merges.

#### 1a. Check for code changes

```
cd $WORK_DIR
git diff --name-only HEAD~1 HEAD -- '*.py' '*.cc' '*.cpp' '*.h' '*.hpp' '*.bzl' 'BUILD' 'WORKSPACE'
```

If output is **non-empty**: source files changed — restart required. Go to **Step 2**.

#### 1b. Check for upstream merge

```
cd $WORK_DIR
git log -1 --format="%s" HEAD
```

If the latest commit message starts with `Merge` (e.g., `Merge remote-tracking branch 'upstream/main'`): upstream was merged — restart required. Go to **Step 2**.

#### 1c. Check existing service health

If neither 1a nor 1b triggered a restart:

```
export no_proxy="localhost,127.0.0.1"
curl -sS --max-time 5 http://localhost:8088/health
```

- If response is `"ok"` or `{"status":"ok"}`: service is running and no changes detected. Skip to **Step 3**.
- If connection refused or timeout: no service running. Continue to **Step 2**.

### 2. Launch Service

Use the **xpu-service** skill to launch the service. It will automatically select the correct mode based on `ZE_AFFINITY_MASK`:
- Single value (e.g., `0`) → **Standard Mode** (single GPU on port 8088)
- Two values (e.g., `0,1`) → **PD Mode** (DECODE on second device:9088, PREFILL on first device:8088)

The xpu-service skill handles killing stale processes before launching. Follow its procedure, then return here for verification.

### 3. Wait for Health Check

Poll until healthy (max 180 seconds, check every 5 seconds). Use `--max-time 5` to prevent curl from blocking:
```
export no_proxy="localhost,127.0.0.1"
curl -sS --max-time 5 http://localhost:8088/health
```

For PD mode (ZE_AFFINITY_MASK has two values), also check:
```
curl -sS --max-time 5 http://localhost:9088/health
```

Expected response: `"ok"` or `{"status":"ok"}`

### 4. Function Test

```
export no_proxy="localhost,127.0.0.1"
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
- Report: mode (standard/PD), health check status, whether service was restarted (and why: code change / upstream merge / not running), function test PASS/FAIL, response snippet
- Keep the service running for subsequent benchmark skills
