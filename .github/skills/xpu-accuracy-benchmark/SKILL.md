---
name: xpu-accuracy-benchmark
description: 'Run GSM8K accuracy evaluation on rtp-llm-xpu using lm-eval. Use when: testing model accuracy, validating math reasoning, checking correctness after changes. Requires running service on port 8088.'
---

# Accuracy Benchmark (GSM8K via lm-eval)

## Inputs (from .env)

- **WORK_DIR** — workspace path
- **MODEL_NAME** — model name (e.g., `Qwen3-8B`)
- **MODEL_TYPE** — model type (e.g., `qwen_3`)
- **MODEL_PATH** — checkpoint path (e.g., `/workspace/Qwen3-8B`)
- **TP_SIZE** — tensor parallelism size
- **ZE_AFFINITY_MASK** — XPU device mask
- **FRONTEND_SERVER_COUNT** — number of frontend servers

## When to Use
- Validating model accuracy after merge or code changes
- Checking that optimizations don't degrade correctness
- Math reasoning evaluation

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

### 3. Run lm-eval

```
cd $WORK_DIR
export HF_ENDPOINT="https://hf-mirror.com"
export no_proxy="localhost,127.0.0.1"

lm-eval --model local-chat-completions \
    --tasks gsm8k \
    --model_args "model=$MODEL_NAME,base_url=http://localhost:8088/v1/chat/completions,enable_thinking=True,think_end_token='</think>',max_gen_toks=1024" \
    --apply_chat_template \
    --num_fewshot 3 \
    --batch_size 1 \
    --limit 50
```

This runs 10 GSM8K items with 8-shot prompting. Takes ~5-10 minutes.

### 4. Extract Metrics

From the lm-eval output table, extract:
- **flexible-extract** score (0.0-1.0) — lenient answer extraction
- **strict-match** score (0.0-1.0) — exact match

Expected baseline for Qwen3-8B: ~0.7 or above on both metrics with 10 items.

## Output
- Report: flexible-extract score, strict-match score, number of items evaluated
