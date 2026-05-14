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
- **ZE_AFFINITY_MASK** — XPU device mask. `0` = single GPU, `0,1` = PD disaggregation on 2 GPUs
- **FRONTEND_SERVER_COUNT** — number of frontend servers

## Determine Mode

Check `ZE_AFFINITY_MASK`:
- If it contains a comma (e.g., `0,1`): use **PD Mode** (Step 2B). The first device is PREFILL, the second is DECODE.
- If single value (e.g., `0`): use **Standard Mode** (Step 2A).

## When to Use
- Validating model accuracy after sync or code changes
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

#### 2A. Standard Mode (single XPU)

Start in an **async terminal**:
```
cd $WORK_DIR
export PYTHONPATH=$(pwd):$PYTHONPATH
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_ENDPOINT="https://hf-mirror.com"
export no_proxy="localhost,127.0.0.1"
ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK FRONTEND_SERVER_COUNT=$FRONTEND_SERVER_COUNT \
python3 rtp_llm/start_server.py \
  --checkpoint_path $MODEL_PATH \
  --model_type $MODEL_TYPE \
  --tp_size $TP_SIZE \
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

Poll until healthy (max 180 seconds):
```
curl -sS --max-time 5 http://localhost:8088/health
curl -sS --max-time 5 http://localhost:9088/health
```

Expected response: `"ok"` or `{"status":"ok"}`

### 3. Run lm-eval

Set up environment:
```
export HF_ENDPOINT="https://hf-mirror.com"
export no_proxy="localhost,127.0.0.1"
```

For PD mode, use `num_concurrent=4` and `timeout=300` to stay within KV cache capacity:

```
cd $WORK_DIR
lm-eval --model local-chat-completions \
    --tasks gsm8k \
    --model_args "model=$MODEL_NAME,base_url=http://localhost:8088/v1/chat/completions,num_concurrent=4,max_retries=3,max_gen_toks=1024,timeout=300" \
    --apply_chat_template \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit 10
```

For standard mode, `num_concurrent` can be higher (e.g., 8).

This runs 10 GSM8K items with 5-shot prompting using greedy decoding. Takes ~5-10 minutes.

**Note:** Do NOT use `--gen_kwargs` with `max_new_tokens` — the `local-chat-completions` backend does not convert it to `max_tokens`. Use `max_gen_toks` in `--model_args` instead.

### 4. Extract Metrics

From the lm-eval output table, extract:
- **flexible-extract** score (0.0-1.0) — lenient answer extraction
- **strict-match** score (0.0-1.0) — exact match

Expected baseline for Qwen3-8B: ~0.7 or above on flexible-extract with 10 items.

## Output
- Report: mode (standard/PD), flexible-extract score, strict-match score, number of items evaluated
