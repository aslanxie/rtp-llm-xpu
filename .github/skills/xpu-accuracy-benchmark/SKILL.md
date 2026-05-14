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
- **ZE_AFFINITY_MASK** — XPU device mask. Single value (e.g., `0`) = single GPU. Two values (e.g., `0,1`) = PD disaggregation.
- **FRONTEND_SERVER_COUNT** — number of frontend servers

## When to Use
- Validating model accuracy after sync or code changes
- Checking that optimizations don't degrade correctness
- Math reasoning evaluation

## Procedure

Follow these steps EXACTLY in order. Do NOT run any commands outside this procedure.

### 1. Check Service Health

```
export no_proxy="localhost,127.0.0.1"
curl -sS --max-time 5 http://localhost:8088/health
```

- If response is `"ok"` or `{"status":"ok"}`: service is running. Skip to **Step 3**.
- If connection refused or timeout: no service running. Continue to **Step 2**.

### 2. Launch Service

Use the **xpu-service** skill to launch the service with `--think_mode 0`. It will automatically select the correct mode based on `ZE_AFFINITY_MASK`:
- Single value (e.g., `0`) → **Standard Mode** (single GPU on port 8088)
- Two values (e.g., `0,1`) → **PD Mode** (DECODE on second device:9088, PREFILL on first device:8088)

**Important:** Always use `--think_mode 0` for lm-eval. Thinking mode (`--think_mode 1`) consumes the entire generation budget on reasoning tokens, leaving no room for the actual answer — making lm-eval scores unreliable.

Follow the xpu-service skill procedure, then return here.

### 3. Run lm-eval

Set up environment:
```
export HF_ENDPOINT="https://hf-mirror.com"
export no_proxy="localhost,127.0.0.1"
```

For PD mode, use `num_concurrent=4` and `timeout=300` to stay within KV cache capacity:

```
cd $WORK_DIR
lm_eval --model local-chat-completions \
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

Expected baseline for Qwen3-8B: ~0.5 or above on flexible-extract with 16 items (PD mode, 5-shot).

## Output
- Report: mode (standard/PD), flexible-extract score, strict-match score, number of items evaluated
