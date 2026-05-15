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

### 1. Ensure Service Is Ready

Use the **xpu-verify** skill to ensure the service is running with up-to-date code. It will automatically:
- Detect code changes or upstream merges and restart the service if needed
- Launch the service if not running
- Skip restart if the service is healthy and no changes detected

Pass `--think_mode 0` when launching. Thinking mode (`--think_mode 1`) consumes the entire generation budget on reasoning tokens, leaving no room for the actual answer — making lm-eval scores unreliable.

Follow the xpu-verify skill procedure, then return here.

### 2. Run lm-eval

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
    --model_args "model=$MODEL_NAME,base_url=http://localhost:8088/v1/chat/completions,num_concurrent=4,max_retries=3,max_length=4096,max_gen_toks=2048,timeout=300" \
    --apply_chat_template \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit 64
```

For standard mode, `num_concurrent` can be higher (e.g., 8).

This runs 64 GSM8K items with 5-shot prompting using greedy decoding.

**Note:** Do NOT use `--gen_kwargs` with `max_new_tokens` — the `local-chat-completions` backend does not convert it to `max_tokens`. Use `max_gen_toks` in `--model_args` instead.

### 3. Extract Metrics

From the lm-eval output table, extract:
- **flexible-extract** score (0.0-1.0) — lenient answer extraction
- **strict-match** score (0.0-1.0) — exact match

Expected baseline for Qwen3-8B: ~0.5 or above on flexible-extract with 64 items (5-shot, think_mode 0).

## Output
- Report: mode (standard/PD), flexible-extract score, strict-match score, number of items evaluated
