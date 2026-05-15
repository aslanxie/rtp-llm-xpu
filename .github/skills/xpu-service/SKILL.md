---
name: xpu-service
description: 'Launch rtp-llm-xpu service in standard or PD-disaggregated mode on Intel XPU, with optional thinking mode. Use when: starting service for verify/benchmark/accuracy tests, switching between single-GPU and dual-GPU PD setups, toggling Qwen3 thinking mode.'
---

# rtp-llm-xpu Service Launch

## Inputs (from .env)

- **WORK_DIR** — workspace path (default: `/workspace/rtp-llm-xpu`)
- **MODEL_NAME** — e.g., `Qwen3-8B`
- **MODEL_TYPE** — e.g., `qwen_3`
- **MODEL_PATH** — checkpoint path, e.g., `/workspace/Qwen3-8B`
- **TP_SIZE** — tensor parallelism (default `1`)
- **ZE_AFFINITY_MASK** — single value (e.g., `0`) = single GPU, two comma-separated values (e.g., `2,3`) = PD disaggregation
- **FRONTEND_SERVER_COUNT** — number of frontend servers (default `1`)

## Determine Mode

Parse `ZE_AFFINITY_MASK`:
- If it contains a comma (e.g., `2,3`): **PD Mode**. Split on comma — first value = PREFILL device, second value = DECODE device.
- If single value (e.g., `0`): **Standard Mode**.

**Important:** Use the actual device IDs from `ZE_AFFINITY_MASK`. Do NOT hardcode `0` and `1`.
For example, if `ZE_AFFINITY_MASK=2,3`, then PREFILL runs on device `2` and DECODE on device `3`.

## When to Use
- Starting service before accuracy / perf benchmark
- Switching between single-GPU and PD disaggregation
- Toggling thinking mode for Qwen3

## Common Setup (run once per shell)

```
cd $WORK_DIR
export PYTHONPATH=$(pwd):$PYTHONPATH
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

## Kill Stale Processes (before any new launch)

```
pkill -9 -f 'rtp_llm/start_server.py' 2>/dev/null
pkill -9 -f 'rtp_llm_backend_server' 2>/dev/null
sleep 2
ps -ef | grep rtp_llm | grep -v grep || echo "Clean"
```

---

## Mode A: Standard (single XPU, e.g., ZE_AFFINITY_MASK=0)

Launch on the specified XPU, client port 8088. Run in an **async terminal**:

```
ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK FRONTEND_SERVER_COUNT=$FRONTEND_SERVER_COUNT \
python3 rtp_llm/start_server.py \
  --checkpoint_path $MODEL_PATH \
  --model_type $MODEL_TYPE \
  --tp_size $TP_SIZE \
  --seq_size_per_block 64 \
  --concurrency_limit 16 \
  --warm_up 0 \
  2>&1 | tee ./logs/standard.log
```

Health: `curl -sS --max-time 5 http://localhost:8088/health`

---

## Mode B: PD Disaggregation (2 XPUs, e.g., ZE_AFFINITY_MASK=2,3)

Parse device IDs: split `ZE_AFFINITY_MASK` on comma → `PREFILL_DEVICE` (first), `DECODE_DEVICE` (second).
Example: `ZE_AFFINITY_MASK=2,3` → `PREFILL_DEVICE=2`, `DECODE_DEVICE=3`.

Launch DECODE **first**, then PREFILL. Each in a separate **async terminal**.

### Shared env (paste into BOTH terminals)

```
cd $WORK_DIR
export PYTHONPATH=$(pwd):$PYTHONPATH
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="localhost,127.0.0.1"
export MODEL_SERVICE_CONFIG='{"service_id":"local","use_local":true,"role_endpoints":[{"group":"default","prefill_endpoint":{"type":"Vipserver","address":"127.0.0.1:8088","protocol":"http","path":"/"},"decode_endpoint":{"type":"Vipserver","address":"127.0.0.1:9088","protocol":"http","path":"/"}}]}'
```

### B1. DECODE server (DECODE_DEVICE, port 9088) — START FIRST

```
export REMOTE_RPC_SERVER_IP=localhost
export REMOTE_SERVER_PORT=8088
export START_PORT=9088
ZE_AFFINITY_MASK=$DECODE_DEVICE FRONTEND_SERVER_COUNT=1 \
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
  2>&1 | tee ./logs/decode.log
```

Wait until `curl -sS --max-time 5 http://localhost:9088/health` returns ok.

### B2. PREFILL server (PREFILL_DEVICE, port 8088) — START SECOND

```
export REMOTE_RPC_SERVER_IP=localhost
export REMOTE_SERVER_PORT=9088
export START_PORT=8088
ZE_AFFINITY_MASK=$PREFILL_DEVICE FRONTEND_SERVER_COUNT=1 \
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
  2>&1 | tee ./logs/prefill.log
```

Client always talks to port **8088** (PREFILL endpoint).

---

## Thinking Mode (Qwen3 / reasoning models)

Qwen3 has chain-of-thought thinking via `<think>...</think>`. Generations are much longer in thinking mode.

### Enable (default for Qwen3)

No extra flags needed. For accuracy eval with thinking, use `--num_fewshot 0` in lm-eval.

### Disable (recommended for perf bench and few-shot accuracy eval)

Add to the launch command (apply to STANDARD or to BOTH PD servers):
```
  --think_mode 0
```

For lm-eval, use `--num_fewshot 5` when thinking is disabled.

### Per-request control (no server restart)

POST body for `/v1/chat/completions`:
```json
{
  "model": "Qwen3-8B",
  "messages": [...],
  "extra_body": {"chat_config": {"enable_thinking": false}}
}
```

---

## Key Server Args Reference

| Arg | Purpose | Default | Notes |
|-----|---------|---------|-------|
| `--role_type` | `PREFILL` / `DECODE` / unset | unset | Required for PD |
| `--seq_size_per_block` | KV cache page size | 64 | Smaller → less fragmentation |
| `--concurrency_limit` | Max parallel requests | 8 | Raise to 16/64 for PD |
| `--max_context_batch_size` | Max concurrent prefills | (auto) | Lower for tight KV |
| `--concurrency_with_block` | Block-aware admission | false | Use `true` for PD |
| `--cache_store_rdma_mode` | 0=TCP, 1=RDMA | 0 | TCP for local PD |
| `--think_mode` | `0` = off, `1` = on | 0 | Qwen3 thinking |
| `--warm_up` | Warmup iters | 1 | Use 0 for faster startup |

## Health Check

```
curl -sS --max-time 5 http://localhost:8088/health   # PREFILL or STANDARD
curl -sS --max-time 5 http://localhost:9088/health   # DECODE (PD only)
```

## Smoke Test

```
curl -sS http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'$MODEL_NAME'",
    "messages": [{"role":"user","content":"What is 2+2?"}],
    "max_tokens": 64
  }'
```

## Output
- Report: which mode launched (standard/PD), device IDs used, both ports healthy (if PD), smoke test PASS/FAIL
- Leave servers running for subsequent benchmark / accuracy skills
