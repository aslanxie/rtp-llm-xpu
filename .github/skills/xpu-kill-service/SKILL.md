---
name: xpu-kill-service
description: 'Kill running rtp-llm-xpu service processes. Use when: stopping service before restart, freeing XPU resources, cleaning up after benchmarks.'
---

# Kill rtp-llm-xpu Service

## When to Use
- Before restarting the service with new config or code changes
- Freeing XPU device memory after benchmarks
- Cleaning up stale service processes

## Procedure

### 1. Find and Kill Service Processes

Kill all rtp_llm related processes (start_server, backend, frontend):

```bash
pkill -f "rtp_llm/start_server.py" 2>/dev/null
pkill -f "rtp_llm_backend_server" 2>/dev/null
pkill -f "rtp_llm_frontend_server" 2>/dev/null
```

### 2. Verify No Processes Remain

```bash
ps aux | grep -E "rtp_llm|start_server" | grep -v grep
```

If processes still exist, force kill:

```bash
pkill -9 -f "rtp_llm/start_server.py" 2>/dev/null
pkill -9 -f "rtp_llm_backend_server" 2>/dev/null
pkill -9 -f "rtp_llm_frontend_server" 2>/dev/null
```

### 3. Confirm Ports Are Free

```bash
ss -tlnp | grep -E ':(8088|9088) '
```

Expected: no output (ports released).

If ports are still bound, wait a few seconds and recheck — the OS may need time to release them.

## Output
- Report whether processes were found and killed
- Confirm ports 8088 and 9088 are free
