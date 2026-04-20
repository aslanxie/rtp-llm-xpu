"""Wrapper around vllm-xpu-kernels ops for use in rtp-llm.

Provides optimized SYCL/DPC++ kernels for Intel XPU via vllm-xpu-kernels.
Falls back to PyTorch native ops when vllm-xpu-kernels is not available.
"""

import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)

_VLLM_XPU_AVAILABLE = False
_FA2_AVAILABLE = False
_MOE_AVAILABLE = False

# If vllm-xpu-kernels is pip-installed, import works directly.
# Only use VLLM_XPU_KERNELS_PATH for development / editable installs.
_vllm_xpu_root = os.environ.get("VLLM_XPU_KERNELS_PATH", "")
if _vllm_xpu_root and os.path.isdir(_vllm_xpu_root) and _vllm_xpu_root not in sys.path:
    sys.path.insert(0, _vllm_xpu_root)

try:
    import vllm_xpu_kernels._C  # noqa: F401
    _VLLM_XPU_AVAILABLE = True
    logger.info("vllm-xpu-kernels _C loaded")
except ImportError as exc:
    logger.warning("vllm-xpu-kernels _C not available: %s", exc)

try:
    import vllm_xpu_kernels._vllm_fa2_C  # noqa: F401
    _FA2_AVAILABLE = True
    logger.info("vllm-xpu-kernels FA2 loaded")
except ImportError as exc:
    logger.warning("vllm-xpu-kernels FA2 not available: %s", exc)

try:
    import vllm_xpu_kernels._moe_C  # noqa: F401
    import vllm_xpu_kernels._xpu_C  # noqa: F401
    _MOE_AVAILABLE = True
    logger.info("vllm-xpu-kernels MoE loaded")
except ImportError as exc:
    logger.warning("vllm-xpu-kernels MoE not available: %s", exc)


def is_available():
    return _VLLM_XPU_AVAILABLE

def is_fa2_available():
    return _FA2_AVAILABLE

def is_moe_available():
    return _MOE_AVAILABLE


def rms_norm(result, input, weight, epsilon):
    if _VLLM_XPU_AVAILABLE:
        torch.ops._C.rms_norm(result, input, weight, epsilon)
    else:
        variance = input.pow(2).mean(-1, keepdim=True)
        normed = input * torch.rsqrt(variance + epsilon)
        result.copy_(normed * weight)


def fused_add_rms_norm(input, residual, weight, epsilon):
    if _VLLM_XPU_AVAILABLE:
        torch.ops._C.fused_add_rms_norm(input, residual, weight, epsilon)
    else:
        residual.add_(input)
        variance = residual.pow(2).mean(-1, keepdim=True)
        normed = residual * torch.rsqrt(variance + epsilon)
        input.copy_(normed * weight)


def silu_and_mul(out, input):
    if _VLLM_XPU_AVAILABLE:
        torch.ops._C.silu_and_mul(out, input)
    else:
        d = input.shape[-1] // 2
        x, gate = input[..., :d], input[..., d:]
        out.copy_(torch.nn.functional.silu(gate) * x)


def gelu_and_mul(out, input):
    if _VLLM_XPU_AVAILABLE:
        torch.ops._C.gelu_and_mul(out, input)
    else:
        d = input.shape[-1] // 2
        x, gate = input[..., :d], input[..., d:]
        out.copy_(torch.nn.functional.gelu(gate) * x)


def rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox=True):
    if _VLLM_XPU_AVAILABLE:
        torch.ops._C.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)
    else:
        pass  # fallback not needed if vllm-xpu-kernels available


def flash_attn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k,
                      max_seqlen_q, max_seqlen_k,
                      softmax_scale=None, causal=True,
                      block_table=None, seqused_k=None):
    if _FA2_AVAILABLE:
        from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
        return flash_attn_varlen_func(
            q, k, v,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_k=cu_seqlens_k,
            seqused_k=seqused_k,
            softmax_scale=softmax_scale,
            causal=causal,
            block_table=block_table,
        )
    else:
        return _sdpa_varlen_fallback(q, k, v, cu_seqlens_q, cu_seqlens_k,
                                     max_seqlen_q, max_seqlen_k, softmax_scale, causal)


def _sdpa_varlen_fallback(q, k, v, cu_seqlens_q, cu_seqlens_k,
                          max_seqlen_q, max_seqlen_k, softmax_scale, causal):
    import torch.nn.functional as F
    batch_size = cu_seqlens_q.numel() - 1
    outputs = []
    scale = softmax_scale or (q.shape[-1] ** -0.5)
    for i in range(batch_size):
        q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        k_start, k_end = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()
        qi = q[q_start:q_end].unsqueeze(0).transpose(1, 2)
        ki = k[k_start:k_end].unsqueeze(0).transpose(1, 2)
        vi = v[k_start:k_end].unsqueeze(0).transpose(1, 2)
        oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=causal, scale=scale)
        outputs.append(oi.transpose(1, 2).squeeze(0))
    if outputs:
        return torch.cat(outputs, dim=0)
    return q.new_empty(0, q.shape[1], q.shape[2])
