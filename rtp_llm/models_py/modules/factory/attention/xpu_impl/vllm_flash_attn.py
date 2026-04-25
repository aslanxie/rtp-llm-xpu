"""XPU Flash Attention with RoPE and KV Cache using vllm-xpu-kernels.

Supports batched multi-request decode and prefill for continuous batching.
Uses the framework's LayerKVCache for paged KV storage. Uses flash_attn_varlen
to handle variable-length sequences in a single kernel call.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs

logger = logging.getLogger(__name__)

# Module-level lazy import for flash attention (avoids per-call import overhead)
_flash_attn_varlen = None
def _get_flash_attn_varlen():
    global _flash_attn_varlen
    if _flash_attn_varlen is None:
        from rtp_llm.models_py.modules.base.xpu.vllm_xpu_ops import flash_attn_varlen
        _flash_attn_varlen = flash_attn_varlen
    return _flash_attn_varlen


# ── RoPE ────────────────────────────────────────────────────────────────────

_COS_SIN_CACHE: Dict = {}
_COS_SIN_CACHE_MAX_SIZE = 32


def _get_cos_sin_cache(rope_config, head_dim, max_pos, dtype, device):
    rotary_dim = getattr(rope_config, 'dim', 0) or head_dim
    base = getattr(rope_config, 'base', 10000.0) or 10000.0
    key = (base, rotary_dim, max_pos, dtype, str(device))
    if key in _COS_SIN_CACHE:
        return _COS_SIN_CACHE[key]
    # Evict oldest entries if cache is full
    if len(_COS_SIN_CACHE) >= _COS_SIN_CACHE_MAX_SIZE:
        oldest_key = next(iter(_COS_SIN_CACHE))
        del _COS_SIN_CACHE[oldest_key]
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    cache = cache.to(dtype=dtype, device=device)
    _COS_SIN_CACHE[key] = cache
    return cache


def _apply_rotary_emb_neox(x, cos, sin):
    d2 = cos.shape[-1]
    x1, x2 = x[..., :d2], x[..., d2:2*d2]
    x_pass = x[..., 2*d2:]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2, x_pass), dim=-1)


def _need_rope(attn_configs):
    if getattr(attn_configs, 'need_rope_kv_cache', False):
        return True
    return getattr(attn_configs.rope_config, 'style', 0) != 0


def _apply_rope(q, k, positions, rope_config, head_dim, num_heads, num_kv_heads, device, dtype):
    rotary_dim = getattr(rope_config, 'dim', 0) or head_dim
    is_neox = getattr(rope_config, 'is_neox_style', True)
    raw_max = max(int(positions.max().item()) + 1, 4096)
    # Round up to next power-of-2 to reduce unique cache entries
    max_pos = 1 << (raw_max - 1).bit_length()
    cos_sin_cache = _get_cos_sin_cache(rope_config, head_dim, max_pos, dtype, device)
    try:
        from rtp_llm.models_py.modules.base.xpu.vllm_xpu_ops import rotary_embedding as vllm_rope
        q_c, k_c = q.contiguous(), k.contiguous()
        vllm_rope(positions, q_c, k_c, head_dim, cos_sin_cache, is_neox)
        return q_c, k_c
    except Exception:
        pass
    num_tokens = q.shape[0]
    cos_sin = cos_sin_cache[positions.long()]
    half = cos_sin.shape[-1] // 2
    cos = cos_sin[:, :half].unsqueeze(1)
    sin = cos_sin[:, half:].unsqueeze(1)
    q_r = q.view(num_tokens, num_heads, head_dim)
    k_r = k.view(num_tokens, num_kv_heads, head_dim)
    q_r = _apply_rotary_emb_neox(q_r, cos, sin)
    k_r = _apply_rotary_emb_neox(k_r, cos, sin)
    return q_r.reshape(num_tokens, -1), k_r.reshape(num_tokens, -1)


def _split_qkv_and_rope(qkv, attn_inputs, num_heads, num_kv_heads, head_dim, rope_config, need_rope):
    """Split QKV tensor and apply RoPE. Returns q, k, v as [tokens, heads, dim]."""
    total_tokens = qkv.shape[0]
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    if need_rope:
        positions = attn_inputs.position_ids
        if positions is None:
            positions = torch.arange(total_tokens, dtype=torch.long, device=qkv.device)
        q_flat = qkv[:, :q_size].contiguous()
        k_flat = qkv[:, q_size:q_size + kv_size].contiguous()
        q_flat, k_flat = _apply_rope(
            q_flat, k_flat, positions, rope_config, head_dim,
            num_heads, num_kv_heads, qkv.device, qkv.dtype,
        )
        q = q_flat.view(total_tokens, num_heads, head_dim)
        k = k_flat.view(total_tokens, num_kv_heads, head_dim)
    else:
        q = qkv[:, :q_size].view(total_tokens, num_heads, head_dim)
        k = qkv[:, q_size:q_size + kv_size].view(total_tokens, num_kv_heads, head_dim)

    v = qkv[:, q_size + kv_size:].view(total_tokens, num_kv_heads, head_dim)
    return q, k, v


# ── Paged KV cache helpers ──────────────────────────────────────────────

def _write_to_paged_cache(k, v, kv_cache, block_ids_cpu, start_pos, num_kv_heads, head_dim):
    """Write k,v [N, kv_heads, dim] to paged LayerKVCache.

    Vectorized: computes block/offset mapping for all N tokens at once
    and writes via advanced indexing instead of a Python while-loop.
    """
    tpb = kv_cache.seq_size_per_block
    cache = kv_cache.kv_cache_base  # [num_blocks, 2, kv_heads, tpb, head_dim]
    bids = block_ids_cpu.reshape(-1)
    N = k.shape[0]
    if N == 0:
        return
    # Compute block slot and offset for each token
    abs_positions = torch.arange(start_pos, start_pos + N, dtype=torch.long)
    blk_slots = abs_positions // tpb
    offsets = abs_positions % tpb
    # Clamp to available block IDs
    valid_mask = blk_slots < bids.numel()
    if not valid_mask.all():
        blk_slots = blk_slots[valid_mask]
        offsets = offsets[valid_mask]
        k = k[:valid_mask.sum()]
        v = v[:valid_mask.sum()]
    block_indices = bids[blk_slots].long().to(cache.device)
    offsets = offsets.to(cache.device)
    # k shape: [N, kv_heads, dim] -> write to cache[block_indices, 0, :, offsets, :]
    cache[block_indices, 0, :, offsets, :] = k
    cache[block_indices, 1, :, offsets, :] = v


def _read_from_paged_cache(kv_cache, block_ids_cpu, total_len, num_kv_heads, head_dim):
    """Read K,V [total_len, kv_heads, dim] from paged LayerKVCache.

    Vectorized: computes block/offset mapping for all positions at once
    and gathers via advanced indexing instead of a Python while-loop.
    """
    tpb = kv_cache.seq_size_per_block
    cache = kv_cache.kv_cache_base  # [num_blocks, 2, kv_heads, tpb, head_dim]
    bids = block_ids_cpu.reshape(-1)
    if total_len == 0:
        return cache.new_empty(0, num_kv_heads, head_dim), cache.new_empty(0, num_kv_heads, head_dim)
    # Compute block slot and offset for each position
    positions = torch.arange(total_len, dtype=torch.long)
    blk_slots = positions // tpb
    offsets = positions % tpb
    block_indices = bids[blk_slots].long().to(cache.device)
    offsets_dev = offsets.to(cache.device)
    # Gather: cache[block_indices, 0/1, :, offsets, :] -> [N, kv_heads, dim]
    k = cache[block_indices, 0, :, offsets_dev, :].contiguous()
    v = cache[block_indices, 1, :, offsets_dev, :].contiguous()
    return k, v


# ── Attention implementations ───────────────────────────────────────────────

class XpuVllmPrefillImpl(FMHAImplBase):
    """Prefill: full sequence attention, stores K/V to framework\'s LayerKVCache."""

    def __init__(self, attn_configs, attn_inputs, parallelism_config=None):
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.rope_config = attn_configs.rope_config
        self.need_rope = _need_rope(attn_configs)
        self.fmha_params = None

    @staticmethod
    def support(attn_configs, attn_inputs):
        return attn_inputs.is_prefill

    def forward(self, qkv, kv_cache=None, layer_idx=0):
        flash_attn_varlen = _get_flash_attn_varlen()
        total_tokens = qkv.shape[0]
        q, k, v = _split_qkv_and_rope(
            qkv, self.attn_inputs, self.num_heads, self.num_kv_heads,
            self.head_dim, self.rope_config, self.need_rope,
        )

        # Write K,V to paged LayerKVCache for future decode steps
        if kv_cache is not None:
            block_ids_all = self.attn_inputs.kv_cache_block_id_device
            if block_ids_all is None:
                block_ids_all = self.attn_inputs.kv_cache_block_id_host
            if block_ids_all is not None and block_ids_all.numel() > 0:
                input_lengths = self.attn_inputs.input_lengths
                if input_lengths is not None and input_lengths.numel() > 1:
                    # Batched prefill: write each request separately
                    offsets = torch.cat([torch.zeros(1, dtype=torch.int32), input_lengths.cpu().cumsum(0)])
                    for req_idx in range(input_lengths.numel()):
                        start = int(offsets[req_idx])
                        end = int(offsets[req_idx + 1])
                        bids = block_ids_all[req_idx].cpu()
                        _write_to_paged_cache(
                            k[start:end], v[start:end], kv_cache, bids, 0,
                            self.num_kv_heads, self.head_dim,
                        )
                else:
                    bids = block_ids_all[0].cpu()
                    _write_to_paged_cache(k, v, kv_cache, bids, 0,
                                          self.num_kv_heads, self.head_dim)

        cu_seqlens = self.attn_inputs.cu_seqlens
        if cu_seqlens is None or cu_seqlens.numel() <= 1:
            cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.int32, device=qkv.device)
        else:
            cu_seqlens = cu_seqlens.to(device=qkv.device, dtype=torch.int32)
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

        output = flash_attn_varlen(
            q.contiguous(), k.contiguous(), v.contiguous(),
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            causal=True,
        )
        return output.reshape(total_tokens, -1)


class XpuVllmDecodeImpl(FMHAImplBase):
    """Decode: process new token(s), read K/V from framework\'s LayerKVCache.

    Supports batched decode with multiple requests. Uses flash_attn_varlen
    with per-request cu_seqlens to handle different KV lengths.
    """

    def __init__(self, attn_configs, attn_inputs, parallelism_config=None):
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.rope_config = attn_configs.rope_config
        self.need_rope = _need_rope(attn_configs)
        self.fmha_params = None

    @staticmethod
    def support(attn_configs, attn_inputs):
        return not attn_inputs.is_prefill

    def forward(self, qkv, kv_cache=None, layer_idx=0):
        if kv_cache is not None:
            return self._paged_decode(qkv, kv_cache, layer_idx)
        # Fallback: no paged cache, self-attend over current tokens
        flash_attn_varlen = _get_flash_attn_varlen()
        q, k, v = _split_qkv_and_rope(
            qkv, self.attn_inputs, self.num_heads, self.num_kv_heads,
            self.head_dim, self.rope_config, self.need_rope,
        )
        N = qkv.shape[0]
        cu = torch.tensor([0, N], dtype=torch.int32, device=qkv.device)
        output = flash_attn_varlen(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                    max_seqlen_q=N, max_seqlen_k=N, causal=True)
        return output.reshape(N, -1)

    def _paged_decode(self, qkv, kv_cache, layer_idx):
        """Decode using paged LayerKVCache with block_table support.

        Uses flash_attn_varlen block_table parameter so the kernel reads
        K,V directly from the paged cache — no manual gather needed.
        Vectorized: position_ids, KV writes, block gather, and table remap
        all use tensor ops instead of Python for-loops.
        """
        flash_attn_varlen = _get_flash_attn_varlen()
        seq_lengths = self.attn_inputs.sequence_lengths
        block_ids_all = self.attn_inputs.kv_cache_block_id_device
        if block_ids_all is None:
            block_ids_all = self.attn_inputs.kv_cache_block_id_host

        try:
            num_requests = seq_lengths.numel() if seq_lengths is not None else 0
        except (RuntimeError, AttributeError):
            num_requests = 0
        if num_requests == 0:
            num_requests = 1

        # Set position_ids for RoPE based on sequence lengths (vectorized)
        if self.need_rope:
            self.attn_inputs.position_ids = seq_lengths.to(
                dtype=torch.long, device=qkv.device,
            ) if seq_lengths is not None else torch.zeros(
                num_requests, dtype=torch.long, device=qkv.device,
            )

        q_new, k_new, v_new = _split_qkv_and_rope(
            qkv, self.attn_inputs, self.num_heads, self.num_kv_heads,
            self.head_dim, self.rope_config, self.need_rope,
        )

        # Reshape block_ids to [num_requests, max_blocks]
        bids_2d = block_ids_all.reshape(num_requests, -1)

        # Write new K,V token to paged cache for each request
        # (each request writes 1 token, so per-request call is unavoidable
        #  since tokens go to different block positions)
        seq_lens_cpu = seq_lengths.cpu() if seq_lengths is not None else torch.zeros(num_requests, dtype=torch.long)
        kv_lens = (seq_lens_cpu + 1).tolist()
        for i in range(num_requests):
            _write_to_paged_cache(
                k_new[i:i+1], v_new[i:i+1], kv_cache, bids_2d[i].cpu(),
                int(seq_lens_cpu[i].item()), self.num_kv_heads, self.head_dim,
            )

        # Gather needed blocks and build block_table (vectorized)
        cache = kv_cache.kv_cache_base
        tpb = kv_cache.seq_size_per_block

        bids_2d_cpu = bids_2d.cpu()
        kv_lens_t = torch.tensor(kv_lens, dtype=torch.long)
        n_blocks_per_req = (kv_lens_t + tpb - 1) // tpb
        max_blocks_needed = int(n_blocks_per_req.max().item())

        # Truncate to needed columns and collect unique block IDs
        needed_bids_2d = bids_2d_cpu[:, :max_blocks_needed]
        # Mask out padding blocks
        col_idx = torch.arange(max_blocks_needed).unsqueeze(0)
        valid_mask = col_idx < n_blocks_per_req.unsqueeze(1)
        needed_flat = needed_bids_2d[valid_mask].long()

        # Unique block IDs with mapping
        unique_bids, inverse = needed_flat.unique(return_inverse=True)
        bid_tensor = unique_bids.to(cache.device)

        # Gather needed blocks: [n_unique, 2, kv_heads, tpb, head_dim]
        gathered = cache[bid_tensor]
        k_cache = gathered[:, 0].transpose(1, 2).contiguous()
        v_cache = gathered[:, 1].transpose(1, 2).contiguous()

        # Build remapped block_table using the inverse mapping
        block_table_flat = torch.zeros(num_requests * max_blocks_needed, dtype=torch.int32)
        block_table_flat[valid_mask.reshape(-1)] = inverse.int()
        block_table = block_table_flat.reshape(num_requests, max_blocks_needed).to(qkv.device)

        max_kv_len = max(kv_lens)
        seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=qkv.device)
        cu_q = torch.arange(0, num_requests + 1, dtype=torch.int32, device=qkv.device)

        output = flash_attn_varlen(
            q_new.contiguous(),
            k_cache,
            v_cache,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=None,
            max_seqlen_q=1,
            max_seqlen_k=max_kv_len,
            causal=False,
            block_table=block_table,
            seqused_k=seqused_k,
        )
        return output.reshape(num_requests, -1)

# Aliases for upstream compatibility
XpuVllmFlashAttnPrefillImpl = XpuVllmPrefillImpl
XpuVllmFlashAttnDecodeImpl = XpuVllmDecodeImpl
