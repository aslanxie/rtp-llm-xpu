"""XPU Flash Attention with RoPE and KV Cache using vllm-xpu-kernels.

Supports batched multi-request decode and prefill for continuous batching.
Each request maintains its own pre-allocated KV cache. Uses flash_attn_varlen
to handle variable-length sequences in a single kernel call.

Optimization: KV cache uses pre-allocated tensors with index-based writes
instead of torch.cat, eliminating O(seq_len) copies per decode step.
"""

import logging
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

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

_DEFAULT_INITIAL_CAPACITY = 16


class XpuKVCache:
    """Per-request KV cache with pre-allocated storage.

    Uses a single [num_layers, capacity, num_kv_heads, head_dim] tensor pair
    for K and V. New tokens are written via index assignment (O(num_new))
    instead of torch.cat (O(seq_len)), eliminating redundant copies.

    The cache tracks two lengths:
    - seq_len: committed length (visible to position_id computation)
    - _end: write head (includes tokens stored in the current step)

    Call commit() after all layers have processed to advance seq_len.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int = 1,
        head_dim: int = 128,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
        initial_capacity: int = _DEFAULT_INITIAL_CAPACITY,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self._initial_capacity = initial_capacity
        self._capacity = 0
        self.seq_len = 0
        self._end = 0
        # Lazy allocation: created on first store() to avoid OOM at init
        self.k_cache = None
        self.v_cache = None

    def store(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Write new K/V tokens at current position. k,v: [num_new, kv_heads, dim]"""
        num_new = k.shape[0]
        end = self.seq_len + num_new
        if self.k_cache is None:
            # Lazy allocation on first store
            cap = max(self._initial_capacity, end)
            self.k_cache = torch.empty(
                self.num_layers, cap, self.num_kv_heads, self.head_dim,
                dtype=k.dtype, device=k.device,
            )
            self.v_cache = torch.empty(
                self.num_layers, cap, self.num_kv_heads, self.head_dim,
                dtype=v.dtype, device=v.device,
            )
            self._capacity = cap
            self.dtype = k.dtype
            self.device = k.device
        elif end > self._capacity:
            self._grow(end)
        self.k_cache[layer_idx, self.seq_len:end] = k
        self.v_cache[layer_idx, self.seq_len:end] = v
        self._end = end

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get full KV cache including tokens stored in current step."""
        return self.k_cache[layer_idx, :self._end], self.v_cache[layer_idx, :self._end]

    def get_seq_len(self, layer_idx: int = 0) -> int:
        """Return committed seq length (before current step's stores)."""
        return self.seq_len

    def commit(self):
        """Advance seq_len to include tokens stored in current step."""
        self.seq_len = self._end

    def reset(self):
        self.seq_len = 0
        self._end = 0
        self._capacity = 0
        self.k_cache = None
        self.v_cache = None

    def _grow(self, min_capacity: int):
        new_capacity = max(min_capacity, self._capacity * 2)
        new_k = torch.empty(
            self.num_layers, new_capacity, self.num_kv_heads, self.head_dim,
            dtype=self.dtype, device=self.device,
        )
        new_v = torch.empty(
            self.num_layers, new_capacity, self.num_kv_heads, self.head_dim,
            dtype=self.dtype, device=self.device,
        )
        if self.seq_len > 0:
            new_k[:, :self.seq_len] = self.k_cache[:, :self.seq_len]
            new_v[:, :self.seq_len] = self.v_cache[:, :self.seq_len]
        self.k_cache = new_k
        self.v_cache = new_v
        self._capacity = new_capacity
        logger.debug(f"[XPU-KV] Grew cache capacity to {new_capacity}")


class XpuKVCacheManager:
    """Manages multiple per-request XpuKVCaches for batched execution.

    Caches are matched by expected KV length. A decode request with
    sequence_length=S expects a cache with S tokens. After storing
    the new token and committing, the cache has S+1 tokens.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int = 1,
        head_dim: int = 128,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
        initial_capacity: int = _DEFAULT_INITIAL_CAPACITY,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.initial_capacity = initial_capacity
        # Map: kv_len -> deque of XpuKVCache
        self._caches: Dict[int, deque] = defaultdict(deque)

    def get_or_create_for_decode(self, expected_kv_len: int) -> XpuKVCache:
        """Get cache for a decode request. expected_kv_len = seq_length."""
        bucket = self._caches.get(expected_kv_len)
        if bucket and len(bucket) > 0:
            return bucket.popleft()
        logger.warning(f"No KV cache found for expected_kv_len={expected_kv_len}, creating new")
        return self._create_cache()

    def create_for_prefill(self) -> XpuKVCache:
        """Create a fresh cache for a prefill request."""
        return self._create_cache()

    def _create_cache(self) -> XpuKVCache:
        return XpuKVCache(
            self.num_layers, self.num_kv_heads, self.head_dim,
            self.dtype, self.device, self.initial_capacity,
        )

    def register(self, cache: XpuKVCache, kv_len: int):
        """Register cache after a step. kv_len = current KV length after commit."""
        self._caches[kv_len].append(cache)

    def num_active(self) -> int:
        return sum(len(b) for b in self._caches.values())


# ── Thread-local KV cache holders ───────────────────────────────────────
_tls = threading.local()

def set_current_kv_cache(kv_cache: Optional['XpuKVCache']):
    _tls.xpu_kv_cache = kv_cache

def get_current_kv_cache() -> Optional['XpuKVCache']:
    return getattr(_tls, 'xpu_kv_cache', None)

def set_current_kv_cache_manager(mgr: Optional['XpuKVCacheManager']):
    _tls.xpu_kv_cache_manager = mgr

def get_current_kv_cache_manager() -> Optional['XpuKVCacheManager']:
    return getattr(_tls, 'xpu_kv_cache_manager', None)

# Per-request caches for the current batch step
def set_batch_kv_caches(caches: Optional[List['XpuKVCache']]):
    _tls.batch_kv_caches = caches

def get_batch_kv_caches() -> Optional[List['XpuKVCache']]:
    return getattr(_tls, 'batch_kv_caches', None)


# ── RoPE ────────────────────────────────────────────────────────────────────

_COS_SIN_CACHE: Dict = {}


def _get_cos_sin_cache(rope_config, head_dim, max_pos, dtype, device):
    rotary_dim = getattr(rope_config, 'dim', 0) or head_dim
    base = getattr(rope_config, 'base', 10000.0) or 10000.0
    key = (base, rotary_dim, max_pos, dtype, str(device))
    if key in _COS_SIN_CACHE:
        return _COS_SIN_CACHE[key]
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
    max_pos = max(int(positions.max().item()) + 1, 4096)
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

    Uses block_ids to map sequence positions to cache blocks.
    block_ids_cpu should be a 1-D (or 2-D with 1 row) CPU tensor of block IDs.
    """
    tpb = kv_cache.seq_size_per_block
    cache = kv_cache.kv_cache_base  # [num_blocks, 2, kv_heads, tpb, head_dim]
    bids = block_ids_cpu.reshape(-1)
    N = k.shape[0]
    pos = 0
    while pos < N:
        abs_pos = start_pos + pos
        blk_slot = abs_pos // tpb
        offset = abs_pos % tpb
        n = min(tpb - offset, N - pos)
        if blk_slot >= bids.numel():
            break
        bid = int(bids[blk_slot])
        cache[bid, 0, :, offset:offset+n, :] = k[pos:pos+n].transpose(0, 1)
        cache[bid, 1, :, offset:offset+n, :] = v[pos:pos+n].transpose(0, 1)
        pos += n


def _read_from_paged_cache(kv_cache, block_ids_cpu, total_len, num_kv_heads, head_dim):
    """Read K,V [total_len, kv_heads, dim] from paged LayerKVCache.

    block_ids_cpu should be a 1-D (or 2-D with 1 row) CPU tensor of block IDs.
    """
    tpb = kv_cache.seq_size_per_block
    cache = kv_cache.kv_cache_base  # [num_blocks, 2, kv_heads, tpb, head_dim]
    bids = block_ids_cpu.reshape(-1)
    k_parts, v_parts = [], []
    remaining = total_len
    blk_slot = 0
    while remaining > 0:
        bid = int(bids[blk_slot])
        n = min(tpb, remaining)
        k_parts.append(cache[bid, 0, :, :n, :].transpose(0, 1).contiguous())
        v_parts.append(cache[bid, 1, :, :n, :].transpose(0, 1).contiguous())
        remaining -= n
        blk_slot += 1
    return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)


# ── Attention implementations ───────────────────────────────────────────────

class XpuVllmPrefillImpl(FMHAImplBase):
    """Prefill: full sequence attention, stores K/V to per-request XpuKVCache.

    Supports batched prefill with multiple requests. Each request gets its
    own KV cache via the batch_kv_caches thread-local list.
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
    """Decode: process new token(s), read K/V from per-request XpuKVCache.

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

        # Set position_ids for RoPE based on sequence lengths
        if self.need_rope:
            positions = []
            for i in range(num_requests):
                pos = int(seq_lengths[i].item()) if seq_lengths is not None else 0
                positions.append(pos)
            self.attn_inputs.position_ids = torch.tensor(
                positions, dtype=torch.long, device=qkv.device,
            )

        q_new, k_new, v_new = _split_qkv_and_rope(
            qkv, self.attn_inputs, self.num_heads, self.num_kv_heads,
            self.head_dim, self.rope_config, self.need_rope,
        )

        # Reshape block_ids to [num_requests, max_blocks]
        bids_2d_for_write = block_ids_all.reshape(num_requests, -1)

        # Write new K,V token to paged cache for each request
        kv_lens = []
        for i in range(num_requests):
            start_pos = int(seq_lengths[i].item()) if seq_lengths is not None else 0
            bids = bids_2d_for_write[i]
            _write_to_paged_cache(
                k_new[i:i+1], v_new[i:i+1], kv_cache, bids.cpu(), start_pos,
                self.num_kv_heads, self.head_dim,
            )
            kv_lens.append(start_pos + 1)

        # Gather only needed blocks, transpose to flash_attn paged format
        # Cache: [num_blocks, 2, kv_heads, tpb, head_dim]
        # flash_attn expects: [gathered_blocks, page_size, nheads_k, head_dim]
        cache = kv_cache.kv_cache_base
        tpb = kv_cache.seq_size_per_block


        # block_ids_all shape can be [batch, 1, max_blocks] or [batch, max_blocks] or [max_blocks]
        # Reshape to [num_requests, max_blocks]
        bids_2d = block_ids_all.reshape(num_requests, -1).cpu()
        max_blocks_needed = max((l + tpb - 1) // tpb for l in kv_lens)

        # Collect unique block IDs needed
        needed_bids = []
        for i in range(num_requests):
            n_blocks = (kv_lens[i] + tpb - 1) // tpb
            for j in range(n_blocks):
                needed_bids.append(int(bids_2d[i, j].item()))

        unique_bids = list(dict.fromkeys(needed_bids))
        bid_to_idx = {bid: idx for idx, bid in enumerate(unique_bids)}
        bid_tensor = torch.tensor(unique_bids, dtype=torch.long, device=cache.device)

        # Gather only needed blocks: [n_unique, 2, kv_heads, tpb, head_dim]
        gathered = cache[bid_tensor]
        # Transpose to flash_attn paged format: [n_unique, tpb, kv_heads, head_dim]
        k_cache = gathered[:, 0].transpose(1, 2).contiguous()
        v_cache = gathered[:, 1].transpose(1, 2).contiguous()

        # Build remapped block_table: [num_requests, max_blocks_needed]
        new_table = []
        for i in range(num_requests):
            n_blocks = (kv_lens[i] + tpb - 1) // tpb
            remapped = []
            for j in range(max_blocks_needed):
                if j < n_blocks:
                    remapped.append(bid_to_idx[int(bids_2d[i, j].item())])
                else:
                    remapped.append(0)
            new_table.append(remapped)
        block_table = torch.tensor(new_table, dtype=torch.int32, device=qkv.device)

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
