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

        # Store K,V to per-request caches
        batch_caches = get_batch_kv_caches()
        if batch_caches is not None and len(batch_caches) > 1:
            # Batched prefill: split K,V by request and store separately
            input_lengths = self.attn_inputs.input_lengths
            if input_lengths is not None and input_lengths.numel() > 1:
                offsets = torch.cat([torch.zeros(1, dtype=torch.int32), input_lengths.cpu().cumsum(0)])
                for req_idx in range(len(batch_caches)):
                    start = int(offsets[req_idx])
                    end = int(offsets[req_idx + 1])
                    batch_caches[req_idx].store(layer_idx, k[start:end], v[start:end])
            else:
                # Fallback: single request or unknown split
                if batch_caches:
                    batch_caches[0].store(layer_idx, k, v)
        else:
            # Single-request path (backward compatible)
            xpu_cache = get_current_kv_cache()
            if xpu_cache is not None:
                xpu_cache.store(layer_idx, k, v)

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
        batch_caches = get_batch_kv_caches()
        if batch_caches is not None and len(batch_caches) > 1:
            # Detect mixed batch: decode_bs from sequence_lengths, rest are prefill
            seq_lens = self.attn_inputs.sequence_lengths
            try:
                decode_bs = seq_lens.numel() if seq_lens is not None else 0
            except (RuntimeError, AttributeError):
                decode_bs = 0
            context_bs = len(batch_caches) - decode_bs
            if context_bs > 0 and decode_bs > 0:
                return self._mixed_batch_forward(qkv, batch_caches, decode_bs, context_bs, layer_idx)
            return self._batched_decode(qkv, batch_caches, layer_idx)
        else:
            return self._single_decode(qkv, layer_idx)

    # Cached decode metadata tensors (created once, reused across layers)
    _cu_q_1: torch.Tensor = None  # [0, 1] for single-token decode
    _cu_k_buf: torch.Tensor = None  # reusable [0, kv_len] tensor

    def _single_decode(self, qkv, layer_idx):
        """Single-request decode (backward compatible)."""
        flash_attn_varlen = _get_flash_attn_varlen()
        new_tokens = qkv.shape[0]
        xpu_cache = get_current_kv_cache()

        # Compute position_ids for decode (only layer 0)
        if self.attn_inputs.position_ids is None and self.need_rope:
            cached_len = xpu_cache.get_seq_len(layer_idx) if xpu_cache is not None else 0
            self.attn_inputs.position_ids = torch.arange(
                cached_len, cached_len + new_tokens,
                dtype=torch.long, device=qkv.device,
            )

        q_new, k_new, v_new = _split_qkv_and_rope(
            qkv, self.attn_inputs, self.num_heads, self.num_kv_heads,
            self.head_dim, self.rope_config, self.need_rope,
        )

        if xpu_cache is not None:
            xpu_cache.store(layer_idx, k_new, v_new)
            k_full, v_full = xpu_cache.get(layer_idx)
        else:
            k_full, v_full = k_new, v_new

        kv_len = k_full.shape[0]

        # Cache cu_seqlens_q for single-token decode (always [0, 1])
        if new_tokens == 1:
            if XpuVllmDecodeImpl._cu_q_1 is None or XpuVllmDecodeImpl._cu_q_1.device != qkv.device:
                XpuVllmDecodeImpl._cu_q_1 = torch.tensor([0, 1], dtype=torch.int32, device=qkv.device)
            cu_q = XpuVllmDecodeImpl._cu_q_1
        else:
            cu_q = torch.tensor([0, new_tokens], dtype=torch.int32, device=qkv.device)
        cu_k = torch.tensor([0, kv_len], dtype=torch.int32, device=qkv.device)

        output = flash_attn_varlen(
            q_new, k_full, v_full,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=new_tokens, max_seqlen_k=kv_len,
            causal=False,
        )
        return output.reshape(new_tokens, -1)

    def _batched_decode(self, qkv, batch_caches, layer_idx):
        """Batched decode with per-request KV caches using flash_attn_varlen."""
        flash_attn_varlen = _get_flash_attn_varlen()

        num_requests = len(batch_caches)

        # Build position_ids for all requests based on their KV cache lengths
        if self.need_rope:
            positions = []
            for i, cache in enumerate(batch_caches):
                cached_len = cache.get_seq_len(layer_idx)
                positions.append(cached_len)
            position_ids = torch.tensor(positions, dtype=torch.long, device=qkv.device)
            # Override position_ids for RoPE
            self.attn_inputs.position_ids = position_ids

        q_new, k_new, v_new = _split_qkv_and_rope(
            qkv, self.attn_inputs, self.num_heads, self.num_kv_heads,
            self.head_dim, self.rope_config, self.need_rope,
        )

        # Store each request's new K,V to its own cache, then gather full KV
        k_parts = []
        v_parts = []
        kv_lens = []

        for i, cache in enumerate(batch_caches):
            # Store this request's new K,V (single token)
            cache.store(layer_idx, k_new[i:i+1], v_new[i:i+1])
            k_full_i, v_full_i = cache.get(layer_idx)
            k_parts.append(k_full_i)
            v_parts.append(v_full_i)
            kv_lens.append(k_full_i.shape[0])

        # Concatenate all KV for flash_attn_varlen
        k_all = torch.cat(k_parts, dim=0)  # [sum_kv_lens, kv_heads, head_dim]
        v_all = torch.cat(v_parts, dim=0)

        # Build cu_seqlens
        # Q: each request has exactly 1 query token
        cu_q = torch.arange(0, num_requests + 1, dtype=torch.int32, device=qkv.device)
        # K: variable lengths per request
        cu_k_list = [0]
        for l in kv_lens:
            cu_k_list.append(cu_k_list[-1] + l)
        cu_k = torch.tensor(cu_k_list, dtype=torch.int32, device=qkv.device)

        max_kv_len = max(kv_lens)

        if layer_idx == 0:
            logger.debug(f'[XPU-BatchDecode] n={num_requests} kv_lens={kv_lens}')

        output = flash_attn_varlen(
            q_new.contiguous(), k_all.contiguous(), v_all.contiguous(),
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=1, max_seqlen_k=max_kv_len,
            causal=False,
        )
        return output.reshape(num_requests, -1)

    def _mixed_batch_forward(self, qkv, batch_caches, decode_bs, context_bs, layer_idx):
        """Handle mixed prefill+decode batch by splitting and processing separately."""
        flash_attn_varlen = _get_flash_attn_varlen()

        input_lengths = self.attn_inputs.input_lengths
        decode_token_count = decode_bs

        decode_qkv = qkv[:decode_token_count]
        prefill_qkv = qkv[decode_token_count:]

        decode_caches = batch_caches[:decode_bs]
        prefill_caches = batch_caches[decode_bs:]

        if layer_idx == 0:
            logger.debug(
                f'[XPU-MixedBatch] decode_bs={decode_bs} context_bs={context_bs} '
                f'decode_tokens={decode_token_count} prefill_tokens={prefill_qkv.shape[0]}'
            )

        # --- Process decode requests ---
        if self.need_rope:
            decode_positions = []
            for cache in decode_caches:
                decode_positions.append(cache.get_seq_len(layer_idx))
            saved_pos = self.attn_inputs.position_ids
            self.attn_inputs.position_ids = torch.tensor(
                decode_positions, dtype=torch.long, device=qkv.device
            )

        q_dec, k_dec, v_dec = _split_qkv_and_rope(
            decode_qkv, self.attn_inputs, self.num_heads, self.num_kv_heads,
            self.head_dim, self.rope_config, self.need_rope,
        )

        # Store decode K,V and gather full KV
        k_parts, v_parts, kv_lens = [], [], []
        for i, cache in enumerate(decode_caches):
            cache.store(layer_idx, k_dec[i:i+1], v_dec[i:i+1])
            k_full_i, v_full_i = cache.get(layer_idx)
            k_parts.append(k_full_i)
            v_parts.append(v_full_i)
            kv_lens.append(k_full_i.shape[0])

        k_all_dec = torch.cat(k_parts, dim=0)
        v_all_dec = torch.cat(v_parts, dim=0)
        cu_q_dec = torch.arange(0, decode_bs + 1, dtype=torch.int32, device=qkv.device)
        cu_k_list = [0]
        for l in kv_lens:
            cu_k_list.append(cu_k_list[-1] + l)
        cu_k_dec = torch.tensor(cu_k_list, dtype=torch.int32, device=qkv.device)

        decode_output = flash_attn_varlen(
            q_dec.contiguous(), k_all_dec.contiguous(), v_all_dec.contiguous(),
            cu_seqlens_q=cu_q_dec, cu_seqlens_k=cu_k_dec,
            max_seqlen_q=1, max_seqlen_k=max(kv_lens),
            causal=False,
        )
        decode_output = decode_output.reshape(decode_bs, -1)

        # --- Process prefill requests ---
        prefill_input_lengths = input_lengths[decode_bs:]
        if self.need_rope:
            prefill_positions = []
            for i in range(context_bs):
                inp_len = int(prefill_input_lengths[i].item())
                prefill_positions.extend(range(inp_len))
            self.attn_inputs.position_ids = torch.tensor(
                prefill_positions, dtype=torch.long, device=qkv.device
            )

        q_pre, k_pre, v_pre = _split_qkv_and_rope(
            prefill_qkv, self.attn_inputs, self.num_heads, self.num_kv_heads,
            self.head_dim, self.rope_config, self.need_rope,
        )

        # Store prefill K,V to caches
        offsets = torch.cat([
            torch.zeros(1, dtype=torch.int32),
            prefill_input_lengths.cpu().cumsum(0)
        ])
        for i, cache in enumerate(prefill_caches):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            cache.store(layer_idx, k_pre[start:end], v_pre[start:end])

        # Build cu_seqlens for prefill (self-attention, causal)
        cu_pre = torch.zeros(context_bs + 1, dtype=torch.int32, device=qkv.device)
        cu_pre[1:] = prefill_input_lengths.to(device=qkv.device, dtype=torch.int32).cumsum(0)
        max_prefill_len = int(prefill_input_lengths.max().item())

        prefill_output = flash_attn_varlen(
            q_pre.contiguous(), k_pre.contiguous(), v_pre.contiguous(),
            cu_seqlens_q=cu_pre, cu_seqlens_k=cu_pre,
            max_seqlen_q=max_prefill_len, max_seqlen_k=max_prefill_len,
            causal=True,
        )
        prefill_output = prefill_output.reshape(prefill_qkv.shape[0], -1)

        # Restore position_ids
        if self.need_rope:
            self.attn_inputs.position_ids = saved_pos if saved_pos is not None else torch.tensor([], dtype=torch.long)

        return torch.cat([decode_output, prefill_output], dim=0)
