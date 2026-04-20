from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FMHAImplBase,
    RMSNorm,
    RMSResNorm,
)
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import DeviceType, LayerKVCache, PyModelInputs, PyModelOutputs, get_exec_ctx
from rtp_llm.utils.model_weight import W


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        layer_idx: int,
        weights: Dict[str, torch.Tensor],
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        attn_configs = config.getAttentionConfigs(parallelism_config.get_attn_tp_size())
        self.self_attn = CausalAttention(
            attn_configs,
            parallelism_config,
            weights,
            config.layernorm_eps,
            quant_config,
            hw_kernel_config,
            layer_idx,
        )
        self.mlp = DenseMLP(
            config.activation_type,
            parallelism_config,
            weights,
            quant_config,
            hw_kernel_config,
        )
        self.input_layernorm = RMSResNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSResNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Fused: residual = residual + hidden_states, hidden_states = RMSNorm(residual)
        hidden_states = self.input_layernorm(hidden_states, residual)
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states, fmha_impl=fmha_impl, kv_cache=kv_cache
        )

        # Fused: residual = residual + hidden_states, hidden_states = RMSNorm(residual)
        hidden_states = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Qwen3Model(GptModelBase):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        quant_config: Optional[object] = None,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )

        self.embed_tokens = Embedding(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    config,
                    parallelism_config,
                    idx,
                    weights.weights[idx],
                    quant_config,
                    py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSResNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )

        # Store KV cache shape params for XPU pre-allocated cache
        attn_cfg = config.getAttentionConfigs(parallelism_config.get_attn_tp_size())
        self._kv_head_num = attn_cfg.kv_head_num
        self._head_dim = attn_cfg.size_per_head

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        # Set up per-request XpuKVCaches for XPU device.
        # Supports both single-request and batched execution.
        batch_caches = None
        if self.device_type == DeviceType.Xpu:
            from rtp_llm.models_py.modules.factory.attention.xpu_impl.vllm_flash_attn import (
                XpuKVCache, XpuKVCacheManager,
                set_current_kv_cache, set_batch_kv_caches,
            )
            has_mgr = hasattr(self, '_xpu_kv_cache_mgr') and self._xpu_kv_cache_mgr is not None
            if not has_mgr:
                _sample_w = self.layers[0].input_layernorm.weight
                self._xpu_kv_cache_mgr = XpuKVCacheManager(
                    self.layer_num,
                    num_kv_heads=self._kv_head_num,
                    head_dim=self._head_dim,
                    dtype=_sample_w.dtype,
                    device=_sample_w.device,
                )
                # # _logging.getLogger(__name__).debug(f'[XPU-MGR] Created new KVCacheManager id={id(self._xpu_kv_cache_mgr)}')
            else:
                pass
                # # _logging.getLogger(__name__).debug(f'[XPU-MGR] Reusing KVCacheManager id={id(self._xpu_kv_cache_mgr)} n={self._xpu_kv_cache_mgr.num_active()}')
            mgr = self._xpu_kv_cache_mgr

            attn_in = inputs.attention_inputs
            seq_lengths = attn_in.sequence_lengths  # [decode_batch_size]
            input_lengths = attn_in.input_lengths    # [batch_size]
            try:
                decode_bs = seq_lengths.numel() if seq_lengths is not None else 0
            except (RuntimeError, AttributeError):
                decode_bs = 0
            try:
                batch_size = input_lengths.numel() if input_lengths is not None else 0
            except (RuntimeError, AttributeError):
                batch_size = 0
            context_bs = batch_size - decode_bs

            batch_caches = []

            # Decode requests (first decode_bs entries in the batch)
            for i in range(decode_bs):
                seq_len = int(seq_lengths[i].item())
                expected_kv_len = seq_len
                cache = mgr.get_or_create_for_decode(expected_kv_len)
                batch_caches.append(cache)
                # # _logging.getLogger(__name__).debug(
                # # f"[XPU-KV] Decode req {i}: seq_len={seq_len} expected_kv={expected_kv_len} "
                # # f"cache_len={cache.get_seq_len(0)} mgr_remaining={mgr.num_active()}"
                # # )

            # Context/prefill requests (remaining entries)
            for i in range(context_bs):
                inp_len = int(input_lengths[decode_bs + i].item())
                cache = mgr.create_for_prefill()
                batch_caches.append(cache)
                # # _logging.getLogger(__name__).debug(
                # # f"[XPU-KV] Prefill req {i}: input_len={inp_len}"
                # # )

            # # _logging.getLogger(__name__).debug(
            # # f"[XPU-BATCH] is_prefill={attn_in.is_prefill} decode_bs={decode_bs} "
            # # f"context_bs={context_bs} batch_size={batch_size} "
            # # f"seq_lens={[int(seq_lengths[i].item()) for i in range(decode_bs)] if decode_bs > 0 else []} "
            # # f"input_lens={[int(input_lengths[i].item()) for i in range(batch_size)]} "
            # # f"mgr_caches={list(mgr._caches.keys())}"
            # # )
            if len(batch_caches) == 1:
                # Single request: use legacy thread-local for backward compat
                set_current_kv_cache(batch_caches[0])
                set_batch_kv_caches(None)
            else:
                set_current_kv_cache(None)
                set_batch_kv_caches(batch_caches)
                # # _logging.getLogger(__name__).debug(
                # # f"[XPU] Batched forward: decode={decode_bs} context={context_bs} "
                # # f"total_tokens={input_ids.shape[0]}"
                # # )

        residual = torch.zeros_like(hidden_states)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            select_block_map_for_layer(inputs.attention_inputs, i)
            hidden_states, residual = decoder_layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states = self.norm(hidden_states, residual)

        # Register updated caches back to the manager
        if self.device_type == DeviceType.Xpu and batch_caches is not None and hasattr(self, '_xpu_kv_cache_mgr'):
            from rtp_llm.models_py.modules.factory.attention.xpu_impl.vllm_flash_attn import (
                set_current_kv_cache, set_batch_kv_caches,
            )
            mgr = self._xpu_kv_cache_mgr
            for cache in batch_caches:
                cache.commit()
                kv_len = cache.get_seq_len(0)
                mgr.register(cache, kv_len)
            # NOTE: Do NOT clean up caches here. Other active requests not in
            # this batch still need their caches. The C++ scheduler may batch
            # only a subset of active requests per step.
            set_current_kv_cache(None)
            set_batch_kv_caches(None)

        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = [
    "Qwen3Model",
]
