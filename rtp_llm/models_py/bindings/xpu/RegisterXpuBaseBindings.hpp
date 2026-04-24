#pragma once

#include <torch/extension.h>
#include <ATen/ATen.h>

namespace py = pybind11;

namespace torch_ext {

// ── Helper: RMSNorm (pure PyTorch) ─────────────────────────────────────────
static void xpu_rmsnorm_impl(at::Tensor& output,
                              const at::Tensor& input,
                              const at::Tensor& weight,
                              double eps) {
    auto float_input = input.to(at::kFloat);
    auto variance = float_input.pow(2).mean(-1, /*keepdim=*/true);
    auto normed = float_input * at::rsqrt(variance + eps);
    output.copy_((weight * normed).to(input.scalar_type()));
}

// ── Helper: LayerNorm (pure PyTorch) ────────────────────────────────────────
static void xpu_layernorm_impl(at::Tensor& output,
                                const at::Tensor& input,
                                const at::Tensor& weight,
                                const at::Tensor& beta,
                                double eps) {
    auto norm_shape = weight.sizes().vec();
    auto result = at::layer_norm(input, norm_shape, weight, beta, eps);
    output.copy_(result);
}

void registerBaseXpuBindings(py::module& rtp_ops_m) {

    // ── Debug ───────────────────────────────────────────────────────────────
    rtp_ops_m.def("debug_kernel",
                  [](const at::Tensor& /*data*/,
                     int64_t /*start_row*/, int64_t /*start_col*/,
                     int64_t /*m*/, int64_t /*n*/,
                     int64_t /*row_len*/, int64_t /*info_id*/) {
                  },
                  "Debug kernel (no-op on XPU)",
                  py::arg("data"),
                  py::arg("start_row"),
                  py::arg("start_col"),
                  py::arg("m"),
                  py::arg("n"),
                  py::arg("row_len"),
                  py::arg("info_id"));

    // ── Write cache store ───────────────────────────────────────────────────
    rtp_ops_m.def("write_cache_store",
                  [](const at::Tensor& /*input_lengths*/,
                     const at::Tensor& /*prefix_lengths*/,
                     const at::Tensor& /*kv_cache_block_id_host*/,
                     py::object /*cache_store_member*/,
                     py::object /*kv_cache*/) {
                  },
                  "WriteCacheStoreOp (no-op on XPU)",
                  py::arg("input_lengths"),
                  py::arg("prefix_lengths"),
                  py::arg("kv_cache_block_id_host"),
                  py::arg("cache_store_member"),
                  py::arg("kv_cache"));

    // ── RMSNorm ─────────────────────────────────────────────────────────────
    rtp_ops_m.def("rmsnorm",
                  [](at::Tensor& output,
                     const at::Tensor& input,
                     const at::Tensor& weight,
                     double eps,
                     int64_t /*cuda_stream*/) {
                      xpu_rmsnorm_impl(output, input, weight, eps);
                  },
                  "RMSNorm kernel (PyTorch fallback on XPU)",
                  py::arg("output"),
                  py::arg("input"),
                  py::arg("weight"),
                  py::arg("eps"),
                  py::arg("cuda_stream") = 0);

    // ── Fused Add RMSNorm ───────────────────────────────────────────────────
    rtp_ops_m.def("fused_add_rmsnorm",
                  [](at::Tensor& input,
                     at::Tensor& residual,
                     const at::Tensor& weight,
                     double eps,
                     int64_t /*cuda_stream*/) {
                      input.add_(residual);
                      residual.copy_(input);
                      auto float_input = input.to(at::kFloat);
                      auto variance = float_input.pow(2).mean(-1, /*keepdim=*/true);
                      auto normed = float_input * at::rsqrt(variance + eps);
                      input.copy_((weight * normed).to(input.scalar_type()));
                  },
                  "Fused Add RMSNorm kernel (PyTorch fallback on XPU)",
                  py::arg("input"),
                  py::arg("residual"),
                  py::arg("weight"),
                  py::arg("eps"),
                  py::arg("cuda_stream") = 0);

    // ── SiLU and Mul ────────────────────────────────────────────────────────
    rtp_ops_m.def("silu_and_mul",
                  [](at::Tensor& output,
                     const at::Tensor& input,
                     int64_t /*cuda_stream*/) {
                      int64_t d = input.size(-1) / 2;
                      auto gate = input.narrow(-1, 0, d);
                      auto up   = input.narrow(-1, d, d);
                      output.copy_(at::silu(gate) * up);
                  },
                  "SiLU and Mul kernel (PyTorch fallback on XPU)",
                  py::arg("output"),
                  py::arg("input"),
                  py::arg("cuda_stream") = 0);

    // ── Fused QK RMSNorm ────────────────────────────────────────────────────
    rtp_ops_m.def("fused_qk_rmsnorm",
                  [](at::Tensor& IO,
                     const at::Tensor& q_gamma,
                     const at::Tensor& k_gamma,
                     double layernorm_eps,
                     int64_t q_group_num,
                     int64_t k_group_num,
                     int64_t m,
                     int64_t n,
                     int64_t norm_size) {
                      int64_t q_size = q_group_num * norm_size;
                      int64_t k_size = k_group_num * norm_size;
                      auto q_flat = IO.narrow(0, 0, m).narrow(1, 0, q_size)
                                       .reshape({m * q_group_num, norm_size});
                      auto q_out = at::empty_like(q_flat);
                      xpu_rmsnorm_impl(q_out, q_flat, q_gamma, layernorm_eps);
                      IO.narrow(0, 0, m).narrow(1, 0, q_size).copy_(
                          q_out.reshape({m, q_size}));
                      auto k_flat = IO.narrow(0, 0, m).narrow(1, q_size, k_size)
                                       .reshape({m * k_group_num, norm_size});
                      auto k_out = at::empty_like(k_flat);
                      xpu_rmsnorm_impl(k_out, k_flat, k_gamma, layernorm_eps);
                      IO.narrow(0, 0, m).narrow(1, q_size, k_size).copy_(
                          k_out.reshape({m, k_size}));
                  },
                  "Fused QK RMSNorm kernel (decomposed on XPU)",
                  py::arg("IO"),
                  py::arg("q_gamma"),
                  py::arg("k_gamma"),
                  py::arg("layernorm_eps"),
                  py::arg("q_group_num"),
                  py::arg("k_group_num"),
                  py::arg("m"),
                  py::arg("n"),
                  py::arg("norm_size"));

    // ── LayerNorm ───────────────────────────────────────────────────────────
    rtp_ops_m.def("layernorm",
                  [](at::Tensor& output,
                     const at::Tensor& input,
                     const at::Tensor& weight,
                     const at::Tensor& beta,
                     double eps) {
                      xpu_layernorm_impl(output, input, weight, beta, eps);
                  },
                  "LayerNorm kernel (PyTorch fallback on XPU)",
                  py::arg("output"),
                  py::arg("input"),
                  py::arg("weight"),
                  py::arg("beta"),
                  py::arg("eps"));

    // ── Fused Add LayerNorm ─────────────────────────────────────────────────
    rtp_ops_m.def("fused_add_layernorm",
                  [](at::Tensor& input,
                     at::Tensor& residual,
                     const at::Tensor& bias,
                     const at::Tensor& weight,
                     const at::Tensor& beta,
                     double eps) {
                      input.add_(residual);
                      if (bias.numel() > 0) {
                          input.add_(bias);
                      }
                      residual.copy_(input);
                      auto norm_shape = weight.sizes().vec();
                      auto normed = at::layer_norm(input, norm_shape, weight, beta, eps);
                      input.copy_(normed);
                  },
                  "Fused Add LayerNorm kernel (PyTorch fallback on XPU)",
                  py::arg("input"),
                  py::arg("residual"),
                  py::arg("bias"),
                  py::arg("weight"),
                  py::arg("beta"),
                  py::arg("eps"));

    // ── Per-token group quant int8 ──────────────────────────────────────────
    rtp_ops_m.def("per_token_group_quant_int8",
                  [](const at::Tensor& input,
                     at::Tensor& output_q,
                     at::Tensor& output_s,
                     int64_t group_size,
                     double eps,
                     double int8_min,
                     double int8_max,
                     bool /*scale_ue8m0*/) {
                      auto float_input = input.to(at::kFloat);
                      auto shape = float_input.sizes().vec();
                      int64_t last_dim = shape.back();
                      int64_t num_groups = last_dim / group_size;
                      auto reshaped = float_input.reshape({-1, num_groups, group_size});
                      auto abs_max = reshaped.abs().amax(-1, true).clamp_min(eps);
                      auto scale = abs_max / int8_max;
                      auto quantized = (reshaped / scale).clamp(int8_min, int8_max).round().to(at::kChar);
                      output_q.copy_(quantized.reshape(shape).to(output_q.scalar_type()));
                      output_s.copy_(scale.squeeze(-1).to(output_s.scalar_type()));
                  },
                  "Int8 per-token group quantization (PyTorch fallback on XPU)",
                  py::arg("input"),
                  py::arg("output_q"),
                  py::arg("output_s"),
                  py::arg("group_size"),
                  py::arg("eps"),
                  py::arg("int8_min"),
                  py::arg("int8_max"),
                  py::arg("scale_ue8m0"));

    // ── Per-token group quant fp8 ───────────────────────────────────────────
    rtp_ops_m.def("per_token_group_quant_fp8",
                  [](const at::Tensor& input,
                     at::Tensor& output_q,
                     at::Tensor& output_s,
                     int64_t group_size,
                     double eps,
                     double fp8_min,
                     double fp8_max,
                     bool /*scale_ue8m0*/) {
                      auto float_input = input.to(at::kFloat);
                      auto shape = float_input.sizes().vec();
                      int64_t last_dim = shape.back();
                      int64_t num_groups = last_dim / group_size;
                      auto reshaped = float_input.reshape({-1, num_groups, group_size});
                      auto abs_max = reshaped.abs().amax(-1, true).clamp_min(eps);
                      auto scale = abs_max / fp8_max;
                      auto quantized = (reshaped / scale).clamp(fp8_min, fp8_max);
                      output_q.copy_(quantized.reshape(shape).to(output_q.scalar_type()));
                      output_s.copy_(scale.squeeze(-1).to(output_s.scalar_type()));
                  },
                  "FP8 per-token group quantization (PyTorch fallback on XPU)",
                  py::arg("input"),
                  py::arg("output_q"),
                  py::arg("output_s"),
                  py::arg("group_size"),
                  py::arg("eps"),
                  py::arg("fp8_min"),
                  py::arg("fp8_max"),
                  py::arg("scale_ue8m0"));

    // ── Per-token group quant fp8 v2 ────────────────────────────────────────
    rtp_ops_m.def("per_token_group_quant_fp8_v2",
                  [](const at::Tensor& input,
                     at::Tensor& output_q,
                     at::Tensor& output_s,
                     int64_t group_size,
                     double eps,
                     double fp8_min,
                     double fp8_max,
                     bool /*scale_ue8m0*/,
                     bool fuse_silu_and_mul,
                     int64_t masked_m) {
                      at::Tensor actual_input;
                      if (fuse_silu_and_mul) {
                          int64_t d = input.size(-1) / 2;
                          auto gate = input.narrow(-1, 0, d);
                          auto up   = input.narrow(-1, d, d);
                          actual_input = at::silu(gate) * up;
                      } else {
                          actual_input = input;
                      }
                      auto float_input = actual_input.to(at::kFloat);
                      if (masked_m > 0 && masked_m < float_input.size(0)) {
                          float_input = float_input.narrow(0, 0, masked_m);
                      }
                      auto shape = float_input.sizes().vec();
                      int64_t last_dim = shape.back();
                      int64_t num_groups = last_dim / group_size;
                      auto reshaped = float_input.reshape({-1, num_groups, group_size});
                      auto abs_max = reshaped.abs().amax(-1, true).clamp_min(eps);
                      auto scale = abs_max / fp8_max;
                      auto quantized = (reshaped / scale).clamp(fp8_min, fp8_max);
                      auto q_view = quantized.reshape(shape).to(output_q.scalar_type());
                      auto s_view = scale.squeeze(-1).to(output_s.scalar_type());
                      if (masked_m > 0 && masked_m < output_q.size(0)) {
                          output_q.narrow(0, 0, masked_m).copy_(q_view);
                          output_s.narrow(0, 0, masked_m).copy_(s_view);
                      } else {
                          output_q.copy_(q_view);
                          output_s.copy_(s_view);
                      }
                  },
                  "FP8 per-token group quantization v2 (PyTorch fallback on XPU)",
                  py::arg("input"),
                  py::arg("output_q"),
                  py::arg("output_s"),
                  py::arg("group_size"),
                  py::arg("eps"),
                  py::arg("fp8_min"),
                  py::arg("fp8_max"),
                  py::arg("scale_ue8m0"),
                  py::arg("fuse_silu_and_mul"),
                  py::arg("masked_m"));

    // ── MoE TopK Softmax ────────────────────────────────────────────────────
    rtp_ops_m.def("moe_topk_softmax",
                  [](at::Tensor& topk_weights,
                     at::Tensor& topk_indices,
                     at::Tensor& token_expert_indices,
                     const at::Tensor& gating_output) {
                      int64_t k = topk_weights.size(-1);
                      auto softmaxed = at::softmax(gating_output.to(at::kFloat), -1);
                      auto topk_result = softmaxed.topk(k, -1);
                      topk_weights.copy_(std::get<0>(topk_result).to(topk_weights.scalar_type()));
                      topk_indices.copy_(std::get<1>(topk_result).to(topk_indices.scalar_type()));
                      token_expert_indices.copy_(topk_indices.reshape({-1}).to(token_expert_indices.scalar_type()));
                  },
                  "MoE TopK Softmax (PyTorch fallback on XPU)",
                  py::arg("topk_weights"),
                  py::arg("topk_indices"),
                  py::arg("token_expert_indices"),
                  py::arg("gating_output"));

    // ── Embedding ───────────────────────────────────────────────────────────
    rtp_ops_m.def("embedding",
                  [](at::Tensor& output,
                     const at::Tensor& input,
                     const at::Tensor& weight) {
                      auto result = at::embedding(weight, input.to(at::kLong));
                      output.copy_(result.to(output.scalar_type()));
                  },
                  "Embedding lookup (PyTorch fallback on XPU)",
                  py::arg("output"),
                  py::arg("input"),
                  py::arg("weight"));

    // ── Embedding BERT ──────────────────────────────────────────────────────
    rtp_ops_m.def("embedding_bert",
                  [](at::Tensor& output,
                     const at::Tensor& input,
                     const at::Tensor& weight,
                     const at::Tensor& combo_position_ids,
                     const at::Tensor& position_encoding,
                     const at::Tensor& combo_tokens_type_ids,
                     const at::Tensor& token_type_embedding,
                     float input_embedding_scalar) {
                      auto word_emb = at::embedding(weight, input.to(at::kLong));
                      auto pos_emb  = at::embedding(position_encoding, combo_position_ids.to(at::kLong));
                      auto type_emb = at::embedding(token_type_embedding, combo_tokens_type_ids.to(at::kLong));
                      auto result = (word_emb * input_embedding_scalar + pos_emb + type_emb);
                      output.copy_(result.to(output.scalar_type()));
                  },
                  "BERT Embedding lookup (PyTorch fallback on XPU)",
                  py::arg("output"),
                  py::arg("input"),
                  py::arg("weight"),
                  py::arg("combo_position_ids"),
                  py::arg("position_encoding"),
                  py::arg("combo_tokens_type_ids"),
                  py::arg("token_type_embedding"),
                  py::arg("input_embedding_scalar") = 1.0f);

    // ── Reuse KV cache indexed batched ──────────────────────────────────────
    rtp_ops_m.def("reuse_kv_cache_indexed_batched",
                  [](at::Tensor& /*final_compressed_kv*/,
                     at::Tensor& /*final_k_pe*/,
                     const at::Tensor& /*compressed_kv*/,
                     const at::Tensor& /*k_pe*/,
                     const at::Tensor& /*kv_cache_base*/,
                     const at::Tensor& /*reuse_cache_page_indice*/,
                     py::object /*batch_reuse_info_vec*/,
                     const at::Tensor& /*qo_indptr*/,
                     int64_t /*tokens_per_block*/) {
                  },
                  "Reuse KV cache indexed batched (no-op on XPU)",
                  py::arg("final_compressed_kv"),
                  py::arg("final_k_pe"),
                  py::arg("compressed_kv"),
                  py::arg("k_pe"),
                  py::arg("kv_cache_base"),
                  py::arg("reuse_cache_page_indice"),
                  py::arg("batch_reuse_info_vec"),
                  py::arg("qo_indptr"),
                  py::arg("tokens_per_block"));

    // ── MLA K Merge ─────────────────────────────────────────────────────────
    rtp_ops_m.def("mla_k_merge",
                  [](at::Tensor& k_out,
                     const at::Tensor& k_nope,
                     const at::Tensor& k_pe) {
                      auto merged = at::cat({k_nope, k_pe}, -1);
                      k_out.copy_(merged);
                  },
                  "MLA K merge (PyTorch fallback on XPU)",
                  py::arg("k_out"),
                  py::arg("k_nope"),
                  py::arg("k_pe"));

    // ── CUDA Graph Copy (no-op on XPU) ──────────────────────────────────────
    rtp_ops_m.def("cuda_graph_copy_small2large",
                  [](const at::Tensor&, at::Tensor&,
                     int64_t, int64_t, int64_t,
                     const at::Tensor&, int64_t, const at::Tensor&) {
                  },
                  "CUDA Graph small-to-large copy (no-op on XPU)",
                  py::arg("input_tensor"),
                  py::arg("output_tensor"),
                  py::arg("batch_size"),
                  py::arg("max_batch_size"),
                  py::arg("max_seq_len"),
                  py::arg("input_lengths"),
                  py::arg("hidden_size"),
                  py::arg("cu_seq_len"));

    rtp_ops_m.def("cuda_graph_copy_large2small",
                  [](const at::Tensor&, at::Tensor&,
                     int64_t, int64_t, int64_t,
                     const at::Tensor&, int64_t, const at::Tensor&) {
                  },
                  "CUDA Graph large-to-small copy (no-op on XPU)",
                  py::arg("input_tensor"),
                  py::arg("output_tensor"),
                  py::arg("batch_size"),
                  py::arg("max_batch_size"),
                  py::arg("max_seq_len"),
                  py::arg("input_lengths"),
                  py::arg("hidden_size"),
                  py::arg("cu_seq_len"));

    // ── Fast TopK v2 ────────────────────────────────────────────────────────
    rtp_ops_m.def("fast_topk_v2",
                  [](at::Tensor& score,
                     at::Tensor& indices,
                     at::Tensor& lengths,
                     py::object /*row_starts*/) {
                      int64_t k = indices.size(-1);
                      auto topk_result = score.topk(k, -1);
                      score.narrow(-1, 0, k).copy_(std::get<0>(topk_result));
                      indices.copy_(std::get<1>(topk_result).to(indices.scalar_type()));
                  },
                  "Fast TopK v2 (PyTorch fallback on XPU)",
                  py::arg("score"),
                  py::arg("indices"),
                  py::arg("lengths"),
                  py::arg("row_starts") = py::none());

    // ── Fast TopK Transform Fused ───────────────────────────────────────────
    rtp_ops_m.def("fast_topk_transform_fused",
                  [](at::Tensor&, at::Tensor&, at::Tensor&,
                     py::object, at::Tensor&, py::object) {
                  },
                  "Fast TopK Transform Fused (no-op on XPU)",
                  py::arg("score"),
                  py::arg("lengths"),
                  py::arg("dst_page_table"),
                  py::arg("src_page_table") = py::none(),
                  py::arg("cu_seqlens_q"),
                  py::arg("row_starts") = py::none());

    // ── Fast TopK Transform Ragged Fused ────────────────────────────────────
    rtp_ops_m.def("fast_topk_transform_ragged_fused",
                  [](at::Tensor&, at::Tensor&, at::Tensor&,
                     at::Tensor&, py::object) {
                  },
                  "Fast TopK Transform Ragged Fused (no-op on XPU)",
                  py::arg("score"),
                  py::arg("lengths"),
                  py::arg("topk_indices_ragged"),
                  py::arg("topk_indices_offset"),
                  py::arg("row_starts") = py::none());

    // ── Indexer K quant and cache ───────────────────────────────────────────
    rtp_ops_m.def("indexer_k_quant_and_cache",
                  [](const at::Tensor&, py::object,
                     const at::Tensor&, int64_t, int64_t) {
                      TORCH_CHECK(false,
                          "indexer_k_quant_and_cache not implemented on XPU.");
                  },
                  "Indexer K quant and cache (not implemented on XPU)",
                  py::arg("k"),
                  py::arg("kv_cache"),
                  py::arg("slot_mapping"),
                  py::arg("quant_block_size"),
                  py::arg("scale_fmt"));

    // ── CP Gather indexer K quant cache ─────────────────────────────────────
    rtp_ops_m.def("cp_gather_indexer_k_quant_cache",
                  [](py::object, at::Tensor&, at::Tensor&,
                     const at::Tensor&, const at::Tensor&) {
                  },
                  "CP Gather indexer K quant cache (no-op on XPU)",
                  py::arg("kv_cache"),
                  py::arg("dst_k"),
                  py::arg("dst_scale"),
                  py::arg("block_table"),
                  py::arg("cu_seq_lens"));

    // ── CP Gather and upconvert FP8 KV cache ───────────────────────────────
    rtp_ops_m.def("cp_gather_and_upconvert_fp8_kv_cache",
                  [](py::object, at::Tensor&, at::Tensor&,
                     const at::Tensor&, const at::Tensor&,
                     const at::Tensor&, int64_t) {
                  },
                  "CP Gather and upconvert FP8 KV cache (no-op on XPU)",
                  py::arg("src_cache"),
                  py::arg("dst_compressed_kv"),
                  py::arg("dst_k_pe"),
                  py::arg("block_table"),
                  py::arg("seq_lens"),
                  py::arg("workspace_starts"),
                  py::arg("batch_size"));

    // ── Concat and cache MLA ────────────────────────────────────────────────
    rtp_ops_m.def("concat_and_cache_mla",
                  [](const at::Tensor&, const at::Tensor&,
                     py::object, const at::Tensor&,
                     const std::string&, double) {
                      TORCH_CHECK(false,
                          "concat_and_cache_mla not implemented on XPU.");
                  },
                  "Concat and cache MLA (not implemented on XPU)",
                  py::arg("kv_c"),
                  py::arg("k_pe"),
                  py::arg("kv_cache"),
                  py::arg("slot_mapping"),
                  py::arg("kv_cache_dtype"),
                  py::arg("scale"));
}

}  // namespace torch_ext
