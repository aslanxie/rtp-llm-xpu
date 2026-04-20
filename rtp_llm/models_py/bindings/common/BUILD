
load("//bazel:arch_select.bzl", "torch_deps")
load("//:def.bzl", "copts")

cc_library(
    name = "common",
    srcs = glob(
        ["*.cc"],
        exclude = [
            "CudaGraphPrefillCopy.cc",
            "FusedQKRmsNorm.cc",
            "RtpEmbeddingLookup.cc",
            "RtpNorm.cc",
        ],
    ) + select({
        "//:using_cuda": [
            "CudaGraphPrefillCopy.cc",
            "FusedQKRmsNorm.cc",
            "RtpEmbeddingLookup.cc",
            "RtpNorm.cc",
        ],
        "//:using_rocm": [
            "CudaGraphPrefillCopy.cc",
            "FusedQKRmsNorm.cc",
            "RtpEmbeddingLookup.cc",
            "RtpNorm.cc",
        ],
        "//:using_xpu": [],
        "//conditions:default": [],
    }),
    hdrs = glob(["*.h"]),
    deps = [
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/cpp/core:op_data",
        "//rtp_llm/cpp/core:exec_ops_hdr",
        "//rtp_llm/cpp/core:cache_store_async_writer",
    ] + torch_deps() + select({
        "//:using_cuda": [
            "//rtp_llm/models_py/bindings/common/kernels:kernels_cu",
            "//rtp_llm/cpp/cuda/ops:cuda_impl",
        ],
        "//:using_rocm": [
            "//rtp_llm/models_py/bindings/common/kernels:kernels_cu",
            "//rtp_llm/cpp/core:exec_ctx_rocm",
        ],
        "//:using_xpu": [],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    copts = copts(),
    alwayslink = 1,
)
