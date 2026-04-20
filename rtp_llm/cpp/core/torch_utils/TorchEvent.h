#pragma once

#include "rtp_llm/cpp/core/Event.h"
#include <torch/all.h>
#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#elif USING_XPU
#include <ATen/xpu/XPUContext.h>
#endif

namespace rtp_llm {

struct TorchEvent: public AsyncEvent {
#if USING_XPU
    TorchEvent(const torch::Stream& stream = c10::xpu::getCurrentXPUStream()) {
        event = std::make_shared<torch::Event>(torch::kXPU);
#else
    TorchEvent(const torch::Stream& stream = c10::cuda::getCurrentCUDAStream()) {
        event = std::make_shared<torch::Event>(torch::kCUDA);
#endif
        event->record(stream);
    };

    ~TorchEvent() override = default;

    void synchronize() const override {
        event->synchronize();
    }

    bool checkReadiness() const override {
        return event->query();
    }

    std::shared_ptr<torch::Event> event;
};

}  // namespace rtp_llm
