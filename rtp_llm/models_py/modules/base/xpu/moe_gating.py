"""XPU MoE gating - PyTorch fallback."""
import torch
import torch.nn as nn


class SigmoidGateScaleAdd(nn.Module):
    def forward(self, gate: torch.Tensor, shared: torch.Tensor, experts: torch.Tensor) -> torch.Tensor:
        assert gate.shape == shared.shape == experts.shape, (
            f"SigmoidGateScaleAdd shape mismatch: gate={gate.shape}, "
            f"shared={shared.shape}, experts={experts.shape}"
        )
        experts.add_(torch.sigmoid(gate) * shared)
        return experts
