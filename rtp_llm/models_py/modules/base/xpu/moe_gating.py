"""XPU MoE gating - PyTorch fallback."""
import torch
import torch.nn as nn


class SigmoidGateScaleAdd(nn.Module):
    def forward(self, gate: torch.Tensor, shared: torch.Tensor, experts: torch.Tensor) -> torch.Tensor:
        assert gate.ndim == 2 and gate.shape[0] == experts.shape[0], (
            f"SigmoidGateScaleAdd: gate must be [T, *], got {gate.shape} "
            f"vs experts {experts.shape}"
        )
        assert shared.shape == experts.shape, (
            f"SigmoidGateScaleAdd: shared/experts shape mismatch: "
            f"shared={shared.shape}, experts={experts.shape}"
        )
        experts.add_(torch.sigmoid(gate) * shared)
        return experts
