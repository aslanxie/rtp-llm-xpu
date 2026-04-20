"""Device-agnostic utility functions for GPU device management.

Provides unified API for CUDA, ROCm, and Intel XPU device operations.
"""

import logging
import os
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


def gpu_is_available() -> bool:
    """Check if any GPU device is available (CUDA, ROCm, or XPU)."""
    if torch.cuda.is_available():
        return True
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return True
    return False


def gpu_device_count() -> int:
    """Return the number of available GPU devices."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.device_count()
    return 0


def gpu_set_device(device_id: int) -> None:
    """Set the current GPU device."""
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.set_device(device_id)


def gpu_current_device() -> int:
    """Get the current GPU device index."""
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.current_device()
    return 0


def gpu_device_name(device_id: int = 0) -> str:
    """Get the name of a GPU device."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(device_id)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.get_device_name(device_id)
    return "cpu"


def gpu_memory_info(device_id: int = 0):
    """Return (free, total) memory in bytes for the GPU device."""
    if torch.cuda.is_available():
        return torch.cuda.mem_get_info(device_id)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.mem_get_info(device_id)
    import psutil
    vmem = psutil.virtual_memory()
    return (vmem.available, vmem.total)


def get_device_string() -> str:
    """Return the device string for tensor placement ('cuda', 'xpu', or 'cpu')."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def get_visible_device_list() -> List[str]:
    """Get list of visible device indices from environment or hardware detection."""
    # Check XPU affinity mask first
    xpu_mask = os.environ.get("ZE_AFFINITY_MASK", None)
    if xpu_mask is not None and hasattr(torch, "xpu") and torch.xpu.is_available():
        return xpu_mask.split(",")

    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_devices is not None:
        return cuda_devices.split(",")

    return [str(i) for i in range(gpu_device_count())]
