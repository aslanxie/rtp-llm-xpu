import os
from enum import IntEnum

import torch


class DeviceType(IntEnum):
    Cpu = 0
    Cuda = 1
    Yitian = 2
    ArmCpu = 3
    ROCm = 4
    Ppu = 5
    Xpu = 6


_DEVICE_TYPE_OVERRIDE = {
    "cpu": DeviceType.Cpu,
    "cuda": DeviceType.Cuda,
    "yitian": DeviceType.Yitian,
    "armcpu": DeviceType.ArmCpu,
    "rocm": DeviceType.ROCm,
    "ppu": DeviceType.Ppu,
    "xpu": DeviceType.Xpu,
}


def get_device_type() -> DeviceType:
    # Explicit override wins so a mixed XPU+CUDA host is never silently
    # resolved by detection-order alone. Set RTP_LLM_DEVICE_TYPE=cuda|xpu|...
    override = os.environ.get("RTP_LLM_DEVICE_TYPE", "").strip().lower()
    if override:
        if override in _DEVICE_TYPE_OVERRIDE:
            return _DEVICE_TYPE_OVERRIDE[override]
        import logging
        logging.getLogger(__name__).warning(
            "Ignoring unknown RTP_LLM_DEVICE_TYPE=%r; valid values: %s",
            override, sorted(_DEVICE_TYPE_OVERRIDE))

    xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
    cuda_available = torch.cuda.is_available()
    if xpu_available and cuda_available:
        # Both backends visible: detection order picks XPU, which may not be
        # what the operator intended. Warn so it is not a silent surprise.
        import logging
        logging.getLogger(__name__).warning(
            "Both XPU and CUDA are available; selecting XPU by detection order. "
            "Set RTP_LLM_DEVICE_TYPE=cuda to force CUDA.")
    if xpu_available:
        return DeviceType.Xpu
    if cuda_available:
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            return DeviceType.ROCm
        if (
            os.environ.get("PPU_HOME")
            or "ppu" in getattr(torch, "__version__", "").lower()
        ):
            return DeviceType.Ppu
        return DeviceType.Cuda
    return DeviceType.Cpu


def is_cuda() -> bool:
    return get_device_type() == DeviceType.Cuda


def is_hip() -> bool:
    return get_device_type() == DeviceType.ROCm


def is_xpu() -> bool:
    return get_device_type() == DeviceType.Xpu
