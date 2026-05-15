"""Device-agnostic utility functions for GPU device management.

This module re-exports functions from rtp_llm.device.device_impl
for backward compatibility. New code should import directly from
rtp_llm.device.device_impl.
"""

from rtp_llm.device.device_impl import (
    get_device_string,
    get_visible_device_list,
    gpu_current_device,
    gpu_device_count,
    gpu_device_name,
    gpu_is_available,
    gpu_memory_info,
    gpu_set_device,
)

__all__ = [
    "gpu_is_available",
    "gpu_device_count",
    "gpu_set_device",
    "gpu_current_device",
    "gpu_device_name",
    "gpu_memory_info",
    "get_device_string",
    "get_visible_device_list",
]
