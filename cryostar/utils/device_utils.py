"""
Device detection and configuration utilities for cross-platform GPU support.

This module provides unified device detection for CUDA (NVIDIA), MPS (Apple Silicon),
and CPU backends, along with helpers for distributed training configuration.
"""

import warnings
from typing import Tuple

import torch


def get_autocast_device_type() -> str:
    """
    Get the appropriate device type for autocast based on available hardware.

    Returns
    -------
    str
        Device type string: "cuda", "cpu", or "mps" (future support)

    Notes
    -----
    - CUDA: Returns "cuda" for NVIDIA GPUs
    - MPS: Returns "cpu" as fallback since MPS autocast is not fully supported yet
    - CPU: Returns "cpu" as default fallback

    The function includes error handling to gracefully fallback to CPU if device
    detection fails.
    """
    try:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS doesn't fully support autocast yet, use CPU autocast
            # Core computations will still run on MPS device
            return "cpu"
    except Exception as e:
        warnings.warn(f"Device detection failed: {e}. Falling back to CPU autocast.")

    return "cpu"


def get_accelerator() -> str:
    """
    Detect the best available hardware accelerator.

    Returns
    -------
    str
        Accelerator string for PyTorch Lightning: "mps", "gpu", or "cpu"

    Notes
    -----
    Priority order:
    1. MPS (Apple Silicon) - if available
    2. GPU/CUDA (NVIDIA) - if available
    3. CPU - fallback
    """
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "gpu"
    except Exception as e:
        warnings.warn(f"Accelerator detection failed: {e}. Falling back to CPU.")

    return "cpu"


def get_distributed_backend(accelerator: str) -> str:
    """
    Get the appropriate distributed training backend for the given accelerator.

    Parameters
    ----------
    accelerator : str
        The accelerator type: "mps", "gpu", or "cpu"

    Returns
    -------
    str
        Distributed backend: "nccl" for CUDA/GPU, "gloo" for MPS/CPU

    Notes
    -----
    - NCCL: NVIDIA's optimized backend for CUDA GPUs (best performance)
    - Gloo: Facebook's backend, works on CPU and MPS (more portable)

    Using NCCL on non-NVIDIA hardware will cause crashes.
    """
    if accelerator == "gpu":
        return "nccl"
    else:
        # MPS and CPU use gloo
        return "gloo"


def validate_device_config(accelerator: str, devices: int) -> Tuple[str, int]:
    """
    Validate and adjust device configuration for the given accelerator.

    Parameters
    ----------
    accelerator : str
        The accelerator type: "mps", "gpu", or "cpu"
    devices : int
        Requested number of devices

    Returns
    -------
    tuple of (str, int)
        Validated (accelerator, devices) pair

    Notes
    -----
    - MPS currently only supports single-device training
    - Will warn and adjust devices to 1 if MPS is used with devices > 1
    """
    if accelerator == "mps" and devices > 1:
        warnings.warn(
            f"MPS accelerator does not support multi-device training. "
            f"Requested {devices} devices, but setting to 1."
        )
        devices = 1

    return accelerator, devices


def get_device_info() -> dict:
    """
    Get comprehensive device information for debugging.

    Returns
    -------
    dict
        Dictionary containing device availability and configuration
    """
    info = {
        'cuda_available': False,
        'cuda_device_count': 0,
        'mps_available': False,
        'recommended_accelerator': 'cpu',
        'recommended_backend': 'gloo'
    }

    try:
        info['cuda_available'] = torch.cuda.is_available()
        if info['cuda_available']:
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    try:
        info['mps_available'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    except Exception:
        pass

    # Set recommendations
    accelerator = get_accelerator()
    info['recommended_accelerator'] = accelerator
    info['recommended_backend'] = get_distributed_backend(accelerator)

    return info
