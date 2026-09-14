#  MAT - Toolkit to analyze media
#  Copyright (c) 2025.  RedRem95
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

# Order matters, first supported type wins.
# int8_float16 needs compute capability 7.0+. Pascal cards (GTX 10xx) only get int8_float32.
_CT2_PREFERENCE = ("int8_float16", "int8_float32", "int8", "float32")


def resolve_device(requested: str = "auto") -> str:
    """"auto" becomes "cuda" when torch sees a GPU, otherwise "cpu". Everything else is passed through."""
    if requested != "auto":
        return requested
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def ct2_compute_type(device: str, requested: str = "auto") -> str:
    """
    Compute type for CTranslate2 models (faster-whisper).
    Anything other than "auto" is passed through unchanged.
    """
    if requested != "auto":
        return requested
    import ctranslate2
    supported = ctranslate2.get_supported_compute_types(device.split(":")[0])
    for compute_type in _CT2_PREFERENCE:
        if compute_type in supported:
            return compute_type
    return "default"


def torch_dtype(device: str, preferred: str = "bfloat16"):
    """
    dtype for loading a torch model. Many new models are published in bfloat16, but bfloat16 needs compute capability
    8.0+ (RTX 30xx and newer). GTX 10xx and RTX 20xx get float16 instead, CPU always gets float32.
    """
    import torch
    if preferred not in ("bfloat16", "float16", "float32"):
        raise ValueError(f"Unknown dtype {preferred}, use bfloat16, float16 or float32")
    if not device.startswith("cuda") or preferred == "float32":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(torch.device(device))
    if preferred == "bfloat16" and major >= 8:
        return torch.bfloat16
    return torch.float16


def free_gpu_memory() -> None:
    """Call after `del model`, so the next step gets the GPU memory back. The 1080 Ti only has 11 GB."""
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


__all__ = ["resolve_device", "ct2_compute_type", "torch_dtype", "free_gpu_memory"]
