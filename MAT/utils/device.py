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


__all__ = ["ct2_compute_type"]
