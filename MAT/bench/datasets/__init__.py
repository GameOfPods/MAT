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
"""Dataset types for bench files. A new one subclasses Dataset, has a `type` and gets @register."""
from MAT.bench.datasets.base import (
    DATASETS, NAME_PATTERN, Clip, Dataset, DatasetOptions, pack, register, resolve_path,
)
from MAT.bench.datasets import reference, fleurs, voxconverse, ami, bundestag  # noqa: F401, registers them

__all__ = ["DATASETS", "NAME_PATTERN", "Clip", "Dataset", "DatasetOptions", "pack", "register", "resolve_path"]
