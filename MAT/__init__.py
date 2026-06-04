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
import logging
import os
import sys

__author__ = 'RedRem95'
from MAT.__version__ import __version__

logging.basicConfig(
    format="{asctime} - {levelname:^8} - {name}: {message}",
    style="{",
    encoding='utf-8',
    datefmt="%Y.%m.%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)

del os
del logging
del sys

from MAT.tools import *
from MAT.pipelines import *
from MAT.reader import *

__all__ = ['__version__', "__author__", "MATResult", "ResultTypes"]
