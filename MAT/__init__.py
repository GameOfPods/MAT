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



def _metadata(field: str) -> str:
    """Version (or another field) from the installed package metadata, which comes from pyproject.toml. A source
    checkout that was never installed reads pyproject.toml itself."""
    from importlib import metadata
    from pathlib import Path

    try:
        if field == "version":
            return metadata.version("MAT")
        return metadata.metadata("MAT")[field] or ""
    except metadata.PackageNotFoundError:
        import tomllib

        project = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        return project["version"] if field == "version" else ""


__version__ = _metadata("version")
# "RedRem <alex@...>" in the metadata, only the name
__author__ = _metadata("Author-email").split("<")[0].strip() or "RedRem"

# Logs go to stderr, stdout is for command output like `MAT config init > mat.toml`
logging.basicConfig(
    format="{asctime} - {levelname:^8} - {name}: {message}",
    style="{",
    encoding='utf-8',
    datefmt="%Y.%m.%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)

from MAT.utils.quiet import colorize_console, quiet_dependencies

# libraries that log a lot get turned down, `MAT run --verbose` undoes it
quiet_dependencies()
# warnings and errors get a color when you're looking at a terminal, log files stay plain
colorize_console()

del os
del logging
del sys

from MAT.tools import *
from MAT.pipelines import *
from MAT.reader import *

__all__ = ['__version__', "__author__", "MATResult", "PodcastResult", "BookResult"]
