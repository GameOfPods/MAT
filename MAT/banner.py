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
"""The header MAT prints when it starts a run, and the versions that go into it."""
import sys

AUTHOR = "RedRem"
HOMEPAGE = "github.com/GameOfPods/MAT"
LICENSE = "GPL-3.0"

_ART = r"""
  __  __    _  _____
 |  \/  |  / \|_   _|
 | |\/| | / _ \ | |
 | |  | |/ ___ \| |
 |_|  |_/_/   \_\_|
"""


def versions() -> str:
    """One line for logs: which MAT writes which result format."""
    from MAT import __version__
    from mat_format import __version__ as format_version

    return f"MAT {__version__}, result format {format_version}"


def banner() -> str:
    from MAT import __version__
    from mat_format import __version__ as format_version

    rows = [
        ("", "Media Analytics Toolset"),
        ("version", __version__),
        ("result format", format_version),
        ("by", AUTHOR),
        ("license", LICENSE),
        ("home", HOMEPAGE),
    ]
    width = max(len(key) for key, _ in rows)
    lines = [f"  {key.rjust(width)}  {value}" if key else f"  {' ' * width}  {value}" for key, value in rows]
    return _ART.strip("\n") + "\n\n" + "\n".join(lines) + "\n"


def print_banner(stream=None) -> None:
    """To stderr like the logs, stdout stays for command output."""
    stream = stream or sys.stderr
    stream.write(banner() + "\n")
    stream.flush()


__all__ = ["AUTHOR", "HOMEPAGE", "LICENSE", "versions", "banner", "print_banner"]
