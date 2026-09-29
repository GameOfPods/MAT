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

# plain ASCII for consoles that can't show the block characters
_ART_ASCII = [
    r" __  __    _  _____",
    r"|  \/  |  / \|_   _|",
    r"| |\/| | / _ \ | |",
    r"| |  | |/ ___ \| |",
    r"|_|  |_/_/   \_\_|",
]
# figlet font "ANSI Shadow"
_ART_BLOCK = [
    "███╗   ███╗ █████╗ ████████╗",
    "████╗ ████║██╔══██╗╚══██╔══╝",
    "██╔████╔██║███████║   ██║",
    "██║╚██╔╝██║██╔══██║   ██║",
    "██║ ╚═╝ ██║██║  ██║   ██║",
    "╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝",
]
_FRAME_ASCII = "+-+|++"
_FRAME_BLOCK = "╔═╗║╚╝"


def versions() -> str:
    """One line for logs: which MAT writes which result format."""
    from MAT import __version__
    from mat_format import __version__ as format_version

    return f"MAT {__version__}, result format {format_version}"


def supports_blocks(stream=None) -> bool:
    """Whether the block letters will come out right. MAT_BANNER=block or ascii decides by hand (none turns the
    banner off, see print_banner). Otherwise the
    stream's encoding has to be able to write them, and the terminal must not be a dumb one. Whether the font has
    the glyphs can't be asked, a UTF-8 terminal almost always has them."""
    import os

    choice = os.environ.get("MAT_BANNER", "").strip().lower()
    if choice in ("block", "ascii"):
        return choice == "block"
    if os.environ.get("TERM", "") == "dumb":
        return False
    stream = stream or sys.stderr
    try:
        "".join(_ART_BLOCK + [_FRAME_BLOCK]).encode(getattr(stream, "encoding", None) or "ascii")
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def banner(blocks: bool = False) -> str:
    """The MAT letters, the name and the versions in a frame, centered on each other."""
    from MAT import __version__
    from mat_format import __version__ as format_version

    art = _ART_BLOCK if blocks else _ART_ASCII
    top_left, horizontal, top_right, vertical, bottom_left, bottom_right = _FRAME_BLOCK if blocks else _FRAME_ASCII
    title = " ".join("MEDIA ANALYTICS TOOLSET") if blocks else "Media Analytics Toolset"
    rows = [("version", __version__), ("result format", format_version), ("by", AUTHOR), ("license", LICENSE),
            ("home", HOMEPAGE)]
    key_width = max(len(key) for key, _ in rows)
    info = [f"{key.rjust(key_width)}  {value}" for key, value in rows]

    art_width, info_width = max(len(line) for line in art), max(len(line) for line in info)
    inner = max(art_width, info_width, len(title)) + 6

    def block(lines, width):
        # a block of lines keeps its own alignment and is centered as a whole
        return [(" " * ((inner - width) // 2) + line.ljust(width)).ljust(inner) for line in lines]

    body = block(art, art_width) + [""] + [title.center(inner)] + [""] + block(info, info_width)
    lines = [top_left + horizontal * inner + top_right]
    lines += [vertical + line.ljust(inner) + vertical for line in body]
    lines += [bottom_left + horizontal * inner + bottom_right]
    return "\n".join(lines) + "\n"


def print_banner(stream=None) -> None:
    """To stderr like the logs, stdout stays for command output. MAT_BANNER=none leaves it out, for scripts and logs
    that should only have log lines. The version line in the log comes anyway."""
    import os

    if os.environ.get("MAT_BANNER", "").strip().lower() == "none":
        return
    stream = stream or sys.stderr
    stream.write(banner(blocks=supports_blocks(stream)) + "\n")
    stream.flush()


__all__ = ["AUTHOR", "HOMEPAGE", "LICENSE", "versions", "supports_blocks", "banner", "print_banner"]
