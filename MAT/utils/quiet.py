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
"""
Keeps the console readable.

Most libraries use Python logging, so a level is enough. NeMo brings its own logger with its own handlers, prints
to stdout in its own format and never reaches our log file. `quiet_nemo` takes those handlers away and lets the
records go through Python logging like everything else, so `--log-file` gets them too.
"""
import logging
import os
import sys
import warnings
from typing import Dict

# Loggers that say a lot and rarely anything we need. `MAT run --verbose` keeps them all.
NOISY: Dict[str, int] = {
    "nemo_logger": logging.WARNING,
    "nv_one_logger": logging.ERROR,
    "lightning": logging.WARNING,
    "lightning_fabric": logging.WARNING,
    "pytorch_lightning": logging.WARNING,
    "pytorch_lightning.utilities.migration.utils": logging.ERROR,
    "transformers": logging.WARNING,
    "huggingface_hub": logging.WARNING,
    "speechbrain": logging.WARNING,
    "matplotlib": logging.WARNING,
    "numba": logging.WARNING,
    "filelock": logging.WARNING,
    "fsspec": logging.WARNING,
    "urllib3": logging.WARNING,
    "asyncio": logging.WARNING,
    "torio": logging.WARNING,
    "datasets": logging.WARNING,
    "sentence_transformers": logging.WARNING,
}

# (message pattern, category) of warnings that show up on every run and that we can't do anything about
WARNINGS = [
    ("torchaudio._backend", UserWarning),
    ("TypedStorage is deprecated", UserWarning),
    ("`resume_download` is deprecated", FutureWarning),
    ("std\\(\\): degrees of freedom", UserWarning),
    ("Trying to infer the `batch_size`", UserWarning),
    ("audioop", DeprecationWarning),
]


COLORS = {"WARNING": "\033[33m", "ERROR": "\033[31m", "CRITICAL": "\033[31;1m"}
RESET = "\033[0m"


def use_colors() -> bool:
    """Colors only for a terminal. Piped into a file or with NO_COLOR set (https://no-color.org) they'd be noise."""
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


class ColorFormatter(logging.Formatter):
    """Paints the level of a record, so a warning stands out between hundreds of info lines."""

    def format(self, record: logging.LogRecord) -> str:
        color = COLORS.get(record.levelname)
        if not color:
            return super().format(record)
        original = record.levelname
        try:
            record.levelname = f"{color}{original}{RESET}"
            return super().format(record)
        finally:
            record.levelname = original


def colorize_console() -> None:
    """Gives the handlers that write to a terminal a coloring formatter. The log file keeps plain text."""
    if not use_colors():
        return
    for handler in logging.getLogger().handlers:
        stream = getattr(handler, "stream", None)
        if stream is None or not getattr(stream, "isatty", lambda: False)():
            continue
        current = handler.formatter
        if current is None or isinstance(current, ColorFormatter):
            continue
        handler.setFormatter(ColorFormatter(fmt=current._fmt, datefmt=current.datefmt, style="{"))


def quiet_dependencies(verbose: bool = False) -> None:
    """Turns down libraries that log a lot. Called when MAT is imported, `--verbose` undoes it."""
    if verbose:
        for name in NOISY:
            logging.getLogger(name).setLevel(logging.NOTSET)
        quiet_nemo(verbose=True)
        return
    # tokenizers warns about forking on every worker start
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for name, level in NOISY.items():
        logging.getLogger(name).setLevel(level)
    for message, category in WARNINGS:
        warnings.filterwarnings("ignore", message=message, category=category)
    quiet_nemo()


def quiet_nemo(verbose: bool = False) -> None:
    """Hands NeMo's logging over to Python logging. Does nothing while NeMo isn't imported, so call it again after
    importing it."""
    module = sys.modules.get("nemo.utils")
    nemo_logging = getattr(module, "logging", None)
    if nemo_logging is None:
        return
    try:
        # its own handlers write to stdout in NeMo's format, ours write to stderr and to --log-file
        nemo_logging.remove_stream_handlers()
        nemo_logging.setLevel(logging.INFO if verbose else logging.WARNING)
        inner = getattr(nemo_logging, "_logger", None)
        if inner is not None:
            inner.propagate = True
    except Exception as e:  # never let logging setup break a run
        logging.getLogger(__name__).debug(f"Could not quiet NeMo logging: {e}")


__all__ = ["NOISY", "WARNINGS", "quiet_dependencies", "quiet_nemo"]
