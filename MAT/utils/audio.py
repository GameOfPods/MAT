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
Splitting long audio for models that only take a limited length at once.

`plan_windows` cuts at the quietest point near the end of each window, so cuts rarely land in the middle of a word.
Windows can overlap. Every window owns the time range between its cut points, and `Window.owns` decides which window
keeps a result from the overlap, so nothing is counted twice when the pieces get merged again.
"""
import math
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class Window:
    start: float
    end: float
    # results with their middle in [keep_start, keep_end) belong to this window
    keep_start: float
    keep_end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def owns(self, start: float, end: float) -> bool:
        """Takes absolute times in seconds, not times relative to the window."""
        middle = (start + end) / 2
        return self.keep_start <= middle < self.keep_end


def frame_energy(samples: np.ndarray, sample_rate: int, frame_seconds: float = 0.1) -> np.ndarray:
    """RMS per frame. Works in blocks, so hours of audio don't get copied to float at once."""
    if samples.ndim > 1:
        samples = samples.mean(axis=-1)
    frame = max(1, int(round(sample_rate * frame_seconds)))
    frames = len(samples) // frame
    energy = np.empty(frames, dtype=np.float64)
    block = 10_000
    for first in range(0, frames, block):
        last = min(frames, first + block)
        chunk = samples[first * frame:last * frame].astype(np.float64).reshape(last - first, frame)
        energy[first:last] = np.sqrt((chunk ** 2).mean(axis=1))
    return energy


def _quietest(energy: np.ndarray, earliest: float, latest: float, frame_seconds: float) -> float:
    first = max(0, int(math.ceil(earliest / frame_seconds)))
    last = min(len(energy) - 1, int(latest / frame_seconds) - 1)
    if first > last:
        return latest
    part = energy[first:last + 1]
    # the latest of the quietest frames, so windows stay as long as possible
    index = first + len(part) - 1 - int(np.argmin(part[::-1]))
    return min(latest, index * frame_seconds + frame_seconds / 2)


def plan_windows(samples: np.ndarray, sample_rate: int, max_length: float, overlap: float = 0.0,
                 search: float = 30.0, frame_seconds: float = 0.1) -> List[Window]:
    """
    Windows of at most `max_length` seconds covering the whole audio. Each cut is placed at the quietest frame in the
    last `search` seconds of the window, neighboring windows share `overlap` seconds around the cut.
    """
    if max_length <= 0:
        raise ValueError("max_length has to be positive")
    if overlap < 0 or overlap * 2 >= max_length:
        raise ValueError("overlap has to be between 0 and half of max_length")
    duration = len(samples) / sample_rate
    if duration <= max_length:
        return [Window(start=0.0, end=duration, keep_start=-math.inf, keep_end=math.inf)]

    energy = frame_energy(samples, sample_rate, frame_seconds)
    windows = []
    start, keep_start = 0.0, -math.inf
    while duration - start > max_length:
        latest = start + max_length - overlap / 2
        earliest = max(start + overlap / 2 + frame_seconds, latest - search)
        cut = _quietest(energy, earliest, latest, frame_seconds)
        windows.append(Window(start=start, end=min(duration, cut + overlap / 2), keep_start=keep_start, keep_end=cut))
        keep_start = cut
        start = cut - overlap / 2
    windows.append(Window(start=start, end=duration, keep_start=keep_start, keep_end=math.inf))
    return windows


__all__ = ["Window", "frame_energy", "plan_windows"]
