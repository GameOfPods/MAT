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
Sound events in an episode: music, laughter, applause, or anything you can describe in words (CLAP).

The audio is cut into overlapping windows, every window gets a score per label, and neighbouring windows above the
threshold are joined into one event.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from MAT.tools import Tool, ToolInput, ToolResult
from MAT.utils.config import Config


@dataclass(frozen=True)
class AudioEvent:
    label: str
    start: float
    end: float
    score: Optional[float] = None


class EventResult(ToolResult):
    def __init__(self, events: Optional[List[AudioEvent]] = None):
        self._events = list(events or [])

    @property
    def events(self) -> List[AudioEvent]:
        return list(self._events)


class EventInput(ToolInput):
    def __init__(self, input_file: str):
        self._input_file = input_file

    @property
    def input_file(self) -> str:
        return self._input_file


class EventTool(Tool[EventInput, EventResult], ABC):
    @abstractmethod
    def process(self, origin_data: EventInput, config: Config) -> Optional[EventResult]:
        pass


def windows(duration: float, length: float, hop: float) -> List[Tuple[float, float]]:
    """Start and end of every window. The last one ends with the audio, so nothing at the end is missed."""
    if duration <= length:
        return [(0.0, duration)]
    spans, start = [], 0.0
    while start + length < duration:
        spans.append((start, start + length))
        start += hop
    spans.append((max(0.0, duration - length), duration))
    return spans


def join_windows(spans: Sequence[Tuple[float, float]], scores: Sequence[Dict[str, float]], threshold: float,
                 min_duration: float = 0.0) -> List[AudioEvent]:
    """Events from the scores of every window. Windows overlap, so an event starts and ends in the middle of the
    overlap with the window before and after it, not at the window edges, which would make it a window too long."""

    def event(label: str, first: int, last: int, best: float) -> AudioEvent:
        start, end = spans[first][0], spans[last][1]
        if first > 0:
            start = max(start, (spans[first - 1][1] + spans[first][0]) / 2)
        if last < len(spans) - 1:
            end = min(end, (spans[last][1] + spans[last + 1][0]) / 2)
        return AudioEvent(label=label, start=round(start, 2), end=round(end, 2), score=round(best, 3))

    events: List[AudioEvent] = []
    for label in sorted({name for window in scores for name in window}):
        run: Optional[List] = None
        for i, window in enumerate(scores):
            score = window.get(label, 0.0)
            if score >= threshold:
                run = [i, i, score] if run is None else [run[0], i, max(run[2], score)]
            elif run is not None:
                events.append(event(label, *run))
                run = None
        if run is not None:
            events.append(event(label, *run))
    return sorted((e for e in events if e.end - e.start >= min_duration), key=lambda e: (e.start, e.label))


def load_audio(path: str, sample_rate: int):
    """Mono float32 samples between -1 and 1."""
    import numpy as np
    from pydub import AudioSegment

    sound = AudioSegment.from_file(path).set_channels(1).set_frame_rate(sample_rate).set_sample_width(2)
    return np.frombuffer(sound.raw_data, dtype=np.int16).astype(np.float32) / 32768.0


from MAT.registry import load_optional  # noqa: E402

load_optional("MAT.tools.events.audioset", slot="events", name="audioset", extra="events")
load_optional("MAT.tools.events.clap", slot="events", name="clap", extra="events")

__all__ = ["AudioEvent", "EventResult", "EventInput", "EventTool", "windows", "join_windows", "load_audio"]
