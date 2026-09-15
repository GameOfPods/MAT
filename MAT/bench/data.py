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
"""Reference items and the transcript format of MAT (`speaker [start - end]: text`) used for own references."""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

UNKNOWN_SPEAKER = "<Unknown>"

_LINE = re.compile(r"^(?P<speakers>.*?)\s*\[(?P<start>[^\]\s]+)\s*-\s*(?P<end>[^\]\s]+)\]\s*:\s?(?P<text>.*)$")


class TranscriptError(ValueError):
    pass


@dataclass(frozen=True)
class Turn:
    """One line of a reference: who talks from start to end and what they say. No speakers means unknown."""
    start: Optional[float]
    end: Optional[float]
    speakers: Tuple[str, ...] = ()
    text: str = ""

    @property
    def timed(self) -> bool:
        return self.start is not None and self.end is not None


@dataclass
class Item:
    """One audio file of a dataset with everything that's known about what's said in it."""
    dataset: str
    id: str
    audio: Path
    language: Optional[str] = None
    turns: List[Turn] = field(default_factory=list)
    # Speaker time ranges for DER when a dataset has them separately (AMI, VoxConverse). Otherwise the turns are used.
    speaker_segments: Optional[Dict[str, List[Tuple[float, float]]]] = None
    # whether the turns hold the spoken words (WER) and whether turns or speaker_segments hold speakers (DER)
    has_words: bool = True
    has_speakers: bool = True
    # only this part of the audio is scored, None means from the start or to the end
    start: Optional[float] = None
    end: Optional[float] = None

    def reference_segments(self) -> Dict[str, List[Tuple[float, float]]]:
        if self.speaker_segments is not None:
            return self.speaker_segments
        segments: Dict[str, List[Tuple[float, float]]] = {}
        for turn in self.turns:
            if turn.timed:
                for speaker in turn.speakers:
                    segments.setdefault(speaker, []).append((turn.start, turn.end))
        return segments


def _time(value: str, line_no: int) -> Optional[float]:
    if value == "None":
        return None
    try:
        return float(value)
    except ValueError:
        raise TranscriptError(f"line {line_no}: {value!r} isn't a time in seconds")


def parse_transcript(text: str) -> List[Turn]:
    """Reads MAT's transcript.txt format. Empty lines and lines starting with # are skipped."""
    turns = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _LINE.match(stripped)
        if match is None:
            raise TranscriptError(f'line {line_no}: expected "speaker [start - end]: text", got {stripped[:80]!r}')
        speakers = tuple(s.strip() for s in match["speakers"].split("&")
                         if s.strip() and s.strip() != UNKNOWN_SPEAKER)
        turns.append(Turn(start=_time(match["start"], line_no), end=_time(match["end"], line_no),
                          speakers=speakers, text=match["text"].strip()))
    return turns


def format_turn(turn: Turn) -> str:
    return f"{' & '.join(turn.speakers) or UNKNOWN_SPEAKER} [{turn.start} - {turn.end}]: {turn.text}"


def turns_from_result(podcast) -> List[Turn]:
    """The segments (lines) of a mat_format PodcastResult as turns."""
    return [Turn(start=s.start, end=s.end, speakers=tuple(s.speakers), text=s.text) for s in podcast.segments]


def load_transcript(path: Path) -> List[Turn]:
    """transcript.txt in MAT's format, or a whole MAT result (folder or zip)."""
    path = Path(path)
    if not path.exists():
        raise TranscriptError(f"{path} doesn't exist")
    if path.is_dir() or path.suffix.lower() == ".zip":
        from mat_format import MATResult

        podcast = MATResult.read(path).podcast
        if podcast is None:
            raise TranscriptError(f"{path} has no podcast result")
        return turns_from_result(podcast)
    try:
        return parse_transcript(path.read_text(encoding="utf-8"))
    except TranscriptError as e:
        raise TranscriptError(f"{path}: {e}")


def safe_name(text: str) -> str:
    """Something usable as a folder name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._") or "item"


__all__ = ["UNKNOWN_SPEAKER", "TranscriptError", "Turn", "Item", "parse_transcript", "format_turn",
           "turns_from_result", "load_transcript", "safe_name"]
