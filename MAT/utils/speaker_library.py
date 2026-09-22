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
Remembers voices between episodes.

A speaker that got a name once (from a gold clip, or by hand) keeps a few embeddings here. In the next episode the
same voice is found again without clips and without asking an LLM, and it keeps the same id, which is what a
program reading our results needs for per speaker statistics.

The file is plain JSON, so it can be looked at, edited and put into a backup.
"""
import json
import logging
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_LOGGER = logging.getLogger(__name__)

FILE_NAME = "speakers.json"
# how many embeddings we keep per speaker. A few from different episodes beat one from a single recording
MAX_EMBEDDINGS = 10
# a match needs this much more similarity than the runner up, otherwise two voices are too close to tell apart
DEFAULT_MARGIN = 0.05


@dataclass
class LibrarySpeaker:
    library_id: str
    name: str
    # where the name came from: gold (a clip matched), llm (the transcript said so), manual (you wrote it)
    source: str = "manual"
    embeddings: List[List[float]] = field(default_factory=list)
    episodes: List[str] = field(default_factory=list)
    updated: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    import numpy as np

    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if norm == 0 else float(np.dot(a, b) / norm)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-") or "speaker"


class SpeakerLibrary:
    """Voices we know, kept in one JSON file. Nothing is written until save() is called."""

    def __init__(self, folder: Path, speakers: Optional[List[LibrarySpeaker]] = None):
        self.folder = Path(folder)
        self.speakers: List[LibrarySpeaker] = list(speakers or [])

    @property
    def file(self) -> Path:
        return self.folder / FILE_NAME

    @classmethod
    def open(cls, path) -> "SpeakerLibrary":
        folder = Path(os.path.expandvars(str(path))).expanduser()
        file = folder / FILE_NAME
        if not file.is_file():
            return cls(folder)
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except ValueError as e:
            raise ValueError(f"{file} isn't readable JSON: {e}")
        speakers = []
        for entry in data.get("speakers") or []:
            known = {name: entry[name] for name in LibrarySpeaker.__dataclass_fields__ if name in entry}
            if known.get("library_id") and known.get("name"):
                speakers.append(LibrarySpeaker(**known))
        _LOGGER.info(f"Speaker library: {len(speakers)} known voices in {file}")
        return cls(folder, speakers)

    def save(self) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "speakers": [speaker.as_dict() for speaker in self.speakers]}
        # written next to the file and moved, so a crash can't leave half a library behind
        temporary = self.file.with_name(self.file.name + ".part")
        temporary.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.file)
        return self.file

    def by_name(self, name: str) -> Optional[LibrarySpeaker]:
        return next((speaker for speaker in self.speakers if speaker.name.casefold() == name.casefold()), None)

    def match(self, embedding: Sequence[float], threshold: float,
              margin: float = DEFAULT_MARGIN) -> Optional[Tuple[LibrarySpeaker, float]]:
        """The known voice this embedding belongs to, or None. A match has to be clearly better than the next best,
        otherwise we'd rename a speaker after somebody who only sounds similar."""
        scored = sorted(((speaker, max((_cosine(embedding, known) for known in speaker.embeddings), default=-1.0))
                         for speaker in self.speakers), key=lambda pair: pair[1], reverse=True)
        if not scored or scored[0][1] < threshold:
            return None
        if len(scored) > 1 and scored[0][1] - scored[1][1] < margin:
            _LOGGER.info(f"Not using the speaker library for this voice: {scored[0][0].name} "
                         f"({scored[0][1]:.2f}) and {scored[1][0].name} ({scored[1][1]:.2f}) are too close")
            return None
        return scored[0]

    def remember(self, name: str, embedding: Sequence[float], source: str = "manual",
                 episode: Optional[str] = None) -> LibrarySpeaker:
        """Adds a voice or gives a known one another embedding."""
        speaker = self.by_name(name)
        if speaker is None:
            speaker = LibrarySpeaker(library_id=f"{_slug(name)}-{uuid.uuid4().hex[:6]}", name=name, source=source)
            self.speakers.append(speaker)
        speaker.embeddings.append([float(value) for value in embedding])
        del speaker.embeddings[:-MAX_EMBEDDINGS]
        if episode and episode not in speaker.episodes:
            speaker.episodes.append(episode)
        speaker.updated = datetime.now().isoformat(timespec="seconds")
        return speaker


__all__ = ["FILE_NAME", "MAX_EMBEDDINGS", "DEFAULT_MARGIN", "LibrarySpeaker", "SpeakerLibrary"]
