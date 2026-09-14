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
Reads MAT results (format 2) from a result folder or a zip of it.

    result = MATResult.read("results/episode_2026-09-14_20-15-02.zip")
    if result.podcast:
        print(result.podcast.speaker_names, result.podcast.summary)
"""
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

FORMAT_VERSION = 2


@dataclass(frozen=True)
class Word:
    start: Optional[float]
    end: Optional[float]
    text: str
    speakers: Tuple[str, ...]


def _timeline(data: Optional[Dict[str, List[List[float]]]]) -> Dict[str, List[Tuple[float, float]]]:
    return {speaker: [(float(start), float(end)) for start, end in times] for speaker, times in (data or {}).items()}


def _words(data: Optional[List[Dict[str, Any]]]) -> List[Word]:
    return [Word(start=w.get("start"), end=w.get("end"), text=w.get("text") or "",
                 speakers=tuple(w.get("speakers") or ())) for w in data or []]


@dataclass(frozen=True)
class PodcastResult:
    language: Optional[str]
    media: Dict[str, Any]
    speakers: Dict[str, List[Tuple[float, float]]]
    diarization: Dict[str, List[Tuple[float, float]]]
    words: List[Word]
    segments: List[Word]
    transcript: str
    summary: Optional[str]
    models: Dict[str, Any]
    events: List[Any]
    entities: List[Any]

    @property
    def duration(self) -> Optional[float]:
        return self.media.get("duration")

    @property
    def speaker_names(self) -> set:
        return set(self.speakers)

    @classmethod
    def from_json(cls, data: Dict[str, Any], transcript: Optional[str] = None) -> "PodcastResult":
        segments = _words(data.get("segments"))
        if transcript is None:
            transcript = "\n".join(f"{' & '.join(s.speakers) or '<Unknown>'} [{s.start} - {s.end}]: {s.text}"
                                   for s in segments)
        return cls(language=data.get("language"), media=data.get("media") or {},
                   speakers=_timeline(data.get("speakers")), diarization=_timeline(data.get("diarization")),
                   words=_words(data.get("words")), segments=segments, transcript=transcript,
                   summary=data.get("summary"), models=data.get("models") or {}, events=data.get("events") or [],
                   entities=data.get("entities") or [])


@dataclass(frozen=True)
class Entity:
    label: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Sentence:
    text: str
    lemmas: Dict[str, int]
    entities: List[Entity]

    def entities_by_label(self) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {}
        for entity in self.entities:
            result.setdefault(entity.label, []).append(entity.text)
        return result


@dataclass(frozen=True)
class Chapter:
    heading: str
    heading_raw: str
    paragraphs: List[str]
    sentences: List[Sentence]


@dataclass(frozen=True)
class BookResult:
    title: str
    language: Optional[str]
    models: Dict[str, Any]
    chapters: List[Chapter]

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "BookResult":
        chapters = [Chapter(
            heading=c["heading"], heading_raw=c.get("heading_raw", c["heading"]), paragraphs=c.get("paragraphs") or [],
            sentences=[Sentence(text=s["text"], lemmas=s.get("lemmas") or {},
                                entities=[Entity(**e) for e in s.get("entities") or []])
                       for s in c.get("sentences") or []],
        ) for c in data.get("chapters") or []]
        return cls(title=data.get("title") or "", language=data.get("language"), models=data.get("models") or {},
                   chapters=chapters)


class MATResult:
    def __init__(self, path: Path, meta: Dict[str, Any], root: Union[Path, zipfile.Path]):
        self._path = path
        self._meta = meta
        self._root = root

    @classmethod
    def read(cls, path: Union[str, os.PathLike]) -> "MATResult":
        path = Path(path)
        if path.is_dir():
            root = path
        elif path.is_file():
            root = zipfile.Path(path)
        else:
            raise FileNotFoundError(f"{path} doesn't exist")
        meta_file = root / "meta.json"
        if not meta_file.exists():
            raise ValueError(f"{path} is not a MAT result, meta.json is missing")
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("format") != FORMAT_VERSION:
            found = meta.get("format", meta.get("version", "unknown"))
            raise ValueError(f"{path} uses result format {found}, this MAT reads format {FORMAT_VERSION}. "
                             f"Results from MAT 0.2 and older can't be read anymore.")
        return cls(path=path, meta=meta, root=root)

    def _text(self, *parts: str) -> Optional[str]:
        file = self._root.joinpath(*parts)
        return file.read_text(encoding="utf-8") if file.exists() else None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def meta(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self._meta))

    @property
    def format(self) -> int:
        return self._meta["format"]

    @property
    def mat_version(self) -> str:
        return self._meta.get("mat_version", "")

    @property
    def created(self) -> str:
        return self._meta.get("created", "")

    @property
    def input_name(self) -> str:
        return self._meta.get("input", {}).get("name", "")

    @property
    def pipelines(self) -> List[str]:
        return list(self._meta.get("pipelines", []))

    @property
    def podcast(self) -> Optional[PodcastResult]:
        if "podcast" not in self.pipelines:
            return None
        return PodcastResult.from_json(json.loads(self._text("podcast", "result.json")),
                                       transcript=self._text("podcast", "transcript.txt"))

    @property
    def book(self) -> Optional[BookResult]:
        if "book" not in self.pipelines:
            return None
        return BookResult.from_json(json.loads(self._text("book", "result.json")))


__all__ = ["MATResult", "PodcastResult", "BookResult", "Word", "Chapter", "Sentence", "Entity", "FORMAT_VERSION"]
