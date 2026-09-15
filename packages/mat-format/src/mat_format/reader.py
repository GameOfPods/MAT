"""Reads MAT results (format 2) from a result folder or a zip of it."""
import json
import os
import zipfile
from functools import cached_property
from pathlib import Path
from typing import Optional, Union

from mat_format.models import FORMAT_VERSION, BookResult, Meta, PodcastResult


class MATResult:
    def __init__(self, path: Path, meta: Meta, root: Union[Path, zipfile.Path]):
        self._path = path
        self._meta = meta
        self._root = root

    @classmethod
    def read(cls, path: Union[str, os.PathLike]) -> "MATResult":
        path = Path(path)
        if path.is_dir():
            root: Union[Path, zipfile.Path] = path
        elif path.is_file():
            root = zipfile.Path(path)
        else:
            raise FileNotFoundError(f"{path} doesn't exist")
        meta_file = root / "meta.json"
        if not meta_file.exists():
            raise ValueError(f"{path} is not a MAT result, meta.json is missing")
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        if data.get("format") != FORMAT_VERSION:
            found = data.get("format", data.get("version", "unknown"))
            raise ValueError(f"{path} uses result format {found}, this reader reads format {FORMAT_VERSION}. "
                             f"Results from MAT 0.2 and older can't be read anymore.")
        return cls(path=path, meta=Meta.model_validate(data), root=root)

    def text(self, *parts: str) -> Optional[str]:
        """Content of a file inside the result, for example text("podcast", "summary.md"). None if it's missing."""
        file = self._root.joinpath(*parts)
        return file.read_text(encoding="utf-8") if file.exists() else None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def meta(self) -> Meta:
        return self._meta

    @cached_property
    def podcast(self) -> Optional[PodcastResult]:
        if "podcast" not in self._meta.pipelines:
            return None
        return PodcastResult.model_validate_json(self.text("podcast", "result.json"))

    @cached_property
    def book(self) -> Optional[BookResult]:
        if "book" not in self._meta.pipelines:
            return None
        return BookResult.model_validate_json(self.text("book", "result.json"))

    def transcript(self) -> Optional[str]:
        """podcast/transcript.txt, or the same lines built from the segments if the file is missing."""
        if self.podcast is None:
            return None
        text = self.text("podcast", "transcript.txt")
        if text is not None:
            return text
        return "\n".join(f"{' & '.join(s.speakers) or '<Unknown>'} [{s.start} - {s.end}]: {s.text}"
                         for s in self.podcast.segments) + "\n"


__all__ = ["MATResult"]
