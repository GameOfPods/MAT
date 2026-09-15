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
import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union

from pydantic import Field

from MAT.bench.data import Item, Turn
from MAT.utils.config import ConfigError, Options

_LOGGER = logging.getLogger(__name__)

NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


class DatasetOptions(Options):
    type: str = Field(description="Dataset type, `MAT bench datasets` lists them.")
    name: Optional[str] = Field(None, pattern=NAME_PATTERN,
                                description="Name in the report and in the result folders. Default: the type.")
    path: Optional[str] = Field(None, description="Where the dataset already is. Without it, it gets downloaded "
                                                  "into the cache.")
    limit: Optional[int] = Field(None, ge=-1, description="Only use the first files. 0 or -1 uses all of them.")
    collar: Optional[float] = Field(None, ge=0, description="DER collar in seconds for this dataset. Default: "
                                                           "[bench] collar, else the default of the type.")


class Dataset(ABC):
    type: ClassVar[str] = ""
    description: ClassVar[str] = ""
    license: ClassVar[str] = ""
    size: ClassVar[str] = ""
    # shown under the dataset's table in the report
    note: ClassVar[str] = ""
    # DER collar when neither the dataset nor [bench] sets one
    default_collar: ClassVar[float] = 0.0
    Options: ClassVar[Type[DatasetOptions]] = DatasetOptions

    def __init__(self, options: DatasetOptions, cache: Path, base_dir: Path = Path(".")):
        self.options = options
        self.cache_root = Path(cache)
        self.base_dir = Path(base_dir)
        # `MAT bench --limit`, replaces options.limit when set
        self.limit_override: Optional[int] = None
        self._indexes: Dict[Tuple[Path, str], Dict[str, Path]] = {}

    @property
    def name(self) -> str:
        return self.options.name or self.options.type

    @property
    def limit(self) -> Optional[int]:
        """How many samples to use (per language for datasets with several), None means all. 0 and -1 mean all."""
        value = self.options.limit if self.limit_override is None else self.limit_override
        return None if value is None or value <= 0 else value

    @property
    def cache(self) -> Path:
        return self.cache_root / self.type

    def collar(self, bench_collar: Optional[float] = None) -> float:
        if self.options.collar is not None:
            return self.options.collar
        return self.default_collar if bench_collar is None else bench_collar

    def local_path(self) -> Optional[Path]:
        return None if self.options.path is None else resolve_path(self.base_dir, self.options.path)

    def find(self, root: Path, file_name: str) -> Optional[Path]:
        """A file with this name somewhere below root. The folder is scanned once per suffix."""
        suffix = Path(file_name).suffix
        key = (root, suffix)
        if key not in self._indexes:
            self._indexes[key] = {p.name: p for p in root.rglob(f"*{suffix}")
                                  if p.is_file() and not is_junk(p.as_posix())}
        return self._indexes[key].get(file_name)

    def error(self, message: str) -> ConfigError:
        return ConfigError(f"[dataset {self.name}] {message}")

    @abstractmethod
    def items(self) -> Iterator[Item]:
        """Downloads what's missing and yields the items. Called again for `MAT bench report`, so it has to be fast
        once everything is in the cache."""


DATASETS: Dict[str, Type[Dataset]] = {}


def register(cls: Type[Dataset]) -> Type[Dataset]:
    DATASETS[cls.type] = cls
    return cls


def is_junk(name: str) -> bool:
    """macOS metadata files that end up in archives (the VoxConverse zips have ._<id>.wav next to every file)."""
    parts = name.replace("\\", "/").split("/")
    return parts[-1].startswith("._") or "__MACOSX" in parts


def resolve_path(base_dir: Union[str, os.PathLike], value: Union[str, os.PathLike]) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else Path(base_dir) / path


@dataclass
class Clip:
    id: str
    audio: Path
    text: str
    speaker: Optional[str] = None


SAMPLE_RATE = 16000


def pack(clips: Sequence[Clip], folder: Path, prefix: str, max_seconds: float,
         gap: float = 1.0) -> List[Tuple[Path, List[Turn]]]:
    """Joins short clips into 16 kHz mono files of up to max_seconds, with gap seconds of silence in between. Models
    load once per file, so this is much faster than one file per sentence, and it also tests longer audio.
    The files are cached in folder, keyed by the clip ids."""
    from pydub import AudioSegment

    folder.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(("\n".join(c.id for c in clips) + f"\n{max_seconds}\n{gap}").encode()).hexdigest()[:10]
    manifest = folder / f"{prefix}-{key}.json"
    if manifest.exists():
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        packed = [(folder / e["audio"], [Turn(t["start"], t["end"], tuple(t["speakers"]), t["text"])
                                         for t in e["turns"]]) for e in entries]
        if all(path.is_file() for path, _ in packed):
            return packed

    bytes_per_second = SAMPLE_RATE * 2
    silence = b"\0" * int(gap * SAMPLE_RATE) * 2
    packed: List[Tuple[Path, List[Turn]]] = []
    buffer, turns = bytearray(), []

    def flush():
        path = folder / f"{prefix}-{key}-{len(packed) + 1:02d}.wav"
        AudioSegment(data=bytes(buffer), sample_width=2, frame_rate=SAMPLE_RATE, channels=1).export(
            str(path), format="wav")
        packed.append((path, list(turns)))

    for clip in clips:
        audio = AudioSegment.from_file(str(clip.audio)).set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2)
        data = audio.raw_data
        if buffer and (len(buffer) + len(silence) + len(data)) / bytes_per_second > max_seconds:
            flush()
            buffer, turns = bytearray(), []
        if buffer:
            buffer += silence
        start = len(buffer) / bytes_per_second
        buffer += data
        turns.append(Turn(round(start, 3), round(len(buffer) / bytes_per_second, 3),
                          (clip.speaker,) if clip.speaker else (), clip.text))
    if buffer:
        flush()
    manifest.write_text(json.dumps([{"audio": path.name, "turns": [
        {"start": t.start, "end": t.end, "speakers": list(t.speakers), "text": t.text} for t in turns]}
                                    for path, turns in packed], ensure_ascii=False, indent=1), encoding="utf-8")
    _LOGGER.info(f"Packed {len(clips)} clips into {len(packed)} files in {folder}")
    return packed


__all__ = ["NAME_PATTERN", "DatasetOptions", "Dataset", "DATASETS", "register", "is_junk", "resolve_path", "Clip",
           "pack"]
