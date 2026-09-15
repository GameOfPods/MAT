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
"""Own references (audio plus a corrected transcript.txt) and plain audio files without references."""
import glob
import hashlib
import tomllib
from pathlib import Path
from typing import Iterator, List, Optional

from pydantic import Field, ValidationError

from MAT.bench.data import Item, TranscriptError, load_transcript, safe_name
from MAT.bench.datasets.base import Dataset, DatasetOptions, register, resolve_path
from MAT.utils.config import Options


class ReferenceOptions(DatasetOptions):
    path: str = Field(description="A reference folder (with reference.toml) or a folder with one reference folder "
                                  "per episode.")


class ReferenceFile(Options):
    audio: str = Field(description="The audio file, relative to reference.toml or absolute.")
    language: Optional[str] = Field(None, description="Language code like de or en. Default: detected.")
    start: Optional[float] = Field(None, ge=0, description="Only score from here on (seconds).")
    end: Optional[float] = Field(None, gt=0, description="Only score up to here (seconds).")
    transcript: str = Field("transcript.txt", description="The reference in MAT's transcript.txt format, or a MAT "
                                                          "result folder or zip.")


@register
class References(Dataset):
    type = "reference"
    description = "Your own references: audio plus a corrected MAT transcript. WER, cpWER, DER and speaker count."
    license = "yours"
    size = "-"
    note = "Own references, created with `MAT bench reference` and corrected by hand."
    # Line times only cover the words, diarizers also cover the short pauses around them. Without a collar those
    # pauses count as false alarms: 20 % DER on the 30 s sample against its own result, 6 % with 0.25 s.
    default_collar = 0.25
    Options = ReferenceOptions

    def items(self) -> Iterator[Item]:
        root = self.local_path()
        if (root / "reference.toml").is_file():
            folders = [root]
        else:
            folders = sorted(p.parent for p in root.glob("*/reference.toml"))
        if not folders:
            raise self.error(f"no reference.toml in {root} or its subfolders")
        for folder in folders[:self.limit]:
            yield self._item(folder)

    def _item(self, folder: Path) -> Item:
        file = folder / "reference.toml"
        try:
            reference = ReferenceFile.model_validate(tomllib.loads(file.read_text(encoding="utf-8")))
        except (tomllib.TOMLDecodeError, ValidationError) as e:
            raise self.error(f"can't read {file}: {e}")
        if reference.start is not None and reference.end is not None and reference.end <= reference.start:
            raise self.error(f"{file}: end has to be after start")
        audio = resolve_path(folder, reference.audio)
        if not audio.is_file():
            raise self.error(f"{file}: audio file {audio} not found")
        try:
            turns = load_transcript(resolve_path(folder, reference.transcript))
        except TranscriptError as e:
            raise self.error(str(e))
        return Item(dataset=self.name, id=safe_name(folder.name), audio=audio, language=reference.language,
                    turns=turns, has_words=True, has_speakers=any(t.speakers for t in turns),
                    start=reference.start, end=reference.end)


class AudioOptions(DatasetOptions):
    files: List[str] = Field(description="Audio files or glob patterns, relative to the bench file.")
    language: Optional[str] = Field(None, description="Language code like de or en. Default: detected.")


@register
class AudioFiles(Dataset):
    type = "audio"
    description = "Audio without references. Speed, memory, speaker count and agreement with the first system."
    license = "yours"
    size = "-"
    note = "No references, only speed, memory, speaker count and how close the systems are to the first one."
    Options = AudioOptions

    def items(self) -> Iterator[Item]:
        paths = set()
        for pattern in self.options.files:
            expanded = str(resolve_path(self.base_dir, pattern))
            paths.update(Path(p) for p in glob.glob(expanded, recursive=True) if Path(p).is_file())
        if not paths:
            raise self.error(f"no audio files found for {', '.join(self.options.files)}")
        for path in sorted(paths)[:self.limit]:
            digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:6]
            yield Item(dataset=self.name, id=safe_name(f"{path.stem}-{digest}"), audio=path,
                       language=self.options.language, has_words=False, has_speakers=False)


__all__ = ["References", "ReferenceFile", "AudioFiles"]
