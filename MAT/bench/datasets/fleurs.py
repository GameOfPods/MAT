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
FLEURS (google/fleurs on Hugging Face): read Wikipedia sentences.

Layout per language: data/<code>/<split>.tsv (no header: id, file name, raw transcription, normalized transcription,
characters, number of samples, gender) and data/<code>/audio/<split>.tar.gz with <split>/<file name>.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Tuple

from pydantic import Field

from MAT.bench.data import Item
from MAT.bench.datasets.base import Clip, Dataset, DatasetOptions, pack, register
from MAT.bench.download import download, extract_from_tar

BASE_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
LANGUAGES = {"de": "de_de", "en": "en_us"}


class FleursOptions(DatasetOptions):
    languages: List[str] = Field(["de", "en"], description="de, en or any FLEURS code like fr_fr.")
    split: Literal["dev", "test", "train"] = Field("test", description="dev, test or train.")
    limit: Optional[int] = Field(100, ge=1, description="Sentences per language.")
    pack_minutes: float = Field(10, gt=0, description="Sentences are joined into files of up to this many minutes.")


@dataclass
class FleursRow:
    id: str
    file: str
    text: str
    samples: int


def parse_tsv(text: str) -> List[FleursRow]:
    rows = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        rows.append(FleursRow(id=parts[0], file=parts[1], text=parts[2],
                              samples=int(parts[5]) if parts[5].isdigit() else 0))
    return rows


def select_rows(rows: List[FleursRow], limit: Optional[int]) -> List[FleursRow]:
    """Every sentence is recorded by several speakers. Take one recording per sentence, in file order."""
    seen, selected = set(), []
    for row in rows:
        if row.id in seen:
            continue
        seen.add(row.id)
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return selected


@register
class Fleurs(Dataset):
    type = "fleurs"
    description = "FLEURS read sentences (many languages). WER only, sentences get joined into longer files."
    license = "CC-BY-4.0"
    size = "whole audio archive of a split: de test 570 MB, en test 290 MB"
    note = "Read sentences, clean audio, one speaker per sentence. Only says how well words are recognized."
    Options = FleursOptions

    def items(self) -> Iterator[Item]:
        split = self.options.split
        for language in self.options.languages:
            code = LANGUAGES.get(language, language)
            tsv, audio_dir, archive = self._locate(code, split)
            rows = select_rows(parse_tsv(tsv.read_text(encoding="utf-8")), self.options.limit)
            if not rows:
                raise self.error(f"{tsv} has no sentences")
            wavs: Dict[str, Path] = {}
            wanted: Dict[str, Path] = {}
            for row in rows:
                existing = None if audio_dir is None else audio_dir / row.file
                if existing is not None and existing.is_file():
                    wavs[row.file] = existing
                    continue
                target = self.cache / code / split / row.file
                wavs[row.file] = target
                if not target.is_file():
                    wanted[f"{split}/{row.file}"] = target
            if wanted:
                if archive is None:
                    archive = download(f"{BASE_URL}/{code}/audio/{split}.tar.gz",
                                       self.cache / code / "audio" / f"{split}.tar.gz")
                extract_from_tar(archive, wanted)
            clips = [Clip(id=row.file, audio=wavs[row.file], text=row.text) for row in rows]
            packed = pack(clips, self.cache / "packed", f"{code}-{split}", self.options.pack_minutes * 60)
            for index, (audio, turns) in enumerate(packed, start=1):
                yield Item(dataset=self.name, id=f"{code}-{split}-{index:02d}", audio=audio,
                           language=code.split("_")[0], turns=turns, has_words=True, has_speakers=False)

    def _locate(self, code: str, split: str) -> Tuple[Path, Optional[Path], Optional[Path]]:
        """tsv file, folder with extracted audio (if any) and the local audio archive (if any)."""
        root = self.local_path()
        if root is None:
            return download(f"{BASE_URL}/{code}/{split}.tsv", self.cache / code / f"{split}.tsv"), None, None
        for folder in (root / code, root / "data" / code, root):
            tsv = folder / f"{split}.tsv"
            if tsv.is_file():
                archive = folder / "audio" / f"{split}.tar.gz"
                return tsv, folder / "audio" / split, archive if archive.is_file() else None
        raise self.error(f"no {code}/{split}.tsv in {root}")


__all__ = ["Fleurs", "parse_tsv", "select_rows"]
