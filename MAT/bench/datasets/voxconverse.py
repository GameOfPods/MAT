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
VoxConverse (https://github.com/joonson/voxconverse, annotations v0.3): diarization of YouTube debates and shows.

Audio comes as voxconverse_<split>_wav.zip from the VGG website, RTTMs from the GitHub repository (<split>/<id>.rttm).
Only the picked files are read out of the zip with range requests.
"""
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Tuple

from pydantic import Field

from MAT.bench.data import Item, safe_name
from MAT.bench.datasets.base import Dataset, DatasetOptions, is_junk, register
from MAT.bench.download import download, extract_member, open_zip

AUDIO_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_{split}_wav.zip"
RTTM_URL = "https://raw.githubusercontent.com/joonson/voxconverse/master/{split}/{id}.rttm"


class VoxConverseOptions(DatasetOptions):
    split: Literal["dev", "test"] = Field("test", description="dev or test.")
    limit: Optional[int] = Field(5, ge=-1, description="Number of files, most are 5 to 20 minutes long. 0 or -1 "
                                                       "uses all of them.")
    files: List[str] = Field(default_factory=list, description="File ids (like aepyx) instead of the first files.")


def parse_rttm(text: str) -> Dict[str, List[Tuple[float, float]]]:
    segments: Dict[str, List[Tuple[float, float]]] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start, duration = float(parts[3]), float(parts[4])
        segments.setdefault(parts[7], []).append((start, start + duration))
    return segments


@register
class VoxConverse(Dataset):
    type = "voxconverse"
    description = "VoxConverse debates, news and shows with 1 to 20+ speakers. DER and speaker count."
    license = "CC-BY-4.0, copyright of the audio stays with the video owners (research use)"
    size = "only the picked files, 20 to 200 MB each"
    note = "Mostly English, many short turns and overlapping speech. Harder than a podcast with a few hosts."
    Options = VoxConverseOptions

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._zip = None

    def items(self) -> Iterator[Item]:
        split = self.options.split
        root = self.local_path()
        ids = list(self.options.files) or self._ids(root, split)
        for file_id in ids[:self.limit]:
            rttm = self._rttm(root, split, file_id)
            yield Item(dataset=self.name, id=safe_name(file_id), audio=self._audio(root, split, file_id),
                       speaker_segments=parse_rttm(rttm.read_text(encoding="utf-8")),
                       has_words=False, has_speakers=True)

    def _ids(self, root: Optional[Path], split: str) -> List[str]:
        if root is not None:
            folder = root / split if (root / split).is_dir() else root
            ids = sorted(p.stem for p in folder.rglob("*.rttm") if not is_junk(p.as_posix()))
            if not ids:
                raise self.error(f"no .rttm files in {folder}")
            return ids
        listing = self.cache / split / "files.txt"
        if not listing.exists():
            names = [Path(n).stem for n in self._archive(split).namelist() if n.endswith(".wav") and not is_junk(n)]
            listing.parent.mkdir(parents=True, exist_ok=True)
            listing.write_text("\n".join(sorted(names)) + "\n", encoding="utf-8")
        return listing.read_text(encoding="utf-8").split()

    def _rttm(self, root: Optional[Path], split: str, file_id: str) -> Path:
        if root is not None:
            found = self.find(root, f"{file_id}.rttm")
            if found is None:
                raise self.error(f"no {file_id}.rttm in {root}")
            return found
        return download(RTTM_URL.format(split=split, id=file_id), self.cache / split / "rttm" / f"{file_id}.rttm")

    def _audio(self, root: Optional[Path], split: str, file_id: str) -> Path:
        if root is not None:
            found = self.find(root, f"{file_id}.wav")
            if found is not None:
                return found
        target = self.cache / split / "audio" / f"{file_id}.wav"
        if not target.exists():
            archive = self._archive(split)
            members = {Path(n).name: n for n in archive.namelist() if not is_junk(n)}
            if f"{file_id}.wav" not in members:
                raise self.error(f"{file_id}.wav isn't in the VoxConverse {split} audio")
            extract_member(archive, members[f"{file_id}.wav"], target)
        return target

    def _archive(self, split: str):
        if self._zip is None:
            self._zip = open_zip(AUDIO_URL.format(split=split))
        return self._zip


__all__ = ["VoxConverse", "parse_rttm"]
