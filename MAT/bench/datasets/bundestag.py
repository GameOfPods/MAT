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
ASR Bundestag (https://opendata.iisys.de/dataset/asr-bundestag/, arXiv 2302.06008): German parliament speech.

asr_bundestag_<subset>.zip has Kaldi style lists per split (<split>/text: "utterance words...", <split>/wav.scp:
"utterance path") and the snippets in wavs/. The clean zip is 59 GB, so single files are read with range requests.
"""
import re
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Tuple

from pydantic import Field

from MAT.bench.data import Item
from MAT.bench.datasets.base import Clip, Dataset, DatasetOptions, pack, register
from MAT.bench.download import extract_member, open_zip

URL = "https://opendata.iisys.de/opendata/Datasets/Bundestag/asr_bundestag_{subset}.zip"


class BundestagOptions(DatasetOptions):
    subset: Literal["clean", "dirty"] = Field("clean", description="clean or dirty.")
    split: str = Field("test", description="Folder with the lists in the archive: test, train_dev or train_nodev.")
    limit: Optional[int] = Field(200, ge=-1, description="Number of snippets, 0 or -1 uses all of them. They're sorted "
                                                         "by session and time, so snippets of one speech end up next "
                                                         "to each other.")
    pack_minutes: float = Field(10, gt=0, description="Snippets are joined into files of up to this many minutes.")


def parse_kaldi(text: str) -> Dict[str, str]:
    # the key ends at the first space or tab, ASR Bundestag uses tabs
    entries = {}
    for line in text.splitlines():
        parts = line.strip().split(maxsplit=1)
        if parts:
            entries[parts[0]] = parts[1].strip() if len(parts) > 1 else ""
    return entries


def utterance_key(utterance: str) -> Tuple:
    # 7546297_0_522_3: session, part and position, compared as numbers where possible
    return tuple((0, int(part), "") if part.isdigit() else (1, 0, part) for part in re.split(r"[_-]", utterance))


def wav_name(utterance: str, scp_value: Optional[str]) -> str:
    for token in (scp_value or "").split():
        if token.endswith(".wav"):
            return Path(token).name
    return f"{utterance}.wav"


@register
class Bundestag(Dataset):
    type = "bundestag"
    description = "ASR Bundestag German parliament speech. WER only, snippets get joined into longer files."
    license = "Bundestag terms of use: no commercial use or advertising"
    size = "only the picked snippets (a few hundred KB each), the clean zip itself is 59 GB"
    note = ("References are lowercase without punctuation and write numbers as words (vierhundertsechzig). Whisper "
            "writes digits, so WER comes out higher than the real error rate. Formal speech, one speaker at a time.")
    Options = BundestagOptions

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._zip = None

    def items(self) -> Iterator[Item]:
        subset, split = self.options.subset, self.options.split
        folder = self._folder()
        prefix = f"asr_bundestag_{subset}"
        if folder is not None:
            text_file, scp_file = folder / split / "text", folder / split / "wav.scp"
        else:
            base = self.cache / subset
            text_file, scp_file = base / split / "text", base / split / "wav.scp"
            for target in (text_file, scp_file):
                if not target.exists():
                    self._extract(f"{prefix}/{split}/{target.name}", target)
        if not text_file.is_file():
            raise self.error(f"{text_file} not found")
        texts = parse_kaldi(text_file.read_text(encoding="utf-8"))
        scp = parse_kaldi(scp_file.read_text(encoding="utf-8")) if scp_file.is_file() else {}
        clips = []
        for utterance in sorted(texts, key=utterance_key)[:self.limit]:
            name = wav_name(utterance, scp.get(utterance))
            if folder is not None and (folder / "wavs" / name).is_file():
                audio = folder / "wavs" / name
            else:
                audio = self.cache / subset / "wavs" / name
                if not audio.exists():
                    self._extract(f"{prefix}/wavs/{name}", audio)
            clips.append(Clip(id=utterance, audio=audio, text=texts[utterance]))
        packed = pack(clips, self.cache / "packed", f"{subset}-{split}", self.options.pack_minutes * 60)
        for audio, turns in packed:
            # the file name has a key of the picked snippets, so another limit doesn't reuse old results
            yield Item(dataset=self.name, id=audio.stem, audio=audio, language="de",
                       turns=turns, has_words=True, has_speakers=False)

    def _folder(self) -> Optional[Path]:
        """The extracted dataset folder, None when files come from a zip (local or remote)."""
        root = self.local_path()
        if root is None or root.is_file():
            return None
        candidate = root / f"asr_bundestag_{self.options.subset}"
        return candidate if candidate.is_dir() else root

    def _extract(self, member: str, target: Path) -> None:
        if self._zip is None:
            root = self.local_path()
            self._zip = open_zip(root if root is not None and root.is_file()
                                 else URL.format(subset=self.options.subset))
        try:
            extract_member(self._zip, member, target)
        except KeyError:
            raise self.error(f"{member} isn't in the archive")


__all__ = ["Bundestag", "parse_kaldi", "utterance_key", "wav_name"]
