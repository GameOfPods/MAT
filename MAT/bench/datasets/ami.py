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
AMI meeting corpus (https://groups.inf.ed.ac.uk/ami/corpus/).

- audio: amicorpus/<meeting>/audio/<meeting>.Mix-Headset.wav from the Edinburgh mirror
- words: words/<meeting>.<speaker letter>.words.xml in ami_public_manual_1.6.2.zip
- DER reference, meeting lists and scoring ranges: BUT's AMI-diarization-setup (only_words RTTMs, uems, lists)
"""
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Tuple
from xml.etree import ElementTree

from pydantic import Field

from MAT.bench.data import Item, Turn, safe_name
from MAT.bench.datasets.base import Dataset, DatasetOptions, register
from MAT.bench.datasets.voxconverse import parse_rttm
from MAT.bench.download import download

AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{meeting}/audio/{meeting}.Mix-Headset.wav"
WORDS_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
SETUP_URL = "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main"

Word = Tuple[float, float, str]


class AmiOptions(DatasetOptions):
    split: Literal["dev", "test", "train"] = Field("test", description="dev, test or train (BUT's lists).")
    limit: Optional[int] = Field(2, ge=1, description="Number of meetings, each 15 to 50 minutes long.")
    meetings: List[str] = Field(default_factory=list, description="Meeting ids (like ES2004a) instead of the "
                                                                   "first ones of the split.")
    turn_gap: float = Field(1.0, ge=0, description="Words of one speaker less than this many seconds apart form "
                                                   "one reference line.")


def parse_words(xml: bytes) -> List[Word]:
    """Timed words of one speaker. Punctuation and untimed entries are skipped."""
    words = []
    for element in ElementTree.fromstring(xml).iter("w"):
        start, end, text = element.get("starttime"), element.get("endtime"), (element.text or "").strip()
        if element.get("punc") == "true" or not text or start is None or end is None:
            continue
        try:
            words.append((float(start), float(end), text))
        except ValueError:
            continue
    return words


def words_to_turns(words_by_speaker: Dict[str, List[Word]], gap: float) -> List[Turn]:
    turns = []
    for speaker, words in words_by_speaker.items():
        current: Optional[List] = None
        for start, end, text in sorted(words):
            if current is not None and start - current[1] <= gap:
                current[1] = max(current[1], end)
                current[2].append(text)
                continue
            if current is not None:
                turns.append(Turn(current[0], current[1], (speaker,), " ".join(current[2])))
            current = [start, end, [text]]
        if current is not None:
            turns.append(Turn(current[0], current[1], (speaker,), " ".join(current[2])))
    return sorted(turns, key=lambda t: (t.start, t.end))


@register
class Ami(Dataset):
    type = "ami"
    description = "AMI meetings with 3 to 5 speakers, mixed headset audio. WER, cpWER, DER and speaker count."
    license = "CC-BY-4.0"
    size = "30 to 90 MB per meeting plus 23 MB of annotations"
    note = ("English meetings with many short backchannels. WER is scored over the whole meeting, so it's higher "
            "than numbers from papers that score per utterance.")
    Options = AmiOptions

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._words_zip: Optional[zipfile.ZipFile] = None

    def items(self) -> Iterator[Item]:
        split = self.options.split
        root = self.local_path()
        meetings = list(self.options.meetings) or self._meetings(root, split)
        for meeting in meetings[:self.options.limit]:
            words = self._words(root, meeting)
            if not words:
                raise self.error(f"no word annotations for {meeting}")
            segments, start, end = self._diarization(root, split, meeting)
            yield Item(dataset=self.name, id=safe_name(meeting), audio=self._audio(root, meeting), language="en",
                       turns=words_to_turns(words, self.options.turn_gap), speaker_segments=segments,
                       has_words=True, has_speakers=True, start=start, end=end)

    def _setup_file(self, root: Optional[Path], relative: str) -> Optional[Path]:
        if root is not None and (root / relative).is_file():
            return root / relative
        return download(f"{SETUP_URL}/{relative}", self.cache / "setup" / relative)

    def _meetings(self, root: Optional[Path], split: str) -> List[str]:
        return self._setup_file(root, f"lists/{split}.meetings.txt").read_text(encoding="utf-8").split()

    def _words(self, root: Optional[Path], meeting: str) -> Dict[str, List[Word]]:
        folder = None if root is None else root / "words"
        if folder is not None and folder.is_dir():
            return {p.name.split(".")[1]: parse_words(p.read_bytes())
                    for p in sorted(folder.glob(f"{meeting}.*.words.xml"))}
        if self._words_zip is None:
            local = None if root is None else next(iter(sorted(root.glob("ami_public_manual*.zip"))), None)
            self._words_zip = zipfile.ZipFile(local or download(WORDS_URL, self.cache / Path(WORDS_URL).name))
        prefix = f"words/{meeting}."
        return {name.split(".")[1]: parse_words(self._words_zip.read(name))
                for name in sorted(self._words_zip.namelist())
                if name.startswith(prefix) and name.endswith(".words.xml")}

    def _diarization(self, root: Optional[Path], split: str,
                     meeting: str) -> Tuple[Dict[str, List[Tuple[float, float]]], Optional[float], Optional[float]]:
        rttm = self._setup_file(root, f"only_words/rttms/{split}/{meeting}.rttm")
        uem = self._setup_file(root, f"uems/{split}/{meeting}.uem")
        start = end = None
        parts = uem.read_text(encoding="utf-8").split()
        if len(parts) >= 4:
            start, end = float(parts[2]), float(parts[3])
        return parse_rttm(rttm.read_text(encoding="utf-8")), start, end

    def _audio(self, root: Optional[Path], meeting: str) -> Path:
        name = f"{meeting}.Mix-Headset.wav"
        if root is not None:
            found = self.find(root, name)
            if found is not None:
                return found
        return download(AUDIO_URL.format(meeting=meeting), self.cache / "audio" / name)


__all__ = ["Ami", "parse_words", "words_to_turns"]
