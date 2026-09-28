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

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, List, Set
from copy import copy

import pydub

from MAT.tools import ToolResult, ToolInput, Tool
from MAT.tools.speakeridentification import SpeakerIdentificationTool, SpeakerIdentificationInput
from MAT.utils.config import Config


class DiarizationResult(ToolResult):
    def __init__(self, diarization: Dict[str, List[Tuple[float, float]]] = None) -> None:
        from collections import defaultdict
        self._diarization: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for k, v in (diarization.items() if diarization is not None else {}.items()):
            for f, t in v:
                self.add_diarization(speaker=k, f=f, t=t)

    def add_diarization(self, speaker: str, f: float, t: float) -> None:
        self._diarization[speaker].append((f, t))

    def get_diarization(self, speaker: str) -> List[Tuple[float, float]]:
        return [x for x in (copy(self._diarization[speaker]) if speaker in self._diarization else [])]

    @property
    def speaker(self) -> Set[str]:
        return set(self._diarization.keys())

    def speaker_matching(self, identifier: SpeakerIdentificationTool, config: Config,
                         audio: pydub.AudioSegment, seconds: Optional[float] = None) -> "DiarizationResult":
        """Asks the identifier about all speakers at once (one model load), with up to `seconds` of each."""
        from MAT.utils.audio import speaker_clip

        speakers = sorted(self.speaker)
        clips = [speaker_clip(audio, self.get_diarization(speaker=speaker), seconds) for speaker in speakers]
        m = identifier.process(origin_data=SpeakerIdentificationInput(*[(c, c.frame_rate) for c in clips]),
                               config=config)
        names = m.get_speaker()
        if len(names) != len(speakers):
            raise Exception(f"Speaker matching gave {len(names)} answers for {len(speakers)} speakers")
        final_speaker: Dict[str, List[Tuple[float, float]]] = {}
        for speaker, found in zip(speakers, names):
            # None means no match (or no gold labels given). Keep the diarizer label then, otherwise all
            # unmatched speakers end up under the same None key and overwrite each other.
            # If two diarizer speakers match the same gold speaker their segments get merged.
            name = speaker if found is None else found
            final_speaker.setdefault(name, []).extend(self.get_diarization(speaker=speaker))
        return DiarizationResult(diarization=final_speaker)

    def to_dict(self):
        return {s: self.get_diarization(speaker=s) for s in self.speaker}


class DiarizerInput(ToolInput):
    def __init__(self, in_file: str):
        self._in_file = in_file

    @property
    def in_file(self) -> str:
        return self._in_file


class DiarizationTool(Tool[DiarizerInput, DiarizationResult], ABC):
    @abstractmethod
    def process(self, origin_data: DiarizerInput, config: Config) -> Optional[DiarizationResult]:
        pass


from MAT.registry import load_optional

load_optional("MAT.tools.diarizators.nemo", slot="diarizer", name="sortformer", extra="sortformer")
load_optional("MAT.tools.diarizators.nemo", slot="diarizer", name="sortformer-streaming", extra="sortformer")
load_optional("MAT.tools.diarizators.pyannote", slot="diarizer", name="pyannote-diarization", extra="pyannote")
# runs in its own environment, `MAT external install diarizen` builds it
load_optional("MAT.tools.diarizators.diarizen", slot="diarizer", name="diarizen", extra="diarizen")

__all__ = ["DiarizationResult", "DiarizerInput", "DiarizationTool"]
