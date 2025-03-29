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
from typing import Iterable, Optional, Tuple, Dict, List, Set
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
                         audio: pydub.AudioSegment) -> "DiarizationResult":
        final_speaker = {}
        for speaker in self.speaker:
            a_t = pydub.AudioSegment.empty()
            for fr, to in self.get_diarization(speaker=speaker):
                a_t += audio[fr * 1000:to * 1000]
            m = identifier.process(origin_data=SpeakerIdentificationInput((a_t, a_t.frame_rate)), config=config)
            if len(m.get_speaker()) != 1:
                raise Exception(f"Diarization failed for {speaker}")
            final_speaker[m.get_speaker()[0]] = [x for x in self.get_diarization(speaker=speaker)]
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


from MAT.tools.diarizators.nemo import DiarizerNEMO

__all__ = ["DiarizationResult", "DiarizerInput", "DiarizationTool", "DiarizerNEMO"]
__all__.extend(["__all__"])
