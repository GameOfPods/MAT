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
from typing import Optional, List, Set, TYPE_CHECKING
from dataclasses import dataclass

from MAT.tools import ToolResult, ToolInput, Tool
from MAT.utils.config import Config

if TYPE_CHECKING:
    from MAT.tools.diarizators import DiarizationResult


@dataclass
class WordTuple:
    start: Optional[float]
    end: Optional[float]
    word: Optional[str]


@dataclass
class WordTupleSpeaker:
    word: WordTuple
    speaker: Set[str]


class TranscriptionResult(ToolResult):
    def __init__(self, word_timings: List[WordTuple] = None, language: str = None, duration_after_vad: float = None,
                 duration: float = None):
        self._word_timings: Optional[List[WordTuple]] = word_timings
        self._language = language
        self._duration_after_vad = duration_after_vad
        self._duration = duration

    @property
    def word_timings(self) -> Optional[List[WordTuple]]:
        return self._word_timings

    @property
    def language(self) -> str:
        return self._language

    @property
    def duration_after_vad(self) -> float:
        return self._duration_after_vad

    @property
    def duration(self) -> float:
        return self._duration


class TranscriptionInput(ToolInput):
    def __init__(self, input_file: str):
        self._input_file = input_file

    @property
    def input_file(self) -> str:
        return self._input_file


class TransciptionTool(Tool[TranscriptionInput, TranscriptionResult], ABC):
    @abstractmethod
    def process(self, origin_data: TranscriptionInput, config: Config) -> Optional[TranscriptionResult]:
        pass


class TranscribeDiarizeResult(TranscriptionResult):
    def __init__(self, diarization: "DiarizationResult" = None, **kwargs):
        super().__init__(**kwargs)
        self._diarization = diarization

    @property
    def diarization(self) -> Optional["DiarizationResult"]:
        return self._diarization


class TranscribeDiarizeTool(TransciptionTool, ABC):
    """
    A model that transcribes and diarizes in one pass. Registered in the transcriber slot, the podcast pipeline then
    takes the diarization from it and doesn't run the diarizer.
    """

    @abstractmethod
    def process(self, origin_data: TranscriptionInput, config: Config) -> Optional[TranscribeDiarizeResult]:
        pass


from MAT.registry import load_optional

load_optional("MAT.tools.transcriptors.whisper", slot="transcriber", name="whisper", extra="whisper")

__all__ = ["TranscriptionResult", "TranscriptionInput", "TransciptionTool", "TranscribeDiarizeResult",
           "TranscribeDiarizeTool", "WordTuple", "WordTupleSpeaker"]
