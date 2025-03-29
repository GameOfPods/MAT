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
from typing import Iterable, Optional, Tuple, Union

from torch import Tensor
import numpy as np
import pydub

from MAT.tools import ToolResult, ToolInput, Tool
from MAT.utils.config import Config


class SpeakerIdentificationResult(ToolResult):
    def __init__(self, *speaker: str) -> None:
        self._speaker = speaker

    def get_speaker(self):
        return self._speaker


class SpeakerIdentificationInput(ToolInput):
    def __init__(self, *audio_files: Tuple[Union[Tensor, np.ndarray, pydub.AudioSegment], int]):
        self._audio_files = audio_files

    def get_audio_files(self):
        return self._audio_files


@property
def in_file(self) -> str:
    return self._in_file


class SpeakerIdentificationTool(Tool[SpeakerIdentificationInput, SpeakerIdentificationResult], ABC):
    @abstractmethod
    def process(self, origin_data: SpeakerIdentificationInput, config: Config) -> Optional[SpeakerIdentificationResult]:
        pass


from MAT.tools.speakeridentification.pyannote import SpeakerIdetificationPyannote
from MAT.tools.speakeridentification.speech_brain import SpeakerIdetificationSpeechBrain

__all__ = ["SpeakerIdentificationResult", "SpeakerIdentificationInput", "SpeakerIdentificationTool"]
__all__.extend(["SpeakerIdetificationPyannote", "SpeakerIdetificationSpeechBrain"])
__all__.extend(["__all__"])
