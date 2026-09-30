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
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
import pydub

if TYPE_CHECKING:  # torch is only needed once a model runs, importing it here slowed down `MAT --help`
    from torch import Tensor

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


class SpeakerIdentificationTool(Tool[SpeakerIdentificationInput, SpeakerIdentificationResult], ABC):
    def can_match(self, config: Config) -> bool:
        """Whether this backend can match anything at all. When it can't (no gold labels), the pipeline skips the
        step instead of concatenating hours of audio for an answer that is None anyway."""
        return True

    @abstractmethod
    def process(self, origin_data: SpeakerIdentificationInput, config: Config) -> Optional[SpeakerIdentificationResult]:
        pass


from MAT.registry import load_optional

load_optional("MAT.tools.speakeridentification.pyannote", slot="identifier", name="pyannote", extra="pyannote")

__all__ = ["SpeakerIdentificationResult", "SpeakerIdentificationInput", "SpeakerIdentificationTool"]
