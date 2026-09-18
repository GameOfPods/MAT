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
Names speakers from what is said in the transcript ("Danke, Alex"), for speakers that gold label clips didn't match.

Gold clips compare voices, this reads the conversation. They answer different questions, so both can run: the
identifier goes first and whatever it matched stays untouched.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from MAT.tools import Tool, ToolInput, ToolResult
from MAT.utils.config import Config


@dataclass(frozen=True)
class SpeakerName:
    speaker: str
    name: str
    # the transcript line the name is based on, so a wrong name can be traced back
    evidence: str = ""


class SpeakerNamingResult(ToolResult):
    def __init__(self, names: Optional[List[SpeakerName]] = None):
        self._names = list(names or [])

    @property
    def names(self) -> List[SpeakerName]:
        return list(self._names)

    def as_dict(self) -> Dict[str, str]:
        return {found.speaker: found.name for found in self._names}


@dataclass
class SpeakerNamingInput(ToolInput):
    # transcript lines in MAT's format: "speaker [start - end]: text"
    lines: List[str] = field(default_factory=list)
    speakers: List[str] = field(default_factory=list)
    language: Optional[str] = None


class SpeakerNamingTool(Tool[SpeakerNamingInput, SpeakerNamingResult], ABC):
    @abstractmethod
    def process(self, origin_data: SpeakerNamingInput, config: Config) -> Optional[SpeakerNamingResult]:
        pass


from MAT.registry import load_optional  # noqa: E402

load_optional("MAT.tools.speakernaming.llm", slot="namer", name="llm-names", extra="llm")

__all__ = ["SpeakerName", "SpeakerNamingResult", "SpeakerNamingInput", "SpeakerNamingTool"]
