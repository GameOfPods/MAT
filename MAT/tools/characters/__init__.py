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
Judges for the character list: are two names one person, and which character does an ambiguous short name mean in
one sentence. The candidates come from MAT/utils/characters.py, a judge only answers what it's asked.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from MAT.tools import Tool, ToolInput, ToolResult
from MAT.utils.config import Config


@dataclass
class PairQuestion:
    id: int
    a: str
    b: str
    sentences_a: List[str]
    sentences_b: List[str]
    together: List[str] = field(default_factory=list)


@dataclass
class MentionQuestion:
    id: int
    name: str
    sentence: str
    options: List[str]


class CharacterJudgeInput(ToolInput):
    def __init__(self, pairs: List[PairQuestion], mentions: List[MentionQuestion], language: Optional[str] = None):
        self.pairs = pairs
        self.mentions = mentions
        self.language = language


class CharacterJudgeResult(ToolResult):
    def __init__(self, same: Optional[Dict[int, str]] = None, mentions: Optional[Dict[int, str]] = None):
        # pair id -> the sentence that shows it, only for pairs that are one person
        self.same = dict(same or {})
        # mention id -> the option it belongs to, only where the judge was sure
        self.mentions = dict(mentions or {})


class CharacterJudgeTool(Tool[CharacterJudgeInput, CharacterJudgeResult], ABC):
    @abstractmethod
    def process(self, origin_data: CharacterJudgeInput, config: Config) -> Optional[CharacterJudgeResult]:
        pass


from MAT.registry import load_optional  # noqa: E402

load_optional("MAT.tools.characters.llm", slot="characters", name="llm-characters", extra="llm")

__all__ = ["PairQuestion", "MentionQuestion", "CharacterJudgeInput", "CharacterJudgeResult", "CharacterJudgeTool"]
