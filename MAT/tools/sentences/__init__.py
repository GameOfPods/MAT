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
Sentences of a transcript. Speaker turns are a bad unit for everything that reads text: a turn can be one word or
three minutes long, and whisper sometimes writes a whole episode in lowercase without punctuation.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from MAT.tools import Tool, ToolInput, ToolResult
from MAT.utils.config import Config


class SentenceInput(ToolInput):
    def __init__(self, texts: Sequence[str], language: Optional[str] = None):
        self.texts = list(texts)
        self.language = language


class SentenceResult(ToolResult):
    def __init__(self, sentences: List[List[str]]):
        # per text its sentences. Joined together they give the text back exactly, spaces included
        self.sentences = sentences


class SentenceTool(Tool[SentenceInput, SentenceResult], ABC):
    @abstractmethod
    def process(self, origin_data: SentenceInput, config: Config) -> Optional[SentenceResult]:
        pass


from MAT.registry import load_optional  # noqa: E402

load_optional("MAT.tools.sentences.sat", slot="sentences", name="sat", extra="sentences")

__all__ = ["SentenceInput", "SentenceResult", "SentenceTool"]
