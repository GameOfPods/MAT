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
from typing import Optional, Iterable, Dict, List
from dataclasses import dataclass

from MAT.utils.config import Config
from MAT.tools import ToolResult, ToolInput, Tool


@dataclass
class SplitterResult(ToolResult):
    sentences: Iterable[str]
    words: Iterable[Dict[str, int]]


@dataclass
class SplitterInput(ToolInput):
    text: str


class SplitterTool(Tool[SplitterInput, SplitterResult], ABC):
    @abstractmethod
    def process(self, origin_data: SplitterInput, config: Config) -> Optional[SplitterResult]:
        pass


from MAT.tools.text_splitter.spacy import SplitterSpacy

__all__ = ["SplitterResult", "SplitterInput", "SplitterTool", "SplitterSpacy"]
__all__.extend(["__all__"])
