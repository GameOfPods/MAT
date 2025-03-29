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
from typing import Iterable, Optional, Tuple

from MAT.tools import ToolResult, ToolInput, Tool
from MAT.utils.config import Config


class SummaryResult(ToolResult):
    def __init__(self, *text: str):
        self._text = text

    @property
    def text(self) -> Iterable[str]:
        return self._text


class SummaryInput(ToolInput):

    def __init__(self, *text: str):
        self._text = text

    @property
    def text(self):
        yield from self._text


class SummaryTool(Tool[SummaryInput, SummaryResult], ABC):
    @abstractmethod
    def process(self, origin_data: SummaryInput, config: Config) -> Optional[SummaryResult]:
        pass


from MAT.tools.summary.llm import SummaryLLM

__all__ = ["SummaryResult", "SummaryInput", "SummaryTool", "SummaryLLM"]
__all__.extend(["__all__"])
