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
from copy import deepcopy
from typing import Iterable, Optional, Tuple, Dict, List

from MAT.tools import ToolResult, ToolInput, Tool
from MAT.utils.config import Config


class NERResult(ToolResult):
    def __init__(self, *ner: Dict[str, List[Tuple[str, int, int]]]):
        self._ner = ner

    @property
    def ner(self):
        return deepcopy(self._ner)


class NERInput(ToolInput):

    def __init__(self, *text: str):
        self._text = text

    @property
    def text(self):
        return list(self._text)


class NERTool(Tool[NERInput, NERResult], ABC):
    @abstractmethod
    def process(self, origin_data: NERInput, config: Config) -> Optional[NERResult]:
        pass


from MAT.tools.ner.ner_gliner import NERGliner

__all__ = ["NERResult", "NERInput", "NERTool", "NERGliner"]
__all__.extend(["__all__"])
