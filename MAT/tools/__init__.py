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
from typing import Generic, TypeVar, Iterable, Optional

from MAT.utils.config import ConfigClass, Config


class ToolResult(ABC):
    pass


class ToolInput(ABC):
    pass


T_out = TypeVar("T_out", bound=ToolResult)
T_in = TypeVar("T_in", bound=ToolInput)


class Tool(Generic[T_in, T_out], ConfigClass, ABC):

    @abstractmethod
    def process(self, origin_data: T_in, config: Config) -> Optional[T_out]:
        pass


from MAT.tools.ner import *
from MAT.tools.ner import __all__ as ner_all

from MAT.tools.summary import *
from MAT.tools.summary import __all__ as summary_all

from MAT.tools.diarizators import *
from MAT.tools.diarizators import __all__ as diarizators_all

from MAT.tools.speakeridentification import *
from MAT.tools.speakeridentification import __all__ as speakeridentification_all

from MAT.tools.transcriptors import *
from MAT.tools.transcriptors import __all__ as transcription_all

from MAT.tools.text_splitter import *
from MAT.tools.text_splitter import __all__ as splitter_all

__all__ = ["Tool", "ToolInput", "ToolResult"]
__all__ += ner_all + summary_all + speakeridentification_all + transcription_all + splitter_all

del ner_all
del summary_all
