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
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict as dataclass_as_dict
from typing import TypeVar, Generic, Iterable, Callable, Dict, Any, Type
from time import perf_counter_ns
from datetime import timedelta

from MAT.tools import ToolResult
from MAT.utils import get_all_concrete_subclasses
from MAT.utils.config import Config, ConfigClass


@dataclass
class PipelineResult(ToolResult):
    def as_dict(self):
        return dataclass_as_dict(self)


@dataclass
class PipelineStepResult:
    name: str
    data: Any


@dataclass
class PipelineStepInput:
    file: str
    config: Config
    previous_results: Dict[str, PipelineStepResult]


T_out = TypeVar("T_out", bound=PipelineResult)


class Pipeline(Generic[T_out], ConfigClass, ABC):
    _LOGGER = logging.getLogger(__name__)

    @classmethod
    def name(cls):
        return cls.__name__

    @classmethod
    @abstractmethod
    def accept(cls, f: str) -> bool:
        raise NotImplemented()

    def process(self, file: str, config: Config) -> T_out:
        step_results: Dict[str, PipelineStepResult] = {}
        self.__class__._LOGGER.info(f"Running pipeline {self.__class__.__name__} on {file}")
        all_steps = list(self._get_steps())
        for i, step in enumerate(all_steps):
            t1 = perf_counter_ns() / 1e+6
            res = step(PipelineStepInput(file=file, config=config, previous_results=step_results))
            step_results[res.name] = res
            t2 = perf_counter_ns() / 1e+6
            self.__class__._LOGGER.info(f"Step {i + 1}/{len(all_steps)} done: {res.name} in "
                                        f"{f'{t2 - t1:.3f}ms' if t2 - t1 < 1000 else str(timedelta(milliseconds=int(t2 - t1)))}")
        return self._finalize_result(step_results=step_results)

    @abstractmethod
    def _get_steps(self) -> Iterable[Callable[[PipelineStepInput], PipelineStepResult]]:
        pass

    @abstractmethod
    def _finalize_result(self, step_results: Dict[str, PipelineStepResult]) -> T_out:
        raise NotImplemented

    @classmethod
    def get_pipelines(cls, f: str) -> Iterable[Type["Pipeline"]]:
        return [x for x in get_all_concrete_subclasses(cls=cls) if x.accept(f=f)]


from MAT.pipelines.Podcast import *
from MAT.pipelines.Podcast import __all__ as podcast_all

from MAT.pipelines.Book import *
from MAT.pipelines.Book import __all__ as book_all

__all__ = ["PipelineResult", "Pipeline"] + podcast_all + book_all
