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
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Optional, Type
from time import perf_counter_ns
from datetime import timedelta

from MAT.tools import Tool, ToolResult
from MAT.utils import get_all_concrete_subclasses
from MAT.utils.config import Config, ConfigError, Configurable


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

    def data(self, step_name: str) -> Any:
        result = self.previous_results.get(step_name)
        return None if result is None else result.data


@dataclass(frozen=True)
class Slot:
    # "none" is allowed as backend, the step is skipped then
    optional: bool = False


class Pipeline(Configurable, ABC):
    _LOGGER = logging.getLogger(__name__)
    # slot name -> Slot. The pipeline's Options need a str field with the same name holding the backend name.
    slots: ClassVar[Dict[str, Slot]] = {}

    def __init__(self):
        # slot -> Tool.describe() of the backends that ran, ends up in the result
        self.models: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def name(cls) -> str:
        return cls.section

    @classmethod
    def all(cls) -> List[Type["Pipeline"]]:
        return get_all_concrete_subclasses(Pipeline)

    @classmethod
    @abstractmethod
    def accept(cls, f: str) -> bool:
        raise NotImplementedError()

    @classmethod
    def get_pipelines(cls, f: str) -> Iterable[Type["Pipeline"]]:
        return [x for x in cls.all() if x.accept(f=f)]

    @classmethod
    def check_slots(cls, config: Config) -> List[str]:
        from MAT import registry

        try:
            options = config.options(cls)
        except ConfigError as e:
            return [str(e)]
        errors = []
        for slot, spec in cls.slots.items():
            choice = getattr(options, slot)
            if choice == "none":
                if not spec.optional:
                    errors.append(f'[{cls.section}] {slot} can\'t be "none"')
                continue
            if registry.find(choice, slot) is None:
                installed = ", ".join(b.name for b in registry.backends(slot)) or "nothing"
                errors.append(f'[{cls.section}] unknown {slot} "{choice}". Installed: {installed}')
        return errors

    def backend(self, slot: str, config: Config) -> Optional[Tool]:
        from MAT import registry

        choice = getattr(config.options(self), slot)
        if choice == "none":
            if not self.slots[slot].optional:
                raise ConfigError(f'[{self.section}] {slot} can\'t be "none"')
            return None
        tool = registry.get(slot, choice).cls()
        try:
            self.models[slot] = tool.describe(config)
        except Exception as e:
            self._LOGGER.debug(f"Could not describe {slot} {choice}: {e}")
            self.models[slot] = {"backend": choice}
        return tool

    def process(self, file: str, config: Config) -> PipelineResult:
        self.models = {}
        step_results: Dict[str, PipelineStepResult] = {}
        self.__class__._LOGGER.info(f"Running pipeline {self.section} on {file}")
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
    def _finalize_result(self, step_results: Dict[str, PipelineStepResult]) -> PipelineResult:
        raise NotImplementedError()


from MAT.pipelines.Podcast import *
from MAT.pipelines.Podcast import __all__ as podcast_all

from MAT.pipelines.Book import *
from MAT.pipelines.Book import __all__ as book_all

__all__ = ["PipelineResult", "PipelineStepResult", "PipelineStepInput", "Pipeline", "Slot"] + podcast_all + book_all
