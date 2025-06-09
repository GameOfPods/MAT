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
from enum import Enum, auto as enum_auto
from typing import List
from uuid import uuid4


class EmitStyle(Enum):
    CREATE = enum_auto()
    ENTER = enum_auto()
    VALUE_CHANGE = enum_auto()
    EXIT = enum_auto()


class ProgressHandler(ABC):
    @abstractmethod
    def emit(self, bar: "Progress", style: EmitStyle):
        raise NotImplementedError()


class Progress:
    _PROGRESS_BARS = {}
    _HANDLERS: List[ProgressHandler] = []

    @classmethod
    def get_progress_bar(cls, id: str):
        return cls._PROGRESS_BARS.get(id, None)

    @classmethod
    def add_handler(cls, handler: ProgressHandler):
        cls._HANDLERS.append(handler)

    @classmethod
    def _emit(cls, bar: "Progress", style: EmitStyle):
        for handler in cls._HANDLERS:
            handler.emit(bar, style)

    def __init__(self, id: str = None, name: str = None, desc: str = None, total: int = 100):
        self._id = id if id is not None else str(uuid4())
        i = 0
        new_id = self._id
        while new_id in Progress._PROGRESS_BARS:
            i += 1
            new_id = f"{self._id}-{i}"
        self._id = new_id
        self.__class__._PROGRESS_BARS[new_id] = self
        self._name = ""
        self._description = ""
        self._value = 0
        self.total = total

        self.name = name if name is not None else ""
        self.description = desc if desc is not None else ""
        self.__class__._emit(bar=self, style=EmitStyle.CREATE)

    @property
    def progress_id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self.__class__._emit(bar=self, style=EmitStyle.VALUE_CHANGE)

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = value
        self.__class__._emit(bar=self, style=EmitStyle.VALUE_CHANGE)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = max(0, min(self.total, value))
        self.__class__._emit(bar=self, style=EmitStyle.VALUE_CHANGE)

    @property
    def value_percent(self):
        return self._value / self.total

    def increment(self, n=1):
        self.value = self.value + n

    def __enter__(self) -> "Progress":
        self.__class__._emit(bar=self, style=EmitStyle.ENTER)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__class__._emit(bar=self, style=EmitStyle.EXIT)
