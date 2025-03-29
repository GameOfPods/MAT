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
import sys
import os
import uuid
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from copy import deepcopy
import logging
from typing import Dict, Any, List, Type, Union

from MAT.utils import get_all_concrete_subclasses


class ConfigElement:
    def __init__(self, default_value: Any, argparse_kwargs: Dict[str, Any]):
        self._default_value = default_value
        self._argparse_kwargs = argparse_kwargs
        if "dest" in self._argparse_kwargs:
            del self._argparse_kwargs["dest"]
        if "default" in self._argparse_kwargs:
            del self._argparse_kwargs["default"]

    @property
    def default_value(self) -> Any:
        return self._default_value

    @property
    def argparse_kwargs(self) -> Dict[str, Any]:
        return self._argparse_kwargs

    def __copy__(self):
        return ConfigElement(self._default_value, deepcopy(self._argparse_kwargs))


class ConfigClass(ABC):

    @classmethod
    def config_name(cls) -> str:
        return cls.__name__.lower()

    @classmethod
    @abstractmethod
    def config_keys(cls) -> Dict[str, ConfigElement]:
        raise NotImplementedError()


class Config:
    _LOGGER = logging.getLogger(__name__)

    def __init__(self):
        self._work_dir = os.path.join(os.getcwd(), f".MAT.{uuid.uuid4()}")
        subclasses: List[Type[ConfigClass]] = get_all_concrete_subclasses(ConfigClass)
        self._config_builder: Dict[str, Dict[str, ConfigElement]] = {}
        self._config: Dict[str, Dict[str, Any]] = {}
        for config in subclasses:
            if "_" in config.config_name():
                raise ValueError(f"Config class {config.config_name()} can not contain underscores")
            if any("_" in x for x in config.config_keys().keys()):
                raise ValueError(f"Config keys cannot not contain underscores [{config.config_keys().keys()}]")
            if config.config_name() in self._config_builder:
                self.__class__._LOGGER.error(f"Config with name {config.config_name()} already exists. Check code.")
                sys.exit(1)
            self._config_builder[config.config_name()] = deepcopy(config.config_keys())
            self._config[config.config_name()] = {x: y.default_value for x, y in config.config_keys().items()}
        return

    def parse_config(self, config: Dict[str, Dict[str, Any]]):
        for key, value in config.items():
            if key in self._config:
                for config_key, config_value in value.items():
                    self._config[key][config_key] = config_value

    def create_argparse(self, argparse: ArgumentParser):
        for config_name, config_builder in self._config_builder.items():
            for config_key, config_value in config_builder.items():
                kwargs = config_value.argparse_kwargs
                if "metavar" not in kwargs and kwargs.get("action", "") not in ("store_false", "store_true"):
                    kwargs["metavar"] = config_key.upper()
                argparse.add_argument(f'--{config_name}_{config_key}',
                                      default=config_value.default_value, dest=f"{config_name}_{config_key}",
                                      **kwargs)

    def parse_argparse(self, namespace: Namespace):
        for key, value in namespace.__dict__.items():
            try:
                config_name, config_key = key.split("_")
                if config_name in self._config:
                    if config_key in self._config[config_name]:
                        self._config[config_name][config_key] = value
            except ValueError:
                pass

    @property
    def config(self) -> Dict[str, Dict[str, Any]]:
        return deepcopy(self._config)

    def get_config(self, key: Union[ConfigClass, Type[ConfigClass]]) -> dict:
        return deepcopy(self._config[key.config_name()])

    def set_work_directory(self, work_directory: str):
        self._work_dir = work_directory

    @property
    def work_directory(self):
        return self._work_dir

    def __str__(self):
        return self._config.__str__()

    def pformat(self):
        from pprint import pformat
        return pformat(self._config)
