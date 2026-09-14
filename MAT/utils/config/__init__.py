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
Typed configuration.

Every pipeline and backend has an `Options` pydantic model and a config `section` (for example `whisper`). Values
come from the defaults, then a TOML file, then `--set section.key=value` on the command line. Keys are written with
dashes in files and on the command line (`beam-size`) and with underscores in Python (`options.beam_size`).
"""
import copy
import json
import logging
import math
import os
import re
import textwrap
import tomllib
import uuid
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

from pydantic import BaseModel, ConfigDict, ValidationError

_LOGGER = logging.getLogger(__name__)


def _kebab(name: str) -> str:
    return name.replace("_", "-")


class ConfigError(ValueError):
    pass


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", alias_generator=_kebab, populate_by_name=True, validate_default=True)


class Configurable:
    section: ClassVar[str] = ""
    description: ClassVar[str] = ""
    Options: ClassVar[Type[Options]] = Options


def parse_value(text: str) -> Any:
    # JSON covers numbers, true/false, null, lists and objects. Everything else is a plain string.
    try:
        return json.loads(text)
    except ValueError:
        return text


def parse_override(item: str) -> Tuple[str, str, Any]:
    key, sep, value = item.partition("=")
    section, dot, option = key.strip().partition(".")
    if not sep or not dot or not section or not option:
        raise ConfigError(f'Expected section.option=value, got "{item}"')
    return section, _kebab(option), parse_value(value.strip())


def configurables() -> Dict[str, Type[Configurable]]:
    import MAT.tools  # noqa: F401, loads the backends
    from MAT import registry
    from MAT.pipelines import Pipeline

    result: Dict[str, Type[Configurable]] = {pipeline.section: pipeline for pipeline in Pipeline.all()}
    for backend in registry.backends():
        if backend.name in result:
            raise RuntimeError(f"Config section [{backend.name}] is used twice")
        result[backend.name] = backend.cls
    return result


def _messages(section: str, error: ValidationError, options: Type[Options]) -> List[str]:
    messages = []
    known = ", ".join(field.alias or _kebab(name) for name, field in options.model_fields.items())
    for err in error.errors():
        location = ".".join(str(x) for x in err["loc"])
        if err["type"] == "extra_forbidden":
            messages.append(f'[{section}] unknown option "{location}". Options: {known}')
        else:
            messages.append(f"[{section}] {location}: {err['msg']}")
    return messages


class Config:
    def __init__(self, values: Optional[Dict[str, Dict[str, Any]]] = None, work_directory: Optional[str] = None):
        self._values: Dict[str, Dict[str, Any]] = {}
        for section, options in (values or {}).items():
            self._values[section] = {_kebab(k): copy.deepcopy(v) for k, v in options.items()}
        self._work_dir = work_directory or os.path.join(os.getcwd(), f".MAT.{uuid.uuid4()}")
        self._cache: Dict[str, Options] = {}

    @classmethod
    def load(cls, file: Optional[Union[str, os.PathLike]] = None, overrides: Sequence[str] = (),
             validate: bool = True) -> "Config":
        values: Dict[str, Dict[str, Any]] = {}
        if file:
            try:
                with open(file, "rb") as f:
                    data = tomllib.load(f)
            except FileNotFoundError:
                raise ConfigError(f"Config file {file} doesn't exist")
            except tomllib.TOMLDecodeError as e:
                raise ConfigError(f"Can't read config file {file}: {e}")
            for section, options in data.items():
                if not isinstance(options, dict):
                    raise ConfigError(f'"{section}" in {file} has to be a [section], not a single value')
                values[section] = {_kebab(k): v for k, v in options.items()}
        for item in overrides:
            section, option, value = parse_override(item)
            values.setdefault(section, {})[option] = value
        config = cls(values)
        if validate:
            config.validate()
        return config

    def set(self, section: str, option: str, value: Any) -> None:
        self._values.setdefault(section, {})[_kebab(option)] = value
        self._cache.pop(section, None)

    @property
    def values(self) -> Dict[str, Dict[str, Any]]:
        return copy.deepcopy(self._values)

    def validate(self) -> None:
        from MAT import registry
        from MAT.pipelines import Pipeline

        known = configurables()
        errors: List[str] = []
        for section, options in self._values.items():
            if section in known:
                try:
                    known[section].Options.model_validate(options)
                except ValidationError as e:
                    errors.extend(_messages(section, e, known[section].Options))
            elif registry.find(section) is not None:
                _LOGGER.warning(f"Ignoring config section [{section}], that backend isn't installed")
            else:
                errors.append(f"Unknown config section [{section}]. Known sections: {', '.join(sorted(known))}")
        if not errors:
            for pipeline in Pipeline.all():
                errors.extend(pipeline.check_slots(self))
        if errors:
            raise ConfigError("\n".join(errors))

    def options(self, configurable: Union[Configurable, Type[Configurable]]) -> Any:
        cls = configurable if isinstance(configurable, type) else type(configurable)
        if cls.section not in self._cache:
            try:
                self._cache[cls.section] = cls.Options.model_validate(self._values.get(cls.section, {}))
            except ValidationError as e:
                raise ConfigError("\n".join(_messages(cls.section, e, cls.Options))) from e
        return self._cache[cls.section]

    def set_work_directory(self, work_directory: str) -> None:
        self._work_dir = work_directory

    @property
    def work_directory(self) -> str:
        return self._work_dir


_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _toml_key(key: str) -> str:
    return key if _BARE_KEY.match(key) else json.dumps(key)


def _literal_multiline_ok(value: str) -> bool:
    return ("\n" in value and "'''" not in value and not value.endswith("'")
            and all(c in "\n\t" or (ord(c) >= 32 and c != "\x7f") for c in value))


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"TOML can't store {value}")
        return repr(value)
    if isinstance(value, str):
        # long prompts stay readable as multi-line literal strings
        return f"'''\n{value}'''" if _literal_multiline_ok(value) else json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(f"{_toml_key(str(k))} = {toml_value(v)}" for k, v in value.items() if v is not None)
        return "{" + items + "}"
    raise TypeError(f"Can't write {type(value).__name__} to TOML")


def render_sections(classes: Iterable[Type[Configurable]], config: Optional[Config] = None,
                    comments: bool = True) -> str:
    config = config or Config()
    lines: List[str] = []
    for cls in classes:
        if lines:
            lines.append("")
        if comments and cls.description:
            lines.extend(f"# {line}" for line in textwrap.wrap(cls.description, 100))
        lines.append(f"[{cls.section}]")
        values = config.options(cls).model_dump(by_alias=True, mode="json")
        for name, field in cls.Options.model_fields.items():
            key = field.alias or _kebab(name)
            if comments and field.description:
                lines.extend(f"# {line}" for line in textwrap.wrap(field.description, 100))
            value = values.get(key)
            # TOML has no null, unset options stay commented out
            lines.append(f"# {_toml_key(key)} =" if value is None else f"{_toml_key(key)} = {toml_value(value)}")
    return "\n".join(lines) + "\n"


def describe_options(options: Type[Options]) -> str:
    lines = []
    for name, field in options.model_fields.items():
        key = field.alias or _kebab(name)
        annotation = field.annotation
        type_name = annotation.__name__ if isinstance(annotation, type) else str(annotation).replace("typing.", "")
        default = field.get_default(call_default_factory=True)
        shown = json.dumps(default, ensure_ascii=False, default=str)
        if len(shown) > 60:
            shown = shown[:57] + "..."
        lines.append(f"  {key}  ({type_name}, default {shown})")
        if field.description:
            lines.extend(f"      {line}" for line in textwrap.wrap(field.description, 90))
    return "\n".join(lines)


__all__ = ["ConfigError", "Options", "Configurable", "Config", "parse_value", "parse_override", "configurables",
           "toml_value", "render_sections", "describe_options"]
