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
Registry of the backends MAT can use, grouped by slot (transcriber, diarizer, ...).

A backend lives in its own module. The module checks its libraries first with `require(...)`, which only looks them up
with importlib (no heavy imports) and raises `MissingDependencies` if the matching uv extra isn't installed. Then it
defines its class with `@register(slot, name)`. The package `__init__.py` loads the module with `load_optional(...)`,
so a backend with missing libraries is skipped and listed by `MAT backends` instead of breaking `import MAT`.
"""
import importlib
import importlib.util
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

_LOGGER = logging.getLogger(__name__)


class MissingDependencies(ImportError):
    def __init__(self, extra: str, missing: Tuple[str, ...]):
        self.extra = extra
        self.missing = tuple(missing)
        super().__init__(f"missing {', '.join(self.missing)} (install with: uv sync --extra {extra})")


class BackendError(LookupError):
    pass


@dataclass(frozen=True)
class Backend:
    slot: str
    name: str
    cls: type
    description: str = ""


@dataclass(frozen=True)
class SkippedBackend:
    slot: str
    name: str
    extra: str
    reason: str


_BACKENDS: Dict[str, Dict[str, Backend]] = {}
_SKIPPED: Dict[str, Dict[str, SkippedBackend]] = {}


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        # find_spec("a.b") imports "a" first and raises if that is missing
        return False


def require(*modules: str, extra: str) -> None:
    missing = tuple(module for module in modules if not _module_available(module))
    if missing:
        raise MissingDependencies(extra=extra, missing=missing)


def register(slot: str, name: str, description: str = ""):
    def decorate(cls):
        existing = _BACKENDS.get(slot, {}).get(name)
        if existing is not None and existing.cls is not cls:
            raise RuntimeError(f"Backend {slot}/{name} is registered twice ({existing.cls} and {cls})")
        cls.slot = slot
        cls.backend_name = name
        cls.section = name
        cls.description = description
        _BACKENDS.setdefault(slot, {})[name] = Backend(slot=slot, name=name, cls=cls, description=description)
        _SKIPPED.get(slot, {}).pop(name, None)
        return cls

    return decorate


def unregister(slot: str, name: str) -> None:
    _BACKENDS.get(slot, {}).pop(name, None)
    _SKIPPED.get(slot, {}).pop(name, None)


def load_optional(module: str, slot: str, name: str, extra: str) -> Optional[Backend]:
    try:
        importlib.import_module(module)
    except MissingDependencies as e:
        _SKIPPED.setdefault(slot, {})[name] = SkippedBackend(slot=slot, name=name, extra=e.extra, reason=str(e))
        _LOGGER.debug(f"Skipping backend {slot}/{name}: {e}")
        return None
    except ImportError as e:
        # the libraries are there but don't load, for example a broken CUDA install
        reason = f"importing {module} failed: {e}"
        _SKIPPED.setdefault(slot, {})[name] = SkippedBackend(slot=slot, name=name, extra=extra, reason=reason)
        _LOGGER.warning(f"Skipping backend {slot}/{name}, {reason}")
        return None
    backend = _BACKENDS.get(slot, {}).get(name)
    if backend is None:
        raise RuntimeError(f"{module} was loaded for backend {slot}/{name} but didn't register it")
    return backend


def slots() -> List[str]:
    return sorted(set(_BACKENDS) | set(_SKIPPED))


def backends(slot: Optional[str] = None) -> List[Backend]:
    selected = [slot] if slot is not None else sorted(_BACKENDS)
    return [b for s in selected for b in sorted(_BACKENDS.get(s, {}).values(), key=lambda b: b.name)]


def skipped(slot: Optional[str] = None) -> List[SkippedBackend]:
    selected = [slot] if slot is not None else sorted(_SKIPPED)
    return [b for s in selected for b in sorted(_SKIPPED.get(s, {}).values(), key=lambda b: b.name)]


def find(name: str, slot: Optional[str] = None) -> Optional[Union[Backend, SkippedBackend]]:
    for backend in backends(slot):
        if backend.name == name:
            return backend
    for backend in skipped(slot):
        if backend.name == name:
            return backend
    return None


def get(slot: str, name: str) -> Backend:
    backend = _BACKENDS.get(slot, {}).get(name)
    if backend is not None:
        return backend
    installed = ", ".join(b.name for b in backends(slot)) or "nothing"
    skipped_backend = _SKIPPED.get(slot, {}).get(name)
    if skipped_backend is not None:
        raise BackendError(f'{slot} "{name}" isn\'t installed, {skipped_backend.reason}. Installed: {installed}')
    raise BackendError(f'Unknown {slot} "{name}". Installed: {installed}')


__all__ = ["MissingDependencies", "BackendError", "Backend", "SkippedBackend", "require", "register", "unregister",
           "load_optional", "slots", "backends", "skipped", "find", "get"]
