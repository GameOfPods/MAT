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
Backends that can't share MAT's dependencies.

Some libraries pin versions we can't follow (DiariZen wants torch 2.1.1 and its own pyannote fork). Those live in
`envs/<name>` with their own virtual environment, and MAT talks to them by running `envs/<name>/run.py` with a JSON
request and reading a JSON answer. No server, no protocol, and the audio is handed over as a file.

Adding another one: a folder under `envs/` with `install.sh` and `run.py`, an entry in ENVIRONMENTS, and a backend
that calls `run_external`. See docs/external-environments.md.
"""
import json
import logging
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from MAT.registry import MissingDependencies

_LOGGER = logging.getLogger(__name__)

ENVS_DIR_VAR = "MAT_ENVS_DIR"


@dataclass(frozen=True)
class ExternalEnvironment:
    name: str
    description: str
    # both relative to the environment folder
    script: str = "run.py"
    install_script: str = "install.sh"


ENVIRONMENTS: Dict[str, ExternalEnvironment] = {
    "diarizen": ExternalEnvironment(
        name="diarizen",
        description="DiariZen diarization (WavLM and Conformer). Pinned to torch 2.1.1 with its own pyannote fork, "
                    "weights are non-commercial (CC BY-NC 4.0).",
    ),
}


class ExternalError(RuntimeError):
    pass


class MissingEnvironment(MissingDependencies):
    """A MissingDependencies, so a backend without its environment is skipped like one with a missing extra."""

    def __init__(self, name: str):
        self.environment = name
        self.extra = name
        self.missing = ()
        ImportError.__init__(self, f'the "{name}" environment isn\'t built (build it with: MAT external install '
                                  f'{name})')


def envs_directory() -> Path:
    """envs/ next to the MAT package, or $MAT_ENVS_DIR."""
    configured = os.environ.get(ENVS_DIR_VAR)
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parent.parent.parent / "envs"


def environment_folder(name: str) -> Path:
    return envs_directory() / name


def interpreter(name: str) -> Path:
    """The Python of that environment. $MAT_EXTERNAL_PYTHON_<NAME> wins, which is what the tests use."""
    override = os.environ.get(f"MAT_EXTERNAL_PYTHON_{name.upper().replace('-', '_')}")
    if override:
        return Path(override)
    folder = environment_folder(name) / ".venv"
    windows = folder / "Scripts" / "python.exe"
    return windows if windows.exists() else folder / "bin" / "python"


def is_available(name: str) -> bool:
    return interpreter(name).exists()


def require_environment(name: str) -> None:
    """Raises MissingEnvironment when the environment isn't built. Call it at the top of a backend module."""
    if not is_available(name):
        raise MissingEnvironment(name)


def _tail(text: Optional[str], lines: int = 15) -> str:
    return "\n".join((text or "").strip().splitlines()[-lines:])


def run_external(name: str, request: Dict[str, Any], timeout: float = 3600.0,
                 script: Optional[str] = None) -> Dict[str, Any]:
    """Runs the environment's script with the request and returns its answer."""
    require_environment(name)
    environment = ENVIRONMENTS.get(name) or ExternalEnvironment(name=name, description="")
    folder = environment_folder(name)
    command = [str(interpreter(name)), str(folder / (script or environment.script))]
    with tempfile.TemporaryDirectory(prefix=f"mat-{name}-") as temporary:
        request_file, result_file = Path(temporary) / "request.json", Path(temporary) / "result.json"
        request_file.write_text(json.dumps(request), encoding="utf-8")
        command += [str(request_file), str(result_file)]
        _LOGGER.debug("Running " + " ".join(shlex.quote(part) for part in command))
        try:
            finished = subprocess.run(command, capture_output=True, text=True, timeout=timeout, cwd=str(folder))
        except subprocess.TimeoutExpired:
            raise ExternalError(f"{name} didn't finish within {timeout:.0f} s")
        except OSError as e:
            raise ExternalError(f"Could not start {name}: {e}")
        for line in (finished.stdout or "").splitlines():
            _LOGGER.debug(f"{name}: {line}")
        if finished.returncode != 0:
            raise ExternalError(f"{name} stopped with exit code {finished.returncode}:\n{_tail(finished.stderr)}")
        if not result_file.exists():
            raise ExternalError(f"{name} wrote no result:\n{_tail(finished.stderr)}")
        try:
            return json.loads(result_file.read_text(encoding="utf-8"))
        except ValueError as e:
            raise ExternalError(f"{name} wrote a result that isn't JSON: {e}")


def install(name: str) -> int:
    """Runs the install script of an environment and lets it write to the console."""
    if name not in ENVIRONMENTS:
        raise ExternalError(f'Unknown environment "{name}". Known: {", ".join(sorted(ENVIRONMENTS))}')
    folder = environment_folder(name)
    script = folder / ENVIRONMENTS[name].install_script
    if not script.exists():
        raise ExternalError(f"{script} doesn't exist")
    _LOGGER.info(f"Building the {name} environment with {script}")
    return subprocess.run(["bash", str(script)], cwd=str(folder)).returncode


__all__ = ["ENVIRONMENTS", "ExternalEnvironment", "ExternalError", "MissingEnvironment", "envs_directory",
           "environment_folder", "interpreter", "is_available", "require_environment", "run_external", "install"]
