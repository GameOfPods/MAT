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
Results of slow steps (transcription, diarization, sound events) kept on disk, so running an episode again doesn't
redo them. A rerun after a failed summary, or trying another summary model on the same episode, starts at the
summary.

The key is the file content (sha1), the step, the backend with its options as the run uses them, the versions of its
packages and of MAT, and whatever else the step depends on (the vocabulary for the transcription). Change any of it
and the step runs again. The files are pickles: delete the folder whenever you like.
"""
import hashlib
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

_LOGGER = logging.getLogger(__name__)

DEFAULT_FOLDER = "~/.cache/mat/steps"


class StepCache:
    def __init__(self, folder: Optional[str]):
        self.folder = Path(os.path.expandvars(folder)).expanduser() if folder else None

    @property
    def enabled(self) -> bool:
        return self.folder is not None

    @staticmethod
    def key(backend, config, **extra) -> Dict[str, Any]:
        """What a cached result depends on besides the file."""
        from MAT import __version__

        options = type(backend).effective_options(config).model_dump(mode="json")
        return {"mat": __version__, "backend": backend.backend_name, "options": options,
                "packages": backend.describe(config).get("packages", {}), **extra}

    def _path(self, file: str, step: str, key: Dict[str, Any]) -> Path:
        from MAT.utils import get_hash_file

        digest = hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]
        return self.folder / get_hash_file(file) / f"{step}-{digest}.pkl"

    def get(self, file: str, step: str, key: Dict[str, Any]) -> Optional[Any]:
        if not self.enabled:
            return None
        path = self._path(file, step, key)
        if not path.is_file():
            return None
        try:
            with open(path, "rb") as f:
                value = pickle.load(f)
        except Exception as e:
            # written by an older MAT whose classes changed, or cut short: just run the step again
            _LOGGER.info(f"Can't read the cached {step} ({e.__class__.__name__}), running it again")
            return None
        _LOGGER.info(f"Using the cached {step} from {path.parent.name[:10]}/{path.name}")
        return value

    def put(self, file: str, step: str, key: Dict[str, Any], value: Any) -> None:
        if not self.enabled or value is None:
            return
        path = self._path(file, step, key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".part")
            with open(temporary, "wb") as f:
                pickle.dump(value, f)
            temporary.replace(path)
        except OSError as e:
            _LOGGER.warning(f"Can't write the step cache in {self.folder}: {e}")


__all__ = ["StepCache"]
