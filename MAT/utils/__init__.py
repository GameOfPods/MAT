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
import functools
import os
from typing import Callable, Dict, Any, Optional


def get_hash(obj) -> str:
    import hashlib
    return hashlib.sha1(obj).hexdigest()

def get_hash_file(file_path: str) -> str:
    """sha1 of a file, read in pieces. Remembered per path, size and modification time, the writer and the step
    cache both need it and a 3 hour episode takes a moment to hash."""
    stat = os.stat(file_path)
    return _hash_file(os.path.abspath(file_path), stat.st_size, stat.st_mtime_ns)


@functools.lru_cache(maxsize=64)
def _hash_file(path: str, size: int, mtime: int) -> str:
    import hashlib

    digest = hashlib.sha1()
    with open(path, "rb") as file_handle:
        for block in iter(lambda: file_handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def get_all_concrete_subclasses(cls):
    concrete_subclasses = []
    for subclass in cls.__subclasses__():
        if not getattr(subclass, '__abstractmethods__', False):
            concrete_subclasses.append(subclass)
        concrete_subclasses.extend(get_all_concrete_subclasses(subclass))
    return concrete_subclasses


def _permanent(error: BaseException) -> bool:
    """Errors a retry can't fix: no login, no access, no such model. Waiting a minute for each is just slow."""
    try:
        from huggingface_hub.errors import GatedRepoError, LocalTokenNotFoundError, RepositoryNotFoundError
    except ImportError:
        return False
    return isinstance(error, (GatedRepoError, LocalTokenNotFoundError, RepositoryNotFoundError))


def timeout_retry(func: Callable, func_args: tuple, func_kwargs: Dict[str, Any], time_out: int = 60, retries: int = 5):
    retries = max(0, retries)
    time_out = max(0, time_out)
    last_e: Optional[BaseException] = None
    from time import sleep
    while retries >= 0:
        try:
            return func(*func_args, **func_kwargs)
        except Exception as e:
            if _permanent(e):
                raise
            last_e = e
        retries -= 1
        # no point in waiting after the last attempt
        if retries >= 0:
            sleep(time_out)
    if last_e is not None:
        raise last_e
    return None



__all__ = ["get_all_concrete_subclasses"]

try:
    import pydub
    import numpy as np


    def pydub_to_np(audio: pydub.AudioSegment) -> tuple[np.ndarray, int]:
        """
        Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
        where each value is in range [-1.0, 1.0].
        Returns tuple (audio_np_array, sample_rate).
        """
        return np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels)) / (
                1 << (8 * audio.sample_width - 1)), audio.frame_rate


    __all__.append("pydub_to_np")

finally:
    pass

try:
    from roman import toRoman
except ImportError:
    # noinspection PyPep8Naming
    def toRoman(n):
        return n
