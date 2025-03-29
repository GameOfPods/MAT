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


def get_all_concrete_subclasses(cls):
    concrete_subclasses = []
    for subclass in cls.__subclasses__():
        if not getattr(subclass, '__abstractmethods__', False):
            concrete_subclasses.append(subclass)
        concrete_subclasses.extend(get_all_concrete_subclasses(subclass))
    return concrete_subclasses


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
