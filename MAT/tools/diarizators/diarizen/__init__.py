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
DiariZen, run in its own environment (envs/diarizen).

It pins torch 2.1.1 and its own pyannote fork, so it can't live next to MAT's dependencies. MAT decodes the audio
and hands the file to envs/diarizen/run.py, see docs/external-environments.md.
"""
import logging
import os
from typing import Optional
from uuid import uuid4

from pydantic import Field

from MAT.registry import register
from MAT.utils.external import require_environment

# without the environment the backend is skipped, like a missing extra
require_environment("diarizen")

from MAT.tools.diarizators import DiarizationResult, DiarizationTool, DiarizerInput  # noqa: E402
from MAT.utils.config import Config, Options  # noqa: E402


class DiarizenOptions(Options):
    model: str = Field("BUT-FIT/diarizen-wavlm-large-s80-md",
                       description="DiariZen model. The weights are non-commercial (CC BY-NC 4.0).")
    device: str = Field("auto", description='"auto" uses the GPU if there is one, or set "cpu" / "cuda".')
    batch_size: int = Field(8, ge=1, description="Audio chunks the model sees at once. DiariZen's own config asks "
                                                 "for 32, which runs out of memory on an 11 GB card. Lower it if it "
                                                 "still does, raise it on a bigger card for speed.")
    timeout: float = Field(3600, gt=0, description="Seconds to wait for the DiariZen process before giving up.")


@register("diarizer", "diarizen",
          description="DiariZen (WavLM and Conformer) in its own environment, non-commercial weights")
class DiarizerDiariZen(DiarizationTool):
    Options = DiarizenOptions
    packages = ()
    _LOGGER = logging.getLogger(__name__)

    def process(self, origin_data: DiarizerInput, config: Config) -> Optional[DiarizationResult]:
        from pydub import AudioSegment

        from MAT.utils.device import resolve_device
        from MAT.utils.external import run_external

        options = config.options(self)
        device = resolve_device(options.device)
        folder = os.path.join(config.work_directory, f"diarizen.{uuid4()}")
        os.makedirs(folder, exist_ok=True)
        # decoded here, so the other environment only needs to read a plain wav
        audio_file = os.path.join(folder, "audio.wav")
        sound = AudioSegment.from_file(origin_data.in_file).set_channels(1).set_frame_rate(16000).set_sample_width(2)
        sound.export(audio_file, format="wav")

        self._LOGGER.info(f"Diarizing {sound.duration_seconds:.0f} s with {options.model} in the diarizen "
                          f"environment on {device}")
        answer = run_external("diarizen", {"audio": audio_file, "model": options.model, "device": device,
                                           "batch_size": options.batch_size}, timeout=options.timeout)
        return self._to_result(answer)

    @staticmethod
    def _to_result(answer: dict) -> DiarizationResult:
        speakers = answer.get("speakers") or {}
        names = {label: f"sprecher_{i}" for i, label in enumerate(sorted(speakers))}
        result = DiarizationResult()
        for label, times in speakers.items():
            for start, end in times:
                result.add_diarization(speaker=names[label], f=float(start), t=float(end))
        return result


__all__ = ["DiarizerDiariZen", "DiarizenOptions"]
