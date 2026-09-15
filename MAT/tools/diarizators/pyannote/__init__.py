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
from typing import Optional

from pydantic import Field, model_validator

from MAT.registry import register, require

require("pyannote.audio", extra="pyannote")

from MAT.tools.diarizators import DiarizationResult, DiarizationTool, DiarizerInput  # noqa: E402
from MAT.utils.config import Config, Options  # noqa: E402


class PyannoteDiarizationOptions(Options):
    model: str = Field("pyannote/speaker-diarization-community-1",
                       description="pyannote diarization pipeline. Accept its terms on Hugging Face and log in with "
                                   "`hf auth login` first.")
    device: str = Field("auto", description='"auto" uses the GPU if there is one, or set "cpu" / "cuda".')
    num_speakers: Optional[int] = Field(None, ge=1, description="Exact number of speakers, if you know it.")
    min_speakers: Optional[int] = Field(None, ge=1, description="Lowest number of speakers when estimating.")
    max_speakers: Optional[int] = Field(None, ge=1, description="Highest number of speakers when estimating.")
    exclusive: bool = Field(False, description="Use the exclusive diarization with one speaker at a time, made for "
                                               "matching transcript words to speakers. Overlapping speech goes to "
                                               "the most likely speaker.")
    no_hf_token: bool = Field(False, description="Don't send your Hugging Face token when loading the model.")

    @model_validator(mode="after")
    def _speaker_range(self):
        if self.min_speakers and self.max_speakers and self.min_speakers > self.max_speakers:
            raise ValueError("min-speakers can't be larger than max-speakers")
        return self


@register("diarizer", "pyannote-diarization",
          description="pyannote speaker diarization (community-1), any number of speakers")
class DiarizerPyannote(DiarizationTool):
    Options = PyannoteDiarizationOptions
    packages = ("pyannote-audio",)
    _LOGGER = logging.getLogger(__name__)

    def process(self, origin_data: DiarizerInput, config: Config) -> Optional[DiarizationResult]:
        import numpy as np
        import torch
        from pydub import AudioSegment

        from MAT.utils.device import free_gpu_memory, resolve_device

        options = config.options(self)
        device = resolve_device(options.device)
        sound = AudioSegment.from_file(origin_data.in_file).set_channels(1).set_frame_rate(16000).set_sample_width(2)
        # Decoded here and passed as a waveform. pyannote decodes files with torchcodec, which doesn't support every
        # FFmpeg version (torchcodec 0.7 stops at FFmpeg 7).
        waveform = torch.from_numpy(np.frombuffer(sound.raw_data, dtype=np.int16).astype(np.float32) / 32768.0)[None]
        speakers = {name: value for name, value in (("num_speakers", options.num_speakers),
                                                    ("min_speakers", options.min_speakers),
                                                    ("max_speakers", options.max_speakers)) if value is not None}
        self._LOGGER.info(f"Diarizing {sound.duration_seconds:.0f} s with {options.model} on {device}"
                          + (f" ({', '.join(f'{k}={v}' for k, v in speakers.items())})" if speakers else ""))
        pipeline = self._load_pipeline(options, device)
        try:
            output = pipeline({"waveform": waveform, "sample_rate": sound.frame_rate}, **speakers)
        finally:
            del pipeline
            free_gpu_memory()
        return self._to_result(output, exclusive=options.exclusive)

    @staticmethod
    def _load_pipeline(options: PyannoteDiarizationOptions, device: str):
        import torch
        from pyannote.audio import Pipeline

        from MAT.utils import timeout_retry

        pipeline = timeout_retry(func=Pipeline.from_pretrained, func_args=(options.model,),
                                 func_kwargs={"token": not options.no_hf_token}, time_out=60, retries=5)
        if pipeline is None:
            raise RuntimeError(f"Could not load {options.model}. Accept its terms on Hugging Face and log in with "
                               f"`hf auth login`")
        pipeline.to(torch.device(device))
        return pipeline

    @staticmethod
    def _to_result(output, exclusive: bool) -> DiarizationResult:
        # pyannote 4 returns an object with both diarizations, older versions the annotation itself
        annotation = getattr(output, "exclusive_speaker_diarization" if exclusive else "speaker_diarization", output)
        names = {label: f"sprecher_{i}" for i, label in enumerate(sorted(annotation.labels()))}
        result = DiarizationResult()
        for turn, _, label in annotation.itertracks(yield_label=True):
            result.add_diarization(speaker=names[label], f=float(turn.start), t=float(turn.end))
        return result


__all__ = ["DiarizerPyannote", "PyannoteDiarizationOptions"]
