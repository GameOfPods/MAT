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
import os
from typing import Dict, Any, Tuple, List, Union, Optional

import numpy as np
import pydub
from pydantic import Field, field_validator
from torch import Tensor

from MAT.registry import register, require

require("pyannote.audio", "scipy", extra="pyannote")

from MAT.tools.speakeridentification import SpeakerIdentificationTool, SpeakerIdentificationInput, \
    SpeakerIdentificationResult  # noqa: E402
from MAT.utils.config import Config, Options  # noqa: E402


class PyannoteOptions(Options):
    gold_labels: Optional[str] = Field(None, description="Folder with one audio clip per speaker, named after the "
                                                         "speaker (alice.mp3). Without it the diarizer labels are "
                                                         "kept.")
    no_hf_token: bool = Field(False, description="Don't send your Hugging Face token when loading the model.")
    device: str = Field("auto", description='"auto" uses the GPU if there is one, or set "cpu" / "cuda".')
    model: str = Field("pyannote/embedding", description="pyannote speaker embedding model.")
    similarity_threshold: float = Field(0.3, description="Minimum cosine similarity to a gold label clip for a match.")

    @field_validator("gold_labels")
    @classmethod
    def _gold_labels_is_folder(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not os.path.isdir(value):
            raise ValueError(f"{value} is not a folder")
        return value


@register("identifier", "pyannote", description="pyannote speaker embeddings compared to gold label clips")
class SpeakerIdetificationPyannote(SpeakerIdentificationTool):
    Options = PyannoteOptions
    packages = ("pyannote-audio",)
    _LOGGER = logging.getLogger(__name__)

    def process(self, origin_data: SpeakerIdentificationInput, config: Config) -> Optional[SpeakerIdentificationResult]:
        import pathlib
        from MAT.utils.device import resolve_device

        options = config.options(self)
        if options.gold_labels is None:
            return SpeakerIdentificationResult(*[None for _ in origin_data.get_audio_files()])

        gold = {pathlib.Path(x).stem: os.path.join(options.gold_labels, x) for x in os.listdir(options.gold_labels)}
        self.__class__._LOGGER.info(f"Found gold labels for {len(gold)} speakers: {', '.join(sorted(gold.keys()))}")

        identification = self.identify(
            device=resolve_device(options.device), use_hf_token=not options.no_hf_token,
            similarity_threshold=options.similarity_threshold, model=options.model,
            gold={k: (pydub.AudioSegment.from_file(v), -1) for k, v in gold.items()},
            audios=origin_data.get_audio_files()
        )

        return SpeakerIdentificationResult(*identification)

    @staticmethod
    def identify(
            model: str,
            gold: Dict[str, Tuple[Union[Tensor, np.ndarray, pydub.AudioSegment], int]],
            audios: List[Tuple[Union[Tensor, np.ndarray, pydub.AudioSegment], int]],
            similarity_threshold: float = 0.3, device: str = "cpu",
            use_hf_token: Any = True,
    ) -> List[Optional[str]]:
        from pyannote.audio import Model, Inference
        from scipy.spatial.distance import cosine
        import torchaudio.transforms
        import torch

        # pyannote.audio 4 renamed use_auth_token to token
        pyannote_model = Model.from_pretrained(model, token=use_hf_token)
        classifier = Inference(pyannote_model, window="whole")
        classifier.to(torch.device(device))

        def _get_embedding(wave: Union[Tensor, np.ndarray, pydub.AudioSegment], sample: int):
            if isinstance(wave, pydub.AudioSegment):
                from MAT.utils import pydub_to_np
                wave, sample = pydub_to_np(audio=wave)
                wave = wave.transpose()
            if isinstance(wave, np.ndarray):
                wave = torch.from_numpy(wave)
            if wave.shape[0] > 1:
                wave = wave.mean(dim=0, keepdim=True)
            if sample != 16000:
                wave = torchaudio.transforms.Resample(orig_freq=sample, new_freq=16000)(wave)
            with torch.no_grad():
                wave.to(device)
                retries = 10
                while True:
                    try:
                        embedding = classifier({"waveform": wave, "sample_rate": 16000})
                        break
                    except Exception:
                        retries -= 1
                        if retries <= 0:
                            return None
                return embedding

        gold_embeddings = {}

        for k, (wave_form, sample_rate) in gold.items():
            gold_embeddings[k] = _get_embedding(wave=wave_form, sample=sample_rate)

        ret = []
        for wave_form, sample_rate in audios:
            test_embedding = _get_embedding(wave=wave_form, sample=sample_rate)
            similarity_scores = []
            for k, gold_embedding in gold_embeddings.items():
                if test_embedding is None or gold_embedding is None:
                    similarity_scores.append((k, 0))
                else:
                    similarity = 1 - cosine(test_embedding, gold_embedding)
                    similarity_scores.append((k, similarity))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            if similarity_scores[0][1] > similarity_threshold:
                ret.append(similarity_scores[0][0])
            else:
                ret.append(None)

        from MAT.utils.device import free_gpu_memory
        del classifier
        del pyannote_model
        free_gpu_memory()

        return ret
