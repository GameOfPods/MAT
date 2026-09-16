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
    model: str = Field("pyannote/wespeaker-voxceleb-resnet34-LM",
                       description="Speaker embedding model. WeSpeaker needs no Hugging Face login and separated "
                                   "speakers better than the older pyannote/embedding in our tests.")
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

    def can_match(self, config: Config) -> bool:
        return config.options(self).gold_labels is not None

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
        """Best gold speaker for every audio, None if no similarity is above the threshold. Several audios can get
        the same speaker."""
        names, matrix = SpeakerIdetificationPyannote.similarities(model=model, gold=gold, audios=audios, device=device,
                                                                  use_hf_token=use_hf_token)
        result = []
        for row in matrix:
            best = int(np.argmax(row)) if names else None
            result.append(names[best] if best is not None and row[best] > similarity_threshold else None)
        return result

    @staticmethod
    def similarities(
            model: str,
            gold: Dict[str, Tuple[Union[Tensor, np.ndarray, pydub.AudioSegment], int]],
            audios: List[Tuple[Union[Tensor, np.ndarray, pydub.AudioSegment], int]],
            device: str = "cpu", use_hf_token: Any = True,
    ) -> Tuple[List[str], np.ndarray]:
        """Gold speaker names and the cosine similarity of every audio (rows) to every gold speaker (columns). 0 where
        no embedding could be made."""
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

        names = list(gold)
        gold_embeddings = [_get_embedding(wave=gold[name][0], sample=gold[name][1]) for name in names]
        matrix = np.zeros((len(audios), len(names)))
        for row, (wave_form, sample_rate) in enumerate(audios):
            test_embedding = _get_embedding(wave=wave_form, sample=sample_rate)
            for column, gold_embedding in enumerate(gold_embeddings):
                if test_embedding is not None and gold_embedding is not None:
                    matrix[row, column] = 1 - cosine(test_embedding, gold_embedding)

        from MAT.utils.device import free_gpu_memory
        del classifier
        del pyannote_model
        free_gpu_memory()

        return names, matrix
