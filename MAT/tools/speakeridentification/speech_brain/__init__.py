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
from typing import Dict, Any, Tuple, List, Union, Optional

import numpy as np
import pydub
from torch import Tensor

from MAT.utils.config import Config
from MAT.tools.speakeridentification import SpeakerIdentificationTool, SpeakerIdentificationInput, \
    SpeakerIdentificationResult
from MAT.utils.config import ConfigElement


class SpeakerIdetificationSpeechBrain(SpeakerIdentificationTool):
    @classmethod
    def config_name(cls) -> str:
        return "SpeechBrain"

    @classmethod
    def config_keys(cls) -> Dict[str, ConfigElement]:
        return {

        }

    def process(self, origin_data: SpeakerIdentificationInput, config: Config) -> Optional[SpeakerIdentificationResult]:
        pass

    @staticmethod
    def identify(
            model: str,
            gold: Dict[str, Tuple[Union[Tensor, np.ndarray, pydub.AudioSegment], int]],
            audios: List[Tuple[Union[Tensor, np.ndarray, pydub.AudioSegment], int]],
            similarity_threshold: float = 0.3, device: str = "cpu",
    ) -> List[Optional[str]]:
        from speechbrain.inference.speaker import EncoderClassifier
        from scipy.spatial.distance import cosine
        import torchaudio.transforms
        import torch

        classifier = EncoderClassifier.from_hparams(source=model)
        classifier.to(device)
        classifier.eval()

        def _get_embedding(wave: Union[Tensor, np.ndarray, pydub.AudioSegment], sample: int):
            if isinstance(wave, pydub.AudioSegment):
                from MAT.utils import pydub_to_np
                wave, sample = pydub_to_np(audio=wave)
            if isinstance(wave, np.ndarray):
                wave = torch.from_numpy(wave)
            if sample != 16000:
                wave = torchaudio.transforms.Resample(orig_freq=sample, new_freq=16000)(wave)
            with torch.no_grad():
                wave.to(device)
                embedding = classifier.encode_batch(wave).squeeze(1).cpu().numpy()
                embedding = np.mean(embedding, axis=0)
                return embedding

        gold_embeddings = {}

        for k, (wave_form, sample_rate) in gold.items():
            gold_embeddings[k] = _get_embedding(wave=wave_form, sample=sample_rate)

        ret = []
        for wave_form, sample_rate in audios:
            test_embedding = _get_embedding(wave=wave_form, sample=sample_rate)
            similarity_scores = []
            for k, gold_embedding in gold_embeddings.items():
                similarity = 1 - cosine(test_embedding, gold_embedding)
                similarity_scores.append((k, similarity))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            if similarity_scores[0][1] > similarity_threshold:
                ret.append(similarity_scores[0][0])
            else:
                ret.append(None)

        return ret
