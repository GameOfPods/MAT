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
from typing import Dict, Any, Tuple, List, Union, Optional

import numpy as np
import pydub
from torch import Tensor

from MAT.utils.config import Config
from MAT.tools.speakeridentification import SpeakerIdentificationTool, SpeakerIdentificationInput, \
    SpeakerIdentificationResult
from MAT.utils.config import ConfigElement


class SpeakerIdetificationPyannote(SpeakerIdentificationTool):
    _LOGGER = logging.getLogger(__name__)

    @classmethod
    def config_name(cls) -> str:
        return "Pyannote-Identification"

    @classmethod
    def config_keys(cls) -> Dict[str, ConfigElement]:
        import torch
        def gold_label_folder(_folder: str) -> str:
            from argparse import ArgumentTypeError
            from pydub import AudioSegment
            import os
            if not os.path.exists(_folder):
                raise ArgumentTypeError("Provided gold folder has to exist")
            if not os.path.isdir(_folder):
                raise ArgumentTypeError("Provided gold folder has to be a folder")
            for file in os.listdir(_folder):
                # noinspection PyBroadException
                try:
                    AudioSegment.from_file(os.path.join(_folder, file))
                    return _folder
                except Exception:
                    pass
            raise ArgumentTypeError("There has to be at least one gold audio file in the folder")

        return {
            "gold-labels": ConfigElement(
                default_value=None,
                argparse_kwargs={
                    "help": "Folder containing audio files to be used as speaker gold labels. Will use the file name without the extension as speaker name. If not provided, skip speaker matching.",
                    "required": False, "type": gold_label_folder,
                }
            ),
            "no-hf-token": ConfigElement(
                default_value=True,
                argparse_kwargs={
                    "help": "Whether to not use a Hugging Face token",
                    "action": "store_false",
                }
            ),
            "device": ConfigElement(
                default_value="cuda" if torch.cuda.is_available() else "cpu",
                argparse_kwargs={
                    "help": "Model to run the model on. Default: %(default)s",
                    "type": str,
                }
            ),
            "model": ConfigElement(
                default_value="pyannote/embedding",
                argparse_kwargs={
                    "help": "Model to use for pyannote embedding. Default: %(default)s", "type": str,
                }
            ),
            "similarity-threshold": ConfigElement(
                default_value=0.3,
                argparse_kwargs={
                    "help": "Threshold for similarity comparison. Default: %(default)s", "type": float,
                }
            )
        }

    def process(self, origin_data: SpeakerIdentificationInput, config: Config) -> Optional[SpeakerIdentificationResult]:
        import os
        import pathlib

        cfg = config.get_config(key=self)
        gold_folder = cfg["gold-labels"]
        if gold_folder is None:
            return SpeakerIdentificationResult(*[None for _ in origin_data.get_audio_files()])

        gold = {pathlib.Path(x).stem: os.path.join(gold_folder, x) for x in os.listdir(gold_folder)}
        self.__class__._LOGGER.info(f"Found gold labels for {len(gold)} speakers: {', '.join(sorted(gold.keys()))}")

        identification = self.identify(
            device=cfg["device"], use_hf_token=not cfg["no-hf-token"], similarity_threshold=cfg["similarity-threshold"],
            model=cfg["model"],
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

        pyannote_model = Model.from_pretrained(model, use_auth_token=use_hf_token)
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
                pass
            if sample != 16000:
                wave = torchaudio.transforms.Resample(orig_freq=sample, new_freq=16000)(wave)
            with torch.no_grad():
                wave.to(device)
                retries = 10
                while True:
                    try:
                        embedding = classifier({"waveform": wave, "sample_rate": 16000})
                        break
                    except Exception as e:
                        retries -= 1
                        if retries <= 0:
                            return None
                # embedding = np.mean(embedding, axis=0)
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

        return ret
