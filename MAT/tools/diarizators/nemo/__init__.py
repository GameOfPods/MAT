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
import os.path
from collections import defaultdict
from typing import Dict, Optional, Tuple, List
import logging
from uuid import uuid4

from MAT.utils.config import ConfigElement, Config
from MAT.tools.diarizators import DiarizationTool, DiarizerInput, DiarizationResult


class DiarizerNEMO(DiarizationTool):
    from pydub import AudioSegment
    _LOGGER = logging.getLogger(__name__)

    @classmethod
    def config_name(cls) -> str:
        return "NeMo"

    @classmethod
    def config_keys(cls) -> Dict[str, ConfigElement]:
        import torch
        return {
            "model": ConfigElement(
                default_value="nvidia/diar_sortformer_4spk-v1",
                argparse_kwargs={
                    "help": "Model for whisper diarizer. Default: %(default)s",
                    "type": str,
                }
            ),
            "device": ConfigElement(
                default_value="cuda" if torch.cuda.is_available() else "cpu",
                argparse_kwargs={
                    "help": "Model to run the model on. Default: %(default)s",
                    "type": str,
                }
            ),
            "segment-length": ConfigElement(
                default_value=5 * 60,
                argparse_kwargs={
                    "help": "Length of audio segments in seconds. Needed to save VRAM. Default: %(default)s",
                    "type": int,
                }
            )
        }

    def process(self, origin_data: DiarizerInput, config: Config) -> Optional[DiarizationResult]:
        from nemo.collections.asr.models import SortformerEncLabelModel
        from math import ceil
        from pydub import AudioSegment

        cfg = config.get_config(key=self)

        nemo_dir = os.path.join(config.work_directory, f"nemo.{uuid4()}")
        os.makedirs(nemo_dir, exist_ok=True)

        sound = AudioSegment.from_file(origin_data.in_file).set_channels(1)
        mono_files = []
        for i in range(ceil(sound.duration_seconds / cfg["segment-length"])):
            audio_file_mono = os.path.join(nemo_dir, f"mono.{uuid4()}.{i}.wav")
            sound[i * cfg["segment-length"] * 1000:(i + 1) * cfg["segment-length"] * 1000].export(audio_file_mono,
                                                                                                  format="wav")
            mono_files.append(audio_file_mono)

        diar_model = SortformerEncLabelModel.from_pretrained(cfg["model"], map_location=cfg["device"])
        diar_model.eval()

        predicted_segments, predicted_probs = diar_model.diarize(
            audio=mono_files, batch_size=1, include_tensor_outputs=True
        )

        del diar_model
        if cfg["device"] == "cuda":
            import torch
            torch.cuda.empty_cache()

        clean_segments: List[Dict[str, List[Tuple[float, float]]]] = []
        for predicted_segment in predicted_segments:
            clean_segments.append(defaultdict(list))
            for segment in predicted_segment:
                f, t, speaker = segment.strip().split(" ")
                clean_segments[-1][speaker].append((float(f), float(t)))

        combinations = [self.combine(segments=segments, mono_file=mono_file) for segments, mono_file in
                        zip(clean_segments, mono_files)]

        speaker_id_template = "sprecher_{id}"
        ret_global_id = [0]

        def get_next_speaker_id() -> str:
            _r = speaker_id_template.format(id=ret_global_id[0])
            ret_global_id[0] += 1
            return _r

        for i in range(len(combinations)):
            combination = [(k, v) for k, v in combinations[i].items()]
            clean_segment = clean_segments[i]
            if i == 0:
                identification = [None] * len(combination)
            else:
                gold = defaultdict(lambda: AudioSegment.empty())
                for j in range(max(0, i - 5), i):
                    for k, v in combinations[j].items():
                        gold[k] += v
                from MAT.tools.speakeridentification.pyannote import SpeakerIdetificationPyannote as Identifier
                identification = Identifier().identify(
                    model="pyannote/embedding",  # "speechbrain/spkrec-ecapa-voxceleb",
                    gold={k: (v, v.frame_rate) for k, v in gold.items()},
                    audios=[(x, x.frame_rate) for _, x in combination],
                    device=cfg["device"],
                )
            combinations[i] = {}
            clean_segments[i] = {}
            for new_speaker, (old_speaker, audio) in zip(identification, combination):
                if new_speaker is None:
                    new_speaker = get_next_speaker_id()
                combinations[i][new_speaker] = audio
                clean_segments[i][new_speaker] = clean_segment[old_speaker]

        ret = DiarizationResult()
        for i, clean_segment in enumerate(clean_segments):
            offset = i * cfg["segment-length"]
            for k, v in clean_segment.items():
                for f, t in v:
                    ret.add_diarization(speaker=k, f=float(f) + offset, t=float(t) + offset)

        return ret

    @staticmethod
    def combine(segments: Dict[str, List[Tuple[float, float]]], mono_file) -> Dict[str, AudioSegment]:
        from pydub import AudioSegment
        combined = {}
        orig = AudioSegment.from_file(mono_file)
        for speaker, times in segments.items():
            combined[speaker] = AudioSegment.empty()
            for f, t in times:
                combined[speaker] += orig[f * 1000:t * 1000]
        return combined

    @classmethod
    def _create_config(cls, audio_file: str, domain: str, out_dir: str) -> Tuple["OmegaConf", str]:
        import os
        import io
        from uuid import uuid4
        import json
        from omegaconf import OmegaConf
        import requests

        config_io = io.StringIO(
            requests.get(
                f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_{domain}.yaml"
            ).text
        )

        config = OmegaConf.load(config_io)

        meta = {
            "audio_filepath": audio_file,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "rttm_filepath": None,
            "uem_filepath": None,
        }

        _manifest_file = os.path.join(out_dir, f"input_manifest_{uuid4()}.json")
        with open(_manifest_file, "w") as fp:
            json.dump(meta, fp)
            fp.write("\n")

        pretrained_vad = "vad_multilingual_marblenet"
        pretrained_speaker_model = "titanet_large"
        config.num_workers = 0
        config.diarizer.manifest_filepath = _manifest_file
        config.diarizer.out_dir = out_dir
        config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
        config.diarizer.oracle_vad = False
        config.diarizer.clustering.parameters.oracle_num_speakers = False
        config.diarizer.vad.model_path = pretrained_vad
        config.diarizer.vad.parameters.onset = 0.8
        config.diarizer.vad.parameters.offset = 0.6
        config.diarizer.vad.parameters.pad_offset = -0.05
        config.diarizer.msdd_model.model_path = f"diar_msdd_{domain}"

        return config, out_dir
