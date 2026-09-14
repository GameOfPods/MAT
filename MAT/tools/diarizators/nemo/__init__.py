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
import os.path
from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple, List
from uuid import uuid4

from pydantic import Field

from MAT.registry import register, require

# pyannote links the speakers of neighboring audio pieces
require("nemo", "pyannote.audio", extra="sortformer")

from MAT.tools.diarizators import DiarizationTool, DiarizerInput, DiarizationResult  # noqa: E402
from MAT.utils.audio import Window  # noqa: E402
from MAT.utils.config import Config, Options  # noqa: E402


class SortformerOptions(Options):
    model: str = Field("nvidia/diar_sortformer_4spk-v1", description="NeMo Sortformer model.")
    device: str = Field("auto", description='"auto" uses the GPU if there is one, or set "cpu" / "cuda".')
    segment_length: int = Field(5 * 60, ge=30, description="Longest piece of audio in seconds the model sees at "
                                                           "once. Longer audio is cut at a quiet spot and the "
                                                           "speakers are linked between pieces. Lower it if the GPU "
                                                           "runs out of memory.")


@register("diarizer", "sortformer", description="NVIDIA NeMo Sortformer, up to 4 speakers per audio piece")
class DiarizerNEMO(DiarizationTool):
    Options = SortformerOptions
    packages = ("nemo-toolkit", "pyannote-audio")
    _LOGGER = logging.getLogger(__name__)

    def process(self, origin_data: DiarizerInput, config: Config) -> Optional[DiarizationResult]:
        import numpy as np
        from MAT.utils import timeout_retry
        from MAT.utils.audio import plan_windows
        from MAT.utils.device import free_gpu_memory, resolve_device
        from nemo.collections.asr.models import SortformerEncLabelModel
        from pydub import AudioSegment

        options = config.options(self)
        device = resolve_device(options.device)

        nemo_dir = os.path.join(config.work_directory, f"nemo.{uuid4()}")
        os.makedirs(nemo_dir, exist_ok=True)

        sound = AudioSegment.from_file(origin_data.in_file).set_channels(1)
        if sound.sample_width not in (2, 4):
            sound = sound.set_sample_width(2)
        # a view on pydub's buffer, no copy of hours of audio
        samples = np.frombuffer(sound.raw_data, dtype=np.int16 if sound.sample_width == 2 else np.int32)
        # cut at quiet spots instead of every segment-length seconds, so cuts don't land in the middle of a word
        windows = plan_windows(samples, sound.frame_rate, max_length=options.segment_length)
        self._LOGGER.info(f"Diarizing {len(windows)} audio piece(s) of at most {options.segment_length} s")
        mono_files = []
        for i, window in enumerate(windows):
            audio_file_mono = os.path.join(nemo_dir, f"mono.{uuid4()}.{i}.wav")
            sound[int(window.start * 1000):int(window.end * 1000)].export(audio_file_mono, format="wav")
            mono_files.append(audio_file_mono)

        diar_model: SortformerEncLabelModel = timeout_retry(
            func=SortformerEncLabelModel.from_pretrained,
            func_args=(options.model,),
            func_kwargs={"map_location": device},
            time_out=60,
            retries=5,
        )
        diar_model.eval()

        predicted_segments, predicted_probs = diar_model.diarize(
            audio=mono_files, batch_size=1, include_tensor_outputs=True
        )

        del diar_model
        free_gpu_memory()

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
                    model="pyannote/embedding",
                    gold={k: (v, v.frame_rate) for k, v in gold.items()},
                    audios=[(x, x.frame_rate) for _, x in combination],
                    device=device,
                )
            combinations[i] = {}
            clean_segments[i] = {}
            for new_speaker, (old_speaker, audio) in zip(identification, combination):
                if new_speaker is None:
                    new_speaker = get_next_speaker_id()
                combinations[i][new_speaker] = audio
                clean_segments[i][new_speaker] = clean_segment[old_speaker]

        return self._merge_windows(windows, clean_segments)

    @staticmethod
    def _merge_windows(windows: Sequence[Window],
                       segments_per_window: Sequence[Dict[str, List[Tuple[float, float]]]]) -> DiarizationResult:
        """Shift the segments of every audio piece to absolute times, keep each only in the piece that owns it."""
        ret = DiarizationResult()
        for window, segments in zip(windows, segments_per_window):
            for speaker, times in segments.items():
                for f, t in times:
                    start, end = float(f) + window.start, float(t) + window.start
                    if window.owns(start, end):
                        ret.add_diarization(speaker=speaker, f=start, t=end)
        return ret

    @staticmethod
    def combine(segments: Dict[str, List[Tuple[float, float]]], mono_file) -> Dict[str, "AudioSegment"]:
        from pydub import AudioSegment
        combined = {}
        orig = AudioSegment.from_file(mono_file)
        for speaker, times in segments.items():
            combined[speaker] = AudioSegment.empty()
            for f, t in times:
                combined[speaker] += orig[f * 1000:t * 1000]
        return combined
