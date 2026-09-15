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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
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
    link_threshold: float = Field(0.3, description="Minimum similarity of the pyannote speaker embeddings for a "
                                                   "speaker of one audio piece to be linked to a speaker of an "
                                                   "earlier piece. Below it the speaker counts as new.")
    embedding_model: str = Field("pyannote/embedding", description="pyannote speaker embedding model for linking "
                                                                   "speakers between pieces.")


# Similarity function for linking: gold speaker -> their audio from earlier pieces, audios of the new piece ->
# (gold speaker names, similarity matrix with one row per audio and one column per gold speaker)
Similarity = Callable[[Dict[str, List[Any]], List[Any]], Tuple[List[str], Any]]


def link_speakers(similarity, local: Sequence[str], known: Sequence[str],
                  threshold: float) -> Dict[str, Optional[str]]:
    """Maps the speakers of a new piece to known speakers, one to one with the highest total similarity. Pairs at or
    below the threshold stay unmatched (None)."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    mapping: Dict[str, Optional[str]] = {speaker: None for speaker in local}
    matrix = np.nan_to_num(np.asarray(similarity, dtype=float), nan=-1.0)
    if not local or not known or matrix.size == 0:
        return mapping
    for row, column in zip(*linear_sum_assignment(matrix, maximize=True)):
        if matrix[row, column] > threshold:
            mapping[local[row]] = known[column]
    return mapping


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

        def similarity(gold: Dict[str, List[AudioSegment]], audios: List[AudioSegment]):
            from MAT.tools.speakeridentification.pyannote import SpeakerIdetificationPyannote as Identifier

            joined = {name: sum(parts[1:], parts[0]) for name, parts in gold.items()}
            return Identifier.similarities(model=options.embedding_model, device=device,
                                           gold={name: (a, a.frame_rate) for name, a in joined.items()},
                                           audios=[(a, a.frame_rate) for a in audios])

        linked = self._link_pieces(clean_segments, combinations, similarity, threshold=options.link_threshold)
        return self._merge_windows(windows, linked)

    @classmethod
    def _link_pieces(cls, segments_per_piece: Sequence[Dict[str, List[Tuple[float, float]]]],
                     audio_per_piece: Sequence[Dict[str, Any]], similarity: Similarity, threshold: float,
                     history: int = 5) -> List[Dict[str, List[Tuple[float, float]]]]:
        """Gives the local speakers of every piece global names. Speakers of later pieces are compared with the audio
        of the speakers in up to `history` earlier pieces and linked one to one, so two local speakers never end up
        under the same name and no segments get lost. Unmatched speakers get a new name."""
        linked_segments: List[Dict[str, List[Tuple[float, float]]]] = []
        linked_audio: List[Dict[str, List[Any]]] = []
        next_id = 0
        for index, (segments, audios) in enumerate(zip(segments_per_piece, audio_per_piece)):
            local = list(segments)
            mapping: Dict[str, Optional[str]] = {speaker: None for speaker in local}
            gold: Dict[str, List[Any]] = defaultdict(list)
            for previous in linked_audio[max(0, index - history):index]:
                for name, parts in previous.items():
                    gold[name].extend(parts)
            if local and gold:
                known, matrix = similarity(dict(gold), [audios[speaker] for speaker in local])
                mapping = link_speakers(matrix, local, known, threshold)
                cls._LOGGER.info(f"Linking audio piece {index + 1}: " + ", ".join(
                    f"{speaker} -> {mapping[speaker] or 'new'}" for speaker in local))
            piece_segments: Dict[str, List[Tuple[float, float]]] = {}
            piece_audio: Dict[str, List[Any]] = {}
            for speaker in local:
                name = mapping[speaker]
                if name is None:
                    name = f"sprecher_{next_id}"
                    next_id += 1
                piece_segments.setdefault(name, []).extend(segments[speaker])
                if speaker in audios:
                    piece_audio.setdefault(name, []).append(audios[speaker])
            linked_segments.append(piece_segments)
            linked_audio.append(piece_audio)
        return linked_segments

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
