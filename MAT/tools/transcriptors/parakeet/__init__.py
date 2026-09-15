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
from typing import List, Optional, Sequence
from uuid import uuid4

from pydantic import Field

from MAT.registry import register, require

require("nemo", extra="parakeet")

from MAT.tools.transcriptors import TranscriptionInput, TransciptionTool, TranscriptionResult, WordTuple  # noqa: E402
from MAT.utils.audio import Window  # noqa: E402
from MAT.utils.config import Config, Options  # noqa: E402


class ParakeetOptions(Options):
    model: str = Field("nvidia/parakeet-tdt-0.6b-v3", description="NeMo ASR model with word timestamps.")
    device: str = Field("auto", description='"auto" uses the GPU if there is one, or set "cpu" / "cuda".')
    segment_length: int = Field(10 * 60, ge=30, description="Longest piece of audio in seconds transcribed at once. "
                                                            "Longer audio is cut at a quiet spot. Lower it if the "
                                                            "GPU runs out of memory.")
    local_attention: bool = Field(True, description="Local attention (256 frames to each side) instead of full "
                                                    "attention. Needs far less GPU memory on long pieces, NVIDIA "
                                                    "says it works for up to 3 hours.")
    language: Optional[str] = Field(None, description="Language code stored in the result. Parakeet detects the "
                                                      "language on its own but doesn't report it, so by default "
                                                      "it's guessed from the transcript.")


def guess_language(text: str) -> Optional[str]:
    """Language code (de, en, ...) of a text, None if there's nothing to go by."""
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException

    if not text.strip():
        return None
    # langdetect gives different answers on short texts unless it's seeded
    DetectorFactory.seed = 0
    try:
        return detect(text)
    except LangDetectException:
        return None


@register("transcriber", "parakeet", description="NVIDIA Parakeet TDT 0.6B v3, 25 European languages, word timestamps")
class TranscriptorParakeet(TransciptionTool):
    Options = ParakeetOptions
    packages = ("nemo-toolkit",)
    _LOGGER = logging.getLogger(__name__)

    def process(self, origin_data: TranscriptionInput, config: Config) -> Optional[TranscriptionResult]:
        import numpy as np
        from pydub import AudioSegment

        from MAT.utils.audio import plan_windows
        from MAT.utils.device import free_gpu_memory, resolve_device

        options = config.options(self)
        device = resolve_device(options.device)
        sound = AudioSegment.from_file(origin_data.input_file).set_channels(1).set_frame_rate(16000)
        sound = sound.set_sample_width(2)
        windows = plan_windows(np.frombuffer(sound.raw_data, dtype=np.int16), sound.frame_rate,
                               max_length=options.segment_length)
        folder = os.path.join(config.work_directory, f"parakeet.{uuid4()}")
        os.makedirs(folder, exist_ok=True)
        files = []
        for i, window in enumerate(windows):
            path = os.path.join(folder, f"piece.{i}.wav")
            sound[int(window.start * 1000):int(window.end * 1000)].export(path, format="wav")
            files.append(path)

        self._LOGGER.info(f"Transcribing {len(windows)} audio piece(s) of at most {options.segment_length} s with "
                          f"{options.model} on {device}")
        model = self._load_model(options, device)
        try:
            hypotheses = model.transcribe(files, batch_size=1, timestamps=True, verbose=False)
        finally:
            del model
            free_gpu_memory()
        # some NeMo versions return (best hypotheses, all hypotheses)
        if isinstance(hypotheses, tuple):
            hypotheses = hypotheses[0]

        words = self._merge_pieces(windows, [self._words(h) for h in hypotheses])
        language = options.language or guess_language(" ".join(w.word for w in words))
        self._LOGGER.info(f"Transcribed {len(words)} words, language {language}")
        return TranscriptionResult(word_timings=words, language=language, duration=sound.duration_seconds)

    @staticmethod
    def _load_model(options: ParakeetOptions, device: str):
        from nemo.collections.asr.models import ASRModel

        from MAT.utils import timeout_retry

        model = timeout_retry(func=ASRModel.from_pretrained, func_args=(options.model,),
                              func_kwargs={"map_location": device}, time_out=60, retries=5)
        model.eval()
        if options.local_attention:
            model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[256, 256])
        return model

    @staticmethod
    def _words(hypothesis) -> List[WordTuple]:
        timestamp = getattr(hypothesis, "timestamp", None) or {}
        return [WordTuple(start=float(w["start"]), end=float(w["end"]), word=str(w["word"]))
                for w in timestamp.get("word") or [] if str(w.get("word", "")).strip()]

    @staticmethod
    def _merge_pieces(windows: Sequence[Window], pieces: Sequence[List[WordTuple]]) -> List[WordTuple]:
        """Shifts the word times of every piece to absolute times. A word belongs to the piece that owns its middle."""
        words = []
        for window, piece in zip(windows, pieces):
            for word in piece:
                start, end = word.start + window.start, word.end + window.start
                if window.owns(start, end):
                    words.append(WordTuple(start=round(start, 3), end=round(end, 3), word=word.word))
        return sorted(words, key=lambda w: (w.start, w.end))


__all__ = ["TranscriptorParakeet", "ParakeetOptions", "guess_language"]
