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
from typing import Optional, List

from pydantic import Field

from MAT.registry import register, require

require("faster_whisper", "whisperx", "ctranslate2", extra="whisper")

from MAT.tools.transcriptors import TranscriptionInput, TransciptionTool, TranscriptionResult, WordTuple  # noqa: E402
from MAT.utils.config import Config, Options  # noqa: E402


class WhisperOptions(Options):
    model: str = Field("large-v3-turbo", description="Whisper model name (large-v3-turbo, large-v3, medium, ...) "
                                                     "or a folder with a CTranslate2 model.")
    device: str = Field("auto", description='"auto" uses the GPU if there is one, or set "cpu" / "cuda".')
    compute_type: str = Field("auto", description='CTranslate2 compute type. "auto" picks the fastest one the device '
                                                  'supports (int8_float32 on CPU and GTX 10xx cards).')
    cpu_count: int = Field(default_factory=lambda: os.cpu_count() or 1, ge=1, description="CPU threads to use.")
    beam_size: int = Field(5, ge=1, description="Beam size for decoding.")


@register("transcriber", "whisper", description="faster-whisper, words aligned with whisperx")
class TransciptorWhisper(TransciptionTool):
    Options = WhisperOptions
    packages = ("faster-whisper", "whisperx", "ctranslate2")
    _LOGGER = logging.getLogger(__name__)

    def process(self, origin_data: TranscriptionInput, config: Config) -> Optional[TranscriptionResult]:
        import sys
        from time import perf_counter
        from datetime import timedelta
        from faster_whisper import WhisperModel, decode_audio
        import whisperx
        from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH
        import tqdm
        import math

        from MAT.utils.device import ct2_compute_type, resolve_device

        options = config.options(self)
        device = resolve_device(options.device)
        compute_type = ct2_compute_type(device=device, requested=options.compute_type)
        self._LOGGER.info(f"Loading whisper {options.model} on {device} with compute type {compute_type}")
        model = WhisperModel(options.model, device=device, compute_type=compute_type, cpu_threads=options.cpu_count)
        # decode once, faster-whisper and the whisperx alignment both work on 16 kHz mono float arrays
        audio = decode_audio(origin_data.input_file)

        # Detect the language first. Only when whisperx has no alignment model for it we ask whisper itself for word
        # timestamps, they cost extra time and the whisperx alignment is more precise.
        language, language_probability, _ = model.detect_language(audio=audio, vad_filter=True)
        has_align_model = language in set().union(DEFAULT_ALIGN_MODELS_TORCH.keys(), DEFAULT_ALIGN_MODELS_HF.keys())
        self._LOGGER.info(f"Detected language {language} ({language_probability:.0%}). "
                          f"{'Aligning words with whisperx' if has_align_model else 'No whisperx alignment model, using whisper word timestamps'}")

        segments, info = model.transcribe(audio, language=language, beam_size=options.beam_size, vad_filter=True,
                                          word_timestamps=not has_align_model)

        segment_lengths = []
        segments_as_dict = []

        t1 = perf_counter()
        with tqdm.tqdm(segments, unit="segment", leave=False, desc="Transcribing") as pb:
            for segment in pb:
                segment_lengths.append(segment.end - segment.start)
                avg_length = sum(segment_lengths) / len(segment_lengths)
                pb.total = math.ceil(info.duration / avg_length)
                pb.set_description(f"Transcribing {avg_length:.1f}s segments")
                # Segment is a dataclass in newer faster-whisper versions and a NamedTuple in older ones
                try:
                    segments_as_dict.append(segment.__dict__)
                except AttributeError:
                    segments_as_dict.append(segment._asdict())
        t2 = perf_counter()

        sys.stdout.flush()
        sys.stderr.flush()
        self._LOGGER.info(f"Finished transcription of {len(segment_lengths)} segments in {timedelta(seconds=t2-t1)}")
        self._LOGGER.info(f"Transcribed {len(segment_lengths)} segments, "
                          f"that total to {timedelta(seconds=sum(segment_lengths))} of audio. "
                          f"Audio file has a length of {timedelta(seconds=info.duration)}")

        if len(segments_as_dict) == 0:
            self._LOGGER.warning(f"No speech found in {origin_data.input_file}")
            return TranscriptionResult(word_timings=[], language=info.language, duration=info.duration,
                                       duration_after_vad=info.duration_after_vad)

        if has_align_model:
            align_model, meta = whisperx.load_align_model(language_code=info.language, device=device)
            aligned = whisperx.align(
                transcript=segments_as_dict,
                model=align_model,
                align_model_metadata=meta,
                audio=audio,
                device=device,
                print_progress=False,
            )
            word_timestamps = TransciptorWhisper._fix_broken_times(
                words=[
                    WordTuple(
                        start=x.get("start", None), end=x.get("end", None), word=x.get("word", None)
                    ) for x in aligned["word_segments"]
                ],
                init=segments_as_dict[0].get("start", None),
                fin=segments_as_dict[-1].get("end", None),
            )

        else:
            word_timestamps = TransciptorWhisper._words_from_segments(segments=segments_as_dict)

        return TranscriptionResult(word_timings=word_timestamps, language=info.language, duration=info.duration,
                                   duration_after_vad=info.duration_after_vad)

    @staticmethod
    def _words_from_segments(segments: List[dict]) -> List[WordTuple]:
        # Used when whisperx has no alignment model, whisper is asked for word timestamps then.
        # If a segment still has no words it becomes one entry with the segment timings.
        ret = []
        for s in segments:
            words = s.get("words") or []
            if len(words) == 0:
                ret.append(WordTuple(start=s.get("start"), end=s.get("end"), word=(s.get("text") or "").strip()))
                continue
            for w in words:
                if isinstance(w, dict):
                    ret.append(WordTuple(start=w.get("start"), end=w.get("end"), word=w.get("word")))
                elif hasattr(w, "start"):
                    ret.append(WordTuple(start=w.start, end=w.end, word=w.word))
                else:
                    ret.append(WordTuple(start=w[0], end=w[1], word=w[2]))
        return ret

    @staticmethod
    def _merge_words(words: List[WordTuple], fin: float, idx: int = 0) -> Optional[float]:
        # if current word is the last word
        if idx >= len(words) - 1:
            return words[-1].start
        n = idx + 1
        while idx < len(words) - 1:
            if words[n].start is None:
                words[idx].word += f" {words[n].word}" if words[n].word else ""
                words[n].word = None
                if words[n].end is not None:
                    return words[n].end
                if n + 1 >= len(words):
                    return fin
                n += 1

            else:
                return words[n].start
        return words[-1].start

    @staticmethod
    def _fix_broken_times(words: List[WordTuple], init: Optional[float] = 0, fin: Optional[float] = None) -> List[
        WordTuple]:
        if len(words) == 0:
            return words
        if words[0].start is None:
            words[0].start = init if init is not None else 0
        if words[0].end is None:
            words[0].end = TransciptorWhisper._merge_words(words=words, idx=0, fin=fin)

        res = [words[0]]

        for i, w in enumerate(words[1:], start=1):
            if w.word is None:
                continue
            if w.start is None:
                w.start = words[i - 1].end
            if w.end is None:
                w.end = TransciptorWhisper._merge_words(words=words, idx=i, fin=fin)
            res.append(w)
        return res
