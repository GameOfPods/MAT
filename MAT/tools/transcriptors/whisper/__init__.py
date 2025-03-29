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
from typing import Dict, Optional, List

from MAT.tools.transcriptors import TranscriptionInput, TransciptionTool, TranscriptionResult, WordTuple
from MAT.utils.config import ConfigElement, Config


class TransciptorWhisper(TransciptionTool):
    _LOGGER = logging.getLogger(__name__)

    @classmethod
    def config_name(cls) -> str:
        return "Whisper"

    @classmethod
    def config_keys(cls) -> Dict[str, ConfigElement]:
        import torch
        from os import cpu_count
        return {
            "device": ConfigElement(
                default_value="cuda" if torch.cuda.is_available() else "cpu",
                argparse_kwargs={
                    "help": "Device to run the model on. Default: %(default)s",
                    "type": str,
                }
            ),
            "model": ConfigElement(
                default_value="large-v2",
                argparse_kwargs={
                    "help": "Model to use for whisper model. Default: %(default)s", "type": str,
                }
            ),
            "cpu-count": ConfigElement(
                default_value=cpu_count(),
                argparse_kwargs={
                    "help": "Amount of cpu cores to use. Default: %(default)s", "type": int,
                }
            ),
            "compute-type": ConfigElement(
                default_value="int8",
                argparse_kwargs={
                    "help": "Compute type. Default: %(default)s", "type": str,
                }
            ),
            "beam-size": ConfigElement(
                default_value=5,
                argparse_kwargs={
                    "help": "Beam size for Whisper transcription. Default: %(default)s", "type": int,
                }
            )
        }

    def process(self, origin_data: TranscriptionInput, config: Config) -> Optional[TranscriptionResult]:
        from faster_whisper import WhisperModel
        import whisperx
        from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH
        import tqdm

        cfg = config.get_config(key=self)
        model = WhisperModel(cfg["model"], device=cfg["device"], compute_type=cfg["compute-type"],
                             cpu_threads=cfg["cpu-count"])
        segments, info = model.transcribe(origin_data.input_file, beam_size=cfg["beam-size"], vad_filter=True, )

        segments_as_dict = []
        for segment in tqdm.tqdm(segments, unit="segment", leave=False, desc="Transcribing"):
            segments_as_dict.append(segment.__dict__)

        if info.language in set().union(DEFAULT_ALIGN_MODELS_TORCH.keys(), DEFAULT_ALIGN_MODELS_HF.keys()):
            align_model, meta = whisperx.load_align_model(language_code=info.language, device=cfg["device"])
            aligned = whisperx.align(
                transcript=segments_as_dict,
                model=align_model,
                align_model_metadata=meta,
                audio=origin_data.input_file,
                device=cfg["device"],
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
            word_timestamps = []
            for s in segments_as_dict:
                for w in s["words"]:
                    word_timestamps.append(WordTuple(start=w[0], end=w[1], word=w[2]))

        return TranscriptionResult(word_timings=word_timestamps, language=info.language, duration=info.duration,
                                   duration_after_vad=info.duration_after_vad)

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
