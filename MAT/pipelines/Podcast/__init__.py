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
from dataclasses import dataclass, asdict as dataclass_as_dict
from typing import List, Dict, Any, Iterable, Callable
import logging

from MAT.pipelines import Pipeline, PipelineResult, ToolResult, T_out, PipelineStepInput, PipelineStepResult
from MAT.utils.config import ConfigElement, Config

from MAT.tools import (
    TransciptorWhisper, TranscriptionInput, TranscriptionResult, WordTupleSpeaker,
    DiarizerNEMO, DiarizerInput, DiarizationResult,
    SpeakerIdetificationPyannote,
    SummaryLLM, SummaryInput, SummaryResult
)
from MAT.utils.diarization import (
    align_diarization_with_transcription, squish_word_speaker, word_speaker_to_transcript
)


@dataclass
class MediaInfo:
    file_name: str
    duration: float
    duration_after_vad: float
    sample_rate: int
    max_dbfs: float
    rms: int
    language: str

    def as_dict(self):
        return dataclass_as_dict(self)


@dataclass
class PodcastOutput(PipelineResult):
    media_info: MediaInfo
    transcription: TranscriptionResult
    diarization: DiarizationResult
    diarization_matched: DiarizationResult
    word_speaker: List[WordTupleSpeaker]
    squished_speaker: List[WordTupleSpeaker]
    full_transcript: str
    summary: SummaryResult

    def as_dict(self):
        return dataclass_as_dict(self)


class PodcastPipeline(Pipeline):

    @classmethod
    def config_name(cls) -> str:
        return "Podcast"

    @classmethod
    def config_keys(cls) -> Dict[str, ConfigElement]:
        return {}

    @classmethod
    def accept(cls, f: str) -> bool:
        from pydub import AudioSegment
        # noinspection PyBroadException
        try:
            AudioSegment.from_file(f)
            return True
        except:
            pass
        return False

    def _get_steps(self) -> Iterable[Callable[[PipelineStepInput], PipelineStepResult]]:
        import pydub
        def transcribe(step_input: PipelineStepInput) -> PipelineStepResult:
            d = TransciptorWhisper().process(origin_data=TranscriptionInput(step_input.file), config=step_input.config)
            return PipelineStepResult(
                name="Transcription",
                data=d
            )

        def diarize(step_input: PipelineStepInput) -> PipelineStepResult:
            d = DiarizerNEMO().process(origin_data=DiarizerInput(in_file=step_input.file), config=step_input.config)
            return PipelineStepResult(
                name="Diarization",
                data=d
            )

        def speaker_matching(step_input: PipelineStepInput) -> PipelineStepResult:
            try:
                diarization_result: DiarizationResult = step_input.previous_results["Diarization"].data
                if diarization_result is None:
                    raise AttributeError()
            except (IndexError, KeyError, ValueError, TypeError, AttributeError):
                return PipelineStepResult(name="Speaker Matching", data=None)
            sm = SpeakerIdetificationPyannote()
            a = pydub.AudioSegment.from_file(step_input.file)
            matched_speaker = diarization_result.speaker_matching(identifier=sm, audio=a, config=step_input.config)
            return PipelineStepResult(
                name="Speaker Matching",
                data=matched_speaker
            )

        def creating_speaker_transcript(step_input: PipelineStepInput) -> PipelineStepResult:
            try:
                matched_speaker: DiarizationResult = step_input.previous_results["Speaker Matching"].data
                transcription: TranscriptionResult = step_input.previous_results["Transcription"].data
                if matched_speaker is None or transcription is None:
                    raise AttributeError()
            except (IndexError, KeyError, ValueError, TypeError, AttributeError):
                return PipelineStepResult(name="Finalizing transcript", data=None)
            word_speaker = align_diarization_with_transcription(diarization=matched_speaker, transcript=transcription)
            squished_speaker = squish_word_speaker(word_speaker=word_speaker)
            full_transcript = "\n".join(word_speaker_to_transcript(word_speaker=squished_speaker))
            return PipelineStepResult(
                name="Finalizing transcript",
                data=(word_speaker, squished_speaker, full_transcript)
            )

        def summarize_transcript(step_input: PipelineStepInput) -> PipelineStepResult:
            try:
                full_transcript: str = step_input.previous_results["Finalizing transcript"].data[2]
                if full_transcript is None:
                    raise AttributeError()
            except (IndexError, KeyError, ValueError, TypeError, AttributeError):
                return PipelineStepResult(name="Summarize transcript", data=None)
            summary = SummaryLLM().process(origin_data=SummaryInput(full_transcript), config=step_input.config)
            return PipelineStepResult(
                name="Summarize transcript",
                data=summary
            )

        def media_infos(step_input: PipelineStepInput) -> PipelineStepResult:
            try:
                transcription: TranscriptionResult = step_input.previous_results["Transcription"].data
                lang = transcription.language
                duration_av = transcription.duration_after_vad
            except (IndexError, KeyError, ValueError, TypeError, AttributeError):
                lang = None
                duration_av = None
            a = pydub.AudioSegment.from_file(step_input.file)
            return PipelineStepResult(
                name="Media Info",
                data=MediaInfo(
                    file_name=os.path.basename(step_input.file),
                    duration=a.duration_seconds, sample_rate=a.frame_rate, max_dbfs=a.max_dBFS, rms=a.rms,
                    language="" if lang is None else lang,
                    duration_after_vad=a.duration_seconds if duration_av is None else duration_av,
                )
            )

        return [transcribe, diarize, speaker_matching, creating_speaker_transcript, summarize_transcript, media_infos]

    def _finalize_result(self, step_results: Dict[str, PipelineStepResult]) -> PodcastOutput:

        def _try_get(k: str):
            try:
                return step_results[k].data
            except (IndexError, KeyError, ValueError, TypeError, AttributeError):
                return None

        transcription = _try_get("Transcription")
        diarization = _try_get("Diarization")
        diarization_matched = _try_get("Speaker Matching")
        transcripts = _try_get("Finalizing transcript")
        word_speaker, squished_speaker, full_transcript = (None, None, None) if transcripts is None else transcription
        summary = _try_get("Summarize transcript")
        media_info = _try_get("Media Info")

        return PodcastOutput(
            media_info=media_info,
            transcription=transcription,
            diarization=diarization, diarization_matched=diarization_matched,
            word_speaker=word_speaker, squished_speaker=squished_speaker, full_transcript=full_transcript,
            summary=summary,
        )


__all__ = ["PodcastOutput", "PodcastPipeline"]
__all__ += ["__all__"]
