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
from dataclasses import dataclass, field, asdict as dataclass_as_dict
from typing import Any, List, Dict, Iterable, Callable

from pydantic import Field

from MAT.pipelines import Pipeline, PipelineResult, PipelineStepInput, PipelineStepResult, Slot
from MAT.tools import (
    TranscriptionInput, TranscriptionResult, TranscribeDiarizeTool, WordTupleSpeaker,
    DiarizerInput, DiarizationResult,
    SummaryInput, SummaryResult
)
from MAT.utils.config import Options
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
    models: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self):
        return dataclass_as_dict(self)


class PodcastOptions(Options):
    transcriber: str = Field("whisper", description="Speech to text backend.")
    diarizer: str = Field("sortformer", description="Diarization backend. Not used when the transcriber also "
                                                    "does the diarization.")
    identifier: str = Field("pyannote", description='Matches speakers to gold label clips. "none" keeps the '
                                                    'diarizer labels.')
    summarizer: str = Field("llm", description='Summary backend. "none" skips the summary.')


class PodcastPipeline(Pipeline):
    section = "podcast"
    description = "Transcript, speakers and summary for audio files (anything ffmpeg can decode)."
    Options = PodcastOptions
    slots = {"transcriber": Slot(), "diarizer": Slot(), "identifier": Slot(optional=True),
             "summarizer": Slot(optional=True)}

    @classmethod
    def accept(cls, f: str) -> bool:
        from pydub import AudioSegment
        # noinspection PyBroadException
        try:
            AudioSegment.from_file(f)
            return True
        except Exception:
            return False

    def _get_steps(self) -> Iterable[Callable[[PipelineStepInput], PipelineStepResult]]:
        import pydub
        used = {}

        def transcribe(step_input: PipelineStepInput) -> PipelineStepResult:
            transcriber = self.backend("transcriber", step_input.config)
            used["transcriber"] = transcriber
            d = transcriber.process(origin_data=TranscriptionInput(step_input.file), config=step_input.config)
            return PipelineStepResult(name="Transcription", data=d)

        def diarize(step_input: PipelineStepInput) -> PipelineStepResult:
            if isinstance(used.get("transcriber"), TranscribeDiarizeTool):
                self._LOGGER.info("The transcriber also diarizes, not running the diarizer")
                transcription = step_input.data("Transcription")
                return PipelineStepResult(name="Diarization", data=getattr(transcription, "diarization", None))
            diarizer = self.backend("diarizer", step_input.config)
            d = diarizer.process(origin_data=DiarizerInput(in_file=step_input.file), config=step_input.config)
            return PipelineStepResult(name="Diarization", data=d)

        def speaker_matching(step_input: PipelineStepInput) -> PipelineStepResult:
            diarization_result: DiarizationResult = step_input.data("Diarization")
            if diarization_result is None:
                return PipelineStepResult(name="Speaker Matching", data=None)
            identifier = self.backend("identifier", step_input.config)
            if identifier is None:
                return PipelineStepResult(name="Speaker Matching", data=diarization_result)
            if not identifier.can_match(step_input.config):
                # without gold labels the identifier answers None for every speaker, and building its audio would
                # decode and copy the whole episode first
                self._LOGGER.info(f"{identifier.backend_name} has nothing to match against, keeping the diarizer names")
                self.models.pop("identifier", None)
                return PipelineStepResult(name="Speaker Matching", data=diarization_result)
            a = pydub.AudioSegment.from_file(step_input.file)
            matched_speaker = diarization_result.speaker_matching(identifier=identifier, audio=a,
                                                                  config=step_input.config)
            return PipelineStepResult(name="Speaker Matching", data=matched_speaker)

        def creating_speaker_transcript(step_input: PipelineStepInput) -> PipelineStepResult:
            matched_speaker: DiarizationResult = step_input.data("Speaker Matching")
            transcription: TranscriptionResult = step_input.data("Transcription")
            if matched_speaker is None or transcription is None:
                return PipelineStepResult(name="Finalizing transcript", data=None)
            word_speaker = align_diarization_with_transcription(diarization=matched_speaker, transcript=transcription)
            squished_speaker = squish_word_speaker(word_speaker=word_speaker)
            full_transcript = "\n".join(word_speaker_to_transcript(word_speaker=squished_speaker))
            return PipelineStepResult(name="Finalizing transcript", data=(word_speaker, squished_speaker, full_transcript))

        def summarize_transcript(step_input: PipelineStepInput) -> PipelineStepResult:
            from os.path import basename
            transcripts = step_input.data("Finalizing transcript")
            full_transcript = transcripts[2] if transcripts is not None else None
            if full_transcript is None:
                return PipelineStepResult(name="Summarize transcript", data=None)
            summarizer = self.backend("summarizer", step_input.config)
            if summarizer is None:
                return PipelineStepResult(name="Summarize transcript", data=None)
            # A failing summary (API down, provider queue timeout, no key) must not throw away the transcript and
            # diarization we already have, so log it and write the episode without a summary.
            try:
                summary = summarizer.process(
                    origin_data=SummaryInput(full_transcript, additional_metadata={"filename": basename(step_input.file)}),
                    config=step_input.config
                )
            except Exception as e:
                self.__class__._LOGGER.exception(f"Summary failed for {basename(step_input.file)}, "
                                                 f"writing the results without a summary", exc_info=e)
                summary = None
            return PipelineStepResult(name="Summarize transcript", data=summary)

        def media_infos(step_input: PipelineStepInput) -> PipelineStepResult:
            transcription: TranscriptionResult = step_input.data("Transcription")
            lang = getattr(transcription, "language", None)
            duration_av = getattr(transcription, "duration_after_vad", None)
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
            result = step_results.get(k)
            return None if result is None else result.data

        transcripts = _try_get("Finalizing transcript")
        word_speaker, squished_speaker, full_transcript = (None, None, None) if transcripts is None else transcripts

        return PodcastOutput(
            media_info=_try_get("Media Info"),
            transcription=_try_get("Transcription"),
            diarization=_try_get("Diarization"), diarization_matched=_try_get("Speaker Matching"),
            word_speaker=word_speaker, squished_speaker=squished_speaker, full_transcript=full_transcript,
            summary=_try_get("Summarize transcript"),
            models=dict(self.models),
        )


__all__ = ["PodcastOutput", "PodcastPipeline", "MediaInfo"]
