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
from typing import Any, List, Dict, Iterable, Callable, Literal, Optional, Set, Tuple

from pydantic import Field

from MAT.pipelines import Pipeline, PipelineResult, PipelineStepInput, PipelineStepResult, Slot
from MAT.tools import (
    TranscriptionInput, TranscriptionResult, TranscribeDiarizeTool, WordTupleSpeaker,
    DiarizerInput, DiarizationResult,
    SpeakerNamingInput,
    SummaryInput, SummaryResult
)
from MAT.utils.config import Options
from MAT.utils.diarization import (
    align_diarization_with_transcription, assign_speakers, squish_word_speaker, word_speaker_to_transcript
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
    # speaker id -> {"name": ..., "library_id": ...} for speakers the library knows
    speaker_library: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def as_dict(self):
        return dataclass_as_dict(self)


def _rename_speakers(renames: Dict[str, str], diarization: DiarizationResult,
                     word_speaker: List[WordTupleSpeaker], squished: List[WordTupleSpeaker]):
    """Gives speakers new names everywhere: in the diarization, in the words and in the transcript lines."""
    def rename(words: List[WordTupleSpeaker]) -> List[WordTupleSpeaker]:
        return [WordTupleSpeaker(word=w.word, speaker={renames.get(s, s) for s in w.speaker}) for w in words]

    renamed_squished = rename(squished)
    renamed_diarization = DiarizationResult({renames.get(s, s): diarization.get_diarization(speaker=s)
                                             for s in diarization.speaker})
    return (renamed_diarization, rename(word_speaker), renamed_squished,
            "\n".join(word_speaker_to_transcript(word_speaker=renamed_squished)))


def _speaker_state(data: Callable[[str], Any]) -> Optional[Dict[str, Any]]:
    """The speakers as far as the pipeline got: after naming, after the library, or straight from the transcript.
    A dict with diarization, word_speaker, squished, transcript and speakers (name -> {"name", "library_id"} for
    voices the library knows)."""
    for step in ("Speaker Names", "Speaker Library"):
        state = data(step)
        if state is not None:
            return state
    transcripts, matched = data("Finalizing transcript"), data("Speaker Matching")
    if transcripts is None or matched is None:
        return None
    return {"diarization": matched, "word_speaker": transcripts[0], "squished": transcripts[1],
            "transcript": transcripts[2], "speakers": {}}


def _with_renames(state: Dict[str, Any], renames: Dict[str, str], **extra) -> Dict[str, Any]:
    diarization, word_speaker, squished, transcript = _rename_speakers(
        renames, state["diarization"], state["word_speaker"], state["squished"])
    return dict(state, diarization=diarization, word_speaker=word_speaker, squished=squished, transcript=transcript,
                **extra)


def _speaker_audio(audio, diarization: DiarizationResult, speaker: str, seconds: float):
    """Up to `seconds` of what one speaker says, enough for an embedding without copying a whole episode."""
    from MAT.utils.audio import speaker_clip

    return speaker_clip(audio, diarization.get_diarization(speaker=speaker), seconds)


class PodcastOptions(Options):
    transcriber: str = Field("whisper", description="Speech to text backend.")
    diarizer: str = Field("pyannote-diarization",
                          description="Diarization backend. Not used when the transcriber also does the "
                                      "diarization. The default needs a Hugging Face login, sortformer doesn't.")
    identifier: str = Field("pyannote", description='Matches speakers to gold label clips. "none" keeps the '
                                                    'diarizer labels.')
    namer: str = Field("none", description='Names the speakers that the identifier left unnamed, from what is said '
                                           'in the transcript. "none" skips it. Gold labels always win.')
    word_speakers: Literal["single", "overlap"] = Field(
        "single", description='"single" gives every word the one speaker who talks longest during it and fills small '
                              'gaps from the words around it. "overlap" gives a word every speaker who talks during '
                              'it (lines like "alice & bob") and leaves words outside all segments without one.')
    min_turn: float = Field(0.5, ge=0, description='Seconds. With word-speakers "single", a shorter turn in the middle '
                                                   'of somebody else\'s sentence goes back to them. 0 turns it off.')
    max_gap: float = Field(1.0, ge=0, description='Seconds. With word-speakers "single", a word outside all segments '
                                                  'takes the closest speaker this near to it.')
    match_seconds: float = Field(120.0, gt=0, description="Seconds of every diarizer speaker compared with the gold "
                                                          "label clips. More isn't better: all of a speaker's hour "
                                                          "long audio matched nobody in a real episode.")
    speaker_library: Optional[str] = Field(None, description="Folder with the speaker library. A voice that got a "
                                                             "name once is recognized in later episodes and keeps "
                                                             "the same id. Not set: no library.")
    speaker_library_threshold: float = Field(0.6, ge=0, le=1, description="How similar a voice has to be to a known "
                                                                          "one to count as the same person.")
    speaker_library_seconds: float = Field(120.0, gt=0, description="Seconds of a speaker the library listens to "
                                                                    "when it makes their voice print.")
    speaker_library_learns: Literal["gold", "all", "never"] = Field(
        "gold", description='What the library learns: "gold" only names that came from gold label clips, "all" also '
                            'names the transcript gave us, "never" only reads and writes nothing.')
    summarizer: str = Field("llm", description='Summary backend. "none" skips the summary.')


class PodcastPipeline(Pipeline):
    section = "podcast"
    description = "Transcript, speakers and summary for audio files (anything ffmpeg can decode)."
    Options = PodcastOptions
    slots = {"transcriber": Slot(), "diarizer": Slot(), "identifier": Slot(optional=True),
             "namer": Slot(optional=True), "summarizer": Slot(optional=True)}

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
            matched_speaker = diarization_result.speaker_matching(
                identifier=identifier, audio=a, config=step_input.config,
                seconds=step_input.config.options(self).match_seconds)
            return PipelineStepResult(name="Speaker Matching", data=matched_speaker)

        def creating_speaker_transcript(step_input: PipelineStepInput) -> PipelineStepResult:
            matched_speaker: DiarizationResult = step_input.data("Speaker Matching")
            transcription: TranscriptionResult = step_input.data("Transcription")
            if matched_speaker is None or transcription is None:
                return PipelineStepResult(name="Finalizing transcript", data=None)
            options = step_input.config.options(self)
            if options.word_speakers == "overlap":
                word_speaker = align_diarization_with_transcription(diarization=matched_speaker,
                                                                    transcript=transcription)
            else:
                word_speaker = assign_speakers(matched_speaker, transcription.word_timings or [],
                                               max_gap=options.max_gap, min_turn=options.min_turn)
            squished_speaker = squish_word_speaker(word_speaker=word_speaker)
            full_transcript = "\n".join(word_speaker_to_transcript(word_speaker=squished_speaker))
            return PipelineStepResult(name="Finalizing transcript", data=(word_speaker, squished_speaker, full_transcript))

        def speaker_library(step_input: PipelineStepInput) -> PipelineStepResult:
            # runs before the LLM naming: a voice the library knows needs no LLM call
            options = step_input.config.options(self)
            state = _speaker_state(step_input.data)
            if not options.speaker_library or state is None:
                return PipelineStepResult(name="Speaker Library", data=None)
            try:
                from MAT.tools.speakeridentification.pyannote import SpeakerIdetificationPyannote as Embedder
            except ImportError as e:
                self.__class__._LOGGER.warning(f"The speaker library needs the pyannote extra, skipping it: {e}")
                return PipelineStepResult(name="Speaker Library", data=None)
            import pydub

            from MAT.utils.device import resolve_device
            from MAT.utils.speaker_library import SpeakerLibrary

            diarization: DiarizationResult = state["diarization"]
            raw: DiarizationResult = step_input.data("Diarization")
            labels = raw.speaker if raw is not None else set()
            # the identifier renamed whatever it matched to a gold clip, so those names are no diarizer labels
            from_gold = diarization.speaker - labels

            library = SpeakerLibrary.open(options.speaker_library)
            audio = pydub.AudioSegment.from_file(step_input.file)
            speakers = sorted(diarization.speaker)
            clips = [_speaker_audio(audio, diarization, speaker, options.speaker_library_seconds)
                     for speaker in speakers]
            identifier_options = step_input.config.options(Embedder)
            vectors = Embedder.embeddings(model=identifier_options.model,
                                          audios=[(clip, clip.frame_rate) for clip in clips],
                                          device=resolve_device(identifier_options.device),
                                          use_hf_token=not identifier_options.no_hf_token)

            episode = os.path.basename(step_input.file)
            renames: Dict[str, str] = {}
            known_speakers: Dict[str, Dict[str, str]] = dict(state.get("speakers") or {})
            voices: Dict[str, Any] = {}
            changed = False
            for speaker, vector in zip(speakers, vectors):
                if vector is None:
                    continue
                # a gold clip beats the library, that voice keeps the name the clips gave it
                found = None if speaker in from_gold else library.match(vector, options.speaker_library_threshold)
                if found is not None:
                    entry, score = found
                    self.__class__._LOGGER.info(f"The library knows {speaker} as {entry.name} ({score:.2f})")
                    if entry.name != speaker:
                        renames[speaker] = entry.name
                    known_speakers[entry.name] = {"name": entry.name, "library_id": entry.library_id}
                    voices[entry.name] = vector
                    if options.speaker_library_learns != "never":
                        library.remember(entry.name, vector, source=entry.source, episode=episode)
                        changed = True
                    continue
                voices[speaker] = vector
                if speaker in labels:
                    continue  # still a diarizer label, maybe the LLM names it later
                if options.speaker_library_learns == "never":
                    self.__class__._LOGGER.info(f"Not putting {speaker} into the library, speaker-library-learns "
                                                f"is never")
                    continue
                entry = library.remember(speaker, vector, source="gold", episode=episode)
                self.__class__._LOGGER.info(f"The library learned the voice of {speaker} (from the gold labels)")
                known_speakers[speaker] = {"name": entry.name, "library_id": entry.library_id}
                changed = True

            if changed:
                self.__class__._LOGGER.info(f"Speaker library saved to {library.save()}")
            # voices go along so the naming step can put names from the transcript into the library
            return PipelineStepResult(name="Speaker Library",
                                      data=_with_renames(state, renames, speakers=known_speakers, voices=voices))

        def name_speakers(step_input: PipelineStepInput) -> PipelineStepResult:
            state = _speaker_state(step_input.data)
            if state is None:
                return PipelineStepResult(name="Speaker Names", data=None)
            namer = self.backend("namer", step_input.config)
            if namer is None:
                return PipelineStepResult(name="Speaker Names", data=None)

            diarization: DiarizationResult = state["diarization"]
            raw: DiarizationResult = step_input.data("Diarization")
            # gold clips and the library renamed what they recognized. Only diarizer labels are up for naming.
            unnamed = diarization.speaker & (raw.speaker if raw is not None else diarization.speaker)
            if not unnamed:
                self.__class__._LOGGER.info("Every speaker has a name already, not asking the LLM")
                return PipelineStepResult(name="Speaker Names", data=None)
            transcription: TranscriptionResult = step_input.data("Transcription")
            try:
                found = namer.process(
                    origin_data=SpeakerNamingInput(lines=state["transcript"].splitlines(),
                                                   speakers=sorted(diarization.speaker),
                                                   language=getattr(transcription, "language", None)),
                    config=step_input.config,
                )
            except Exception as e:
                self.__class__._LOGGER.exception("Naming the speakers failed, keeping the names we have", exc_info=e)
                return PipelineStepResult(name="Speaker Names", data=None)

            known = state.get("speakers") or {}
            renames = {}
            for name in (found.names if found is not None else []):
                if name.speaker in unnamed:
                    renames[name.speaker] = name.name
                elif name.speaker != name.name:
                    source = "speaker library" if name.speaker in known else "gold labels"
                    self.__class__._LOGGER.warning(
                        f'The transcript calls {name.speaker} "{name.name}" ({name.evidence}), but the {source} '
                        f"matched that voice to {name.speaker}. Keeping {name.speaker}.")
            if not renames:
                return PipelineStepResult(name="Speaker Names", data=None)
            self.__class__._LOGGER.info("Named " + ", ".join(f"{old} -> {new}" for old, new in renames.items()))

            known_speakers = dict(known)
            options = step_input.config.options(self)
            voices = state.get("voices") or {}
            if options.speaker_library and voices:
                if options.speaker_library_learns == "all":
                    from MAT.utils.speaker_library import SpeakerLibrary

                    library = SpeakerLibrary.open(options.speaker_library)
                    episode = os.path.basename(step_input.file)
                    for old, new in renames.items():
                        if voices.get(old) is None:
                            continue
                        entry = library.remember(new, voices[old], source="llm", episode=episode)
                        known_speakers[new] = {"name": entry.name, "library_id": entry.library_id}
                        self.__class__._LOGGER.info(f"The library learned the voice of {new} (from the transcript)")
                    self.__class__._LOGGER.info(f"Speaker library saved to {library.save()}")
                else:
                    self.__class__._LOGGER.info(f"Not putting {', '.join(renames.values())} into the library, the "
                                                f"names came from the transcript and speaker-library-learns is "
                                                f"{options.speaker_library_learns}")
            return PipelineStepResult(name="Speaker Names",
                                      data=_with_renames(state, renames, speakers=known_speakers))

        def summarize_transcript(step_input: PipelineStepInput) -> PipelineStepResult:
            from os.path import basename
            state = _speaker_state(step_input.data)
            full_transcript = None if state is None else state["transcript"]
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

        return [transcribe, diarize, speaker_matching, creating_speaker_transcript, speaker_library, name_speakers,
                summarize_transcript, media_infos]

    def _finalize_result(self, step_results: Dict[str, PipelineStepResult]) -> PodcastOutput:

        def _try_get(k: str):
            result = step_results.get(k)
            return None if result is None else result.data

        transcripts = _try_get("Finalizing transcript")
        word_speaker, squished_speaker, full_transcript = (None, None, None) if transcripts is None else transcripts
        matched = _try_get("Speaker Matching")
        # the library and the naming step return everything again with the new names
        state = _speaker_state(_try_get)
        if state is not None:
            matched, word_speaker = state["diarization"], state["word_speaker"]
            squished_speaker, full_transcript = state["squished"], state["transcript"]
        library = {} if state is None else state.get("speakers") or {}

        return PodcastOutput(
            media_info=_try_get("Media Info"),
            transcription=_try_get("Transcription"),
            diarization=_try_get("Diarization"), diarization_matched=matched,
            word_speaker=word_speaker, squished_speaker=squished_speaker, full_transcript=full_transcript,
            summary=_try_get("Summarize transcript"),
            models=dict(self.models),
            speaker_library=library,
        )


__all__ = ["PodcastOutput", "PodcastPipeline", "MediaInfo"]
