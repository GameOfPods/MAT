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
import uuid
from dataclasses import dataclass, field, asdict as dataclass_as_dict
from typing import Any, List, Dict, Iterable, Callable, Literal, Optional, Sequence, Set, Tuple, Union

from pydantic import Field

from MAT.pipelines import Pipeline, PipelineResult, PipelineStepInput, PipelineStepResult, Slot
from MAT.tools import (
    TranscriptionInput, TranscriptionResult, TranscribeDiarizeTool, WordTupleSpeaker,
    DiarizerInput, DiarizationResult,
    SpeakerNamingInput,
    SummaryInput, SummaryResult,
    AudioEvent, EventInput,
)
from MAT.utils.config import Config, ConfigError, Options
from MAT.utils.step_cache import DEFAULT_FOLDER as DEFAULT_CACHE
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
    # {"label", "text", "start", "end", "speakers"} per named entity in the transcript
    entities: List[Dict[str, Any]] = field(default_factory=list)
    # {"start", "end", "speakers", "text", "first_word", "last_word"} per sentence
    sentences: List[Dict[str, Any]] = field(default_factory=list)
    # sound events like music or laughter
    events: List[AudioEvent] = field(default_factory=list)

    def as_dict(self):
        return dataclass_as_dict(self)


def _rename_speakers(renames: Dict[str, str], diarization: DiarizationResult,
                     word_speaker: List[WordTupleSpeaker], squished: List[WordTupleSpeaker]):
    """Gives speakers new names everywhere: in the diarization, in the words and in the transcript lines. Two
    speakers that get the same name (the library knows both clusters as one voice) become one, with the segments of
    both. A dict keyed by the new name kept only one of them, and the words of the other had no segments (#4)."""
    def rename(words: List[WordTupleSpeaker]) -> List[WordTupleSpeaker]:
        return [WordTupleSpeaker(word=w.word, speaker={renames.get(s, s) for s in w.speaker}) for w in words]

    renamed_squished = rename(squished)
    merged: Dict[str, List[Tuple[float, float]]] = {}
    for speaker in sorted(diarization.speaker):
        merged.setdefault(renames.get(speaker, speaker), []).extend(diarization.get_diarization(speaker=speaker))
    renamed_diarization = DiarizationResult({name: sorted(segments) for name, segments in merged.items()})
    return (renamed_diarization, rename(word_speaker), renamed_squished,
            "\n".join(word_speaker_to_transcript(word_speaker=renamed_squished)))


def load_vocabulary(value: Union[None, str, Sequence[str]]) -> List[str]:
    """podcast.vocabulary as a list: the list itself, or the lines of a text file without comments and blanks."""
    if not value:
        return []
    if isinstance(value, str):
        path = os.path.expanduser(os.path.expandvars(value))
        if not os.path.isfile(path):
            raise ConfigError(f"podcast.vocabulary: {value} is no file. Give a file or a list of words.")
        with open(path, encoding="utf-8") as f:
            lines = [line.split("#", 1)[0].strip() for line in f]
        return [line for line in lines if line]
    return [str(word).strip() for word in value if str(word).strip()]


Spans = List[Tuple[int, int, int]]  # (start, end) in the text and the index of the word


def words_as_text(words: Sequence[WordTupleSpeaker], indexes: Sequence[int]) -> Tuple[str, Spans]:
    """The words joined with single spaces, and where each of them is in that text."""
    text, spans = "", []
    for index in indexes:
        token = (words[index].word.word or "").strip()
        if not token:
            continue
        if text:
            text += " "
        spans.append((len(text), len(text) + len(token), index))
        text += token
    return text, spans


def speaker_turns(words: Sequence[WordTupleSpeaker]) -> List[List[int]]:
    """Indexes of the words, grouped into runs of the same speaker."""
    turns: List[List[int]] = []
    for index, word in enumerate(words):
        if turns and word.speaker == words[turns[-1][-1]].speaker:
            turns[-1].append(index)
        else:
            turns.append([index])
    return turns


def split_sentences(words: Sequence[WordTupleSpeaker], splitter, config: Config,
                    language: Optional[str] = None) -> List[Dict[str, Any]]:
    """Sentences of a transcript, never across a change of speaker: {"start", "end", "speakers", "text",
    "first_word", "last_word"}, the last two are indexes into the words."""
    from MAT.tools.sentences import SentenceInput

    turns = [words_as_text(words, turn) for turn in speaker_turns(words)]
    turns = [(text, spans) for text, spans in turns if text]
    if not turns:
        return []
    result = splitter.process(origin_data=SentenceInput([text for text, _ in turns], language=language),
                              config=config)
    sentences = []
    for (text, spans), pieces in zip(turns, result.sentences):
        offset = 0
        for piece in pieces:
            begin, finish = offset, offset + len(piece)
            offset = finish
            covered = [index for a, b, index in spans if a < finish and b > begin]
            if not covered or not piece.strip():
                continue
            # a word split by the model counts for the sentence it starts in
            covered = [index for index in covered if not sentences or index > sentences[-1]["last_word"]]
            if not covered:
                continue
            starts = [words[i].word.start for i in covered if words[i].word.start is not None]
            ends = [words[i].word.end for i in covered if words[i].word.end is not None]
            sentences.append({"start": min(starts) if starts else None, "end": max(ends) if ends else None,
                              "speakers": sorted(words[covered[0]].speaker), "text": piece.strip(),
                              "first_word": covered[0], "last_word": covered[-1]})
    return sentences


def transcript_entities(words: List[WordTupleSpeaker], ner, config: Config, max_words: int = 150,
                        sentences: Optional[Sequence[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Named entities of a transcript with times and speakers. Runs NER per sentence when there are sentences,
    else per speaker turn, in pieces of at most max_words (the models read about 400 tokens), and maps the character
    spans back to words."""
    from MAT.tools.ner import NERInput

    if sentences:
        groups = [list(range(s["first_word"], s["last_word"] + 1)) for s in sentences]
    else:
        groups = speaker_turns(words)
    pieces = []
    for group in groups:
        for first in range(0, len(group), max_words):
            text, spans = words_as_text(words, group[first:first + max_words])
            if text:
                pieces.append((text, spans))
    if not pieces:
        return []

    from MAT.utils.characters import clean, looks_like_a_name

    result = ner.process(origin_data=NERInput(*[text for text, _ in pieces]), config=config)
    entities = []
    for (text, spans), found in zip(pieces, result.ner):
        for label, hits in found.items():
            for entity_text, start, end in hits:
                # NER calls "er" or "König" a PERSON now and then, that's never a person's name
                if label.casefold() == "person" and not looks_like_a_name(clean(entity_text)):
                    continue
                covered = [words[index] for a, b, index in spans if a < end and b > start]
                if not covered:
                    continue
                starts = [w.word.start for w in covered if w.word.start is not None]
                ends = [w.word.end for w in covered if w.word.end is not None]
                entities.append({"label": label, "text": entity_text, "start": min(starts) if starts else None,
                                 "end": max(ends) if ends else None,
                                 "speakers": sorted({s for w in covered for s in w.speaker})})
    return sorted(entities, key=lambda e: (e["start"] is None, e["start"] or 0.0, e["label"]))


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
    vocabulary: Optional[Union[List[str], str]] = Field(
        None, description="Names and words of the show that get misheard, as a list or a text file with one per "
                          "line (# starts a comment). Whisper expects them while transcribing and the summary "
                          "spells them that way. Keep it short, whisper reads about 600 characters.")
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
    sentences: str = Field("sat", description='Splits the transcript into sentences, which named entities use '
                                               'too. "none" skips it.')
    entities: str = Field("none", description='Named entities in the transcript (people, places, ...) with speaker '
                                              'and time. "gliner" turns it on, the labels are gliner.labels.')
    events: str = Field("none", description='Sound events: "audioset" for music, laughter and applause, "clap" for '
                                            'labels you describe in words (jingle, intro music). "none" skips it.')
    summarizer: str = Field("llm", description='Summary backend. "none" skips the summary.')
    cache: Optional[str] = Field(DEFAULT_CACHE, description="Folder for the results of transcription, "
                                                                   "diarization and sound events, so running an "
                                                                   "episode again starts after them. Empty or "
                                                                   "`MAT run --no-cache` turns it off.")


class PodcastPipeline(Pipeline):
    section = "podcast"
    description = "Transcript, speakers and summary for audio files (anything ffmpeg can decode)."
    Options = PodcastOptions
    slots = {"transcriber": Slot(), "diarizer": Slot(), "identifier": Slot(optional=True),
             "namer": Slot(optional=True), "sentences": Slot(optional=True), "entities": Slot(optional=True, kind="ner"),
             "events": Slot(optional=True), "summarizer": Slot(optional=True)}
    required_steps = {"prepare_audio", "transcribe"}

    def __init__(self):
        super().__init__()
        # the 16 kHz mono audio of the file being processed, decoded once by the first step
        self._segment = None

    @classmethod
    def preflight(cls, config: Config) -> None:
        load_vocabulary(config.options(cls).vocabulary)  # a missing file stops the run before the first episode
        super().preflight(config)

    @classmethod
    def accept(cls, f: str) -> bool:
        return cls.why_not(f) is None

    @classmethod
    def why_not(cls, f: str) -> Optional[str]:
        import shutil
        import subprocess

        # ffprobe only reads the header, decoding a 3 hour episode just to say yes took seconds and gigabytes
        if shutil.which("ffprobe"):
            try:
                probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                                        f], capture_output=True, text=True, timeout=60)
            except (OSError, subprocess.SubprocessError) as e:
                return f"ffprobe failed ({e})"
            if "audio" in probe.stdout.split():
                return None
            last = [line for line in probe.stderr.strip().splitlines() if line.strip()][-1:]
            return f"ffmpeg can't read it as audio: {last[0].strip()}" if last else "no audio stream in it"
        from pydub import AudioSegment
        # noinspection PyBroadException
        try:
            AudioSegment.from_file(f)
            return None
        except FileNotFoundError as e:
            # pydub raises this when ffmpeg itself is missing, too
            return f"ffmpeg or the file is missing ({e})"
        except Exception as e:
            # pydub puts ffmpeg's whole output into the message, the last line says what went wrong
            last = [line for line in str(e).strip().splitlines() if line.strip()][-1:] or [e.__class__.__name__]
            return f"ffmpeg can't read it as audio: {last[0].strip()}"

    def _get_steps(self) -> Iterable[Callable[[PipelineStepInput], PipelineStepResult]]:
        import pydub
        used = {}

        def cached(step_input: PipelineStepInput, step: str, backend, compute, **extra):
            """compute() unless the cache has this step for this file, backend and options already."""
            from MAT.utils.step_cache import StepCache

            cache = StepCache(step_input.config.options(self).cache)
            key = StepCache.key(backend, step_input.config, **extra) if cache.enabled else {}
            value = cache.get(step_input.file, step, key)
            if value is None:
                value = compute()
                cache.put(step_input.file, step, key, value)
            return value

        def audio_path(step_input: PipelineStepInput) -> str:
            """The 16 kHz mono wav the audio step wrote, the input file if it didn't run."""
            return (step_input.data("Audio") or {}).get("path") or step_input.file

        def audio_segment(step_input: PipelineStepInput) -> "pydub.AudioSegment":
            if self._segment is None:
                self._segment = pydub.AudioSegment.from_file(audio_path(step_input))
            return self._segment

        def prepare_audio(step_input: PipelineStepInput) -> PipelineStepResult:
            # Decoded once here. Every model wants 16 kHz mono anyway and reading a wav is quick, where every step
            # used to decode the mp3 again (4 to 5 times per episode).
            sound = pydub.AudioSegment.from_file(step_input.file)
            info = {"duration": sound.duration_seconds, "sample_rate": sound.frame_rate, "max_dbfs": sound.max_dBFS,
                    "rms": sound.rms}
            self._segment = sound.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            del sound
            os.makedirs(step_input.config.work_directory, exist_ok=True)
            path = os.path.join(step_input.config.work_directory, f"audio.{uuid.uuid4().hex}.wav")
            self._segment.export(path, format="wav")
            return PipelineStepResult(name="Audio", data=dict(info, path=path))

        def transcribe(step_input: PipelineStepInput) -> PipelineStepResult:
            transcriber = self.backend("transcriber", step_input.config)
            used["transcriber"] = transcriber
            vocabulary = load_vocabulary(step_input.config.options(self).vocabulary)
            d = cached(step_input, "transcription", transcriber, vocabulary=vocabulary,
                       compute=lambda: transcriber.process(
                           origin_data=TranscriptionInput(audio_path(step_input), vocabulary=vocabulary),
                           config=step_input.config))
            return PipelineStepResult(name="Transcription", data=d)

        def diarize(step_input: PipelineStepInput) -> PipelineStepResult:
            if isinstance(used.get("transcriber"), TranscribeDiarizeTool):
                self._LOGGER.info("The transcriber also diarizes, not running the diarizer")
                transcription = step_input.data("Transcription")
                return PipelineStepResult(name="Diarization", data=getattr(transcription, "diarization", None))
            diarizer = self.backend("diarizer", step_input.config)
            d = cached(step_input, "diarization", diarizer,
                       compute=lambda: diarizer.process(origin_data=DiarizerInput(in_file=audio_path(step_input)),
                                                        config=step_input.config))
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
            a = audio_segment(step_input)
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
            audio = audio_segment(step_input)
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
                metadata = {"filename": basename(step_input.file)}
                vocabulary = load_vocabulary(step_input.config.options(self).vocabulary)
                if vocabulary:
                    metadata["names and words of the show, spelled right"] = ", ".join(vocabulary)
                language = getattr(step_input.data("Transcription"), "language", None)
                summary = summarizer.process(origin_data=SummaryInput(full_transcript, additional_metadata=metadata,
                                                                      language=language),
                                             config=step_input.config)
            except Exception as e:
                self.__class__._LOGGER.exception(f"Summary failed for {basename(step_input.file)}, "
                                                 f"writing the results without a summary", exc_info=e)
                summary = None
            return PipelineStepResult(name="Summarize transcript", data=summary)

        def find_sentences(step_input: PipelineStepInput) -> PipelineStepResult:
            state = _speaker_state(step_input.data)
            if state is None or not state["word_speaker"]:
                return PipelineStepResult(name="Sentences", data=None)
            splitter = self.backend("sentences", step_input.config)
            if splitter is None:
                return PipelineStepResult(name="Sentences", data=None)
            language = getattr(step_input.data("Transcription"), "language", None)
            return PipelineStepResult(name="Sentences", data=split_sentences(
                state["word_speaker"], splitter, step_input.config, language=language))

        def find_entities(step_input: PipelineStepInput) -> PipelineStepResult:
            state = _speaker_state(step_input.data)
            if state is None or not state["word_speaker"]:
                return PipelineStepResult(name="Entities", data=None)
            ner = self.backend("entities", step_input.config)
            if ner is None:
                return PipelineStepResult(name="Entities", data=None)
            entities = transcript_entities(state["word_speaker"], ner, step_input.config,
                                           sentences=step_input.data("Sentences"))
            self.__class__._LOGGER.info(f"Found {len(entities)} entities, "
                                        f"{len({(e['label'], e['text'].casefold()) for e in entities})} different")
            return PipelineStepResult(name="Entities", data=entities)

        def find_events(step_input: PipelineStepInput) -> PipelineStepResult:
            tagger = self.backend("events", step_input.config)
            if tagger is None:
                return PipelineStepResult(name="Audio events", data=None)
            events = cached(step_input, "events", tagger, compute=lambda: getattr(
                tagger.process(origin_data=EventInput(step_input.file), config=step_input.config), "events", None))
            return PipelineStepResult(name="Audio events", data=events)

        def media_infos(step_input: PipelineStepInput) -> PipelineStepResult:
            transcription: TranscriptionResult = step_input.data("Transcription")
            lang = getattr(transcription, "language", None)
            duration_av = getattr(transcription, "duration_after_vad", None)
            info = step_input.data("Audio")
            if info is None:
                a = pydub.AudioSegment.from_file(step_input.file)
                info = {"duration": a.duration_seconds, "sample_rate": a.frame_rate, "max_dbfs": a.max_dBFS,
                        "rms": a.rms}
            return PipelineStepResult(
                name="Media Info",
                data=MediaInfo(
                    file_name=os.path.basename(step_input.file),
                    duration=info["duration"], sample_rate=info["sample_rate"], max_dbfs=info["max_dbfs"],
                    rms=info["rms"], language="" if lang is None else lang,
                    duration_after_vad=info["duration"] if duration_av is None else duration_av,
                )
            )

        return [prepare_audio, transcribe, diarize, speaker_matching, creating_speaker_transcript, speaker_library,
                name_speakers, find_sentences, find_entities, find_events, summarize_transcript, media_infos]

    def _finalize_result(self, step_results: Dict[str, PipelineStepResult]) -> PodcastOutput:

        def _try_get(k: str):
            result = step_results.get(k)
            return None if result is None else result.data

        # the decoded audio is only needed while the steps run
        self._segment = None
        prepared = (_try_get("Audio") or {}).get("path")
        if prepared and os.path.isfile(prepared):
            os.remove(prepared)

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
            entities=_try_get("Entities") or [],
            events=_try_get("Audio events") or [],
            sentences=_try_get("Sentences") or [],
        )


__all__ = ["PodcastOutput", "PodcastPipeline", "MediaInfo"]
