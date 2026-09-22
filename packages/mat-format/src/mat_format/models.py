"""
Data model of MAT results in format 2.

MAT's writer builds these models and dumps them to JSON, the reader parses the JSON back into them and the JSON schemas
in `schemas/` are generated from them. Change a model and the schema changes with it.

Rules for changes:
- adding a field (optional or not) keeps the format version, readers ignore fields they don't know
- renaming or removing a field, or changing its type or meaning, needs a new format version
"""
from typing import Dict, List, Literal, Optional, Set

from pydantic import BaseModel, ConfigDict, Field

FORMAT_VERSION = 2


class _Model(BaseModel):
    # unknown fields are ignored, so files from a newer MAT with extra fields still load
    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)


class ModelInfo(BaseModel):
    """Which backend and model produced a step, and the versions of the Python packages it used."""
    model_config = ConfigDict(extra="allow", frozen=True)

    backend: str = Field(description="MAT backend name, for example whisper or sortformer.")
    model: Optional[str] = Field(None, description="Model the backend used, if it has one.")
    packages: Dict[str, str] = Field(default_factory=dict, description="Python package name to version.")


# ---------------------------------------------------------------- meta.json


class Input(_Model):
    """The media file that was processed."""

    name: str = Field(description="File name without folders.")
    path: str = Field(description="Absolute path on the machine that ran MAT.")
    sha1: str = Field(description="SHA-1 of the file content, hex encoded.")


class Meta(_Model):
    """Content of meta.json in the root of every result."""

    format: Literal[2] = Field(FORMAT_VERSION, description="Result format version.")
    mat_version: str = Field(description="Version of MAT that wrote the result.")
    created: str = Field(description="Local time the result was written, ISO 8601 without time zone.")
    input: Input
    pipelines: List[str] = Field(description='Pipelines that ran, each has a folder of the same name: "podcast", '
                                             '"book".')


# ---------------------------------------------------------------- podcast/result.json


class TimeRange(_Model):
    """Seconds from the start of the audio file."""

    start: float
    end: float


class Speaker(_Model):
    """A speaker and when they talk."""

    id: str = Field(description="Speaker label. The gold label name (for example alice) if the speaker was matched "
                                "to a gold label clip, otherwise the diarizer label (for example sprecher_0). Only "
                                "unique inside one result, not across episodes.")
    name: Optional[str] = Field(None, description="Real name of the person, when MAT knows one: from a gold label "
                                                  "clip, from the transcript or from the speaker library. Null when "
                                                  "only the diarizer label is known.")
    library_id: Optional[str] = Field(None, description="Id of this voice in the speaker library. Stays the same in "
                                                        "every episode the voice shows up in, which makes statistics "
                                                        "per person possible. Null when no library was used.")
    segments: List[TimeRange] = Field(description="Time ranges this speaker talks, sorted by start.")


class Word(_Model):
    """A word or, in segments, several words of the same speakers merged into one line."""

    start: Optional[float] = Field(description="Seconds from the start of the audio file, null if unknown.")
    end: Optional[float] = Field(description="Seconds from the start of the audio file, null if unknown.")
    text: str
    speakers: List[str] = Field(description="Speaker ids talking during this word. Empty if nobody matched, more "
                                            "than one if speakers overlap.")


class Media(_Model):
    """Technical data of the audio file."""

    duration: float = Field(description="Length in seconds.")
    speech_duration: Optional[float] = Field(description="Seconds of speech after voice activity detection.")
    sample_rate: int = Field(description="Samples per second of the decoded audio.")
    max_dbfs: Optional[float] = Field(description="Loudest sample in dBFS, null for silent audio.")
    rms: Optional[float] = Field(description="Root mean square of the samples.")


class Event(_Model):
    """A sound event like music or laughter. Not filled yet, planned for a later MAT version."""

    label: str
    start: float
    end: float
    score: Optional[float] = Field(None, description="Confidence between 0 and 1, if the model gives one.")


class TranscriptEntity(_Model):
    """A named entity found in the transcript. Not filled yet, planned for a later MAT version."""

    label: str = Field(description="Entity type, for example PERSON.")
    text: str
    start: Optional[float] = Field(None, description="Seconds from the start of the audio file.")
    end: Optional[float] = Field(None, description="Seconds from the start of the audio file.")
    speakers: List[str] = Field(default_factory=list, description="Speaker ids that said it.")


class PodcastResult(_Model):
    """Content of podcast/result.json."""

    schema_: Literal["mat.podcast"] = Field("mat.podcast", alias="schema")
    format: Literal[2] = Field(FORMAT_VERSION, description="Result format version.")
    models: Dict[str, ModelInfo] = Field(description="Step name (transcriber, diarizer, identifier, summarizer) to "
                                                     "the backend that ran it. Steps that were skipped are missing.")
    language: Optional[str] = Field(description="Detected language as ISO 639-1 code, for example de.")
    media: Optional[Media]
    speakers: List[Speaker] = Field(description="Speakers after matching them to gold label clips. This is the "
                                                "list to use.")
    diarization: List[Speaker] = Field(description="Speakers as the diarizer found them, before matching.")
    words: List[Word] = Field(description="Every transcribed word, in order.")
    segments: List[Word] = Field(description="Consecutive words with the same speakers merged into lines.")
    summary: Optional[str] = Field(description="Summary as Markdown, null if the summary was skipped or failed.")
    events: List[Event]
    entities: List[TranscriptEntity]

    @property
    def speaker_ids(self) -> Set[str]:
        return {speaker.id for speaker in self.speakers}

    def speaker(self, speaker_id: str) -> Optional[Speaker]:
        return next((speaker for speaker in self.speakers if speaker.id == speaker_id), None)


# ---------------------------------------------------------------- book/result.json


class TextEntity(_Model):
    """A named entity inside a sentence."""

    label: str = Field(description="Entity type, for example PERSON.")
    text: str
    start: int = Field(description="Character offset in the sentence text where the entity starts.")
    end: int = Field(description="Character offset in the sentence text where the entity ends (exclusive).")


class Sentence(_Model):
    text: str
    lemmas: Dict[str, int] = Field(description="Lemma to count, without stop words and punctuation.")
    entities: List[TextEntity]

    def entities_by_label(self) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {}
        for entity in self.entities:
            result.setdefault(entity.label, []).append(entity.text)
        return result


class Chapter(_Model):
    heading: str = Field(description='Heading to show. Repeated headings get a roman numeral, for example "Part II".')
    heading_raw: str = Field(description="Heading as it is in the book.")
    paragraphs: List[str]
    sentences: List[Sentence] = Field(description="Empty if sentence splitting didn't run.")


class BookResult(_Model):
    """Content of book/result.json."""

    schema_: Literal["mat.book"] = Field("mat.book", alias="schema")
    format: Literal[2] = Field(FORMAT_VERSION, description="Result format version.")
    models: Dict[str, ModelInfo] = Field(description="Step name (splitter, ner) to the backend that ran it.")
    title: str
    language: Optional[str] = Field(description="Detected language as ISO 639-1 code.")
    chapters: List[Chapter]


__all__ = ["FORMAT_VERSION", "ModelInfo", "Input", "Meta", "TimeRange", "Speaker", "Word", "Media", "Event",
           "TranscriptEntity", "PodcastResult", "TextEntity", "Sentence", "Chapter", "BookResult"]
