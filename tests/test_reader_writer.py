import shutil
from pathlib import Path

import pytest

from MAT.pipelines.Book import BookOutput, Chapter
from MAT.pipelines.Podcast import PodcastOutput, MediaInfo
from MAT.reader import MATResult, ResultTypes
from MAT.tools import DiarizationResult, SummaryResult, TranscriptionResult, WordTuple, WordTupleSpeaker
from MAT.writer import Writer


@pytest.fixture
def input_file(tmp_path):
    f = tmp_path / "episode.wav"
    f.write_bytes(b"not really audio, the writer only hashes it")
    return f


def _podcast_output() -> PodcastOutput:
    words = [WordTuple(0.0, 0.5, "hello"), WordTuple(1.0, 1.5, "hi")]
    word_speaker = [WordTupleSpeaker(words[0], {"alice"}), WordTupleSpeaker(words[1], {"bob"})]
    diarization = DiarizationResult({"alice": [(0.0, 0.9)], "bob": [(0.9, 2.0)]})
    return PodcastOutput(
        media_info=MediaInfo(file_name="episode.wav", duration=2.0, duration_after_vad=1.5, sample_rate=16000,
                             max_dbfs=-1.0, rms=100, language="en"),
        transcription=TranscriptionResult(word_timings=words, language="en", duration=2.0, duration_after_vad=1.5),
        diarization=diarization,
        diarization_matched=diarization,
        word_speaker=word_speaker,
        squished_speaker=word_speaker,
        full_transcript="alice [0.0 - 0.5]: hello\nbob [1.0 - 1.5]: hi",
        summary=SummaryResult("a summary"),
    )


def _book_output() -> BookOutput:
    return BookOutput(title="Book", language="en", chapter_data=[
        Chapter(heading="Chapter 1", content=["Alice went home. Bob stayed."],
                sentences=["Alice went home.", "Bob stayed."],
                sentence_words=[{"Alice": 1, "home": 1}, {"Bob": 1, "stay": 1}],
                ner=[{"PERSON": [("Alice", 0, 5)]}, {"PERSON": [("Bob", 0, 3)]}]),
        # a chapter where splitting and NER never ran, so those keys are not written at all
        Chapter(heading="Epilogue", content=["The end."]),
    ])


@pytest.mark.parametrize("as_zip", [False, True])
def test_podcast_roundtrip(tmp_path, input_file, as_zip):
    folder = Writer().store(file=str(input_file), output=str(tmp_path / "out"), pipeline_results=[_podcast_output()])
    path = Path(shutil.make_archive(folder, "zip", folder)) if as_zip else Path(folder)

    result = MATResult.read(path)
    assert result is not None
    podcasts = list(result.get_results(ResultTypes.PODCAST))
    assert len(podcasts) == 1
    p = podcasts[0]
    assert p.language == "en"
    assert p.speaker_names == {"alice", "bob"}
    assert "hello" in p.transcript
    assert p.summary == "a summary"


@pytest.mark.parametrize("as_zip", [False, True])
def test_book_roundtrip(tmp_path, input_file, as_zip):
    folder = Writer().store(file=str(input_file), output=str(tmp_path / "out"), pipeline_results=[_book_output()])
    path = Path(shutil.make_archive(folder, "zip", folder)) if as_zip else Path(folder)

    books = list(MATResult.read(path).get_results(ResultTypes.BOOK))
    assert len(books) == 1
    chapters = books[0].chapters
    assert [c.heading for c in chapters] == ["Chapter 1", "Epilogue"]
    assert chapters[0].sentences[0].ner == {"PERSON": ["Alice"]}
    assert chapters[1].sentences == []


def test_writer_does_not_collide(tmp_path, input_file):
    writer = Writer()
    a = writer.store(file=str(input_file), output=str(tmp_path), pipeline_results=[])
    b = writer.store(file=str(input_file), output=str(tmp_path), pipeline_results=[])
    assert a != b
