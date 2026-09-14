import json
import shutil
from pathlib import Path

import pytest

from MAT.pipelines.Book import BookOutput, Chapter
from MAT.pipelines.Podcast import MediaInfo, PodcastOutput
from MAT.reader import MATResult
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
    return PodcastOutput(
        media_info=MediaInfo(file_name="episode.wav", duration=2.0, duration_after_vad=1.5, sample_rate=16000,
                             max_dbfs=-1.0, rms=100, language="en"),
        transcription=TranscriptionResult(word_timings=words, language="en", duration=2.0, duration_after_vad=1.5),
        diarization=DiarizationResult({"sprecher_0": [(0.0, 0.9)], "sprecher_1": [(0.9, 2.0)]}),
        diarization_matched=DiarizationResult({"alice": [(0.0, 0.9)], "bob": [(0.9, 2.0)]}),
        word_speaker=word_speaker,
        squished_speaker=word_speaker,
        full_transcript="alice [0.0 - 0.5]: hello\nbob [1.0 - 1.5]: hi",
        summary=SummaryResult("# A summary"),
        models={"transcriber": {"backend": "whisper", "model": "large-v3-turbo"}},
    )


def _book_output() -> BookOutput:
    return BookOutput(title="Book", language="en", models={"ner": {"backend": "gliner"}}, chapter_data=[
        Chapter(heading="Part", heading_beautified="Part I", content=["Alice went home. Bob stayed."],
                sentences=["Alice went home.", "Bob stayed."],
                sentence_words=[{"Alice": 1, "home": 1}, {"Bob": 1, "stay": 1}],
                ner=[{"PERSON": [("Alice", 0, 5)], "LOCATION": []}, {"PERSON": [("Bob", 0, 3)]}]),
        # splitting and NER never ran for this one
        Chapter(heading="Epilogue", content=["The end."]),
    ])


def _write(tmp_path, input_file, results, as_zip):
    folder = Writer().store(file=str(input_file), output=str(tmp_path / "out"), pipeline_results=results)
    return Path(shutil.make_archive(folder, "zip", folder)) if as_zip else Path(folder)


def test_layout_and_meta(tmp_path, input_file):
    folder = _write(tmp_path, input_file, [_podcast_output(), _book_output()], as_zip=False)
    assert sorted(p.name for p in folder.iterdir()) == ["book", "meta.json", "podcast"]
    assert sorted(p.name for p in (folder / "podcast").iterdir()) == \
        ["diarization.rttm", "result.json", "summary.md", "transcript.txt"]
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["format"] == 2
    assert meta["input"]["name"] == "episode.wav"
    assert meta["pipelines"] == ["podcast", "book"]


@pytest.mark.parametrize("as_zip", [False, True])
def test_podcast_roundtrip(tmp_path, input_file, as_zip):
    result = MATResult.read(_write(tmp_path, input_file, [_podcast_output()], as_zip))

    assert result.format == 2
    assert result.book is None
    podcast = result.podcast
    assert podcast.language == "en"
    assert podcast.duration == 2.0
    assert podcast.speaker_names == {"alice", "bob"}
    assert podcast.speakers["alice"] == [(0.0, 0.9)]
    assert podcast.diarization == {"sprecher_0": [(0.0, 0.9)], "sprecher_1": [(0.9, 2.0)]}
    assert [(w.text, w.speakers) for w in podcast.words] == [("hello", ("alice",)), ("hi", ("bob",))]
    assert podcast.transcript.splitlines() == ["alice [0.0 - 0.5]: hello", "bob [1.0 - 1.5]: hi"]
    assert podcast.summary == "# A summary"
    assert podcast.models["transcriber"]["model"] == "large-v3-turbo"


@pytest.mark.parametrize("as_zip", [False, True])
def test_book_roundtrip(tmp_path, input_file, as_zip):
    book = MATResult.read(_write(tmp_path, input_file, [_book_output()], as_zip)).book

    assert book.title == "Book"
    assert book.models == {"ner": {"backend": "gliner"}}
    assert [(c.heading, c.heading_raw) for c in book.chapters] == [("Part I", "Part"), ("Epilogue", "Epilogue")]
    first = book.chapters[0].sentences[0]
    assert first.text == "Alice went home."
    assert first.lemmas == {"Alice": 1, "home": 1}
    assert first.entities_by_label() == {"PERSON": ["Alice"]}
    assert book.chapters[1].sentences == []
    assert book.chapters[1].paragraphs == ["The end."]


def test_podcast_without_summary_and_transcript(tmp_path, input_file):
    output = _podcast_output()
    output.summary = None
    folder = _write(tmp_path, input_file, [output], as_zip=False)
    assert not (folder / "podcast" / "summary.md").exists()
    assert MATResult.read(folder).podcast.summary is None


def test_old_results_are_rejected(tmp_path):
    folder = tmp_path / "old"
    folder.mkdir()
    (folder / "meta.json").write_text(json.dumps({"version": "1", "MAT_version": "0.2.0", "pipelines": []}))
    with pytest.raises(ValueError, match="format"):
        MATResult.read(folder)


def test_writer_does_not_collide(tmp_path, input_file):
    writer = Writer()
    a = writer.store(file=str(input_file), output=str(tmp_path), pipeline_results=[])
    b = writer.store(file=str(input_file), output=str(tmp_path), pipeline_results=[])
    assert a != b
