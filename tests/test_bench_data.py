import tomllib
from pathlib import Path

import pytest

from MAT.bench.commands import create_reference
from MAT.bench.data import TranscriptError, Turn, format_turn, load_transcript, parse_transcript
from MAT.bench.datasets.reference import AudioFiles, AudioOptions, ReferenceOptions, References
from MAT.utils.config import ConfigError

EXAMPLE = Path(__file__).resolve().parent.parent / "packages" / "mat-format" / "examples" / "podcast-sample"


def test_parse_transcript():
    turns = parse_transcript(
        "# comment\n"
        "sprecher_0 [6.721 - 7.062]: hello\n"
        "\n"
        "sprecher_0 & sprecher_1 [10.795 - 10.895]: i\n"
        "<Unknown> [30.76 - 30.863]: Nein.\n"
        "alice [None - None]: no times\n")
    assert turns == [
        Turn(6.721, 7.062, ("sprecher_0",), "hello"),
        Turn(10.795, 10.895, ("sprecher_0", "sprecher_1"), "i"),
        Turn(30.76, 30.863, (), "Nein."),
        Turn(None, None, ("alice",), "no times"),
    ]


def test_parse_transcript_round_trip():
    turn = Turn(1.5, 2.0, ("alice", "bob"), "hi there")
    assert parse_transcript(format_turn(turn)) == [turn]


def test_parse_transcript_names_the_bad_line():
    with pytest.raises(TranscriptError, match="line 2"):
        parse_transcript("alice [0 - 1]: ok\nthis line is broken\n")
    with pytest.raises(TranscriptError, match="line 1"):
        parse_transcript("alice [zero - 1]: ok\n")


def test_load_transcript_from_a_result_folder():
    turns = load_transcript(EXAMPLE)
    lines = [line for line in (EXAMPLE / "podcast" / "transcript.txt").read_text().splitlines() if line.strip()]
    assert len(turns) == len(lines)
    assert turns == parse_transcript("\n".join(lines))


def _reference(folder: Path, toml: str, transcript: str) -> Path:
    folder.mkdir(parents=True)
    (folder / "episode.wav").write_bytes(b"RIFF")
    (folder / "reference.toml").write_text(toml)
    (folder / "transcript.txt").write_text(transcript)
    return folder


def test_reference_dataset(tmp_path):
    _reference(tmp_path / "refs" / "ep1", 'audio = "episode.wav"\nlanguage = "de"\nstart = 1.0\nend = 9.5\n',
               "alice [1.0 - 3.0]: hallo\nbob [3.0 - 9.5]: tschüss\n")
    _reference(tmp_path / "refs" / "ep2", 'audio = "episode.wav"\n', "<Unknown> [0 - 1]: nur text\n")
    dataset = References(ReferenceOptions(type="reference", name="own", path=str(tmp_path / "refs")), cache=tmp_path)
    first, second = list(dataset.items())
    assert (first.dataset, first.id, first.language, first.start, first.end) == ("own", "ep1", "de", 1.0, 9.5)
    assert first.audio == tmp_path / "refs" / "ep1" / "episode.wav"
    assert first.has_words and first.has_speakers
    assert first.reference_segments() == {"alice": [(1.0, 3.0)], "bob": [(3.0, 9.5)]}
    assert second.has_words and not second.has_speakers


def test_reference_dataset_single_folder_and_errors(tmp_path):
    folder = _reference(tmp_path / "ep", 'audio = "missing.wav"\n', "alice [0 - 1]: hi\n")
    dataset = References(ReferenceOptions(type="reference", path=str(folder)), cache=tmp_path)
    with pytest.raises(ConfigError, match="missing.wav"):
        list(dataset.items())
    (folder / "reference.toml").write_text('audio = "episode.wav"\nstart = 5\nend = 2\n')
    with pytest.raises(ConfigError, match="end has to be after start"):
        list(dataset.items())
    (folder / "reference.toml").write_text('audio = "episode.wav"\nspeed = 2\n')
    with pytest.raises(ConfigError, match="speed"):
        list(dataset.items())


def test_audio_dataset(tmp_path):
    for name in ("b.mp3", "a.mp3", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    dataset = AudioFiles(AudioOptions(type="audio", files=["*.mp3"], limit=1), cache=tmp_path, base_dir=tmp_path)
    (item,) = dataset.items()
    assert item.audio.name == "a.mp3" and item.id.startswith("a-")
    assert not item.has_words and not item.has_speakers


def test_create_reference_from_a_result(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")
    with pytest.raises(ConfigError, match="--audio"):
        create_reference(EXAMPLE, tmp_path / "missing-audio")

    folder = create_reference(EXAMPLE, tmp_path / "ref", audio=audio, start=10, end=20)
    settings = tomllib.loads((folder / "reference.toml").read_text())
    turns = parse_transcript((folder / "transcript.txt").read_text())
    assert settings["audio"] == str(audio.resolve())
    assert settings["language"] == "en"
    assert settings["start"] == min(t.start for t in turns) and settings["end"] == max(t.end for t in turns)
    assert all(10 <= (t.start + t.end) / 2 < 20 for t in turns)

    (item,) = References(ReferenceOptions(type="reference", path=str(folder)), cache=tmp_path).items()
    assert item.start == settings["start"] and len(item.turns) == len(turns)

    with pytest.raises(ConfigError, match="isn't empty"):
        create_reference(EXAMPLE, folder, audio=audio)
