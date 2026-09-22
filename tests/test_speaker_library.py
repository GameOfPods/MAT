import json

import pytest

from MAT.utils.speaker_library import MAX_EMBEDDINGS, SpeakerLibrary

ALEX = [1.0, 0.0, 0.0]
TOBI = [0.0, 1.0, 0.0]
ALEX_AGAIN = [0.96, 0.05, 0.0]


def test_empty_folder_gives_an_empty_library(tmp_path):
    library = SpeakerLibrary.open(tmp_path / "does-not-exist")
    assert library.speakers == []
    assert library.match(ALEX, threshold=0.5) is None


def test_remember_and_read_back(tmp_path):
    library = SpeakerLibrary.open(tmp_path)
    alex = library.remember("Alex", ALEX, source="gold", episode="ep1")
    assert alex.library_id.startswith("alex-") and alex.updated
    library.save()

    again = SpeakerLibrary.open(tmp_path)
    assert [speaker.name for speaker in again.speakers] == ["Alex"]
    assert again.speakers[0].library_id == alex.library_id
    assert again.speakers[0].episodes == ["ep1"]

    found = again.match(ALEX_AGAIN, threshold=0.8)
    assert found is not None and found[0].name == "Alex" and found[1] > 0.9


def test_the_same_person_keeps_one_entry(tmp_path):
    library = SpeakerLibrary.open(tmp_path)
    first = library.remember("Alex", ALEX, episode="ep1")
    second = library.remember("alex", ALEX_AGAIN, episode="ep2")
    assert first.library_id == second.library_id
    assert len(library.speakers) == 1
    assert second.episodes == ["ep1", "ep2"]
    assert len(second.embeddings) == 2


def test_only_the_newest_embeddings_are_kept(tmp_path):
    library = SpeakerLibrary.open(tmp_path)
    for i in range(MAX_EMBEDDINGS + 5):
        library.remember("Alex", [float(i), 1.0, 0.0])
    kept = library.speakers[0].embeddings
    assert len(kept) == MAX_EMBEDDINGS
    assert kept[-1][0] == float(MAX_EMBEDDINGS + 4)


def test_a_voice_nobody_knows_is_no_match(tmp_path):
    library = SpeakerLibrary.open(tmp_path)
    library.remember("Alex", ALEX)
    assert library.match(TOBI, threshold=0.5) is None


def test_two_voices_too_close_together_are_no_match(tmp_path, caplog):
    import logging

    caplog.set_level(logging.INFO)
    library = SpeakerLibrary.open(tmp_path)
    library.remember("Alex", [1.0, 0.0, 0.0])
    library.remember("Chris", [0.99, 0.01, 0.0])
    assert library.match([1.0, 0.005, 0.0], threshold=0.5) is None
    assert "too close" in caplog.text


def test_saving_twice_leaves_no_leftovers(tmp_path):
    library = SpeakerLibrary.open(tmp_path)
    library.remember("Alex", ALEX)
    library.save()
    library.save()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["speakers.json"]
    assert json.loads((tmp_path / "speakers.json").read_text())["version"] == 1


def test_broken_file_says_so(tmp_path):
    (tmp_path / "speakers.json").write_text("{not json")
    with pytest.raises(ValueError, match="isn't readable JSON"):
        SpeakerLibrary.open(tmp_path)


def test_entries_without_a_name_are_ignored(tmp_path):
    (tmp_path / "speakers.json").write_text(json.dumps({"speakers": [
        {"library_id": "x-1", "name": "Alex", "embeddings": [ALEX]},
        {"library_id": "y-2", "embeddings": [TOBI]},
        {"name": "no id"},
        {"library_id": "z-3", "name": "Extra", "embeddings": [], "unknown_field": 1},
    ]}))
    library = SpeakerLibrary.open(tmp_path)
    assert [speaker.name for speaker in library.speakers] == ["Alex", "Extra"]
