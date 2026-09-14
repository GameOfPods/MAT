import builtins

import pytest

from MAT.cli import main


def test_yes_skips_the_question(tmp_path, monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda *a: pytest.fail("should not ask with --yes"))
    code = main(["run", "-i", str(tmp_path / "nothing*.mp3"), "-o", str(tmp_path / "out"), "-wd", str(tmp_path),
                 "--yes"])
    assert code == 0


def test_without_yes_it_asks(tmp_path, monkeypatch):
    questions = []

    def answer_no(prompt):
        questions.append(prompt)
        return "n"

    monkeypatch.setattr(builtins, "input", answer_no)
    with pytest.raises(SystemExit):
        main(["run", "-i", str(tmp_path / "nothing*.mp3"), "-o", str(tmp_path / "out"), "-wd", str(tmp_path)])
    assert len(questions) == 1


def test_bad_config_stops_before_processing(tmp_path, capsys):
    code = main(["run", "-i", str(tmp_path / "nothing*.mp3"), "-o", str(tmp_path / "out"), "--yes",
                 "--set", "podcast.transcriber=nope"])
    assert code == 2
    assert 'unknown transcriber "nope"' in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
