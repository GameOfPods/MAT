import tomllib

import pytest

from MAT.cli import main


def test_run_help_stays_short(capsys):
    with pytest.raises(SystemExit):
        main(["run", "-h"])
    out = capsys.readouterr().out
    assert "--transcriber" in out
    # backend options are not flags, they are listed by `MAT backends show`
    assert "beam-size" not in out
    assert len(out.splitlines()) < 80


def test_backends_list(capsys):
    assert main(["backends"]) == 0
    out = capsys.readouterr().out
    assert "transcriber" in out
    assert "whisper" in out


def test_backends_show(capsys):
    assert main(["backends", "show", "whisper"]) == 0
    out = capsys.readouterr().out
    assert "beam-size" in out
    assert "--set whisper." in out


def test_backends_show_unknown(capsys):
    assert main(["backends", "show", "nope"]) == 2
    assert "Unknown backend" in capsys.readouterr().err


def test_config_init_is_valid_toml(capsys):
    assert main(["config", "init", "--summarizer", "none"]) == 0
    data = tomllib.loads(capsys.readouterr().out)
    assert data["podcast"]["summarizer"] == "none"
    assert "whisper" in data
    # the summarizer is off, so its section isn't in the file
    assert "llm" not in data


def test_config_init_stays_valid_toml_when_a_backend_is_missing(tmp_path, monkeypatch, capsys, caplog):
    import logging
    import sys
    import textwrap
    from MAT import registry

    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "mat_fake_absent.py").write_text(textwrap.dedent("""
        from MAT.registry import require
        require("mat_definitely_missing", extra="demo")
    """))
    try:
        registry.load_optional("mat_fake_absent", slot="diarizer", name="test-absent", extra="demo")
        with caplog.at_level(logging.WARNING):
            assert main(["config", "init", "--set", "podcast.diarizer=test-absent"]) == 0
        # the warning is logged, stdout only has the TOML file
        data = tomllib.loads(capsys.readouterr().out)
        assert data["podcast"]["diarizer"] == "test-absent"
        assert "test-absent isn't installed" in caplog.text
    finally:
        registry.unregister("diarizer", "test-absent")
        sys.modules.pop("mat_fake_absent", None)


def test_config_init_does_not_overwrite(tmp_path, capsys):
    target = tmp_path / "mat.toml"
    target.write_text("keep me")
    assert main(["config", "init", "-o", str(target)]) == 2
    assert target.read_text() == "keep me"


def test_config_show_merges_file_and_set(tmp_path, capsys):
    file = tmp_path / "mat.toml"
    file.write_text("[whisper]\nbeam-size = 3\n")
    assert main(["config", "show", "-c", str(file), "--set", "whisper.model=medium"]) == 0
    data = tomllib.loads(capsys.readouterr().out)
    assert data["whisper"]["beam-size"] == 3
    assert data["whisper"]["model"] == "medium"


def test_config_show_reports_typos(capsys):
    assert main(["config", "show", "--set", "whisper.beam-sise=3"]) == 2
    assert "beam-sise" in capsys.readouterr().err
