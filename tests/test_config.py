import logging
import sys
import textwrap
import tomllib

import pytest

from MAT import registry
from MAT.pipelines.Podcast import PodcastPipeline
from MAT.tools.speakeridentification.pyannote import SpeakerIdetificationPyannote
from MAT.tools.summary.llm import SummaryLLM
from MAT.tools.transcriptors.whisper import TransciptorWhisper
from MAT.utils.config import Config, ConfigError, parse_override, render_sections


def test_defaults_are_typed():
    options = Config().options(TransciptorWhisper)
    assert options.model == "large-v3-turbo"
    assert options.beam_size == 5
    assert Config().options(SpeakerIdetificationPyannote).no_hf_token is False


def test_set_wins_over_file(tmp_path):
    file = tmp_path / "mat.toml"
    file.write_text('[whisper]\nbeam-size = 3\nmodel = "medium"\n')
    options = Config.load(file=file, overrides=["whisper.beam-size=8"]).options(TransciptorWhisper)
    assert options.beam_size == 8
    assert options.model == "medium"


def test_underscore_and_dash_keys_are_the_same(tmp_path):
    file = tmp_path / "mat.toml"
    file.write_text("[whisper]\nbeam_size = 3\n")
    assert Config.load(file=file, overrides=["whisper.beam-size=4"]).options(TransciptorWhisper).beam_size == 4


@pytest.mark.parametrize("item, expected", [
    ("whisper.model=large-v3", ("whisper", "model", "large-v3")),
    ("whisper.beam-size=8", ("whisper", "beam-size", 8)),
    ("whisper.beam_size=8", ("whisper", "beam-size", 8)),
    ('llm.extra-body={"thinking": {"type": "disabled"}}', ("llm", "extra-body", {"thinking": {"type": "disabled"}})),
    ("podcast.summarizer=none", ("podcast", "summarizer", "none")),
    ("llm.temperature=null", ("llm", "temperature", None)),
])
def test_parse_override(item, expected):
    assert parse_override(item) == expected


@pytest.mark.parametrize("item", ["whisper.model", "model=x", ".model=x", "whisper.=x"])
def test_parse_override_rejects_bad_input(item):
    with pytest.raises(ConfigError):
        parse_override(item)


def test_unknown_section_fails():
    with pytest.raises(ConfigError, match=r"Unknown config section \[whisperr\]"):
        Config({"whisperr": {}}).validate()


def test_unknown_option_fails_and_lists_options():
    with pytest.raises(ConfigError, match='unknown option "beam-sise".*beam-size'):
        Config({"whisper": {"beam-sise": 3}}).validate()


def test_wrong_type_fails():
    with pytest.raises(ConfigError, match="beam-size"):
        Config({"whisper": {"beam-size": "many"}}).validate()


def test_unknown_backend_in_slot_fails():
    with pytest.raises(ConfigError, match='unknown transcriber "nope"'):
        Config({"podcast": {"transcriber": "nope"}}).validate()


def test_missing_gold_label_folder_fails(tmp_path):
    with pytest.raises(ConfigError, match="is not a folder"):
        Config({"pyannote": {"gold-labels": str(tmp_path / "missing")}}).validate()


def test_section_of_missing_backend_only_warns(tmp_path, monkeypatch, caplog):
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "mat_fake_skipped.py").write_text(textwrap.dedent("""
        from MAT.registry import require
        require("mat_definitely_missing", extra="demo")
    """))
    try:
        registry.load_optional("mat_fake_skipped", slot="diarizer", name="test-skipped", extra="demo")
        with caplog.at_level(logging.WARNING):
            Config({"test-skipped": {"anything": 1}}).validate()
        assert "test-skipped" in caplog.text
    finally:
        registry.unregister("diarizer", "test-skipped")
        sys.modules.pop("mat_fake_skipped", None)


def test_rendered_config_loads_back():
    config = Config({"whisper": {"beam-size": 7}, "podcast": {"summarizer": "none"}})
    data = tomllib.loads(render_sections([PodcastPipeline, TransciptorWhisper, SummaryLLM], config))
    assert data["whisper"]["beam-size"] == 7
    assert data["podcast"]["summarizer"] == "none"
    # unset options are commented out, not written as empty values
    assert "temperature" not in data["llm"]
    loaded = Config(data)
    assert loaded.options(TransciptorWhisper) == config.options(TransciptorWhisper)
    # the long multi-line prompts survive the trip
    assert loaded.options(SummaryLLM) == config.options(SummaryLLM)
