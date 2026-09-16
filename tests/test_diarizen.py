import os
import sys

import pytest
from pydub.generators import Sine

os.environ.setdefault("MAT_EXTERNAL_PYTHON_DIARIZEN", sys.executable)  # the module needs the environment to exist

from MAT.tools.diarizators import DiarizerInput  # noqa: E402
from MAT.tools.diarizators.diarizen import DiarizerDiariZen  # noqa: E402
from MAT.utils.config import Config  # noqa: E402


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "episode.mp3"
    Sine(440).to_audio_segment(duration=1500).set_channels(2).export(str(path), format="mp3")
    return path


def test_process_hands_over_a_wav_and_maps_the_speakers(audio, tmp_path, monkeypatch):
    seen = {}

    def fake_run(name, request, timeout=3600.0, script=None):
        seen.update(name=name, request=request, timeout=timeout)
        return {"speakers": {"SPEAKER_01": [[1.0, 2.0]], "SPEAKER_00": [[0.0, 0.5], [2.5, 3.0]]}}

    monkeypatch.setattr("MAT.utils.external.run_external", fake_run)
    config = Config({"diarizen": {"device": "cpu", "timeout": 60}}, work_directory=str(tmp_path / "work"))
    result = DiarizerDiariZen().process(DiarizerInput(str(audio)), config=config)

    assert seen["name"] == "diarizen" and seen["timeout"] == 60
    assert seen["request"]["model"] == "BUT-FIT/diarizen-wavlm-large-s80-md"
    assert seen["request"]["device"] == "cpu"
    assert seen["request"]["audio"].endswith(".wav") and os.path.isfile(seen["request"]["audio"])
    assert result.to_dict() == {"sprecher_0": [(0.0, 0.5), (2.5, 3.0)], "sprecher_1": [(1.0, 2.0)]}


def test_empty_answer():
    assert DiarizerDiariZen._to_result({}).to_dict() == {}
