from types import SimpleNamespace

import pytest
import torch
from pyannote.core import Annotation, Segment
from pydub.generators import Sine

from MAT.tools.diarizators import DiarizerInput
from MAT.tools.diarizators.pyannote import DiarizerPyannote
from MAT.utils.config import Config, ConfigError


class FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        full = Annotation()
        full[Segment(0.0, 1.0)] = "SPEAKER_01"
        full[Segment(0.5, 1.5)] = "SPEAKER_00"
        exclusive = Annotation()
        exclusive[Segment(0.0, 0.5)] = "SPEAKER_01"
        exclusive[Segment(0.5, 1.5)] = "SPEAKER_00"
        return SimpleNamespace(speaker_diarization=full, exclusive_speaker_diarization=exclusive)


@pytest.fixture
def run(tmp_path, monkeypatch):
    audio = tmp_path / "episode.wav"
    Sine(440).to_audio_segment(duration=2000).set_channels(2).export(str(audio), format="wav")
    fake = FakePipeline()
    loaded = {}

    def load(options, device):
        loaded.update(device=device, model=options.model)
        return fake

    monkeypatch.setattr(DiarizerPyannote, "_load_pipeline", staticmethod(load))

    def run(values):
        config = Config({"pyannote-diarization": values}, work_directory=str(tmp_path))
        return fake, loaded, DiarizerPyannote().process(DiarizerInput(str(audio)), config=config)

    return run


def test_passes_decoded_audio_and_speaker_counts(run):
    fake, loaded, result = run({"device": "cpu", "min-speakers": 2, "max-speakers": 4})
    audio, kwargs = fake.calls[0]
    assert kwargs == {"min_speakers": 2, "max_speakers": 4}
    assert audio["sample_rate"] == 16000
    assert audio["waveform"].dtype == torch.float32 and audio["waveform"].shape[0] == 1
    assert abs(audio["waveform"].shape[1] - 32000) <= 16
    assert loaded == {"device": "cpu", "model": "pyannote/speaker-diarization-community-1"}
    assert result.to_dict() == {"sprecher_0": [(0.5, 1.5)], "sprecher_1": [(0.0, 1.0)]}


def test_exclusive_diarization(run):
    _, _, result = run({"device": "cpu", "exclusive": True})
    assert result.to_dict() == {"sprecher_0": [(0.5, 1.5)], "sprecher_1": [(0.0, 0.5)]}


def test_older_pyannote_returns_the_annotation():
    annotation = Annotation()
    annotation[Segment(1.0, 2.0)] = "A"
    assert DiarizerPyannote._to_result(annotation, exclusive=False).to_dict() == {"sprecher_0": [(1.0, 2.0)]}


def test_speaker_range_is_checked(tmp_path):
    config = Config({"pyannote-diarization": {"min-speakers": 5, "max-speakers": 2}}, work_directory=str(tmp_path))
    with pytest.raises(ConfigError, match="max-speakers"):
        config.options(DiarizerPyannote)
