import numpy as np
import pytest

from MAT.tools.events import AudioEvent, join_windows, windows
from MAT.utils.config import Config


def test_windows_cover_the_whole_audio():
    assert windows(8.0, 10.0, 5.0) == [(0.0, 8.0)]
    assert windows(23.0, 10.0, 5.0) == [(0.0, 10.0), (5.0, 15.0), (10.0, 20.0), (13.0, 23.0)]


def test_neighbouring_windows_become_one_event_with_edges_in_the_overlap():
    spans = [(0.0, 10.0), (5.0, 15.0), (10.0, 20.0), (15.0, 25.0)]
    scores = [{"music": 0.1}, {"music": 0.8}, {"music": 0.9}, {"music": 0.1, "laughter": 0.6}]
    events = join_windows(spans, scores, threshold=0.5)
    # music in windows 2 and 3: from the middle of the overlap with window 1 to the middle of the one with window 4
    assert events == [AudioEvent("music", 7.5, 17.5, 0.9), AudioEvent("laughter", 17.5, 25.0, 0.6)]
    assert join_windows(spans, scores, threshold=0.5, min_duration=9) == [AudioEvent("music", 7.5, 17.5, 0.9)]


class FakeExtractor:
    sampling_rate = 16000

    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def __call__(self, batch, sampling_rate, return_tensors):
        import torch

        class Inputs(dict):
            def to(self, device):
                return self

        # the loudness of each window stands in for its features
        return Inputs(input_values=torch.tensor([[float(np.abs(w).mean())] for w in batch]))


class FakeAST:
    config = type("C", (), {"id2label": {0: "Music", 1: "Laughter", 2: "Speech"}})()

    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_values):
        import torch

        loud = input_values[:, 0]
        # loud windows are music, quiet ones speech
        logits = torch.stack([(loud - 0.3) * 50, torch.full_like(loud, -10.0), (0.3 - loud) * 50], dim=1)
        return type("Out", (), {"logits": logits})()


def test_audioset_tags_music_windows(monkeypatch, tmp_path):
    transformers = pytest.importorskip("transformers")
    from pydub.generators import Sine

    from MAT.tools.events import EventInput
    from MAT.tools.events.audioset import EventsAudioSet

    monkeypatch.setattr(transformers, "ASTFeatureExtractor", FakeExtractor)
    monkeypatch.setattr(transformers, "ASTForAudioClassification", FakeAST)
    audio = Sine(440).to_audio_segment(duration=20000, volume=-40) + \
        Sine(440).to_audio_segment(duration=20000, volume=-1)
    path = tmp_path / "episode.wav"
    audio.export(str(path), format="wav")

    config = Config({"audioset": {"device": "cpu", "groups": {"music": ["Music", "Not a class"],
                                                               "laughter": ["Laughter"]}}})
    events = EventsAudioSet().process(EventInput(str(path)), config).events
    assert [e.label for e in events] == ["music"]
    assert 17.5 <= events[0].start <= 22.5 and events[0].end == 40.0
