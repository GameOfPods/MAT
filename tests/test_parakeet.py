import math
from types import SimpleNamespace

import pytest
from pydub.generators import Sine

from MAT.tools import TranscriptionInput, WordTuple
from MAT.tools.transcriptors.parakeet import TranscriptorParakeet, guess_language
from MAT.utils.audio import Window
from MAT.utils.config import Config


class FakeModel:
    def __init__(self, words):
        self.words = words
        self.calls = []

    def transcribe(self, files, **kwargs):
        self.calls.append((list(files), kwargs))
        return [SimpleNamespace(timestamp={"word": self.words}) for _ in files]


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "episode.wav"
    Sine(440).to_audio_segment(duration=2000).set_channels(2).export(str(path), format="wav")
    return path


def test_process_returns_the_word_timestamps(audio, tmp_path, monkeypatch):
    model = FakeModel([{"word": "Hallo", "start": 0.1, "end": 0.4}, {"word": "zusammen,", "start": 0.5, "end": 0.9},
                       {"word": " ", "start": 1.0, "end": 1.0}])
    loaded = {}

    def load(options, device):
        loaded.update(device=device, local_attention=options.local_attention)
        return model

    monkeypatch.setattr(TranscriptorParakeet, "_load_model", staticmethod(load))
    config = Config({"parakeet": {"device": "cpu", "language": "de"}}, work_directory=str(tmp_path / "work"))
    result = TranscriptorParakeet().process(TranscriptionInput(str(audio)), config=config)

    assert result.word_timings == [WordTuple(0.1, 0.4, "Hallo"), WordTuple(0.5, 0.9, "zusammen,")]
    assert (result.language, result.duration) == ("de", pytest.approx(2.0))
    files, kwargs = model.calls[0]
    assert len(files) == 1 and kwargs["timestamps"] is True
    assert loaded == {"device": "cpu", "local_attention": True}


def test_merge_pieces_shifts_and_keeps_words_once():
    windows = [Window(0.0, 65.0, -math.inf, 60.0), Window(55.0, 120.0, 60.0, math.inf)]
    # "b" is in the overlap of both pieces, its middle (60.25 s) belongs to the second one
    pieces = [[WordTuple(10.0, 10.5, "a"), WordTuple(59.0, 61.5, "b")],
              [WordTuple(4.0, 6.5, "b"), WordTuple(20.0, 20.4, "c")]]
    assert TranscriptorParakeet._merge_pieces(windows, pieces) == [
        WordTuple(10.0, 10.5, "a"), WordTuple(59.0, 61.5, "b"), WordTuple(75.0, 75.4, "c")]


def test_words_without_timestamps():
    assert TranscriptorParakeet._words("just text") == []
    assert TranscriptorParakeet._words(SimpleNamespace(timestamp={})) == []


def test_guess_language():
    assert guess_language("Das ist ein ziemlich langer deutscher Satz über das Wetter und die Politik.") == "de"
    assert guess_language("This is a fairly long English sentence about the weather and politics.") == "en"
    assert guess_language("   ") is None
