from types import SimpleNamespace

import faster_whisper
import numpy as np
import pytest
import whisperx
from faster_whisper.transcribe import Word

from MAT.tools import TransciptorWhisper, TranscriptionInput, WordTuple
from MAT.utils.config import Config


class FakeWhisperModel:
    language = "en"
    transcribe_calls = []

    def __init__(self, *args, **kwargs):
        pass

    def detect_language(self, audio=None, vad_filter=False, **kwargs):
        return self.language, 0.98, []

    def transcribe(self, audio, **kwargs):
        FakeWhisperModel.transcribe_calls.append(kwargs)
        words = [Word(start=0.1, end=0.4, word=" hello", probability=1.0),
                 Word(start=0.5, end=0.9, word=" there", probability=1.0)] if kwargs.get("word_timestamps") else None
        segment = SimpleNamespace(start=0.0, end=1.0, text=" hello there", words=words)
        info = SimpleNamespace(language=self.language, duration=1.0, duration_after_vad=1.0)
        return iter([segment]), info


@pytest.fixture
def fake_whisper(monkeypatch):
    FakeWhisperModel.transcribe_calls = []
    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(faster_whisper, "decode_audio", lambda path: np.zeros(16000, dtype=np.float32))
    config = Config()
    config.parse_config({"Whisper": {"device": "cpu", "compute-type": "int8"}})
    return config


def test_language_with_alignment_model_uses_whisperx(fake_whisper, monkeypatch):
    FakeWhisperModel.language = "en"
    seen = {}

    def fake_align(transcript, model, align_model_metadata, audio, device, **kwargs):
        seen["audio"] = audio
        return {"word_segments": [{"start": 0.1, "end": 0.4, "word": "hello"}, {"start": 0.5, "end": 0.9, "word": "there"}]}

    monkeypatch.setattr(whisperx, "load_align_model", lambda language_code, device: (object(), {}))
    monkeypatch.setattr(whisperx, "align", fake_align)

    result = TransciptorWhisper().process(TranscriptionInput("episode.mp3"), config=fake_whisper)

    call = FakeWhisperModel.transcribe_calls[0]
    assert call["language"] == "en" and call["word_timestamps"] is False
    assert isinstance(seen["audio"], np.ndarray)
    assert [w.word for w in result.word_timings] == ["hello", "there"]


def test_language_without_alignment_model_asks_whisper_for_words(fake_whisper, monkeypatch):
    FakeWhisperModel.language = "xx"
    monkeypatch.setattr(whisperx, "align", lambda *a, **kw: pytest.fail("whisperx alignment should not run"))

    result = TransciptorWhisper().process(TranscriptionInput("episode.mp3"), config=fake_whisper)

    call = FakeWhisperModel.transcribe_calls[0]
    assert call["language"] == "xx" and call["word_timestamps"] is True
    assert result.language == "xx"
    assert result.word_timings == [WordTuple(0.1, 0.4, " hello"), WordTuple(0.5, 0.9, " there")]
