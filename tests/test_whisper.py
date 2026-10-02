from faster_whisper.transcribe import Word

from MAT.tools import WordTuple
from MAT.tools.transcriptors.whisper import TransciptorWhisper


def test_words_from_segments_without_word_timestamps():
    segments = [{"start": 0.0, "end": 2.0, "text": " Hello there.", "words": None}]
    assert TransciptorWhisper._words_from_segments(segments) == [WordTuple(0.0, 2.0, "Hello there.")]


def test_words_from_segments_with_words():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "a b", "words": [Word(start=0.0, end=0.4, word="a", probability=1.0),
                                                           {"start": 0.5, "end": 1.0, "word": "b"}]},
        {"start": 1.0, "end": 2.0, "text": "c", "words": [(1.0, 2.0, "c")]},
    ]
    assert TransciptorWhisper._words_from_segments(segments) == [
        WordTuple(0.0, 0.4, "a"), WordTuple(0.5, 1.0, "b"), WordTuple(1.0, 2.0, "c"),
    ]


def test_fix_broken_times_fills_missing_start_and_end():
    words = [WordTuple(None, 0.5, "a"), WordTuple(None, None, "b"), WordTuple(1.0, 1.5, "c")]
    fixed = TransciptorWhisper._fix_broken_times(words, init=0.1, fin=2.0)
    assert all(w.start is not None and w.end is not None for w in fixed)
    assert fixed[0].start == 0.1


def test_faster_whisper_can_decode_with_the_installed_pyav(tmp_path):
    """PyAV 19 dropped an argument faster-whisper 1.2.1 passes to av.open, and every transcription failed on the GPU
    box with "open() got an unexpected keyword argument 'metadata_errors'". pyproject.toml caps av for that."""
    import wave

    import numpy as np
    from faster_whisper import decode_audio

    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(np.zeros(16000, dtype=np.int16).tobytes())
    assert len(decode_audio(str(path))) == 16000
