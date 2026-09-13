import pydub

from MAT.tools import DiarizationResult, SpeakerIdentificationResult, TranscriptionResult, WordTuple
from MAT.utils.diarization import align_diarization_with_transcription, squish_word_speaker, word_speaker_to_transcript


class FakeIdentifier:
    def __init__(self, answer):
        self.answer = answer

    def process(self, origin_data, config):
        return SpeakerIdentificationResult(self.answer)


def test_speaker_matching_without_match_keeps_labels():
    diarization = DiarizationResult({"sprecher_0": [(0.0, 1.0)], "sprecher_1": [(1.0, 2.0)]})
    matched = diarization.speaker_matching(identifier=FakeIdentifier(None), config=None,
                                           audio=pydub.AudioSegment.silent(duration=3000))
    assert matched.to_dict() == {"sprecher_0": [(0.0, 1.0)], "sprecher_1": [(1.0, 2.0)]}


def test_speaker_matching_merges_same_person():
    diarization = DiarizationResult({"sprecher_0": [(0.0, 1.0)], "sprecher_1": [(1.0, 2.0)]})
    matched = diarization.speaker_matching(identifier=FakeIdentifier("alice"), config=None,
                                           audio=pydub.AudioSegment.silent(duration=3000))
    assert matched.speaker == {"alice"}
    assert sorted(matched.get_diarization("alice")) == [(0.0, 1.0), (1.0, 2.0)]


def test_alignment_and_transcript():
    diarization = DiarizationResult({"A": [(0.0, 1.0)], "B": [(1.0, 3.0)]})
    transcript = TranscriptionResult(word_timings=[
        WordTuple(0.1, 0.4, "hello"), WordTuple(0.5, 0.9, "there"), WordTuple(1.2, 1.8, "hi"),
    ])
    word_speaker = align_diarization_with_transcription(diarization=diarization, transcript=transcript)
    assert [w.speaker for w in word_speaker] == [{"A"}, {"A"}, {"B"}]

    squished = squish_word_speaker(word_speaker)
    assert [w.word.word for w in squished] == ["hello there", "hi"]
    assert word_speaker_to_transcript(squished) == ["A [0.1 - 0.9]: hello there", "B [1.2 - 1.8]: hi"]
