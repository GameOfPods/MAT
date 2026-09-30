import pydub

from MAT.tools import DiarizationResult, SpeakerIdentificationResult, TranscriptionResult, WordTuple
from MAT.utils.diarization import (
    align_diarization_with_transcription, assign_speakers, squish_word_speaker, word_speaker_to_transcript,
)


class FakeIdentifier:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def process(self, origin_data, config):
        clips = origin_data.get_audio_files()
        self.calls.append([round(clip.duration_seconds, 1) for clip, _ in clips])
        answers = self.answer if isinstance(self.answer, list) else [self.answer] * len(clips)
        return SpeakerIdentificationResult(*answers)


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


def test_speaker_matching_asks_once_with_a_slice_of_everyone():
    diarization = DiarizationResult({"sprecher_0": [(0.0, 4.0), (5.0, 9.0), (10.0, 14.0)],
                                     "sprecher_1": [(4.0, 5.0)]})
    identifier = FakeIdentifier(["alice", None])
    matched = diarization.speaker_matching(identifier=identifier, config=None, seconds=6,
                                           audio=pydub.AudioSegment.silent(duration=15000))
    # one call for both speakers, the first one cut after the segment that crosses 6 seconds
    assert identifier.calls == [[8.0, 1.0]]
    assert matched.speaker == {"alice", "sprecher_1"}


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


def _speakers(diarization, words, **kwargs):
    return [sorted(w.speaker) for w in assign_speakers(DiarizationResult(diarization), words, **kwargs)]


def test_every_word_gets_the_speaker_who_talks_longest_during_it():
    words = [WordTuple(0.0, 1.0, "one"), WordTuple(1.8, 2.4, "two")]
    # "two" overlaps A for 0.2 s and B for 0.4 s
    assert _speakers({"A": [(0.0, 2.0)], "B": [(2.0, 3.0)]}, words) == [["A"], ["B"]]


def test_overlapping_speech_keeps_whoever_was_talking():
    words = [WordTuple(0.0, 0.5, "so"), WordTuple(1.0, 1.5, "yes")]
    # B is talking from the start, A joins and both cover "yes" completely
    assert _speakers({"B": [(0.0, 2.0)], "A": [(0.8, 2.0)]}, words) == [["B"], ["B"]]


def test_words_outside_all_segments_take_the_speaker_around_them():
    words = [WordTuple(0.0, 0.5, "a"), WordTuple(1.0, 1.5, "b"), WordTuple(2.0, 2.5, "c")]
    assert _speakers({"A": [(0.0, 0.6), (1.9, 3.0)]}, words) == [["A"], ["A"], ["A"]]


def test_a_lonely_word_takes_the_closest_segment_or_nobody():
    words = [WordTuple(0.0, 0.5, "a."), WordTuple(0.9, 1.1, "b"), WordTuple(9.0, 9.5, "c")]
    speakers = _speakers({"A": [(0.0, 0.5)], "B": [(1.2, 2.0)]}, words, min_turn=0)
    assert speakers == [["A"], ["B"], []]


def test_words_without_times_follow_their_neighbours():
    words = [WordTuple(0.0, 0.5, "a"), WordTuple(None, None, "2026"), WordTuple(0.6, 1.0, "c")]
    assert _speakers({"A": [(0.0, 1.0)]}, words) == [["A"], ["A"], ["A"]]


def test_a_short_turn_inside_a_sentence_goes_back_to_the_speaker():
    words = [WordTuple(0.0, 0.4, "Also"), WordTuple(0.5, 0.7, "den"), WordTuple(0.75, 0.9, "Titel"),
             WordTuple(1.0, 1.4, "hält"), WordTuple(1.5, 2.0, "er.")]
    diarization = {"A": [(0.0, 0.7), (0.9, 2.0)], "B": [(0.7, 0.9)]}
    assert _speakers(diarization, words) == [["A"]] * 5
    assert _speakers(diarization, words, min_turn=0) == [["A"], ["A"], ["B"], ["A"], ["A"]]


def test_a_short_answer_after_a_finished_sentence_stays():
    words = [WordTuple(0.0, 0.5, "Right?"), WordTuple(0.6, 0.8, "Ja."), WordTuple(1.0, 1.5, "Okay")]
    diarization = {"A": [(0.0, 0.55), (0.9, 2.0)], "B": [(0.55, 0.9)]}
    assert _speakers(diarization, words) == [["A"], ["B"], ["A"]]
