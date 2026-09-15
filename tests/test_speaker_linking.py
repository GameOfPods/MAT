import numpy as np

from MAT.tools.diarizators.nemo import DiarizerNEMO, link_speakers
from MAT.tools.speakeridentification.pyannote import SpeakerIdetificationPyannote


def test_link_speakers_is_one_to_one():
    mapping = link_speakers(np.array([[0.9, 0.1], [0.8, 0.2]]), ["x", "y"], ["a", "b"], threshold=0.3)
    assert mapping == {"x": "a", "y": None}


def test_link_speakers_takes_the_best_total():
    mapping = link_speakers(np.array([[0.9, 0.85], [0.8, 0.1]]), ["x", "y"], ["a", "b"], threshold=0.3)
    assert mapping == {"x": "b", "y": "a"}


def test_link_speakers_without_known_speakers():
    assert link_speakers(np.zeros((2, 0)), ["x", "y"], [], threshold=0.3) == {"x": None, "y": None}


def _same_voice(gold, audios):
    """Fake embeddings: the audio is just the voice name, similarity is 1 for the same voice."""
    names = list(gold)
    return names, np.array([[1.0 if voice in gold[name] else 0.0 for name in names] for voice in audios])


def test_link_pieces_keeps_every_segment():
    segments = [{"0": [(0, 1)], "1": [(1, 2)]},
                {"0": [(0, 1)], "1": [(1, 2)], "2": [(2, 3)]}]
    # in the second piece bob comes first and alice was split into two local speakers
    audio = [{"0": "alice", "1": "bob"},
             {"0": "bob", "1": "alice", "2": "alice"}]
    first, second = DiarizerNEMO._link_pieces(segments, audio, _same_voice, threshold=0.3)
    assert first == {"sprecher_0": [(0, 1)], "sprecher_1": [(1, 2)]}
    assert second["sprecher_1"] == [(0, 1)]
    # the old code put both alice parts under sprecher_0 and the second one replaced the first
    assert set(second) == {"sprecher_0", "sprecher_1", "sprecher_2"}
    assert sorted(t for times in second.values() for t in times) == [(0, 1), (1, 2), (2, 3)]


def test_link_pieces_uses_the_history_and_empty_pieces():
    segments = [{"0": [(0, 1)]}, {}, {"0": [(5, 6)], "1": [(6, 7)]}]
    audio = [{"0": "alice"}, {}, {"0": "carol", "1": "alice"}]
    linked = DiarizerNEMO._link_pieces(segments, audio, _same_voice, threshold=0.3)
    assert linked == [{"sprecher_0": [(0, 1)]}, {}, {"sprecher_1": [(5, 6)], "sprecher_0": [(6, 7)]}]


def test_link_pieces_single_piece_does_not_compare():
    def fail(gold, audios):
        raise AssertionError("no linking needed for one piece")

    (linked,) = DiarizerNEMO._link_pieces([{"0": [(0, 1)], "1": [(2, 3)]}], [{"0": "a", "1": "b"}], fail, 0.3)
    assert linked == {"sprecher_0": [(0, 1)], "sprecher_1": [(2, 3)]}


def _joined_voice(gold, audios):
    """Fake embeddings for merging: a speaker's audio is the joined voice names, the same text means the same voice."""
    names = list(gold)
    return names, np.array([[1.0 if audio == "".join(gold[name]) else 0.0 for name in names] for audio in audios])


def test_merge_joins_speakers_that_sound_alike():
    segments = [{"sprecher_0": [(0, 1)], "sprecher_1": [(1, 2)]}, {"sprecher_2": [(2, 3)], "sprecher_1": [(3, 4)]}]
    audio = [{"sprecher_0": ["alice"], "sprecher_1": ["bob"]}, {"sprecher_2": ["alice"], "sprecher_1": ["bob"]}]
    merged = DiarizerNEMO._merge_speakers(segments, audio, _joined_voice, threshold=0.5)
    # sprecher_2 sounds like sprecher_0, bob keeps his own speaker
    assert merged == [{"sprecher_0": [(0, 1)], "sprecher_1": [(1, 2)]}, {"sprecher_0": [(2, 3)], "sprecher_1": [(3, 4)]}]


def test_merge_keeps_different_speakers_and_renumbers():
    segments = [{"sprecher_0": [(0, 1)], "sprecher_3": [(1, 2)], "sprecher_10": [(2, 3)]}]
    audio = [{"sprecher_0": ["alice"], "sprecher_3": ["bob"], "sprecher_10": ["carol"]}]
    merged = DiarizerNEMO._merge_speakers(segments, audio, _joined_voice, threshold=0.5)
    assert merged == [{"sprecher_0": [(0, 1)], "sprecher_1": [(1, 2)], "sprecher_2": [(2, 3)]}]


def test_link_pieces_merges_only_when_asked():
    segments = [{"0": [(0, 1)]}, {"0": [(0, 1)], "1": [(1, 2)]}]
    audio = [{"0": "alice"}, {"0": "alice", "1": "alice"}]
    assert len(set().union(*DiarizerNEMO._link_pieces(segments, audio, _same_voice, threshold=0.3))) == 2
    merged = DiarizerNEMO._link_pieces(segments, audio, _same_voice, threshold=0.3, merge_threshold=0.5)
    assert set().union(*merged) == {"sprecher_0"}


def test_identify_keeps_the_best_match_per_audio(monkeypatch):
    matrix = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.2]])
    monkeypatch.setattr(SpeakerIdetificationPyannote, "similarities", staticmethod(lambda **kwargs: (["a", "b"], matrix)))
    assert SpeakerIdetificationPyannote.identify(model="m", gold={}, audios=[]) == ["a", "a", None]
    monkeypatch.setattr(SpeakerIdetificationPyannote, "similarities",
                        staticmethod(lambda **kwargs: ([], np.zeros((1, 0)))))
    assert SpeakerIdetificationPyannote.identify(model="m", gold={}, audios=[]) == [None]
