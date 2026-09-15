import pytest

from MAT.bench.metrics import cp_word_errors, diarization_errors, midpoint_in, normalize, word_errors


def test_normalize_drops_case_punctuation_and_fillers():
    assert normalize("Hallo, Welt! Äh, das ist's.", "de") == ["hallo", "welt", "das", "ists"]


def test_normalize_splits_hyphens_and_keeps_numbers():
    assert normalize("Mm-hmm, a well-known 5%", "en") == ["a", "well", "known", "5"]
    assert normalize("Straße", "de") == normalize("STRASSE", "de")


def test_normalize_unknown_language_drops_all_fillers():
    assert normalize("uh äh ok", None) == ["ok"]


def test_word_errors():
    errors = word_errors("a b c d".split(), "a x c".split())
    assert (errors.substitutions, errors.deletions, errors.insertions) == (1, 1, 0)
    assert errors.rate == 0.5


def test_word_errors_empty_sides():
    assert word_errors([], ["a"]).insertions == 1
    assert word_errors([], ["a"]).rate is None
    assert word_errors(["a", "b"], []).rate == 1.0


def test_cp_word_errors_finds_the_speaker_mapping():
    reference = {"alice": ["hello", "there"], "bob": ["good", "morning"]}
    hypothesis = {"s1": ["good", "morning"], "s0": ["hello", "there"]}
    errors, mapping = cp_word_errors(reference, hypothesis)
    assert errors.errors == 0 and errors.reference_words == 4
    assert mapping == {"alice": "s0", "bob": "s1"}


def test_cp_word_errors_counts_extra_and_missing_speakers():
    reference = {"alice": ["hello", "there"], "bob": ["good", "morning"]}
    errors, _ = cp_word_errors(reference, {"s0": ["hello", "there"], "s1": ["good", "morning"], "s2": ["noise"]})
    assert (errors.insertions, errors.rate) == (1, 0.25)
    errors, mapping = cp_word_errors(reference, {"s0": ["hello", "there", "good", "morning"]})
    assert errors.errors == 4
    assert None in mapping.values()


def test_wrong_speaker_counts_in_cpwer_but_not_in_wer():
    reference = {"alice": ["one", "two"], "bob": ["three", "four"]}
    hypothesis = {"s0": ["one", "two", "three"], "s1": ["four"]}
    assert cp_word_errors(reference, hypothesis)[0].errors == 2
    assert word_errors("one two three four".split(), "one two three four".split()).errors == 0


def test_diarization_errors_ignore_label_names():
    reference = {"alice": [(0, 10)], "bob": [(10, 20)]}
    errors = diarization_errors(reference, {"x": [(0, 10)], "y": [(10, 20)]}, 0, 20)
    assert errors.rate == 0


def test_diarization_errors_parts():
    reference = {"alice": [(0, 10)], "bob": [(10, 20)]}
    errors = diarization_errors(reference, {"x": [(0, 10)]}, 0, 20)
    assert errors.missed == pytest.approx(10) and errors.total == pytest.approx(20)
    errors = diarization_errors(reference, {"x": [(0, 20)]}, 0, 20)
    assert errors.confusion == pytest.approx(10)


def test_diarization_errors_only_inside_the_window():
    errors = diarization_errors({"alice": [(0, 10)]}, {"x": [(5, 10)]}, 5, 10)
    assert errors.rate == 0 and errors.total == pytest.approx(5)


def test_midpoint_in():
    assert midpoint_in(None, None, None, None)
    assert not midpoint_in(None, None, 0, 10)
    assert midpoint_in(9, 10.5, 0, 10)
    assert not midpoint_in(9.8, 11, 0, 10)
