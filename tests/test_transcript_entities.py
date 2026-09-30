from MAT.pipelines.Podcast import transcript_entities
from MAT.tools import WordTuple, WordTupleSpeaker
from MAT.tools.ner import NERResult
from MAT.utils.config import Config


class FakeNER:
    def __init__(self, find):
        self.find = find
        self.texts = []

    def process(self, origin_data, config):
        self.texts = origin_data.text
        answers = []
        for text in origin_data.text:
            found = {"PERSON": []}
            for name in self.find:
                start = text.find(name)
                if start >= 0:
                    found["PERSON"].append((name, start, start + len(name)))
            answers.append(found)
        return NERResult(*answers)


def _words(speaker, text, start):
    words = []
    for i, token in enumerate(text.split()):
        words.append(WordTupleSpeaker(word=WordTuple(start + i, start + i + 0.5, " " + token), speaker={speaker}))
    return words


def test_entities_get_time_and_speaker_of_their_words():
    words = _words("alex", "Heute geht es um Stannis Baratheon und Davos.", 0.0) + \
        _words("max", "Davos ist toll.", 20.0)
    ner = FakeNER(["Stannis Baratheon", "Davos"])
    entities = transcript_entities(words, ner, Config({}))
    # one text per speaker turn, words joined with single spaces
    assert ner.texts == ["Heute geht es um Stannis Baratheon und Davos.", "Davos ist toll."]
    assert [(e["text"], e["start"], e["end"], e["speakers"]) for e in entities] == [
        ("Stannis Baratheon", 4.0, 5.5, ["alex"]),
        ("Davos", 7.0, 7.5, ["alex"]),
        ("Davos", 20.0, 20.5, ["max"]),
    ]


def test_long_turns_are_cut_into_pieces():
    words = _words("alex", " ".join(f"w{i}" for i in range(7)), 0.0)
    ner = FakeNER([])
    transcript_entities(words, ner, Config({}), max_words=3)
    assert ner.texts == ["w0 w1 w2", "w3 w4 w5", "w6"]


def test_entity_counts_sum_up_per_label():
    from mat_format import PodcastResult, TranscriptEntity

    result = PodcastResult.model_construct(entities=[
        TranscriptEntity(label="PERSON", text="Stannis"), TranscriptEntity(label="PERSON", text="stannis"),
        TranscriptEntity(label="LOCATION", text="Drachenstein"), TranscriptEntity(label="PERSON", text="Davos")])
    assert result.entity_counts() == {"PERSON": {"Stannis": 2, "Davos": 1}, "LOCATION": {"Drachenstein": 1}}


class FakeSplitter:
    """Splits after every word that ends with a full stop, like a sentence model would."""

    def __init__(self):
        self.seen = []

    def process(self, origin_data, config):
        import re

        from MAT.tools.sentences import SentenceResult

        self.seen = origin_data.texts
        return SentenceResult([re.findall(r".*?\.\s*|.+$", text) for text in origin_data.texts])


def test_sentences_follow_the_words_and_never_cross_speakers():
    from MAT.pipelines.Podcast import split_sentences

    words = _words("alex", "Hallo zusammen. Heute geht es um Davos.", 0.0) + _words("max", "Schön. Ja", 20.0)
    splitter = FakeSplitter()
    sentences = split_sentences(words, splitter, Config({}), language="de")
    assert splitter.seen == ["Hallo zusammen. Heute geht es um Davos.", "Schön. Ja"]
    assert [(s["text"], s["speakers"], s["first_word"], s["last_word"]) for s in sentences] == [
        ("Hallo zusammen.", ["alex"], 0, 1), ("Heute geht es um Davos.", ["alex"], 2, 6),
        ("Schön.", ["max"], 7, 7), ("Ja", ["max"], 8, 8)]
    assert (sentences[1]["start"], sentences[1]["end"]) == (2.0, 6.5)


def test_entities_run_per_sentence_when_there_are_sentences():
    from MAT.pipelines.Podcast import split_sentences

    words = _words("alex", "Hallo zusammen. Heute geht es um Stannis Baratheon.", 0.0)
    sentences = split_sentences(words, FakeSplitter(), Config({}))
    ner = FakeNER(["Stannis Baratheon"])
    entities = transcript_entities(words, ner, Config({}), sentences=sentences)
    assert ner.texts == ["Hallo zusammen.", "Heute geht es um Stannis Baratheon."]
    assert [(e["text"], e["start"], e["end"]) for e in entities] == [("Stannis Baratheon", 6.0, 7.5)]


def test_pronouns_and_titles_are_no_person():
    words = _words("alex", "Er sagt der König und Davos kommen.", 0.0)
    entities = transcript_entities(words, FakeNER(["Er", "König", "Davos"]), Config({}))
    assert [e["text"] for e in entities] == ["Davos"]
