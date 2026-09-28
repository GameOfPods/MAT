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
