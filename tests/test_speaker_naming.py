import json

import pytest

from MAT import registry
from MAT.pipelines import PipelineStepInput, PipelineStepResult
from MAT.pipelines.Podcast import PodcastPipeline
from MAT.tools import WordTuple, WordTupleSpeaker
from MAT.tools.diarizators import DiarizationResult
from MAT.tools.speakernaming import SpeakerName, SpeakerNamingResult, SpeakerNamingTool
from MAT.tools.speakernaming.llm import SpeakerNamingLLM
from MAT.utils.config import Config

KNOWN = ["sprecher_0", "sprecher_1"]


def _answer(**entry):
    return json.dumps({"speakers": [{"id": "sprecher_0", "name": "Alex", "confidence": "high",
                                     "evidence": "sprecher_1 [12.3 - 13.0]: danke alex", **entry}]})


def test_a_good_answer_is_taken():
    (name,) = SpeakerNamingLLM._parse(_answer(), known=KNOWN)
    assert (name.speaker, name.name) == ("sprecher_0", "Alex")
    assert "danke alex" in name.evidence


def test_text_around_the_json_is_fine():
    assert SpeakerNamingLLM._parse(f"Sure!\n```json\n{_answer()}\n```", known=KNOWN)


@pytest.mark.parametrize("entry, why", [
    ({"confidence": "low"}, "not sure"),
    ({"confidence": "medium"}, "not sure enough"),
    ({"evidence": ""}, "nothing to back it up"),
    ({"name": ""}, "no name"),
    ({"name": "sprecher_0"}, "that's the label, not a name"),
    ({"name": "12345"}, "not a name"),
    ({"id": "somebody_else"}, "not a speaker of this episode"),
])
def test_everything_shaky_is_dropped(entry, why):
    assert SpeakerNamingLLM._parse(_answer(**entry), known=KNOWN) == [], why


def test_two_speakers_cant_get_the_same_name():
    answer = json.dumps({"speakers": [
        {"id": "sprecher_0", "name": "Alex", "confidence": "high", "evidence": "a"},
        {"id": "sprecher_1", "name": "alex", "confidence": "high", "evidence": "b"},
    ]})
    assert [n.speaker for n in SpeakerNamingLLM._parse(answer, known=KNOWN)] == ["sprecher_0"]


def test_answers_that_are_not_json():
    assert SpeakerNamingLLM._parse("I think the first one is Alex", known=KNOWN) == []
    assert SpeakerNamingLLM._parse('{"speakers": [', known=KNOWN) == []
    assert SpeakerNamingLLM._parse("", known=KNOWN) == []


def test_samples_take_the_opening_and_some_lines_of_everyone():
    from MAT.tools.speakernaming import SpeakerNamingInput

    lines = [f"sprecher_{i % 2} [{i * 60}.0 - {i * 60}.5]: line {i}" for i in range(40)]
    opening, samples = SpeakerNamingLLM._samples(SpeakerNamingInput(lines=lines, speakers=KNOWN),
                                                 opening_minutes=10, lines_per_speaker=3)
    assert opening.splitlines() == lines[:11]
    assert len(samples.splitlines()) == 6


@registry.register("namer", "test-namer")
class FakeNamer(SpeakerNamingTool):
    answer = []

    def process(self, origin_data, config):
        FakeNamer.seen = origin_data
        return SpeakerNamingResult(FakeNamer.answer)


def teardown_module():
    registry.unregister("namer", "test-namer")


def _words(speaker: str, text: str, start: float):
    return WordTupleSpeaker(word=WordTuple(start=start, end=start + 1, word=text), speaker={speaker})


def _name_step(answer):
    FakeNamer.answer = answer
    pipeline = PodcastPipeline()
    step = next(s for s in pipeline._get_steps() if s.__name__ == "name_speakers")
    raw = DiarizationResult({"sprecher_0": [(0.0, 5.0)], "sprecher_1": [(5.0, 9.0)]})
    # the identifier matched one voice to a gold clip, that speaker is called alex now
    matched = DiarizationResult({"alex": [(0.0, 5.0)], "sprecher_1": [(5.0, 9.0)]})
    squished = [_words("alex", "hallo zusammen", 0.0), _words("sprecher_1", "danke alex", 5.0)]
    transcript = "alex [0.0 - 1.0]: hallo zusammen\nsprecher_1 [5.0 - 6.0]: danke alex"
    previous = {
        "Diarization": PipelineStepResult(name="Diarization", data=raw),
        "Speaker Matching": PipelineStepResult(name="Speaker Matching", data=matched),
        "Finalizing transcript": PipelineStepResult(name="Finalizing transcript",
                                                    data=(squished, squished, transcript)),
        "Transcription": PipelineStepResult(name="Transcription", data=None),
    }
    config = Config({"podcast": {"namer": "test-namer"}})
    return pipeline, step(PipelineStepInput(file="episode.mp3", config=config, previous_results=previous))


def test_unnamed_speaker_gets_the_name():
    _, result = _name_step([SpeakerName(speaker="sprecher_1", name="Tobi", evidence="sprecher_0: danke tobi")])
    diarization, word_speaker, squished, transcript = result.data
    assert sorted(diarization.speaker) == ["alex", "Tobi"] or sorted(diarization.speaker) == ["Tobi", "alex"]
    assert "Tobi [5.0 - 6.0]: danke alex" in transcript
    assert all(speaker in ("alex", "Tobi") for word in squished for speaker in word.speaker)


def test_gold_label_wins_and_the_mismatch_is_logged(caplog):
    _, result = _name_step([SpeakerName(speaker="alex", name="Chris", evidence="sprecher_1: danke chris")])
    assert result.data is None
    assert "gold label" in caplog.text.lower()
    assert "Chris" in caplog.text


def test_nothing_found_keeps_everything():
    _, result = _name_step([])
    assert result.data is None


def test_the_namer_sees_every_speaker():
    _name_step([])
    assert sorted(FakeNamer.seen.speakers) == ["alex", "sprecher_1"]
    assert len(FakeNamer.seen.lines) == 2
