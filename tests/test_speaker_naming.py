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


def _name_step(answer, library=None, values=None):
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
    if library is not None:
        previous["Speaker Library"] = PipelineStepResult(name="Speaker Library", data=library)
    config = Config({"podcast": dict({"namer": "test-namer"}, **(values or {}))})
    return pipeline, step(PipelineStepInput(file="episode.mp3", config=config, previous_results=previous))


def test_unnamed_speaker_gets_the_name():
    _, result = _name_step([SpeakerName(speaker="sprecher_1", name="Tobi", evidence="sprecher_0: danke tobi")])
    diarization, squished, transcript = (result.data[k] for k in ("diarization", "squished", "transcript"))
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


def _namer_options(values):
    config = Config(values)
    return SpeakerNamingLLM._apply_preset(SpeakerNamingLLM._inherit(config.options(SpeakerNamingLLM), config))


def test_without_its_own_preset_the_namer_follows_the_summary():
    options = _namer_options({"llm": {"preset": "ollama", "model": "qwen3:8b", "chunk-size": 12000}})
    assert (options.preset, options.service, options.model) == ("ollama", "Ollama", "qwen3:8b")


def test_its_own_preset_means_nothing_comes_from_the_summary():
    # the same preset as [llm] still cuts the link, so llm.model doesn't leak over
    options = _namer_options({"llm": {"preset": "openai", "model": "deepseek-flash"},
                              "llm-names": {"preset": "openai"}})
    assert options.model == "gpt-5.6-terra"
    options = _namer_options({"llm": {"model": "deepseek-flash"}, "llm-names": {"preset": "ollama"}})
    assert (options.service, options.model) == ("Ollama", "gpt-5.6-terra")


def test_a_model_set_for_the_namer_always_wins():
    options = _namer_options({"llm": {"preset": "ollama", "model": "qwen3:8b"}, "llm-names": {"model": "qwen3:14b"}})
    assert (options.service, options.model) == ("Ollama", "qwen3:14b")


def test_the_result_names_the_model_that_was_asked(caplog):
    import logging

    from MAT.tools.summary.llm import SummaryLLM

    # a GPU run with llm.preset=ollama wrote "llm-names (gpt-5.6-terra)" and service OpenAI into the result
    config = Config({"llm": {"preset": "ollama", "model": "qwen3:8b"}})
    with caplog.at_level(logging.INFO):
        namer = SpeakerNamingLLM().describe(config)
        summary = SummaryLLM().describe(config)
    assert (namer["model"], namer["service"]) == ("qwen3:8b", "Ollama")
    assert (summary["model"], summary["service"]) == ("qwen3:8b", "Ollama")
    # where the settings come from is said once, before the first file, not by every step
    assert "from [llm]" not in caplog.text


def test_nothing_set_anywhere_keeps_the_defaults():
    options = _namer_options({})
    assert (options.preset, options.service, options.model) == ("none", "OpenAI", "gpt-5.6-terra")


def _library_state(tmp_path, known):
    """What the library step hands on: sprecher_1 is still a label, known maps names to library entries."""
    diarization = DiarizationResult({"alex": [(0.0, 5.0)], "sprecher_1": [(5.0, 9.0)]})
    squished = [_words("alex", "hallo zusammen", 0.0), _words("sprecher_1", "danke alex", 5.0)]
    return {"diarization": diarization, "word_speaker": squished, "squished": squished,
            "transcript": "alex [0.0 - 1.0]: hallo zusammen\nsprecher_1 [5.0 - 6.0]: danke alex",
            "speakers": known, "voices": {"alex": [1.0, 0.0], "sprecher_1": [0.0, 1.0]}}


def test_the_llm_is_not_asked_when_the_library_named_everyone(tmp_path):
    state = _library_state(tmp_path, {"alex": {"name": "alex", "library_id": "alex-1"}})
    state["diarization"] = DiarizationResult({"alex": [(0.0, 5.0)], "tobi": [(5.0, 9.0)]})
    FakeNamer.seen = None
    _, result = _name_step([], library=state)
    assert result.data is None and FakeNamer.seen is None


def test_names_from_the_transcript_go_into_the_library_only_when_allowed(tmp_path):
    folder = tmp_path / "library"
    answer = [SpeakerName(speaker="sprecher_1", name="Tobi", evidence="danke tobi")]
    _, result = _name_step(answer, library=_library_state(tmp_path, {}),
                           values={"speaker-library": str(folder)})
    assert "Tobi" in result.data["diarization"].speaker
    assert not (folder / "speakers.json").exists()

    _, result = _name_step(answer, library=_library_state(tmp_path, {}),
                           values={"speaker-library": str(folder), "speaker-library-learns": "all"})
    stored = json.loads((folder / "speakers.json").read_text())["speakers"]
    assert [(s["name"], s["source"], s["embeddings"]) for s in stored] == [("Tobi", "llm", [[0.0, 1.0]])]
    assert result.data["speakers"]["Tobi"]["library_id"].startswith("tobi-")


def test_a_library_name_beats_the_llm_and_the_mismatch_is_logged(tmp_path, caplog):
    state = _library_state(tmp_path, {"alex": {"name": "alex", "library_id": "alex-1"}})
    _, result = _name_step([SpeakerName(speaker="alex", name="Chris", evidence="danke chris")], library=state)
    assert result.data is None
    assert "speaker library" in caplog.text and "Chris" in caplog.text


def test_a_quote_has_to_be_in_what_the_model_saw():
    from MAT.tools.speakernaming.llm import quoted_in

    text = "sprecher_1 [12.3 - 13.0]: Danke, Alex! Schön, dass du da bist.\nsprecher_0 [14.0 - 15.0]: Gerne."
    assert quoted_in("sprecher_1 [12.3 - 13.0]: danke alex", text)
    assert quoted_in("Danke, Alex -- Gerne", text)
    assert not quoted_in("Danke, Max", text)
    answer = json.dumps({"speakers": [{"id": "sprecher_1", "name": "Max", "confidence": "high",
                                       "evidence": "sprecher_0: Hallo Max"}]})
    assert SpeakerNamingLLM._parse(answer, known=["sprecher_0", "sprecher_1"], lines=text) == []


@pytest.mark.parametrize("base, expected", [
    ("", "schema"), ("https://api.openai.com/v1", "schema"), ("https://api.deepseek.com", "json"),
    ("http://localhost:8080/v1", "off")])
def test_structured_output_follows_the_server(monkeypatch, base, expected):
    from MAT.tools.summary.llm import structured_output

    monkeypatch.setenv("OPENAI_API_BASE", base)
    assert structured_output("OpenAI") == expected
    assert structured_output("Ollama") == "schema"
    assert structured_output("OpenAI", "off") == "off"


def test_the_schema_reaches_the_client(monkeypatch):
    import langchain_ollama
    import langchain_openai

    from MAT.tools.speakernaming.llm import ANSWER_SCHEMA
    from MAT.tools.summary.llm import LLM

    seen = {}
    monkeypatch.setattr(langchain_ollama, "ChatOllama", lambda **kwargs: seen.setdefault("ollama", kwargs))
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: seen.setdefault("openai", kwargs))
    LLM.Ollama.get_llm(model="m", max_tokens=10, schema=ANSWER_SCHEMA, structured="schema").factory(None)
    LLM.OpenAI.get_llm(model="m", max_tokens=10, schema=ANSWER_SCHEMA, structured="json").factory(None)
    assert seen["ollama"]["format"] == ANSWER_SCHEMA
    assert seen["openai"]["model_kwargs"] == {"response_format": {"type": "json_object"}}


# lines and answers from real episodes on the GPU box, shortened
HAMILTON = (
    "sprecher_2 [0.22 - 12.32]: Und damit herzlich willkommen zurück zu Kino-Kompromisse mit meiner Freundin Carla.\n"
    "sprecher_2 [24.635 - 27.9]: 6, schon ein Gast. Wir haben Max dabei.\n"
    "sprecher_1 [28.804 - 31.92]: Einen wunderschönen guten Tag.\n"
    "sprecher_1 [133.824 - 180.2]: Es geht um das Leben von Alexander Hamilton.\n"
    "sprecher_1 [311.702 - 393.996]: Warum ist es bei dir nicht so, Alex?\n"
    "sprecher_2 [394.036 - 436.453]: Nee, leider gar nicht. Carla, du guckst gerade schon wieder so.\n"
    "sprecher_0 [437.0 - 440.0]: Ich bin Carla und ich gucke immer so.\n"
)


@pytest.mark.parametrize("speaker, name, evidence, wrong", [
    # the speaker says the name themself: they talk to or about someone else
    ("sprecher_2", "Carla", "sprecher_2 [0.22 - 12.32]: Und damit herzlich willkommen zurück zu Kino-Kompromisse mit "
                            "meiner Freundin Carla.", True),
    # "Alex" isn't in "Alexander Hamilton"
    ("sprecher_0", "Alex", "sprecher_1 [133.824 - 180.2]: Es geht um das Leben von Alexander Hamilton.", True),
    # somebody else says it and the named speaker answers: fine
    ("sprecher_1", "Max", "sprecher_2 [24.635 - 27.9]: 6, schon ein Gast. Wir haben Max dabei. "
                          "sprecher_1 [28.804 - 31.92]: Einen wunderschönen guten Tag.", False),
    ("sprecher_2", "Alex", "sprecher_1 [311.702 - 393.996]: Warum ist es bei dir nicht so, Alex? / "
                           "sprecher_2 [394.036 - 436.453]: Nee, leider gar nicht.", False),
    # a quote without its label is looked up in the lines
    ("sprecher_2", "Carla", "Carla, du guckst gerade schon wieder so.", True),
    ("sprecher_0", "Carla", "Carla, du guckst gerade schon wieder so. -- Ich bin Carla und ich gucke immer so.", False),
    # introducing yourself is the exception
    ("sprecher_0", "Carla", "sprecher_0 [437.0 - 440.0]: Ich bin Carla und ich gucke immer so.", False),
    # three people: sprecher_1 asks Alex something, and sprecher_2 answers. qwen3:8b named sprecher_0
    ("sprecher_0", "Alex", "sprecher_1 [311.702 - 393.996]: Warum ist es bei dir nicht so, Alex?", True),
    ("sprecher_2", "Alex", "sprecher_1 [311.702 - 393.996]: Warum ist es bei dir nicht so, Alex?", False),
    # what OpenAI answered: only the line with the name, the named speaker is the next to talk
    ("sprecher_1", "Max", "sprecher_2 [24.635 - 27.9]: 6, schon ein Gast. Wir haben Max dabei.", False),
])
def test_evidence_that_names_someone_else(speaker, name, evidence, wrong):
    from MAT.tools.speakernaming.llm import contradicted

    assert bool(contradicted(speaker, name, evidence, HAMILTON, speakers=3)) == wrong
    answer = json.dumps({"speakers": [{"id": speaker, "name": name, "confidence": "high", "evidence": evidence}]})
    names = SpeakerNamingLLM._parse(answer, known=["sprecher_0", "sprecher_1", "sprecher_2"], lines=HAMILTON)
    assert (names == []) == wrong
