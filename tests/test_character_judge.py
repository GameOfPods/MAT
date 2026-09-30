import json

from MAT import registry
from MAT.tools.characters import (
    CharacterJudgeInput, CharacterJudgeResult, CharacterJudgeTool, MentionQuestion, PairQuestion,
)
from MAT.tools.characters.llm import CharacterJudgeLLM
from MAT.utils.config import Config

SENTENCE = "Davos, den alle den Zwiebelritter nannten, stand am Strand."


class ScriptedLLM:
    """Answers the pair and mention prompts from a function of the prompt, like a model would."""

    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def invoke(self, messages):
        prompt = messages[-1].content
        self.prompts.append(prompt)
        return type("Answer", (), {"content": json.dumps(self.answer(prompt))})()


def _judge(monkeypatch, answer):
    llm = ScriptedLLM(answer)
    monkeypatch.setattr(CharacterJudgeLLM, "client", staticmethod(lambda options, schema=None, num_ctx=None: llm))
    pairs = [PairQuestion(0, "Davos", "Zwiebelritter", ["Davos schwieg."], ["Der Zwiebelritter lachte."], [SENTENCE]),
             PairQuestion(1, "Arya", "Sansa", ["Arya ran."], ["Sansa sang."], [])]
    mentions = [MentionQuestion(0, "Stark", "Arya Stark ran, the Stark girl laughed.", ["Arya Stark", "Eddard Stark"])]
    result = CharacterJudgeLLM().process(CharacterJudgeInput(pairs, mentions, language="de"), Config({}))
    return result, llm


def test_same_needs_a_real_sentence_and_both_orders(monkeypatch):
    def answer(prompt):
        if "Pair 0" in prompt:
            return {"pairs": [{"pair": 0, "answer": "same", "evidence": SENTENCE},
                              # made up evidence doesn't count
                              {"pair": 1, "answer": "same", "evidence": "Arya and Sansa are the same girl."}]}
        return {"mentions": [{"id": 0, "answer": "Arya Stark"}]}

    result, llm = _judge(monkeypatch, answer)
    assert result.same == {0: SENTENCE}
    assert result.mentions == {0: "Arya Stark"}
    # every question twice, the second time the other way round
    assert len(llm.prompts) == 4
    assert '"Davos" and "Zwiebelritter"' in llm.prompts[0] and '"Zwiebelritter" and "Davos"' in llm.prompts[1]
    assert "German" in llm.prompts[0]


def test_answers_that_change_with_the_order_are_dropped(monkeypatch):
    calls = {"n": 0}

    def answer(prompt):
        calls["n"] += 1
        if "Pair 0" in prompt:
            first = '"Davos" and "Zwiebelritter"' in prompt
            return {"pairs": [{"pair": 0, "answer": "same" if first else "unsure", "evidence": SENTENCE}]}
        # the first option in the list wins, whichever it is: a model reading positions, not sentences
        options = "Arya Stark" if prompt.index('"Arya Stark"') < prompt.index('"Eddard Stark"') else "Eddard Stark"
        return {"mentions": [{"id": 0, "answer": options}]}

    result, _ = _judge(monkeypatch, answer)
    assert result.same == {} and result.mentions == {}


@registry.register("characters", "test-judge")
class FakeJudge(CharacterJudgeTool):
    def process(self, origin_data, config):
        FakeJudge.seen = origin_data
        return CharacterJudgeResult(same={q.id: SENTENCE for q in origin_data.pairs})


def teardown_module():
    registry.unregister("characters", "test-judge")


def test_the_book_pipeline_joins_what_the_judge_confirmed():
    from MAT.pipelines import PipelineStepInput, PipelineStepResult
    from MAT.pipelines.Book import BookPipeline, Chapter

    chapter = Chapter(heading="Davos", content=[], sentences=[SENTENCE, "Der Zwiebelritter lachte.", "Davos schwieg."],
                      ner=[{"PERSON": [("Davos", 0, 5), ("Zwiebelritter", 20, 33)]},
                           {"PERSON": [("Zwiebelritter", 4, 17)]}, {"PERSON": [("Davos", 0, 5)]}])
    pipeline = BookPipeline()
    step = next(s for s in pipeline._get_steps() if s.__name__ == "character_task")
    config = Config({"book": {"character-judge": "test-judge"}})
    result = step(PipelineStepInput(file="book.epub", config=config,
                                    previous_results={"NER": PipelineStepResult("NER", [chapter]),
                                                      "Language": PipelineStepResult("Language", "de")}))
    assert [(c["name"], c["mentions"]) for c in result.data] == [("Davos", 4)]
    assert result.data[0]["joined"] == [{"name": "Zwiebelritter", "evidence": SENTENCE}]
    assert FakeJudge.seen.language == "de"


def test_ollama_gets_a_context_that_fits_the_prompt(monkeypatch):
    seen = []
    llm = ScriptedLLM(lambda prompt: {"pairs": [], "mentions": []})
    monkeypatch.setattr(CharacterJudgeLLM, "client",
                        staticmethod(lambda options, schema=None, num_ctx=None: seen.append(num_ctx) or llm))
    pairs = [PairQuestion(i, f"Name{i}", f"Other{i}", ["A long sentence about somebody. " * 20] * 5,
                          ["Another long sentence. " * 20] * 5, []) for i in range(20)]
    CharacterJudgeLLM().process(CharacterJudgeInput(pairs, [], language="de"),
                                Config({"llm-characters": {"preset": "ollama", "model": "qwen3:8b"}}))
    # two calls (both orders), each with room for a prompt far above Ollama's default of a few thousand tokens
    assert len(seen) == 2 and all(n > 20000 for n in seen)
