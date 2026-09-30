import pytest

from MAT import registry
from MAT.pipelines import PipelineStepInput, PipelineStepResult
from MAT.pipelines.Book import BookPipeline, Chapter
from MAT.tools import SummaryInput, SummaryResult
from MAT.tools.summary import SummaryTool
from MAT.utils.config import Config


def _steps():
    pipeline = BookPipeline()
    return pipeline, {step.__name__: step for step in pipeline._get_steps()}


def test_characters_come_from_person_entities_of_all_chapters():
    chapters = [
        Chapter(heading="Davos", content=[], sentences=["a", "b"],
                ner=[{"PERSON": [("Stannis", 0, 7)], "LOCATION": [("Dragonstone", 0, 11)]},
                     {"PERSON": [("Stannis Baratheon", 0, 17)]}]),
        Chapter(heading="Jon", content=[], sentences=["c"], ner=[{"PERSON": [("Stannis", 0, 7), ("Jon", 0, 3)]}]),
    ]
    _, steps = _steps()
    result = steps["character_task"](PipelineStepInput(
        file="book.epub", config=Config({"book": {"full-name-mentions": 1}}),
        previous_results={"NER": PipelineStepResult("NER", chapters)}))
    assert result.data == [{"name": "Stannis Baratheon", "mentions": 3,
                            "variants": {"Stannis": 2, "Stannis Baratheon": 1}, "chapters": {"Davos": 2, "Jon": 1},
                            "joined": []}]


@registry.register("summarizer", "test-chapters")
class FakeSummarizer(SummaryTool):
    def process(self, origin_data, config):
        FakeSummarizer.seen = origin_data
        return SummaryResult(*[f"summary of {text.splitlines()[0]}" for text in origin_data.text])


def teardown_module():
    registry.unregister("summarizer", "test-chapters")


def test_every_chapter_gets_its_own_summary():
    chapters = [Chapter(heading="Prologue", content=["It was dark."]), Chapter(heading="Davos", content=["Fire."])]
    _, steps = _steps()
    config = Config({"book": {"chapter-summarizer": "test-chapters"}})
    result = steps["summary_task"](PipelineStepInput(
        file="book.epub", config=config, previous_results={"Chapters": PipelineStepResult("Chapters", chapters)}))
    assert [c.summary for c in result.data] == ["summary of Prologue", "summary of Davos"]
    assert FakeSummarizer.seen.kind == "chapter"


def test_chapter_summaries_use_the_chapter_prompts(monkeypatch):
    from langchain_core.language_models.fake import FakeListLLM

    from MAT.tools.summary.llm import LLM, SummaryLLM

    prompts = []

    class Recording(FakeListLLM):
        def _call(self, prompt, stop=None, run_manager=None, **kwargs):
            prompts.append(prompt)
            return super()._call(prompt, stop=stop, run_manager=run_manager, **kwargs)

    monkeypatch.setattr(LLM, "get_llm", lambda self, **kwargs: Recording(responses=["ok"] * 4))
    monkeypatch.setattr(SummaryLLM, "_server_context", classmethod(lambda cls, options: None))
    SummaryLLM().process(SummaryInput("Davos\n\nFire on the beach.", kind="chapter"), config=Config({}))
    assert "You summarize chapters of books" in prompts[0] and "Here is a chapter" in prompts[0]


def test_book_flag_is_written_with_dashes():
    from MAT.cli import build_parser

    args = build_parser().parse_args(["run", "-i", "x", "-o", "y", "--chapter-summarizer", "none"])
    assert args.chapter_summarizer == "none"


def test_spacy_sentences_lose_their_line_breaks():
    pytest.importorskip("en_core_web_md")
    from MAT.tools.text_splitter import SplitterInput
    from MAT.tools.text_splitter.spacy import SplitterSpacy

    splitter = SplitterSpacy()
    result = splitter.process(SplitterInput("Alice met Bob.\nThey both worked for Acme.", language="en"), Config({}))
    assert list(result.sentences) == ["Alice met Bob.", "They both worked for Acme."]
    # words with offsets into their own sentence, so they line up with the NER spans
    second = list(result.sentences)[1]
    for start, end, pos, article in result.tokens[1]:
        assert second[start:end].strip() == second[start:end] != ""
    assert [(second[s:e], pos) for s, e, pos, _ in result.tokens[1]][0] == ("They", "PRON")
    assert result.tokens[0][0][:3] == (0, 5, "PROPN")
    # a second chapter reuses the loaded model
    splitter.process(SplitterInput("Bob left.", language="en"), Config({}))
    assert list(splitter._loaded) == ["en_core_web_md"]
