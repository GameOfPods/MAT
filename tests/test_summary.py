from typing import List

from langchain_core.language_models.fake import FakeListLLM

from MAT.tools import SummaryInput, SummaryLLM
from MAT.tools.summary.llm import LLM
from MAT.utils.config import Config


class RecordingLLM(FakeListLLM):
    prompts: List[str] = []

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        self.prompts.append(prompt)
        return super()._call(prompt, stop=stop, run_manager=run_manager, **kwargs)


def test_build_template_keeps_custom_placeholder():
    assert SummaryLLM._build_template("sys", "{additional_metadata} {text}") == "sys\n\n{additional_metadata} {text}"
    assert "{additional_metadata}" in SummaryLLM._build_template("sys", "{text}")


def test_metadata_reaches_every_prompt(monkeypatch):
    llm = RecordingLLM(responses=["first summary", "refined summary"] * 10)
    monkeypatch.setattr(LLM, "get_llm", lambda self, **kwargs: llm)

    config = Config()
    # small chunks so the refine prompt gets used too. The splitter has a fixed overlap of 200, so stay above that
    config.parse_config({"LLM-Summarizer": {"chunk-size": 300}})
    text = "\n\n".join(f"This is paragraph number {i} of a longer transcript about nothing." for i in range(80))

    result = SummaryLLM().process(SummaryInput(text, additional_metadata={"filename": "episode.mp3"}), config=config)

    assert len(list(result.text)) == 1
    assert len(llm.prompts) > 1
    assert all("filename: episode.mp3" in p for p in llm.prompts)
