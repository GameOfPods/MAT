from typing import List

import pytest
from langchain_core.language_models.fake import FakeListLLM

from MAT.tools import SummaryInput
from MAT.tools.summary.llm import LLM, SummaryLLM
from MAT.utils.config import Config


class RecordingLLM(FakeListLLM):
    prompts: List[str] = []

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        self.prompts.append(prompt)
        return super()._call(prompt, stop=stop, run_manager=run_manager, **kwargs)


def test_splitter_overlap_follows_chunk_size():
    assert SummaryLLM._get_splitter(20, len)._chunk_overlap == 2
    assert SummaryLLM._get_splitter(15000, len)._chunk_overlap == 200
    assert SummaryLLM._get_splitter(300, len, chunk_overlap=50)._chunk_overlap == 50


@pytest.mark.parametrize("overlap", [-1, 20, 30])
def test_splitter_rejects_bad_overlap(overlap):
    with pytest.raises(ValueError):
        SummaryLLM._get_splitter(20, len, chunk_overlap=overlap)


def test_build_template_keeps_custom_placeholder():
    assert SummaryLLM._build_template("sys", "{additional_metadata} {text}") == "sys\n\n{additional_metadata} {text}"
    assert "{additional_metadata}" in SummaryLLM._build_template("sys", "{text}")


def test_metadata_reaches_every_prompt(monkeypatch):
    llm = RecordingLLM(responses=["first summary", "refined summary"] * 10)
    monkeypatch.setattr(LLM, "get_llm", lambda self, **kwargs: llm)

    # small chunks so the refine prompt gets used too (chunk sizes below 200 used to crash)
    config = Config({"llm": {"chunk-size": 50}})
    text = "\n\n".join(f"This is paragraph number {i} of a longer transcript about nothing." for i in range(12))

    result = SummaryLLM().process(SummaryInput(text, additional_metadata={"filename": "episode.mp3"}), config=config)

    assert len(list(result.text)) == 1
    assert len(llm.prompts) > 1
    assert all("filename: episode.mp3" in p for p in llm.prompts)
