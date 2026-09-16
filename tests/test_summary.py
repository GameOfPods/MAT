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


def test_fill_replaces_placeholders_and_leaves_the_rest_alone():
    filled = SummaryLLM._fill("{additional_metadata}|{text}|{existing_answer}|{unknown}",
                              text="T", metadata="M", existing_answer="E")
    assert filled == "M|T|E|{unknown}"


def test_prompt_placeholders():
    from MAT.tools.summary.llm.prompts import PROMPT, REFINE_PROMPT

    assert "{text}" in PROMPT
    assert "{text}" in REFINE_PROMPT and "{existing_answer}" in REFINE_PROMPT


def test_prompts_keep_the_model_inside_the_transcript(monkeypatch):
    llm = RecordingLLM(responses=["first summary", "refined summary"] * 10)
    monkeypatch.setattr(LLM, "get_llm", lambda self, **kwargs: llm)

    config = Config({"llm": {"chunk-size": 50}})
    text = "\n\n".join(f"sprecher_0 [{i}.0 - {i}.5]: paragraph number {i} about nothing." for i in range(12))
    SummaryLLM().process(SummaryInput(text, additional_metadata={}), config=config)

    assert len(llm.prompts) > 1
    # on a real episode the old system message allowed outside knowledge and the model invented a comparison
    assert all("Use only what the transcript says" in prompt for prompt in llm.prompts)
    assert all("Write no sentence about the text being a summary" in prompt for prompt in llm.prompts)
    # langchain's generic defaults are gone
    assert not any("concise summary of the following" in prompt for prompt in llm.prompts)
    assert "Don't repeat what the summary already says" in llm.prompts[-1]


def test_short_transcript_is_one_call_without_the_refine_prompt(monkeypatch):
    llm = RecordingLLM(responses=["the only summary"] * 5)
    monkeypatch.setattr(LLM, "get_llm", lambda self, **kwargs: llm)

    result = SummaryLLM().process(SummaryInput("sprecher_0 [0.0 - 1.0]: short episode about nothing.",
                                               additional_metadata={}), config=Config({}))

    assert list(result.text) == ["the only summary"]
    assert len(llm.prompts) == 1
    assert "The summary so far" not in llm.prompts[0]


def test_long_transcript_refines_chunk_by_chunk(monkeypatch):
    llm = RecordingLLM(responses=[f"summary {i}" for i in range(1, 13)])
    monkeypatch.setattr(LLM, "get_llm", lambda self, **kwargs: llm)

    config = Config({"llm": {"chunk-size": 50}})
    text = "\n\n".join(f"sprecher_0 [{i}.0 - {i}.5]: paragraph number {i} about nothing." for i in range(12))
    result = SummaryLLM().process(SummaryInput(text, additional_metadata={}), config=config)

    assert len(llm.prompts) > 1
    assert "The summary so far" not in llm.prompts[0]
    # every later call gets the summary built so far
    assert all("The summary so far" in prompt for prompt in llm.prompts[1:])
    assert "summary 1" in llm.prompts[1]
    assert list(result.text) == [f"summary {len(llm.prompts)}"]


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
