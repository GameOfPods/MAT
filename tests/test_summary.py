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


def _options(**kwargs):
    from MAT.tools.summary.llm import LLMOptions

    return LLMOptions(**kwargs)


def test_preset_fills_what_you_did_not_set():
    filled = SummaryLLM._apply_preset(_options(preset="ollama"))
    assert filled.service == "Ollama"
    assert (filled.max_tokens, filled.reasoning_effort, filled.idle_timeout) == (4096, "none", 300.0)

    # your own values win over the preset
    mine = SummaryLLM._apply_preset(_options(preset="ollama", **{"max-tokens": 99, "reasoning-effort": "high"}))
    assert (mine.max_tokens, mine.reasoning_effort, mine.service) == (99, "high", "Ollama")


def test_no_preset_changes_nothing():
    plain = _options(model="x")
    assert SummaryLLM._apply_preset(plain) is plain


def test_explicit_chunk_size_wins(monkeypatch):
    monkeypatch.setattr(SummaryLLM, "_server_context", classmethod(lambda cls, options: 200_000))
    assert SummaryLLM._resolve_chunk_size(_options(**{"chunk-size": 5000}), reserved=100, len_fun=len) == 5000


def test_auto_chunk_size_fills_the_server_context(monkeypatch):
    monkeypatch.setattr(SummaryLLM, "_server_context", classmethod(lambda cls, options: 32768))
    size = SummaryLLM._resolve_chunk_size(_options(**{"max-tokens": 16384}), reserved=1000, len_fun=len)
    assert size == int((32768 - 16384 - 1000) * 0.9)


def test_auto_chunk_size_is_capped_and_uses_the_table(monkeypatch):
    from MAT.tools.summary.llm import MAX_CHUNK_SIZE

    monkeypatch.setattr(SummaryLLM, "_server_context", classmethod(lambda cls, options: None))
    # deepseek reports nothing, the table says a million, the cap keeps it sane
    assert SummaryLLM._resolve_chunk_size(_options(model="deepseek-flash"), reserved=100, len_fun=len) == MAX_CHUNK_SIZE


def test_auto_chunk_size_falls_back_for_unknown_models(monkeypatch):
    from MAT.tools.summary.llm import FALLBACK_CHUNK_SIZE

    monkeypatch.setattr(SummaryLLM, "_server_context", classmethod(lambda cls, options: None))
    assert SummaryLLM._resolve_chunk_size(_options(model="something-local"), reserved=100,
                                          len_fun=len) == FALLBACK_CHUNK_SIZE


def test_server_context_reads_llama_cpp_and_model_listings(monkeypatch):
    import requests

    answers = {}

    class Response:
        def __init__(self, payload):
            self.payload, self.ok = payload, payload is not None

        def json(self):
            return self.payload

    monkeypatch.setenv("OPENAI_API_BASE", "http://server:8080/v1")
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: Response(answers.get(url.rsplit("/", 1)[-1])))

    mine = _options(model="mine")

    # llama.cpp: what the server really loaded
    answers.clear()
    answers["props"] = {"default_generation_settings": {"n_ctx": 8192}}
    assert SummaryLLM._server_context(mine) == 8192

    # a listing that reports the context of the model
    answers.clear()
    answers["models"] = {"data": [{"id": "other", "context_length": 999}, {"id": "mine", "max_model_len": 40960}]}
    assert SummaryLLM._server_context(mine) == 40960

    answers.clear()
    answers["models"] = {"data": [{"id": "mine", "meta": {"n_ctx_train": 131072}}]}
    assert SummaryLLM._server_context(mine) == 131072

    # nothing useful, and a server that errors, both give None instead of breaking the run
    answers.clear()
    answers["models"] = {"data": [{"id": "mine"}]}
    assert SummaryLLM._server_context(mine) is None
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: (_ for _ in ()).throw(OSError("no server")))
    assert SummaryLLM._server_context(mine) is None


def test_no_api_base_means_no_probe(monkeypatch):
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    assert SummaryLLM._server_context(_options(model="mine")) is None


def test_ollama_context_comes_from_api_show(monkeypatch):
    import requests

    seen = {}

    class Response:
        ok = True

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.update(url=url, body=json)
        return Response({"model_info": {"general.architecture": "qwen3", "qwen3.context_length": 40960}})

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    options = _options(service="Ollama", model="qwen3:8b")

    assert SummaryLLM._server_context(options) == 40960
    assert seen["url"] == "http://localhost:11434/api/show"
    assert seen["body"] == {"model": "qwen3:8b"}


def test_ollama_url_takes_option_then_environment(monkeypatch):
    from MAT.tools.summary.llm import ollama_url

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert ollama_url() == "http://localhost:11434"
    monkeypatch.setenv("OLLAMA_HOST", "gpu-box:11434")
    assert ollama_url() == "http://gpu-box:11434"
    assert ollama_url("https://somewhere/") == "https://somewhere"


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
    # chunk-size is "auto" by default, don't ask a server that someone's environment happens to point at
    monkeypatch.setattr(SummaryLLM, "_server_context", staticmethod(lambda model: None))

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
