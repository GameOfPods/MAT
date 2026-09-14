import asyncio
from typing import Any, List

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk

from MAT.tools.summary.llm import LLM
from MAT.tools.summary.llm.robust import LLMEmptyResponse, LLMStreamTimeout, StreamingChatModel, is_retryable


class ScriptedChat(BaseChatModel):
    """Streams (delay, text) steps, optionally raises an error instead."""
    steps: List[Any] = []
    error: Any = None

    @property
    def _llm_type(self):
        return "scripted"

    def _generate(self, *args, **kwargs):
        raise NotImplementedError()

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        if self.error is not None:
            raise self.error
        for delay, text in self.steps:
            await asyncio.sleep(delay)
            yield ChatGenerationChunk(message=AIMessageChunk(content=text))


def make_model(attempts: List[ScriptedChat], **kwargs) -> StreamingChatModel:
    calls = []

    def factory(http_client):
        calls.append(http_client)
        return attempts[len(calls) - 1]

    model = StreamingChatModel(factory=factory, **kwargs)
    model.__dict__["_test_calls"] = calls
    return model


@pytest.fixture
def sleeps(monkeypatch):
    waits = []
    monkeypatch.setattr("time.sleep", lambda s: waits.append(s))
    return waits


def test_streams_and_merges(sleeps):
    model = make_model([ScriptedChat(steps=[(0, "Hello"), (0, " "), (0, "world")])])
    assert model.invoke([HumanMessage("hi")]).content == "Hello world"
    assert sleeps == []


def test_empty_thinking_chunks_count_as_progress(sleeps):
    # thinking chunks have no content but keep arriving, the idle timer must not fire
    steps = [(0.05, "")] * 6 + [(0, "done")]
    model = make_model([ScriptedChat(steps=steps)], first_token_timeout=1, idle_timeout=0.1)
    assert model.invoke([HumanMessage("hi")]).content == "done"


def test_first_token_timeout_retries_then_succeeds(sleeps):
    slow = ScriptedChat(steps=[(1.0, "late")])
    fast = ScriptedChat(steps=[(0, "ok")])
    model = make_model([slow, fast], first_token_timeout=0.05, idle_timeout=1, max_retries=2)
    assert model.invoke([HumanMessage("hi")]).content == "ok"
    assert sleeps == [30.0]


def test_idle_timeout_gives_up_after_retries(sleeps):
    stalls = [ScriptedChat(steps=[(0, "start"), (1.0, "never")]) for _ in range(2)]
    model = make_model(stalls, first_token_timeout=1, idle_timeout=0.05, max_retries=1)
    with pytest.raises(LLMStreamTimeout):
        model.invoke([HumanMessage("hi")])
    assert sleeps == [30.0]


def test_busy_error_is_retried(sleeps):
    busy = ScriptedChat(error=ValueError(
        "{'message': 'We were unable to start processing your request within the 900-second timeout limit. "
        "Please try again later.'}"))
    model = make_model([busy, ScriptedChat(steps=[(0, "ok")])], max_retries=2)
    assert model.invoke([HumanMessage("hi")]).content == "ok"
    assert sleeps == [30.0]


def test_other_errors_are_not_retried(sleeps):
    model = make_model([ScriptedChat(error=ValueError("model does not exist"))], max_retries=2)
    with pytest.raises(ValueError, match="does not exist"):
        model.invoke([HumanMessage("hi")])
    assert sleeps == []


def test_empty_answer_is_retried_with_growing_waits(sleeps):
    model = make_model([ScriptedChat(steps=[]) for _ in range(3)], max_retries=2)
    with pytest.raises(LLMEmptyResponse):
        model.invoke([HumanMessage("hi")])
    assert sleeps == [30.0, 120.0]


def test_is_retryable_openai_errors():
    import openai
    request = httpx.Request("POST", "https://api.example.com/chat/completions")
    assert is_retryable(openai.APIConnectionError(request=request))
    assert is_retryable(openai.RateLimitError("slow down", response=httpx.Response(429, request=request), body=None))
    assert is_retryable(openai.InternalServerError("oops", response=httpx.Response(503, request=request), body=None))
    assert not is_retryable(openai.BadRequestError("bad", response=httpx.Response(400, request=request), body=None))


@pytest.mark.parametrize("effort, expected", [("low", "low"), ("unset", None), (None, None)])
def test_get_llm_passes_options(monkeypatch, effort, expected):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    llm = LLM.OpenAI.get_llm(model="gpt-5.6-terra", max_tokens=1000, reasoning_effort=effort,
                             extra_body={"thinking": {"type": "disabled"}}, first_token_timeout=5,
                             idle_timeout=2, max_retries=1)
    assert isinstance(llm, StreamingChatModel)
    assert (llm.first_token_timeout, llm.idle_timeout, llm.max_retries) == (5, 2, 1)
    inner = llm.factory(httpx.AsyncClient())
    assert inner.reasoning_effort == expected
    assert inner.extra_body == {"thinking": {"type": "disabled"}}
    assert inner.max_retries == 0 and inner.stream_usage is True
    # langchain's 120 s chunk timeout would also hit the first token while waiting in a provider queue
    assert inner.stream_chunk_timeout is None
