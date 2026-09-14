#  MAT - Toolkit to analyze media
#  Copyright (c) 2025.  RedRem95
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
"""
Chat model wrapper that streams every answer and doesn't wait forever.

Providers like DeepSeek answer HTTP 200 right away and then send empty keep-alive lines until the model is done.
The HTTP read timeout resets on those lines, so a request stuck in the provider queue looks like a slow model.
This wrapper watches the streamed chunks instead: no first chunk within `first_token_timeout` or no next chunk within
`idle_timeout` cancels the request, and it gets retried with a growing wait.
Thinking-only chunks (DeepSeek's reasoning_content) arrive as empty chunks and count as progress too.
"""
import asyncio
import logging
import time
from typing import Any, Callable, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

_LOGGER = logging.getLogger(__name__)

# Wait before retry 1, 2, 3, ... A busy provider needs some time, retrying right away only queues us again.
RETRY_WAITS = (30.0, 120.0, 300.0)

_BUSY_MARKERS = ("unable to start processing", "overloaded", "server busy", "try again later",
                 "too many requests", "rate limit")


class LLMStreamTimeout(TimeoutError):
    pass


class LLMEmptyResponse(RuntimeError):
    pass


def is_retryable(error: BaseException) -> bool:
    if isinstance(error, (LLMStreamTimeout, LLMEmptyResponse)):
        return True
    try:
        import openai
        if isinstance(error, openai.APIConnectionError):  # includes APITimeoutError
            return True
        if isinstance(error, openai.APIStatusError):
            return error.status_code == 429 or error.status_code >= 500
        if isinstance(error, openai.APIError):
            # error inside a 200 response, for example an error event in the stream
            return True
    except ImportError:
        pass
    # DeepSeek's queue timeout comes as HTTP 200 with an error body, langchain raises that as a plain ValueError
    message = str(error).lower()
    return any(marker in message for marker in _BUSY_MARKERS)


class StreamingChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Builds the real chat model for one call. Gets a fresh httpx.AsyncClient, because every call runs in its own
    # event loop (asyncio.run) and pooled async connections can't move between loops.
    factory: Callable[[Any], BaseChatModel]
    first_token_timeout: float = 900.0
    idle_timeout: float = 120.0
    max_retries: int = 2
    progress_interval: float = 30.0

    @property
    def _llm_type(self) -> str:
        return "mat-streaming"

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager=None,
                  **kwargs: Any) -> ChatResult:
        attempt = 0
        while True:
            try:
                message = asyncio.run(self._stream_once(messages, stop=stop, **kwargs))
                return ChatResult(generations=[ChatGeneration(message=message)])
            except Exception as e:
                if attempt >= self.max_retries or not is_retryable(e):
                    raise
                wait = RETRY_WAITS[min(attempt, len(RETRY_WAITS) - 1)]
                attempt += 1
                _LOGGER.warning(f"LLM call failed ({type(e).__name__}: {e}). "
                                f"Try {attempt + 1} of {self.max_retries + 1} in {wait:.0f} s")
                time.sleep(wait)

    async def _stream_once(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                           **kwargs: Any) -> AIMessage:
        import httpx

        started = time.monotonic()
        last_log = started
        merged: Optional[AIMessageChunk] = None
        chunks = 0
        # no read timeout here, the chunk timeouts below do that job
        async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=60.0)) as http_client:
            stream = self.factory(http_client).astream(messages, stop=stop, **kwargs).__aiter__()
            try:
                while True:
                    timeout = self.first_token_timeout if chunks == 0 else self.idle_timeout
                    try:
                        chunk = await asyncio.wait_for(stream.__anext__(), timeout=timeout)
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError as e:
                        what = "the first token" if chunks == 0 else "the next token"
                        raise LLMStreamTimeout(f"No {what} from the LLM within {timeout:.0f} s") from e
                    except ValueError as e:
                        # langchain raises this itself when the stream ended without a single chunk
                        if "no generation chunks" in str(e).lower():
                            raise LLMEmptyResponse("The LLM returned an empty answer, "
                                                   "the provider might be overloaded") from e
                        raise
                    chunks += 1
                    merged = chunk if merged is None else merged + chunk
                    now = time.monotonic()
                    if chunks == 1:
                        _LOGGER.info(f"First token after {now - started:.1f} s")
                        last_log = now
                    elif now - last_log >= self.progress_interval:
                        _LOGGER.info(f"Still receiving the answer, {chunks} chunks after {now - started:.0f} s")
                        last_log = now
            finally:
                if hasattr(stream, "aclose"):
                    await stream.aclose()

        if merged is None or not merged.content:
            raise LLMEmptyResponse("The LLM returned an empty answer, the provider might be overloaded")
        _LOGGER.info(f"Answer complete: {chunks} chunks in {time.monotonic() - started:.1f} s")
        return AIMessage(content=merged.content, additional_kwargs=merged.additional_kwargs,
                         response_metadata=merged.response_metadata, usage_metadata=merged.usage_metadata)


__all__ = ["StreamingChatModel", "LLMStreamTimeout", "LLMEmptyResponse", "is_retryable", "RETRY_WAITS"]
