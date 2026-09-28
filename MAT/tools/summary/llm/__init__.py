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
from typing import Any, Dict, Literal, Optional, List, Union
import os
from enum import Enum, auto as enum_auto
import logging

from pydantic import Field, field_validator

from MAT.registry import register, require

require("langchain_core", "langchain_openai", "langchain_text_splitters", "tiktoken", extra="llm")

from MAT.utils.config import Config, Options  # noqa: E402
from MAT.tools.summary import SummaryTool, SummaryInput, SummaryResult  # noqa: E402
from MAT.tools.summary.llm.prompts import MAP_PROMPT, PROMPT, REDUCE_PROMPT, REFINE_PROMPT, SYSTEM_MESSAGE  # noqa: E402

# Used for "auto" when the server doesn't report a context size. Providers that don't say: DeepSeek (1M tokens for
# deepseek-flash and deepseek-pro since V4, prompt and answer share it), OpenAI doesn't report it either.
KNOWN_CONTEXTS: Dict[str, int] = {"deepseek": 1_000_000}
# Very long inputs make summaries worse in the middle, so we don't fill a million tokens even when we could
MAX_CHUNK_SIZE = 100_000
MIN_CHUNK_SIZE = 2_000
FALLBACK_CHUNK_SIZE = 32_000

# Ollama wants a yes or no for thinking, we have OpenAI's scale. "low" means "think as little as possible" for us.
REASONING_TO_THINKING: Dict[str, bool] = {"none": False, "low": False, "medium": True, "high": True, "xhigh": True,
                                          "max": True}

# Starting points for the two ways to run this. Anything set in the config or with --set wins over them.
PRESETS: Dict[str, Dict[str, Any]] = {
    # a hosted API: queues can be long, thinking models are common, answers may be big
    "openai": {"service": "OpenAI", "reasoning_effort": "low", "max_tokens": 16384, "chunk_size": "auto",
               "first_token_timeout": 900.0, "idle_timeout": 120.0},
    # Ollama on your own machine: no queue, but a GTX 1080 Ti chews on a long prompt for minutes
    "ollama": {"service": "Ollama", "reasoning_effort": "none", "max_tokens": 4096, "chunk_size": "auto",
               "first_token_timeout": 1800.0, "idle_timeout": 300.0},
    # llama.cpp and vLLM speak the OpenAI API, point OPENAI_API_BASE at them
    "llamacpp": {"service": "OpenAI", "reasoning_effort": "unset", "max_tokens": 4096, "chunk_size": "auto",
                 "first_token_timeout": 1800.0, "idle_timeout": 300.0},
}


def check_llm_access(service: str, model: str, base_url: Optional[str], what: str, skip: str) -> None:
    """Stops a run before the first file when the LLM can't work: no API key, or an Ollama that doesn't answer or
    doesn't have the model. `what` is the step ("The summary"), `skip` how to run without it."""
    from MAT.utils.config import ConfigError

    if service == "Ollama":
        import requests

        url = ollama_url(base_url)
        try:
            response = requests.get(f"{url}/api/tags", timeout=5)
            response.raise_for_status()
            names = {entry.get("name") for entry in response.json().get("models") or []}
        except Exception as e:
            raise ConfigError(f"{what} uses Ollama at {url}, but it doesn't answer ({e.__class__.__name__}). Start "
                              f"it (systemctl start ollama, or ollama serve), point llm.base-url or OLLAMA_HOST at "
                              f"it, or {skip}.") from e
        wanted = {model, f"{model}:latest"}
        if names and not names & wanted:
            raise ConfigError(f"{what} wants {model}, but Ollama at {url} only has {', '.join(sorted(names))}. "
                              f"Run `ollama pull {model}` or pick one of those.")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        where = os.environ.get("OPENAI_API_BASE") or "the OpenAI API"
        raise ConfigError(f"{what} uses {where} and needs OPENAI_API_KEY (any value for a local server behind "
                          f"OPENAI_API_BASE). Set it, use a local model (--set llm.preset=ollama), or {skip}.")


def ollama_url(base_url: Optional[str] = None) -> str:
    """Where Ollama listens: the option, then $OLLAMA_HOST, then the default. A bare host:port gets a scheme."""
    url = (base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").strip().rstrip("/")
    return url if url.startswith(("http://", "https://")) else f"http://{url}"


class LLM(Enum):
    OpenAI = enum_auto()
    Ollama = enum_auto()

    def get_llm(self, model: str, max_tokens: int, temperature: float = None, reasoning_effort: Optional[str] = None,
                extra_body: Optional[dict] = None, first_token_timeout: float = 900.0, idle_timeout: float = 120.0,
                max_retries: int = 2, base_url: Optional[str] = None, num_ctx: Optional[int] = None):
        from MAT.tools.summary.llm.robust import StreamingChatModel

        if self == self.Ollama:
            from logging import WARNING
            logging.getLogger("httpx").setLevel(WARNING)
            try:
                from langchain_ollama import ChatOllama
            except ImportError as e:
                raise ImportError("The Ollama service needs langchain-ollama, install the llm extra") from e

            kwargs: Dict[str, Any] = {"model": model, "num_predict": max_tokens,
                                      "base_url": ollama_url(base_url)}
            if temperature is not None:
                kwargs["temperature"] = temperature
            # Ollama loads a context of its own choosing unless we say otherwise, and cuts the prompt silently
            if num_ctx is not None:
                kwargs["num_ctx"] = num_ctx
            thinking = REASONING_TO_THINKING.get(reasoning_effort)
            if thinking is not None:
                kwargs["reasoning"] = thinking
            logging.getLogger("LLM-Service").info(f"Using ollama at {kwargs['base_url']}"
                                                  + (f" with a context of {num_ctx} tokens" if num_ctx else ""))

            def ollama_factory(_http_async_client):
                return ChatOllama(**kwargs)

            return StreamingChatModel(factory=ollama_factory, first_token_timeout=first_token_timeout,
                                      idle_timeout=idle_timeout, max_retries=max_retries)

        if self == self.OpenAI:
            from logging import WARNING
            logging.getLogger("httpx").setLevel(WARNING)
            from langchain_openai import ChatOpenAI
            if "OPENAI_API_BASE" in os.environ:
                logging.getLogger("LLM-Service").info(f"Using openai api located at {os.environ['OPENAI_API_BASE']}")

            kwargs = {"model": model, "max_tokens": max_tokens, "stream_usage": True,
                      # retries are done by StreamingChatModel, with longer waits than the SDK uses
                      "max_retries": 0,
                      # langchain's own chunk timeout (120 s by default) also applies to the first token and would
                      # cancel waits in a provider queue, StreamingChatModel handles both timeouts
                      "stream_chunk_timeout": None}
            if temperature is not None:
                kwargs["temperature"] = temperature
            if reasoning_effort is not None and reasoning_effort != "unset":
                kwargs["reasoning_effort"] = reasoning_effort
            if extra_body:
                kwargs["extra_body"] = extra_body

            def factory(http_async_client):
                return ChatOpenAI(http_async_client=http_async_client, **kwargs)

            return StreamingChatModel(factory=factory, first_token_timeout=first_token_timeout,
                                      idle_timeout=idle_timeout, max_retries=max_retries)

        raise ValueError(f"LLM of type {self} not defined")

    @classmethod
    def parse_str(cls, name: str):
        for x in cls:
            if x.name == name:
                return x
        raise ValueError(f"LLM of type {name} not defined")


class LLMOptions(Options):
    preset: Literal["none", "openai", "ollama", "llamacpp"] = Field(
        "none", description="Starting point for the other options: openai for a hosted API, ollama for a local "
                            "Ollama, llamacpp for a local OpenAI compatible server. Everything you set yourself "
                            "wins over it.")
    service: str = Field("OpenAI", description="LLM provider. OpenAI works with every OpenAI compatible API, point "
                                               "OPENAI_API_BASE at it. Ollama talks to Ollama directly, which is the "
                                               "only way to know and set its context size.")
    base_url: Optional[str] = Field(None, description="Where Ollama listens. Default: $OLLAMA_HOST, else "
                                                      "http://localhost:11434. The OpenAI service uses "
                                                      "$OPENAI_API_BASE instead.")
    model: str = Field("gpt-5.6-terra", description="Model name at the provider.")
    temperature: Optional[float] = Field(None, description="Sampling temperature. Not sent by default, reasoning "
                                                           "models reject it.")
    max_tokens: int = Field(16384, ge=1, description="Maximum tokens per answer. For reasoning models this includes "
                                                     "the thinking tokens.")
    chunk_size: Union[int, Literal["auto"]] = Field("auto", description="Tokens per chunk. One chunk means one call, "
                                                                        "more chunks are refined one after another. "
                                                                        '"auto" asks the server for the context size '
                                                                        "and fills it, so a transcript that fits is "
                                                                        "summarized in a single call.")
    chunk_overlap: Optional[int] = Field(None, ge=0, description="Tokens shared by neighboring chunks. Not set: 10% "
                                                                 "of the chunk size, at most 200.")
    reasoning_effort: Optional[str] = Field("low", description='How much a reasoning model may think. low, medium '
                                                               'and high work with OpenAI and DeepSeek. "unset" '
                                                               'doesn\'t send the parameter.')
    extra_body: Optional[Dict[str, Any]] = Field(None, description='Extra JSON fields for the request, for provider '
                                                                   'specific switches. DeepSeek thinking off: '
                                                                   '{"thinking": {"type": "disabled"}}')
    first_token_timeout: float = Field(900.0, gt=0, description="Seconds to wait for the first streamed token. "
                                                                "Covers queueing at the provider and prompt "
                                                                "processing.")
    idle_timeout: float = Field(120.0, gt=0, description="Seconds to wait between two streamed tokens.")
    max_retries: int = Field(2, ge=0, description="Retries after timeouts, connection problems or a busy provider.")
    system_message: str = Field(SYSTEM_MESSAGE, description="Instructions put in front of every prompt.")
    prompt: str = Field(PROMPT, description="Prompt for the first chunk, has to contain {text}.")
    prompt_refine: str = Field(REFINE_PROMPT, description="Prompt for the following chunks, has to contain "
                                                          "{existing_answer} and {text}.")
    strategy: Literal["refine", "map-reduce"] = Field(
        "refine", description="For transcripts that need more than one call. refine: every call rewrites the "
                              "summary with the next chunk. map-reduce: notes per chunk, then one call that writes "
                              "the summary from all notes, which keeps late chunks from taking over and suits small "
                              "local models.")
    prompt_map: str = Field(MAP_PROMPT, description="map-reduce: prompt for the notes on one chunk, has to contain "
                                                    "{text}. {part} and {parts} are filled in too.")
    prompt_reduce: str = Field(REDUCE_PROMPT, description="map-reduce: prompt that turns the notes into the summary, "
                                                          "has to contain {text}.")

    @field_validator("service")
    @classmethod
    def _known_service(cls, value: str) -> str:
        names = [x.name for x in LLM]
        if value not in names:
            raise ValueError(f"unknown service {value}, choose from {', '.join(names)}")
        return value


@register("summarizer", "llm", description="Summary from an OpenAI compatible API or a local Ollama")
class SummaryLLM(SummaryTool):
    Options = LLMOptions
    packages = ("langchain-core", "langchain-openai")
    _LOGGER = logging.getLogger(__name__)

    def describe(self, config: Config) -> Dict[str, Any]:
        info = super().describe(config)
        info["service"] = config.options(self).service
        if "OPENAI_API_BASE" in os.environ:
            info["api_base"] = os.environ["OPENAI_API_BASE"]
        return info

    def process(self, origin_data: SummaryInput, config: Config) -> Optional[SummaryResult]:
        from langchain_core.messages import HumanMessage, SystemMessage

        options = self._apply_preset(config.options(self))
        return_summaries: List[str] = []

        len_fun = self._get_len_fun()
        metadata = "\n".join(f"{k}: {v}" for k, v in origin_data.additional_metadata.items())
        system = self._fill(options.system_message, metadata=metadata)
        if metadata and "{additional_metadata}" not in options.system_message:
            system = f"{system}\n\nAdditional information about the source:\n{metadata}"

        # the instructions and the answer have to fit next to the transcript. A refine call also carries the summary
        # so far, which can be as long as one answer.
        reserved = len_fun(system) + len_fun(options.prompt)
        if options.strategy == "map-reduce":
            # a map call carries one chunk, a reduce call notes of at most one chunk's size
            refine_reserved = len_fun(system) + max(len_fun(options.prompt_map), len_fun(options.prompt_reduce))
        else:
            refine_reserved = len_fun(system) + len_fun(options.prompt_refine) + options.max_tokens
        longest = max((len_fun(text) for text in origin_data.text), default=0)
        chunk_size = self._resolve_chunk_size(options, reserved=reserved, len_fun=len_fun,
                                              refine_reserved=refine_reserved, longest=longest)
        splitter = self._get_splitter(chunk_size, len_fun=len_fun, chunk_overlap=options.chunk_overlap)

        # Ollama needs to be told how much context to load, the others take what the prompt brings
        refines = longest > chunk_size
        num_ctx = (chunk_size + options.max_tokens + (refine_reserved if refines else reserved)
                   if options.service == "Ollama" else None)
        llm = LLM.parse_str(name=options.service).get_llm(
            model=options.model,
            max_tokens=options.max_tokens,
            temperature=options.temperature,
            reasoning_effort=options.reasoning_effort,
            extra_body=options.extra_body,
            first_token_timeout=options.first_token_timeout,
            idle_timeout=options.idle_timeout,
            max_retries=options.max_retries,
            base_url=options.base_url,
            num_ctx=num_ctx,
        )
        self.__class__._LOGGER.info(f'Loaded {options.service} as summarization LLM with model {options.model}')

        for text in origin_data.text:
            chunks = splitter.split_text(text)
            self.__class__._LOGGER.info(f"Summarizing {len_fun(text)} tokens in {len(chunks)} chunk(s) of at most "
                                        f"{chunk_size} tokens")
            if len(chunks) > 1 and options.strategy == "map-reduce":
                return_summaries.append(self._map_reduce(llm, options, system, metadata, chunks, chunk_size, len_fun))
                continue
            summary: Optional[str] = None
            for number, chunk in enumerate(chunks, start=1):
                if summary is None:
                    prompt = self._fill(options.prompt, text=chunk, metadata=metadata)
                else:
                    self.__class__._LOGGER.info(f"Refining the summary with chunk {number} of {len(chunks)}")
                    prompt = self._fill(options.prompt_refine, text=chunk, metadata=metadata, existing_answer=summary)
                # the instructions go in as a real system message, not glued in front of the transcript
                answer = llm.invoke([SystemMessage(system), HumanMessage(prompt)])
                summary = str(getattr(answer, "content", answer) or "").strip()
            return_summaries.append(summary or "")

        return SummaryResult(*return_summaries)

    @classmethod
    def _map_reduce(cls, llm, options: "LLMOptions", system: str, metadata: str, chunks: List[str],
                    chunk_size: int, len_fun) -> str:
        """Notes per chunk, then the summary from the notes. Notes that don't fit into one call together are
        reduced in groups first."""
        from langchain_core.messages import HumanMessage, SystemMessage

        def ask(prompt: str) -> str:
            answer = llm.invoke([SystemMessage(system), HumanMessage(prompt)])
            return str(getattr(answer, "content", answer) or "").strip()

        notes = []
        for number, chunk in enumerate(chunks, start=1):
            cls._LOGGER.info(f"Notes on chunk {number} of {len(chunks)}")
            notes.append(ask(cls._fill(options.prompt_map, text=chunk, metadata=metadata, part=str(number),
                                       parts=str(len(chunks)))))

        def joined(group: List[str]) -> str:
            return "\n\n".join(f"Part {i}:\n{note}" for i, note in enumerate(group, start=1))

        while len(notes) > 1 and len_fun(joined(notes)) > chunk_size:
            groups: List[List[str]] = [[]]
            for note in notes:
                if groups[-1] and len_fun(joined(groups[-1] + [note])) > chunk_size:
                    groups.append([])
                groups[-1].append(note)
            if len(groups) >= len(notes):
                break  # every note is as big as a chunk, grouping gets nowhere
            cls._LOGGER.info(f"The notes don't fit into one call, joining them in {len(groups)} groups first")
            notes = [ask(cls._fill(options.prompt_reduce, text=joined(group), metadata=metadata)) for group in groups]
        cls._LOGGER.info(f"Writing the summary from the notes on {len(chunks)} chunks")
        return ask(cls._fill(options.prompt_reduce, text=joined(notes), metadata=metadata))

    @staticmethod
    def _fill(template: str, text: str = "", metadata: str = "", existing_answer: str = "", part: str = "",
              parts: str = "") -> str:
        """Fills the placeholders of a prompt. Replacing instead of str.format, so braces in a transcript or in a
        prompt of your own don't blow up the run."""
        for name, value in (("{text}", text), ("{additional_metadata}", metadata),
                            ("{existing_answer}", existing_answer), ("{part}", part), ("{parts}", parts)):
            template = template.replace(name, value)
        return template

    @classmethod
    def _apply_preset(cls, options: "LLMOptions") -> "LLMOptions":
        """Fills the options a preset knows about, except the ones set in the config or with --set."""
        values = PRESETS.get(options.preset)
        if not values:
            return options
        update = {name: value for name, value in values.items() if name not in options.model_fields_set}
        if not update:
            return options
        cls._LOGGER.info(f"Preset {options.preset} sets " + ", ".join(f"{k.replace('_', '-')}={v}"
                                                                      for k, v in sorted(update.items())))
        return options.model_copy(update=update)

    @classmethod
    def effective_options(cls, config: Config) -> "LLMOptions":
        return cls._apply_preset(config.options(cls))

    @classmethod
    def preflight(cls, config: Config) -> None:
        options = cls.effective_options(config)
        check_llm_access(options.service, options.model, options.base_url, "The summary", "--summarizer none")

    @classmethod
    def _resolve_chunk_size(cls, options: "LLMOptions", reserved: int, len_fun, refine_reserved: Optional[int] = None,
                            longest: Optional[int] = None) -> int:
        """Tokens per chunk. An explicit number wins, "auto" asks the server, then the table, then the fallback.
        With "auto", a transcript that fits gets one call. One that doesn't gets chunks small enough that a refine
        call (refine_reserved: prompt plus the summary so far) still fits into the context."""
        if options.chunk_size != "auto":
            return int(options.chunk_size)
        context, source = cls._server_context(options), "the server"
        if context is None:
            context = next((size for name, size in KNOWN_CONTEXTS.items() if options.model.startswith(name)), None)
            source = "our table of known models"
        if context is None:
            cls._LOGGER.info(f"{options.model} doesn't say how much context it has, using {FALLBACK_CHUNK_SIZE} "
                             f"tokens per chunk. Set llm.chunk-size if you know better.")
            return FALLBACK_CHUNK_SIZE
        def fitting(prompt: int) -> int:
            # 10 % for the tokenizer counting differently than the model does
            return max(MIN_CHUNK_SIZE, min(int((context - options.max_tokens - prompt) * 0.9), MAX_CHUNK_SIZE))

        chunk = fitting(reserved)
        if refine_reserved is not None and longest is not None and longest > chunk:
            chunk = fitting(refine_reserved)
            cls._LOGGER.info(f"{options.model} has {context} tokens of context according to {source}. The transcript "
                             f"needs several calls, using {chunk} tokens per chunk to leave room for the summary so "
                             f"far")
        else:
            cls._LOGGER.info(f"{options.model} has {context} tokens of context according to {source}, "
                             f"using {chunk} tokens per chunk")
        return chunk

    @classmethod
    def _server_context(cls, options: "LLMOptions") -> Optional[int]:
        """Context size the server reports for the model, None when it doesn't say. Never raises, a summary
        shouldn't fail because a server answers something unexpected."""
        import requests

        def ask(url, payload=None, headers=None):
            try:
                if payload is None:
                    response = requests.get(url, headers=headers or {}, timeout=10)
                else:
                    response = requests.post(url, json=payload, headers=headers or {}, timeout=10)
                return response.json() if response.ok else None
            except Exception as e:
                logging.getLogger(__name__).debug(f"Could not ask {url}: {e}")
                return None

        if options.service == "Ollama":
            shown = ask(f"{ollama_url(options.base_url)}/api/show", payload={"model": options.model}) or {}
            # the key is named after the architecture, for example llama.context_length
            for key, value in (shown.get("model_info") or {}).items():
                if key.endswith(".context_length") and isinstance(value, int) and value > 0:
                    return value
            length = (shown.get("details") or {}).get("context_length")
            return length if isinstance(length, int) and length > 0 else None

        base = os.environ.get("OPENAI_API_BASE")
        if not base:
            return None
        base = base.rstrip("/")
        headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"} if "OPENAI_API_KEY" in os.environ else {}

        def get(url):
            return ask(url, headers=headers)

        # llama.cpp reports what it really loaded, which can be smaller than what the model could do
        loaded = ((get(f"{base}/props") or {}).get("default_generation_settings") or {}).get("n_ctx")
        if isinstance(loaded, int) and loaded > 0:
            return loaded
        for entry in (get(f"{base}/models") or {}).get("data") or []:
            if not isinstance(entry, dict) or entry.get("id") != options.model:
                continue
            meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
            # names used by OpenRouter, vLLM and llama.cpp
            for value in (entry.get("context_length"), entry.get("max_model_len"), entry.get("max_context_length"),
                          meta.get("n_ctx"), meta.get("n_ctx_train")):
                if isinstance(value, int) and value > 0:
                    return value
        return None

    @classmethod
    def _get_len_fun(cls):
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        def _len_fun(_txt: str) -> int:
            return len(enc.encode(_txt, ))

        return _len_fun

    @classmethod
    def _get_splitter(cls, chunk_size: int, len_fun, chunk_overlap: Optional[int] = None):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # The splitter's own default overlap is 200, which crashed for chunk sizes below 200
        if chunk_overlap is None:
            chunk_overlap = min(200, chunk_size // 10)
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(f"chunk-overlap ({chunk_overlap}) has to be between 0 and chunk-size ({chunk_size})")
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n",
                        "\n",
                        ".",
                        ",",
                        " ",
                        "​",
                        "，",
                        "、",
                        "．",
                        "。",
                        "",
                        ],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len_fun,
        )
        return splitter
