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
from typing import Any, Dict, Optional, List
import os
from enum import Enum, auto as enum_auto
import logging

from pydantic import Field, field_validator

from MAT.registry import register, require

require("langchain_classic", "langchain_core", "langchain_openai", "langchain_text_splitters", "tiktoken", extra="llm")

from MAT.utils.config import Config, Options  # noqa: E402
from MAT.tools.summary import SummaryTool, SummaryInput, SummaryResult  # noqa: E402
from MAT.tools.summary.llm.prompts import SYSTEM_MESSAGE, PROMPT, REFINE_PROMPT  # noqa: E402


class LLM(Enum):
    OpenAI = enum_auto()

    def get_llm(self, model: str, max_tokens: int, temperature: float = None, reasoning_effort: Optional[str] = None,
                extra_body: Optional[dict] = None, first_token_timeout: float = 900.0, idle_timeout: float = 120.0,
                max_retries: int = 2):
        from MAT.tools.summary.llm.robust import StreamingChatModel

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
    service: str = Field("OpenAI", description="LLM provider. OpenAI works with every OpenAI compatible API, point "
                                               "OPENAI_API_BASE at it.")
    model: str = Field("gpt-5.6-terra", description="Model name at the provider.")
    temperature: Optional[float] = Field(None, description="Sampling temperature. Not sent by default, reasoning "
                                                           "models reject it.")
    max_tokens: int = Field(16384, ge=1, description="Maximum tokens per answer. For reasoning models this includes "
                                                     "the thinking tokens.")
    chunk_size: int = Field(32000, ge=1, description="The transcript is split into chunks of this many tokens. The "
                                                     "first chunk is summarized, the summary is then refined with "
                                                     "each following chunk.")
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

    @field_validator("service")
    @classmethod
    def _known_service(cls, value: str) -> str:
        names = [x.name for x in LLM]
        if value not in names:
            raise ValueError(f"unknown service {value}, choose from {', '.join(names)}")
        return value


@register("summarizer", "llm", description="LangChain refine summary with an OpenAI compatible model")
class SummaryLLM(SummaryTool):
    Options = LLMOptions
    packages = ("langchain-classic", "langchain-openai")
    _LOGGER = logging.getLogger(__name__)

    def describe(self, config: Config) -> Dict[str, Any]:
        info = super().describe(config)
        info["service"] = config.options(self).service
        if "OPENAI_API_BASE" in os.environ:
            info["api_base"] = os.environ["OPENAI_API_BASE"]
        return info

    def process(self, origin_data: SummaryInput, config: Config) -> Optional[SummaryResult]:
        from langchain_classic.chains.summarize import load_summarize_chain
        from langchain_core.prompts import PromptTemplate
        from langchain_core.documents import Document

        options = config.options(self)
        return_summaries: List[str] = []

        llm = LLM.parse_str(name=options.service).get_llm(
            model=options.model,
            max_tokens=options.max_tokens,
            temperature=options.temperature,
            reasoning_effort=options.reasoning_effort,
            extra_body=options.extra_body,
            first_token_timeout=options.first_token_timeout,
            idle_timeout=options.idle_timeout,
            max_retries=options.max_retries,
        )
        self.__class__._LOGGER.info(f'Loaded {options.service} as summarization LLM with model {options.model}')

        len_fun = self._get_len_fun()
        splitter = self._get_splitter(options.chunk_size, len_fun=len_fun, chunk_overlap=options.chunk_overlap)

        for text in origin_data.text:
            doc = Document(text)
            split_doc = splitter.split_documents([doc])
            self.__class__._LOGGER.info(
                f"Split text into {len(split_doc)} documents. "
                f"Original text length: {len_fun(doc.page_content)}. "
                f"Chunk size: {options.chunk_size}"
            )
            chain = load_summarize_chain(
                llm,
                chain_type="refine",
                question_prompt=PromptTemplate.from_template(
                    self._build_template(options.system_message, options.prompt)
                ),
                refine_prompt=PromptTemplate.from_template(
                    self._build_template(options.system_message, options.prompt_refine)
                ),
                return_intermediate_steps=True,
                input_key="input_documents",
                output_key="output_text",
            )

            additional_metadata = "\n".join(
                f"{k}: {v}" for k, v in origin_data.additional_metadata.items()
            ) if len(origin_data.additional_metadata) > 0 else ""

            summary = chain.invoke({
                "input_documents": split_doc,
                "additional_metadata": additional_metadata,
            }, config={"max_concurrency": 1})
            return_summaries.append(summary["output_text"])

        return SummaryResult(*return_summaries)

    @staticmethod
    def _build_template(system_message: str, prompt: str) -> str:
        # The default langchain prompts only know {text} (and {existing_answer}), so the metadata we pass into the
        # chain was silently dropped. Add it unless the prompt already uses it.
        if "{additional_metadata}" in system_message or "{additional_metadata}" in prompt:
            return f"{system_message}\n\n{prompt}"
        return f"{system_message}\n\nAdditional information about the source:\n{{additional_metadata}}\n\n{prompt}"

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
