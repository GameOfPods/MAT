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
from typing import Dict, Optional, List
import json
import os
from enum import Enum, auto as enum_auto
import logging

from MAT.utils.config import ConfigElement, Config
from MAT.tools.summary import SummaryTool, SummaryInput, SummaryResult


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


class SummaryLLM(SummaryTool):
    _LOGGER = logging.getLogger(__name__)

    @classmethod
    def config_name(cls) -> str:
        return "LLM-Summarizer"

    @classmethod
    def config_keys(cls) -> Dict[str, ConfigElement]:
        from MAT.tools.summary.llm.prompts import SYSTEM_MESSAGE, PROMPT, REFINE_PROMPT
        return {
            "service": ConfigElement(
                default_value=list(LLM)[0].name,
                argparse_kwargs={
                    "help": "LLM Service to use. Choose one of the available. Available: %(choices)s. Default: %(default)s",
                    "choices": [x.name for x in LLM],
                    "type": str,
                }
            ),
            "model": ConfigElement(
                default_value="gpt-5.6-terra",
                argparse_kwargs={
                    "help": "model name for your selected LLM-provider. Default: %(default)s",
                    "type": str,
                }
            ),
            "temperature": ConfigElement(
                default_value=None,
                argparse_kwargs={
                    "help": "temperature for your llm model",
                    "type": float,
                }
            ),
            "max-tokens": ConfigElement(
                # reasoning models (gpt-5 and newer) count their thinking tokens here too, 4096 could cut them off
                default_value=16384,
                argparse_kwargs={
                    "help": "Maximum number of tokens the model may generate per call. For reasoning models this "
                            "includes the thinking tokens. Default: %(default)s",
                    "type": int,
                }
            ),
            "chunk-size": ConfigElement(
                default_value=32000,
                argparse_kwargs={
                    "help": "Chunk size for summarization. "
                            "Original text will be split into chunks of this size and then the summarization will be "
                            "run on the first and refined with the following chunks. Default: %(default)s",
                    "type": int,
                }
            ),
            "chunk-overlap": ConfigElement(
                default_value=None,
                argparse_kwargs={
                    "help": "How many tokens neighboring chunks share. Must be smaller than the chunk size. "
                            "Default: 10%% of the chunk size, at most 200",
                    "type": int,
                }
            ),
            "reasoning-effort": ConfigElement(
                default_value="low",
                argparse_kwargs={
                    "help": "How much a reasoning model may think before answering. low, medium and high work with "
                            "OpenAI and DeepSeek, OpenAI also knows none, DeepSeek also max. Use \"unset\" to not "
                            "send the parameter at all (for servers that reject it). Default: %(default)s",
                    "type": str,
                }
            ),
            "extra-body": ConfigElement(
                default_value=None,
                argparse_kwargs={
                    "help": "Extra JSON fields for the request body, for provider specific switches. "
                            "Example to turn DeepSeek thinking off: '{\"thinking\": {\"type\": \"disabled\"}}'",
                    "type": json.loads,
                }
            ),
            "first-token-timeout": ConfigElement(
                default_value=900.0,
                argparse_kwargs={
                    "help": "Seconds to wait for the first streamed token of an answer. Covers queueing at the "
                            "provider and prompt processing. Default: %(default)s",
                    "type": float,
                }
            ),
            "idle-timeout": ConfigElement(
                default_value=120.0,
                argparse_kwargs={
                    "help": "Seconds to wait between two streamed tokens. Every token resets it. "
                            "Default: %(default)s",
                    "type": float,
                }
            ),
            "max-retries": ConfigElement(
                default_value=2,
                argparse_kwargs={
                    "help": "How often a call is tried again after a timeout, connection problem or a busy "
                            "provider. Default: %(default)s",
                    "type": int,
                }
            ),
            "system-message": ConfigElement(
                default_value=SYSTEM_MESSAGE,
                argparse_kwargs={
                    "help": "Message to be passed to the model as system message. "
                            "Dont touch if you dont know what you are doing.",
                    "type": str,
                }
            ),
            "prompt": ConfigElement(
                default_value=PROMPT,
                argparse_kwargs={
                    "help": "Default summary prompt message for llm. "
                            "Dont touch if you don't know what you are doing.",
                    "type": str,
                }
            ),
            "prompt-refine": ConfigElement(
                default_value=REFINE_PROMPT,
                argparse_kwargs={
                    "help": "Prompt that is passed to model to refine summary with chunked text. "
                            "Dont touch if you dont know what you are doing.",
                    "type": str,
                }
            )
        }

    _LOGGER = logging.getLogger(__name__)

    def process(self, origin_data: SummaryInput, config: Config) -> Optional[SummaryResult]:
        try:
            from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
            from langchain.chains.mapreduce import MapReduceChain
            from langchain.chains.summarize import load_summarize_chain
        except ImportError as e:
            from langchain_classic.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
            from langchain_classic.chains.mapreduce import MapReduceChain
            from langchain_classic.chains.summarize import load_summarize_chain
        try:
            from langchain.prompts import PromptTemplate
            from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
            from langchain.docstore.document import Document
        except ImportError:
            from langchain_core.prompts import PromptTemplate
            from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
            from langchain_core.documents import Document

        cfg = config.get_config(key=self)
        return_summaries: List[str] = []

        llm = LLM.parse_str(name=cfg["service"]).get_llm(
            model=cfg["model"],
            max_tokens=cfg["max-tokens"],
            temperature=cfg["temperature"],
            reasoning_effort=cfg["reasoning-effort"],
            extra_body=cfg["extra-body"],
            first_token_timeout=cfg["first-token-timeout"],
            idle_timeout=cfg["idle-timeout"],
            max_retries=cfg["max-retries"],
        )
        self.__class__._LOGGER.info(f'Loaded {cfg["service"]} as summarization LLM with model {cfg["model"]}')

        len_fun = self._get_len_fun()
        splitter = self._get_splitter(cfg["chunk-size"], len_fun=len_fun, chunk_overlap=cfg["chunk-overlap"])

        for text in origin_data.text:
            doc = Document(text)
            split_doc = splitter.split_documents([doc])
            self.__class__._LOGGER.info(
                f"Split text into {len(split_doc)} documents. "
                f"Original text length: {len_fun(doc.page_content)}. "
                f"Chunk size: {cfg['chunk-size']}"
            )
            chain = load_summarize_chain(
                llm,
                chain_type="refine",
                question_prompt=PromptTemplate.from_template(
                    self._build_template(cfg['system-message'], cfg['prompt'])
                ),
                refine_prompt=PromptTemplate.from_template(
                    self._build_template(cfg['system-message'], cfg['prompt-refine'])
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
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")

            def _len_fun(_txt: str) -> int:
                return len(enc.encode(_txt, ))

            return _len_fun
        except ImportError:
            cls._LOGGER.error("Could not import tiktoken. Will use python length for text length estimation")

            def _len_fun(_txt: str) -> int:
                return len(_txt)

            return _len_fun

    @classmethod
    def _get_splitter(cls, chunk_size: int, len_fun, chunk_overlap: Optional[int] = None):

        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
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
                        "\u200b",
                        "\uff0c",
                        "\u3001",
                        "\uff0e",
                        "\u3002",
                        "",
                        ],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len_fun,
        )
        return splitter
