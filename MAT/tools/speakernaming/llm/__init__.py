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
Asks an LLM what the speakers are called, based on the transcript.

Only names with "high" confidence and a quoted line are kept, everything else is thrown away here. The connection
settings work like the summary ones, see MAT/tools/summary/llm.
"""
import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

from MAT.registry import register, require

require("langchain_core", "langchain_openai", "tiktoken", extra="llm")

from MAT.tools.speakernaming import (  # noqa: E402
    SpeakerName, SpeakerNamingInput, SpeakerNamingResult, SpeakerNamingTool,
)
from MAT.tools.speakernaming.llm.prompts import PROMPT, SYSTEM_MESSAGE  # noqa: E402
from MAT.utils.config import Config, Options  # noqa: E402

# A name we accept: letters, optionally a second part after a space, hyphen or apostrophe. No digits and no
# underscores, so a model that echoes "sprecher_0" back at us doesn't get through.
_NAME = re.compile(r"^[^\W\d_]+(?:[ '’\-][^\W\d_]+)*$", re.UNICODE)


class SpeakerNamingOptions(Options):
    preset: Literal["none", "openai", "ollama", "llamacpp"] = Field(
        "none", description="Starting point for the other options, like llm.preset.")
    service: str = Field("OpenAI", description="LLM provider, OpenAI or Ollama. Same meaning as in [llm].")
    model: str = Field("gpt-5.6-terra", description="Model name at the provider.")
    base_url: Optional[str] = Field(None, description="Where Ollama listens, default $OLLAMA_HOST or localhost.")
    max_tokens: int = Field(2048, ge=1, description="Maximum tokens for the answer. The answer is a short JSON.")
    reasoning_effort: Optional[str] = Field("low", description='Thinking effort, "unset" doesn\'t send it.')
    first_token_timeout: float = Field(900.0, gt=0, description="Seconds to wait for the first streamed token.")
    idle_timeout: float = Field(120.0, gt=0, description="Seconds to wait between two streamed tokens.")
    max_retries: int = Field(2, ge=0, description="Retries after timeouts or a busy provider.")
    opening_minutes: float = Field(10.0, gt=0, description="Minutes from the start of the episode the model sees. "
                                                           "People introduce themselves early.")
    lines_per_speaker: int = Field(40, ge=1, description="Lines of every speaker the model sees on top of that.")
    system_message: str = Field(SYSTEM_MESSAGE, description="Instructions, sent as a system message.")
    prompt: str = Field(PROMPT, description="Prompt, has to contain {speakers}, {opening} and {samples}.")


@register("namer", "llm-names", description="Names speakers from what is said in the transcript")
class SpeakerNamingLLM(SpeakerNamingTool):
    Options = SpeakerNamingOptions
    packages = ("langchain-core", "langchain-openai")
    _LOGGER = logging.getLogger(__name__)

    def process(self, origin_data: SpeakerNamingInput, config: Config) -> Optional[SpeakerNamingResult]:
        from langchain_core.messages import HumanMessage, SystemMessage

        from MAT.tools.summary.llm import LLM, SummaryLLM

        options = self._apply_preset(config.options(self))
        if not origin_data.speakers or not origin_data.lines:
            return SpeakerNamingResult()

        prompt = self._fill(options, origin_data)
        len_fun = SummaryLLM._get_len_fun()
        # Ollama has to be told how much context to load, the others take what the prompt brings
        num_ctx = (len_fun(options.system_message) + len_fun(prompt) + options.max_tokens + 512
                   if options.service == "Ollama" else None)
        llm = LLM.parse_str(name=options.service).get_llm(
            model=options.model, max_tokens=options.max_tokens, reasoning_effort=options.reasoning_effort,
            first_token_timeout=options.first_token_timeout, idle_timeout=options.idle_timeout,
            max_retries=options.max_retries, base_url=options.base_url, num_ctx=num_ctx,
        )
        self._LOGGER.info(f"Asking {options.model} for the names of {len(origin_data.speakers)} speakers")
        answer = llm.invoke([SystemMessage(options.system_message), HumanMessage(prompt)])
        found = self._parse(str(getattr(answer, "content", answer) or ""), known=origin_data.speakers)
        for name in found:
            self._LOGGER.info(f"{name.speaker} is called {name.name} ({name.evidence})")
        if not found:
            self._LOGGER.info("The transcript doesn't say who is who, keeping the diarizer names")
        return SpeakerNamingResult(found)

    @classmethod
    def _apply_preset(cls, options: SpeakerNamingOptions) -> SpeakerNamingOptions:
        from MAT.tools.summary.llm import PRESETS

        values = PRESETS.get(options.preset)
        if not values:
            return options
        fields = type(options).model_fields
        update = {name: value for name, value in values.items()
                  if name in fields and name not in options.model_fields_set}
        return options.model_copy(update=update) if update else options

    @staticmethod
    def _fill(options: SpeakerNamingOptions, origin_data: SpeakerNamingInput) -> str:
        opening, samples = SpeakerNamingLLM._samples(origin_data, options.opening_minutes,
                                                     options.lines_per_speaker)
        prompt = options.prompt
        for placeholder, value in (("{speakers}", ", ".join(origin_data.speakers)), ("{opening}", opening),
                                   ("{samples}", samples)):
            prompt = prompt.replace(placeholder, value)
        return prompt

    @staticmethod
    def _samples(origin_data: SpeakerNamingInput, opening_minutes: float, lines_per_speaker: int):
        """The first minutes of the episode plus some lines of every speaker, so late joiners aren't missed."""
        opening, per_speaker = [], {speaker: [] for speaker in origin_data.speakers}
        limit = opening_minutes * 60
        for line in origin_data.lines:
            start = re.search(r"\[([\d.]+) - ", line)
            if start and float(start.group(1)) <= limit:
                opening.append(line)
            speaker = line.split(" [", 1)[0]
            if speaker in per_speaker and len(per_speaker[speaker]) < lines_per_speaker:
                per_speaker[speaker].append(line)
        samples = "\n".join(line for speaker in origin_data.speakers for line in per_speaker[speaker])
        return "\n".join(opening), samples

    @classmethod
    def _parse(cls, answer: str, known: List[str]) -> List[SpeakerName]:
        """Keeps only what is usable: a known speaker, high confidence, a real name and a quoted line."""
        start, end = answer.find("{"), answer.rfind("}")
        if start < 0 or end <= start:
            cls._LOGGER.warning("The model didn't answer with JSON, no names taken from it")
            return []
        try:
            data: Dict[str, Any] = json.loads(answer[start:end + 1])
        except ValueError as e:
            cls._LOGGER.warning(f"The model's JSON is broken ({e}), no names taken from it")
            return []

        names, used = [], set()
        for entry in data.get("speakers") or []:
            if not isinstance(entry, dict):
                continue
            speaker, name = str(entry.get("id") or ""), str(entry.get("name") or "").strip()
            confidence, evidence = str(entry.get("confidence") or "").lower(), str(entry.get("evidence") or "").strip()
            if speaker not in known or not name:
                continue
            if confidence != "high":
                cls._LOGGER.info(f"Not naming {speaker} {name}, the model isn't sure ({confidence or 'no confidence'})")
                continue
            if not evidence:
                cls._LOGGER.info(f"Not naming {speaker} {name}, the model gave no line to back it up")
                continue
            if not _NAME.match(name) or len(name) > 60 or name.casefold() == speaker.casefold():
                cls._LOGGER.info(f'Not naming {speaker}, "{name}" doesn\'t look like a name')
                continue
            if name.casefold() in used:
                cls._LOGGER.warning(f"Not naming {speaker} {name}, that name is already taken by another speaker")
                continue
            used.add(name.casefold())
            names.append(SpeakerName(speaker=speaker, name=name, evidence=evidence))
        return names


__all__ = ["SpeakerNamingLLM", "SpeakerNamingOptions"]
