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
from typing import Any, Dict, List, Optional

from pydantic import Field

from MAT.registry import register, require

require("langchain_core", "langchain_openai", "tiktoken", extra="llm")

from MAT.tools.speakernaming import (  # noqa: E402
    SpeakerName, SpeakerNamingInput, SpeakerNamingResult, SpeakerNamingTool,
)
from MAT.tools.speakernaming.llm.prompts import PROMPT, SYSTEM_MESSAGE  # noqa: E402
from MAT.tools.summary.llm.task import INHERITED, LLMTask, LLMTaskOptions  # noqa: E402,F401
from MAT.utils.config import Config  # noqa: E402

# what the model has to answer, for servers that can enforce it (see structured_output)
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"speakers": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "string"}, "name": {"type": "string"},
                       "confidence": {"type": "string", "enum": ["high", "low"]}, "evidence": {"type": "string"}},
        "required": ["id", "name", "confidence", "evidence"], "additionalProperties": False}}},
    "required": ["speakers"], "additionalProperties": False,
}


def quoted_in(quote: str, text: str) -> bool:
    """Whether a quote is really in the text. Case, spaces and punctuation don't count, and a quote that
    starts with the line's label ("sprecher_1 [12.3 - 13.0]: ...") or leaves out the start of the line is fine.
    Parts joined with "--" or "..." have to be there each."""
    def simple(value: str) -> str:
        return " ".join(re.sub(r"[^\w\s]", " ", value.casefold()).split())

    haystack = simple(text)
    parts = [p for p in re.split(r"\s*(?:--|\.\.\.|…)\s*", quote) if p.strip()]
    if not parts:
        return False
    for part in parts:
        # drop a leading "speaker [start - end]:" so the check is about what was said
        part = re.sub(r"^\s*\S+(?:\s*&\s*\S+)*\s*\[[^\]]*\]\s*:\s*", "", part)
        if simple(part) and simple(part) not in haystack:
            return False
    return True


# A name we accept: letters, optionally a second part after a space, hyphen or apostrophe. No digits and no
# underscores, so a model that echoes "sprecher_0" back at us doesn't get through.
_NAME = re.compile(r"^[^\W\d_]+(?:[ '’\-][^\W\d_]+)*$", re.UNICODE)


class SpeakerNamingOptions(LLMTaskOptions):
    opening_minutes: float = Field(10.0, gt=0, description="Minutes from the start of the episode the model sees. "
                                                           "People introduce themselves early.")
    lines_per_speaker: int = Field(40, ge=1, description="Lines of every speaker the model sees on top of that.")
    system_message: str = Field(SYSTEM_MESSAGE, description="Instructions, sent as a system message.")
    prompt: str = Field(PROMPT, description="Prompt, has to contain {speakers}, {opening} and {samples}.")


@register("namer", "llm-names", description="Names speakers from what is said in the transcript")
class SpeakerNamingLLM(LLMTask, SpeakerNamingTool):
    TASK = "Speaker naming"
    SKIP = "--namer none"
    Options = SpeakerNamingOptions
    packages = ("langchain-core", "langchain-openai")
    _LOGGER = logging.getLogger(__name__)

    def process(self, origin_data: SpeakerNamingInput, config: Config) -> Optional[SpeakerNamingResult]:
        from langchain_core.messages import HumanMessage, SystemMessage

        from MAT.tools.summary.llm import SummaryLLM

        options = self._apply_preset(self._inherit(config.options(self), config))
        if not origin_data.speakers or not origin_data.lines:
            return SpeakerNamingResult()

        prompt = self._fill(options, origin_data)
        len_fun = SummaryLLM._get_len_fun()
        # Ollama has to be told how much context to load, the others take what the prompt brings
        num_ctx = (len_fun(options.system_message) + len_fun(prompt) + options.max_tokens + 512
                   if options.service == "Ollama" else None)
        llm = self.client(options, schema=ANSWER_SCHEMA, num_ctx=num_ctx)
        self._LOGGER.info(f"Asking {options.model} for the names of {len(origin_data.speakers)} speakers")
        answer = llm.invoke([SystemMessage(options.system_message), HumanMessage(prompt)])
        found = self._parse(str(getattr(answer, "content", answer) or ""), known=origin_data.speakers,
                            lines=prompt)
        for name in found:
            self._LOGGER.info(f"{name.speaker} is called {name.name} ({name.evidence})")
        if not found:
            self._LOGGER.info("The transcript doesn't say who is who, keeping the diarizer names")
        return SpeakerNamingResult(found)

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
    def _parse(cls, answer: str, known: List[str], lines: Optional[str] = None) -> List[SpeakerName]:
        """Keeps only what is usable: a known speaker, high confidence, a real name and a quoted line. With lines
        (what the model was shown) the quote has to be in there, an invented line doesn't count."""
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
            if lines is not None and not quoted_in(evidence, lines):
                cls._LOGGER.info(f"Not naming {speaker} {name}, the quoted line isn't in the transcript: {evidence}")
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
