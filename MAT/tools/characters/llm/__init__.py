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
llm-characters: an LLM judges the candidates of the character list.

Every question is asked twice, the second time with the names (or the characters to choose from) in the other
order. An answer only counts when both agree, and a "same" only counts with a sentence that really is in what the
model was shown.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from pydantic import Field

from MAT.registry import register, require

require("langchain_core", "langchain_openai", extra="llm")

from MAT.tools.characters import (  # noqa: E402
    CharacterJudgeInput, CharacterJudgeResult, CharacterJudgeTool, MentionQuestion, PairQuestion,
)
from MAT.tools.characters.llm.prompts import MENTION_PROMPT, PAIR_PROMPT, SYSTEM_MESSAGE  # noqa: E402
from MAT.tools.summary.llm.task import LLMTask, LLMTaskOptions  # noqa: E402
from MAT.utils.config import Config  # noqa: E402

LANGUAGES = {"de": "German", "en": "English", "fr": "French"}

PAIR_SCHEMA = {
    "type": "object",
    "properties": {"pairs": {"type": "array", "items": {
        "type": "object",
        "properties": {"pair": {"type": "integer"},
                       "answer": {"type": "string", "enum": ["same", "different", "unsure"]},
                       "evidence": {"type": "string"}},
        "required": ["pair", "answer", "evidence"], "additionalProperties": False}}},
    "required": ["pairs"], "additionalProperties": False,
}
MENTION_SCHEMA = {
    "type": "object",
    "properties": {"mentions": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "answer": {"type": "string"}},
        "required": ["id", "answer"], "additionalProperties": False}}},
    "required": ["mentions"], "additionalProperties": False,
}


class CharacterJudgeOptions(LLMTaskOptions):
    batch_size: int = Field(20, ge=1, description="Questions per call.")
    system_message: str = Field(SYSTEM_MESSAGE, description="Instructions, sent as a system message.")
    pair_prompt: str = Field(PAIR_PROMPT, description="Prompt for name pairs, has {language} and {pairs}.")
    mention_prompt: str = Field(MENTION_PROMPT, description="Prompt for ambiguous names, has {language} and "
                                                            "{mentions}.")


@register("characters", "llm-characters", description="Decides with an LLM which names of a book are one character")
class CharacterJudgeLLM(LLMTask, CharacterJudgeTool):
    Options = CharacterJudgeOptions
    packages = ("langchain-core", "langchain-openai")
    TASK = "Character judging"
    SKIP = "--character-judge none"
    _LOGGER = logging.getLogger(__name__)

    def process(self, origin_data: CharacterJudgeInput, config: Config) -> Optional[CharacterJudgeResult]:
        options = self.effective_options(config)
        language = LANGUAGES.get((origin_data.language or "")[:2], origin_data.language or "an unknown language")
        pair_llm = self.client(options, schema=PAIR_SCHEMA) if origin_data.pairs else None
        mention_llm = self.client(options, schema=MENTION_SCHEMA) if origin_data.mentions else None
        same: Dict[int, str] = {}
        for batch in _batches(origin_data.pairs, options.batch_size):
            first = self._ask(pair_llm, options, options.pair_prompt, language, "{pairs}", _pairs_text(batch, False))
            second = self._ask(pair_llm, options, options.pair_prompt, language, "{pairs}", _pairs_text(batch, True))
            same.update(self._agreed_pairs(batch, first, second))
        chosen: Dict[int, str] = {}
        for batch in _batches(origin_data.mentions, options.batch_size):
            first = self._ask(mention_llm, options, options.mention_prompt, language, "{mentions}",
                              _mentions_text(batch, False))
            second = self._ask(mention_llm, options, options.mention_prompt, language, "{mentions}",
                               _mentions_text(batch, True))
            chosen.update(self._agreed_mentions(batch, first, second))
        self._LOGGER.info(f"{len(same)} of {len(origin_data.pairs)} name pairs are one character, "
                          f"{len(chosen)} of {len(origin_data.mentions)} ambiguous names decided")
        return CharacterJudgeResult(same=same, mentions=chosen)

    def _ask(self, llm, options, template: str, language: str, placeholder: str, items: str) -> Dict[str, Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = template.replace("{language}", language).replace(placeholder, items)
        try:
            answer = llm.invoke([SystemMessage(options.system_message), HumanMessage(prompt)])
        except Exception as e:
            self._LOGGER.warning(f"The model didn't answer ({e.__class__.__name__}: {e}), these questions stay open")
            return {}
        text = str(getattr(answer, "content", answer) or "")
        start, end = text.find("{"), text.rfind("}")
        try:
            return json.loads(text[start:end + 1]) if 0 <= start < end else {}
        except ValueError:
            self._LOGGER.warning("The model's JSON is broken, these questions stay open")
            return {}

    @classmethod
    def _agreed_pairs(cls, batch: Sequence[PairQuestion], first: Dict, second: Dict) -> Dict[int, str]:
        from MAT.tools.speakernaming.llm import quoted_in

        def answers(data: Dict) -> Dict[int, Dict]:
            result = {}
            for entry in data.get("pairs") or []:
                if isinstance(entry, dict) and isinstance(entry.get("pair"), int):
                    result[entry["pair"]] = entry
            return result

        one, two = answers(first), answers(second)
        agreed = {}
        for question in batch:
            a, b = one.get(question.id, {}), two.get(question.id, {})
            if a.get("answer") != "same" or b.get("answer") != "same":
                continue
            evidence = str(a.get("evidence") or b.get("evidence") or "").strip()
            shown = "\n".join(question.sentences_a + question.sentences_b + question.together)
            if not evidence or not quoted_in(evidence, shown):
                cls._LOGGER.info(f"Not joining {question.a} and {question.b}, no sentence backs it up")
                continue
            cls._LOGGER.info(f"{question.a} and {question.b} are one character: {evidence}")
            agreed[question.id] = evidence
        return agreed

    @classmethod
    def _agreed_mentions(cls, batch: Sequence[MentionQuestion], first: Dict, second: Dict) -> Dict[int, str]:
        def answers(data: Dict) -> Dict[int, str]:
            return {entry["id"]: str(entry.get("answer") or "").strip() for entry in data.get("mentions") or []
                    if isinstance(entry, dict) and isinstance(entry.get("id"), int)}

        one, two = answers(first), answers(second)
        agreed = {}
        for question in batch:
            answer = one.get(question.id)
            if answer and answer == two.get(question.id) and answer in question.options:
                agreed[question.id] = answer
        return agreed


def _batches(items: Sequence, size: int) -> List[Sequence]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _quote(sentences: Sequence[str]) -> str:
    return "\n".join(f'  - "{s}"' for s in sentences) or "  (none)"


def _pairs_text(batch: Sequence[PairQuestion], swapped: bool) -> str:
    parts = []
    for q in batch:
        a, b, sa, sb = (q.b, q.a, q.sentences_b, q.sentences_a) if swapped else (q.a, q.b, q.sentences_a, q.sentences_b)
        parts.append(f'Pair {q.id}: "{a}" and "{b}"\nSentences with "{a}":\n{_quote(sa)}\n'
                     f'Sentences with "{b}":\n{_quote(sb)}\nSentences with both:\n{_quote(q.together)}')
    return "\n\n".join(parts)


def _mentions_text(batch: Sequence[MentionQuestion], swapped: bool) -> str:
    parts = []
    for q in batch:
        options = list(reversed(q.options)) if swapped else list(q.options)
        parts.append(f'{q.id}: "{q.sentence}"\n   name: "{q.name}", characters: '
                     + ", ".join(f'"{o}"' for o in options))
    return "\n\n".join(parts)


__all__ = ["CharacterJudgeLLM", "CharacterJudgeOptions", "PAIR_SCHEMA", "MENTION_SCHEMA"]
