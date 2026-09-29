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
Prompts for llm-characters. The model is scored on correct answers only, a wrong "same" merges two people in every
count that follows, so "unsure" has to be the easy way out. No outside knowledge: "Reek is Theon" is exactly what a
model knows from training, and a spoiler for someone reading that chapter.
"""

SYSTEM_MESSAGE = (
    "You decide questions about the characters of a book, only from the sentences you are given.\n"
    "\n"
    "You are scored only on correct decisions. A wrong answer is the worst outcome: it merges two people, or gives "
    "one person's lines to another, in every count that follows. \"unsure\" costs nothing. When in doubt, answer "
    "\"unsure\".\n"
    "\n"
    "Rules:\n"
    "- Decide only from the quoted sentences. Don't use what you know about the book, its sequels or adaptations. "
    "If the sentences don't say it, answer \"unsure\", even if you are sure from memory.\n"
    "- Two people with the same family name are not the same person.\n"
    "- Answer with the JSON object that is asked for, nothing else."
)

PAIR_PROMPT = (
    "The book is written in {language}.\n"
    "\n"
    "For every pair: are the two names the same character?\n"
    "\"same\" needs a sentence that shows it: both names for one person in one sentence, or a sentence that says one "
    "is the other's name or nickname. Quote that sentence exactly in evidence. Without such a sentence the answer is "
    "\"unsure\" or \"different\".\n"
    "\n"
    "{pairs}\n"
    "\n"
    'JSON: {"pairs": [{"pair": <number>, "answer": "same" | "different" | "unsure", '
    '"evidence": "<quoted sentence or empty>"}]}'
)

MENTION_PROMPT = (
    "The book is written in {language}.\n"
    "\n"
    "Each item is one sentence with a name that fits more than one character. Which character does the name mean in "
    "that sentence? Answer with one of the listed characters, exactly as written, or \"unsure\". Pick a character only "
    "if the sentence leaves no other reading.\n"
    "\n"
    "{mentions}\n"
    "\n"
    'JSON: {"mentions": [{"id": <number>, "answer": "<one of the characters>" | "unsure"}]}'
)

__all__ = ["SYSTEM_MESSAGE", "PAIR_PROMPT", "MENTION_PROMPT"]
