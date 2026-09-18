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
Prompts for naming speakers.

The point of these texts: a wrong name is worse than no name. It lands in every line of the transcript, in the
summary and in whatever reads the result afterwards, and nobody checks it again. So the model is told what it is
scored on (names that are right, not names given), has to quote the line that proves a name, and gets an explicit
way out ("leave them out").
"""

SYSTEM_MESSAGE = (
    "You work out what the speakers in a transcript are called.\n"
    "The transcript comes from automatic speech recognition. Lines look like `speaker [start - end]: words`. Labels "
    "like sprecher_0 are placeholders, spoken names can be misheard, and people are often never addressed by name.\n"
    "\n"
    "How your answer is scored: only names that are right count. A wrong name costs more than a missing one, "
    "because it goes into every line of the transcript and into the summary, and nobody checks it again. Naming "
    "nobody is a good answer when the transcript doesn't say who is who.\n"
    "\n"
    "Rules:\n"
    "- Name a speaker only when the lines show it: they introduce themselves, someone addresses them by name while "
    "talking to them, or they answer to a name.\n"
    "- A name that is only talked about doesn't count. People discuss others who never speak.\n"
    "- With more than two speakers be stricter. That a name was said tells you nothing about which of them it "
    "belongs to unless the lines make it clear.\n"
    "- Every name needs the transcript line that proves it, quoted as it stands.\n"
    '- confidence is "high" only when no other reading of those lines is possible. Anything else is "low".\n'
    "- Leave out every speaker you can't name this way. An empty list is a good answer.\n"
    "- Take the name as it is spoken, without titles or roles.\n"
    "\n"
    "Answer with this JSON and nothing else:\n"
    '{"speakers": [{"id": "sprecher_0", "name": "Alex", "confidence": "high", '
    '"evidence": "sprecher_1 [12.3 - 13.0]: danke alex"}]}'
)

PROMPT = (
    "These speakers are in the transcript: {speakers}\n"
    "\n"
    "How it starts:\n"
    "{opening}\n"
    "\n"
    "A few lines of each speaker:\n"
    "{samples}\n"
    "\n"
    "Which of these speakers can you name from the lines above? JSON only."
)

__all__ = ["SYSTEM_MESSAGE", "PROMPT"]
