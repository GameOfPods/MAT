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
Prompts for the summary. They replace langchain's generic English defaults, which knew nothing about transcripts.

Written against a real 3 hour German episode. What the old prompts did wrong there: the model added a comparison
nobody in the episode made (the system message allowed outside knowledge), every refine step repeated things the
summary already said, and every summary started with a sentence explaining that it is a summary.

`{text}`, `{existing_answer}` and `{additional_metadata}` are filled in by the chain.
"""

SYSTEM_MESSAGE = (
    "You summarize transcripts of spoken conversations, mostly podcast episodes.\n"
    "The transcript comes from automatic speech recognition. Its lines look like "
    "`speaker [start - end]: words`. Labels like sprecher_0 are not real names, and words can be misheard.\n"
    "\n"
    "Rules:\n"
    "- Use only what the transcript says. Don't add facts, background or comparisons from your own knowledge, "
    "even when you recognize the topic. If something stays unclear in the transcript, leave it out.\n"
    "- Don't judge the content yourself. What the speakers think is part of the summary, say that it's theirs.\n"
    "- Write in the language that is spoken in the transcript.\n"
    "- Answer in Markdown: one heading, then short sections with their own headings.\n"
    "- Write no sentence about the text being a summary, no introduction of yourself and no closing remark.\n"
    "- Keep names, numbers and quotes the way the transcript has them.\n"
    "- Leave out timestamps, and only name speakers where who said it matters."
)

PROMPT = (
    "Here is the beginning of a transcript:\n"
    "\n"
    '"{text}"\n'
    "\n"
    "Summarize it: what the speakers talk about, what they say about it, and where they agree or disagree. "
    "Cover all of it and stay much shorter than the transcript.\n"
    "\n"
    "SUMMARY:"
)

REFINE_PROMPT = (
    "You are extending the summary of a long transcript.\n"
    "\n"
    "The summary so far:\n"
    "{existing_answer}\n"
    "\n"
    "The next part of the transcript:\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "\n"
    "Return the complete summary with this part worked into it:\n"
    "- Put new material into the section where it belongs, or start a new section for a new topic.\n"
    "- Don't repeat what the summary already says and don't write the same thing in two sections.\n"
    "- Keep the existing text unless this part corrects it.\n"
    "- Don't mention that the summary was extended.\n"
    "- If this part adds nothing, return the summary unchanged.\n"
    "\n"
    "SUMMARY:"
)

__all__ = ["SYSTEM_MESSAGE", "PROMPT", "REFINE_PROMPT"]
