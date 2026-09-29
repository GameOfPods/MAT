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

`{text}`, `{existing_answer}`, `{part}`, `{parts}` and `{additional_metadata}` are filled in by MAT.

Two ways to handle a transcript that doesn't fit into one call: refine (PROMPT for the first chunk, REFINE_PROMPT for
every further one, each call rewrites the whole summary) or map-reduce (MAP_PROMPT writes notes per chunk
independently, REDUCE_PROMPT turns the notes into the summary). On the GPU box a small local model appended new
sections after its own conclusion when refining, map-reduce doesn't give it the chance.
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

MAP_PROMPT = (
    "Here is part {part} of {parts} of a transcript:\n"
    "\n"
    '"{text}"\n'
    "\n"
    "Write notes on this part for a summary of the whole transcript: every topic, what the speakers say about it, "
    "and where they agree or disagree. Short bullet points under a heading per topic. Nothing about this being a "
    "part, no introduction and no conclusion.\n"
    "\n"
    "NOTES:"
)

REDUCE_PROMPT = (
    "Here are notes on the parts of one transcript, in order:\n"
    "\n"
    "{text}\n"
    "\n"
    "Write the summary of the whole transcript from them:\n"
    "- Join what belongs together, also across parts, and say nothing twice.\n"
    "- Give every part the room its content needs, the first one no more than the last.\n"
    "- Don't mention the parts or the notes.\n"
    "\n"
    "SUMMARY:"
)

# Book chapters. Same shape as the transcript prompts, but about a written story, and without spoilers: a chapter
# summary must not know what happens later.
CHAPTER_SYSTEM_MESSAGE = (
    "You summarize chapters of books.\n"
    "\n"
    "Rules:\n"
    "- Use only what the chapter says. Nothing from later chapters, other books, films or series, even when you know "
    "the book. A reader who is at this chapter must not learn anything new about the rest of the story.\n"
    "- Write in the language of the chapter.\n"
    "- Answer in Markdown: short paragraphs, no heading for the whole summary.\n"
    "- Write no sentence about the text being a summary, no introduction of yourself and no closing remark.\n"
    "- Keep names the way the chapter writes them."
)

CHAPTER_PROMPT = (
    "Here is a chapter:\n"
    "\n"
    '"{text}"\n'
    "\n"
    "Summarize it: what happens, in order, who takes part, and what changes for them. Stay much shorter than the "
    "chapter.\n"
    "\n"
    "SUMMARY:"
)

CHAPTER_REFINE_PROMPT = (
    "You are extending the summary of a long chapter.\n"
    "\n"
    "The summary so far:\n"
    "{existing_answer}\n"
    "\n"
    "The next part of the chapter:\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "\n"
    "Return the complete summary with this part added in the order of events. Don't repeat what it already says "
    "and don't mention that it was extended.\n"
    "\n"
    "SUMMARY:"
)

CHAPTER_MAP_PROMPT = (
    "Here is part {part} of {parts} of a chapter:\n"
    "\n"
    '"{text}"\n'
    "\n"
    "Write notes on this part for a summary of the whole chapter: what happens, in order, and who takes part. Short "
    "bullet points, nothing about this being a part.\n"
    "\n"
    "NOTES:"
)

CHAPTER_REDUCE_PROMPT = (
    "Here are notes on the parts of one chapter, in order:\n"
    "\n"
    "{text}\n"
    "\n"
    "Write the summary of the whole chapter from them, in the order of events, saying nothing twice and without "
    "mentioning the parts or the notes.\n"
    "\n"
    "SUMMARY:"
)

# option name -> prompt, what a chapter summary uses instead of the transcript prompts
CHAPTER_PROMPTS = {
    "system_message": CHAPTER_SYSTEM_MESSAGE,
    "prompt": CHAPTER_PROMPT,
    "prompt_refine": CHAPTER_REFINE_PROMPT,
    "prompt_map": CHAPTER_MAP_PROMPT,
    "prompt_reduce": CHAPTER_REDUCE_PROMPT,
}

__all__ = ["SYSTEM_MESSAGE", "PROMPT", "REFINE_PROMPT", "MAP_PROMPT", "REDUCE_PROMPT", "CHAPTER_PROMPTS"]
