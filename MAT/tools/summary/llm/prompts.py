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

try:
    from langchain.chains.summarize.refine_prompts import PROMPT, REFINE_PROMPT
except ImportError as e:
    from langchain_classic.chains.summarize.refine_prompts import PROMPT, REFINE_PROMPT

PROMPT = PROMPT.template
REFINE_PROMPT = REFINE_PROMPT.template

SYSTEM_MESSAGE = ("You are a helpful assistant that helps to create precise and complete summaries. "
                  "Always keep your answers precise and only use what you either find in a prompt or in the source "
                  "material that is to be summarized. You may use contextual information you know on the source "
                  "material.\n"
                  "Only return the requested summary without any introduction of yourself or other system messages.\n"
                  "Dont include personal opinions on the content.\n"
                  "Always start the summary with a heading, an introduction of the text and it being a summary.\n"
                  "If the prompt contains an already existing summary of the text extend the existing introduction. "
                  "Extend it with the new information. Then return the new summary with the old and new information "
                  "added together. Always take the previous information into account. If it is an extension "
                  "of an existing summary dont mention that and just mention it beeing a summary.\n"
                  "Always return the summary in valid Markdown format. Try to return the summary in the language of the"
                  "original content.")
