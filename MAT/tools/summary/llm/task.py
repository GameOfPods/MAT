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
What every small LLM job (speaker names, character names) shares with the summary: its own config section that
follows [llm] until it has a preset of its own, the presets, the access check before a run, and a client that can be
told to answer in JSON.
"""
from typing import Literal, Optional

from pydantic import Field

from MAT.utils.config import Config, Options

# what a task takes from [llm] while it has no preset of its own
INHERITED = ("preset", "service", "model", "base_url")


class LLMTaskOptions(Options):
    preset: Literal["none", "openai", "ollama", "llamacpp"] = Field(
        "none", description="Starting point for the other options, like llm.preset. While it isn't set, preset, "
                            "service, model and base-url come from [llm] wherever you set them there. Once it's "
                            "set (even to the same value) this section stands on its own.")
    service: str = Field("OpenAI", description="LLM provider, OpenAI or Ollama. Same meaning as in [llm].")
    model: str = Field("gpt-5.6-terra", description="Model name at the provider.")
    base_url: Optional[str] = Field(None, description="Where Ollama listens, default $OLLAMA_HOST or localhost.")
    max_tokens: int = Field(2048, ge=1, description="Maximum tokens for the answer. The answer is a short JSON.")
    reasoning_effort: Optional[str] = Field("low", description='Thinking effort, "unset" doesn\'t send it.')
    first_token_timeout: float = Field(900.0, gt=0, description="Seconds to wait for the first streamed token.")
    idle_timeout: float = Field(120.0, gt=0, description="Seconds to wait between two streamed tokens.")
    max_retries: int = Field(2, ge=0, description="Retries after timeouts or a busy provider.")
    structured: Literal["auto", "schema", "json", "off"] = Field(
        "auto", description='Make the model answer in JSON: "schema" lets the server enforce the answer format '
                            '(Ollama, OpenAI), "json" only asks for JSON (DeepSeek), "off" relies on the prompt. '
                            'auto picks by server.')


class LLMTask:
    """Mixin for a backend whose Options derive from LLMTaskOptions. TASK names the job in logs and errors, SKIP says
    how to run without it."""
    TASK = "An LLM task"
    SKIP = ""

    @classmethod
    def effective_options(cls, config: Config, log: bool = False):
        return cls._apply_preset(cls._inherit(config.options(cls), config, log=log))

    @classmethod
    def preflight(cls, config: Config) -> None:
        from MAT.tools.summary.llm import check_llm_access

        # the one place that says where the settings come from, the steps and the cache ask again later
        options = cls.effective_options(config, log=True)
        check_llm_access(options.service, options.model, options.base_url, cls.TASK, cls.SKIP)

    def describe(self, config: Config):
        # what the task really talks to, not the section's defaults (a run with llm.preset=ollama wrote gpt-5.6-terra)
        info = super().describe(config)
        options = self.effective_options(config)
        info.update(model=options.model, service=options.service)
        if options.base_url:
            info["api_base"] = options.base_url
        return info

    @classmethod
    def _inherit(cls, options, config: Config, log: bool = False):
        """Without a preset of its own the task talks to the same model as the summary: whatever of preset, service,
        model and base-url you set in [llm] and not here. With its own preset nothing comes from [llm], so a cloud
        model name can't end up at Ollama."""
        from MAT.tools.summary.llm import SummaryLLM

        if "preset" in options.model_fields_set:
            if log:
                cls._LOGGER.info(f"{cls.TASK} uses its own settings ({cls.section}.preset = {options.preset})")
            return options
        summary = config.options(SummaryLLM)
        taken = {name: getattr(summary, name) for name in INHERITED
                 if name in summary.model_fields_set and name not in options.model_fields_set}
        if not taken:
            return options
        if log:
            cls._LOGGER.info(f"{cls.TASK} takes " + ", ".join(f"{name.replace('_', '-')}={value}"
                                                              for name, value in sorted(taken.items())) + " from [llm]")
        return options.model_copy(update=taken)

    @classmethod
    def _apply_preset(cls, options):
        from MAT.tools.summary.llm import PRESETS

        values = PRESETS.get(options.preset)
        if not values:
            return options
        fields = type(options).model_fields
        update = {name: value for name, value in values.items()
                  if name in fields and name not in options.model_fields_set}
        return options.model_copy(update=update) if update else options

    @staticmethod
    def client(options, schema: Optional[dict] = None, num_ctx: Optional[int] = None):
        from MAT.tools.summary.llm import LLM, structured_output

        return LLM.parse_str(name=options.service).get_llm(
            model=options.model, max_tokens=options.max_tokens, reasoning_effort=options.reasoning_effort,
            first_token_timeout=options.first_token_timeout, idle_timeout=options.idle_timeout,
            max_retries=options.max_retries, base_url=options.base_url, num_ctx=num_ctx,
            schema=schema, structured=structured_output(options.service, options.structured))


__all__ = ["INHERITED", "LLMTaskOptions", "LLMTask"]
