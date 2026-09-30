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
"""SaT (Segment any Text, wtpsplit): sentence borders without relying on punctuation or casing."""
import logging
from typing import Optional

from pydantic import Field

from MAT.registry import register, require

require("wtpsplit", extra="sentences")

from MAT.tools.sentences import SentenceInput, SentenceResult, SentenceTool  # noqa: E402
from MAT.utils.config import Config, Options  # noqa: E402


class SatOptions(Options):
    model: str = Field("sat-3l-sm", description="SaT model. The -sm models were trained to split text without "
                                                "punctuation and casing, which is what ASR output often is.")
    device: str = Field("auto", description='"auto" uses the GPU if there is one, or set "cpu" / "cuda".')
    threshold: Optional[float] = Field(None, gt=0, lt=1, description="Probability for a sentence border. Not set: "
                                                                     "the model's own default. Lower gives more, "
                                                                     "shorter sentences.")


@register("sentences", "sat", description="SaT (wtpsplit), sentences without relying on punctuation")
class SentencesSaT(SentenceTool):
    Options = SatOptions
    packages = ("wtpsplit",)
    memory_hint = "--set sat.device=cpu, the small models are quick there too"
    _LOGGER = logging.getLogger(__name__)

    def process(self, origin_data: SentenceInput, config: Config) -> Optional[SentenceResult]:
        from wtpsplit import SaT

        from MAT.utils.device import free_gpu_memory, resolve_device

        options = config.options(self)
        if not origin_data.texts:
            return SentenceResult([])
        device = resolve_device(options.device)
        model = SaT(options.model)
        if device != "cpu":
            # float32: the model is small, and Pascal cards gain nothing from half precision here
            model.to(device)
        try:
            kwargs = {} if options.threshold is None else {"threshold": options.threshold}
            sentences = [list(pieces) for pieces in model.split(origin_data.texts, **kwargs)]
        finally:
            del model
            free_gpu_memory()
        self._LOGGER.info(f"{sum(len(s) for s in sentences)} sentences in {len(sentences)} speaker turns")
        return SentenceResult(sentences)


__all__ = ["SentencesSaT", "SatOptions"]
