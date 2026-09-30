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
"""CLAP: sound events described in words ("a short jingle"), for things AudioSet has no class for."""
import logging
from typing import Dict, Optional

from pydantic import Field

from MAT.registry import register, require

require("transformers", "torch", extra="events")

from MAT.tools.events import EventInput, EventResult, EventTool, join_windows, load_audio, windows  # noqa: E402
from MAT.utils.config import Config, Options  # noqa: E402

DEFAULT_LABELS = {
    "jingle": "a short jingle or intro music",
    "music": "music playing",
    "laughter": "people laughing",
    "applause": "applause",
}
# what most of an episode sounds like. They compete with the labels but never become events.
DEFAULT_BACKGROUND = {
    "speech": "people talking in a conversation",
    "voice": "a person speaking",
    "silence": "silence",
}


class ClapOptions(Options):
    model: str = Field("laion/clap-htsat-unfused", description="CLAP model.")
    device: str = Field("auto", description='"auto" uses the GPU if there is one, or set "cpu" / "cuda".')
    labels: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_LABELS),
                                   description="Label in the result = description CLAP compares the audio with.")
    background: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_BACKGROUND),
                                       description="Descriptions of the normal sound of an episode. They take part "
                                                   "in the comparison so plain talk doesn't become an event.")
    threshold: float = Field(0.8, gt=0, lt=1, description="Minimum share of a label among all descriptions. At 0.5 a "
                                                          "3 hour episode had 189 \"jingles\", nearly all talk. "
                                                          "Above 0.8: intro, outro, one jingle and 3 laughs.")
    window: float = Field(10.0, gt=0, description="Seconds per window. CLAP reads at most 10 s.")
    hop: float = Field(5.0, gt=0, description="Seconds from one window to the next.")
    min_duration: float = Field(0.0, ge=0, description="Shorter events are dropped.")
    batch_size: int = Field(16, ge=1, description="Windows per model call.")


@register("events", "clap", description="CLAP, sound events described in words (jingle, intro music, ...)")
class EventsClap(EventTool):
    Options = ClapOptions
    packages = ("transformers",)
    memory_hint = "--set clap.batch-size=4"
    _LOGGER = logging.getLogger(__name__)

    def process(self, origin_data: EventInput, config: Config) -> Optional[EventResult]:
        import torch
        from transformers import ClapModel, ClapProcessor

        from MAT.utils.device import free_gpu_memory, resolve_device

        options = config.options(self)
        device = resolve_device(options.device)
        names = list(options.labels) + [f"background:{name}" for name in options.background]
        prompts = list(options.labels.values()) + list(options.background.values())
        processor = ClapProcessor.from_pretrained(options.model)
        model = ClapModel.from_pretrained(options.model).to(device).eval()
        try:
            with torch.no_grad():
                text = processor(text=prompts, return_tensors="pt", padding=True).to(device)
                text_features = torch.nn.functional.normalize(model.get_text_features(**text), dim=-1)
            rate = processor.feature_extractor.sampling_rate
            samples = load_audio(origin_data.input_file, rate)
            spans = windows(len(samples) / rate, options.window, options.hop)
            self._LOGGER.info(f"Looking for {', '.join(options.labels)} in {len(spans)} windows with {options.model} "
                              f"on {device}")
            scale = model.logit_scale_a.exp()
            scores = []
            for first in range(0, len(spans), options.batch_size):
                batch = [samples[int(s * rate):int(e * rate)] for s, e in spans[first:first + options.batch_size]]
                inputs = processor(audios=batch, sampling_rate=rate, return_tensors="pt").to(device)
                with torch.no_grad():
                    audio_features = torch.nn.functional.normalize(model.get_audio_features(**inputs), dim=-1)
                    shares = (scale * audio_features @ text_features.T).softmax(dim=-1).cpu().numpy()
                for row in shares:
                    scores.append({name: float(value) for name, value in zip(names, row)
                                   if not name.startswith("background:")})
        finally:
            del model
            free_gpu_memory()
        events = join_windows(spans, scores, options.threshold, options.min_duration)
        self._LOGGER.info(f"Found {len(events)} events")
        return EventResult(events)


__all__ = ["EventsClap", "ClapOptions", "DEFAULT_LABELS", "DEFAULT_BACKGROUND"]
