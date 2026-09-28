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
"""AudioSet tagging with AST: fixed sound classes like music, laughter and applause."""
import logging
from typing import Dict, List, Optional

from pydantic import Field

from MAT.registry import register, require

require("transformers", "torch", extra="events")

from MAT.tools.events import EventInput, EventResult, EventTool, join_windows, load_audio, windows  # noqa: E402
from MAT.utils.config import Config, Options  # noqa: E402

# our label -> AudioSet class names that count for it. The score of a label is the highest of its classes.
DEFAULT_GROUPS = {
    "music": ["Music", "Song", "Musical instrument"],
    "laughter": ["Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle"],
    "applause": ["Applause", "Clapping", "Cheering"],
}


class AudioSetOptions(Options):
    model: str = Field("MIT/ast-finetuned-audioset-10-10-0.4593", description="AudioSet classifier (AST).")
    device: str = Field("auto", description='"auto" uses the GPU if there is one, or set "cpu" / "cuda".')
    groups: Dict[str, List[str]] = Field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_GROUPS.items()},
                                         description="Label in the result = AudioSet classes that count for it.")
    threshold: float = Field(0.3, gt=0, lt=1, description="Minimum score of a window for its label.")
    window: float = Field(10.0, gt=0, description="Seconds per window. AST was trained on 10 s clips.")
    hop: float = Field(5.0, gt=0, description="Seconds from one window to the next. Smaller finds edges better "
                                              "and takes longer.")
    min_duration: float = Field(0.0, ge=0, description="Shorter events are dropped.")
    batch_size: int = Field(16, ge=1, description="Windows per model call.")


@register("events", "audioset", description="AudioSet classes (AST): music, laughter, applause")
class EventsAudioSet(EventTool):
    Options = AudioSetOptions
    packages = ("transformers",)
    memory_hint = "--set audioset.batch-size=4"
    _LOGGER = logging.getLogger(__name__)

    def process(self, origin_data: EventInput, config: Config) -> Optional[EventResult]:
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        from MAT.utils.device import free_gpu_memory, resolve_device

        options = config.options(self)
        device = resolve_device(options.device)
        extractor = ASTFeatureExtractor.from_pretrained(options.model)
        model = ASTForAudioClassification.from_pretrained(options.model).to(device).eval()
        try:
            names = {name: index for index, name in model.config.id2label.items()}
            groups = {}
            for label, classes in options.groups.items():
                known = [names[c] for c in classes if c in names]
                missing = [c for c in classes if c not in names]
                if missing:
                    self._LOGGER.warning(f"{label}: {', '.join(missing)} aren't AudioSet classes of {options.model}")
                if known:
                    groups[label] = known

            rate = extractor.sampling_rate
            samples = load_audio(origin_data.input_file, rate)
            spans = windows(len(samples) / rate, options.window, options.hop)
            self._LOGGER.info(f"Looking for {', '.join(groups)} in {len(spans)} windows with {options.model} "
                              f"on {device}")
            scores = []
            for first in range(0, len(spans), options.batch_size):
                batch = [samples[int(s * rate):int(e * rate)] for s, e in spans[first:first + options.batch_size]]
                inputs = extractor(batch, sampling_rate=rate, return_tensors="pt").to(device)
                with torch.no_grad():
                    probabilities = torch.sigmoid(model(**inputs).logits).cpu().numpy()
                for row in probabilities:
                    scores.append({label: float(max(row[i] for i in indexes)) for label, indexes in groups.items()})
        finally:
            del model
            free_gpu_memory()
        events = join_windows(spans, scores, options.threshold, options.min_duration)
        self._LOGGER.info(f"Found {len(events)} events")
        return EventResult(events)


__all__ = ["EventsAudioSet", "AudioSetOptions", "DEFAULT_GROUPS"]
