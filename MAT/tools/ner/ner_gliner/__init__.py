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
import logging
from typing import Optional, Dict, List, Tuple, Literal, Union
from dataclasses import dataclass

from pydantic import Field

from MAT.registry import register, require

require("gliner2", extra="gliner")

from MAT.tools.ner import NERTool, NERResult, NERInput  # noqa: E402
from MAT.utils.config import Config, Options  # noqa: E402


@dataclass
class GLiNERResult:
    text: str
    label: str
    start: int
    end: int


DEFAULT_LABELS = {
    "PERSON": "name of a person or character",
    "LOCATION": "name of a place, city or country",
    "ORGANIZATION": "name of a company, group or institution",
    "DATE": "a date, year or day",
}


class GlinerOptions(Options):
    version: Literal[1, 2] = Field(2, description="GLiNER generation. 1 needs a GLiNER v1 model.")
    model: str = Field("fastino/gliner2-multi-v1", description="GLiNER model. multi-v1 made far fewer mistakes than "
                                                             "large-v1 on German and English in our test (common "
                                                             "nouns like Frau or river as entities) and is twice as "
                                                             "fast.")
    labels: Union[List[str], Dict[str, str]] = Field(
        default_factory=lambda: dict(DEFAULT_LABELS),
        description="Entity labels to look for, as a list or as label = description. GLiNER2 uses the "
                    "descriptions: with them it stopped calling pronouns (you, I) a PERSON in our tests.")
    device: str = Field("auto", description='"auto" uses the GPU if there is one, or set "cpu" / "cuda".')
    batch_size: int = Field(8, ge=1, description="Texts per model call (GLiNER2).")
    threshold: float = Field(0.5, gt=0, lt=1, description="Minimum confidence for an entity (GLiNER2).")


@register("ner", "gliner", description="GLiNER / GLiNER2 zero shot named entities")
class NERGliner(NERTool):
    Options = GlinerOptions
    packages = ("gliner", "gliner2")
    memory_hint = "--set gliner.batch-size=2, or --set gliner.device=cpu"
    _LOGGER = logging.getLogger(__name__)

    def process(self, origin_data: NERInput, config: Config) -> Optional[NERResult]:
        from MAT.utils.device import free_gpu_memory, resolve_device

        options = config.options(self)
        device = resolve_device(options.device)
        labels = list(options.labels)
        # GLiNER2 takes {label: description}, GLiNER 1 only the labels
        asked = dict(options.labels) if isinstance(options.labels, dict) else labels
        texts = origin_data.text
        self._LOGGER.info(f"Looking for {', '.join(labels)} in {len(texts)} texts with {options.model} on {device}")
        # All labels in one call. Asked one label at a time, the model finds something for every label, so Bob and
        # Paris also came back as ORGANIZATION.
        if options.version == 1:
            from gliner import GLiNER

            model = GLiNER.from_pretrained(options.model).to(device)
            found = [[GLiNERResult(text=r["text"], start=r["start"], end=r["end"], label=r["label"])
                      for r in model.predict_entities(text, labels, threshold=options.threshold)] for text in texts]
        else:
            from gliner2 import GLiNER2

            GLiNER2._print_config = lambda *args, **kwargs: None
            model = GLiNER2.from_pretrained(options.model)
            model.to(device)
            answers = model.batch_extract_entities(texts, asked, batch_size=options.batch_size,
                                                   threshold=options.threshold, include_spans=True) if texts else []
            found = [[GLiNERResult(text=e["text"], start=e["start"], end=e["end"], label=label)
                      for label, entities in (answer.get("entities") or {}).items() for e in entities]
                     for answer in answers]
        del model
        free_gpu_memory()

        ret: List[Dict[str, List[Tuple[str, int, int]]]] = []
        for entities in found:
            per_label: Dict[str, List[Tuple[str, int, int]]] = {label: [] for label in labels}
            for e in entities:
                per_label.setdefault(e.label, []).append((e.text, e.start, e.end))
            ret.append(per_label)
        return NERResult(*ret)


__all__ = ["NERGliner"]
