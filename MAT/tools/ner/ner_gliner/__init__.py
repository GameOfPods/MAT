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
from typing import Optional, Dict, List, Tuple, Callable, Literal, Union
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


class GlinerOptions(Options):
    version: Literal[1, 2] = Field(2, description="GLiNER generation. 1 needs a GLiNER v1 model.")
    model: str = Field("fastino/gliner2-large-v1", description="GLiNER model.")
    labels: List[str] = Field(default_factory=lambda: ["PERSON", "LOCATION", "ORGANIZATION", "DATE"],
                              description="Entity labels to look for.")


@register("ner", "gliner", description="GLiNER / GLiNER2 zero shot named entities")
class NERGliner(NERTool):
    Options = GlinerOptions
    packages = ("gliner", "gliner2")
    _LOGGER = logging.getLogger(__name__)

    def process(self, origin_data: NERInput, config: Config) -> Optional[NERResult]:
        import tqdm
        model: Union["GLiNER", "GLiNER2"] = None
        get_entities: Callable[[str, List[str]], List[GLiNERResult]]
        options = config.options(self)
        match options.version:
            case 1:
                from gliner import GLiNER
                model = GLiNER.from_pretrained(options.model)

                def get_entities(_txt: str, _labels: List[str]) -> List[GLiNERResult]:
                    _result = model.predict_entities(_txt, _labels)
                    _ret = []
                    for _r in _result:
                        _ret.append(GLiNERResult(text=_r["text"], start=_r["start"], end=_r["end"], label=_r["label"]))
                    return _ret

            case 2:
                from gliner2 import GLiNER2
                GLiNER2._print_config = lambda *args, **kwargs: None
                model = GLiNER2.from_pretrained(options.model)

                def get_entities(_txt: str, _labels: List[str]) -> List[GLiNERResult]:
                    _result = model.extract_entities(_txt, _labels, include_spans=True)
                    _ret = []
                    for _label, _entities in _result["entities"].items():
                        for _e in _entities:
                            _ret.append(GLiNERResult(text=_e["text"], start=_e["start"], end=_e["end"], label=_label))
                    return _ret

        labels = options.labels
        self.__class__._LOGGER.debug(f"Running GliNER{options.version}-{options.model} with labels: {', '.join(labels)}")

        ret: List[Dict[str, List[Tuple[str, int, int]]]] = []
        for txt in tqdm.tqdm(origin_data.text, leave=False, desc="NER on sentence", unit="sentences"):
            ret.append({})
            for label in labels:
                ret[-1][label] = []
            for label in labels:
                result = get_entities(_txt=txt, _labels=[label])
                for r in result:
                    ret[-1][r.label].append((r.text, r.start, r.end))

        if model is not None:
            del model

        return NERResult(*ret)


__all__ = ["NERGliner"]
