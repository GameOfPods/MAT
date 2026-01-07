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
from typing import Optional, Dict, List, Tuple, Callable, Union
from dataclasses import dataclass

from MAT.tools.ner import NERTool, NERResult, NERInput
from MAT.utils.config import ConfigElement, Config


@dataclass
class GLiNERResult:
    text: str
    label: str
    start: int
    end: int


class NERGliner(NERTool):
    @classmethod
    def config_name(cls) -> str:
        return "GliNER"

    @classmethod
    def config_keys(cls) -> Dict[str, ConfigElement]:
        return {
            "version": ConfigElement(
                default_value=2,
                argparse_kwargs={
                    "help": "GliNER version to use. Choose from %(choices)s [Default: %(default)s]",
                    "type": int, "choices": [1, 2]
                }
            ),
            "model": ConfigElement(
                default_value="fastino/gliner2-large-v1",
                argparse_kwargs={
                    "help": "GliNER model to use [Default: %(default)s]",
                    "type": str,
                }
            ),
            "labels": ConfigElement(
                default_value=["PERSON", "LOCATION", "ORGANIZATION", "DATE"],
                argparse_kwargs={
                    "help": "GliNER labels to use [Default: %(default)s]",
                    "type": str, "nargs": "+",
                }
            ),
        }

    _LOGGER = logging.getLogger(__name__)

    def process(self, origin_data: NERInput, config: Config) -> Optional[NERResult]:
        import tqdm
        model: Union["GLiNER", "GLiNER2"] = None
        get_entities: Callable[[str, List[str]], List[GLiNERResult]]
        cfg = config.get_config(key=self.__class__)
        match cfg["version"]:
            case 1:
                from gliner import GLiNER
                model = GLiNER.from_pretrained(cfg["model"])

                def get_entities(_txt: str, _labels: List[str]) -> List[GLiNERResult]:
                    _result = model.predict_entities(txt, [label])
                    _ret = []
                    for _r in _result:
                        _ret.append(GLiNERResult(text=_r["text"], start=_r["start"], end=_r["end"], label=_r["label"]))
                    return _ret

            case 2:
                from gliner2 import GLiNER2
                GLiNER2._print_config = lambda *args, **kwargs: None
                model = GLiNER2.from_pretrained(cfg["model"])

                def get_entities(_txt: str, _labels: List[str]) -> List[GLiNERResult]:
                    _result = model.extract_entities(txt, [label], include_spans=True)
                    _ret = []
                    for _label, _entities in _result["entities"].items():
                        for _e in _entities:
                            _ret.append(GLiNERResult(text=_e["text"], start=_e["start"], end=_e["end"], label=_label))
                    return _ret
            case _:
                self.__class__._LOGGER.error(f"Unknown GliNER version: {cfg['version']}")
                return None

        labels = cfg["labels"]
        self.__class__._LOGGER.info(f"Running GliNER{cfg['version']}-{cfg['model']} with labels: {', '.join(labels)}")

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
