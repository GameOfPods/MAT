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
from typing import Iterable, Optional, Dict, List, Tuple
import logging

from MAT.tools.ner import NERTool, NERResult, NERInput
from MAT.utils.config import ConfigElement, Config


class NERGliner(NERTool):
    @classmethod
    def config_name(cls) -> str:
        return "GliNER"

    @classmethod
    def config_keys(cls) -> Dict[str, ConfigElement]:
        return {
            "model": ConfigElement(
                default_value="urchade/gliner_multi-v2.1",
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
            )
        }

    _LOGGER = logging.getLogger(__name__)

    def process(self, origin_data: NERInput, config: Config) -> Optional[NERResult]:
        from gliner import GLiNER
        import tqdm

        cfg = config.get_config(key=self.__class__)

        model = GLiNER.from_pretrained(cfg["model"])
        labels = cfg["labels"]

        self.__class__._LOGGER.info(f"Running GliNER with labels: {', '.join(labels)}")

        ret: List[Dict[str, List[Tuple[str, int, int]]]] = []
        for txt in tqdm.tqdm(origin_data.text, leave=False, desc="NER on sentence", unit="sentences"):
            ret.append({})
            for label in labels:
                ret[-1][label] = []
                result = model.predict_entities(txt, [label])
                for r in result:
                    ret[-1][label].append((r["text"], r["start"], r["end"]))

        return NERResult(*ret)
