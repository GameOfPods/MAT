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
from typing import Dict, Optional
import logging

from MAT.tools.text_splitter import SplitterInput, SplitterResult, SplitterTool
from MAT.utils.config import ConfigElement, Config


class SplitterSpacy(SplitterTool):
    _LOGGER = logging.getLogger(__name__)
    _DEFAULT_MODELS = {
        "en": "en_core_web_trf",
        "de": "de_core_news_lg",
        "fr": "fr_dep_news_trf",
        None: "xx_sent_ud_sm"
    }

    @classmethod
    def config_name(cls) -> str:
        return "SpaCy"

    @classmethod
    def config_keys(cls) -> Dict[str, ConfigElement]:
        return {
            "model": ConfigElement(
                default_value=None,
                argparse_kwargs={
                    "help": "Model to use for spacy. If not given will try to guess best model from language",
                    "type": str,
                }
            )
        }

    def process(self, origin_data: SplitterInput, config: Config) -> Optional[SplitterResult]:
        from pprint import pformat
        from collections import Counter
        cfg = config.get_config(self)
        model = cfg["model"]
        if model is None:
            self.__class__._LOGGER.debug("No model specified. Using default model")
            try:
                from langdetect import detect
                lang = detect(origin_data.text)
            except ImportError:
                lang = None
            model = self.__class__._DEFAULT_MODELS.get(lang, self.__class__._DEFAULT_MODELS[None])
        spacy_module_kwargs = {}
        self.__class__._LOGGER.debug(f"Using {model} SpaCy model. With arguments: {pformat(spacy_module_kwargs)}")
        try:
            import spacy
            nlp = spacy.load(model, **spacy_module_kwargs)
        except:
            from spacy_download import load_spacy
            nlp = load_spacy(model, **spacy_module_kwargs)

        doc = nlp(origin_data.text)

        return SplitterResult(
            sentences=[x.text for x in doc.sents],
            words=[Counter(e.lemma_ for e in s if not any([e.is_space, e.is_punct, e.is_stop])) for s in doc.sents]
        )
