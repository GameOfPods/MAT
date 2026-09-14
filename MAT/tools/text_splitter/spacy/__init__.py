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
from typing import Optional

from pydantic import Field

from MAT.registry import register, require

require("spacy", "spacy_download", extra="spacy")

from MAT.tools.text_splitter import SplitterInput, SplitterResult, SplitterTool  # noqa: E402
from MAT.utils.config import Config, Options  # noqa: E402


class SpacyOptions(Options):
    model: Optional[str] = Field(None, description="spaCy model. Not set: picked by language (en_core_web_trf, "
                                                   "de_core_news_lg, fr_dep_news_trf, xx_sent_ud_sm for others). "
                                                   "Missing models get downloaded.")


@register("splitter", "spacy", description="spaCy sentences and lemma counts")
class SplitterSpacy(SplitterTool):
    Options = SpacyOptions
    packages = ("spacy",)
    _LOGGER = logging.getLogger(__name__)
    _DEFAULT_MODELS = {
        "en": "en_core_web_trf",
        "de": "de_core_news_lg",
        "fr": "fr_dep_news_trf",
        None: "xx_sent_ud_sm"
    }

    def process(self, origin_data: SplitterInput, config: Config) -> Optional[SplitterResult]:
        from collections import Counter
        model = config.options(self).model
        if model is None:
            self.__class__._LOGGER.debug("No model specified. Guessing best model by language")
            from langdetect import detect
            lang = detect(origin_data.text)
            model = self.__class__._DEFAULT_MODELS.get(lang, self.__class__._DEFAULT_MODELS[None])
        self.__class__._LOGGER.debug(f"Using {model} SpaCy model")
        try:
            import spacy
            nlp = spacy.load(model)
        except OSError:
            from spacy_download import load_spacy
            nlp = load_spacy(model)

        doc = nlp(origin_data.text)

        ret = SplitterResult(
            sentences=[x.text for x in doc.sents],
            words=[Counter(e.lemma_ for e in s if not any([e.is_space, e.is_punct, e.is_stop])) for s in doc.sents]
        )

        del doc

        import gc
        gc.collect()

        return ret
