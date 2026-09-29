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
    model: Optional[str] = Field(None, description="spaCy model. Not set: picked by the language of the book "
                                                   "(en_core_web_md, de_core_news_md, fr_core_news_md, xx_sent_ud_sm "
                                                   "for others). The English and German ones come with the spacy "
                                                   "extra, others get downloaded.")


@register("splitter", "spacy", description="spaCy sentences and lemma counts")
class SplitterSpacy(SplitterTool):
    Options = SpacyOptions
    packages = ("spacy",)
    _LOGGER = logging.getLogger(__name__)
    # md: the lemmas and sentence borders we need are as good as with lg, at a tenth of the size. The transformer
    # models (*_trf) need spacy-transformers, which pins transformers to an old version.
    _DEFAULT_MODELS = {
        "en": "en_core_web_md",
        "de": "de_core_news_md",
        "fr": "fr_core_news_md",
        None: "xx_sent_ud_sm"
    }

    def __init__(self):
        # a book calls this once per chapter, the model is loaded once
        self._loaded = {}

    def process(self, origin_data: SplitterInput, config: Config) -> Optional[SplitterResult]:
        from collections import Counter

        model = config.options(self).model
        if model is None:
            language = origin_data.language
            if language is None:
                from langdetect import detect
                language = detect(origin_data.text)
            model = self._DEFAULT_MODELS.get(language, self._DEFAULT_MODELS[None])
        nlp = self._loaded.get(model)
        if nlp is None:
            self._LOGGER.info(f"Using spaCy model {model}")
            try:
                import spacy
                nlp = spacy.load(model)
            except OSError:
                from spacy_download import load_spacy
                nlp = load_spacy(model)
            self._loaded = {model: nlp}

        doc = nlp(origin_data.text)
        # spaCy keeps the line break after a sentence ("Alice met Bob.\n"), nobody wants that stored
        sentences = [(sent.text.strip(), sent) for sent in doc.sents]
        sentences = [(text, sent) for text, sent in sentences if text]
        ret = SplitterResult(
            sentences=[text for text, _ in sentences],
            words=[Counter(e.lemma_ for e in sent if not any([e.is_space, e.is_punct, e.is_stop]))
                   for _, sent in sentences]
        )
        del doc
        return ret
