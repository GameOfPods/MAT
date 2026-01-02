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
from dataclasses import dataclass
from typing import Dict, Iterable, Callable, List, Tuple, Optional
import logging

from MAT import SplitterSpacy, SplitterInput
from MAT.pipelines import PipelineResult, Pipeline, PipelineStepResult, T_out, PipelineStepInput
from MAT.utils.config import ConfigElement


@dataclass
class Chapter:
    heading: str
    content: List[str]
    heading_beautified: Optional[str] = None
    sentences: Optional[List[str]] = None
    sentence_words: Optional[List[Dict[str, int]]] = None
    ner: Optional[List[Dict[str, List[Tuple[str, int, int]]]]] = None

    def get_beautiful_heading(self) -> str:
        return self.heading_beautified if self.heading_beautified is not None else self.heading

    @property
    def words(self) -> Optional[Dict[str, int]]:
        if self.sentence_words is None:
            return None
        from collections import defaultdict
        r = defaultdict(lambda: 0)
        for d in self.sentence_words:
            for k, v in d.items():
                r[k] += v
        return r


@dataclass
class BookOutput(PipelineResult):
    title: str
    language: Optional[str]
    chapter_data: List[Chapter]


class BookPipeline(Pipeline):
    import re
    _LOGGER = logging.getLogger(__name__)
    _SPECIAL_CHAPTERS = {"prologue", "introduction", "epilogue", "prolog", "epilog"}
    _CHAPTER_NUMBER_REGEX = {re.compile(r"chapter \d+$"), re.compile(r"kapitel \d+$")}

    @classmethod
    def accept(cls, f: str) -> bool:
        try:
            from ebooklib import epub
            epub.read_epub(f, options={"ignore_ncx": True})
            return True
        except:
            return False

    @classmethod
    def config_keys(cls) -> Dict[str, ConfigElement]:
        return {
            "chapter-names": ConfigElement(
                default_value=tuple(),
                argparse_kwargs={
                    "help": "Chapter names to use for this book",
                    "nargs": "*",
                }
            )
        }

    def _get_steps(self) -> Iterable[Callable[[PipelineStepInput], PipelineStepResult]]:
        from MAT.utils import get_hash_pipeline
        from ebooklib import epub, ITEM_DOCUMENT, ITEM_NAVIGATION
        def parse_book(step_input: PipelineStepInput) -> PipelineStepResult:
            book: epub.EpubBook = epub.read_epub(step_input.file, options={"ignore_ncx": True})
            return PipelineStepResult(
                name="Book",
                data=book
            )

        def validate_chapters(step_input: PipelineStepInput) -> PipelineStepResult:
            from bs4 import BeautifulSoup
            import re
            from collections import Counter
            try:
                book: epub.EpubBook = step_input.previous_results["Book"].data
                if book is None:
                    raise AttributeError()
            except (IndexError, KeyError, ValueError, TypeError, AttributeError):
                return PipelineStepResult(name="Chapters", data=None)
            nav = list(book.get_items_of_type(ITEM_NAVIGATION))[0].get_content().decode()
            items = sorted((x for x in book.get_items_of_type(ITEM_DOCUMENT) if x.get_name() in nav),
                           key=lambda x: nav.index(x.get_name()))
            chapters = []
            for item in items:
                content = item.get_body_content()
                soup = BeautifulSoup(content, "html.parser", from_encoding="utf-8")
                story = [x.get_text() for x in soup.find_all("p")]
                headings = [x.get_text() for x in soup.find_all(re.compile(r"^h[1-6]$"))]
                if len(headings) != 1 or len(headings[0]) <= 0:
                    continue
                chapters.append((headings[0].strip(), story))
            heading_c = Counter(x[0] for x in chapters)
            book_valid_chapters = step_input.config.get_config(self).get("chapter-names", list())
            valid_chapters = set(
                k for k, v in heading_c.items() if self._chapter_valid(k, heading_c, book_valid_chapters))
            invalid_chapters = set(heading_c.keys()) - valid_chapters
            self.__class__._LOGGER.info(f"Valid chapters  : {' '.join(f'<<{x}>>' for x in sorted(valid_chapters))}")
            self.__class__._LOGGER.info(f"Invalid chapters: {' '.join(f'<<{x}>>' for x in sorted(invalid_chapters))}")
            chapters = [Chapter(heading=x, content=y) for x, y in chapters if x in valid_chapters]
            return PipelineStepResult(name="Chapters", data=chapters)

        def beautify_chapters(step_input: PipelineStepInput) -> PipelineStepResult:
            try:
                chapters: List[Chapter] = step_input.previous_results["Chapters"].data
                if chapters is None or len(chapters) <= 0:
                    raise AttributeError()
            except (IndexError, KeyError, ValueError, TypeError, AttributeError):
                return PipelineStepResult(name="Chapters Beauty", data=None)
            try:
                from MAT.utils import toRoman
                from collections import Counter
                counter1 = Counter(x.heading for x in chapters)
                counter2 = Counter()

                for c in chapters:
                    if counter1[c.heading] > 1:
                        counter2[c.heading] += 1
                        c.heading_beautified = f"{c.heading} {toRoman(counter2[c.heading])}"

                return PipelineStepResult(name="Chapters Beauty", data=chapters)
            except ImportError:
                return PipelineStepResult(name="Chapters Beauty", data=chapters)

        def get_language(step_input: PipelineStepInput) -> PipelineStepResult:
            from langdetect import detect
            try:
                chapters: List[Chapter] = step_input.previous_results["Chapters"].data
                if chapters is None or len(chapters) <= 0:
                    raise AttributeError()
            except (IndexError, KeyError, ValueError, TypeError, AttributeError):
                return PipelineStepResult(name="Language", data=None)
            full_text = "\n".join("\n".join(x.content) for x in chapters)
            lang = detect(full_text)
            return PipelineStepResult(name="Language", data=lang)

        def splitting_task(step_input: PipelineStepInput) -> PipelineStepResult:
            import tqdm
            try:
                _prev_res = step_input.previous_results
                chapters: List[Chapter] = _prev_res.get("Chapters Beauty", _prev_res["Chapters"]).data
                if chapters is None or len(chapters) <= 0:
                    raise AttributeError()
            except (IndexError, KeyError, ValueError, TypeError, AttributeError):
                return PipelineStepResult(name="Word Counter", data=None)
            from MAT import SplitterSpacy, SplitterInput, SplitterResult

            splitter = SplitterSpacy()
            for c in tqdm.tqdm(chapters, leave=False, desc="Working on chapters", unit="chapter"):
                splitted = splitter.process(origin_data=SplitterInput("\n".join(c.content)), config=step_input.config)
                c.sentences = list(splitted.sentences) if splitted.sentences is not None else None
                c.sentence_words = list(splitted.words) if splitted.words is not None else None
            return PipelineStepResult(name="Word Counter", data=chapters)

        def ner_task(step_input: PipelineStepInput) -> PipelineStepResult:
            try:
                _prev_res = step_input.previous_results
                chapters: List[Chapter] = _prev_res["Word Counter"].data
                if chapters is None or len(chapters) <= 0:
                    raise AttributeError()
            except (IndexError, KeyError, ValueError, TypeError, AttributeError):
                return PipelineStepResult(name="NER", data=None)

            from MAT.tools.ner.ner_gliner import NERGliner, NERInput
            ner_gliner = NERGliner()
            for c in chapters:
                if c.sentences is None or len(c.sentences) <= 0:
                    continue
                res = ner_gliner.process(origin_data=NERInput(*c.sentences), config=step_input.config)
                c.ner = list(res.ner)

            return PipelineStepResult(name="NER", data=chapters)

        return [parse_book, validate_chapters, beautify_chapters, get_language, splitting_task, ner_task]

    def _finalize_result(self, step_results: Dict[str, PipelineStepResult]) -> BookOutput:

        def _try_get(k: str):
            try:
                return step_results[k].data
            except (IndexError, KeyError, ValueError, TypeError, AttributeError):
                return None

        book = _try_get("Book")
        chapters = _try_get("Chapters")
        language = _try_get("Language")

        return BookOutput(
            title=book.title if book is not None else "No title",
            language=language if language is not None else "",
            chapter_data=chapters if chapters is not None else [],
        )

    @classmethod
    def _chapter_valid(cls, chapter_name: str, chapter_counter: Dict[str, int],
                       book_valid_chapters: List[str] = None) -> bool:
        if book_valid_chapters is not None and len(book_valid_chapters) > 0:
            if chapter_name.strip().lower() in [x.strip().lower() for x in book_valid_chapters]:
                return True
        if chapter_name.isnumeric():
            return True
        if chapter_counter.get(chapter_name, 0) > 1 or chapter_counter.get(chapter_name.lower(), 0) > 1:
            return True
        if chapter_name in cls._SPECIAL_CHAPTERS or chapter_name.lower() in cls._SPECIAL_CHAPTERS:
            return True
        if any(x.fullmatch(chapter_name.lower()) for x in cls._CHAPTER_NUMBER_REGEX):
            return True
        return False


__all__ = ["BookPipeline", "BookOutput"]
