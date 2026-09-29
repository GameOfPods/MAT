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
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Callable, List, Tuple, Optional
import logging

from pydantic import Field

from MAT.pipelines import PipelineResult, Pipeline, PipelineStepResult, PipelineStepInput, Slot
from MAT.tools.ner import NERInput
from MAT.tools.text_splitter import SplitterInput
from MAT.utils.config import Options


@dataclass
class Chapter:
    heading: str
    content: List[str]
    heading_beautified: Optional[str] = None
    sentences: Optional[List[str]] = None
    sentence_words: Optional[List[Dict[str, int]]] = None
    ner: Optional[List[Dict[str, List[Tuple[str, int, int]]]]] = None
    summary: Optional[str] = None

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
    models: Dict[str, Any] = field(default_factory=dict)
    # {"name", "mentions", "variants", "chapters"} per character, see MAT/utils/characters.py
    characters: List[Dict[str, Any]] = field(default_factory=list)


class BookOptions(Options):
    splitter: str = Field("spacy", description="Splits chapters into sentences and counts lemmas.")
    ner: str = Field("gliner", description='Named entities per sentence. "none" skips it.')
    character_labels: List[str] = Field(["PERSON"], description="NER labels that count as characters for the "
                                                                "character list.")
    min_mentions: int = Field(2, ge=1, description="Characters mentioned less often stay off the character list.")
    character_judge: str = Field("none", description='Decides with an LLM which names are one character, from '
                                                      'sentences that say so ("X, den alle Y nannten"), English '
                                                      'nicknames and ambiguous short names like a family name. '
                                                      '"llm-characters" turns it on, settings in [llm-characters].')
    chapter_summarizer: str = Field("none", description='Summary of every chapter, with the [llm] settings and '
                                                        'prompts written for chapters (no spoilers from later '
                                                        'chapters). "llm" turns it on.')
    chapter_names: List[str] = Field(default_factory=list, description="Headings that always count as chapters.")


class BookPipeline(Pipeline):
    section = "book"
    description = "Chapters, sentences, lemma counts and named entities for EPUB books."
    Options = BookOptions
    slots = {"splitter": Slot(), "ner": Slot(optional=True),
             "character_judge": Slot(optional=True, kind="characters"),
             "chapter_summarizer": Slot(optional=True, kind="summarizer")}
    required_steps = {"parse_book", "validate_chapters"}
    _LOGGER = logging.getLogger(__name__)
    _SPECIAL_CHAPTERS = {"prologue", "introduction", "epilogue", "prolog", "epilog"}
    _CHAPTER_NUMBER_REGEX = {re.compile(r"chapter \d+$"), re.compile(r"kapitel \d+$")}

    @classmethod
    def accept(cls, f: str) -> bool:
        return cls.why_not(f) is None

    @classmethod
    def why_not(cls, f: str) -> Optional[str]:
        try:
            from ebooklib import epub
            epub.read_epub(f, options={"ignore_ncx": True})
            return None
        except Exception as e:
            return f"not an EPUB ({e.__class__.__name__})"

    def _get_steps(self) -> Iterable[Callable[[PipelineStepInput], PipelineStepResult]]:
        from ebooklib import epub, ITEM_DOCUMENT, ITEM_NAVIGATION

        def parse_book(step_input: PipelineStepInput) -> PipelineStepResult:
            book: epub.EpubBook = epub.read_epub(step_input.file, options={"ignore_ncx": True})
            return PipelineStepResult(name="Read Book", data=book)

        def validate_chapters(step_input: PipelineStepInput) -> PipelineStepResult:
            from bs4 import BeautifulSoup
            from collections import Counter
            book: epub.EpubBook = step_input.data("Read Book")
            if book is None:
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
            book_valid_chapters = step_input.config.options(self).chapter_names
            valid_chapters = set(
                k for k, v in heading_c.items() if self._chapter_valid(k, heading_c, book_valid_chapters))
            invalid_chapters = set(heading_c.keys()) - valid_chapters
            self.__class__._LOGGER.info(f"Valid chapters   ({len(valid_chapters)}): "
                                        f"{' '.join(f'<<{x}>>' for x in sorted(valid_chapters))}")
            self.__class__._LOGGER.info(f"Invalid chapters ({len(invalid_chapters)}): "
                                        f"{' '.join(f'<<{x}>>' for x in sorted(invalid_chapters))}")
            chapters = [Chapter(heading=x, content=y) for x, y in chapters if x in valid_chapters]
            self.__class__._LOGGER.info(f"Going forward with {len(chapters)} chapters")
            return PipelineStepResult(name="Chapters", data=chapters)

        def beautify_chapters(step_input: PipelineStepInput) -> PipelineStepResult:
            from collections import Counter
            from MAT.utils import toRoman
            chapters: List[Chapter] = step_input.data("Chapters")
            if not chapters:
                return PipelineStepResult(name="Chapters Beauty", data=None)
            counter1 = Counter(x.heading for x in chapters)
            counter2 = Counter()
            for c in chapters:
                if counter1[c.heading] > 1:
                    counter2[c.heading] += 1
                    c.heading_beautified = f"{c.heading} {toRoman(counter2[c.heading])}"
            return PipelineStepResult(name="Chapters Beauty", data=chapters)

        def get_language(step_input: PipelineStepInput) -> PipelineStepResult:
            from langdetect import detect
            chapters: List[Chapter] = step_input.data("Chapters")
            if not chapters:
                return PipelineStepResult(name="Language", data=None)
            full_text = "\n".join("\n".join(x.content) for x in chapters)
            lang = detect(full_text)
            self.__class__._LOGGER.info(f"Detected language: {lang}")
            return PipelineStepResult(name="Language", data=lang)

        def splitting_task(step_input: PipelineStepInput) -> PipelineStepResult:
            import tqdm
            chapters: List[Chapter] = step_input.data("Chapters Beauty") or step_input.data("Chapters")
            if not chapters:
                return PipelineStepResult(name="Word Counter", data=None)
            splitter = self.backend("splitter", step_input.config)
            language = step_input.data("Language")
            for c in tqdm.tqdm(chapters, leave=False, desc="Working on chapters", unit="chapter"):
                splitted = splitter.process(origin_data=SplitterInput("\n".join(c.content), language=language),
                                            config=step_input.config)
                c.sentences = list(splitted.sentences) if splitted.sentences is not None else None
                c.sentence_words = list(splitted.words) if splitted.words is not None else None
            return PipelineStepResult(name="Word Counter", data=chapters)

        def ner_task(step_input: PipelineStepInput) -> PipelineStepResult:
            chapters: List[Chapter] = step_input.data("Word Counter")
            if not chapters:
                return PipelineStepResult(name="NER", data=None)
            ner = self.backend("ner", step_input.config)
            if ner is None:
                return PipelineStepResult(name="NER", data=chapters)
            # all sentences of the book in one call, so the model loads once and not once per chapter
            sentences = [sentence for c in chapters for sentence in (c.sentences or [])]
            self.__class__._LOGGER.info(f"Using {ner.backend_name} on {len(sentences)} sentences of "
                                        f"{len(chapters)} chapters")
            found = list(ner.process(origin_data=NERInput(*sentences), config=step_input.config).ner)
            for c in chapters:
                if c.sentences:
                    c.ner, found = found[:len(c.sentences)], found[len(c.sentences):]
            return PipelineStepResult(name="NER", data=chapters)

        def character_task(step_input: PipelineStepInput) -> PipelineStepResult:
            from MAT.utils.characters import Mention, build, cluster

            chapters: List[Chapter] = step_input.data("NER")
            if not chapters:
                return PipelineStepResult(name="Characters", data=None)
            options = step_input.config.options(self)
            wanted = {label.casefold() for label in options.character_labels}
            mentions = [Mention(chapter=c.get_beautiful_heading(), name=text,
                                sentence=c.sentences[i] if c.sentences and i < len(c.sentences) else "")
                        for c in chapters for i, sentence in enumerate(c.ner or [])
                        for label, found in sentence.items() if label.casefold() in wanted for text, _, _ in found]
            clusters = cluster(mentions)
            joins, resolved = self._judge_characters(clusters, step_input)
            characters = build(clusters, min_mentions=options.min_mentions, joins=joins, resolved=resolved)
            self.__class__._LOGGER.info(f"{len(characters)} characters, most mentioned: "
                                        + ", ".join(f"{c['name']} ({c['mentions']})" for c in characters[:8]))
            return PipelineStepResult(name="Characters", data=characters)

        def summary_task(step_input: PipelineStepInput) -> PipelineStepResult:
            from MAT.tools.summary import SummaryInput

            chapters: List[Chapter] = step_input.data("Chapters Beauty") or step_input.data("Chapters")
            if not chapters:
                return PipelineStepResult(name="Chapter summaries", data=None)
            summarizer = self.backend("chapter_summarizer", step_input.config)
            if summarizer is None:
                return PipelineStepResult(name="Chapter summaries", data=None)
            book = step_input.data("Read Book")
            texts = [f"{c.get_beautiful_heading()}\n\n" + "\n\n".join(c.content) for c in chapters]
            self.__class__._LOGGER.info(f"Summarizing {len(texts)} chapters")
            # like the podcast summary: a failing LLM must not cost the rest of the results
            try:
                result = summarizer.process(
                    origin_data=SummaryInput(*texts, kind="chapter",
                                             additional_metadata={"book": getattr(book, "title", "") or ""}),
                    config=step_input.config)
            except Exception as e:
                self.__class__._LOGGER.exception("Chapter summaries failed, writing the book without them",
                                                 exc_info=e)
                return PipelineStepResult(name="Chapter summaries", data=None)
            for chapter, summary in zip(chapters, list(result.text) if result is not None else []):
                chapter.summary = summary or None
            return PipelineStepResult(name="Chapter summaries", data=chapters)

        return [parse_book, validate_chapters, beautify_chapters, get_language, splitting_task, ner_task,
                character_task, summary_task]

    def _judge_characters(self, clusters, step_input: PipelineStepInput):
        """Asks the character judge about the candidates, returns what it confirmed: (joins, resolved)."""
        from MAT.tools.characters import CharacterJudgeInput, MentionQuestion, PairQuestion
        from MAT.utils.characters import candidates

        judge = self.backend("character_judge", step_input.config)
        if judge is None:
            return [], {}
        language = step_input.data("Language")
        pairs, mentions = candidates(clusters, language=language)
        if not pairs and not mentions:
            self.__class__._LOGGER.info("No name pairs or ambiguous names to ask about")
            return [], {}
        self.__class__._LOGGER.info(f"Asking {judge.backend_name} about {len(pairs)} name pairs and "
                                    f"{len(mentions)} ambiguous names")
        questions = [PairQuestion(id=i, a=clusters.display(p.a), b=clusters.display(p.b), sentences_a=p.sentences_a,
                                  sentences_b=p.sentences_b, together=p.together) for i, p in enumerate(pairs)]
        options = {i: {clusters.display(o): o for o in m.options} for i, m in enumerate(mentions)}
        mention_questions = [MentionQuestion(id=i, name=m.name, sentence=m.sentence, options=list(options[i]))
                             for i, m in enumerate(mentions)]
        try:
            result = judge.process(CharacterJudgeInput(questions, mention_questions, language=language),
                                   step_input.config)
        except Exception as e:
            # the list without the judge is still right, just less merged
            self.__class__._LOGGER.exception("Judging the characters failed, keeping the list as the rules made it",
                                             exc_info=e)
            return [], {}
        joins = [(pairs[i].a, pairs[i].b, evidence) for i, evidence in result.same.items()]
        resolved = {mentions[i].index: options[i][name] for i, name in result.mentions.items() if name in options[i]}
        return joins, resolved

    def _finalize_result(self, step_results: Dict[str, PipelineStepResult]) -> BookOutput:

        def _try_get(k: str):
            result = step_results.get(k)
            return None if result is None else result.data

        book = _try_get("Read Book")
        chapters = _try_get("Chapters")
        language = _try_get("Language")

        return BookOutput(
            title=book.title if book is not None else "No title",
            language=language if language is not None else "",
            chapter_data=chapters if chapters is not None else [],
            models=dict(self.models),
            characters=_try_get("Characters") or [],
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
