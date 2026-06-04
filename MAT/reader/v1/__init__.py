from typing import Any, Dict, Iterator, Optional, List, Tuple, Set
from abc import ABC
from pathlib import Path

from MAT.reader import MATResult, ParsedResult, ResultTypes


class MATResultV1(MATResult):

    @classmethod
    def supported_version(cls) -> str:
        return "1"

    def __init__(self, path: Path):
        super().__init__(path)
        import json

        self._content = self.get_content(files=self.filelist)

        self._meta = json.loads(self._content["meta.json"].decode("utf-8"))
        self.get_logger().info(f"Parsed result from {self.filename}. Was created using {self.MAT_version}")

    def meta(self, key) -> Any:
        return self._meta[key]

    def get_results(self, t: ResultTypes) -> Iterator[ParsedResult]:
        if t == ResultTypes.PODCAST:
            for r in self.get_podcast_results:
                yield r
        elif t == ResultTypes.BOOK:
            for r in self.get_book_results:
                yield r
        else:
            self.get_logger().error(f"{self.__class__.__name__} does not support Result {t.name}")

    @property
    def filename(self) -> str:
        return self.meta("file_name")

    @property
    def MAT_version(self) -> str:
        return self.meta("MAT_version")

    @property
    def get_book_results(self) -> Iterator[ParsedResult]:
        from MAT.pipelines.Book import BookOutput
        for pipeline in self.meta("pipelines"):
            if pipeline["type"] == str(BookOutput):
                content = {
                    k.split(self.pathsep, 1)[1]: v
                    for k, v in self._content.items() if k.startswith(pipeline["folder"] + self.pathsep)}
                yield ParsedBookResult(content=content, parent=self)

    @property
    def get_podcast_results(self) -> Iterator[ParsedResult]:
        from MAT.pipelines.Podcast import PodcastOutput
        for pipeline in self.meta("pipelines"):
            if pipeline["type"] == str(PodcastOutput):
                content = {
                    k.split(self.pathsep, 1)[1]: v
                    for k, v in self._content.items() if k.startswith(pipeline["folder"] + self.pathsep)}
                yield ParsedPodcastResult(content=content, parent=self)


class ParsedBookResult(ParsedResult):
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Sentence:
        sentence: str
        ner: Dict[str, List[str]]
        words: Dict[str, int]

    @dataclass(frozen=True)
    class Chapter:
        heading_raw: str
        heading: str
        paragraphs: List[str]
        sentences: List["ParsedBookResult.Sentence"]

    def __init__(self, content: Dict[str, Any], parent: MATResult):
        super().__init__(parent)
        import json
        self._content = {}
        if "book.json" in content:
            self._content = json.loads(content["book.json"].decode("utf-8"))

    @property
    def language(self) -> str:
        return self._content["language"]

    @property
    def title(self) -> str:
        return self._content["title"]

    @property
    def chapters(self) -> List["ParsedBookResult.Chapter"]:
        ret = []
        for chapter in self._content["chapters"]:
            if len(chapter["sentence_words"]) != len(chapter["sentences"]) or len(chapter["sentences"]) != len(chapter["ner"]):
                self.parent.get_logger().error("The sentence data was not recorded correctly")
                exit(1)

            sentences = []
            for sentence, sentence_words, ner in zip(chapter["sentences"], chapter["sentence_words"], chapter["ner"]):
                sentences.append(ParsedBookResult.Sentence(
                    sentence=sentence, words=sentence_words,
                    ner={k: [x["ent"] for x in v] for k, v in ner.items()}
                ))

            ret.append(ParsedBookResult.Chapter(
                heading_raw=chapter["heading_raw"],
                heading=chapter["heading"],
                paragraphs=chapter["content"],
                sentences=sentences
            ))
        return ret


class ParsedPodcastResult(ParsedResult):

    def __init__(self, content: Dict[str, bytes], parent: MATResultV1):
        super().__init__(parent)
        import json
        self._diarization: Optional[Dict[str, List[Tuple[float, float]]]] = None
        if "diarization.json" in content:
            self._diarization = json.loads(content["diarization.json"].decode("utf-8"))
        elif "diarization.rttm" in content:
            from rttm_manager import RTTMImporter
            self._diarization = {}
            for line in content["diarization.rttm"].decode("utf-8").splitlines():
                rttm_line = RTTMImporter._get_rttm_from_line(line)
                if rttm_line.speaker_name not in self._diarization:
                    self._diarization[rttm_line.speaker_name] = []
                self._diarization[rttm_line.speaker_name].append((
                    rttm_line.turn_onset,
                    rttm_line.turn_onset + rttm_line.turn_duration
                ))

        self._media = json.loads(content["media.json"].decode("utf-8"))
        self._summary = content["summary.txt"].decode("utf-8") if "summary.txt" in content else ""
        self._transcript = content["transcript.txt"].decode("utf-8") if "transcript.txt" in content else ""

    @property
    def language(self) -> str:
        return self._media["language"]

    @property
    def duration(self) -> str:
        return self._media["duration"]

    @property
    def duration_after_vad(self) -> str:
        return self._media["duration_after_vad"]

    @property
    def diarization(self) -> Dict[str, List[Tuple[float, float]]]:
        if self._diarization is None:
            return {}
        from copy import deepcopy
        return deepcopy(self._diarization)

    @property
    def speaker_names(self) -> Set[str]:
        return set(self._diarization.keys() if self._diarization is not None else [])

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def transcript(self) -> str:
        return self._transcript
