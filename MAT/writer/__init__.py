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
"""
Writes results in format 2.

    <input name>_<time>/
        meta.json               format, MAT version, input file, pipelines that ran
        podcast/result.json     everything the podcast pipeline found, plus transcript.txt, summary.md, diarization.rttm
        book/result.json

`MAT.reader.MATResult` reads it back, from the folder or from a zip of it.
"""
import json
import os
import pathlib
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from MAT.pipelines import PipelineResult, PodcastOutput, BookOutput
from MAT.tools import DiarizationResult, WordTupleSpeaker

FORMAT_VERSION = 2


def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")


def _timeline(diarization: Optional[DiarizationResult]) -> Optional[Dict[str, List[List[float]]]]:
    if diarization is None:
        return None
    return {speaker: [[start, end] for start, end in sorted(diarization.get_diarization(speaker))]
            for speaker in sorted(diarization.speaker)}


def _words(words: Optional[List[WordTupleSpeaker]]) -> List[Dict[str, Any]]:
    return [{"start": w.word.start, "end": w.word.end, "text": w.word.word,
             "speakers": sorted(s for s in w.speaker if s is not None)} for w in words or []]


def _summary_text(output: PodcastOutput) -> Optional[str]:
    return None if output.summary is None else "\n\n---\n\n".join(output.summary.text)


def podcast_json(output: PodcastOutput) -> Dict[str, Any]:
    media = output.media_info.as_dict() if output.media_info is not None else None
    return {
        "schema": "mat.podcast",
        "format": FORMAT_VERSION,
        "models": output.models,
        "language": (media or {}).get("language") or None,
        "media": media,
        # speakers after matching them to gold labels, diarization is what the diarizer said
        "speakers": _timeline(output.diarization_matched),
        "diarization": _timeline(output.diarization),
        "words": _words(output.word_speaker),
        # words of the same speaker merged into lines
        "segments": _words(output.squished_speaker),
        "summary": _summary_text(output),
        # filled by later stages (audio events, named entities in transcripts)
        "events": [],
        "entities": [],
    }


def book_json(output: BookOutput) -> Dict[str, Any]:
    chapters = []
    for chapter in output.chapter_data:
        sentences = []
        for i, text in enumerate(chapter.sentences or []):
            lemmas = chapter.sentence_words[i] if chapter.sentence_words and i < len(chapter.sentence_words) else {}
            ner = chapter.ner[i] if chapter.ner and i < len(chapter.ner) else {}
            entities = [{"label": label, "text": entity, "start": start, "end": end}
                        for label, found in ner.items() for entity, start, end in found]
            sentences.append({"text": text, "lemmas": dict(lemmas), "entities": entities})
        chapters.append({
            "heading": chapter.get_beautiful_heading(),
            "heading_raw": chapter.heading,
            "paragraphs": chapter.content,
            "sentences": sentences,
        })
    return {
        "schema": "mat.book",
        "format": FORMAT_VERSION,
        "models": output.models,
        "title": output.title,
        "language": output.language or None,
        "chapters": chapters,
    }


def write_podcast(output: PodcastOutput, folder: str) -> None:
    _write_json(os.path.join(folder, "result.json"), podcast_json(output))
    if output.full_transcript:
        _write_text(os.path.join(folder, "transcript.txt"), output.full_transcript)
    summary = _summary_text(output)
    if summary:
        _write_text(os.path.join(folder, "summary.md"), summary)
    if output.diarization_matched is not None and output.media_info is not None:
        from pydantic import ValidationError
        try:
            from rttm_manager import export_rttm, RTTM
            time_line = sorted((start, end, speaker) for speaker in output.diarization_matched.speaker
                               for start, end in output.diarization_matched.get_diarization(speaker=speaker))
            rttms = [RTTM(type="SPEAKER", file_id=output.media_info.file_name, channel_id=1, speaker_name=speaker,
                          turn_onset=start, turn_duration=end - start) for start, end, speaker in time_line]
            export_rttm(rttms=rttms, file_path=os.path.join(folder, "diarization.rttm"))
        except (ImportError, ValidationError):
            pass


def write_book(output: BookOutput, folder: str) -> None:
    _write_json(os.path.join(folder, "result.json"), book_json(output))


class Writer:
    def __init__(self):
        self._writers: Dict[Type[PipelineResult], Tuple[str, Callable[[PipelineResult, str], None]]] = {}
        self.register_writer(PodcastOutput, "podcast", write_podcast)
        self.register_writer(BookOutput, "book", write_book)

    def register_writer(self, result_type: Type[PipelineResult], folder_name: str,
                        function: Callable[[PipelineResult, str], None]) -> None:
        self._writers[result_type] = (folder_name, function)

    def store(self, file: str, output: str, pipeline_results: List[PipelineResult]) -> str:
        from MAT.utils import get_hash_file
        from MAT import __version__
        now = datetime.now()
        base_folder = os.path.join(output, f"{pathlib.Path(file).stem}_{now.strftime('%Y-%m-%d_%H-%M-%S')}")
        # two inputs with the same name finishing in the same second would collide otherwise
        folder, i = base_folder, 1
        while os.path.exists(folder):
            folder = f"{base_folder}_{i}"
            i += 1
        os.makedirs(folder, exist_ok=False)
        meta = {
            "format": FORMAT_VERSION,
            "mat_version": __version__,
            "created": now.isoformat(timespec="seconds"),
            "input": {
                "name": os.path.basename(file),
                "path": os.path.abspath(file),
                "sha1": get_hash_file(file_path=os.path.abspath(file)),
            },
            "pipelines": [],
        }
        for result in pipeline_results:
            if type(result) not in self._writers:
                raise TypeError(f"No writer registered for {type(result).__name__}")
            name, write = self._writers[type(result)]
            os.makedirs(os.path.join(folder, name), exist_ok=False)
            write(result, os.path.join(folder, name))
            meta["pipelines"].append(name)
        _write_json(os.path.join(folder, "meta.json"), meta)
        return os.path.abspath(folder)


__all__ = ["Writer", "FORMAT_VERSION", "podcast_json", "book_json"]
