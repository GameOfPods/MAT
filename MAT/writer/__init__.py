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
from typing import List, Callable, Dict, Type
import os
import pathlib
from datetime import datetime
import json

from MAT.pipelines import PipelineResult, PodcastOutput, BookOutput, BookPipeline


class Writer:
    __api_version__ = "1"

    def __init__(self):
        self._storage_functions: Dict[Type[PipelineResult], Callable[[PipelineResult, str], None]] = {}
        self.register_writer(PodcastOutput, self.store_podcast_output)
        self.register_writer(BookOutput, self.store_book_output)

    def register_writer(self, t: Type[PipelineResult], f: Callable[[PipelineResult, str], None]):
        self._storage_functions[t] = f

    def store(self, file: str, output: str, pipeline_results: List[PipelineResult]) -> str:
        now = datetime.now()
        folder = os.path.join(output, f"{pathlib.Path(file).stem}_{now.strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(folder, exist_ok=False)
        meta = {
            "version": self.__api_version__,
            "file_name": os.path.basename(file),
            "file_name_wo_extension": pathlib.Path(file).stem,
            "full_file": os.path.abspath(file),
            "creation_time": now.isoformat(),
            "pipelines": []
        }
        for i, res in enumerate(pipeline_results):
            # noinspection PyBroadException
            try:
                pipe_folder = f"{i}.{type(res).__name__}"
                os.makedirs(os.path.join(folder, pipe_folder), exist_ok=False)
                self._storage_functions[type(res)](res, os.path.join(folder, pipe_folder))
                meta["pipelines"].append({
                    "folder": pipe_folder,
                    "type": str(type(res)),
                })
            except Exception as e:
                raise e
        with open(os.path.join(folder, "meta.json"), "w") as f:
            json.dump(meta, f)
        return os.path.abspath(folder)

    @staticmethod
    def store_book_output(output: BookOutput, folder: str):
        r = {
            "title": output.title if output.title is not None else "",
            "language": output.language if output.language is not None else "",
            "chapters": []
        }
        for c in output.chapter_data:
            r_c = {
                "heading_raw": c.heading,
                "heading": c.get_beautiful_heading(),
                "content": c.content,
            }
            if c.sentence_words is not None:
                r_c["sentence_words"] = c.sentence_words
            if c.sentences is not None:
                r_c["sentences"] = c.sentences
            if c.ner is not None:
                r_c["ner"] = []
                for n in c.ner:
                    r_c["ner"].append({})
                    for k, v in n.items():
                        r_c["ner"][-1][k] = []
                        for ent, fr, to in v:
                            r_c["ner"][-1][k].append({
                                "ent": ent,
                                "from": fr,
                                "to": to,
                            })
            r["chapters"].append(r_c)
        with open(os.path.join(folder, "book.json"), "w") as f:
            json.dump(r, f)

    @staticmethod
    def store_podcast_output(output: PodcastOutput, folder: str):
        if output.media_info is not None:
            with open(os.path.join(folder, "media.json"), "w") as f:
                json.dump(output.media_info.as_dict(), f, indent=2)

        if output.full_transcript is not None:
            with open(os.path.join(folder, "transcript.txt"), "w") as f:
                f.write(output.full_transcript)

        if output.word_speaker is not None:
            with open(os.path.join(folder, "transcript.json"), "w") as f:
                r = {"transcript": []}
                for x in output.word_speaker:
                    r["transcript"].append(
                        {"speaker": tuple(x.speaker), "word": x.word.word, "start": x.word.start, "finish": x.word.end}
                    )
                json.dump(r, f)

        if output.summary is not None:
            with open(os.path.join(folder, "summary.txt"), "w") as f:
                r = []
                for i, t in enumerate(output.summary.text):
                    r.append(t)
                f.write("\n---\n".join(r))

        if output.diarization_matched is not None:
            with open(os.path.join(folder, "diarization.json"), "w") as f:
                json.dump(output.diarization_matched.to_dict(), f)

        if output.diarization_matched is not None and output.media_info is not None:
            from pydantic import ValidationError
            try:
                from rttm_manager import export_rttm, RTTM
                time_line = []
                for speaker in output.diarization_matched.speaker:
                    for fr, to in output.diarization_matched.get_diarization(speaker=speaker):
                        time_line.append((fr, to, speaker))
                time_line.sort()
                rttms = []
                for fr, to, speaker in time_line:
                    rttms.append(RTTM(
                        type="SPEAKER", file_id=output.media_info.file_name, channel_id=1, speaker_name=speaker,
                        turn_onset=fr, turn_duration=to - fr
                    ))
                export_rttm(rttms=rttms, file_path=os.path.join(folder, "diarization.rttm"))
            except (ImportError, ValidationError):
                pass
