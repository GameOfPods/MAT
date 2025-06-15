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
from collections import defaultdict
from typing import List

from MAT.tools.transcriptors import TranscriptionResult, WordTupleSpeaker, WordTuple
from MAT.tools.diarizators import DiarizationResult


def word_speaker_match(word: WordTuple, speach_from, speach_to) -> float:
    if word.start > speach_to or word.end < speach_from:
        return 0
    common_start = max(word.start, speach_from)
    common_end = min(word.end, speach_to)
    if common_start == common_end:
        return 0
    return (common_end - common_start) / (word.end - word.start)


def align_diarization_with_transcription(diarization: DiarizationResult, transcript: TranscriptionResult) -> List[
    WordTupleSpeaker]:
    aligned_transcript = []

    # First, prepare a timeline of all speaker segments
    speaker_timeline = []
    for speaker in diarization.speaker:
        for f, t in diarization.get_diarization(speaker=speaker):
            speaker_timeline.append({"speaker": speaker, "time": f, "type": "start"})
            speaker_timeline.append({"speaker": speaker, "time": t, "type": "end"})

    speaker_timeline.sort(key=lambda x: x["time"])

    # Track active speakers at any point in time
    active_speakers = defaultdict(lambda: 0)
    speaker_intervals = []

    for event in speaker_timeline:
        if event["type"] == "start":
            active_speakers[event["speaker"]] += 1
        else:
            active_speakers[event["speaker"]] -= 1

        # Record the set of active speakers for this time point
        speaker_intervals.append({
            "time": event["time"],
            "speakers": set([k for k, v in active_speakers.items() if v > 0])
        })

    word_speaker: List[WordTupleSpeaker] = []

    # Assign words to speakers
    for word in transcript.word_timings:
        word_compare_time = (word.start + word.end) / 2
        word_compare_time = word.start

        # Find the speaker interval this word belongs to
        active_speakers_for_word = set()
        for i in range(len(speaker_intervals) - 1):
            if speaker_intervals[i]["time"] <= word_compare_time < speaker_intervals[i + 1]["time"]:
                active_speakers_for_word = speaker_intervals[i]["speakers"]
                break

        word_speaker.append(WordTupleSpeaker(word=word, speaker=active_speakers_for_word))

    word_speaker: List[WordTupleSpeaker] = []

    for word in transcript.word_timings:
        speaker_matching = [
            (s, max(word_speaker_match(word, f, t) for f, t in diarization.get_diarization(speaker=s))) for s in
            diarization.speaker
        ]
        good_speakers, overlap = set(), 1
        while len(good_speakers) <= 0 and overlap > 0:
            good_speakers = set(x for x, g in speaker_matching if g >= overlap)
            overlap -= .1
        word_speaker.append(WordTupleSpeaker(word=word, speaker=good_speakers))

    return word_speaker


def squish_word_speaker(word_speaker: List[WordTupleSpeaker]) -> List[WordTupleSpeaker]:
    current_utterance = {
        "speakers": set(),
        "start": None,
        "end": None,
        "text": ""
    }

    utterances = []

    for word in word_speaker:
        # If this is a new utterance or speakers have changed
        if current_utterance["start"] is None or word.speaker != current_utterance["speakers"]:

            # Save previous utterance if it exists
            if current_utterance["start"] is not None:
                current_utterance["text"] = current_utterance["text"].strip()
                utterances.append(current_utterance)

            # Start a new utterance
            current_utterance = {
                "speakers": word.speaker,
                "start": word.word.start,
                "end": word.word.end,
                "text": word.word.word + " "
            }
        else:
            # Continue the current utterance
            current_utterance["end"] = word.word.end
            current_utterance["text"] += word.word.word + " "

    # Add the last utterance
    if current_utterance["start"] is not None:
        current_utterance["text"] = current_utterance["text"].strip()
        utterances.append(current_utterance)

    return [WordTupleSpeaker(word=WordTuple(start=x["start"], end=x["end"], word=x["text"]), speaker=x["speakers"]) for
            x in utterances]


def word_speaker_to_transcript(word_speaker: List[WordTupleSpeaker]) -> List[str]:
    lines = []

    for line in word_speaker:
        speaker = " & ".join(sorted("<Unknown>" if x is None else x for x in line.speaker)) if len(line.speaker) > 0 else "<Unknown>"
        timeing = f"{line.word.start} - {line.word.end}"
        lines.append(f"{speaker} [{timeing}]: {line.word.word}")

    return lines
