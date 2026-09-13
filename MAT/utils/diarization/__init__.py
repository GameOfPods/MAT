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
    word_speaker: List[WordTupleSpeaker] = []

    # For every word take the speakers with the biggest time overlap. Start at full overlap and lower the bar
    # in 0.1 steps until at least one speaker matches.
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
