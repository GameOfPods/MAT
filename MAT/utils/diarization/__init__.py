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
import bisect
from typing import List, Optional, Sequence, Tuple

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


# a word ending with one of these ends a sentence, so a short line after it can be a real answer ("Ja.")
_SENTENCE_END = (".", "?", "!", "…", ":", ";")


class _Timeline:
    """The segments of one speaker, overlaps joined, for quick lookups by time."""

    def __init__(self, segments: Sequence[Tuple[float, float]]):
        merged: List[List[float]] = []
        for start, end in sorted(segments):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        self.starts = [x[0] for x in merged]
        self.ends = [x[1] for x in merged]

    def overlap(self, start: float, end: float) -> float:
        total, i = 0.0, bisect.bisect_left(self.starts, end) - 1
        while i >= 0 and self.ends[i] > start:
            total += min(end, self.ends[i]) - max(start, self.starts[i])
            i -= 1
        return total

    def distance(self, start: float, end: float) -> float:
        i = bisect.bisect_left(self.starts, end)
        best = float("inf")
        if i < len(self.starts):
            best = self.starts[i] - end
        if i > 0:
            best = min(best, max(0.0, start - self.ends[i - 1]))
        return best


def assign_speakers(diarization: DiarizationResult, words: Sequence[WordTuple], max_gap: float = 1.0,
                    min_turn: float = 0.5) -> List[WordTupleSpeaker]:
    """
    One speaker per word. A word goes to the speaker who talks longest during it, a tie (overlapping speech) goes to
    whoever said the word before. Words outside every segment take the speaker around them, or the closest segment
    within max_gap seconds. A turn shorter than min_turn seconds inside another speaker's sentence goes back to that
    speaker, which removes the one word fragments at speaker changes. The set is only empty when nothing fits.
    """
    timelines = {speaker: _Timeline(diarization.get_diarization(speaker)) for speaker in sorted(diarization.speaker)}
    chosen: List[Optional[str]] = []
    previous: Optional[str] = None
    for word in words:
        speaker = None
        if word.start is not None and word.end is not None and timelines:
            start, end = word.start, max(word.end, word.start + 0.01)
            overlaps = {s: t.overlap(start, end) for s, t in timelines.items()}
            best = max(overlaps.values())
            if best > 0:
                # both talk during the whole word (overlapping segments): keep the one who was already talking
                tied = [s for s, value in overlaps.items() if value >= best]
                speaker = previous if previous in tied else tied[0]
        chosen.append(speaker)
        if speaker is not None:
            previous = speaker

    # words nobody talks during: same speaker on both sides wins, else the closest segment
    before: List[Optional[str]] = []
    for speaker in chosen:
        before.append(speaker if speaker is not None else (before[-1] if before else None))
    after: List[Optional[str]] = []
    for speaker in reversed(chosen):
        after.append(speaker if speaker is not None else (after[-1] if after else None))
    after.reverse()
    for i, word in enumerate(words):
        if chosen[i] is not None:
            continue
        if before[i] is not None and before[i] == after[i]:
            chosen[i] = before[i]
        elif word.start is not None and word.end is not None and timelines:
            distance, speaker = min((t.distance(word.start, word.end), s) for s, t in timelines.items())
            if distance <= max_gap:
                chosen[i] = speaker

    # short turns inside a sentence of someone else
    runs: List[List[int]] = []
    for i, speaker in enumerate(chosen):
        if runs and chosen[runs[-1][0]] == speaker:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    for k in range(1, len(runs) - 1):
        first, last = runs[k]
        around = chosen[runs[k - 1][1]]
        if around is None or around != chosen[runs[k + 1][0]] or chosen[first] == around:
            continue
        before = (words[first - 1].word or "").strip()
        if before.endswith(_SENTENCE_END):
            continue
        start, end = words[first].start, words[last].end
        if start is None or end is None or end - start < min_turn:
            for i in range(first, last + 1):
                chosen[i] = around

    return [WordTupleSpeaker(word=word, speaker=set() if speaker is None else {speaker})
            for word, speaker in zip(words, chosen)]


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
