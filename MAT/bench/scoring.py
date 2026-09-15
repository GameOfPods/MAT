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
"""Scores a MAT podcast result against an item's reference, or against the result of another system."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from MAT.bench.data import UNKNOWN_SPEAKER, Item, Turn
from MAT.bench.metrics import (
    DiarizationErrors, WordErrors, cp_word_errors, diarization_errors, midpoint_in, normalize, word_errors,
)


@dataclass
class Score:
    wer: Optional[WordErrors] = None
    cpwer: Optional[WordErrors] = None
    der: Optional[DiarizationErrors] = None
    reference_speakers: Optional[int] = None
    hypothesis_speakers: Optional[int] = None


def duration_of(podcast) -> float:
    if podcast.media is not None:
        return podcast.media.duration
    ends = [w.end for w in podcast.words if w.end is not None]
    ends += [r.end for s in podcast.speakers for r in s.segments]
    return max(ends, default=0.0)


def hypothesis_segments(podcast) -> Dict[str, List[Tuple[float, float]]]:
    return {s.id: [(r.start, r.end) for r in s.segments] for s in podcast.speakers}


def _speakers_in(segments: Dict[str, Sequence[Tuple[float, float]]], start: float, end: float) -> int:
    return sum(1 for ranges in segments.values() if any(e > start and s < end for s, e in ranges))


def score(item: Item, podcast, collar: float = 0.0) -> Score:
    language = item.language or podcast.language
    window_start = item.start if item.start is not None else 0.0
    window_end = item.end if item.end is not None else duration_of(podcast)
    hypothesis = hypothesis_segments(podcast)
    result = Score(hypothesis_speakers=_speakers_in(hypothesis, window_start, window_end))

    if item.has_words:
        turns = [t for t in item.turns if midpoint_in(t.start, t.end, item.start, item.end)]
        words = [w for w in podcast.words if midpoint_in(w.start, w.end, item.start, item.end)]
        result.wer = word_errors([x for t in turns for x in normalize(t.text, language)],
                                 [x for w in words for x in normalize(w.text, language)])
        if item.has_speakers:
            reference_words: Dict[str, List[str]] = {}
            hypothesis_words: Dict[str, List[str]] = {}
            for turn in turns:
                speaker = turn.speakers[0] if turn.speakers else UNKNOWN_SPEAKER
                reference_words.setdefault(speaker, []).extend(normalize(turn.text, language))
            for word in words:
                # a word during overlapping speech counts for the first of its speakers only
                speaker = word.speakers[0] if word.speakers else UNKNOWN_SPEAKER
                hypothesis_words.setdefault(speaker, []).extend(normalize(word.text, language))
            result.cpwer = cp_word_errors(reference_words, hypothesis_words)[0]

    if item.has_speakers:
        reference = item.reference_segments()
        result.der = diarization_errors(reference, hypothesis, window_start, window_end, collar)
        result.reference_speakers = _speakers_in(reference, window_start, window_end)
    return result


def agreement(item: Item, podcast, baseline, collar: float = 0.0) -> Score:
    """Scores podcast as if the baseline result (another system) were the reference."""
    pseudo = Item(dataset=item.dataset, id=item.id, audio=item.audio, language=item.language or baseline.language,
                  turns=[Turn(w.start, w.end, tuple(w.speakers[:1]), w.text) for w in baseline.words],
                  speaker_segments=hypothesis_segments(baseline), has_words=True, has_speakers=True,
                  start=item.start, end=item.end)
    return score(pseudo, podcast, collar)


__all__ = ["Score", "score", "agreement", "duration_of", "hypothesis_segments"]
