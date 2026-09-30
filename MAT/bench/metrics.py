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
Text normalization, WER, cpWER and DER.

WER uses jiwer, DER pyannote.metrics (both in the `bench` extra). cpWER is computed here: every reference speaker
gets the hypothesis speaker that gives the fewest word errors (Hungarian algorithm from scipy), which is the same
definition meeteval uses, without its compiled dependencies.
"""
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# Hesitations are dropped from reference and hypothesis. Whisper mostly leaves them out, the AMI references keep them.
FILLERS = {
    "en": {"uh", "um", "uhm", "umm", "hmm", "hm", "mm", "mhm", "mmhmm", "ah", "eh", "er", "erm"},
    "de": {"äh", "ähm", "öh", "öhm", "eh", "ehm", "hm", "hmm", "mhm", "mm", "uh", "um"},
}
_ALL_FILLERS = set().union(*FILLERS.values())
_APOSTROPHES = "'’ʼ`´"
_SPLIT = re.compile(r"[-‐‑‒–—/]")


def normalize(text: Optional[str], language: Optional[str] = None) -> List[str]:
    """Lowercase words without punctuation and fillers. Numbers are left as they are (5 and five stay different)."""
    text = unicodedata.normalize("NFKC", text or "").casefold()
    for apostrophe in _APOSTROPHES:
        text = text.replace(apostrophe, "")
    text = _SPLIT.sub(" ", text)
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    fillers = FILLERS.get((language or "")[:2].lower(), _ALL_FILLERS)
    return [word for word in text.split() if word not in fillers]


def midpoint_in(start: Optional[float], end: Optional[float],
                window_start: Optional[float], window_end: Optional[float]) -> bool:
    """Whether something from start to end belongs to the window. Without a window everything does."""
    if window_start is None and window_end is None:
        return True
    if start is None or end is None:
        return False
    middle = (start + end) / 2
    return (window_start is None or middle >= window_start) and (window_end is None or middle < window_end)


@dataclass
class WordErrors:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_words: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> Optional[float]:
        return None if self.reference_words == 0 else self.errors / self.reference_words

    def __add__(self, other: "WordErrors") -> "WordErrors":
        return WordErrors(self.substitutions + other.substitutions, self.deletions + other.deletions,
                          self.insertions + other.insertions, self.reference_words + other.reference_words)


def word_errors(reference: Sequence[str], hypothesis: Sequence[str]) -> WordErrors:
    if not reference:
        return WordErrors(insertions=len(hypothesis))
    if not hypothesis:
        return WordErrors(deletions=len(reference), reference_words=len(reference))
    import jiwer

    out = jiwer.process_words(" ".join(reference), " ".join(hypothesis))
    return WordErrors(out.substitutions, out.deletions, out.insertions, len(reference))


def cp_word_errors(reference: Dict[str, List[str]],
                   hypothesis: Dict[str, List[str]]) -> Tuple[WordErrors, Dict[str, Optional[str]]]:
    """cpWER counts plus which hypothesis speaker got assigned to each reference speaker (None if none was left)."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    references, hypotheses = sorted(reference), sorted(hypothesis)
    size = max(len(references), len(hypotheses))
    if size == 0:
        return WordErrors(), {}
    table: Dict[Tuple[int, int], WordErrors] = {}
    cost = np.zeros((size, size))
    for i in range(size):
        ref_words = reference[references[i]] if i < len(references) else []
        for j in range(size):
            hyp_words = hypothesis[hypotheses[j]] if j < len(hypotheses) else []
            table[i, j] = word_errors(ref_words, hyp_words)
            cost[i, j] = table[i, j].errors
    total, mapping = WordErrors(), {}
    for i, j in zip(*linear_sum_assignment(cost)):
        total += table[i, j]
        if i < len(references):
            mapping[references[i]] = hypotheses[j] if j < len(hypotheses) else None
    return total, mapping


@dataclass
class DiarizationErrors:
    missed: float = 0.0
    false_alarm: float = 0.0
    confusion: float = 0.0
    # seconds of reference speech, overlapping speakers count once per speaker
    total: float = 0.0

    @property
    def errors(self) -> float:
        return self.missed + self.false_alarm + self.confusion

    @property
    def rate(self) -> Optional[float]:
        return None if self.total <= 0 else self.errors / self.total

    def __add__(self, other: "DiarizationErrors") -> "DiarizationErrors":
        return DiarizationErrors(self.missed + other.missed, self.false_alarm + other.false_alarm,
                                 self.confusion + other.confusion, self.total + other.total)


def _annotation(segments: Dict[str, Sequence[Tuple[float, float]]]):
    from pyannote.core import Annotation, Segment

    annotation = Annotation()
    for speaker, ranges in segments.items():
        for index, (start, end) in enumerate(ranges):
            if end > start:
                annotation[Segment(start, end), f"{speaker}#{index}"] = speaker
    return annotation


def diarization_errors(reference: Dict[str, Sequence[Tuple[float, float]]],
                       hypothesis: Dict[str, Sequence[Tuple[float, float]]],
                       start: float, end: float, collar: float = 0.0) -> DiarizationErrors:
    """DER parts between start and end, overlapping speech is scored."""
    if end <= start:
        return DiarizationErrors()
    from pyannote.core import Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
    details = metric(_annotation(reference), _annotation(hypothesis), uem=Timeline([Segment(start, end)]),
                     detailed=True)
    return DiarizationErrors(missed=details["missed detection"], false_alarm=details["false alarm"],
                             confusion=details["confusion"], total=details["total"])


__all__ = ["FILLERS", "normalize", "midpoint_in", "WordErrors", "word_errors", "cp_word_errors",
           "DiarizationErrors", "diarization_errors"]
