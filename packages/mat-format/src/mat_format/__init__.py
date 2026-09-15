"""
Data model, JSON schemas and reader for MAT results.

    from mat_format import MATResult
    result = MATResult.read("results/episode_2026-09-15_20-15-02.zip")
"""
from mat_format.models import (
    FORMAT_VERSION, BookResult, Chapter, Event, Input, Media, Meta, ModelInfo, PodcastResult, Sentence, Speaker,
    TextEntity, TimeRange, TranscriptEntity, Word,
)
from mat_format.reader import MATResult

__all__ = ["FORMAT_VERSION", "MATResult", "Meta", "Input", "ModelInfo", "PodcastResult", "Media", "Speaker",
           "TimeRange", "Word", "Event", "TranscriptEntity", "BookResult", "Chapter", "Sentence", "TextEntity"]
