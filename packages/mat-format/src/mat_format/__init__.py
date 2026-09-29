"""
Data model, JSON schemas and reader for MAT results.

    from mat_format import MATResult
    result = MATResult.read("results/episode_2026-09-15_20-15-02.zip")
"""
from mat_format.models import (
    FORMAT_VERSION, BookResult, Chapter, Character, Event, FailedStep, Input, Media, Meta, ModelInfo, PodcastResult,
    Sentence, Speaker, TextEntity, TimeRange, TranscriptEntity, Word,
)
from mat_format.reader import MATResult


def _version() -> str:
    """From the installed package, which takes it from pyproject.toml. A source checkout reads the file itself."""
    from importlib import metadata
    from pathlib import Path

    try:
        return metadata.version("mat-format")
    except metadata.PackageNotFoundError:
        import tomllib

        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        return tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]


__version__ = _version()

__all__ = ["FORMAT_VERSION", "MATResult", "Meta", "FailedStep", "Input", "ModelInfo", "PodcastResult",
           "Media", "Speaker", "TimeRange", "Word", "Event", "TranscriptEntity", "BookResult", "Chapter", "Character",
           "Sentence", "TextEntity"]
