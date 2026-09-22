import json

import pytest
from pydub.generators import Sine

from MAT.pipelines import PipelineStepInput, PipelineStepResult
from MAT.pipelines.Podcast import PodcastPipeline
from MAT.tools import WordTuple, WordTupleSpeaker
from MAT.tools.diarizators import DiarizationResult
from MAT.utils.config import Config
from MAT.utils.speaker_library import SpeakerLibrary

ALEX = [1.0, 0.0, 0.0]
TOBI = [0.0, 1.0, 0.0]


@pytest.fixture
def episode(tmp_path):
    path = tmp_path / "episode.wav"
    Sine(440).to_audio_segment(duration=12000).export(str(path), format="wav")
    return path


def _voices(vectors):
    """Stands in for the embedding model, keeps the real config section of the pyannote identifier."""
    from MAT.tools.speakeridentification.pyannote import SpeakerIdetificationPyannote

    class FakeEmbedder(SpeakerIdetificationPyannote):
        seen = {}

        @staticmethod
        def embeddings(model, audios, device="cpu", use_hf_token=True):
            FakeEmbedder.seen["count"] = len(audios)
            FakeEmbedder.seen["seconds"] = [round(clip.duration_seconds, 1) for clip, _ in audios]
            FakeEmbedder.seen["model"] = model
            return list(vectors)

    return FakeEmbedder


def _run(episode, tmp_path, monkeypatch, *, matched, raw, named=None, vectors, values=None):
    embedder = _voices(vectors)
    pipeline = PodcastPipeline()
    step = next(s for s in pipeline._get_steps() if s.__name__ == "speaker_library")

    import MAT.tools.speakeridentification.pyannote as module

    monkeypatch.setattr(module, "SpeakerIdetificationPyannote", embedder)

    words = [WordTupleSpeaker(word=WordTuple(start=0.0, end=2.0, word="hallo"), speaker={s})
             for s in sorted(matched.speaker)]
    transcript = "\n".join(f"{s} [0.0 - 2.0]: hallo" for s in sorted(matched.speaker))
    previous = {
        "Diarization": PipelineStepResult(name="Diarization", data=raw),
        "Speaker Matching": PipelineStepResult(name="Speaker Matching", data=matched),
        "Finalizing transcript": PipelineStepResult(name="Finalizing transcript", data=(words, words, transcript)),
        "Speaker Names": PipelineStepResult(name="Speaker Names", data=named),
    }
    settings = {"speaker-library": str(tmp_path / "library"), "speaker-library-threshold": 0.8}
    settings.update(values or {})
    config = Config({"podcast": settings})
    return step(PipelineStepInput(file=str(episode), config=config, previous_results=previous)), embedder


def test_a_known_voice_gets_its_name_back(episode, tmp_path, monkeypatch):
    library = SpeakerLibrary.open(tmp_path / "library")
    library.remember("Alex", ALEX, source="gold", episode="older.mp3")
    library.save()

    raw = DiarizationResult({"sprecher_0": [(0.0, 5.0)]})
    result, _ = _run(episode, tmp_path, monkeypatch, matched=raw, raw=raw, vectors=[ALEX])

    assert sorted(result.data["diarization"].speaker) == ["Alex"]
    assert result.data["transcript"].startswith("Alex [0.0 - 2.0]")
    assert result.data["speakers"]["Alex"]["library_id"].startswith("alex-")


def test_a_gold_name_is_never_overwritten_by_the_library(episode, tmp_path, monkeypatch):
    library = SpeakerLibrary.open(tmp_path / "library")
    library.remember("Chris", ALEX, source="gold")
    library.save()

    raw = DiarizationResult({"sprecher_0": [(0.0, 5.0)]})
    matched = DiarizationResult({"alex": [(0.0, 5.0)]})  # a gold clip matched this voice
    result, _ = _run(episode, tmp_path, monkeypatch, matched=matched, raw=raw, vectors=[ALEX])

    assert sorted(result.data["diarization"].speaker) == ["alex"]


def test_gold_names_are_learned_and_unnamed_speakers_are_not(episode, tmp_path, monkeypatch):
    raw = DiarizationResult({"sprecher_1": [(0.0, 5.0)]})
    matched = DiarizationResult({"alex": [(0.0, 5.0)], "sprecher_1": [(5.0, 9.0)]})
    result, _ = _run(episode, tmp_path, monkeypatch, matched=matched, raw=raw, vectors=[ALEX, TOBI])

    stored = json.loads((tmp_path / "library" / "speakers.json").read_text())["speakers"]
    assert [speaker["name"] for speaker in stored] == ["alex"]
    assert stored[0]["source"] == "gold" and stored[0]["episodes"] == ["episode.wav"]
    assert "sprecher_1" not in result.data["speakers"]


def test_llm_names_are_only_learned_when_allowed(episode, tmp_path, monkeypatch):
    raw = DiarizationResult({"sprecher_0": [(0.0, 5.0)]})
    # the naming step renamed the speaker from the transcript
    named = (DiarizationResult({"Tobi": [(0.0, 5.0)]}), [], [], "Tobi [0.0 - 2.0]: hallo")
    matched = raw

    _run(episode, tmp_path, monkeypatch, matched=matched, raw=raw, named=named, vectors=[TOBI])
    assert not (tmp_path / "library" / "speakers.json").exists()

    _run(episode, tmp_path, monkeypatch, matched=matched, raw=raw, named=named, vectors=[TOBI],
         values={"speaker-library-learns": "all"})
    stored = json.loads((tmp_path / "library" / "speakers.json").read_text())["speakers"]
    assert [(s["name"], s["source"]) for s in stored] == [("Tobi", "llm")]


def test_no_library_folder_means_the_step_does_nothing(episode, tmp_path, monkeypatch):
    raw = DiarizationResult({"sprecher_0": [(0.0, 5.0)]})
    result, _ = _run(episode, tmp_path, monkeypatch, matched=raw, raw=raw, vectors=[ALEX],
                     values={"speaker-library": None})
    assert result.data is None


def test_only_a_slice_of_each_speaker_is_listened_to(episode, tmp_path, monkeypatch):
    raw = DiarizationResult({"sprecher_0": [(0.0, 4.0), (4.0, 8.0), (8.0, 12.0)]})
    _, embedder = _run(episode, tmp_path, monkeypatch, matched=raw, raw=raw, vectors=[ALEX],
                       values={"speaker-library-seconds": 5})
    assert embedder.seen["count"] == 1
    # stops after the segment that crosses the limit instead of taking all 12 seconds
    assert embedder.seen["seconds"] == [8.0]
