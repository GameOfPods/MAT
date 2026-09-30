import pytest

from MAT.pipelines import PipelineStepInput
from MAT.pipelines.Podcast import PodcastPipeline, load_vocabulary
from MAT.tools import TranscriptionResult
from MAT.utils.config import Config, ConfigError


def test_vocabulary_from_a_list_or_a_file(tmp_path):
    assert load_vocabulary(None) == []
    assert load_vocabulary([" Samwell ", "", "Azor Ahai"]) == ["Samwell", "Azor Ahai"]
    words = tmp_path / "show.txt"
    words.write_text("# Game of Pods\nSamwell\n\nAzor Ahai  # the prophecy\n", encoding="utf-8")
    assert load_vocabulary(str(words)) == ["Samwell", "Azor Ahai"]


def test_a_missing_vocabulary_file_stops_the_run_early(tmp_path):
    config = Config({"podcast": {"vocabulary": str(tmp_path / "nope.txt"), "diarizer": "sortformer",
                                 "summarizer": "none", "identifier": "none"}})
    with pytest.raises(ConfigError, match="vocabulary"):
        PodcastPipeline.preflight(config)


def test_the_transcriber_and_the_summary_get_the_vocabulary(monkeypatch, tmp_path):
    from MAT import registry
    from MAT.tools.summary import SummaryTool
    from MAT.tools.transcriptors import TransciptionTool

    seen = {}

    @registry.register("transcriber", "test-vocabulary")
    class Transcriber(TransciptionTool):
        def process(self, origin_data, config):
            seen["vocabulary"] = origin_data.vocabulary
            return TranscriptionResult(word_timings=[], language="de", duration=1.0)

    @registry.register("summarizer", "test-vocabulary")
    class Summarizer(SummaryTool):
        def process(self, origin_data, config):
            seen["metadata"] = origin_data.additional_metadata
            return None

    try:
        config = Config({"podcast": {"vocabulary": ["Samwell", "Azor Ahai"], "transcriber": "test-vocabulary",
                                     "summarizer": "test-vocabulary"}})
        pipeline = PodcastPipeline()
        steps = {step.__name__: step for step in pipeline._get_steps()}
        steps["transcribe"](PipelineStepInput(file="episode.mp3", config=config, previous_results={}))
        assert seen["vocabulary"] == ["Samwell", "Azor Ahai"]

        from MAT.pipelines import PipelineStepResult
        from MAT.tools.diarizators import DiarizationResult
        previous = {"Finalizing transcript": PipelineStepResult("Finalizing transcript", ([], [], "a [0 - 1]: hi")),
                    "Speaker Matching": PipelineStepResult("Speaker Matching", DiarizationResult({"a": [(0, 1)]}))}
        steps["summarize_transcript"](PipelineStepInput(file="episode.mp3", config=config, previous_results=previous))
        assert "Samwell, Azor Ahai" in seen["metadata"].values()
    finally:
        registry.unregister("transcriber", "test-vocabulary")
        registry.unregister("summarizer", "test-vocabulary")
