import MAT.pipelines.Podcast as podcast_module
from MAT.pipelines import PipelineStepInput, PipelineStepResult
from MAT.pipelines.Podcast import PodcastPipeline
from MAT.utils.config import Config


class FailingSummary:
    def process(self, origin_data, config):
        raise ValueError("We were unable to start processing your request within the 900-second timeout limit.")


def test_failing_summary_keeps_the_episode(monkeypatch):
    monkeypatch.setattr(podcast_module, "SummaryLLM", FailingSummary)
    summarize = next(step for step in PodcastPipeline()._get_steps() if step.__name__ == "summarize_transcript")
    previous = {"Finalizing transcript": PipelineStepResult(name="Finalizing transcript", data=([], [], "a: hello"))}

    result = summarize(PipelineStepInput(file="episode.mp3", config=Config(), previous_results=previous))

    assert result.name == "Summarize transcript"
    assert result.data is None
