from MAT import registry
from MAT.pipelines import PipelineStepInput, PipelineStepResult
from MAT.pipelines.Podcast import PodcastPipeline
from MAT.tools import SummaryTool
from MAT.utils.config import Config


@registry.register("summarizer", "test-failing")
class FailingSummary(SummaryTool):
    def process(self, origin_data, config):
        raise ValueError("We were unable to start processing your request within the 900-second timeout limit.")


def teardown_module():
    registry.unregister("summarizer", "test-failing")


def _summarize(config):
    summarize = next(step for step in PodcastPipeline()._get_steps() if step.__name__ == "summarize_transcript")
    previous = {"Finalizing transcript": PipelineStepResult(name="Finalizing transcript", data=([], [], "a: hello"))}
    return summarize(PipelineStepInput(file="episode.mp3", config=config, previous_results=previous))


def test_failing_summary_keeps_the_episode():
    result = _summarize(Config({"podcast": {"summarizer": "test-failing"}}))
    assert result.name == "Summarize transcript"
    assert result.data is None


def test_summarizer_none_is_skipped():
    assert _summarize(Config({"podcast": {"summarizer": "none"}})).data is None
