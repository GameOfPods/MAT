import pydub
import pytest

from MAT import registry
from MAT.pipelines.Podcast import PodcastPipeline
from MAT.tools import (DiarizationResult, DiarizationTool, SummaryResult, SummaryTool, TranscribeDiarizeResult,
                       TranscribeDiarizeTool, WordTuple)
from MAT.utils.config import Config


@registry.register("transcriber", "test-joint", description="fake model that transcribes and diarizes")
class JointTranscriber(TranscribeDiarizeTool):
    def process(self, origin_data, config):
        return TranscribeDiarizeResult(
            word_timings=[WordTuple(0.1, 0.5, "hello"), WordTuple(1.1, 1.5, "there")],
            language="en", duration=2.0, duration_after_vad=2.0,
            diarization=DiarizationResult({"host": [(0.0, 1.0)], "guest": [(1.0, 2.0)]}),
        )


@registry.register("diarizer", "test-never")
class NeverDiarizer(DiarizationTool):
    def process(self, origin_data, config):
        raise AssertionError("the diarizer must not run when the transcriber diarizes")


@registry.register("summarizer", "test-echo")
class EchoSummarizer(SummaryTool):
    def process(self, origin_data, config):
        return SummaryResult("summary of " + next(origin_data.text).splitlines()[0])


def teardown_module():
    for slot, name in [("transcriber", "test-joint"), ("diarizer", "test-never"), ("summarizer", "test-echo")]:
        registry.unregister(slot, name)


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "episode.wav"
    pydub.AudioSegment.silent(duration=2000, frame_rate=16000).export(path, format="wav")
    return path


def _config(tmp_path, **podcast):
    values = {"transcriber": "test-joint", "diarizer": "test-never", "identifier": "none", "summarizer": "test-echo"}
    values.update(podcast)
    config = Config({"podcast": values}, work_directory=str(tmp_path / "work"))
    config.validate()
    return config


def test_joint_transcriber_replaces_the_diarizer(tmp_path, audio):
    result = PodcastPipeline().process(file=str(audio), config=_config(tmp_path))

    assert result.diarization_matched.speaker == {"host", "guest"}
    assert result.full_transcript.splitlines() == ["host [0.1 - 0.5]: hello", "guest [1.1 - 1.5]: there"]
    assert list(result.summary.text) == ["summary of host [0.1 - 0.5]: hello"]
    assert result.models["transcriber"]["backend"] == "test-joint"
    assert result.models["summarizer"]["backend"] == "test-echo"
    # diarizer never ran, identifier and summarizer "none" are not listed
    assert set(result.models) == {"transcriber", "summarizer"}


def test_none_skips_the_summary(tmp_path, audio):
    result = PodcastPipeline().process(file=str(audio), config=_config(tmp_path, summarizer="none"))
    assert result.summary is None
    assert "summarizer" not in result.models
    assert result.full_transcript


def test_required_slot_cannot_be_none(tmp_path):
    from MAT.utils.config import ConfigError
    with pytest.raises(ConfigError, match='transcriber can\'t be "none"'):
        Config({"podcast": {"transcriber": "none"}}).validate()
