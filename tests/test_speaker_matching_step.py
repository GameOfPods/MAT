import pytest

from MAT import registry
from MAT.pipelines import PipelineStepInput, PipelineStepResult
from MAT.pipelines.Podcast import PodcastPipeline
from MAT.tools.diarizators import DiarizationResult
from MAT.tools.speakeridentification import SpeakerIdentificationResult, SpeakerIdentificationTool
from MAT.utils.config import Config


@registry.register("identifier", "test-nothing-to-match")
class NothingToMatch(SpeakerIdentificationTool):
    def can_match(self, config):
        return False

    def process(self, origin_data, config):
        raise AssertionError("must not run without something to match against")


@registry.register("identifier", "test-matching")
class AlwaysMatches(SpeakerIdentificationTool):
    def process(self, origin_data, config):
        return SpeakerIdentificationResult("alice")


def teardown_module():
    registry.unregister("identifier", "test-nothing-to-match")
    registry.unregister("identifier", "test-matching")


def _match(identifier: str):
    diarization = DiarizationResult({"sprecher_0": [(0.0, 1.0)]})
    pipeline = PodcastPipeline()
    step = next(s for s in pipeline._get_steps() if s.__name__ == "speaker_matching")
    previous = {"Diarization": PipelineStepResult(name="Diarization", data=diarization)}
    # the file doesn't exist, so decoding the audio would raise
    result = step(PipelineStepInput(file="not-here.mp3", config=Config({"podcast": {"identifier": identifier}}),
                                    previous_results=previous))
    return pipeline, diarization, result


def test_identifier_without_gold_labels_is_skipped():
    pipeline, diarization, result = _match("test-nothing-to-match")
    assert result.data is diarization
    # it didn't run, so it doesn't belong in the models of the result either
    assert "identifier" not in pipeline.models


def test_identifier_none_keeps_the_diarizer_names():
    pipeline, diarization, result = _match("none")
    assert result.data is diarization


def test_identifier_that_matches_reads_the_audio():
    with pytest.raises(Exception):
        _match("test-matching")


def test_pyannote_can_match_depends_on_gold_labels(tmp_path):
    from MAT.tools.speakeridentification.pyannote import SpeakerIdetificationPyannote

    identifier = SpeakerIdetificationPyannote()
    assert identifier.can_match(Config({})) is False
    assert identifier.can_match(Config({"pyannote": {"gold-labels": str(tmp_path)}})) is True
