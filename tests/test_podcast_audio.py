import os

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from MAT.pipelines import PipelineStepInput, PipelineStepResult
from MAT.pipelines.Podcast import PodcastPipeline
from MAT.utils.config import Config


@pytest.fixture
def stereo_mp3(tmp_path):
    tone = Sine(440).to_audio_segment(duration=2000).set_frame_rate(44100)
    path = tmp_path / "episode.mp3"
    AudioSegment.from_mono_audiosegments(tone, tone).export(str(path), format="mp3")
    return path


def test_accept_probes_instead_of_decoding(stereo_mp3, tmp_path):
    assert PodcastPipeline.accept(str(stereo_mp3))
    text = tmp_path / "notes.mp3"
    text.write_text("not audio")
    reason = PodcastPipeline.why_not(str(text))
    assert reason is not None and reason.startswith("ffmpeg can't read it as audio")


def test_audio_is_decoded_once_into_a_16_khz_mono_wav(stereo_mp3, tmp_path):
    pipeline = PodcastPipeline()
    steps = {step.__name__: step for step in pipeline._get_steps()}
    config = Config({}, work_directory=str(tmp_path / "work"))
    result = steps["prepare_audio"](PipelineStepInput(file=str(stereo_mp3), config=config, previous_results={}))
    wav = AudioSegment.from_file(result.data["path"])
    assert (wav.frame_rate, wav.channels) == (16000, 1)
    # media info describes the original file, not the working copy
    assert result.data["sample_rate"] == 44100 and abs(result.data["duration"] - 2.0) < 0.1

    # the working copy goes away with the result
    pipeline._finalize_result({"Audio": PipelineStepResult("Audio", result.data)})
    assert not os.path.exists(result.data["path"]) and pipeline._segment is None
