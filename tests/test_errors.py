"""The messages a user gets when something outside MAT is missing or too small."""
import pytest

from MAT.utils.config import Config, ConfigError
from MAT.utils.device import GpuOutOfMemory, is_out_of_memory


class _Backend:
    backend_name = "whisper"
    memory_hint = "a smaller model (--set whisper.model=medium)"


def _failing_pipeline(error):
    """The podcast pipeline with one step that fails while a backend is running. Not a subclass, those would be
    picked up as real pipelines by every later test."""
    from MAT.pipelines import PodcastPipeline

    pipeline = PodcastPipeline()

    def transcribe(step_input):
        pipeline._running = _Backend()
        raise error

    pipeline._get_steps = lambda: [transcribe]
    return pipeline


@pytest.mark.parametrize("error", [
    RuntimeError("CUDA failed with error out of memory"),
    MemoryError("batch_size (32) is probably too large, try a smaller one"),
])
def test_out_of_memory_names_the_backend_and_the_fix(error):
    assert is_out_of_memory(error)
    with pytest.raises(GpuOutOfMemory) as caught:
        _failing_pipeline(error).process(file="episode.mp3", config=Config({}))
    message = str(caught.value)
    assert message.startswith("whisper ran out of GPU memory")
    assert "whisper.model=medium" in message and "OLLAMA_KEEP_ALIVE" in message


def test_other_errors_stay_what_they_are():
    with pytest.raises(ValueError, match="broken"):
        _failing_pipeline(ValueError("broken")).process(file="episode.mp3", config=Config({}))


def test_summary_without_api_key_stops_before_the_first_file(monkeypatch):
    from MAT.tools.summary.llm import SummaryLLM

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY.*--summarizer none"):
        SummaryLLM.preflight(Config({}))
    monkeypatch.setenv("OPENAI_API_KEY", "anything")
    SummaryLLM.preflight(Config({}))


def test_ollama_that_is_down_or_lacks_the_model(monkeypatch):
    import requests

    from MAT.tools.speakernaming.llm import SpeakerNamingLLM
    from MAT.tools.summary.llm import SummaryLLM

    def down(url, timeout=None):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", down)
    config = Config({"llm": {"preset": "ollama", "model": "qwen3:8b"}})
    with pytest.raises(ConfigError, match="doesn't answer.*ollama serve"):
        SummaryLLM.preflight(config)
    # the namer follows [llm], so it checks the same server
    with pytest.raises(ConfigError, match="Speaker naming uses Ollama"):
        SpeakerNamingLLM.preflight(config)

    class Tags:
        def raise_for_status(self):
            pass

        def json(self):
            return {"models": [{"name": "llama3:latest"}]}

    monkeypatch.setattr(requests, "get", lambda url, timeout=None: Tags())
    with pytest.raises(ConfigError, match="ollama pull qwen3:8b"):
        SummaryLLM.preflight(config)
    SummaryLLM.preflight(Config({"llm": {"preset": "ollama", "model": "llama3"}}))


def _pipeline_with(steps):
    from MAT.pipelines import PodcastPipeline

    pipeline = PodcastPipeline()
    pipeline._get_steps = lambda: steps
    pipeline._finalize_result = lambda step_results: type("Result", (), {"steps": dict(step_results)})()
    return pipeline


def test_an_optional_step_that_fails_is_left_out():
    from MAT.pipelines import PipelineStepResult

    def transcribe(step_input):
        return PipelineStepResult("Transcription", "words")

    def summarize_transcript(step_input):
        raise RuntimeError("the LLM is gone")

    def media_infos(step_input):
        return PipelineStepResult("Media Info", "info")

    pipeline = _pipeline_with([transcribe, summarize_transcript, media_infos])
    result = pipeline.process(file="episode.mp3", config=Config({}))
    assert sorted(result.steps) == ["Media Info", "Transcription"]
    assert result.failed_steps == [{"step": "summarize_transcript", "error": "RuntimeError: the LLM is gone"}]


def test_a_required_step_that_fails_fails_the_file():
    def transcribe(step_input):
        raise RuntimeError("no model")

    with pytest.raises(RuntimeError, match="no model"):
        _pipeline_with([transcribe]).process(file="episode.mp3", config=Config({}))


def test_failed_steps_end_up_in_meta_json(tmp_path):
    from mat_format import MATResult

    from MAT.writer import Writer
    from tests.test_result_format import podcast_output

    output = podcast_output()
    output.failed_steps = [{"step": "find_events", "error": "RuntimeError: boom"}]
    episode = tmp_path / "episode.wav"
    episode.write_bytes(b"x")
    folder = Writer().store(file=str(episode), output=str(tmp_path / "out"), pipeline_results=[output])
    meta = MATResult.read(folder).meta
    assert [(f.pipeline, f.step, f.error) for f in meta.failed_steps] == [("podcast", "find_events", "RuntimeError: boom")]
