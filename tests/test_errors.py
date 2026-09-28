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
