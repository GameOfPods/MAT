from MAT.utils.config import Config
from MAT.utils.step_cache import StepCache


class Backend:
    backend_name = "fake"

    class Options:
        pass

    @classmethod
    def effective_options(cls, config):
        from MAT.utils.config import Options

        class FakeOptions(Options):
            model: str = "a"

        return FakeOptions(**config.values.get("fake", {}))

    def describe(self, config):
        return {"backend": "fake", "packages": {"fake": "1.0"}}


def test_a_result_comes_back_for_the_same_file_and_options(tmp_path):
    episode = tmp_path / "episode.mp3"
    episode.write_bytes(b"audio")
    cache = StepCache(str(tmp_path / "cache"))
    key = StepCache.key(Backend(), Config({}), vocabulary=["Samwell"])
    assert cache.get(str(episode), "transcription", key) is None
    cache.put(str(episode), "transcription", key, {"words": 3})
    assert cache.get(str(episode), "transcription", key) == {"words": 3}

    # other options, other vocabulary or another file: run again
    assert cache.get(str(episode), "transcription", StepCache.key(Backend(), Config({"fake": {"model": "b"}}),
                                                                 vocabulary=["Samwell"])) is None
    assert cache.get(str(episode), "transcription", StepCache.key(Backend(), Config({}), vocabulary=[])) is None
    episode.write_bytes(b"other audio")
    assert cache.get(str(episode), "transcription", key) is None


def test_off_and_broken_files_just_mean_running_the_step(tmp_path):
    episode = tmp_path / "episode.mp3"
    episode.write_bytes(b"audio")
    off = StepCache(None)
    off.put(str(episode), "diarization", {}, "x")
    assert off.get(str(episode), "diarization", {}) is None

    cache = StepCache(str(tmp_path / "cache"))
    cache.put(str(episode), "diarization", {"a": 1}, "x")
    path = next((tmp_path / "cache").rglob("*.pkl"))
    path.write_bytes(b"not a pickle")
    assert cache.get(str(episode), "diarization", {"a": 1}) is None


def test_no_cache_flag_turns_it_off():
    from MAT.cli import _load_config, build_parser

    args = build_parser().parse_args(["run", "-i", "x", "-o", "y", "--no-cache"])
    from MAT.pipelines.Podcast import PodcastPipeline

    assert _load_config(args).options(PodcastPipeline).cache is None
