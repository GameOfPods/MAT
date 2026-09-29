import pytest


@pytest.fixture(autouse=True)
def no_hugging_face_login_check(monkeypatch):
    """Unit tests don't touch the network. The default diarizer checks its Hugging Face access before a run, which
    fails in CI (no login) and is slow everywhere. Tests of that check patch auth_check themselves."""
    try:
        import huggingface_hub
    except ImportError:
        return
    monkeypatch.setattr(huggingface_hub, "auth_check", lambda repo_id, **kwargs: None)


@pytest.fixture(autouse=True)
def no_step_cache_in_home(monkeypatch):
    """Pipeline tests must not write results of fake backends into ~/.cache/mat/steps. Tests of the cache give it
    a folder of their own."""
    from MAT.utils import step_cache

    original = step_cache.StepCache.__init__

    def init(self, folder):
        original(self, None if folder == step_cache.DEFAULT_FOLDER else folder)

    monkeypatch.setattr(step_cache.StepCache, "__init__", init)
