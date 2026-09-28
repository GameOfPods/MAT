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
