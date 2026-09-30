import pytest

from MAT.utils import timeout_retry


def test_timeout_retry_does_not_sleep_after_last_try(monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    def fail():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        timeout_retry(func=fail, func_args=(), func_kwargs={}, time_out=3, retries=2)
    assert sleeps == [3, 3]


def test_timeout_retry_returns_value(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise ValueError()
        return "ok"

    assert timeout_retry(func=flaky, func_args=(), func_kwargs={}, time_out=1, retries=3) == "ok"


def test_timeout_retry_gives_up_at_once_on_a_missing_login():
    from huggingface_hub.errors import GatedRepoError

    from MAT.utils import timeout_retry

    calls = []

    def load():
        calls.append(1)
        raise GatedRepoError("401 Client Error", response=None)

    with pytest.raises(GatedRepoError):
        timeout_retry(func=load, func_args=(), func_kwargs={}, time_out=60, retries=5)
    assert len(calls) == 1
