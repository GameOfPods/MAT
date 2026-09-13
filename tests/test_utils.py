from argparse import ArgumentParser

import pytest

from MAT.tools import SpeakerIdetificationPyannote
from MAT.utils import timeout_retry
from MAT.utils.config import Config


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


@pytest.mark.parametrize("argv, expected", [([], False), (["--Pyannote-Identification_no-hf-token"], True)])
def test_no_hf_token_flag(argv, expected):
    config = Config()
    parser = ArgumentParser()
    config.create_argparse(argparse=parser)
    config.parse_argparse(namespace=parser.parse_args(argv))
    assert config.get_config(SpeakerIdetificationPyannote)["no-hf-token"] is expected
