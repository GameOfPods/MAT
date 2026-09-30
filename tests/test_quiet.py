import logging
import sys
from types import SimpleNamespace

from MAT.utils.quiet import NOISY, quiet_dependencies, quiet_nemo


class FakeNemoLogger:
    def __init__(self):
        self._logger = logging.getLogger("fake_nemo_logger")
        self._logger.propagate = False
        self.removed = False
        self.level = None

    def remove_stream_handlers(self):
        self.removed = True

    def setLevel(self, level):
        self.level = level


def test_quiet_dependencies_sets_levels_and_verbose_restores(monkeypatch):
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    quiet_dependencies()
    assert logging.getLogger("nv_one_logger").level == NOISY["nv_one_logger"]
    assert logging.getLogger("pytorch_lightning").level == logging.WARNING
    import os

    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"

    quiet_dependencies(verbose=True)
    assert logging.getLogger("nv_one_logger").level == logging.NOTSET


def test_quiet_nemo_without_nemo():
    sys.modules.pop("nemo.utils", None)
    quiet_nemo()  # does nothing, must not raise


def test_quiet_nemo_takes_over_the_logging(monkeypatch):
    fake = FakeNemoLogger()
    monkeypatch.setitem(sys.modules, "nemo.utils", SimpleNamespace(logging=fake))
    quiet_nemo()
    assert fake.removed is True
    assert fake.level == logging.WARNING
    # records now go to MAT's handlers and into --log-file
    assert fake._logger.propagate is True

    quiet_nemo(verbose=True)
    assert fake.level == logging.INFO


def test_quiet_nemo_survives_a_broken_logger(monkeypatch):
    broken = SimpleNamespace(logging=SimpleNamespace())
    monkeypatch.setitem(sys.modules, "nemo.utils", broken)
    quiet_nemo()


def test_quiet_whisperx_sends_its_records_through_our_logging(monkeypatch, capsys):
    import logging
    import sys
    import types

    from MAT.utils.quiet import quiet_whisperx

    logger = logging.getLogger("whisperx")
    # what whisperx sets up on import: its own stdout handler, nothing reaches the root logger
    monkeypatch.setitem(sys.modules, "whisperx", types.ModuleType("whisperx"))
    monkeypatch.setattr(logger, "handlers", [logging.StreamHandler(sys.stdout)])
    monkeypatch.setattr(logger, "propagate", False)
    quiet_whisperx()
    assert logger.propagate
    assert all(isinstance(h, logging.NullHandler) for h in logger.handlers) and logger.handlers
    logging.getLogger("whisperx.asr").error("from whisperx")
    assert "from whisperx" not in capsys.readouterr().out
