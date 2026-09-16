import sys

import pytest

from MAT.utils import external
from MAT.utils.external import ExternalEnvironment, ExternalError, MissingEnvironment, run_external

ECHO = """
import json, sys
from pathlib import Path
request = json.loads(Path(sys.argv[1]).read_text())
print("working")
Path(sys.argv[2]).write_text(json.dumps({"echo": request}))
"""


@pytest.fixture
def environment(tmp_path, monkeypatch):
    """A fake environment that runs with the Python of the test run."""
    monkeypatch.setenv(external.ENVS_DIR_VAR, str(tmp_path))
    monkeypatch.setenv("MAT_EXTERNAL_PYTHON_ECHO", sys.executable)
    monkeypatch.setitem(external.ENVIRONMENTS, "echo", ExternalEnvironment(name="echo", description="test"))
    folder = tmp_path / "echo"
    folder.mkdir()
    (folder / "run.py").write_text(ECHO)
    return folder


def test_round_trip(environment):
    assert run_external("echo", {"audio": "x.wav", "model": "m"}) == {"echo": {"audio": "x.wav", "model": "m"}}


def test_environment_variables_reach_the_process(environment, monkeypatch):
    # the other environment has to see HF_HOME and HF_TOKEN, otherwise it downloads the models a second time
    (environment / "run.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        "Path(sys.argv[2]).write_text(json.dumps({name: os.environ.get(name) for name in ('HF_HOME', 'MAT_TEST_VAR')}))\n")
    monkeypatch.setenv("HF_HOME", "/somewhere/huggingface")
    monkeypatch.setenv("MAT_TEST_VAR", "passed along")
    assert run_external("echo", {}) == {"HF_HOME": "/somewhere/huggingface", "MAT_TEST_VAR": "passed along"}


def test_missing_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(external.ENVS_DIR_VAR, str(tmp_path))
    monkeypatch.delenv("MAT_EXTERNAL_PYTHON_DIARIZEN", raising=False)
    assert external.is_available("diarizen") is False
    with pytest.raises(MissingEnvironment, match="MAT external install diarizen") as error:
        run_external("diarizen", {})
    # skipped like a backend with a missing extra
    assert error.value.extra == "diarizen"


def test_failing_script_keeps_the_error(environment):
    (environment / "run.py").write_text("import sys\nprint('boom', file=sys.stderr)\nsys.exit(3)\n")
    with pytest.raises(ExternalError, match="exit code 3") as error:
        run_external("echo", {})
    assert "boom" in str(error.value)


def test_script_without_result(environment):
    (environment / "run.py").write_text("print('nothing to see')\n")
    with pytest.raises(ExternalError, match="wrote no result"):
        run_external("echo", {})


def test_broken_result(environment):
    (environment / "run.py").write_text("import sys\nfrom pathlib import Path\n"
                                        "Path(sys.argv[2]).write_text('not json')\n")
    with pytest.raises(ExternalError, match="isn't JSON"):
        run_external("echo", {})


def test_timeout(environment):
    (environment / "run.py").write_text("import time\ntime.sleep(5)\n")
    with pytest.raises(ExternalError, match="didn't finish"):
        run_external("echo", {}, timeout=0.5)


def test_install_checks_the_name(tmp_path, monkeypatch):
    monkeypatch.setenv(external.ENVS_DIR_VAR, str(tmp_path))
    with pytest.raises(ExternalError, match="Unknown environment"):
        external.install("nope")
    with pytest.raises(ExternalError, match="doesn't exist"):
        external.install("diarizen")


def test_environment_folder_default(monkeypatch):
    # test_diarizen.py sets this so its module can be imported without the environment
    monkeypatch.delenv("MAT_EXTERNAL_PYTHON_DIARIZEN", raising=False)
    monkeypatch.delenv(external.ENVS_DIR_VAR, raising=False)
    assert external.envs_directory().name == "envs"
    assert (external.envs_directory() / "diarizen" / "run.py").is_file()
    assert sorted(external.ENVIRONMENTS) == ["diarizen"]
    assert external.interpreter("diarizen") == external.envs_directory() / "diarizen" / ".venv" / "bin" / "python"
