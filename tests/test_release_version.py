import importlib.util
from pathlib import Path

import pytest

import MAT

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_release_version.py"
spec = importlib.util.spec_from_file_location("check_release_version", SCRIPT)
check_release_version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_release_version)


def test_reads_the_same_version_as_mat():
    assert check_release_version.read_version() == MAT.__version__


def test_matching_tag_passes():
    assert check_release_version.check("v0.3.0", "0.3.0") is None
    assert check_release_version.check("v0.3.0rc1", "0.3.0rc1") is None


@pytest.mark.parametrize("tag", ["v0.2.9", "0.3.0", "v0.3", "v0.3.0-final", "release-0.3.0"])
def test_other_tags_fail(tag):
    error = check_release_version.check(tag, "0.3.0")
    assert error is not None
    assert '"v0.3.0"' in error


def test_main_exit_codes(capsys):
    assert check_release_version.main([f"v{MAT.__version__}"]) == 0
    assert check_release_version.main(["v999.0.0"]) == 1
    assert "doesn't match" in capsys.readouterr().err


def test_missing_version_line(tmp_path):
    file = tmp_path / "__version__.py"
    file.write_text("version = '1.0'\n")
    with pytest.raises(ValueError):
        check_release_version.read_version(file)
