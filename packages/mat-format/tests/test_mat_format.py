"""Tests that only need mat-format, pytest and jsonschema, so they also run without MAT installed."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from mat_format import MATResult
from mat_format import schema as format_schema

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_DIRS = sorted(p for p in EXAMPLES.iterdir() if p.is_dir())


def _validator(file_name):
    schema = json.loads((format_schema.SCHEMA_DIR / file_name).read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema)


def test_schema_files_match_the_models():
    for file_name, schema in format_schema.generate().items():
        path = format_schema.SCHEMA_DIR / file_name
        assert path.exists(), f"{file_name} is missing, is it packaged?"
        assert path.read_text(encoding="utf-8") == format_schema.render(schema), \
            f"{file_name} is outdated, run: uv run python -m mat_format.schema"


def test_schemas_are_valid_json_schema():
    for file_name in format_schema.SCHEMAS:
        jsonschema.Draft202012Validator.check_schema(
            json.loads((format_schema.SCHEMA_DIR / file_name).read_text(encoding="utf-8")))


def test_there_are_examples():
    assert {p.name for p in EXAMPLE_DIRS} >= {"podcast-sample", "book-sample"}


@pytest.mark.parametrize("example", EXAMPLE_DIRS, ids=lambda p: p.name)
def test_example_matches_the_schemas_and_loads(example):
    meta = json.loads((example / "meta.json").read_text(encoding="utf-8"))
    _validator("meta.schema.json").validate(meta)
    for pipeline in meta["pipelines"]:
        _validator(f"{pipeline}-result.schema.json").validate(
            json.loads((example / pipeline / "result.json").read_text(encoding="utf-8")))
    result = MATResult.read(example)
    assert result.meta.format == 2
    assert result.podcast is not None or result.book is not None


def test_zip_and_folder_read_the_same(tmp_path):
    folder = EXAMPLES / "podcast-sample"
    zipped = shutil.make_archive(str(tmp_path / "podcast-sample"), "zip", folder)
    from_folder, from_zip = MATResult.read(folder), MATResult.read(zipped)
    assert from_zip.podcast == from_folder.podcast
    assert from_zip.transcript() == from_folder.transcript()


def test_output_option_writes_every_schema(tmp_path):
    subprocess.run([sys.executable, "-m", "mat_format.schema", "--output", str(tmp_path)], check=True,
                   capture_output=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(format_schema.SCHEMAS)
    for file_name, schema in format_schema.generate().items():
        assert (tmp_path / file_name).read_text(encoding="utf-8") == format_schema.render(schema)


def test_does_not_pull_in_mat_or_torch():
    code = "import sys, mat_format; assert 'torch' not in sys.modules and 'MAT' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
