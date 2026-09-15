import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from mat_format import schema as format_schema
from test_result_format import book_output, podcast_output, write

EXAMPLES = Path(__file__).parent.parent / "packages" / "mat-format" / "examples"


def _validator(file_name):
    schema = json.loads((format_schema.SCHEMA_DIR / file_name).read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())


def _validate_result_folder(folder: Path):
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    _validator("meta.schema.json").validate(meta)
    for pipeline in meta["pipelines"]:
        data = json.loads((folder / pipeline / "result.json").read_text(encoding="utf-8"))
        _validator(f"{pipeline}-result.schema.json").validate(data)


def test_committed_schemas_match_the_models():
    for file_name, schema in format_schema.generate().items():
        committed = (format_schema.SCHEMA_DIR / file_name).read_text(encoding="utf-8")
        assert committed == format_schema.render(schema), \
            f"{file_name} is outdated, run: uv run python -m mat_format.schema"


def test_schemas_are_valid_json_schema():
    for file_name in format_schema.SCHEMAS:
        jsonschema.Draft202012Validator.check_schema(
            json.loads((format_schema.SCHEMA_DIR / file_name).read_text(encoding="utf-8")))


def test_writer_output_matches_the_schemas(tmp_path):
    input_file = tmp_path / "episode.wav"
    input_file.write_bytes(b"x")
    _validate_result_folder(write(tmp_path, input_file, [podcast_output(), book_output()]))


def test_schema_catches_broken_files(tmp_path):
    input_file = tmp_path / "episode.wav"
    input_file.write_bytes(b"x")
    folder = write(tmp_path, input_file, [podcast_output()])
    data = json.loads((folder / "podcast" / "result.json").read_text())
    data["speakers"][0]["segments"][0]["start"] = "zero"
    with pytest.raises(jsonschema.ValidationError):
        _validator("podcast-result.schema.json").validate(data)


def test_examples_match_the_schemas():
    examples = sorted(p for p in EXAMPLES.iterdir() if p.is_dir())
    assert examples, f"no examples in {EXAMPLES}"
    for example in examples:
        _validate_result_folder(example)


def test_mat_format_does_not_pull_in_mat_or_torch():
    code = "import sys, mat_format; assert 'torch' not in sys.modules and 'MAT' not in sys.modules, sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
