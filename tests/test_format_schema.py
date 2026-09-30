"""Writer output against the schemas. Schema and example checks that don't need MAT are in packages/mat-format/tests."""
import json
from pathlib import Path

import jsonschema
import pytest

from mat_format import schema as format_schema
from test_result_format import book_output, podcast_output, write


def _validator(file_name):
    schema = json.loads((format_schema.SCHEMA_DIR / file_name).read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema)


def test_writer_output_matches_the_schemas(tmp_path):
    input_file = tmp_path / "episode.wav"
    input_file.write_bytes(b"x")
    folder: Path = write(tmp_path, input_file, [podcast_output(), book_output()])
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    _validator("meta.schema.json").validate(meta)
    for pipeline in meta["pipelines"]:
        _validator(f"{pipeline}-result.schema.json").validate(
            json.loads((folder / pipeline / "result.json").read_text(encoding="utf-8")))


def test_schema_catches_broken_files(tmp_path):
    input_file = tmp_path / "episode.wav"
    input_file.write_bytes(b"x")
    folder = write(tmp_path, input_file, [podcast_output()])
    data = json.loads((folder / "podcast" / "result.json").read_text())
    data["speakers"][0]["segments"][0]["start"] = "zero"
    with pytest.raises(jsonschema.ValidationError):
        _validator("podcast-result.schema.json").validate(data)
