import importlib.util
import zipfile
from pathlib import Path

import mat_format
from mat_format import schema as format_schema

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_format_bundle.py"
spec = importlib.util.spec_from_file_location("build_format_bundle", SCRIPT)
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


def test_the_zip_has_everything_a_reader_needs(tmp_path):
    schemas = tmp_path / "schemas"
    schemas.mkdir()
    for name, content in format_schema.generate().items():
        (schemas / name).write_text(format_schema.render(content), encoding="utf-8")
    target = bundle.build(schemas, tmp_path / "out", tag="v9.9.9")
    with zipfile.ZipFile(target) as archive:
        names = set(archive.namelist())
        readme = archive.read("README.md").decode()
        spec_text = archive.read("result-format.md").decode()
    assert {"README.md", "LICENSE", "result-format.md", "schemas/meta.schema.json",
            "examples/podcast-sample/meta.json"} <= names
    assert f"MAT result format {mat_format.__version__}" in readme and "tree/v9.9.9" in readme
    # links point inside the zip, or to GitHub at the tag
    assert "](../" not in spec_text
    assert "](schemas/meta.schema.json)" in spec_text and "blob/v9.9.9/packages/" in spec_text
