"""
Generates the JSON schemas in schemas/ from the models.

    python -m mat_format.schema          write the schema files
    python -m mat_format.schema --check  exit with 1 if the files don't match the models
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, Type

from pydantic import BaseModel

from mat_format.models import FORMAT_VERSION, BookResult, Meta, PodcastResult

SCHEMA_DIR = Path(__file__).parent / "schemas"
BASE_URL = "https://raw.githubusercontent.com/GameOfPods/MAT/master/packages/mat-format/src/mat_format/schemas/"

# schema file name -> (model, file it describes)
SCHEMAS: Dict[str, Tuple[Type[BaseModel], str]] = {
    "meta.schema.json": (Meta, "meta.json"),
    "podcast-result.schema.json": (PodcastResult, "podcast/result.json"),
    "book-result.schema.json": (BookResult, "book/result.json"),
}


def generate() -> Dict[str, dict]:
    schemas = {}
    for file_name, (model, target) in SCHEMAS.items():
        schema = model.model_json_schema(mode="serialization", by_alias=True)
        schemas[file_name] = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": BASE_URL + file_name,
            "$comment": f"MAT result format {FORMAT_VERSION}, describes {target}. Generated from "
                        f"mat_format/models.py, don't edit by hand.",
            **schema,
        }
    return schemas


def render(schema: dict) -> str:
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Only check, don't write")
    args = parser.parse_args()

    outdated = []
    for file_name, schema in generate().items():
        path = SCHEMA_DIR / file_name
        if not path.exists() or path.read_text(encoding="utf-8") != render(schema):
            outdated.append(path)
            if not args.check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(render(schema), encoding="utf-8")
                print(f"wrote {path}")
    if args.check and outdated:
        print("outdated schema files, run `python -m mat_format.schema`: " + ", ".join(map(str, outdated)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
