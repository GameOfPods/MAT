"""
Builds mat-result-format.zip, the package other programs need to read MAT results without MAT.

    python scripts/build_format_bundle.py --schemas dist/schemas --output dist [--tag v0.3.0]

The zip has the JSON schemas, the spec (with links that work inside the zip), the example results, a README with
the format version and the license of the format files. The release workflow attaches it twice: as
mat-result-format.zip and as mat-result-format-<format version>.zip. Needs mat-format installed, for its version.
"""
import argparse
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
FORMAT = ROOT / "packages" / "mat-format"
REPOSITORY = "https://github.com/GameOfPods/MAT"

README = """# MAT result format {version}

Everything a program needs to read results of MAT (Media Analytics Toolset) without MAT itself.

- `result-format.md`: the spec. Layout of a result folder or zip, every field, the versioning rules and the changelog.
- `schemas/`: JSON Schemas (draft 2020-12) for `meta.json`, `podcast/result.json` and `book/result.json`. Every file
  has `"version": "{version}"` at the top.
- `examples/`: real results, one podcast and one book, written by MAT {example_version}.
- `LICENSE`: Apache 2.0, for everything in this zip.

Writing a reader:

1. Read `meta.json`. Stop if `format` isn't {major}, that's a format your reader doesn't know.
2. `format_version` ({version} here) tells which fields to expect, the changelog in the spec says which minor version
   added what. Ignore fields you don't know, newer results can have more.
3. Validate against the schemas while you develop, and test with the examples.

Python programs can use the `mat-format` package instead (attached to the same release as a wheel), it has the
reader and the data model.

Source: {repository}{tree}
"""


def spec_for_bundle(text: str, tag: Optional[str]) -> str:
    """The spec links into the repository. Inside the zip the schemas and examples sit next to it, and the models
    are linked on GitHub."""
    ref = tag or "master"
    text = text.replace("../packages/mat-format/src/mat_format/schemas/", "schemas/")
    text = text.replace("../packages/mat-format/examples", "examples")
    return re.sub(r"\]\(\.\./([^)]+)\)", lambda m: f"]({REPOSITORY}/blob/{ref}/{m.group(1)})", text)


def build(schemas: Path, output: Path, tag: Optional[str] = None) -> Path:
    import mat_format

    import json

    version = mat_format.__version__
    # the MAT version that really wrote the examples, not the one being released
    example_version = json.loads((FORMAT / "examples" / "podcast-sample" / "meta.json").read_text())["mat_version"]
    with tempfile.TemporaryDirectory() as temporary:
        bundle = Path(temporary)
        shutil.copytree(schemas, bundle / "schemas")
        shutil.copytree(FORMAT / "examples", bundle / "examples")
        shutil.copy(FORMAT / "LICENSE", bundle / "LICENSE")
        spec = (ROOT / "docs" / "result-format.md").read_text(encoding="utf-8")
        (bundle / "result-format.md").write_text(spec_for_bundle(spec, tag), encoding="utf-8")
        (bundle / "README.md").write_text(README.format(
            version=version, major=version.split(".")[0], example_version=example_version, repository=REPOSITORY,
            tree=f"/tree/{tag}" if tag else ""), encoding="utf-8")

        output.mkdir(parents=True, exist_ok=True)
        target = output / "mat-result-format.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(bundle).as_posix())
    return target


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schemas", type=Path, required=True, help="Folder with the generated schema files")
    parser.add_argument("--output", type=Path, required=True, help="Where the zip goes")
    parser.add_argument("--tag", help="Release tag, for the links and the MAT version in the README")
    args = parser.parse_args(argv)
    print(build(args.schemas, args.output, args.tag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
