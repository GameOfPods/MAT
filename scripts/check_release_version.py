"""
Checks that a release tag matches the MAT version in MAT/__version__.py.

The tag has to be "v" plus the version, for example v0.2.0 for __version__ = "0.2.0".
Used by .github/workflows/release-schemas.yml, works without installing MAT.

    python scripts/check_release_version.py v0.2.0
"""
import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

VERSION_FILE = Path(__file__).resolve().parent.parent / "MAT" / "__version__.py"


def read_version(path: Path = VERSION_FILE) -> str:
    match = re.search(r"""^__version__\s*=\s*["']([^"']+)["']""", path.read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise ValueError(f"No __version__ found in {path}")
    return match.group(1)


def check(tag: str, version: str) -> Optional[str]:
    """None if the tag fits the version, otherwise the error message."""
    expected = f"v{version}"
    if tag == expected:
        return None
    return (f'Release tag "{tag}" doesn\'t match the MAT version {version}. '
            f'Tag the release as "{expected}" or change MAT/__version__.py and tag again.')


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tag", help="Release tag, for example v0.2.0")
    args = parser.parse_args(argv)

    version = read_version()
    error = check(args.tag, version)
    if error is not None:
        # ::error:: shows up as an annotation on the workflow run
        print(f"::error::{error}" if os.environ.get("GITHUB_ACTIONS") else error, file=sys.stderr)
        return 1
    print(f"Release tag {args.tag} matches MAT {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
