#  MAT - Toolkit to analyze media
#  Copyright (c) 2025.  RedRem95
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
import sys
from typing import Sequence

from MAT import __version__


def main(args: Sequence[str] = None):
    from argparse import ArgumentParser
    import os
    import glob
    import logging

    _LOGGER = logging.getLogger("MAT")

    parser = ArgumentParser(description=f'MAT-{__version__}', prog='MAT', epilog=f'Thanks for using MAT-{__version__}')
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('-i', '--input', type=str, required=True, nargs="+", dest="input", metavar="INPUT",
                        help='Input file description. Uses glob wildcards to search for files.')
    parser.add_argument('--verbose', action="store_true", dest="verbose")
    parser.add_argument('--input-recursive', action="store_true", dest="input_recursive")
    parser.add_argument('-o', '--output', type=str, required=True, dest="output", metavar="OUTPUT",
                        help='Folder to store output files to. Will be created if it does not exist.')
    parser.add_argument('--output-zip', action="store_true", dest="output_zip",
                        help="Create zip archive of output files instead of folder")

    args = parser.parse_args(args=args)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if os.path.exists(args.output) and os.path.isfile(args.output):
        sys.stderr.write(f"Output \"{args.output}\" already exists and is a file. Please specify a different output.")
        sys.exit(1)
    os.makedirs(args.output, exist_ok=True)

    input_files = set()
    for filename in args.input:
        input_files = input_files.union(os.path.abspath(x) for x in glob.glob(filename, recursive=args.input_recursive))
    input_files = set(x for x in input_files if os.path.exists(x) and os.path.isfile(x))

    _LOGGER.info(f"Found {len(input_files)} files to process")


if __name__ == "__main__":
    main()
