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
import shutil
import sys
import uuid
from typing import Sequence, List

from MAT import __version__


def main(args: Sequence[str] = None) -> List[str]:
    from argparse import ArgumentParser
    import os
    import glob
    import logging
    import json
    logging.getLogger("pytorch_lightning.utilities.migration.utils").setLevel(logging.WARN)
    _LOGGER = logging.getLogger("MAT")

    from MAT.utils.config import Config
    config = Config()

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
    parser.add_argument("-wd", "--work-dir", type=str, dest="work_dir", metavar="WORKDIR", default=os.getcwd(),
                        help="Set working directory to work in. Will create temporary folder that will be deleted in the end. Default %(default)s")

    parser.add_argument("-c", "--config", type=str, required=False, default=None, dest="config", metavar="CONFIG",
                        help="Path to config file for MAT tool")
    parser.add_argument("--export-config", action="store_true", dest="export_config",
                        help="If set will export the config used for this run as config.json un the export folder.")

    config.create_argparse(argparse=parser)

    args = parser.parse_args(args=args)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if os.path.exists(args.output) and os.path.isfile(args.output):
        sys.stderr.write(f"Output \"{args.output}\" already exists and is a file. Please specify a different output.")
        sys.exit(1)
    os.makedirs(args.output, exist_ok=True)

    _LOGGER.info(f"Export folder set to {args.output}")

    input_files = set()
    for filename in args.input:
        input_files = input_files.union(os.path.abspath(x) for x in glob.glob(filename, recursive=args.input_recursive))
    input_files = set(x for x in input_files if os.path.exists(x) and os.path.isfile(x))

    _LOGGER.info(f"Found {len(input_files)} files to process")

    config.parse_argparse(namespace=args)

    if args.config is not None and len(args.config) > 0 and os.path.isfile(args.config):
        try:
            with open(args.config, "r") as f:
                config.parse_config(config=json.load(f))
        finally:
            pass

    config.set_work_directory(os.path.join(os.path.abspath(args.work_dir), f".MAT.{uuid.uuid4()}"))
    while os.path.exists(config.work_directory):
        config.set_work_directory(os.path.join(os.path.abspath(args.work_dir), f".MAT.{uuid.uuid4()}"))
    os.makedirs(config.work_directory, exist_ok=False)
    _LOGGER.info(f"Working directory set to {config.work_directory}")

    from MAT.pipelines import Pipeline
    from MAT.writer import Writer
    from MAT.utils.progress import Progress
    writer = Writer()
    r = []

    with Progress(name="Processing files", desc="", total=len(input_files)) as pb:
        for file in sorted(input_files):
            pb.description = f"{file}"
            pb.increment(n=1)
            try:
                res = []
                for pipe_class in Pipeline.get_pipelines(f=file):
                    pipe = pipe_class()
                    res.append(pipe.process(file=file, config=config))
                written_folder = writer.store(file=file, output=args.output, pipeline_results=res)
                r.append(written_folder)
                if args.export_config:
                    with open(os.path.join(written_folder, "config.json"), "w") as f:
                        json.dump(config.config, f)
                        _LOGGER.info(f'Saved config to "{os.path.join(args.output, "config.json")}"')
            except Exception as e:
                import traceback
                _LOGGER.exception(f"Got error during execution for file {file}", exc_info=e)
                with open(os.path.join(args.output, f"{os.path.basename(file)}.error.txt"), "w") as f:
                    f.write(f"{e.__class__.__name__}:\n{str(e)}\n")
                    f.write(f"{'=' * 20}")
                    f.write("Full Error:\n")
                    f.write(''.join(traceback.format_exception(type(e), e, e.__traceback__)))


    shutil.rmtree(config.work_directory)
    return r

if __name__ == "__main__":
    main()
