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
"""
Command line interface: `MAT run`, `MAT backends` and `MAT config`.

Backend options are not argparse flags. They come from a TOML file (`-c`) or `--set section.key=value`, and
`MAT backends show NAME` lists them. That keeps `MAT run -h` short no matter how many backends are installed.
"""
import argparse
import glob
import logging
import os
import shutil
import sys
import traceback
import uuid
from datetime import timedelta
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Type

from MAT import __version__

_LOGGER = logging.getLogger("MAT")


def _slot_owners() -> Dict[str, Type]:
    from MAT.pipelines import Pipeline

    owners = {}
    for pipeline in Pipeline.all():
        for slot in pipeline.slots:
            if slot in owners:
                raise RuntimeError(f"Slot {slot} is used by {owners[slot].section} and {pipeline.section}")
            owners[slot] = pipeline
    return owners


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("config")
    group.add_argument("-c", "--config", metavar="FILE", help="TOML config file, `MAT config init` writes one")
    group.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                       help="Set one option, overrides the config file. Can be used many times, "
                            "for example --set whisper.model=large-v3")


def _add_slot_flags(parser: argparse.ArgumentParser) -> None:
    from MAT import registry

    group = parser.add_argument_group("backends", "Pick the backend for each step. `MAT backends` lists them, "
                                                  "`MAT backends show NAME` shows their options.")
    for slot, pipeline in _slot_owners().items():
        choices = [b.name for b in registry.backends(slot)]
        if pipeline.slots[slot].optional:
            choices.append("none")
        field = pipeline.Options.model_fields[slot]
        group.add_argument(f"--{slot}", choices=choices, default=None, metavar="NAME",
                           help=f"{field.description} Installed: {', '.join(choices) or 'nothing'}. "
                                f"Default: {field.default}")


def _slot_overrides(args: argparse.Namespace) -> List[str]:
    return [f"{pipeline.section}.{slot}={getattr(args, slot)}"
            for slot, pipeline in _slot_owners().items() if getattr(args, slot, None)]


def _load_config(args: argparse.Namespace, validate: bool = True):
    from MAT.utils.config import Config

    overrides = list(getattr(args, "set", None) or []) + _slot_overrides(args)
    return Config.load(file=getattr(args, "config", None), overrides=overrides, validate=validate)


def _selected_sections(config) -> List[Type]:
    """Pipelines plus the backends their slots point at, in pipeline order."""
    from MAT import registry
    from MAT.pipelines import Pipeline

    sections = []
    for pipeline in Pipeline.all():
        sections.append(pipeline)
        options = config.options(pipeline)
        for slot in pipeline.slots:
            choice = getattr(options, slot)
            found = registry.find(choice, slot) if choice != "none" else None
            if isinstance(found, registry.Backend) and found.cls not in sections:
                sections.append(found.cls)
            elif isinstance(found, registry.SkippedBackend):
                _LOGGER.warning(f"{slot} {choice} isn't installed ({found.reason})")
    return sections


def build_parser() -> argparse.ArgumentParser:
    import MAT.tools  # noqa: F401, loads the backends
    import MAT.pipelines  # noqa: F401

    parser = argparse.ArgumentParser(prog="MAT", description=f"MAT {__version__}, Media Analytics Toolset",
                                     epilog="Run `MAT COMMAND -h` for the options of a command.")
    parser.add_argument("-v", "--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    run = commands.add_parser("run", help="Process media files",
                              description="Process media files. Every file goes through each pipeline that "
                                          "accepts it: podcast for audio, book for EPUB.")
    run.add_argument("-i", "--input", nargs="+", required=True, metavar="GLOB",
                     help="Input files, glob patterns work. Quote them so your shell doesn't expand them.")
    run.add_argument("-o", "--output", required=True, metavar="FOLDER", help="Output folder, created if missing")
    run.add_argument("-y", "--yes", action="store_true", help="Don't ask before processing the found files")
    run.add_argument("--input-recursive", action="store_true", help="Allow ** in input globs")
    _add_config_args(run)
    _add_slot_flags(run)
    output = run.add_argument_group("output")
    output.add_argument("--output-zip", action="store_true", help="Zip every result folder")
    output.add_argument("--keep-uncompressed", action="store_true", help="Keep the folder next to the zip")
    output.add_argument("--export-config", action="store_true",
                        help="Save the config that was used as config.toml in every result")
    output.add_argument("-wd", "--work-dir", default=os.getcwd(), metavar="FOLDER",
                        help="Where temporary files go, default: current directory")
    logs = run.add_argument_group("logging")
    logs.add_argument("--verbose", action="store_true", help="Debug logging")
    logs.add_argument("--log-file", metavar="FILE", help="Also write the log to this file")
    logs.add_argument("--log-file-append", action="store_true", help="Append to the log file instead of replacing it")

    backends = commands.add_parser("backends", help="List backends or show the options of one",
                                   description="`MAT backends` lists all backends per step and whether they are "
                                               "installed. `MAT backends show NAME` prints the options of one.")
    backends.add_argument("action", nargs="?", choices=["list", "show"], default="list")
    backends.add_argument("name", nargs="?", help="Backend name, for show")

    config = commands.add_parser("config", help="Create or check config files")
    config_commands = config.add_subparsers(dest="config_command", required=True, metavar="ACTION")
    init = config_commands.add_parser("init", help="Print a commented config file for the chosen backends")
    _add_config_args(init)
    _add_slot_flags(init)
    init.add_argument("-o", "--output", metavar="FILE", help="Write the file instead of printing it")
    show = config_commands.add_parser("show", help="Print the merged config (defaults, config file, --set)")
    _add_config_args(show)
    _add_slot_flags(show)

    from MAT.bench.commands import add_bench_parser

    add_bench_parser(commands)
    return parser


def cmd_backends(args: argparse.Namespace) -> int:
    from MAT import registry
    from MAT.utils.config import describe_options

    if args.action == "list":
        if args.name:
            sys.stderr.write("MAT: use `MAT backends show NAME` to see one backend\n")
            return 2
        for slot in registry.slots():
            print(slot)
            for backend in registry.backends(slot):
                print(f"  {backend.name:<14} installed      {backend.description}")
            for backend in registry.skipped(slot):
                print(f"  {backend.name:<14} not installed  {backend.reason}")
        return 0

    if not args.name:
        sys.stderr.write("MAT: `MAT backends show` needs a backend name\n")
        return 2
    found = registry.find(args.name)
    if found is None:
        raise registry.BackendError(f'Unknown backend "{args.name}". `MAT backends` lists all of them')
    if isinstance(found, registry.SkippedBackend):
        print(f"{found.name} ({found.slot}) is not installed: {found.reason}")
        return 1
    print(f"{found.name} ({found.slot}): {found.description}")
    print(f"Config section [{found.name}], or on the command line: --set {found.name}.OPTION=VALUE")
    print()
    print(describe_options(found.cls.Options))
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    from MAT.utils.config import ConfigError, render_sections

    if args.config_command == "init":
        config = _load_config(args)
        text = ("# MAT config file, created by `MAT config init`.\n"
                "# Values in here override the defaults. `--set section.key=value` overrides this file.\n\n"
                + render_sections(_selected_sections(config), config=config, comments=True))
        if args.output:
            if os.path.exists(args.output):
                raise ConfigError(f"{args.output} already exists, not overwriting it")
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Wrote {args.output}")
        else:
            sys.stdout.write(text)
        return 0

    config = _load_config(args)
    sys.stdout.write(render_sections(_selected_sections(config), config=config, comments=False))
    return 0


def _find_inputs(patterns: Sequence[str], recursive: bool) -> List[str]:
    files = set()
    for pattern in patterns:
        files.update(os.path.abspath(x) for x in glob.glob(pattern, recursive=recursive))
    return sorted((x for x in files if os.path.isfile(x)), key=lambda x: (os.path.basename(x), x))


def _confirm(input_files: List[str]) -> bool:
    answers = {"yes": ("y", "yes", "j", "t", "1"), "no": ("n", "no", "f", "0"), "list": ("l", "list")}
    while True:
        answer = input(f"Found {len(input_files)} files to process. Continue? [y/n/l] ").strip().lower()
        if answer in answers["yes"]:
            return True
        if answer in answers["no"]:
            return False
        if answer in answers["list"]:
            _LOGGER.info("Files to process:\n" + "\n".join(input_files))
        else:
            _LOGGER.error(f"Did not recognize {answer}. Answer y, n or l")


def cmd_run(args: argparse.Namespace) -> int:
    from MAT.pipelines import Pipeline
    from MAT.utils.config import render_sections
    from MAT.utils.progress import Progress
    from MAT.writer import Writer

    logging.getLogger("pytorch_lightning.utilities.migration.utils").setLevel(logging.WARN)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.log_file:
        logging.getLogger().addHandler(
            logging.FileHandler(args.log_file, mode="a" if args.log_file_append else "w", encoding="utf-8"))

    config = _load_config(args)
    _selected_sections(config)  # warns about chosen backends that aren't installed

    if os.path.isfile(args.output):
        sys.stderr.write(f'MAT: output "{args.output}" is a file, give a folder\n')
        return 2
    os.makedirs(args.output, exist_ok=True)
    _LOGGER.info(f"Export folder set to {args.output}")

    input_files = _find_inputs(args.input, args.input_recursive)
    _LOGGER.info(f"Found {len(input_files)} files to process")
    if not args.yes and not _confirm(input_files):
        _LOGGER.error("Please check your input and try again.")
        sys.exit(1)

    work_directory = os.path.join(os.path.abspath(args.work_dir), f".MAT.{uuid.uuid4()}")
    os.makedirs(work_directory, exist_ok=False)
    config.set_work_directory(work_directory)
    _LOGGER.info(f"Working directory set to {work_directory}")

    writer = Writer()
    failed = 0
    t_whole_start = perf_counter()
    try:
        with Progress(name="Processing files", desc="", total=len(input_files)) as pb:
            for file in input_files:
                t_file_start = perf_counter()
                pb.description = file
                pb.increment(n=1)
                try:
                    results = []
                    for pipeline_class in Pipeline.get_pipelines(f=file):
                        t_start = perf_counter()
                        pipeline = pipeline_class()
                        results.append(pipeline.process(file=file, config=config))
                        _LOGGER.info(f"{pipeline.name()} took {timedelta(seconds=perf_counter() - t_start)} on {file}")
                    written_folder = writer.store(file=file, output=args.output, pipeline_results=results)
                    if args.export_config:
                        with open(os.path.join(written_folder, "config.toml"), "w", encoding="utf-8") as f:
                            f.write(render_sections(_selected_sections(config), config=config, comments=False))
                    if args.output_zip:
                        zipped_folder = shutil.make_archive(written_folder, "zip", written_folder)
                        if not args.keep_uncompressed:
                            shutil.rmtree(written_folder)
                        written_folder = zipped_folder
                    _LOGGER.info(f"Wrote {written_folder}")
                except Exception as e:
                    failed += 1
                    _LOGGER.exception(f"Got error during execution for file {file}", exc_info=e)
                    with open(os.path.join(args.output, f"{os.path.basename(file)}.error.txt"), "w") as f:
                        f.write(f"{e.__class__.__name__}:\n{e}\n{'=' * 20}\nFull Error:\n")
                        f.write("".join(traceback.format_exception(type(e), e, e.__traceback__)))
                _LOGGER.info(f"File {file} took {timedelta(seconds=perf_counter() - t_file_start)}")
    finally:
        shutil.rmtree(work_directory, ignore_errors=True)
    _LOGGER.info(f"Whole Process took {timedelta(seconds=perf_counter() - t_whole_start)} "
                 f"for {len(input_files)} files, {failed} failed")
    return 1 if failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    from MAT.registry import BackendError
    from MAT.utils.config import ConfigError

    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return cmd_run(args)
        if args.command == "backends":
            return cmd_backends(args)
        if args.command == "bench":
            from MAT.bench.commands import cmd_bench

            return cmd_bench(args)
        return cmd_config(args)
    except (ConfigError, BackendError) as e:
        sys.stderr.write(f"MAT: {e}\n")
        return 2


__all__ = ["main", "build_parser"]
