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
"""`MAT bench run|report|download|datasets|reference`."""
import argparse
import importlib.util
import logging
import os
from pathlib import Path
from typing import Optional

from MAT.utils.config import ConfigError

_LOGGER = logging.getLogger("MAT.bench")


def add_bench_parser(commands) -> None:
    bench = commands.add_parser(
        "bench", help="Benchmark backends on datasets",
        description="Run podcast systems (backends and their settings) on datasets and compare speed, GPU memory, "
                    "WER, cpWER and DER. Summaries never run. See docs/benchmarks.md.")
    actions = bench.add_subparsers(dest="bench_command", required=True, metavar="ACTION")

    def bench_file(parser, output: bool):
        parser.add_argument("-c", "--config", required=True, metavar="FILE",
                            help="Bench file, benchmarks/example.toml shows all parts")
        if output:
            parser.add_argument("-o", "--output", required=True, metavar="FOLDER",
                                help="Where results, results.csv and report.md go")
        parser.add_argument("--dataset", action="append", default=[], metavar="NAME",
                            help="Only this dataset, can be given more than once")
        parser.add_argument("--limit", type=int, metavar="N", help="Only the first N files of every dataset")
        parser.add_argument("--cache", metavar="FOLDER",
                            help="Dataset cache, overrides the bench file and $MAT_BENCH_CACHE")
        parser.add_argument("--verbose", action="store_true", help="Debug logging")

    run = actions.add_parser("run", help="Run the systems on the datasets and write the report",
                             description="Results that finished in an earlier run in the same output folder are "
                                         "skipped, so an aborted benchmark continues where it stopped.")
    bench_file(run, output=True)
    run.add_argument("--system", action="append", default=[], metavar="NAME",
                     help="Only this system, can be given more than once")
    run.add_argument("--rerun", action="store_true", help="Run again even where a result exists")

    report = actions.add_parser("report", help="Score the stored results again and rewrite the report",
                                description="Useful after correcting a reference or changing the collar.")
    bench_file(report, output=True)
    report.add_argument("--system", action="append", default=[], metavar="NAME", help="Only this system")

    download = actions.add_parser("download", help="Download and prepare the datasets without running anything")
    bench_file(download, output=False)

    actions.add_parser("datasets", help="List the dataset types and their options")

    reference = actions.add_parser(
        "reference", help="Create a reference to correct by hand from a MAT result",
        description="Copies the transcript of a MAT result (or the part between --start and --end) into a new "
                    "reference folder. Correct transcript.txt there, then add the folder as a reference dataset.")
    reference.add_argument("result", metavar="RESULT", help="MAT result folder or zip")
    reference.add_argument("-o", "--output", required=True, metavar="FOLDER", help="New reference folder")
    reference.add_argument("--audio", metavar="FILE",
                           help="The audio file. Default: the input path stored in the result")
    reference.add_argument("--start", type=float, metavar="SECONDS", help="Only lines from here on")
    reference.add_argument("--end", type=float, metavar="SECONDS", help="Only lines up to here")
    reference.add_argument("--language", metavar="CODE", help="Default: the language of the result")


def _require_metrics() -> None:
    missing = [module for module in ("jiwer", "pyannote.metrics") if importlib.util.find_spec(module) is None]
    if missing:
        raise ConfigError(f"MAT bench needs {', '.join(missing)}. Install the bench extra, for example "
                          f"`uv sync --extra bench`")


def cmd_bench(args: argparse.Namespace) -> int:
    if args.bench_command == "datasets":
        return _list_datasets()
    if args.bench_command == "reference":
        folder = create_reference(Path(args.result), Path(args.output), audio=Path(args.audio) if args.audio else None,
                                  start=args.start, end=args.end, language=args.language)
        print(f"Wrote {folder}. Correct {folder / 'transcript.txt'}, see docs/benchmarks.md")
        return 0

    from MAT.bench.runner import BenchFile, collect_rows, prepare, run_benchmark

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    bench = BenchFile.load(args.config, cache=os.path.abspath(args.cache) if args.cache else None)

    if args.bench_command == "download":
        datasets, _ = bench.pick(args.dataset)
        for dataset, items in prepare(datasets, args.limit):
            print(f"{dataset.name}: {len(items)} files ready")
        return 0

    _require_metrics()
    if args.bench_command == "run":
        rows = run_benchmark(bench, Path(args.output), datasets=args.dataset, systems=args.system,
                             limit=args.limit, rerun=args.rerun)
    else:
        from MAT.bench.report import write_report

        datasets, systems = bench.pick(args.dataset, args.system)
        prepared = prepare(datasets, args.limit)
        rows = collect_rows(Path(args.output), prepared, systems, bench.settings.collar)
        write_report(Path(args.output), rows, prepared, systems, bench.settings.collar)
    failed = sum(1 for row in rows if row.get("error"))
    print(f"Wrote {Path(args.output) / 'report.md'} and results.csv, {len(rows)} runs, {failed} failed")
    return 1 if failed else 0


def _list_datasets() -> int:
    from MAT.bench.datasets import DATASETS
    from MAT.utils.config import describe_options

    for name, cls in sorted(DATASETS.items()):
        print(f"{name}: {cls.description}")
        print(f"  license: {cls.license}")
        print(f"  download: {cls.size}")
        print("  options:")
        print("\n".join(f"  {line}" for line in describe_options(cls.Options).splitlines()))
        print()
    return 0


def create_reference(result: Path, output: Path, audio: Optional[Path] = None, start: Optional[float] = None,
                     end: Optional[float] = None, language: Optional[str] = None) -> Path:
    from mat_format import MATResult

    from MAT.bench.data import TranscriptError, parse_transcript
    from MAT.bench.metrics import midpoint_in
    from MAT.utils.config import toml_value

    if start is not None and end is not None and end <= start:
        raise ConfigError("--end has to be after --start")
    try:
        mat = MATResult.read(result)
    except (OSError, ValueError) as e:
        raise ConfigError(f"Can't read {result}: {e}")
    if mat.podcast is None:
        raise ConfigError(f"{result} has no podcast result")
    audio = audio or Path(mat.meta.input.path)
    if not audio.is_file():
        raise ConfigError(f"Audio file {audio} not found, pass --audio")
    if output.exists() and any(output.iterdir()):
        raise ConfigError(f"{output} exists and isn't empty")

    kept = []
    for line in (mat.transcript() or "").splitlines():
        try:
            turns = parse_transcript(line)
        except TranscriptError:
            continue
        if turns and midpoint_in(turns[0].start, turns[0].end, start, end):
            kept.append((turns[0], line))
    if not kept:
        raise ConfigError("No transcript lines in that time range")

    toml = ["# Reference for `MAT bench`, created from a MAT result. See docs/benchmarks.md.",
            "# Correct transcript.txt: fix the words and the speaker of every line and split lines where the",
            "# speaker changes. Times only need to be about right.",
            f"audio = {toml_value(str(audio.resolve()))}"]
    language = language or mat.podcast.language
    if language:
        toml.append(f"language = {toml_value(language)}")
    # the scored range starts and ends with the kept lines, so no line is cut in half
    if start is not None:
        toml.append(f"start = {min(t.start for t, _ in kept)}")
    if end is not None:
        toml.append(f"end = {max(t.end for t, _ in kept)}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "reference.toml").write_text("\n".join(toml) + "\n", encoding="utf-8")
    (output / "transcript.txt").write_text("\n".join(line for _, line in kept) + "\n", encoding="utf-8")
    return output


__all__ = ["add_bench_parser", "cmd_bench", "create_reference"]
