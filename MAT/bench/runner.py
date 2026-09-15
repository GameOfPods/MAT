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
Bench file, run loop and scoring of the stored results.

Every run of a system on an item lands in <output>/results/<system>/<dataset>/<item>/ as `result/` (a normal MAT
result) plus `bench.json` (time, step times, GPU memory, error). Runs that finished are skipped next time, so an
aborted benchmark continues where it stopped. Metrics are always computed from the stored results.
"""
import importlib.util
import itertools
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import Field, ValidationError

from MAT.bench.data import Item
from MAT.bench.datasets import DATASETS, NAME_PATTERN, Dataset, resolve_path
from MAT.bench.download import cache_root
from MAT.utils.config import Config, ConfigError, Options

_LOGGER = logging.getLogger(__name__)


class BenchSettings(Options):
    cache: Optional[str] = Field(None, description="Folder for downloaded datasets, relative to the bench file. "
                                                   "Default: $MAT_BENCH_CACHE or ~/.cache/mat/bench")
    collar: Optional[float] = Field(None, ge=0, description="DER collar in seconds for all datasets. Default: 0.25 "
                                                           "for own references (their line times only cover the "
                                                           "words), 0 for the public datasets.")


class SystemSettings(Options):
    name: str = Field(pattern=NAME_PATTERN, description="Name in the report and in the result folders.")
    config: Optional[str] = Field(None, description="MAT config file (TOML), relative to the bench file.")
    set: List[str] = Field(default_factory=list, description='Settings like `MAT run --set`, for example '
                                                             '"whisper.model=large-v3".')


def _validate(model, raw: Any, where: str):
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} has to be a table")
    try:
        return model.model_validate(raw)
    except ValidationError as e:
        messages = []
        for err in e.errors():
            location = ".".join(str(x) for x in err["loc"]) or "value"
            if err["type"] == "extra_forbidden":
                messages.append(f'{where}: unknown option "{location}"')
            else:
                messages.append(f"{where}: {location}: {err['msg']}")
        raise ConfigError("\n".join(messages))


def _unique(names: Sequence[str], what: str) -> None:
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ConfigError(f"{what} names have to be unique, used more than once: {', '.join(duplicates)}")


@dataclass
class BenchFile:
    settings: BenchSettings
    datasets: List[Dataset]
    systems: List[SystemSettings]
    base_dir: Path

    @classmethod
    def load(cls, path, cache: Optional[str] = None) -> "BenchFile":
        path = Path(path)
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ConfigError(f"Bench file {path} doesn't exist")
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"Can't read bench file {path}: {e}")
        return cls.from_dict(data, base_dir=path.resolve().parent, cache=cache)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_dir: Path, cache: Optional[str] = None) -> "BenchFile":
        unknown = set(data) - {"bench", "dataset", "system"}
        if unknown:
            raise ConfigError(f"Unknown entries in the bench file: {', '.join(sorted(unknown))}. "
                              f"Expected [bench], [[dataset]] and [[system]]")
        settings = _validate(BenchSettings, data.get("bench", {}), "[bench]")
        if cache:
            root = Path(cache).expanduser()
        elif settings.cache:
            root = resolve_path(base_dir, settings.cache)
        else:
            root = cache_root()

        raw_datasets = data.get("dataset", [])
        if not isinstance(raw_datasets, list) or not raw_datasets:
            raise ConfigError("The bench file needs at least one [[dataset]]")
        datasets = []
        for number, raw in enumerate(raw_datasets, start=1):
            kind = raw.get("type") if isinstance(raw, dict) else None
            if kind not in DATASETS:
                raise ConfigError(f'[[dataset]] number {number}: unknown type "{kind}". '
                                  f'Types: {", ".join(sorted(DATASETS))}')
            options = _validate(DATASETS[kind].Options, raw, f"[[dataset]] number {number} ({kind})")
            datasets.append(DATASETS[kind](options, cache=root, base_dir=base_dir))
        _unique([d.name for d in datasets], "Dataset")

        raw_systems = data.get("system", [])
        if not isinstance(raw_systems, list):
            raise ConfigError("[[system]] has to be a list of tables")
        systems = [_validate(SystemSettings, raw, f"[[system]] number {number}")
                   for number, raw in enumerate(raw_systems, start=1)] or [SystemSettings(name="default")]
        _unique([s.name for s in systems], "System")
        return cls(settings=settings, datasets=datasets, systems=systems, base_dir=base_dir)

    def pick(self, datasets: Sequence[str] = (), systems: Sequence[str] = ()) -> Tuple[List[Dataset],
                                                                                     List[SystemSettings]]:
        return (_pick(self.datasets, datasets, "dataset", lambda d: d.name),
                _pick(self.systems, systems, "system", lambda s: s.name))


def _pick(values, names: Sequence[str], what: str, name_of):
    if not names:
        return list(values)
    known = {name_of(v): v for v in values}
    missing = [n for n in names if n not in known]
    if missing:
        raise ConfigError(f"Unknown {what} {', '.join(missing)}. In the bench file: {', '.join(known)}")
    return [known[n] for n in names]


def system_config(system: SystemSettings, base_dir: Path) -> Config:
    """The MAT config of a system. Summaries never run, the identifier is off unless the system sets it."""
    file = resolve_path(base_dir, system.config) if system.config else None
    config = Config.load(file=file, overrides=system.set, validate=False)
    podcast = config.values.get("podcast", {})
    if podcast.get("summarizer", "none") != "none":
        _LOGGER.warning(f"System {system.name}: summaries don't run in benchmarks, ignoring the summarizer")
    config.set("podcast", "summarizer", "none")
    if "identifier" not in podcast:
        config.set("podcast", "identifier", "none")
    try:
        config.validate()
    except ConfigError as e:
        raise ConfigError(f"System {system.name}: {e}")
    return config


def prepare(datasets: Sequence[Dataset], limit: Optional[int] = None) -> List[Tuple[Dataset, List[Item]]]:
    """Items of every dataset, downloading what's missing."""
    prepared = []
    for dataset in datasets:
        _LOGGER.info(f"Preparing dataset {dataset.name} ({dataset.type})")
        items = dataset.items()
        prepared.append((dataset, list(itertools.islice(items, limit) if limit else items)))
    return prepared


class GpuMemorySampler:
    """Highest GPU memory of this process according to nvidia-smi. CTranslate2 (whisper) doesn't report to torch,
    so torch's own counter misses it."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak: Optional[int] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def sample() -> Optional[int]:
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        pid, total = str(os.getpid()), 0
        for line in out.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0] == pid and parts[1].isdigit():
                total += int(parts[1])
        return total

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self.sample()
            if value is not None:
                self.peak = max(self.peak or 0, value)
            self._stop.wait(self.interval)

    def __enter__(self) -> "GpuMemorySampler":
        if shutil.which("nvidia-smi") is not None:
            self._thread = threading.Thread(target=self._run, name="gpu-memory", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)


@dataclass
class RunInfo:
    wall_seconds: float = 0.0
    steps: Dict[str, float] = field(default_factory=dict)
    peak_gpu_mib: Optional[int] = None
    torch_peak_mib: Optional[int] = None
    error: Optional[str] = None


def _cuda_torch():
    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    return torch if torch.cuda.is_available() else None


def run_pipeline(item: Item, config: Config, target: Path) -> RunInfo:
    """Runs the podcast pipeline on one item and stores the MAT result in target."""
    from MAT.pipelines import PodcastPipeline
    from MAT.writer import Writer

    torch = _cuda_torch()
    if torch is not None:
        torch.cuda.reset_peak_memory_stats()
    pipeline = PodcastPipeline()
    with GpuMemorySampler() as gpu:
        start = time.perf_counter()
        output = pipeline.process(file=str(item.audio), config=config)
        wall = time.perf_counter() - start
    torch_peak = None if torch is None else int(torch.cuda.max_memory_allocated() / 2 ** 20)

    staging = target.parent / f".{target.name}.staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    written = Writer().store(file=str(item.audio), output=str(staging), pipeline_results=[output])
    shutil.rmtree(target, ignore_errors=True)
    shutil.move(written, target)
    shutil.rmtree(staging, ignore_errors=True)
    return RunInfo(wall_seconds=wall, steps=dict(pipeline.step_seconds), peak_gpu_mib=gpu.peak,
                   torch_peak_mib=torch_peak)


def run_folder(output: Path, system: str, item: Item) -> Path:
    return output / "results" / system / item.dataset / item.id


def load_run(folder: Path) -> Optional[RunInfo]:
    try:
        data = json.loads((folder / "bench.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None
    return RunInfo(**{k: v for k, v in data.items() if k in RunInfo.__dataclass_fields__})


Process = Callable[[Item, Config, Path], RunInfo]


def run_benchmark(bench: BenchFile, output: Path, datasets: Sequence[str] = (), systems: Sequence[str] = (),
                  limit: Optional[int] = None, rerun: bool = False,
                  process: Process = run_pipeline) -> List[Dict[str, Any]]:
    """Runs every picked system on every item, then writes results.csv and report.md. Returns the rows."""
    from MAT.bench.report import write_report

    output = Path(output)
    picked_datasets, picked_systems = bench.pick(datasets, systems)
    # check every system config before anything long runs
    configs = {system.name: system_config(system, bench.base_dir) for system in picked_systems}
    prepared = prepare(picked_datasets, limit)
    total = sum(len(items) for _, items in prepared) * len(picked_systems)
    work = output / ".work"
    count = 0
    for system in picked_systems:
        for _, items in prepared:
            for item in items:
                count += 1
                folder = run_folder(output, system.name, item)
                previous = load_run(folder)
                if not rerun and previous is not None and previous.error is None and (folder / "result").is_dir():
                    _LOGGER.info(f"[{count}/{total}] {system.name} on {item.dataset}/{item.id}: done before, skipping")
                    continue
                _LOGGER.info(f"[{count}/{total}] {system.name} on {item.dataset}/{item.id}")
                folder.mkdir(parents=True, exist_ok=True)
                work.mkdir(parents=True, exist_ok=True)
                configs[system.name].set_work_directory(str(work))
                try:
                    info = process(item, configs[system.name], folder / "result")
                except Exception as e:
                    _LOGGER.exception(f"{system.name} failed on {item.dataset}/{item.id}", exc_info=e)
                    info = RunInfo(error=f"{type(e).__name__}: {e}")
                finally:
                    shutil.rmtree(work, ignore_errors=True)
                (folder / "bench.json").write_text(json.dumps(asdict(info), indent=1), encoding="utf-8")
    rows = collect_rows(output, prepared, picked_systems, bench.settings.collar)
    write_report(output, rows, prepared, picked_systems, bench.settings.collar)
    return rows


def _snake(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    return None if value is None else round(value, digits)


def collect_rows(output: Path, prepared: Sequence[Tuple[Dataset, List[Item]]],
                 systems: Sequence[SystemSettings], bench_collar: Optional[float] = None) -> List[Dict[str, Any]]:
    """One row per system and item with times and metrics, computed from the stored results."""
    from mat_format import MATResult

    from MAT.bench.scoring import agreement, duration_of, score

    rows = []
    for dataset, items in prepared:
        collar = dataset.collar(bench_collar)
        for item in items:
            baseline = None
            for index, system in enumerate(systems):
                folder = run_folder(output, system.name, item)
                info = load_run(folder)
                row: Dict[str, Any] = {"dataset": item.dataset, "item": item.id, "system": system.name,
                                       "language": item.language}
                podcast = None
                if info is None:
                    row["error"] = "not run"
                elif info.error is not None:
                    row["error"] = info.error
                else:
                    try:
                        podcast = MATResult.read(folder / "result").podcast
                    except (OSError, ValueError) as e:
                        row["error"] = f"can't read the result: {e}"
                if podcast is not None:
                    duration = duration_of(podcast)
                    row.update(language=item.language or podcast.language, audio_seconds=_round(duration, 3),
                               wall_seconds=_round(info.wall_seconds, 3),
                               rtfx=_round(duration / info.wall_seconds, 2) if info.wall_seconds else None,
                               peak_gpu_mib=info.peak_gpu_mib, torch_peak_mib=info.torch_peak_mib)
                    row.update({f"step_{_snake(name)}_seconds": _round(seconds, 3)
                                for name, seconds in info.steps.items()})
                    _add_score(row, "", score(item, podcast, collar))
                    if index == 0:
                        baseline = podcast
                    elif baseline is not None:
                        _add_score(row, "agree_", agreement(item, podcast, baseline, collar))
                elif index == 0:
                    baseline = None
                rows.append(row)
    return rows


def _add_score(row: Dict[str, Any], prefix: str, result) -> None:
    if result.wer is not None:
        row.update({f"{prefix}wer": _round(result.wer.rate), f"{prefix}wer_errors": result.wer.errors,
                    f"{prefix}wer_words": result.wer.reference_words})
    if result.cpwer is not None:
        row.update({f"{prefix}cpwer": _round(result.cpwer.rate), f"{prefix}cpwer_errors": result.cpwer.errors,
                    f"{prefix}cpwer_words": result.cpwer.reference_words})
    if result.der is not None:
        row.update({f"{prefix}der": _round(result.der.rate), f"{prefix}der_missed": _round(result.der.missed, 3),
                    f"{prefix}der_false_alarm": _round(result.der.false_alarm, 3),
                    f"{prefix}der_confusion": _round(result.der.confusion, 3),
                    f"{prefix}der_total": _round(result.der.total, 3)})
    if not prefix:
        row["ref_speakers"] = result.reference_speakers
        row["hyp_speakers"] = result.hypothesis_speakers


__all__ = ["BenchSettings", "SystemSettings", "BenchFile", "system_config", "prepare", "GpuMemorySampler",
           "RunInfo", "run_pipeline", "run_folder", "load_run", "run_benchmark", "collect_rows"]
