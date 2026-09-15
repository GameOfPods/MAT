import json
import shutil
from pathlib import Path

import pytest
from mat_format import Media, MATResult, Speaker, TimeRange, Word

from MAT.bench.report import summarize
from MAT.bench.runner import BenchFile, GpuMemorySampler, RunInfo, run_benchmark, system_config
from MAT.utils.config import ConfigError

EXAMPLE = Path(__file__).resolve().parent.parent / "packages" / "mat-format" / "examples" / "podcast-sample"


def _bench(tmp_path, **extra):
    refs = tmp_path / "refs" / "ep1"
    refs.mkdir(parents=True)
    (refs / "episode.wav").write_bytes(b"RIFF")
    (refs / "reference.toml").write_text('audio = "episode.wav"\nlanguage = "en"\n')
    (refs / "transcript.txt").write_text("alice [0.0 - 2.0]: Hello there.\nbob [2.0 - 4.0]: Good morning!\n")
    data = {"dataset": [{"type": "reference", "name": "refs", "path": str(tmp_path / "refs")}]}
    data.update(extra)
    return BenchFile.from_dict(data, base_dir=tmp_path, cache=str(tmp_path / "cache"))


def test_bench_file_errors(tmp_path):
    with pytest.raises(ConfigError, match="Unknown entries"):
        BenchFile.from_dict({"datasets": []}, base_dir=tmp_path)
    with pytest.raises(ConfigError, match="at least one"):
        BenchFile.from_dict({}, base_dir=tmp_path)
    with pytest.raises(ConfigError, match='unknown type "nope"'):
        BenchFile.from_dict({"dataset": [{"type": "nope"}]}, base_dir=tmp_path)
    with pytest.raises(ConfigError, match='unknown option "speed"'):
        BenchFile.from_dict({"dataset": [{"type": "fleurs", "speed": 1}]}, base_dir=tmp_path)
    with pytest.raises(ConfigError, match="used more than once: fleurs"):
        BenchFile.from_dict({"dataset": [{"type": "fleurs"}, {"type": "fleurs"}]}, base_dir=tmp_path)
    with pytest.raises(ConfigError, match="name"):
        BenchFile.from_dict({"dataset": [{"type": "fleurs"}], "system": [{"name": "bad name"}]}, base_dir=tmp_path)


def test_bench_file_defaults(tmp_path):
    bench = BenchFile.from_dict({"dataset": [{"type": "fleurs", "pack-minutes": 5}]}, base_dir=tmp_path)
    assert [s.name for s in bench.systems] == ["default"]
    assert bench.datasets[0].options.pack_minutes == 5
    with pytest.raises(ConfigError, match="Unknown system x"):
        bench.pick(systems=["x"])


def test_collar_defaults_and_overrides(tmp_path):
    datasets = [{"type": "fleurs"}, {"type": "reference", "path": "x"},
                {"type": "reference", "name": "strict", "path": "x", "collar": 0}]
    fleurs, own, strict = BenchFile.from_dict({"dataset": datasets}, base_dir=tmp_path).datasets
    assert (fleurs.collar(None), own.collar(None), strict.collar(None)) == (0.0, 0.25, 0)
    assert (fleurs.collar(0.1), own.collar(0.1), strict.collar(0.1)) == (0.1, 0.1, 0)


def test_system_config_never_summarizes(tmp_path, caplog):
    bench = BenchFile.from_dict({"dataset": [{"type": "fleurs"}], "system": [
        {"name": "a", "set": ["podcast.summarizer=llm"]},
        {"name": "b", "set": ["podcast.identifier=pyannote", "whisper.beam-size=2"]}]}, base_dir=tmp_path)
    a, b = (system_config(s, tmp_path) for s in bench.systems)
    assert a.values["podcast"] == {"summarizer": "none", "identifier": "none"}
    assert "summaries don't run" in caplog.text
    assert b.values["podcast"]["identifier"] == "pyannote"
    assert b.values["whisper"]["beam-size"] == 2
    with pytest.raises(ConfigError, match="System c"):
        system_config(type(bench.systems[0])(name="c", set=["whisper.nope=1"]), tmp_path)


def _fake_process(drop=()):
    """Writes a MAT result that says what the reference says, minus the words in drop."""
    calls = []

    def process(item, config, target):
        calls.append(item.id)
        words = []
        for turn in item.turns:
            texts = turn.text.split()
            step = (turn.end - turn.start) / len(texts)
            speaker = {"alice": "s0", "bob": "s1"}[turn.speakers[0]]
            words += [Word(start=turn.start + i * step, end=turn.start + (i + 1) * step, text=text,
                           speakers=[speaker]) for i, text in enumerate(texts) if text not in drop]
        speakers = [Speaker(id="s0", segments=[TimeRange(start=0, end=2)]),
                    Speaker(id="s1", segments=[TimeRange(start=2, end=4)])]
        example = MATResult.read(EXAMPLE).podcast
        podcast = example.model_copy(update={
            "words": words, "segments": words, "speakers": speakers, "diarization": speakers, "language": "en",
            "media": Media(duration=4.0, speech_duration=4.0, sample_rate=16000, max_dbfs=None, rms=None)})
        shutil.copytree(EXAMPLE, target)
        (target / "podcast" / "result.json").write_text(podcast.model_dump_json(by_alias=True))
        return RunInfo(wall_seconds=2.0, steps={"Transcription": 1.5}, peak_gpu_mib=1000)

    process.calls = calls
    return process


def _rows_by_system(rows):
    return {row["system"]: row for row in rows}


def test_run_benchmark_scores_and_reports(tmp_path):
    bench = _bench(tmp_path, system=[{"name": "perfect"}, {"name": "sloppy"}])
    processes = {"perfect": _fake_process(), "sloppy": _fake_process(drop=("there.",))}
    output = tmp_path / "out"

    def process(item, config, target):
        return processes[target.parts[-4]](item, config, target)

    rows = _rows_by_system(run_benchmark(bench, output, process=process))
    perfect, sloppy = rows["perfect"], rows["sloppy"]
    assert (perfect["wer"], perfect["cpwer"], perfect["der"]) == (0, 0, 0)
    assert (perfect["ref_speakers"], perfect["hyp_speakers"], perfect["rtfx"]) == (2, 2, 2.0)
    assert perfect["step_transcription_seconds"] == 1.5
    assert sloppy["wer"] == 0.25 and sloppy["cpwer"] == 0.25
    assert sloppy["agree_wer"] == 0.25 and sloppy["agree_der"] == 0
    assert "agree_wer" not in perfect

    report = (output / "report.md").read_text()
    assert "## refs" in report and "| perfect |" in report and "## Agreement with perfect" in report
    assert (output / "results.csv").read_text().splitlines()[0].startswith("dataset,item,system")
    bench_json = json.loads((output / "results" / "perfect" / "refs" / "ep1" / "bench.json").read_text())
    assert bench_json["error"] is None


def test_run_benchmark_resumes_and_records_failures(tmp_path):
    bench = _bench(tmp_path)
    output = tmp_path / "out"
    first = _fake_process()
    run_benchmark(bench, output, process=first)
    assert first.calls == ["ep1"]

    def broken(item, config, target):
        raise RuntimeError("model exploded")

    (row,) = run_benchmark(bench, output, process=broken)
    assert "error" not in row
    assert row["wer"] == 0

    (row,) = run_benchmark(bench, output, process=broken, rerun=True)
    assert row["error"] == "RuntimeError: model exploded"
    assert "failed" in (output / "report.md").read_text()

    second = _fake_process()
    run_benchmark(bench, output, process=second)
    assert second.calls == ["ep1"]


def test_summarize_uses_totals():
    rows = [{"wer_errors": 1, "wer_words": 10, "audio_seconds": 60, "wall_seconds": 10},
            {"wer_errors": 9, "wer_words": 90, "audio_seconds": 120, "wall_seconds": 20},
            {"error": "boom", "wer_errors": 100, "wer_words": 100}]
    summary = summarize(rows)
    assert summary["wer"] == 0.1 and summary["rtfx"] == 6.0
    assert (summary["files"], summary["failed"]) == (3, 1)
    assert summary["der"] is None


def test_gpu_sampler_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr("MAT.bench.runner.shutil.which", lambda name: None)
    with GpuMemorySampler() as sampler:
        pass
    assert sampler.peak is None
