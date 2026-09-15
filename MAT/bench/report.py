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
"""results.csv (one row per system and file) and report.md (tables per dataset)."""
import csv
import importlib.metadata
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    fields: List[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row[key] for key in fields})


def _ratio(rows: Sequence[Dict[str, Any]], error_keys: Sequence[str], total_key: str) -> Optional[float]:
    scored = [r for r in rows if r.get(total_key) is not None]
    total = sum(r[total_key] for r in scored)
    if not total:
        return None
    return sum(r.get(key) or 0 for r in scored for key in error_keys) / total


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Totals over files: error rates are total errors over the total reference, not a mean of per file rates."""
    ok = [r for r in rows if not r.get("error")]
    audio = sum(r.get("audio_seconds") or 0 for r in ok)
    wall = sum(r.get("wall_seconds") or 0 for r in ok)
    peaks = [r["peak_gpu_mib"] for r in ok if r.get("peak_gpu_mib") is not None]
    off = [abs(r["hyp_speakers"] - r["ref_speakers"]) for r in ok
           if r.get("ref_speakers") is not None and r.get("hyp_speakers") is not None]
    der_parts = ["der_missed", "der_false_alarm", "der_confusion"]
    return {
        "files": len(rows), "failed": len(rows) - len(ok), "audio_seconds": audio,
        "rtfx": audio / wall if wall else None, "peak_gpu_mib": max(peaks) if peaks else None,
        "wer": _ratio(ok, ["wer_errors"], "wer_words"), "cpwer": _ratio(ok, ["cpwer_errors"], "cpwer_words"),
        "der": _ratio(ok, der_parts, "der_total"), "speakers_off": sum(off) / len(off) if off else None,
        "agree_wer": _ratio(ok, ["agree_wer_errors"], "agree_wer_words"),
        "agree_der": _ratio(ok, [f"agree_{p}" for p in der_parts], "agree_der_total"),
    }


def _percent(value: Optional[float]) -> str:
    return "-" if value is None else f"{value * 100:.1f} %"


COLUMNS = [
    ("files", "files", lambda s: str(s["files"])),
    ("audio", "audio_seconds", lambda s: f"{s['audio_seconds'] / 60:.1f} min"),
    ("RTFx", "rtfx", lambda s: "-" if s["rtfx"] is None else f"{s['rtfx']:.1f}x"),
    ("peak GPU", "peak_gpu_mib", lambda s: "-" if s["peak_gpu_mib"] is None else f"{s['peak_gpu_mib']} MiB"),
    ("WER", "wer", lambda s: _percent(s["wer"])),
    ("cpWER", "cpwer", lambda s: _percent(s["cpwer"])),
    ("DER", "der", lambda s: _percent(s["der"])),
    ("speakers off", "speakers_off", lambda s: "-" if s["speakers_off"] is None else f"{s['speakers_off']:.1f}"),
    ("failed", "failed", lambda s: str(s["failed"])),
]
# columns that are left out when no file of the dataset has a value
OPTIONAL = {"peak_gpu_mib", "wer", "cpwer", "der", "speakers_off"}


def _table(header: Sequence[str], lines: Sequence[Sequence[str]]) -> List[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(l) + " |"
                                                                              for l in lines]


def environment() -> Dict[str, str]:
    from MAT import __version__

    info = {"MAT": __version__}
    try:
        info["torch"] = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        pass
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
            if out:
                info["GPU"] = out[0]
        except (OSError, subprocess.SubprocessError):
            pass
    info["CPU"] = platform.processor() or platform.machine()
    return info


def render_markdown(rows: Sequence[Dict[str, Any]], prepared: Sequence[Tuple[Any, List[Any]]],
                    systems: Sequence[Any], bench_collar: Optional[float] = None,
                    env: Optional[Dict[str, str]] = None) -> str:
    env = environment() if env is None else env
    lines = ["# MAT benchmark", "",
             f"Created {datetime.now().strftime('%Y-%m-%d %H:%M')}, " + ", ".join(f"{k} {v}" for k, v in env.items()),
             "", "## Systems", ""]
    lines += _table(["system", "settings"],
                    [[s.name, ", ".join(filter(None, [f"config {s.config}" if s.config else "", *s.set]))
                      or "MAT defaults"] for s in systems])

    for dataset, _ in prepared:
        dataset_rows = [r for r in rows if r["dataset"] == dataset.name]
        summaries = [(s.name, summarize([r for r in dataset_rows if r["system"] == s.name])) for s in systems]
        shown = [c for c in COLUMNS if c[1] not in OPTIONAL or any(summary[c[1]] is not None
                                                                   for _, summary in summaries)]
        der = f", DER collar {dataset.collar(bench_collar)} s" if any(s["der"] is not None for _, s in summaries) else ""
        lines += ["", f"## {dataset.name}", "", f"{dataset.type}, license: {dataset.license}{der}. {dataset.note}", ""]
        lines += _table(["system"] + [c[0] for c in shown],
                        [[name] + [c[2](summary) for c in shown] for name, summary in summaries])

    if len(systems) > 1:
        lines += ["", f"## Agreement with {systems[0].name}", "",
                  f"The results of {systems[0].name} used as reference. Useful for files without a reference.", ""]
        table = []
        for dataset, _ in prepared:
            for system in systems[1:]:
                summary = summarize([r for r in rows if r["dataset"] == dataset.name and r["system"] == system.name])
                table.append([dataset.name, system.name, _percent(summary["agree_wer"]),
                              _percent(summary["agree_der"])])
        lines += _table(["dataset", "system", f"WER vs {systems[0].name}", f"DER vs {systems[0].name}"], table)

    lines += ["", "## How to read this", "",
              "- RTFx: seconds of audio per second of processing, model loading included.",
              "- Peak GPU: highest memory of the MAT process in nvidia-smi, including the CUDA context.",
              "- WER: word errors over the whole file after lowercasing and removing punctuation and fillers "
              "(uh, äh, ...). Numbers aren't normalized, 5 and five count as different words.",
              "- cpWER: like WER, but every reference speaker is compared only with the words of the system speaker "
              "assigned to them, so wrong speakers count as errors.",
              "- DER: missed speech, false alarms and speaker confusion over all reference speech, overlaps included.",
              "- speakers off: average difference between the number of reference and found speakers.",
              "- Rates are totals over all files of a dataset, so long files weigh more.", ""]
    return "\n".join(lines)


def write_report(output: Path, rows: Sequence[Dict[str, Any]], prepared, systems,
                 bench_collar: Optional[float] = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output / "results.csv")
    (output / "report.md").write_text(render_markdown(rows, prepared, systems, bench_collar), encoding="utf-8")


__all__ = ["write_csv", "summarize", "render_markdown", "write_report", "environment"]
