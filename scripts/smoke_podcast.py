"""
Quick end to end check of the podcast pipeline on a short clip.

Uses the 30 second two speaker sample that ships with pyannote.audio unless you pass --audio.
The summary step is skipped unless you pass --summary (needs OPENAI_API_KEY or OPENAI_API_BASE).

    uv run python scripts/smoke_podcast.py --device cuda
    uv run python scripts/smoke_podcast.py --device cuda --transcriber parakeet --diarizer pyannote-diarization
"""
import argparse
import shutil
import tempfile
from importlib import resources
from pathlib import Path
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", type=Path, default=None, help="Audio file, default: pyannote sample.wav")
    parser.add_argument("--device", default=None, help="cpu or cuda, default: cuda if available")
    parser.add_argument("--transcriber", default="whisper", help="Transcriber backend, default: %(default)s")
    parser.add_argument("--diarizer", default="sortformer", help="Diarizer backend, default: %(default)s")
    parser.add_argument("--whisper-model", default=None, help="Override the whisper model")
    parser.add_argument("--summary", action="store_true", help="Run the real LLM summary instead of a fake one")
    parser.add_argument("--out", type=Path, default=None, help="Output folder, default: a temp folder")
    args = parser.parse_args()

    import torch
    from MAT.pipelines import Pipeline
    from MAT.pipelines.Podcast import PodcastPipeline
    from MAT.reader import MATResult
    from MAT.utils.config import Config
    from MAT.writer import Writer

    audio = args.audio or Path(str(resources.files("pyannote.audio") / "sample" / "sample.wav"))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out or Path(tempfile.mkdtemp(prefix="mat-smoke-"))
    print(f"audio: {audio}\ndevice: {device}\noutput: {out}")
    if device.startswith("cuda"):
        print(f"gpu: {torch.cuda.get_device_name(0)}, compute capability {torch.cuda.get_device_capability(0)}")

    assert PodcastPipeline in Pipeline.get_pipelines(f=str(audio)), "PodcastPipeline did not accept the audio file"

    values = {
        "podcast": {"summarizer": "llm" if args.summary else "none", "transcriber": args.transcriber,
                    "diarizer": args.diarizer},
        "whisper": {"device": device}, "parakeet": {"device": device}, "sortformer": {"device": device},
        "pyannote-diarization": {"device": device}, "pyannote": {"device": device},
    }
    if args.whisper_model:
        values["whisper"]["model"] = args.whisper_model
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    config = Config(values, work_directory=str(work))
    config.validate()

    start = perf_counter()
    result = PodcastPipeline().process(file=str(audio), config=config)
    took = perf_counter() - start

    print("=== transcript")
    print(result.full_transcript)
    print(f"speakers: {sorted(result.diarization_matched.speaker)}")
    print(f"language: {result.media_info.language}, duration: {result.media_info.duration:.1f}s, took {took:.1f}s")
    if device.startswith("cuda"):
        # CTranslate2 (whisper) allocates outside of torch, so this only covers the torch models
        print(f"peak torch gpu memory: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

    folder = Writer().store(file=str(audio), output=str(out / "results"), pipeline_results=[result])
    zipped = shutil.make_archive(folder, "zip", folder)
    for path in (folder, zipped):
        read_back = MATResult.read(path)
        assert read_back.podcast is not None and read_back.transcript(), f"could not read back {path}"
        assert read_back.podcast.models["transcriber"].backend == args.transcriber, f"models missing in {path}"
        assert read_back.podcast.models["diarizer"].backend == args.diarizer, f"models missing in {path}"
    assert len(result.diarization_matched.speaker) >= 2, "expected at least two speakers in the sample"
    print("SMOKE PODCAST OK")


if __name__ == "__main__":
    main()
