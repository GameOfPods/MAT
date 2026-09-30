"""
Runs DiariZen inside its own environment. MAT starts this with two file names: a JSON request and where the JSON
answer goes. Keep it in sync with MAT/tools/diarizators/diarizen/__init__.py.

    python run.py request.json result.json

Request:  {"audio": "/path/file.wav", "model": "BUT-FIT/diarizen-wavlm-large-s80-md", "device": "cuda",
           "batch_size": 8}
Answer:   {"speakers": {"SPEAKER_00": [[start, end], ...]}}
"""
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    from diarizen.pipelines.inference import DiariZenPipeline

    pipeline = DiariZenPipeline.from_pretrained(request["model"])
    device = request.get("device")
    if device and device != "auto":
        import torch

        try:
            pipeline.to(torch.device(device))
        except AttributeError:
            # older pipelines are moved by their own config
            pass

    # The model config asks for 32, which needs more than 11 GB. Both names exist in their pyannote fork:
    # segmentation_batch_size is a property writing into the Inference object, embedding_batch_size a plain attribute.
    batch_size = request.get("batch_size")
    if batch_size:
        for name in ("segmentation_batch_size", "embedding_batch_size"):
            try:
                setattr(pipeline, name, int(batch_size))
            except AttributeError:
                print(f"could not set {name}", file=sys.stderr)

    annotation = pipeline(request["audio"])
    speakers = {}
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        speakers.setdefault(str(speaker), []).append([float(turn.start), float(turn.end)])
    Path(sys.argv[2]).write_text(json.dumps({"speakers": speakers}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
