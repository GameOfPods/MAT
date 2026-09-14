<p align="center">
  <img src="logo.png" alt="MAT logo" width="280">
</p>

# MAT - Media Analytics Toolset

MAT is a command line tool that takes podcast episodes and EPUB books and pulls structured data out of them. For audio you get a transcript with speakers and timestamps, a diarization and an LLM summary. For books you get chapters, sentences, word counts and named entities.

You point it at some files, it picks the pipeline that fits each file, runs it and writes the results to a folder (or a zip). There's also a small reader API to load those results in Python later.

The project is pre-alpha. Options and the output format can still change.

## What the pipelines do

### Podcasts

Used for any file ffmpeg can decode.

1. **Transcription** with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`large-v2` by default). If [whisperx](https://github.com/m-bain/whisperX) has an alignment model for the detected language we get word timings. If not, we fall back to segment timings.
2. **Diarization** with NVIDIA NeMo Sortformer (`nvidia/diar_sortformer_4spk-v1`). Audio is split into 5 minute chunks to keep memory use down. Speakers in different chunks are linked with pyannote embeddings.
3. **Speaker names** (optional). Give MAT a folder with one short clip per person, named after the person (`alice.mp3`, `bob.wav`). Each diarized speaker is compared against those clips. Speakers without a match keep names like `sprecher_0`.
4. **Transcript**. Every word gets the speaker with the most time overlap, then words are merged into lines like `alice [12.3 - 15.8]: ...`.
5. **Summary** with an OpenAI compatible model through LangChain. Long transcripts are split into chunks and the summary gets refined chunk by chunk.
6. **Media info**: duration, sample rate, loudness, language.

Files written per episode: `media.json`, `transcript.txt`, `transcript.json`, `summary.txt`, `diarization.json`, `diarization.rttm`.

### Books

Used for EPUB files.

1. Read the book and walk the chapters in table of contents order.
2. Keep headings that look like real chapters: numbers, `Chapter 12` / `Kapitel 12`, prologue/epilogue, headings that repeat, or names you pass with `--bookpipeline_chapter-names`. Repeated headings get roman numerals (`Part I`, `Part II`).
3. Detect the language.
4. Split sentences and count lemmas with spaCy. The model is picked by language (English, German, French, multilingual fallback) unless you set `--SpaCy_model`.
5. Named entities per sentence with GLiNER2 (`PERSON`, `LOCATION`, `ORGANIZATION`, `DATE` by default).

Everything ends up in `book.json`.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- ffmpeg 4 to 7 on your `PATH` (torchcodec, which pyannote uses for decoding, doesn't support newer versions yet)
- For the GPU: an NVIDIA driver that supports CUDA 12.6. GTX 10xx cards need the 580 driver branch, later branches dropped them.
- Disk space for models. The first run downloads a few GB into the Hugging Face and torch caches.
- For summaries: `OPENAI_API_KEY`. Set `OPENAI_API_BASE` if you want to use another OpenAI compatible server.
- For speaker names and for audio longer than 5 minutes: a Hugging Face token with access to the gated [`pyannote/embedding`](https://huggingface.co/pyannote/embedding) model. Accept the terms on the model page, then run `huggingface-cli login` or set `HF_TOKEN`.

A GPU helps a lot but isn't needed. Everything runs on CPU, it's just slow.

## Install

```bash
git clone https://github.com/GameOfPods/MAT.git
cd MAT
uv sync
```

`uv sync` installs torch with CUDA 12.6. We stay on 12.6 on purpose: it's the newest PyTorch build that still runs on GTX 10xx cards, and it works on newer cards and on CPU as well.

On a machine without an NVIDIA GPU you can use the smaller CPU build. uv doesn't remember that choice, so the flags go on every `uv sync` and `uv run`:

```bash
uv sync --no-default-groups --group dev --group cpu
uv run --no-default-groups --group dev --group cpu MAT --help
```

If you forget the flags once, uv installs the CUDA build again. That still works, it's just a big download.

## Usage

```bash
uv run MAT -i "episodes/*.mp3" -o results
```

MAT lists how many files it found and asks before it starts. Answer `y` to go, `n` to stop or `l` to print the file list.

Main options:

| Option | What it does |
|---|---|
| `-i`, `--input` | One or more input globs. Quote them so your shell doesn't expand them. |
| `--input-recursive` | Allow `**` in globs |
| `-o`, `--output` | Output folder, created if missing |
| `--output-zip` | Zip each result folder |
| `--keep-uncompressed` | Keep the folder next to the zip |
| `-wd`, `--work-dir` | Where temp files go (default: current directory) |
| `-c`, `--config` | JSON config file |
| `--export-config` | Save the config that was used as `config.json` in each result |
| `--verbose`, `--log-file`, `--log-file-append` | Logging |

If one file fails, MAT writes `<file name>.error.txt` into the output folder and moves on to the next file.

### Tool options

Each tool has its own options, named `--<Tool>_<option>`. `uv run MAT --help` lists all of them. Some examples:

```bash
uv run MAT -i episode.mp3 -o results \
  --Whisper_model medium \
  --Pyannote-Identification_gold-labels speakers/ \
  --LLM-Summarizer_model gpt-4o-mini
```

The same options work in a JSON config file. The top level keys are the tool names:

```json
{
  "Whisper": {"model": "medium", "beam-size": 5},
  "Pyannote-Identification": {"gold-labels": "speakers/"},
  "LLM-Summarizer": {"model": "gpt-4o-mini"}
}
```

Values from the config file currently override command line flags. The easiest way to get a complete file to start from is a run with `--export-config`.

## Output

```
results/
└── episode_2026-09-13_20-15-02/
    ├── meta.json
    └── 0.PodcastOutput/
        ├── media.json
        ├── transcript.txt
        ├── transcript.json
        ├── summary.txt
        ├── diarization.json
        └── diarization.rttm
```

`meta.json` has the result format version, the MAT version, the input file name and hash, and which pipeline wrote which subfolder.

## Reading results in Python

```python
from pathlib import Path
from MAT import MATResult, ResultTypes

result = MATResult.read(Path("results/episode_2026-09-13_20-15-02.zip"))  # folder or zip

for podcast in result.get_results(ResultTypes.PODCAST):
    print(podcast.speaker_names)
    print(podcast.summary)

for book in result.get_results(ResultTypes.BOOK):
    for chapter in book.chapters:
        print(chapter.heading, len(chapter.sentences))
```

Importing `MAT` loads torch and all tools, so this takes a few seconds.

## Development

```bash
uv sync
uv run pytest tests
```

The unit tests don't download any models. For a real check there are two smoke scripts that run the actual pipelines on tiny inputs: the 30 second two speaker clip that ships with pyannote, and a generated EPUB. Run them after dependency updates, and on the GPU box after every install:

```bash
uv run python scripts/smoke_podcast.py --device cuda
uv run python scripts/smoke_book.py
```

The podcast script fakes the summary unless you pass `--summary`. The first run downloads the models (a few GB).

Some notes on the code:

- Tools, pipelines and readers are found by subclassing. A new tool has to be imported in its package `__init__.py`, otherwise MAT never sees it.
- Config names and keys can't contain `_`, because `_` separates tool and option in the CLI flags.
- Pipeline steps pass data to each other by step name. A step that is missing its input returns `None` instead of raising.

`pagebreak.lua` is a pandoc filter that puts a page break before every heading when you convert Markdown (like `summary.txt`) to docx: `pandoc summary.txt -f markdown -o summary.docx --lua-filter pagebreak.lua`.

Known problems and planned work are in [docs/roadmap.md](docs/roadmap.md).

## License

GPL-3.0. See [LICENSE](LICENSE).
