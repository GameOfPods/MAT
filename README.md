<p align="center">
  <img src="logo.png" alt="MAT logo" width="280">
</p>

# MAT - Media Analytics Toolset

MAT is a command line tool that takes podcast episodes and EPUB books and pulls structured data out of them. For audio you get a transcript with speakers and timestamps, a diarization and an LLM summary. For books you get chapters, sentences, word counts and named entities.

You point it at some files, it picks the pipeline that fits each file, runs it and writes the results to a folder (or a zip). There's also a small reader API to load those results in Python later.

Every step of a pipeline is done by a backend (Whisper for speech to text, Sortformer for diarization, ...). Backends are installed as uv extras and picked per run, so new models can be added without touching the pipelines.

The project is pre-alpha. Options and the output format can still change.

## What the pipelines do

### Podcasts

Used for any file ffmpeg can decode.

1. **Transcription** (`whisper`): [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with `large-v3-turbo` by default. If [whisperx](https://github.com/m-bain/whisperX) has an alignment model for the detected language, it aligns the words. If not, whisper's own word timestamps are used.
2. **Diarization** (`sortformer`): NVIDIA NeMo Sortformer (`nvidia/diar_sortformer_4spk-v1`). Audio longer than 5 minutes is cut into pieces at quiet spots to keep memory use down, and the speakers of neighboring pieces are linked with pyannote embeddings.
3. **Speaker names** (`pyannote`, optional): give MAT a folder with one short clip per person, named after the person (`alice.mp3`, `bob.wav`). Each diarized speaker is compared against those clips. Speakers without a match keep names like `sprecher_0`.
4. **Transcript**: every word gets the speaker with the most time overlap, then words are merged into lines like `alice [12.3 - 15.8]: ...`.
5. **Summary** (`llm`, optional): an OpenAI compatible model through LangChain. Long transcripts are split into chunks and the summary gets refined chunk by chunk.
6. **Media info**: duration, sample rate, loudness, language.

### Books

Used for EPUB files.

1. Read the book and walk the chapters in table of contents order.
2. Keep headings that look like real chapters: numbers, `Chapter 12` / `Kapitel 12`, prologue/epilogue, headings that repeat, or names you pass with `--set 'book.chapter-names=["Prolog", "Nachwort"]'`. Repeated headings get roman numerals (`Part I`, `Part II`).
3. Detect the language.
4. Split sentences and count lemmas (`spacy`). The spaCy model is picked by language (English, German, French, multilingual fallback) unless you set `spacy.model`.
5. Named entities per sentence (`gliner`, optional): GLiNER2 with `PERSON`, `LOCATION`, `ORGANIZATION`, `DATE` by default.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- ffmpeg on your `PATH`. The current pipelines work with any recent version (tested with 6.1 and 9.0). torchcodec 0.7, which pyannote uses when it has to open audio files itself, only supports ffmpeg 4 to 7, so that becomes relevant once pyannote reads files directly.
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

`uv sync` installs torch with CUDA 12.6 and all backends. We stay on CUDA 12.6 on purpose: it's the newest PyTorch build that still runs on GTX 10xx cards, and it works on newer cards and on CPU as well.

You can also install only the backends you need. Each one is an extra:

| Extra | Backend | Step |
|---|---|---|
| `whisper` | `whisper` | transcriber |
| `sortformer` | `sortformer` | diarizer |
| `pyannote` | `pyannote` | identifier |
| `llm` | `llm` | summarizer |
| `spacy` | `spacy` | splitter |
| `gliner` | `gliner` | ner |

```bash
uv sync --no-default-groups --group cu126 --extra whisper --extra sortformer --extra pyannote
```

Backends that aren't installed are skipped. `MAT backends` shows what is installed and what to install for the rest.

On a machine without an NVIDIA GPU you can use the smaller CPU build of torch. uv doesn't remember that choice, so the flags go on every `uv sync` and `uv run`:

```bash
uv sync --no-default-groups --group dev --group cpu --group backends
uv run --no-default-groups --group dev --group cpu --group backends MAT --help
```

If you forget the flags once, uv installs the CUDA build again. That still works, it's just a big download.

## Usage

```bash
uv run MAT run -i "episodes/*.mp3" -o results
```

MAT lists how many files it found and asks before it starts. Answer `y` to go, `n` to stop or `l` to print the file list. Pass `--yes` to skip the question, for example in scripts or cron jobs.

The commands:

| Command | What it does |
|---|---|
| `MAT run` | Process files |
| `MAT backends` | List all backends per step, installed or not |
| `MAT backends show NAME` | Options of one backend with type, default and description |
| `MAT config init` | Print a commented config file for the chosen backends |
| `MAT config show` | Print the config a run would use (defaults, config file and `--set` merged) |

Options of `MAT run`:

| Option | What it does |
|---|---|
| `-i`, `--input` | One or more input globs. Quote them so your shell doesn't expand them. |
| `-o`, `--output` | Output folder, created if missing |
| `-y`, `--yes` | Don't ask before processing |
| `--input-recursive` | Allow `**` in globs |
| `-c`, `--config` | TOML config file |
| `--set SECTION.KEY=VALUE` | Set one option, can be used many times |
| `--transcriber`, `--diarizer`, `--identifier`, `--summarizer`, `--splitter`, `--ner` | Pick the backend for a step. `none` skips optional steps. |
| `--output-zip` | Zip each result folder |
| `--keep-uncompressed` | Keep the folder next to the zip |
| `--export-config` | Save the config that was used as `config.toml` in each result |
| `-wd`, `--work-dir` | Where temp files go (default: current directory) |
| `--verbose`, `--log-file`, `--log-file-append` | Logging |

If one file fails, MAT writes `<file name>.error.txt` into the output folder and moves on to the next file. The exit code is 1 if any file failed.

### Backend options

Every pipeline and backend has a config section named after it: `podcast`, `book`, `whisper`, `sortformer`, `pyannote`, `llm`, `spacy`, `gliner`. `MAT backends show whisper` lists the options of a section.

Set single options on the command line:

```bash
uv run MAT run -i episode.mp3 -o results \
  --set whisper.model=medium \
  --set pyannote.gold-labels=speakers/ \
  --summarizer none
```

Values are read as JSON when that works (`8`, `true`, `null`, `["a", "b"]`, `{"k": "v"}`), anything else is a plain string.

For more than a few options use a TOML file. `MAT config init` writes one with every option, its default and a short description:

```bash
uv run MAT config init > mat.toml
uv run MAT run -i "episodes/*.mp3" -o results -c mat.toml
```

```toml
[podcast]
summarizer = "llm"

[whisper]
model = "large-v3-turbo"
beam-size = 5

[pyannote]
gold-labels = "speakers/"
```

The order is: defaults, then the config file, then `--set`. Unknown sections, misspelled options and wrong types stop the run before any file is processed. Sections for backends that aren't installed only give a warning.

### Summaries

The summary uses `gpt-5.6-terra` by default. `gpt-5.6-luna` is a lot cheaper and fine for most episodes. With `OPENAI_API_BASE` pointing at another OpenAI compatible server (DeepSeek, Ollama, llama.cpp) the model name is whatever that server calls the model.

Answers are streamed. If no first token arrives within 15 minutes (`llm.first-token-timeout`, covers the provider queue) or tokens stop for 2 minutes (`llm.idle-timeout`), the call is cancelled and tried again after 30 seconds, then 2 minutes (`llm.max-retries`, default 2). If the summary still fails, the episode is written without it and the error is in the log.

Reasoning models think before they answer, which costs time and tokens. MAT asks for `llm.reasoning-effort = "low"` by default. Provider specific switches go into `llm.extra-body`, for example to turn thinking off completely on DeepSeek:

```bash
OPENAI_API_BASE=https://api.deepseek.com OPENAI_API_KEY=sk-... uv run MAT run -i episode.mp3 -o results --yes \
  --set llm.model=deepseek-flash \
  --set 'llm.extra-body={"thinking": {"type": "disabled"}}'
```

## Output

```
results/
└── episode_2026-09-14_20-15-02/
    ├── meta.json
    ├── config.toml              with --export-config
    └── podcast/
        ├── result.json
        ├── transcript.txt
        ├── summary.md
        └── diarization.rttm
```

`meta.json` has the result format (`2`), the MAT version, the input file with its SHA-1 and the pipelines that ran. An EPUB gets a `book/` folder with a `result.json` instead. `result.json` holds all data: the models and package versions that were used, media info, speakers with their time ranges, every word, the merged lines and the summary. The other files are convenience copies.

The format is described in [docs/result-format.md](docs/result-format.md), with JSON schemas for other languages and example results in `packages/mat-format/`. Results from MAT 0.2.0 and older use a different layout and can't be read anymore.

## Reading results in Python

The reader is a separate small package, `mat-format` in `packages/mat-format`. It only needs pydantic, so other projects can read results without installing MAT and its ML libraries. Unlike MAT it's licensed under Apache 2.0.

```python
from mat_format import MATResult

result = MATResult.read("results/episode_2026-09-14_20-15-02.zip")  # folder or zip

if result.podcast:
    for speaker in result.podcast.speakers:
        print(speaker.id, sum(s.end - s.start for s in speaker.segments), "seconds")
    print(result.transcript())
    print(result.podcast.summary)

if result.book:
    for chapter in result.book.chapters:
        print(chapter.heading, len(chapter.sentences))
```

Inside MAT the same classes are available as `from MAT import MATResult`.

If you change the data model in `packages/mat-format/src/mat_format/models.py`, regenerate the schemas with `uv run python -m mat_format.schema` and update `docs/result-format.md`. Adding a field is fine within a format version, renaming, removing or changing a field needs a new one.

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

The podcast script skips the summary unless you pass `--summary`. The first run downloads the models (a few GB).

GitHub Actions run on every pull request and push to `master` (`.github/workflows/ci.yml`):

- **Tests (all backends, CPU)**: unit tests of MAT and mat-format, schema check, a quick check of the command line
- **Install without backends**: MAT has to start and list every backend as not installed
- **mat-format (Python 3.10, 3.12, 3.13)**: mat-format installed on its own and tested without MAT

When a release is published, `.github/workflows/release-schemas.yml` attaches the JSON schemas and a zip with schemas, spec and examples to it.

### Adding a backend

1. Add an extra with its libraries to `pyproject.toml`.
2. Write a module next to the existing ones, for example `MAT/tools/transcriptors/parakeet/__init__.py`:
   - call `require("nemo", extra="parakeet")` at the top, before anything else from the library is imported
   - define an `Options` pydantic model (subclass of `MAT.utils.config.Options`) with a description per field
   - decorate the class with `@register("transcriber", "parakeet", description="...")` and implement `process`
   - import heavy libraries inside `process`, not at module level
3. Load it at the end of the step's package `__init__.py`: `load_optional("MAT.tools.transcriptors.parakeet", slot="transcriber", name="parakeet", extra="parakeet")`.
4. Add a unit test with a fake model.

A model that transcribes and diarizes in one go subclasses `TranscribeDiarizeTool`, the podcast pipeline then skips the diarizer. Helpers for backends are in `MAT/utils/device.py` (device, dtype, freeing GPU memory) and `MAT/utils/audio.py` (cutting long audio at quiet spots).

Some notes on the code:

- Pipeline steps pass data to each other by step name. A step that is missing its input returns `None` instead of raising.
- Logs go to stderr, command output (like `MAT config init`) to stdout.

`pagebreak.lua` is a pandoc filter that puts a page break before every heading when you convert Markdown to docx: `pandoc summary.md -o summary.docx --lua-filter pagebreak.lua`.

Known problems and planned work are in [docs/roadmap.md](docs/roadmap.md).

## License

MAT is GPL-3.0, see [LICENSE](LICENSE).

The result format package in `packages/mat-format` (data model, reader, JSON schemas and examples) is Apache 2.0, see [packages/mat-format/LICENSE](packages/mat-format/LICENSE), so programs under other licenses can read MAT results.
