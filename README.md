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

1. **Transcription** (`whisper` or `parakeet`): [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with `large-v3-turbo` by default. If [whisperx](https://github.com/m-bain/whisperX) has an alignment model for the detected language, it aligns the words. If not, whisper's own word timestamps are used. Or NVIDIA [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (`--transcriber parakeet`), 25 European languages with word timestamps from the model itself. It's the fast option: a 3 hour German episode took 2 minutes instead of 12 on a GTX 1080 Ti, with about the same number of words and a transcript that reads about as well, but it needs up to 9.6 GB of GPU memory (whisper 4.4 GB). It doesn't report the language, so MAT guesses it from the transcript.
2. **Diarization** (`pyannote-diarization`, `sortformer`, `sortformer-streaming` or `diarizen`): [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) by default, with its exclusive diarization (one speaker at a time). It has no speaker limit, takes the whole file at once and did best in our benchmark apart from the non-commercial DiariZen (VoxConverse DER 8.2 % against 22.8 % for Sortformer). It needs a Hugging Face login, MAT checks that before the first file. Or NVIDIA NeMo Sortformer (`nvidia/diar_sortformer_4spk-v1`, `--diarizer sortformer`), which needs no login and less GPU memory but finds at most 4 speakers per piece: audio longer than 5 minutes is cut into pieces at quiet spots, and the speakers of neighboring pieces are linked with WeSpeaker embeddings. Or [streaming Sortformer v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) (NVIDIA Open Model License, trained mostly on English), which keeps a speaker cache and handles hours of audio without cutting it into pieces.
3. **Speaker names** (`pyannote`, optional): give MAT a folder with one short clip per person, named after the person (`alice.mp3`, `bob.wav`). Each diarized speaker is compared against those clips. Speakers without a match keep names like `sprecher_0`.
4. **Speaker names from the transcript** (`llm-names`, off by default): asks an LLM who is who, based on what is said ("Danke, Alex"). It only runs for speakers the clips didn't match, needs a quoted line as proof and high confidence, and a gold clip always wins. Turn it on with `--namer llm-names`. It talks to the same model as the summary (`[llm]`) unless you give `[llm-names]` a preset of its own. An 8B model isn't good enough for this: in our test qwen3:8b gave one of three people the wrong name, DeepSeek got all three right.
5. **Transcript**: every word gets the one speaker who talks longest during it. Words between segments take the speaker around them, and a turn under half a second in the middle of someone else's sentence goes back to them (`podcast.min-turn`). Then words are merged into lines like `alice [12.3 - 15.8]: ...`. `podcast.word-speakers = "overlap"` keeps the old behavior with lines like `alice & bob`.
6. **Entities** (`gliner`, off by default): people, places, organizations and dates in the transcript with the time and the speaker who said them, from [GLiNER2](https://github.com/fastino-ai/GLiNER2). Turn it on with `--entities gliner`, the labels are `gliner.labels`. `PodcastResult.entity_counts()` in mat-format adds them up per episode. On CPU it takes about 30 minutes for a 3 hour episode, on a GPU a small part of that.
7. **Sound events** (`audioset` or `clap`, off by default): music, laughter and applause with an [AudioSet tagger](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) (`--events audioset`), or anything you describe in words with [CLAP](https://huggingface.co/laion/clap-htsat-unfused) (`--events clap`, labels in `clap.labels`, like `jingle = "a short jingle or intro music"`). Both look at 10 second windows every 5 seconds, so start and end are good to a few seconds. They go into `events` in `result.json`.
8. **Summary** (`llm`, optional): an OpenAI compatible API or a local Ollama. A transcript that fits into the model's context is summarized in one call, a longer one is split into chunks and refined chunk by chunk.
9. **Media info**: duration, sample rate, loudness, language.

### Books

Used for EPUB files.

1. Read the book and walk the chapters in table of contents order.
2. Keep headings that look like real chapters: numbers, `Chapter 12` / `Kapitel 12`, prologue/epilogue, headings that repeat, or names you pass with `--set 'book.chapter-names=["Prolog", "Nachwort"]'`. Repeated headings get roman numerals (`Part I`, `Part II`).
3. Detect the language.
4. Split sentences and count lemmas (`spacy`). The spaCy model is picked by language unless you set `spacy.model`: `en_core_web_md` and `de_core_news_md` come with the `spacy` extra, French and the multilingual fallback get downloaded the first time.
5. Named entities per sentence (`gliner`, optional): GLiNER2 with `PERSON`, `LOCATION`, `ORGANIZATION`, `DATE` by default, all sentences of the book in one go.
6. Character list: the `PERSON` entities of all chapters, with titles dropped and short names joined to the one full name they belong to ("Eddard" and "Lord Eddard Stark" count for Eddard Stark, "Stark" alone stays separate because it fits several people). Mentions per chapter, names mentioned less than `book.min-mentions` (2) times are left out. Nicknames like "Ned" aren't joined.
7. Chapter summaries (`--chapter-summarizer llm`, off by default): one summary per chapter with the `[llm]` settings and prompts written for books, which tell the model to use nothing from later chapters or from what it knows about the book.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- ffmpeg on your `PATH`. The current pipelines work with any recent version (tested with 6.1 and 9.0). torchcodec 0.7, which pyannote uses when it has to open audio files itself, only supports ffmpeg 4 to 7, so that becomes relevant once pyannote reads files directly.
- For the GPU: an NVIDIA driver that supports CUDA 12.6. GTX 10xx cards need the 580 driver branch, later branches dropped them.
- Disk space for models. The first run downloads a few GB into the Hugging Face and torch caches.
- For summaries: `OPENAI_API_KEY`, or a local Ollama. Set `OPENAI_API_BASE` if you want to use another OpenAI compatible server. `MAT run` checks the key (or that Ollama answers and has the model) before the first file, `--summarizer none` skips summaries.
- A Hugging Face login for the default diarizer, [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) (gated): accept its terms on the model page, then run `hf auth login` or set `HF_TOKEN`. `MAT run` checks the access before the first file and says so if it's missing. Without an account use `--diarizer sortformer`, everything else (WeSpeaker for speaker names, Parakeet, whisper) needs no login.

A GPU helps a lot but isn't needed. Everything runs on CPU, it's just slow.

### GPU memory

The steps run one after another and every backend frees the GPU before the next one starts, so what counts is the biggest single step. Measured on a GTX 1080 Ti, whole MAT process including the CUDA context:

| Backend | Up to | To use less |
|---|---|---|
| whisper `large-v3-turbo` | 4.7 GB | `whisper.model=medium`, `whisper.compute-type=int8` |
| parakeet | 9.6 GB | `parakeet.segment-length=300` |
| pyannote-diarization (default) | 6.9 GB | `--diarizer sortformer` |
| sortformer, 300 s pieces | 5.5 GB | `sortformer.segment-length=150` (about 2 GB) |
| sortformer-streaming | 2.0 GB | |
| diarizen | 7.3 GB | `diarizen.batch-size=4` |

Everything else on the card counts too. The desktop takes about 1 GB, and a local LLM server keeps its model loaded after it answered (Ollama for 5 minutes, unless `OLLAMA_KEEP_ALIVE=0`), so an 8B model sitting there collides with the next episode's transcription. MAT doesn't start or stop your LLM server. When a step runs out of memory, MAT says which backend it was and which of these settings helps.

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
| `parakeet` | `parakeet` | transcriber |
| `sortformer` | `sortformer`, `sortformer-streaming` | diarizer |
| `pyannote` | `pyannote-diarization` | diarizer |
| `pyannote` | `pyannote` | identifier |
| `bench` | - | `MAT bench` metrics |
| `llm` | `llm` | summarizer |
| `spacy` | `spacy` | splitter |
| `gliner` | `gliner` | ner, entities |
| `events` | `audioset`, `clap` | events |

```bash
uv sync --no-default-groups --group cu126 --extra whisper --extra sortformer --extra pyannote
```

A few backends can't share these dependencies. [DiariZen](https://github.com/BUTSpeechFIT/DiariZen) pins torch 2.1.1 and its own pyannote fork, so it gets its own environment and MAT runs it as a separate process:

```bash
uv run MAT external list
uv run MAT external install diarizen
uv run MAT run -i episode.mp3 -o out --diarizer diarizen
```

Without that environment the backend is just skipped, like a missing extra. See [docs/external-environments.md](docs/external-environments.md), also for adding more of them. DiariZen's weights are non-commercial (CC BY-NC 4.0).

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

Every pipeline and backend has a config section named after it: `podcast`, `book`, `whisper`, `parakeet`, `sortformer`, `sortformer-streaming`, `pyannote-diarization`, `pyannote`, `llm`, `spacy`, `gliner`. `MAT backends show whisper` lists the options of a section.

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

### Banner

`MAT run` and `MAT bench run` start with a banner showing the MAT version and the result format version. It uses block letters when the console can write them (UTF-8, not `TERM=dumb`) and plain ASCII otherwise. `MAT_BANNER=block` or `MAT_BANNER=ascii` picks one by hand, `MAT_BANNER=none` leaves the banner out for scripts and clean logs (the log line with both versions stays).

### Step cache

Transcription, diarization and sound events are kept in `~/.cache/mat/steps`, keyed by the file content, the backend, its options and the versions involved. Running an episode again, for example after a failed summary or with another summary model, starts right after them. `--no-cache` runs everything again, `--set podcast.cache=/other/folder` moves the cache, and deleting the folder is always fine.

### Show vocabulary

Names that whisper gets wrong ("Samuel" for Samwell) can go into a vocabulary, as a list or a text file with one name per line:

```bash
uv run MAT run -i episode.mp3 -o results --yes --set podcast.vocabulary=~/shows/game-of-pods.txt
```

Whisper expects these words in every 30 second window and the summary spells them that way. Parakeet ignores the list. Keep it to the names that actually go wrong, whisper reads about 600 characters of it.

### Speaker library

MAT can remember voices between episodes. Point it at a folder and every speaker whose name it knows gets a voice print stored there:

```bash
uv run MAT run -i episode.mp3 -o results --yes \
  --set podcast.speaker-library=~/mat-speakers \
  --set pyannote.gold-labels=speakers/
```

In the next episode the same voices are recognized without clips, keep their names, and keep the same `library_id` in `result.json`, so you can count speaking time per person across a whole season.

By default only names that came from gold clips are learned (`podcast.speaker-library-learns = "gold"`). `"all"` also stores names that the LLM read out of the transcript, `"never"` only reads. The library is a single `speakers.json` you can open, fix by hand or delete. It keeps one voice print per speaker and episode (at most 10 per speaker), so running an episode again replaces its print.

### Summaries

The summary uses `gpt-5.6-terra` by default. `gpt-5.6-luna` is a lot cheaper and fine for most episodes. With `OPENAI_API_BASE` pointing at another OpenAI compatible server (DeepSeek, llama.cpp, vLLM) the model name is whatever that server calls the model.

`llm.preset` sets the options that usually go together, and anything you set yourself wins over it:

```bash
uv run MAT run -i episode.mp3 -o results --yes --set llm.preset=ollama --set llm.model=qwen3:8b
```

- `openai`: a hosted API. Long queues are normal, so MAT waits up to 15 minutes for the first token.
- `ollama`: talks to Ollama directly instead of through its OpenAI endpoint, which is the only way to learn the context size of a model and to send `num_ctx`. Through `/v1` Ollama quietly uses its own small default and cuts off the rest of the transcript. Server: `llm.base-url`, else `$OLLAMA_HOST`, else `http://localhost:11434`.
- `llamacpp`: a local OpenAI compatible server. Point `OPENAI_API_BASE` at it, MAT asks it how much context it loaded.

How much transcript goes into one call is `llm.chunk-size`, `auto` by default: MAT asks the server for the context size and fills it, so an episode that fits is summarized in a single call instead of chunk by chunk.

An episode that doesn't fit is handled by `llm.strategy`. `refine` (default) hands the summary so far plus the next chunk to every call. `map-reduce` writes notes for every chunk on its own and then the summary from all notes in one call, so late chunks can't take over and the model never gets to add sections after its own conclusion. That's worth trying with small local models.

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

`scripts/full_test.sh` runs everything in one go: install, unit tests, schema check, both smoke scripts and a complete `MAT run` on an audio file of your choice, which then gets validated against the result schemas. It asks for the device (`cuda` or `cpu`), an output folder, the audio file and optional tokens, or takes them from environment variables (see the top of the script). Each question shows its variable name, and the settings of a run are saved to `full_test.env` in the output folder without the tokens, so `source` that file to skip the questions next time. Tool output goes to log files, the console only shows the steps and results:

```bash
bash scripts/full_test.sh 2>&1 | tee full_test.log
```

### Benchmarks

`MAT bench` runs podcast systems (backends and their settings) on datasets and compares speed, GPU memory, WER, cpWER and DER. It downloads FLEURS, VoxConverse, AMI and ASR Bundestag into a cache folder, and it can use your own corrected transcripts as references. Summaries never run in a benchmark.

```bash
uv run MAT bench datasets
uv run MAT bench run -c benchmarks/example.toml -o bench-results --limit 1
```

See [docs/benchmarks.md](docs/benchmarks.md) for the bench file, the datasets and how to make a reference from one of your episodes.

GitHub Actions run on every pull request and push to `master` (`.github/workflows/ci.yml`):

- **Tests (all backends, CPU)**: unit tests of MAT and mat-format, schema check, a quick check of the command line
- **Install without backends**: MAT has to start and list every backend as not installed
- **mat-format (Python 3.10, 3.12, 3.13)**: mat-format installed on its own and tested without MAT

When a release is published, `.github/workflows/release-schemas.yml` checks that the tag matches the MAT version and then attaches the JSON schemas and a zip with schemas, spec and examples to it.

To release:

1. Set `version` in `pyproject.toml`, for example `0.3.0`, and merge that into `master`. The result format has its own version in `packages/mat-format/pyproject.toml`, it only changes when the format does.
2. Create the release on GitHub with the tag `v0.3.0`, exactly `v` plus the version.

A tag that doesn't match fails the release workflow and nothing gets attached. Fix it by deleting the release and tag and creating them again with the right tag. `python scripts/check_release_version.py v0.3.0` does the same check locally.

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
