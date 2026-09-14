# CLAUDE.md

Notes for working on MAT (Media Analytics Toolset) with Claude Code.

## What this is

CLI that runs ML pipelines on media files and writes results to a folder/zip. Two pipelines:

- `PodcastPipeline` (any audio pydub/ffmpeg can open): faster-whisper + whisperx alignment, NeMo Sortformer diarization, pyannote speaker naming, LangChain/OpenAI summary.
- `BookPipeline` (EPUB): chapter detection, langdetect, spaCy sentence split + lemma counts, GLiNER/GLiNER2 NER.

`MAT/reader` loads written results back (format version `"1"` in `meta.json`).

## Commands

This dev machine has no GPU and uses the CPU torch build. Put the group flags on **every** uv call here, otherwise uv
swaps in the default CUDA 12.6 build (a ~3 GB download):

```bash
uv sync --no-default-groups --group dev --group cpu
uv run --no-default-groups --group dev --group cpu pytest tests        # unit tests, no model downloads
uv run --no-default-groups --group dev --group cpu MAT --help
uv run --no-default-groups --group dev --group cpu python scripts/smoke_podcast.py --device cpu   # ~2-3 min
uv run --no-default-groups --group dev --group cpu python scripts/smoke_book.py                   # ~1 min
```

The GPU box uses the defaults: `uv sync`, `uv run python scripts/smoke_podcast.py --device cuda`.

When bumping torch, bump `torchcodec` with it (0.7 <-> torch 2.8, 0.8 <-> 2.9, ...). A mismatch only fails at runtime.

## How the code is wired

- Plugins are found by subclassing. `get_all_concrete_subclasses` finds pipelines, `ConfigClass`es and readers, so a new tool only exists once its module is imported from the package `__init__.py`.
- Every `ConfigClass` gets CLI flags `--<config_name>_<key>`. `_` is the separator, so config names and keys must not contain underscores. JSON config files use the same names as top level keys and are applied after the CLI flags.
- Pipeline steps are closures `PipelineStepInput -> PipelineStepResult(name, data)`. Later steps look up earlier results by the step name string. Missing data means `data=None`, not an exception.
- The writer stores `str(type(result))` in `meta.json` and the v1 reader matches on it. Renaming or moving `PodcastOutput`/`BookOutput` breaks reading old results. Don't change the written layout without a new reader version.

## Machines

- The dev machine is low powered: CPU only, no CUDA. Don't run full pipelines on long audio. Use the smoke scripts in `scripts/` (30 second sample that ships with pyannote.audio, generated EPUB).
- No `OPENAI_API_KEY` and no Hugging Face token here. Fake the summary step and keep audio under 5 minutes so NeMo doesn't need the gated `pyannote/embedding` model.
- Cached models: `mobiuslabsgmbh/faster-whisper-large-v3-turbo`, `Systran/faster-whisper-large-v2`, `nvidia/diar_sortformer_4spk-v1`, `fastino/gliner2-large-v1`. spaCy models are pip-installed at runtime by `spacy_download` and removed again by every `uv sync` (exact sync), so the book smoke script downloads `en_core_web_sm` again after a sync.
- Real runs and benchmarks happen on a separate GPU box with a GTX 1080 Ti (Pascal, compute capability 6.1, 11 GB), set up with uv. Claude can't reach it, the user runs GPU smoke tests and `MAT bench` there and shares the results.
  Known setup: driver 580.178.04, FFmpeg 9.0.1 (too new for torchcodec 0.7), the desktop already uses about 930 MiB of GPU memory. Stage 2 smoke numbers are in `docs/roadmap.md`.

## GTX 1080 Ti limits

Check every new dependency against these before adding it:

- PyTorch only from the CUDA 12.6 index (Pascal kernels exist there up to torch 2.14). cu128/cu129/cu130 wheels fail with "no kernel image".
- No bfloat16, no FlashAttention 2, no vLLM. Load bfloat16 models in float16 or float32 through transformers.
- faster-whisper/CTranslate2: use `int8_float32` or `float32`.
- 11 GB VRAM: one model on the GPU at a time. Free it (`del`, `torch.cuda.empty_cache()`) before the next step.

## Backends (planned, stage 4 of the roadmap)

Until stage 4 lands the pipeline still hardcodes its tools. The target pattern:

- One uv extra per backend. The backend module checks its libraries with `importlib.util.find_spec` at the top and raises `ImportError`, the package `__init__.py` imports it inside `try/except ImportError`. Missing extra means the backend is skipped, not a crash.
- Backend options are pydantic models, config files are TOML, CLI is `MAT run` / `MAT backends` / `MAT config` / `MAT bench` with `--set backend.option=value` overrides.
- Every new backend gets a unit test with a mocked model and a benchmark run on the GPU box before it can become a default.
- Library conflicts: newer libraries usually win, but ask the user before dropping or isolating a backend.

## Git, commits, PRs

- Work on a branch, not on `master`.
- Commits end with `Co-Authored-By: Claude <noreply@anthropic.com>`.
- PR descriptions end with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Never add a link to the Claude chat or session in commits or PRs. `.claude/settings.json` sets this up (`attribution`, `sessionUrl: false`).

## Writing style for human facing text

README, docs, CLI help, log messages, code comments and PR text use simple English and read like a developer on the project wrote them:

- short, direct sentences with concrete technical details
- plain developer words, contractions are fine, "I/we" where it fits
- no marketing language, filler phrases, dramatic intros, rhetorical questions or forced conclusions
- don't overuse em dashes

## Open work

`docs/roadmap.md` has the list of known problems and what's planned next.
