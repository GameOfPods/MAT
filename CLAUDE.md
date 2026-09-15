# CLAUDE.md

Notes for working on MAT (Media Analytics Toolset) with Claude Code.

## What this is

CLI that runs ML pipelines on media files and writes results to a folder/zip. Two pipelines:

- `podcast` (any audio pydub/ffmpeg can open), slots: `transcriber` (whisper), `diarizer` (sortformer), `identifier` (pyannote or none), `summarizer` (llm or none).
- `book` (EPUB), slots: `splitter` (spacy), `ner` (gliner or none).

`MAT/reader` loads written results back (format 2, see `MAT/writer`).

## Commands

This dev machine has no GPU and uses the CPU torch build. Put the group flags on **every** uv call here, otherwise uv
swaps in the default CUDA 12.6 build (a ~3 GB download):

```bash
uv sync --no-default-groups --group dev --group cpu --group backends
uv run --no-default-groups --group dev --group cpu --group backends pytest tests        # unit tests, no model downloads
uv run --no-default-groups --group dev --group cpu --group backends MAT backends
uv run --no-default-groups --group dev --group cpu --group backends python scripts/smoke_podcast.py --device cpu   # ~1-3 min
uv run --no-default-groups --group dev --group cpu --group backends python scripts/smoke_book.py                   # ~1 min
```

`.venv/bin/python -m pytest tests` and `.venv/bin/MAT ...` also work once the env is synced.

The GPU box uses the defaults: `uv sync`, `uv run python scripts/smoke_podcast.py --device cuda`.

`bash scripts/full_test.sh 2>&1 | tee full_test.log` runs install, tests, smoke scripts and a complete run on one audio
file. Settings come from `MAT_TEST_*` env vars or get asked for. Never put paths or names from the user's machines into
it as defaults.

When bumping torch, bump `torchcodec` with it (0.7 <-> torch 2.8, 0.8 <-> 2.9, ...). A mismatch only fails at runtime.

## How the code is wired

- Backends: `MAT/registry.py`. A backend module calls `require(modules..., extra=...)` first (find_spec only, raises
  `MissingDependencies`), then defines its class with `@register(slot, name, description=...)`. The step package
  `__init__.py` loads it with `load_optional(module, slot, name, extra)`, so a missing extra skips the backend.
  Heavy imports stay inside `process`. Every backend has its own uv extra, `all` has all of them and the default
  `backends` dependency group installs `all`.
- Config: `MAT/utils/config`. Every pipeline/backend has an `Options` pydantic model (extra="forbid") and a section
  (the backend name, or `podcast`/`book`). Keys are kebab-case in TOML and `--set` (`whisper.beam-size`), snake_case in
  Python (`config.options(self).beam_size`). Values: defaults < `-c file.toml` < `--set`. `Config.validate()` fails on
  unknown sections/options/types and unknown slot backends, and only warns for sections of backends that aren't installed.
- CLI: `MAT/cli.py` with `run`, `backends [show NAME]`, `config init|show`. Backend options are never argparse flags
  (keeps `run -h` short), only the slot flags (`--transcriber` ...) are. `main()` returns an exit code.
  Logs go to stderr, stdout is for command output.
- Pipelines: `MAT/pipelines`. `Pipeline.backend(slot, config)` creates the chosen backend and records `describe()`
  (backend, model, package versions) in `pipeline.models`, which ends up in the result. Steps are closures
  `PipelineStepInput -> PipelineStepResult(name, data)` and read earlier results with `step_input.data(name)`.
  A `TranscribeDiarizeTool` as transcriber replaces the diarizer step.
- Output format 2: `<stem>_<time>/meta.json` plus `podcast/` or `book/` with `result.json` (and `transcript.txt`,
  `summary.md`, `diarization.rttm`). Spec in `docs/result-format.md`. The data model, reader and JSON schemas live in the
  uv workspace package `packages/mat-format` (only pydantic, no MAT imports). `MAT/writer` builds those models.
  After changing `mat_format/models.py`: run `.venv/bin/python -m mat_format.schema`, update the spec, and remember
  that renaming/removing/retyping a field needs a new format version (adding fields doesn't). Other programs
  (Mosaicast, Java) read results through the schemas, so don't break the format casually.
- Benchmarks: `MAT/bench` (`MAT bench`, docs in `docs/benchmarks.md`). Dataset types subclass
  `MAT/bench/datasets/base.py:Dataset` and get `@register`, they download into the cache or read an existing copy
  (`path`). Summaries never run in benchmarks. Unit tests use fake `process` functions and tiny generated files, no
  models or downloads. Results of the user's own episodes stay local, never commit them.
- Helpers for backends: `MAT/utils/device.py` (`resolve_device`, `ct2_compute_type`, `torch_dtype`, `free_gpu_memory`),
  `MAT/utils/audio.py` (`plan_windows` cuts long audio at quiet spots, `Window.owns` decides who keeps results in overlaps).

## Machines

- The dev machine is low powered: CPU only, no CUDA. Don't run full pipelines on long audio. Use the smoke scripts in `scripts/` (30 second sample that ships with pyannote.audio, generated EPUB).
- No `OPENAI_API_KEY` and no Hugging Face token here. Keep `--summarizer none` (the smoke script does that) and keep audio under 5 minutes so Sortformer doesn't need the gated `pyannote/embedding` model to link audio pieces.
- Cached models: `mobiuslabsgmbh/faster-whisper-large-v3-turbo`, `Systran/faster-whisper-large-v2`, `nvidia/diar_sortformer_4spk-v1`, `fastino/gliner2-large-v1`. spaCy models are pip-installed at runtime by `spacy_download` and removed again by every `uv sync` (exact sync), so the book smoke script downloads `en_core_web_sm` again after a sync.
- Real runs and benchmarks happen on a separate GPU box with a GTX 1080 Ti (Pascal, compute capability 6.1, 11 GB), set up with uv. Claude can't reach it, the user runs GPU smoke tests and `MAT bench` there and shares the results.
  Known setup: driver 580.178.04, FFmpeg 9.0.1 (too new for torchcodec 0.7), the desktop already uses about 930 MiB of GPU memory. It has an HF token and API keys. Stage 2 smoke numbers are in `docs/roadmap.md`.

## GTX 1080 Ti limits

Check every new dependency against these before adding it:

- PyTorch only from the CUDA 12.6 index (Pascal kernels exist there up to torch 2.14). cu128/cu129/cu130 wheels fail with "no kernel image".
- No bfloat16, no FlashAttention 2, no vLLM. Load bfloat16 models with `torch_dtype(device)` (float16 on Pascal) through transformers.
- faster-whisper/CTranslate2: use `int8_float32` or `float32` (`ct2_compute_type` does that).
- 11 GB VRAM: one model on the GPU at a time. `del model` then `free_gpu_memory()` before the next step.

## Adding a backend

- New extra in `pyproject.toml`, module with `require` + `Options` + `@register`, `load_optional` in the step package.
- Unit test with a mocked model, benchmark run on the GPU box before it can become a default.
- Library conflicts: newer libraries usually win, but ask the user before dropping or isolating a backend.

## CI

- `.github/workflows/ci.yml` runs on PRs and pushes to `master`: "Tests (all backends, CPU)" (`uv sync --locked`, so
  `uv.lock` has to be committed and current), "Install without backends", "mat-format (Python 3.10/3.12/3.13)"
  (mat-format installed alone, tests in `packages/mat-format/tests` must not import MAT).
- `.github/workflows/release-schemas.yml` runs on every published release. First `scripts/check_release_version.py`
  checks that the tag is `v` + `MAT/__version__.py` (0.2.0 -> v0.2.0), then it attaches the generated schemas and
  `mat-result-format.zip`. Keep the schema file names stable, other projects download them from
  `releases/latest/download/`.
- `mat-format`'s major version (`packages/mat-format/pyproject.toml`) has to equal `FORMAT_VERSION`, a test checks it.
- `packages/mat-format` is Apache 2.0, MAT is GPL-3.0. Don't copy GPL code into `mat_format`.

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
