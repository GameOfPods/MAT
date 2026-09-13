# CLAUDE.md

Notes for working on MAT (Media Analytics Toolset) with Claude Code.

## What this is

CLI that runs ML pipelines on media files and writes results to a folder/zip. Two pipelines:

- `PodcastPipeline` (any audio pydub/ffmpeg can open): faster-whisper + whisperx alignment, NeMo Sortformer diarization, pyannote speaker naming, LangChain/OpenAI summary.
- `BookPipeline` (EPUB): chapter detection, langdetect, spaCy sentence split + lemma counts, GLiNER/GLiNER2 NER.

`MAT/reader` loads written results back (format version `"1"` in `meta.json`).

## Commands

```bash
uv sync                      # install (torch comes from the CPU index, see [tool.uv.sources])
uv run pytest tests          # unit tests, no model downloads, a few seconds
uv run MAT --help            # lists every tool option
uv run MAT -i "file.mp3" -o out/
```

## How the code is wired

- Plugins are found by subclassing. `get_all_concrete_subclasses` finds pipelines, `ConfigClass`es and readers, so a new tool only exists once its module is imported from the package `__init__.py`.
- Every `ConfigClass` gets CLI flags `--<config_name>_<key>`. `_` is the separator, so config names and keys must not contain underscores. JSON config files use the same names as top level keys and are applied after the CLI flags.
- Pipeline steps are closures `PipelineStepInput -> PipelineStepResult(name, data)`. Later steps look up earlier results by the step name string. Missing data means `data=None`, not an exception.
- The writer stores `str(type(result))` in `meta.json` and the v1 reader matches on it. Renaming or moving `PodcastOutput`/`BookOutput` breaks reading old results. Don't change the written layout without a new reader version.

## Dev machine limits

- The dev machine is low powered: CPU only, no CUDA. Don't run full pipelines on long audio. For smoke tests use a clip of about 30 seconds (pyannote ships `sample.wav`, it's in the uv cache) and a tiny generated EPUB.
- No `OPENAI_API_KEY` and no Hugging Face token here. Fake the summary step and keep audio under 5 minutes so NeMo doesn't need the gated `pyannote/embedding` model.
- Cached models: `Systran/faster-whisper-large-v2`, `nvidia/diar_sortformer_4spk-v1`, `fastino/gliner2-large-v1`.

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
