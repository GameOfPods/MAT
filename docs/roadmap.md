# Roadmap

Known problems and what we want to do about them. Stage 1 is done on the `fix/restore-v0.2` branch. Every later stage gets its own branch.

**Next up:** a GPU box run of stages 7 and 8 (full_test turns entities and sound events on, a book with `--chapter-summarizer llm`), then the summary comparison (refine, map-reduce, one call) and stage 9. Parakeet as the default transcriber waits for the corrected German reference.

Target hardware for real runs is a separate box with a GTX 1080 Ti (11 GB). Development and unit tests happen on a CPU-only machine. The notes on models and hardware at the bottom explain most of the choices below.

Things we decided up front:

- German and English episodes, one language per episode
- Episodes are often longer than 2 hours. Usually 4 speakers or fewer, but any number has to work.
- Non-commercial model weights are fine
- The output format can change without keeping old results readable
- Defaults get picked from benchmark numbers, not from blog posts
- When libraries conflict, the newer one usually wins. We decide case by case.

## Stage 1: get 0.2.0 working again

Environment

- [x] Python 3.11, torch 2.5.1 from the CPU index, `numpy<2`. `uv lock` resolves, `import MAT` and `MAT --help` work.
- [x] pytest as dev dependency, unit tests in `tests/` (no model downloads)

Bugs

- [x] Reader could not read result folders. It joined paths with `os.pathsep` (`:` on linux) instead of `os.sep`. Only zips worked.
- [x] Without gold labels all speakers got the identity `None` and overwrote each other, so only one speaker survived in `diarization_matched` and the transcript. Unmatched speakers now keep their diarizer label, speakers matched to the same person get merged.
- [x] Whisper fallback for languages without a whisperx alignment model crashed (`words` is `None`, `Word` is not a tuple). It now uses segment timings.
- [x] Whisper crashed with `IndexError` on files without speech.
- [x] Book reader raised `KeyError` for chapters without `sentences`/`sentence_words`/`ner` and called `exit(1)` on bad data. Missing keys are handled now and bad data raises `ValueError`.
- [x] The summary chain got `additional_metadata` (the file name) but the prompts had no placeholder, so it was ignored. It's added to the prompt now unless a custom prompt already uses it.

Smaller fixes

- [x] GLiNER helper functions used the loop variables `txt`/`label` from the outer scope instead of their own arguments
- [x] `--Pyannote-Identification_no-hf-token` did the opposite of its name (passing it turned the token on)
- [x] `align_diarization_with_transcription` computed a full alignment and threw it away
- [x] `timeout_retry` slept once more after the last failed try
- [x] Writer crashed when two results with the same file name were written in the same second
- [x] Stray module level `@property in_file` in `speakeridentification`
- [x] `--export-config` logged the wrong path
- [x] Diarizer exports were missing from `MAT.tools.__all__`
- [x] Readers raise instead of calling `exit(1)` when two readers claim the same version

Docs

- [x] README, CLAUDE.md, this file

## Stage 2: platform upgrade

- [x] Python 3.12, numpy 2
- [x] torch 2.8 with torchcodec 0.7. The torch build is picked with uv dependency groups: `cu126` is the default (also runs on CPU), `cpu` is opt-in. We use groups because uv has no default extras, and without a default a plain `uv run` installs the PyPI build, which doesn't run on the 1080 Ti.
- [x] pyannote.audio 4.0.7, whisperx 3.8.6, faster-whisper 1.2.1, NeMo 3.0, langchain 1.4, gliner2 2.0, transformers 4.53
- [x] Fix the API changes that come with that (pyannote `use_auth_token` became `token`, gliner2 needs its `local` extra). whisperx alignment and NeMo Sortformer kept their signatures.
- [x] Whisper default `large-v3-turbo` instead of `large-v2`. `compute-type` defaults to `auto`, which picks the fastest type the device supports (`int8_float32` on the 1080 Ti and on CPU).
- [x] Smoke scripts in `scripts/` so they can run on the GPU box
- [x] Unit tests and CPU smoke scripts pass on the dev machine, both smoke scripts pass on the GTX 1080 Ti (driver 580.178.04, torch 2.8.0+cu126, CTranslate2 `int8_float32`). First numbers on the 30 second sample, models cached: whisper decoding 0.75 s for 20.7 s of speech (about 28x realtime), whole pipeline 5.7 s, most of it model loading. Peak GPU memory about 1.6 GB on top of the desktop.

## Stage 3: behavior and usability

- [x] Add `--yes` so MAT can run without the interactive confirmation (cron, containers)
- [x] `--LLM-Summarizer_chunk-size` below 200 crashed because the splitter overlap was fixed at 200. New option `--LLM-Summarizer_chunk-overlap`, default 10% of the chunk size and at most 200.
- [x] Default summary model was `gpt-4`, now `gpt-5.6-terra`. `max-tokens` default went from 4096 to 16384 because reasoning models count their thinking tokens there. Not tested against the real API yet (no key on the dev machine).
- [x] Whisper detects the language first and asks faster-whisper for word timestamps when whisperx has no alignment model, instead of using segment timings. The audio is decoded once and shared with the alignment.
- [x] `large-v3-turbo` returned lowercase text without punctuation on the pyannote sample when running on CPU. On the GTX 1080 Ti the same model and audio came out with normal punctuation and casing, also on a real German episode (SPOILER! 5.33, 12.9 minutes, 64 s for the whole pipeline, about 12x realtime, 3.6 GB peak torch memory). CPU inference quirk, nothing to do.

LLM calls (found while testing a DeepSeek summary on the GPU box, the run sat silent after `HTTP 200` for many minutes):

- [x] Streaming on by default (`MAT/tools/summary/llm/robust.py`). Logs the time to the first token, a progress line every 30 seconds while the answer comes in, and the total at the end. The refine chain still gets the full text.
- [x] Timeouts based on streamed tokens. Right now the client has no timeout and can wait forever. DeepSeek answers `200` right away and then sends empty lines until the model is done, and the HTTP read timeout resets on those lines, so it can't tell a slow model from a stuck request. MAT has to watch the stream chunks itself, with async streaming and a timeout on every "wait for the next chunk", and cancel the request when it runs out:
  - `--LLM-Summarizer_first-token-timeout`, default 15 minutes. Covers queueing and prompt processing. DeepSeek's docs say they close the connection after 10 minutes, but in our test run it took 900 seconds until they gave up. If MAT gives up earlier, it only lands at the back of the queue again. A local model on the 1080 Ti can also need minutes for a 32k token prompt.
  - `--LLM-Summarizer_idle-timeout`, default 2 minutes. Maximum gap between two tokens, every token resets it.
  - `--LLM-Summarizer_max-retries`, default 2, for timeouts, connection errors and "server busy" errors, with a growing wait between tries (for example 30 s, then 2 min). DeepSeek reports a queue timeout as HTTP 200 with an error in the body (`We were unable to start processing your request within the 900-second timeout limit`), langchain raises that as a plain `ValueError`, so MAT has to recognize it.
  - No total time limit, `max-tokens` already bounds the answer.
  - Thinking tokens (DeepSeek streams them as `reasoning_content`) don't reach MAT as text, but langchain's OpenAI client still yields an empty chunk for each of them, so they reset the idle timer.
  - langchain-openai has its own `stream_chunk_timeout` (120 s by default, also for the first token). MAT turns it off, otherwise it would cancel waits in a provider queue after 2 minutes.
  - An empty answer (stream ends without a chunk, langchain raises `No generation chunks were returned`) counts as a busy provider and gets retried.
- [x] `--LLM-Summarizer_reasoning-effort`, default `low`, the lowest value both OpenAI (`none`, `low`, `medium`, `high`, `xhigh`, default `medium`) and DeepSeek (`low`, `high`, `max`, thinking on by default at `high`) accept. `unset` doesn't send the parameter. Summaries don't need much thinking, and thinking tokens cost time, money and `max-tokens` on every refine step. Plus `--LLM-Summarizer_extra-body` (JSON) for provider specific switches like DeepSeek's `thinking` or `enable_thinking` on local servers.
- [x] A failed summary must not drop the transcript and diarization of the episode. The episode is written without `summary.txt` and the error goes to the log. Ctrl+C still aborts the whole run. Moved up from stage 10 for the summary step, stage 10 still covers the general case. In the DeepSeek test run the summary failed after 15 minutes in their queue and the finished transcript of the episode was lost with it.
- [x] `--LLM-Summarizer_chunk-size` default from 15000 to 32000 tokens. Every current API model handles that. Automatic sizing comes in stage 7.
- [ ] Test the LLM changes against the real APIs (DeepSeek, OpenAI) on the GPU box. The dev machine has no API key, the unit tests use scripted fake models. DeepSeek works for a 17 minute episode that fits into one call (streamed, 9.4 s, German summary). Several refine calls work too (Ollama, 3 chunks, 2026-09-27). Still open: a busy DeepSeek queue, OpenAI.

## Stage 4: pluggable backends, new CLI and config

Today every tool adds `--<Tool>_<option>` flags to one argparse parser. With more backends and all extras installed `MAT -h` would list hundreds of options. The pipeline also hardcodes Whisper, NeMo and pyannote, so a new backend means editing `PodcastPipeline`.

Config and CLI

- [x] Options of each backend become a typed pydantic model instead of `ConfigElement` dicts. Type, default and help text live in one place. Keys look like `whisper.beam-size`, so the "no underscores" rule goes away.
- [x] Config files are TOML (read with stdlib `tomllib`). Order: defaults, then config file, then `--set key=value` on the command line.
- [x] Subcommands (`MAT bench` comes with stage 5):
  - `MAT run -i ... -o ... [--transcriber X] [--diarizer Y] [-c mat.toml] [--set parakeet.batch-size=8] [--yes]`. `-h` only shows the core options and the backend slots with the choices that are installed.
  - `MAT backends` lists every backend per slot, either available or skipped with the extra you need to install
  - `MAT backends show <name>` prints the options of one backend
  - `MAT config init [--transcriber X ...]` writes a commented TOML file with only the chosen backends
  - `MAT config show -c mat.toml` prints the merged config. Keys for backends that aren't installed log a warning, typos fail.
  - `MAT bench` (stage 5)

Backends

- [x] `PodcastPipeline` gets the slots `transcriber`, `diarizer`, `identifier` and `summarizer`, `BookPipeline` the slots `splitter` and `ner`. Choices are the backends registered for the slot, `none` skips `identifier`, `summarizer` and `ner`.
- [x] One uv extra per backend. So far `whisper`, `sortformer`, `pyannote`, `llm`, `spacy`, `gliner`, plus `all`. The default dependency group `backends` installs `all`, so a plain `uv sync` still gets everything. Stage 6 adds `parakeet`, `diarizen`, `qwen-asr`, `moss`, ...
- [x] Backend registry (`MAT/registry.py`): a backend module calls `require(...)` (find_spec only) and registers with `@register(slot, name)`, the package `__init__.py` loads it with `load_optional(...)`. A missing extra skips the backend, `MAT backends` lists it with the extra to install. Checked with an install without any backend extra.
- [x] New base class `TranscribeDiarizeTool` for models that do both. The pipeline skips the separate diarization step when one of them is picked.
- [x] Shared helpers: `resolve_device`, `torch_dtype` (bfloat16 only on compute capability 8.0+, float16 on Pascal), `free_gpu_memory` in `MAT/utils/device.py`, `plan_windows` in `MAT/utils/audio.py` (cuts long audio at the quietest spot near the window end, optional overlap, `Window.owns` decides which piece keeps a result). Sortformer uses it instead of hard 5 minute cuts.
- [x] Run a long episode (more than 5 minutes, so Sortformer cuts it into pieces) on the GPU box. Linking the pieces needs the gated `pyannote/embedding` model, which the dev machine can't download. `scripts/full_test.sh` on a 17 minute German episode with two hosts: 4 pieces, linked into 3 speakers (one of them only 2 s, see stage 6), all 9 test steps ok, result valid against the schemas.

Output format

- [x] New format 2, old results don't need to stay readable: `meta.json` plus one folder per pipeline (`podcast/`, `book/`) with a `result.json` holding models and package versions, media info, speakers, raw diarization, words, segments, summary, and empty `events`/`entities` for later. `transcript.txt`, `summary.md` and `diarization.rttm` next to it.
- [x] Stable pipeline names (`podcast`, `book`) instead of `str(type(...))`. The v1 reader is removed, `MATResult.read` rejects other formats with a clear error.

Found on the way:

- [x] The console script exited with code 1 on success, because `main` returned a list of result folders. `MAT` now returns a real exit code (1 if a file failed, 2 for config errors).
- [x] Logs went to stdout and got mixed into command output. They go to stderr now, so `MAT config init > mat.toml` gives a clean file.

Result format definition (other programs like Mosaicast, which is Java, need to read results without MAT):

- [x] `packages/mat-format`: uv workspace package with the pydantic data model of format 2, the reader and the JSON schemas. Only needs pydantic, MAT's writer uses it, `MAT.reader` re-exports it.
- [x] JSON schemas (draft 2020-12) generated from the models with `python -m mat_format.schema`. Tests fail when the committed schemas don't match the models and check writer output and the example results against them.
- [x] `docs/result-format.md`: layout (folder or zip), versioning rules (readers ignore unknown fields, `format` only changes on breaking changes), every field, the convenience files, how to read it from other languages.
- [x] Real example results (30 second podcast sample, smoke test EPUB) in `packages/mat-format/examples/`.
- [x] Cleaned up the format before documenting it: speakers and time ranges are objects, `language` only once, `duration_after_vad` is `speech_duration`, `events` and `entities` have a defined shape, loudness of silence is `null` instead of `-Infinity`.
- [x] `mat-format` (package, schemas, examples) is Apache 2.0, MAT stays GPL-3.0. Projects under other licenses can use the reader and the schemas.
- [x] Every published release gets the schemas attached under fixed names (`releases/latest/download/podcast-result.schema.json`) plus a zip with schemas, spec and examples.
- [x] The release workflow first checks that the tag is `v` + the MAT version (`scripts/check_release_version.py`). A wrong tag fails the run and nothing gets attached. GitHub can't stop the release itself from being published.

## Stage 5: benchmark suite

We have no reference transcripts of our own episodes, so the suite uses public datasets for quality and our episodes for speed and for comparing backends with each other.

- [x] `MAT bench run|report|download|datasets|reference` (`MAT/bench`, docs in `docs/benchmarks.md`, example bench file in `benchmarks/example.toml`). A bench file lists systems (backends plus `--set` style settings, summaries never run) and datasets. Every run is stored as a normal MAT result plus `bench.json`, finished runs are skipped, so aborted benchmarks continue and new systems only run themselves. Metrics are always computed from the stored results.
- [x] Speed as RTFx plus seconds per pipeline step, peak GPU memory of the process from `nvidia-smi` (CTranslate2 doesn't report to torch) and torch's own peak.
- [x] WER with jiwer after lowercasing, removing punctuation and German/English fillers. DER with pyannote.metrics. cpWER computed with jiwer and scipy's Hungarian algorithm instead of meeteval, which only ships as source on PyPI and would need compiling on the GPU box.
- [x] Own references in MAT's `transcript.txt` format, so a MAT draft only needs correcting. `MAT bench reference RESULT --start --end` makes one, a scored time range lets the systems run on the whole episode but get compared on the corrected part.
- [x] Datasets, downloaded into a configurable cache (for example on a NAS) or read from an existing copy: FLEURS de/en (read sentences packed into 10 minute files), VoxConverse (DER, 1 to 20+ speakers), AMI (WER, cpWER, DER, meetings), ASR Bundestag (German, WER). Big zips (VoxConverse 4 GB, Bundestag 59 GB) are read with HTTP range requests, so only the picked files get downloaded. This American Life is left out, its license for audio isn't clear.
- [x] Audio without references (`audio` dataset): speed, memory, speaker count and agreement with the first system.
- [x] Report as Markdown and CSV in the output folder.
- [x] Checked on the dev machine: every dataset loader against the real servers (Bundestag and VoxConverse files out of the big zips, AMI meeting with annotations, FLEURS lists), and `MAT bench reference` plus `MAT bench run` with real models on CPU on the 30 second sample: WER and cpWER 0 against its own result, DER 20 % without a collar. Reference lines only cover words, diarizers also cover the pauses around them, so own references use a 0.25 s collar by default (6 % there), public datasets stay at 0.
- [ ] Waiting for the user's corrected reference of about 10 minutes of a German episode.
- [ ] First run with the current backends on the GPU box, public dataset numbers into `docs/benchmarks/`. Every new backend from stage 6 gets benchmarked when it lands.
- [ ] Numbers aren't normalized (5 vs five), which hurts WER on ASR Bundestag. Consider a German/English number normalizer if it matters for picking defaults.
- [ ] Models load again for every file (stage 9). Packing FLEURS and Bundestag sentences into 10 minute files works around that for short clips.
- [ ] Packed files lose speech, and that's what made Parakeet look broken on German. On English FLEURS (2026-09-28 run) Parakeet returned 1081 of 2196 words, with stretches of up to 150 s without a single word inside one 10 minute piece, and whisper lost whole sentences too (188 deletions, 15.7 % WER). German FLEURS scored as one text, which doesn't depend on any timing, comes out at 6.6 % for whisper and 6.9 % for Parakeet. Checked on the dev machine with 30 English sentences packed into 5 minutes:
  - Not the digital zeros in the gaps: quiet noise there changed nothing for Parakeet (50.9 % WER against 40.6 %).
  - Whisper: FLEURS has recordings 40 dB quieter than their neighbours (peaks around -44 dBFS), and the voice activity filter dropped them as silence. Bringing every sentence to -20 dBFS before packing: 21.0 % -> 6.9 % WER, 699 of 700 words. `pack()` does that now, and it's part of the cache key, so old packs get rebuilt.
  - Parakeet: leveling didn't help (47.7 %). The piece length does: 600 s pieces 40.6 %, 120 s 30.1 %, 60 s 22.0 %, 30 s 8.7 % (677 of 700 words). Local attention off: 37.9 %. So Parakeet loses sentences when a long piece is made of unrelated recordings. AMI (continuous meetings) was fine with 600 s pieces, 17.4 % WER against whisper's 19.5 %, so real episodes may not be affected. Check on a real German episode by comparing Parakeet's word count with whisper's, and keep `parakeet-sortformer-30s` in `benchmarks/stage6.toml` next to the default.
  Rerun on the GPU box (2026-09-28): FLEURS WER whisper 11.1 % -> 6.9 %, Parakeet with 30 s pieces 5.9 % (600 s pieces 19.8 %). Bundestag whisper 10.1 %, Parakeet 20.7 % with 30 s pieces and 31.0 % with 600 s, still missing short snippets. On a real 3 hour German episode Parakeet with its default 600 s pieces has no hole longer than 20 s and 18601 words against whisper's 19162, transcribes in 2:08 instead of 11:47, and reads about as well (whisper gets "Azor Hai" and "Lightbringer" right, Parakeet "Celtiger"). So the losses are about packed unrelated snippets, not episodes. A real WER on German needs the corrected reference.

## Stage 6: new speech backends

Each one gets its extra, a backend class, a unit test with a mocked model and a benchmark run on the 1080 Ti.

Known conflict: transformers is capped at `<4.53.3` by spacy-transformers (needed for the `*_trf` spaCy models) and at `<5` by gliner2. Cohere Transcribe needs transformers 5.4+ and Granite Speech 4.1 needs 5.8+. Decide when we get there, for example drop the transformer based spaCy models or wait for new releases.

- [x] 6a: Parakeet TDT 0.6B v3 (`parakeet`), pyannote community-1 (`pyannote-diarization`), WeSpeaker embeddings for linking and for gold label matching (replaces the old gated `pyannote/embedding`). Checked with the real models on the 30 second sample on CPU: Parakeet 81 words with punctuation, language detected, 7.6 % WER against our whisper result; community-1 DER 5.2 %, Sortformer 7.1 %, streaming Sortformer 9.5 %. That sample is far too short to judge quality, the GPU box has to benchmark them (`benchmarks/stage6.toml`). Watch out: pyannote decodes files with torchcodec, and torchcodec 0.7 (the one that fits torch 2.8) only supports FFmpeg 4 to 7. The GPU box has FFmpeg 9. Either pass decoded audio to pyannote (MAT already does that for speaker matching), install FFmpeg 7 there, or move to torch 2.9+ with a newer torchcodec.
- [x] Streaming Sortformer v2.1 as `sortformer-streaming`: the Sortformer backend with the streaming model and its speaker cache settings (NVIDIA's high latency preset). It handles hours of audio in one piece, so no linking between pieces. NeMo 3.0 already has everything, no install from main needed.
- [x] 6b: DiariZen (`diarizen`) as the first backend in its own environment. It pins torch 2.1.1 and its own pyannote fork, so `envs/diarizen` has its own venv and MAT starts `envs/diarizen/run.py` as a process (JSON request, JSON answer, wav on disk). The wiring is reusable (`MAT/utils/external.py`, `MAT external install NAME`, docs in `docs/external-environments.md`), but only DiariZen uses it for now. Checked on the dev machine: `install.sh` builds the environment (torch 2.1.1+cpu, 2.1 GB plus the cloned repo) and the whole path works, MAT decodes the audio, the process runs DiariZen and the result comes back as a normal diarization. On the 30 second sample: DER 4.9 % against community-1 5.2 %, Sortformer 7.1 % and streaming Sortformer 9.5 %, which says nothing about real episodes. On the GPU box every file failed with `MemoryError: batch_size (32) is probably too large`, DiariZen's own config value, which doesn't fit into 11 GB (on CPU it passed because RAM is plentiful). `diarizen.batch-size` now sets it, default 8. With that it runs and wins clearly: VoxConverse DER 3.6 % (community-1 9.3 %, Sortformer 25.9 %), AMI DER 12.8 % (16.3 %, 16.9 %) and the best cpWER there (22.2 %). It costs speed: 3.0 to 3.8x realtime against 6.5 to 12.3x for Sortformer, at 6.8 GB. Weights are non-commercial (CC BY-NC 4.0).
- [ ] Transformers is capped at `<4.53.3` by spacy-transformers and `<5` by gliner2, which blocks Cohere Transcribe (needs 5.4+) and Granite Speech (5.8+). Cheaper than another environment: drop `spacy-transformers` (we don't default to the `*_trf` spaCy models) and check what gliner2 really needs. Update 2026-09-29: spacy-transformers is out (stage 8), only gliner2's `<5` is left.
- [ ] 6c: Qwen3-ASR 1.7B with Qwen3-ForcedAligner for timestamps, MOSS-Transcribe-Diarize and Granite Speech 4.1 2B-plus (both transcribe and diarize). MOSS handles 90 minutes and Granite 9 minutes per pass, so both use the long-audio splitter.
- [ ] 6d: Cohere Transcribe. It needs the language up front and has no timestamps, so language comes from a first pass and timestamps from Qwen3-ForcedAligner.
- [x] Pick new defaults from the benchmark numbers. First run on the GPU box (2026-09-16, `benchmarks/stage6.toml`, one to two files per dataset, so treat single numbers with care):

  | diarizer | VoxConverse DER | speakers off | AMI DER | AMI cpWER | RTFx | peak GPU |
  |---|---|---|---|---|---|---|
  | sortformer (default) | 25.9 % | 4.5 | 16.9 % | 24.1 % | 6.5 to 12.3x | 3.6 to 4.4 GB |
  | sortformer-streaming | 26.3 % | 4.5 | 18.1 % | 23.5 % | 7.3 to 13.9x | 1.7 to 3.0 GB |
  | pyannote-diarization | 9.3 % | 0.5 | 16.3 % | 24.7 % | 5.5 to 7.9x | 6.1 GB |
  | diarizen | 3.6 % | 0.0 | 12.8 % | 22.2 % | 3.0 to 3.8x | 6.8 GB |

  Sortformer's 4 speaker limit is what kills VoxConverse (up to 20 speakers there). The exclusive output of community-1 made DER worse (18.5 % against 16.3 % on AMI) and cpWER slightly better, so it's not a free win for word matching.

  Transcribers: whisper large-v3-turbo against Parakeet gives FLEURS 17.1 % against 20.9 % WER and AMI 20.2 % against 19.3 %, but Parakeet is much faster (AMI 25.7x against 12.3x) and needs 9.2 GB against 4.4 GB.

  Second run (2026-09-28, the limits of the bench file: 5 FLEURS files, 5 VoxConverse files, 4 AMI meetings, 4 Bundestag files):

  | system | VoxConverse DER | speakers off | AMI DER | AMI cpWER | AMI words with 2+ speakers | RTFx on AMI | peak GPU |
  |---|---|---|---|---|---|---|---|
  | whisper + sortformer (default) | 22.8 % | 4.4 | 17.6 % | 23.9 % | 7.0 % | 12.0x | 4.5 GB |
  | whisper + sortformer-streaming | 23.1 % | 4.8 | 23.8 % | 31.5 % | 5.6 % | 12.7x | 2.0 GB |
  | whisper + community-1 | 8.4 % | 1.2 | 14.7 % | 23.3 % | 5.9 % | 7.4x | 6.8 GB |
  | whisper + community-1 exclusive | 8.2 % | 1.2 | 16.9 % | 22.9 % | 0.1 % | 8.4x | 6.8 GB |
  | whisper + diarizen | 5.2 % | 0.8 | 13.3 % | 21.6 % | 6.9 % | 4.2x | 7.2 GB |
  | parakeet + community-1 | 8.4 % | 1.2 | 14.7 % | 21.2 % | | 12.9x | 9.6 GB |

  Parakeet wins on English meetings (AMI WER 17.4 % against 19.5 %, twice as fast) at 9.5 GB. Its FLEURS and Bundestag numbers (29.9 % and 38.7 %) come from the packing problem under stage 5, not from the model. Streaming Sortformer is the low memory option (2 GB) but the worst at matching words to speakers on AMI.

  Open decision: `pyannote-diarization` is better than Sortformer everywhere but needs a Hugging Face login (gated) and more memory, `diarizen` is better again but non-commercial and needs its own environment. A default that needs an account is a real cost for anyone installing MAT. Decided (2026-09-28): the default diarizer is `pyannote-diarization` with `exclusive` on. `MAT run` checks the Hugging Face access before the first file and says what to do (accept the terms, `hf auth login`, or `--diarizer sortformer`), and missing access is no longer retried 5 times a minute apart. Whisper stays the default transcriber until the German reference gives Parakeet a real WER; the README calls Parakeet the fast option.
- [x] Parakeet on ASR Bundestag (54.9 % WER against 8.5 %) and whisper's high FLEURS WER: both come from the packed files losing speech, see stage 5.

Transcript quality (seen on the first real German episode):

- [x] Many tiny fragments assigned to two speakers at once, like `sprecher_0 & sprecher_1 [28.49 - 28.678]: Nicht`, plus single `<Unknown>` words. Sortformer segments overlap and words at speaker changes match both. Try pyannote community-1's exclusive diarization, assign each word to the single speaker with the most overlap, and merge very short fragments into the neighboring line. On a 17 minute episode 112 of 2331 words had two speakers and 95 had none. On a 3 hour episode (2754 transcript lines) 467 lines have no speaker at all (17 %) and 367 have two, so this is the most visible transcript problem we have. One 2 second ghost speaker survived as well, out of three that the linking created.
  Counted per word it's smaller than per line: on a 3 hour episode (Sortformer) 2.2 % of 19162 words have two speakers and 3.0 % have none, but they make 10 % and 18 % of the 1873 lines, because each stray word becomes a line of its own. In the benchmark, community-1's exclusive output brings AMI words with two speakers from 7.0 % (Sortformer) and 5.9 % (community-1) down to 0.1 %, with the best cpWER of the community-1 runs. The GPU box test with community-1 on a German episode ran with Sortformer by mistake, so a German number is still missing.
  Done (2026-09-28): every word gets the one speaker who talks longest during it, a tie in overlapping speech stays with whoever was talking, words between segments take the speaker around them (or the closest segment within `podcast.max-gap`), and a turn under `podcast.min-turn` (0.5 s) inside someone else's sentence goes back to them. `podcast.word-speakers = "overlap"` keeps the old way. Re-scored from the stored GPU box results without models: the 3 hour episode goes from 1873 lines (187 with two speakers, 332 without) to 1033 lines (0 and 2), the 2 hour one from 2122 lines (206, 499) to about 900 (0, 2). AMI cpWER: Sortformer 23.9 % -> 23.2 %, community-1 exclusive 22.9 % -> 22.8 %, community-1 23.3 % -> 23.9 %, DiariZen 21.6 % -> 22.6 %. Meetings are full of real backchannels, so pulling short turns back costs a little there; min-turn 0 gives 23.0 %, 23.8 % and 22.3 % for the last three. Checked on the GPU box on the 3 hour episode: 1033 lines, none with two speakers, 3 words without one, and the step takes 43 ms instead of 4.6 s.
- [x] Speaker time doesn't add up: on the same episode the speakers have 770 s together, but voice activity detection found 991 s of speech. Found with `MAT bench` on AMI IS1009a (14 minutes, 4 speakers): DER 69 %, of that 431 s missed speech out of 696 s. Raw Sortformer (measured on CPU without linking) only misses 99 s with 300 s pieces and 85 s with 150 s pieces, the pipeline missed 431 s and 321 s. Cause: when linking pieces, every local speaker picked its best earlier speaker on its own, two of them could pick the same one, and the second one's segments replaced the first one's. Now the speakers of a piece are linked one to one (Hungarian assignment on the pyannote similarities, `sortformer.link-threshold`), unmatched ones become new speakers and segments are only ever added. Checked on the GPU box: DER 69.1 % -> 29.0 % (300 s pieces) and 58.3 % -> 27.9 % (150 s), cpWER 61.7 % -> 28.8 % and 76.4 % -> 27.6 %, missed speech is now exactly what Sortformer itself misses (99 s and 85 s).
- [ ] Linking Sortformer pieces left a third speaker with one 2 s segment on an episode with two hosts. Either a real short voice (clip, jingle) or a piece that didn't match. Still happens after the linking fix: AMI IS1009a has 4 speakers, MAT finds 5 with 300 s pieces and 7 with 150 s pieces, leaving 52 s and 39 s of confusion plus 51 s and 70 s of false alarm. Measured on the dev machine on 4 AMI meetings (Sortformer pieces cached, then MAT's real linking code, DER over all four):

  | embedding | threshold | DER | confusion | speakers found (4 each) |
  |---|---|---|---|---|
  | `pyannote/embedding` | 0.1 | 23.7 % | 548 s | 5, 6, 5, 4 |
  | `pyannote/embedding` | 0.3 (default) | 24.0 % | 567 s | 5, 7, 9, 8 |
  | `pyannote/embedding` | 0.4 | 19.6 % | 303 s | 6, 11, 12, 9 |
  | `wespeaker-voxceleb-resnet34-LM` | 0.1 | 17.6 % | 179 s | 4, 5, 5, 4 |
  | `wespeaker-voxceleb-resnet34-LM` | 0.3 | 17.7 % | 187 s | 5, 7, 7, 6 |
  | `wespeaker-voxceleb-resnet34-LM` | 0.4 | 18.0 % | 204 s | 5, 7, 8, 8 |

  WeSpeaker is better in both DER and speaker count, and with it a low threshold wins, while the old model needs a high one. So `sortformer.embedding-model` should default to WeSpeaker with a low `link-threshold`, to be confirmed on the GPU box. The missed speech (562 s) and false alarms (313 s) are Sortformer itself and don't change with linking.
- [x] Extra speakers left over after linking: `sortformer.merge-threshold` joins speakers that sound alike over all their audio. Swept link (0.1 to 0.3) against merge (0.3 to 0.7) with WeSpeaker on the same 4 AMI meetings: merging never helped. Up to 0.6 it merges different people (a meeting ends up with 1 to 3 speakers, DER up to 32 %), at 0.7 it changes nothing. Speaker audio from different pieces of a meeting is apparently less similar than different speakers can be. The option stays, off by default.
- [x] Defaults changed from the sweeps: `sortformer.embedding-model` is `pyannote/wespeaker-voxceleb-resnet34-LM` and `link-threshold` is 0.1 (best combination: DER 17.6 %, 4, 5, 5, 4 speakers for 4 real ones). The speaker identifier uses WeSpeaker too, which also drops the Hugging Face login for the default setup (`pyannote/embedding` is gated, WeSpeaker isn't). Both need a check on the GPU box, the identifier one with real gold label clips.
- [ ] Sortformer 4spk-v1 can't take a 14 minute meeting as one piece on the 1080 Ti (out of memory, 6.5 GB more needed). Smaller pieces use much less memory (150 s: 2.0 GB peak for the whole pipeline, 300 s: 5.1 GB) but need more linking. Benchmark `segment-length` against DER after the fix, and compare with streaming Sortformer v2.1, which handles long audio without pieces.
- [x] Show specific names get misheard ("Samuel" for Samwell, "Spoiler-Tile"). faster-whisper has `initial_prompt` and `hotwords`. Add a vocabulary option (per show, for example a text file with character and place names) and check which backends of stage 6 support something similar. Done: `podcast.vocabulary` (a list or a text file) goes to whisper as `hotwords`, which faster-whisper puts in front of every 30 s window, and into the summary's metadata so the LLM spells the names right. Parakeet has no simple way to take it and says so. On the 30 s sample whisper also started writing capitals and punctuation with a vocabulary, which it didn't without one. Still to see on a real episode.

## Stage 7: podcast extras

- [x] LLM settings for either the OpenAI API or a local server: `llm.preset` is `openai`, `ollama` or `llamacpp` and fills service, thinking, answer length and the timeouts (a hosted API queues, a local card chews on a long prompt for minutes). Anything set in the config or with `--set` wins over the preset.
- [x] Say what a run needs and why it can run out of memory, instead of recommending models. 11 GB minus about 1 GB for the desktop doesn't fit an LLM next to the speech models. Measured peaks: whisper large-v3-turbo 4.4 GB, Parakeet 9.2 to 9.6 GB, Sortformer 5.1 GB at 300 s pieces (2.0 GB at 150 s), community-1 6.1 GB, DiariZen 6.8 GB. An 8B model at Q4 wants roughly 6 GB on top of that, so with Parakeet it never fits and with whisper it's tight. Done: table in the README (Requirements, GPU memory).
  Inside MAT this is handled: the steps run one after another and every backend drops its model and calls `free_gpu_memory()` before the next one starts. The problem is a local LLM server, because it's a different process and keeps the model loaded after answering (Ollama's `keep_alive` is 5 minutes by default), so it can still be sitting in VRAM while the next episode starts transcribing.
  Decided (2026-09-22): MAT doesn't manage anybody's LLM. It doesn't start Ollama, doesn't stop it and doesn't unload models behind the user's back. Bringing a model up, locally or hosted, is theirs to arrange, because we have no control over that machine and would own every failure of it ("MAT killed my model", "MAT hangs on a port") without being able to fix them.
  What we owe them instead: numbers and a clear error. Write the peak GPU memory per backend into the docs as an "up to" value (whisper large-v3-turbo 4.4 GB, Parakeet 9.6 GB, Sortformer 5.1 GB at 300 s pieces and 2.0 GB at 150 s, community-1 6.1 GB, DiariZen 6.8 GB), say that the steps run one after another so only one of ours is in VRAM at a time, and say that a local LLM server keeps its model loaded after answering (Ollama: 5 minutes by default), which is why it collides with the next episode. Then people can pick: a smaller model, `segment-length 150`, the LLM on CPU, another machine, or a hosted API.
  Optional and still user controlled: pass `llm.keep-alive` through to Ollama so someone who wants the model dropped right after the summary can ask for it. That's a request parameter, not us managing a server.
- [x] pyannote prints a `ReproducibilityWarning` about TF32 through `warnings` on every run with the GPU. Silence it in `MAT/utils/quiet.py`. Done.
- [x] `MAT config show` and `--export-config` write the values before the preset fills them in (`service = "OpenAI"` next to `preset = "ollama"`). Show what the run will really use. Done: `effective_options` on every configurable, the LLM backends fill in their preset (and the namer what it takes from `[llm]`) there.
- [x] Better error messages, starting with the ones we've already run into. A CUDA out of memory should say which backend was loading, how much it wanted and which option lowers it (`sortformer.segment-length`, `diarizen.batch-size`, a smaller whisper model), instead of a raw torch traceback. DiariZen's failure on the GPU box was `MemoryError: batch_size (32) is probably too large`, which took a code read to understand, and the same will happen to anyone who puts an LLM next to Parakeet. Same treatment for the other traps: a gated model without a Hugging Face login, a missing `OPENAI_API_KEY`, an Ollama that isn't running, and an audio file ffmpeg can't open. Done: a step that runs out of GPU memory raises GpuOutOfMemory naming the backend, its `memory_hint` and the LLM server trap. `MAT run` and `MAT bench` run `preflight` for every backend before the first file: no Hugging Face access for community-1, no `OPENAI_API_KEY`, an Ollama that doesn't answer or lacks the model. A file nothing can read gets the reason (ffmpeg's last line) instead of an empty result.
- [x] Ollama as its own provider (`llm.service = "Ollama"`, `langchain-ollama`). It asks `/api/show` for the context length of the model and sends `num_ctx` with every call, so Ollama loads what we actually need instead of its default (often 2048 or 4096 tokens) and cutting the prompt without saying so. `llm.base-url`, else `$OLLAMA_HOST`, else localhost.
- [x] `chunk-size = auto`, now the default: explicit value wins, then ask the server (llama.cpp `meta.n_ctx` in `/v1/models`, vLLM `max_model_len`, OpenRouter `context_length`, Ollama `/api/show`), then a small built-in table of known API models (OpenAI and DeepSeek don't report it), then 32000. Chunk size = context - `max-tokens` - prompt size - 10% margin (MAT counts with OpenAI's tokenizer, other models count German text differently), capped at 100000 tokens because very long inputs make summaries worse in the middle. With 1M context API models a 2 hour episode then fits into one call.
- [ ] Review and rework the summary pipeline, including the prompts. Known so far:
  - [done] The "system message" was pasted into the user prompt instead of being sent as a system message
  - [done] The question and refine prompts were langchain's old English defaults ("Write a concise summary of the following"). Ours live in `MAT/tools/summary/llm/prompts.py` and ask for the language that is spoken in the transcript.
  - [done] Typos in the system message ("Dont", "beeing", "language of theoriginal"), gone with the new one
  - [done] The deprecated `langchain_classic` refine chain is gone. MAT splits the transcript itself, sends one call when it fits and refines chunk by chunk when it doesn't, and the dependency is out of the `llm` extra.
  - Refine processes chunks one after another, so late chunks can dominate. Compare with one call per episode (now possible with large contexts) and map-reduce for local models with small contexts. `llm.strategy = "map-reduce"` exists now (notes per chunk, then one call over all notes, grouped when they don't fit). Still to do: compare refine, map-reduce and one call on real episodes on the GPU box before changing a default or a preset.
  - [done] The summary is Markdown and gets saved as `summary.md` since format 2
  - Decide what a good summary of an episode should contain (structure, length, topics, speakers, spoilers) and check results against a few episodes, together with the stage 5 benchmark
  - Seen on a real 3 hour episode (German, 3 chunks, 3 refine calls in 2:18, no retries): the chain keeps the beginning and covers the whole episode, but it repeats itself (the same scene described in two sections, one sentence three times), the middle chunks get much more room than the first, and the forced "this is a summary of the text" opening is useless for show notes.
  - [done] Refine calls don't fit the Ollama context: `num_ctx` and the chunk size leave room for the refine prompt and the answer, but not for the summary so far that goes into `{existing_answer}`. From the second chunk on a call can need a full answer more than was loaded, and Ollama then cuts the start of the prompt without an error. llama.cpp with `chunk-size = auto` would reject it instead. Now `chunk-size = auto` gives a transcript that fits one call the whole context, and one that doesn't gets chunks that leave room for the summary so far. Ollama's `num_ctx` counts `max-tokens` once more when there will be refine calls.
  - Checked on the GPU box (2026-09-27): DeepSeek writes good German summaries of a 70 minute and a 3 hour episode in one call each (30 s). qwen3:8b on Ollama wrote English for German episodes although the prompt asks for the transcript's language, invented facts ("his son Davos"), gave a 3 hour episode 350 generic words and added new sections after its own "Conclusion" when refining. Not usable for summaries on this card, which the docs should say.
  - [done] The system message said "You may use contextual information you know on the source material", and the model used it: it compared a scene to "dem Haka der Neuseeländer", which nobody in the episode said (they talked about Icelandic football rituals). The prompts in `MAT/tools/summary/llm/prompts.py` are ours now: transcript aware (they explain the `speaker [start - end]: text` lines), no outside knowledge, no "this is a summary" opening, and the refine prompt says where new material goes and that nothing may be repeated. Still to check against a real episode.
- [x] Speaker names from the transcript when there are no gold clips (`namer = "llm-names"`, off by default). Runs after the identifier and only touches speakers that still carry a diarizer label, so a gold clip always wins. The model has to quote the line that proves a name and say "high" confidence, everything else is dropped, and names that look like labels or repeat another speaker's name are dropped too. A confident name for a speaker the clips already matched is logged as a mismatch (in red on a terminal) and ignored. The model sees the first minutes of the episode plus a few lines of every speaker.
- [x] Optional speaker library (`podcast.speaker-library`, a folder with one JSON file): keeps a few voice prints of every named speaker and recognizes them in later episodes, so a voice named once keeps its name and its `library_id` from then on. `speakers[].name` and `speakers[].library_id` are new optional fields in mat-format 2.1 (added fields, so the format stays 2), which is what Mosaicast needs to count per person across episodes. What may be learned is `speaker-library-learns`: `gold` (default, only names that came from gold clips), `all` (also names the transcript gave us) or `never`. A match needs `speaker-library-threshold` similarity and a clear gap to the runner up, otherwise two similar voices would swap names.
- [x] Tested on the GPU box (2026-09-27). Gold clips taught the library two voices on one episode, and it named both on two other episodes without clips, one of them from another season: similarity 0.96 and 0.94, same `library_id` everywhere. LLM naming on a three person episode (Carla split into two diarizer speakers, so four): DeepSeek named all three correctly with sound quotes. qwen3:8b on Ollama got Max right and gave Carla's voice the name Alex, quoting a line in which that voice greets "my friend Alex", and the wrong name then went into the summary. Both left the second Carla piece unnamed, as the duplicate rule says. An 8B model isn't good enough for naming, the docs have to say that.
- [x] `speakers[].name` is only filled by the library. The spec says a name from gold clips or the transcript goes there too, but results with gold clips or LLM names have `"name": null` (the name is only in `id`). Fill it whenever the id isn't a diarizer label. Done.
- [x] Running the same episode again stores its voice prints again (three runs of one episode gave three identical prints), and with 10 prints per speaker that pushes out other episodes. Keep one print per speaker and episode and replace it on a rerun. Done: every print remembers its episode (`embedding_episodes`), a rerun replaces it, and old files lose their duplicates when they're read.
- [x] Gold clip matching loads the embedding model once per diarizer speaker and embeds all of that speaker's audio (26 to 30 s on a 70 minute episode, and "Found gold labels" is logged once per speaker). Load it once and use a slice of every speaker, like the library does with `speaker-library-seconds`. It was worse than slow: on the 3 hour episode (50+ minutes per host) neither host matched a gold clip and a 2 s ghost speaker got the name alex, while the library recognized both from 2 minutes at 0.96 and 0.94. Done: every speaker gets up to `podcast.match-seconds` (120) and all go to the identifier in one call.
- [x] `llm-names` ignores `[llm]`: `--set llm.model=qwen3:8b` changes the summary model but the namer still asks for `gpt-5.6-terra`, which was a trap on the GPU box. Rule: as long as `llm-names.preset` isn't set, the namer takes preset, service, model and base-url from `[llm]`. Once `llm-names.preset` is set, even to the same value as `llm.preset`, the namer is configured on its own and uses its own defaults, never `llm.model`, so a cloud model name can't end up at Ollama. An explicit `llm-names.model` always wins. "Set" means written in the TOML or given with `--set`, the same check the presets use. Log where the namer's model came from. Done.
- [x] The library runs after the LLM naming, so an episode with only known voices still pays for one LLM call. Move the matching in front of the naming and only ask the LLM about the speakers that are left. Done: the library runs first, the LLM is only asked when a diarizer label is left, and names it finds go into the library afterwards when `speaker-library-learns` is `all`.
- [x] NER on transcripts with GLiNER2: entities with speaker and timestamp, summed up per episode. Done: `--entities gliner` (off by default) runs GLiNER2 on every speaker turn (pieces of at most 150 words), maps the character spans back to words and writes label, text, start, end and speakers into `entities`. mat-format 2.2 adds `PodcastResult.entity_counts()`. The labels have descriptions now ("name of a person or character"): with bare labels GLiNER2 called "you" and "I" a PERSON. On the German episode it finds the characters and places, and also the Seven (Vater, Mutter, Jungfrau, Krieger), which is right for that show. CPU speed: 60 lines in 100 s, so about 30 minutes for a 3 hour episode. Needs a GPU box run for the speed there, then decide whether it should be on by default. Once, 60 lines on CPU crashed with a core dump that didn't happen again in four more tries.
- [x] Audio events: AudioSet tagger (AST or BEATs) for music, laughter and applause, CLAP for custom labels like "jingle". Mark them in the transcript, optionally skip music before transcription and diarization. Done: `--events audioset` (AST, fixed AudioSet classes grouped into music, laughter, applause) and `--events clap` (labels described in words, with background descriptions of normal talk that compete but never become events). 10 s windows every 5 s, neighbours above the threshold join into one event, written to `events`. Checked on CPU with made up music around the 30 s sample: AudioSet found both music parts and nothing in the speech, CLAP called both "jingle" and, with the first background descriptions, also called the phone call "music" (fixed with "people talking in a conversation" and "a person speaking"). About a minute for a minute of audio on CPU, so a GPU box run on real episodes is needed for speed and thresholds. Not done: marking events in transcript.txt (its format is also what `MAT bench reference` reads) and skipping music before transcription.

## Stage 8: books

- [x] GLiNER is asked about one label at a time, so it tends to find something for every label. In a test run `Bob` and `Paris` also came back as `ORGANIZATION`. Pass all labels in one call. Done in stage 7: all labels in one call, Bob is a PERSON again.
- [x] spaCy sentences keep their trailing newline (`"Alice met Bob.\n"`). Strip them before storing. Done: sentences are stripped, empty ones dropped.
- [x] German and English spaCy model defaults Done: `en_core_web_md` and `de_core_news_md` (and `fr_core_news_md`). md instead of lg or trf: same lemmas and sentence borders for our use at a fraction of the size, and the trf models needed spacy-transformers. The language comes from the book's language step now instead of being guessed per chapter, and the model is loaded once per book instead of once per chapter.
- [x] GLiNER2 and spaCy always run on CPU, there is no device option. On the GPU box the book smoke run didn't touch the GPU and NER took 12 s for 5 sentences. Add a device option and move GLiNER2 (and transformer based spaCy models) to the GPU. GLiNER2 done in stage 7 (`gliner.device`, `gliner.batch-size`). spaCy still runs on CPU.
- [x] spaCy models get pip-installed at runtime by `spacy_download`, and every `uv sync` removes them again because it only keeps declared packages. Declare the default models as dependencies (direct wheel URLs) so they stay installed. Done: both are in the `spacy` extra, the wheels come from `[tool.uv.sources]`. `spacy-transformers` is gone from the extra, which also removes one of the two caps on transformers (gliner2's `<5` is left).
- [x] Character list per book: merge name variants, count mentions per chapter Done: `characters` in book/result.json (mat-format 2.3). Titles are dropped, a short name joins the one full name it is part of, a name that fits several stays on its own. Nicknames ("Ned") stay separate, that's what coreference or an LLM would have to do.
- [x] Chapter summaries with the same LLM settings as podcasts Done: `--chapter-summarizer llm`, one summary per chapter in `chapters[].summary`, with prompts for books (`CHAPTER_PROMPTS`) that forbid anything from later chapters or outside knowledge. Not tried on a real book with a real LLM yet.
- [x] Try coreference resolution for characters in German and English. Keep it only if the results are usable. Tried coreferee 1.5 (2026-09, supports spaCy 3.8, English and German, installed in a throwaway environment) on a five sentence Davos/Stannis passage: in English it linked Davos with "her" and Stannis with "woman", in German Stannis with "Frau", and missed "the king" = Stannis in both. Wrong links are worse than none for a character list, so it's not in. maverick-coref-de (KONVENS 2025) is German only research code without a clear license. If this comes back, it's as an LLM job next to the chapter summaries.

## After stage 8: proposals

- [ ] Nicknames, ambiguous short names and coreference for character counts: [proposal](proposals/character-aliases-and-coreference.md). Alias file, pattern candidates, an LLM that judges candidate pairs with evidence, and a coreference experiment where two systems have to agree.
- [ ] NLP tools research (sentences with SaT, GLiNER2 multi for German, keywords, episode chapters, structured LLM output): [proposal](proposals/nlp-tools-2026.md).

## Stage 9: speed

Drop whatever stage 4 already solved.

- [ ] Numbers from the GPU box so far, models cached: 12.9 minute episode 64 s for the whole pipeline (stage 3), 17 minute episode 121 s (stage 4, about 8.5x realtime). Of those 121 s transcription with word alignment took 94 s, diarization 10 s, speaker matching 2 s, summary 10 s. Find out why the second episode was slower per minute (alignment, model loading, reading from the NAS) with the stage 5 benchmark before optimizing.

- [x] Audio is decoded 4+ times per file (`accept`, diarization, speaker matching, media info). Decode once and share. Done: the first podcast step decodes the file once into a 16 kHz mono wav in the work directory (every model wants that anyway) and keeps it in memory for speaker matching and the library. Media info is taken from the original before resampling. On the dev machine 30 minutes of mp3 take 6.4 s to decode and the wav 0.01 s, so a 3 hour episode saves around 3 minutes of decoding. Sound events still read the original, CLAP wants 48 kHz.
- [x] Speaker matching ran even when it couldn't match anything. With the default `identifier = pyannote` and no gold labels it decoded the whole file and concatenated every segment per speaker, and the identifier then returned None without looking at the audio: 2:35 of a 30:30 run on a 3 hour episode. Identifiers now answer `can_match(config)` (pyannote: are there gold labels), the pipeline skips the step when they can't and leaves the identifier out of the models in the result.
- [x] Cache step results per input file (keyed by file hash plus the options of the step) in the work directory or a cache folder, so a re-run after a failed summary doesn't transcribe and diarize the whole episode again. `MAT/utils` already has an unused `get_hash_pipeline`. Done: `MAT/utils/step_cache.py`, on by default in `~/.cache/mat/steps` (`podcast.cache`, `MAT run --no-cache`). Transcription, diarization and sound events are cached, keyed by the sha1 of the file, the backend, its options as the run uses them, package versions, the MAT version and the vocabulary. On the 30 s sample the second run took 1 s instead of 74 s with the same result.json. Benchmarks and the smoke test never use it. The unused `get_hash_pipeline` is gone, the file hash is read in pieces and remembered, the writer needed it too.
- [x] `PodcastPipeline.accept` decodes the whole file just to check the type. Use a probe instead. Done: ffprobe reads the header (0.07 s instead of a full decode), a full decode is the fallback without ffprobe.
- [x] Word/speaker alignment is O(words x segments). Use a sweep over sorted segments. Done with the new word assignment in stage 7 (bisect per speaker, 43 ms instead of 4.6 s on a 3 hour episode). `word-speakers = "overlap"` keeps the old slow way.
- [ ] Models are loaded again for every file. Keep them between files when memory allows.
- [x] `import MAT` pulls in torch and all tools, so even `MAT --help` is slow. Import heavy libraries lazily. Done: two type hints imported torch at startup. `MAT --help` 1.9 s -> 0.4 s on the dev machine, a test keeps torch out of `import MAT.cli`.

## Stage 10: robustness and cleanup

- [ ] One failing step (for example the summary without an API key) drops all results of the file. Make steps fail on their own and keep the rest.
- [ ] Steps that skip because of missing input do it silently. Log a warning.
- [x] `SpeakerIdetificationSpeechBrain.process` was a stub that returned `None`. Removed in stage 4 together with the `speechbrain` dependency, the backend registry makes it easy to add a real one later.
- [x] `DiarizerNEMO._create_config` (old MSDD setup, downloads yaml from GitHub) was unused. Removed in stage 4.
- [x] CI: GitHub Actions for pull requests and pushes to `master`. Unit tests with all backends on CPU torch, an install without any backend, mat-format alone on Python 3.10/3.12/3.13. Dependabot keeps the action versions current.
- [ ] Mark the CI jobs as required status checks in the GitHub branch protection of `master` (repository settings, can't be done from a file).
- [ ] Clean up stale `mat.egg-info`/`MAT.egg-info` folders and decide if `.idea/` belongs in the repo

## Notes on models and hardware (September 2026)

### GTX 1080 Ti

It's a Pascal card (compute capability 6.1) and newer software is dropping it:

- PyTorch removed Pascal from its CUDA 12.8, 12.9 and 13 wheels. The CUDA 12.6 wheels still have it, and PyTorch publishes those up to 2.14 ([issue](https://github.com/pytorch/pytorch/issues/157517), [notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)). That's new enough for pyannote.audio 4 (torch 2.8+), whisperx 3.8 and NeMo 3.0.
- NVIDIA driver branch 580 is the last one with Pascal support. Keep the GPU box on it.
- No bfloat16, no FlashAttention 2, no vLLM. Models published in bfloat16 run in float16 or float32 through transformers, slower than their published numbers.
- CTranslate2 (faster-whisper) runs `int8_float32` natively, `float16` falls back to `float32` ([docs](https://opennmt.net/CTranslate2/quantization.html)).
- 11 GB means one model on the GPU at a time.

### Speech recognition

Whisper has no v4. The newest checkpoints are `large-v3` and `large-v3-turbo`, turbo is about 4x faster and roughly one WER point worse ([whisper](https://github.com/openai/whisper/discussions/1762)). Overview of current open models: [MarkTechPost](https://www.marktechpost.com/2026/07/23/best-open-speech-recognition-asr-models-in-2026-wer-languages-latency-and-license-compared/), [Open ASR Leaderboard](https://arxiv.org/abs/2510.06961).

| Model | Size | License | Timestamps | Why it's interesting for us |
|---|---|---|---|---|
| [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | 0.6B | CC-BY-4.0 | words, native | 25 European languages, runs on NeMo which we already use, local attention for up to 3 h, very fast |
| Whisper large-v3-turbo | 0.8B | MIT | via whisperx | drop-in upgrade of what we have |
| [Qwen3-ASR 1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 1.7B | Apache 2.0 | via Qwen3-ForcedAligner | 30 languages, aligner covers German and English |
| [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | 2B | Apache 2.0 | none | top of the leaderboard, needs the language up front |
| [Granite Speech 4.1 2B-plus](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus) | 2B | Apache 2.0 | words + speakers | transcription and speaker labels in one model, 9 min per pass |
| [MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) | 0.9B | Apache 2.0 | segments + speakers | 50+ languages, 90 min per pass, also tags audio events |

Left out: VibeVoice-ASR (about 9B, too big), Canary-Qwen 2.5B and [Granite Speech 5.0](https://slator.com/ibm-granite-speech-5-transcription-model/) (English only), [Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) (made for streaming, which we don't need).

### Diarization

- [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1): CC-BY-4.0, no speaker limit, has an "exclusive" output made for matching with ASR words. Needs pyannote.audio 4 ([deps](https://raw.githubusercontent.com/pyannote/pyannote-audio/main/pyproject.toml)).
- [DiariZen](https://github.com/BUTSpeechFIT/DiariZen): best open numbers right now (AMI-SDM DER 13.9 vs 22.4 for pyannote 3.1), non-commercial weights, pinned to an old pyannote fork and torch 2.1.1
- [Streaming Sortformer v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1): max 4 speakers, mostly English training data
- Comparison across several languages including German: [Benchmarking Diarization Models](https://arxiv.org/html/2509.26177v1)

### LLMs and other bits

- llama.cpp and Ollama still run on Pascal, without FlashAttention ([example on a GTX 1080](https://mdda.net/blog/tech/dl/llama-cpp-moe-on-an-old-gtx-1080)). Qwen 3.5 and Gemma 4 are Apache 2.0 and have sizes that fit in 11 GB at 4 bit ([comparison](https://www.betterclaw.io/blog/gemma-4-vs-qwen-3-5)). Both serve an OpenAI compatible API, so our summary client works with `OPENAI_API_BASE`.
- Audio events: [OpenBEATs](https://arxiv.org/pdf/2507.14129), [open vocabulary sound event detection](https://arxiv.org/html/2507.16343)
- Speaker embeddings: [Kiwano](https://arxiv.org/html/2606.22369) compares ECAPA2, ReDimNet and WeSpeaker models
- Coreference for German is still mostly research code: [CRAC 2026](https://aclanthology.org/2026.codi-1.22/)
- German long-form reference data: [ASR Bundestag](https://arxiv.org/pdf/2302.06008)
