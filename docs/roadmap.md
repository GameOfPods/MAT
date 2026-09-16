# Roadmap

Known problems and what we want to do about them. Stage 1 is done on the `fix/restore-v0.2` branch. Every later stage gets its own branch.

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
- [ ] Test the LLM changes against the real APIs (DeepSeek, OpenAI) on the GPU box. The dev machine has no API key, the unit tests use scripted fake models. DeepSeek works for a 17 minute episode that fits into one call (streamed, 9.4 s, German summary). Still open: a long episode with several refine calls, a busy DeepSeek queue, OpenAI.

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

## Stage 6: new speech backends

Each one gets its extra, a backend class, a unit test with a mocked model and a benchmark run on the 1080 Ti.

Known conflict: transformers is capped at `<4.53.3` by spacy-transformers (needed for the `*_trf` spaCy models) and at `<5` by gliner2. Cohere Transcribe needs transformers 5.4+ and Granite Speech 4.1 needs 5.8+. Decide when we get there, for example drop the transformer based spaCy models or wait for new releases.

- [ ] 6a: Parakeet TDT 0.6B v3 (transcriber), pyannote community-1 (diarizer), WeSpeaker embeddings from pyannote 4 for gold label matching (replaces the old gated `pyannote/embedding`). Watch out: pyannote decodes files with torchcodec, and torchcodec 0.7 (the one that fits torch 2.8) only supports FFmpeg 4 to 7. The GPU box has FFmpeg 9. Either pass decoded audio to pyannote (MAT already does that for speaker matching), install FFmpeg 7 there, or move to torch 2.9+ with a newer torchcodec.
- [ ] 6b: DiariZen (diarizer). Its pyannote fork pins torch 2.1.1, so this needs a port, a separate environment or a subprocess. Streaming Sortformer v2.1 (diarizer, max 4 speakers, handles long audio without our chunk linking).
- [ ] 6c: Qwen3-ASR 1.7B with Qwen3-ForcedAligner for timestamps, MOSS-Transcribe-Diarize and Granite Speech 4.1 2B-plus (both transcribe and diarize). MOSS handles 90 minutes and Granite 9 minutes per pass, so both use the long-audio splitter.
- [ ] 6d: Cohere Transcribe. It needs the language up front and has no timestamps, so language comes from a first pass and timestamps from Qwen3-ForcedAligner.
- [ ] Pick new defaults from the benchmark numbers

Transcript quality (seen on the first real German episode):

- [ ] Many tiny fragments assigned to two speakers at once, like `sprecher_0 & sprecher_1 [28.49 - 28.678]: Nicht`, plus single `<Unknown>` words. Sortformer segments overlap and words at speaker changes match both. Try pyannote community-1's exclusive diarization, assign each word to the single speaker with the most overlap, and merge very short fragments into the neighboring line. On a 17 minute episode 112 of 2331 words had two speakers and 95 had none.
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
- [ ] Show specific names get misheard ("Samuel" for Samwell, "Spoiler-Tile"). faster-whisper has `initial_prompt` and `hotwords`. Add a vocabulary option (per show, for example a text file with character and place names) and check which backends of stage 6 support something similar.

## Stage 7: podcast extras

- [ ] LLM settings for either the OpenAI API or a local OpenAI compatible server (Ollama, llama.cpp). Document a few model picks that fit on the 1080 Ti.
- [ ] Ollama as its own provider (`langchain-ollama`). Chat through Ollama's OpenAI compatible `/v1` endpoint already works, but only the native API (`/api/show`, `/api/chat`) exposes the context size and lets us set `num_ctx` per request. Through `/v1` the server default applies (often 2048 or 4096 tokens) and Ollama cuts longer prompts without an error.
- [ ] `chunk-size = auto`: explicit value wins, then ask the server (llama.cpp `meta.n_ctx` in `/v1/models`, vLLM `max_model_len`, OpenRouter `context_length`, Ollama `/api/show`), then a small built-in table of known API models (OpenAI and DeepSeek don't report it), then 32000. Chunk size = context - `max-tokens` - prompt size - 10% margin (MAT counts with OpenAI's tokenizer, other models count German text differently), capped at 100000 tokens because very long inputs make summaries worse in the middle. With 1M context API models a 2 hour episode then fits into one call.
- [ ] Review and rework the summary pipeline, including the prompts. Known so far:
  - The "system message" is pasted into the user prompt instead of being sent as a system message
  - The question and refine prompts are langchain's old English defaults ("Write a concise summary of the following"). Write our own, and decide how German and English episodes are handled.
  - Typos in the system message ("Dont", "beeing", "language of theoriginal")
  - It uses deprecated `langchain_classic` chains, and some imports (`ConditionalPromptSelector`, `MapReduceChain`) are unused. Consider plain client calls instead of the legacy chain.
  - Refine processes chunks one after another, so late chunks can dominate. Compare with one call per episode (now possible with large contexts) and map-reduce for local models with small contexts.
  - The summary is Markdown but gets saved as `summary.txt`
  - Decide what a good summary of an episode should contain (structure, length, topics, speakers, spoilers) and check results against a few episodes, together with the stage 5 benchmark
- [ ] Speaker names from the transcript when there are no gold clips: send the first minutes and some lines per speaker to the LLM, get name guesses with the lines that support them
- [ ] Optional speaker library: keep embeddings of named speakers and match them automatically in later episodes. This gives stable speaker ids across episodes, which Mosaicast needs for per speaker stats. Add them as new fields on `Speaker` in `mat-format` (for example `name`, `library_id`), so the format version stays 2.
- [ ] NER on transcripts with GLiNER2: entities with speaker and timestamp, summed up per episode
- [ ] Audio events: AudioSet tagger (AST or BEATs) for music, laughter and applause, CLAP for custom labels like "jingle". Mark them in the transcript, optionally skip music before transcription and diarization.

## Stage 8: books

- [ ] GLiNER is asked about one label at a time, so it tends to find something for every label. In a test run `Bob` and `Paris` also came back as `ORGANIZATION`. Pass all labels in one call.
- [ ] spaCy sentences keep their trailing newline (`"Alice met Bob.\n"`). Strip them before storing.
- [ ] German and English spaCy model defaults
- [ ] GLiNER2 and spaCy always run on CPU, there is no device option. On the GPU box the book smoke run didn't touch the GPU and NER took 12 s for 5 sentences. Add a device option and move GLiNER2 (and transformer based spaCy models) to the GPU.
- [ ] spaCy models get pip-installed at runtime by `spacy_download`, and every `uv sync` removes them again because it only keeps declared packages. Declare the default models as dependencies (direct wheel URLs) so they stay installed.
- [ ] Character list per book: merge name variants, count mentions per chapter
- [ ] Chapter summaries with the same LLM settings as podcasts
- [ ] Try coreference resolution for characters in German and English. Keep it only if the results are usable.

## Stage 9: speed

Drop whatever stage 4 already solved.

- [ ] Numbers from the GPU box so far, models cached: 12.9 minute episode 64 s for the whole pipeline (stage 3), 17 minute episode 121 s (stage 4, about 8.5x realtime). Of those 121 s transcription with word alignment took 94 s, diarization 10 s, speaker matching 2 s, summary 10 s. Find out why the second episode was slower per minute (alignment, model loading, reading from the NAS) with the stage 5 benchmark before optimizing.

- [ ] Audio is decoded 4+ times per file (`accept`, diarization, speaker matching, media info). Decode once and share.
- [ ] Cache step results per input file (keyed by file hash plus the options of the step) in the work directory or a cache folder, so a re-run after a failed summary doesn't transcribe and diarize the whole episode again. `MAT/utils` already has an unused `get_hash_pipeline`.
- [ ] `PodcastPipeline.accept` decodes the whole file just to check the type. Use a probe instead.
- [ ] Word/speaker alignment is O(words x segments). Use a sweep over sorted segments.
- [ ] Models are loaded again for every file. Keep them between files when memory allows.
- [ ] `import MAT` pulls in torch and all tools, so even `MAT --help` is slow. Import heavy libraries lazily.

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
