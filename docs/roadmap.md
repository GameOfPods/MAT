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

- [ ] Python 3.12, numpy 2
- [ ] torch 2.8 or newer. Two uv extras, `cpu` for the dev machine and `cu126` for the GPU box, marked as conflicting in `[tool.uv] conflicts`. Only the cu126 wheels still run on the 1080 Ti (see notes).
- [ ] pyannote.audio 4, whisperx 3.8, faster-whisper 1.2, NeMo 3.0, current langchain
- [ ] Fix the API changes that come with that (for example pyannote `use_auth_token` became `token`, whisperx alignment and loading)
- [ ] Whisper default `large-v3-turbo` instead of `large-v2`. Pick `compute-type` from the device (`int8_float32` on the 1080 Ti).
- [ ] Done when unit tests and the CPU smoke scripts pass here and the smoke scripts also run on the GPU box

## Stage 3: behavior and usability

- [ ] Add `--yes` so MAT can run without the interactive confirmation (cron, containers)
- [ ] `--LLM-Summarizer_chunk-size` below 200 crashes because the splitter overlap is fixed at 200. Make the overlap relative or configurable.
- [ ] Default summary model is `gpt-4`. Pick a current default.
- [ ] Whisper: detect the language first and ask faster-whisper for word timestamps when there is no alignment model, instead of segment timings

## Stage 4: pluggable backends, new CLI and config

Today every tool adds `--<Tool>_<option>` flags to one argparse parser. With more backends and all extras installed `MAT -h` would list hundreds of options. The pipeline also hardcodes Whisper, NeMo and pyannote, so a new backend means editing `PodcastPipeline`.

Config and CLI

- [ ] Options of each backend become a typed pydantic model instead of `ConfigElement` dicts. Type, default and help text live in one place. Keys look like `whisper.beam-size`, so the "no underscores" rule goes away.
- [ ] Config files are TOML (read with stdlib `tomllib`). Order: defaults, then config file, then `--set key=value` on the command line.
- [ ] Subcommands:
  - `MAT run -i ... -o ... [--transcriber X] [--diarizer Y] [-c mat.toml] [--set parakeet.batch-size=8] [--yes]`. `-h` only shows the core options and the backend slots with the choices that are installed.
  - `MAT backends` lists every backend per slot, either available or skipped with the extra you need to install
  - `MAT backends show <name>` prints the options of one backend
  - `MAT config init [--transcriber X ...]` writes a commented TOML file with only the chosen backends
  - `MAT config show -c mat.toml` prints the merged config. Keys for backends that aren't installed log a warning, typos fail.
  - `MAT bench` (stage 5)

Backends

- [ ] `PodcastPipeline` gets the slots `transcriber`, `diarizer`, `identifier` and `summarizer`. Choices come from the subclasses of each tool base class, keyed by `config_name()`.
- [ ] One uv extra per backend (`whisper`, `parakeet`, `pyannote`, `diarizen`, `qwen-asr`, `moss`, `events`, ...)
- [ ] Backend modules get imported from their package `__init__.py` inside `try/except ImportError`. Each backend module checks its libraries with `importlib.util.find_spec` at the top, so a missing extra skips the backend without importing torch at startup.
- [ ] New base class `TranscribeDiarizeTool` for models that do both. The pipeline skips the separate diarization step when one of them is picked.
- [ ] Shared helpers: pick device and dtype (Pascal aware), unload a model and free GPU memory, split long audio into windows at pauses (with overlap) and merge the results. Any backend with a length limit uses the same splitter.

Output format

- [ ] New format, old results don't need to stay readable: one JSON per pipeline with a schema version, the models and versions that were used, segments, words, speakers, events and entities. Human readable `transcript.txt` and `summary.md` next to it.
- [ ] Stable type names instead of `str(type(...))`. Remove the v1 reader.

## Stage 5: benchmark suite

We have no reference transcripts of our own episodes, so the suite uses public datasets for quality and our episodes for speed and for comparing backends with each other.

- [ ] `benchmarks/` and a `MAT bench` subcommand that runs every installed backend on a list of datasets
- [ ] Speed as real time factor, peak GPU memory (torch stats plus `nvidia-smi` sampling, because CTranslate2 doesn't report to torch)
- [ ] WER with jiwer and text normalization for German and English. DER with pyannote.metrics. cpWER with meeteval where there are speaker labeled references.
- [ ] Datasets (small subsets, check each license first): FLEURS de/en for short WER, ASR Bundestag for German long-form, VoxConverse and AMI for diarization with varying speaker counts, This American Life for English long-form podcasts
- [ ] Our own episodes without references: speed, memory, estimated speaker count, how much backends agree with each other
- [ ] Report as Markdown and CSV in `docs/benchmarks/`, run on the GPU box
- [ ] First run with the current backends. Every new backend from stage 6 gets benchmarked when it lands.

## Stage 6: new speech backends

Each one gets its extra, a backend class, a unit test with a mocked model and a benchmark run on the 1080 Ti.

- [ ] 6a: Parakeet TDT 0.6B v3 (transcriber), pyannote community-1 (diarizer), WeSpeaker embeddings from pyannote 4 for gold label matching (replaces the old gated `pyannote/embedding`)
- [ ] 6b: DiariZen (diarizer). Its pyannote fork pins torch 2.1.1, so this needs a port, a separate environment or a subprocess. Streaming Sortformer v2.1 (diarizer, max 4 speakers, handles long audio without our chunk linking).
- [ ] 6c: Qwen3-ASR 1.7B with Qwen3-ForcedAligner for timestamps, MOSS-Transcribe-Diarize and Granite Speech 4.1 2B-plus (both transcribe and diarize). MOSS handles 90 minutes and Granite 9 minutes per pass, so both use the long-audio splitter.
- [ ] 6d: Cohere Transcribe. It needs the language up front and has no timestamps, so language comes from a first pass and timestamps from Qwen3-ForcedAligner.
- [ ] Pick new defaults from the benchmark numbers

## Stage 7: podcast extras

- [ ] LLM settings for either the OpenAI API or a local OpenAI compatible server (Ollama, llama.cpp). Document a few model picks that fit on the 1080 Ti.
- [ ] Speaker names from the transcript when there are no gold clips: send the first minutes and some lines per speaker to the LLM, get name guesses with the lines that support them
- [ ] Optional speaker library: keep embeddings of named speakers and match them automatically in later episodes
- [ ] NER on transcripts with GLiNER2: entities with speaker and timestamp, summed up per episode
- [ ] Audio events: AudioSet tagger (AST or BEATs) for music, laughter and applause, CLAP for custom labels like "jingle". Mark them in the transcript, optionally skip music before transcription and diarization.

## Stage 8: books

- [ ] GLiNER is asked about one label at a time, so it tends to find something for every label. In a test run `Bob` and `Paris` also came back as `ORGANIZATION`. Pass all labels in one call.
- [ ] spaCy sentences keep their trailing newline (`"Alice met Bob.\n"`). Strip them before storing.
- [ ] German and English spaCy model defaults
- [ ] Character list per book: merge name variants, count mentions per chapter
- [ ] Chapter summaries with the same LLM settings as podcasts
- [ ] Try coreference resolution for characters in German and English. Keep it only if the results are usable.

## Stage 9: speed

Drop whatever stage 4 already solved.

- [ ] Audio is decoded 4+ times per file (`accept`, diarization, speaker matching, media info). Decode once and share.
- [ ] `PodcastPipeline.accept` decodes the whole file just to check the type. Use a probe instead.
- [ ] Word/speaker alignment is O(words x segments). Use a sweep over sorted segments.
- [ ] Models are loaded again for every file. Keep them between files when memory allows.
- [ ] `import MAT` pulls in torch and all tools, so even `MAT --help` is slow. Import heavy libraries lazily.

## Stage 10: robustness and cleanup

- [ ] One failing step (for example the summary without an API key) drops all results of the file. Make steps fail on their own and keep the rest.
- [ ] Steps that skip because of missing input do it silently. Log a warning.
- [ ] `SpeakerIdetificationSpeechBrain.process` is a stub that returns `None`. Finish it or remove it.
- [ ] `DiarizerNEMO._create_config` (old MSDD setup, downloads yaml from GitHub) is unused. Remove it.
- [ ] CI: GitHub Actions with the `cpu` extra and `pytest`
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
