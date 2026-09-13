# Roadmap

Known problems and what we want to do about them. Stage 1 is what the `fix/restore-v0.2` branch does. Later stages are not started yet.

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

## Stage 2: behavior and usability

These change how MAT behaves, so each one needs a decision first.

- [ ] Add `--yes` so MAT can run without the interactive confirmation (cron, containers)
- [ ] Config precedence: right now the JSON config file overrides CLI flags. Explicit CLI flags should win.
- [ ] `--LLM-Summarizer_chunk-size` below 200 crashes because the splitter overlap is fixed at 200. Make the overlap relative or configurable.
- [ ] Whisper: instead of segment timings, detect the language first and ask faster-whisper for word timestamps when there is no alignment model
- [ ] GPU install: torch is pinned `<2.6`, but the `pytorch-cu126` index only has torch 2.6+. Pick a matching CUDA index (cu124 for 2.5) and document how to switch.
- [ ] Find out why torch is pinned below 2.6 and write it down (or lift the pin)
- [ ] Default summary model is `gpt-4`. Pick a current default.
- [ ] GLiNER is asked about one label at a time, so it tends to find something for every label. In a test run `Bob` and `Paris` also came back as `ORGANIZATION`. Passing all labels in one call should let the model choose.
- [ ] spaCy sentences keep their trailing newline (`"Alice met Bob.\n"`). Strip them before storing.
- [ ] `Config` calls `sys.exit(1)` on duplicate config names. Raise instead.

## Stage 3: speed

- [ ] Audio is decoded 4+ times per file (`accept`, diarization, speaker matching, media info). Decode once and share.
- [ ] `PodcastPipeline.accept` decodes the whole file just to check the type. Use a probe instead.
- [ ] Word/speaker alignment is O(words x segments). Use a sweep over sorted segments.
- [ ] Models are loaded again for every file (Whisper, Sortformer, GLiNER, spaCy). Keep them between files.
- [ ] NeMo chunk linking loads the pyannote model once per chunk
- [ ] `import MAT` pulls in torch and all tools, so even `MAT --help` is slow. Import heavy libraries lazily.

## Stage 4: robustness and cleanup

- [ ] Store a stable result type name (`PodcastOutput`) instead of `str(type(...))`. Needs result format version 2, keep the v1 reader.
- [ ] One failing step (for example the summary without an API key) drops all results of the file. Make steps fail on their own and keep the rest.
- [ ] Steps that skip because of missing input do it silently. Log a warning.
- [ ] `SpeakerIdetificationSpeechBrain.process` is a stub that returns `None`. Finish it or remove it.
- [ ] `DiarizerNEMO._create_config` (old MSDD setup, downloads yaml from GitHub) is unused. Remove it.
- [ ] CI: GitHub Actions with `uv sync` and `pytest` on CPU
- [ ] Clean up stale `mat.egg-info`/`MAT.egg-info` folders and decide if `.idea/` belongs in the repo
