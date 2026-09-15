# Benchmarks

`MAT bench` runs podcast *systems* on datasets and compares them. A system is MAT's podcast pipeline with a choice of backends and settings, for example whisper `large-v3-turbo` with Sortformer. For every system and file it measures:

- speed (RTFx) and the time of every pipeline step
- peak GPU memory
- WER, cpWER, DER and the number of speakers, where the dataset has a reference

Summaries never run in a benchmark. They'd cost API tokens and no dataset has reference summaries to compare with.

## Quick start

The `bench` extra (jiwer and pyannote.metrics) is part of `all`, so a normal `uv sync` has it.

```bash
uv run MAT bench datasets                                    # dataset types, licenses and options
uv run MAT bench download -c benchmarks/example.toml         # optional, download everything first
uv run MAT bench run -c benchmarks/example.toml -o bench-results
```

Useful options of `run`:

- `--limit N` replaces the `limit` of every dataset in the bench file with N samples (sentences, files, meetings or snippets, per language for FLEURS). `--limit 1` is good for a first try, `--limit 0` or `-1` uses all of every dataset. Without `--limit` the limits of the bench file apply.
- `--dataset NAME` and `--system NAME` pick single datasets or systems (both can be repeated)
- `--cache FOLDER` sets the dataset cache
- `--rerun` runs again where a result already exists

Every finished run is stored, and runs that finished before are skipped. So an aborted benchmark continues where it stopped, and a new system added to the bench file only runs the new system. `MAT bench report -c ... -o ...` scores the stored results again without running anything, for example after correcting a reference.

## Bench file

A TOML file with three parts. `benchmarks/example.toml` has all of them. Relative paths are relative to the bench file.

```toml
[bench]
cache = "/mnt/nas/mat-datasets"   # default: $MAT_BENCH_CACHE or ~/.cache/mat/bench
collar = 0.25                     # optional, DER collar for all datasets, see Metrics

[[system]]
name = "turbo"
set = ["whisper.model=large-v3-turbo"]    # same as `MAT run --set`

[[system]]
name = "tuned"
config = "tuned.toml"                     # a MAT config file, `set` still wins over it

[[dataset]]
type = "fleurs"
languages = ["de", "en"]
limit = 100
```

- `[[system]]`: `name` (letters, digits, `.`, `_`, `-`), optional `config` and `set`. The identifier is off unless a system sets `podcast.identifier`, because datasets have no gold speaker clips. Without any `[[system]]` a single system `default` with MAT's defaults runs.
- `[[dataset]]`: `type` plus the options of that type. Every dataset has `name` (default: the type, needed when one type is used twice), `limit`, `path` (see below) and `collar`.

## Datasets

| type | what | metrics | license | download |
|---|---|---|---|---|
| `reference` | your own episodes with corrected transcripts | WER, cpWER, DER, speakers | yours | - |
| `audio` | any audio files, no reference | speed, memory, speakers, agreement | yours | - |
| `fleurs` | read Wikipedia sentences, many languages | WER | CC-BY-4.0 | whole audio archive of a split: de test 570 MB, en test 290 MB |
| `voxconverse` | YouTube debates, news and shows, 1 to 20+ speakers | DER, speakers | CC-BY-4.0, audio copyright stays with the video owners | only the picked files, read out of the zip |
| `ami` | meetings with 3 to 5 speakers | WER, cpWER, DER, speakers | CC-BY-4.0 | 30 to 90 MB per meeting plus 23 MB annotations |
| `bundestag` | German parliament speech (ASR Bundestag) | WER | Bundestag terms of use, no commercial use or advertising | only the picked snippets, read out of the 59 GB zip |

`MAT bench datasets` prints the options of every type. The notes below say what to keep in mind when reading the numbers.

`limit` counts samples in the unit of the dataset. Without `limit` in the bench file the type's default applies, `0` or `-1` uses everything, and `MAT bench run --limit N` replaces the limits of all datasets.

| type | default `limit` | counts | all of it |
|---|---|---|---|
| `reference`, `audio` | everything | reference folders, audio files | - |
| `fleurs` | 100 | sentences per language | about 350 per language in the test split, 1 to 1.5 hours |
| `voxconverse` | 5 | files | 232 files, about 43 hours (test) |
| `ami` | 2 | meetings | 16 meetings, about 9 hours (test) |
| `bundestag` | 200 | snippets | the test split of the clean subset, many hours |

**fleurs** (`languages`, `split`, `limit` = sentences per language, `pack-minutes`): every sentence exists from several speakers, MAT takes one recording per sentence. The sentences get joined into files of up to `pack-minutes` with a second of silence in between, so the models load once per file and not once per sentence. Clean read speech, so it only says how well words are recognized.

**voxconverse** (`split`, `limit`, `files`): mostly English. Lots of short turns and overlapping speech, much harder for diarization than a podcast with a few hosts. Annotations v0.3 from the GitHub repository.

**ami** (`split`, `limit`, `meetings`, `turn-gap`): mixed headset recordings (all microphones summed). The word reference comes from the manual AMI annotations, the DER reference, meeting lists and scored time ranges from [BUT's AMI diarization setup](https://github.com/BUTSpeechFIT/AMI-diarization-setup) (`only_words`). WER is scored over the whole meeting, so it's higher than numbers in papers that score single utterances.

**bundestag** (`subset`, `split`, `limit` = snippets, `pack-minutes`): the references are lowercase without punctuation and write numbers as words (`vierhundertsechzig`). Whisper writes digits, so WER comes out higher than the real error rate. Snippets are sorted by session and time before packing, so neighbouring parts of a speech end up together.

### Using an existing copy

Set `path` to a folder (or file) that already has the dataset, then nothing gets downloaded from it. Missing files still land in the cache.

- `fleurs`: the Hugging Face layout, `path/data/<code>/<split>.tsv` (or `path/<code>/<split>.tsv`) with the audio extracted in `audio/<split>/` or as `audio/<split>.tar.gz`
- `voxconverse`: `<id>.rttm` files and `<id>.wav` files anywhere below `path`
- `ami`: `words/<meeting>.<letter>.words.xml` or `ami_public_manual_1.6.2.zip`, `<meeting>.Mix-Headset.wav` anywhere below `path`, optionally the BUT setup folders `lists/`, `only_words/rttms/` and `uems/`
- `bundestag`: the extracted `asr_bundestag_clean` folder (or its parent), or the zip file itself

## Own references

A reference is a folder with the audio path and a corrected transcript in MAT's `transcript.txt` format. The easiest way is to let MAT write a draft and fix it:

1. Run MAT on the episode: `uv run MAT run -i episode.mp3 -o results --summarizer none --yes`
2. Make a reference from the result, only for the part you want to correct (10 minutes is plenty):

   ```bash
   uv run MAT bench reference results/episode_2026-09-16_10-00-00 -o references/episode-01 --start 600 --end 1200
   ```

   This writes `references/episode-01/reference.toml` and `transcript.txt` with the lines of that part. The audio path is taken from the result, pass `--audio` if the file moved.
3. Correct `transcript.txt` while listening to that part.
4. Add the references to the bench file:

   ```toml
   [[dataset]]
   type = "reference"
   name = "own"
   path = "references"    # one folder with a reference, or a folder of reference folders
   ```

How to correct the transcript:

- Keep the layout `speaker [start - end]: text`, one line per turn.
- Fix the words: remove what wasn't said, add what's missing, fix names. Write numbers the way you want them compared (whisper writes digits).
- Fix the speaker of every line. Use any labels you like (`alex`, `host2`), but the same one for the same person in the whole file. They don't have to match MAT's labels, the metrics find the best mapping.
- Split a line where the speaker changes and guess the time of the split. Times only need to be about right. The start and end of a line tell DER when that speaker talks, so leave gaps where nobody talks.
- Overlapping speech gets one line per speaker with overlapping times. Lines with `a & b` count for both speakers in DER, but their words only for `a` in cpWER, so better split them.
- Keep `<Unknown>` only for voices you really can't tell. Lines starting with `#` are ignored.

`reference.toml`:

```toml
audio = "/media/podcasts/episode.mp3"   # absolute or relative to reference.toml
language = "de"                         # optional, default: detected
start = 600.3                           # optional, only this part is scored
end = 1199.8
transcript = "transcript.txt"           # optional, can also point to a MAT result folder or zip
```

Only words and speaker time between `start` and `end` are scored, so the systems still run on the whole episode (with long audio handling and speaker linking) but get compared on the corrected part. A reference made from one system's draft is a bit biased towards that system in unclear cases (spelling of names, filler words), so correct carefully.

## Metrics

- **RTFx**: seconds of audio per second of processing, model loading included. `results.csv` also has the seconds of every pipeline step.
- **Peak GPU**: the highest GPU memory of the MAT process according to `nvidia-smi`, including the CUDA context. `torch_peak_mib` in `results.csv` is what torch allocated itself, it misses CTranslate2 (whisper).
- **WER**: word errors (substitutions, deletions, insertions) over the number of reference words, for the whole file. Before comparing, text is lowercased, punctuation is removed, hyphens split words and fillers like uh, um, äh, hm are dropped. Numbers aren't normalized, `5` and `five` are different words.
- **cpWER**: every reference speaker is compared with the words of one system speaker, picked so the total errors are lowest. Words given to the wrong speaker count as errors, extra speakers count as insertions. A word with several speakers counts for the first one.
- **DER**: missed speech, false alarm and speaker confusion over the total reference speech, overlapping speech included (pyannote.metrics). The collar ignores that many seconds on both sides of every reference boundary. It's 0.25 s for own references and 0 for the public datasets, `collar` in `[bench]` changes it for all datasets and `collar` in a `[[dataset]]` for one. Own references need it because a line's times only cover its words, while diarizers also mark the short pauses around them: the 30 second sample scored against its own MAT result has 20 % DER without a collar, 6 % with 0.25 s and 2 % with 0.5 s. The report shows the collar of every dataset.
- **speakers off**: average difference between the number of reference speakers and found speakers.
- **Agreement**: with more than one system, every system is also scored against the first one as if its result were the reference. That's the only quality signal for `audio` datasets without references. Low numbers mean the systems produce similar results, not that both are right.

Rates in the report are totals over all files of a dataset (total errors over total reference), so long files weigh more than short ones.

## Output

```
bench-results/
  report.md              tables per dataset, agreement, how to read them
  results.csv            one row per system and file with all numbers
  results/<system>/<dataset>/<file>/
    result/              the normal MAT result (meta.json, podcast/result.json, transcript.txt, ...)
    bench.json           time, step times, GPU memory, error
```

The output folder has full transcripts of everything that ran, including your own episodes. Keep it out of git. Numbers for public datasets from the GPU box go into `docs/benchmarks/`.

## Cache

Downloads go to `--cache`, else `cache` in the bench file, else `$MAT_BENCH_CACHE`, else `~/.cache/mat/bench`. A cache on a network share can be used by several machines. Downloads are written as `.part` files and renamed when complete, so an aborted download is continued next time, but don't download the same dataset from two machines at the same time.

The cache has one folder per dataset type. Packed FLEURS and Bundestag files live in `<type>/packed/` and get reused as long as the same sentences are picked.
