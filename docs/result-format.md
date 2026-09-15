# MAT result format

This describes what MAT writes, so other programs can read it without MAT. The current version is **format 2**.

Machine readable definitions (JSON Schema, draft 2020-12):

- [`meta.schema.json`](../packages/mat-format/src/mat_format/schemas/meta.schema.json) for `meta.json`
- [`podcast-result.schema.json`](../packages/mat-format/src/mat_format/schemas/podcast-result.schema.json) for `podcast/result.json`
- [`book-result.schema.json`](../packages/mat-format/src/mat_format/schemas/book-result.schema.json) for `book/result.json`

The schemas are generated from the pydantic models in [`packages/mat-format/src/mat_format/models.py`](../packages/mat-format/src/mat_format/models.py), which MAT's writer uses. A test fails if the schema files and the models don't match. Real example results are in [`packages/mat-format/examples/`](../packages/mat-format/examples).

## Layout

One result per input file. It's either a folder or a zip of that folder (`--output-zip`). In the zip the files sit at the root, there is no extra top folder.

```
episode_2026-09-15_20-15-02/        or episode_2026-09-15_20-15-02.zip
├── meta.json                       always
├── config.toml                     only with --export-config
├── podcast/                        for audio files
│   ├── result.json                 all data, use this
│   ├── transcript.txt              convenience copy
│   ├── summary.md                  convenience copy, missing without summary
│   └── diarization.rttm            convenience copy
└── book/                           for EPUB files
    └── result.json
```

`result.json` is the source of truth. The other files are generated from the same data for people and tools that want them. All text files are UTF-8, all JSON is strict (no `NaN` or `Infinity`).

## Versioning

- `meta.json` and every `result.json` have a `format` field. It only changes when something breaks: a field is renamed or removed, a type changes, or the meaning of a field changes.
- New fields can show up within a format version. **Readers must ignore fields they don't know.** The schemas don't forbid extra properties for that reason.
- Check `format` in `meta.json` first and stop if it's a version your reader doesn't support.
- The major version of the `mat-format` Python package is the format version.

## Conventions

- Times are seconds (float) from the start of the audio file.
- Optional values are written as `null`, not left out. Lists are empty, not `null`.
- Speaker ids are only unique inside one result. `sprecher_0` in one episode has nothing to do with `sprecher_0` in the next. Stable speaker ids across episodes are planned and will be added as new fields.

## meta.json

| Field | Type | Meaning |
|---|---|---|
| `format` | integer | Always `2` |
| `mat_version` | string | MAT version that wrote it |
| `created` | string | Local time, ISO 8601 without time zone, for example `2026-09-15T20:15:02` |
| `input.name` | string | Input file name |
| `input.path` | string | Absolute path of the input on the machine that ran MAT |
| `input.sha1` | string | SHA-1 of the input file, hex |
| `pipelines` | list of string | Pipelines that ran, each has a folder with that name: `podcast`, `book` |

## podcast/result.json

| Field | Type | Meaning |
|---|---|---|
| `schema` | string | Always `mat.podcast` |
| `format` | integer | Always `2` |
| `models` | object | Step name to model info, see below. Steps that were skipped (for example `summarizer` set to `none`) are missing. |
| `language` | string or null | Detected language, ISO 639-1 (`de`, `en`) |
| `media` | object or null | `duration` (s), `speech_duration` (s of speech after voice activity detection, or null), `sample_rate`, `max_dbfs` (null for silence), `rms` |
| `speakers` | list of speaker | Speakers after matching them to gold label clips. Use this list. |
| `diarization` | list of speaker | Speakers as the diarizer found them, before matching |
| `words` | list of word | Every word in order |
| `segments` | list of word | Consecutive words with the same speakers merged into lines |
| `summary` | string or null | Markdown. Null if the summary was skipped or failed. |
| `events` | list | Sound events (`label`, `start`, `end`, `score`). Always empty in MAT 0.2, planned. |
| `entities` | list | Named entities in the transcript (`label`, `text`, `start`, `end`, `speakers`). Always empty in MAT 0.2, planned. |

A **speaker** is `{"id": "alice", "segments": [{"start": 6.72, "end": 7.28}, ...]}`. The id is the gold label name when the speaker was matched, otherwise the diarizer label like `sprecher_0`. Segments are sorted by start.

A **word** is `{"start": 6.73, "end": 6.86, "text": "Hello?", "speakers": ["alice"]}`. `start` and `end` can be null if the aligner couldn't place the word. `speakers` is empty when no speaker talks at that time and has more than one entry when speakers overlap.

**Model info** is `{"backend": "whisper", "model": "large-v3-turbo", "packages": {"faster-whisper": "1.2.1"}}`. Backends can add more fields, for example the summarizer adds `service` and `api_base`.

Speaking time per speaker, for example, is the sum of `end - start` over their `segments`.

## book/result.json

| Field | Type | Meaning |
|---|---|---|
| `schema` | string | Always `mat.book` |
| `format` | integer | Always `2` |
| `models` | object | Step name (`splitter`, `ner`) to model info |
| `title` | string | Book title |
| `language` | string or null | Detected language, ISO 639-1 |
| `chapters` | list of chapter | In table of contents order |

A **chapter** has `heading` (repeated headings get a roman numeral, like `Part II`), `heading_raw` (as in the book), `paragraphs` (list of strings) and `sentences`.

A **sentence** has `text`, `lemmas` (lemma to count, without stop words and punctuation) and `entities`. An entity is `{"label": "PERSON", "text": "Alice", "start": 0, "end": 5}`, where `start` and `end` are character offsets in the sentence text (end exclusive).

`sentences` is empty when sentence splitting didn't run, `entities` is empty when named entity recognition was skipped.

## The convenience files

- `transcript.txt`: one line per segment, `SPEAKERS [START - END]: TEXT`. Several speakers are joined with ` & `, no speaker is `<Unknown>`. Example: `alice [6.721 - 7.062]: Hello?`
- `summary.md`: the `summary` field.
- `diarization.rttm`: the `speakers` list in [RTTM](https://github.com/nryant/dscore#rttm) format, for diarization tools.

## Reading it

**Python:** install the `mat-format` package from `packages/mat-format`. It only needs pydantic.

```python
from mat_format import MATResult

result = MATResult.read("episode_2026-09-15_20-15-02.zip")
if result.podcast:
    for speaker in result.podcast.speakers:
        print(speaker.id, sum(s.end - s.start for s in speaker.segments))
```

**Other languages:** open the folder or zip, read `meta.json`, check `format`, then parse the `result.json` of each pipeline in `pipelines`. To get classes instead of maps, generate them from the schemas, for Java for example with [jsonschema2pojo](https://www.jsonschema2pojo.org/). To check files against the schemas, use a JSON Schema validator for draft 2020-12, for Java for example [networknt json-schema-validator](https://github.com/networknt/json-schema-validator). Keep unknown properties allowed so newer MAT versions don't break your reader.

## Changelog

- **Format 2** (MAT 0.2, stage 4 of the roadmap): first documented format. Folder per pipeline with `result.json`, models and package versions, speakers as objects, `speech_duration`, typed `events` and `entities`.
- **Format 1** (MAT 0.2.0 and older): `0.PodcastOutput/` style folders with several JSON files, undocumented, not readable anymore.
