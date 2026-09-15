# mat-format

Data model, JSON schemas and a reader for the results [MAT](https://github.com/GameOfPods/MAT) writes. It only needs pydantic, so you can read MAT results without installing MAT and its ML libraries.

```python
from mat_format import MATResult

result = MATResult.read("results/episode_2026-09-15_20-15-02.zip")  # folder or zip
if result.podcast:
    for speaker in result.podcast.speakers:
        print(speaker.id, sum(s.end - s.start for s in speaker.segments))
    print(result.transcript())
```

The format itself is described in [docs/result-format.md](../../docs/result-format.md). The JSON schemas are in `src/mat_format/schemas/` and are generated from the models in `src/mat_format/models.py`:

```bash
uv run python -m mat_format.schema          # rewrite the schema files
uv run python -m mat_format.schema --check  # fail if they are outdated
```

The major version of this package is the result format version.

Every GitHub release of MAT has the schemas attached under fixed names, so other projects can always download the newest ones, for example `https://github.com/GameOfPods/MAT/releases/latest/download/podcast-result.schema.json`.

## License

This package, the schemas and the examples are licensed under the Apache License 2.0, see [LICENSE](LICENSE). MAT itself is GPL-3.0.
