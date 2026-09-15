#  MAT - Toolkit to analyze media
#  Copyright (c) 2025.  RedRem95
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
"""
`MAT bench`: runs podcast systems (backends plus their settings) on datasets and compares speed, GPU memory, WER,
cpWER and DER. Summaries never run in a benchmark.

- `data.py`: reference items and the transcript.txt format used for your own references
- `datasets/`: loaders for your own references, plain audio, FLEURS, VoxConverse, AMI and ASR Bundestag
- `metrics.py`, `scoring.py`: text normalization and the metrics
- `runner.py`: bench file, run loop and resuming, `report.py`: results.csv and report.md
- `commands.py`: the `MAT bench` subcommands

See docs/benchmarks.md.
"""
