# Backends in their own environment

Most backends are uv extras and run inside MAT's environment. Some libraries pin versions we can't follow, for example DiariZen with torch 2.1.1 and its own pyannote.audio fork. Those get their own environment under `envs/<name>` and MAT runs them as a separate process.

```
envs/diarizen/
  install.sh      builds envs/diarizen/.venv
  run.py          runs inside that environment: JSON in, JSON out
  .venv/          not in git
  DiariZen/       cloned by install.sh, not in git
```

MAT decodes the audio, writes a wav, and starts `envs/<name>/.venv/bin/python envs/<name>/run.py request.json result.json`. That's the whole protocol: two files, one process, no server. Models are loaded in that process, so only one of the two environments holds GPU memory at a time.

The process inherits MAT's environment variables, so `HF_HOME`, `HF_HUB_CACHE`, `HF_TOKEN`, `CUDA_VISIBLE_DEVICES` and proxy settings apply to it as well and both environments share one model cache. The install script inherits them too. A test keeps it that way, so don't hand `subprocess.run` its own environment.

## DiariZen

DiariZen ([BUT Brno](https://github.com/BUTSpeechFIT/DiariZen)) is WavLM plus Conformer on a pyannote style pipeline and has the best open diarization numbers (AMI-SDM 14.0 %, VoxConverse 9.2 % in their README). **The weights are CC BY-NC 4.0, so no commercial use.**

```bash
uv run MAT external list                 # what exists and what is built
uv run MAT external install diarizen     # clone, build the environment, check the import
uv run MAT run -i episode.mp3 -o out --diarizer diarizen
```

The environment takes about 2 GB with the CPU torch build and more with the CUDA one, plus the cloned repository and the model weights in the Hugging Face cache. A run pays for the process start and loading the model: the 30 second sample took 62 seconds on CPU, so this is worth it for episodes, not for clips.

`install.sh` uses the CUDA 12.1 torch build, which works on GTX 10xx cards. For a machine without a GPU:

```bash
TORCH_INDEX=https://download.pytorch.org/whl/cpu bash envs/diarizen/install.sh
```

Options: `diarizen.model` (`BUT-FIT/diarizen-wavlm-large-s80-md`, `...-v2`, or the `base` model), `diarizen.device`, `diarizen.timeout`. `MAT backends show diarizen` prints them.

Without the environment the backend is simply skipped, like an extra that isn't installed. `MAT backends` then shows it as not installed with the install hint.

## Adding another environment

1. `envs/<name>/install.sh` builds `envs/<name>/.venv`. Keep the versions the library asks for.
2. `envs/<name>/run.py` reads a JSON request and writes a JSON answer. Keep it small, it can't import MAT.
3. Add an `ExternalEnvironment` entry to `ENVIRONMENTS` in `MAT/utils/external.py`.
4. Write the backend as usual, but call `require_environment("<name>")` instead of `require(...)` at the top of the module, and `run_external("<name>", request)` in `process`.
5. Add `envs/<name>/.venv` and anything the install script downloads to `.gitignore`.

Keep the request and answer to plain JSON plus file paths for audio. Numpy arrays, models or Python objects don't belong in there.

## What this costs

Every environment is a second dependency set to maintain, a few GB on disk and 10 to 20 seconds of process and model loading per run. That's worth it for a model that clearly beats what we have, and not worth it for convenience. Backends that work inside MAT's environment stay there.
