#!/usr/bin/env bash
# Builds the DiariZen environment next to this script.
#
# DiariZen pins torch 2.1.1 and ships its own pyannote.audio fork, so it can't live in MAT's environment. MAT runs
# it as a separate process, see docs/external-environments.md.
#
#   bash envs/diarizen/install.sh              # CUDA 12.1 build (works on GTX 10xx)
#   TORCH_INDEX=https://download.pytorch.org/whl/cpu bash envs/diarizen/install.sh
#
# About 6 GB on disk. The model weights are non-commercial (CC BY-NC 4.0).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${DIARIZEN_REPO:-$HERE/DiariZen}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}

command -v uv > /dev/null || { echo "uv is needed, see https://docs.astral.sh/uv/"; exit 2; }
command -v git > /dev/null || { echo "git is needed"; exit 2; }

if [ ! -d "$REPO" ]; then
  echo "== Cloning DiariZen into $REPO"
  git clone --depth 1 https://github.com/BUTSpeechFIT/DiariZen.git "$REPO"
fi

echo "== Creating the environment ($HERE/.venv, Python $PYTHON_VERSION)"
uv venv --python "$PYTHON_VERSION" "$HERE/.venv"
PYTHON=$HERE/.venv/bin/python

# torch first, everything else is pinned against it by constraints.txt
echo "== Installing torch 2.1.1 from $TORCH_INDEX"
uv pip install --python "$PYTHON" torch==2.1.1 torchaudio==2.1.1 torchvision==0.16.1 --index-url "$TORCH_INDEX"

echo "== Installing DiariZen and its pyannote fork"
uv pip install --python "$PYTHON" -r "$REPO/requirements.txt" -c "$REPO/constraints.txt"
uv pip install --python "$PYTHON" -e "$REPO" -c "$REPO/constraints.txt"
uv pip install --python "$PYTHON" -e "$REPO/pyannote-audio" -c "$REPO/constraints.txt"

echo "== Checking"
"$PYTHON" -c "import torch; from diarizen.pipelines.inference import DiariZenPipeline; print('diarizen ok, torch', torch.__version__)"
echo "Done. MAT finds the environment on its own, check with: MAT external list"
