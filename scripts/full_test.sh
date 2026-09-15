#!/usr/bin/env bash
# Full test of a MAT checkout: install, unit tests, schema check, both smoke scripts and one complete run on a real
# audio file, which gets read back and validated against the result schemas. Works with a CUDA GPU and on CPU only
# machines (on CPU the complete run takes a long time for long audio).
#
#   bash scripts/full_test.sh 2>&1 | tee full_test.log
#
# Settings come from these environment variables. Anything that isn't set gets asked for at the start, tokens with
# hidden input. Set a variable to an empty string to skip the question and use the "empty" behavior.
#   MAT_TEST_DEVICE     cuda or cpu, cpu installs the CPU torch build
#   MAT_TEST_OUT        folder for logs and results, has to be empty or not exist yet
#   MAT_TEST_AUDIO      audio file for the complete run, empty skips it. Longer than 5 minutes also tests
#                       diarization in pieces and speaker linking.
#   HF_TOKEN            Hugging Face token with access to pyannote/embedding, empty uses a saved `hf auth login`
#   OPENAI_API_KEY      API key for the summary, empty runs without a summary
#   OPENAI_API_BASE     OpenAI compatible endpoint, empty uses the OpenAI API
#   MAT_TEST_LLM_MODEL  model for the summary, empty uses MAT's default
#
# Output of the tools goes to log files in $MAT_TEST_OUT/logs, the console only shows the steps and their results.
# Tokens are never printed or written to a file.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

die() {
  echo "full_test: $*" >&2
  exit 2
}

# ask NAME QUESTION [secret]   keeps NAME if it's set (also when empty), otherwise reads it from the terminal
ask() {
  local name=$1 question=$2 secret=${3:-} value=""
  [ -n "${!name+x}" ] && return
  { : </dev/tty; } 2>/dev/null || die "$name isn't set and there's no terminal to ask for it"
  printf '%s: ' "$question" >/dev/tty
  if [ "$secret" = secret ]; then
    read -rs value </dev/tty
    echo >/dev/tty
  else
    read -r value </dev/tty
  fi
  printf -v "$name" '%s' "$value"
}

# ---------------------------------------------------------------- settings
ask MAT_TEST_DEVICE "Device, cuda or cpu"
case "$MAT_TEST_DEVICE" in
  cuda | cpu) ;;
  *) die "MAT_TEST_DEVICE has to be cuda or cpu, got '$MAT_TEST_DEVICE'" ;;
esac

ask MAT_TEST_OUT "Output folder for logs and results (empty or new)"
[ -n "$MAT_TEST_OUT" ] || die "MAT_TEST_OUT is needed"
[ -e "$MAT_TEST_OUT" ] && [ ! -d "$MAT_TEST_OUT" ] && die "$MAT_TEST_OUT isn't a folder"
[ -d "$MAT_TEST_OUT" ] && [ -n "$(ls -A "$MAT_TEST_OUT")" ] && die "$MAT_TEST_OUT isn't empty"

ask MAT_TEST_AUDIO "Audio file for the complete run (Enter skips it)"
[ -z "$MAT_TEST_AUDIO" ] || [ -f "$MAT_TEST_AUDIO" ] || die "audio file not found: $MAT_TEST_AUDIO"

ask HF_TOKEN "Hugging Face token (Enter uses a saved login)" secret
ask OPENAI_API_KEY "LLM API key (Enter runs without a summary)" secret
if [ -n "$OPENAI_API_KEY" ]; then
  ask OPENAI_API_BASE "LLM endpoint, OpenAI compatible (Enter uses the OpenAI API)"
  ask MAT_TEST_LLM_MODEL "LLM model (Enter uses MAT's default)"
else
  OPENAI_API_BASE=${OPENAI_API_BASE:-}
  MAT_TEST_LLM_MODEL=${MAT_TEST_LLM_MODEL:-}
fi

mkdir -p "$MAT_TEST_OUT/logs" || die "can't create $MAT_TEST_OUT"
OUT=$(cd "$MAT_TEST_OUT" && pwd)
LOGS=$OUT/logs

if [ "$MAT_TEST_DEVICE" = cpu ]; then
  SYNC=(uv sync --locked --no-default-groups --group dev --group cpu --group backends)
  # backends pick cuda when they see a GPU, hide it
  export CUDA_VISIBLE_DEVICES=""
else
  SYNC=(uv sync --locked)
fi

# ---------------------------------------------------------------- helpers
STEPS=0
FAILED=()

detail() { sed 's/^/      /'; }

# run NAME LOG COMMAND...   output goes to $LOGS/LOG.log, prints time, peak GPU memory and the log tail on failure
run() {
  local name=$1 log=$LOGS/$2.log sampler="" rc start seconds peak=""
  shift 2
  STEPS=$((STEPS + 1))
  printf '%s  %-24s ' "$(date +%H:%M)" "$name"
  if [ "$MAT_TEST_DEVICE" = cuda ] && command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > "$LOGS/$2.gpu" 2>/dev/null &
    sampler=$!
  fi
  start=$(date +%s.%N)
  "$@" > "$log" 2>&1
  rc=$?
  seconds=$(awk "BEGIN {printf \"%.1f\", $(date +%s.%N) - $start}")
  if [ -n "$sampler" ]; then
    kill "$sampler" 2>/dev/null
    wait "$sampler" 2>/dev/null
    peak=", peak GPU memory $(sort -n "$LOGS/$2.gpu" | tail -1) MiB"
  fi
  if [ "$rc" -eq 0 ]; then
    echo "ok, ${seconds} s${peak}"
  else
    FAILED+=("$name")
    echo "FAILED with exit code $rc after ${seconds} s${peak}, log: $log"
    tail -15 "$log" | detail
  fi
  return "$rc"
}

# ---------------------------------------------------------------- environment
echo "MAT full test"
echo "  commit:  $(git log -1 --format='%h %s') ($(git branch --show-current))"
echo "  device:  $MAT_TEST_DEVICE"
if [ "$MAT_TEST_DEVICE" = cuda ]; then
  echo "  gpu:     $(nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader 2>&1 | head -1)"
fi
echo "  ffmpeg:  $(ffmpeg -version 2>&1 | head -1)"
echo "  uv:      $(uv --version)"
echo "  memory:  $(free -g | awk '/^Mem:/ {print $2 " GB RAM, " $7 " GB available"}')"
echo "  audio:   $([ -n "$MAT_TEST_AUDIO" ] && basename "$MAT_TEST_AUDIO" || echo "none, skipping the complete run")"
echo "  HF token given: $([ -n "$HF_TOKEN" ] && echo yes || echo no), saved login: $([ -f "$HOME/.cache/huggingface/token" ] && echo yes || echo no)"
echo "  summary: $([ -n "$OPENAI_API_KEY" ] && echo "yes, model ${MAT_TEST_LLM_MODEL:-default}" || echo no)"
echo "  logs:    $LOGS"
echo

# ---------------------------------------------------------------- steps
if ! run "install" install "${SYNC[@]}"; then
  echo "Install failed, stopping"
  exit 1
fi

run "torch and device" torch uv run --no-sync python -W ignore -c "
import sys, torch
print(f'torch {torch.__version__}, CUDA build {torch.version.cuda}, CUDA available {torch.cuda.is_available()}')
if sys.argv[1] == 'cuda':
    if not torch.cuda.is_available():
        sys.exit('CUDA is not available')
    import ctranslate2
    print(f'{torch.cuda.get_device_name(0)}, compute capability {torch.cuda.get_device_capability(0)}')
    print(f'CTranslate2 compute types: {sorted(ctranslate2.get_supported_compute_types(\"cuda\"))}')
" "$MAT_TEST_DEVICE" && detail < "$LOGS/torch.log"

run "unit tests" pytest uv run --no-sync pytest tests packages/mat-format/tests -q -p no:cacheprovider
tail -1 "$LOGS/pytest.log" | detail

run "schema check" schema uv run --no-sync python -m mat_format.schema --check

run "backends" backends uv run --no-sync MAT backends && {
  echo "installed:     $(awk '$2 == "installed" {printf "%s ", $1}' "$LOGS/backends.log")"
  echo "not installed: $(awk '$2 == "not" {printf "%s ", $1}' "$LOGS/backends.log")"
} | detail

run "smoke podcast" smoke_podcast uv run --no-sync python scripts/smoke_podcast.py --device "$MAT_TEST_DEVICE" \
  --out "$OUT/smoke_podcast" && grep -E "^(language|speakers|peak torch)" "$LOGS/smoke_podcast.log" | detail

run "smoke book" smoke_book uv run --no-sync python scripts/smoke_book.py --out "$OUT/smoke_book" \
  && grep -E "^took" "$LOGS/smoke_book.log" | detail

if [ -n "$MAT_TEST_AUDIO" ]; then
  ARGS=(run --yes --export-config -o "$OUT/run" -i "$MAT_TEST_AUDIO")
  if [ -z "$OPENAI_API_KEY" ]; then
    ARGS+=(--summarizer none)
  elif [ -n "$MAT_TEST_LLM_MODEL" ]; then
    ARGS+=(--set "llm.model=$MAT_TEST_LLM_MODEL")
  fi
  # the tokens only go into the environment of this one command
  run "complete run" run env \
    ${HF_TOKEN:+HF_TOKEN="$HF_TOKEN"} \
    ${OPENAI_API_KEY:+OPENAI_API_KEY="$OPENAI_API_KEY"} \
    ${OPENAI_API_BASE:+OPENAI_API_BASE="$OPENAI_API_BASE"} \
    uv run --no-sync MAT "${ARGS[@]}"
  grep -E "Detected language|Diarizing|also diarizes|Found gold labels|Step [0-9]+/[0-9]+ done|Answer complete|LLM call failed|Summary failed|isn't installed" \
    "$LOGS/run.log" | sed -E 's/^[0-9: ,-]+ - +[A-Z]+ +- [^:]*: //' | detail

  run "check result" check uv run --no-sync python -W ignore - "$OUT/run" <<'PY'
import json
import sys
from pathlib import Path

import jsonschema
from mat_format import MATResult
from mat_format import schema as format_schema

folders = sorted(p for p in Path(sys.argv[1]).glob("*") if p.is_dir())
if not folders:
    sys.exit(f"no result folder in {sys.argv[1]}")
folder = folders[-1]


def validate(schema_file, data_file):
    schema = json.loads((format_schema.SCHEMA_DIR / schema_file).read_text())
    jsonschema.Draft202012Validator(schema).validate(json.loads(data_file.read_text()))


validate("meta.schema.json", folder / "meta.json")
for pipeline in json.loads((folder / "meta.json").read_text())["pipelines"]:
    validate(f"{pipeline}-result.schema.json", folder / pipeline / "result.json")
print("schemas valid, files:", " ".join(sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())))

result = MATResult.read(folder)
podcast = result.podcast
print("models:", ", ".join(f"{slot}={info.backend}/{info.model}" for slot, info in podcast.models.items()))
print(f"language {podcast.language}, duration {podcast.media.duration:.0f} s, speech {podcast.media.speech_duration:.0f} s")
print("diarizer labels:", ", ".join(s.id for s in podcast.diarization))
for speaker in podcast.speakers:
    print(f"speaker {speaker.id}: {sum(s.end - s.start for s in speaker.segments):.0f} s in {len(speaker.segments)} segments")
words = podcast.words
print(f"words {len(words)}, without times {sum(w.start is None or w.end is None for w in words)}, "
      f"more than one speaker {sum(len(w.speakers) > 1 for w in words)}, no speaker {sum(not w.speakers for w in words)}, "
      f"segments {len(podcast.segments)}")
print("transcript start:")
for line in result.transcript().splitlines()[:8]:
    print("  " + line[:160])
print("summary start:")
for line in (podcast.summary or "(no summary)")[:700].splitlines():
    print("  " + line)
PY
  detail < "$LOGS/check.log"
fi

# ---------------------------------------------------------------- result
echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All $STEPS steps ok"
else
  echo "${#FAILED[@]} of $STEPS steps failed: ${FAILED[*]}"
fi
echo "Logs and results: $OUT"
[ ${#FAILED[@]} -eq 0 ]
