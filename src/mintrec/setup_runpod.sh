#!/usr/bin/env bash
# Persistent RunPod environment for the multimodal-distillation project.
#
# Creates a venv on the PERSISTENT /workspace volume that INHERITS the base
# image's torch/CUDA (so we never re-download multi-GB torch wheels), then
# installs only the extra packages on top. Everything lives under /workspace,
# so it survives pod stop/restart/recreate.
#
# Run ONCE after attaching a fresh pod (idempotent — safe to re-run):
#     bash setup_runpod.sh
# Then EVERY session just:
#     source /workspace/venv/bin/activate
set -euo pipefail

VENV=/workspace/venv
REQ="$(dirname "$(realpath "$0")")/requirements-runpod.txt"

# keep pip cache on the persistent volume too (faster re-installs)
export PIP_CACHE_DIR=/workspace/.pip-cache

# 1) system codecs: librosa(mp4 audio) + decord/av video decoding need ffmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo ">> installing ffmpeg ..."
  apt-get update -qq && apt-get install -y -qq ffmpeg
fi

# 2) venv on the persistent volume, inheriting system torch/CUDA
if [ ! -d "$VENV" ]; then
  echo ">> creating venv at $VENV (inheriting system torch) ..."
  python -m venv "$VENV" --system-site-packages
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# 3) sanity: confirm inherited torch sees CUDA BEFORE installing on top
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| available", torch.cuda.is_available())
PY

# 4) extras only (no torch — inherited). bitsandbytes matches inherited CUDA.
echo ">> installing project deps ..."
pip install --upgrade pip
pip install -r "$REQ"

# 5) register a Jupyter kernel so the notebook uses THIS venv
python -m ipykernel install --user --name mmkd --display-name "mmkd (venv)"

echo
echo ">> done. Next session just run:  source $VENV/bin/activate"
