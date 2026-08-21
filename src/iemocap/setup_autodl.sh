#!/usr/bin/env bash
# AutoDL environment for the IEMOCAP teacher stage.
#
# Image to pick when creating the instance:
#     PyTorch 2.8.0  /  CUDA 12.x  /  Python 3.10-3.12     on RTX 4090 (24 GB)
# 2.8.0 is the newest offered and the closest to the 2.9.1+cu126 used locally;
# bitsandbytes, peft and transformers all support it, and the 4090 (sm_89)
# needs CUDA 12.x anyway.
#
# Everything persistent lives under /root/autodl-tmp (the data disk). The
# system disk is small and is reset when the image is rebuilt, so the venv,
# the pip cache and the HuggingFace cache all go on the data disk -- the
# Qwen2.5-Omni-3B weights alone are several GB and should survive a restart.
#
# Cost tip: run this in AutoDL's no-GPU mode (~0.1 CNY/h). Uploading data,
# installing packages and downloading the model need no GPU; only start a GPU
# instance once `python -c "import peft, bitsandbytes"` succeeds.
#
# Run ONCE per instance (idempotent):
#     bash src/iemocap/setup_autodl.sh
# Then EVERY session:
#     source /root/autodl-tmp/venv/bin/activate
set -euo pipefail

PERSIST=/root/autodl-tmp
VENV="$PERSIST/venv"
REQ="$(dirname "$(realpath "$0")")/requirements-autodl.txt"

export PIP_CACHE_DIR="$PERSIST/.pip-cache"
export HF_HOME="$PERSIST/.hf"

mkdir -p "$PERSIST" "$PIP_CACHE_DIR" "$HF_HOME"

# ffmpeg: librosa's audio backend. IEMOCAP wavs are plain PCM so soundfile
# alone would usually do, but installing it removes a whole class of surprises.
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo ">> installing ffmpeg ..."
  apt-get update -qq && apt-get install -y -qq ffmpeg
fi

if [ ! -d "$VENV" ]; then
  echo ">> creating venv at $VENV (inheriting the image's torch) ..."
  python -m venv "$VENV" --system-site-packages
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Confirm the inherited torch sees the GPU BEFORE installing anything on top.
# In no-GPU mode `available` prints False, which is expected.
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0),
          f"| {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
PY

echo ">> installing project deps ..."
pip install --upgrade pip -q
pip install -r "$REQ"

python - <<'PY'
import peft, bitsandbytes, transformers, sklearn
print("peft", peft.__version__, "| bitsandbytes", bitsandbytes.__version__,
      "| transformers", transformers.__version__, "| sklearn", sklearn.__version__)
PY

cat <<EOF

>> done.

Persistent paths
    venv      $VENV
    HF cache  $HF_HOME     (export HF_HOME=$HF_HOME in every new shell)
    pip cache $PIP_CACHE_DIR

Every session:
    source $VENV/bin/activate
    export HF_HOME=$HF_HOME

NOTE ON MODEL DOWNLOAD
    The AutoDL instance is in mainland China regardless of where you are, so
    the pull of Qwen/Qwen2.5-Omni-3B happens from there. If it stalls or fails,
    try AutoDL's own accelerator first, then a mirror:
        source /etc/network_turbo                  # AutoDL academic acceleration
        export HF_ENDPOINT=https://hf-mirror.com   # fallback
    Neither is enabled by default here.

Next
    tar xzf iemocap_teacher_subset.tar.gz -C $PERSIST/data
    python src/iemocap/teacher/extract_features.py --split val --limit 5   # smoke
EOF
