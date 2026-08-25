"""
Log-mel cache for the IEMOCAP student. Run once.

Same mel config as FSC and MIntRec so the three students compare directly:
16 kHz, n_fft 400 (25 ms), hop 160 (10 ms), 64 mels, fmax 8000.

Only the clip length changes. IEMOCAP durations are long-tailed - mean 4.55 s,
p50 3.58, p90 8.69, p95 11.06, max 34.14 - so 8.0 s, which covers about the
90th percentile without paying for the tail where padding would take over the
input. FSC used 3 s and MIntRec 6 s. DSResNet-SE pools globally before the
projection head, so clip length changes compute but not parameter count.

Long utterances are centre-cropped rather than cut from the start, since the
emotional part of a turn isn't reliably at the beginning.

    python src/iemocap/student/precompute_logmel.py
    python src/iemocap/student/precompute_logmel.py --splits val --limit 20
"""

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from iemocap.paths import IEMOCAP_STUDENT  # noqa: E402
from iemocap.teacher.data import SPLITS, load_split, wav_path  # noqa: E402

SR, N_FFT, HOP, N_MELS, FMAX = 16000, 400, 160, 64, 8000
MAX_SECONDS = 8.0
TARGET_LEN = int(MAX_SECONDS * SR)
OUT_DIR = IEMOCAP_STUDENT / "logmel"


def fit_length(wav):
    """centre-crop or right-pad to TARGET_LEN."""
    n = len(wav)
    if n > TARGET_LEN:
        start = (n - TARGET_LEN) // 2
        return wav[start:start + TARGET_LEN], True
    if n < TARGET_LEN:
        return np.pad(wav, (0, TARGET_LEN - n)), False
    return wav, False


def logmel(wav):
    mel = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                         n_mels=N_MELS, fmax=FMAX)
    return librosa.power_to_db(mel, ref=1.0).T.astype(np.float32)   # [T, n_mels]


def main():
    ap = argparse.ArgumentParser(description="Pre-compute IEMOCAP student log-mels.")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        df = load_split(split, limit=args.limit)
        if df.empty:                      # LOSO has no val
            print(f"{split}: absent in this protocol's manifest, skipping")
            continue
        X, ids, labels, n_crop = [], [], [], 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc=split):
            wav, _ = librosa.load(str(wav_path(row)), sr=SR, mono=True)
            wav, cropped = fit_length(wav)
            n_crop += cropped
            X.append(logmel(wav))
            ids.append(row["turn_id"])
            labels.append(int(row["label"]))
        X = torch.from_numpy(np.stack(X)).to(torch.float16)
        torch.save({"X": X, "labels": torch.tensor(labels, dtype=torch.long), "ids": ids},
                   OUT_DIR / f"{split}.pt")
        print(f"{split}: {tuple(X.shape)}  centre-cropped(>{MAX_SECONDS}s)={n_crop}", flush=True)

    with open(OUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump({"sr": SR, "n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS, "fmax": FMAX,
                   "max_seconds": MAX_SECONDS, "target_len": TARGET_LEN,
                   "long_clip_policy": "centre-crop", "dtype": "float16"}, f, indent=2)
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    main()
