"""
Pre-compute 39-dim MFCCs for the MS-SENet student, following the official
TIM-Net / MS-SENet feature pipeline exactly.

Source: https://github.com/Jiaxin-Ye/TIM-Net_SER  Code/extract_feature.py,
which MS-SENet consumes as a pre-built .npy. Reproduced verbatim:

    signal, fs = librosa.load(path)              # NO sr argument -> 22050 Hz
    # symmetric zero-pad, or CENTRE crop, to exactly mean_signal_length
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=39)   # librosa defaults:
                                                 # n_fft=2048, hop_length=512
    feature = mfcc.T                             # [T, 39]

Two consequences worth stating rather than silently absorbing:

* The sample rate is 22050, not the 16 kHz used everywhere else in this
  project, because `librosa.load` is called without `sr`. IEMOCAP audio is
  16 kHz, so this UPSAMPLES it. That adds no information, but it is what the
  published numbers were produced with, so it is kept.

* `mean_signal_length` is 310000 for IEMOCAP (the official path constant is
  `IEMOCAP_MFCC_310`), i.e. 14.06 s at 22050 Hz -- far longer than the 8 s
  used for the DSResNet-SE log-mel cache. With hop 512 that gives 606 frames
  at ~43 fps, against 801 frames at 100 fps for the log-mel. The MS-SENet
  input is therefore coarser in time and longer in span.

Crop/pad matches the official code: shorter signals are padded symmetrically
with zeros, longer ones are centre-cropped.

No normalisation is applied by default -- the official pipeline feeds raw MFCC
values straight in, relying on the frontend's BatchNorm. `--normalize` exists
so the effect of that choice can be measured rather than assumed.

Outputs:
    data/iemocap/student/mfcc39/{train,val,test}.pt   X [N, 606, 39] float16
    data/iemocap/student/mfcc39/config.json

Usage:
    python src/iemocap/student/precompute_mfcc.py
    python src/iemocap/student/precompute_mfcc.py --splits val --limit 20
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

SR = 22050                 # librosa.load default -- the official code passes no sr
MEAN_SIGNAL_LENGTH = 310000  # official IEMOCAP setting (IEMOCAP_MFCC_310)
N_MFCC = 39
N_FFT = 2048               # librosa defaults, as used by the official extractor
HOP = 512
OUT_DIR = IEMOCAP_STUDENT / "mfcc39"


def fit_length(signal, target=MEAN_SIGNAL_LENGTH):
    """Symmetric zero-pad or centre-crop, exactly as the official extractor."""
    n = len(signal)
    if n < target:
        pad = target - n
        rem = pad % 2
        pad //= 2
        return np.pad(signal, (pad, pad + rem), "constant", constant_values=0), False
    off = (n - target) // 2
    return signal[off:off + target], n > target


def mfcc39(signal, sr=SR):
    m = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    return m.T.astype(np.float32)              # [T, 39]


def main():
    ap = argparse.ArgumentParser(description="Pre-compute official-style 39-dim MFCCs.")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--normalize", action="store_true",
                    help="standardise per coefficient with TRAIN statistics "
                         "(the official pipeline does not)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache, n_crop = {}, {}
    for split in args.splits:
        df = load_split(split, limit=args.limit)
        X, ids, labels, crops = [], [], [], 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc=split, mininterval=5.0):
            sig, fs = librosa.load(str(wav_path(row)), sr=SR)
            sig, cropped = fit_length(sig)
            crops += cropped
            X.append(mfcc39(sig, fs))
            ids.append(row["turn_id"])
            labels.append(int(row["label"]))
        cache[split] = (np.stack(X), torch.tensor(labels, dtype=torch.long), ids)
        n_crop[split] = crops

    mu = sd = None
    if args.normalize and "train" in cache:
        tr = cache["train"][0]
        mu, sd = tr.mean((0, 1), keepdims=True), tr.std((0, 1), keepdims=True).clip(1e-6)

    for split, (X, y, ids) in cache.items():
        if mu is not None:
            X = (X - mu) / sd
        Xt = torch.from_numpy(X).to(torch.float16)
        torch.save({"X": Xt, "labels": y, "ids": ids}, OUT_DIR / f"{split}.pt")
        print(f"{split}: {tuple(Xt.shape)}  centre-cropped={n_crop[split]}", flush=True)

    with open(OUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump({"sr": SR, "mean_signal_length": MEAN_SIGNAL_LENGTH,
                   "seconds": round(MEAN_SIGNAL_LENGTH / SR, 3), "n_mfcc": N_MFCC,
                   "n_fft": N_FFT, "hop": HOP, "normalized": bool(args.normalize),
                   "long_clip_policy": "centre-crop", "pad_policy": "symmetric-zero",
                   "source": "TIM-Net_SER/Code/extract_feature.py (official)"},
                  f, indent=2)
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    main()
