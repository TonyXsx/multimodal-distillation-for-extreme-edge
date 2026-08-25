"""
39-dim MFCC cache for the MS-SENet student, copying the official TIM-Net /
MS-SENet feature pipeline.

From https://github.com/Jiaxin-Ye/TIM-Net_SER, Code/extract_feature.py, which
MS-SENet reads as a prebuilt .npy:

    signal, fs = librosa.load(path)              # no sr arg, so 22050 Hz
    # symmetric zero-pad or centre crop to mean_signal_length
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=39)
    feature = mfcc.T                             # [T, 39]

Two things about this that are worth writing down rather than just absorbing.

The sample rate is 22050, not the 16 kHz used everywhere else here, because
librosa.load gets no sr. IEMOCAP audio is 16 kHz so this upsamples it, which
adds nothing, but it is what the published numbers came from so it stays.

mean_signal_length is 310000 for IEMOCAP (their constant is IEMOCAP_MFCC_310),
so 14.06 s at 22050 Hz, much longer than the 8 s in the log-mel cache. With hop
512 that is 606 frames at ~43 fps against 801 at 100 fps. So the MS-SENet input
is coarser in time and covers longer.

Crop and pad match the official code: short signals padded symmetrically with
zeros, long ones centre-cropped.

No normalisation by default, the official pipeline feeds raw MFCCs in and
relies on the frontend BatchNorm. --normalize is there so that choice can be
measured instead of assumed.

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

SR = 22050                 # librosa.load default, the official code passes no sr
MEAN_SIGNAL_LENGTH = 310000  # official IEMOCAP setting
N_MFCC = 39
N_FFT = 2048               # librosa defaults, same as the official extractor
HOP = 512
OUT_DIR = IEMOCAP_STUDENT / "mfcc39"


def fit_length(signal, target=MEAN_SIGNAL_LENGTH):
    """symmetric zero-pad or centre-crop, same as the official extractor."""
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
    return m.T.astype(np.float32)


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
