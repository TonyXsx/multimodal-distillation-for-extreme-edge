"""
Augmented log-mel cache for the IEMOCAP train split: speed perturbation + VTLP.

Both augmentations were picked for the same reason. The failure mode on this
project is generalising across speakers, and each split has exactly two of
them, so anything that manufactures speaker variety attacks the real problem.
Noise and reverb target channel robustness instead, which is not the issue.

Speed perturbation: resample to get x(alpha*t) with alpha in {0.9, 1.0, 1.1},
so tempo and pitch move together (Ko et al. 2015). The pitch part is what
matters here - phase-vocoder time stretching keeps pitch and wouldn't perturb
the speaker at all.

VTLP (Jaitly & Hinton 2013): a piecewise linear warp of the frequency axis,
applied to the power spectrogram before the mel filterbank.

    f' = f * a                                          for f <= f_hi*min(a,1)/a
    f' = F - (F - f_hi*min(a,1)) / (F - f_hi*min(a,1)/a) * (F - f)   otherwise

with F = sr/2 and f_hi = 4800 Hz. It fakes a different vocal tract length, i.e.
a different speaker. The SER literature recommends it for IEMOCAP specifically
because it increases the effective number of speakers, which is exactly what
6 training speakers is short of.

Only train gets expanded. Val and test stay identical to what
precompute_logmel.py wrote, so old numbers still compare. Copy 0 of each
utterance is the untouched original, the rest get a speed factor and an
independently drawn VTLP factor.

Every copy stores orig_idx pointing back at the utterance it came from, and
inherits that utterance's teacher logits and bottleneck targets. The teacher
heard the original audio, and its judgement doesn't change because the copy is
played faster or with shifted formants.

That asymmetry is the whole point. CE gets N times more (input, label) pairs;
distillation gets N times more (input, teacher output) pairs, so the student
has to reproduce the same teacher response under variation the teacher never
saw. If augmentation is going to help KD more than it helps CE, this is how.

Mel config is imported from precompute_logmel.py rather than restated.

    python src/iemocap/student/precompute_logmel_aug.py
    python src/iemocap/student/precompute_logmel_aug.py --copies 5 --limit 20
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
from iemocap.paths import IEMOCAP_DATA  # noqa: E402,F401
from iemocap.student.precompute_logmel import (  # noqa: E402  shared config, not copied
    FMAX, HOP, N_FFT, N_MELS, OUT_DIR, SR, fit_length,
)
from iemocap.teacher.data import load_split, wav_path  # noqa: E402

F_HI = 4800.0          # VTLP boundary freq, from the paper
VTLP_RANGE = (0.9, 1.1)
SPEEDS = (1.0, 0.9, 1.1)   # copy 0 is the original


def speed_perturb(y, sr, alpha):
    """x(alpha*t). resample to sr/alpha then pretend the result is sr."""
    if alpha == 1.0:
        return y
    return librosa.resample(y, orig_sr=sr, target_sr=int(round(sr / alpha)))


def vtlp_interp(alpha, n_freq, sr=SR, f_hi=F_HI):
    """index/weight arrays for warping the frequency axis.

    Returns (lo, hi, frac), so the warped spectrum is
    S[lo] * (1 - frac) + S[hi] * frac. Built once per alpha and reused for
    every frame, the warp doesn't depend on time.
    """
    F = sr / 2.0
    freqs = np.linspace(0.0, F, n_freq)
    boundary = f_hi * min(alpha, 1.0) / alpha
    scale = (F - f_hi * min(alpha, 1.0)) / (F - boundary)
    warped = np.where(freqs <= boundary, freqs * alpha, F - scale * (F - freqs))
    # invert it: for each output bin, which input freq lands there
    src = np.interp(freqs, warped, freqs)
    pos = np.clip(src / F * (n_freq - 1), 0, n_freq - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.clip(lo + 1, 0, n_freq - 1)
    return lo, hi, (pos - lo).astype(np.float32)


def logmel_vtlp(y, vtlp_alpha, mel_fb, cache):
    """power spectrogram -> optional VTLP warp -> mel -> dB. identical to
    precompute_logmel.logmel when vtlp_alpha is 1.0."""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2
    if vtlp_alpha != 1.0:
        key = round(vtlp_alpha, 4)
        if key not in cache:
            cache[key] = vtlp_interp(vtlp_alpha, S.shape[0])
        lo, hi, frac = cache[key]
        S = S[lo] * (1.0 - frac)[:, None] + S[hi] * frac[:, None]
    return librosa.power_to_db(mel_fb @ S, ref=1.0).T.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Speed + VTLP log-mel cache (train only).")
    ap.add_argument("--copies", type=int, default=3, help="copies per utterance (copy 0 = clean)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="train_aug.pt")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    mel_fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmax=FMAX)
    cache = {}

    df = load_split("train", limit=args.limit)
    X, ids, labels, orig, sp_log, vt_log, n_crop = [], [], [], [], [], [], 0
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="train+aug",
                                      mininterval=5.0)):
        wav, _ = librosa.load(str(wav_path(row)), sr=SR)
        for c in range(args.copies):
            if c == 0:
                sp, vt = 1.0, 1.0
            else:
                sp = SPEEDS[c % len(SPEEDS)] if c < len(SPEEDS) else float(rng.choice([0.9, 1.1]))
                vt = float(rng.uniform(*VTLP_RANGE))
            w, cropped = fit_length(speed_perturb(wav, SR, sp))
            n_crop += cropped
            X.append(logmel_vtlp(w, vt, mel_fb, cache))
            ids.append(f"{row['turn_id']}#c{c}")
            labels.append(int(row["label"]))
            orig.append(i)
            sp_log.append(sp)
            vt_log.append(vt)

    Xt = torch.from_numpy(np.stack(X)).to(torch.float16)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"X": Xt, "labels": torch.tensor(labels, dtype=torch.long), "ids": ids,
                "orig_idx": torch.tensor(orig, dtype=torch.long),
                "speed": torch.tensor(sp_log, dtype=torch.float32),
                "vtlp": torch.tensor(vt_log, dtype=torch.float32)},
               OUT_DIR / args.out)
    print(f"train+aug: {tuple(Xt.shape)}  from {len(df)} utterances x {args.copies} copies"
          f"  centre-cropped={n_crop}", flush=True)

    with open(OUT_DIR / "config_aug.json", "w", encoding="utf-8") as f:
        json.dump({"copies": args.copies, "n_source": len(df), "n_total": len(X),
                   "speeds": list(SPEEDS[:args.copies]), "vtlp_range": list(VTLP_RANGE),
                   "vtlp_f_hi": F_HI, "seed": args.seed,
                   "speed_method": "resample to sr/alpha, replayed at sr (tempo+pitch)",
                   "vtlp_method": "piecewise-linear frequency warp of the power "
                                  "spectrogram before the mel filterbank",
                   "splits_augmented": ["train"],
                   "teacher_signal_policy": "copies inherit the source utterance's "
                                            "teacher logits and bottleneck targets"},
                  f, indent=2)
    print(f"-> {OUT_DIR / args.out}")


if __name__ == "__main__":
    main()
