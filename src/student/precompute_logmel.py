"""
Precompute 64-bin log-mel features for the student (one-time).

Decoding FSC audio + computing mel on every epoch would dominate runtime
(~hours). Instead we decode once, compute fixed-length log-mel, and cache to
disk so the 4-ablation x N-epoch student training reads from RAM.

Alignment: samples are kept in FSC order (same as the teacher feature files),
so cache index i corresponds to teacher signal index i. We also store the file
id per sample and assert-match against the teacher features at train time.

Mel config: 16 kHz, n_fft=400 (25 ms), hop=160 (10 ms), n_mels=64, fmax=8000.
Waveforms are padded/truncated to MAX_SECONDS so every sample has T frames.

Output:
    data/student/logmel_cache/
        train_logmel.pt   { logmel:[N,1,T,64] fp16, labels, sample_ids, mean[64], std[64] }
        val_logmel.pt     { logmel:[N,1,T,64] fp16, labels, sample_ids }
        config.json
"""

import io
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT  = Path(r"D:\msc_AI\individual_project\multimodal-distillation-for-extreme-edge")
DATA     = PROJECT / "data"
LABEL_CFG = DATA / "fsc_small_ablation" / "config.json"     # reuse identical label2id
OUT_DIR  = DATA / "student" / "logmel_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Mel config ──────────────────────────────────────────────────────────────────
SR          = 16000
N_FFT       = 400
HOP         = 160
N_MELS      = 64
FMAX        = 8000
MAX_SECONDS = 3.0
TARGET_LEN  = int(MAX_SECONDS * SR)         # pad/truncate waveforms to this


def wav_to_logmel(wav):
    if len(wav) < TARGET_LEN:
        wav = np.pad(wav, (0, TARGET_LEN - len(wav)))
    else:
        wav = wav[:TARGET_LEN]
    mel = librosa.feature.melspectrogram(
        y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmax=FMAX
    )
    logmel = librosa.power_to_db(mel, ref=1.0)          # [n_mels, T]
    return logmel.T.astype(np.float32)                  # [T, n_mels]


def load_fsc():
    with open(LABEL_CFG, encoding="utf-8") as f:
        label2id = json.load(f)["label2id"]
    fsc = load_dataset("s3prl/superb", name="ic", cache_dir=str(DATA))
    feats = fsc["train"].features
    an, on, ln = feats["action"].names, feats["object"].names, feats["location"].names
    fsc = fsc.cast_column("audio", Audio(decode=False))

    def add(ex):
        intent = f"{an[ex['action']]}_{on[ex['object']]}_{ln[ex['location']]}"
        ex["intent"] = intent
        ex["label_id"] = label2id[intent]
        return ex

    return fsc.map(add, desc="intent/label_id"), label2id


def process(ds):
    logmels, labels, ids = [], [], []
    n_trunc = 0
    for i in tqdm(range(len(ds)), desc="logmel"):
        ex = ds[i]
        wav, _ = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32", always_2d=False)
        if len(wav) > TARGET_LEN:
            n_trunc += 1
        lm = wav_to_logmel(wav)                          # [T, 64]
        logmels.append(torch.from_numpy(lm))
        labels.append(int(ex["label_id"]))
        ids.append(str(ex.get("file", i)))
    X = torch.stack(logmels).unsqueeze(1)                # [N, 1, T, 64]
    return X, torch.tensor(labels, dtype=torch.long), ids, n_trunc


def main():
    fsc, label2id = load_fsc()

    print("\n=== TRAIN ===")
    Xtr, ytr, idtr, tr_trunc = process(fsc["train"])
    print("=== VAL ===")
    Xva, yva, idva, va_trunc = process(fsc["validation"])
    print(f"\nshapes: train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}")
    print(f"truncated (> {MAX_SECONDS}s): train {tr_trunc}  val {va_trunc}")

    # Per-bin normalization stats from TRAIN only (applied at train time).
    mean = Xtr.mean(dim=(0, 1, 2))                        # [64]
    std  = Xtr.std(dim=(0, 1, 2)).clamp_min(1e-6)         # [64]

    torch.save({"logmel": Xtr.to(torch.float16), "labels": ytr, "sample_ids": idtr,
                "mean": mean, "std": std}, OUT_DIR / "train_logmel.pt")
    torch.save({"logmel": Xva.to(torch.float16), "labels": yva, "sample_ids": idva},
               OUT_DIR / "val_logmel.pt")

    cfg = {"sr": SR, "n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS, "fmax": FMAX,
           "max_seconds": MAX_SECONDS, "target_len": TARGET_LEN,
           "n_frames": int(Xtr.shape[2]), "train_n": int(Xtr.shape[0]),
           "val_n": int(Xva.shape[0]), "normalization": "per-bin, train mean/std"}
    with open(OUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nSaved -> {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
