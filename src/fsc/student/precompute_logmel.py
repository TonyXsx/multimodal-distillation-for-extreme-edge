"""
Cache 64-bin log-mel for the student. Run once.

Decoding audio and computing mel every epoch takes hours, so do it once and
keep it on disk. The ablation training then reads straight from RAM.

Samples stay in FSC order, same as the teacher feature files, so cache index i
matches teacher index i. The file id is stored per sample and checked against
the teacher features at train time.

mel: 16 kHz, n_fft 400 (25 ms), hop 160 (10 ms), 64 mels, fmax 8000. Waveforms
padded or cut to MAX_SECONDS so T is fixed.

writes data/student/logmel_cache/{train,val,test}_logmel.pt + config.json.
Splits that already exist are skipped, so adding test doesn't redo train/val.
test is final eval only, no teacher signal needed there.
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


import sys
_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import DATA_ROOT                          # noqa: E402

DATA     = DATA_ROOT
LABEL_CFG = DATA / "fsc_small_ablation" / "config.json"     # same label2id
OUT_DIR  = DATA / "student" / "logmel_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)


SR          = 16000
N_FFT       = 400
HOP         = 160
N_MELS      = 64
FMAX        = 8000
MAX_SECONDS = 3.0
TARGET_LEN  = int(MAX_SECONDS * SR)


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


# (hf_split, out filename, save norm stats)
SPLITS = [("train", "train_logmel.pt", True),
          ("validation", "val_logmel.pt", False),
          ("test", "test_logmel.pt", False)]


def main():
    fsc, label2id = load_fsc()

    n_frames = None
    for hf_split, fname, save_stats in SPLITS:
        out_path = OUT_DIR / fname
        if out_path.exists():
            print(f"exists, skip: {fname}")
            continue
        print(f"\n=== {hf_split} ===")
        X, y, ids, trunc = process(fsc[hf_split])
        n_frames = int(X.shape[2])
        print(f"{hf_split}: {tuple(X.shape)}  truncated(> {MAX_SECONDS}s)={trunc}")
        payload = {"logmel": X.to(torch.float16), "labels": y, "sample_ids": ids}
        if save_stats:  # train only, per-bin stats get applied to every split later
            payload["mean"] = X.mean(dim=(0, 1, 2))
            payload["std"] = X.std(dim=(0, 1, 2)).clamp_min(1e-6)
        torch.save(payload, out_path)
        print(f"saved -> {out_path}")

    cfg = {"sr": SR, "n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS, "fmax": FMAX,
           "max_seconds": MAX_SECONDS, "target_len": TARGET_LEN,
           "n_frames": n_frames if n_frames is not None else TARGET_LEN // HOP + 1,
           "train_n": len(fsc["train"]), "val_n": len(fsc["validation"]),
           "test_n": len(fsc["test"]), "normalization": "per-bin, train mean/std"}
    with open(OUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nSaved -> {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
