"""
Caches the student inputs for MIntRec2.0: log-mel plus low-res video frames.
Same reasoning as fsc/student/precompute_logmel.py - decoding video and audio
every epoch across 10 KD conditions would take over the runtime.

The raw-media helpers are imported from
mintrec.teacher_probe.extract_features_local (load_split_df, build_stem2path,
find_video, load_audio, extract_frames) rather than rewritten, so sample order
and ids match how the teacher features were extracted.

Audio: 16 kHz, n_fft 400, hop 160, 64 mels, fmax 8000, same as FSC. Padded or
cut to 6.0 s, which covers most MIntRec clips - a 40-clip check gave p50 2.4 s,
p90 4.2, p95 5.0, max 8.1.

Video: 8 frames, matching the teacher, resized to a 64x64 RGB thumbnail. Plain
resize, no aspect ratio kept, this is a sanity-check student not a vision
pipeline.

Labels come from the label2id inside the QLoRA teacher config.json rather than
being recomputed, so the id space matches the teacher logits dimension order.

Splits that already have a cache file are skipped. Test is cached here too
since the student needs it for the final eval, but no teacher signal is ever
attached to it - that rule lives in kd_common.py.
"""

import json
import sys
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
from tqdm import tqdm

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import MINTREC_DATA                                       # noqa: E402
from mintrec.teacher_probe.extract_features_local import (                   # noqa: E402
    load_split_df, build_stem2path, find_video, load_audio, extract_frames,
)

OUT_DIR = MINTREC_DATA / "student" / "feature_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_CFG = MINTREC_DATA / "teacher_qlora" / "3b_tva_tr_last_r32" / "config.json"

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 64
FMAX = 8000
MAX_SECONDS = 6.0
TARGET_LEN = int(MAX_SECONDS * SR)

N_FRAMES = 8
FRAME_SIZE = 64


def wav_to_logmel(wav):
    if len(wav) < TARGET_LEN:
        wav = np.pad(wav, (0, TARGET_LEN - len(wav)))
    else:
        wav = wav[:TARGET_LEN]
    mel = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                          n_mels=N_MELS, fmax=FMAX)
    logmel = librosa.power_to_db(mel, ref=1.0)
    return logmel.T.astype(np.float32)


def frames_to_tensor(pil_frames):
    arr = np.stack([cv2.resize(np.array(f), (FRAME_SIZE, FRAME_SIZE)) for f in pil_frames])  # [F,H,W,3]
    if arr.shape[0] < N_FRAMES:                 # repeat the last frame to pad
        pad = np.repeat(arr[-1:], N_FRAMES - arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.shape[0] > N_FRAMES:
        arr = arr[:N_FRAMES]
    return arr.astype(np.uint8)


def process_split(df, s2p, label2id, limit=None):
    n = len(df) if limit is None else min(limit, len(df))
    logmels, frames_all, labels, ids, n_missing = [], [], [], [], 0
    for i in tqdm(range(n), desc="precompute"):
        row = df.iloc[i]
        vp = find_video(row, s2p)
        if vp is None:
            n_missing += 1
            continue
        try:
            wav = load_audio(vp)
            if wav.size == 0:
                n_missing += 1
                continue
            pil_frames = extract_frames(vp, N_FRAMES, max_side=FRAME_SIZE * 2)
        except Exception as ex:
            print(f"  skip {row['id']}: {type(ex).__name__}: {ex}")
            n_missing += 1
            continue
        logmels.append(torch.from_numpy(wav_to_logmel(wav)))
        frames_all.append(torch.from_numpy(frames_to_tensor(pil_frames)))
        labels.append(label2id[row["label"]])
        ids.append(row["id"])
    X_logmel = torch.stack(logmels).unsqueeze(1)                 # [N,1,T,64]
    X_frames = torch.stack(frames_all)                           # uint8
    return X_logmel, X_frames, torch.tensor(labels, dtype=torch.long), ids, n_missing


SPLITS = [("train", "train_features.pt", True),
          ("dev", "dev_features.pt", False),
          ("test", "test_features.pt", False)]


def main():
    with open(LABEL_CFG, encoding="utf-8") as f:
        label2id = json.load(f)["label2id"]

    s2p, n_mp4 = build_stem2path()
    print(f"Indexed {n_mp4:,} clips; {len(label2id)} intent classes")

    n_frames_time = None
    for split, fname, save_stats in SPLITS:
        out_path = OUT_DIR / fname
        if out_path.exists():
            print(f"exists, skip: {fname}")
            continue
        print(f"\n=== {split} ===")
        df = load_split_df(split)
        X_logmel, X_frames, y, ids, n_missing = process_split(df, s2p, label2id)
        n_frames_time = int(X_logmel.shape[2])
        print(f"{split}: logmel {tuple(X_logmel.shape)}  frames {tuple(X_frames.shape)}  "
              f"missing={n_missing}")
        payload = {"logmel": X_logmel.to(torch.float16), "frames": X_frames,
                   "labels": y, "sample_ids": ids}
        if save_stats:
            payload["mean"] = X_logmel.mean(dim=(0, 1, 2))
            payload["std"] = X_logmel.std(dim=(0, 1, 2)).clamp_min(1e-6)
        torch.save(payload, out_path)
        print(f"saved -> {out_path}")

    cfg = {"sr": SR, "n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS, "fmax": FMAX,
           "max_seconds": MAX_SECONDS, "target_len": TARGET_LEN,
           "n_frames_time": n_frames_time, "n_frames_video": N_FRAMES, "frame_size": FRAME_SIZE,
           "normalization": "logmel: per-bin, train mean/std. frames: uint8, /255 at train time."}
    with open(OUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nSaved -> {OUT_DIR}\nDone.")


if __name__ == "__main__":
    main()
