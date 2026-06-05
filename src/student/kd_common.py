"""
Shared infrastructure for all student KD experiments.

This module holds the pieces that every student training/tuning script needs —
data loading, teacher-signal construction, KD losses, SpecAugment, evaluation,
stratified subsetting, and the common training constants — so the experiment
scripts (train_student.py, train_student_2x2.py, tune_kd_hparams.py) only
contain their own experiment logic and import the rest from here.

Inputs it reads:
  data/student/logmel_cache/{train,val}_logmel.pt          (precompute_logmel.py)
  data/teacher_features/<feat>/{train,val}_features.pt      (full_feature_extraction.py)
  data/teacher_probe/<feat>/checkpoints/B2_*.pt            (train_probe.py)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

PROJECT = Path(r"D:\msc_AI\individual_project\multimodal-distillation-for-extreme-edge")
sys.path.insert(0, str(PROJECT / "src" / "teacher_probe"))
from train_probe import Probe   # noqa: E402  (teacher probe class, for build_teacher_signals)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA     = PROJECT / "data"
LOGMEL   = DATA / "student" / "logmel_cache"
FEAT_TAG = "fsc_full__qwen2.5-omni-3b-4bit__pf_audiomean_L24-27-30-34"
FEAT_DIR = DATA / "teacher_features" / FEAT_TAG
PROBE_CKPT_DIR = DATA / "teacher_probe" / FEAT_TAG / "checkpoints"

# ── Shared training constants ─────────────────────────────────────────────────────
# SpecAugment + label smoothing are part of the shared training recipe (the
# strong baseline), applied identically across experiments so the only variable
# is the studied factor (KD loss / data fraction / capacity).
EPOCHS       = 70
LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 256
DROPOUT      = 0.2
LABEL_SMOOTH = 0.1
SEED         = 42
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# ── Teacher signals ──────────────────────────────────────────────────────────────
def build_teacher_signals():
    """Return dict split -> (z_64 [N,64], logits [N,31], sample_ids) from the B2 probe."""
    ckpt_path = next(PROBE_CKPT_DIR.glob("B2_*.pt"))
    ckpt = torch.load(ckpt_path, weights_only=False)
    mean, std = ckpt["standardizer"]["mean"], ckpt["standardizer"]["std"]
    feat_name = ckpt["feature_name"]

    probe = Probe(2048, ckpt["hidden_dims"], ckpt["n_classes"], dropout=ckpt["dropout"])
    probe.load_state_dict(ckpt["state_dict"])
    probe.eval().to(DEVICE)

    out = {}
    for split, fname in [("train", "train_features.pt"), ("val", "val_features.pt")]:
        d = torch.load(FEAT_DIR / fname, weights_only=False)
        X = ((d["features"][feat_name].float() - mean) / std).to(DEVICE)
        with torch.no_grad():
            logits, z = probe(X, return_bottleneck=True)
        out[split] = (z.cpu(), logits.cpu(), d["sample_ids"])
    return out, ckpt_path.name


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data():
    """Returns (Xtr, ytr, ztr, ltr), (Xva, yva, zva, lva): normalized log-mel +
    aligned teacher bottleneck (z) and teacher logits (l)."""
    tr = torch.load(LOGMEL / "train_logmel.pt", weights_only=False)
    va = torch.load(LOGMEL / "val_logmel.pt",   weights_only=False)
    mean = tr["mean"].view(1, 1, 1, -1)
    std  = tr["std"].view(1, 1, 1, -1)

    Xtr = (tr["logmel"].float() - mean) / std
    Xva = (va["logmel"].float() - mean) / std
    ytr, yva = tr["labels"].long(), va["labels"].long()

    teacher, probe_name = build_teacher_signals()
    ztr, ltr, idtr = teacher["train"]
    zva, lva, idva = teacher["val"]

    # Critical: teacher signals and student inputs must be the same samples, same order.
    assert tr["sample_ids"] == idtr, "TRAIN sample_id mismatch (student vs teacher)"
    assert va["sample_ids"] == idva, "VAL sample_id mismatch (student vs teacher)"

    print(f"Teacher probe   : {probe_name}")
    print(f"train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}")
    return (Xtr, ytr, ztr, ltr), (Xva, yva, zva, lva)


# ── Losses ──────────────────────────────────────────────────────────────────────
def kd_logit_loss(student_logits, teacher_logits, t):
    return F.kl_div(
        F.log_softmax(student_logits / t, dim=1),
        F.softmax(teacher_logits / t, dim=1),
        reduction="batchmean",
    ) * (t * t)


def kd_feature_loss(student_z, teacher_z):
    return 1.0 - F.cosine_similarity(student_z, teacher_z, dim=1).mean()


# ── SpecAugment ──────────────────────────────────────────────────────────────────
def spec_augment(x, n_freq=2, n_time=2, f_max=12, t_max=40):
    """Per-batch time/freq masking (training only). x: [B,1,T,F] normalized log-mel."""
    B, _, T, F_ = x.shape
    x = x.clone()
    for _ in range(n_freq):
        f = int(torch.randint(0, f_max + 1, (1,)))
        if f > 0:
            f0 = int(torch.randint(0, max(1, F_ - f), (1,)))
            x[:, :, :, f0:f0 + f] = 0.0
    for _ in range(n_time):
        t = int(torch.randint(0, t_max + 1, (1,)))
        if t > 0:
            t0 = int(torch.randint(0, max(1, T - t), (1,)))
            x[:, :, t0:t0 + t, :] = 0.0
    return x


# ── Eval ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    preds = []
    for i in range(0, X.shape[0], BATCH_SIZE):
        _, logits = model(X[i:i + BATCH_SIZE].to(DEVICE))
        preds.append(logits.argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    yt = y.numpy()
    return {
        "acc": accuracy_score(yt, preds),
        "macro_f1": f1_score(yt, preds, average="macro"),
        "weighted_f1": f1_score(yt, preds, average="weighted"),
    }


# ── Stratified subset (for limited-data settings) ─────────────────────────────────
def stratified_indices(labels, frac, seed):
    """Sorted indices of a stratified `frac` subset of `labels`, RNG-seeded by `seed`."""
    rng = np.random.RandomState(seed)
    y = labels.numpy()
    idx = []
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        k = max(1, int(round(len(c_idx) * frac)))
        idx.extend(rng.choice(c_idx, size=k, replace=False))
    return np.sort(np.array(idx))
