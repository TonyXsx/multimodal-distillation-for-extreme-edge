"""
FSC student facade: FSC-specific data loading + teacher-signal construction +
the FSC training constants. Generic, dataset-agnostic pieces (KD losses,
SpecAugment, evaluate, stratified subsetting, Probe, models) live in src/common/
and are RE-EXPORTED here so the FSC scripts can import everything from one place.

Inputs it reads:
  data/student/logmel_cache/{train,val,test}_logmel.pt     (precompute_logmel.py)
  data/teacher_features/<feat>/{train,val}_features.pt      (full_feature_extraction.py)
  data/teacher_probe/<feat>/checkpoints/B2_*.pt            (train_probe.py)
"""

import sys
from pathlib import Path

import torch

# ── Make src/ importable (file-relative, no hardcoded drive) ──────────────────────
_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.config import DATA_ROOT                                 # noqa: E402
from common.probe import Probe                                     # noqa: E402
from common.training import DEVICE, evaluate, stratified_indices   # noqa: E402,F401 (re-export)
from common.losses import kd_logit_loss, kd_feature_loss           # noqa: E402,F401 (re-export)
from common.augment import spec_augment                            # noqa: E402,F401 (re-export)

# ── FSC paths ─────────────────────────────────────────────────────────────────────
DATA     = DATA_ROOT
LOGMEL   = DATA_ROOT / "student" / "logmel_cache"
FEAT_TAG = "fsc_full__qwen2.5-omni-3b-4bit__pf_audiomean_L24-27-30-34"
FEAT_DIR = DATA_ROOT / "teacher_features" / FEAT_TAG
PROBE_CKPT_DIR = DATA_ROOT / "teacher_probe" / FEAT_TAG / "checkpoints"

# ── Shared training constants (strong baseline recipe) ────────────────────────────
EPOCHS       = 70
LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 256
DROPOUT      = 0.2
LABEL_SMOOTH = 0.1
SEED         = 42


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
