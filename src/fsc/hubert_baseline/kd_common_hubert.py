"""
HuBERT-teacher facade for the FSC student KD scripts (audio-only-teacher
baseline). Mirrors fsc/student/kd_common.py's build_teacher_signals() /
load_data(), pointed at the frozen HuBERT-large probe instead of Qwen's.
Everything else (student log-mel cache, training constants, generic
losses/eval/augment) is imported UNCHANGED from kd_common, so the KD recipe
is byte-identical to the Qwen run -- teacher identity is the only variable
that differs between the two pipelines.
"""

import sys
from pathlib import Path

import torch

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.probe import Probe                                        # noqa: E402
from fsc.student.kd_common import (                                    # noqa: E402,F401 (re-export)
    DATA, LOGMEL, DEVICE, evaluate, spec_augment,
    kd_logit_loss, kd_feature_loss, stratified_indices,
    EPOCHS, LR, WEIGHT_DECAY, BATCH_SIZE, LABEL_SMOOTH, SEED,
)

FEAT_TAG = "fsc_full__hubert-large-ll60k__last_layer_mean"
FEAT_DIR = DATA / "teacher_features" / FEAT_TAG
PROBE_CKPT_DIR = DATA / "teacher_probe" / FEAT_TAG / "checkpoints"


def build_teacher_signals():
    """Return dict split -> (z_64 [N,64], logits [N,31], sample_ids) from the
    HuBERT B2-equivalent probe."""
    ckpt_path = next(PROBE_CKPT_DIR.glob("B2_*.pt"))
    ckpt = torch.load(ckpt_path, weights_only=False)
    mean, std = ckpt["standardizer"]["mean"], ckpt["standardizer"]["std"]
    feat_name = ckpt["feature_name"]

    probe = Probe(ckpt["in_dim"], ckpt["hidden_dims"], ckpt["n_classes"], dropout=ckpt["dropout"])
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


def load_data():
    """Returns (Xtr, ytr, ztr, ltr), (Xva, yva, zva, lva): normalized log-mel
    (identical cache to the Qwen run) + aligned HuBERT-teacher bottleneck (z)
    and teacher logits (l)."""
    tr = torch.load(LOGMEL / "train_logmel.pt", weights_only=False)
    va = torch.load(LOGMEL / "val_logmel.pt", weights_only=False)
    mean = tr["mean"].view(1, 1, 1, -1)
    std = tr["std"].view(1, 1, 1, -1)

    Xtr = (tr["logmel"].float() - mean) / std
    Xva = (va["logmel"].float() - mean) / std
    ytr, yva = tr["labels"].long(), va["labels"].long()

    teacher, probe_name = build_teacher_signals()
    ztr, ltr, idtr = teacher["train"]
    zva, lva, idva = teacher["val"]

    # Critical: teacher signals and student inputs must be the same samples, same order.
    assert tr["sample_ids"] == idtr, "TRAIN sample_id mismatch (student vs HuBERT teacher)"
    assert va["sample_ids"] == idva, "VAL sample_id mismatch (student vs HuBERT teacher)"

    print(f"Teacher probe   : {probe_name}  (HuBERT-large-ll60k, audio-only)")
    print(f"train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}")
    return (Xtr, ytr, ztr, ltr), (Xva, yva, zva, lva)
