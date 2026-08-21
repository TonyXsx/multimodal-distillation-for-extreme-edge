"""
IEMOCAP student facade: cached inputs, teacher signals, shared constants.

Discipline carried over from FSC and MIntRec:
  * student inputs come from the pre-computed log-mel cache, never re-decoded
    during training;
  * teacher signals exist for TRAIN only -- test never receives one;
  * a strict id assert ties inputs to teacher signals, so a silent
    misalignment between the two caches cannot happen.

Teacher signals (all pre-computed, nothing here re-runs the teacher):

    logits   [N, 4]   the LoRA head's own output -- always the Logit-KD target
    z_audio  [N, 64]  bottleneck probe on `audio_mean_l27` (adapted arm)
    z_last   [N, 64]  bottleneck probe on `last_token` (adapted arm)

Both feature targets are kept because which one is right is an empirical
question this dataset can finally answer. On MIntRec every student sat on the
macro-F1 ~0.06 noise floor, so the comparison there was meaningless. Here the
probe results already point the other way from MIntRec: on held-out test the
CLEAN audio feature (`audio_mean_l27`, UA 0.7861) edges out the privileged
readout (`last_token`, 0.7806), because this time the audio tower itself was
adapted. If that ordering survives into the student, the "aligning a
text-free student to a text-shaped target" worry is settled for this setting.

Training constants are the tuned FSC recipe, unchanged, so nothing here is a
new hyperparameter search: T=8, lambda_logit = lambda_feature = 1.0.
"""

import sys
from pathlib import Path

import torch

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.losses import kd_feature_loss, kd_logit_loss, rkd_loss  # noqa: E402,F401
from common.models.audio_student import DSResNetSE, model_summary  # noqa: E402,F401
from common.training import DEVICE  # noqa: E402,F401
from iemocap.paths import IEMOCAP_PROBE, IEMOCAP_STUDENT, find_adapted_features  # noqa: E402
from iemocap.teacher.data import CLASSES  # noqa: E402

LOGMEL_DIR = IEMOCAP_STUDENT / "logmel"
BOTTLENECK_DIR = IEMOCAP_PROBE / "bottleneck" / "adapted"
ADAPTED_FEATS = find_adapted_features()

# Student architecture: the FSC "small" DSResNet-SE, unchanged.
SMALL_KW = {"channels": (16, 32, 64, 96, 128), "proj_hidden": None}
PROJ_DIM = 64
N_CLASSES = len(CLASSES)

# Tuned FSC recipe, reused verbatim.
EPOCHS = 70
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
DROPOUT = 0.2
LABEL_SMOOTH = 0.1
T_KD = 8.0
LAM_LOGIT = 1.0
LAM_FEATURE = 1.0
SEED = 42

FEATURE_TARGETS = {"audio": "audio_mean_l27", "lasttoken": "last_token"}


def load_inputs(split):
    d = torch.load(LOGMEL_DIR / f"{split}.pt", weights_only=False)
    return d["X"], d["labels"], d["ids"]


def normalizer(Xtr):
    """Train-set mean/std over the log-mel cache (float32 for stability)."""
    x = Xtr.float()
    return x.mean(), x.std().clamp_min(1e-6)


def load_teacher_signals(ids):
    """Teacher logits + both 64-d bottleneck targets, aligned to `ids`.

    Raises if the orderings disagree rather than silently zipping mismatched
    rows -- the two caches are produced by different scripts and only agree
    because both iterate the manifest in its stored order.
    """
    feats = torch.load(ADAPTED_FEATS / "train_features.pt", weights_only=False)
    if list(feats["sample_ids"]) != list(ids):
        raise RuntimeError("teacher feature ids do not match the log-mel cache ids")
    out = {"logits": feats["features"]["logits"].float()}
    for name, key in FEATURE_TARGETS.items():
        rep = torch.load(BOTTLENECK_DIR / key / "bottleneck_reps.pt", weights_only=False)["train"]
        if list(rep["ids"]) != list(ids):
            raise RuntimeError(f"bottleneck ids for {key} do not match the log-mel cache ids")
        out[f"z_{name}"] = rep["z"].float()
    return out
