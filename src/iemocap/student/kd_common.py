"""
IEMOCAP student side: cached inputs, teacher signals, shared constants.

Same discipline as the FSC and MIntRec tracks. Inputs come from the log-mel
cache and are never re-decoded during training, teacher signals exist for train
only so test never gets one, and an id assert ties the two caches together so
they can't quietly go out of sync.

Teacher signals, all precomputed, nothing here re-runs the teacher:

    logits   [N, 4]   the LoRA head output, always the logit-KD target
    z_audio  [N, 64]  bottleneck probe on audio_mean_l27
    z_last   [N, 64]  bottleneck probe on last_token

Both feature targets are kept because which one is right is a question this
dataset can actually answer - on MIntRec every student sat on the macro-F1
noise floor so the comparison meant nothing. The probe results here already
point the other way from MIntRec: on test the clean audio feature
(audio_mean_l27, UA 0.7861) just beats the privileged readout (last_token,
0.7806), because this time the audio tower was adapted too. If that ordering
holds up in the student then the worry about aligning a text-free student to a
text-shaped target is answered for this setting.

Training constants are the tuned FSC recipe, reused as is. No new search here:
T=8, lambda_logit = lambda_feature = 1.0.
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
from iemocap.paths import IEMOCAP_PROBE, IEMOCAP_STUDENT, TEACHER, find_adapted_features  # noqa: E402
from iemocap.teacher.data import CLASSES  # noqa: E402

LOGMEL_DIR = IEMOCAP_STUDENT / "logmel"
BOTTLENECK_DIR = IEMOCAP_PROBE / "bottleneck" / "adapted"
ADAPTED_FEATS = find_adapted_features()

# the FSC small DSResNet-SE, unchanged
SMALL_KW = {"channels": (16, 32, 64, 96, 128), "proj_hidden": None}
PROJ_DIM = 64
N_CLASSES = len(CLASSES)

# tuned FSC recipe, reused as is
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

# the hubert teacher has no prompt and no transcript, so it has an audio
# target and nothing to put opposite it
FEATURE_TARGETS = ({"audio": "audio_mean_l27", "lasttoken": "last_token"} if TEACHER == "qwen"
                   else {"audio": "hubert_mean_l18"})


def available_splits():
    """which splits have a log-mel cache. the LOSO folds have no val set, so ask
    instead of assuming."""
    return tuple(s for s in ("train", "val", "test") if (LOGMEL_DIR / f"{s}.pt").exists())


def load_inputs(split):
    d = torch.load(LOGMEL_DIR / f"{split}.pt", weights_only=False)
    return d["X"], d["labels"], d["ids"]


def normalizer(Xtr):
    """train mean/std over the log-mel cache, in fp32."""
    x = Xtr.float()
    return x.mean(), x.std().clamp_min(1e-6)


def load_teacher_signals(ids):
    """teacher logits and both 64-d targets, aligned to ids.

    Raises if the orders disagree instead of zipping mismatched rows. The two
    caches come from different scripts and only line up because both walk the
    manifest in its stored order.
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
