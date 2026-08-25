"""
MIntRec2.0 student side: data loading and teacher signals.

Same shape and same discipline as fsc/student/kd_common.py. Student inputs
(log-mel + video frames) come from the precomputed cache and are never
re-decoded during training. Teacher signals exist for train and dev only, test
never gets one - the test log-mel and frames are loaded separately at final eval
time by whichever run_final_kd needs them. A sample_id assert ties the inputs to
the signals so they cannot quietly go out of sync.

Three teacher signals, all pre-extracted, nothing here re-runs the teacher:

  logits         [N, 30]  the QLoRA head output, from extract_with_lora.py.
                          always the logit-KD target, both students.
  z_audiohidden  [N, 64]  bottleneck probe on audio_mean_l27. feature-KD target
                          for the audio-only "...audiohidden" runs.
  z_lasttoken    [N, 64]  bottleneck probe on last_token. target for the
                          "...lasttoken" runs and for the AV fusion alignment.
"""

import sys
from pathlib import Path

import torch

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.config import MINTREC_DATA                              # noqa: E402
from common.training import DEVICE, evaluate                        # noqa: E402,F401 (re-export)
from common.losses import kd_logit_loss, kd_feature_loss             # noqa: E402,F401 (re-export)

FEATURE_CACHE = MINTREC_DATA / "student" / "feature_cache"
QLORA_DIR = MINTREC_DATA / "teacher_features" / "mintrec2.0__qwen2.5-omni-3b-4bit-QLORA__tva_tr__adapter_ep3"
BOTTLENECK_DIR = MINTREC_DATA / "teacher_probe" / "qlora_bottleneck"


EPOCHS = 70
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128            # MIntRec train (6,165) is smaller than FSC, 128 keeps the step count sane
DROPOUT = 0.2
LABEL_SMOOTH = 0.1
SEED = 42
T_KD = 8.0
LAM_LOGIT = 1.0
LAM_FEATURE = 1.0


def _load_split_inputs(split):
    d = torch.load(FEATURE_CACHE / f"{split}_features.pt", weights_only=False)
    return d


def build_teacher_signals():
    """split -> (logits[N,30], z_audiohidden[N,64], z_lasttoken[N,64], sample_ids)."""
    qtr = torch.load(QLORA_DIR / "train_features.pt", weights_only=False)
    qdv = torch.load(QLORA_DIR / "dev_features.pt", weights_only=False)
    bh_tr = torch.load(BOTTLENECK_DIR / "audio_mean_l27" / "bottleneck_reps.pt", weights_only=False)
    bh_lt = torch.load(BOTTLENECK_DIR / "last_token" / "bottleneck_reps.pt", weights_only=False)

    out = {}
    for split, q, key in (("train", qtr, "train"), ("dev", qdv, "dev")):
        logits = q["features"]["logits"].float()
        ids_q = q["sample_ids"]
        z_audio = bh_tr[key]["emb"].float()
        ids_audio = bh_tr[key]["sample_ids"]
        z_last = bh_lt[key]["emb"].float()
        ids_last = bh_lt[key]["sample_ids"]
        assert ids_q == ids_audio == ids_last, f"{split}: teacher-signal sample_id mismatch"
        out[split] = (logits, z_audio, z_last, ids_q)
    return out


def load_data():
    """train_data, dev_data. each has logmel [N,1,T,64], frames [N,F,3,H,W] in
    [0,1], labels, logits, z_audiohidden, z_lasttoken."""
    tr = _load_split_inputs("train")
    dv = _load_split_inputs("dev")
    mean = tr["mean"].view(1, 1, 1, -1)
    std = tr["std"].view(1, 1, 1, -1)

    teacher = build_teacher_signals()

    out = {}
    for split, d in (("train", tr), ("dev", dv)):
        logmel = (d["logmel"].float() - mean) / std
        frames = d["frames"].permute(0, 1, 4, 2, 3).float() / 255.0     # -> [N,F,3,H,W] in [0,1]
        logits, z_audio, z_last, ids_t = teacher[split]
        assert d["sample_ids"] == ids_t, f"{split}: student/teacher sample_id mismatch"
        out[split] = {
            "logmel": logmel, "frames": frames, "labels": d["labels"].long(),
            "logits": logits, "z_audiohidden": z_audio, "z_lasttoken": z_last,
        }
        print(f"{split:5} logmel {tuple(logmel.shape)}  frames {tuple(frames.shape)}")
    return out["train"], out["dev"]


def load_test():
    """test log-mel and frames only. the student is audio+visual at inference so
    there is no teacher signal to load."""
    te = _load_split_inputs("test")
    tr = _load_split_inputs("train")
    mean = tr["mean"].view(1, 1, 1, -1)
    std = tr["std"].view(1, 1, 1, -1)
    logmel = (te["logmel"].float() - mean) / std
    frames = te["frames"].permute(0, 1, 4, 2, 3).float() / 255.0
    return logmel, frames, te["labels"].long()
