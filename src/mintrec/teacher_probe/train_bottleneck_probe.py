"""
Bottleneck probes on the adapted MIntRec2.0 teacher, 2048 -> 64 -> 30.

Same B2 recipe as fsc/teacher_probe/train_probe.py, same fixed hyperparameters
and the same "dev is the final eval set, no early stopping" rule, so the 64-d
bottleneck can be a feature-KD target in exactly the shape FSC used.

Why this stage exists: extract_with_lora.py and probe_qlora.py gave us raw
2048-d hidden states (audio_mean_l27, last_token) and a strong 30-way logits
head, but no compact version of those hidden states. On FSC the B2 checkpoint
is itself the feature-KD target, so this closes the same gap for MIntRec by
training a B2-shaped probe on each of the two candidate features.

Two probes, bottleneck fixed at 64 to match the student proj_dim so
kd_feature_loss needs no extra projector:

  audio_mean_l27  the clean audio-attending feature, used by the audio-only
                  student's "..._audiohidden" runs
  last_token      the privileged all-modality readout, used by the
                  "..._lasttoken" runs and the AV fusion alignment

Logit-KD does not use either probe's classifier output, it uses the extracted
logits directly - that's the real tuned head at 65.3% test acc, stronger than a
from-scratch probe. These two exist only to give feature-KD a low-dim target.

Same constants as the FSC probe: 50 epochs, AdamW(1e-3, wd 1e-4), batch 256,
dropout 0.1, CE, standardised on train stats, acc and macro-F1 reported once on
dev with no selection on it. Test features are never loaded here.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import MINTREC_DATA, MINTREC_OUTPUTS   # noqa: E402
from common.probe import Probe                            # noqa: E402

QLORA_DIR = MINTREC_DATA / "teacher_features" / "mintrec2.0__qwen2.5-omni-3b-4bit-QLORA__tva_tr__adapter_ep3"
OUT_ROOT = MINTREC_DATA / "teacher_probe" / "qlora_bottleneck"
OUT_PLOT = MINTREC_OUTPUTS / "teacher_probe"
OUT_PLOT.mkdir(parents=True, exist_ok=True)

# same hyperparameters as the FSC B2 probe
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
DROPOUT = 0.1
BOTTLENECK = 64
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURES = ["audio_mean_l27", "last_token"]


def load_feature(name):
    tr = torch.load(QLORA_DIR / "train_features.pt", weights_only=False)
    dv = torch.load(QLORA_DIR / "dev_features.pt", weights_only=False)
    Xtr = tr["features"][name].float()
    Xdv = dv["features"][name].float()
    ytr = tr["labels"].long()
    ydv = dv["labels"].long()

    mean = Xtr.mean(dim=0, keepdim=True)
    std = Xtr.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xtr_n = (Xtr - mean) / std
    Xdv_n = (Xdv - mean) / std
    return (Xtr_n, ytr, tr["sample_ids"]), (Xdv_n, ydv, dv["sample_ids"]), (mean, std)


def train_one(name):
    print(f"\n=== {name} ===")
    (Xtr, ytr, tr_ids), (Xdv, ydv, dv_ids), (mean, std) = load_feature(name)
    n_classes = int(max(ytr.max(), ydv.max()).item()) + 1
    in_dim = Xtr.shape[1]
    print(f"train {tuple(Xtr.shape)}  dev {tuple(Xdv.shape)}  classes {n_classes}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = Probe(in_dim, [BOTTLENECK], n_classes, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    Xtr_d, ytr_d = Xtr.to(DEVICE), ytr.to(DEVICE)
    Xdv_d, ydv_d = Xdv.to(DEVICE), ydv.to(DEVICE)
    n = Xtr_d.shape[0]

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_d[idx]), ytr_d[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        tr_pred = model(Xtr_d).argmax(1)
        dv_pred = model(Xdv_d).argmax(1)
    train_acc = (tr_pred == ytr_d).float().mean().item()
    eval_acc = (dv_pred == ydv_d).float().mean().item()
    eval_f1 = f1_score(ydv.numpy(), dv_pred.cpu().numpy(), average="macro")
    print(f"[{name}] bottleneck={BOTTLENECK}  dev_acc={eval_acc:.4f}  macroF1={eval_f1:.4f}  (train_acc={train_acc:.4f})")

    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "feature_name": name, "hidden_dims": [BOTTLENECK], "n_classes": n_classes,
        "in_dim": in_dim, "dropout": DROPOUT,
        "state_dict": model.state_dict(),
        "standardizer": {"mean": mean, "std": std},
        "eval_acc": eval_acc, "eval_macro_f1": eval_f1, "train_acc": train_acc,
    }, out_dir / "checkpoint.pt")

    with torch.no_grad():
        _, btr = model(Xtr_d, return_bottleneck=True)
        _, bdv = model(Xdv_d, return_bottleneck=True)
    torch.save({
        "feature_name": name, "bottleneck_dim": BOTTLENECK,
        "train": {"emb": btr.cpu().to(torch.float16), "labels": ytr, "sample_ids": tr_ids},
        "dev": {"emb": bdv.cpu().to(torch.float16), "labels": ydv, "sample_ids": dv_ids},
        "note": "bottleneck activation (post LayerNorm-GELU); MIntRec student Feature-KD target",
    }, out_dir / "bottleneck_reps.pt")
    print(f"Saved -> {out_dir}")

    return {"feature": name, "bottleneck": BOTTLENECK, "eval_acc": round(eval_acc, 4),
            "eval_macro_f1": round(eval_f1, 4), "train_acc": round(train_acc, 4)}


def main():
    print(f"Device: {DEVICE}")
    results = [train_one(name) for name in FEATURES]

    csv_path = OUT_PLOT / "bottleneck_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "bottleneck", "eval_acc", "eval_macro_f1", "train_acc"])
        w.writeheader(); w.writerows(results)
    print(f"\nResults CSV -> {csv_path}")
    for r in results:
        print(r)
    print("Done.")


if __name__ == "__main__":
    main()
