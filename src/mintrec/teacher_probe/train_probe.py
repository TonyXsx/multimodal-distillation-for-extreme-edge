"""
Linear probe on the frozen teacher audio features for MIntRec2.0.

How much intent information is in the audio-token hidden states of a teacher
that also attended over text and video? This is the go/no-go number for using
the frozen teacher as a KD target, same role as the FSC probe stage.

Same protocol as the FSC probe. Dev is the eval set so nothing selects on it.
Fixed 50 epochs, AdamW(1e-3, wd 1e-4), batch 256, CE, standardised with train
mean/std. Each feature gets a linear head and a 1024 MLP head as a nonlinear
upper bound, reported once after training.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import sys
_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import MINTREC_DATA, MINTREC_OUTPUTS   # noqa: E402
from common.probe import Probe                            # noqa: E402


def build_feat_tag(dtype):
    return f"mintrec2.0_multimodal__qwen2.5-omni-3b-{dtype}__pf_text-video-audio__audiomean"

OUT_DIR = MINTREC_OUTPUTS / "teacher_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)


EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (arch id, hidden dims, description)
ARCHS = [
    ("A1", [],     "linear"),
    ("A2", [1024], "1024-MLP upper bound"),
]


def standardize(Xtr, Xev):
    mean = Xtr.mean(dim=0, keepdim=True)
    std = Xtr.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (Xtr - mean) / std, (Xev - mean) / std


def train_one(hidden, Xtr, ytr, Xev, yev, n_classes):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = Probe(Xtr.shape[1], hidden, n_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    Xtr_d, ytr_d = Xtr.to(DEVICE), ytr.to(DEVICE)
    Xev_d, yev_d = Xev.to(DEVICE), yev.to(DEVICE)
    n = Xtr_d.shape[0]
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            loss_fn(model(Xtr_d[idx]), ytr_d[idx]).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        tr_pred = model(Xtr_d).argmax(1)
        ev_pred = model(Xev_d).argmax(1)
    train_acc = (tr_pred == ytr_d).float().mean().item()
    dev_acc = (ev_pred == yev_d).float().mean().item()
    dev_f1 = f1_score(yev.numpy(), ev_pred.cpu().numpy(), average="macro")
    return train_acc, dev_acc, dev_f1


def main():
    ap = argparse.ArgumentParser(description="Linear/MLP probe on frozen multimodal-teacher audio features.")
    ap.add_argument("--dtype", choices=["bf16", "4bit"], default="bf16",
                    help="Which teacher-precision features to probe (must match extract_features.py).")
    args = ap.parse_args()

    feat_dir = MINTREC_DATA / "teacher_features" / build_feat_tag(args.dtype)
    print(f"Device: {DEVICE}  |  features: {feat_dir.name}")
    tr = torch.load(feat_dir / "train_features.pt", weights_only=False)
    dv = torch.load(feat_dir / "dev_features.pt", weights_only=False)
    ytr = tr["labels"].long()
    yev = dv["labels"].long()
    n_classes = int(max(ytr.max(), yev.max()).item()) + 1
    feat_names = list(tr["features"].keys())
    print(f"train N={len(ytr)}  dev N={len(yev)}  classes={n_classes}")
    print(f"features: {feat_names}\n")

    results = []
    for feat in feat_names:
        Xtr = tr["features"][feat].float()
        Xev = dv["features"][feat].float()
        Xtr_n, Xev_n = standardize(Xtr, Xev)
        for arch_id, hidden, desc in ARCHS:
            tr_acc, dev_acc, dev_f1 = train_one(hidden, Xtr_n, ytr, Xev_n, yev, n_classes)
            print(f"  {feat:<28} [{arch_id}] dev_acc={dev_acc:.4f}  macroF1={dev_f1:.4f}  (train_acc={tr_acc:.4f})")
            results.append({
                "feature": feat, "arch": arch_id, "arch_desc": desc,
                "dev_acc": dev_acc, "dev_macro_f1": dev_f1, "train_acc": tr_acc,
            })

    results.sort(key=lambda r: r["dev_acc"], reverse=True)
    csv_path = OUT_DIR / f"results_{args.dtype}.csv"   # keep bf16 and 4bit side by side
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "arch", "arch_desc",
                                          "dev_acc", "dev_macro_f1", "train_acc"])
        w.writeheader()
        w.writerows(results)

    best = results[0]
    print(f"\nResults CSV -> {csv_path}")
    print(f"Best: {best['feature']} [{best['arch']}]  dev_acc={best['dev_acc']:.4f}  macroF1={best['dev_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
